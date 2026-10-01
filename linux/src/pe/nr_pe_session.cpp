#include "nr_pe_session.hpp"
#include "nr_pe_log.hpp"
#include "nr_log.hpp"
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

namespace nr::pe {
namespace {

using nr::pe::log;

// What the process has left of its own address space.
//
// This exists because of one log line: `neural rendering unavailable:
// std::bad_alloc`, from a 32-bit game at 4K with 11979 MB of VRAM free, 482 ms
// into a build - and nothing else. A `std::bad_alloc` is `operator new`
// refusing, which in a 32-bit process almost always means the 2 GB of virtual
// address space is gone, not that the machine is out of memory; a Vulkan
// failure would have come back through NRVK_CHECK with a VkResult instead. The
// two numbers that tell those apart are the total free and the **largest
// contiguous free block** - a 300 MB request fails against 900 MB of free space
// scattered in 60 MB pieces, and only the second number shows it.
//
// A 64-bit process has terabytes and this reads as such; the walk costs one
// VirtualQuery per region either way, and it is only ever called on a failure.
std::string address_space_report() {
    MEMORY_BASIC_INFORMATION region{};
    unsigned long long free_total = 0, largest = 0;
    auto* address = static_cast<unsigned char*>(nullptr);
    while (VirtualQuery(address, &region, sizeof region) == sizeof region) {
        if (region.State == MEM_FREE) {
            free_total += region.RegionSize;
            if (region.RegionSize > largest) largest = region.RegionSize;
        }
        auto* next = static_cast<unsigned char*>(region.BaseAddress) + region.RegionSize;
        if (next <= address) break;   // no progress, or the pointer wrapped
        address = next;
    }
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof memory;
    GlobalMemoryStatusEx(&memory);
    char text[256];
    std::snprintf(text, sizeof text,
                  "address space: %.0f MB free in this process, largest contiguous block %.0f MB "
                  "(of %.0f MB total); %.0f MB of RAM free",
                  double(free_total) / (1 << 20), double(largest) / (1 << 20),
                  double(memory.ullTotalVirtual) / (1 << 20),
                  double(memory.ullAvailPhys) / (1 << 20));
    return text;
}

}  // namespace

struct Session::Impl {
    std::string root;
    std::string status;
    // ---- one built network per extent ----------------------------------------
    //
    // There used to be exactly one, rebuilt whenever anything about it changed.
    // OptiScaler keeps neural-rendering features alive at *two* extents at once
    // - pre-upscale at the render resolution and post-upscale at the display
    // one - and creates and destroys them on every toggle, so a single slot
    // thrashed: 1920x1080, then 3840x2160, then 1920x1080 again, a full rebuild
    // each time, with frames passing through unenhanced in between. That is the
    // stall the user reported and it is also why "passes" appeared to do
    // nothing: the network was never finished long enough to run.
    //
    // So built networks are kept, keyed by everything that makes one different,
    // and a switch between two of them is a pointer assignment. They are
    // evicted least-recently-used, and only when the card cannot hold the next
    // one (fits_in_memory) - the arena is 1.05 GB at 1080p and 4.07 GB at 4K, so
    // this is not a cache that can be allowed to grow.
    struct Built {
        std::unique_ptr<Runtime> runtime;
        uint32_t width{}, height{};
        VkFormat format{VK_FORMAT_UNDEFINED};
        bool linear{};
        float scale{1.0f};
        // The pass ceiling this one was built for: it sizes one history image
        // per pass, so a network built for four cannot serve a request for six.
        uint32_t passes{4};
        bool prep{};   // RuntimeConfig::preprocess
        uint64_t used{};   // LRU stamp
    };
    std::vector<Built> cache;
    uint64_t use_stamp{};
    // The entry in `cache` the run paths are using. Raw, because the cache owns
    // them; null until the first build lands.
    Runtime* runtime{};
    uint32_t width{}, height{};
    VkFormat format{VK_FORMAT_UNDEFINED};
    DeviceHandles handles{};
    // Our own render-resolution image, handed to the upscaler in place of the
    // game's colour. Owned as a D3D12 resource so the upscaler can take it, and
    // used as a VkImage so the network can write it - the same allocation seen
    // from both sides, which is the whole advantage of being inside vkd3d.
    ID3D12Resource* output{};
    ResourceHandle output_vk{};
    bool failed{};   // one hard failure disables the session rather than retrying into a crash
    bool typeless_noted{};
    float history_strength{1.0f};

    // A Vulkan game has no D3D12 allocation to borrow, so the output is ours.
    // Our own command buffer, for the paths that are not handed one.
    VkCommandPool pool{};
    VkCommandBuffer own_cmd{};
    // Building the runtime uploads the weights, which is a submit, not a
    // recording. Every path therefore has to find a queue before it can build
    // anything - the earlier code passed VK_NULL_HANDLE here and the runtime
    // rejects that outright, so no path had ever built a network.
    QueueAccess access{};
    ID3D12CommandQueue* owned_queue{};
    // DXVK's interop device, held for the life of the session: the queue lock
    // closures call into it every frame.
    IDXGIVkInteropDevice* dxvk{};

    VkImage vk_output{};
    VkImageView vk_output_view{};
    VkDeviceMemory vk_output_memory{};
    uint32_t vk_w{}, vk_h{};
    VkFormat vk_format{VK_FORMAT_UNDEFINED};

    // The one thing vkd3d-proton cannot be told any other way: that the compute
    // bind point of its command buffer no longer holds the pipeline it thinks it
    // does. See `forget_bound_pipeline` below for the whole story; these three
    // objects exist only to give it a ClearUAV to run.
    ID3D12Resource* invalidator{};
    ID3D12DescriptorHeap* invalidator_cpu_heap{};
    ID3D12DescriptorHeap* invalidator_gpu_heap{};
    bool invalidator_failed{};
    bool ensure_invalidator(ID3D12Device* device);
    void forget_bound_pipeline(ID3D12Device* device, ID3D12GraphicsCommandList* list);
    void release_invalidator();

    bool ensure_output(ID3D12Device* device, uint32_t w, uint32_t h, DXGI_FORMAT dxgi);
    void release_output();
    bool vk_output_valid(uint32_t w, uint32_t h, VkFormat format) const {
        return vk_output && vk_w == w && vk_h == h && vk_format == format;
    }
    bool ensure_vk_output(uint32_t w, uint32_t h, VkFormat format);
    void release_vk_output();
    // `linear`: the colour is scene-referred linear light (an upscaler's input
    // in a float format), so the runtime encodes it; false for every swapchain.
    // Builds in the background: returns false, with `status` saying so, until
    // the network is ready, and the caller leaves the frame alone meanwhile.
    // `controls` asking for a preprocess makes every runtime from then on one
    // that can run it (prep_want); see RuntimeConfig::preprocess.
    bool ensure_runtime(const Controls& controls, uint32_t w, uint32_t h, VkFormat format, bool linear);
    bool ensure_runtime_(const Controls& controls, uint32_t w, uint32_t h, VkFormat format, bool linear);
    // The preprocess meter in the log every five seconds while it runs, so NR on/off at one spot
    // can be compared (the feedback loop that made upstream drop frame metering).
    std::chrono::steady_clock::time_point meter_logged{};
    static bool is_linear_format(VkFormat f) {
        return f == VK_FORMAT_R16G16B16A16_SFLOAT || f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 ||
               f == VK_FORMAT_R32G32B32A32_SFLOAT ||
               // Taken only through the rest-of-DXGI colour formats
               // (nr_pe_interop.cpp colour_format_fallback), so no frame that
               // ran before changes its answer here.
               f == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 || f == VK_FORMAT_R32G32B32_SFLOAT;
    }
    // The background build. Everything the thread needs is copied in; it hands
    // back either a runtime or an error under `build_lock`, and the render
    // thread adopts whichever on its next call.
    std::thread build_thread;
    std::mutex build_lock;
    bool building{}, build_done{}, build_linear{}, build_prep{};
    uint32_t build_passes{kMaxPasses};
    uint32_t build_w{}, build_h{};
    VkFormat build_format{VK_FORMAT_UNDEFINED};
    std::unique_ptr<Runtime> built;
    std::string build_error;
    double build_seconds{};
    // A build that ran out of *host* memory is a moment, not a verdict: the one
    // that failed did so 168 ms after a fullscreen transition, while DXVK was
    // re-making a 4K swapchain and the game was reloading into the same 2 GB.
    // The same build a few seconds later may well fit, so it is retried with a
    // backoff instead of latching `failed` and declining every frame for the
    // rest of the session, which is what used to happen. Every other exception
    // still latches - a device that cannot run this network will not start to.
    bool build_oom{};
    unsigned oom_failures{};
    static constexpr unsigned kOomRetries = 4;
    std::chrono::steady_clock::time_point retry_at{};
    bool linear{};
    float white_point{1.0f};
    // Model Resolution. A change rebuilds the network, like a resolution change.
    float model_scale{1.0f}, build_scale{1.0f}, built_scale{1.0f};
    bool native_compose{};   // Session::set_native_compose
    // Controls::preprocess was asked for once: from then on every runtime is
    // built able to run it, so the hotkey switches at once after the first
    // time. Never cleared; until then nothing about the runtime changes.
    bool prep_want{};
    // Every runtime is built able to run this many passes; Controls::passes
    // picks the count per frame.
    // What a network is built for unless the host asks for more. Four is what
    // shipped; the ceiling is 16 and raising it costs one model-sized history
    // image per pass, which is why it is not simply set there.
    static constexpr uint32_t kMaxPasses = 4;
    static constexpr uint32_t kPassCeiling = 16;
    uint32_t max_passes_want{kMaxPasses};
    bool ensure_own_command_buffer();
    bool ensure_d3d12_device(ID3D12Device* device);
    bool fits_in_memory(uint32_t w, uint32_t h);
    // Make `entry` the one the run paths use. Returns false when there is no
    // such entry.
    bool select_runtime(uint32_t w, uint32_t h, VkFormat format, bool want_linear);
    bool select_bridge_runtime(uint32_t w, uint32_t h, VkFormat format, bool want_linear);
    void trigger_build(const Controls& controls, uint32_t w, uint32_t h, VkFormat format, bool want_linear);
    bool have_runtime(uint32_t w, uint32_t h) const {
        for (const auto& e : cache) if (e.width == w && e.height == h) return true;
        return false;
    }
    bool have_runtime_format(uint32_t w, uint32_t h, VkFormat f) const {
        for (const auto& e : cache) if (e.width == w && e.height == h && e.format == f) return true;
        return false;
    }

    // Format bridge for instant cutscene and scene transition support without passthrough
    VkImage bridge_image{};
    VkDeviceMemory bridge_memory{};
    uint32_t bridge_w{}, bridge_h{};
    VkFormat bridge_format{VK_FORMAT_UNDEFINED};
    bool bridged{false};
    VkFormat bridged_target_format{VK_FORMAT_UNDEFINED};
    bool ensure_bridge(uint32_t w, uint32_t h, VkFormat fmt);
    void release_bridge();

    // Dual format proactive warm-up queue
    VkFormat warmup_format{VK_FORMAT_UNDEFINED};
    uint32_t warmup_w{}, warmup_h{};
    bool warmup_linear{};

    // Free least-recently-used entries until the card can hold another network
    // at this extent. False means it cannot hold one even with the cache empty.
    bool make_room(uint32_t w, uint32_t h);
    void drop(size_t index, const char* why = "to make room");
    // Feature ids, handed out by Session::create_feature. Monotonic, never
    // reused, so a stale id can only ever miss.
    uint64_t next_feature{1};
    // One of everything per swapchain image. A binary semaphore may have exactly
    // one pending signal-wait pair, and the present waiting on ours is still
    // outstanding when the next frame is recorded - so a single one reused every
    // frame is undefined, which is what a black screen followed by a crash looks
    // like. Same argument for the command buffer and the fence.
    struct PresentSlot {
        VkCommandBuffer cmd{};
        VkFence fence{};
        VkSemaphore done{};
        bool pending{};
    };
    std::vector<PresentSlot> slots;
    VkCommandPool slot_pool{};
    uint32_t slot_family{~0u};
    bool ensure_slots(uint32_t count);
    bool wait_slot(PresentSlot& slot);
    void release_slots();
    // Resetting a command buffer the GPU is still reading is undefined, and at
    // a 4K working extent the GPU is a long way behind by the time the next
    // frame arrives. One fence, waited on before each reuse.
    VkFence own_fence{};
    uint32_t pool_family{~0u};
    // An extent seen but not yet built for, and how many consecutive frames it
    // has held. See run_present_vulkan.
    uint32_t pending_w{}, pending_h{}, pending_frames{};
    bool own_cmd_pending{};
    bool wait_for_own_cmd();
};

void Session::Impl::release_vk_output() {
    if (vk_output_view) { vkDestroyImageView(handles.device, vk_output_view, nullptr); vk_output_view = VK_NULL_HANDLE; }
    if (vk_output) { vkDestroyImage(handles.device, vk_output, nullptr); vk_output = VK_NULL_HANDLE; }
    if (vk_output_memory) { vkFreeMemory(handles.device, vk_output_memory, nullptr); vk_output_memory = VK_NULL_HANDLE; }
    vk_w = vk_h = 0; vk_format = VK_FORMAT_UNDEFINED;
}

bool Session::Impl::ensure_vk_output(uint32_t w, uint32_t h, VkFormat format) {
    release_vk_output();
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D; ci.format = format; ci.extent = {w, h, 1};
    ci.mipLevels = ci.arrayLayers = 1; ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(handles.device, &ci, nullptr, &vk_output) != VK_SUCCESS) {
        status = "could not create the render-resolution output image";
        return false;
    }
    VkMemoryRequirements req{}; vkGetImageMemoryRequirements(handles.device, vk_output, &req);
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(handles.physical, &mem);
    uint32_t type = mem.memoryTypeCount;
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
        if ((req.memoryTypeBits & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { type = i; break; }
    if (type == mem.memoryTypeCount) { release_vk_output(); status = "no device-local memory"; return false; }
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size; alloc.memoryTypeIndex = type;
    if (vkAllocateMemory(handles.device, &alloc, nullptr, &vk_output_memory) != VK_SUCCESS ||
        vkBindImageMemory(handles.device, vk_output, vk_output_memory, 0) != VK_SUCCESS) {
        release_vk_output(); status = "could not back the output image"; return false;
    }
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = vk_output; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(handles.device, &vi, nullptr, &vk_output_view) != VK_SUCCESS) {
        release_vk_output(); status = "could not view the output image"; return false;
    }
    vk_w = w; vk_h = h; vk_format = format;
    return true;
}

void Session::Impl::release_bridge() {
    if (bridge_image) {
        vkDestroyImage(handles.device, bridge_image, nullptr);
        bridge_image = VK_NULL_HANDLE;
    }
    if (bridge_memory) {
        vkFreeMemory(handles.device, bridge_memory, nullptr);
        bridge_memory = VK_NULL_HANDLE;
    }
    bridge_w = bridge_h = 0;
    bridge_format = VK_FORMAT_UNDEFINED;
}

bool Session::Impl::ensure_bridge(uint32_t w, uint32_t h, VkFormat format) {
    if (bridge_image && bridge_w == w && bridge_h == h && bridge_format == format)
        return true;
    release_bridge();
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = {w, h, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(handles.device, &ci, nullptr, &bridge_image) != VK_SUCCESS) {
        status = "could not create the format bridge image";
        return false;
    }
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(handles.device, bridge_image, &req);
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(handles.physical, &mem);
    uint32_t type = mem.memoryTypeCount;
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            type = i;
            break;
        }
    }
    if (type == mem.memoryTypeCount) {
        release_bridge();
        status = "no device-local memory for bridge";
        return false;
    }
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    if (vkAllocateMemory(handles.device, &alloc, nullptr, &bridge_memory) != VK_SUCCESS ||
        vkBindImageMemory(handles.device, bridge_image, bridge_memory, 0) != VK_SUCCESS) {
        release_bridge();
        status = "could not back the bridge image";
        return false;
    }
    bridge_w = w;
    bridge_h = h;
    bridge_format = format;
    return true;
}

static void transition_layout(VkCommandBuffer cmd, VkImage image,
                              VkImageLayout old_layout, VkImageLayout new_layout,
                              VkAccessFlags src_access, VkAccessFlags dst_access,
                              VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = src_access;
    b.dstAccessMask = dst_access;
    b.oldLayout = old_layout;
    b.newLayout = new_layout;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

bool Session::Impl::wait_for_own_cmd() {
    if (!own_cmd_pending || !own_fence) return true;
    // A second is far longer than any frame; reaching it means the device is
    // lost or hung, and continuing to submit into that is how a hang becomes a
    // crash.
    const VkResult r = vkWaitForFences(handles.device, 1, &own_fence, VK_TRUE, 1000000000ull);
    if (r != VK_SUCCESS) {
        status = "the previous neural rendering submit never completed";
        failed = true;
        return false;
    }
    vkResetFences(handles.device, 1, &own_fence);
    own_cmd_pending = false;
    return true;
}

bool Session::Impl::ensure_own_command_buffer() {
    // A pool belongs to one queue family for its whole life. If the family we
    // submit on has changed - and it does, because the queue a game presents on
    // is not necessarily the one we found when looking for a compute queue -
    // the pool has to go with it. Submitting a buffer from another family's pool
    // is invalid and takes the process down.
    if (own_cmd && pool_family != access.family) {
        wait_for_own_cmd();
        if (pool) { vkDestroyCommandPool(handles.device, pool, nullptr); pool = VK_NULL_HANDLE; }
        own_cmd = VK_NULL_HANDLE;
    }
    if (own_cmd) return true;
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.queueFamilyIndex = access.family;
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(handles.device, &pi, nullptr, &pool) != VK_SUCCESS) {
        status = "could not create a command pool on the game's device";
        return false;
    }
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(handles.device, &ai, &own_cmd) != VK_SUCCESS) {
        status = "could not allocate a command buffer";
        return false;
    }
    pool_family = access.family;
    if (own_fence) return true;
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(handles.device, &fi, nullptr, &own_fence) != VK_SUCCESS) {
        status = "could not create the fence that paces our own submits";
        return false;
    }
    return true;
}

bool Session::Impl::wait_slot(PresentSlot& slot) {
    if (!slot.pending) return true;
    const VkResult r = vkWaitForFences(handles.device, 1, &slot.fence, VK_TRUE, 1000000000ull);
    if (r != VK_SUCCESS) {
        status = "the previous neural rendering submit never completed";
        failed = true;
        return false;
    }
    vkResetFences(handles.device, 1, &slot.fence);
    slot.pending = false;
    return true;
}

void Session::Impl::release_slots() {
    for (auto& slot : slots) {
        wait_slot(slot);
        if (slot.done) vkDestroySemaphore(handles.device, slot.done, nullptr);
        if (slot.fence) vkDestroyFence(handles.device, slot.fence, nullptr);
    }
    slots.clear();
    if (slot_pool) { vkDestroyCommandPool(handles.device, slot_pool, nullptr); slot_pool = VK_NULL_HANDLE; }
    slot_family = ~0u;
}

bool Session::Impl::ensure_slots(uint32_t count) {
    if (count == 0) count = 1;
    if (slots.size() == count && slot_family == access.family) return true;
    release_slots();

    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.queueFamilyIndex = access.family;
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(handles.device, &pi, nullptr, &slot_pool) != VK_SUCCESS) {
        status = "could not create a command pool on the game's present family";
        return false;
    }
    std::vector<VkCommandBuffer> buffers(count);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = slot_pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = count;
    if (vkAllocateCommandBuffers(handles.device, &ai, buffers.data()) != VK_SUCCESS) {
        status = "could not allocate the per-image command buffers";
        return false;
    }
    slots.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        slots[i].cmd = buffers[i];
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (vkCreateFence(handles.device, &fi, nullptr, &slots[i].fence) != VK_SUCCESS ||
            vkCreateSemaphore(handles.device, &si, nullptr, &slots[i].done) != VK_SUCCESS) {
            status = "could not create the per-image fence and semaphore";
            release_slots();
            return false;
        }
    }
    slot_family = access.family;
    log("[nr] %u present slots on queue family %u", count, access.family);
    return true;
}

// Will the network fit in what is left of the card?
//
// The activation arena scales with the working extent and it is not small: 1.05
// GB at 1920x1088 and 4.07 GB at 3840x2176 with every value in its own bytes,
// about 500 bytes per pixel; 384 MB and 1.53 GB since values share memory by
// lifetime (nr_graph.cpp `reuse`), about 185 bytes per pixel (200 below, for
// the small extents' fixed part). NR_ARENA_REUSE=0 turns sharing off. A game at 4K has already taken most of a 16 GB card, so the
// fallback - which runs at output resolution, because it works on the finished
// frame - asks for four gigabytes that are not there. Allocation then fails
// somewhere inside the graph build, or succeeds by evicting the game's own
// resources, and what the player sees is a black frame and a crash.
//
// Refusing with a number in the log is strictly better than either.
static uint64_t arena_bytes_per_pixel() {
    const char* e = std::getenv("NR_ARENA_REUSE");
    return e && *e && std::atoi(e) == 0 ? 520 : 200;
}

bool Session::Impl::fits_in_memory(uint32_t w, uint32_t h) {
    // The padded working extent, which is what the arena is sized from.
    const uint64_t pixels = uint64_t((w + 63) / 64 * 64) * ((h + 63) / 64 * 64);
    const uint64_t need = pixels * arena_bytes_per_pixel() + (300ull << 20);   // arena, plus weights and scratch

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    props.pNext = &budget;
    vkGetPhysicalDeviceMemoryProperties2(handles.physical, &props);

    // The largest device-local heap's free space, by budget where the driver
    // reports one and by heap size where it does not.
    uint64_t available = 0;
    const auto& mem = props.memoryProperties;
    for (uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
        if (!(mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
        const uint64_t heap = mem.memoryHeaps[i].size;
        const uint64_t used = budget.heapUsage[i];
        const uint64_t cap = budget.heapBudget[i] ? budget.heapBudget[i] : heap;
        const uint64_t free = cap > used ? cap - used : 0;
        if (free > available) available = free;
    }
    const double need_mb = double(need) / (1 << 20);
    const double free_mb = double(available) / (1 << 20);
    if (available && need > available) {
        char message[256];
        std::snprintf(message, sizeof message,
                      "the network needs about %.0f MB at %ux%u and only %.0f MB is free; "
                      "not building it. At output resolution this is expected - run the "
                      "network before the upscaler instead, where the extent is smaller.",
                      need_mb, w, h, free_mb);
        status = message;
        log("[nr] %s", message);
        return false;
    }
    log("[nr] %ux%u needs about %.0f MB, %.0f MB free", w, h, need_mb, free_mb);
    return true;
}

bool Session::Impl::select_runtime(uint32_t w, uint32_t h, VkFormat format, bool want_linear) {
    for (auto& e : cache) {
        if (e.width != w || e.height != h || e.format != format || e.linear != want_linear ||
            e.scale != model_scale || e.passes != max_passes_want || e.prep != prep_want)
            continue;
        e.used = ++use_stamp;
        if (runtime != e.runtime.get() || bridged) {
            log("[nr] runtime cache hit: %ux%u fmt %u (model %ux%u, scale %.2f%s), %u built, "
                "%u live features", e.width, e.height, unsigned(e.format), e.runtime->model_width(),
                e.runtime->model_height(), e.scale, e.linear ? ", linear" : "",
                unsigned(cache.size()), e.runtime->live_features());
        }
        runtime = e.runtime.get();
        width = e.width; height = e.height; this->format = e.format;
        linear = e.linear; built_scale = e.scale;
        bridged = false;
        return true;
    }
    return false;
}

bool Session::Impl::select_bridge_runtime(uint32_t w, uint32_t h, VkFormat format, bool want_linear) {
    for (auto& e : cache) {
        if (e.width != w || e.height != h || e.scale != model_scale || e.prep != prep_want)
            continue;
        const bool both_linear = is_linear_format(e.format) && is_linear_format(format);
        const bool both_unorm = !is_linear_format(e.format) && !is_linear_format(format);
        if (!both_linear && !both_unorm)
            continue;

        e.used = ++use_stamp;
        if (runtime != e.runtime.get() || !bridged || bridged_target_format != format) {
            log("[nr] format bridge active: serving request %ux%u fmt %u through resident network fmt %u",
                w, h, unsigned(format), unsigned(e.format));
        }
        runtime = e.runtime.get();
        width = e.width; height = e.height; this->format = e.format;
        linear = e.linear; built_scale = e.scale;
        bridged = true;
        bridged_target_format = format;
        return true;
    }
    return false;
}

void Session::Impl::drop(size_t index, const char* why) {
    auto& e = cache[index];
    log("[nr] evicting the %ux%u network %s", e.width, e.height, why);
    if (runtime == e.runtime.get()) { runtime = nullptr; width = height = 0; }
    // Nothing of ours may still be reading it. On the paths that submit for
    // themselves this waits on our own fence; on the paths that record into the
    // game's command list there is nothing of ours to wait for, and eviction
    // only happens on a frame where the caller has already been told to pass
    // through - which is exactly when it is safe.
    wait_for_own_cmd();
    cache.erase(cache.begin() + long(index));
}

bool Session::Impl::make_room(uint32_t w, uint32_t h) {
    while (true) {
        if (fits_in_memory(w, h)) return true;
        if (cache.empty()) return false;
        size_t oldest = 0;
        for (size_t i = 1; i < cache.size(); ++i)
            if (cache[i].used < cache[oldest].used) oldest = i;
        drop(oldest);
    }
}

bool Session::Impl::ensure_runtime(const Controls& controls, uint32_t w, uint32_t h, VkFormat format,
                                   bool want_linear) {
    const bool ok = ensure_runtime_(controls, w, h, format, want_linear);
    if (ok && controls.preprocess.active() && controls.preprocess.exposure == 1) {
        const auto now = std::chrono::steady_clock::now();
        const auto m = runtime->preprocess_meter();
        if (now - meter_logged > std::chrono::seconds(5) && m.first == m.first) {
            meter_logged = now;
            const float bias = controls.preprocess.bias_ev;
            log("[nr] preprocess exposure: auto %+.2f EV (this frame asks %+.2f) + ExposureBias %+.2f = %+.2f EV",
                m.first, m.second, bias, m.first + bias);
        }
    }
    return ok;
}

void Session::Impl::trigger_build(const Controls& controls, uint32_t w, uint32_t h, VkFormat format,
                                  bool want_linear) {
    if (building) return;
    if (!make_room(w, h)) return;
    RuntimeConfig config;
    config.root = root; config.width = w; config.height = h; config.colour_format = format;
    config.linear_input = want_linear; config.white_point = white_point;
    config.model_scale = model_scale;
    config.max_passes = native_compose ? 1u : max_passes_want;
    config.native_compose = native_compose;
    config.preprocess = prep_want;
    config.preprocess_unknee = want_linear || (native_compose && is_linear_format(format));
    if (native_compose) config.model_scale = 1.0f;
    TemporalConfig temporal; temporal.enable = true;
    temporal.history_strength = history_strength;
    HostDevice host{};
    host.instance = handles.instance; host.physical = handles.physical; host.device = handles.device;
    host.queue = access.queue; host.queue_family = access.family;
    const QueueAccess access_copy = access;
    building = true; build_w = w; build_h = h; build_format = format; build_linear = want_linear;
    build_scale = model_scale; build_passes = max_passes_want; build_prep = prep_want;
    status = "building the network in the background; frames pass through until it is ready";
    log("[nr] building the network at %ux%u fmt %u (model scale %.2f) in the background%s", w, h,
        unsigned(format), model_scale,
        want_linear ? " (linear-light colour: encoding with a white point, see dlssnr-amd.ini white_point)" : "");
    build_thread = std::thread([this, host, config, temporal, access_copy] {
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_ptr<Runtime> made;
        std::string error;
        bool oom = false;
        try {
            HostDevice locked = host;
            locked.queue_lock = access_copy.lock_quiet;
            locked.queue_unlock = access_copy.unlock_quiet;
            made = std::make_unique<Runtime>(locked, config, ControlMaskConfig{}, temporal);
        } catch (const std::bad_alloc&) {
            oom = true;
            error = "neural rendering unavailable: out of host memory. " + address_space_report();
        } catch (const std::exception& e) {
            error = std::string("neural rendering unavailable: ") + e.what();
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::lock_guard<std::mutex> guard(build_lock);
        built = std::move(made); build_error = error; build_seconds = seconds; build_done = true;
        build_oom = oom;
    });
}

bool Session::Impl::ensure_runtime_(const Controls& controls, uint32_t w, uint32_t h, VkFormat format,
                                    bool want_linear) {
    if (controls.preprocess.active() && !prep_want) {
        prep_want = true;
        log("[nr] preprocess asked for: networks are rebuilt able to run it (once)");
    }
    // Adopt a finished background build first.
    {
        std::lock_guard<std::mutex> guard(build_lock);
        if (build_done) {
            if (build_thread.joinable()) build_thread.join();
            build_done = false; building = false;
            if (built) {
                Built entry;
                entry.width = build_w; entry.height = build_h; entry.format = build_format;
                entry.linear = build_linear; entry.scale = build_scale;
                entry.passes = build_passes; entry.prep = build_prep;
                entry.used = ++use_stamp;
                entry.runtime = std::move(built);
                log("[nr] network built at %ux%u (model %ux%u, scale %.2f) on queue family %u in "
                    "%.1f s%s; %u networks resident",
                    entry.width, entry.height, entry.runtime->model_width(),
                    entry.runtime->model_height(), entry.scale, access.family, build_seconds,
                    entry.linear ? "; the colour is linear light and is encoded for the network" : "",
                    unsigned(cache.size() + 1));
                for (size_t i = cache.size(); i-- > 0;)
                    if (entry.prep && !cache[i].prep && cache[i].width == entry.width &&
                        cache[i].height == entry.height)
                        drop(i, "built without the preprocess");
                cache.push_back(std::move(entry));
                auto& adopted = cache.back();
                runtime = adopted.runtime.get();
                width = adopted.width; height = adopted.height; this->format = adopted.format;
                linear = adopted.linear; built_scale = adopted.scale;
                bridged = false;
                oom_failures = 0;
                retry_at = {};

                // Proactive dual format warm-up:
                VkFormat counterpart = VK_FORMAT_UNDEFINED;
                if (adopted.format == VK_FORMAT_B10G11R11_UFLOAT_PACK32)
                    counterpart = VK_FORMAT_R16G16B16A16_SFLOAT;
                else if (adopted.format == VK_FORMAT_R16G16B16A16_SFLOAT)
                    counterpart = VK_FORMAT_B10G11R11_UFLOAT_PACK32;

                if (counterpart != VK_FORMAT_UNDEFINED && !have_runtime_format(adopted.width, adopted.height, counterpart)) {
                    warmup_w = adopted.width;
                    warmup_h = adopted.height;
                    warmup_format = counterpart;
                    warmup_linear = adopted.linear;
                    log("[nr] queued background warm-up for companion format %u at %ux%u",
                        unsigned(counterpart), adopted.width, adopted.height);
                }
            } else {
                status = build_error;
                log("[nr] %s", status.c_str());
                if (build_oom && oom_failures < kOomRetries) {
                    const unsigned wait = 4u << oom_failures;
                    ++oom_failures;
                    retry_at = std::chrono::steady_clock::now() + std::chrono::seconds(wait);
                    log("[nr] retrying the build in %u s (attempt %u of %u)", wait, oom_failures,
                        kOomRetries);
                } else {
                    failed = true;
                }
                return false;
            }
        }
    }

    if (select_runtime(w, h, format, want_linear)) return true;

    // Trigger exact native build in background if not already running:
    if (!building) {
        if (!access.valid()) {
            status = "no submittable queue on the game's device; the weights cannot be uploaded";
            failed = true;
            return false;
        }
        if (retry_at != std::chrono::steady_clock::time_point{}) {
            if (std::chrono::steady_clock::now() < retry_at) {
                status = build_error;
                return false;
            }
            retry_at = {};
        }
        trigger_build(controls, w, h, format, want_linear);
    }

    // While native build is compiling in background, if we have a resident network
    // at the same resolution, serve the frame immediately via format bridge!
    if (select_bridge_runtime(w, h, format, want_linear)) {
        return true;
    }

    // If idle and warm-up format is queued, trigger companion format warm-up:
    if (!building && warmup_format != VK_FORMAT_UNDEFINED) {
        if (!have_runtime_format(warmup_w, warmup_h, warmup_format) && fits_in_memory(warmup_w, warmup_h)) {
            const VkFormat wf = warmup_format;
            const uint32_t ww = warmup_w, wh = warmup_h;
            const bool wl = warmup_linear;
            warmup_format = VK_FORMAT_UNDEFINED;
            log("[nr] starting background warm-up build for format %u at %ux%u", unsigned(wf), ww, wh);
            trigger_build(controls, ww, wh, wf, wl);
        } else {
            warmup_format = VK_FORMAT_UNDEFINED;
        }
    }

    if (building) {
        status = "building the network in the background; frames pass through until it is ready";
        return false;
    }
    return false;
}

void Session::Impl::release_output() {
    if (output) { output->Release(); output = nullptr; }
    output_vk = ResourceHandle{};
}

void Session::Impl::release_invalidator() {
    if (invalidator_gpu_heap) { invalidator_gpu_heap->Release(); invalidator_gpu_heap = nullptr; }
    if (invalidator_cpu_heap) { invalidator_cpu_heap->Release(); invalidator_cpu_heap = nullptr; }
    if (invalidator) { invalidator->Release(); invalidator = nullptr; }
}

// A 2x1 R32_UINT texture and the two descriptors a ClearUAV needs. Two texels
// because the clear below asks for ONE of them: see forget_bound_pipeline.
bool Session::Impl::ensure_invalidator(ID3D12Device* device) {
    if (invalidator && invalidator_cpu_heap && invalidator_gpu_heap) return true;
    if (invalidator_failed || !device) return false;

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = 2;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R32_UINT;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                 IID_PPV_ARGS(&invalidator));
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 1;
    if (SUCCEEDED(hr))
        hr = device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&invalidator_cpu_heap));
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (SUCCEEDED(hr))
        hr = device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&invalidator_gpu_heap));
    if (FAILED(hr)) {
        release_invalidator();
        invalidator_failed = true;
        log("[nr] could not build the pipeline invalidator (0x%08lX); a second neural pass in one "
            "command list will leave the caller's next dispatch bound to our kernel",
            (unsigned long)hr);
        return false;
    }

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R32_UINT;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(invalidator, nullptr, &uav,
                                      invalidator_cpu_heap->GetCPUDescriptorHandleForHeapStart());
    device->CreateUnorderedAccessView(invalidator, nullptr, &uav,
                                      invalidator_gpu_heap->GetCPUDescriptorHandleForHeapStart());
    return true;
}

// **Why a ClearUAV of two texels is the fix for multipass.** Measured with
// a D3D12 test harness, which drives our _nvngx.dll the way
// OptiScaler does, with OptiScaler's own codec shader between and after the
// evaluates, and reads every surface back:
//
//   one pass  - the caller's dispatch after the evaluate is correct.
//   two passes- the clamp (after the FIRST evaluate) is correct; every dispatch
//               after the SECOND evaluate runs our last network kernel instead
//               of the caller's shader. Its target is never written, so the
//               frame keeps whatever it held: the unedited picture. That is the
//               NR-off picture, exactly as reported in the game.
//
// The cause is in vkd3d-proton (libs/vkd3d/command.c). It tracks TWO pipelines:
// `state`/`current_pipeline`, the D3D12-level one, and `command_buffer_pipeline`,
// what it believes is really bound in the VkCommandBuffer. Our
// SetPipelineState(nullptr) below clears the first, so the caller's next
// SetPipelineState is not deduplicated at the D3D12 level - but the bind itself
// is still skipped, because the recomputed VkPipeline equals
// `command_buffer_pipeline`. The first dispatch after a recording escapes only
// because that caller pipeline had never been bound in this command buffer yet.
//
// vkd3d has exactly one mechanism for "something bound its own pipeline behind
// my back": d3d12_command_list_invalidate_current_pipeline(list, meta_shader=true),
// which sets `command_buffer_pipeline = VK_NULL_HANDLE` - "just pretend we never
// bound anything". It is not on any interface; it runs when vkd3d dispatches one
// of its own meta shaders. ClearUnorderedAccessViewUint takes that path whenever
// its fast path cannot serve the clear, and a rect covering PART of the image is
// refused by the fast path (d3d12_command_list_clear_uav_builtin: "Accept a
// single full-subresource rect"). So: two texels, clear one.
//
// It costs one tiny meta dispatch per evaluate and touches no state the caller
// can observe - unlike ClearState, which would do the same job by also throwing
// away the caller's render targets and bindings.
void Session::Impl::forget_bound_pipeline(ID3D12Device* device, ID3D12GraphicsCommandList* list) {
    if (!ensure_invalidator(device)) return;
    const UINT zero[4] = {0, 0, 0, 0};
    const D3D12_RECT rect{0, 0, 1, 1};   // half of a 2x1 image: the fast path must refuse it
    list->ClearUnorderedAccessViewUint(invalidator_gpu_heap->GetGPUDescriptorHandleForHeapStart(),
                                       invalidator_cpu_heap->GetCPUDescriptorHandleForHeapStart(),
                                       invalidator, zero, 1, &rect);
}

bool Session::Impl::ensure_output(ID3D12Device* device, uint32_t w, uint32_t h, DXGI_FORMAT dxgi) {
    if (output) {
        const auto desc = output->GetDesc();
        if (desc.Width == w && desc.Height == h && desc.Format == dxgi) return true;
        release_output();
    }
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = w; desc.Height = h; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
    desc.Format = dxgi; desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                               __uuidof(ID3D12Resource),
                                               reinterpret_cast<void**>(&output)))) {
        status = "could not allocate the render-resolution output";
        return false;
    }
    output_vk = colour_handle(device, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!output_vk.usable()) {
        release_output();
        status = "the output has no Vulkan image; this is not vkd3d-proton";
        return false;
    }
    return true;
}

namespace {

// One place turns a D3D12 resource into the ColourFrame the runtime wants, so
// colour, motion and depth cannot drift apart in layout handling.
bool fill(ColourFrame* out, const ResourceHandle& handle) {
    if (!handle.usable()) return false;
    out->image = handle.image;
    out->format = handle.format;
    out->width = handle.width; out->height = handle.height;
    out->before = out->after = handle.layout;
    out->usage = handle.usage;
    return true;
}

// Narrow a guide to the part of it the game rendered into: its subrect's base and extent, which the
// runtime reads from (EngineFrame::motion_x/depth_x and the narrowed width/height) - its blits take
// the region as their source rectangle, the in-place motion sample maps uv into it. A base outside
// the texture keeps the whole allocation (logged once).
void apply_guide_subrect(ColourFrame* guide, const Session::Subrect& want, const char* what,
                         uint32_t* base_x, uint32_t* base_y) {
    if (!guide->image) return;
    const uint32_t aw = guide->width, ah = guide->height;
    const int which = what[0] == 'd' ? 0 : 1;
    if (want.x >= aw || want.y >= ah) {
        static bool outside[2] = {false, false};
        if (!outside[which]) {
            outside[which] = true;
            log("[nr] the %s guide's subrect starts at (%u,%u), outside its %ux%u texture; the whole texture is used",
                what, want.x, want.y, aw, ah);
        }
        return;
    }
    // The region the game rendered into: from the base, the given extent (or the
    // rest of the texture), clipped to the texture - NVIDIA's own reading of the
    // subrect (base and extent, nvngx_dlssnr.dll 0x180022278).
    const uint32_t w = want.width ? std::min(want.width, aw - want.x) : aw - want.x;
    const uint32_t h = want.height ? std::min(want.height, ah - want.y) : ah - want.y;
    *base_x = want.x; *base_y = want.y;
    // A depth region that is not the whole image is read through the runtime's
    // copy, not sampled in place. Motion keeps its usage: it is sampled in place
    // through the region's base and share of the allocation.
    if (which == 0 && (w < aw || h < ah)) guide->usage = 0;
    guide->width = w; guide->height = h;
    if (want.x || want.y) {
        static bool noted[2] = {false, false};
        if (!noted[which]) {
            noted[which] = true;
            log("[nr] the %s guide's region starts at (%u,%u): %ux%u of the %ux%u texture", what, want.x, want.y,
                w, h, aw, ah);
        }
    }
}

}  // namespace

bool Session::Impl::ensure_d3d12_device(ID3D12Device* device) {
    if (handles.valid() && access.valid()) return true;
    if (!handles.valid()) {
        handles = device_handles(device);
        if (!handles.valid()) {
            status = "no Vulkan device underneath; not running on vkd3d-proton";
            failed = true;
            log("[nr] %s", status.c_str());
            return false;
        }
    }
    if (!access.valid() && !d3d12_queue_access(device, &access, &owned_queue)) {
        status = "vkd3d-proton would not hand out a queue; the weights cannot be uploaded";
        failed = true;
        log("[nr] %s", status.c_str());
        return false;
    }
    return true;
}

Session::Session(std::string root) : impl_(std::make_unique<Impl>()) {
    impl_->root = root.empty() ? std::string(nr::pe::module_folder()) : std::move(root);
    // The runtime's own diagnostics - the build's phase breakdown above all -
    // went to a stdout a game does not have. Send them to the same file
    // everything else here writes to.
    nr::set_log_sink([](const char* line) { log("%s", line); });
    log("[nr] session root %s", impl_->root.c_str());
    // NR_INPUT_CHECK=1 (NR_INPUT_CHECK_DEFAULT for a build that has it on without asking):
    // score what the game hands the network and what it gets back; pictures in
    // dlssnr-amd-check\ beside the log. NR_INPUT_CHECK_PICTURES=n sets how many captures
    // are written as pictures (default 4).
    static const bool input_check = [] {
#ifdef NR_INPUT_CHECK_DEFAULT
        bool on = NR_INPUT_CHECK_DEFAULT != 0;
#else
        bool on = false;
#endif
        char v[16] = {};
        if (GetEnvironmentVariableA("NR_INPUT_CHECK", v, sizeof v) > 0) on = v[0] == '1';
        if (!on) return false;
        int pictures = 4;
        if (GetEnvironmentVariableA("NR_INPUT_CHECK_PICTURES", v, sizeof v) > 0) pictures = std::atoi(v);
        std::string folder = nr::pe::module_folder();
        folder += folder.empty() ? "dlssnr-amd-check" : "\\dlssnr-amd-check";
        if (!CreateDirectoryA(folder.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) folder.clear();
        nr::set_input_check(true, folder, pictures);
        return true;
    }();
    (void)input_check;
}

void Session::set_white_point(float v) {
    impl_->white_point = v;
    for (auto& e : impl_->cache) e.runtime->set_white_point(v);
}

void Session::set_native_compose(bool v) { impl_->native_compose = v; }

void Session::set_max_passes(uint32_t v) {
    // Never below what shipped, so the common 1..4 never pays for a rebuild;
    // above it the next frame rebuilds, like a resolution change.
    const uint32_t want = v < Session::Impl::kMaxPasses ? Session::Impl::kMaxPasses
                        : (v > Session::Impl::kPassCeiling ? Session::Impl::kPassCeiling : v);
    impl_->max_passes_want = want;
}

void Session::set_model_scale(float v) {
    // Clamped to the range the runtime accepts; the rebuild happens on the
    // next frame, in the background, like a resolution change.
    impl_->model_scale = v < 0.25f ? 0.25f : (v > 1.0f ? 1.0f : v);
}

void Session::set_history_strength(float v) {
    impl_->history_strength = v;
    for (auto& e : impl_->cache) e.runtime->set_history_strength(v);
}

Session::~Session() {
    // A build still running owns nothing the session is about to destroy, but
    // it will hand back a runtime that must be destroyed too: wait for it.
    if (impl_->build_thread.joinable()) impl_->build_thread.join();
    impl_->built.reset();
    impl_->runtime = nullptr;
    impl_->cache.clear();
    impl_->release_output();
    impl_->release_invalidator();
    impl_->release_vk_output();
    impl_->release_bridge();
    if (impl_->pool) vkDestroyCommandPool(impl_->handles.device, impl_->pool, nullptr);
    if (impl_->owned_queue) impl_->owned_queue->Release();
    if (impl_->dxvk) impl_->dxvk->Release();
    if (impl_->own_fence) {
        // Nothing may be freed while the GPU is still reading it.
        impl_->wait_for_own_cmd();
        vkDestroyFence(impl_->handles.device, impl_->own_fence, nullptr);
    }
    impl_->release_slots();
}

void Session::wait_idle() {
    impl_->wait_for_own_cmd();
    for (auto& slot : impl_->slots) impl_->wait_slot(slot);
}

const std::string& Session::status() const { return impl_->status; }

bool Session::building() const { return impl_->building; }
bool Session::failed() const { return impl_->failed; }
bool Session::is_bridged() const { return impl_->bridged; }

// The runtime that is actually running frames, not the one being built: during a
// rebuild the old one is still the one paying for the picture on screen.
float Session::gpu_ms() const {
    return impl_->runtime ? impl_->runtime->average_gpu_ms() : 0.0f;
}

uint64_t Session::create_feature() {
    const uint64_t id = impl_->next_feature++;
    log("[nr] feature #%u created", static_cast<unsigned>(id));
    return id;
}

void Session::release_feature(uint64_t feature) {
    if (!feature) return;
    // Every network this feature may have been evaluated against holds a
    // history for it. None of them is freed here; see Runtime::release_feature.
    for (auto& e : impl_->cache) e.runtime->release_feature(feature);
    log("[nr] feature #%u released", static_cast<unsigned>(feature));
}

ID3D12Resource* Session::run(ID3D12Device* device, ID3D12GraphicsCommandList* list,
                             const EngineResources& resources, const Controls& controls) {
    auto& s = *impl_;
    ID3D12Resource* const colour = resources.colour;
    const D3D12_RESOURCE_STATES colour_state = resources.colour_state;
    if (s.failed || !device || !list || !colour) return nullptr;
    s.status.clear();

    if (!s.ensure_d3d12_device(device)) return nullptr;
    const VkCommandBuffer cmd = command_buffer(list);
    if (!cmd) {
        s.status = "the command list exposes no Vulkan handle";
        s.failed = true;
        log("[nr] %s", s.status.c_str());
        return nullptr;
    }
    const auto colour_vk = colour_handle(device, colour, colour_state);
    if (!colour_vk.usable()) {
        s.status = describe_rejection(colour_vk, "colour");
        return nullptr;
    }
    if (resource_is_typeless(colour_vk) && !s.typeless_noted) {
        s.typeless_noted = true;
        log("[nr] the colour resource is typeless (DXGI format %u); read through vkd3d's "
            "representative format %u - if the picture is wrong, start here",
            unsigned(colour_vk.dxgi), unsigned(colour_vk.format));
    }
    const auto colour_desc = colour->GetDesc();
    if (!s.ensure_output(device, colour_vk.width, colour_vk.height, colour_desc.Format))
        return nullptr;

    // The network's extent is the *render* resolution, which changes with the
    // upscaler's quality preset. Rebuild rather than scale: this pass is only
    // cheap because it runs at exactly the size the game rendered.
    if (!s.ensure_runtime(controls, colour_vk.width, colour_vk.height, colour_vk.format,
                          !resources.colour_encoded &&
                              Session::Impl::is_linear_format(colour_vk.format)))
        return nullptr;

    // The game keeps using its own colour buffer, so the network works on our
    // copy and the upscaler is pointed at that instead.
    ColourFrame frame{};
    frame.image = s.output_vk.image;
    frame.format = colour_vk.format;
    frame.width = colour_vk.width; frame.height = colour_vk.height;
    frame.before = frame.after = VK_IMAGE_LAYOUT_GENERAL;

    EngineFrame engine{};
    engine.colour = frame;
    engine.feature = resources.feature;
    engine.reset = resources.reset;
    engine.motion_scale_x = resources.motion_scale_x;
    engine.motion_scale_y = resources.motion_scale_y;
    engine.depth_inverted = resources.depth_inverted;
    const bool motion_vk = resources.motion &&
        fill(&engine.motion, resource_handle(device, resources.motion, resources.motion_state));
    if (motion_vk) {
        engine.motion_texture_width = engine.motion.width; engine.motion_texture_height = engine.motion.height;
        apply_guide_subrect(&engine.motion, resources.motion_subrect, "motion-vector", &engine.motion_x, &engine.motion_y);
    }
    if (resources.depth &&
        fill(&engine.depth, resource_handle(device, resources.depth, resources.depth_state)))
        apply_guide_subrect(&engine.depth, resources.depth_subrect, "depth", &engine.depth_x, &engine.depth_y);

    try {
        // Seed our output with the game's colour, then enhance it in place.
        VkImageCopy copy{};
        copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.extent = {colour_vk.width, colour_vk.height, 1};
        // The caller told us the D3D12 state; vkd3d told us the layout that
        // implies. Transition from it and put it back, or the game's next use
        // of its own buffer sees a layout it never asked for.
        VkImageMemoryBarrier in{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        in.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        in.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        in.oldLayout = colour_vk.layout; in.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        in.srcQueueFamilyIndex = in.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        in.image = colour_vk.image;
        in.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageMemoryBarrier out = in;
        out.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        out.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        out.oldLayout = VK_IMAGE_LAYOUT_GENERAL; out.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        out.image = s.output_vk.image;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &in);
        vkCmdCopyImage(cmd, colour_vk.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       s.output_vk.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
        std::swap(in.oldLayout, in.newLayout);
        in.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        in.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &in);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &out);

        if (motion_vk) s.runtime->record_engine(cmd, engine, controls);
        else s.runtime->record(cmd, frame, controls);
    } catch (const std::exception& e) {
        s.status = std::string("neural rendering failed: ") + e.what();
        s.failed = true;
        log("[nr] %s", s.status.c_str());
        return nullptr;
    }
    return s.output;
}

bool Session::run_after(ID3D12Device* device, ID3D12GraphicsCommandList* list,
                        ID3D12Resource* output, D3D12_RESOURCE_STATES output_state,
                        const EngineResources& resources, const Controls& controls,
                        ID3D12Resource* input, D3D12_RESOURCE_STATES input_state) {
    auto& s = *impl_;
    if (s.failed || !device || !list || !output) return false;
    s.status.clear();
    if (!s.ensure_d3d12_device(device)) return false;
    const VkCommandBuffer cmd = command_buffer(list);
    if (!cmd) { s.status = "the command list exposes no Vulkan handle"; s.failed = true; return false; }
    const auto target = colour_handle(device, output, output_state);
    if (!target.usable()) { s.status = describe_rejection(target, "output"); return false; }
    if (!s.ensure_runtime(controls, target.width, target.height, target.format,
                          !resources.colour_encoded &&
                              Session::Impl::is_linear_format(target.format))) return false;

    // Format bridge branch: instant execution when resident network format differs from target (e.g. cutscenes)
    if (s.bridged) {
        if (!s.ensure_bridge(target.width, target.height, s.format)) return false;

        transition_layout(cmd, target.image, target.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                          VK_ACCESS_TRANSFER_READ_BIT,
                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        transition_layout(cmd, s.bridge_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          0, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {int32_t(target.width), int32_t(target.height), 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstOffsets[1] = {int32_t(target.width), int32_t(target.height), 1};
        vkCmdBlitImage(cmd, target.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       s.bridge_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

        transition_layout(cmd, s.bridge_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                          VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

        ColourFrame frame{};
        frame.image = s.bridge_image;
        frame.format = s.format;
        frame.width = target.width; frame.height = target.height;
        frame.before = frame.after = VK_IMAGE_LAYOUT_GENERAL;
        frame.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

        EngineFrame engine{};
        engine.colour = frame;
        engine.feature = resources.feature;
        engine.reset = resources.reset;
        engine.motion_scale_x = resources.motion_scale_x;
        engine.motion_scale_y = resources.motion_scale_y;
        engine.depth_inverted = resources.depth_inverted;
        const bool motion_vk = resources.motion &&
            fill(&engine.motion, resource_handle(device, resources.motion, resources.motion_state));
        if (motion_vk) {
            engine.motion_texture_width = engine.motion.width; engine.motion_texture_height = engine.motion.height;
            apply_guide_subrect(&engine.motion, resources.motion_subrect, "motion-vector", &engine.motion_x, &engine.motion_y);
        }
        if (resources.depth &&
            fill(&engine.depth, resource_handle(device, resources.depth, resources.depth_state)))
            apply_guide_subrect(&engine.depth, resources.depth_subrect, "depth", &engine.depth_x, &engine.depth_y);

        const auto everything = [&](VkAccessFlags src_access, VkAccessFlags dst_access) {
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = src_access; mb.dstAccessMask = dst_access;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 0, 1, &mb, 0, nullptr, 0, nullptr);
        };
        everything(VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        try {
            if (motion_vk) s.runtime->record_engine(cmd, engine, controls);
            else s.runtime->record(cmd, frame, controls);
        } catch (const std::exception& e) {
            s.status = std::string("neural rendering failed: ") + e.what();
            s.failed = true;
            log("[nr] %s", s.status.c_str());
            return false;
        }
        everything(VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);

        transition_layout(cmd, s.bridge_image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        transition_layout(cmd, target.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        vkCmdBlitImage(cmd, s.bridge_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

        transition_layout(cmd, target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, target.layout,
                          VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

        list->SetPipelineState(nullptr);
        list->SetComputeRootSignature(nullptr);
        s.forget_bound_pipeline(device, list);
        return true;
    }

    // In place: the upscaler already wrote this and nobody is waiting for a
    // different resource, so there is nothing to rebind.
    ColourFrame frame{};
    frame.image = target.image;
    frame.format = target.format;
    frame.width = target.width; frame.height = target.height;
    frame.before = frame.after = target.layout;
    frame.usage = target.usage;

    EngineFrame engine{};
    engine.colour = frame;
    // A separate input: the network samples it in place and stores its answer
    // into the output, so the output needs no copy of the colour first. Only
    // the engine path with the model applied; anything else is the caller's
    // seed copy and the in-place run.
    if (input) {
        if (!resources.motion || !s.runtime || !s.runtime->takes_target(controls)) return false;
        const auto source = colour_handle(device, input, input_state);
        if (!source.usable() || source.format != target.format || source.width != target.width ||
            source.height != target.height || !(source.usage & VK_IMAGE_USAGE_SAMPLED_BIT))
            return false;
        engine.target = frame;
        engine.colour.image = source.image;
        engine.colour.before = engine.colour.after = source.layout;
        engine.colour.usage = source.usage;
    }
    engine.feature = resources.feature;
    engine.reset = resources.reset;
    engine.motion_scale_x = resources.motion_scale_x;
    engine.motion_scale_y = resources.motion_scale_y;
    engine.depth_inverted = resources.depth_inverted;
    const bool motion_vk = resources.motion &&
        fill(&engine.motion, resource_handle(device, resources.motion, resources.motion_state));
    if (motion_vk) {
        engine.motion_texture_width = engine.motion.width; engine.motion_texture_height = engine.motion.height;
        apply_guide_subrect(&engine.motion, resources.motion_subrect, "motion-vector", &engine.motion_x, &engine.motion_y);
    }
    if (resources.depth &&
        fill(&engine.depth, resource_handle(device, resources.depth, resources.depth_state)))
        apply_guide_subrect(&engine.depth, resources.depth_subrect, "depth", &engine.depth_x, &engine.depth_y);
    if (input && !motion_vk) return false;   // the plain path works in place: the caller seeds
    // Full memory barriers on either side of the network, because the caller's
    // barriers are vkd3d's translation of D3D12 resource states and know
    // nothing about ours. Its "UAV -> SRV" on the output waits for compute
    // shader writes; the network's last write into the output is a
    // vkCmdCopyImage - a TRANSFER-stage write - so a dispatch that follows
    // immediately (OptiScaler's clamp between two NR passes, its resolve) can
    // start before that copy has landed and reads what was there before: the
    // seed copy of the input. One pass survived only because other work
    // happened to sit between the two. The same in reverse for our reads.
    const auto everything = [&](VkAccessFlags src_access, VkAccessFlags dst_access) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = src_access; mb.dstAccessMask = dst_access;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    everything(VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    try {
        if (motion_vk) s.runtime->record_engine(cmd, engine, controls);
        else s.runtime->record(cmd, frame, controls);
    } catch (const std::exception& e) {
        s.status = std::string("neural rendering failed: ") + e.what();
        s.failed = true;
        log("[nr] %s", s.status.c_str());
        return false;
    }
    // The network was recorded as raw Vulkan into vkd3d-proton's command buffer:
    // it bound its own compute pipelines, descriptor sets and push constants.
    // vkd3d does not know, and it de-duplicates state on the D3D12 side -
    // SetPipelineState returns early when `list->state == state`, the root
    // signature likewise - so a caller that sets the SAME pipeline state
    // before and after us (OptiScaler's clamp between two NR passes, then its
    // resolve) never gets vkCmdBindPipeline re-emitted and dispatches our last
    // kernel instead of its own. NVIDIA's DLL never trips this because a cubin
    // launch (vkCmdCuLaunchKernelNVX) touches no Vulkan bind point. Setting the
    // compute bind point's tracked state to null makes the next real set differ
    // and vkd3d re-emits the pipeline, the descriptor offsets and the push
    // constants (SetComputeRootSignature -> invalidate_root_parameters). The
    // graphics bind point was not touched by us and is left alone.
    everything(VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    list->SetPipelineState(nullptr);
    list->SetComputeRootSignature(nullptr);
    // ...and the part those two cannot reach: vkd3d's memory of what is bound in
    // the Vulkan command buffer itself. Without this, the second neural pass in
    // one command list leaves every following caller dispatch running our kernel.
    s.forget_bound_pipeline(device, list);
    return true;
}

bool Session::run_present(ID3D12Device* device, ID3D12GraphicsCommandList* list,
                          ID3D12Resource* back_buffer, unsigned width, unsigned height,
                          const Controls& controls) {
    auto& s = *impl_;
    if (s.failed || !device || !list || !back_buffer) return false;
    s.status.clear();
    if (!s.ensure_d3d12_device(device)) return false;
    const VkCommandBuffer cmd = command_buffer(list);
    if (!cmd) { s.status = "the command list exposes no Vulkan handle"; s.failed = true; return false; }
    const auto target = colour_handle(device, back_buffer, D3D12_RESOURCE_STATE_PRESENT);
    if (!target.usable()) { s.status = describe_rejection(target, "back buffer"); return false; }
    if (!s.ensure_runtime(controls, target.width, target.height, target.format, false)) return false;

    // Here the network works on the game's own buffer: there is nothing to hand
    // anyone afterwards, the frame just has to come out enhanced.
    ColourFrame frame{};
    frame.image = target.image;
    frame.format = target.format;
    frame.width = target.width; frame.height = target.height;
    frame.before = frame.after = target.layout;
    try {
        TemporalFrame temporal{};
        s.runtime->record_temporal(cmd, frame, controls, temporal);
    } catch (const std::exception& e) {
        s.status = std::string("neural rendering failed: ") + e.what();
        s.failed = true;
        log("[nr] %s", s.status.c_str());
        return false;
    }
    return true;
}

bool Session::run_d3d11(const D3D11Frame& frame, const Controls& controls) {
    auto& s = *impl_;
    if (s.failed || !frame.device || !frame.target) return false;
    s.status.clear();

    // DXVK's interop device is the whole of this path: it is where the Vulkan
    // handles, the submission queue and the queue lock all come from.
    if (!s.dxvk) {
        log("[nr] d3d11: asking DXVK for its Vulkan handles");
        s.handles = d3d11_handles(frame.device, &s.dxvk);
        log("[nr] d3d11: instance %p physical %p device %p interop %p",
            static_cast<void*>(s.handles.instance), static_cast<void*>(s.handles.physical),
            static_cast<void*>(s.handles.device), static_cast<void*>(s.dxvk));
        if (!s.handles.valid() || !s.dxvk) {
            if (s.dxvk) { s.dxvk->Release(); s.dxvk = nullptr; }
            s.status = "no Vulkan device underneath; this D3D11 game is not on DXVK";
            s.failed = true;
            log("[nr] %s", s.status.c_str());
            return false;
        }
        if (!d3d11_queue_access(s.dxvk, &s.access)) {
            s.status = "DXVK would not hand out its submission queue";
            s.failed = true;
            log("[nr] %s", s.status.c_str());
            return false;
        }
    }

    // Said once per target, not once per frame: this pair used to be two of the
    // fifteen flushed lines every frame wrote (see first_sight in
    // nr_pe_interop.cpp). What the target is only changes when it changes.
    static VkImage traced_target = VK_NULL_HANDLE;
    const auto target = d3d11_image(frame.target, frame.device);
    if (!target) {
        s.status = "the D3D11 texture exposes no Vulkan image";
        return false;
    }
    if (target.image != traced_target) {
        traced_target = target.image;
        log("[nr] d3d11: queue family %u; target %ux%u format %u layout %u usage 0x%x",
            s.access.family, target.width, target.height, unsigned(target.format),
            unsigned(target.layout), unsigned(target.usage));
    }
    // The pass transfers into and out of whatever image it is given. DXVK tells
    // us what the image may be used for, so this is a check rather than a hope -
    // and a texture without these bits gets declined with a reason instead of
    // producing validation errors inside somebody's game.
    constexpr VkImageUsageFlags required =
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if ((target.usage & required) != required) {
        s.status = "the game's texture cannot be copied to and from; nothing to run on";
        return false;
    }
    if (!s.ensure_runtime(controls, target.width, target.height, target.format,
                          (frame.upscaler_input || frame.linear_hdr) && !frame.estimate_motion &&
                          Session::Impl::is_linear_format(target.format))) return false;
    if (!s.ensure_own_command_buffer()) return false;
    if (!s.wait_for_own_cmd()) return false;

    ColourFrame colour{};
    colour.image = target.image;
    colour.format = target.format;
    colour.width = target.width; colour.height = target.height;
    // Transition from the layout DXVK believes the image is in, and put it back:
    // the game's next use of its own texture must see what it left there.
    colour.before = colour.after = target.layout;

    EngineFrame engine{};
    engine.colour = colour;
    engine.reset = frame.reset;
    engine.motion_scale_x = frame.motion_scale_x;
    engine.motion_scale_y = frame.motion_scale_y;
    engine.depth_inverted = frame.depth_inverted;
    const bool motion = frame.motion && fill(&engine.motion, d3d11_image(frame.motion, frame.device));
    if (frame.depth) fill(&engine.depth, d3d11_image(frame.depth, frame.device));

    try {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandBuffer(s.own_cmd, 0);
        vkBeginCommandBuffer(s.own_cmd, &bi);
        if (motion) s.runtime->record_engine(s.own_cmd, engine, controls);
        else if (frame.estimate_motion) {
            TemporalFrame temporal{}; temporal.reset = frame.reset;
            s.runtime->record_temporal(s.own_cmd, colour, controls, temporal);
        } else {
            s.runtime->record(s.own_cmd, colour, controls);
        }
        vkEndCommandBuffer(s.own_cmd);
    } catch (const std::exception& e) {
        s.status = std::string("neural rendering failed: ") + e.what();
        s.failed = true;
        log("[nr] %s", s.status.c_str());
        return false;
    }

    // Everything the game has recorded goes in front of us - that is what the
    // flush inside the lock is for - and then our submit happens with the queue
    // held, because DXVK shares one VkQueue with everything else in the process.
    VkResult r;
    {
        QueueAccess::Held held(s.access);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &s.own_cmd;
        r = vkQueueSubmit(s.access.queue, 1, &si, s.own_fence);
        if (r == VK_SUCCESS) s.own_cmd_pending = true;
    }
    if (r != VK_SUCCESS) {
        s.status = "the submit on the game's queue failed";
        return false;
    }
    return true;
}

bool Session::run_present_vulkan(const DeviceHandles& handles, VkQueue queue, uint32_t family,
                                 VkImage image, VkFormat format, uint32_t width, uint32_t height,
                                 uint32_t slot_index, uint32_t slot_count,
                                 const VkSemaphore* wait, uint32_t wait_count,
                                 VkSemaphore* signalled, bool temporal,
                                 const Controls& controls) {
    auto& s = *impl_;
    if (s.failed || !image || !queue || !handles.valid() || !signalled) return false;
    s.status.clear();
    s.handles = handles;
    s.access.queue = queue;
    s.access.family = family;
    s.access.lock = nullptr;      // this queue is the game's and we are inside its present
    s.access.unlock = nullptr;

    // A game that alternates between two swapchain extents - 007 First Light
    // cycles 3840x2160 and 3840x2096 at startup and on every mode change - would
    // otherwise rebuild the whole network on each one: four gigabytes
    // reallocated, the weights re-uploaded, and the history thrown away. A
    // discarded history closes the temporal gate for that frame, which changes
    // the picture visibly, and doing it every other frame is exactly what a
    // flicker is.
    //
    // So an extent that does not match the built network does not rebuild it: it
    // is counted, the frame is left alone, and the rebuild happens only once the
    // new extent has actually settled. A real resolution change costs half a
    // second of untouched frames; an alternation costs nothing at all.
    if (s.runtime && (s.width != width || s.height != height) && !s.have_runtime(width, height)) {
        if (s.pending_w != width || s.pending_h != height) {
            s.pending_w = width; s.pending_h = height; s.pending_frames = 0;
        }
        if (++s.pending_frames < 30) return false;
        log("[nr] the swapchain has been %ux%u for %u frames; rebuilding from %ux%u",
            width, height, s.pending_frames, s.width, s.height);
    }
    s.pending_frames = 0;

    if (!s.ensure_slots(slot_count)) return false;
    if (slot_index >= s.slots.size()) return false;
    auto& slot = s.slots[slot_index];

    // Every slot, not just this one.
    //
    // The runtime's contract is explicit that the caller serialises uses of its
    // scratch arena, and the per-image slots defeat exactly that: with three
    // swapchain images, three of our command buffers can be executing at once,
    // all reading and writing one activation arena and one history image. The
    // result is frames that alternate between right and wrong, which is what a
    // flickering picture is.
    //
    // So the slots stay - a binary semaphore still needs one per image, and the
    // present depends on that - but only one pass is ever in flight. It costs a
    // pipeline bubble, which is the price of the arena being shared.
    for (auto& other : s.slots) if (!s.wait_slot(other)) return false;
    // Only after the slot is free: rebuilding the runtime destroys images this
    // slot's command buffer may still be reading.
    if (!s.ensure_runtime(controls, width, height, format, false)) return false;

    // A swapchain image at present time is in PRESENT_SRC. The pass transfers in
    // and out of it, and leaves it exactly as it found it: the present that
    // follows is entitled to the layout it was promised.
    ColourFrame colour{};
    colour.image = image;
    colour.format = format;
    colour.width = width; colour.height = height;
    colour.before = colour.after = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    static bool first = true;
    if (first) log("[nr] first fallback frame: %ux%u format %d on queue family %u, image %u of %u",
                   width, height, int(format), family, slot_index, slot_count);
    try {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandBuffer(slot.cmd, 0);
        vkBeginCommandBuffer(slot.cmd, &bi);
        if (temporal) {
            TemporalFrame frame{};
            s.runtime->record_temporal(slot.cmd, colour, controls, frame);
        } else {
            s.runtime->record(slot.cmd, colour, controls);
        }
        vkEndCommandBuffer(slot.cmd);
    } catch (const std::exception& e) {
        s.status = std::string("neural rendering failed: ") + e.what();
        s.failed = true;
        log("[nr] %s", s.status.c_str());
        return false;
    }

    // One wait stage per semaphore, as the spec requires.
    std::vector<VkPipelineStageFlags> stages(wait_count, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = wait_count;
    si.pWaitSemaphores = wait_count ? wait : nullptr;
    si.pWaitDstStageMask = wait_count ? stages.data() : nullptr;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &slot.cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &slot.done;
    if (vkQueueSubmit(queue, 1, &si, slot.fence) != VK_SUCCESS) {
        s.status = "the submit before present failed";
        return false;
    }
    slot.pending = true;
    *signalled = slot.done;
    if (first) { first = false; log("[nr] first fallback frame submitted"); }
    return true;
}

void Session::set_vulkan_queue(VkQueue queue, uint32_t family) {
    if (!queue) return;
    impl_->access.queue = queue;
    impl_->access.family = family;
    // No lock: this queue belongs to the game and nothing else in this process
    // shares it through an interop layer, so there is no mutex to take. The
    // caller records into the command buffer the upscaler handed over, which the
    // game submits itself; only the one-off weight upload submits here.
    impl_->access.lock = nullptr;
    impl_->access.unlock = nullptr;
}

VkImageView Session::vulkan_output_view() const { return impl_->vk_output_view; }

VkImage Session::run_vulkan(const DeviceHandles& handles, VkCommandBuffer cmd,
                            const VulkanFrame& frame, const Controls& controls) {
    auto& s = *impl_;
    if (s.failed || !cmd || !frame.colour || !handles.valid()) return VK_NULL_HANDLE;
    s.status.clear();
    s.handles = handles;

    // A Vulkan game gives us no D3D12 resource to hand back, so the output is a
    // plain VkImage of our own rather than one borrowed from a D3D12 allocation.
    if (!s.vk_output_valid(frame.width, frame.height, frame.colour_format)) {
        if (!s.ensure_vk_output(frame.width, frame.height, frame.colour_format))
            return VK_NULL_HANDLE;
    }
    if (!s.ensure_runtime(controls, frame.width, frame.height, frame.colour_format,
                          (frame.upscaler_input || frame.linear_hdr) &&
                              Session::Impl::is_linear_format(frame.colour_format)))
        return VK_NULL_HANDLE;

    ColourFrame colour{};
    colour.image = s.vk_output;
    colour.format = frame.colour_format;
    colour.width = frame.width; colour.height = frame.height;
    colour.before = colour.after = VK_IMAGE_LAYOUT_GENERAL;

    EngineFrame engine{};
    engine.colour = colour;
    // The caller's colour read in place and its output written directly (the engine path with
    // the model applied): the copy into vk_output, the copy into the runtime's input and the
    // write-back all go. Anything else keeps the copies below.
    const bool direct = frame.output && frame.motion && s.runtime && s.runtime->takes_target(controls) &&
                        (frame.colour_usage & VK_IMAGE_USAGE_SAMPLED_BIT) &&
                        !(std::getenv("NR_VK_COPY") && std::atoi(std::getenv("NR_VK_COPY")));
    if (direct) {
        engine.colour.image = frame.colour;
        engine.colour.before = engine.colour.after = frame.colour_layout;
        engine.colour.usage = frame.colour_usage;
        engine.target = colour;
        engine.target.image = frame.output;
        engine.target.before = engine.target.after = frame.output_layout;
        engine.target.usage = frame.output_usage;
    }
    engine.feature = frame.feature;
    engine.reset = frame.reset;
    engine.motion_scale_x = frame.motion_scale_x;
    engine.motion_scale_y = frame.motion_scale_y;
    engine.depth_inverted = frame.depth_inverted;
    if (frame.motion) {
        engine.motion.image = frame.motion;
        engine.motion.format = frame.motion_format;
        engine.motion.width = frame.motion_width;
        engine.motion.height = frame.motion_height;
        engine.motion.before = engine.motion.after = frame.motion_layout;
        engine.motion_texture_width = engine.motion.width; engine.motion_texture_height = engine.motion.height;
        apply_guide_subrect(&engine.motion, frame.motion_subrect, "motion-vector", &engine.motion_x, &engine.motion_y);
    }
    if (frame.depth) {
        engine.depth.image = frame.depth;
        engine.depth.format = frame.depth_format;
        engine.depth.width = frame.depth_width;
        engine.depth.height = frame.depth_height;
        engine.depth.before = engine.depth.after = frame.depth_layout;
        apply_guide_subrect(&engine.depth, frame.depth_subrect, "depth", &engine.depth_x, &engine.depth_y);
    }

    if (direct) {
        try {
            s.runtime->record_engine(cmd, engine, controls);
        } catch (const std::exception& e) {
            s.status = std::string("neural rendering failed: ") + e.what();
            s.failed = true;
            log("[nr] %s", s.status.c_str());
            return VK_NULL_HANDLE;
        }
        return frame.output;
    }
    try {
        VkImageCopy copy{};
        copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.extent = {frame.width, frame.height, 1};
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.oldLayout = frame.colour_layout; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = frame.colour;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
        vkCmdCopyImage(cmd, frame.colour, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       s.vk_output, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
        std::swap(b.oldLayout, b.newLayout);
        b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
        if (frame.motion) s.runtime->record_engine(cmd, engine, controls);
        else s.runtime->record(cmd, colour, controls);
    } catch (const std::exception& e) {
        s.status = std::string("neural rendering failed: ") + e.what();
        s.failed = true;
        log("[nr] %s", s.status.c_str());
        return VK_NULL_HANDLE;
    }
    return s.vk_output;
}

}  // namespace nr::pe
