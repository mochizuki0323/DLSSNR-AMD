#pragma once
// The network on a Vulkan device of its own, beside a game that runs on the native D3D12/D3D11
// runtime of the AMD Windows driver.
//
// Under vkd3d-proton/DXVK the game's device is a Vulkan device and the network records straight into
// it. Natively it is not, so the two meet the way two APIs on one GPU are meant to: images the D3D
// side creates as shared resources and the Vulkan side imports (VK_KHR_external_memory_win32), and a
// shared D3D fence the Vulkan side imports as a timeline semaphore (VK_KHR_external_semaphore_win32).
// Nothing crosses the CPU and nothing waits on it: the D3D queue signals, our queue waits on the GPU,
// runs the network, signals, and the D3D queue waits on the GPU before it reads the result.
//
// One device per process, on the adapter the game renders with (matched by LUID), with exactly the
// features the network is measured with (nrvk::Context::create) plus the external-memory pair and
// timeline semaphores.
#include "nr_pe_interop.hpp"

#include <windows.h>
#include <d3d10.h>
#include <d3d11.h>
#include <d3d12.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace nr::pe::bridge {

// An image both APIs see. `d3d` is the D3D side (an ID3D12Resource or an ID3D11Texture2D) and is
// owned here; `image` is the same memory as a VkImage on our device. The Vulkan side keeps it in
// VK_IMAGE_LAYOUT_GENERAL for its whole life (moved there once, right after the import); the D3D12
// side leaves it in D3D12_RESOURCE_STATE_COMMON at every hand-over.
struct Image {
    IUnknown* d3d{};
    DXGI_FORMAT dxgi{DXGI_FORMAT_UNKNOWN};   // the format the D3D side was created with
    VkImage image{};
    VkDeviceMemory memory{};
    VkFormat format{VK_FORMAT_UNDEFINED};
    VkImageUsageFlags usage{};
    uint32_t width{}, height{};
    bool storage{};   // created with unordered access / STORAGE (the format allows it)
    const void* device{};   // the D3D device that created the D3D side (its resources are not another's)
    explicit operator bool() const { return image != VK_NULL_HANDLE; }
    bool matches(uint32_t w, uint32_t h, DXGI_FORMAT f, const void* d) const {
        return image && width == w && height == h && dxgi == f && device == d;
    }
};

// The format a shared copy of a resource is created with: the typed member of a typeless family (the
// copy into it is a CopyResource, which only needs the family to match), the resource's own format
// otherwise. DXGI_FORMAT_UNKNOWN for what the bridge does not carry.
DXGI_FORMAT shared_format(DXGI_FORMAT format);

// A recording of the network for one frame. Made with Device::begin, finished with Device::end, then
// either submitted (run_between / submit) or discarded.
struct Job;

class Device {
  public:
    // The process's device on the adapter with this LUID, created on the first call for that adapter.
    // Null with `why` set when the GPU or driver cannot carry the bridge; that answer is kept and
    // repeated for that adapter.
    static Device* get(const LUID& luid, std::string* why);
    const LUID& luid() const { return luid_; }

    DeviceHandles handles() const { return handles_; }
    VkQueue queue() const { return queue_; }
    uint32_t family() const { return family_; }
    // Taken around every submit on our queue (the network's weight upload included).
    void lock() { queue_mutex_.lock(); }
    void unlock() { queue_mutex_.unlock(); }

    bool create_d3d12(ID3D12Device* device, uint32_t width, uint32_t height, DXGI_FORMAT format, Image* out,
                      std::string* why);
    bool create_d3d11(ID3D11Device* device, uint32_t width, uint32_t height, DXGI_FORMAT format, Image* out,
                      std::string* why);
    // D3D10 has no fences: its shared textures carry a keyed mutex (legacy shared handle), and the
    // hand-over is the mutex - the D3D10 side releases key 1, our submit acquires 1 and releases 2,
    // the D3D10 side acquires 2. Needs VK_KHR_win32_keyed_mutex.
    bool keyed_mutex() const { return keyed_mutex_; }
    bool create_d3d10(ID3D10Device* device, uint32_t width, uint32_t height, DXGI_FORMAT format, Image* out,
                      std::string* why);
    // Submits `job` acquiring every image in `images` at `acquire_key` and releasing it at `release_key`.
    bool submit_keyed(Job* job, uint64_t generation, const std::vector<const Image*>& images, uint64_t acquire_key,
                      uint64_t release_key);
    // Frees both sides. The caller has made sure neither API is still using it.
    void destroy(Image& image);
    // Frees the Vulkan side now (no job of ours can use it) and releases the D3D side once the hand-over
    // queue is past everything given to it so far: a game command list recorded with our copies may
    // still run after the session that made them is gone.
    void bury(Image& image);

    // Recording. `begin` hands out a command buffer in the recording state (waiting for an old one only
    // when every one is still on the GPU) and holds the pool's lock until `end` or `abort`; `end` closes it, with the acquire of `images` from the D3D
    // side recorded before the caller's work and their release after it.
    //
    // `owner` is what the recording reads and writes besides the images (the network, nr::Runtime): it
    // must outlive every job of it that can still run (busy). A job is one use: `generation` tells a
    // later run_between/discard that holds an old pointer that the job has been handed out again since.
    Job* begin(std::string* why, const void* owner);
    VkCommandBuffer command_buffer(Job* job) const;
    uint64_t generation(const Job* job) const;
    // Called (once, on whatever thread) when a recorded job is never run as recorded: discarded,
    // refused, submitted out of recording order. The owner's history then cannot be trusted.
    void on_drop(Job* job, std::function<void()> f);
    void acquire(Job* job, const std::vector<const Image*>& images);
    bool end(Job* job, std::string* why);
    // Never submitted (the frame's command list was thrown away): the buffer goes back to the pool.
    // A stale generation does nothing.
    void discard(Job* job, uint64_t generation);
    // Give up a recording between begin and end (the caller's recording threw): as discard, and the
    // recording lock taken by begin is released.
    void abort(Job* job);
    // Every job of `owner` that has not been submitted is given up (the owner is going away); on_drop
    // is not called.
    void cancel(const void* owner);
    // Can a job still use `owner` / `image`: recorded and not yet submitted, or submitted and not finished.
    bool busy(const void* owner);
    bool busy(VkImage image);

    // The GPU-side hand-over on a D3D12 queue: the queue signals the shared fence after everything it
    // has been given so far, our queue waits for that, runs `job`, signals, and the D3D12 queue waits
    // for that before anything given to it afterwards. False when the job did not run (stale, failed
    // submit): the D3D12 queue then signals the value itself, so it never waits for a signal that will
    // not come, and whatever it reads afterwards is what it put there. False with nothing queued at all
    // when the D3D12 side refused a signal.
    //
    // One D3D12 queue per process carries the hand-overs (the first one asked): frames on another
    // queue would share the shared images and our one queue with it without an order between them,
    // so there the job is not run (false).
    bool run_between(ID3D12CommandQueue* queue, Job* job, uint64_t generation);
    // The same on D3D11's immediate context (ID3D11DeviceContext4 and a shared ID3D11Fence).
    bool run_between(ID3D11DeviceContext* context, Job* job, uint64_t generation);

    // Where the hand-over queue is now (taken after a run_between), and whether that queue has since
    // got past everything it was given with that frame, the copy back out of the shared images
    // included. A D3D12 resource may only be released then: D3D12 does not keep it alive for its GPU work.
    uint64_t hand_over_mark();
    bool hand_over_passed(uint64_t mark);

    // Have all our submits finished? Waits at most a second.
    bool wait_idle();
    // The device is lost (a D3D side reported removal): nothing more is submitted.
    void lost(const char* why);
    // Tests only (bridge_probe exercises D3D12, D3D11 and D3D10 in one process): the next run_between
    // takes its queue or context as the hand-over queue. Nothing may be in flight.
    void forget_hand_over_queue();

    ~Device();

  private:
    Device() = default;
    bool create(const LUID& luid, std::string* why);
    Job* begin_locked(std::string* why, const void* owner);
    void drop_locked(Job* job, std::vector<std::function<void()>>& calls);
    bool done(Job* job);   // jobs_mutex_ held: submitted and its fence signalled
    LUID luid_{};
    struct Buried { IUnknown* d3d; uint64_t mark; };
    std::vector<Buried> buried_;   // under sync_mutex_
    void sweep_buried_locked(uint64_t completed);
    IUnknown* hand_over_queue_{};   // the one D3D12 queue (or D3D11 context) the hand-overs run on
    bool lost_{};
    bool import_image(HANDLE handle, VkExternalMemoryHandleTypeFlagBits type, uint32_t width, uint32_t height,
                      VkFormat format, bool storage, Image* out, std::string* why);
    bool external_ok(VkFormat format, VkImageUsageFlags usage, VkExternalMemoryHandleTypeFlagBits type,
                     bool* dedicated);
    bool submit(Job* job, uint64_t generation, VkSemaphore wait, uint64_t wait_value, VkSemaphore signal,
                uint64_t signal_value);
    VkSemaphore import_fence(HANDLE handle, std::string* why);
    bool initial_layout(VkImage image, std::string* why);

    VkInstance instance_{};
    VkPhysicalDevice physical_{};
    VkDevice device_{};
    VkQueue queue_{};
    uint32_t family_{};
    DeviceHandles handles_{};
    VkPhysicalDeviceMemoryProperties memory_{};
    std::mutex queue_mutex_;
    PFN_vkGetMemoryWin32HandlePropertiesKHR get_handle_properties_{};
    PFN_vkImportSemaphoreWin32HandleKHR import_semaphore_{};
    bool keyed_mutex_{};

    std::mutex jobs_mutex_;
    // The command pool is externally synchronised: held from begin() to end()/abort(), across the
    // caller's recording, and by anything else that allocates from the pool.
    std::mutex record_mutex_;
    VkCommandPool pool_{};
    std::vector<std::unique_ptr<Job>> jobs_;

    // One shared fence per D3D queue (or D3D11 context), imported once. Values only grow: every signal
    // on it, from either side, is allocated here under `sync_mutex_`.
    struct Timeline {
        IUnknown* owner{};   // the ID3D12CommandQueue or ID3D11DeviceContext, held
        IUnknown* fence{};   // ID3D12Fence or ID3D11Fence
        VkSemaphore semaphore{};
        uint64_t value{};
    };
    std::mutex sync_mutex_;
    std::vector<Timeline> timelines_;
    Timeline* timeline_d3d12(ID3D12CommandQueue* queue);
    Timeline* timeline_d3d11(ID3D11DeviceContext* context);
};

// For hooks that see every Vulkan device in the process (ReShade injected as dxgi.dll hooks vulkan-1.dll
// too, and so does our own device watcher): true while this thread is creating the bridge's device,
// and for the bridge's device itself. Such a device is ours and is never a game's.
bool creating_device();
bool is_own_device(VkDevice device);

// Small D3D-side helpers shared by the routes.

// A depth buffer as one float per texel in an R32_FLOAT image, by a compute pass (D3D12 and D3D11
// cannot copy a depth format into a colour one). The view format is chosen from the resource's own
// format; false when the resource cannot be read as a shader resource.
class DepthCopyD3D12 {
  public:
    ~DepthCopyD3D12();
    // Records the pass into `list`; `src` must be in a shader-readable state, `dst` (R32_FLOAT, created
    // with unordered access) in UNORDERED_ACCESS. Changes the list's descriptor heaps, compute root
    // signature and pipeline.
    bool record(ID3D12Device* device, ID3D12GraphicsCommandList* list, ID3D12Resource* src,
                ID3D12Resource* dst, std::string* why);
    // The pipeline exists (built on first use); asked before anything is recorded.
    bool ready(ID3D12Device* device, std::string* why) { return ensure(device, why); }
  private:
    bool ensure(ID3D12Device* device, std::string* why);
    ID3D12RootSignature* root_{};
    ID3D12PipelineState* pso_{};
    ID3D12DescriptorHeap* heap_{};
    UINT increment_{};
    uint32_t next_{};
    bool failed_{};
};

class DepthCopyD3D11 {
  public:
    ~DepthCopyD3D11();
    // Runs the pass on the immediate context and puts the compute state it touches back.
    bool run(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Resource* src, ID3D11Texture2D* dst,
             std::string* why);
  private:
    ID3D11ComputeShader* cs_{};
    bool failed_{};
    // Views are cached for the last source/destination pair.
    ID3D11Resource* src_{};
    ID3D11Texture2D* dst_{};
    ID3D11ShaderResourceView* srv_{};
    ID3D11UnorderedAccessView* uav_{};
    void drop_views();
};

// The shader-readable view format of a depth (or depth-like) resource format; UNKNOWN when none.
DXGI_FORMAT depth_view_format(DXGI_FORMAT format);

}  // namespace nr::pe::bridge
