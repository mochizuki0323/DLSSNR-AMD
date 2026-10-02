#include "nr_gamescope_bridge.hpp"
#include "nr_pe_log.hpp"

#ifdef _WIN32
namespace nr::pe::gamescope {

Bridge::Bridge(const DeviceHandles& handles, const QueueAccess& access,
               const BridgeConfig& config)
    : handles_(handles), access_(access), config_(config), verbose_(config.verbose) {}

Bridge::~Bridge() {}

bool Bridge::connect() { return false; }
bool Bridge::is_connected() const { return false; }
bool Bridge::is_enabled() const { return false; }
VkImage Bridge::process_frame(Session&, VkCommandBuffer, const Session::VulkanFrame&, const Controls&) {
    return VK_NULL_HANDLE;
}
void Bridge::shutdown() {}

}  // namespace nr::pe::gamescope
#else
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <cstring>
#include <cerrno>
#include <vector>
#include <time.h>

// The bridge lives inside the Proton/Wine process, where nr::pe::log writes
// to dlssnr-amd.log. Every message is prefixed with "[nr] gamescope:" so it
// stands out in a log full of unrelated game activity.

namespace nr::pe::gamescope {
namespace {
using nr::pe::log;

inline uint64_t get_bridge_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
}

// ─── construction / destruction ───────────────────────────────────────────────

Bridge::Bridge(const DeviceHandles& handles, const QueueAccess& access,
               const BridgeConfig& config)
    : handles_(handles), access_(access), config_(config),
      verbose_(config.verbose)
{
}

Bridge::~Bridge() {
    shutdown();
}

// ─── extension loading ───────────────────────────────────────────────────────

bool Bridge::load_extension_functions() {
    ext_.vkGetMemoryFdKHR = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        vkGetDeviceProcAddr(handles_.device, "vkGetMemoryFdKHR"));
    ext_.vkGetSemaphoreFdKHR = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
        vkGetDeviceProcAddr(handles_.device, "vkGetSemaphoreFdKHR"));
    ext_.vkImportSemaphoreFdKHR = reinterpret_cast<PFN_vkImportSemaphoreFdKHR>(
        vkGetDeviceProcAddr(handles_.device, "vkImportSemaphoreFdKHR"));

    if (!ext_.vkGetMemoryFdKHR || !ext_.vkGetSemaphoreFdKHR ||
        !ext_.vkImportSemaphoreFdKHR) {
        log("[nr] gamescope: device is missing VK_KHR_external_memory_fd or "
            "VK_KHR_external_semaphore_fd entry points");
        return false;
    }
    return true;
}

// ─── capability check ─────────────────────────────────────────────────────────

bool Bridge::verify_external_memory_support(VkFormat format) {
    // Ask the driver whether this format can be backed by dma-buf memory that
    // another process can import. If it cannot, the bridge has nothing to work
    // with and falls back silently.
    VkPhysicalDeviceExternalImageFormatInfo ext_info{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO};
    ext_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkPhysicalDeviceImageFormatInfo2 info{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
    info.pNext = &ext_info;
    info.format = format;
    info.type = VK_IMAGE_TYPE_2D;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                 VK_IMAGE_USAGE_SAMPLED_BIT |
                 VK_IMAGE_USAGE_STORAGE_BIT;

    VkExternalImageFormatProperties ext_props{
        VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
    VkImageFormatProperties2 props{
        VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
    props.pNext = &ext_props;

    VkResult r = vkGetPhysicalDeviceImageFormatProperties2(
        handles_.physical, &info, &props);
    if (r != VK_SUCCESS) {
        log("[nr] gamescope: format %d not supported for external memory "
            "(VkResult %d)", int(format), int(r));
        return false;
    }

    const auto& compat = ext_props.externalMemoryProperties.compatibleHandleTypes;
    if (!(compat & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)) {
        log("[nr] gamescope: format %d does not support OPAQUE_FD export",
            int(format));
        return false;
    }
    return true;
}

// ─── connect ──────────────────────────────────────────────────────────────────

bool Bridge::connect() {
    if (connected_) return true;
    if (!config_.enabled) return false;

    if (!load_extension_functions()) {
        extensions_ok_ = false;
        return false;
    }
    extensions_ok_ = true;

    sock_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock_fd_ < 0) {
        log("[nr] gamescope: socket(): %s", strerror(errno));
        return false;
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, config_.socket_path.c_str(),
            sizeof(addr.sun_path) - 1);

    if (::connect(sock_fd_, reinterpret_cast<sockaddr*>(&addr),
                  sizeof(addr)) < 0) {
        // Not an error worth logging at every frame: the host may not be
        // running yet (Gamescope starts the bridge on demand).
        close(sock_fd_);
        sock_fd_ = -1;
        return false;
    }

    // ── handshake: exchange device UUIDs ──
    VkPhysicalDeviceIDProperties id_props{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &id_props;
    vkGetPhysicalDeviceProperties2(handles_.physical, &props2);

    HandshakeMsg hs{};
    memcpy(hs.device_uuid, id_props.deviceUUID, VK_UUID_SIZE);
    hs.driver_version = props2.properties.driverVersion;
    hs.max_width = props2.properties.limits.maxImageDimension2D;
    hs.max_height = props2.properties.limits.maxImageDimension2D;
    hs.external_memory_handle_types =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    hs.in_engine_nr_active = 1; // OptiScaler in-engine capture & NR active

    if (!send_message(sock_fd_, MessageType::MSG_HANDSHAKE,
                      &hs, sizeof(hs), nullptr, 0)) {
        log("[nr] gamescope: failed to send handshake");
        close(sock_fd_); sock_fd_ = -1;
        return false;
    }

    MessageHeader hdr{};
    HandshakeAckMsg ack{};
    uint32_t fd_count = 0;
    if (!recv_message(sock_fd_, &hdr, &ack, sizeof(ack),
                      nullptr, 0, &fd_count) ||
        hdr.type != MessageType::MSG_HANDSHAKE_ACK) {
        log("[nr] gamescope: no valid handshake response");
        close(sock_fd_); sock_fd_ = -1;
        return false;
    }

    if (!ack.accepted) {
        log("[nr] gamescope: handshake rejected: %s", ack.rejection_reason);
        close(sock_fd_); sock_fd_ = -1;
        return false;
    }

    connected_ = true;
    log("[nr] gamescope: connected to bridge, device UUID validated");
    return true;
}

bool Bridge::is_connected() const { return connected_; }

bool Bridge::is_enabled() const {
    return config_.enabled && connected_ && extensions_ok_;
}

// ─── resource helpers ─────────────────────────────────────────────────────────

void Bridge::destroy_slot(Slot& s) {
    if (s.view)      { vkDestroyImageView(handles_.device, s.view, nullptr);   s.view = VK_NULL_HANDLE; }
    if (s.image)     { vkDestroyImage(handles_.device, s.image, nullptr);      s.image = VK_NULL_HANDLE; }
    if (s.memory)    { vkFreeMemory(handles_.device, s.memory, nullptr);       s.memory = VK_NULL_HANDLE; }
    if (s.semaphore) { vkDestroySemaphore(handles_.device, s.semaphore, nullptr); s.semaphore = VK_NULL_HANDLE; }
    s.width = s.height = 0;
    s.format = VK_FORMAT_UNDEFINED;
    s.alloc_size = 0;
}

void Bridge::destroy_all_resources() {
    for (uint32_t i = 0; i < kSlots; ++i) {
        destroy_slot(imported_[i]);
        destroy_slot(exported_[i]);
    }
}

// ─── import a frame from Gamescope ────────────────────────────────────────────

bool Bridge::import_frame(const FrameOfferMsg& offer, int dma_buf_fd,
                          int sem_fd) {
    Slot& s = imported_[slot_idx_];

    // Destroy the old slot if the resolution or format changed.
    if (s.image && (s.width != offer.width || s.height != offer.height ||
                    s.format != offer.format)) {
        destroy_slot(s);
    }

    // ── import the dma-buf fd as VkDeviceMemory ──
    // The fd transfers ownership to the driver; we must not close it after a
    // successful import.
    VkImportMemoryFdInfoKHR import_fd{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
    import_fd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    import_fd.fd = dma_buf_fd;

    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.pNext = &import_fd;
    alloc.allocationSize = offer.allocation_size;
    alloc.memoryTypeIndex = offer.memory_type_index;

    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkResult r = vkAllocateMemory(handles_.device, &alloc, nullptr, &mem);
    if (r != VK_SUCCESS) {
        log("[nr] gamescope: vkAllocateMemory (import) failed: %d", int(r));
        close(dma_buf_fd);
        if (sem_fd >= 0) close(sem_fd);
        return false;
    }
    // fd is now owned by the driver; do not close it.

    // ── create a VkImage backed by the imported memory ──
    VkExternalMemoryImageCreateInfo ext_ci{
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    ext_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.pNext = &ext_ci;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = offer.format;
    ci.extent = {offer.width, offer.height, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = offer.tiling;
    ci.usage = offer.usage_flags;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage img = VK_NULL_HANDLE;
    r = vkCreateImage(handles_.device, &ci, nullptr, &img);
    if (r != VK_SUCCESS) {
        log("[nr] gamescope: vkCreateImage (import) failed: %d", int(r));
        vkFreeMemory(handles_.device, mem, nullptr);
        if (sem_fd >= 0) close(sem_fd);
        return false;
    }

    r = vkBindImageMemory(handles_.device, img, mem, 0);
    if (r != VK_SUCCESS) {
        log("[nr] gamescope: vkBindImageMemory (import) failed: %d", int(r));
        vkDestroyImage(handles_.device, img, nullptr);
        vkFreeMemory(handles_.device, mem, nullptr);
        if (sem_fd >= 0) close(sem_fd);
        return false;
    }

    // ── import the semaphore ──
    VkSemaphore sema = VK_NULL_HANDLE;
    if (sem_fd >= 0) {
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        r = vkCreateSemaphore(handles_.device, &sci, nullptr, &sema);
        if (r == VK_SUCCESS) {
            VkImportSemaphoreFdInfoKHR imp{
                VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR};
            imp.semaphore = sema;
            imp.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
            imp.fd = sem_fd;
            r = ext_.vkImportSemaphoreFdKHR(handles_.device, &imp);
            if (r != VK_SUCCESS) {
                log("[nr] gamescope: vkImportSemaphoreFdKHR failed: %d", int(r));
                vkDestroySemaphore(handles_.device, sema, nullptr);
                sema = VK_NULL_HANDLE;
                close(sem_fd);
            }
            // fd owned by driver on success.
        } else {
            close(sem_fd);
        }
    }

    // ── create view ──
    VkImageView view = VK_NULL_HANDLE;
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = offer.format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(handles_.device, &vi, nullptr, &view);

    // Destroy the previous slot's resources *after* creating the new ones so
    // that a failure above does not leave us with nothing.
    destroy_slot(s);

    s.image = img;
    s.memory = mem;
    s.view = view;
    s.semaphore = sema;
    s.width = offer.width;
    s.height = offer.height;
    s.format = offer.format;
    s.alloc_size = offer.allocation_size;

    return true;
}

// ─── export the result back to Gamescope ──────────────────────────────────────

bool Bridge::export_result(VkImage result, uint32_t w, uint32_t h,
                           VkFormat fmt, uint64_t frame_id) {
    Slot& s = exported_[slot_idx_];

    // (Re)create the export image if the dimensions changed.
    if (!s.image || s.width != w || s.height != h || s.format != fmt) {
        destroy_slot(s);

        VkExternalMemoryImageCreateInfo ext_ci{
            VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
        ext_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.pNext = &ext_ci;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = fmt;
        ci.extent = {w, h, 1};
        ci.mipLevels = ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                   VK_IMAGE_USAGE_SAMPLED_BIT;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkResult r = vkCreateImage(handles_.device, &ci, nullptr, &s.image);
        if (r != VK_SUCCESS) {
            log("[nr] gamescope: vkCreateImage (export) failed: %d", int(r));
            return false;
        }

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(handles_.device, s.image, &req);

        VkExportMemoryAllocateInfo exp{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
        exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        // Find a device-local memory type.
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(handles_.physical, &mem_props);
        uint32_t type_idx = mem_props.memoryTypeCount;
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
            if ((req.memoryTypeBits & (1u << i)) &&
                (mem_props.memoryTypes[i].propertyFlags &
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                type_idx = i;
                break;
            }
        }
        if (type_idx == mem_props.memoryTypeCount) {
            log("[nr] gamescope: no device-local memory type for export image");
            vkDestroyImage(handles_.device, s.image, nullptr);
            s.image = VK_NULL_HANDLE;
            return false;
        }

        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.pNext = &exp;
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = type_idx;

        r = vkAllocateMemory(handles_.device, &alloc, nullptr, &s.memory);
        if (r != VK_SUCCESS || vkBindImageMemory(handles_.device, s.image, s.memory, 0) != VK_SUCCESS) {
            log("[nr] gamescope: export image allocation failed: %d", int(r));
            destroy_slot(s);
            return false;
        }

        s.width = w; s.height = h; s.format = fmt; s.alloc_size = req.size;

        // Create the export semaphore.
        VkExportSemaphoreCreateInfo exp_sem{
            VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
        exp_sem.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        sci.pNext = &exp_sem;
        vkCreateSemaphore(handles_.device, &sci, nullptr, &s.semaphore);

        log("[nr] gamescope: export resources created %ux%u format %d",
            w, h, int(fmt));
    }

    // ── export the memory fd ──
    VkMemoryGetFdInfoKHR get_fd{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    get_fd.memory = s.memory;
    get_fd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int mem_fd = -1;
    VkResult r = ext_.vkGetMemoryFdKHR(handles_.device, &get_fd, &mem_fd);
    if (r != VK_SUCCESS) {
        log("[nr] gamescope: vkGetMemoryFdKHR failed: %d", int(r));
        return false;
    }

    // ── export the semaphore fd ──
    VkSemaphoreGetFdInfoKHR get_sem{
        VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    get_sem.semaphore = s.semaphore;
    get_sem.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    int sem_fd = -1;
    ext_.vkGetSemaphoreFdKHR(handles_.device, &get_sem, &sem_fd);

    // ── send MSG_FRAME_RESULT ──
    FrameResultMsg result_msg{};
    result_msg.frame_id = frame_id;
    result_msg.width = w;
    result_msg.height = h;
    result_msg.format = fmt;
    result_msg.layout = VK_IMAGE_LAYOUT_GENERAL;
    result_msg.allocation_size = s.alloc_size;

    // Find the memory type index used for the export allocation.
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(handles_.device, s.image, &req);
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(handles_.physical, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags &
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            result_msg.memory_type_index = i;
            break;
        }
    }

    int fds[2] = {mem_fd, sem_fd >= 0 ? sem_fd : -1};
    uint32_t fd_count = sem_fd >= 0 ? 2 : 1;
    bool ok = send_message(sock_fd_, MessageType::MSG_FRAME_RESULT,
                           &result_msg, sizeof(result_msg), fds, fd_count);
    // The fds are consumed by sendmsg (duplicated to the receiver); close ours.
    if (mem_fd >= 0) close(mem_fd);
    if (sem_fd >= 0) close(sem_fd);

    if (!ok) {
        log("[nr] gamescope: failed to send MSG_FRAME_RESULT");
        return false;
    }
    return true;
}

// ─── the frame loop ──────────────────────────────────────────────────────────

VkImage Bridge::process_frame(Session& session, VkCommandBuffer cmd,
                              const Session::VulkanFrame& original_frame,
                              const Controls& controls) {
    if (!connected_) return VK_NULL_HANDLE;

    // ── poll for a frame offer ──
    pollfd pfd{};
    pfd.fd = sock_fd_;
    pfd.events = POLLIN;
    int pr = poll(&pfd, 1, static_cast<int>(config_.poll_timeout_us / 1000));
    if (pr <= 0) {
        // No frame available: let the caller use the normal path.
        return VK_NULL_HANDLE;
    }

    // ── receive MSG_FRAME_OFFER ──
    MessageHeader hdr{};
    FrameOfferMsg offer{};
    int fds[kMaxFds];
    uint32_t fd_count = 0;

    if (!recv_message(sock_fd_, &hdr, &offer, sizeof(offer),
                      fds, kMaxFds, &fd_count) ||
        hdr.type != MessageType::MSG_FRAME_OFFER || fd_count < 1) {
        // Protocol error or wrong message type: drop and fall back.
        if (hdr.type == MessageType::MSG_SHUTDOWN) {
            log("[nr] gamescope: host sent shutdown");
            shutdown();
        }
        return VK_NULL_HANDLE;
    }

    int dma_buf_fd = fds[0];
    int sem_fd = fd_count >= 2 ? fds[1] : -1;

    if (verbose_) {
        log("[nr] gamescope: Gamescope → Proton import (frame %llu, %ux%u)",
            (unsigned long long)offer.frame_id, offer.width, offer.height);
    }

    // ── import the frame ──
    if (!import_frame(offer, dma_buf_fd, sem_fd)) {
        log("[nr] gamescope: import failed for frame %llu",
            (unsigned long long)offer.frame_id);
        return VK_NULL_HANDLE;
    }

    // ── verify format support on first use or change ──
    if (offer.format != current_fmt_ &&
        !verify_external_memory_support(offer.format)) {
        log("[nr] gamescope: format %d unsupported for external memory; "
            "falling back", int(offer.format));
        connected_ = false;
        return VK_NULL_HANDLE;
    }
    current_w_ = offer.width;
    current_h_ = offer.height;
    current_fmt_ = offer.format;

    // ── fill a VulkanFrame pointing to the imported image ──
    // The imported image now holds the game's frame; we tell the session to
    // run the network on it exactly as the existing Vulkan path does.
    Session::VulkanFrame bridge_frame{};
    bridge_frame.feature = original_frame.feature;
    bridge_frame.colour = imported_[slot_idx_].image;
    bridge_frame.colour_layout = static_cast<VkImageLayout>(offer.layout);
    bridge_frame.colour_format = offer.format;
    bridge_frame.width = offer.width;
    bridge_frame.height = offer.height;
    // Carry over engine guides from the original frame if available.
    bridge_frame.motion = original_frame.motion;
    bridge_frame.motion_layout = original_frame.motion_layout;
    bridge_frame.motion_format = original_frame.motion_format;
    bridge_frame.motion_width = original_frame.motion_width;
    bridge_frame.motion_height = original_frame.motion_height;
    bridge_frame.depth = original_frame.depth;
    bridge_frame.depth_layout = original_frame.depth_layout;
    bridge_frame.depth_format = original_frame.depth_format;
    bridge_frame.depth_width = original_frame.depth_width;
    bridge_frame.depth_height = original_frame.depth_height;
    bridge_frame.depth_subrect = original_frame.depth_subrect;
    bridge_frame.motion_subrect = original_frame.motion_subrect;
    bridge_frame.depth_inverted = original_frame.depth_inverted;
    bridge_frame.motion_scale_x = original_frame.motion_scale_x;
    bridge_frame.motion_scale_y = original_frame.motion_scale_y;
    bridge_frame.reset = original_frame.reset;
    bridge_frame.upscaler_input = original_frame.upscaler_input;
    bridge_frame.linear_hdr = original_frame.linear_hdr;

    // ── run the Mochizuki NR network ──
    DeviceHandles h = handles_;
    uint64_t t_fg_start = get_bridge_time_ns();
    VkImage result = session.run_vulkan(h, cmd, bridge_frame, controls);
    uint64_t t_fg_end = get_bridge_time_ns();
    uint64_t fg_dur_ns = t_fg_end - t_fg_start;

    log("[Telemetry Timing] Frame Generation: FrameID: %llu, Duration: %.3f ms (%llu ns), start: %llu ns, end: %llu ns, status: %s",
        (unsigned long long)offer.frame_id,
        fg_dur_ns / 1'000'000.0, (unsigned long long)fg_dur_ns,
        (unsigned long long)t_fg_start, (unsigned long long)t_fg_end,
        result ? "SUCCESS" : "DECLINED/BUILDING");

    if (verbose_) {
        log("[nr] gamescope: Mochizuki run_vulkan%s",
            result ? "" : " (declined - building or failed)");
    }

    if (!result) {
        // The network is still building or failed; the caller uses the
        // original frame and nothing is sent back.
        return VK_NULL_HANDLE;
    }

    // ── copy the result into the export slot and send it back ──
    // The result is the session's vk_output, in GENERAL layout.  We blit it
    // into our exportable image so Gamescope can import it.
    Slot& exp = exported_[slot_idx_];
    if (!exp.image || exp.width != offer.width || exp.height != offer.height ||
        exp.format != offer.format) {
        // The export_result method handles (re)creation internally.
    }

    // Barrier: result -> TRANSFER_SRC, export -> TRANSFER_DST
    VkImageMemoryBarrier barriers[2]{};
    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[0].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barriers[0].srcQueueFamilyIndex = barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].image = result;
    barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    // We need the export image to exist first; ensure it here.
    if (!export_result(result, offer.width, offer.height, offer.format,
                       offer.frame_id)) {
        log("[nr] gamescope: export_result failed");
        return VK_NULL_HANDLE;
    }

    if (verbose_) {
        log("[nr] gamescope: NR output exported (frame %llu)",
            (unsigned long long)offer.frame_id);
    }

    slot_idx_ = (slot_idx_ + 1) % kSlots;
    frame_counter_++;
    return result;
}

// ─── resize ──────────────────────────────────────────────────────────────────

void Bridge::notify_resize(uint32_t w, uint32_t h, VkFormat fmt) {
    if (w == current_w_ && h == current_h_ && fmt == current_fmt_) return;

    vkDeviceWaitIdle(handles_.device);
    destroy_all_resources();

    current_w_ = w;
    current_h_ = h;
    current_fmt_ = fmt;

    if (connected_) {
        ResizeMsg msg{};
        msg.width = w;
        msg.height = h;
        msg.format = fmt;
        send_message(sock_fd_, MessageType::MSG_RESIZE,
                     &msg, sizeof(msg), nullptr, 0);
        log("[nr] gamescope: resize to %ux%u format %d", w, h, int(fmt));
    }
}

// ─── shutdown ─────────────────────────────────────────────────────────────────

void Bridge::shutdown() {
    if (!connected_ && sock_fd_ < 0) return;

    if (connected_) {
        ShutdownMsg msg{};
        msg.reason = 0;
        send_message(sock_fd_, MessageType::MSG_SHUTDOWN,
                     &msg, sizeof(msg), nullptr, 0);
    }

    vkDeviceWaitIdle(handles_.device);
    destroy_all_resources();

    if (sock_fd_ >= 0) {
        close(sock_fd_);
        sock_fd_ = -1;
    }
    connected_ = false;
    log("[nr] gamescope: bridge shut down");
}

}  // namespace nr::pe::gamescope
#endif
