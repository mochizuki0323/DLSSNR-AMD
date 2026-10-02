#include "nr_gamescope_host.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <cstring>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <vector>
#include <time.h>

// The host lives in Gamescope's native Linux process. It has no access to
// nr::pe::log (which writes through Wine's file API), so it logs to stderr
// where Gamescope's own messages go.

namespace {
inline uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void host_log(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}
}  // namespace

namespace nr::pe::gamescope {

// ─── construction / destruction ──────────────────────────────────────────────

Host::Host(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
           VkQueue queue, uint32_t queue_family, const std::string& socket_path)
    : instance_(instance), physical_(physical), device_(device),
      queue_(queue), queue_family_(queue_family),
      socket_path_(socket_path.empty() ? default_socket_path() : socket_path)
{
}

Host::~Host() {
    shutdown();
}

// ─── memory type helper ─────────────────────────────────────────────────────

uint32_t Host::find_device_local_memory(VkMemoryRequirements req) const {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(physical_, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags &
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            return i;
    }
    return props.memoryTypeCount;  // sentinel: not found
}

// ─── start ──────────────────────────────────────────────────────────────────

bool Host::start() {
    // Load extension function pointers.
    ext_.vkGetMemoryFdKHR = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR"));
    ext_.vkGetSemaphoreFdKHR = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
        vkGetDeviceProcAddr(device_, "vkGetSemaphoreFdKHR"));
    ext_.vkImportSemaphoreFdKHR = reinterpret_cast<PFN_vkImportSemaphoreFdKHR>(
        vkGetDeviceProcAddr(device_, "vkImportSemaphoreFdKHR"));

    if (!ext_.vkGetMemoryFdKHR || !ext_.vkGetSemaphoreFdKHR) {
        host_log("[nr] gamescope: VK_KHR_external_memory_fd or "
                 "VK_KHR_external_semaphore_fd not available");
        return false;
    }

    // Query our device UUID for the handshake.
    VkPhysicalDeviceIDProperties id_props{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &id_props;
    vkGetPhysicalDeviceProperties2(physical_, &props2);
    memcpy(device_uuid_, id_props.deviceUUID, VK_UUID_SIZE);

    // Create the listening socket.
    listen_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        host_log("[nr] gamescope: socket(): %s", strerror(errno));
        return false;
    }

    // Remove any stale socket from a previous crash.
    unlink(socket_path_.c_str());

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        host_log("[nr] gamescope: bind(%s): %s",
                 socket_path_.c_str(), strerror(errno));
        close(listen_fd_); listen_fd_ = -1;
        return false;
    }

    if (listen(listen_fd_, 1) < 0) {
        host_log("[nr] gamescope: listen(): %s", strerror(errno));
        close(listen_fd_); listen_fd_ = -1;
        return false;
    }

    // Create command pool for our own submits.
    VkCommandPoolCreateInfo pool_ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_ci.queueFamilyIndex = queue_family_;
    pool_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device_, &pool_ci, nullptr, &cmd_pool_) != VK_SUCCESS) {
        host_log("[nr] gamescope: failed to create command pool");
        close(listen_fd_); listen_fd_ = -1;
        return false;
    }

    VkCommandBufferAllocateInfo alloc_ci{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc_ci.commandPool = cmd_pool_;
    alloc_ci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_ci.commandBufferCount = kSlots;
    vkAllocateCommandBuffers(device_, &alloc_ci, cmd_bufs_);

    host_log("[nr] gamescope: host listening on %s", socket_path_.c_str());
    return true;
}

// ─── accept connection ──────────────────────────────────────────────────────

bool Host::accept_connection() {
    if (listen_fd_ < 0) return false;

    // Non-blocking accept so we do not stall the compositor.
    int flags = fcntl(listen_fd_, F_GETFL, 0);
    fcntl(listen_fd_, F_SETFL, flags | O_NONBLOCK);

    client_fd_ = accept(listen_fd_, nullptr, nullptr);
    fcntl(listen_fd_, F_SETFL, flags);  // restore

    if (client_fd_ < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK)
            host_log("[nr] gamescope: accept(): %s", strerror(errno));
        return false;
    }

    // Receive the client's handshake.
    MessageHeader hdr{};
    HandshakeMsg hs{};
    uint32_t fd_count = 0;
    if (!recv_message(client_fd_, &hdr, &hs, sizeof(hs),
                      nullptr, 0, &fd_count) ||
        hdr.type != MessageType::MSG_HANDSHAKE) {
        host_log("[nr] gamescope: failed to receive handshake");
        close(client_fd_); client_fd_ = -1;
        return false;
    }

    // Validate: same physical GPU.
    bool uuid_match = memcmp(device_uuid_, hs.device_uuid, VK_UUID_SIZE) == 0;

    HandshakeAckMsg ack{};
    ack.accepted = uuid_match;
    if (!uuid_match) {
        strncpy(ack.rejection_reason, "device UUID mismatch",
                sizeof(ack.rejection_reason) - 1);
        host_log("[nr] gamescope: rejecting client: device UUID mismatch");
    }

    send_message(client_fd_, MessageType::MSG_HANDSHAKE_ACK,
                 &ack, sizeof(ack), nullptr, 0);

    if (!uuid_match) {
        close(client_fd_); client_fd_ = -1;
        return false;
    }

    active_ = true;
    in_engine_nr_active_ = (hs.in_engine_nr_active != 0);
    host_log("[nr] gamescope: client connected, same GPU validated (in_engine_nr_active=%d)",
             int(in_engine_nr_active_));
    return true;
}

bool Host::is_active() const { return active_; }

// ─── export resource management ─────────────────────────────────────────────

bool Host::create_export_resources(uint32_t w, uint32_t h, VkFormat fmt) {
    destroy_export_resources();
    width_ = w; height_ = h; format_ = fmt;

    for (uint32_t i = 0; i < kSlots; ++i) {
        Slot& s = export_slots_[i];

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

        VkResult r = vkCreateImage(device_, &ci, nullptr, &s.image);
        if (r != VK_SUCCESS) {
            host_log("[nr] gamescope: vkCreateImage (export) failed: %d", int(r));
            return false;
        }

        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device_, s.image, &req);

        uint32_t type_idx = find_device_local_memory(req);
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(physical_, &mem_props);
        if (type_idx == mem_props.memoryTypeCount) {
            host_log("[nr] gamescope: no device-local memory type");
            return false;
        }

        VkExportMemoryAllocateInfo exp{
            VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
        exp.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.pNext = &exp;
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = type_idx;

        r = vkAllocateMemory(device_, &alloc, nullptr, &s.memory);
        if (r != VK_SUCCESS) {
            host_log("[nr] gamescope: vkAllocateMemory (export) failed: %d", int(r));
            return false;
        }
        vkBindImageMemory(device_, s.image, s.memory, 0);

        s.alloc_size = req.size;
        s.mem_type_idx = type_idx;

        // Create exportable semaphore.
        VkExportSemaphoreCreateInfo exp_sem{
            VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
        exp_sem.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        sci.pNext = &exp_sem;
        vkCreateSemaphore(device_, &sci, nullptr, &s.semaphore);
    }

    host_log("[nr] gamescope: export resources created %ux%u format %u",
             w, h, unsigned(fmt));
    return true;
}

void Host::destroy_export_resources() {
    for (uint32_t i = 0; i < kSlots; ++i) {
        Slot& s = export_slots_[i];
        if (s.semaphore) { vkDestroySemaphore(device_, s.semaphore, nullptr); s.semaphore = VK_NULL_HANDLE; }
        if (s.image)     { vkDestroyImage(device_, s.image, nullptr);         s.image = VK_NULL_HANDLE; }
        if (s.memory)    { vkFreeMemory(device_, s.memory, nullptr);          s.memory = VK_NULL_HANDLE; }
        s.alloc_size = 0;
    }
}

void Host::destroy_import_slot(Slot& s) {
    if (s.semaphore) { vkDestroySemaphore(device_, s.semaphore, nullptr); s.semaphore = VK_NULL_HANDLE; }
    if (s.image)     { vkDestroyImage(device_, s.image, nullptr);         s.image = VK_NULL_HANDLE; }
    if (s.memory)    { vkFreeMemory(device_, s.memory, nullptr);          s.memory = VK_NULL_HANDLE; }
    s.alloc_size = 0;
}

// ─── send_frame ─────────────────────────────────────────────────────────────

bool Host::send_frame(VkCommandBuffer cmd, VkImage game_frame, VkFormat format,
                      uint32_t width, uint32_t height,
                      VkImageLayout current_layout) {
    if (!active_) return false;

    // (Re)create export resources on resolution/format change.
    if (width != width_ || height != height_ || format != format_) {
        vkDeviceWaitIdle(device_);
        if (!create_export_resources(width, height, format))
            return false;
    }

    uint32_t idx = slot_idx_;
    Slot& s = export_slots_[idx];

    VkCommandBuffer submit_cmd = cmd;
    bool own_cmd = false;
    if (submit_cmd == VK_NULL_HANDLE) {
        submit_cmd = cmd_bufs_[idx];
        own_cmd = true;
        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandBuffer(submit_cmd, 0);
        vkBeginCommandBuffer(submit_cmd, &begin_info);
    }

    // ── barrier: game frame → TRANSFER_SRC, export → TRANSFER_DST ──
    VkImageMemoryBarrier barriers[2]{};
    // Source frame.
    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[0].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barriers[0].oldLayout = current_layout;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barriers[0].srcQueueFamilyIndex = barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].image = game_frame;
    barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    // Export destination.
    barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[1].srcAccessMask = 0;
    barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barriers[1].srcQueueFamilyIndex = barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].image = s.image;
    barriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    vkCmdPipelineBarrier(submit_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 2, barriers);

    // ── copy ──
    VkImageCopy copy{};
    copy.srcSubresource = copy.dstSubresource =
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.extent = {width, height, 1};
    vkCmdCopyImage(submit_cmd, game_frame, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    // ── restore game frame layout ──
    barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barriers[0].newLayout = current_layout;
    vkCmdPipelineBarrier(submit_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &barriers[0]);

    if (own_cmd) {
        vkEndCommandBuffer(submit_cmd);
        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &submit_cmd;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &s.semaphore;
        vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE);
    }

    // ── export memory fd ──
    VkMemoryGetFdInfoKHR get_fd{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    get_fd.memory = s.memory;
    get_fd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int mem_fd = -1;
    VkResult r = ext_.vkGetMemoryFdKHR(device_, &get_fd, &mem_fd);
    if (r != VK_SUCCESS) {
        host_log("[nr] gamescope: vkGetMemoryFdKHR failed: %d", int(r));
        return false;
    }

    // ── export semaphore fd ──
    VkSemaphoreGetFdInfoKHR get_sem{
        VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    get_sem.semaphore = s.semaphore;
    get_sem.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    int sem_fd = -1;
    if (ext_.vkGetSemaphoreFdKHR)
        ext_.vkGetSemaphoreFdKHR(device_, &get_sem, &sem_fd);

    // ── send MSG_FRAME_OFFER ──
    FrameOfferMsg offer{};
    offer.frame_id = frame_counter_;
    offer.width = width;
    offer.height = height;
    offer.format = format;
    offer.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    offer.allocation_size = s.alloc_size;
    offer.memory_type_index = s.mem_type_idx;
    offer.tiling = VK_IMAGE_TILING_OPTIMAL;
    offer.usage_flags = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        VK_IMAGE_USAGE_SAMPLED_BIT;
    offer.semaphore_handle_type =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fds[2] = {mem_fd, sem_fd >= 0 ? sem_fd : -1};
    uint32_t fd_count = sem_fd >= 0 ? 2 : 1;
    bool ok = send_message(client_fd_, MessageType::MSG_FRAME_OFFER,
                           &offer, sizeof(offer), fds, fd_count);

    // The fds are duplicated by the kernel on sendmsg; close our copies.
    if (mem_fd >= 0) close(mem_fd);
    if (sem_fd >= 0) close(sem_fd);

    if (!ok) {
        host_log("[nr] gamescope: failed to send frame offer, client disconnected");
        if (client_fd_ >= 0) { close(client_fd_); client_fd_ = -1; }
        active_ = false;
        return false;
    }

    host_log("[nr] gamescope: Gamescope frame exported (frame %llu, %ux%u)",
             (unsigned long long)frame_counter_, width, height);

    slot_idx_ = (slot_idx_ + 1) % kSlots;
    frame_counter_++;
    return true;
}

// ─── receive_result ─────────────────────────────────────────────────────────

bool Host::receive_result(VkCommandBuffer cmd, VkImage target,
                          VkImageLayout target_layout) {
    if (!active_) return false;

    // ── receive MSG_FRAME_RESULT ──
    MessageHeader hdr{};
    FrameResultMsg result_msg{};
    int fds[kMaxFds];
    uint32_t fd_count = 0;

    // Block briefly waiting for the result. Use 1ms timeout during video/fallback mode to avoid FPS drops.
    uint64_t t_ipc_poll_start = get_time_ns();
    pollfd pfd{};
    pfd.fd = client_fd_;
    pfd.events = POLLIN;
    int timeout_ms = (consecutive_timeouts_ >= 2) ? 1 : 16;
    int poll_res = poll(&pfd, 1, timeout_ms);
    uint64_t t_ipc_poll_end = get_time_ns();
    uint64_t ipc_sync_dur_ns = t_ipc_poll_end - t_ipc_poll_start;

    if (poll_res <= 0) {
        consecutive_timeouts_++;
        host_log("[Telemetry Timing] IPC Synchronization TIMEOUT/FAIL (count: %u, timeout: %d ms): Duration: %.3f ms (%llu ns), start: %llu ns, end: %llu ns, poll_res: %d (errno: %d)",
                 consecutive_timeouts_, timeout_ms,
                 ipc_sync_dur_ns / 1'000'000.0, (unsigned long long)ipc_sync_dur_ns,
                 (unsigned long long)t_ipc_poll_start, (unsigned long long)t_ipc_poll_end,
                 poll_res, errno);
        return false;
    }
    consecutive_timeouts_ = 0;

    if (pfd.revents & (POLLHUP | POLLERR)) {
        host_log("[nr] gamescope: client disconnected (hup/err)");
        if (client_fd_ >= 0) { close(client_fd_); client_fd_ = -1; }
        active_ = false;
        return false;
    }

    if (!recv_message(client_fd_, &hdr, &result_msg, sizeof(result_msg),
                      fds, kMaxFds, &fd_count) ||
        hdr.type != MessageType::MSG_FRAME_RESULT || fd_count < 1) {
        host_log("[nr] gamescope: client disconnected or invalid message");
        if (client_fd_ >= 0) { close(client_fd_); client_fd_ = -1; }
        active_ = false;
        return false;
    }

    host_log("[Telemetry Timing] IPC Synchronization: FrameID: %llu, Duration: %.3f ms (%llu ns), start: %llu ns, end: %llu ns, poll_res: %d",
             (unsigned long long)result_msg.frame_id,
             ipc_sync_dur_ns / 1'000'000.0, (unsigned long long)ipc_sync_dur_ns,
             (unsigned long long)t_ipc_poll_start, (unsigned long long)t_ipc_poll_end,
             poll_res);

    int mem_fd = fds[0];
    int sem_fd = fd_count >= 2 ? fds[1] : -1;

    host_log("[nr] gamescope: Proton → Gamescope import (frame %llu)",
             (unsigned long long)result_msg.frame_id);

    // ── import the result memory ──
    uint32_t imp_idx = result_msg.frame_id % kSlots;
    Slot& is = import_slots_[imp_idx];
    destroy_import_slot(is);

    VkImportMemoryFdInfoKHR import_fd{
        VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
    import_fd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    import_fd.fd = mem_fd;

    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.pNext = &import_fd;
    alloc.allocationSize = result_msg.allocation_size;
    alloc.memoryTypeIndex = result_msg.memory_type_index;

    VkResult r = vkAllocateMemory(device_, &alloc, nullptr, &is.memory);
    if (r != VK_SUCCESS) {
        host_log("[nr] gamescope: import alloc failed: %d", int(r));
        close(mem_fd);
        if (sem_fd >= 0) close(sem_fd);
        return false;
    }
    // fd now owned by driver.

    // ── create VkImage backed by imported memory ──
    VkExternalMemoryImageCreateInfo ext_ci{
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    ext_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.pNext = &ext_ci;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = result_msg.format;
    ci.extent = {result_msg.width, result_msg.height, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    r = vkCreateImage(device_, &ci, nullptr, &is.image);
    if (r != VK_SUCCESS) {
        host_log("[nr] gamescope: import create image failed: %d", int(r));
        vkFreeMemory(device_, is.memory, nullptr);
        is.memory = VK_NULL_HANDLE;
        if (sem_fd >= 0) close(sem_fd);
        return false;
    }
    vkBindImageMemory(device_, is.image, is.memory, 0);

    // ── import semaphore ──
    if (sem_fd >= 0 && ext_.vkImportSemaphoreFdKHR) {
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vkCreateSemaphore(device_, &sci, nullptr, &is.semaphore);
        VkImportSemaphoreFdInfoKHR imp{
            VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR};
        imp.semaphore = is.semaphore;
        imp.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        imp.fd = sem_fd;
        if (ext_.vkImportSemaphoreFdKHR(device_, &imp) != VK_SUCCESS) {
            close(sem_fd);
        }
        // fd owned by driver on success.
    } else if (sem_fd >= 0) {
        close(sem_fd);
    }

    // ── record copy ──
    bool own_cmd = (cmd == VK_NULL_HANDLE);
    VkCommandBuffer submit_cmd = cmd;
    if (own_cmd) {
        submit_cmd = cmd_bufs_[imp_idx];
        vkResetCommandBuffer(submit_cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(submit_cmd, &bi);
    }

    // ── copy the imported result into the compositor's target ──
    VkImageMemoryBarrier barriers[2]{};
    // Imported result → TRANSFER_SRC
    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[0].srcAccessMask = 0;
    barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barriers[0].srcQueueFamilyIndex = barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].image = is.image;
    barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    // Target → TRANSFER_DST
    barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[1].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barriers[1].srcQueueFamilyIndex = barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].image = target;
    barriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    vkCmdPipelineBarrier(submit_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 2, barriers);

    VkImageCopy copy{};
    copy.srcSubresource = copy.dstSubresource =
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.extent = {result_msg.width, result_msg.height, 1};
    vkCmdCopyImage(submit_cmd, is.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    // Restore target layout.
    barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
    barriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barriers[1].newLayout = target_layout;
    vkCmdPipelineBarrier(submit_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &barriers[1]);

    if (own_cmd) {
        vkEndCommandBuffer(submit_cmd);
        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &submit_cmd;
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        if (is.semaphore != VK_NULL_HANDLE) {
            submit_info.waitSemaphoreCount = 1;
            submit_info.pWaitSemaphores = &is.semaphore;
            submit_info.pWaitDstStageMask = &wait_stage;
        }
        vkQueueSubmit(queue_, 1, &submit_info, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue_);
    }

    host_log("[nr] gamescope: Gamescope presenting processed frame");
    return true;
}

// ─── shutdown ───────────────────────────────────────────────────────────────

void Host::shutdown() {
    if (active_) {
        ShutdownMsg msg{};
        msg.reason = 0;
        send_message(client_fd_, MessageType::MSG_SHUTDOWN,
                     &msg, sizeof(msg), nullptr, 0);
    }

    vkDeviceWaitIdle(device_);

    destroy_export_resources();
    for (uint32_t i = 0; i < kSlots; ++i)
        destroy_import_slot(import_slots_[i]);

    if (cmd_pool_) {
        vkDestroyCommandPool(device_, cmd_pool_, nullptr);
        cmd_pool_ = VK_NULL_HANDLE;
    }

    if (client_fd_ >= 0) { close(client_fd_); client_fd_ = -1; }
    if (listen_fd_ >= 0) {
        close(listen_fd_);
        listen_fd_ = -1;
        unlink(socket_path_.c_str());
    }
    active_ = false;
    host_log("[nr] gamescope: host shut down");
}

}  // namespace nr::pe::gamescope
