#pragma once
// Proton-side bridge to Gamescope's compositor: imports frames from the host
// process, runs them through Mochizuki NR, and exports the results back.
//
// The bridge operates entirely on the game's Vulkan device, which under Proton
// is the same physical GPU that Gamescope composites on. The two processes
// share VkImage allocations through dma-buf file descriptors (VK_KHR_external_
// memory_fd), and synchronise GPU work through external semaphores
// (VK_KHR_external_semaphore_fd). No pixel ever touches the CPU.
//
// When the bridge is disabled, not connected, or encounters any error, the
// caller falls back to the existing in-process NR path and the game renders
// exactly as it always has.
#include "nr_pe_session.hpp"
#include "nr_pe_interop.hpp"
#include "nr_gamescope_ipc.hpp"
#include "nr_gamescope_config.hpp"
#include "nr_runtime.hpp"
#include <vulkan/vulkan.h>
#include <string>
#include <memory>

namespace nr::pe::gamescope {

class Bridge {
public:
    // `handles` is the game's resolved Vulkan device (from vkd3d/DXVK interop).
    // `access` provides the submittable queue and its lock.
    // `config` determines whether the bridge is enabled and where the socket is.
    Bridge(const DeviceHandles& handles, const QueueAccess& access,
           const BridgeConfig& config);
    ~Bridge();

    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;

    // Attempt to connect to the Gamescope host socket and validate that both
    // processes share the same physical GPU. Returns false on any failure;
    // the caller must treat that as "use the normal path" rather than as a
    // hard error.
    bool connect();

    // True after a successful connect() and until shutdown() or a fatal error.
    bool is_connected() const;

    // True when the bridge is configured on, connected, and the driver
    // supports every extension the interprocess path requires.
    bool is_enabled() const;

    // The frame loop. If the bridge is active, this:
    //   1. receives a frame offer from Gamescope (dma-buf fd + metadata)
    //   2. imports the fd as a VkImage on this device
    //   3. calls session.run_vulkan() to run the NR network
    //   4. exports the result as a dma-buf fd
    //   5. sends the result back to Gamescope
    //   6. returns the processed VkImage (in GENERAL layout)
    //
    // On any failure, returns VK_NULL_HANDLE and the caller must fall back
    // to the existing path. The function never throws.
    VkImage process_frame(Session& session, VkCommandBuffer cmd,
                          const Session::VulkanFrame& original_frame,
                          const Controls& controls);

    // Notify that the game's resolution or format changed. The bridge tears
    // down its current resources and sends a RESIZE message to the host.
    void notify_resize(uint32_t w, uint32_t h, VkFormat fmt);

    // Orderly teardown: waits for GPU idle, frees all Vulkan resources, sends
    // a SHUTDOWN message, and closes the socket.
    void shutdown();

private:
    // Extension function pointers, loaded once from the device.
    struct ExtFunctions {
        PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = nullptr;
        PFN_vkGetSemaphoreFdKHR vkGetSemaphoreFdKHR = nullptr;
        PFN_vkImportSemaphoreFdKHR vkImportSemaphoreFdKHR = nullptr;
    };

    // One slot in the double-buffered resource ring.
    struct Slot {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkSemaphore semaphore = VK_NULL_HANDLE;
        uint32_t width = 0, height = 0;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkDeviceSize alloc_size = 0;
    };

    bool load_extension_functions();
    bool verify_external_memory_support(VkFormat format);
    bool import_frame(const FrameOfferMsg& offer, int dma_buf_fd, int sem_fd);
    bool export_result(VkImage result, uint32_t w, uint32_t h, VkFormat fmt,
                       uint64_t frame_id);
    void destroy_slot(Slot& slot);
    void destroy_all_resources();

    DeviceHandles handles_;
    QueueAccess access_;
    BridgeConfig config_;
    ExtFunctions ext_;

    int sock_fd_ = -1;
    bool connected_ = false;
    bool extensions_ok_ = false;
    bool verbose_ = false;

    uint64_t frame_counter_ = 0;

    // Double-buffered: [0] and [1] alternate to avoid stalls.
    static constexpr uint32_t kSlots = 2;
    Slot imported_[kSlots];    // frames received from Gamescope
    Slot exported_[kSlots];    // results sent back to Gamescope
    uint32_t slot_idx_ = 0;

    uint32_t current_w_ = 0, current_h_ = 0;
    VkFormat current_fmt_ = VK_FORMAT_UNDEFINED;
};

}  // namespace nr::pe::gamescope
