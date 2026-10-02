#pragma once
// Gamescope-side host: captures game frames, exports them to the Proton-side
// bridge via dma-buf, receives processed frames, and integrates them back
// into Gamescope's compositor pipeline.
//
// This is a native Linux component. It does NOT use Wine/Windows headers.
// It runs in the Gamescope compositor process, which has its own Vulkan
// device on the same physical GPU the game runs on.
#include "nr_gamescope_ipc.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <memory>

namespace nr::pe::gamescope {

class Host {
public:
    Host(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
         VkQueue queue, uint32_t queue_family,
         const std::string& socket_path = "");
    ~Host();

    Host(const Host&) = delete;
    Host& operator=(const Host&) = delete;

    // Create the listening socket and validate Vulkan capabilities.
    bool start();

    // Accept an incoming Proton-side connection and validate device UUID.
    bool accept_connection();

    // True when a client is connected and the handshake passed.
    bool is_active() const;

    // Capture a game frame from the compositor pipeline.
    // Creates/reuses an exportable image, copies game_frame into it, exports
    // the dma-buf fd and a ready semaphore, and sends them to the bridge.
    bool send_frame(VkCommandBuffer cmd, VkImage game_frame, VkFormat format,
                    uint32_t width, uint32_t height,
                    VkImageLayout current_layout);

    // Receive the NR-processed frame from the bridge.
    // Imports the result dma-buf fd, copies it into target.
    bool receive_result(VkCommandBuffer cmd, VkImage target,
                        VkImageLayout target_layout);

    // Orderly teardown.
    void shutdown();

    const std::string& socket_path() const { return socket_path_; }

private:
    // Extension function pointers.
    struct ExtFunctions {
        PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = nullptr;
        PFN_vkGetSemaphoreFdKHR vkGetSemaphoreFdKHR = nullptr;
        PFN_vkImportSemaphoreFdKHR vkImportSemaphoreFdKHR = nullptr;
    };

    // Double-buffered resource slot.
    struct Slot {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkSemaphore semaphore = VK_NULL_HANDLE;
        VkDeviceSize alloc_size = 0;
        uint32_t mem_type_idx = 0;
    };

    bool create_export_resources(uint32_t w, uint32_t h, VkFormat fmt);
    void destroy_export_resources();
    void destroy_import_slot(Slot& s);
    uint32_t find_device_local_memory(VkMemoryRequirements req) const;

    VkInstance instance_;
    VkPhysicalDevice physical_;
    VkDevice device_;
    VkQueue queue_;
    uint32_t queue_family_;
    std::string socket_path_;

    int listen_fd_ = -1;
    int client_fd_ = -1;
    bool active_ = false;

    uint64_t frame_counter_ = 0;

    // Export resources (double-buffered).
    static constexpr uint32_t kSlots = 2;
    Slot export_slots_[kSlots];
    Slot import_slots_[kSlots];

    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_bufs_[kSlots] = {};

    uint32_t slot_idx_ = 0;
    uint32_t width_ = 0, height_ = 0;
    VkFormat format_ = VK_FORMAT_UNDEFINED;

    ExtFunctions ext_;
    uint8_t device_uuid_[VK_UUID_SIZE] = {};
    uint32_t consecutive_timeouts_ = 0;
    bool in_engine_nr_active_ = false;

public:
    bool is_in_engine_nr_active() const { return in_engine_nr_active_; }
    bool is_in_fallback_mode() const { return consecutive_timeouts_ >= 2; }
    uint32_t consecutive_timeouts() const { return consecutive_timeouts_; }
};

}  // namespace nr::pe::gamescope
