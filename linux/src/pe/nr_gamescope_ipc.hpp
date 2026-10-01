#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>

namespace nr {
namespace pe {
namespace gamescope {

// Defines versions and bounds for the protocol between Gamescope and the DLSS-NR runtime.
constexpr uint32_t kProtocolVersion = 1;
constexpr uint32_t kMagic = 0x4E524753; // 'NRGS'
constexpr uint32_t kMaxFormats = 8;
constexpr uint32_t kMaxFds = 4;
constexpr uint32_t kMaxReasonLen = 64;

// IPC message categories guiding the lifecycle of the shared Vulkan resources.
enum class MessageType : uint32_t {
    MSG_HANDSHAKE = 0,
    MSG_HANDSHAKE_ACK = 1,
    MSG_FRAME_OFFER = 2,
    MSG_FRAME_RESULT = 3,
    MSG_RESIZE = 4,
    MSG_SHUTDOWN = 5
};

// Wire format envelope ensuring parsing alignment and payload bounds checking.
struct alignas(8) MessageHeader {
    uint32_t magic;
    uint32_t version;
    MessageType type;
    uint32_t payload_size;
    uint32_t fd_count;
};

// Establishes hardware locality; we must ensure producer and consumer share the same GPU node.
struct alignas(8) HandshakeMsg {
    uint8_t device_uuid[VK_UUID_SIZE];
    uint32_t driver_version;
    uint32_t format_count;
    VkFormat supported_formats[kMaxFormats];
    uint32_t max_width;
    uint32_t max_height;
    VkExternalMemoryHandleTypeFlags external_memory_handle_types;
    uint32_t padding; // Ensure 8-byte alignment overall
};

// Provides explicit rejection paths avoiding silent stalls on capability mismatches.
struct alignas(8) HandshakeAckMsg {
    bool accepted;
    char rejection_reason[kMaxReasonLen];
};

// Describes incoming memory shapes so the consumer can wrap the dma-buf fd into a VkImage.
struct alignas(8) FrameOfferMsg {
    uint64_t frame_id;
    VkDeviceSize allocation_size;
    uint32_t width;
    uint32_t height;
    VkFormat format;
    VkImageLayout layout;
    uint32_t memory_type_index;
    VkImageTiling tiling;
    VkImageUsageFlags usage_flags;
    uint32_t semaphore_handle_type;
};

// Finalizes the loop, handing ownership of the processed dma-buf back to the compositor.
struct alignas(8) FrameResultMsg {
    uint64_t frame_id;
    VkDeviceSize allocation_size;
    uint32_t width;
    uint32_t height;
    VkFormat format;
    VkImageLayout layout;
    uint32_t memory_type_index;
    uint32_t padding; // Ensure 8-byte alignment overall
};

// Reallocation signal; resource churn is heavy so explicit tracking avoids leaks.
struct alignas(8) ResizeMsg {
    uint32_t width;
    uint32_t height;
    VkFormat format;
};

// Disconnect signal avoiding timeout-based heuristics.
struct alignas(8) ShutdownMsg {
    uint32_t reason;
};

static_assert(sizeof(MessageHeader) == 24, "Invalid MessageHeader size");
static_assert(sizeof(HandshakeMsg) == 72, "Invalid HandshakeMsg size");
static_assert(sizeof(HandshakeAckMsg) == 72, "Invalid HandshakeAckMsg size");
static_assert(sizeof(FrameOfferMsg) == 48, "Invalid FrameOfferMsg size");
static_assert(sizeof(FrameResultMsg) == 40, "Invalid FrameResultMsg size");
static_assert(sizeof(ResizeMsg) == 16, "Invalid ResizeMsg size");
static_assert(sizeof(ShutdownMsg) == 8, "Invalid ShutdownMsg size");

// Pack payloads into msghdr struct and manage SCM_RIGHTS for zero-copy file descriptor sharing.
bool send_message(int sockfd, MessageType type, const void* payload, uint32_t payload_size, const int* fds, uint32_t fd_count);

// Extract control messages and handle partial reads transparently.
bool recv_message(int sockfd, MessageHeader* header, void* payload, uint32_t max_payload, int* fds, uint32_t max_fds, uint32_t* fd_count_out);

// Dynamic socket path binding to user session and caller namespace.
std::string default_socket_path();

} // namespace gamescope
} // namespace pe
} // namespace nr
