#include "nr_gamescope_ipc.hpp"
#if __has_include("nr_pe_log.hpp")
#include "nr_pe_log.hpp"
#else
#include <cstdio>
#include <cstdarg>
namespace nr::pe {
inline void log(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}
}
#endif

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <string>

namespace nr {
namespace pe {
namespace gamescope {

bool send_message(int sockfd, MessageType type, const void* payload, uint32_t payload_size, const int* fds, uint32_t fd_count) {
    if (fd_count > kMaxFds) {
        log("[nr] gamescope: send_message failed, too many fds (%u)", fd_count);
        return false;
    }

    MessageHeader header{};
    header.magic = kMagic;
    header.version = kProtocolVersion;
    header.type = type;
    header.payload_size = payload_size;
    header.fd_count = fd_count;

    iovec iov[2];
    iov[0].iov_base = &header;
    iov[0].iov_len = sizeof(header);
    
    int iov_count = 1;
    if (payload_size > 0 && payload != nullptr) {
        iov[1].iov_base = const_cast<void*>(payload);
        iov[1].iov_len = payload_size;
        iov_count = 2;
    }

    msghdr msg{};
    msg.msg_iov = iov;
    msg.msg_iovlen = iov_count;

    // SCM_RIGHTS buffer is required if passing descriptors.
    union {
        char buf[CMSG_SPACE(kMaxFds * sizeof(int))];
        struct cmsghdr align;
    } u{};

    if (fd_count > 0 && fds != nullptr) {
        msg.msg_control = u.buf;
        msg.msg_controllen = CMSG_SPACE(fd_count * sizeof(int));

        cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN(fd_count * sizeof(int));

        std::memcpy(CMSG_DATA(cmsg), fds, fd_count * sizeof(int));
    }

    // Use MSG_NOSIGNAL to avoid SIGPIPE if the connection drops.
    ssize_t bytes_sent = sendmsg(sockfd, &msg, MSG_NOSIGNAL);
    if (bytes_sent < 0) {
        // Log explicitly since network paths are flaky by nature.
        log("[nr] gamescope: sendmsg failed: %s (errno %d)", std::strerror(errno), errno);
        return false;
    }

    return true;
}

bool recv_message(int sockfd, MessageHeader* header, void* payload, uint32_t max_payload, int* fds, uint32_t max_fds, uint32_t* fd_count_out) {
    if (!header) return false;

    iovec iov[2];
    iov[0].iov_base = header;
    iov[0].iov_len = sizeof(MessageHeader);
    
    int iov_count = 1;
    if (max_payload > 0 && payload != nullptr) {
        iov[1].iov_base = payload;
        iov[1].iov_len = max_payload;
        iov_count = 2;
    }

    msghdr msg{};
    msg.msg_iov = iov;
    msg.msg_iovlen = iov_count;

    union {
        char buf[CMSG_SPACE(kMaxFds * sizeof(int))];
        struct cmsghdr align;
    } u{};

    msg.msg_control = u.buf;
    msg.msg_controllen = sizeof(u.buf);

    ssize_t bytes_recv = recvmsg(sockfd, &msg, 0);
    if (bytes_recv <= 0) {
        if (bytes_recv < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            // Non-blocking wait, valid state.
            return false;
        }
        if (bytes_recv < 0) {
            log("[nr] gamescope: recvmsg failed: %s (errno %d)", std::strerror(errno), errno);
        }
        return false;
    }

    if (static_cast<size_t>(bytes_recv) < sizeof(MessageHeader)) {
        log("[nr] gamescope: recvmsg truncated header (got %zd bytes)", bytes_recv);
        return false;
    }

    if (header->magic != kMagic || header->version != kProtocolVersion) {
        log("[nr] gamescope: protocol mismatch (magic: %x, version: %u)", header->magic, header->version);
        return false;
    }

    if (fd_count_out) {
        *fd_count_out = 0;
    }

    // Extract sideband descriptors robustly, handling CMSG padding.
    for (cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
            size_t data_len = cmsg->cmsg_len - CMSG_LEN(0);
            uint32_t count = data_len / sizeof(int);
            
            if (count > max_fds) {
                log("[nr] gamescope: too many descriptors received (%u > %u)", count, max_fds);
                // Protect from fd leaks by closing excess fds.
                const int* received_fds = reinterpret_cast<const int*>(CMSG_DATA(cmsg));
                for (uint32_t i = 0; i < count; ++i) {
                    close(received_fds[i]);
                }
                return false;
            }

            if (fds && fd_count_out) {
                std::memcpy(fds, CMSG_DATA(cmsg), count * sizeof(int));
                *fd_count_out = count;
            }
        }
    }

    return true;
}

std::string default_socket_path() {
    // Shared /tmp path across host and container (pressure-vessel / bwrap namespaces).
    uid_t uid = getuid();
    pid_t pid = getpid();
    return "/tmp/dlssnr-gamescope-" + std::to_string(uid) + "-" + std::to_string(pid) + ".sock";
}


} // namespace gamescope
} // namespace pe
} // namespace nr
