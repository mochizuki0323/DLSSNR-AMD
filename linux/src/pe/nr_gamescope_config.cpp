#include "nr_gamescope_config.hpp"
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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/types.h>

namespace nr::pe::gamescope {
namespace {

using nr::pe::log;

// Strip leading/trailing whitespace; the INI parser this project already uses
// does the same, but this file reads its own section so it needs its own.
std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}

bool to_bool(const std::string& v) {
    const auto t = trim(v);
    return t == "1" || t == "true" || t == "True" || t == "TRUE" ||
           t == "yes" || t == "Yes" || t == "YES";
}

// Read the [Gamescope] section from the INI file. Anything it does not find
// stays at the struct's default, which is "off".
void read_ini_section(BridgeConfig& cfg, const std::string& path) {
    std::ifstream f(path);
    if (!f) return;

    bool in_section = false;
    std::string line;
    while (std::getline(f, line)) {
        const auto trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == ';' || trimmed[0] == '#')
            continue;
        if (trimmed.front() == '[') {
            in_section = trimmed == "[Gamescope]";
            continue;
        }
        if (!in_section) continue;

        const auto eq = trimmed.find('=');
        if (eq == std::string::npos) continue;
        const auto key = trim(trimmed.substr(0, eq));
        const auto val = trim(trimmed.substr(eq + 1));

        if (key == "Enabled") cfg.enabled = to_bool(val);
        else if (key == "SocketPath") cfg.socket_path = val;
        else if (key == "PollTimeoutUs") cfg.poll_timeout_us = static_cast<uint32_t>(std::strtoul(val.c_str(), nullptr, 10));
        else if (key == "Verbose") cfg.verbose = to_bool(val);
    }
}

}  // namespace

BridgeConfig load_bridge_config(const std::string& ini_path) {
    BridgeConfig cfg;

    // INI file first: it is the canonical configuration.
    if (!ini_path.empty()) {
        read_ini_section(cfg, ini_path);
    } else {
        if (access("dlssnr-amd.ini", R_OK) == 0) {
            read_ini_section(cfg, "dlssnr-amd.ini");
        } else if (const char* home = std::getenv("HOME")) {
            std::string user_cfg = std::string(home) + "/.config/dlssnr-amd.ini";
            if (access(user_cfg.c_str(), R_OK) == 0) {
                read_ini_section(cfg, user_cfg);
            }
        }
    }

    // Environment overrides: a developer testing the bridge should not have to
    // edit the INI inside a prefix every time, and an environment variable is
    // what every other NR_ knob uses.
    if (const char* env = std::getenv("NR_GAMESCOPE_BRIDGE"))
        cfg.enabled = env[0] == '1';
    if (const char* env = std::getenv("NR_GAMESCOPE_SOCKET"))
        cfg.socket_path = env;
    if (const char* env = std::getenv("NR_GAMESCOPE_VERBOSE"))
        cfg.verbose = env[0] == '1';

    // Default socket path: /tmp/dlssnr-gamescope-<uid>-<pid>.sock
    // /tmp is bind-mounted across container boundaries (e.g. pressure-vessel/bwrap).
    if (cfg.enabled && cfg.socket_path.empty()) {
        cfg.socket_path = "/tmp/dlssnr-gamescope-" +
#if defined(_WIN32) && !defined(__CYGWIN__)
                          std::to_string(getpid()) + ".sock";
#else
                          std::to_string(getuid()) + "-" +
                          std::to_string(getpid()) + ".sock";
#endif
    }

    if (cfg.enabled) {
        log("[nr] gamescope: bridge enabled, socket %s, poll %u us%s",
            cfg.socket_path.c_str(), cfg.poll_timeout_us,
            cfg.verbose ? ", verbose" : "");
    }

    return cfg;
}

bool bridge_requested() {
    // Quick check without reading the INI: the environment variable alone is
    // enough to activate the bridge, so a caller can bail out early when it is
    // absent and the INI has not been loaded yet.
    if (const char* env = std::getenv("NR_GAMESCOPE_BRIDGE"))
        return env[0] == '1';
    if (access("dlssnr-amd.ini", R_OK) == 0) {
        BridgeConfig cfg;
        read_ini_section(cfg, "dlssnr-amd.ini");
        if (cfg.enabled) return true;
    }
    if (const char* home = std::getenv("HOME")) {
        std::string user_cfg = std::string(home) + "/.config/dlssnr-amd.ini";
        if (access(user_cfg.c_str(), R_OK) == 0) {
            BridgeConfig cfg;
            read_ini_section(cfg, user_cfg);
            if (cfg.enabled) return true;
        }
    }
    return false;
}

}  // namespace nr::pe::gamescope
