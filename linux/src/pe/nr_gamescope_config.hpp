#pragma once
// Configuration for the Gamescope ↔ OptiScaler interprocess bridge.
//
// The bridge is opt-in: it does nothing unless the user asks for it through the
// environment (NR_GAMESCOPE_BRIDGE=1) or the INI file ([Gamescope] Enabled=1).
// When it is off the existing paths run exactly as they always have, and this
// file is the only place that knows whether it should be on.
//
// The socket path defaults to /run/user/<uid>/dlssnr-gamescope-<pid>.sock,
// which XDG_RUNTIME_DIR already owns and cleans up. A user who needs a fixed
// path for testing sets SocketPath in the INI or NR_GAMESCOPE_SOCKET in the
// environment.
#include <cstdint>
#include <string>

namespace nr::pe::gamescope {

struct BridgeConfig {
    // Master switch. False means the bridge never connects and every frame
    // takes the existing OptiScaler → Mochizuki path inside the Proton
    // process, which is the default and the only behaviour that shipped.
    bool enabled = false;

    // Unix domain socket path. Empty selects the default, which includes the
    // PID so that several games can run under Gamescope at once without
    // colliding.
    std::string socket_path;

    // How long the Proton side waits for a frame offer from Gamescope before
    // giving up and letting the frame pass through unenhanced. In
    // microseconds; the default is one millisecond, which is short enough to
    // stay inside a frame budget and long enough for a socket round trip.
    uint32_t poll_timeout_us = 1000;

    // Log every frame's round trip instead of only the first and last.
    // Noisy, but the only way to confirm the bridge is actually running on
    // every frame rather than falling through.
    bool verbose = false;
};

// Read from dlssnr-amd.ini's [Gamescope] section and the environment.
// `ini_path` is the path to the INI file; the caller already knows it.
BridgeConfig load_bridge_config(const std::string& ini_path);

// True when either the environment or the INI says the bridge should be on.
// This is a quick check the caller can use before doing any work at all.
bool bridge_requested();

}  // namespace nr::pe::gamescope
