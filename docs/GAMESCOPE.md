# Gamescope Compositor Bridge for DLSS-NR on AMD GPUs

This document describes the Gamescope compositor integration for DLSS 5 Neural Reconstruction (DLSS-NR) on AMD hardware under Linux.

---

## Overview

The Gamescope bridge enables DLSS-NR processing at the compositor level by establishing a high-performance, zero-copy inter-process communication (IPC) channel between Gamescope (the host compositor) and Proton/OptiScaler (the guest runtime running inside Steam Linux Runtime / Wine).

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Gamescope Compositor (Host)                                 │
│                                                             │
│   vulkan_init()                                             │
│     └─► nr::pe::gamescope::Host                             │
│           ├─► Unix Domain Socket: /tmp/dlssnr-gamescope-*.sock│
│           └─► Shares VkInstance, VkDevice, VkQueue          │
│                                                             │
│   vulkan_composite()                                        │
│     ├─► send_frame()    ──► Export VkImage via Opaque FD    │
│     │                       (VK_KHR_external_memory_fd)     │
│     └─► receive_result()◄── Import Result via Opaque FD     │
│                             (VK_KHR_external_semaphore_fd) │
└──────────────────────────┬───▲──────────────────────────────┘
                           │   │  Zero-Copy GPU Texture
            Unix IPC (SCM_RIGHTS) │  (No CPU readback)
                           ▼   │
┌──────────────────────────┴───┴──────────────────────────────┐
│ Proton / OptiScaler Guest (Wine / SteamLinuxRuntime)        │
│                                                             │
│   dlssnr_core.dll / nvngx_dlssnr.dll                        │
│     └─► nr::pe::gamescope::Bridge                           │
│           ├─► Connects to /tmp/dlssnr-gamescope-*.sock      │
│           ├─► Imports Gamescope's source texture            │
│           ├─► Runs Mochizuki Neural Reconstruction Network  │
│           │   (Vulkan Compute / RDNA4 WMMA FP8)             │
│           └─► Exports processed image back to Gamescope     │
└─────────────────────────────────────────────────────────────┘
```

---

## Technical Specifications

1. **Zero-Copy Cross-Process Sharing**:
   - Textures are exported and imported across process boundaries using `VK_KHR_external_memory_fd` (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`).
   - Inter-process synchronization uses `VK_KHR_external_semaphore_fd` (`VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT`).
   - No CPU readback, copy buffers, or shared system RAM copies are used; memory remains on GPU VRAM at all times.

2. **IPC Protocol & Container Compatibility**:
   - IPC messages are exchanged via Unix Domain Sockets using standard POSIX `sendmsg`/`recvmsg` with `SCM_RIGHTS` for descriptor passing.
   - Sockets are placed in `/tmp/dlssnr-gamescope-<uid>-<pid>.sock`.
   - **Crucial design detail**: `/tmp` is bind-mounted directly across container namespaces in Steam Linux Runtime (`pressure-vessel` / bubblewrap), unlike `/run/user/<uid>` which is kept in an isolated private tmpfs mount. This allows transparent host-to-container communication.

3. **Graceful Failure & Safe Fallback**:
   - If `[Gamescope] Enabled=0` or `NR_GAMESCOPE_BRIDGE=0`: Gamescope and OptiScaler behave identically to vanilla execution.
   - If the socket connection fails or times out: Gamescope automatically falls back to rendering the original unmodified frame without dropping or stalling presentation.
   - If the bridge process disconnects or exits: Gamescope recovers instantly without crashing.

---

## Gamescope Source Integration

The host-side implementation requires patching Gamescope's Vulkan rendering pipeline:

1. **Host Instantiation**:
   In `rendervulkan.cpp` (`vulkan_init`), instantiate `nr::pe::gamescope::Host` with Gamescope's `VkInstance`, `VkPhysicalDevice`, `VkDevice`, `VkQueue`, and queue family index.

2. **Per-Frame Interception**:
   In `rendervulkan.cpp` (`vulkan_composite`), call `g_dlssnr_host->send_frame()` with the compositor's source layer texture, wait for the neural reconstruction output via `g_dlssnr_host->receive_result()`, and substitute `frameInfo->layers[0].tex` with the processed texture.

3. **Meson Build**:
   Add `dlssnr/nr_gamescope_host.cpp`, `dlssnr/nr_gamescope_ipc.cpp`, and `dlssnr/nr_gamescope_config.cpp` to Gamescope's `src/meson.build`.

---

## Configuration & Launch

### Configuration (`dlssnr-amd.ini`)

Add the following section to `dlssnr-amd.ini` in the game folder:

```ini
[Gamescope]
Enabled=1
Verbose=0
```

### Steam Launch Options

```bash
WINEDLLOVERRIDES="dxgi=n,b" NR_GAMESCOPE_BRIDGE=1 /path/to/gamescope-dlssnr -W 1920 -H 1080 -w 1920 -h 1080 -f -- %command%
```

> **Note**: Do not use the `-e` (`--steam`) flag when launching in standard desktop mode. The `-e` flag expects Steam Gamepad UI (Big Picture / Steam Deck session) to manage window mapping; on a standard desktop environment (KDE/GNOME), it causes the window to remain unmapped or minimized. Use `-f` (fullscreen) or `-b` (borderless) with explicit `-W` and `-H` dimensions.

---

## Validation & Test Data

Tested and validated on hardware:
- **GPU**: AMD Radeon RX 9060 XT (RADV GFX1200 / RDNA4)
- **OS**: Linux 6.18 x86_64, Mesa 25.1-devel
- **Gamescope Version**: 3.16.25+
- **Proton Runtime**: GE-Proton Latest (with Steam Linux Runtime 4)
- **Target Application**: *CONTROL Resonant* (DirectX 12 / VKD3D + OptiScaler 0.8.4)

### Verification Results
- Host socket created at `/tmp/dlssnr-gamescope-1000-<pid>.sock`.
- Handshake validated successfully across container boundary (`device UUID match: AMD Radeon RX 9060 XT`).
- OptiScaler initialized via `dlssnr_core.dll` (`dxgi.dll` hook).
- Verified concurrent execution with AMD FidelityFX Frame Generation (`FGInput=dlssg`, `FGOutput=fsrfg`).
- Confirmed zero compositor stalls on handshake miss or disconnect.
