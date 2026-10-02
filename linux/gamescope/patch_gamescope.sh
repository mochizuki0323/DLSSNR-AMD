#!/usr/bin/env bash
# patch_gamescope.sh: Applies the DLSS-NR host bridge integration to Gamescope source.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path-to-gamescope-source-tree>" >&2
    exit 1
fi

GS_DIR="$(realpath "$1")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PE_SRC="$(realpath "$SCRIPT_DIR/../src/pe")"

if [ ! -f "$GS_DIR/src/rendervulkan.cpp" ]; then
    echo "Error: $GS_DIR does not appear to be a valid Gamescope source tree." >&2
    exit 1
fi

echo "==> Copying DLSS-NR host sources to $GS_DIR/src/dlssnr/..."
mkdir -p "$GS_DIR/src/dlssnr"
cp "$PE_SRC/nr_gamescope_host.hpp" "$GS_DIR/src/dlssnr/"
cp "$PE_SRC/nr_gamescope_host.cpp" "$GS_DIR/src/dlssnr/"
cp "$PE_SRC/nr_gamescope_ipc.hpp" "$GS_DIR/src/dlssnr/"
cp "$PE_SRC/nr_gamescope_ipc.cpp" "$GS_DIR/src/dlssnr/"
cp "$PE_SRC/nr_gamescope_config.hpp" "$GS_DIR/src/dlssnr/"
cp "$PE_SRC/nr_gamescope_config.cpp" "$GS_DIR/src/dlssnr/"

echo "==> Applying gamescope-dlssnr.patch..."
patch -d "$GS_DIR" -p1 < "$SCRIPT_DIR/gamescope-dlssnr.patch"

echo "==> Successfully patched Gamescope for DLSS-NR bridge!"
echo "Now build Gamescope normally with:"
echo "  meson setup build $GS_DIR"
echo "  ninja -C build src/gamescope"
