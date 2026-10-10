#!/usr/bin/env bash
# Download the third-party pieces the build uses, each pinned to the exact version the
# packages were built and tested with, into toolchain/ and artifacts/ref/ (neither is in git).
#
#   bash fetch_deps.sh             what the Linux package needs
#   bash fetch_deps.sh --windows   plus what the Windows package needs (GE-Proton 11-7 for DXVK,
#                                  and its licence)
#
# Already present entries are kept. Needs git, curl, tar and python3.
#
# Nothing from NVIDIA is downloaded: the model comes from your own nvngx_dlssnr.dll
# (linux/package/model-tools/extract_model.sh).
set -euo pipefail
cd -- "$(dirname -- "$0")"
windows=0
for a in "$@"; do
    case "$a" in --windows) windows=1;; *) echo "usage: fetch_deps.sh [--windows]" >&2; exit 2;; esac
done
mkdir -p toolchain artifacts/ref/downloads

# A git tree at one commit, without the history.
repo() {  # <dir> <url> <commit>
    local dir=$1 url=$2 commit=$3
    [[ -d "$dir" ]] && { echo "have $dir"; return; }
    echo "fetch $dir"
    git init -q "$dir"
    git -C "$dir" fetch -q --depth 1 "$url" "$commit"
    git -C "$dir" checkout -q FETCH_HEAD
}

# A file, checked against its SHA-256.
get() {  # <file> <url> <sha256>
    local file=$1 url=$2 sum=$3
    if [[ ! -f "$file" ]]; then
        echo "fetch $file"
        curl -fsSL --retry 3 -o "$file.part" "$url"
        mv -- "$file.part" "$file"
    fi
    echo "$sum  $file" | sha256sum -c --quiet - || { echo "checksum mismatch: $file" >&2; exit 1; }
}

# ---- build tools ------------------------------------------------------------------------------
repo toolchain/Vulkan-Headers https://github.com/KhronosGroup/Vulkan-Headers.git \
     e3b1eec08173d6b825cd3ac88c885a63b621504a                                      # v1.4.357
# The shaders are only byte-identical to the tested ones with this exact glslang.
get artifacts/ref/downloads/glslang-16.5.0-linux-x86_64-release.tar.gz \
    https://github.com/KhronosGroup/glslang/releases/download/16.5.0/glslang-16.5.0-linux-x86_64-release.tar.gz \
    b9b1f96acb898a62251b171f7695efcecfc206a530299054071919b06820f657
if [[ ! -x toolchain/glslang/bin/glslang ]]; then
    mkdir -p toolchain/glslang
    tar -C toolchain/glslang -xzf artifacts/ref/downloads/glslang-16.5.0-linux-x86_64-release.tar.gz
fi

# ---- sources and headers ----------------------------------------------------------------------
# DLSS5-Feeder (MIT): MinHook, the ReShade add-on headers and Dear ImGui under external/, and the
# DLSS5_Feed.fx effect the ReShade route ships.
repo artifacts/ref/DLSS5-Feeder https://github.com/jlrouzies-fr/DLSS5-Feeder.git \
     76a08db83652617cfe4192eeecf9e46bfc008382
repo artifacts/ref/reshade-shaders https://github.com/crosire/reshade-shaders.git \
     6db142b4b1a05c764222e5b0bd9a644b7ccfe1dc
# vort_Shaders (MIT): the motion vectors of the ReShade route.
repo artifacts/ref/vort_Shaders https://github.com/vortigern11/vort_Shaders.git \
     b410b9f0c0fbb83c8cb42164aaf1655fab386f4a
# The Vulkan loader source is cloned by linux/build/build_vulkan_loader.sh itself.

# ---- binaries the packages carry unmodified ---------------------------------------------------
# ReShade 6.8.0 with add-on support: the DLLs come out of the official installer, which is a zip.
get artifacts/ref/downloads/ReShade_Setup_6.8.0_Addon.exe \
    https://reshade.me/downloads/ReShade_Setup_6.8.0_Addon.exe \
    afe4c8f13048306307983b8b3d41d5bf00a86820440b0e57dea10950e1176445
rs=artifacts/ref/reshade-6.8.0
if [[ ! -f "$rs/ReShade64.dll" ]]; then
    mkdir -p "$rs"
    python3 - artifacts/ref/downloads/ReShade_Setup_6.8.0_Addon.exe "$rs" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    for name in ("ReShade32.dll", "ReShade64.dll"):
        open(f"{sys.argv[2]}/{name}", "wb").write(z.read(name))
PY
fi
sha256sum -c --quiet - <<EOF
da430e0a9c6eecefa0d1b27d05e16c426fb5d04e808b194d914eaac4b31bc0f8  $rs/ReShade32.dll
0cee63f9c9f13f3ac909c5b4903f4dbb4b719a7ab3b4f13b0deaf83c814b94f7  $rs/ReShade64.dll
EOF
get "$rs/ReShade-LICENSE.md" https://raw.githubusercontent.com/crosire/reshade/v6.8.0/LICENSE.md \
    237ded5b8344f820113efab1e65e91e1f159d9202c5b4856606a0590d3ffdab0
# OptiScaler-NR (GPL-3.0), the OptiScaler route's host, shipped as released: 0.8.91. 0.8.4 still works with the same
# core (NR_OPTI_ZIP=, see linux/package/optiscaler/README.md).
get artifacts/ref/downloads/OptiScaler-NR-v0.8.4.zip \
    https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/releases/download/v0.8.4/OptiScaler-NR-v0.8.4.zip \
    8789912859882e66b3f3a1aa768db947da779dfd65225df69ea919052e73a2e4
get artifacts/ref/downloads/OptiScaler-NR-v0.8.91.zip \
    https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/releases/download/v0.8.91/OptiScaler-NR-v0.8.91.zip \
    19a2852bb3f88e09075e3ccc66e0318c52e83a5901d9020a10384ae49d59ff77

# ---- Windows package only ---------------------------------------------------------------------
if [[ $windows == 1 ]]; then
    # DXVK as GE-Proton 11-7 ships it (PE build, it runs on Windows): the DX9 route's d3d9.dll and dxgi.dll.
    ge=GE-Proton11-7-x86_64
    if [[ ! -f "artifacts/ref/downloads/$ge.tar.gz" ]]; then
        echo "fetch $ge.tar.gz"
        curl -fsSL --retry 3 -o "artifacts/ref/downloads/$ge.tar.gz.part" \
            "https://github.com/GloriousEggroll/proton-ge-custom/releases/download/GE-Proton11-7/$ge.tar.gz"
        mv -- "artifacts/ref/downloads/$ge.tar.gz.part" "artifacts/ref/downloads/$ge.tar.gz"
    fi
    echo "7db87e9787e20c35cbdac26018431d5794626b626e4067b050684e45a88cc2ca229d7d263519eafb2e168cde5bef57611065d159d3685aaec152ccb9abe3073f  artifacts/ref/downloads/$ge.tar.gz" |
        sha512sum -c --quiet - || { echo "checksum mismatch: $ge.tar.gz" >&2; exit 1; }
    [[ -d "toolchain/$ge" ]] || tar -C toolchain -xzf "artifacts/ref/downloads/$ge.tar.gz"
    mkdir -p artifacts/ref/dxvk
    get artifacts/ref/dxvk/LICENSE \
        https://raw.githubusercontent.com/doitsujin/dxvk/601930949d111edbbcf9dd463948426d9f8f6ddd/LICENSE \
        a5cb1a6ded7d2d7e92d550ba28edd21be2d1d4044662b399887351023e30ce64
fi
echo "dependencies ready"
