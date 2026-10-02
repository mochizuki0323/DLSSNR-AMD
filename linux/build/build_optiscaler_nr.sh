#!/usr/bin/env bash
# The three DLLs that give an OptiScaler DLSS-NR fork an AMD backend, cross-compiled to PE.
#
#   _nvngx.dll             the NGX core. Two jobs: the capability parameter block every fork needs,
#                          and feature 18 itself -- wilsjo2 >= 0.8.1 loads no forwarder and calls
#                          NVSDK_NGX_D3D12_CreateFeature(cmd, 18, params, &handle) straight into this.
#                          This is the DLL [Libraries] NvngxPath points at.
#   nvngx.dll_dlssnr.dll   the 28-export forwarder, for Dagherbou's lineage, which loads it beside
#                          itself and calls dlssnr_call_* / dlssnr_vk_*.
#   nvngx_dlssnr.dll       a byte copy of the forwarder; the older fork checks this file exists before
#                          it will create a feature at all (DlssNr_Dx12.cpp looks for the "snippet"
#                          beside itself or the executable and fails with "nvngx_dlssnr.dll was not
#                          found" if it is missing). Nothing loads it -- no snippet is ever resolved.
#
# All three share linux/src/pe/nr_dlssnr_model.cpp, which is the model: one nr::pe::Session per graphics
# API, the feature handles, the history rule and the control mapping. The two entry points cannot
# drift because there is only one body between them.
#
# OptiScaler itself is not built here and is not modified. Compile flags, include paths and the
# Vulkan lazy-import trick are lifted from linux/build/build_package.sh, which is the PE build
# this shares every object with below the ABI layer.
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."

arch=${NR_ARCH:-x86_64}
case "$arch" in x86_64|i686) ;; *) echo "NR_ARCH must be x86_64 or i686" >&2; exit 2;; esac
if [[ "$arch" == i686 ]]; then default_out=artifacts/optiscaler/nr32; else default_out=artifacts/optiscaler/nr; fi
out=$(realpath -m -- "${1:-$default_out}")
case "$out" in "$(pwd)"/*) ;; *) echo 'build directory must be in the project' >&2; exit 2;; esac
mkdir -p -- "$out"

cxx=${NR_MINGW:-$arch-w64-mingw32-g++}
cc=${NR_MINGW_CC:-$arch-w64-mingw32-gcc}
objdump=${NR_OBJDUMP:-$arch-w64-mingw32-objdump}
command -v "$cxx" >/dev/null || { echo "no mingw cross compiler ($cxx)" >&2; exit 1; }

minhook=artifacts/ref/DLSS5-Feeder/external/minhook
[[ -d "$minhook" ]] || { echo "missing reference headers: $minhook" >&2; exit 1; }
[[ -d toolchain/Vulkan-Headers/include ]] || { echo "missing toolchain/Vulkan-Headers" >&2; exit 1; }

# Same flag set as the game module: these objects are the same code, and building them twice with
# different flags is how two builds of one network start disagreeing.
# The version the DLL was built from (version.sh), printed in the first line of its log. Without it there is no
# way to tell from a game whether a fix is in the binary that ran.
stamp=$(bash linux/build/version.sh)
common=(-std=c++17 -O2 -DNDEBUG -DNR_BUILD_STAMP="\"$stamp\"" -Itoolchain/Vulkan-Headers/include -Ilinux/src -Ilinux/src/core -Ilinux/src/layer -Ilinux/src/pe -I"$out")

# The network the package ships (linux/build/arch/rdna4.sh): the host must be built
# with the same constants as the shaders in dlssnr-amd/shaders.
source linux/build/arch/rdna4.sh
"$cxx" "${common[@]}" "${NR_PRODUCT_DEFINES[@]}" -c linux/src/core/nr_runtime.cpp -o "$out/nr_runtime.o"
"$cxx" "${common[@]}" -c linux/src/core/nr_native_plan.cpp -o "$out/nr_native_plan.o"
"$cxx" "${common[@]}" -c linux/src/pe/nr_pe_log.cpp -o "$out/log.o"
"$cxx" "${common[@]}" -c linux/src/pe/nr_pe_interop.cpp -o "$out/interop.o"
"$cxx" "${common[@]}" -I"$minhook/include" -c linux/src/pe/nr_pe_vkdevice.cpp -o "$out/vkdevice.o"
"$cxx" "${common[@]}" -c linux/src/pe/nr_pe_session.cpp -o "$out/session.o"
"$cxx" "${common[@]}" -c linux/src/pe/nr_pe_config.cpp -o "$out/config.o"
"$cxx" "${common[@]}" -c linux/src/pe/nr_dlssnr_model.cpp -o "$out/model.o"
"$cxx" "${common[@]}" -c linux/src/pe/nr_gamescope_bridge.cpp -o "$out/gamescope_bridge.o"
"$cxx" "${common[@]}" -c linux/src/pe/nr_gamescope_config.cpp -o "$out/gamescope_config.o"
"$cxx" "${common[@]}" -c linux/src/pe/nr_dlssnr_forwarder.cpp -o "$out/forwarder.o"
"$cxx" "${common[@]}" -c linux/src/pe/nr_ngx_core.cpp -o "$out/ngx_core.o"
"$cxx" "${common[@]}" -I"$minhook/include" -c linux/src/pe/nr_pe_optifix.cpp -o "$out/optifix.o"

# MinHook, for the device watcher the Vulkan path asks for a queue through.
for unit in hook buffer trampoline; do
    "$cc" -O2 -I"$minhook/include" -I"$minhook/src" -c "$minhook/src/$unit.c" -o "$out/mh_$unit.o"
done
hde=hde64; [[ "$arch" == i686 ]] && hde=hde32
"$cc" -O2 -I"$minhook/include" -I"$minhook/src" -c "$minhook/src/hde/$hde.c" -o "$out/mh_hde.o"

ldflags=(-ld3d12 -ldxgi -lole32 -static -static-libgcc -static-libstdc++)
[[ "$arch" == i686 ]] && ldflags+=(-Wl,--kill-at)

# Vulkan is resolved at run time rather than imported, exactly as in build_package.sh: a static
# import of vulkan-1.dll forces winevulkan up before the game's graphics stack, which has killed a
# real game. This link is expected to fail; its failure is the input to the next step.
shared_objs=("$out/model.o" "$out/gamescope_bridge.o" "$out/gamescope_config.o" "$out/session.o" "$out/config.o" "$out/interop.o" "$out/log.o" "$out/vkdevice.o"
             "$out/nr_runtime.o" "$out/nr_native_plan.o" "$out"/mh_*.o)
forwarder_objs=("$out/forwarder.o" "${shared_objs[@]}")
core_objs=("$out/ngx_core.o" "$out/optifix.o" "${shared_objs[@]}")
# Both links contribute undefined vk* names; the probe link takes every object either DLL uses so one
# generated thunk file serves both.
undefined=$({ "$cxx" -shared -o /dev/null "${forwarder_objs[@]}" "$out/ngx_core.o" "$out/optifix.o" "${ldflags[@]}" 2>&1 || true; } |
    grep -o "undefined reference to \`_*vk[A-Za-z0-9_@]*'" | sed "s/.*\`//; s/'//" | sort -u)
count=$(echo "$undefined" | grep -c . || true)
plain() { echo "$1" | sed 's/^_//; s/@[0-9]*$//'; }

{
    echo '// Generated by linux/build/build_optiscaler_nr.sh. One tail jump per entry point.'
    echo '    .text'
    for sym in $undefined; do
        base=$(plain "$sym")
        if [[ "$arch" == i686 ]]; then
            label="_${sym#_}"
            printf '    .globl %s\n%s:\n    jmp *_nr_vk_p_%s\n' "$label" "$label" "$base"
        else
            printf '    .globl %s\n%s:\n    jmp *nr_vk_p_%s(%%rip)\n' "$sym" "$sym" "$base"
        fi
    done
    echo '    .data'
    for sym in $undefined; do
        base=$(plain "$sym")
        if [[ "$arch" == i686 ]]; then
            printf '    .globl _nr_vk_p_%s\n    .align 4\n_nr_vk_p_%s:\n    .long _nr_vk_unresolved\n' "$base" "$base"
        else
            printf '    .globl nr_vk_p_%s\n    .align 8\nnr_vk_p_%s:\n    .quad nr_vk_unresolved\n' "$base" "$base"
        fi
    done
} > "$out/vulkan_lazy.s"

{
    echo '// Generated by linux/build/build_optiscaler_nr.sh.'
    echo '#include <windows.h>'
    echo 'long nr_vk_unresolved(void) { return -3; }   // VK_ERROR_INITIALIZATION_FAILED'
    for sym in $undefined; do printf 'extern void* nr_vk_p_%s;\n' "$(plain "$sym")"; done
    echo 'void nr_vk_load(void) {'
    echo '    HMODULE m = GetModuleHandleW(L"vulkan-1.dll");'
    echo '    if (!m) m = LoadLibraryW(L"vulkan-1.dll");'
    echo '    if (!m) return;'
    for sym in $undefined; do
        base=$(plain "$sym")
        printf '    { void* p = (void*)GetProcAddress(m, "%s"); if (p) nr_vk_p_%s = p; else OutputDebugStringA("nr_vk_load: no export %s"); }\n' "$base" "$base" "$base"
    done
    echo '}'
} > "$out/vulkan_lazy.c"

"$cc" -c "$out/vulkan_lazy.s" -o "$out/vulkan_lazy_s.o"
"$cc" -O2 -c "$out/vulkan_lazy.c" -o "$out/vulkan_lazy_c.o"
echo "vulkan entry points resolved at run time: $count"

"$cxx" -shared -o "$out/nvngx.dll_dlssnr.dll" \
    "${forwarder_objs[@]}" "$out/vulkan_lazy_s.o" "$out/vulkan_lazy_c.o" "${ldflags[@]}"

# The core is no longer a stub: it carries the same model, because wilsjo2's fork drives feature 18
# through it and loads nothing else.
"$cxx" -shared -o "$out/_nvngx.dll" \
    "${core_objs[@]}" "$out/vulkan_lazy_s.o" "$out/vulkan_lazy_c.o" "${ldflags[@]}"

# The presence check. Dagherbou's fork will not create a feature unless a file by this name sits
# beside it or the executable, because on NVIDIA that file is the model. Here it is the forwarder
# again, and nothing ever calls into it.
#
# wilsjo2's fork looks for the same name for a different reason: DlssNr_CompatibilityRuntime opens it
# with LoadLibraryExW and drives NVIDIA's own runtime through it, but ONLY if our CreateFeature(18)
# fails and returns no handle (DlssNr_Proxy.cpp:180). If that ever happened it would find our
# forwarder, which exports no NVSDK_NGX_D3D12_Init_Ext, and give up with "is missing required NR
# exports" -- preserving our failure rather than running an NVIDIA snippet on an AMD card. That is the
# safe outcome, and it is the second reason CreateFeature must succeed.
cp -- "$out/nvngx.dll_dlssnr.dll" "$out/nvngx_dlssnr.dll"

echo
echo "built in $out:"
for dll in nvngx.dll_dlssnr.dll nvngx_dlssnr.dll _nvngx.dll; do
    printf '  %-24s %s bytes\n' "$dll" "$(stat -c %s "$out/$dll")"
done

echo
echo "forwarder exports (dlssnr_*): $("$objdump" -p "$out/nvngx.dll_dlssnr.dll" | grep -c 'dlssnr_')"
"$objdump" -p "$out/nvngx.dll_dlssnr.dll" | grep -o '\bdlssnr_[A-Za-z0-9_]*' | sort -u | sed 's/^/    /'
echo
echo "core exports (NVSDK_NGX_*): $("$objdump" -p "$out/_nvngx.dll" | grep -c 'NVSDK_NGX_')"
"$objdump" -p "$out/_nvngx.dll" | grep -o '\bNVSDK_NGX_[A-Za-z0-9_]*' | sort -u | sed 's/^/    /'
