# Third-party components

Apart from the AMD FSR 1 headers, none of these are stored in this repository. `fetch_deps.sh` downloads them at pinned versions;
the build compiles against some of them and the packages carry some of them unmodified, each with
its licence file.

| Component | Licence | Used for |
| --- | --- | --- |
| [DLSS5-Feeder](https://github.com/jlrouzies-fr/DLSS5-Feeder) | MIT | The ReShade add-on (`*/src/pe/nr_reshade_addon.cpp`) follows its design; `DLSS5_Feed.fx` is shipped. Its bundled MinHook, ReShade add-on headers and Dear ImGui headers are compiled in. |
| [MinHook](https://github.com/TsudaKageyu/minhook) | BSD-2-Clause | Compiled into the DLLs (from DLSS5-Feeder's copy). |
| [Dear ImGui](https://github.com/ocornut/imgui) | MIT | Headers, for the ReShade add-on's settings page. |
| [ReShade](https://github.com/crosire/reshade) 6.8.0 | BSD-3-Clause | Add-on headers; `ReShade32/64.dll` shipped in the ReShade routes. |
| [reshade-shaders](https://github.com/crosire/reshade-shaders) | BSD-3-Clause | `ReShade.fxh`, `ReShadeUI.fxh` shipped. |
| [vort_Shaders](https://github.com/vortigern11/vort_Shaders) | MIT | Motion vectors for the ReShade routes, shipped. |
| [OptiScaler-NR](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass) 0.8.91 | GPL-3.0 | The OptiScaler route's host, shipped as released; this project's DLLs are separate modules it loads. |
| [AMD FidelityFX Super Resolution 1](https://github.com/GPUOpen-Effects/FidelityFX-FSR) (`ffx_a.h`, `ffx_fsr1.h`) | MIT | Stored in `linux/shaders/passes/include/fsr1/` and `windows/shaders/passes/include/fsr1/` with its licence; compiled into `runtime_upscale.spv` (the Classic enlargement's FSR 1 choice), which ships with the licence beside it. |
| [Vulkan-Loader](https://github.com/KhronosGroup/Vulkan-Loader) 1.4.357 | Apache-2.0 | Built with the patches in `*/vulkan-loader/` and shipped; the changes are stated in `PATCHES.diff`. |
| [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) 1.4.357 | Apache-2.0 / MIT | Build only. |
| [glslang](https://github.com/KhronosGroup/glslang) 16.5.0 | BSD-3-Clause and others | Build only (GLSL -> SPIR-V). |
| [DXVK](https://github.com/doitsujin/dxvk) from [GE-Proton](https://github.com/GloriousEggroll/proton-ge-custom) 11-7 | zlib | Windows package only (the DX9 route), shipped unmodified. |

NVIDIA's `nvngx_dlssnr.dll` and the weights in it are NVIDIA's property. This project contains no part
of them; `linux/package/model-tools/` reads them from a copy the user supplies. The layer table in
`*/src/core/nr_native_plan_data.hpp` and `linux/package/model-tools/descriptor.json` describe the
network's structure (layer types, shapes, offsets), not its weights. `linux/data/int4/` holds the int4
mixed option's tables: per matrix the clip step index of each row, the correction of each weight against
plain rounding, and calibration means of the inputs; no weight values. The int4 weights are made at
install from the user's own model.

## DLSS5-Feeder licence notice

```
MIT License

Copyright (c) 2026 Jean-Laurent ROUZIES

Portions derived from dlss5-dx11-bridge, Copyright (c) 2026 NIGos (MIT), see external/bridge-1.0.19/LICENSE.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
