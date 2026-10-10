# Architecture

The technical side of the project: how each route reaches the game, why it is Vulkan, and where the
code is. For installing and using it, see the [README](../README.md).

## The network

The NR network of DLSS 5 (`nvngx_dlssnr.dll` 310.8.0) is reimplemented as Vulkan compute shaders in
GLSL: FP8 matrix multiplication through `VK_KHR_cooperative_matrix` and `VK_EXT_shader_float8`, which
compile to the RDNA4 WMMA instructions. Precision follows the original operation by operation: FP8
where NVIDIA's kernels use FP8, FP16/FP32 everywhere else. The output is close to NVIDIA's but not
bit-identical. The weights are read from the user's own DLL at install time
(`linux/package/model-tools/`); nothing from NVIDIA is in this repository.

## Routes into a game

The network is Vulkan compute, so it has to run on a Vulkan device. On Linux it runs on the **game's own**
one: its passes are recorded into the game's command stream, between the game's work and the frame's
presentation, with no second device, no copy between APIs and no cross-API synchronisation. Every
route below is a way of getting from a game's graphics API to that Vulkan device and to the frame's
colour, depth and motion vectors.


**D3D12 and D3D11 on Linux.** Under Proton these games already run on Vulkan: vkd3d-proton translates
D3D12 and DXVK translates D3D10/11. Both expose interop interfaces that hand over the underlying
`VkDevice`, queue, command buffers and images. The network's shaders are recorded straight into the
game's own command buffer, at the point where the frame is processed.

**`optiscaler` route.** [OptiScaler-NR](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass)
hooks the game's DLSS/FSR/XeSS upscaler, gathers colour, depth and the engine's own motion vectors,
and calls NVIDIA's NGX API for feature 18 (DLSS-NR). This project provides that API: an NGX core
(`dlssnr_core.dll`, which OptiScaler loads through `[Libraries] NvngxPath`) implements the NGX entry
points and runs our network instead of NVIDIA's. It is deliberately not named `_nvngx.dll`: Streamline
games would take a loaded module of that name for NVIDIA's own core and lose their DLSS option.
OptiScaler's file is not modified; a few defects of the bundled release are corrected in memory when
it loads the core (`linux/src/pe/nr_pe_optifix.cpp`, listed in `linux/package/optiscaler/README.md`).
The motion vectors and depth are the engine's own, not estimated as in the ReShade routes.

**`reshade` route.** For games without an upscaler to hook, a ReShade add-on (the design follows
[DLSS5-Feeder](https://github.com/jlrouzies-fr/DLSS5-Feeder)) takes the back buffer, depth from
ReShade and motion vectors computed by the vort_Shaders motion-estimation effect, and runs the
network on it through the same DXVK / vkd3d-proton interop. ReShade is installed as `dxgi.dll`.

**`dx9` / `vulkan` routes.** D3D9 is the hard case. ReShade's D3D9 backend only gives an add-on D3D9
objects, and D3D9 has no compute shaders at all, so there is nothing to run the network on. Under
Proton, though, a D3D9 game's `d3d9.dll` is DXVK, which renders it with Vulkan. So the route loads
ReShade not into D3D9 but as a **Vulkan layer underneath DXVK**: ReShade then sees a Vulkan game, and
the add-on gets a real `VkDevice`. That needs three pieces of plumbing:

- A Vulkan loader that loads layers. Wine's own `vulkan-1.dll` does not, so the route ships the
  Khronos loader built for Windows with three small patches (`linux/vulkan-loader/`): it reads its
  driver and layer manifests from a `vk-override/` folder next to itself (under Wine every process
  looks elevated, so the stock loader ignores the `VK_*` environment variables), it never calls DXGI
  (its DXGI-based adapter sorting deadlocks inside DXVK's `dxgi.dll`), and it zeroes the padding of
  `vkGetDeviceQueue2`'s argument (Wine 11 compares the whole struct).
- A `winevulkan.dll` forwarder. DXVK loads `winevulkan.dll` directly, ReShade looks for
  `vulkan-1.dll`; a forwarder whose exports all point at our `vulkan-1.dll` gives both the same loader
  instance. The real `winevulkan.dll` is then loaded by the loader as the driver, by its full path.
- A depth setting. Old D3D9 games use conventional depth, not reversed-Z; `dx9` installs a ReShade
  preset with `RESHADE_DEPTH_INPUT_IS_REVERSED=0`, otherwise it is the `vulkan` route.

Native Vulkan games use the same route: ReShade as a layer under the game's own Vulkan device.

**Windows.** A native D3D game on Windows has no Vulkan device underneath it. DirectX 10/11/12 games
keep the system's D3D, and the network runs on a Vulkan device of its own, created on the adapter the
game renders with (matched by LUID; `windows/src/pe/nr_pe_bridge.hpp`). The frame crosses over in D3D
textures created as shared and imported into Vulkan, and the two sides are ordered on the GPU by a
shared D3D fence imported as a Vulkan timeline semaphore:

- D3D12, OptiScaler route: the evaluate call sits in the middle of the game's command list. The list
  is cut there (`nr_pe_d3d12split.cpp`): ExecuteCommandLists submits the part before it, the copies into
  the shared textures and a fence signal; the network runs on the Vulkan queue after that value and
  signals the next; the rest of the list runs after it, with the state it had at the cut replayed.
  A list the cut cannot follow (open queries or render passes, predication, bundles it has not seen)
  runs whole, without NR, for that frame.
- D3D12 and D3D11 in the ReShade route, D3D11 in the OptiScaler route: the work is ordered on the
  queue (D3D12) or on the immediate context through `ID3D11DeviceContext4` and a shared `ID3D11Fence`.
  OptiScaler runs the upscaler of DX11 games on a D3D12 device of its own (the installer sets
  `Dx11Upscaler=ffx_12`), and NR runs there.
- D3D10 has no fences: its shared textures carry a keyed mutex. Depth is not handed over yet: the D3D11
  path converts it with a compute shader, which D3D10 does not have, and a pixel-shader copy is not
  implemented.

D3D9 shares neither textures nor fences, so DX9 games run on DXVK with ReShade as a Vulkan layer
underneath it, as on Linux.

The Windows version runs the same network through AMD's Windows shader compiler, which produces slower
code for the largest kernels than Mesa's ACO does on Linux. With `[Network] ACO Mode = 1` it imports
machine code made by ACO for the Linux shaders instead (`windows/data/aco/records`), through
`VK_KHR_pipeline_binary`: the driver compiles a shader with the same interface to get its pipeline
container, ACO's code is put into that container (`windows/src/core/nr_pal_binary.hpp`,
`nr_pipeline_binary.hpp`), and the imported binaries are kept in `dlssnr-amd\aco-cache`.

## Why Vulkan and not HIP/ROCm

HIP could run this network as well - ROCm supports RDNA4 and its WMMA instructions. Vulkan was chosen
because it fits the job better:

- **The code runs inside the game.** OptiScaler, ReShade and every add-on are Windows DLLs loaded into
  the game's process; on Linux that process is a Wine/Proton process. Vulkan is there already, because
  DXVK and vkd3d-proton are how the game draws. HIP would need ROCm installed and a bridge from the
  Wine process to the Linux runtime.
- **Same device, no copies.** With Vulkan the network records into the game's own command buffers on
  the game's own device. HIP would need its own context next to the game's Vulkan device, the frame's
  images imported or copied into it, and synchronisation between the two APIs every frame.
- **One code base for Linux and Windows.** The AMD Windows driver exposes the same Vulkan extensions
  (`VK_KHR_cooperative_matrix`, `VK_EXT_shader_float8`), so the Windows version is the same shaders and
  host code, compiled by a different driver.
- **The hardware is reachable.** Cooperative matrices with FP8 operands compile to the RDNA4 WMMA
  instructions, which is where the network's time goes. The network is one sequential chain of about
  150 layers (only one adjacent pair is independent), so HIP streams or graphs would have nothing to
  run in parallel.

## Layout


```
linux/                 the Linux version (the reference implementation)
  src/core/              network runtime: graph, Vulkan helpers, planner
  src/pe/                the Windows DLLs as run under Proton: OptiScaler backend, ReShade add-on
  shaders/rdna4/         the network (pipelines.json lists every pipeline and its defines)
  shaders/passes/        the passes around it (encode, transfer, motion estimation)
  build/                 build_network.py, build_package.sh, build_optiscaler_nr.sh, ...
  package/               installer, README and model-tools/ (model extraction)
  vulkan-loader/         patches for the Khronos loader used by the vulkan/dx9 routes
windows/               a separate copy for the AMD Windows driver, changed on its own
  data/aco/              ACO machine code of the Linux network, for [Network] ACO Mode = 1
fetch_deps.sh          pinned third-party downloads
```

`linux/` and `windows/` share no files on purpose: workarounds for the Windows driver never
touch the Linux version.

The host code and the shaders must be built with the same defines (`*/build/arch/rdna4.sh`); the
runtime refuses a shader folder that disagrees.
