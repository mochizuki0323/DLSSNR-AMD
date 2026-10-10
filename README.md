# DLSSNR-AMD

Runs the neural rendering (NR) model of NVIDIA DLSS 5 in games on AMD Radeon RX 9000 (RDNA4)
graphics cards.

- The network is reimplemented as Vulkan compute shaders.
- The model is not included. The installer extracts it from your own copy of `nvngx_dlssnr.dll`
  (version 310.8.0).
- The Linux build, on an RX 9070 XT, checked against NVIDIA's own `nvngx_dlssnr.dll` run on an
  RTX 5090 through NGX (the way a game calls it), with the same input and settings. Single frames:
  PSNR 45.56 dB at 1080p, 47.99 dB at 1440p and 49.06 dB at 4K (higher is closer; the input frame
  itself scores 26-29 dB), SSIM 0.996-0.997 (1 = identical); NVIDIA's DLL gives byte-identical output
  from run to run, so the difference is between the two implementations, not noise. Moving sequences
  with motion vectors and history (a still scene, a moving object, a panning camera, and NR after an
  upscaler; 10 frames each): PSNR 46.3-47.0 dB at 1080p, 50.1-50.9 dB at 1440p and 51.1-51.6 dB at 4K,
  SSIM 0.997-0.998, and the output changes from frame to frame by about as much as NVIDIA's. Method,
  pictures and data: [docs/ngx-verification](docs/ngx-verification/NGX-VERIFICATION.md).
- **Tested only on an RX 9070 XT.** Other cards are not guaranteed to work.
- **The Windows version is an experimental preview and has not been tested much.** Game crashes,
  driver resets and other unexpected problems can happen. With its default settings the network is
  slower than on Linux. With [ACO Mode](#windows-experimental-preview) it loads machine code that was
  made in advance with Mesa's ACO compiler (the Linux driver's compiler) and ships in the package, and
  comes close to Linux (see [Performance](#performance)). ACO Mode has been tested on one driver version only (AMD Software
  26.9.2); see [Windows](#windows-experimental-preview).

This is an independent project. It is not affiliated with, endorsed by or supported by NVIDIA or
AMD. DLSS is a trademark of NVIDIA Corporation.

## Before you use it

This is an early project (version 0.0.x) and its testing is limited:

- one graphics card, one Linux desktop, Mesa 26.2 and GE-Proton 11-7;
- a small number of games, at a few resolutions and settings;

Other cards, drivers, Proton versions, distributions and games have not been tested. Expect problems:
games that fail to start or crash, visual artifacts, settings that do not behave as described, or
performance that differs from the numbers below. Problems are fixed as they are found, and the project
will keep changing, including in ways that break earlier setups.

- The installer puts DLLs into the game folder. Keep a backup of anything you care about; `remove`
  deletes what was installed.
- **Do not use it in online games with anti-cheat.** Injected DLLs can get an account banned.
- It is provided as is, without warranty (see [LICENSE](LICENSE)).

If something goes wrong, please open an issue with the game, the route, your card and driver, and
these logs:

- `dlssnr-amd.log` and `OptiScaler.log` or `ReShade.log` (whichever route you use), all in the game
  folder. On Windows, option 5 of `install.bat` puts them into one zip.
- On Linux, if you can, a Proton log as well: set the game's launch options in Steam to
  `PROTON_LOG=1 %command%`, run the game until the problem shows, and attach `steam-<appid>.log` from
  your home folder.

## Status

| Platform | Needs | State |
| --- | --- | --- |
| **Linux** (Steam / Proton) | Mesa 26.2 or newer, GE-Proton 11-7 (tested) | The main version, used in games. |
| **Windows** | AMD Software 26.9.2 (tested) | **Experimental preview, not tested much.** Slower than Linux by default, close to it with ACO Mode; game crashes, driver resets and other problems can happen. |

Both need an RX 9000 series (RDNA4) card; older cards (RX 7000 and earlier) lack the FP8 matrix
instructions the network needs.

## Performance

GPU time of the network per frame on an RX 9070 XT, **offline benchmark** (network only), measured
with v0.0.3 (the int4 mixed row with v0.0.4, the Windows ACO Mode row with v0.0.4-win-preview on
AMD Software 26.9.2):

| | 1080p | 1440p | 4K |
| --- | --- | --- | --- |
| Linux | 5.60 ms | 9.70 ms | 21.89 ms |
| Linux, [int4 mixed](#int4-mixed-optional-linux) | 4.99 ms | 8.69 ms | 19.64 ms |
| Windows | 7.21 ms | 12.59 ms | 27.32 ms |
| Windows, [ACO Mode](#windows-experimental-preview) On | 5.70 ms | 9.96 ms | 22.49 ms |

In game (Linux, RX 9070 XT):

| Game | Setting | Output | Render | Model | Without NR | With NR |
| --- | --- | --- | --- | --- | --- | --- |
| 007 First Light (v0.0.2) | FSR4 Performance, NR before upscaler | 4K | 1080p | 1080p | 126 fps | 68 fps |
| 007 First Light (v0.0.2) | FSR4 Performance, NR after upscaler | 4K | 1080p | 4K | 126 fps | 30 fps |
| Kingdom Come: Deliverance II (v0.0.2) | FSR4 Performance, NR before upscaler | 4K | 1080p | 1080p | 85 fps | 54 fps |
| Kingdom Come: Deliverance II (v0.0.2) | FSR4 Performance, NR after upscaler | 4K | 1080p | 4K | 85 fps | 27 fps |
| Dying Light: The Beast | FSR4 Performance, NR before upscaler | 4K | 1080p | 1080p | 97 fps | 56 fps |
| Dying Light: The Beast | FSR4 Performance, NR after upscaler | 4K | 1080p | 4K | 97 fps | 27 fps |
| Tomb Raider (2013) | Model resolution 50% | 4K | 4K | 1080p | 130 fps | 60 fps |
| Tomb Raider (2013) | Model resolution 100% | 4K | 4K | 4K | 130 fps | 30 fps |
| 7 Days to Die | FSR4 Performance, NR before upscaler | 4K | 1080p | 1080p | 136 fps | 68 fps |
| 7 Days to Die | FSR4 Performance, NR after upscaler | 4K | 1080p | 4K | 136 fps | 29 fps |

Rows marked (v0.0.2) were measured with version 0.0.2, the others with 0.0.1.

Running the model below the output resolution about doubles the frame rate. In the ReShade routes
that is the **Model resolution** setting. In the OptiScaler route it is mainly OptiScaler-NR's
**"Generate model before upscaler"** option, which runs the model on the game's render-resolution
frame before FSR4 upscales it; OptiScaler-NR has a **Model resolution** setting as well.

None of these measurements use frame generation; the OptiScaler route can turn it on.

## Supported games

| Game uses | Linux | Windows |
| --- | --- | --- |
| **DirectX 12** | `optiscaler` if the game has DLSS/FSR/XeSS, otherwise `reshade` | OptiScaler or ReShade |
| **DirectX 11** | `optiscaler` if the game has DLSS/FSR/XeSS, otherwise `reshade` | OptiScaler or ReShade |
| **DirectX 10** | `reshade` | ReShade |
| **DirectX 9** | `dx9` | ReShade (DX9) |
| **Vulkan** | `vulkan` | ReShade |
| OpenGL | not supported | not supported |

- **`optiscaler`**: for games with a DLSS, FSR or XeSS option. Uses
  [OptiScaler-NR](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass) with this project as its
  DLSS-NR backend, and the game's own motion vectors. 64-bit games only.
- **`reshade`**: for other DirectX 10/11/12 games. Runs through ReShade; motion is estimated from the
  picture by a ReShade shader.
- **`vulkan` / `dx9`**: for Vulkan and DirectX 9 games. On Linux, DirectX 9 games run on Vulkan through
  DXVK, and ReShade is loaded as a Vulkan layer underneath it. `dx9` also sets the depth direction old
  DirectX 9 games use.

32-bit games work on Linux (use the i686 package; not with `optiscaler`). The Windows version is
64-bit only.

## Preprocess (optional)

`[Preprocess]` in `dlssnr-amd.ini` in the game folder changes the picture the NR network is shown (exposure,
display curve, contrast, saturation), and so how NR edits the picture. It has two uses: a personal look in any
game, and fixing games that do not hand their exposure to the upscaler (below). The installer writes the
file; the ReShade routes also show the settings on the Add-ons page.

- **Enabled**: off by default; when off, none of it runs. Turned on, it starts from auto exposure and the
  filmic curve.
- **Exposure**:
  - `auto`: auto exposure after Unreal Engine's design, for games that compute their exposure but do not hand
    it to the upscaler.
  - `off`: the exposure stays as the game or OptiScaler set it.
  - `fixed`: `ExposureBias` alone.
- **ExposureBias** (EV, -8 to +8, decimals allowed): with `fixed` it is the whole gain; with `auto` it is added
  to what auto works out.
- **Curve, Contrast, Saturation**: change the picture NR is shown, and so how it edits the picture.
- **Hotkey** (Ctrl+F10 by default): switches the preprocess on and off for the current run, to compare on
  the same picture. It does not change the file.
- **Sound**: a short sound when it switches: two notes going up for on, going down for off. `Sound = 0`
  turns it off.

Some games hand the upscaler a picture without its exposure, and NR then works on a frame that is several
stops too dark: 007 First Light turns green and grainy with NR. Such games are fixed by the preprocess: set
`Enabled = 1` (its defaults, auto exposure and the filmic curve, are meant for this).

For a personal look in a game that gives its exposure properly, `auto` is not needed: use `Exposure = off`
(or `fixed` for a small shift) and change `Curve`, `Contrast` or `Saturation`. This departs from the original
look; it may be better or worse.

The first time the preprocess is turned on, NR rebuilds; a second or two of frames go without NR.
The file itself explains every setting.

## int4 mixed (optional, Linux)

A second network, faster than the default one (see Performance): part of its computation runs in int4,
a lower precision than the original network uses, so its picture differs somewhat from the default
network's. Both Linux packages have it; it is chosen at install.

- The installer asks once (Enter = no), or pass `--int4` or `--no-int4`.
- With it, the Steam launch options the installer prints have two more entries, `VK_ADD_LAYER_PATH`
  and `VK_INSTANCE_LAYERS` (a Vulkan layer in the game folder that int4 mixed needs). Copy the whole
  line.
- Once installed it is on. In game, Ctrl+F11 switches between int4 mixed and the default network, to
  compare them on the same picture; a switch builds the other network in the background, which takes a
  few seconds (`KeepBoth = 1` keeps both networks in video memory and makes switching instant). The
  settings are in `[Int4Mixed]` in `dlssnr-amd.ini`.
- With int4 mixed, NR takes much longer to take effect when a game starts. Until then the picture goes
  without NR; this does not mean int4 mixed is not working.
- Its weights are made during installation from your model and the tables in the package, and checked
  against a known SHA-256. The tables (`linux/data/int4/`) come from this project's own calibration on
  1,112 game frames and darkened copies of some of them. int4 mixed is the network these tables define;
  another calibration would give a different network.

## Why Vulkan

HIP would work too: ROCm supports RDNA4 and its matrix (WMMA) instructions. Vulkan fits this job
better:

- The network runs on the game's own Vulkan device and queue (DXVK / vkd3d-proton under Proton). The
  frame never leaves that device, and no second GPU context or cross-API synchronisation is needed.
- Nothing extra to install: Vulkan comes with the graphics driver. On Linux the HIP runtime
  (`libamdhip64`) is not part of Mesa and has to be installed separately (ROCm or the distribution's
  packages), plus a bridge to reach it from a game running in Proton (a Windows process). On Windows
  the AMD driver includes it (`amdhip64_7.dll` in current drivers).
- RDNA4's matrix instructions are available in Vulkan through `VK_KHR_cooperative_matrix`, both in
  Mesa (Linux) and in the AMD Windows driver.

How the routes work in detail, and where the code is: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Install (Linux)

Download a `DLSSNR-AMD-Vulkan-Linux-*-x86_64.tar.gz` (64-bit games) or `-i686.tar.gz` (32-bit games) from
the releases, unpack it and run:

```sh
bash install.sh "/path/to/steamapps/common/<game folder>" --dll /path/to/nvngx_dlssnr_310.8.0.zip
```

The folder is the one holding the game's exe. The installer lists the routes, asks whether to add
[int4 mixed](#int4-mixed-optional-linux), extracts the model (below) and prints the Steam launch
options to use. `--dll` is needed only for the first install:
the extracted model is kept and reused for every later one. Needs `bash` and `python3`. See
[linux/package/README.txt](linux/package/README.txt).

## Install (Windows)

Download `DLSSNR-AMD-Vulkan-Windows-*-preview-x86_64.zip` from the releases, unpack it and double-click
`install.bat`. Pick the game's exe (the one that actually runs, not a launcher), then the route. The
first install asks for `nvngx_dlssnr.dll` 310.8.0 or a zip that contains it and extracts the model
(below); later installs from the same package do not ask again. Read
[windows/package/README.txt](windows/package/README.txt) and [Windows](#windows-experimental-preview)
first.

## The model

The weights are NVIDIA's and are not part of this project. They are extracted from your own copy of
`nvngx_dlssnr.dll`, which must be **version 310.8.0**.

- Input: the DLL itself, or a zip that contains exactly one `nvngx_dlssnr.dll` (it may be in a
  subfolder).
- A DLL of any other version is refused.
- All 599 extracted entries are checked against known hashes; the model file is written only if every
  one matches. On Linux it takes about 20 seconds and needs `bash` and `python3`. The Windows package
  does the same with `model-tools\dlssnr_extract_model.exe` (no Python needed), in about a second.
- The extracted model is kept in the package's own `dlssnr-amd/dlssnr.bin`. Later installs from
  that package without `--dll` check its SHA256 and install it from there, so the extraction runs
  only once per package. With a new package, use `--dll` once more or copy that file over.

Two ways to run it:

```sh
# while installing: the model goes to <game folder>/dlssnr-amd/dlssnr.bin
bash install.sh "<game folder>" --dll nvngx_dlssnr_310.8.0.zip

# on its own: model-tools/ in the package, linux/package/model-tools/ in this repository
bash model-tools/extract_model.sh nvngx_dlssnr_310.8.0.zip dlssnr.bin
```

On success it prints `599 entries, 140.9 MiB` (147,756,560 bytes). Put the file at
`<game folder>/dlssnr-amd/dlssnr.bin`, or pass it to a package build with `NR_MODEL=`.

On Windows, `install.bat` extracts the model by itself; to run the tool on its own:

```bat
model-tools\dlssnr_extract_model.exe nvngx_dlssnr.dll dlssnr.bin
```

It takes the DLL itself; unpack it from the zip first.

## Windows (experimental preview)

**The Windows version is an experimental preview and has not been tested much.** Game crashes, driver
resets and other unexpected problems can happen. It is built from the same code as the Linux version
and changed only where the AMD Windows driver needs it. It is updated less often than the Linux
version, and some releases may be Linux only.

DirectX 10/11/12 games keep the system's own D3D. NR runs on a Vulkan device of its own on the same
card, and each frame is handed between the two through shared textures and a shared fence, on the
GPU. DirectX 9 games run on Vulkan through DXVK, because D3D9 cannot share a frame with another API.

**ACO Mode.** `[Network] ACO Mode` in `dlssnr-amd.ini` chooses where the network's machine code comes
from. Off (`ACO Mode = 0`, the default): the AMD driver compiles the network. On (`ACO Mode = 1`): the
package carries the network as machine code made in advance with Mesa's ACO compiler (the Linux
driver's compiler), and the AMD driver loads that code instead of compiling the network. ACO's machine
code for this network runs faster than the AMD driver's (see Performance). It takes effect when the
game restarts.
**ACO Mode has been tested on one driver version only (AMD Software 26.9.2, RX 9070 XT). With other
drivers it may fail: game crashes, wrong pictures or driver resets. If that happens, turn it off.**
When ACO Mode cannot be used at all, NR runs as with it off and `dlssnr-amd.log` says why.

Known issues:

- With ACO Mode off it is slower than the Linux version (see Performance).
- The first time NR runs in a game, the network compiles for about a minute before it takes effect
  (on Linux about 10-20 seconds); this happens once for each game. Until then the picture looks as
  without NR, which does not mean the mod is not working: give it a minute.
- DirectX 9 games run entirely on DXVK, which changes their performance and behaviour; overlays such
  as Steam's may not show.
- In DirectX 10 games the game's depth buffer is not passed to NR yet (not implemented).
- Games whose FSR runs in their own shaders never hand it to OptiScaler: pick the game's DLSS option
  instead, which OptiScaler offers.
- The network pauses itself when video or system memory runs short.
- With ACO Mode off the network is built by the AMD Windows driver's shader compiler, which rounds
  differently from the Linux version in places; on a 1080p test frame the two outputs are 48.5 dB
  PSNR apart.

## Build

Tested on Ubuntu 24.04. Everything, including the Windows DLLs, is cross-compiled on Linux.

```sh
sudo apt install git curl python3 cmake ninja-build mingw-w64 g++-multilib
bash fetch_deps.sh                          # pinned third-party pieces -> toolchain/, artifacts/ref/
bash linux/build/build_vulkan_loader.sh     # the patched Vulkan loader for the vulkan/dx9 routes
NR_ARCH=x86_64 bash linux/build/build_package.sh
NR_ARCH=i686   bash linux/build/build_package.sh
```

The packages land in `linux/package/`. They contain no model; `install.sh --dll` extracts it. To put a
model you extracted yourself into a package for your own use, set `NR_MODEL=/path/to/dlssnr.bin`.
Do not share packages that contain the model.

### Windows package

```sh
bash fetch_deps.sh --windows                # adds GE-Proton 11-7 (DXVK)
bash windows/build/build_package.sh
```

The result is `windows/package/DLSSNR-AMD-Vulkan-Windows-*-preview-x86_64.zip`; on Windows, run `install.bat` from it.
It contains no model; the installer extracts it on the first install. `NR_MODEL=/path/to/dlssnr.bin`
puts one into the package for your own use. Do not share packages that contain the model.

## Licence

MIT for the code in this repository, see [LICENSE](LICENSE). The packages also carry third-party
components under their own licences, listed in [THIRD_PARTY.md](THIRD_PARTY.md). NVIDIA's model
weights are NVIDIA's property and are neither included nor distributed.
