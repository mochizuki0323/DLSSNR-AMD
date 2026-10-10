DLSSNR-AMD-Vulkan (Windows, experimental preview)
=================================================

Runs the neural rendering (NR) model of DLSS 5 in games on AMD graphics cards. The network is
reimplemented in Vulkan; NVIDIA's runtime is neither needed nor called.

This is an experimental preview and has not been tested much: game crashes, driver resets and
other unexpected problems can happen. It is slower than the Linux version and updated less
often, and some releases may be Linux only. If something breaks, uninstall with option 5 of
install.bat.

DX10/11/12 games keep the system's own D3D; nothing is translated. NR runs on a Vulkan device
of its own on the same graphics card, and the frame is handed between the two through shared
textures and a shared fence, all on the GPU. Only DX9 games are moved to Vulkan with DXVK
(D3D9 cannot share a frame with another API).

Requirements
------------
- RX 9000 series (RDNA4) graphics card, AMD Software 26.9.2 (tested)
- 64-bit games
- Do not use it in online games with anti-cheat; start games with EasyAntiCheat with EAC off

Install
-------
Double-click install.bat, pick the game's exe (the one that actually runs - for Unreal Engine
games it is under Binaries\Win64, not the launcher outside), then pick a route:

  1) OptiScaler   the game has a DLSS, FSR or XeSS option (DX11 / DX12)
  2) ReShade      other DX10/11/12 games
  3) ReShade      old DX9 games
  4) ReShade      Vulkan games
  5) Uninstall
  6) Collect logs when something goes wrong: writes a zip to the desktop to attach to an issue

You can also drag the game's exe onto install.bat. Games under Program Files ask for
administrator rights.

Model
-----
The package does not contain NVIDIA's model. The first install asks for nvngx_dlssnr.dll
(version 310.8.0 only) or a zip that contains it. The installer extracts the model with
model-tools\dlssnr_extract_model.exe (it only reads the weight data; the DLL is never loaded or
run), checks every entry against known hashes and writes dlssnr-amd\dlssnr.bin only if all
match. The model stays in this package, so later installs from it (into other games too) do not
ask again; with a newer package, choose the DLL once more or copy that file over.

Use
---
OptiScaler: turn on DLSS (or FSR / XeSS) in the game's graphics settings. Insert opens the
            OptiScaler menu, the NR settings are on the DLSS Neural Rendering page.
            If the NR page keeps showing "Waiting for the upscaler to run", set
            [Spoofing] Dxgi=false in OptiScaler.ini and try again.
ReShade:    Home opens ReShade, the settings are on the Add-ons page; the same settings are
            kept in dlssnr-amd.ini in the game folder and changes apply live.
            Model resolution, and Model passes of 5 or more, rebuild the network; for the few
            seconds that takes the picture is shown without NR.
            When Model resolution is below 100%, Enlargement chooses how the model's result is
            brought back to full resolution: Matched residual (the default) and Edge-aware
            lighting + colour enlarge the change the model made and apply it to the
            full-resolution picture; Classic enlarges the model's output picture (with a choice
            of scalers, including FSR 1).
            HDR: scRGB (linear) HDR works; this route leaves HDR10 (PQ/HLG) pictures alone, so
            set the game to SDR or scRGB.

Preprocess (optional, off by default): [Preprocess] in dlssnr-amd.ini in the game folder (the
installer writes the file; ReShade also shows it on the Add-ons page). It changes the picture
the network is shown (exposure, display curve, contrast, saturation), and so how NR edits the
picture. Two uses: a personal look in any game (it departs from the original look; the result
may be better or worse), and games that do not hand their exposure to the upscaler, which it
fixes (007 First Light turns green and grainy with NR otherwise). Ctrl+F10 switches it for the
current run, to compare. The file explains every setting.

The first time NR runs in a game the network has to compile; it takes effect after about a
minute (much longer than on Linux, where it takes 10-20 seconds). Until then the picture looks
as without NR, which does not mean the mod is not working: give it a minute. After that it is
cached. This happens once for each game.

Files put into the game folder
------------------------------
    dlssnr-amd\                  model and shaders
    dlssnr-amd-install.txt       the installation record, which uninstall follows
    dlssnr-amd.ini               settings, written at install
    OptiScaler route:            dxgi.dll (OptiScaler), dlssnr_core.dll and OptiScaler's own files
    ReShade route (DX10/11/12):  dxgi.dll (ReShade), dlssnr_amd.addon64, ReShade's settings and
                                 shaders
    ReShade route (DX9):         d3d9.dll dxgi.dll (DXVK), vulkan-1.dll (Vulkan loader; never
                                 calls DXGI and loads the ReShade layer from the game folder),
                                 ReShade's files
    ReShade route (Vulkan):      vulkan-1.dll and ReShade's files
Files of the same name already in the game folder are first moved to dlssnr-amd-backup\ and
put back on uninstall.
Installing again (another route, another package) first uninstalls by the installation record,
and dlssnr-amd.ini is replaced with a new default file: earlier settings are not kept, so back
the file up first if you need them.

Logs: dlssnr-amd.log, and OptiScaler.log or ReShade.log, all in the game folder.

Known limits
------------
- DX9 games run entirely on DXVK; the frame rate can differ from native D3D9, and overlays such
  as Steam's may not show.
- In DX10 games the game's depth buffer is not passed to NR yet (not implemented).
- Games whose FSR runs in their own shaders never hand it to OptiScaler: pick the game's DLSS
  option instead, which OptiScaler offers.
- When video memory or system memory (including virtual memory) runs short, the network pauses
  by itself and the picture returns to normal until memory frees up. Set the page file to
  "System managed size".
- 64-bit package only.

Third-party components
----------------------
DXVK (zlib licence) comes from GE-Proton 11-7; the version is in dxvk\version.txt. Source:
https://github.com/doitsujin/dxvk . The licence files of ReShade 6.8.0, OptiScaler-NR 0.8.91,
the Vulkan Loader (with a description of the changes), vort_Shaders and DLSS5-Feeder are in
their folders.
