#pragma once
// In-memory fixes for OptiScaler-NR v0.8.4 and v0.8.91, applied from this module's DllMain -
// OptiScaler loads it during its own initialisation, before the game creates a Vulkan device or
// presents. OptiScaler's file on disk is not touched, and each fix is found by exact byte signatures
// (v0.8.91 matches the v0.8.4 ones for fixes 1, 3, 4 and has its own for 2, 5 and 7): any other build
// is left alone and the log says so. NR_OPTISCALER_FIX=0 turns them all off.
//
// 2. Finished Picture on DXVK/vkd3d-proton: see fix_finished_picture in nr_pe_optifix.cpp.
// 3. A 0 x 0 (window-sized) swapchain taken for an overlay and never wrapped (Helldivers 2): see
//    fix_window_sized_swapchain.
// 4. A D24S8 or D32S8 depth guide copied into a colour texture, which can hang the GPU under
//    vkd3d-proton (Helldivers 2, Kingdom Come: Deliverance II, S.T.A.L.K.E.R. 2): see
//    fix_depth_stencil_guide.
// 6. A feature released while the GPU still has its work - its timestamp heaps freed under a pending
//    list (S.T.A.L.K.E.R. 2's first FSR context): see release_hold.
// 5. IsHDR clear but AutoExposure set on a float colour (007 First Light, Helldivers 2): treated as
//    linear HDR so OptiScaler encodes it: see fix_autoexposure_hdr.
//
// 1. The device-extension query:
//
// The fork's vkCreateDevice hook asks which device extensions the physical device offers
// (DlssNr::VkExt::SupportedDeviceExtensions) through vkGetInstanceProcAddr on
// State::Instance().VulkanInstance - the most recently *created* instance, never cleared when that
// instance is destroyed. A game that creates and destroys several instances during startup
// (007 First Light: Streamline's adapter probe, AMD AGS's D3D11 device, DXVK/vkd3d factories) can
// reach the next vkCreateDevice with that handle already dead. The Linux Vulkan loader validates it
// ("vkGetInstanceProcAddr: Invalid instance") and aborts the process before a window opens.
//
// The fix passes a null instance to that one call. vkGetInstanceProcAddr(NULL, <device-level name>)
// returns NULL by the specification, the function returns an empty list, and the NR hook adds no
// extensions. What it added before on a Radeon was VK_KHR_buffer_device_address, a name for a
// feature that vkd3d-proton and DXVK already use as Vulkan 1.3 core; the NVX extensions it exists
// for are never available here. OptiScaler's file on disk is not touched.
//
// The code is found by an exact byte signature whose string operand must be
// "vkEnumerateDeviceExtensionProperties"; exactly one match is required, otherwise nothing is
// changed and the log says so. NR_OPTISCALER_FIX=0 turns it off.
namespace nr::pe {

void fix_optiscaler();

}  // namespace nr::pe
