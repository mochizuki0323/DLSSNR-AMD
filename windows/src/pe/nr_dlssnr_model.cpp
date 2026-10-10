// The DLSS-NR model behind both doors; see nr_dlssnr_model.hpp for which two and why.
//
// Everything here was windows/src/pe/nr_dlssnr_forwarder.cpp's anonymous namespace and the bodies of its
// dlssnr_call_create / dlssnr_call_evaluate_v2 / dlssnr_call_release and dlssnr_vk_* exports, lifted
// out so _nvngx.dll's NVSDK_NGX_*_CreateFeature(18) runs the same code rather than a copy of it. The
// comments that came with it are the record of why each line is what it is and travel with the code.
//
// The call sequence, argument order and return conventions were read from, and must keep matching:
//
//   OptiScaler_DLSSNR/OptiScaler/dlssnr/forwarder/dlssnr_forwarder.cpp   (the ABI)
//   OptiScaler_DLSSNR/OptiScaler/shaders/dlssnr/DlssNr_Dx12.cpp          (the D3D12 caller)
//   OptiScaler_DLSSNR/OptiScaler/dlssnr/DlssNrFeature_Vk.cpp             (the Vulkan caller)
//   artifacts/ref/wilsjo2-OptiScaler-DLSSNR-PreSR-Multipass/OptiScaler/dlssnr/DlssNr_Proxy.cpp
//   artifacts/ref/wilsjo2-OptiScaler-DLSSNR-PreSR-Multipass/OptiScaler/dlssnr/DlssNrFeature_Vk_Model.cpp
//   artifacts/ref/wilsjo2-OptiScaler-DLSSNR-PreSR-Multipass/OptiScaler/shaders/dlssnr/DlssNr_Dx12.cpp
//
// The division of labour is the same as the game module's: nr::pe::Session owns the device, the
// queue, the background build of nr::Runtime and the EngineFrame, and this file is only the model's
// shape and the control mapping. Nothing here duplicates windows/src/pe/nr_pe_session.cpp.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>

#include "nr_dlssnr_model.hpp"
#include "nr_pe_config.hpp"
#include "nr_pe_interop.hpp"
#include "nr_pe_log.hpp"
#include "nr_pe_session.hpp"
#include "nr_pe_vkdevice.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using nr::pe::log;

namespace nr::dlssnr {

// The per-feature staging buffers of the NR_DEBUG_READBACK=1 instrumentation, defined below with the
// rest of it. Null in every build and every run where the variable is not set.
struct DebugReadback;

// A feature is the handle, the controls and one id; everything expensive is the session's.
//
// The id is what makes this an independent object rather than a label. NVIDIA's DLL gives every
// feature its own temporal state and OptiScaler is built on that -- one feature per model pass, two
// extents alive at once, everything destroyed and recreated whenever a control moves -- so the id
// travels with every evaluate and the runtime keeps a history, a first-frame latch and a parity for
// each one. Creating a feature costs the id and, on its first evaluate, one image.
struct Feature {
    unsigned int width = 0;
    unsigned int height = 0;
    int preset = 0;
    int ui_correction = 1;
    nr::Controls controls;
    uint64_t id = 0;
    // For the rate-limited per-evaluate line: the first three and then every 300th.
    uint64_t evaluates = 0;
    // NR_DEBUG_READBACK=1 only; owned here, retired with the feature. See the readback section.
    DebugReadback* debug = nullptr;
};

namespace {

std::mutex g_lock;

// One session for the whole process per API, not one per NGX feature. A feature here is a handle and
// a set of controls; the expensive thing -- the device, the queue and a ~300 MB weight upload -- is
// the session's, and it survives OptiScaler tearing a feature down and building another at a new
// resolution (which it does whenever the working scale changes). Session::ensure_runtime already
// rebuilds the network for a new extent or format on its own.
//
// This is also why both entry points share this translation unit: two copies of these two pointers
// in one process would be two networks and twice the weights. A process that loads both DLLs is
// handled on top of that, in nr_ngx_core.cpp, which delegates to the forwarder's exports when the
// forwarder module is already there.
std::unique_ptr<nr::pe::Session> g_d3d12_session;
std::unique_ptr<nr::pe::Session> g_vk_session;

// OptiScaler runs its own resolve (the RenoDX-derived Resolve, model resolution, passes) after
// the model, exactly as it does after NVIDIA's DLL. So the model's answer must be the DLL's own
// picture and nothing more: the post block's tail with the saturated Intensity lerp, no transfer
// pass, no colour/model-scale controls of ours (RuntimeConfig::native_compose).
nr::pe::Session& d3d12_session() {
    if (!g_d3d12_session) {
        g_d3d12_session = std::make_unique<nr::pe::Session>(std::string());
        g_d3d12_session->set_native_compose(true);
    }
    return *g_d3d12_session;
}

nr::pe::Session& vk_session() {
    if (!g_vk_session) {
        g_vk_session = std::make_unique<nr::pe::Session>(std::string());
        g_vk_session->set_native_compose(true);
    }
    return *g_vk_session;
}

// [Preprocess] in dlssnr-amd.ini beside this module: the file re-read at most once a second, the
// hotkey every evaluate. Written with the defaults (off) when absent. OptiScaler knows nothing of it:
// the change is made on the proxy it hands us and taken back out of the answer before it gets it back.
nr::Preprocess preprocess_now() {
    static nr::pe::PreprocessFile file;
    static nr::pe::PreprocessConfig config;
    static nr::pe::PreprocessSwitch toggle;
    static ULONGLONG next = 0;
    const ULONGLONG now = GetTickCount64();
    if (now >= next) {
        next = now + 1000;
        const char* folder = nr::pe::module_folder();
        if (*folder) file.poll(std::string(folder) + "\\dlssnr-amd.ini", config);
    }
    return toggle.frame(config);
}

// Multi-pass, and why nothing here has to be given up for it.
//
// The multipass fork's Passes slider creates one NGX feature per pass and evaluates them in a chain,
// each with its own handle, tuning and reset flag (DlssNr_Dx12.cpp: g_nr.passFeature[], PassTuning,
// the ping-pong between g_nr.output and g_nr.passScratch; DlssNr_Proxy.cpp keeps every handle alive
// for the life of the pass). Every create here returns a distinct handle with a distinct id, and the
// runtime keeps one temporal history per id.
//
// This replaces a rule that used to live here: the first handle on a session "owned" the single
// history and every later one was evaluated with reset forced on, so passes 2..n ran as pure spatial
// enhancements and the history ended up holding the last pass's picture rather than the first's.
// That was a limitation of this side and of nothing in the reference, and it is gone: a feature is
// an independent object, exactly as it is in the DLL.

// The per-evaluate line, rate limited per feature: the first three, then every three hundredth. It
// is the only way to read "the model ran" or "it passed the frame through, and why" off a log
// without a debugger, and a user reporting that a control does nothing is asking exactly that.
void log_evaluate(Feature* f, const char* api, unsigned int w, unsigned int h, const void* in,
                  const void* out, bool reset, bool ran, const char* why, float gpu_ms, float network_ms) {
    const uint64_t n = ++f->evaluates;
    if (n > 3 && n % 300 != 0) return;
    // owner is always 1 now: every feature owns its own temporal history.
    // `gpu_ms` is the pass's own cost from the GPU's timestamps, `network_ms` the network's part of
    // it (the rest is the work around it: input, composition, copies); zero means no result yet.
    char cost[64] = "";
    if (gpu_ms > 0.0f && network_ms > 0.0f)
        std::snprintf(cost, sizeof cost, ", %.2f ms (network %.2f, route %.2f)", gpu_ms, network_ms,
                      gpu_ms > network_ms ? gpu_ms - network_ms : 0.0f);
    else if (gpu_ms > 0.0f) std::snprintf(cost, sizeof cost, ", %.2f ms", gpu_ms);
    log("[nr] evaluate #%u (%s, %u) %ux%u in=%p out=%p reset=%d owner=1%s -> %s%s%s",
        static_cast<unsigned>(f->id), api, static_cast<unsigned>(n), w, h, in,
        out, reset ? 1 : 0, cost, ran ? "ran" : "passthrough(", ran ? "" : (why ? why : "unknown"),
        ran ? "" : ")");
}

// The six NGX controls, mapped one to one onto nr::Controls.
//
// Three fields of nr::Controls have no NGX counterpart and are left at their defaults, because the
// surface carries no key for them:
//   - `colour`  : OptiScaler applies its own ColourStrength in its resolve shader, after us.
//   - `passes`  : OptiScaler runs its own passes as separate features; see the multipass note.
//   - `apply_model`: there is no "run but discard" in the NGX surface.
nr::Controls controls_from(const Controls6& in) {
    nr::Controls c;
    c.enabled = true;
    c.intensity = in.intensity;
    c.style = in.style;
    c.local_structure = in.local_structure;
    c.local_tone = in.local_tone;
    c.skin_structure = in.skin_structure;
    c.automatic_mask = in.use_auto_mask != 0;
    return c;
}

// The motion vectors times DLSSNR.MVecScale are in pixels of the motion texture's own region (its
// subrect, else the whole allocation), and nr::EngineFrame wants normalized screen units: the same
// number over that extent. Not over the colour's: after the upscaler the model runs on the display
// image while the vectors stay at render resolution, and OptiScaler passes the game's scale through
// untouched on purpose ("every resource already carries a subrect saying how big it is", its
// DlssNr_Dx12_Run.cpp). Dividing by the colour width made them half as long at 4K with FSR
// Performance; before the upscaler the two extents are the same and nothing changes.
float normalized_mv_scale(float ngx_scale, unsigned int extent) {
    return extent != 0 ? ngx_scale / static_cast<float>(extent) : ngx_scale;
}

// Which vtable slot the host settled on for floats.
//
// It starts at the MSVC slot our own block publishes (6, see nr_ngx_abi.hpp), which is the same slot
// NVIDIA's driver block keeps floats in, so OptiScaler's probe and this default agree. The setter is
// still honoured: dlssnr_call_set_float_slot is part of the forwarder ABI, and a host that decided on
// a different slot is telling us something about the block it handed over, not about us.
int g_float_slot = kNgxSlotSetFloat;

// Driving the caller's parameter block, exactly as the reference forwarder drives NVIDIA's: index the
// vtable by hand and call. The slot constants are the reference's own (dlssnr_forwarder.cpp
// VT_SET_ULL 0, VT_SET_UINT 3) and are now the same numbers nr_ngx_abi.hpp derives from MSVC's layout
// -- slot 0 is Set(void*), which is what a resource handle is, and slot 3 is Set(int).
using PFN_SetFloat = void (NR_NGX_MSABI*)(void*, const char*, float);
using PFN_SetUInt = void (NR_NGX_MSABI*)(void*, const char*, unsigned int);
using PFN_SetPtr = void (NR_NGX_MSABI*)(void*, const char*, void*);

void* const* vtable_of(void* params) { return *reinterpret_cast<void* const* const*>(params); }

void set_uint(void* params, const char* name, unsigned int v) {
    if (!params) return;
    reinterpret_cast<PFN_SetUInt>(vtable_of(params)[kNgxSlotSetInt])(params, name, v);
}

void set_float(void* params, const char* name, float v) {
    if (!params) return;
    reinterpret_cast<PFN_SetFloat>(vtable_of(params)[g_float_slot])(params, name, v);
}

void set_resource(void* params, const char* name, void* v) {
    if (!params) return;
    reinterpret_cast<PFN_SetPtr>(vtable_of(params)[kNgxSlotSetVoidPtr])(params, name, v);
}

// The last thing that went wrong, for dlssnr_call_error and for the core's own logging. The reference
// keeps this thread-local; ours is under g_lock with everything else, because a Session's status is
// process-wide.
std::string g_last_error;

// The Vulkan device the caller initialised us on, and whether a submittable queue was found for it.
nr::pe::DeviceHandles g_vk_handles;
bool g_vk_queue_known = false;

// One image copy with the barriers around it, for moving our answer into the caller's output image.
// OptiScaler's Vulkan path leaves its output in VK_IMAGE_LAYOUT_GENERAL immediately before evaluate
// (DlssNrFeature_Vk.cpp: Transition(cmdBuffer, g_vk.output, VK_IMAGE_LAYOUT_GENERAL)), and expects it
// back in GENERAL, so that is what this assumes and restores.
void copy_into_output(VkCommandBuffer cmd, VkImage src, VkImageLayout src_layout, VkImage dst,
                      uint32_t w, uint32_t h) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    // The source is whatever the caller left it in: GENERAL for our own output coming out of
    // Session::run_vulkan, SHADER_READ_ONLY_OPTIMAL for OptiScaler's proxy on the pass-through. Both
    // have to be restored, because neither image is ours to relabel.
    b.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.oldLayout = src_layout;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.image = src;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &b);

    VkImageCopy copy{};
    copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.extent = {w, h, 1};
    vkCmdCopyImage(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_GENERAL, 1,
                   &copy);

    std::swap(b.oldLayout, b.newLayout);
    b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &b);

    VkImageMemoryBarrier w2r{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    w2r.srcQueueFamilyIndex = w2r.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    w2r.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    w2r.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    w2r.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    w2r.oldLayout = w2r.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    w2r.image = dst;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &w2r);
}

// Pull a VkImage and its extent out of an NVSDK_NGX_Resource_VK. Buffers are not images and are
// declined rather than reinterpreted.
bool image_of(const void* resource, VkImage* image, VkFormat* format, uint32_t* width,
              uint32_t* height) {
    if (!resource) return false;
    const auto* r = static_cast<const NVSDK_NGX_Resource_VK*>(resource);
    if (r->Type != NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW) return false;
    const auto& info = r->Resource.ImageViewInfo;
    if (!info.Image) return false;
    *image = reinterpret_cast<VkImage>(info.Image);
    *format = static_cast<VkFormat>(info.Format);
    *width = info.Width;
    *height = info.Height;
    return true;
}

void barrier(ID3D12GraphicsCommandList* cmd, ID3D12Resource* res, D3D12_RESOURCE_STATES from,
             D3D12_RESOURCE_STATES to) {
    if (!res || from == to) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    b.Transition.pResource = res;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    cmd->ResourceBarrier(1, &b);
}

nr::pe::Session::Subrect to_session(const Rect& r) { return {r.x, r.y, r.width, r.height}; }

}  // namespace

// ---------------------------------------------------------------------------------------------
// Instrumentation: NR_DEBUG_READBACK=1 (2026-09-16). Off unless the variable is set; when it is
// unset not one byte is allocated, recorded or mapped by any of this.
//
// "The second pass does nothing" is a claim about pixels, and nothing in this log has ever carried a
// pixel. Both features report `-> ran` with distinct buffers and the host's "Model output (raw)" view
// still shows the untouched frame, which is three different faults wearing one face: the network did
// nothing, or it wrote somewhere else, or the host is looking at the wrong image. Only numbers off
// the buffer we were handed can tell them apart.
//
// So: inside the caller's own command list, copy the output twice -- once immediately after the seed
// (where it is a byte copy of the colour input, so it *is* the input) and once after run_after has
// recorded the network -- into two readback buffers, and score the pair. Two copies of one resource
// rather than colour-then-output on purpose: CopyResource has already proved the formats identical,
// so the difference cannot be a decode mismatch, and a seed that did not happen shows up as an input
// that is not the picture rather than as a spurious difference.
//
// ORDERING. There is no fence here, because there is no queue here: this path is only ever given an
// ID3D12GraphicsCommandList, the caller owns the ID3D12CommandQueue and never passes it, and a fence
// signalled on a queue of ours would order nothing on theirs. The ordering is therefore the
// retirement of the *next* evaluate of the same feature: the pair recorded at evaluate n is mapped at
// the top of evaluate n+1, by which point the list holding it has been closed, executed and -- for
// the sampled evaluates 300, 600, ... -- three hundred frames old. For evaluates 1..3 the gap is one
// frame, which a game with frames in flight may not have retired yet, so those three lines can in
// principle read a buffer mid-copy. They are kept because the first frames are the interesting ones
// and a torn read of a *constant* region still reads the constant; treat 1..3 as indicative and the
// 300th as authoritative.
// ---------------------------------------------------------------------------------------------
namespace {

// Read once, exactly like NR_DEBUG_ALWAYS_RESET above.
bool debug_readback_on() {
    static const bool on = [] {
        char v[8] = {};
        const DWORD n = GetEnvironmentVariableA("NR_DEBUG_READBACK", v, sizeof v);
        const bool e = (n > 0 && v[0] == '1');
        if (e) log("[nr] NR_DEBUG_READBACK=1: the model's input and output are copied back and scored");
        return e;
    }();
    return on;
}

// NR_DEBUG_DUMP=1 (2026-09-16, needs NR_DEBUG_READBACK=1): the first scored pair of every feature is
// also written next to the log as two 8-bit PPMs, dlssnr-amd-f<id>-e<n>-{in,out}.ppm, so the
// picture the second pass was given and the picture it returned can be looked at, not just scored.
bool debug_dump_on() {
    static const bool on = [] {
        char v[8] = {};
        const DWORD n = GetEnvironmentVariableA("NR_DEBUG_DUMP", v, sizeof v);
        const bool e = (n > 0 && v[0] == '1');
        if (e) log("[nr] NR_DEBUG_DUMP=1: the first readback pair of every feature is written as PPM");
        return e;
    }();
    return on;
}

// NR_DEBUG_SKIP_PASS=k (2026-09-16): the k-th live D3D12 feature in creation order records nothing;
// its output is the seed copy of its colour and nothing else. OptiScaler creates one feature per
// pass, in pass order, and keeps them all alive, so k is the pass number. With passes=2: k=2 leaves
// pass 1's answer (through the clamp) as the final answer -- if the picture is then the neural
// one, recording the second network is what breaks the frame; k=1 hands pass 2 the untouched
// proxy -- if the picture is then neural, pass 2's answer does reach the resolve.
int debug_skip_pass() {
    static const int k = [] {
        char v[8] = {};
        const DWORD n = GetEnvironmentVariableA("NR_DEBUG_SKIP_PASS", v, sizeof v);
        const int e = (n > 0) ? std::atoi(v) : 0;
        if (e > 0) log("[nr] NR_DEBUG_SKIP_PASS=%d: live D3D12 feature number %d records no network", e, e);
        return e;
    }();
    return k;
}

// NR_DEBUG_NOOP=1: every D3D12 evaluate returns at once - no seed copy, no barrier, nothing
// recorded - so a fault that remains is in the caller's own work around the call, not in ours.
bool debug_noop() {
    static const bool k = [] {
        char v[8] = {};
        const DWORD n = GetEnvironmentVariableA("NR_DEBUG_NOOP", v, sizeof v);
        const bool on = n > 0 && v[0] == '1';
        if (on) log("[nr] NR_DEBUG_NOOP=1: D3D12 evaluates record nothing at all");
        return on;
    }();
    return k;
}

// NR_DEBUG_DEPTH=off: the depth guide is not handed on, so the motion vector is read at the pixel
// itself; NR_DEBUG_DEPTH=flip: the caller's DepthInverted is taken the other way round. For telling a
// depth-related artefact from one that has nothing to do with depth, in the same build.
enum class DebugDepth { Normal, Off, Flip };
DebugDepth debug_depth() {
    static const DebugDepth k = [] {
        char v[8] = {};
        const DWORD n = GetEnvironmentVariableA("NR_DEBUG_DEPTH", v, sizeof v);
        const std::string s = n > 0 && n < sizeof v ? std::string(v, n) : std::string();
        if (s == "off") { log("[nr] NR_DEBUG_DEPTH=off: the depth guide is not used"); return DebugDepth::Off; }
        if (s == "flip") { log("[nr] NR_DEBUG_DEPTH=flip: DepthInverted is taken the other way round"); return DebugDepth::Flip; }
        return DebugDepth::Normal;
    }();
    return k;
}

// Live D3D12 features in creation order, for the pass number above only.
std::vector<Feature*> g_live_d3d12;

// Bytes per pixel of the formats a DLSS-NR output is ever seen in; 0 is "this tool cannot read it".
// R8G8B8A8_TYPELESS and _UNORM_SRGB decode byte-identically to _UNORM (the numbers are then in the
// encoded space, which is the space the difference is being judged in anyway), so they are taken too.
unsigned readback_bpp(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R11G11B10_FLOAT: return 4;
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: return 8;
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: return 4;
        default: return 0;
    }
}

float bits_to_float(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

float half_to_float(uint16_t h) {
    const uint32_t s = static_cast<uint32_t>(h >> 15) & 1u;
    const uint32_t e = static_cast<uint32_t>(h >> 10) & 0x1Fu;
    const uint32_t m = static_cast<uint32_t>(h) & 0x3FFu;
    if (e == 0) return (s ? -1.0f : 1.0f) * static_cast<float>(m) * (1.0f / 1024.0f) * 6.103515625e-05f;
    if (e == 31) return bits_to_float((s << 31) | 0x7F800000u | (m << 13));
    return bits_to_float((s << 31) | ((e - 15u + 127u) << 23) | (m << 13));
}

// The two unsigned floats of R11G11B10: 5-bit exponent, 6- and 5-bit mantissa, no sign bit.
float f11_to_float(uint32_t v) {
    const uint32_t e = (v >> 6) & 0x1Fu, m = v & 0x3Fu;
    if (e == 0) return static_cast<float>(m) * (1.0f / 64.0f) * 6.103515625e-05f;
    if (e == 31) return bits_to_float(0x7F800000u | (m << 17));
    return bits_to_float(((e - 15u + 127u) << 23) | (m << 17));
}

float f10_to_float(uint32_t v) {
    const uint32_t e = (v >> 5) & 0x1Fu, m = v & 0x1Fu;
    if (e == 0) return static_cast<float>(m) * (1.0f / 32.0f) * 6.103515625e-05f;
    if (e == 31) return bits_to_float(0x7F800000u | (m << 18));
    return bits_to_float(((e - 15u + 127u) << 23) | (m << 18));
}

void decode_rgb(const uint8_t* p, DXGI_FORMAT f, float out[3]) {
    switch (f) {
        case DXGI_FORMAT_R11G11B10_FLOAT: {
            uint32_t v = 0;
            std::memcpy(&v, p, 4);
            out[0] = f11_to_float(v & 0x7FFu);
            out[1] = f11_to_float((v >> 11) & 0x7FFu);
            out[2] = f10_to_float((v >> 22) & 0x3FFu);
            break;
        }
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: {
            uint16_t h[3] = {};
            std::memcpy(h, p, sizeof h);
            for (int i = 0; i < 3; ++i) out[i] = half_to_float(h[i]);
            break;
        }
        default:   // the three 8-bit RGBA layouts
            for (int i = 0; i < 3; ++i) out[i] = static_cast<float>(p[i]) * (1.0f / 255.0f);
            break;
    }
}

double luma_of(const float c[3]) {
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];   // Rec.709
}

struct ReadbackStats {
    double in_luma = 0, out_luma = 0;
    double mean_abs[3] = {0, 0, 0};
    double changed = 0;    // fraction, 0..1
    double max_abs = 0;
    double in_max = 0;       // largest input channel: a float frame can exceed 1.0, an 8-bit one cannot
    uint64_t in_over1 = 0;   // samples with any input channel above 1.0
    uint64_t samples = 0;
    uint64_t nonfinite = 0;
};

// Staging buffers live past the feature that owns them, because the copy that fills them is in a
// command list the GPU may not have reached. The runtime already defers a feature's images by 64
// recordings for exactly this reason (see release_d3d12); this is the same rule with its own list,
// counted in evaluates of any feature. Only ever non-empty when the switch is on.
struct RetiredBuffer {
    ID3D12Resource* res = nullptr;
    uint64_t deadline = 0;
};
std::vector<RetiredBuffer> g_retired;
uint64_t g_readback_recordings = 0;
constexpr uint64_t kReadbackRetireAfter = 64;

void retire_buffer(ID3D12Resource* res) {
    if (!res) return;
    g_retired.push_back({res, g_readback_recordings + kReadbackRetireAfter});
}

void sweep_retired() {
    size_t keep = 0;
    for (size_t i = 0; i < g_retired.size(); ++i) {
        if (g_retired[i].deadline <= g_readback_recordings) {
            g_retired[i].res->Release();
        } else {
            g_retired[keep++] = g_retired[i];
        }
    }
    g_retired.resize(keep);
}

}  // namespace

// The two buffers and the footprint that describes them. One per feature, sized from the region that
// is actually scored rather than from the whole image: the centre quarter by area is the centre half
// of each axis, so the copy, the memory and the PCIe traffic are all a quarter of the frame.
struct DebugReadback {
    ID3D12Resource* in = nullptr;    // the output right after the seed copy == the model's input
    ID3D12Resource* out = nullptr;   // the output after run_after recorded the network
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    uint64_t bytes = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    UINT box_x = 0, box_y = 0;       // where the footprint sits in the source image
    bool pending = false;            // a pair is recorded and not yet mapped
    uint64_t pending_n = 0;          // which evaluate of this feature recorded it
    bool have_stats = false;         // stats below belong to stats_n and have not been logged
    uint64_t stats_n = 0;
    ReadbackStats stats{};
    bool dumped = false;             // NR_DEBUG_DUMP=1: the one pair written as PPM
};

namespace {

void readback_free(DebugReadback& rb) {
    retire_buffer(rb.in);
    retire_buffer(rb.out);
    rb.in = rb.out = nullptr;
    rb.pending = false;
    rb.bytes = 0;
}

ID3D12Resource* make_readback_buffer(ID3D12Device* device, uint64_t bytes) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    heap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap.CreationNodeMask = 1;
    heap.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ID3D12Resource* res = nullptr;
    const HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                       D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                       IID_PPV_ARGS(&res));
    if (FAILED(hr)) {
        static bool said = false;
        if (!said) {
            said = true;
            log("[nr] readback: could not allocate a %llu byte readback buffer (0x%08lx)",
                static_cast<unsigned long long>(bytes), static_cast<unsigned long>(hr));
        }
        return nullptr;
    }
    return res;
}

// Size the pair against this frame's output. Returns false when the format is one this cannot decode
// or the allocation failed, in which case nothing is recorded and nothing is logged.
bool readback_ensure(ID3D12Device* device, DebugReadback& rb, ID3D12Resource* tex) {
    const D3D12_RESOURCE_DESC d = tex->GetDesc();
    if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) return false;
    if (readback_bpp(d.Format) == 0) {
        static bool said = false;
        if (!said) {
            said = true;
            log("[nr] readback: unsupported format %u", static_cast<unsigned>(d.Format));
        }
        return false;
    }

    // NR_DEBUG_DUMP_FULL=1: the whole frame rather than the middle quarter.
    //
    // **The crop is not a smaller version of the frame.** The model's bottleneck
    // attends over the whole image, so its answer for the middle of a 4K frame
    // is not the answer it gives that crop on its own - measured at 3.2 points
    // of mean luminance, and the in-game gap being chased is 15. A centre crop
    // therefore cannot be replayed offline against anything, and every
    // comparison made from one has had to be thrown away. Costs 4x the readback
    // (two 32 MB buffers at 4K) and is off by default for that reason.
    static const bool full = [] {
        char v[8] = {}; const DWORD n = GetEnvironmentVariableA("NR_DEBUG_DUMP_FULL", v, sizeof v);
        const bool on = n > 0 && v[0] == '1';
        if (on) log("[nr] NR_DEBUG_DUMP_FULL=1: the readback pair is the whole frame");
        return on;
    }();
    const UINT w = static_cast<UINT>(d.Width), h = static_cast<UINT>(d.Height);
    UINT bx = 0, by = 0, bw = w, bh = h;
    if (!full && w >= 8 && h >= 8) { bx = w / 4; by = h / 4; bw = w / 2; bh = h / 2; }

    if (rb.in && rb.out && rb.format == d.Format && rb.box_x == bx && rb.box_y == by &&
        rb.fp.Footprint.Width == bw && rb.fp.Footprint.Height == bh)
        return true;

    // The extent or the format moved under us (OptiScaler recreates a feature for that, but a
    // debugging switch does not get to assume it). The old pair may still be a copy destination in a
    // list in flight, so it is retired rather than released.
    readback_free(rb);

    D3D12_RESOURCE_DESC sub = d;
    sub.Width = bw;
    sub.Height = bh;
    sub.DepthOrArraySize = 1;
    sub.MipLevels = 1;
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    device->GetCopyableFootprints(&sub, 0, 1, 0, &fp, &rows, &row_bytes, &total);
    if (total == 0) return false;

    rb.in = make_readback_buffer(device, total);
    rb.out = rb.in ? make_readback_buffer(device, total) : nullptr;
    if (!rb.in || !rb.out) {
        readback_free(rb);
        return false;
    }
    rb.fp = fp;
    rb.bytes = total;
    rb.format = d.Format;
    rb.box_x = bx;
    rb.box_y = by;
    log("[nr] readback: scoring %ux%u at (%u,%u) of %ux%u fmt %u, %llu bytes x2",
        bw, bh, bx, by, w, h, static_cast<unsigned>(d.Format),
        static_cast<unsigned long long>(total));
    return true;
}

// One CopyTextureRegion with the state round trip around it. `state` is the state the caller holds
// the resource in across the evaluate and the state it is put back into, so nothing downstream can
// tell this happened.
void readback_record(ID3D12GraphicsCommandList* cmd, const DebugReadback& rb, ID3D12Resource* src,
                     D3D12_RESOURCE_STATES state, ID3D12Resource* dst) {
    if (!dst) return;
    barrier(cmd, src, state, D3D12_RESOURCE_STATE_COPY_SOURCE);

    D3D12_TEXTURE_COPY_LOCATION dl{};
    dl.pResource = dst;
    dl.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dl.PlacedFootprint = rb.fp;

    D3D12_TEXTURE_COPY_LOCATION sl{};
    sl.pResource = src;
    sl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    sl.SubresourceIndex = 0;

    D3D12_BOX box{};
    box.left = rb.box_x;
    box.top = rb.box_y;
    box.front = 0;
    box.right = rb.box_x + rb.fp.Footprint.Width;
    box.bottom = rb.box_y + rb.fp.Footprint.Height;
    box.back = 1;
    cmd->CopyTextureRegion(&dl, 0, 0, 0, &sl, &box);

    barrier(cmd, src, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
}

// Map the pair recorded one evaluate ago and score it over every 16th pixel of the copied region.
// NR_DEBUG_DUMP=1: one mapped readback buffer as a binary PPM beside the log, values clamped to
// [0,1] and scaled to 8 bits with no transfer curve (the proxy is display-referred already).
void dump_ppm(const DebugReadback& rb, const uint8_t* base, uint64_t feature_id, const char* which) {
    std::string path = nr::pe::log_path() ? nr::pe::log_path() : "";
    const size_t cut = path.find_last_of("\\/");
    path = (cut == std::string::npos) ? std::string() : path.substr(0, cut + 1);
    char name[96];
    std::snprintf(name, sizeof name, "dlssnr-amd-f%u-e%u-%s.ppm", static_cast<unsigned>(feature_id),
                  static_cast<unsigned>(rb.pending_n), which);
    path += name;
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) { log("[nr] dump: cannot write %s", path.c_str()); return; }
    const UINT w = rb.fp.Footprint.Width, h = rb.fp.Footprint.Height;
    const size_t pitch = rb.fp.Footprint.RowPitch;
    const unsigned bpp = readback_bpp(rb.format);
    std::fprintf(fp, "P6\n%u %u\n255\n", static_cast<unsigned>(w), static_cast<unsigned>(h));
    std::vector<uint8_t> row(static_cast<size_t>(w) * 3);
    for (UINT y = 0; y < h; ++y) {
        const uint8_t* src = base + static_cast<size_t>(y) * pitch;
        for (UINT x = 0; x < w; ++x) {
            float c[3];
            decode_rgb(src + static_cast<size_t>(x) * bpp, rb.format, c);
            for (int k = 0; k < 3; ++k) {
                const float v = std::isfinite(c[k]) ? std::clamp(c[k], 0.0f, 1.0f) : 0.0f;
                row[static_cast<size_t>(x) * 3 + k] = static_cast<uint8_t>(v * 255.0f + 0.5f);
            }
        }
        std::fwrite(row.data(), 1, row.size(), fp);
    }
    std::fclose(fp);
    log("[nr] dump: wrote %s", path.c_str());
}

bool readback_drain(DebugReadback& rb, uint64_t feature_id) {
    if (!rb.pending || !rb.in || !rb.out) return false;
    rb.pending = false;

    const uint8_t* a = nullptr;
    const uint8_t* b = nullptr;
    D3D12_RANGE all{0, static_cast<SIZE_T>(rb.bytes)};
    const D3D12_RANGE none{0, 0};
    if (FAILED(rb.in->Map(0, &all, reinterpret_cast<void**>(const_cast<uint8_t**>(&a)))) || !a)
        return false;
    if (FAILED(rb.out->Map(0, &all, reinterpret_cast<void**>(const_cast<uint8_t**>(&b)))) || !b) {
        rb.in->Unmap(0, &none);
        return false;
    }

    // EVERY sampled pair whose input carries a picture, overwriting the last (pair 1 of a feature is
    // often mapped before the GPU reached it and reads as zeros).
    //
    // It used to write the first pair only, and the first pair is worthless: the readback samples
    // evaluates 1, 2, 3 and then every 300th, and at evaluate 2-3 the network is usually still
    // building, so the pair on disk was a passthrough - input and output identical, 0.00% changed.
    // Overwriting means the files left after a twenty-second run are from evaluate 300 or 600, which
    // is the picture the model is actually being given in the game. That input is the one thing the
    // offline harness cannot synthesize: OptiScaler's encoded proxy, with the game's own exposure
    // and white point in it.
    if (debug_dump_on() && rb.pending_n >= 2) {
        rb.dumped = true;
        dump_ppm(rb, a, feature_id, "in");
        dump_ppm(rb, b, feature_id, "out");
    }

    const UINT w = rb.fp.Footprint.Width, h = rb.fp.Footprint.Height;
    const size_t pitch = rb.fp.Footprint.RowPitch;
    const unsigned bpp = readback_bpp(rb.format);
    ReadbackStats s{};
    for (UINT y = 0; y < h; y += 16) {
        const uint8_t* ra = a + static_cast<size_t>(y) * pitch;
        const uint8_t* rbrow = b + static_cast<size_t>(y) * pitch;
        for (UINT x = 0; x < w; x += 16) {
            float ci[3], co[3];
            decode_rgb(ra + static_cast<size_t>(x) * bpp, rb.format, ci);
            decode_rgb(rbrow + static_cast<size_t>(x) * bpp, rb.format, co);
            bool finite = true;
            for (int k = 0; k < 3; ++k)
                if (!std::isfinite(ci[k]) || !std::isfinite(co[k])) finite = false;
            if (!finite) { ++s.nonfinite; continue; }
            const double cmax = std::max({double(ci[0]), double(ci[1]), double(ci[2])});
            if (cmax > s.in_max) s.in_max = cmax;
            if (cmax > 1.0) ++s.in_over1;
            s.in_luma += luma_of(ci);
            s.out_luma += luma_of(co);
            double biggest = 0;
            for (int k = 0; k < 3; ++k) {
                const double d = std::fabs(static_cast<double>(co[k]) - static_cast<double>(ci[k]));
                s.mean_abs[k] += d;
                if (d > biggest) biggest = d;
            }
            if (biggest > s.max_abs) s.max_abs = biggest;
            if (biggest > 1.0 / 255.0) s.changed += 1.0;
            ++s.samples;
        }
    }
    rb.in->Unmap(0, &none);
    rb.out->Unmap(0, &none);

    if (s.samples) {
        const double inv = 1.0 / static_cast<double>(s.samples);
        s.in_luma *= inv;
        s.out_luma *= inv;
        for (int k = 0; k < 3; ++k) s.mean_abs[k] *= inv;
        s.changed *= inv;
    }
    rb.stats = s;
    rb.stats_n = rb.pending_n;
    rb.have_stats = true;
    return true;
}

// The one line, emitted immediately after the per-evaluate line of the evaluate that drained it.
void readback_log(const Feature* f, DebugReadback& rb) {
    if (!rb.have_stats) return;
    rb.have_stats = false;
    const ReadbackStats& s = rb.stats;
    log("[nr] readback #%u (%u) in luma %.4f out luma %.4f mean|d| %.5f %.5f %.5f changed %.2f%% "
        "max %.5f",
        static_cast<unsigned>(f->id), static_cast<unsigned>(rb.stats_n), s.in_luma, s.out_luma,
        s.mean_abs[0], s.mean_abs[1], s.mean_abs[2], s.changed * 100.0, s.max_abs);
    log("[nr] readback #%u (%u) input max %.4f, %.3f%% of samples above 1.0", static_cast<unsigned>(f->id),
        static_cast<unsigned>(rb.stats_n), s.in_max,
        s.samples ? 100.0 * double(s.in_over1) / double(s.samples) : 0.0);
    if (s.nonfinite)
        log("[nr] readback #%u (%u) %llu of %llu samples were not finite and were dropped",
            static_cast<unsigned>(f->id), static_cast<unsigned>(rb.stats_n),
            static_cast<unsigned long long>(s.nonfinite),
            static_cast<unsigned long long>(s.nonfinite + s.samples));
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// The parameter block.
// ---------------------------------------------------------------------------------------------

void set_float_slot(int slot) {
    if (slot >= 0 && slot < 8) g_float_slot = slot;
}

void probe_float(void* params, const char* name, float value, int slot) {
    if (!params || slot < 0 || slot >= 8) return;
    void** vt = *reinterpret_cast<void***>(params);
    reinterpret_cast<PFN_SetFloat>(vt[slot])(params, name, value);
}

const char* last_error() { return g_last_error.c_str(); }

// The controls, written into the block under the model's own names. Nothing on this side reads them
// back -- our network takes them as arguments -- but the reference writes them here and OptiScaler
// reads them back to report what landed (DlssNr_Dx12.cpp, "DLSS-NR readback ..."). A block that
// answers 0xBAD00000 to every one of those is indistinguishable from a broken integration, so they
// are written.
void write_create_keys(void* params, unsigned int width, unsigned int height, int preset,
                       int ui_correction, const Controls6& c) {
    if (!params) return;
    set_uint(params, "DLSSNR.Enabled", 1u);
    set_uint(params, "DLSSNR.Width", width);
    set_uint(params, "DLSSNR.Height", height);
    set_uint(params, "DLSSNR.Hint.Render.Preset", static_cast<unsigned int>(preset));
    set_uint(params, "DLSSNR.UICorrection", static_cast<unsigned int>(ui_correction));
    set_float(params, "DLSSNR.Intensity", c.intensity);
    set_uint(params, "DLSSNR.Style", static_cast<unsigned int>(c.style));
    set_float(params, "DLSSNR.LocalStructureStrength", c.local_structure);
    set_float(params, "DLSSNR.LocalToneStrength", c.local_tone);
    set_float(params, "DLSSNR.SkinStructureStrength", c.skin_structure);
    // An explicit ControlMask takes precedence over the automatic one, and the block outlives every
    // feature, so it is cleared rather than left holding someone else's resource.
    set_resource(params, "DLSSNR.ControlMask", nullptr);
    set_uint(params, "DLSSNR.UseAutoMask", c.use_auto_mask != 0 ? 1u : 0u);
}

void write_evaluate_keys(void* params, const Controls6& c, bool reset, int depth_inverted) {
    if (!params) return;
    set_float(params, "DLSSNR.Intensity", c.intensity);
    set_uint(params, "DLSSNR.Style", static_cast<unsigned int>(c.style));
    set_float(params, "DLSSNR.LocalStructureStrength", c.local_structure);
    set_float(params, "DLSSNR.LocalToneStrength", c.local_tone);
    set_float(params, "DLSSNR.SkinStructureStrength", c.skin_structure);
    set_resource(params, "DLSSNR.ControlMask", nullptr);
    set_uint(params, "DLSSNR.UseAutoMask", c.use_auto_mask != 0 ? 1u : 0u);
    set_uint(params, "DLSSNR.Reset", reset ? 1u : 0u);
    set_uint(params, "DLSSNR.DepthInverted", static_cast<unsigned int>(depth_inverted));
}

// ---------------------------------------------------------------------------------------------
// Direct3D 12: through vkd3d-proton, or the native runtime through the bridge.
// ---------------------------------------------------------------------------------------------

// Creates the persistent feature. Unlike the reference this records nothing into `cmd`: NGX builds
// its feature on the command list and so must outlive its execution, whereas nr::Runtime is built on
// the session's own queue on a background thread.
Feature* create_d3d12(ID3D12Device* device, ID3D12GraphicsCommandList* cmd, void* params,
                      unsigned int width, unsigned int height, int preset, int ui_correction,
                      const Controls6& controls, int* out_result) {
    std::lock_guard<std::mutex> guard(g_lock);
    g_last_error.clear();
    auto fail = [&](int code) -> Feature* {
        if (out_result) *out_result = code;
        return nullptr;
    };

    if (!device || !cmd) return fail(static_cast<int>(NVSDK_NGX_Result_FAIL_InvalidParameter));

    // Under vkd3d-proton the network records into the game's own command buffer, which has to be
    // there; on the native runtime it runs on the bridge (nr_pe_bridge.hpp) and needs neither.
    const bool bridged = d3d12_session().bridged(device);
    if (!bridged && nr::pe::command_buffer(cmd) == VK_NULL_HANDLE) {
        g_last_error = "the command list exposes no VkCommandBuffer";
        log("[nr] model create (D3D12): %s", g_last_error.c_str());
        return fail(static_cast<int>(NVSDK_NGX_Result_FAIL_PlatformError));
    }

    auto* feature = new Feature();
    feature->width = width;
    feature->height = height;
    feature->preset = preset;
    feature->ui_correction = ui_correction;
    feature->controls = controls_from(controls);

    write_create_keys(params, width, height, preset, ui_correction, controls);

    // Construct the session now so its log line lands at create, not mid-frame. It is one object for
    // the whole process and it keeps every network it has built, so this create rebuilds nothing:
    // a feature at an extent that has been seen before starts running on its first evaluate.
    feature->id = d3d12_session().create_feature();
    g_live_d3d12.push_back(feature);

    if (out_result) *out_result = static_cast<int>(NVSDK_NGX_Result_Success);
    log("[nr] model create (D3D12): feature #%u, %ux%u, preset %d, intensity %.3f, style %d",
        static_cast<unsigned>(feature->id), width, height, preset, controls.intensity,
        controls.style);
    return feature;
}

// The evaluate. OptiScaler's contract is output = model(color); ours is a network that enhances an
// image in place, so the two are bridged by seeding the output with the colour and enhancing that.
//
// The seed is a D3D12 CopyResource rather than a hand-written vkCmdCopyImage: the resources are the
// caller's and their D3D12 states are known exactly (DlssNr_Dx12.cpp leaves the model input in
// NON_PIXEL_SHADER_RESOURCE and the output in UNORDERED_ACCESS across the call), so letting vkd3d
// place the barriers is both shorter and the only version that cannot disagree with what vkd3d
// believes the layouts are.
//
// It also means the pass degrades to a pass-through: while the network is still building on its
// background thread, the output is a byte copy of the input and OptiScaler's resolve composites
// exactly the proxy it started with.
//
// Colour and output are at the working resolution; depth and motion come from the game's own upscaler
// call and may be at a different one, so each carries its own subrect base and extent, and the motion
// vectors arrive already scaled by the working/active ratio the caller computed
// (DlssNr_Dx12.cpp: guideMvScaleX * workWidth/width). The reference forwarder computes no ratio of its
// own -- it writes the number it is given straight into DLSSNR.MVecScaleX -- and neither does this.
int evaluate_d3d12(ID3D12GraphicsCommandList* cmd, Feature* f, void* params, ID3D12Resource* color,
                   ID3D12Resource* depth, ID3D12Resource* motion, ID3D12Resource* output,
                   unsigned int width, unsigned int height, const Rect& depth_rect,
                   const Rect& motion_rect, int depth_inverted, int reset, const Controls6& controls,
                   float mv_scale_x, float mv_scale_y) {
    std::lock_guard<std::mutex> guard(g_lock);
    g_last_error.clear();

    if (!f || !cmd || !color || !output) {
        g_last_error = "invalid model feature or resources";
        return 0;   // the reference's "bad arguments" answer
    }
    if (debug_noop()) return static_cast<int>(NVSDK_NGX_Result_Success);

    ID3D12Device* device = nullptr;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))) || !device) {
        g_last_error = "the command list belongs to no D3D12 device";
        return 0;
    }

    // NR_DEBUG_READBACK=1: which evaluate this is, and the pair the previous one recorded.
    //
    // The drain has to come first, before anything is recorded into this list, because the pair is
    // two buffers reused every time and the copy below would overwrite what is about to be read. The
    // *line* it produces is emitted after the per-evaluate line, further down.
    const bool readback = debug_readback_on();
    const uint64_t this_evaluate = f->evaluates + 1;
    const bool readback_sample = readback && (this_evaluate <= 3 || this_evaluate % 300 == 0);
    if (readback) {
        ++g_readback_recordings;
        sweep_retired();
        if (!f->debug) f->debug = new DebugReadback();
        readback_drain(*f->debug, f->id);
    }

    // Seed: output := color. The states are the ones DlssNr_Dx12.cpp holds across the call.
    // Only when the network does not write the output from the colour itself (see below).
    const D3D12_RESOURCE_STATES kColorState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    const D3D12_RESOURCE_STATES kOutputState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    const auto seed = [&] {
        barrier(cmd, color, kColorState, D3D12_RESOURCE_STATE_COPY_SOURCE);
        barrier(cmd, output, kOutputState, D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyResource(output, color);
        barrier(cmd, color, D3D12_RESOURCE_STATE_COPY_SOURCE, kColorState);
        barrier(cmd, output, D3D12_RESOURCE_STATE_COPY_DEST, kOutputState);
    };

    // (a) the model's input: the colour (what the seed would put in the output, byte for byte).
    bool readback_recorded = false;
    if (readback_sample && readback_ensure(device, *f->debug, output)) {
        readback_record(cmd, *f->debug, color, kColorState, f->debug->in);
        readback_recorded = true;
    }

    f->controls = controls_from(controls);
    f->controls.preprocess = preprocess_now();

    nr::pe::Session::EngineResources resources{};
    resources.feature = f->id;
    resources.colour = color;
    resources.colour_state = kColorState;
    resources.motion = motion;
    // Both guides are OptiScaler's own clones, made readable by ReadableGuide and left in
    // NON_PIXEL_SHADER_RESOURCE for the call (DlssNr_Dx12.cpp lines 2440-2448 transition them back).
    resources.motion_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    resources.depth = debug_depth() == DebugDepth::Off ? nullptr : depth;
    resources.depth_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    // The guide subrects, straight through. Depth takes the render extent, motion takes its own --
    // they differ whenever the game's motion vectors are at display resolution
    // (DlssNr_Guides.h ResolveGuideRegions, lowResolutionMotion). Base and extent are both honoured,
    // as NVIDIA's DLL does: nr_pe_session.cpp apply_guide_subrect.
    resources.depth_subrect = to_session(depth_rect);
    resources.motion_subrect = to_session(motion_rect);
    resources.depth_inverted = (depth_inverted != 0) != (debug_depth() == DebugDepth::Flip);
    // The caller's flag and nothing else: every feature has its own history now.
    resources.reset = reset != 0;
    // Diagnostic (2026-09-16): NR_DEBUG_ALWAYS_RESET=1 suppresses history on every evaluate, to
    // separate "the second pass's history is wrong" from "the second pass's output is lost".
    {
        static const int always_reset = [] {
            char v[8] = {}; DWORD n = GetEnvironmentVariableA("NR_DEBUG_ALWAYS_RESET", v, sizeof v);
            const int on = (n > 0 && v[0] == '1') ? 1 : 0;
            if (on) log("[nr] NR_DEBUG_ALWAYS_RESET=1: every evaluate runs without history");
            return on;
        }();
        if (always_reset) resources.reset = true;
    }
    {
        const D3D12_RESOURCE_DESC md = motion ? motion->GetDesc() : D3D12_RESOURCE_DESC{};
        const unsigned int mw = motion_rect.width ? motion_rect.width : motion ? unsigned(md.Width) : width;
        const unsigned int mh = motion_rect.height ? motion_rect.height : motion ? unsigned(md.Height) : height;
        resources.motion_scale_x = normalized_mv_scale(mv_scale_x, mw);
        resources.motion_scale_y = normalized_mv_scale(mv_scale_y, mh);
    }
    // OptiScaler's DlssNr encode has already tone-mapped this proxy into the game's own format
    // (DlssNr_Dx12.cpp DispatchPass -> colorCopy), so a float format here is display-referred, not
    // scene-referred linear light, and the runtime must not encode it again.
    //
    // A float colour that reports no HDR but asks for AutoExposure (007 First Light, Helldivers 2) is
    // scene-linear; OptiScaler now encodes it as linear HDR too (nr_pe_optifix.cpp, fix 5). What it
    // cannot do is expose it: such a game hands over no exposure, the encode falls back to a fixed
    // white point, and 007's frame reaches the network about five stops too dark. Re-encoding here
    // would not help; the white point has to come from the frame, before the encode.
    resources.colour_encoded = true;

    write_evaluate_keys(params, controls, resources.reset, depth_inverted);

    auto& session = d3d12_session();
    // NR_DEBUG_SKIP_PASS: this feature's pass number is its rank among the live features.
    int pass_number = 1;
    for (const Feature* o : g_live_d3d12)
        if (o->id < f->id) ++pass_number;
    const bool skipped = debug_skip_pass() > 0 && pass_number == debug_skip_pass();
    // The network reads the colour and writes the output (no seed copy; its post block stores
    // into the output when it can). Where it cannot - still building, a failure, a format or
    // extent mismatch, the model not applied - the output is seeded and the pass runs in place
    // as before. NR_SEED_COPY=1 keeps the old order for comparisons.
    static const bool seed_always = [] { const char* e = std::getenv("NR_SEED_COPY"); return e && std::atoi(e); }();
    bool ran = false;
    if (!skipped && !seed_always) {
        // the seed copy's D3D12 transitions were also what made vkd3d-proton
        // flush transfer work it batches on the CPU (a caller's CopyTextureRegion into the colour)
        // into the command buffer before our raw Vulkan commands. A global UAV barrier - a real
        // D3D12 command - does that without the copy.
        D3D12_RESOURCE_BARRIER uav{};
        uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav.UAV.pResource = nullptr;
        cmd->ResourceBarrier(1, &uav);
        ran = session.run_after(device, cmd, output, kOutputState, resources, f->controls, color, kColorState);
    }
    if (!ran) {
        seed();
        ran = skipped ? false : session.run_after(device, cmd, output, kOutputState, resources, f->controls);
    }

    // (b) the model's answer, from the same resource in the same state, so the only difference
    // between the two buffers is what run_after recorded between them.
    if (readback_recorded) {
        readback_record(cmd, *f->debug, output, kOutputState, f->debug->out);
        f->debug->pending = true;
        f->debug->pending_n = this_evaluate;
    }

    device->Release();

    // **Only a real failure is a failure.** A non-empty status used to be enough to answer
    // FAIL_UnableToInitializeFeature, and the status a session sets while it is *building* is not
    // empty -- so the first evaluate after every create returned 0xBAD0000B, OptiScaler latched
    // "DLSS-NR failed this session; use Retry", and the model never ran at all. The seed copy above
    // has already put the input in the output, so a frame that could not be enhanced is a complete,
    // unenhanced frame: that is a Success with a passthrough in the log, exactly as the reference
    // behaves while NGX is still initialising. Only Session::failed() - a hard, latched fault - is
    // worth telling the host about, because only that one will not fix itself.
    const char* why = ran ? nullptr
                      : skipped ? "skipped(NR_DEBUG_SKIP_PASS)"
                      : session.failed() ? "failed"
                      : session.building() ? "building" : "declined";
    log_evaluate(f, "D3D12", width, height, static_cast<const void*>(color),
                 static_cast<const void*>(output), resources.reset, ran, why, session.gpu_ms(), session.network_ms());
    if (readback && f->debug) readback_log(f, *f->debug);
    // The Vulkan identities behind the D3D12 pointers, for the multipass question: does pass k's
    // colour really alias the buffer pass k-1 wrote, or the untouched model input? Same rate limit.
    if (f->evaluates <= 3 || f->evaluates % 300 == 0) {
        const auto cvk = nr::pe::resource_handle(device, color, kColorState);
        const auto ovk = nr::pe::resource_handle(device, output, kOutputState);
        const D3D12_RESOURCE_DESC cd = color->GetDesc(), od = output->GetDesc();
        log("[nr]   colour vk=%p %ux%u fmt %u layout %u | output vk=%p %ux%u fmt %u layout %u",
            (void*)cvk.image, (unsigned)cd.Width, (unsigned)cd.Height, (unsigned)cd.Format, (unsigned)cvk.layout,
            (void*)ovk.image, (unsigned)od.Width, (unsigned)od.Height, (unsigned)od.Format, (unsigned)ovk.layout);
    }
    if (!ran && !skipped && !session.status().empty()) {
        g_last_error = session.status();
        static std::string reported;
        if (reported != g_last_error) {
            reported = g_last_error;
            log("[nr] model evaluate (D3D12): %s", g_last_error.c_str());
        }
    }
    if (!ran && session.failed())
        return static_cast<int>(NVSDK_NGX_Result_FAIL_UnableToInitializeFeature);
    return static_cast<int>(NVSDK_NGX_Result_Success);
}

void release_d3d12(Feature* f) {
    std::lock_guard<std::mutex> guard(g_lock);
    // **No wait.** This runs on the render thread, and it used to call Session::wait_idle() before
    // freeing anything the GPU might still be reading. That was the wrong tool twice over: on the
    // D3D12 path nothing of ours is ever submitted by us, so there is no fence of ours to wait on,
    // and a device-wide wait inside a game's frame is a stall by construction. What actually keeps
    // the GPU safe is deferral: OptiScaler has already parked this feature for 32 evaluates
    // (DlssNr_Dx12.cpp ParkNrFeature) before releasing it, and the runtime retires the feature's
    // images for a further 64 recordings before destroying them.
    if (f && g_d3d12_session) g_d3d12_session->release_feature(f->id);
    g_live_d3d12.erase(std::remove(g_live_d3d12.begin(), g_live_d3d12.end(), f), g_live_d3d12.end());
    // The readback pair goes the same way and for the same reason: a copy into it may still be in a
    // list the GPU has not reached, so it is retired for 64 further evaluates rather than released
    // here. Nothing to do at all unless NR_DEBUG_READBACK=1 ever allocated it.
    if (f && f->debug) {
        readback_free(*f->debug);
        delete f->debug;
        f->debug = nullptr;
    }
    delete f;
    // The session outlives the feature on purpose; see the comment on g_d3d12_session.
}

// ---------------------------------------------------------------------------------------------
// Vulkan.
// ---------------------------------------------------------------------------------------------

int vk_init(void* instance, void* physical_device, void* device) {
    std::lock_guard<std::mutex> guard(g_lock);

    if (!instance || !physical_device || !device) {
        return static_cast<int>(NVSDK_NGX_Result_FAIL_InvalidParameter);
    }

    g_vk_handles.instance = static_cast<VkInstance>(instance);
    g_vk_handles.physical = static_cast<VkPhysicalDevice>(physical_device);
    g_vk_handles.device = static_cast<VkDevice>(device);

    // A queue is not optional: constructing nr::Runtime submits the weight upload and waits for it.
    // A Vulkan game hands us no queue anywhere in this ABI -- NVIDIA's core has one because the game
    // gave it one at NVSDK_NGX_VULKAN_Init, and neither the forwarder signature nor
    // VULKAN_CreateFeature1 passes it on.
    //
    // UNDICTATED CHOICE, flagged in the report: the device watcher (windows/src/pe/nr_pe_vkdevice.cpp) is the
    // game module's answer -- it hooks vkGetDeviceQueue and remembers what the game asked for. It only
    // works if it was installed before the game created its device, which is true when dlssnr_amd.dll
    // is loaded as the game's dxgi/winmm proxy and false when only these NGX DLLs are present. There
    // is no second source for the answer that is not a guess, so when the watcher has nothing we
    // decline rather than call vkGetDeviceQueue on a family the game may never have requested (which
    // is undefined behaviour, not a failed call).
    nr::pe::vkdevice::note_handles(g_vk_handles.instance, g_vk_handles.physical, g_vk_handles.device);

    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = 0;
    g_vk_queue_known = nr::pe::vkdevice::queue_for(g_vk_handles.device, &queue, &family);

    if (!g_vk_queue_known) {
        log("[nr] Vulkan model init: no VkQueue known for this device; the Vulkan path needs "
            "dlssnr_amd.dll loaded early enough to watch vkGetDeviceQueue");
        return static_cast<int>(NVSDK_NGX_Result_FAIL_UnableToInitializeFeature);
    }

    vk_session().set_vulkan_queue(queue, family);
    log("[nr] Vulkan model init: device %p, queue family %u", device, family);
    return static_cast<int>(NVSDK_NGX_Result_Success);
}

bool vk_ready() {
    std::lock_guard<std::mutex> guard(g_lock);
    return g_vk_handles.valid() && g_vk_queue_known;
}

Feature* create_vk(void* cmd_buffer, void* params, unsigned int width, unsigned int height,
                   int preset, int ui_correction, const Controls6& controls, int* out_result) {
    std::lock_guard<std::mutex> guard(g_lock);

    if (!cmd_buffer || !g_vk_handles.valid() || !g_vk_queue_known) {
        if (out_result) *out_result = static_cast<int>(NVSDK_NGX_Result_FAIL_NotInitialized);
        return nullptr;
    }

    auto* feature = new Feature();
    feature->width = width;
    feature->height = height;
    feature->preset = preset;
    feature->ui_correction = ui_correction;
    feature->controls = controls_from(controls);
    feature->id = vk_session().create_feature();

    write_create_keys(params, width, height, preset, ui_correction, controls);

    // Nothing is recorded into cmd_buffer. The reference has to, because NGX builds its feature there;
    // ours builds on a background thread against the session's own queue and the first frames pass
    // through unenhanced until it lands.
    if (out_result) *out_result = static_cast<int>(NVSDK_NGX_Result_Success);
    log("[nr] model create (Vulkan): feature #%u, %ux%u, preset %d",
        static_cast<unsigned>(feature->id), width, height, preset);
    return feature;
}

int evaluate_vk(void* cmd_buffer, Feature* f, void* params, const void* color, const void* depth,
                const void* motion, const void* output, unsigned int width, unsigned int height,
                const Rect& depth_rect, const Rect& motion_rect, int depth_inverted, int reset,
                const Controls6& controls, float mv_scale_x, float mv_scale_y) {
    std::lock_guard<std::mutex> guard(g_lock);

    if (!f || !cmd_buffer || !color || !output || !g_vk_handles.valid()) return -1;

    auto cmd = static_cast<VkCommandBuffer>(cmd_buffer);

    nr::pe::Session::VulkanFrame frame{};
    if (!image_of(color, &frame.colour, &frame.colour_format, &frame.width, &frame.height)) return -1;

    VkImage out_image = VK_NULL_HANDLE;
    VkFormat out_format = VK_FORMAT_UNDEFINED;
    uint32_t out_w = 0, out_h = 0;
    if (!image_of(output, &out_image, &out_format, &out_w, &out_h)) return -1;

    // The caller's own subrect arguments are authoritative over the descriptor's extent: it passes the
    // working size, and the descriptor may describe a larger allocation.
    if (width) frame.width = width;
    if (height) frame.height = height;

    // The layouts the caller left these in. OptiScaler transitions its output to GENERAL right before
    // this call; the colour it hands over is its own proxy, which its encode leaves in
    // SHADER_READ_ONLY_OPTIMAL (DlssNrFeature_Vk.cpp lines 937/963, DlssNrFeature_Vk_Model.cpp the
    // same). Neither is carried in NVSDK_NGX_Resource_VK, which has no layout field at all, so these
    // are read from the caller's source rather than asked for -- flagged in the report as the one
    // undictated Vulkan assumption.
    frame.colour_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    frame.upscaler_input = false;   // OptiScaler already tone-mapped this; see EngineResources::colour_encoded

    // The output written directly (Session::run_vulkan): only when it is the colour's twin - same
    // format, and both descriptors' extents are the working size (no subrect to crop). The colour is
    // an NGX input (sampled; this path already copies from it) and the output an NGX output this
    // path already copies into; NGX's ReadWrite flag is the UAV one, i.e. storage usage - OptiScaler
    // creates both with STORAGE|SAMPLED|TRANSFER and marks them ReadWrite.
    {
        uint32_t cw = 0, ch = 0; VkImage ci{}; VkFormat cf{};
        image_of(color, &ci, &cf, &cw, &ch);
        if (out_format == frame.colour_format && out_w == frame.width && out_h == frame.height &&
            cw == frame.width && ch == frame.height) {
            frame.output = out_image;
            frame.output_layout = VK_IMAGE_LAYOUT_GENERAL;
            frame.colour_usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                 (static_cast<const NVSDK_NGX_Resource_VK*>(color)->ReadWrite ? VK_IMAGE_USAGE_STORAGE_BIT : 0u);
            frame.output_usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                 (static_cast<const NVSDK_NGX_Resource_VK*>(output)->ReadWrite ? VK_IMAGE_USAGE_STORAGE_BIT : 0u);
        }
    }

    if (image_of(motion, &frame.motion, &frame.motion_format, &frame.motion_width,
                 &frame.motion_height)) {
        frame.motion_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    if (image_of(depth, &frame.depth, &frame.depth_format, &frame.depth_width, &frame.depth_height)) {
        frame.depth_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    frame.motion_subrect = to_session(motion_rect);
    frame.depth_subrect = to_session(depth_rect);

    frame.depth_inverted = depth_inverted != 0;
    frame.feature = f->id;
    // The caller's flag and nothing else: every feature has its own history now.
    frame.reset = reset != 0;
    frame.motion_scale_x = normalized_mv_scale(
        mv_scale_x, motion_rect.width ? motion_rect.width : frame.motion_width ? frame.motion_width : frame.width);
    frame.motion_scale_y = normalized_mv_scale(
        mv_scale_y, motion_rect.height ? motion_rect.height : frame.motion_height ? frame.motion_height : frame.height);

    f->controls = controls_from(controls);
    f->controls.preprocess = preprocess_now();
    write_evaluate_keys(params, controls, frame.reset, depth_inverted);

    auto& session = vk_session();
    VkImage answer = session.run_vulkan(g_vk_handles, cmd, frame, f->controls);

    const bool ran = answer != VK_NULL_HANDLE;
    if (ran && answer == out_image) {
        // Written in place by the network (Session::run_vulkan with VulkanFrame::output).
    } else if (!ran) {
        // Still building, or declined. The output has to carry a picture either way, so it gets the
        // input unchanged -- the pass becomes a no-op rather than a black frame. OptiScaler's resolve
        // blends output against the same proxy, so an identical copy composites to exactly the proxy.
        copy_into_output(cmd, frame.colour, frame.colour_layout, out_image, frame.width, frame.height);
    } else {
        // Session::run_vulkan leaves its own output in VK_IMAGE_LAYOUT_GENERAL.
        copy_into_output(cmd, answer, VK_IMAGE_LAYOUT_GENERAL, out_image, frame.width, frame.height);
    }
    // As the D3D12 path: building is not failing, and a frame that carries the input unchanged is a
    // complete frame. See the note there.
    const char* why = ran ? nullptr
                      : session.failed() ? "failed"
                      : session.building() ? "building" : "declined";
    // A C-style cast through uintptr_t: VkImage is a pointer on x86_64 and a
    // uint64_t on i686, and static_cast is legal for only one of them.
    log_evaluate(f, "Vulkan", frame.width, frame.height, (const void*)(uintptr_t)(frame.colour),
                 (const void*)(uintptr_t)(out_image), frame.reset, ran, why, session.gpu_ms(), session.network_ms());
    if (!ran && !session.status().empty()) {
        g_last_error = session.status();
        static std::string reported;
        if (reported != g_last_error) {
            reported = g_last_error;
            log("[nr] model evaluate (Vulkan): %s", g_last_error.c_str());
        }
    }
    if (!ran && session.failed())
        return static_cast<int>(NVSDK_NGX_Result_FAIL_UnableToInitializeFeature);
    return static_cast<int>(NVSDK_NGX_Result_Success);
}

void release_vk(Feature* f) {
    std::lock_guard<std::mutex> guard(g_lock);
    if (f && g_vk_session) g_vk_session->release_feature(f->id);
    delete f;
    // The session outlives the feature on purpose; see the comment on g_vk_session.
}

}  // namespace nr::dlssnr
