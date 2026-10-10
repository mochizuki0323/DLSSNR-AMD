#pragma once
#include <utility>
#include <vector>
#include <vulkan/vulkan.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace nr {

// One later pass's own model controls. Absent fields inherit pass 1's, except
// the tone, which the original's own hosts default to 0 on every pass after the
// first - a second pass that re-applies the tone control compounds it, measured
// at x1.052 -> x1.102 -> x1.136 over three passes.
struct PassControls {
    bool used = false;              // false: this pass inherits pass 1 entirely
    int style = 0;
    float intensity = 1.0f;
    float local_tone = 0.0f;
    float local_structure = 1.0f;
    float skin_structure = -1.0f;
    bool automatic_mask = true;
};

// [Preprocess] in dlssnr-amd.ini: what the network is shown is changed before
// it runs and every change is taken back out of its answer
// (windows/shaders/passes/runtime_prep.comp). Opt-in; the defaults below are what
// Enabled=1 starts from (auto exposure, filmic curve), and each field has
// a value that leaves the frame as the host handed it over (exposure off,
// curve 0, contrast and saturation 1).
// Needs a runtime built with RuntimeConfig::preprocess.
struct Preprocess {
    bool enabled = false;
    // 0 off; 1 auto: a game engine's histogram auto exposure
    // (runtime_prep.comp meter()), plus bias_ev; 2 fixed: bias_ev alone
    int exposure = 1;
    float bias_ev = 0.0f;        // -8..8
    // 0 none, 1 neutral, 2 reinhard, 3 filmic, 4 gt, 5 aces, 6 agx
    int curve = 3;
    float contrast = 1.0f;       // about mid grey, 0.5..2
    float saturation = 1.0f;     // 0.05..2
    // Does any of it change anything? False records nothing.
    bool active() const {
        return enabled && (exposure == 1 || (exposure == 2 && bias_ev != 0.0f) || curve != 0 ||
                           contrast != 1.0f || saturation != 1.0f);
    }
    bool operator==(const Preprocess& o) const {
        return enabled == o.enabled && exposure == o.exposure && bias_ev == o.bias_ev && curve == o.curve &&
               contrast == o.contrast && saturation == o.saturation;
    }
    bool operator!=(const Preprocess& o) const { return !(*this == o); }
};

// In-process interface shared by game adapters and the native menu.
// Controls::transfer
constexpr int kEnlargeMatched = 0, kEnlargeEdgeAware = 1, kEnlargeClassic = 2;

struct Controls {
    bool enabled = true;
    bool apply_model = true;
    float intensity = 1.0f;
    // The composition onto the frame (windows/shaders/passes/runtime_transfer.comp), as
    // OptiScaler DLSS-NR's resolve: Detail strength (TransferStrength, 0..2),
    // Colour strength (0..4, above 1 boosts saturation) and Highlight guard
    // (MaxRatio, 1..8). Not used with native compose, where the host composes.
    float detail_strength = 1.0f;
    float colour_strength = 1.0f;
    float max_ratio = 2.0f;
    // Below 100% Model resolution, how the model's result is enlarged to the frame
    // (runtime_transfer.comp), kEnlarge*: matched residual (the default), edge-aware
    // lighting + colour, classic. Per frame; nothing is rebuilt.
    int transfer = 0;
    // Classic's scaler: 0 bilinear, 1 Catmull-Rom, 2 Lanczos3, 3 FSR 1 EASU.
    int classic_scaler = 1;
    // Edge-aware: the enlargement weighted by the full-resolution proxy (else plain
    // bilinear), and the weight's range in stops. Not user settings.
    bool transfer_guided = true;
    float transfer_sigma = 0.25f;
    // How many times the network runs on the frame, 1..RuntimeConfig::max_passes.
    // Pass k+1 takes pass k's output as its colour, with the same motion and
    // depth, and its own history; the transfer pass runs once at the end
    // against the frame the first pass saw. What the NVIDIA-side "cascade"
    // mods do. Cost is linear in the count.
    int passes = 1;
    int style = 0;
    float local_tone = 1.0f;
    float local_structure = 1.0f;
    float skin_structure = -1.0f;
    bool automatic_mask = true;
    // Per-pass overrides for passes 2..N; index 0 is pass 2. An entry with
    // `used == false`, or a pass past the end of this, inherits pass 1 with the
    // tone zeroed. Empty is exactly the behaviour that shipped before.
    std::vector<PassControls> per_pass{};
    Preprocess preprocess{};
};

struct HostDevice {
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    uint32_t queue_family{};
    // Vulkan layers receive inner physical-device handles. Route physical
    // queries through their next instance dispatch, not the outer loader.
    // Null for ordinary application-owned (outer-loader) handles.
    PFN_vkGetInstanceProcAddr physical_dispatch{};
    // The host API's queue lock, taken by the constructor around each of its
    // submits only. Empty: the caller serialises the whole construction.
    std::function<void()> queue_lock, queue_unlock;
    // bufferDeviceAddress is enabled on `device` (vkd3d-proton's always; DXVK's and a Vulkan
    // game's when the device watch added the network's features; the bridge's own device). The graph
    // then reads the activation arena through a buffer_reference where that is faster (Windows
    // network only).
    bool buffer_device_address = false;
};

struct RuntimeConfig {
    std::string root;
    std::string plan;  // empty: compiled native planner; optional file must match it
    std::string accumulation = "fp32";
    std::string adapter_shaders;  // empty: ROOT/build/runtime
    uint32_t width{}, height{};
    VkFormat colour_format = VK_FORMAT_R32G32B32A32_SFLOAT;
    // The colour is scene-referred linear light - the buffer an upscaler is
    // handed before the tone mapper, not a finished frame. The runtime then
    // shows the network an sRGB proxy (windows/shaders/passes/runtime_encode.comp) and carries
    // the edit back onto the untouched frame as a luminance ratio
    // (runtime_decode.comp). Off for every swapchain path: a swapchain image in
    // SRGB_NONLINEAR holds encoded [0,1] values whatever its container.
    bool linear_input = false;
    // Divides the linear frame before encoding; the level that becomes 1.0 in
    // the proxy. 1.0 when the game's exposure is unknown. See set_white_point.
    float white_point = 1.0f;
    // The network's extent as a fraction of the frame's, (0, 1]. Below 1 the
    // frame is downscaled into the network's input (aligned to 8) and the edit
    // is carried back onto the full-resolution frame by the transfer pass, so
    // the cost tracks scale^2 while the game's own resolution is untouched.
    // The "Model Resolution" of the NVIDIA-side mods.
    float model_scale = 1.0f;
    // The most passes Controls::passes may ask for; each one above the first
    // costs a history image at the model extent.
    uint32_t max_passes = 1;
    // What a later pass is shown. false (the default): the previous pass's
    // answer as it is, which is what the NVIDIA-side cascade mods do and the
    // only honest setting while the network's own tone drift is
    // being fixed at its source - hiding it here would only mask the defect.
    // true: the first pass's input
    // carrying only the previous pass's local structure - its luminance ratio
    // with the low-pass removed, on the input's own hue - so passes add detail
    // and never stack the model's tone compression or colour shift. false: the
    // previous pass's answer as it is (what the NVIDIA-side cascade mods do
    // with a network that has no such drift; ours has).
    bool cascade_detail_only = false;
    // The pinned DLL's own composition and nothing else: the post block's tail
    // writes base + net*gate, then out = sat(base + sat(intensity)*(nr - base))
    // (its fp8_simple_blend epilogue). No transfer
    // pass, so `colour` and `model_scale` have no effect. What a host that does
    // its own resolve afterwards (OptiScaler) must be given.
    bool native_compose = false;
    // Able to run Controls::preprocess. Costs the frame-format input copy and
    // the post block's direct store (the input has to be a float image the
    // preprocess can rewrite), whether or not a frame asks for it; off, the
    // runtime is exactly what it was.
    bool preprocess = false;
    // The frame is a proxy of linear light made with OptiScaler's soft knee
    // (its linear-HDR encode, or linear_input's own): the preprocess undoes the
    // knee first, so its curve is the only one. False for an SDR frame.
    bool preprocess_unknee = false;
};

// SDR encoded RGB, source-sized and upright. Accepts RGBA32F and 8-bit RGBA/BGRA
// UNORM or SRGB. For SRGB images, copy the encoded bits through a private UNORM
// transfer image; do not apply a transfer function or require mutable game views.
// Image must have TRANSFER_SRC|TRANSFER_DST usage, one colour mip/layer/sample,
// and belong to the host queue family. Format must match RuntimeConfig.
struct ColourFrame {
    VkImage image{};
    VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
    uint32_t width{}, height{};
    VkImageLayout before = VK_IMAGE_LAYOUT_GENERAL;
    VkImageLayout after = VK_IMAGE_LAYOUT_GENERAL;
    VkPipelineStageFlags before_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkAccessFlags before_access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    VkPipelineStageFlags after_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkAccessFlags after_access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    // What the image may be used for, when the caller knows. Zero = unknown.
    // With VK_IMAGE_USAGE_SAMPLED_BIT the engine path samples the colour and
    // depth in place instead of copying them (same values, no copy).
    VkImageUsageFlags usage = 0;
};

// Explicit engine/integration-provided mask. Linear RGBA32F, TRANSFER_SRC,
// one mip/layer/sample, host queue family. Preserve its image/layout/contents.
// R masks the NR1 edit before application strength; G/B condition tone/structure.
// Alpha unused. Subrect width/height zero select the full texture independently.
// This does not acquire a mask from a game or generate a character segmentation.
struct ControlMaskConfig {
    // Both zero disable the optional allocation/pipelines. Size changes require
    // reconstruction, independently of colour extent. Existing RuntimeConfig ABI
    // remains intact; callers opt in through the separate constructor overload.
    uint32_t width{}, height{};
};

struct ControlMaskFrame {
    ColourFrame texture;
    uint32_t x{}, y{}, width{}, height{};
};

// Opt-in temporal path. Allocates the fallback motion estimator's luma
// pyramids and flow field, plus the history image, and selects the temporal
// pre/post pipeline variants. With `enable` false nothing is allocated and the
// runtime records exactly what it records today.
//
// This is the consumer side of the motion provider architecture. The
// estimator is the *fallback* provider: a
// game that exposes engine motion vectors must use those instead, and when that
// path exists it fills the same motion texture rather than replacing any of
// this. An estimate never reports itself as engine data.
struct TemporalConfig {
    bool enable = false;
    // The original's post block blends the reprojected previous output into the
    // current one under a weight the network emits, scaled by a device scalar
    // (its parameter block's +104, clamped to [0, 1]): the model's blend_scale,
    // 0.7397. This multiplies it; 1 is the original. 0 disables the output-side
    // blend and leaves only the pre block's history features.
    float history_strength = 1.0f;
    // Variant SPIR-V directory; empty selects ROOT/build/windows/rdna4/network/temporal.
    // These are variants, not production binaries: the temporal build
    // verifies the shipping pre/post SPIR-V are unchanged when it builds them.
    std::string shaders;
};

// Per-frame temporal state. `reset` is `DLSSNR.Reset`, one of the three terms
// of the original's gate: it suppresses history
// for this frame only. The other two terms - the first-frame latch and whether
// a motion source exists at all - the runtime knows and resolves itself.
struct TemporalFrame {
    bool reset = false;
    // Which feature's temporal history this frame belongs to; see
    // Runtime::release_feature. Zero is the runtime's own single history, which
    // is what every caller that does not know about features gets.
    uint64_t feature = 0;
};

struct RecordResult {
    uint32_t network_dispatches{};
    bool applied{};
    float effective_skin{}, effective_background{};

};

struct ControlMaskResult {
    RecordResult frame;
    bool consumed{};
};

struct TemporalResult {
    RecordResult frame;
    // The original's gate, as it resolved for this frame: the first-frame latch
    // was set, `reset` was clear, and a motion source was available. False means
    // the history features held the current colour, which is what the original
    // does in the same situation - not a degraded mode of our own.
    bool history_consumed{};
    // Provenance, for the UI. "none", "estimated" or - once the upscaler
    // interception exists - "engine". Never report an estimate as engine data.
    const char* motion_provider = "none";
};

// One frame's worth of engine-provided resources, in the caller's own Vulkan
// device. This is the *engine* motion provider, and
// the point of the whole provider split: the consumer below is identical to the
// fallback's, so neural rendering runs on real motion vectors when a game has
// them and on an estimate when it does not, with nothing else changing.
//
// Colour is the pre-upscale, render-resolution image an upscaler is about to
// consume. Running here is what "pre-upscale neural rendering" means: the
// network sees the frame before the upscaler and before the UI, which is where
// the original runs and is not where a Present-time adapter can reach.
struct EngineFrame {
    ColourFrame colour;        // render-resolution, modified in place
    // Backward motion vectors. Any 2-channel format; `motion_scale` converts
    // the texture's units into normalized screen units, exactly as the
    // original's MVecScaleX/Y do.
    ColourFrame motion;
    float motion_scale_x = 1.0f, motion_scale_y = 1.0f;
    // The motion texture's whole allocation when `motion.width/height` name a
    // region of it (a subrect); zero: the same extent.
    uint32_t motion_texture_width = 0, motion_texture_height = 0;
    // Where the motion and depth regions start in their textures (the subrect
    // bases, in texels); zero: the origin. `width/height` are the regions'.
    uint32_t motion_x = 0, motion_y = 0, depth_x = 0, depth_y = 0;
    // The engine's depth buffer, at render resolution. Optional, and it is never
    // a feature of its own: the original samples depth at the centre and four
    // diagonals and reads the motion vector at whichever of the five is nearest
    // the camera (the pre PTX at :339-378). On a
    // silhouette that is what stops a background vector being used for a
    // foreground pixel. Without it the motion vector is read at the pixel
    // itself, which is what the original does when the game passes no depth.
    ColourFrame depth;
    // Reversed-Z, which is what most modern engines use: nearest is the largest
    // value rather than the smallest. Getting this backwards picks the furthest
    // of the five taps and makes silhouettes worse than no depth at all.
    bool depth_inverted = false;
    // DLSSNR.Reset: suppresses history for this frame only.
    bool reset = false;
    // As TemporalFrame::feature: the independent temporal state this frame
    // belongs to. A host that keeps several NGX features alive on one network -
    // OptiScaler runs one per pass - gives each its own id, and each then has
    // its own history, latch and parity exactly as it would with NVIDIA's DLL,
    // where a feature IS the object that owns them.
    uint64_t feature = 0;
    // where the answer goes when it is not `colour` itself (same format and
    // extent, one pass, the model applied). `colour` is then only read, and the
    // host's seed copy of the colour into the target is not needed.
    ColourFrame target;
};

struct EngineResult {
    RecordResult frame;
    bool history_consumed{};
    const char* motion_provider = "engine";
};

// Diagnostics, off by default: every ~600 recordings of an engine frame, read back what the
// network is handed (colour, the game's motion vectors, depth) and what it hands back, score it
// on a background thread and log the result; the first `pictures` captures also as PNGs in
// `folder`. See nr_input_check.hpp.
void set_input_check(bool on, const std::string& folder, int pictures = 4);

class Runtime {
public:
    // Caller enables the same device features/extensions as nrvk::Context::create.
    // Initialization uploads weights on host.queue; serialize it with host submits.
    Runtime(const HostDevice&, const RuntimeConfig&);
    Runtime(const HostDevice&, const RuntimeConfig&, const ControlMaskConfig&);
    Runtime(const HostDevice&, const RuntimeConfig&, const ControlMaskConfig&, const TemporalConfig&);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    // Records only: no submit, wait, readback, image resizing or host allocation.
    // Caller supplies a recording command buffer on queue_family, synchronizes
    // incoming work and serializes uses of this runtime's scratch arena.
    // Changes to controls affect this recording, including intensity 0/1/2.
    // Wait for all uses before destruction. The mask overload consumes explicit
    // caller resources; neither overload acquires game guides or supplies temporal
    // history, depth/motion integration or a Present hook by itself.
    RecordResult record(VkCommandBuffer, const ColourFrame&, const Controls&);
    ControlMaskResult record(VkCommandBuffer, const ColourFrame&, const Controls&, const ControlMaskFrame*);
    // Named rather than overloaded: `record(cmd, frame, controls, nullptr)` must
    // keep meaning what it already means to every existing caller.
    // Requires a runtime built with TemporalConfig::enable.
    TemporalResult record_temporal(VkCommandBuffer, const ColourFrame&, const Controls&,
                                   const TemporalFrame&);
    // The engine provider: real motion vectors, on the pre-upscale colour image.
    // Requires a runtime built with TemporalConfig::enable, whose extent matches
    // the render resolution rather than the display one.
    EngineResult record_engine(VkCommandBuffer, const EngineFrame&, const Controls&);
    // whether record_engine can take EngineFrame::target with these controls
    // (the model applied, one pass). A host asks before it skips its own seed copy.
    bool takes_target(const Controls&) const;
    // Drop a feature's temporal state (EngineFrame::feature). The images are not
    // freed here: the caller's last use of them may still be executing and this
    // is called from a render thread, so they are retired and destroyed a fixed
    // number of later recordings on, which is the same reasoning as OptiScaler's
    // own 32-evaluate parking of a retired feature. Never blocks.
    void release_feature(uint64_t feature);
    // A recording of this runtime was never executed (the host dropped it): no feature's temporal
    // state can be trusted. Each feature's next frame runs as its first (no history, seed 0). Call
    // between recordings.
    void drop_history();
    // How many features currently hold temporal state, for the log.
    uint32_t live_features() const;

    // The history image, after the last recorded frame completed: the blended
    // image the model wrote, alpha = the blend weight it asked for. For tests
    // and diagnostics; `image` is null when the runtime has no temporal path.
    // The image belongs to the runtime and stays in `layout` for its whole
    // life; a caller may copy from it and must not destroy it.
    struct TemporalHistoryView {
        VkImage image{}; VkImageView view{}; uint32_t width{}, height{};
        VkFormat format{}; VkImageLayout layout{};
    };
    TemporalHistoryView temporal_history() const;
    // The extent the network actually runs at (RuntimeConfig::model_scale applied).
    uint32_t model_width() const;
    uint32_t model_height() const;
    // Change the post block's history strength (TemporalConfig::history_strength)
    // between frames. Takes effect on the next recording; no-op without a
    // temporal path.
    void set_history_strength(float);
    // Change the white point (RuntimeConfig::white_point) between frames. No-op
    // unless the runtime was built with linear_input.
    void set_white_point(float);

    // **What the pass costs on the GPU**, from a timestamp pair around the work
    // this runtime records - everything between the first barrier and the last,
    // which is the number a player comparing it against their frame time wants.
    // It is not the frame time and not a CPU measurement: the recording is
    // submitted inside somebody else's command buffer on two of the three
    // routes, so wall-clock around the call measures the caller, not us.
    //
    // Read from a small ring several recordings behind, never waiting, so
    // asking costs nothing and a result that is not back yet leaves the last one
    // standing. Zero until the first pair comes back, and zero for the whole
    // life of a device whose queue family reports no timestamp bits.
    float last_gpu_ms() const;
    // The same, smoothed over recent frames - what to put in a UI, because the
    // instantaneous number moves too much to read.
    float average_gpu_ms() const;
    // The network's share of average_gpu_ms (every pass's dispatches), the same average; the rest is
    // the work around it (input, composition, copies). Zero until the first result is back.
    float average_network_ms() const;
    // The preprocess meter as the GPU last left it: {smoothed EV, this frame's
    // target EV}, bias not included. For the log; NaN until it has metered.
    std::pair<float, float> preprocess_meter() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    // One body behind the three entry points, so the mask and temporal paths
    // cannot drift apart in barriers or ordering.
    ControlMaskResult record_all(VkCommandBuffer, const ColourFrame&, const Controls&,
                                 const ControlMaskFrame*, const TemporalFrame*, bool*,
                                 const EngineFrame*);
};
}  // namespace nr
