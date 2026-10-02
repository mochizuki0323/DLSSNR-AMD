#pragma once
// One neural-rendering session bound to a game's D3D12 device, running at the
// game's *render* resolution.
//
// Placement, which is the whole point (docs and the neural-upstream reference):
//
//     internal render res -> NR -> the game's DLSS/FSR upscale -> output
//
// rather than the official `internal -> upscale -> NR at output resolution`.
// The network is same-resolution - it enhances, it does not upscale - so running
// it on the smaller image is what makes it affordable, and the upscaler still
// does the job it was going to do.
//
// The network is SPIR-V and the game is D3D12, which on Proton is not a problem:
// vkd3d-proton hands out the command list's VkCommandBuffer and each resource's
// VkImage, so the pass records into the game's own command stream, in order,
// immediately before the upscale it is feeding.
#include "nr_pe_interop.hpp"
#include "nr_runtime.hpp"
#include <memory>
#include <string>

namespace nr::pe {

class Session {
  public:
    // `root` is the directory holding build/ and artifacts/; empty means "next
    // to this module", the same rule the Vulkan layer uses so that nothing has
    // to be configured through the environment.
    explicit Session(std::string root);
    ~Session();

    // The post block's history strength, 0..1; see Config::history. Applies to
    // the running network at once and to any network built later.
    void set_history_strength(float);
    // The white point for linear-light colour buffers; see Config::white_point.
    void set_white_point(float);
    // Model Resolution, 0.25..1: the network's extent as a fraction of the
    // frame's. Changing it rebuilds the network in the background.
    // The pass ceiling a network is built for. Below what shipped is ignored,
    // so 1..4 never rebuilds; above it the next frame rebuilds.
    void set_max_passes(uint32_t);
    void set_model_scale(float);
    // The DLL's own composition only (RuntimeConfig::native_compose): the host
    // does its resolve after us. Set before the first frame; the OptiScaler
    // route sets it, the legacy module does not.
    void set_native_compose(bool);

    // Run the network on `colour` and return the resource to hand the upscaler
    // in its place. Returns null when the pass could not run, in which case the
    // caller must leave the game's parameters exactly as it found them.
    //
    // `motion` may be null: without it the original does not consume history
    // either, and the pass still runs as a
    // spatial enhancement. `reset` is DLSSNR.Reset for this frame.
    // Everything one call of an upscaler carries that the network can use. A
    // struct rather than a parameter list because the list had reached nine and
    // the next thing to add - depth - would have made it eleven, in an order
    // nobody could keep straight across four call sites.
    // A rectangle of a guide texture that actually holds this frame's data.
    struct Subrect {
        uint32_t x{}, y{}, width{}, height{};
    };

    struct EngineResources {
        // Which NGX feature this frame belongs to; see create_feature. Zero is
        // "the session's own single temporal state", which is what the game
        // module's one-pass hooks use.
        uint64_t feature{};
        ID3D12Resource* colour{};
        D3D12_RESOURCE_STATES colour_state{D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
        ID3D12Resource* motion{};
        D3D12_RESOURCE_STATES motion_state{D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
        // Optional. Depth never becomes a feature; it only chooses where the
        // motion vector is sampled, which is what keeps a background vector off
        // a foreground pixel at a silhouette.
        ID3D12Resource* depth{};
        D3D12_RESOURCE_STATES depth_state{D3D12_RESOURCE_STATE_DEPTH_READ};
        bool depth_inverted{true};   // reversed-Z, which is what modern engines use
        // How much of the depth and motion textures the game actually rendered into, and where.
        //
        // Not the same thing as how big they are. A game with dynamic resolution allocates its guides
        // once at the largest size it will ever need and renders into a corner of them; sizing the
        // guide from the resource hands the network the stale margin as well and calls it scene.
        // Zero width or height means "the caller did not say", and the allocation is used.
        //
        // Base and extent are both honoured (apply_guide_subrect in nr_pe_session.cpp).
        Subrect depth_subrect{};
        Subrect motion_subrect{};
        float motion_scale_x{1.0f}, motion_scale_y{1.0f};
        bool reset{};
        // The colour has already been encoded by something upstream, so a float
        // format here is NOT scene-referred linear light and must not be encoded
        // a second time. OptiScaler's DLSS-NR pass tone-maps its proxy before it
        // ever reaches a model, so its path sets this; other callers leave it
        // false and keep judging by format alone.
        bool colour_encoded{};
    };

    ID3D12Resource* run(ID3D12Device* device, ID3D12GraphicsCommandList* list,
                        const EngineResources& resources, const Controls& controls);

    // The post-upscale placement: the upscaler has already written its output at
    // display resolution and the network runs over that, in place. This is where
    // the official pipeline puts it, and it is the more expensive of the two -
    // the network's cost tracks the resolution it runs at. Engine motion vectors
    // still come from the same call, at render resolution; the shader samples
    // them by normalized coordinate, so the resolutions need not match.
    // `input` (optional): read the colour from there and write the answer into
    // `output`, which then needs no seed copy of it. Returns false without
    // recording anything when that is not possible (the caller seeds and runs in
    // place instead): different formats or extents, or the model not applied.
    bool run_after(ID3D12Device* device, ID3D12GraphicsCommandList* list,
                   ID3D12Resource* output, D3D12_RESOURCE_STATES output_state,
                   const EngineResources& resources, const Controls& controls,
                   ID3D12Resource* input = nullptr, D3D12_RESOURCE_STATES input_state = D3D12_RESOURCE_STATE_COMMON);

    // D3D11, through DXVK. There is no command list to record into, so this one
    // owns a command buffer, records into it, and submits it on DXVK's own queue
    // between a flush and the queue lock - the sequence DXVK documents for
    // exactly this.
    //
    // The result goes back into `target` itself rather than into an image of our
    // own. D3D11 offers no way to hand the upscaler a different texture - there
    // is no parameter to rebind and no way to wrap a VkImage as an
    // ID3D11Texture2D - so an out-of-place pass here would run, cost its
    // milliseconds, and be read by nobody. In place is what makes this path real.
    // The game keeps its handle; the pixels behind it change.
    struct D3D11Frame {
        IUnknown* device{};
        IUnknown* target{};   // read and written; the upscaler's input, or its output
        IUnknown* motion{};   // engine motion vectors; null runs the spatial pass
        IUnknown* depth{};    // optional; chooses where the motion vector is read
        bool depth_inverted{true};
        float motion_scale_x{1.0f}, motion_scale_y{1.0f};
        bool reset{};
        // No engine data at all - the Present fallback. Runs the estimator's
        // temporal path instead, and must never be reported as engine data.
        bool estimate_motion{};
        // `target` is an upscaler's input: scene-referred linear light when its
        // format is a float one, and then the runtime encodes it. A finished
        // back buffer - the fallback's, or ReShade's - is never that, whatever
        // its format, so this stays false there.
        bool upscaler_input{};
        // ... unless the swapchain says otherwise. An HDR back buffer IS linear
        // light (scRGB), and the caller is the only one who can know: the format
        // cannot tell an SDR float buffer from an HDR one, and ReShade can
        // (`swapchain::get_color_space`). Set it and the runtime encodes the
        // frame for the model exactly as it does an upscaler's input.
        bool linear_hdr{};
    };
    bool run_d3d11(const D3D11Frame& frame, const Controls& controls);

    // The fallback: no upscaler, so no engine data. The colour is the finished
    // back buffer - already upscaled, already carrying the game's UI - and the
    // motion vectors come from the estimator rather than the engine. Strictly
    // worse than the upscaler hook, and the only thing such a game offers.
    // Returns true when the pass was recorded into the list.
    bool run_present(ID3D12Device* device, ID3D12GraphicsCommandList* list,
                     ID3D12Resource* back_buffer, unsigned width, unsigned height,
                     const Controls& controls);

    // The same pass for a Vulkan game, where the upscaler hands over a
    // VkCommandBuffer and VkImages directly and there is no D3D12 in the way.
    // The device handles come from the upscaler's own context creation.
    struct VulkanFrame {
        uint64_t feature{};   // as EngineResources::feature
        VkImage colour{}; VkImageLayout colour_layout{VK_IMAGE_LAYOUT_UNDEFINED};
        VkFormat colour_format{VK_FORMAT_UNDEFINED};
        uint32_t width{}, height{};
        VkImage motion{}; VkImageLayout motion_layout{VK_IMAGE_LAYOUT_UNDEFINED};
        VkFormat motion_format{VK_FORMAT_UNDEFINED};
        uint32_t motion_width{}, motion_height{};
        VkImage depth{}; VkImageLayout depth_layout{VK_IMAGE_LAYOUT_UNDEFINED};
        VkFormat depth_format{VK_FORMAT_UNDEFINED};
        uint32_t depth_width{}, depth_height{};
        // As EngineResources: the part of each guide the game rendered into. The *_width/_height above
        // stay the allocation, so both numbers are available.
        Subrect depth_subrect{};
        Subrect motion_subrect{};
        bool depth_inverted{true};
        float motion_scale_x{1.0f}, motion_scale_y{1.0f};
        bool reset{};
        bool upscaler_input{};   // as D3D11Frame::upscaler_input
        bool linear_hdr{};       // as D3D11Frame::linear_hdr
        // The caller's output (optional): the network reads `colour` in place and writes
        // here, and run_vulkan returns it - no copy into an image of ours and back.
        // `output_usage` / `colour_usage` as far as the caller's descriptors say.
        VkImage output{}; VkImageLayout output_layout{VK_IMAGE_LAYOUT_GENERAL};
        VkImageUsageFlags output_usage{}, colour_usage{};
    };
    // A Vulkan game's own queue. The two D3D runtimes have an interop object to
    // ask for one; a Vulkan game has nothing of the sort, so the queue is learned
    // by watching the game create it. Without this the network cannot upload its
    // weights and the Vulkan path declines with that reason.
    void set_vulkan_queue(VkQueue queue, uint32_t family);

    // Returns the image to hand the upscaler in place of the game's colour, or
    // null when the pass could not run. With VulkanFrame::output set it is that
    // image when the answer went straight into it.
    VkImage run_vulkan(const DeviceHandles& handles, VkCommandBuffer cmd,
                       const VulkanFrame& frame, const Controls& controls);
    // The universal fallback: the frame exactly as it is about to be shown, in
    // whatever API the game used - under Proton all of them arrive here as a
    // Vulkan swapchain image. No engine data, so the estimator supplies motion
    // and the result is never reported as the engine's.
    //
    // Ordering is the whole difficulty and it is handled here: the pass waits on
    // the semaphores the game's present was going to wait on, and signals one of
    // ours for the present to wait on instead. `signalled` is that semaphore.
    //
    // `slot` is the swapchain image index and `slots` how many there are, and
    // they are not bookkeeping. A binary semaphore may have exactly one pending
    // signal-wait pair at a time, and the present that waits on ours is still
    // outstanding when the next frame arrives - so one semaphore reused every
    // frame is undefined, and shows up as a black screen and then a crash.
    // Everything per-frame here is per image index instead: semaphore, fence and
    // command buffer.
    bool run_present_vulkan(const DeviceHandles& handles, VkQueue queue, uint32_t family,
                            VkImage image, VkFormat format, uint32_t width, uint32_t height,
                            uint32_t slot, uint32_t slots,
                            const VkSemaphore* wait, uint32_t wait_count,
                            VkSemaphore* signalled, bool temporal, const Controls& controls);

    // The view onto the image run_vulkan returns. NGX takes a resource
    // descriptor, not a bare image, so the caller needs both.
    VkImageView vulkan_output_view() const;

    // ---- features ------------------------------------------------------
    //
    // A feature is what NVIDIA's DLL hands back from CreateFeature: an object
    // with its own temporal state, created and destroyed freely while the
    // network behind it stays put. OptiScaler depends on that - one feature per
    // model pass, two extents alive at once, everything torn down and rebuilt
    // whenever a control moves - so this session gives each one an id and each
    // id its own history inside every runtime it is evaluated against.
    //
    // Creating one costs nothing here: the images are allocated by the runtime
    // the first time the feature is actually evaluated, because until then
    // there may be no runtime and no extent to size them from.
    uint64_t create_feature();
    // Never waits. The runtime retires the feature's images and destroys them a
    // fixed number of later recordings on.
    void release_feature(uint64_t feature);

    // Is the network still being built in the background? A caller must treat
    // this as "pass the frame through", NOT as a failure: it is the normal
    // state for the first second of a session, and reporting it as a failure is
    // what makes a host disable the feature for good.
    bool building() const;
    // Has something gone wrong that will not fix itself? This, and only this, is
    // a failure.
    bool failed() const;
    // Is the session running via the format bridge (e.g. cutscene format adaptation)?
    bool is_bridged() const;

    // Wait for any work this session has in flight. Called before anything it
    // may still be reading is destroyed - a retiring swapchain above all, whose
    // images go away while our last submit can still be referencing them.
    void wait_idle();

    // Why the last run declined, for the log and the overlay. Empty while the
    // pass is running normally.
    const std::string& status() const;

    // What the pass costs on the GPU, in milliseconds - the number OptiScaler's
    // own overlay shows for DLSS-NR, measured the same way: a timestamp pair
    // around the work, not a clock around the call. Smoothed; zero until the
    // first result is back, and zero on a queue with no timestamp support.
    float gpu_ms() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace nr::pe
