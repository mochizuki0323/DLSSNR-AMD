#include "nr_runtime.hpp"
#include "nr_native_plan.hpp"
#include "nr_runtime_dispatch.hpp"
#define NR_NO_MAIN 1
#include "nr_graph.cpp"
#include "nr_input_check.hpp"
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <string>
#include <mutex>

// Arena reuse: byte-identical to one value one slot at 16 extents, first and fourth
// frame, model scale 1 and 0.5.
constexpr bool kArenaReuseDefault = true;

namespace nr {
namespace {
std::mutex build_mutex;

void validate(const Controls& c) {
    auto range = [](float v, float lo, float hi) { return std::isfinite(v) && v >= lo && v <= hi; };
    if (!range(c.intensity, 0, 2) || !range(c.detail_strength, 0, 2) || !range(c.colour_strength, 0, 4) ||
        !range(c.max_ratio, 1, 8) || !range(c.local_tone, 0, 2) ||
        !range(c.local_structure, 0, 2) || !range(c.skin_structure, -1, 2) ||
        c.style < 0 || c.style > 2 || c.passes < 1)
        throw std::invalid_argument("invalid NR controls");
}

void barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout,
             VkImageLayout new_layout, VkPipelineStageFlags src_stage,
             VkAccessFlags src_access, VkPipelineStageFlags dst_stage, VkAccessFlags dst_access,
             VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = src_access; b.dstAccessMask = dst_access;
    b.oldLayout = old_layout; b.newLayout = new_layout;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image; b.subresourceRange = {aspect, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// The aspects of a depth/stencil format (zero for a colour one). A barrier on
// such an image has to name them all; a view that samples it names DEPTH only.
VkImageAspectFlags depth_stencil_aspects(VkFormat f) {
    switch (f) {
        case VK_FORMAT_D16_UNORM: case VK_FORMAT_X8_D24_UNORM_PACK32: case VK_FORMAT_D32_SFLOAT:
            return VK_IMAGE_ASPECT_DEPTH_BIT;
        case VK_FORMAT_D16_UNORM_S8_UINT: case VK_FORMAT_D24_UNORM_S8_UINT: case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        default:
            return 0;
    }
}

// `exec_only` drops the memory half. That is correct only between two network
// steps whose pipelines were built with `NR_COHERENT_ACT=1` - every activation
// load and store then carries device scope and reaches L2 itself - which is
// what `s.runner.exec_barrier` says after `build()` read `coherent-act.txt`
// beside the SPVs. Measured in `nr_graph` at -3.7% of the 1080p frame, byte-
// identical; the passes around the network keep the full barrier.
//
// `inv_only` drops the *source* half instead, which is what ships:
// the destination invalidate stays and the availability operation goes, because
// on gfx1201 there is nothing to make available - every cache below the device
// coherence point is write-through. See `runner.inv_barrier` in nr_graph.cpp for
// the argument and the numbers (1080p 7.947 -> 7.829 ms, 4K 29.457 -> 28.939).
void compute_barrier(VkCommandBuffer cmd, bool exec_only = false, bool inv_only = false) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = inv_only ? 0u : VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, exec_only ? 0u : 1u, &b,
                         0, nullptr, 0, nullptr);
}

// runtime_prep.comp's push block; see the shader.
// runtime_transfer.comp's push block, byte for byte.
struct TransferPush {
    uint32_t w, h, model_w, model_h, passthrough;
    float detail, colour, max_ratio, white;
    uint32_t mode, flags;
    float sigma;
};
static_assert(sizeof(TransferPush) == 48, "runtime_transfer.comp's push block");
constexpr uint32_t kTransferEnlarged = 1, kTransferGuided = 2, kTransferRoundU8 = 4;

struct PrepPush {
    uint32_t w, h, mode, flags, curve;
    float bias, contrast, saturation, k, cap;
    float dt;   // seconds since the last metered frame (the meter adapts in real time)
};
enum : uint32_t { kPrepAuto = 1, kPrepUnknee = 2, kPrepRestore = 4, kPrepReset = 8 };
// Each curve's input scale that leaves mid grey (0.18) where it was, so the
// curves shape shadows and highlights and the exposure alone sets brightness.
// Solved numerically, once, for each curve below.
constexpr float kPrepAnchor[7] = {1.0f, 1.0f, 1.2195122f, 2.9275228f, 1.0052344f, 0.7231708f, 0.8083602f};

void dispatch(VkCommandBuffer cmd, const nrvk::Kernel& k, uint32_t x, uint32_t y,
              uint32_t z, const void* push, uint32_t bytes, VkDescriptorSet set = VK_NULL_HANDLE) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0, 1, set ? &set : &k.set, 0, nullptr);
    if (bytes) vkCmdPushConstants(cmd, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, bytes, push);
    vkCmdDispatch(cmd, x, y, z);
}

// Another descriptor set for a kernel's layout, over other resources: buffers
// first, then images in binding order, exactly as nrvk::Kernel::create writes
// its own (a sampler makes a combined image sampler, none a storage image).
VkDescriptorSet kernel_set(VkDevice device, VkDescriptorPool pool, const nrvk::Kernel& k,
                           const std::vector<VkBuffer>& buffers,
                           const std::vector<const nrvk::Context::Image*>& images) {
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &k.dsl;
    VkDescriptorSet set{};
    if (vkAllocateDescriptorSets(device, &ai, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
    const uint32_t nb = uint32_t(buffers.size()), ni = uint32_t(images.size());
    std::vector<VkDescriptorBufferInfo> bi(nb);
    std::vector<VkDescriptorImageInfo> ii(ni);
    std::vector<VkWriteDescriptorSet> w(nb + ni);
    for (uint32_t i = 0; i < nb; ++i) {
        bi[i] = {buffers[i], 0, VK_WHOLE_SIZE};
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = set; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &bi[i];
    }
    for (uint32_t i = 0; i < ni; ++i) {
        ii[i] = {images[i]->sampler, images[i]->view, images[i]->layout};
        w[nb + i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[nb + i].dstSet = set; w[nb + i].dstBinding = nb + i; w[nb + i].descriptorCount = 1;
        w[nb + i].descriptorType = images[i]->sampler ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                      : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[nb + i].pImageInfo = &ii[i];
    }
    vkUpdateDescriptorSets(device, nb + ni, w.data(), 0, nullptr);
    return set;
}

// How a caller's colour format reaches the network's RGBA32F image.
//
//   identical   - RGBA32F: a plain copy, no conversion anywhere.
//   through a   - the four 8-bit formats: the bits are copied into a private
//   transfer      UNORM image and blitted from there, so an *_SRGB view never
//   image         applies its transfer function to the stored values. This is
//                 the existing path.
//   direct blit - the wide formats a Wayland surface offers first: FP16, 16-bit
//                 UNORM and the two 10-bit packed ones, plus the packed 11/11/10
//                 float an engine renders into before its upscaler. None of them
//                 carries an sRGB transfer function, so there is no hazard to
//                 route around and `vkCmdBlitImage` converts directly. The
//                 swapchain layer only accepts these with SRGB_NONLINEAR; the
//                 engine-data path hands over linear scene-referred values and
//                 nothing here rescales or encodes them.
enum class Transfer { Identical, Encoded, DirectBlit };
#ifndef NR_DIRECT_IN
#define NR_DIRECT_IN 1
#endif
#ifndef NR_DIRECT_OUT
#define NR_DIRECT_OUT 1
#endif
#ifndef NR_DIRECT_MOTION
// The engine's motion vectors sampled by the pre and post blocks at full resolution, in place,
// as the original reads them; 0: the old blit into the estimator's quarter-resolution field.
#define NR_DIRECT_MOTION 1
#endif
#ifndef NR_DIRECT_SAMPLE
#define NR_DIRECT_SAMPLE 1
#endif

Transfer transfer_mode(VkFormat format) {
    switch(format) {
        case VK_FORMAT_R32G32B32A32_SFLOAT: return Transfer::Identical;
        // Only an *_SRGB frame needs the private UNORM image: a UNORM frame's own
        // blit is the same UNORM <-> float conversion, without the two copies.
        case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_SRGB: return Transfer::Encoded;
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
        // The engine-data path's usual colour buffer (DXGI R11G11B10_FLOAT):
        // unsigned floats, no alpha, no transfer function, values above 1.0
        // allowed. The blit in carries them unchanged and the blit out clamps
        // anything negative to zero, which is what the format can hold.
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
            return Transfer::DirectBlit;
        default:
            break;
    }
    // Every other RGB colour format a frame can come in, if NR_FORMAT_FALLBACK is
    // not 0. The cases above keep their paths; this is only reached by a format
    // that used to be refused here. Blitting converts each of these to RGBA32F
    // and back by the format's own definition, and the constructor checks this
    // GPU can blit it before anything is built. Left out on purpose: integer
    // formats (a blit cannot convert them to float), sRGB ones other than the
    // two above (a blit would decode their transfer function, which the network
    // must not see), block-compressed and depth formats.
    const char* fallback = std::getenv("NR_FORMAT_FALLBACK");
    if (!(fallback && fallback[0] == '0')) switch(format) {
        case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
        case VK_FORMAT_R32G32B32_SFLOAT: case VK_FORMAT_R16G16B16_SFLOAT:
        case VK_FORMAT_R16G16B16A16_SNORM: case VK_FORMAT_R16G16B16_UNORM: case VK_FORMAT_R16G16B16_SNORM:
        case VK_FORMAT_R8G8B8A8_SNORM: case VK_FORMAT_B8G8R8A8_SNORM:
        case VK_FORMAT_A8B8G8R8_UNORM_PACK32: case VK_FORMAT_A8B8G8R8_SNORM_PACK32:
        case VK_FORMAT_R8G8B8_UNORM: case VK_FORMAT_R8G8B8_SNORM:
        case VK_FORMAT_B8G8R8_UNORM: case VK_FORMAT_B8G8R8_SNORM:
        case VK_FORMAT_A2R10G10B10_SNORM_PACK32: case VK_FORMAT_A2B10G10R10_SNORM_PACK32:
        case VK_FORMAT_R5G6B5_UNORM_PACK16: case VK_FORMAT_B5G6R5_UNORM_PACK16:
        case VK_FORMAT_R5G5B5A1_UNORM_PACK16: case VK_FORMAT_B5G5R5A1_UNORM_PACK16:
        case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
        case VK_FORMAT_R4G4B4A4_UNORM_PACK16: case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
        case VK_FORMAT_A4R4G4B4_UNORM_PACK16: case VK_FORMAT_A4B4G4R4_UNORM_PACK16:
            return Transfer::DirectBlit;
        default:
            break;
    }
    throw std::invalid_argument("unsupported NR colour format, VkFormat " + std::to_string(unsigned(format)));
}

VkFormat encoded_format(VkFormat format) {
    switch(format) {
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
        case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
        default: return format;
    }
}
}  // namespace

// The fallback motion estimator's pyramid, as measured:
// base /4 with four levels, and a per-level
// search radius that a /2 base was shown to be *worse* than, not cheaper than.
constexpr unsigned kTemporalLevels = 4;
constexpr unsigned kTemporalBase = 4;
constexpr int kTemporalRadius[kTemporalLevels] = {2, 3, 3, 4};
constexpr float kTemporalReject = 0.50f;
// The post block's history weight is sigmoid(net) * clamp(s, 0, 1), where s is
// the model's own f16 weight block70.layer0.blend_scale (0x39EB, the post
// kernel's +104 pointer; the DLL looks it up next to the layer's weights at
// 0x180075ce0). The model is pinned to 310.8.0 by its SHA-256, so it is a
// constant here. TemporalConfig::history_strength scales it.
constexpr float kPostBlendScale = 0.73974609375f;
// The pre block's push, PushFSwin + PushPreImage (nr_graph.cpp).
constexpr uint32_t kPrePushBytes = sizeof(PushFSwin) + sizeof(PushPreImage);
constexpr uint32_t kPreSeedAt = sizeof(PushFSwin) + offsetof(PushPreImage, seed);
constexpr uint32_t kPreNoiseFieldAt = sizeof(PushFSwin) + offsetof(PushPreImage, noise_off);

struct Temporal {
    struct LumaPush { uint32_t dst_w, dst_h, src_w, src_h, mode; };
    struct FlowPush {
        uint32_t level_w, level_h, coarse_w, coarse_h;
        int32_t radius; uint32_t first, last; float reject;
    };

    bool enabled{};
    uint32_t lw[kTemporalLevels]{}, lh[kTemporalLevels]{};
    // Ping-pong: this frame's pyramid becomes next frame's previous one. The
    // descriptor sets are baked at creation, so each parity needs its own
    // pipelines; they are four tiny dispatches each and cost nothing to hold.
    nrvk::Context::Image luma[2][kTemporalLevels]{};
    nrvk::Context::Image flow[kTemporalLevels]{};
    nrvk::Context::Image flow0_sampled{};   // the same image, sampled alias
    VkSampler motion_sampler{};             // owned here; the alias does not own its image
    nrvk::Context::Image history{};
    // One pass: two history images in turn. The post variant writes the model's
    // image straight into the one the next frame reads (pre/post read
    // hist(cur), post writes hist(cur ^ 1)), instead of into surf1 and a copy -
    // a whole RGBA32F frame read and written once a frame (1080p 0.085 ms).
    nrvk::Context::Image history_b{};
    nrvk::Kernel pre_pp[2], post_pp[2];
    // What pre_pp/post_pp were built over, so a set can be made that swaps the
    // colour (and depth) for the caller's own images; see Impl::DirectSrc.
    std::vector<VkBuffer> pre_buffers, post_buffers;
    nrvk::Context::Image hist_store[2]{};
    // the post block's history, reconstructed by the pre block (RGBA32F storage,
    // model extent) when the package has the *hp variants; else an empty image.
    nrvk::Context::Image hpass{};
    bool hp{};
    // (research NR_HIST16=1): the one-pass history images in RGBA16F.
    bool h16{};
    bool pingpong{};
    uint32_t hcur{};
    nrvk::Context::Image& hist(uint32_t i) { return i ? history_b : history; }
    // Where the engine's depth is put, so the consumer has one descriptor
    // whether a game supplies depth or not. Allocated always and left cleared
    // when it is not; the shader is told which by the parameter buffer rather
    // than by reading the contents, because an all-zero depth is a legitimate
    // depth.
    nrvk::Context::Image depth{};
    nrvk::Buffer params;
    nrvk::Kernel luma_kernel[2][kTemporalLevels];
    nrvk::Kernel flow_kernel[2][kTemporalLevels];
    nrvk::Kernel pre, post;
    const nrvk::Kernel* original_pre{};
    const nrvk::Kernel* original_post{};
    float history_strength{1.0f};
    uint32_t parity{};
    // The first-frame latch, the original's plan +0x68. Set after any recorded
    // frame, never cleared by `reset` - reset suppresses consumption, it does
    // not un-write the history buffer.
    bool latch{};
    // The pre block's noise seed for the next frame: frames since the first one
    // or the last reset (FeatureState::seed per feature).
    uint32_t seed{};

    void destroy(nrvk::Context& ctx) {
        for (unsigned p = 0; p < 2; ++p)
            for (unsigned k = 0; k < kTemporalLevels; ++k) {
                if (luma_kernel[p][k].device) luma_kernel[p][k].destroy();
                if (flow_kernel[p][k].device) flow_kernel[p][k].destroy();
                ctx.destroy(luma[p][k]);
            }
        for (unsigned k = 0; k < kTemporalLevels; ++k) ctx.destroy(flow[k]);
        ctx.destroy(depth);
        if (pre.device) pre.destroy();
        if (post.device) post.destroy();
        for (unsigned c = 0; c < 2; ++c) {
            if (pre_pp[c].device) pre_pp[c].destroy();
            if (post_pp[c].device) post_pp[c].destroy();
        }
        if (history_b.handle) ctx.destroy(history_b);
        if (hpass.handle) ctx.destroy(hpass);
        // flow0_sampled aliases flow[0]; only its sampler is ours to free.
        if (motion_sampler) vkDestroySampler(ctx.device, motion_sampler, nullptr);
        ctx.destroy(history);
        ctx.destroy(params);
    }
};

struct Runtime::Impl {
    NrSession session;
    nrvk::Kernel alpha, mask_pre, mask_resolve;
    // The linear-light path: the proxy is made in place on tex_in and the
    // original kept here for the decode. Only when RuntimeConfig::linear_input.
    nrvk::Kernel encode, transfer_pass;
    nrvk::Context::Image keep{};
    bool linear{};
    // Preprocess (RuntimeConfig::preprocess): runtime_prep.comp's meter,
    // forward and back modes, the frame as it came in (`prep_keep`, model
    // sized) and the meter's state. `prep_back` is `prep` itself unless later
    // passes overwrite tex_in, when the first pass's input is shown_keep and
    // the frame is put back there for the transfer pass. `transfer_prep` is the
    // transfer pass reading prep_keep as what the network was shown.
    bool prep{}, prep_unknee{};
    nrvk::Kernel prep_k, prep_back_k, transfer_prep;
    nrvk::Context::Image prep_keep{};
    nrvk::Buffer prep_state{};
    // What the last frame asked for. A change resets the meter and bumps
    // prep_gen; each history made under another generation (in the other
    // domain) is not consumed, per feature, as OptiScaler runs each pass as
    // its own feature.
    bool prep_was_on{};
    Preprocess prep_last{};
    std::chrono::steady_clock::time_point prep_metered{};
    uint32_t prep_gen{}, prep_gen_single{};
    float white_point{1.0f};
    // Model Resolution: the network runs at mw x mh, the frame is width x
    // height. When they differ, `keep_full` holds the frame as handed over
    // (linear or sRGB, whichever it is) and `full_out` is where the transfer
    // pass writes the full-resolution answer; both are absent otherwise.
    uint32_t mw{}, mh{};
    bool scaled{};
    bool native_compose{};   // RuntimeConfig::native_compose
    // The post block restores the frame's alpha (and the 8-bit rounding) in its
    // own store, so the alpha pass is skipped: native compose, one pass, no
    // Model Resolution - there the answer is the post block's output as it is.
    bool post_alpha{};
    VkFilter downscale_filter{VK_FILTER_LINEAR};
    nrvk::Context::Image keep_full{}, full_out{};
    // `keep_native`: keep_full in the frame's own format (an *_SRGB frame's UNORM twin) rather than
    // RGBA32F. The frame is copied into it as it is and runtime_downscale.comp makes the model's
    // input from it, sampling where the linear blit would (`taps`, measured from a probe blit at
    // build). `store_native`: the transfer pass (runtime_transfer_store.spv) also writes the answer
    // back into keep_full, each pixel after reading it, with the alpha pass's rounding; then there
    // is no full_out and no alpha pass, and the write-back is a copy.
    bool keep_native{}, store_native{};
    nrvk::Kernel downscale;
    nrvk::Buffer taps{};
    // store_native with the caller's frame itself (usage SAMPLED and STORAGE, keep_full's format):
    // runtime_downscale.comp reads it and the transfer pass writes the answer back into it, so
    // neither the copy into keep_full nor the copy back is made. Views and sets per recording,
    // in a ring retired like DirectSrc's. transfer_binds / transfer_prep_binds: the images the two
    // transfer kernels were created with, [2] (keep) and [3] (answer) replaced by the frame's.
    struct FrameRw { VkImageView view{}; VkDescriptorPool pool{}; VkDescriptorSet down{}, transfer{}, transfer_prep{}; };
    std::vector<FrameRw> frame_ring;
    bool frame_rw_failed{}, frame_rw_logged{};
    std::vector<nrvk::Context::Image> transfer_binds, transfer_prep_binds;
    const FrameRw* frame_rw_source(VkImage image, VkFormat format) {
        if (frame_rw_failed) return nullptr;
        const VkDevice dev = session.ctx.device;
        if (frame_ring.empty()) frame_ring.resize(size_t(kRetireAfter) + 2);
        FrameRw& f = frame_ring[size_t(recordings % frame_ring.size())];
        if (f.view) { vkDestroyImageView(dev, f.view, nullptr); f.view = VK_NULL_HANDLE; }
        f.down = f.transfer = f.transfer_prep = VK_NULL_HANDLE;
        bool ok = true;
        if (!f.pool) {
            const VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                                                  {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6},
                                                  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 10}};
            VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pi.maxSets = 3; pi.poolSizeCount = 3; pi.pPoolSizes = sizes;
            ok = vkCreateDescriptorPool(dev, &pi, nullptr, &f.pool) == VK_SUCCESS;
        } else {
            ok = vkResetDescriptorPool(dev, f.pool, 0) == VK_SUCCESS;
        }
        if (ok) {
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = format;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            ok = vkCreateImageView(dev, &vi, nullptr, &f.view) == VK_SUCCESS;
        }
        if (ok) {
            nrvk::Context::Image sampled = keep_full;   // its sampler; texelFetch ignores filtering
            sampled.handle = image; sampled.view = f.view; sampled.layout = VK_IMAGE_LAYOUT_GENERAL;
            nrvk::Context::Image storage = sampled;
            storage.sampler = VK_NULL_HANDLE;
            nrvk::Context::Image tin = session.tex_in;
            tin.sampler = VK_NULL_HANDLE; tin.layout = VK_IMAGE_LAYOUT_GENERAL;
            f.down = kernel_set(dev, f.pool, downscale, {taps.handle}, {&sampled, &tin});
            auto set_for = [&](const nrvk::Kernel& k, std::vector<nrvk::Context::Image> binds) {
                binds[2] = sampled; binds[3] = storage;
                std::vector<const nrvk::Context::Image*> p;
                for (const auto& b : binds) p.push_back(&b);
                return kernel_set(dev, f.pool, k, {}, p);
            };
            f.transfer = set_for(transfer_pass, transfer_binds);
            if (transfer_prep.device) f.transfer_prep = set_for(transfer_prep, transfer_prep_binds);
            ok = f.down && f.transfer && (!transfer_prep.device || f.transfer_prep);
        }
        if (!ok) {
            if (f.view) { vkDestroyImageView(dev, f.view, nullptr); f.view = VK_NULL_HANDLE; }
            frame_rw_failed = true;
            nr::logf("[nr] reading and writing the frame in place failed; copying instead");
            return nullptr;
        }
        if (!frame_rw_logged) {
            frame_rw_logged = true;
            nr::logf("[nr] Model resolution: frame read and written in place (VkFormat %u)", unsigned(format));
        }
        return &f;
    }
    // Classic with a scaler other than bilinear: the model's answer enlarged to the frame by
    // runtime_upscale.comp.
    nrvk::Context::Image up_result{}, up_shown{};
    nrvk::Kernel upscale_pass, upscale_prep;
    // Multi-pass: what the first pass was shown, kept for the transfer pass
    // because later passes overwrite tex_in; and one history per pass.
    uint32_t max_passes{1};
    nrvk::Context::Image shown_keep{};
    // The detail-only cascade (RuntimeConfig::cascade_detail_only): the log
    // luminance ratio, its separable blur (through tmp), and the pass that
    // writes the next input. Model-sized, GENERAL for life.
    bool cascade_detail{};
    int cascade_radius{4};
    nrvk::Context::Image lograt{}, cascade_tmp{}, lowpass{};
    nrvk::Kernel cascade_lograt, cascade_blur_h, cascade_blur_v, cascade_feed;
    std::vector<nrvk::Context::Image> history_store;
    // ---- what the pass costs, measured on the GPU ----------------------------
    //
    // A timestamp pair around everything `record_all` puts in the caller's
    // command buffer. A **ring** of them, read several recordings behind and
    // never with the WAIT bit: the point of the number is to be free to ask for,
    // and a query that is still in flight must not stall the render thread that
    // is asking. `timing_ok` is false on a queue family with no timestamp bits,
    // and then every accessor answers zero rather than a wrong number.
    static constexpr uint32_t kTimingSlots = 4;
    VkQueryPool timing{};
    bool timing_ok{};
    double timestamp_ns{};       // VkPhysicalDeviceLimits::timestampPeriod
    uint32_t timing_slot{};      // the slot the next recording writes
    uint64_t timings_recorded{};
    float gpu_ms{}, gpu_ms_avg{};
    // Each pass's network is timed too: a pair around its dispatches, after the pair for the
    // whole. network_ms is their sum, the rest of gpu_ms is the route's own work (input,
    // enlargement and composition, copies).
    bool split_timing{};
    uint32_t timing_stride{2};             // queries per slot: 2 + 2 x max_passes
    uint32_t slot_passes[kTimingSlots]{};  // passes timed in each slot
    float net_ms{}, net_ms_avg{};
    // Read the oldest slot in the ring, if its recording has finished. Called
    // once per recording, from the thread that records.
    void read_timing() {
        if (!timing_ok || timings_recorded <= kTimingSlots) return;
        const uint32_t oldest = timing_slot;   // the next to be overwritten
        const uint32_t n = 2 + 2 * slot_passes[oldest];
        uint64_t stamps[2 + 2 * 16] = {};
        if (n > sizeof stamps / sizeof stamps[0]) return;
        if (vkGetQueryPoolResults(session.ctx.device, timing, oldest * timing_stride, n, n * sizeof(uint64_t), stamps,
                                  sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
            return;   // VK_NOT_READY: still running, ask again next frame
        if (stamps[1] <= stamps[0]) return;
        gpu_ms = float(double(stamps[1] - stamps[0]) * timestamp_ns / 1e6);
        if (slot_passes[oldest]) {
            uint64_t net = 0;
            for (uint32_t p = 0; p < slot_passes[oldest]; ++p)
                if (stamps[3 + 2 * p] > stamps[2 + 2 * p]) net += stamps[3 + 2 * p] - stamps[2 + 2 * p];
            net_ms = float(double(net) * timestamp_ns / 1e6);
            net_ms_avg = net_ms_avg > 0.0f ? net_ms_avg + (net_ms - net_ms_avg) * 0.1f : net_ms;
        }
        // A plain exponential average. Ten frames is enough to stop the last
        // digit dancing and short enough that a resolution change shows up
        // immediately rather than being dragged in from before it.
        gpu_ms_avg = gpu_ms_avg > 0.0f ? gpu_ms_avg + (gpu_ms - gpu_ms_avg) * 0.1f : gpu_ms;
    }
    // ---- one temporal state per feature -------------------------------------
    //
    // NVIDIA's DLL gives every NGX feature its own object, and OptiScaler builds
    // on that: one feature per model pass, features at two extents alive at
    // once, features destroyed and rebuilt whenever a control moves. What makes
    // a feature independent is its temporal state and nothing else - the
    // weights, the pipelines and the activation arena are properties of the
    // network and the extent, not of the handle.
    //
    // So a feature here is exactly this: one history image, the first-frame
    // latch and the estimator's parity. `temporal.history` stays the image the
    // pre/post descriptor sets are baked against - it is the *bound* history -
    // and a feature's own image is copied into it on the way in and back out on
    // the way out, but only when the bound feature actually changes. With one
    // feature, which is every caller that does not use this, nothing is copied
    // and the recording is byte-identical to what it always was.
    struct FeatureState {
        nrvk::Context::Image history{};
        bool latch{};
        uint32_t parity{};
        bool cleared{};
        uint32_t prep_gen{};   // Impl::prep_gen its history was made under
        uint32_t seed{};       // Temporal::seed
    };
    std::map<uint64_t, FeatureState> features;
    uint64_t bound_feature{};
    // Released features, and the recording count at which their images may be
    // destroyed. Never a wait: the render thread that releases a feature must
    // not block on the GPU, and OptiScaler has already parked the feature for 32
    // evaluates before it gets here (DlssNr_Dx12.cpp ParkNrFeature).
    static constexpr uint64_t kRetireAfter = 64;
    std::vector<std::pair<nrvk::Context::Image, uint64_t>> retiring;
    uint64_t recordings{};
    // **The tone control is a frame edit and belongs to one pass.** NVIDIA's own
    // description of `LocalToneStrength` is "low-frequency details such as
    // broader lighting and colour response", and it enters the network as five
    // constant features on the pre block - the same constant on every pixel.
    // Running the network again on its own answer therefore applies that whole
    // low-frequency edit a second time, and it compounds: measured on the
    // TR2013 frame against NVIDIA's own output, the block-mean luminance ratio
    // runs 1.052 -> 1.102 -> 1.136 over three passes with the tone control on
    // each time, and 1.052 -> 1.055 -> 1.056 when only the first pass carries
    // it. That drift is the red skin the user sees at more than one pass.
    // Structure is what a later pass can legitimately add more of, so only the
    // tone feature is dropped. One alternate push blob per dispatch, parallel to
    // `session.disp`, empty for every dispatch that is not the pre block.
    //
    // **Not under `cascade_detail_only`.** That feed hands the next pass the
    // first pass's *tone* with this pass's structure on it, so the tone edit has
    // been taken back out of the picture and the network has to make it again;
    // dropping the control there moves the two-pass answer away from the
    // one-pass one instead of towards it (block drift mean 0.0163 -> 0.0273 in
    // `linear_frame_test`). The two are alternative ways to stop the same
    // stacking, so exactly one of them is active at a time.
    // [pass][dispatch]; empty means "use pass 1's push". Pass 0's entry is
    // unused - that pass writes session.disp[i].push in place.
    std::vector<std::vector<std::vector<uint8_t>>> pass_push;
    nrvk::Context::Image mask_image;
    nrvk::Buffer mask_rect;
    // the plain path's pre and post (the graph's own pipelines), for sampling the
    // caller's colour in place without the temporal path (Impl::DirectSrc, plain sets).
    const nrvk::Kernel* plain_pre{};
    const nrvk::Kernel* plain_post{};
    const nrvk::Kernel* original_pre{};
    nrvk::Context::Image encoded;
    // An 8-bit frame: the answer is rounded as frame_image.py does before the
    // format-converting blit (alpha pass or the post block's store).
    bool round_u8{};
    Transfer transfer{Transfer::Identical};
    // The network's input image is in the frame's own format (its UNORM twin
    // for *_SRGB) and a plain copy fills it; see the graph's --tex-in-format.
    bool direct_in{};
    // The post block stores into an image in the frame's (UNORM) format and the
    // write-back is a copy; see the graph's --out0-format.
    bool direct_out{};
    // The engine path with one pass, native compose and ping-pong history can
    // sample the caller's colour and depth in place: the pre and post blocks
    // only sample them (texel centres, the same samplers), so a view of the
    // caller's image gives the values the copy into our own image gave, and
    // the copies go.
    //
    // **The views are made every frame, from the image handed over that frame,
    // and never looked up by handle.** A game (OptiScaler, toggling its NR
    // mode) destroys and recreates these resources, and the driver hands the
    // same VkImage value back for a new image - a cached view keyed by handle
    // then samples memory the old image no longer owns (in game: vertical
    // bands of another frame). Each recording takes the next slot of a ring
    // and retires what that slot held kRetireAfter + 2 recordings ago, the
    // rule the retired feature histories already live by.
    //
    // The same slots carry the game's motion vectors (mview), which the pre
    // and post blocks sample in place whenever the frame allows it, whether or
    // not the colour is: the original reads them per pixel at full resolution,
    // and the estimator's quarter-resolution field they used to be blitted into
    // blended foreground and background vectors at every silhouette and left the
    // five-tap depth choice a quarter of a texel to work with.
    struct DirectSrc {
        VkImageView cview{}, dview{}, mview{}, tview{};
        VkDescriptorPool pool{};
        VkDescriptorSet pre[2]{}, post[2]{};
        // the post block stores into the caller's image through cview
        // (a storage descriptor) and the write-back copy goes.
        bool store{};
    };
    std::vector<DirectSrc> direct_ring;
    bool direct_src_failed{}, direct_src_logged{};
    bool motion_src_failed{}, motion_src_logged{};
    // Formats whose sampled view filters linearly (the motion sampler is linear).
    std::map<VkFormat, bool> linear_ok;
    bool filters_linearly(VkFormat f) {
        auto it = linear_ok.find(f);
        if (it != linear_ok.end()) return it->second;
        VkFormatProperties fp{};
        vkGetPhysicalDeviceFormatProperties(session.ctx.physical, f, &fp);
        const bool ok = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
        linear_ok[f] = ok;
        return ok;
    }
    // A depth/stencil depth buffer is read through a depth-aspect view by
    // runtime_depth.comp (see there). The view is over the caller's image and
    // made per recording, in a ring retired like DirectSrc's.
    nrvk::Kernel depth_copy;
    struct DepthSrc { VkImageView view{}; VkDescriptorPool pool{}; };
    std::vector<DepthSrc> depth_ring;
    bool depth_src_failed{}, depth_src_logged{};
    bool depth_unread{};   // this recording's depth/stencil buffer could not be read
    VkDescriptorSet depth_source(VkImage image, VkFormat format) {
        if (depth_src_failed || !depth_copy.device) return VK_NULL_HANDLE;
        auto& t = temporal;
        const VkDevice dev = session.ctx.device;
        if (depth_ring.empty()) depth_ring.resize(size_t(kRetireAfter) + 2);
        DepthSrc& d = depth_ring[size_t(recordings % depth_ring.size())];
        if (d.view) { vkDestroyImageView(dev, d.view, nullptr); d.view = VK_NULL_HANDLE; }
        bool ok = true;
        if (!d.pool) {
            const VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                                                  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
            VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pi.maxSets = 1; pi.poolSizeCount = 2; pi.pPoolSizes = sizes;
            ok = vkCreateDescriptorPool(dev, &pi, nullptr, &d.pool) == VK_SUCCESS;
        } else {
            ok = vkResetDescriptorPool(dev, d.pool, 0) == VK_SUCCESS;
        }
        if (ok) {
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = format;
            vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            ok = vkCreateImageView(dev, &vi, nullptr, &d.view) == VK_SUCCESS;
        }
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (ok) {
            nrvk::Context::Image src = t.depth;           // its sampler; texelFetch ignores filtering
            src.handle = image; src.view = d.view; src.layout = VK_IMAGE_LAYOUT_GENERAL;
            nrvk::Context::Image dst = t.depth;
            dst.sampler = VK_NULL_HANDLE;                  // the storage alias
            set = kernel_set(dev, d.pool, depth_copy, {}, {&src, &dst});
            ok = set != VK_NULL_HANDLE;
        }
        if (!ok) {
            if (d.view) { vkDestroyImageView(dev, d.view, nullptr); d.view = VK_NULL_HANDLE; }
            depth_src_failed = true;
            nr::logf("[nr] the depth/stencil depth buffer could not be read (VkFormat %u); running without depth",
                     unsigned(format));
            return VK_NULL_HANDLE;
        }
        if (!depth_src_logged) {
            depth_src_logged = true;
            nr::logf("[nr] depth read from a depth/stencil buffer (VkFormat %u)", unsigned(format));
        }
        return set;
    }
    void direct_slot_release(DirectSrc& d) {
        const VkDevice dev = session.ctx.device;
        if (d.cview) vkDestroyImageView(dev, d.cview, nullptr);
        if (d.dview) vkDestroyImageView(dev, d.dview, nullptr);
        if (d.mview) vkDestroyImageView(dev, d.mview, nullptr);
        if (d.tview) vkDestroyImageView(dev, d.tview, nullptr);
        d.cview = d.dview = d.mview = d.tview = VK_NULL_HANDLE;
        d.pre[0] = d.pre[1] = d.post[0] = d.post[1] = VK_NULL_HANDLE;
        d.store = false;
    }
    // Any of colour (with its depth) and motion may be null: that one is our
    // own image, bound as in the sets made at creation.
    const DirectSrc* direct_source(VkImage colour, VkFormat format, VkImage depth, VkImage motion,
                                   VkFormat mformat, uint32_t c, bool store = false, VkImage target = VK_NULL_HANDLE,
                                   bool plain = false) {
        if (!colour && !motion) return nullptr;
        auto& s = session; auto& t = temporal;
        const VkDevice dev = s.ctx.device;
        if (direct_ring.empty()) direct_ring.resize(size_t(kRetireAfter) + 2);
        DirectSrc& d = direct_ring[size_t(recordings % direct_ring.size())];
        direct_slot_release(d);
        auto view = [&](VkImage im, VkFormat f, VkImageView* out) {
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = im; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = f;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            return vkCreateImageView(dev, &vi, nullptr, out) == VK_SUCCESS;
        };
        bool ok = true;
        if (!d.pool) {
            const VkDescriptorPoolSize sizes[] = {
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 7}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4}};
            VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pi.maxSets = 2; pi.poolSizeCount = 3; pi.pPoolSizes = sizes;
            ok = vkCreateDescriptorPool(dev, &pi, nullptr, &d.pool) == VK_SUCCESS;
        } else {
            ok = vkResetDescriptorPool(dev, d.pool, 0) == VK_SUCCESS;
        }
        const bool colour_ok = ok && (!colour || (view(colour, format, &d.cview) &&
                                                  (!depth || view(depth, VK_FORMAT_R32_SFLOAT, &d.dview))));
        const bool motion_ok = ok && (!motion || view(motion, mformat, &d.mview));
        ok = colour_ok && motion_ok && (!(colour && store && target) || view(target, format, &d.tview));
        if (ok) {
            // The caller's images, with our own samplers, in GENERAL; this
            // frame's history parity only.
            nrvk::Context::Image cim = s.tex_in, dim = t.depth, mim = t.flow0_sampled;
            if (colour) { cim.handle = colour; cim.view = d.cview; cim.layout = VK_IMAGE_LAYOUT_GENERAL; }
            if (depth) { dim.handle = depth; dim.view = d.dview; dim.layout = VK_IMAGE_LAYOUT_GENERAL; }
            if (motion) { mim.handle = motion; mim.view = d.mview; mim.layout = VK_IMAGE_LAYOUT_GENERAL; }
            nrvk::Context::Image surf0 = s.surf0;
            // the answer straight into the caller's image (same format as our own
            // answer image, storage usage): the post samples its own pixel's centre with
            // a NEAREST sampler before it stores that pixel, so nothing reads a stored texel.
            if (colour && store) {
                surf0.handle = target ? target : colour; surf0.view = target ? d.tview : d.cview;
                surf0.layout = VK_IMAGE_LAYOUT_GENERAL; surf0.sampler = VK_NULL_HANDLE; d.store = true;
            }
            if (plain) {
                // The graph's own pre and post: the buffers and images they were created with
                // (nr_graph.cpp), the caller's colour for tex_in.
                const VkBuffer a = s.act.handle, w = s.wgt.handle;
                nrvk::Context::Image surf1 = s.surf1;
                d.pre[0] = kernel_set(dev, d.pool, *plain_pre, {a, a, w, w, w, a}, {&cim});
                d.post[0] = kernel_set(dev, d.pool, *plain_post, {a, a, w, w, w}, {&surf0, &surf1, &cim});
                ok = d.pre[0] && d.post[0];
            } else {
            d.pre[c] = t.hp ? kernel_set(dev, d.pool, t.pre_pp[c], t.pre_buffers, {&cim, &mim, &t.hist(c), &dim, &t.hpass})
                            : kernel_set(dev, d.pool, t.pre_pp[c], t.pre_buffers, {&cim, &mim, &t.hist(c), &dim});
            d.post[c] = t.hp ? kernel_set(dev, d.pool, t.post_pp[c], t.post_buffers,
                                          {&surf0, &t.hist_store[c ^ 1u], &cim, &mim, &t.hist(c), &t.hpass})
                             : kernel_set(dev, d.pool, t.post_pp[c], t.post_buffers,
                                          {&surf0, &t.hist_store[c ^ 1u], &cim, &mim, &t.hist(c)});
            ok = d.pre[c] && d.post[c];
            }
        }
        if (!ok) {
            direct_slot_release(d);
            // Whichever view could not be made stops being tried; a set that
            // could not be written (both views made) stops both.
            if (colour && (!colour_ok || motion_ok)) direct_src_failed = true;
            if (motion && (!motion_ok || colour_ok)) motion_src_failed = true;
            nr::logf("[nr] sampling the caller's %s in place failed; copying instead",
                     colour && motion ? "colour and motion vectors" : colour ? "colour" : "motion vectors");
            return nullptr;
        }
        if (colour && !direct_src_logged) {
            direct_src_logged = true;
            nr::logf("[nr] colour%s sampled in place", depth ? " and depth" : "");
        }
        if (motion && !motion_src_logged) {
            motion_src_logged = true;
            nr::logf("[nr] motion vectors sampled in place at full resolution (VkFormat %u)", unsigned(mformat));
        }
        return &d;
    }
    Temporal temporal;
    VkFormat colour_format{};
    uint32_t width{}, height{};

    // ---- the in-game check (diagnostic) --------------------------------------
    //
    // For the in-game green screen (2026-09): a feature that stays wrong until
    // the next one, and nothing in the log. Every kEvery recordings the
    // recording also copies, after the network, the tile-counter region (frame
    // word, the waits' error words, the counters), each persistent run's epoch
    // and error word, and an 8x8 grid of the history the next frame reads, into
    // a host-visible buffer. kLag recordings later - that submission has long
    // run - the render thread reads it and logs whatever is off: a wait that
    // ran out of its bound (it then reads whatever is there), a counter that is
    // not need x frames, a run whose epoch is not the frame count, a history
    // holding NaN/Inf. Never a wait. Off unless NR_GAME_CHECK=1.
    struct Check {
        static constexpr uint64_t kEvery = 32, kLag = 16, kBeat = 1920;
        static constexpr uint32_t kGrid = 8;
        bool off{};
        nrvk::Buffer buf{};
        VkDeviceSize persist_at{}, hist_at{};
        uint64_t written_at{~0ull};   // the recording that copied, until read
        uint64_t feature{};           // the feature bound at that recording
        bool hist{};                  // that copy took the history
        size_t tc_errors{}, persist_errors{};
        size_t off_count{~size_t(0)};
        uint64_t beat_at{};
        std::set<uint64_t> hist_bad;
        std::map<uint64_t, int> feature_checks;
    };
    Check check;
    // NR_INPUT_CHECK (nr_input_check.hpp): what the engine path is handed and hands back.
    incheck::State incheck;
    void check_record(VkCommandBuffer cmd) {
        auto& k = check;
        auto& s = session;
        if (k.off) return;
        if (!k.buf.handle) {
            const char* e = std::getenv("NR_GAME_CHECK");
            if (!e || std::atoi(e) != 1) { k.off = true; return; }
            VkDeviceSize n = VkDeviceSize(s.tc_words) * 4;
            k.persist_at = n;
            n += VkDeviceSize(s.persist_err.size()) * 8;
            k.hist_at = n = (n + 15) & ~VkDeviceSize(15);
            n += VkDeviceSize(Check::kGrid) * Check::kGrid * 16;
            try {
                k.buf = s.ctx.buffer(n, true);
            } catch (const std::exception& x) {
                nr::logf("[nr] check: no host-visible buffer (%s), off", x.what());
                k.off = true;
                return;
            }
            nr::logf("[nr] check on (%ux%u): %zu chained pairs, %zu counters, %zu persistent runs, history %s",
                     width, height, s.tc_pair.size(), s.tc_expect.size(), s.persist_err.size(),
                     temporal.enabled ? "8x8 texels" : "none");
        }
        if (recordings % Check::kEvery) return;
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        std::vector<VkBufferCopy> r;
        if (s.tc_words) r.push_back({VkDeviceSize(s.tc_base) * 4, 0, VkDeviceSize(s.tc_words) * 4});
        for (size_t i = 0; i < s.persist_err.size(); ++i)   // epoch word, error word
            r.push_back({VkDeviceSize(s.persist_err[i]) - 4, k.persist_at + VkDeviceSize(i) * 8, 8});
        if (!r.empty()) vkCmdCopyBuffer(cmd, s.act.handle, k.buf.handle, uint32_t(r.size()), r.data());
        k.hist = temporal.enabled;
        if (k.hist) {
            const auto& h = temporal.hist(temporal.hcur);   // what the next frame reads
            std::vector<VkBufferImageCopy> ir;
            for (uint32_t y = 0; y < Check::kGrid; ++y)
                for (uint32_t x = 0; x < Check::kGrid; ++x) {
                    VkBufferImageCopy c{};
                    c.bufferOffset = k.hist_at + VkDeviceSize(y * Check::kGrid + x) * 16;
                    c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    c.imageOffset = {int32_t((2 * x + 1) * h.w / (2 * Check::kGrid)),
                                     int32_t((2 * y + 1) * h.h / (2 * Check::kGrid)), 0};
                    c.imageExtent = {1, 1, 1};
                    ir.push_back(c);
                }
            vkCmdCopyImageToBuffer(cmd, h.handle, VK_IMAGE_LAYOUT_GENERAL, k.buf.handle,
                                   uint32_t(ir.size()), ir.data());
        }
        mb.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
        k.written_at = recordings;
        k.feature = bound_feature;
    }
    void check_read() {
        auto& k = check;
        auto& s = session;
        if (!k.buf.handle || k.written_at == ~0ull || recordings < k.written_at + Check::kLag) return;
        const uint64_t at = k.written_at;
        k.written_at = ~0ull;
        const unsigned long long fid = (unsigned long long)k.feature;
        const uint32_t* w = k.buf.as<uint32_t>();
        const std::vector<uint32_t> v(w, w + s.tc_words);
        const uint32_t* p = w + k.persist_at / 4;
        const bool have_f = s.tc_words != 0;
        const uint32_t f = have_f ? v[0] : (s.persist_err.empty() ? 0u : p[0]);
        // Waits that ran out: sticky words, so the count only grows.
        size_t te = 0;
        std::string names;
        for (size_t i = 0; i < s.tc_err.size(); ++i)
            if (v[s.tc_err[i]]) {
                ++te;
                if (names.size() < 300 && i < s.tc_pair.size()) names += (names.empty() ? "" : ", ") + s.tc_pair[i];
            }
        if (te > k.tc_errors) {
            nr::logf("[nr] check: %zu tile-counter waits ran out by frame %u (recording %llu, feature %llx): %s",
                     te, f, (unsigned long long)at, fid, names.c_str());
            k.tc_errors = te;
        }
        size_t pe = 0;
        std::string layers;
        for (size_t i = 0; i < s.persist_err.size(); ++i)
            if (p[2 * i + 1]) {
                ++pe;
                layers += " run " + std::to_string(i) + " layer " + std::to_string(p[2 * i + 1] - 1u);
            }
        if (pe > k.persist_errors) {
            nr::logf("[nr] check: %zu persistent-run waits ran out by frame %u (recording %llu, feature %llx):%s",
                     pe, f, (unsigned long long)at, fid, layers.c_str());
            k.persist_errors = pe;
        }
        // After whole frames every counter holds need x frames and every run's epoch is the frame count.
        size_t bad = 0, epoch_bad = 0;
        std::string ex;
        if (have_f) {
            for (const auto& e : s.tc_expect) {
                const uint32_t want = e.need * f;
                if (v[e.word] == want) continue;
                if (bad < 4) {
                    char b[160];
                    std::snprintf(b, sizeof b, " [%s: counter %u = %u, want %u x %u]",
                                  e.pair < s.tc_pair.size() ? s.tc_pair[e.pair].c_str() : "?", e.word, v[e.word], e.need, f);
                    ex += b;
                }
                ++bad;
            }
            for (size_t i = 0; i < s.persist_err.size(); ++i)
                if (p[2 * i] != f) {
                    if (epoch_bad < 4) ex += " [run " + std::to_string(i) + " epoch " + std::to_string(p[2 * i]) + "]";
                    ++epoch_bad;
                }
        }
        if (have_f && bad + epoch_bad != k.off_count) {
            if (bad + epoch_bad)
                nr::logf("[nr] check: frame %u (recording %llu, feature %llx): %zu of %zu counters and %zu of %zu "
                         "run epochs off:%s", f, (unsigned long long)at, fid, bad, s.tc_expect.size(), epoch_bad,
                         s.persist_err.size(), ex.c_str());
            else
                nr::logf("[nr] check: frame %u: all %zu counters = need x frames, %zu run epochs = frames%s", f,
                         s.tc_expect.size(), s.persist_err.size(), k.off_count == ~size_t(0) ? "" : " again");
            k.off_count = bad + epoch_bad;
        }
        // The history the next frame reads.
        if (!k.hist) return;
        const float* h = reinterpret_cast<const float*>(w + k.hist_at / 4);
        const uint32_t n = Check::kGrid * Check::kGrid;
        uint32_t nonfinite = 0;
        float peak = 0.0f;
        double mean[3] = {};
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t c = 0; c < 4; ++c) {
                const float x = h[i * 4 + c];
                if (!std::isfinite(x)) { ++nonfinite; continue; }
                peak = std::max(peak, std::fabs(x));
                if (c < 3) mean[c] += x / n;
            }
        const bool weird = nonfinite || peak > 1024.0f;
        int& seen = k.feature_checks[k.feature];
        ++seen;
        if (weird && k.hist_bad.insert(k.feature).second)
            nr::logf("[nr] check: history of feature %llx at frame %u: %u NaN/Inf of %u values, max |v| %g, "
                     "mean %.4f %.4f %.4f", fid, f, nonfinite, n * 4, double(peak), mean[0], mean[1], mean[2]);
        else if (seen <= 2 || at >= k.beat_at + Check::kBeat) {
            // A new feature's first two looks, then a heartbeat.
            nr::logf("[nr] check: frame %u, feature %llx: history mean %.4f %.4f %.4f, max %.4g; waits ran out "
                     "%zu+%zu, counters off %zu", f, fid, mean[0], mean[1], mean[2], double(peak), k.tc_errors,
                     k.persist_errors, k.off_count == ~size_t(0) ? size_t(0) : k.off_count);
            if (seen > 2) k.beat_at = at;
        }
        if (k.feature_checks.size() > 256) k.feature_checks.clear();
    }

    ~Impl() {
        // Caller has already completed its submitted command buffers.
        auto& s = session;
        if (!s.ctx.device) return;
        for (auto& d : direct_ring) {
            direct_slot_release(d);
            if (d.pool) vkDestroyDescriptorPool(s.ctx.device, d.pool, nullptr);
        }
        for (auto& d : depth_ring) {
            if (d.view) vkDestroyImageView(s.ctx.device, d.view, nullptr);
            if (d.pool) vkDestroyDescriptorPool(s.ctx.device, d.pool, nullptr);
        }
        if (depth_copy.device) depth_copy.destroy();
        if (alpha.device) alpha.destroy();
        if (encode.device) encode.destroy();
        if (transfer_pass.device) transfer_pass.destroy();
        if (prep_k.device) prep_k.destroy();
        if (prep_back_k.device) prep_back_k.destroy();
        if (transfer_prep.device) transfer_prep.destroy();
        if (prep_keep.handle) session.ctx.destroy(prep_keep);
        if (prep_state.handle) session.ctx.destroy(prep_state);
        if (keep.handle) session.ctx.destroy(keep);
        if (keep_full.handle) session.ctx.destroy(keep_full);
        if (downscale.device) downscale.destroy();
        if (taps.handle) session.ctx.destroy(taps);
        for (auto& f : frame_ring) {
            if (f.view) vkDestroyImageView(s.ctx.device, f.view, nullptr);
            if (f.pool) vkDestroyDescriptorPool(s.ctx.device, f.pool, nullptr);
        }
        if (up_result.handle) session.ctx.destroy(up_result);
        if (up_shown.handle) session.ctx.destroy(up_shown);
        if (upscale_pass.device) upscale_pass.destroy();
        if (upscale_prep.device) upscale_prep.destroy();
        if (full_out.handle) session.ctx.destroy(full_out);
        if (shown_keep.handle) session.ctx.destroy(shown_keep);
        if (lograt.handle) session.ctx.destroy(lograt);
        if (cascade_tmp.handle) session.ctx.destroy(cascade_tmp);
        if (lowpass.handle) session.ctx.destroy(lowpass);
        if (cascade_lograt.device) cascade_lograt.destroy();
        if (cascade_blur_h.device) cascade_blur_h.destroy();
        if (cascade_blur_v.device) cascade_blur_v.destroy();
        if (cascade_feed.device) cascade_feed.destroy();
        for (auto& h : history_store) session.ctx.destroy(h);
        for (auto& f : features) if (f.second.history.handle) session.ctx.destroy(f.second.history);
        for (auto& r : retiring) if (r.first.handle) session.ctx.destroy(r.first);
        if (mask_pre.device) mask_pre.destroy();
        if (mask_resolve.device) mask_resolve.destroy();
        if (timing) { vkDestroyQueryPool(s.ctx.device, timing, nullptr); timing = VK_NULL_HANDLE; }
        if (check.buf.handle) s.ctx.destroy(check.buf);
        incheck.destroy(s.ctx);
        temporal.destroy(s.ctx);
        if (s.ctx.pipeline_cache) {
            vkDestroyPipelineCache(s.ctx.device, s.ctx.pipeline_cache, nullptr);
            s.ctx.pipeline_cache = VK_NULL_HANDLE;
        }
        s.ctx.destroy(mask_image);
        s.ctx.destroy(mask_rect);
        if (s.runner.ctx) s.runner.destroy();
        for (auto& entry : s.kern) if (entry.second.device) entry.second.destroy();
        s.ctx.destroy(s.tex_in); s.ctx.destroy(s.surf0); s.ctx.destroy(s.surf1);
        s.ctx.destroy(encoded);
        s.ctx.destroy(s.act); s.ctx.destroy(s.wgt);
        // Do not destroy the host's instance, device or queue.
    }

    // One pass's resolved model controls. A later pass inherits pass 1 with the
    // tone zeroed - that is what the original's own hosts do, and re-applying
    // the tone control every pass compounds it (x1.052 -> x1.102 -> x1.136 over
    // three) - unless the caller gave that pass its own set.
    struct Resolved { int style; float intensity, tone, structure, skin_in; bool mask; };

    Resolved resolve_pass(const Controls& c, uint32_t pass) const {
        Resolved r{c.style, c.intensity, c.local_tone, c.local_structure, c.skin_structure,
                   c.automatic_mask};
        if (pass == 0) return r;
        if (!cascade_detail) r.tone = 0.0f;
        const size_t k = size_t(pass) - 1;
        if (k < c.per_pass.size() && c.per_pass[k].used) {
            const PassControls& o = c.per_pass[k];
            r.style = o.style; r.intensity = o.intensity; r.tone = o.local_tone;
            r.structure = o.local_structure; r.skin_in = o.skin_structure; r.mask = o.automatic_mask;
        }
        return r;
    }

    // Which dispatches carry a control at all. Everything else uses pass 1's
    // push, so a later pass costs four small blobs rather than a copy of the
    // whole table.
    static bool carries_controls(const std::string& kern) {
        return kern.rfind("fswinimagepreds32", 0) == 0 || kern.rfind("imgin", 0) == 0 ||
               kern == "fswinimagepost32" || kern.rfind("imgout", 0) == 0;
    }

    void patch_push(std::vector<uint8_t>& blob, const std::string& kern, const Resolved& r) const {
        const float skin = r.mask ? (r.skin_in < 0 ? r.structure : r.skin_in) : -1.0f;
        const float background = r.mask ? r.structure : -1.0f;
        if (kern.rfind("fswinimagepreds32", 0) == 0) {
            PushPreImage p{};
            std::memcpy(&p, blob.data() + sizeof(PushFSwin), sizeof p);
            p.style = float(r.style) / 128; p.tone = r.tone;
            p.structure = r.mask ? 1 : r.structure;
            p.skin = skin; p.other = background;
            std::memcpy(blob.data() + sizeof(PushFSwin), &p, sizeof p);
        } else if (kern.rfind("imgin", 0) == 0) {
            PushImgIn p{}; std::memcpy(&p, blob.data(), sizeof p);
            p.s434 = float(r.style) / 128; p.aux0 = r.tone;
            p.aux1 = r.mask ? 1 : r.structure;
            p.gate0 = skin; p.gate1 = background;
            std::memcpy(blob.data(), &p, sizeof p);
        } else if (kern == "fswinimagepost32") {
            constexpr size_t offset = sizeof(PushFSwin) + sizeof(PushUps);
            PushImageTail p{}; std::memcpy(&p, blob.data() + offset, sizeof p);
            // Native compose (the OptiScaler and ReShade routes): the fused
            // tail's own blend IS the DLL's post-block Intensity. **Not clamped
            // to 1**: that clamp came from one post variant's `sat(Intensity)`
            // in the PTX, which governs that variant's lerp and not the
            // control - the DLL copies Intensity out of its settings
            // unsaturated (18001cf07) and the Toolkit documents 0..2. The tail
            // takes 0..1 as the lerp and past it as a luminance-ratio power;
            // see windows/shaders/rdna4/fswin_t.comp.
            // Every route, every pass: each pass carries its own Intensity
            // into this blend, as each OptiScaler DLSS-NR pass is its own DLL
            // feature with its own Intensity.
            p.intensity = std::clamp(r.intensity, 0.0f, 2.0f);
            p.w_off = (p.w_off & 0x3FFFFFFFu) |
                      (post_alpha ? 0x80000000u | (round_u8 ? 0x40000000u : 0u) : 0u);
            std::memcpy(blob.data() + offset, &p, sizeof p);
        } else if (kern.rfind("imgout", 0) == 0) {
            PushImgOut p{}; std::memcpy(&p, blob.data(), sizeof p);
            p.nr_intensity = std::clamp(r.intensity, 0.0f, 2.0f);
            std::memcpy(blob.data(), &p, sizeof p);
        }
    }

    void controls(const Controls& c) {
        const uint32_t npass = std::max<uint32_t>(max_passes, 1u);
        pass_push.assign(npass, {});
        for (auto& d : session.disp) patch_push(d.push, d.kern, resolve_pass(c, 0));
        for (uint32_t p = 1; p < npass; ++p) {
            const Resolved r = resolve_pass(c, p);
            pass_push[p].assign(session.disp.size(), {});
            for (size_t i = 0; i < session.disp.size(); ++i) {
                if (!carries_controls(session.disp[i].kern)) continue;
                pass_push[p][i] = session.disp[i].push;
                patch_push(pass_push[p][i], session.disp[i].kern, r);
            }
        }
    }
};

Runtime::Runtime(const HostDevice& host, const RuntimeConfig& config) : Runtime(host,config,ControlMaskConfig{}) {}
Runtime::Runtime(const HostDevice& host, const RuntimeConfig& config, const ControlMaskConfig& mask_config)
    : Runtime(host,config,mask_config,TemporalConfig{}) {}
Runtime::Runtime(const HostDevice& host, const RuntimeConfig& config, const ControlMaskConfig& mask_config,
                 const TemporalConfig& temporal_config)
    : impl_(std::make_unique<Impl>()) {
    runtime_dispatch::Scope physical_scope(host.instance, host.physical_dispatch);
    if (!host.instance || !host.physical || !host.device || !host.queue || !config.width || !config.height)
        throw std::invalid_argument("NR requires complete host device and source extent");
    if (config.accumulation != "fp32" && config.accumulation != "round-fp16")
        throw std::invalid_argument("unsupported accumulation policy");
    const Transfer transfer = transfer_mode(config.colour_format);
    const VkFormat unorm = encoded_format(config.colour_format);
    impl_->transfer = transfer;
    impl_->round_u8 = config.colour_format == VK_FORMAT_R8G8B8A8_UNORM || config.colour_format == VK_FORMAT_R8G8B8A8_SRGB ||
                      config.colour_format == VK_FORMAT_B8G8R8A8_UNORM || config.colour_format == VK_FORMAT_B8G8R8A8_SRGB;
    impl_->colour_format = config.colour_format;
    const auto root = std::filesystem::canonical(config.root);
    // An installed copy keeps everything it loads in dlssnr-amd/ beside the DLL:
    // the model pack and shaders/ (network, runtime/ passes, temporal/). A
    // development tree has build/ and artifacts/ at its root instead.
    const auto data = std::filesystem::is_directory(root / "dlssnr-amd") ? root / "dlssnr-amd" : root;
    const bool installed = data != root;
    // The ACO network (nr_pipeline_binary.hpp) needs a device created with pipeline binaries; one that
    // was not (a game's device made before we could add them) builds with the driver's compiler.
    if (!nr::binary::directory().empty() && !nr::binary::enabled(host.device))
        nr::binary::disable("the device was created without pipeline binaries");
    const std::string aco = nr::binary::directory();
    const auto native_shaders = installed ? data / "shaders" : root / "build";
    const auto network_shaders = !aco.empty() ? std::filesystem::path(aco) / "shaders" : native_shaders;
    // Shaders are read from inside the NR root, or from the ACO bundle when one is in use.
    const auto inside = [&](const std::filesystem::path& p) {
        for (const auto& base : {root, aco.empty() ? root : std::filesystem::canonical(aco)}) {
            const auto rel = p.lexically_relative(base);
            if (!rel.empty() && *rel.begin() != "..") return true;
        }
        return false;
    };
    if (!std::isfinite(config.model_scale) || config.model_scale <= 0.f || config.model_scale > 1.f)
        throw std::invalid_argument("model_scale must be in (0, 1]");
    if (config.max_passes < 1 || config.max_passes > 16)
        throw std::invalid_argument("max_passes must be 1..16");
    // The model extent. At scale 1 it is the frame's own, unaligned, as every
    // game has run so far; below 1 it is rounded to a multiple of 8 (the
    // network's tile) and never above the frame.
    auto model_extent = [&](uint32_t v) {
        if (config.model_scale == 1.0f) return v;
        uint32_t m = uint32_t(std::lround(double(v) * config.model_scale));
        m = (m + 4) / 8 * 8;
        return std::min(std::max(m, 8u), v);
    };
    const uint32_t mw = model_extent(config.width), mh = model_extent(config.height);
    impl_->mw = mw; impl_->mh = mh; impl_->scaled = mw != config.width || mh != config.height;
    impl_->native_compose = config.native_compose;
    if (config.native_compose && impl_->scaled)
        throw std::runtime_error("native_compose needs model_scale 1: the DLL has no transfer pass to resample through");
    impl_->max_passes = config.max_passes;
    const auto native = make_native_plan(mw, mh);
    std::string plan_name = "compiled native descriptor";
    if (!config.plan.empty()) {
        const auto path = std::filesystem::canonical(config.plan);
        const auto rel = path.lexically_relative(root);
        if (rel.empty() || *rel.begin() == "..") throw std::invalid_argument("plan must be inside NR root");
        auto tokens = [](std::istream& input) {
            std::vector<std::string> result; std::string line, word;
            while (std::getline(input, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream words(line); while (words >> word) result.push_back(word);
            }
            return result;
        };
        std::ifstream provided(path); std::istringstream expected(native.text);
        if (!provided || tokens(provided) != tokens(expected))
            throw std::invalid_argument("explicit plan does not match native source-sized descriptor plan");
        plan_name = path.string();
    }
    std::istringstream plan_stream(native.text);
    const auto prepared_plan = parse_plan(plan_stream);
    nr::PhaseTimer timer;
    std::lock_guard<std::mutex> lock(build_mutex);
    auto& s = impl_->session;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(host.physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(host.physical, &count, props.data());
    if (host.queue_family >= count || !(props[host.queue_family].queueFlags & VK_QUEUE_COMPUTE_BIT))
        throw std::invalid_argument("NR host queue must support compute");
    if (transfer != Transfer::Identical && !(props[host.queue_family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
        throw std::invalid_argument("NR colour format conversion requires a graphics-capable queue");
    std::string why;
    if (!s.ctx.adopt(host.instance, host.physical, host.device, host.queue, host.queue_family, &why))
        throw std::runtime_error("NR device unsupported: " + why);
    s.ctx.queue_lock = host.queue_lock; s.ctx.queue_unlock = host.queue_unlock;
    s.ctx.buffer_device_address = host.buffer_device_address;
    s.ctx.require_matrix_config();
    // Where the pass's own cost is measured. Two queries per ring slot. A queue
    // family is allowed to report no timestamp bits - then there is no pool, and
    // every accessor answers zero rather than a number made up from a clock.
    if (props[host.queue_family].timestampValidBits && s.ctx.timestamp_period > 0.0f) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        impl_->split_timing = true;
        impl_->timing_stride = 2 + 2 * config.max_passes;
        qi.queryCount = Impl::kTimingSlots * impl_->timing_stride;
        if (vkCreateQueryPool(s.ctx.device, &qi, nullptr, &impl_->timing) == VK_SUCCESS) {
            impl_->timing_ok = true;
            impl_->timestamp_ns = s.ctx.timestamp_period;
        }
    }
    // Values share arena memory by lifetime (nr_graph.cpp, `reuse`): same
    // arithmetic, same bytes out, a fraction of the memory. NR_ARENA_REUSE=0
    // gives every value its own bytes again, for an A/B.
    auto arena_reuse = [] {
        const char* e = std::getenv("NR_ARENA_REUSE");
        return e && *e ? std::atoi(e) != 0 : kArenaReuseDefault;
    };
    std::vector<std::string> arguments = {"nr_runtime", "--plan", plan_name,
        "--unpacked", (installed ? data / "model" : root / "artifacts").string(),
        "--spv-dir", network_shaders.string(),
        "--host-boundary", arena_reuse() ? "--reuse" : "--no-reuse", "--source-width", std::to_string(mw),
        "--source-height", std::to_string(mh), "--accumulation", config.accumulation,
        "--pipeline-cache", (installed ? data / "pipeline.cache" : root / "dlssnr-amd.pipelinecache").string()};
    // The weights as one file (linux/package/model-tools/pack_model.py); the names inside are
    // the relative paths the graph asks for under --unpacked.
    if (installed) {
        const auto pack = data / "dlssnr.bin";
        if (!std::filesystem::is_regular_file(pack))
            throw std::runtime_error("missing " + pack.string() + " (run install.sh again)");
        arguments.push_back("--model-pack"); arguments.push_back(pack.string());
    }
    // One pass at the frame's own extent, not linear: every kernel only samples
    // the input image, so it can be the frame's own format filled by a copy.
    // Sampling converts exactly as the blit into RGBA32F did (the blit is a
    // sampling shader too) - the same values for a quarter of the bytes, and
    // no conversion pass. Anything else keeps RGBA32F: later passes copy their
    // answer into it, and the linear path writes it as storage.
    {
        const VkFormat in_format = transfer == Transfer::Encoded ? unorm : config.colour_format;
        VkFormatProperties fp{};
        vkGetPhysicalDeviceFormatProperties(host.physical, in_format, &fp);
        const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        impl_->direct_in = NR_DIRECT_IN && transfer != Transfer::Identical && !impl_->scaled &&
                           !config.linear_input && !config.preprocess && config.max_passes == 1 &&
                           mask_config.width == 0 &&
                           (fp.optimalTilingFeatures & need) == need;
        if (impl_->direct_in) {
            arguments.push_back("--tex-in-format");
            arguments.push_back(std::to_string(unsigned(in_format)));
        }
        // The post block's own store: native compose folds the alpha pass into
        // it, so nothing reads the answer image but the write-back, and the
        // store's conversion is the blit's for these formats (byte-compared at
        // 1080p, 6 temporal frames: 8-bit UNORM/SRGB - the post block rounds
        // those to k/255 itself - FP16, 16-bit UNORM, 11/11/10 float). Not the
        // 10-bit UNORM ones: there the blit comes out one step lower on ~18%
        // of channels. Formatless storage writes: the post SPIR-V declares no
        // format (NR_OUT0_NOFORMAT), which Vulkan 1.3 allows for a format with
        // STORAGE_WRITE_WITHOUT_FORMAT.
        const bool store_matches_blit = impl_->round_u8 || config.colour_format == VK_FORMAT_R16G16B16A16_SFLOAT ||
            config.colour_format == VK_FORMAT_R16G16B16A16_UNORM || config.colour_format == VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        VkFormatProperties3 f3{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
        VkFormatProperties2 f2{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &f3};
        vkGetPhysicalDeviceFormatProperties2(host.physical, unorm, &f2);
        impl_->direct_out = NR_DIRECT_OUT && NR_POST_ALPHA && impl_->direct_in && config.native_compose && store_matches_blit &&
                            mask_config.width == 0 &&
                            (f3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT) &&
                            (f3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT);
        nr::logf("[nr] input %s, write-back %s", impl_->direct_in ? "copied in the frame's format"
                 : transfer == Transfer::Identical ? "copied (RGBA32F)" : "converted",
                 impl_->direct_out ? "copied from the post block's own store" : "converted");
        if (impl_->direct_out) {
            arguments.push_back("--out0-format");
            arguments.push_back(std::to_string(unsigned(unorm)));
        }
    }
    std::vector<char*> argv;
    for (auto& a : arguments) argv.push_back(a.data());
    timer.mark("device");
    if (s.build(int(argv.size()), argv.data(), &prepared_plan) != 0 || !s.missing.empty())
        throw std::runtime_error("NR graph initialization failed");
    timer.mark("graph");
    impl_->width = config.width; impl_->height = config.height;
    // The runtime passes are the driver's compile in either case (the ACO bundle's are the same SPIR-V).
    const auto adapters = std::filesystem::canonical(config.adapter_shaders.empty()
        ? native_shaders / "runtime" : std::filesystem::path(config.adapter_shaders));
    if (!inside(adapters))
        throw std::invalid_argument("adapter shaders must be inside NR root");
    if (impl_->scaled) {
        if (mask_config.width) throw std::invalid_argument("the control mask path does not support model_scale below 1");
        // The downscale is a blit from the RGBA32F keep; linear filtering on a
        // 32-bit float format is a feature bit, not a given.
        VkFormatProperties fp{}; vkGetPhysicalDeviceFormatProperties(host.physical, VK_FORMAT_R32G32B32A32_SFLOAT, &fp);
        impl_->downscale_filter = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)
            ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        // The frame kept in its own format (Impl::keep_native) for the formats whose texelFetch is
        // the blit's own conversion; the store into it where that store is the blit's conversion
        // too (the formats the post block's own store covers, see direct_out below).
        // NR_KEEP_RGBA32F=1: the RGBA32F keep and the blits, as before.
        const VkFormat keep_format = transfer == Transfer::Encoded ? unorm : config.colour_format;
        const VkFormat cf = config.colour_format;
        const bool listed = transfer == Transfer::Encoded || cf == VK_FORMAT_R8G8B8A8_UNORM ||
            cf == VK_FORMAT_B8G8R8A8_UNORM || cf == VK_FORMAT_R16G16B16A16_SFLOAT || cf == VK_FORMAT_R16G16B16A16_UNORM ||
            cf == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || cf == VK_FORMAT_A2R10G10B10_UNORM_PACK32 ||
            cf == VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        const bool store_listed = impl_->round_u8 || cf == VK_FORMAT_R16G16B16A16_SFLOAT ||
            cf == VK_FORMAT_R16G16B16A16_UNORM || cf == VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        VkFormatProperties3 k3{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
        VkFormatProperties2 k2{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &k3};
        vkGetPhysicalDeviceFormatProperties2(host.physical, keep_format, &k2);
        const VkFormatFeatureFlags2 kf = k3.optimalTilingFeatures;
        const char* old_keep = std::getenv("NR_KEEP_RGBA32F");
        impl_->keep_native = !(old_keep && std::atoi(old_keep)) && listed && impl_->downscale_filter == VK_FILTER_LINEAR &&
            s.tex_in.format == VK_FORMAT_R32G32B32A32_SFLOAT && (kf & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT) &&
            (kf & VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT) && (kf & VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT);
        impl_->store_native = impl_->keep_native && store_listed && (kf & VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT) &&
            (kf & VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT);
        impl_->keep_full = s.ctx.image(config.width, config.height,
                                       impl_->keep_native ? keep_format : VK_FORMAT_R32G32B32A32_SFLOAT, true, true,
                                       !impl_->keep_native || impl_->store_native);
        s.ctx.transition(impl_->keep_full, VK_IMAGE_LAYOUT_GENERAL);
        if (!impl_->store_native) {
            impl_->full_out = s.ctx.image(config.width, config.height, VK_FORMAT_R32G32B32A32_SFLOAT, false);
            s.ctx.transition(impl_->full_out, VK_IMAGE_LAYOUT_GENERAL);
        }
        for (auto* im : {&impl_->up_result, &impl_->up_shown}) {
            *im = s.ctx.image(config.width, config.height, VK_FORMAT_R32G32B32A32_SFLOAT, false);
            s.ctx.transition(*im, VK_IMAGE_LAYOUT_GENERAL);
        }
        if (impl_->keep_native) {
            // Where the blit samples: a frame-sized probe holding each texel's column and row,
            // blitted down as the frame would be, read back into `taps` on the GPU. Frame and
            // model sized RGBA32F for the length of one submission.
            const uint32_t W = config.width, H = config.height;
            impl_->taps = s.ctx.buffer(VkDeviceSize(mw + mh) * 8);
            auto probe = s.ctx.image(W, H, VK_FORMAT_R32G32B32A32_SFLOAT, false, false);
            auto probed = s.ctx.image(mw, mh, VK_FORMAT_R32G32B32A32_SFLOAT, false, false);
            probe.layout = probed.layout = VK_IMAGE_LAYOUT_GENERAL;
            nrvk::Kernel taps_k;
            taps_k.create(s.ctx, (adapters / "runtime_taps.spv").string(), {impl_->taps.handle}, 12, {&probe, &probed});
            s.ctx.one_shot([&](VkCommandBuffer cmd) {
                barrier(cmd, probe.handle, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
                const uint32_t fill[] = {W, H, 0u};
                dispatch(cmd, taps_k, (W + 63) / 64, H, 1, fill, sizeof fill);
                barrier(cmd, probe.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
                barrier(cmd, probed.handle, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                VkImageBlit down{};
                down.srcSubresource = down.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                down.srcOffsets[1] = {int32_t(W), int32_t(H), 1};
                down.dstOffsets[1] = {int32_t(mw), int32_t(mh), 1};
                vkCmdBlitImage(cmd, probe.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               probed.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &down, impl_->downscale_filter);
                barrier(cmd, probed.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
                const uint32_t read[] = {mw, mh, 1u};
                dispatch(cmd, taps_k, (mw + mh + 63) / 64, 1, 1, read, sizeof read);
                compute_barrier(cmd);
            });
            taps_k.destroy();
            s.ctx.destroy(probe); s.ctx.destroy(probed);
            nrvk::Context::Image tex_in_storage = s.tex_in;
            tex_in_storage.sampler = VK_NULL_HANDLE;
            tex_in_storage.layout = VK_IMAGE_LAYOUT_GENERAL;
            impl_->downscale.create(s.ctx, (adapters / "runtime_downscale.spv").string(), {impl_->taps.handle}, 8,
                                    {&impl_->keep_full, &tex_in_storage});
        }
        nr::logf("[nr] Model resolution %ux%u: frame kept %s, answer %s", mw, mh,
                 impl_->keep_native ? "in its own format" : "as RGBA32F",
                 impl_->store_native ? "stored in its own format" : "RGBA32F, converted on the way back");
    }
    if (config.max_passes > 1) {
        if (mask_config.width) throw std::invalid_argument("the control mask path does not support more than one pass");
        impl_->shown_keep = s.ctx.image(mw, mh, VK_FORMAT_R32G32B32A32_SFLOAT, true);
        s.ctx.transition(impl_->shown_keep, VK_IMAGE_LAYOUT_GENERAL);
        if (config.cascade_detail_only) {
            impl_->cascade_detail = true;
            impl_->cascade_radius = std::max(4, int(mh / 60));
            for (auto* im : {&impl_->lograt, &impl_->cascade_tmp, &impl_->lowpass}) {
                *im = s.ctx.image(mw, mh, VK_FORMAT_R32_SFLOAT, false);
                s.ctx.transition(*im, VK_IMAGE_LAYOUT_GENERAL);
            }
            nrvk::Context::Image tex_in_storage = s.tex_in;
            tex_in_storage.sampler = VK_NULL_HANDLE;
            tex_in_storage.layout = VK_IMAGE_LAYOUT_GENERAL;
            impl_->cascade_lograt.create(s.ctx, (adapters / "cascade_lograt.spv").string(), {}, 12,
                                         {&s.surf0, &impl_->shown_keep, &impl_->lograt});
            impl_->cascade_blur_h.create(s.ctx, (adapters / "cascade_blur.spv").string(), {}, 16,
                                         {&impl_->lograt, &impl_->cascade_tmp});
            impl_->cascade_blur_v.create(s.ctx, (adapters / "cascade_blur.spv").string(), {}, 16,
                                         {&impl_->cascade_tmp, &impl_->lowpass});
            impl_->cascade_feed.create(s.ctx, (adapters / "cascade_feed.spv").string(), {}, 12,
                                       {&impl_->shown_keep, &impl_->lograt, &impl_->lowpass, &tex_in_storage});
        }
    }
    // The alpha pass restores the frame's own alpha onto the answer: the answer
    // is full_out when scaled, and the frame's alpha lives wherever the frame
    // was kept.
    // store_native: the answer is written into keep_full (a storage alias of it).
    nrvk::Context::Image keep_full_storage = impl_->keep_full;
    keep_full_storage.sampler = VK_NULL_HANDLE;
    nrvk::Context::Image* answer = impl_->store_native ? &keep_full_storage : impl_->scaled ? &impl_->full_out : &s.surf0;
    nrvk::Context::Image* frame_alpha = impl_->scaled ? &impl_->keep_full
                                        : (config.max_passes > 1 ? &impl_->shown_keep : &s.tex_in);
    if (!impl_->store_native)
        impl_->alpha.create(s.ctx, (adapters / "runtime_alpha.spv").string(), {}, 12, {answer, frame_alpha});
    if (config.linear_input) {
        impl_->linear = true;
        impl_->white_point = config.white_point;
        // Sampled, because the transfer reads it; the encode writes it through
        // a storage alias (same image and view, no sampler, GENERAL).
        impl_->keep = s.ctx.image(mw, mh, VK_FORMAT_R32G32B32A32_SFLOAT, true);
        s.ctx.transition(impl_->keep, VK_IMAGE_LAYOUT_GENERAL);
        nrvk::Context::Image keep_storage = impl_->keep;
        keep_storage.sampler = VK_NULL_HANDLE;
        // tex_in as a storage image, in GENERAL for the one pass that writes it.
        // Same VkImage and view as the sampled descriptor every kernel reads;
        // the alias owns nothing and is never destroyed.
        nrvk::Context::Image tex_in_storage = s.tex_in;
        tex_in_storage.sampler = VK_NULL_HANDLE;
        tex_in_storage.layout = VK_IMAGE_LAYOUT_GENERAL;
        impl_->encode.create(s.ctx, (adapters / "runtime_encode.spv").string(), {}, 16,
                             {&tex_in_storage, &keep_storage});
    }
    // The transfer runs on every frame at the FRAME extent: `result` and
    // `shown` are model-sized (sampled bilinearly when scaled), `keep` is the
    // frame as handed over - the full-resolution copy when scaled, the
    // untouched linear original on the linear path, the frame itself
    // otherwise - and `out` is the full-resolution answer.
    const char* const transfer_spv = impl_->store_native ? "runtime_transfer_store.spv" : "runtime_transfer.spv";
    {
        nrvk::Context::Image* shown = config.max_passes > 1 ? &impl_->shown_keep : &s.tex_in;
        nrvk::Context::Image* keep = impl_->scaled ? &impl_->keep_full
                                     : impl_->linear ? &impl_->keep : shown;
        // Unscaled, the two enlargement bindings are never read; the answer stands in.
        nrvk::Context::Image* up_r = impl_->scaled ? &impl_->up_result : answer;
        nrvk::Context::Image* up_s = impl_->scaled ? &impl_->up_shown : answer;
        impl_->transfer_pass.create(s.ctx, (adapters / transfer_spv).string(), {}, sizeof(TransferPush),
                               {&s.surf0, shown, keep, answer, up_r, up_s});
        impl_->transfer_binds = {s.surf0, *shown, *keep, *answer, *up_r, *up_s};
        if (impl_->scaled)
            impl_->upscale_pass.create(s.ctx, (adapters / "runtime_upscale.spv").string(), {}, 20,
                                       {&s.surf0, shown, &impl_->up_result, &impl_->up_shown});
    }
    if (config.preprocess && !mask_config.width) {
        impl_->prep = true;
        impl_->prep_unknee = config.preprocess_unknee;
        impl_->prep_keep = s.ctx.image(mw, mh, VK_FORMAT_R32G32B32A32_SFLOAT, true);
        s.ctx.transition(impl_->prep_keep, VK_IMAGE_LAYOUT_GENERAL);
        // Host visible: 16 bytes the log reads (Runtime::preprocess_meter).
        impl_->prep_state = s.ctx.buffer(16, true);
        // Storage aliases, as the linear path's: same image and view, no sampler.
        auto storage = [](const nrvk::Context::Image& im) {
            nrvk::Context::Image alias = im; alias.sampler = VK_NULL_HANDLE; alias.layout = VK_IMAGE_LAYOUT_GENERAL;
            return alias;
        };
        nrvk::Context::Image tex_in_st = storage(s.tex_in), keep_st = storage(impl_->prep_keep),
                             surf0_st = storage(s.surf0);
        const auto spv = (adapters / "runtime_prep.spv").string();
        impl_->prep_k.create(s.ctx, spv, {impl_->prep_state.handle}, sizeof(PrepPush), {&tex_in_st, &keep_st, &surf0_st});
        if (config.max_passes > 1) {
            nrvk::Context::Image shown_st = storage(impl_->shown_keep);
            impl_->prep_back_k.create(s.ctx, spv, {impl_->prep_state.handle}, sizeof(PrepPush), {&shown_st, &keep_st, &surf0_st});
        } else if (!config.native_compose) {
            // One pass: tex_in keeps what the network saw and the transfer pass
            // reads the frame from prep_keep instead, as `shown` and, when
            // tex_in was also the frame as handed over, as `keep`.
            nrvk::Context::Image* keep = impl_->scaled ? &impl_->keep_full
                                         : impl_->linear ? &impl_->keep : &impl_->prep_keep;
            nrvk::Context::Image* answer = impl_->store_native ? &keep_full_storage
                                           : impl_->scaled ? &impl_->full_out : &s.surf0;
            nrvk::Context::Image* up_r = impl_->scaled ? &impl_->up_result : answer;
            nrvk::Context::Image* up_s = impl_->scaled ? &impl_->up_shown : answer;
            impl_->transfer_prep.create(s.ctx, (adapters / transfer_spv).string(), {},
                                        sizeof(TransferPush), {&s.surf0, &impl_->prep_keep, keep, answer, up_r, up_s});
            impl_->transfer_prep_binds = {s.surf0, impl_->prep_keep, *keep, *answer, *up_r, *up_s};
            if (impl_->scaled)
                impl_->upscale_prep.create(s.ctx, (adapters / "runtime_upscale.spv").string(), {}, 20,
                                           {&s.surf0, &impl_->prep_keep, &impl_->up_result, &impl_->up_shown});
        }
    }
    auto require_variant_profile = [&](const std::filesystem::path& directory) {
        std::ifstream manifest(directory / "shader-constants.txt");
        uint32_t profile = 0;
        if (manifest) {
            const auto values = nr::detail::read_shader_manifest(manifest);
            if (values.count("math_profile")) profile = values.at("math_profile");
        } else if (s.math_profile != 0) {
            throw std::runtime_error("fast graph requires a versioned image-variant manifest");
        }
        if (profile != s.math_profile)
            throw std::runtime_error("image variant and graph math profiles differ");
    };
    if (bool(mask_config.width) != bool(mask_config.height))
        throw std::invalid_argument("both control mask dimensions are required");
    if (mask_config.width) {
        if (config.accumulation == "fp32") require_variant_profile(adapters);
        impl_->mask_image=s.ctx.image(mask_config.width,mask_config.height,VK_FORMAT_R32G32B32A32_SFLOAT,true);
        impl_->mask_rect=s.ctx.buffer(24);
        auto& k=impl_->mask_pre;
        // "fswinimagepreds32nh" is the same kernel without the (never reached) upper
        // exponent clamp; the mask variant keeps the clamp, which is the same function.
        const std::string pre_name = s.kern.count("fswinimagepreds32nh") ? "fswinimagepreds32nh" : "fswinimagepreds32";
        if (s.kern.count(pre_name)) {
            impl_->original_pre=&s.kern.at(pre_name);
            k.create(s.ctx,(adapters/"mask_pre_fp32.spv").string(),
                {s.act.handle,s.act.handle,s.wgt.handle,s.wgt.handle,s.wgt.handle,s.act.handle,impl_->mask_rect.handle},
                sizeof(PushFSwin)+sizeof(PushPreImage),{&s.tex_in,&impl_->mask_image});
        } else if (s.kern.count("imgin16fast")) {
            impl_->original_pre=&s.kern.at("imgin16fast");
            k.create(s.ctx,(adapters/"mask_pre_round.spv").string(),
                {s.act.handle,s.wgt.handle,impl_->mask_rect.handle},sizeof(PushImgIn),{&s.tex_in,&impl_->mask_image});
        } else throw std::runtime_error("control mask pre-input variant unavailable");
        impl_->mask_resolve.create(s.ctx,(adapters/"runtime_control_mask.spv").string(),
            {impl_->mask_rect.handle},12,{&s.surf0,&s.tex_in,&impl_->mask_image});
    }
    {
        const std::string pn = s.kern.count("fswinimagepreds32nh") ? "fswinimagepreds32nh" : "fswinimagepreds32";
        if (s.kern.count(pn) && s.kern.count("fswinimagepost32")) {
            impl_->plain_pre = &s.kern.at(pn); impl_->plain_post = &s.kern.at("fswinimagepost32");
        }
    }
    if (temporal_config.enable) {
        auto& t = impl_->temporal;
        // The temporal pre/post blocks are network kernels: the ACO bundle's own SPIR-V when it is in use.
        auto shaders = std::filesystem::canonical(temporal_config.shaders.empty()
            ? (!aco.empty() ? std::filesystem::path(aco) / "shaders/temporal"
               : installed ? data / "shaders/temporal" : root / "build/windows/rdna4/network/temporal")
            : std::filesystem::path(temporal_config.shaders));
        require_variant_profile(shaders);
        if (!inside(shaders))
            throw std::invalid_argument("temporal shaders must be inside NR root");
        const bool pre_nh = s.kern.count("fswinimagepreds32nh") != 0;
        const std::string pre_name = pre_nh ? "fswinimagepreds32nh" : "fswinimagepreds32";
        if (!s.kern.count(pre_name) || !s.kern.count("fswinimagepost32"))
            throw std::runtime_error("the temporal path requires the fused FP32 image kernels");
        t.original_pre = &s.kern.at(pre_name);
        t.original_post = &s.kern.at("fswinimagepost32");
        t.lw[0] = (mw + kTemporalBase - 1) / kTemporalBase;
        t.lh[0] = (mh + kTemporalBase - 1) / kTemporalBase;
        for (unsigned k = 1; k < kTemporalLevels; ++k) {
            t.lw[k] = (t.lw[k - 1] + 1) / 2;
            t.lh[k] = (t.lh[k - 1] + 1) / 2;
        }
        for (unsigned p = 0; p < 2; ++p)
            for (unsigned k = 0; k < kTemporalLevels; ++k)
                t.luma[p][k] = s.ctx.image(t.lw[k], t.lh[k], VK_FORMAT_R32_SFLOAT, false);
        for (unsigned k = 0; k < kTemporalLevels; ++k)
            t.flow[k] = s.ctx.image(t.lw[k], t.lh[k], VK_FORMAT_R32G32_SFLOAT, false);
        // The five-tap reconstruction's merged centre tap is only valid under
        // linear filtering, and the motion field wants the same. nrvk creates
        // NEAREST + MIRRORED_REPEAT because that is the shipping colour
        // kernel's addressing; replace the sampler here rather than change a
        // production header.
        auto linear_sampler = [&] {
            VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            si.magFilter = si.minFilter = VK_FILTER_LINEAR;
            si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            VkSampler sampler{};
            NRVK_CHECK(vkCreateSampler(s.ctx.device, &si, nullptr, &sampler));
            return sampler;
        };
        t.h16 = std::getenv("NR_HIST16") && std::atoi(std::getenv("NR_HIST16")) && config.max_passes == 1 &&
                std::filesystem::exists(shaders / "temporal_post_fp32h16.spv");
        const VkFormat hist_format = t.h16 ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
        t.history = s.ctx.image(mw, mh, hist_format, true);
        vkDestroySampler(s.ctx.device, t.history.sampler, nullptr);
        t.history.sampler = linear_sampler();
        // Kept in GENERAL for its whole life: it is a transfer destination once
        // a frame and a sampled source once a frame, and GENERAL is valid for
        // both. The alternative is two layout transitions per frame to save
        // nothing measurable on an image this size.
        s.ctx.transition(t.history, VK_IMAGE_LAYOUT_GENERAL);
        // The finest flow level is written as a storage image and read as a
        // sampled one. Same VkImage and view, two descriptors; the alias does
        // not own the image, so it must never be handed to ctx.destroy.
        t.motion_sampler = linear_sampler();
        t.flow0_sampled = t.flow[0];
        t.flow0_sampled.sampler = t.motion_sampler;
        // Single channel and float: a depth buffer's value is all this reads, and
        // the blit converts whatever the game hands over.
        t.depth = s.ctx.image(mw, mh, VK_FORMAT_R32_SFLOAT, true);
        vkDestroySampler(s.ctx.device, t.depth.sampler, nullptr);
        t.depth.sampler = linear_sampler();
        s.ctx.transition(t.depth, VK_IMAGE_LAYOUT_GENERAL);
        {
            // runtime_depth.comp's layout: a sampled source, then t.depth as storage.
            nrvk::Context::Image depth_store = t.depth;
            depth_store.sampler = VK_NULL_HANDLE;
            impl_->depth_copy.create(s.ctx, (adapters / "runtime_depth.spv").string(), {}, 24,
                                     {&t.depth, &depth_store});
        }
        t.params = s.ctx.buffer(48);

        for (unsigned p = 0; p < 2; ++p) {
            for (unsigned k = 0; k < kTemporalLevels; ++k) {
                // Binding 1 is the finer level this one halves; in mode 0 it is
                // not read, and the coarsest level is bound there as a valid
                // descriptor that nothing touches.
                auto* src = k ? &t.luma[p][k - 1] : &t.luma[p][kTemporalLevels - 1];
                t.luma_kernel[p][k].create(s.ctx, (shaders / "motion_luma.spv").string(), {},
                                           sizeof(Temporal::LumaPush),
                                           {&s.tex_in, src, &t.luma[p][k]});
                // Likewise, the coarsest flow pass runs with first=1 and never
                // reads flow_in, so it may point at its own output.
                const bool first = k == kTemporalLevels - 1;
                t.flow_kernel[p][k].create(s.ctx, (shaders / "motion_estimate.spv").string(), {},
                                           sizeof(Temporal::FlowPush),
                                           {&t.luma[p][k], &t.luma[1 - p][k],
                                            first ? &t.flow[k] : &t.flow[k + 1], &t.flow[k]});
            }
        }
        t.pingpong = config.max_passes == 1;
        if (t.pingpong) {
            t.history_b = s.ctx.image(mw, mh, t.history.format, true);
            vkDestroySampler(s.ctx.device, t.history_b.sampler, nullptr);
            t.history_b.sampler = linear_sampler();
            s.ctx.transition(t.history_b, VK_IMAGE_LAYOUT_GENERAL);
        }
        // The storage view of a history image the post variant writes: the same
        // image and view, no sampler (an alias; it owns nothing).
        auto storage_of = [](const nrvk::Context::Image& im) {
            nrvk::Context::Image a = im; a.sampler = VK_NULL_HANDLE; return a;
        };
        {
            const bool tp_nh0 = pre_nh && std::filesystem::exists(shaders / "temporal_pre_fp32nh.spv");
            t.hp = std::getenv("NR_HPASS") && std::atoi(std::getenv("NR_HPASS")) &&
                   std::filesystem::exists(shaders / (tp_nh0 ? "temporal_pre_fp32nhhp.spv" : "temporal_pre_fp32hp.spv")) &&
                   std::filesystem::exists(shaders / "temporal_post_fp32hp.spv");
            if (t.hp) {
                t.hpass = s.ctx.image(mw, mh, VK_FORMAT_R32G32B32A32_SFLOAT, false);
                s.ctx.transition(t.hpass, VK_IMAGE_LAYOUT_GENERAL);
            }
        }
        nrvk::Context::Image (&hist_store)[2] = t.hist_store;
        hist_store[0] = storage_of(t.hist(0)); hist_store[1] = storage_of(t.hist(1));
        t.pre_buffers = {s.act.handle, s.act.handle, s.wgt.handle, s.wgt.handle, s.wgt.handle,
                         s.act.handle, t.params.handle};
        t.post_buffers = {s.act.handle, s.act.handle, s.wgt.handle, s.wgt.handle, s.wgt.handle,
                          t.params.handle};
        for (unsigned c = 0; c < (t.pingpong ? 2u : 1u); ++c) {
            nrvk::Kernel& pre = t.pingpong ? t.pre_pp[c] : t.pre;
            nrvk::Kernel& post = t.pingpong ? t.post_pp[c] : t.post;
            // the no-upper-clamp temporal pre when the graph's pre qualified and the
            // package has it (the clamped one is the same function there).
            const bool tp_nh = pre_nh && std::filesystem::exists(shaders / "temporal_pre_fp32nh.spv");
            // (research NR_HPASS=1): the pre block hands the post its history.
            const std::string pre_spv = std::string(tp_nh ? "temporal_pre_fp32nh" : "temporal_pre_fp32") + (t.hp ? "hp.spv" : ".spv");
            pre.create(s.ctx, (shaders / pre_spv).string(),
                       {s.act.handle, s.act.handle, s.wgt.handle, s.wgt.handle, s.wgt.handle,
                        s.act.handle, t.params.handle},
                       sizeof(PushFSwin) + sizeof(PushPreImage),
                       t.hp ? std::vector<nrvk::Context::Image*>{&s.tex_in, &t.flow0_sampled, &t.hist(c), &t.depth, &t.hpass}
                            : std::vector<nrvk::Context::Image*>{&s.tex_in, &t.flow0_sampled, &t.hist(c), &t.depth});
            // The post variant reads the same parameter buffer, the motion field and
            // the history the pre variant does: its blend is gated by the same term.
            post.create(s.ctx, (shaders / (t.hp ? "temporal_post_fp32hp.spv" : t.h16 ? "temporal_post_fp32h16.spv" : "temporal_post_fp32.spv")).string(),
                        {s.act.handle, s.act.handle, s.wgt.handle, s.wgt.handle, s.wgt.handle,
                         t.params.handle},
                        sizeof(PushFSwin) + sizeof(PushUps) + sizeof(PushImageTail),
                        t.hp ? std::vector<nrvk::Context::Image*>{&s.surf0, t.pingpong ? &hist_store[c ^ 1u] : &s.surf1, &s.tex_in,
                                                                         &t.flow0_sampled, &t.hist(c), &t.hpass}
                             : std::vector<nrvk::Context::Image*>{&s.surf0, t.pingpong ? &hist_store[c ^ 1u] : &s.surf1, &s.tex_in,
                                                                         &t.flow0_sampled, &t.hist(c)});
        }
        t.history_strength = temporal_config.history_strength;
        t.enabled = true;
        // One history per pass. Pass k reads and writes store[k] through the
        // bound history image; the copies are model-sized and cost nothing
        // next to a pass.
        if (config.max_passes > 1) {
            impl_->history_store.resize(config.max_passes);
            for (auto& h : impl_->history_store) {
                h = s.ctx.image(mw, mh, VK_FORMAT_R32G32B32A32_SFLOAT, false);
                s.ctx.transition(h, VK_IMAGE_LAYOUT_GENERAL);
            }
        }
    }
    if(transfer != Transfer::Identical) {
        for(VkFormat format : {transfer == Transfer::Encoded ? unorm : config.colour_format,
                               VK_FORMAT_R32G32B32A32_SFLOAT}) {
            VkFormatProperties properties{};vkGetPhysicalDeviceFormatProperties(host.physical,format,&properties);
            const VkFormatFeatureFlags need=VK_FORMAT_FEATURE_BLIT_SRC_BIT|VK_FORMAT_FEATURE_BLIT_DST_BIT;
            if((properties.optimalTilingFeatures&need)!=need)
                throw std::runtime_error("NR format conversion requires source/destination blit support");
        }
    }
    // Only the 8-bit path needs the private transfer image: it exists to keep an
    // *_SRGB view from applying its transfer function to the bits, not to
    // convert anything. The wide Wayland formats blit straight through.
    if(transfer == Transfer::Encoded) {
        auto& im=impl_->encoded;im.w=config.width;im.h=config.height;im.format=unorm;
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType=VK_IMAGE_TYPE_2D;info.format=unorm;info.extent={im.w,im.h,1};
        info.mipLevels=info.arrayLayers=1;info.samples=VK_SAMPLE_COUNT_1_BIT;
        info.tiling=VK_IMAGE_TILING_OPTIMAL;
        info.usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        NRVK_CHECK(vkCreateImage(host.device,&info,nullptr,&im.handle));
        VkMemoryRequirements req{};vkGetImageMemoryRequirements(host.device,im.handle,&req);
        uint32_t type=s.ctx.mem.memoryTypeCount;
        for(uint32_t i=0;i<s.ctx.mem.memoryTypeCount;++i)
            if((req.memoryTypeBits&(1u<<i))&&(s.ctx.mem.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {type=i;break;}
        if(type==s.ctx.mem.memoryTypeCount)throw std::runtime_error("no device-local transfer image memory");
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize=req.size;allocation.memoryTypeIndex=type;
        NRVK_CHECK(vkAllocateMemory(host.device,&allocation,nullptr,&im.memory));
        NRVK_CHECK(vkBindImageMemory(host.device,im.handle,im.memory,0));
        s.ctx.transition(im,VK_IMAGE_LAYOUT_GENERAL);
    }
    impl_->post_alpha = NR_POST_ALPHA && impl_->native_compose && !impl_->scaled &&
                        impl_->max_passes == 1 && s.kern.count("fswinimagepost32");
    timer.mark("adapters");
    // Built: from here on the caller serialises this runtime's submits itself.
    s.ctx.queue_lock = nullptr; s.ctx.queue_unlock = nullptr;
    nr::logf("[nr] runtime %ux%u built in %.2fs: %s", config.width, config.height, timer.total(),
             timer.text().c_str());
}

Runtime::~Runtime() = default;

uint32_t Runtime::model_width() const { return impl_->mw; }
uint32_t Runtime::model_height() const { return impl_->mh; }

void Runtime::set_white_point(float v) {
    if (impl_->linear) impl_->white_point = v > 1e-4f ? v : 1e-4f;
}

void Runtime::set_history_strength(float v) {
    if (impl_->temporal.enabled) impl_->temporal.history_strength = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
}

void Runtime::release_feature(uint64_t feature) {
    if (!feature) return;
    auto it = impl_->features.find(feature);
    if (it == impl_->features.end()) return;
    if (it->second.history.handle)
        impl_->retiring.emplace_back(it->second.history,
                                     impl_->recordings + Impl::kRetireAfter);
    impl_->features.erase(it);
    // The bound history keeps whatever this feature left in it; the next
    // feature to record copies its own state over the top, and a feature that
    // is gone has no state to write back.
    if (impl_->bound_feature == feature) impl_->bound_feature = 0;
}

void Runtime::drop_history() {
    // A recording that advanced the temporal state (latch, parity, seed, the bound feature's copies)
    // never ran. Every feature starts again as on its first frame: the gate stays closed until a
    // recording that does run has written its history, and a fresh image is cleared before use.
    impl_->temporal.latch = false;
    for (auto& f : impl_->features) { f.second.latch = false; f.second.cleared = false; }
    impl_->bound_feature = 0;
}

uint32_t Runtime::live_features() const { return uint32_t(impl_->features.size()); }

Runtime::TemporalHistoryView Runtime::temporal_history() const {
    TemporalHistoryView v{};
    const auto& t = impl_->temporal;
    if (!t.enabled) return v;
    const auto& h = const_cast<Temporal&>(t).hist(t.hcur);
    v.image = h.handle; v.view = h.view;
    v.width = h.w; v.height = h.h;
    v.format = h.format; v.layout = h.layout;
    return v;
}

RecordResult Runtime::record(VkCommandBuffer cmd, const ColourFrame& frame, const Controls& c) {
    return record(cmd,frame,c,nullptr).frame;
}

ControlMaskResult Runtime::record(VkCommandBuffer cmd, const ColourFrame& frame, const Controls& c,
                             const ControlMaskFrame* mask) {
    return record_all(cmd,frame,c,mask,nullptr,nullptr,nullptr);
}

TemporalResult Runtime::record_temporal(VkCommandBuffer cmd, const ColourFrame& frame,
                                        const Controls& c, const TemporalFrame& temporal) {
    if (!impl_->temporal.enabled)
        throw std::invalid_argument("this NR runtime was not built with the temporal path");
    TemporalResult result{};
    bool consumed = false;
    result.frame = record_all(cmd,frame,c,nullptr,&temporal,&consumed,nullptr).frame;
    result.history_consumed = consumed;
    // The only provider that exists today. When the upscaler interception lands
    // it fills the same texture and this becomes "engine" - an estimate must
    // never be reported as engine data.
    result.motion_provider = "estimated";
    return result;
}

bool Runtime::takes_target(const Controls& c) const {
    return c.enabled && c.apply_model && impl_->temporal.enabled &&
           std::min<uint32_t>(std::max(c.passes, 1), impl_->max_passes) == 1;
}

EngineResult Runtime::record_engine(VkCommandBuffer cmd, const EngineFrame& frame,
                                    const Controls& c) {
    auto& t = impl_->temporal;
    if (!t.enabled)
        throw std::invalid_argument("this NR runtime was not built with the temporal path");
    if (!frame.motion.image)
        throw std::invalid_argument("the engine provider requires a motion-vector image");
    EngineResult result{};
    bool consumed = false;
    // The engine's motion texture is blitted into the same image the fallback
    // estimator writes, so exactly one pipeline and one descriptor set serve
    // both providers. It costs one blit and it means the consumer cannot drift
    // between the two paths - which was the point of splitting them.
    TemporalFrame temporal{frame.reset, frame.feature};
    result.frame = record_all(cmd, frame.colour, c, nullptr, &temporal, &consumed, &frame).frame;
    result.history_consumed = consumed;
    result.motion_provider = "engine";
    return result;
}

ControlMaskResult Runtime::record_all(VkCommandBuffer cmd, const ColourFrame& frame, const Controls& c,
                             const ControlMaskFrame* mask, const TemporalFrame* temporal,
                             bool* history_consumed, const EngineFrame* engine) {
    validate(c);
    if (!cmd || !frame.image || frame.format != impl_->colour_format ||
        frame.width != impl_->width || frame.height != impl_->height ||
        !frame.before_stage || !frame.after_stage || frame.before == VK_IMAGE_LAYOUT_UNDEFINED ||
        frame.before == VK_IMAGE_LAYOUT_PREINITIALIZED || frame.after == VK_IMAGE_LAYOUT_UNDEFINED ||
        frame.after == VK_IMAGE_LAYOUT_PREINITIALIZED)
        throw std::invalid_argument("invalid NR colour frame");
    uint32_t mw=0,mh=0;
    if (mask) {
        const auto& t=mask->texture;
        mw=mask->width?mask->width:t.width;mh=mask->height?mask->height:t.height;
        if (!impl_->original_pre || !t.image || t.image==frame.image ||
            t.format!=VK_FORMAT_R32G32B32A32_SFLOAT ||
            t.width!=impl_->mask_image.w || t.height!=impl_->mask_image.h ||
            mask->x>=t.width || mask->y>=t.height || mw>t.width-mask->x || mh>t.height-mask->y ||
            !t.before_stage || !t.after_stage || t.before==VK_IMAGE_LAYOUT_UNDEFINED ||
            t.before==VK_IMAGE_LAYOUT_PREINITIALIZED || t.after==VK_IMAGE_LAYOUT_UNDEFINED ||
            t.after==VK_IMAGE_LAYOUT_PREINITIALIZED)
            throw std::invalid_argument("invalid/unconfigured external control mask");
    }
    // the one fallible allocation of a recording - a new feature's
    // history image - before anything is recorded, so a failure leaves the caller's
    // images in the layouts it gave and its fallback (seed copy, in place) sound.
    if (c.enabled && temporal && impl_->temporal.enabled && temporal->feature && !impl_->features.count(temporal->feature)) {
        const auto& th = impl_->temporal.history;
        Impl::FeatureState made{};
        made.history = impl_->session.ctx.image(th.w, th.h, th.format, false, false);
        impl_->features.emplace(temporal->feature, made);
    }
    ControlMaskResult result{};
    auto& info=result.frame;
    info.effective_skin = c.automatic_mask ? (c.skin_structure < 0 ? c.local_structure : c.skin_structure) : -1;
    info.effective_background = c.automatic_mask ? c.local_structure : -1;
    if (!c.enabled) {
        if (mask && mask->texture.before!=mask->texture.after) {
            const auto& t=mask->texture;
            barrier(cmd,t.image,t.before,t.after,t.before_stage,t.before_access,t.after_stage,t.after_access);
        }
        if (frame.before != frame.after)
            barrier(cmd, frame.image, frame.before, frame.after, frame.before_stage,
                    frame.before_access, frame.after_stage, frame.after_access);
        return result;
    }
    // Timed from here: a disabled pass records nothing and is not a cost.
    const uint32_t timing_slot = impl_->timing_slot;
    if (impl_->timing_ok) {
        vkCmdResetQueryPool(cmd, impl_->timing, timing_slot * impl_->timing_stride, impl_->timing_stride);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, impl_->timing, timing_slot * impl_->timing_stride);
    }
    auto& s = impl_->session;
    Controls effective=c;
    if (mask) {
        effective.automatic_mask=false;effective.intensity=1;
        info.effective_skin=info.effective_background=-1;
        result.consumed=true;
        const auto& t=mask->texture;
        auto& im=impl_->mask_image;
        const float rect[]={float(mask->x),float(mask->y),float(mw),float(mh),1.f/t.width,1.f/t.height};
        VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        b.srcAccessMask=VK_ACCESS_SHADER_READ_BIT;b.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
        b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        b.buffer=impl_->mask_rect.handle;b.size=VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,1,&b,0,nullptr);
        vkCmdUpdateBuffer(cmd,impl_->mask_rect.handle,0,sizeof rect,rect);
        b.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&b,0,nullptr);
        barrier(cmd,t.image,t.before,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,t.before_stage,t.before_access,
                VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_READ_BIT);
        barrier(cmd,im.handle,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_READ_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);
        VkImageCopy copy{};copy.srcSubresource=copy.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
        copy.extent={t.width,t.height,1};
        vkCmdCopyImage(cmd,t.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,im.handle,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);
        barrier(cmd,im.handle,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_READ_BIT);
        barrier(cmd,t.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,t.after,VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,t.after_stage,t.after_access);
    }
    impl_->controls(effective);
    impl_->check_read();
    if (engine && temporal)
        impl_->incheck.begin(s.ctx, cmd, frame, *engine, temporal->feature, impl_->recordings,
                             impl_->temporal.lw[0], impl_->temporal.lh[0], impl_->mw, impl_->mh);
    // The network extent (Model Resolution); the frame is frame.width x frame.height.
    const uint32_t nw = impl_->mw, nh = impl_->mh;
    const uint32_t passes = std::min<uint32_t>(std::max(effective.passes, 1), impl_->max_passes);
    // a separate answer image (EngineFrame::target).
    const ColourFrame* tgt = engine && engine->target.image && engine->target.image != frame.image ? &engine->target : nullptr;
    if (tgt && (!c.apply_model || passes != 1 || mask || tgt->format != frame.format ||
                tgt->width != frame.width || tgt->height != frame.height || !tgt->before_stage || !tgt->after_stage ||
                tgt->before == VK_IMAGE_LAYOUT_PREINITIALIZED || tgt->after == VK_IMAGE_LAYOUT_UNDEFINED ||
                tgt->after == VK_IMAGE_LAYOUT_PREINITIALIZED))
        throw std::invalid_argument("invalid NR target frame");
    const VkImage wb = tgt ? tgt->image : frame.image;
    // The caller's colour (and depth) sampled in place, when this frame allows
    // it: see Impl::DirectSrc. Null means the copies below, as always.
    const Impl::DirectSrc* ds = nullptr;
    // post_alpha: native compose, one pass, scale 1, and no alpha pass - after
    // the network only the write-back touches the frame.
    const bool colour_direct =
        NR_DIRECT_SAMPLE && !impl_->direct_src_failed && impl_->post_alpha && !impl_->prep && engine &&
        temporal && !mask && passes == 1 &&
        impl_->temporal.pingpong && impl_->transfer != Transfer::Encoded &&
        (frame.usage & VK_IMAGE_USAGE_SAMPLED_BIT) && frame.width == nw && frame.height == nh;
    // the depth in place is a separate question. A depth that still needs its blit or
    // conversion (render-resolution guides after SR, a subrect, depth/stencil) goes through
    // t.depth as before, and the colour is sampled in place (and the answer stored) anyway.
    const bool depth_in_place = colour_direct && engine->depth.image &&
        engine->depth.format == VK_FORMAT_R32_SFLOAT && (engine->depth.usage & VK_IMAGE_USAGE_SAMPLED_BIT) &&
        engine->depth_x == 0 && engine->depth_y == 0 &&
        engine->depth.width == impl_->temporal.depth.w && engine->depth.height == impl_->temporal.depth.h;
    // The motion vectors in place: any engine frame with one pass and the
    // ping-pong sets, a sampled image and a linearly filterable format.
    const bool motion_direct =
        NR_DIRECT_MOTION && !impl_->motion_src_failed && engine && temporal && !mask && passes == 1 &&
        impl_->temporal.pingpong && engine->motion.image &&
        (engine->motion.usage & VK_IMAGE_USAGE_SAMPLED_BIT) && impl_->filters_linearly(engine->motion.format);
    // (research NR_DIRECT_STORE=1): the post block stores into the frame itself.
    // With a separate target (storage-capable) the post block stores into it and the
    // write-back copy goes (4K RGBA16F -0.15 ms). In place (the post samples only its own
    // pixel, NEAREST, before it stores it) for 64-bit frames: 4K RGBA16F -0.05 ms; bgra8
    // ties (its copy stays in the cache). NR_DIRECT_STORE=0: never; =1: in place for any format.
    static const int direct_store_env = std::getenv("NR_DIRECT_STORE") ? std::atoi(std::getenv("NR_DIRECT_STORE")) : -1;
    const bool wide = frame.format == VK_FORMAT_R16G16B16A16_SFLOAT || frame.format == VK_FORMAT_R16G16B16A16_UNORM;
    const bool store_direct = colour_direct && impl_->direct_out && c.apply_model && frame.format == s.surf0.format &&
        !impl_->scaled && direct_store_env != 0 &&
        (tgt ? (tgt->usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0
             : (direct_store_env == 1 || wide) && (frame.usage & VK_IMAGE_USAGE_STORAGE_BIT));
    // the plain path (no temporal, no engine frame) samples the caller's colour in place
    // too - the input copy goes. Research switch NR_PLAIN_DIRECT=0 turns it off.
    static const bool plain_env = !(std::getenv("NR_PLAIN_DIRECT") && !std::atoi(std::getenv("NR_PLAIN_DIRECT")));
    const bool plain_direct = plain_env && !temporal && !engine && !mask && passes == 1 && impl_->plain_pre &&
        impl_->plain_post && !impl_->direct_src_failed && impl_->post_alpha && !impl_->prep && !impl_->linear &&
        impl_->transfer != Transfer::Encoded && (frame.usage & VK_IMAGE_USAGE_SAMPLED_BIT) &&
        frame.width == nw && frame.height == nh;
    const bool plain_store = plain_direct && impl_->direct_out && c.apply_model && frame.format == s.surf0.format &&
        !impl_->scaled && direct_store_env != 0 && (direct_store_env == 1 || wide) &&
        (frame.usage & VK_IMAGE_USAGE_STORAGE_BIT);
    const Impl::DirectSrc* sets = nullptr;
    if (plain_direct)
        sets = impl_->direct_source(frame.image, frame.format, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                    VK_FORMAT_UNDEFINED, 0, plain_store, VK_NULL_HANDLE, true);
    else if (colour_direct || motion_direct)
        sets = impl_->direct_source(colour_direct ? frame.image : VK_NULL_HANDLE, frame.format,
                                    depth_in_place ? engine->depth.image : VK_NULL_HANDLE,
                                    motion_direct ? engine->motion.image : VK_NULL_HANDLE,
                                    engine->motion.format, impl_->temporal.hcur, store_direct,
                                    tgt ? tgt->image : VK_NULL_HANDLE);
    if (sets && sets->cview) ds = sets;
    // Model resolution with the frame read and written in place (Impl::FrameRw). NR_FRAME_RW=0: never.
    static const bool frame_rw_env = !(std::getenv("NR_FRAME_RW") && !std::atoi(std::getenv("NR_FRAME_RW")));
    const Impl::FrameRw* fr = frame_rw_env && impl_->store_native && impl_->scaled && !tgt && !mask &&
        (frame.usage & VK_IMAGE_USAGE_SAMPLED_BIT) && (frame.usage & VK_IMAGE_USAGE_STORAGE_BIT) &&
        frame.format == impl_->keep_full.format && frame.width == impl_->keep_full.w && frame.height == impl_->keep_full.h
        ? impl_->frame_rw_source(frame.image, frame.format) : nullptr;
    const bool stored = (ds && ds->store) || fr;
    const bool depth_direct = ds && ds->dview != VK_NULL_HANDLE;   // the caller's depth bound in place
    const bool mv_direct = sets && sets->mview;
    barrier(cmd, frame.image, frame.before, ds || fr ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            frame.before_stage, frame.before_access,
            ds || fr ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
            stored && !tgt ? VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                   : ds ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_TRANSFER_READ_BIT);
    if (stored && tgt)
        barrier(cmd, tgt->image, tgt->before, VK_IMAGE_LAYOUT_GENERAL, tgt->before_stage, tgt->before_access,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    // Where tex_in stands once the frame is in it: a copy or blit leaves it a transfer
    // destination, runtime_downscale.comp (keep_native) a storage image.
    const bool down_native = impl_->scaled && impl_->keep_native;
    const VkImageLayout tin_layout = down_native ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    const VkPipelineStageFlags tin_stage = down_native ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT;
    const VkAccessFlags tin_access = down_native ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_TRANSFER_WRITE_BIT;
    if (!ds)
    barrier(cmd, s.tex_in.handle, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, tin_layout,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, tin_stage, tin_access);
    VkImageCopy region{};
    region.srcSubresource = region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent = {frame.width, frame.height, 1};
    VkImageBlit blit{};
    blit.srcSubresource=blit.dstSubresource=region.srcSubresource;
    blit.srcOffsets[1]=blit.dstOffsets[1]={int32_t(frame.width),int32_t(frame.height),1};
    auto& encoded=impl_->encoded;
    // Model Resolution: the frame lands in keep_full at its own size first,
    // then is filtered down into tex_in. Whatever the frame's format, the
    // downscale is one RGBA32F -> RGBA32F blit.
    VkImage colour_dst = s.tex_in.handle;
    if (impl_->scaled && !fr) {
        colour_dst = impl_->keep_full.handle;
        barrier(cmd, impl_->keep_full.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    }
    if (fr) {
        // Nothing to copy: the downscale and the transfer pass read the frame itself.
    } else if (down_native) {
        // The frame's bits as they are (an *_SRGB frame into its UNORM twin).
        vkCmdCopyImage(cmd, frame.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       colour_dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    } else if (ds) {
        // Nothing to copy: the pre and post blocks sample the frame itself.
    } else if (impl_->direct_in) {
        // The bits as they are (an *_SRGB frame into its UNORM twin: the same
        // bits the private transfer image held); the kernels' sampling does
        // the conversion the blit used to.
        vkCmdCopyImage(cmd, frame.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       colour_dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    } else if(encoded.handle) {
        barrier(cmd,encoded.handle,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);
        vkCmdCopyImage(cmd,frame.image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       encoded.handle,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&region);
        barrier(cmd,encoded.handle,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_READ_BIT);
        vkCmdBlitImage(cmd,encoded.handle,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       colour_dst,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&blit,VK_FILTER_NEAREST);
        barrier(cmd,encoded.handle,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT);
    } else if (impl_->transfer == Transfer::DirectBlit) {
        // FP16 / 16-bit UNORM / 10-bit packed: no sRGB transfer function to route
        // around, so the blit's own format conversion is the whole of it.
        vkCmdBlitImage(cmd, frame.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       colour_dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
    } else {
        vkCmdCopyImage(cmd, frame.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       colour_dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }
    if (down_native) {
        if (!fr)
            barrier(cmd, impl_->keep_full.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        const uint32_t model[] = {nw, nh};
        dispatch(cmd, impl_->downscale, (nw + 7) / 8, (nh + 7) / 8, 1, model, sizeof model, fr ? fr->down : VK_NULL_HANDLE);
    } else if (impl_->scaled) {
        barrier(cmd, impl_->keep_full.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkImageBlit down{};
        down.srcSubresource = down.dstSubresource = region.srcSubresource;
        down.srcOffsets[1] = {int32_t(frame.width), int32_t(frame.height), 1};
        down.dstOffsets[1] = {int32_t(nw), int32_t(nh), 1};
        vkCmdBlitImage(cmd, impl_->keep_full.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       s.tex_in.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &down, impl_->downscale_filter);
        barrier(cmd, impl_->keep_full.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    if (impl_->linear) {
        // Linear light: make the proxy in place and keep the original. The
        // network reads tex_in in SHADER_READ_ONLY like every other frame.
        barrier(cmd, s.tex_in.handle, tin_layout, VK_IMAGE_LAYOUT_GENERAL, tin_stage, tin_access,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        struct { uint32_t w, h; float white, knee; } push{nw, nh, impl_->white_point, 0.75f};
        dispatch(cmd, impl_->encode, (nw + 7) / 8, (nh + 7) / 8, 1, &push, sizeof push);
        barrier(cmd, s.tex_in.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    } else if (!ds) {
        barrier(cmd, s.tex_in.handle, tin_layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, tin_stage, tin_access,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    // Preprocess: meter the frame, keep it, and put what the network is to see
    // in tex_in. A change of settings resets the meter and, below, the history.
    const Preprocess& pp = c.preprocess;
    const bool prep_on = impl_->prep && pp.active() && !mask;
    const Preprocess& was = impl_->prep_last;
    if (impl_->prep && (prep_on != impl_->prep_was_on || (prep_on && pp != was))) ++impl_->prep_gen;
    PrepPush prep_push{};
    if (prep_on) {
        const uint32_t curve = uint32_t(std::clamp(pp.curve, 0, 6));
        const bool metered = pp.exposure == 1;
        prep_push = {nw, nh, 0u,
                     (metered ? kPrepAuto : 0u) | (impl_->prep_unknee ? kPrepUnknee : 0u) |
                         (impl_->prep_was_on ? 0u : kPrepReset),
                     curve, pp.exposure == 0 ? 0.0f : std::clamp(pp.bias_ev, -8.0f, 8.0f),
                     std::clamp(pp.contrast, 0.5f, 2.0f), std::clamp(pp.saturation, 0.05f, 2.0f),
                     kPrepAnchor[curve], 4.0f, 0.0f};
        const auto now = std::chrono::steady_clock::now();
        if (impl_->prep_was_on)
            prep_push.dt = std::chrono::duration<float>(now - impl_->prep_metered).count();
        impl_->prep_metered = now;
        barrier(cmd, s.tex_in.handle, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        if (metered) {
            dispatch(cmd, impl_->prep_k, 1, 1, 1, &prep_push, sizeof prep_push);
            compute_barrier(cmd);
        }
        prep_push.mode = 1;
        dispatch(cmd, impl_->prep_k, (nw + 15) / 16, (nh + 15) / 16, 1, &prep_push, sizeof prep_push);
        barrier(cmd, s.tex_in.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    impl_->prep_was_on = prep_on;
    impl_->prep_last = pp;
    // A GENERAL -> GENERAL copy between two of our own images, fenced for the
    // compute work on either side. Model-sized, used by the multi-pass loop.
    auto copy_general = [&](const nrvk::Context::Image& src, const nrvk::Context::Image& dst) {
        barrier(cmd, src.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        barrier(cmd, dst.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        VkImageCopy c{};
        c.srcSubresource = c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.extent = {src.w, src.h, 1};
        vkCmdCopyImage(cmd, src.handle, VK_IMAGE_LAYOUT_GENERAL, dst.handle, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        barrier(cmd, dst.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        barrier(cmd, src.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    };
    if (impl_->max_passes > 1) {
        // What the first pass sees, for the transfer pass and the alpha
        // restore; tex_in is overwritten by every pass after the first.
        barrier(cmd, s.tex_in.handle, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        barrier(cmd, impl_->shown_keep.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        VkImageCopy c{};
        c.srcSubresource = c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.extent = {nw, nh, 1};
        vkCmdCopyImage(cmd, s.tex_in.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       impl_->shown_keep.handle, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        barrier(cmd, impl_->shown_keep.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        barrier(cmd, s.tex_in.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    compute_barrier(cmd);
    auto& t = impl_->temporal;
    // This frame's feature, if the caller named one. Its state is host-side
    // bookkeeping plus one image; nothing is recorded yet.
    Impl::FeatureState* fs = nullptr;
    if (temporal && t.enabled && temporal->feature) {
        auto it = impl_->features.find(temporal->feature);
        if (it == impl_->features.end()) {
            Impl::FeatureState made{};
            // No settling transition: that is a submit and a fence wait, and
            // this is the render thread inside a recording. The barrier below
            // does it in the command buffer instead.
            made.history = s.ctx.image(t.history.w, t.history.h, t.history.format, false, false);
            it = impl_->features.emplace(temporal->feature, made).first;
        }
        fs = &it->second;
    }
    bool gate = false;
    uint32_t pre_seed = 0;
    if (temporal) {
        // The original's gate, all three terms, resolved here and nowhere else:
        // the first-frame latch, DLSSNR.Reset, and a motion source existing at
        // all.
        uint32_t& gen = fs ? fs->prep_gen : impl_->prep_gen_single;
        gate = (fs ? fs->latch : t.latch) && !temporal->reset && gen == impl_->prep_gen;
        gen = impl_->prep_gen;
        // The pre block's noise seed, the DLL's rule: a counter in the pre node
        // (+0xc8) handed to the kernel and then incremented (0x180060f44), zeroed
        // by Reset (0x1800616e0). The first frame and a reset use seed 0, so
        // their noise is the precomputed field's; it moves every frame after.
        uint32_t& counter = fs ? fs->seed : t.seed;
        if (!(fs ? fs->latch : t.latch) || temporal->reset) counter = 0;
        pre_seed = counter++;
        // diagnostics (wrong picture): price the per-frame noise and the history.
        if (std::getenv("NR_DIAG_SEED0")) pre_seed = 0;
        if (std::getenv("NR_DIAG_TGATE0")) gate = false;
        const uint32_t p = fs ? fs->parity : t.parity;
        if (engine) {
            // Engine motion: sampled in place (mv_direct, put back after the
            // passes), else one blit into the estimator's quarter-resolution
            // field, converting whatever format the game handed us.
            if (mv_direct) {
                barrier(cmd, engine->motion.image, engine->motion.before, VK_IMAGE_LAYOUT_GENERAL,
                        engine->motion.before_stage, engine->motion.before_access,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
            } else {
            barrier(cmd, engine->motion.image, engine->motion.before,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, engine->motion.before_stage,
                    engine->motion.before_access, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT);
            barrier(cmd, t.flow[0].handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkImageBlit mv{};
            mv.srcSubresource = mv.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            mv.srcOffsets[0] = {int32_t(engine->motion_x), int32_t(engine->motion_y), 0};
            mv.srcOffsets[1] = {int32_t(engine->motion_x + engine->motion.width),
                                int32_t(engine->motion_y + engine->motion.height), 1};
            mv.dstOffsets[1] = {int32_t(t.lw[0]), int32_t(t.lh[0]), 1};
            vkCmdBlitImage(cmd, engine->motion.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           t.flow[0].handle, VK_IMAGE_LAYOUT_GENERAL, 1, &mv, VK_FILTER_LINEAR);
            barrier(cmd, t.flow[0].handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
            barrier(cmd, engine->motion.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    engine->motion.after, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT, engine->motion.after_stage,
                    engine->motion.after_access);
            }
            // Depth, the same way and for the same reason: one blit into one
            // image, so the consumer has a single descriptor whether or not the
            // game supplies depth this frame.
            impl_->depth_unread = false;
            if (engine->depth.image && depth_direct) {
                // Sampled in place (Impl::DirectSrc); put back after the passes.
                barrier(cmd, engine->depth.image, engine->depth.before, VK_IMAGE_LAYOUT_GENERAL,
                        engine->depth.before_stage, engine->depth.before_access,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
            } else if (engine->depth.image && depth_stencil_aspects(engine->depth.format)) {
                // A depth/stencil buffer: sampled through a depth-aspect view
                // (runtime_depth.comp), because a blit may not convert it.
                const VkImageAspectFlags aspects = depth_stencil_aspects(engine->depth.format);
                const bool sampled = !engine->depth.usage || (engine->depth.usage & VK_IMAGE_USAGE_SAMPLED_BIT);
                const VkDescriptorSet set =
                    sampled ? impl_->depth_source(engine->depth.image, engine->depth.format) : VK_NULL_HANDLE;
                if (set) {
                    barrier(cmd, engine->depth.image, engine->depth.before, VK_IMAGE_LAYOUT_GENERAL,
                            engine->depth.before_stage, engine->depth.before_access,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT, aspects);
                    barrier(cmd, t.depth.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
                    const uint32_t push[6] = {t.depth.w, t.depth.h, engine->depth.width, engine->depth.height,
                                              engine->depth_x, engine->depth_y};
                    dispatch(cmd, impl_->depth_copy, (t.depth.w + 7) / 8, (t.depth.h + 7) / 8, 1, push,
                             sizeof push, set);
                    barrier(cmd, t.depth.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
                    barrier(cmd, engine->depth.image, VK_IMAGE_LAYOUT_GENERAL, engine->depth.after,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                            engine->depth.after_stage, engine->depth.after_access, aspects);
                } else {
                    impl_->depth_unread = true;
                }
            } else if (engine->depth.image) {
                barrier(cmd, engine->depth.image, engine->depth.before,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, engine->depth.before_stage,
                        engine->depth.before_access, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT);
                barrier(cmd, t.depth.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                VkImageBlit d{};
                // A colour-format depth (R32_FLOAT and the like) only: depth/stencil
                // formats took the branch above, since a blit may not convert them.
                d.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                d.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                d.srcOffsets[0] = {int32_t(engine->depth_x), int32_t(engine->depth_y), 0};
                d.srcOffsets[1] = {int32_t(engine->depth_x + engine->depth.width),
                                   int32_t(engine->depth_y + engine->depth.height), 1};
                d.dstOffsets[1] = {int32_t(t.depth.w), int32_t(t.depth.h), 1};
                // NEAREST, not LINEAR: interpolating across a silhouette invents
                // a depth that is on neither surface, which is precisely the
                // edge this tap exists to get right. Vulkan also forbids linear
                // filtering on a depth aspect.
                vkCmdBlitImage(cmd, engine->depth.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               t.depth.handle, VK_IMAGE_LAYOUT_GENERAL, 1, &d, VK_FILTER_NEAREST);
                barrier(cmd, t.depth.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
                barrier(cmd, engine->depth.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        engine->depth.after, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_TRANSFER_READ_BIT, engine->depth.after_stage,
                        engine->depth.after_access);
            }
        }
        for (unsigned k = 0; !engine && k < kTemporalLevels; ++k) {
            Temporal::LumaPush lp{t.lw[k], t.lh[k],
                                  k ? t.lw[k-1] : nw, k ? t.lh[k-1] : nh,
                                  k ? 1u : 0u};
            dispatch(cmd, t.luma_kernel[p][k], (t.lw[k]+7)/8, (t.lh[k]+7)/8, 1, &lp, sizeof lp);
            compute_barrier(cmd);
        }
        if (gate && !engine) {
            for (int k = int(kTemporalLevels) - 1; k >= 0; --k) {
                const bool first = unsigned(k) == kTemporalLevels - 1, last = k == 0;
                Temporal::FlowPush fp{t.lw[k], t.lh[k],
                                      first ? 1u : t.lw[k+1], first ? 1u : t.lh[k+1],
                                      kTemporalRadius[k], first ? 1u : 0u, last ? 1u : 0u,
                                      kTemporalReject};
                dispatch(cmd, t.flow_kernel[p][k], (t.lw[k]+7)/8, (t.lh[k]+7)/8, 1, &fp, sizeof fp);
                compute_barrier(cmd);
            }
        }
        // The estimator writes normalized backward motion, so the consumer's
        // MVecScale pair is 1.0.
        const bool have_depth = engine && engine->depth.image && !impl_->depth_unread;
        // [8..11]: where the frame's uv lands in the motion texture, uv * [8..9]
        // + [10..11] - the region's share of the allocation and its base, when
        // it is sampled in place; else 1 and 0 (the field covers the frame).
        const uint32_t mtw = engine && engine->motion_texture_width ? engine->motion_texture_width
                             : engine ? engine->motion_x + engine->motion.width : 1;
        const uint32_t mth = engine && engine->motion_texture_height ? engine->motion_texture_height
                             : engine ? engine->motion_y + engine->motion.height : 1;
        const float params[12] = {gate ? 1.0f : 0.0f,
                                  engine ? engine->motion_scale_x : 1.0f,
                                  engine ? engine->motion_scale_y : 1.0f,
                                  t.history_strength * kPostBlendScale,
                                  float(t.history.w), float(t.history.h),
                                  have_depth ? 1.0f : 0.0f,
                                  have_depth && engine->depth_inverted ? 1.0f : 0.0f,
                                  mv_direct ? float(engine->motion.width) / float(mtw) : 1.0f,
                                  mv_direct ? float(engine->motion.height) / float(mth) : 1.0f,
                                  mv_direct ? float(engine->motion_x) / float(mtw) : 0.0f,
                                  mv_direct ? float(engine->motion_y) / float(mth) : 0.0f};
        VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        b.srcAccessMask=VK_ACCESS_SHADER_READ_BIT;b.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
        b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        b.buffer=t.params.handle;b.size=VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,1,&b,0,nullptr);
        vkCmdUpdateBuffer(cmd,t.params.handle,0,sizeof params,params);
        b.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&b,0,nullptr);
    }
    // Bind this feature's history: copy the previously bound feature's state out
    // and this one's in. Only on a change, so the single-feature case records
    // nothing at all here.
    if (fs && impl_->bound_feature != temporal->feature) {
        if (impl_->bound_feature) {
            auto prev = impl_->features.find(impl_->bound_feature);
            if (prev != impl_->features.end()) copy_general(t.hist(t.hcur), prev->second.history);
        }
        if (!fs->cleared) {
            // A fresh feature's image is whatever the allocator left there. The
            // gate is closed on its first frame so nothing reads it, but an
            // unwritten float image can hold NaN and a NaN multiplied by a zero
            // gate is still NaN. This is also where it reaches GENERAL, which
            // it will stay in for the rest of its life.
            barrier(cmd, fs->history.handle, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cmd, fs->history.handle, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
            barrier(cmd, fs->history.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            fs->history.layout = VK_IMAGE_LAYOUT_GENERAL;
            fs->cleared = true;
        }
        copy_general(fs->history, t.hist(t.hcur));
        impl_->bound_feature = temporal->feature;
    }
    // The passes. Pass k > 0 takes pass k-1's answer (surf0, the blended
    // output) as its colour, the same motion and depth, and its own history:
    // store[k] is copied into the bound history image before the pass and
    // the model's write (surf1) copied back after it. With one pass the
    // history round trip is the single copy it always was.
    for (uint32_t pass = 0; pass < passes; ++pass) {
        if (pass > 0 && impl_->cascade_detail) {
            // Detail only: the first pass's input with the previous pass's
            // local structure on it (windows/shaders/passes/cascade_feed.comp).
            barrier(cmd, s.tex_in.handle, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
            struct { uint32_t w, h; float floor_; } lp{nw, nh, 1.0f / 512.0f};
            dispatch(cmd, impl_->cascade_lograt, (nw + 7) / 8, (nh + 7) / 8, 1, &lp, sizeof lp);
            compute_barrier(cmd);
            struct { uint32_t w, h; int32_t radius; uint32_t dir; } bp{nw, nh, impl_->cascade_radius, 0u};
            dispatch(cmd, impl_->cascade_blur_h, (nw + 7) / 8, (nh + 7) / 8, 1, &bp, sizeof bp);
            compute_barrier(cmd);
            bp.dir = 1u;
            dispatch(cmd, impl_->cascade_blur_v, (nw + 7) / 8, (nh + 7) / 8, 1, &bp, sizeof bp);
            compute_barrier(cmd);
            struct { uint32_t w, h; float guard; } fp{nw, nh, 2.0f};
            dispatch(cmd, impl_->cascade_feed, (nw + 7) / 8, (nh + 7) / 8, 1, &fp, sizeof fp);
            barrier(cmd, s.tex_in.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
            compute_barrier(cmd);
            if (temporal) copy_general(impl_->history_store[pass], t.history);
        } else if (pass > 0) {
            barrier(cmd, s.surf0.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            barrier(cmd, s.tex_in.handle, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkImageCopy c{};
            c.srcSubresource = c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            c.extent = {nw, nh, 1};
            vkCmdCopyImage(cmd, s.surf0.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           s.tex_in.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
            barrier(cmd, s.tex_in.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
            barrier(cmd, s.surf0.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            if (temporal) copy_general(impl_->history_store[pass], t.history);
        }
        const bool time_net = impl_->timing_ok && impl_->split_timing;
        const uint32_t net_query = timing_slot * impl_->timing_stride + 2 + 2 * pass;
        if (time_net) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, impl_->timing, net_query);
        for (size_t i = 0; i < s.steps.size(); ++i) {
            const auto& step = s.steps[i];
            const nrvk::Kernel* use = step.k;
            if (mask && step.k==impl_->original_pre) use = &impl_->mask_pre;
            VkDescriptorSet set = VK_NULL_HANDLE;
            if (mask && step.k==impl_->original_pre) {}
            else if (!temporal && ds && step.k==impl_->plain_pre) set = ds->pre[0];
            else if (!temporal && ds && step.k==impl_->plain_post) set = ds->post[0];
            else if (temporal && step.k==t.original_pre) {
                use = t.pingpong ? &t.pre_pp[t.hcur] : &t.pre;
                if (sets) set = sets->pre[t.hcur];
            } else if (temporal && step.k==t.original_post) {
                use = t.pingpong ? &t.post_pp[t.hcur] : &t.post;
                if (sets) set = sets->post[t.hcur];
            }
            // Each pass has its own controls; see `pass_push`. An empty entry
            // means that dispatch carries none, so pass 1's push stands.
            const void* push = step.push;
            if (pass > 0 && pass < impl_->pass_push.size() &&
                i < impl_->pass_push[pass].size() && !impl_->pass_push[pass][i].empty())
                push = impl_->pass_push[pass][i].data();
            // This frame's noise seed, computed in the shader: the precomputed
            // field is seed 0's.
            uint8_t pre_push[kPrePushBytes];
            if (temporal && step.k == t.original_pre && pre_seed) {
                if (step.push_bytes != kPrePushBytes)
                    throw std::runtime_error("temporal pre block: unexpected push size");
                const uint32_t no_field = 0;
                std::memcpy(pre_push, push, kPrePushBytes);
                std::memcpy(pre_push + kPreSeedAt, &pre_seed, 4);
                std::memcpy(pre_push + kPreNoiseFieldAt, &no_field, 4);
                push = pre_push;
            }
            dispatch(cmd, *use, step.gx, step.gy, step.gz, push, step.push_bytes, set);
            // Execution-only between network steps when the arena is coherent,
            // invalidate-only when it is not; the barrier after the last step
            // stays full, because the passes that follow it are not part of
            // either contract - they were not built with device-scoped loads
            // and the arena is not the only thing they read.
            const bool last = i + 1 >= s.steps.size();
            // C=512 dispatches joined by tile counters (nr_graph.cpp NR_TCHAIN) have no barrier.
            if (!last && i < s.runner.no_barrier_after.size() && s.runner.no_barrier_after[i]) continue;
            compute_barrier(cmd, s.runner.exec_barrier && !last,
                            s.runner.inv_barrier && !last);
        }
        if (time_net) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, impl_->timing, net_query + 1);
        if (temporal && !t.pingpong) {
            // History is a copy of what the MODEL wrote - the temporal post variant
            // puts that on the second surface, before the application strength -
            // and it is taken independently of whatever composition follows.
            const nrvk::Context::Image& dst = passes > 1 ? impl_->history_store[pass] : t.history;
            barrier(cmd, s.surf1.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            barrier(cmd, dst.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkImageCopy history_copy{};
            history_copy.srcSubresource = history_copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
            history_copy.extent = {nw, nh, 1};
            vkCmdCopyImage(cmd, s.surf1.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           dst.handle, VK_IMAGE_LAYOUT_GENERAL, 1, &history_copy);
            barrier(cmd, dst.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
            barrier(cmd, s.surf1.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }
    }
    if (temporal) {
        // Leave the bound history holding the FIRST pass's, which is what the
        // next frame's first pass reads.
        if (passes > 1) copy_general(impl_->history_store[0], t.history);
        // The latch is set after every recorded frame and `reset` never clears
        // it: reset suppresses consumption, it does not un-write the buffer.
        if (fs) { fs->latch = true; fs->parity ^= 1u; }
        else { t.latch = true; t.parity ^= 1u; }
        // The post variant wrote this frame's history into the other image.
        if (t.pingpong) t.hcur ^= 1u;
        if (history_consumed) *history_consumed = gate;
    }
    impl_->check_record(cmd);
    // Retired features, destroyed once enough recordings have gone by that the
    // submission holding their last use cannot still be executing.
    ++impl_->recordings;
    for (size_t i = impl_->retiring.size(); i-- > 0;) {
        if (impl_->recordings < impl_->retiring[i].second) continue;
        s.ctx.destroy(impl_->retiring[i].first);
        impl_->retiring.erase(impl_->retiring.begin() + long(i));
    }
    info.network_dispatches = uint32_t(s.steps.size()) * passes;
    if (prep_on) {
        // The answer back into the frame's own domain. With later passes the
        // first pass's input is in shown_keep, and the frame goes back there.
        const bool multi = impl_->max_passes > 1;
        if (!multi)
            barrier(cmd, s.tex_in.handle, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        compute_barrier(cmd);
        prep_push.mode = 2;
        prep_push.flags = (prep_push.flags & ~kPrepReset) | (multi ? kPrepRestore : 0u);
        dispatch(cmd, multi ? impl_->prep_back_k : impl_->prep_k, (nw + 15) / 16, (nh + 15) / 16, 1,
                 &prep_push, sizeof prep_push);
        compute_barrier(cmd);
        if (!multi)
            barrier(cmd, s.tex_in.handle, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    if (c.apply_model) {
        if (mask) {
            struct { uint32_t w,h;float intensity; } push{frame.width,frame.height,c.intensity};
            dispatch(cmd,impl_->mask_resolve,(frame.width+7)/8,(frame.height+7)/8,1,&push,sizeof push);
            compute_barrier(cmd);
        }
        if (!mask && !impl_->native_compose) {
            // The model's answer onto the frame, as OptiScaler DLSS-NR's
            // resolve; see the shader. At the frame extent; the model-sized
            // answer is sampled.
            const bool prep_variant = prep_on && impl_->transfer_prep.device;
            // runtime_transfer.comp's mode: 0 classic, 1 matched residual, 2 edge-aware
            static const uint32_t kShaderMode[3] = {1u, 2u, 0u};
            const int enlargement = std::clamp(c.transfer, 0, 2);
            const uint32_t mode = kShaderMode[enlargement];
            const uint32_t scaler = uint32_t(std::clamp(c.classic_scaler, 0, 3));
            const bool enlarge = impl_->scaled && enlargement == kEnlargeClassic && scaler != 0;
            if (enlarge) {
                const uint32_t up[] = {frame.width, frame.height, nw, nh, scaler};
                dispatch(cmd, prep_variant ? impl_->upscale_prep : impl_->upscale_pass,
                         (frame.width + 7) / 8, (frame.height + 7) / 8, 1, up, sizeof up);
                compute_barrier(cmd);
            }
            const TransferPush push{frame.width, frame.height, nw, nh, impl_->linear ? 0u : 1u,
                                    c.detail_strength, c.colour_strength, c.max_ratio, impl_->white_point, mode,
                                    (enlarge ? kTransferEnlarged : 0u) | (c.transfer_guided ? kTransferGuided : 0u) |
                                        (impl_->store_native && impl_->round_u8 ? kTransferRoundU8 : 0u),
                                    c.transfer_sigma};
            const nrvk::Kernel& transfer = prep_variant ? impl_->transfer_prep : impl_->transfer_pass;
            dispatch(cmd, transfer, (frame.width + 7) / 8, (frame.height + 7) / 8, 1, &push, sizeof push,
                     fr ? (prep_variant ? fr->transfer_prep : fr->transfer) : VK_NULL_HANDLE);
            compute_barrier(cmd);
        }
        const VkImage out = impl_->store_native ? impl_->keep_full.handle
                            : impl_->scaled ? impl_->full_out.handle : s.surf0.handle;
        const uint32_t extent[] = {frame.width, frame.height, uint32_t(impl_->round_u8)};
        if ((!impl_->post_alpha || mask) && !impl_->store_native)
            dispatch(cmd, impl_->alpha, (frame.width + 7) / 8, (frame.height + 7) / 8, 1, extent, sizeof extent);
        if (stored) {
            // The post block's store was the write-back.
        } else {
        barrier(cmd, out, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        if (tgt)
            barrier(cmd, tgt->image, tgt->before, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    tgt->before_stage, tgt->before_access, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        else
        barrier(cmd, frame.image, ds ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                ds ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
                ds ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        if (impl_->direct_out || impl_->store_native) {
            // Already the frame's bits (an *_SRGB frame takes its UNORM twin's).
            vkCmdCopyImage(cmd, out, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           wb, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        } else if(encoded.handle) {
            barrier(cmd,encoded.handle,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);
            vkCmdBlitImage(cmd,out,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           encoded.handle,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&blit,VK_FILTER_NEAREST);
            barrier(cmd,encoded.handle,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_READ_BIT);
            vkCmdCopyImage(cmd,encoded.handle,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           wb,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&region);
            barrier(cmd,encoded.handle,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_TRANSFER_WRITE_BIT);
        } else if (impl_->transfer == Transfer::DirectBlit) {
            vkCmdBlitImage(cmd, out, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           wb, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
        } else {
            vkCmdCopyImage(cmd, out, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           wb, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        }
        barrier(cmd, out, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }
        info.applied = true;
    }
    if (tgt) {
        barrier(cmd, frame.image, ds ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, frame.after,
                ds ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
                ds ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_TRANSFER_READ_BIT, frame.after_stage, frame.after_access);
        if (stored)
            barrier(cmd, tgt->image, VK_IMAGE_LAYOUT_GENERAL, tgt->after, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT, tgt->after_stage, tgt->after_access);
        else
        barrier(cmd, tgt->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, tgt->after, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, tgt->after_stage, tgt->after_access);
    } else if (stored)
        barrier(cmd, frame.image, VK_IMAGE_LAYOUT_GENERAL, frame.after, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, frame.after_stage, frame.after_access);
    else
    barrier(cmd, frame.image, c.apply_model ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                              : ds ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            frame.after, c.apply_model || !ds ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            c.apply_model ? VK_ACCESS_TRANSFER_WRITE_BIT : ds ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_TRANSFER_READ_BIT,
            frame.after_stage, frame.after_access);
    if (depth_direct)
        barrier(cmd, engine->depth.image, VK_IMAGE_LAYOUT_GENERAL, engine->depth.after,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                engine->depth.after_stage, engine->depth.after_access);
    if (mv_direct)
        barrier(cmd, engine->motion.image, VK_IMAGE_LAYOUT_GENERAL, engine->motion.after,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT,
                engine->motion.after_stage, engine->motion.after_access);
    if (engine && temporal)
        impl_->incheck.end(s.ctx, cmd, tgt && (tgt->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ? *tgt : frame, *engine, impl_->temporal.flow[0], impl_->temporal.depth,
                           depth_direct, engine->depth.image && !impl_->depth_unread,
                           history_consumed && *history_consumed, impl_->mw, impl_->mh, mv_direct);
    if (impl_->timing_ok) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, impl_->timing,
                            timing_slot * impl_->timing_stride + 1);
        impl_->slot_passes[timing_slot] = impl_->split_timing ? passes : 0;
        ++impl_->timings_recorded;
        impl_->timing_slot = (timing_slot + 1) % Impl::kTimingSlots;
        // Collect whatever has finished. Never waits; the caller is a render
        // thread and this is a diagnostic.
        impl_->read_timing();
    }
    return result;
}

void set_input_check(bool on, const std::string& folder, int pictures) {
    incheck::g_folder = folder;
    incheck::g_dumps_left = pictures;
    incheck::g_on = on;
    if (on) nr::logf("[nr] input check on: every ~600 recordings the engine inputs and the output are read back "
                     "and scored; pictures of the first %d in which the picture moves in %s", pictures,
                     folder.empty() ? "(none)" : folder.c_str());
}

float Runtime::last_gpu_ms() const { return impl_->gpu_ms; }
float Runtime::average_gpu_ms() const { return impl_->gpu_ms_avg; }
float Runtime::average_network_ms() const { return impl_->split_timing ? impl_->net_ms_avg : 0.0f; }
std::pair<float, float> Runtime::preprocess_meter() const {
    const float* s = static_cast<const float*>(impl_->prep_state.mapped);
    if (!s || s[1] != 1.0f) return {NAN, NAN};
    return {s[0], s[3]};
}
}  // namespace nr
