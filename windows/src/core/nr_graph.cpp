// The network as a graph: one activation arena, one weight arena, one submit.
//
// Every other host in `src/` drives a single kernel against a captured gold.
// This one owns the frame. It reads the frame plan - the ordered dispatch list
// derived from the shipping descriptor - loads every layer's weights once, and
// records the whole chain
// into one command buffer.
//
// **What makes this possible is that the kernels now address the image and
// share one format.** Until `NR_IMAGE` a fused Swin kernel read window-major,
// so the host gathered the windows and applied the shift, and would have had to
// do it again for the next layer - on the CPU, 152 times a frame. And until
// `NR_R_E4M3` the projection kernels read their residual as linear f16, which
// would have needed a second arena in a second format written by every layer.
// Both are gone: one e4m3 tile-blocked arena, indexed by global 4x4-pixel tile,
// and the host does nothing between layers.
//
// Two things this is not, yet: every family - `--report` says which - and the
// OptiScaler entry point. The arenas are already the shapes that adapter wants.
#include "nr_shader_manifest.hpp"
#include "nr_activation_lut.hpp"
#include "nr_log.hpp"
#include "nrvk.hpp"
#include "telemetry.hpp"
#include "tinlayout.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <functional>
#include <vector>

// The ViT GEMM's wide tile, `NR_MTILE`/`NR_NTILE` in gemm1x1.comp. It is a
// shader constant that the host's grid arithmetic also needs, so the shader
// build derives both from the same defines and records the result in
// `shader-constants.txt` beside the SPVs, which this binary checks against
// whatever `--spv-dir` it was pointed at. See the check in `build()`.
#ifndef NR_GEMM_WIDE_MT
#define NR_GEMM_WIDE_MT 128
#endif
#ifndef NR_GEMM_WIDE_NT
#define NR_GEMM_WIDE_NT 256
#endif
// The ViT QKV at small extents (gemmvqkvnorms, 2x6 fragments a wave): at 1080p its
// 640 tokens make 480 waves of the 4x4 tile, 3.75 a SIMD; the 2x6 tile makes 640
// (5 a SIMD). 1440p and 4K keep gemmvqkvnorm.
#ifndef NR_GEMM_QKVS_MT
#define NR_GEMM_QKVS_MT 32
#endif
#ifndef NR_GEMM_QKVS_NT
#define NR_GEMM_QKVS_NT 384
#endif
#ifndef NR_QKVS_MAX_TOKENS
#define NR_QKVS_MAX_TOKENS 768
#endif
// ffwd3 subgroups per workgroup (one group each); the grid scales by 8/this.
#ifndef NR_FFWD_WGW
#define NR_FFWD_WGW 8
#endif
// The post block can restore the frame's alpha itself (fswin_t.comp NR_POST_ALPHA;
// manifest post_alpha), which the runtime uses instead of its alpha pass.
#ifndef NR_POST_ALPHA
#define NR_POST_ALPHA 0
#endif
// The C=512 dispatches (attn, the projections on gemmprojc, ffwd3/ffwd3w) carry
// tile counters (NR_TCHAIN=1 builds, manifest tchain): consecutive ones wait on
// the token tiles they read instead of a barrier (see the NR_TCHAIN block).
// NR_TCHAIN=0 in the environment keeps the barriers.
#ifndef NR_TCHAIN_TILES
#define NR_TCHAIN_TILES 0
#endif
// `ffwd3w` (ffwd3_t.comp NR_FFWD_FM=2: two 16-token tiles a subgroup, each weight
// fragment feeding two MMAs) for C=512 layers of at least this many tokens; 0: never.
// Fewer, heavier subgroups: 4K -14% a layer, 1080p +3%.
// The subgroups of an ffwd3 workgroup share one weight group and take
// consecutive token tiles (ffwd3_t.comp NR_FFWD_GMAJOR); must match the SPVs.
#ifndef NR_GEMM_REMAP_HOST
#define NR_GEMM_REMAP_HOST 0  // gemm1x1 NR_GEMM_REMAP_PC workgroup order; must match the SPVs
#endif
#ifndef NR_FFWD_GMAJOR
#define NR_FFWD_GMAJOR 0
#endif
#ifndef NR_FFWD_FM2_MIN_TOKENS
#define NR_FFWD_FM2_MIN_TOKENS 0
#endif

// upsview/repack move a token's 16-channel run per invocation.
#ifndef NR_UPSVIEW_VEC
#define NR_UPSVIEW_VEC 0
#endif
// The upsample view as the producer's own store when the view is a copy (see
// the F_UPS lowering).
#ifndef NR_UPSVIEW_FOLD
#define NR_UPSVIEW_FOLD 1
#endif
#ifndef NR_REPACK_VEC
#define NR_REPACK_VEC 0
#endif
// Which wide widths use the fused upsample body (bit 1 C64, 2 C128, 4 C256).
#ifndef NR_WIDE_UPS_MASK
#define NR_WIDE_UPS_MASK 7
#endif
// Straggler-queue threshold in percent of a layer's queued items (0: off). Must
// match the shaders' `persist_strag` constant. NR_STRAG_PCT_ENV overrides it for
// experiments only.
#ifndef NR_STRAG_PCT
#define NR_STRAG_PCT 0
#endif
// One workgroup per item for the folded runs of 65..NR_PERSIST_ONE_MAX windows a
// layer, per width (bits as persist_ds: 2 C=128, 4 C=256); those widths'
// fswinpds/fswinpup SPVs must be built with NR_PERSIST_ONE=1. Must match the
// shaders' `persist_one` constant.
#ifndef NR_PERSIST_ONE_MASK
#define NR_PERSIST_ONE_MASK 0
#endif
#ifndef NR_PERSIST_ONE_MAX
#define NR_PERSIST_ONE_MAX 400
#endif
// The pre-block's noise features precomputed once at build into the weight arena
// (noise_field.comp); fswinimagepreds32 must be built with NR_NOISE_FIELD=1. Must
// match the shaders' `noise_field` constant.
#ifndef NR_NOISE_FIELD
#define NR_NOISE_FIELD 0
#endif
struct NoiseJob { uint32_t off, width, height, seed; float noise; };
static std::vector<NoiseJob> g_noise_jobs;
// Fold the ViT token-order repacks into FinalHead's and the last ViT
// projection's stores (gemm1x1 NR_OUT_PERM); 0 keeps the repack dispatches.
#ifndef NR_REPACK_FOLD
#define NR_REPACK_FOLD 0
#endif
// decups channels per invocation (1 = the original per-element grid).
#ifndef NR_DECUPS_VEC
#define NR_DECUPS_VEC 1
#endif
// Fuse the learned downsample projection into the pooled body
// (bit 1: C=32, 2: C=64, 4: C=128, 8: C=256).
#ifndef NR_DS_FUSE
#define NR_DS_FUSE 0
#endif
// Dependency-driven claims in persistent runs (items are only claimed ready).
static size_t g_prof_words = 0, g_prof_off = 0;  // NR_PROF
static uint32_t g_chain_epoch_word = 0xFFFFFFFFu, g_chain_base = 0, g_chain_n = 0;  // NR_CHAIN
static std::set<std::string> g_chain_kern;
static std::vector<uint8_t> g_chain_nobar;
// NR_TCHAIN: barriers replaced by tile counters; the counters and error words
// (activation arena words from g_tc_base, zero-filled with the arena).
static std::vector<uint8_t> g_tc_nobar;
static std::vector<uint32_t> g_tc_words;
static std::vector<size_t> g_tc_err;   // error words, as indices into g_tc_words
static size_t g_tc_base = 0;
static bool tchain_active() {
    const char* e = std::getenv("NR_TCHAIN");
    return NR_TCHAIN_TILES && !(e && *e && std::atoi(e) == 0) && g_chain_epoch_word != 0xFFFFFFFFu;
}
// Which barriers a tile counter replaces. C=512: attn -> its projection ->
// ffwd3 -> FFN projection -> attn. ViT: expand -> contraction -> QKV, and
// attention -> projection -> the next expand (QKV -> attention needs every
// token and keeps its barrier). The arena layout probe asks before any offset
// exists, so this looks only at the kernels; the build checks the data offsets.
static bool tchain_kern(const std::string& k);
#define NR_KBASE(k) (k)
static bool tchain_pair(const std::string& p, const std::string& q) {
    if (!tchain_kern(p) || !tchain_kern(q)) return false;   // both ends built with NR_TCHAIN
    auto g = [](const std::string& k) { return k == "gemmprojc" || k == "gemmproj5"; };
    auto f = [](const std::string& k) { return k == "ffwd3" || k == "ffwd3w"; };
    auto vp = [](const std::string& k) { return k == "gemmproj" || k == "gemmprojw" || k == "gemmprojt"; };
    auto up = [](const std::string& k) { return k == "fswinpup256" || k == "fswinpup128" || k == "fswinpup64"; };
    const uint32_t kinds = std::getenv("NR_TCHAIN_KINDS")   // diagnostic: bit per group below
        ? uint32_t(std::strtoul(std::getenv("NR_TCHAIN_KINDS"), nullptr, 0)) : (nr::binary::directory().empty() ? 0x1u : 0x1ffu);
    // diagnostic NR_TCHAIN_C512=<bits>: 1 attn -> projection, 2 projection -> FFN, 4 FFN -> projection, 8 projection -> attn
    static const uint32_t c512 = std::getenv("NR_TCHAIN_C512") ? uint32_t(std::strtoul(std::getenv("NR_TCHAIN_C512"), nullptr, 0)) : 0xFu;
    return ((kinds & 1) && (((c512 & 1) && p == "attn" && g(q)) || ((c512 & 2) && g(p) && f(q)) ||
                            ((c512 & 4) && f(p) && g(q)) || ((c512 & 8) && g(p) && q == "attn"))) ||
           ((kinds & 2) && p == "gemmvact" && vp(q)) || ((kinds & 4) && vp(p) && (q == "gemmvqkvnorm" || q == "gemmvqkvnorms")) ||
           ((kinds & 8) && p == "vitattn" && vp(q)) || ((kinds & 16) && vp(p) && q == "gemmvact") ||
           // the decoder: C=512 -> the UPS-first C=256 run -> C=128 -> C=64 -> the C=32 fused upsample
           ((kinds & 32) && ((g(p) && up(q)) || (up(p) && (up(q) || q == "fswinfusedup32")))) ||
           // the encoder: each level's downsample -> the next level's DS-last run
           ((kinds & 64) && ((p == "fswindsp32" && q == "fswinpds64") || (p == "fswinpds64" && q == "fswinpds128") ||
                             (p == "fswinpds128" && q == "fswinpds256"))) ||
           // the C=256 run's downsample -> the first C=512 FFN (needs the frame tick)
           ((kinds & 128) && p == "fswinpds256" && f(q));
}
// The kernels built with NR_TCHAIN=1, which carry the five push words right
// after their push block as this file declares it (gemm1x1 pipelines therefore
// all build NR_GEMM_REMAP_PC=1, so their block ends with `remap` like PushGemm).
static bool tchain_kern(const std::string& k) {
    if (!nr::binary::directory().empty()) {
        std::string base = k;
        if (base.size() > 2 && base.compare(base.size()-2, 2, "nh") == 0) base.resize(base.size()-2);
        return base == "attn" || base == "gemmprojc" || base == "ffwd3" || base == "ffwd3w" ||
               base == "gemmvact" || base == "gemmproj" || base == "gemmprojw" || base == "gemmprojt" ||
               base == "gemmvqkvnorm" || base == "gemmvqkvnorms" || base == "vitattn" ||
               base == "fswinpup256" || base == "fswinpup128" || base == "fswinpup64" || base == "fswinfusedup32" ||
               base == "fswinpds256" || base == "fswinpds128" || base == "fswinpds64" || base == "fswindsp32" || base == "fswin32";
    }
    // Windows: only the C=512 chain's kernels are built with NR_TCHAIN. Under LLPC the counter
    // code costs the persistent runs ~0.11 ms at 1080p even unused, more than their chains save.
    return k == "attn" || k == "gemmproj5" || k == "ffwd3" || k == "ffwd3w" || k == "fswin32t" || k == "fswin32nht";
}
#ifndef NR_PERSIST_DF
#define NR_PERSIST_DF 0
#endif
// Fold the level's fused downsample (`fswindsp<C>`) into the persistent run
// that precedes it, as the run's last layer (bit 1: C=64, 2: C=128, 4: C=256).
// The fswinp<C> SPV must be built with NR_PERSIST_DS=1 for every set bit.
// Shared arena: a slot smaller than NR_ARENA_COLD_MAX bytes only takes memory
// freed at least NR_ARENA_COLD dispatches before its first one (see the reuse layout).
#ifndef NR_ARENA_COLD
#define NR_ARENA_COLD 1
#endif
#ifndef NR_ARENA_COLD_MAX
#define NR_ARENA_COLD_MAX 8388608
#endif
#ifndef NR_PERSIST_DS_MASK
#define NR_PERSIST_DS_MASK 0
#endif
// Fold the level's wide fused upsample (`fswinfusedup<C>`) into the
// persistent run that follows it, as the run's first layer (same bits).
#ifndef NR_PERSIST_UPS_MASK
#define NR_PERSIST_UPS_MASK 0
#endif
// The residual projection pipeline (`gemmproj`) gets its own tile, so its
// workgroup can shrink to four waves without moving gemmnores/gemmpool/gemmds.
#ifndef NR_GEMM_PROJ_MT
#define NR_GEMM_PROJ_MT 64
#endif
#ifndef NR_GEMM_PROJ_NT
#define NR_GEMM_PROJ_NT 128
#endif
// ViT's FFN contraction (K = 4096) at large token counts gets a second
// projection pipeline, `gemmprojw`, with 64x64 wave tiles (one load VGPR per
// WMMA instead of two). Same k order per output, so the same bytes. Below
// NR_PROJW_MIN_TOKENS the 32x32 tile keeps more waves in flight and wins
// (with the remapped workgroup order: 640 tokens 25 vs 38 us, 960 tokens 44.7
// vs 40.4 us, so the product sets 768).
#ifndef NR_GEMM_PROJW_MT
#define NR_GEMM_PROJW_MT 64
#endif
#ifndef NR_GEMM_PROJW_NT
#define NR_GEMM_PROJW_NT 128
#endif
#ifndef NR_PROJW_MIN_TOKENS
#define NR_PROJW_MIN_TOKENS 2048
#endif
// The decoder input's 1024->512 GEMM (640 tokens at 1080p) on the wide
// tile is 20 workgroups - 80 waves for 128 SIMDs. `gemmvqkvs` is the same
// no-residual FP16-output product on the projection tile (NR_GEMM_PROJ_MT/NT),
// four times the workgroups; every output keeps its k order, so the same bytes.
#ifndef NR_DECQ_SMALL
#define NR_DECQ_SMALL 0
#endif

namespace {

const char* arg(int c, char** v, const char* k, const char* d = nullptr) {
    for (int i = 1; i + 1 < c; ++i) if (!std::strcmp(v[i], k)) return v[i + 1];
    return d;
}
bool flag(int c, char** v, const char* k) {
    for (int i = 1; i < c; ++i) if (!std::strcmp(v[i], k)) return true;
    return false;
}
// The per-layer and scoring tools read values by slot after the frame; they
// keep the one-value-one-slot layout (no arena reuse) and the plain repacks.
bool slot_reading_diagnostic(int c, char** v) {
    for (const char* f : {"--only", "--seed-dir", "--oracle", "--hash-values", "--dump", "--dump-stage",
                          "--seed", "--score", "--ds-only", "--barrier-report", "--dump-layer"})
        if (flag(c, v, f) || arg(c, v, f)) return true;
    return false;
}

// The model pack: every weight file the graph reads, byte for byte, in one file
// (linux/package/model-tools/pack_model.py). Paths under `root` are looked up by their
// relative name instead of opened; entries are read on demand, so peak memory
// is what it was with loose files.
struct ModelPack {
    std::string root, file;
    std::map<std::string, std::pair<uint64_t, uint64_t>> index;   // name -> offset, size
};
ModelPack g_pack;
bool open_model_pack(const std::string& file, const std::string& root, std::string* why) {
    std::ifstream f(file, std::ios::binary);
    char magic[8];
    uint32_t count = 0, reserved = 0;
    if (!f.read(magic, 8) || std::memcmp(magic, "NRMODEL1", 8) ||
        !f.read(reinterpret_cast<char*>(&count), 4) || !f.read(reinterpret_cast<char*>(&reserved), 4)) {
        if (why) *why = "not a model pack: " + file;
        return false;
    }
    ModelPack p; p.root = root; p.file = file;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t n = 0; uint64_t off = 0, size = 0;
        if (!f.read(reinterpret_cast<char*>(&n), 4) || n > 4096) { if (why) *why = "model pack index is damaged"; return false; }
        std::string name(n, '\0');
        if (!f.read(name.data(), n) || !f.read(reinterpret_cast<char*>(&off), 8) ||
            !f.read(reinterpret_cast<char*>(&size), 8)) { if (why) *why = "model pack index is damaged"; return false; }
        p.index[name] = {off, size};
    }
    g_pack = std::move(p);
    return true;
}
std::vector<uint8_t> slurp(const std::string& p) {
    if (std::getenv("NR_WEIGHT_TRACE")) std::fprintf(stderr, "weight-read %s\n", p.c_str());
    if (!g_pack.file.empty() && p.size() > g_pack.root.size() &&
        p.compare(0, g_pack.root.size(), g_pack.root) == 0 &&
        (p[g_pack.root.size()] == '/' || p[g_pack.root.size()] == '\\')) {
        std::string name = p.substr(g_pack.root.size() + 1);
        for (char& c : name) if (c == '\\') c = '/';
        const auto it = g_pack.index.find(name);
        if (it == g_pack.index.end()) return {};
        std::ifstream f(g_pack.file, std::ios::binary);
        std::vector<uint8_t> v(size_t(it->second.second));
        if (!f.seekg(std::streamoff(it->second.first)) ||
            !f.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size()))) return {};
        return v;
    }
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
}
size_t align(size_t x, size_t a) { return (x + a - 1) / a * a; }
// **The tile row stride truncates: it is the number of *whole* 4-pixel tiles in
// the layer's own width, and a partial tile at the right edge is not addressed.**
//
// At level 1 the width is 540, a multiple of four, so truncating and rounding up
// agree and the choice is invisible. At level 2 it is 270, and the two differ:
// 67 tiles against 68. Measured, not reasoned - `fswin_test --find-tile` takes
// the host model's output for the window's four tiles and searches NVIDIA's
// b005l0 capture for the tile each one actually landed in. Tile 2 lands at gold
// tile **67**, with the runner-up at 14%, and at that stride all four tiles
// match at 72-79% - the same quality the identical weights get on the synthetic
// Gaussian gold. Rounding up put the whole second tile row of every window one
// tile late and cost block 5 half its agreement.
//
// This file had it right once (`s.H / 4`) and lost it to a helper that looked
// more careful. A rounding rule is a measurement here, not a convention.
// `--tiles-up` rounds the stride up instead. Only odd extents differ - 540/4 is
// 135 either way, 270/4 is 67 or 68 - so this changes nothing above the first
// downsample and everything below it, which is exactly where the residual error
// lives: the `_upsample` family degrades monotonically with depth (0.9986 at
// level 2, then 0.83, 0.73, 0.26) with its magnitudes already exact.
bool g_tiles_up = false;
// **The plan's own token counts give the stride as an identity, not a rule.**
// `tokens == 64 * gx * gy` holds at all six levels of `frame_plan.txt` with no
// exception, and `64 = 16 * 2 * 2`, so NVIDIA's tile grid is exactly the window
// grid doubled - `2*gx` by `2*gy` - and the row stride is
//
//     tiles = 2 * ceil(px / 8)
//
//     px    1080  540  270  135   67   33      (H, at the six levels)
//     2*gx   270  136   68   34   18   10
//     px/4   270  135   67   33   16    8      <- what shipped
//     px     1920  960  480  240  120   60     (W)
//     2*gy   480  240  120   60   30   16
//     px/4   480  240  120   60   30   15      <- ditto
//
// Only the top level agrees, which is why every seam this project has chased
// has been *below* the first downsample.
//
// **And it is the allocation, not the stride** - `--tiles-win` measured, and it
// is worse at eighteen layers and better at none:
//
//     trunc   42 >= .99 |  8 | 2 | 86   mean 0.3792
//     win     34 >= .99 |  3 | 10 | 91  mean 0.3633
//     b015l0  0.9995 -> 0.7929   b022l0  0.9989 -> 0.7395
//     b024l2  0.9320 -> 0.5787   the whole C=512 bottleneck follows it down
//
// So `64 * gx * gy` is how big NVIDIA's slot is - padded to whole 8-pixel
// windows - and `px / 4` is how wide a row of it is. Third time this project
// has had to separate a size from a stride; see [[the-ds-seam-is-closed]].
bool g_tiles_win = false;
uint32_t tiles_of(int px) {
    if (g_tiles_win) return uint32_t(2 * ((px + 7) / 8));
    return uint32_t(g_tiles_up ? (px + 3) / 4 : px / 4);
}
// **A `_ds` layer's view raster is padded to whole 8-pixel windows.** The
// capture file sizes say so at every level and nothing else does:
//
//     block   unpadded   padded    capture / C   the block that reads it
//        0   540 x 960  544 x 960    522240      b001  522240
//        4   270 x 480  272 x 480    130560      b005  130560
//        8   135 x 240  136 x 240     32640      b009   32640
//       14    67 x 120   72 x 120      8640      b015    8640
//       22    33 x  60   40 x  64      2560      b023    2560
//
// The view layout is `(ch/16)*(W*H*16) + (y*W + x)*16 + view_sub(ch)`, so **W is
// the row stride** - reading it at the unpadded width shears every row by the
// padding gap, and the encoder's scores fall exactly in step with that gap:
// 0.7% -> r=0.9934, 0.7% -> 0.9877, 7.5% -> 0.6341, 21% -> 0.1107. Monotonic.
static size_t pad8(size_t v) { return (v + 7) / 8 * 8; }
// **A `_ds` view raster is padded to whole 4-pixel tiles**, which is not the
// same as the whole 8-pixel windows the capture's *size* suggests. Found by the
// one statistic that needs no channel model at all: the per-pixel L2 norm of a
// view tensor is an image, so the right raster is the one whose norm image is
// spatially smooth. Roughness (lower is smoother) for b022l0's capture:
//
//     36 x 60, plane 36*60   **0.1588**   <- pad4
//     33 x 60, plane 33*60     0.3532     <- what shipped, unpadded
//     36 x 60, plane 2560      0.5383
//     40 x 64, plane 40*64     0.6705     <- pad8, the capture's own size
//     32 x 60, plane 32*60     0.8402
//
// Half the roughness of anything else. 33 -> 36, 67 -> 68, 135 -> 136,
// 270 -> 272, 540 -> 540: only the last is unchanged, which is why level 1 was
// always right and the damage grew with depth.
static size_t pad4(size_t v) { return (v + 3) / 4 * 4; }
// **The half-resolution raster is the window grid, read out of the `_ds_fp8`
// kernel's own store address.** `codex` traced it from `/tmp/ds8.sass`:
//
//     Wp = 4 * ceil(W / 8)          Hp = 4 * ceil(H / 8)
//     addr = B + 16 * (Wp * (Hp*g + y) + x) + 4*(lane & 3)
//
// which is exactly `tin::view_byte_of(x, y, ch, Wp, Hp)` - and `ceil(W/8)` is
// `gx`, the plan's own window grid. So the raster is `4*gx` by `4*gy`:
//
//     block    W x H      4*gx x 4*gy   what shipped (VW x VH)
//        0  1080 x 1920   540 x 960     540 x 960     <- the one that agreed
//        4   540 x  960   272 x 480     270 x 480
//        8   270 x  480   136 x 240     135 x 240
//       14   135 x  240    68 x 120      67 x 120
//       22    67 x  120    36 x  60      33 x  60
//
// Two things this settles that four sessions of sweeping did not. The half
// extent is `(W+1)/2` - **34 at block 22, not 33** - and the size model
// `((W + 1) / 2 + 3) & ~3` that [[where-the-oracle-stands]] wrote down was
// right all along and was being applied to the *extent* where it is the *row
// stride*. And the trace only worked once the right kernel was open: the cubin
// holds four, `_ds_fp8` at line 7288 and `_ds` at 10856, and the stores I had
// been reading at 13393 were the non-FP8 one's.
//
// **The half-resolution tile count is a measured table, not a rounding rule.**
// Four candidate laws were tried and each fits two of the four levels:
//
//     parent H   VW    floor(VW/4)   ceil(VW/4)   what measures best
//       1080    540       135           135            135
//        540    270        67            68            *67*   r 0.994 vs 0.827
//        270    135        33            34            *33*   r 0.988 vs 0.692
//        135     67        16            17            *17*   r 0.999 vs 0.634
//         67     33         8             9             *9*   r 0.968 vs 0.111
//
// Floor above, ceil below, and no divisibility argument separates them - so it
// is written down as measured rather than derived, the way the `NR_FWAVES` and
// `NR_QUANT_MODE` tables are. `--ds-ow <VW>:<tiles>,...` overrides it, which is
// how the table was found and is how the law should be looked for next.
static std::map<size_t, size_t> g_ds_ow;
// `4*gx` by `4*gy`, straight off the store address above - **at the two deepest
// levels.** At the two shallowest the capture measures better read at the plain
// halved extent, and the difference is 2 pixels at block 4 and 1 at block 8
// against 1 and 3 at blocks 14 and 22:
//
//     block   VW = H/2   4*gx    r at VW    r at 4*gx
//        4      270       272     0.9942     0.8305
//        8      135       136     0.9877     0.6943
//       14       67        68     0.6343     0.9990
//       22       33        36     0.1108     0.9989
//
// Both halves of that table are large and unambiguous, and no rounding rule
// produces both. The store address says `4*ceil(W/8)` and the capture says
// otherwise above 135, so **one of the two is not the tensor I think it is** -
// most likely blocks 4 and 8 have a second consumer whose view differs, since
// they are also the two whose skip partners sit deepest in the decoder. Written
// down as a measured table rather than guessed at; `--ds-raster` overrides.
static bool g_host_shapes = false;
// C=32 steps whose weights keep every Swin exponent input below the upper clamp
// (the audit where the head scales are packed): they take the `<kernel>nh`
// pipeline, built with NR_EXP_NOHI. NR_EXP_NOHI=0 in the environment keeps them.
static std::set<std::pair<int, int>> g_nohi;
static std::vector<float> g_audit_bias;   // the baked position bias of the step being packed
static int g_ds_raster = -1;   // -1 = the table, 0 = always VW, 1 = always 4*gx
static void ds_raster(int gx, int gy, size_t VW, size_t VH, size_t& Wp, size_t& Hp) {
    if(g_host_shapes) { Wp=pad4(VW); Hp=pad4(VH); return; }
    const bool grid = g_ds_raster < 0 ? (VW < 135) : (g_ds_raster == 1);
    Wp = grid ? size_t(gx) * 4 : VW;
    Hp = grid ? size_t(gy) * 4 : VH;
}
// **Our own arena's half-resolution tile stride, which is not the capture's
// raster.** The `_ds` output exists in two formats: NVIDIA writes the *view*,
// whose row is `ds_raster` above, and our arena keeps everything tile-blocked
// because the consumer blocks read tile-blocked. So this one follows the
// consumer - `tiles_of` - and only the comparison converts. Conflating the two
// cost b005l0 0.9993 -> 0.9548 and b015l0 0.9995 -> 0.6624 before they were
// separated. `--ds-ow <VW>:<tiles>` overrides.
// **The arena's half-resolution tile row is the view raster's, `ds_raster / 4`,
// and `gx` is that number already.** Read out of the `_ds` kernels' own store
// address: a 4h/8h `_ds` writes `x` over `[0, pad4(ceil(W/2)))` at a row stride
// of the same, so the level below it is 68 columns wide where `135 / 2` says 67
// - see the frame plan's level ladder. Passing `gx` here rather than
// deriving it keeps the one rule in one place: `4 * gx` is the raster and `gx`
// is its tile count, at every level where the raster is padded.
static size_t ds_tiles(size_t VW, int gx) {
    if(g_host_shapes)return pad4(VW)/4;
    auto it = g_ds_ow.find(VW);
    if (it != g_ds_ow.end()) return it->second;
    // **Not `gx`.** The raster is `4*gx` and the consumer's grid is its own
    // extent's, which is one tile narrower at the two padded levels: NVIDIA
    // writes 68 columns and reads 67 of them. Tried and measured against their
    // own bytes - see the frame plan's level ladder.
    (void)gx;
    // **One rule, and it is `tiles_of`'s.** The special case below - ceil under
    // 34 columns, truncate above - was the shape of the error, not of the
    // tensor: `2*ceil(VW/8)` is 136, 68, 34, 18, 10 at the five halved levels
    // and the old rule got only the last two of those right by accident.
    if (g_tiles_win) return size_t(tiles_of(int(VW)));
    // **Truncating at every level, and the `VW <= 33` ceil that used to be here
    // was wrong.** NVIDIA's own store says so: `cc_split_swin_16h_ffwd_*_512`
    // guards its tile column with `2*ctaid.x < W/4`, which at W=33 is **8**
    // tiles - the 33rd column is not addressed by the consumer either. With the
    // ceil, the seeder laid level 6 out nine tiles to a row while `ffwd3` read
    // it eight, and b023l0 scored 0.32 seeded from a correct capture.
    return VW / 4;
}
// `view_byte_of` ties the row stride and the plane stride to one (W, H). The
// capture's *size* is the padded window grid at every level, and reading the
// whole thing padded measures worse, so the two strides are separable and
// `--ds-view` sweeps them:
//
//   0  row VW, plane VW*VH   what shipped
//   1  row PW, plane PW*PH   wholly padded
//   2  row VW, plane PW*PH   unpadded rows in a padded plane
//   3  row PW, plane VW*VH
static size_t view_byte2(size_t x, size_t y, size_t ch, size_t row, size_t plane) {
    return (ch / 16) * (plane * 16) + (y * row + x) * 16 + tin::view_sub(ch);
}
static int g_ds_view = 0;
// 0 = tiles_of(VW)*4, the truncating grid that shipped; 1 = the padded PW.
static int g_ds_grid = 0;
static bool g_grid_tiles = false;
static void ds_view_strides(size_t VW, size_t VH, size_t& row, size_t& plane,
                            size_t& nx, size_t& ny) {
    const size_t PW = pad8(VW), PH = pad8(VH);
    row   = (g_ds_view == 1 || g_ds_view == 3) ? PW : VW;
    plane = (g_ds_view == 1 || g_ds_view == 2) ? PW * PH : VW * VH;
    nx = (g_ds_view == 1) ? PW : VW;
    ny = (g_ds_view == 1) ? PH : VH;
    // 4: the capture read exactly as it shipped - unpadded raster, unpadded
    //    plane - while only our own token grid moves to `ds_tiles`. The two are
    //    independent: one describes NVIDIA's file, the other our arena.
    if (g_ds_view == 4) { row = VW; plane = VW * VH; nx = VW; ny = VH; }
}
// **FALSIFIED, 2026-09-11: the view token grid is not `M / VH`.** The slot sizes
// this project allocates say it should be - a `_ds` slot holds `tokens / 4`
// tokens over `VH` rows, which divides to exactly `ceil(VW/4)` tiles at all five
// levels where `tiles_of` truncates at four of them:
//
//     block   VW    tiles_of(VW)   M / VH / 4
//        0   540        135           135
//        4   270         67           *68*
//        8   135         33           *34*
//       14    67         16           *17*
//       22    33          8            *9*
//
// and the reading that made it look like a law - a `_ds` output's tile grid *is*
// the parent's window grid, one 4-pixel tile per 8-pixel window - closes on all
// five. **It measures worse:** the encoder board goes 41/7/2/6 to 37/7/6/6, with
// b004l0 0.9938 -> 0.9426, b008l0 -> 0.9403 and b015l0 -> 0.8885.
//
// The error is that `M` is *our own allocation*, sized from the plan's padded
// window grid - so deriving the grid from it proves only that our allocator and
// our derivation agree. The capture is the evidence and it says no. The seam
// stays open; `fswin_test --find-tile` reads the mapping off the gold instead of
// proposing one, and that is what the next attempt should use.
// **The `_ds` half extent at the two deepest levels is not settled**, and this
// records what was measured rather than leaving the next attempt to re-derive
// it. The size model `((W + 1) / 2 + 3) & ~3` says 135 halves to 68, not 67, and
// the three combinations of view raster VW and token grid TW score:
//
//     VW=67 TW=64   b014l0 r=0.632    b015l0 >=0.95   <- what ships
//     VW=68 TW=68   b014l0 r=0.9967   b015l0 0.662
//     VW=68 TW=64   b014l0 r=0.906    b015l0 0.669
//
// No combination satisfies both, so a rule is still missing - the producer's
// view extent and the consumer's tile row stride disagree at level 4 in a way
// they do not at levels 1 and 2, where the extents are even and every variant
// coincides. The next step is `fswin_test --find-tile` at 68x120, the same
// probe that settled the level-2 stride: read the mapping off the capture
// instead of proposing one. Shipping the truncating pair because it is the one
// with no regression, not because it is right.
uint32_t half_of(int px) { return uint32_t((px + 1) / 2); }   // unused; see above

// ---- the plan ------------------------------------------------------------
struct Step {
    int block, layer, div, W, H;
    long tokens;
    int gx, gy, gz, C, heads, ci, co, shifted, nin, in0, in1;
    std::string type, kernel;
    int shift_x = 100, shift_y = 100; // optional signed tile offsets
};

std::vector<Step> parse_plan(std::istream& f) {
    std::vector<Step> v;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream s(line);
        Step t{};
        s >> t.block >> t.layer >> t.div >> t.W >> t.H >> t.tokens
          >> t.gx >> t.gy >> t.gz >> t.C >> t.heads >> t.ci >> t.co
          >> t.shifted >> t.nin >> t.in0 >> t.in1 >> t.type >> t.kernel;
        s >> t.shift_x >> t.shift_y;
        v.push_back(t);
    }
    return v;
}

std::vector<Step> load_plan(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    return parse_plan(f);
}

// A produced activation is identified by the *layer* that made it, not the
// block. A split-Swin block's `Proj` takes its residual from `FfwdProj`'s
// output and the `QKVAttn` between them would overwrite it in a one-slot-per-
// block arena, which is the bug this key exists to make impossible.
int key_of(int block, int layer) { return block * 8 + layer; }
// A `_ds` layer produces three values, not one: the full-resolution block
// output that the U-Net skip partner reads, the 2x2 mean, and the projected
// half-resolution result the next block reads. Layers never reach 6, so these
// two slots cannot collide with a real one.
int skip_key(int block) { return block * 8 + 7; }
int pool_key(int block) { return block * 8 + 6; }
// `_upsample`'s two prologue buffers: the 2C -> C projection at the input's
// (half) resolution reuses `pool_key`, and this is the replicated-and-blended
// full-resolution tensor the block body then reads as its input.
int ups_key(int block) { return block * 8 + 5; }
// The pre-block's FP16 lift writes the block body's input. It needs a slot of
// its own: writing it into the block's *output* slot made the fused Swin block
// read and write one buffer in a single dispatch, and a shifted window means
// one workgroup reads tokens another is overwriting. No block is both an
// `_upsample` and the pre-block, so the number can be shared.
int lift_key(int block) { return block * 8 + 5; }

// ---- the ViT QKV output's three layouts ------------------------------------
//
// `cc_vit_1d_qkv_fp8` writes Q, K and V to three separate allocations in three
// different formats. Q is the standard activation layout - the reference's
// `act_byte` is `tin::byte_of` bit for bit - while K and V arrive at
// the attention as an MMA's B operands by `cp.async.bulk` and carry swizzles of
// their own, verified against the shipping kernel's output at 98.44% and 98.04%.
//
// The maps were measured on a 256-token capture and extended to 1024 by two
// token bits. **The general form is a 256-token group with a linear stride
// between groups**, which is what the 2560-token frame needs and which the
// 1024-token measurement is a special case of. Confirmed on the frame capture by
// the one invariant this layer supplies for free: K is unit-norm per (token,
// head), and the grouped map gives a median 0.9803 across all ten groups against
// 0.64 and 0.55 for the two wrong maps.
size_t vit_k_byte(size_t t, size_t c) {
    const size_t g = t / 256, u = t % 256;
    return g * 262144
         + (((u & 1) << 6) | (((u >> 1) & 1) << 7) | (((u >> 2) & 1) << 8)
          | (((u >> 3) & 1) << 3) | (((u >> 4) & 0xF) << 14)
          | ((c & 1) << 0) | (((c >> 1) & 1) << 4) | (((c >> 2) & 1) << 5)
          | (((c >> 3) & 1) << 1) | (((c >> 4) & 1) << 2) | (((c >> 5) & 0x1F) << 9));
}
size_t vit_v_byte(size_t t, size_t c) {
    const size_t g = t / 256, u = t % 256;
    return g * 262144
         + (((u & 1) << 0) | (((u >> 1) & 1) << 4) | (((u >> 2) & 1) << 5)
          | (((u >> 3) & 1) << 1) | (((u >> 4) & 1) << 2) | (((u >> 5) & 7) << 15)
          | ((c & 1) << 6) | (((c >> 1) & 1) << 7) | (((c >> 2) & 1) << 8)
          | (((c >> 3) & 1) << 3) | (((c >> 4) & 0x3F) << 9));
}

// cc_vit_1d_qkv_fp8 packs Q/K/V in alternating 1024-byte tiles.
// The ordinary ViT 16-row deswizzle is not valid for this record.
size_t vit_qkv_weight_byte(size_t which, size_t row, size_t k) {
    const size_t b = (k & 1) | ((k >> 1 & 1) << 4) | ((k >> 2 & 1) << 5)
        | ((k >> 3 & 1) << 1) | ((k >> 4 & 1) << 2)
        | ((row & 1) << 6) | ((row >> 1 & 1) << 7) | ((row >> 2 & 1) << 8)
        | ((row >> 3 & 1) << 3) | ((row >> 4 & 1) << 9);
    const size_t tile = (row >> 5) | ((k >> 5) << 5);
    return 128 + (3 * tile + which) * 1024 + b;
}

// ---- families ------------------------------------------------------------
enum Fam { F_NONE, F_FSWIN, F_GEMM, F_FFWD3, F_ATTN, F_DS, F_UPS, F_VATTN, F_DECUPS };

struct Shape { Fam fam; int N, K; bool act, residual, wide; };

// N and K are the *matmul* shape, which is not always the plan's ci/co: the ViT
// QKV writes 3C and the descriptor reports C. Everything here is the shape the
// per-family harness was validated at.
Shape shape_of(const Step& s) {
    const std::string& t = s.type;
    if (t.rfind("CCTinlayoutFusedSwin", 0) == 0) {
        // **`_inpview` and `_outview` are the plain kernel here.** They differ
        // from it only in reading or writing NVIDIA's level-boundary "view"
        // format - 16-channel planes, raster pixels, a permuted sub-channel
        // (tin::view_byte_of) - and *we choose the arena format*. Our `_ds`
        // writes tile-blocked, so our `_inpview` reads tile-blocked, and the
        // arithmetic is identical. The view layout matters only when replaying
        // against a capture, which is what the fswin test harness is for.
        //
        // `_ds` and `_upsample` are real work: a second, half-resolution output
        // through a learned 32->64 resample, and a second input from the U-Net
        // skip. Both are addressing, but neither is a no-op.
        // `_ds` is the plain block followed by a 2x2 mean and a learned
        // [2C][C] projection - `<layer>.resample.bin` is 64x32 at C=32 and
        // 512x256 at C=256, so the four pixels are **reduced first** and the
        // projection applied to the result, not a concatenating patch merge.
        // Three dispatches, all of them kernels that already score.
        if (s.kernel.find("_ds_") != std::string::npos) return {F_DS, 2 * s.C, s.C, false, false, false};
        // **`_upsample` is `_ds` mirrored**: a learned 2C -> C projection at the
        // input's half resolution, a 2x2 replication up to the output level, the
        // U-Net skip blended in with a per-channel gain, and then the ordinary
        // block. Prologue-then-body where `_ds` is body-then-epilogue. Three
        // dispatches, and N is the block's own width with K twice it.
        if (s.kernel.find("_upsample") != std::string::npos)
            return {F_UPS, s.co, 2 * s.co, false, false, false};
        return {F_FSWIN, s.C, s.C, false, true, false};
    }
    if (t == "CCSplitSwin16HFfwd")      return {F_FFWD3, 512, 512, false, false, false};
    if (t == "CCSplitSwin16HQKVAttn")   return {F_ATTN, 512, 512, false, false, false};
    if (t == "CCSplitSwin16HFfwdProj")  return {F_GEMM, 512, 512, false, true, false};
    if (t == "CCSplitSwin16HProj")      return {F_GEMM, 512, 512, false, true, false};
    // ProjPool is Proj plus a second, 2x2-pooled output at half resolution -
    // the pool being the mean of the f16 projection value, in the tinlayout, and
    // scored at 97.82% on its own. Only the projection branch is wired here; the
    // pooled branch has no consumer in the graph yet.
    if (t == "CCSplitSwin16HProjPool")  return {F_GEMM, 512, 512, false, true, false};
    if (t == "CCSplitSwin16HFinalHead") return {F_GEMM, 1024, 512, false, false, false};
    if (t == "CCVit1DFfnExpand")        return {F_GEMM, 4096, 1024, true,  false, true};
    if (t == "CCVit1DFfnContract")      return {F_GEMM, 1024, 4096, false, true,  true};
    if (t == "CCVit1DQKV")              return {F_GEMM, 3072, 1024, false, false, true};
    if (t == "CCVit1DProjection")       return {F_GEMM, 1024, 1024, false, true,  true};
    // The bottleneck ViT's global attention. Q, K and V are three channel ranges
    // of the single buffer `CCVit1DQKV` writes, so the shape is C in and C out
    // while the *input* it reads is 3C wide.
    if (t == "CCVit1DAttention")        return {F_VATTN, 1024, 1024, false, false, true};
    // Decoder entry projects 1024->512, nearest-upsamples 2x and adds the
    // weighted b30 projection skip. See the host launch and fresh replay.
    // **Both image adapters are the C=32 equal-width fused Swin block**, with a
    // texture sample and a lift bolted on the front of one and a projection and
    // two surface writes on the back of the other. `unpack_preblock.py`
    // and `unpack_postblock.py` write the body's tensors under the same
    // names the standard family uses, so the body needs no new code at all -
    // only a different directory and a dispatch on either side.
    if (t == "CCTinlayoutFusedPreBlockSwin1H" ||
        t == "CCTinlayoutFusedPostBlockSwin1H")
        return {F_FSWIN, 32, 32, false, true, false};
    if (t == "CCDecInputUpsample")      return {F_DECUPS, s.co, s.ci, false, true, false};
    // Still to write: the four `_upsample` variants (a prologue that combines an
    // upsampled input with a U-Net skip, arithmetic not yet decoded),
    // CCVit1DAttention, CCDecInputUpsample and the two image adapters.
    return {F_NONE};
}

// ---- push constants, one per family --------------------------------------
struct PushFSwin {
    uint32_t x_off, o_off, e_off, ct_off, qkv_off, op_off, b_off, rs_off, ars_off, s_off;
    uint32_t rsd_off, ard_off, mid_off, tiles_x, tiles_y;
    int32_t shift, shift_y;
    uint32_t pool_off, pool_tiles_x, pool_tiles_y;
};
// The push block a merged run of fused-Swin layers takes instead. Twenty-uint
// `PushFSwin` records go to the weight arena and the sync region to the
// activation arena, so the block itself is five scalars whatever the run's
// length. See `NR_PERSIST` in windows/shaders/rdna4/fswin_t.comp.
// gemmds's shear epilogue fields, appended to a fused `_ds` body's push.
struct PushDsProj { uint32_t w_off, o_off, otx, raster, crow, rows, mode, writer_rows, n, clear_x, clear_y; };
static_assert(sizeof(PushFSwin) + sizeof(PushDsProj) <= 128);
struct PushPersist {
    uint32_t layers_off, sync_off, n_layers, spin_limit, total_windows;
    // NR_PERSIST_DF: u32 index of [need[total], cons[total][4], init[n0]]
    // in the weight arena, and n0, the items with no producer in the run.
    uint32_t df_off, df_n0;
    // NR_PERSIST_DS_MASK: u32 index of the DS layer's PushDsProj, 0 = none.
    uint32_t ds_off;
};
// What the shader reads per layer: `PushFSwin` verbatim, then the layer's own
// window grid. **Consecutive layers of one level do not share one** - the grid
// is padded per axis for the shift - so the run carries one per layer and the
// flags are indexed by a running base rather than `layer * windows`.
struct PersistRec { PushFSwin p; uint32_t gx, gy, windows, flag_base; };
static_assert(sizeof(PersistRec) == 96);
static_assert(sizeof(PushFSwin) == 80);
// `fswinp<C>` and nothing else: `fswinpre32`, `fswinpreds32` and `fswinpost32`
// share the prefix and are ordinary per-layer kernels.
static bool is_persist_kern(const std::string& k) {
    return k == "fswinp32" || k == "fswinp64" || k == "fswinp128" || k == "fswinp256" ||
           k == "fswinpds64" || k == "fswinpds128" || k == "fswinpds256" ||
           k == "fswinpup64" || k == "fswinpup128" || k == "fswinpup256";
}
static bool is_plain_fswin(const std::string& k) {
    return k == "fswin32" || k == "fswin64" || k == "fswin128" || k == "fswin256";
}
struct PushGemm {
    uint32_t gd_off, x_off, r_off, o_off, d_off, w_off, g_off, M, N, K, p_off, W;
    // Downsample writer/consumer extents; gemmpool also uses rows to bound
    // its separately allocated half-resolution output.
    uint32_t otx, raster, crow, rows, ds_raster, writer_rows;
    // C>=64 DS: the real pooled extent inside the padded view; zero outside it.
    uint32_t clear_x, clear_y;
    // The activation arena's device address, for gemm1x1.comp's NR_RBDA (Windows); filled once
    // the arena exists, zero when the device has no bufferDeviceAddress.
    uint32_t act_lo, act_hi;
    // gemm1x1 NR_GEMM_REMAP_PC: workgroup order (0 = launch order).
    uint32_t remap;
};
// Every gemm1x1 kernel ("gemm*") takes a PushGemm (the pipeline loop's last branch).
static bool gemm_push_kern(const std::string& k) { return k.rfind("gemm", 0) == 0; }
struct PushQkvNorm { uint32_t src, dst, tokens, scale_off; };
struct PushDecoderUps { uint32_t src, skip, dst, gain_off, IW, OW, OH; };
struct PushRepack { uint32_t src, dst, W, H, C, inverse; };
struct PushFfwd3 { uint32_t x_off, o_off, a_off, q0_off, q2_off, M, C; };
struct PushPool { uint32_t x_off, o_off, tiles_x, tiles_y, otiles_x, raster, crow; };
struct PushUpsView { uint32_t x_off, o_off, M, C, W, H, RW, RH; };
struct PushUps {
    uint32_t p_off, s_off, o_off, g_off, tiles_x, tiles_y, itiles_x, itiles_y, mode;
    uint32_t stiles_x;   // the skip's own row stride in tiles; 0 = the output's
};
// act_lo/act_hi: the activation arena's device address, for vit_attn.comp's NR_VBDA (Windows);
// filled once the arena exists, zero when the device has no bufferDeviceAddress.
struct PushVAttn { uint32_t x_off, o_off, tokens, s_off, mode, act_lo, act_hi; };
struct PushImgIn {
    uint32_t o_off, w_off, tiles_x, W, H, slot;
    float gate, bias_x, bias_y, scale_x, scale_y, post_x, post_y;
    uint32_t uv_swap, seed;
    float s434, aux0, aux1, gate0, gate1, noise, konst, amp_b;
    uint32_t bslot, lift_perm, row_shift, lift_tiled;
    uint32_t lift_t, lift_neg;
    uint32_t source_W, source_H;
};
static_assert(sizeof(PushImgIn)==124);
struct PushImgOut { uint32_t x_off, w_off, tiles_x, W, H; float gate, src; uint32_t sw, sh, base0, base1; float blend, nr_intensity; };
static_assert(sizeof(PushImgOut)==52);
struct PushImageTail { uint32_t w_off; float intensity; };
static_assert(sizeof(PushFSwin)+sizeof(PushUps)+sizeof(PushImageTail)==128);
struct PushPreImage {
    uint32_t lift,source_W,source_H,seed;
    float style,tone,structure,skin,other,noise,constant;
    uint32_t noise_off;   // NR_NOISE_FIELD: u32 index of the precomputed noise features, 0: none
};
static_assert(sizeof(PushPreImage)==48 && sizeof(PushFSwin)+sizeof(PushPreImage)==128);
struct PushAttn  {
    uint32_t x_off, o_off, w_off, b_off, s_off, C, wins_x, tiles_x, tiles_y;
    int32_t shift, shift_y;
};

// ---- CCSplitSwin16HFfwd's three matrices ---------------------------------
// The layer's 524,288-byte record holds A, Q0 and Q2 under **bit-scattered**
// index maps - A's k bits land in byte bits 0,4,5,1,2,14,15,16,17 and so on.
// Measured against the shipping kernel; these are copied verbatim from the
// ffwd3 reference and its test harness. Unpacking them into
// canonical [N][K] and then tile-blocked is load-time work, which is where
// decision 4 puts it.
constexpr size_t FF_GROUPS = 8, FF_R = 64, FF_J = 256, FF_OUT = 64, FF_K = 512;
size_t ff_a(int g, int row, int k) {
    return size_t((k & 1) << 0) | ((k >> 1 & 1) << 4) | ((k >> 2 & 1) << 5)
         | ((k >> 3 & 1) << 1) | ((k >> 4 & 1) << 2)
         | ((k >> 5 & 1) << 14) | ((k >> 6 & 1) << 15)
         | ((k >> 7 & 1) << 16) | ((k >> 8 & 1) << 17)
         | ((row & 1) << 3) | ((row >> 1 & 1) << 6) | ((row >> 2 & 1) << 7)
         | ((row >> 3 & 1) << 8) | ((row >> 4 & 1) << 9) | ((row >> 5 & 1) << 10)
         | ((g & 1) << 11) | ((g >> 1 & 1) << 12) | ((g >> 2 & 1) << 13);
}
size_t ff_q0(int g, int j, int row) {
    return size_t(262144) + size_t(g) * 16384
         + (((row & 1) << 1) | ((row >> 1 & 1) << 0) | ((row >> 2 & 1) << 4)
         | ((row >> 3 & 1) << 5) | ((row >> 4 & 1) << 2) | ((row >> 5 & 1) << 13)
         | ((j & 1) << 6) | ((j >> 1 & 1) << 3) | ((j >> 2 & 1) << 9)
         | ((j >> 3 & 1) << 7) | ((j >> 4 & 1) << 8) | ((j >> 5 & 1) << 10)
         | ((j >> 6 & 1) << 11) | ((j >> 7 & 1) << 12));
}
size_t ff_q2(int g, int n, int j) {
    return size_t(393216) + size_t(g) * 16384
         + (((j & 1) << 0) | ((j >> 1 & 1) << 1) | ((j >> 2 & 1) << 2)
         | ((j >> 3 & 1) << 4) | ((j >> 4 & 1) << 5) | ((j >> 5 & 1) << 11)
         | ((j >> 6 & 1) << 12) | ((j >> 7 & 1) << 13)
         | ((n & 1) << 6) | ((n >> 1 & 1) << 7) | ((n >> 2 & 1) << 8)
         | ((n >> 3 & 1) << 3) | ((n >> 4 & 1) << 9) | ((n >> 5 & 1) << 10));
}

std::vector<float> load_e4m3(const std::string& p, size_t n) {
    const std::vector<uint8_t> b = slurp(p);
    if (b.size() < n) {
        std::fprintf(stderr, "%s: %zu bytes, need %zu\n", p.c_str(), b.size(), n);
        throw std::runtime_error("FP8 weight file is missing or truncated: " + p);
    }
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = tin::e4m3_to_f(b[i]);
    return v;
}
std::vector<float> load_f16(const std::string& p) {
    const std::vector<uint8_t> b = slurp(p);
    std::vector<float> v(b.size() / 2);
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = tin::f16_to_f(uint16_t(b[2 * i] | (b[2 * i + 1] << 8)));
    return v;
}
// **How far apart, in encodings.** "91% of bytes agree" cannot tell a layer
// whose values land one rounding step away from one that computes something
// else in 9% of its output, and those two want completely different work. e4m3
// codes are monotonic within a sign, so the distance between two encodings is
// a subtraction; across signs it is the distance through zero.
inline int e4m3_codes_apart(uint8_t a, uint8_t b) {
    const int ma = a & 0x7F, mb = b & 0x7F;
    if (!ma && !mb) return 0;                       // +0 and -0 are the same value
    if ((a ^ b) & 0x80) return ma + mb;
    return ma > mb ? ma - mb : mb - ma;
}
std::vector<uint8_t> to_e4m3(const std::vector<float>& v) {
    std::vector<uint8_t> o(v.size());
    for (size_t i = 0; i < v.size(); ++i) o[i] = tin::f_to_e4m3(v[i]);
    return o;
}

// ---- 98% of a nine-second build was one nested loop -------------------------
//
// The five big weight matrices are stored as e4m3 bytes and were read with
// `to_e4m3(load_e4m3(path, n))`: byte -> float -> byte. The second half of that
// is `tin::f_to_e4m3`, which finds the nearest encoding by **scanning all 256 of
// them**, so every weight costs 256 float comparisons. Over the ~140 MB of
// matrices in this network that is 8.99 s of a 9.15 s build, measured with
// nr::PhaseTimer (960x640, Mesa 26.2.2).
//
// The composition depends only on the input byte, so it is a 256-entry table -
// and the table is **built by calling the same two functions**, which is what
// makes this byte-identical rather than merely equivalent. Two of its entries
// are not the identity and both are load-bearing:
//
//   0x00 -> 0x80   +0 and -0 are both exact, so the search ties, and its
//                  tie-break keeps the later even index, which is -0.
//   0xFF -> 0x7F   both NaN encodings read as NaN and f_to_e4m3 answers 0x7F.
//
// Anything that changes tin::f_to_e4m3 or tin::e4m3_to_f changes this with it,
// automatically, because the table is their composition and nothing else.
const std::array<uint8_t, 256>& e4m3_requantise_table() {
    static const std::array<uint8_t, 256> table = [] {
        std::array<uint8_t, 256> t{};
        for (int i = 0; i < 256; ++i) t[size_t(i)] = tin::f_to_e4m3(tin::e4m3_to_f(uint8_t(i)));
        return t;
    }();
    return table;
}

// `to_e4m3(load_e4m3(p, n))` with the round trip collapsed.
std::vector<uint8_t> load_e4m3_requantised(const std::string& p, size_t n) {
    const std::vector<uint8_t> b = slurp(p);
    if (b.size() < n) {
        std::fprintf(stderr, "%s: %zu bytes, need %zu\n", p.c_str(), b.size(), n);
        throw std::runtime_error("FP8 weight file is missing or truncated: " + p);
    }
    const auto& table = e4m3_requantise_table();
    std::vector<uint8_t> o(n);
    for (size_t i = 0; i < n; ++i) o[i] = table[b[i]];
    return o;
}

// NVIDIA store the position bias in their own MMA C-fragment order; the host
// puts it back into [i][j] once, at load time, exactly as attn_test does.
std::vector<float> deswizzle_bias(const std::vector<float>& in) {
    std::vector<float> out(64 * 64);
    for (size_t i = 0; i < 64; ++i)
        for (size_t j = 0; j < 64; ++j) {
            const size_t lane = 4 * ((i % 16) % 8) + ((j % 16) % 8) / 2;
            const size_t sl = (j % 2) | (((i % 16) / 8) << 1) | (((j % 16) / 8) << 2);
            out[i * 64 + j] = in[(4 * (i / 16) + (j / 16)) * 256 + lane * 8 + sl];
        }
    return out;
}

struct Disp {
    const Step* s;
    std::string kern;
    uint32_t gx, gy, gz;
    std::vector<uint8_t> push;
    // The lowered steps this dispatch executes, as positions in `run`: one step,
    // or the whole range of a merged persistent run. What arena reuse reads.
    int lo = -1, hi = -1;
};

// The activation arena's slot table, and which lowered steps ask for each slot.
//
// Every offset a dispatch is given comes from here, so a record kept while the
// dispatches are being lowered is the complete list of what each one touches -
// reads and writes alike, side buffers and the U-Net's long edges included,
// with nothing modelled by hand. That is what arena reuse needs and what the
// hand-written liveness it replaces got wrong: it knew the chain slots, not the
// side buffers (see the comment at the allocation below).
struct ValueOffsets {
    std::map<int, size_t> m;
    int step = -1;                                // the step being lowered, or -1
    std::map<int, std::pair<int, int>> used;      // key -> first, last step
    size_t& operator[](int k) { note(k); return m[k]; }
    size_t& at(int k) { note(k); return m.at(k); }
    size_t count(int k) const { return m.count(k); }
    size_t size() const { return m.size(); }
    void note(int k) {
        if (step < 0) return;
        auto it = used.find(k);
        if (it == used.end()) used[k] = {step, step};
        else it->second = {std::min(it->second.first, step), std::max(it->second.second, step)};
    }
};

}  // namespace


// ---- the network as an object -----------------------------------------------
//
// Everything in `build` used to be the top of `main`. It is a struct now for one
// reason: **an injector has to run this graph on the host application's own
// VkDevice**, not on a device of its own, and it has no command line. `build`
// still takes an argv - every tuning flag stays reachable - and the CLI below is
// unchanged, because it aliases these members by reference.
//
// See `nrvk::Context::adopt`. With a fresh `ctx` this is exactly what `main`
// did; with an adopted one the same 171 dispatches record into the host's own
// command buffer.
// The weight arena on the host, in pieces: 4 MB chunks, so no allocation is larger than that (a 32-bit game's
// address space is small and fragmented). A chunk nothing was written to holds zeros without memory (the noise
// field's region), and the arena-layout probe only counts - it needs the offsets, not the bytes.
class WeightBlob {
public:
    static constexpr size_t kChunk = size_t(4) << 20;
    explicit WeightBlob(bool count_only) : count_only_(count_only) {}
    size_t size() const { return size_; }
    // Grow to `n` (zeros). Never shrinks.
    void grow(size_t n) { if (n > size_) size_ = n; }
    // Append `n` bytes at `align`; returns their offset.
    size_t append(const void* p, size_t n, size_t align) {
        const size_t off = (size_ + align - 1) / align * align;
        size_ = off;
        write(off, p, n);
        return off;
    }
    size_t append_zeros(size_t n, size_t align) {
        const size_t off = (size_ + align - 1) / align * align;
        size_ = off + n;
        return off;
    }
    void write(size_t off, const void* src, size_t n) {
        if (off + n > size_) size_ = off + n;
        if (count_only_) return;
        auto b = static_cast<const uint8_t*>(src);
        while (n) {
            const size_t c = off / kChunk, o = off % kChunk, k = std::min(n, kChunk - o);
            if (c >= chunks_.size()) chunks_.resize(c + 1);
            if (!chunks_[c]) chunks_[c].reset(new uint8_t[kChunk]());
            std::memcpy(chunks_[c].get() + o, b, k);
            off += k; b += k; n -= k;
        }
    }
    void read(size_t off, void* dst, size_t n) const {
        auto d = static_cast<uint8_t*>(dst);
        while (n) {
            const size_t c = off / kChunk, o = off % kChunk, k = std::min(n, kChunk - o);
            if (c < chunks_.size() && chunks_[c]) std::memcpy(d, chunks_[c].get() + o, k);
            else std::memset(d, 0, k);
            off += k; d += k; n -= k;
        }
    }
    void clear() { chunks_.clear(); chunks_.shrink_to_fit(); size_ = 0; }
private:
    bool count_only_;
    size_t size_ = 0;
    std::vector<std::unique_ptr<uint8_t[]>> chunks_;
};

struct NrSession {
    nrvk::Context ctx;
    nrvk::Buffer act{}, wgt{};
    nrvk::Context::Image tex_in{}, surf0{}, surf1{};
    std::map<std::string, nrvk::Kernel> kern;
    std::vector<Disp> disp;
    std::vector<Step> plan;
    ValueOffsets voff;
    std::map<int, size_t> vsize;
    // Bytes a slot's readers reach past its size, which must stay zero.
    std::map<int, size_t> overread;
    // The arena-reuse diagnostics below (NR_ARENA_UNWRITTEN, NR_POISON_KEYS,
    // NR_ARENA_GAP / NR_POISON_GAPS): where the values end, and the guard gaps.
    size_t values_end = 0;
    std::vector<std::pair<int, size_t>> gaps;   // value before it, gap offset
    // Arena reuse (`--reuse`): a first build with `arena_probe` set stops after
    // lowering and merging, before anything touches the device, and hands back
    // a layout in which values whose lifetimes do not overlap share memory.
    bool arena_probe = false;
    std::map<int, size_t> arena_layout;
    size_t arena_values = 0;
    std::vector<std::pair<int, int>> arena_disp_steps, arena_expect;
    std::map<std::string, int> missing;
    nrvk::Runner runner;
    std::vector<nrvk::Runner::Step> steps;
    uint32_t in_w = 0, in_h = 0, out_w = 0, out_h = 0;
    uint32_t math_profile = 0;
    // Byte offsets of every merged run's error word, read back after a run.
    std::vector<uint32_t> persist_err;
    // NR_TCHAIN, kept per session for the runtime's in-game check (nr_runtime.cpp
    // Impl::Check; the g_tc_* globals belong to whichever build ran last): the
    // counter region (arena words from tc_base), its error words, and every
    // counter a table names with the units a frame adds to it - after n whole
    // frames it holds need * n.
    struct TcExpect { uint32_t word, need, pair; };
    uint32_t tc_base = 0, tc_words = 0;
    std::vector<uint32_t> tc_err;          // words in the region, one per chained pair
    std::vector<TcExpect> tc_expect;       // word in the region, need, pair
    std::vector<std::string> tc_pair;      // "producer -> consumer" per chained pair
    std::vector<const Step*> run;
    std::map<int, size_t> reader_tokens;   // block -> tokens of whoever reads it
    std::map<int, int> last_layer;   // block -> its final layer
    std::function<int(const Step&)> x_src;
    bool host_boundary = false;
    const char* seed_path = nullptr;
    const char* score_dir = nullptr;
    // 0 built, 1 a fatal setup error, 2 a flag asked for a report and nothing
    // should run.
    int build(int argc, char** argv, const std::vector<Step>* prepared_plan = nullptr);
};

// The layout globals above (g_chain_*, g_tc_*) are one copy for every build, and
// a game can build two networks at once (the D3D12 and the Vulkan session each
// have a background build thread): one build at a time. A session keeps what it
// needs of them (persist_err, tc_*), and recording reads only that. Recursive:
// a build runs its arena-layout probe build on the same thread.
static std::recursive_mutex g_build_lock;

int NrSession::build(int argc, char** argv, const std::vector<Step>* prepared_plan) {
    std::lock_guard<std::recursive_mutex> one_build(g_build_lock);
    // A build lays out its own persistent sync words; an epoch word left from a
    // previous build (another extent) would point into the old layout.
    g_chain_epoch_word = 0xFFFFFFFFu;
    // Where the build's seconds go. Closed phase by phase and logged once at the
    // end through nr::logf, which is the only route that reaches a game's log.
    nr::PhaseTimer timer;
    const char* intensity_text=arg(argc,argv,"--nr-intensity","1");
    char* intensity_end=nullptr;
    const float nr_intensity=std::strtof(intensity_text,&intensity_end);
    if (intensity_end==intensity_text || *intensity_end || !std::isfinite(nr_intensity) || nr_intensity<0 || nr_intensity>2) {
        std::fprintf(stderr,"--nr-intensity (Overall Intensity) must be finite and within 0..2\n");return 1;
    }
    if(nr_intensity!=1 && flag(argc,argv,"--legacy-boundary")) {
        std::fprintf(stderr,"non-default Overall Intensity requires the first-frame host boundary\n");return 1;
    }
    const std::string auto_mask_text=arg(argc,argv,"--auto-mask","1");
    if(auto_mask_text!="0" && auto_mask_text!="1") {
        std::fprintf(stderr,"--auto-mask must be 0 or 1\n");return 1;
    }
    const std::string plan_path =
        arg(argc, argv, "--plan", "windows/src/core/nr_frame.txt");
    const std::string unp = arg(argc, argv, "--unpacked", "artifacts");
    if (const char* pack = arg(argc, argv, "--model-pack")) {
        std::string why;
        if (!open_model_pack(pack, unp, &why)) { std::fprintf(stderr, "%s\n", why.c_str()); return 1; }
        std::printf("model pack: %s (%zu entries)\n", pack, g_pack.index.size());
    }
    const std::string accumulation=arg(argc,argv,"--accumulation","fp32");
    if (accumulation!="fp32" && accumulation!="round-fp16") {
        std::fprintf(stderr,"--accumulation must be fp32 or round-fp16\n");return 1;
    }
    std::string spv_dir = arg(argc, argv, "--spv-dir", "build");
    if (!nr::binary::directory().empty()) spv_dir = nr::binary::directory() + "/shaders";
    if(accumulation=="round-fp16") spv_dir += "/round-fp16";
    std::ifstream policy_file(spv_dir+"/accumulation.txt");
    std::string built_policy;
    policy_file >> built_policy;
    if(built_policy!=accumulation) {
        std::fprintf(stderr,"missing or mismatched %s pipelines in %s; run windows/build/build_network.py\n",
                     accumulation.c_str(),spv_dir.c_str());return 1;
    }
    std::printf("accumulation: %s (FP32 hardware%s)\n",accumulation.c_str(),
                accumulation=="round-fp16"?", explicit FP16 rounding at MMA boundaries":"");
    // ---- the activation arena's coherence, and the barrier that depends on it
    // Every block aliasing the activation arena is declared `coherent` under
    // `-DNR_COHERENT_ACT=1`, which on this driver puts `scope:SCOPE_DEV` on its
    // loads and stores: they bypass L0 and reach L2, the coherence point. With
    // every activation load at device scope the memory half of the
    // inter-dispatch barrier has nothing left to do, and `--coherent-barriers`
    // records only the execution dependency. The SPV directory says which it
    // is, because a runner and a build that disagree here compute stale data.
    bool coherent_act = false;
    {
        std::ifstream coh(spv_dir + "/coherent-act.txt");
        std::string t;
        coh >> t;
        coherent_act = (t == "1");
    }
    // This is a weight-storage contract, independent of MMA accumulation.
    // Snapshots predating the marker and standalone shaders use FP32 bias.
    std::string swin_bias_storage = "fp32";
    const std::string bias_marker = spv_dir + "/swin-bias-storage.txt";
    if (std::filesystem::exists(bias_marker)) {
        std::ifstream file(bias_marker);
        if (!(file >> swin_bias_storage) ||
            (swin_bias_storage != "fp16" && swin_bias_storage != "fp32")) {
            std::fprintf(stderr, "invalid Swin bias storage marker: %s\n", bias_marker.c_str());
            return 1;
        }
    }
    const bool swin_bias_f16 = swin_bias_storage == "fp16";
    std::printf("Swin position bias storage: %s; FP32 logit addition\n", swin_bias_storage.c_str());
    const uint32_t repeats = uint32_t(std::atoi(arg(argc, argv, "--repeats", "1")));
    const uint32_t warm = uint32_t(std::atoi(arg(argc, argv, "--warmup", "0")));
    // ---- scoring against NVIDIA's own capture ------------------------------
    // Until this existed the graph had **no check that could fail**. It ran, it
    // filled every output, and `ffwd3` was writing canonical [token][channel]
    // into an arena every consumer reads tile-blocked - sixteen layers feeding
    // the next sixteen a transposed mess, with nothing in the harness able to
    // notice. A frame time is not a result.
    //
    //   --seed  <cap>/b000l0.bin   start from the capture's own block 0 output
    //   --score <cap>              compare every layer that has a capture
    //
    // The capture is in NVIDIA's tinlayout; the arena is tile-blocked canonical.
    // Both are converted to plain [token][channel] before comparing, which is
    // what the fswin test harness does for a single layer.
    seed_path = arg(argc, argv, "--seed");
    score_dir = arg(argc, argv, "--score");
    if (!score_dir && (seed_path || arg(argc,argv,"--seed-dir") ||
                       arg(argc,argv,"--only") || flag(argc,argv,"--oracle") ||
                       flag(argc,argv,"--hash-values"))) {
        std::fprintf(stderr,"--seed, --seed-dir, --only, --oracle and --hash-values require --score DIR\n");
        return 1;
    }
    if (flag(argc,argv,"--reverse")) {
        std::fprintf(stderr,"--reverse is unsupported; use --oracle --score DIR --seed-dir DIR\n");
        return 1;
    }

    g_tiles_up = flag(argc, argv, "--tiles-up");
    g_tiles_win = flag(argc, argv, "--tiles-win");
    g_ds_view = std::atoi(arg(argc, argv, "--ds-view", "0"));
    g_ds_grid = std::atoi(arg(argc, argv, "--ds-grid", "0"));
    g_grid_tiles = flag(argc, argv, "--grid-tiles");
    g_ds_raster = std::atoi(arg(argc, argv, "--ds-raster", "-1"));
    if (const char* ow = arg(argc, argv, "--ds-ow")) {
        std::istringstream is(ow);
        std::string tok;
        while (std::getline(is, tok, ',')) {
            const size_t colon = tok.find(':');
            if (colon != std::string::npos)
                g_ds_ow[size_t(std::stoul(tok.substr(0, colon)))] =
                    size_t(std::stoul(tok.substr(colon + 1)));
        }
    }
    host_boundary = !flag(argc, argv, "--legacy-boundary");
    if (arg(argc, argv, "--ds-only") && host_boundary &&
        !flag(argc, argv, "--legacy-ds-quant")) {
        std::fprintf(stderr, "--ds-only seeds an FP8 skip, which cannot recover the original half pooling input; use --only with --seed-dir to test the full DS layer, or --legacy-ds-quant for the old diagnostic\n");
        return 1;
    }
    if (flag(argc, argv, "--host-boundary") && !host_boundary) {
        std::fprintf(stderr, "--host-boundary and --legacy-boundary conflict\n"); return 1;
    }
    g_host_shapes = host_boundary;
    g_nohi.clear(); g_audit_bias.clear();
    plan = prepared_plan ? *prepared_plan : load_plan(plan_path);
    std::printf("%zu steps from %s\n", plan.size(), plan_path.c_str());

    for (const Step& s : plan)
        last_layer[s.block] = std::max(last_layer[s.block], s.layer);

    // ---- what runs ---------------------------------------------------------
    for (const Step& s : plan) {
        if (shape_of(s).fam != F_NONE) run.push_back(&s);
        else ++missing[s.type];
    }
    std::printf("%zu of %zu layers have a kernel\n", run.size(), plan.size());
    if (flag(argc, argv, "--report")) {
        for (const auto& kv : missing)
            std::printf("  missing %-34s %3d\n", kv.first.c_str(), kv.second);
        return 2;
    }

    // ---- where each layer's output lives ------------------------------------
    // A value's size is its own token count times its own output width, which is
    // the matmul N and not the plan's `co`: the ViT QKV writes 3C where the
    // descriptor reports C.
    // **A `_ds` layer's output is sized by its *consumer*, not by a quarter of
    // its own tokens.** Read straight off the capture file sizes, which are the
    // only evidence here:
    //
    //     block   tokens/4   capture   the block that reads it
    //        0     518400    522240    b001  522240
    //        4     130560    130560    b005  130560
    //        8      32640     32640    b009   32640
    //       14       8160    *8640*    b015    8640
    //       22       2160    *2560*    b023    2560
    //
    // The two that agree are the even halvings; the three that do not are three
    // of the six layers the encoder oracle still fails, and b022l0's short slot
    // is why `CCSplitSwin16HFfwd` at b023l0 reads 2160 tokens where NVIDIA's
    // kernel read 2560. See [[the-ds-seam-is-not-a-tile-stride]], which ruled
    // out the tile stride by measurement before this turned up.
    for (const Step& s : plan)
        for (int in : {s.in0, s.in1})
            if (in >= 0 && in != s.block && !reader_tokens.count(in))
                reader_tokens[in] = size_t(s.tokens);
    auto ds_out_tokens = [&](const Step& s, size_t fallback) {
        auto it = reader_tokens.find(s.block);
        return it == reader_tokens.end() ? fallback : it->second;
    };
    // The window grid of another block, for the U-Net skip edges: a consumer
    // has to know the row stride its *producer* wrote with, and `gx` is the
    // only place the plan records it.
    std::map<int, int> block_gx;
    for (const Step& s : plan)
        if (!block_gx.count(s.block)) block_gx[s.block] = s.gx;

    auto up_extent = [&](const Step& s) {
        std::pair<int,int> extent{2*s.W,2*s.H};
        if(host_boundary)for(const Step& enc:plan)if(enc.block==s.in1) {
            extent={enc.W,enc.H};
            if(s.co==32)extent={int(pad8(enc.W)),int(pad8(enc.H))};
            break;
        }
        return extent;
    };
    vsize.clear();
    overread.clear();
    for (const Step& s : plan) {
        const Shape sh = shape_of(s);
        const int k = key_of(s.block, s.layer);
        if (sh.fam == F_UPS) {
            // **The plan's extent and token count are the layer's INPUT level;
            // its grid is the OUTPUT level.** b066 carries px 480x270 and
            // tokens 130560 with a 68x120 window grid, which is 544x960 - one
            // level up. So the output token count is the grid's, gx*8 by gy*8,
            // and the output extent is twice the plan's.
            const auto extent=up_extent(s);
            const size_t OT = host_boundary ? pad8(extent.first)*pad8(extent.second)
                                           : size_t(s.gx)*8*size_t(s.gy)*8;
            vsize[pool_key(s.block)] = std::max(vsize[pool_key(s.block)],
                                                size_t(s.tokens) * size_t(s.co) * (host_boundary ? 2 : 1));
            vsize[ups_key(s.block)] = std::max(vsize[ups_key(s.block)], OT * size_t(s.co) * (host_boundary && s.co == 32 ? 2 : 1));
            vsize[k] = std::max(vsize[k], OT * size_t(s.co));
            continue;
        }
        if (sh.fam == F_DS) {
            // tokens is the *input* count; the projected output is a quarter of
            // it and twice as wide, which is exactly the next block's shape.
            vsize[skip_key(s.block)] = std::max(vsize[skip_key(s.block)],
                                                size_t(s.tokens) * size_t(s.C));
            vsize[pool_key(s.block)] = std::max(vsize[pool_key(s.block)],
                                                size_t(s.tokens) / 4 * size_t(s.C));
            vsize[k] = std::max(vsize[k],
                                ds_out_tokens(s, size_t(s.tokens) / 4) * size_t(sh.N));
            // The pooled grid is stored padded to its reader's window grid, and
            // the pooling writes only the real pixels: the padding columns and
            // rows (and, where the reader has more tokens, the tail) are read as
            // zero - NVIDIA's own capture has them at exactly zero. See
            // `overread` below.
            overread[k] = std::max(overread[k], vsize[k]);
            continue;
        }
        // **The pre-block is a `_ds` the size model does not see as one.** Its
        // tail is a bare 2x2 mean with no resample - the extra region in its
        // record is the FP16 lift - so `shape_of` gives it the plain family and
        // neither its pool output nor its lift input was ever sized. Both fell
        // through to `std::map`'s default and sat at **offset 0**, on top of the
        // block's own output. See the aliasing check below the allocator.
        if (s.type == "CCTinlayoutFusedPostBlockSwin1H") {
            // Four times the plan row's token count: this layer doubles.
            // **`sh.N`, not `s.C`.** The plan's C column for this row is 3 -
            // the image's colour channels - and the block's working width is 32.
            if(host_boundary)vsize[pool_key(s.block)]=size_t(s.tokens)*size_t(sh.N);
            vsize[ups_key(s.block)] = std::max(vsize[ups_key(s.block)],
                                               size_t(s.tokens) * 4 * size_t(sh.N) * (host_boundary ? 2 : 1));
            vsize[k] = std::max(vsize[k], size_t(s.tokens) * 4 * size_t(sh.N)
                                        * (host_boundary ? 2 : 1));
        }
        if (s.type == "CCTinlayoutFusedPreBlockSwin1H") {
            // **`sh.N`, not `s.C`.** The plan's C column for this row is 3 -
            // the image's colour channels - and the lift writes the block's
            // working width, 32. Sized by `s.C` the slot was 6.2 MB and the
            // lift wrote 66.4, trampling 60 MB of whatever the allocator had
            // put after it: the adapter's own output came back at rms 242 with
            // 448s in it, which is what reading past a slot looks like.
            vsize[lift_key(s.block)] = std::max(vsize[lift_key(s.block)],
                                                size_t(s.tokens) * size_t(sh.N) * (host_boundary ? 2 : 1));
            vsize[skip_key(s.block)] = std::max(vsize[skip_key(s.block)],
                                                size_t(s.tokens) * size_t(sh.N));
            // **And its own output slot is the *pooled* one, a quarter of the
            // tokens.** Sized at the full resolution it was 66355200 B against
            // a 16711680 B capture, so `--seed-dir` rejected it every run and
            // b001l0 - which reads it - has been scoring 0.55 against an input
            // that was never seeded. The generic rule below cannot know: this
            // layer is a `_ds` that `shape_of` reports as a plain block.
            vsize[k] = std::max(vsize[k],
                                ds_out_tokens(s, size_t(s.tokens) / 4) * size_t(sh.N));
            continue;
        }
        if (host_boundary && sh.fam == F_DECUPS) {
            for (const Step& prev : plan) if (prev.block == s.in0 && prev.layer == 4)
                vsize[pool_key(s.block)] = align(size_t(prev.W)*prev.H,32)*512*2;
        }
        if (s.type == "CCVit1DQKV") vsize[pool_key(s.block)] = align(size_t(s.tokens),32)*3072*2;
        if (host_boundary && s.type == "CCSplitSwin16HProjPool") {
            vsize[pool_key(s.block)] = pad4((s.H + 1) / 2) * pad4((s.W + 1) / 2) * 512;
            // pad4's extra rows and columns are never written and are read as zero.
            overread[pool_key(s.block)] = std::max(overread[pool_key(s.block)], vsize[pool_key(s.block)]);
        }
        // **FinalHead reads more rows than the pool slot holds.** Its GEMM runs
        // M = the plan's token count (768 at 1080p) over the pooled grid, which
        // has pad4(W/2)*pad4(H/2) rows (640): the last 128 rows are whatever
        // lies past the slot. In the one-value-one-slot layout that is bytes no
        // dispatch ever writes, so zero from the build's arena fill, every
        // frame; arena reuse keeps it that way by reserving those bytes for the
        // slot alone (`overread`).
        //
        // `overread` is every slot with bytes that are read but never written.
        // Measured 2026-09-24 by filling the value area with a pattern, running
        // a frame, and poisoning each unwritten range in turn at 1920x1080,
        // 2560x1440 and 3818x1998: this pool slot and the DS output that feeds
        // the C512 level are the only ones whose unwritten bytes change the
        // picture.
        if (host_boundary && s.type == "CCSplitSwin16HFinalHead")
            overread[pool_key(s.block)] = std::max(overread[pool_key(s.block)], size_t(s.tokens) * 512);
        if (host_boundary && s.block == 31 && s.layer == 0)
            vsize[lift_key(31)] = align(size_t(s.tokens),32) * 1024;
        if (host_boundary && s.block == 38 && s.layer == 4)
            vsize[ups_key(38)] = align(size_t(s.tokens),32) * 1024;
        // A partial ViT token tile still spans every channel group. M*C bytes
        // is insufficient for tile-blocked M when M is not 16-aligned. PTX
        // rounds its ViT domain to 32; preserve zero padding in separate slots.
        const size_t stored_tokens = s.type.rfind("CCVit1D",0)==0
            ? align(size_t(s.tokens),32) : size_t(s.tokens);
        size_t n = stored_tokens * size_t(sh.fam == F_NONE ? s.co : sh.N);
        vsize[k] = std::max(vsize[k], n);
    }
    // Intra-block edges come from the family's own structure; cross-block edges
    // name a block and mean its last layer's output.
    auto blk_out = [&](int b) { return key_of(b, last_layer.count(b) ? last_layer[b] : 0); };
    // **`this`, not a copy.** As a `[&]` lambda this dangled the moment `build`
    // returned - the CLI calls it while seeding - and as a `[=]` one it copied
    // a `blk_out` that still referenced the dead `last_layer`. Both compile and
    // the second crashes at run time, which is why `last_layer` is a member.
    x_src = [this](const Step& s) {
        if (host_boundary && s.type == "CCSplitSwin16HFinalHead") return pool_key(s.block);
        if (host_boundary && s.block == 31 && s.layer == 0) return lift_key(31);
        if (host_boundary && s.block == 39) return ups_key(38);
        if (s.layer > 0) return key_of(s.block, s.layer - 1);
        const int b = s.in0 >= 0 ? s.in0 : s.block;
        return key_of(b, last_layer.count(b) ? last_layer.at(b) : 0);
    };
    // Descriptor +0x90 lists intra-block layer sources; +0xa8 lists
    // block-input sources. The old replay reused the immediate predecessor
    // for both pointers, so high agreement with it did not validate an edge.
    auto r_src = [&](const Step& s) {
        if (flag(argc, argv, "--legacy-replay-edges"))
            return s.layer > 0 ? key_of(s.block, s.layer - 1) : blk_out(s.in0);
        if (s.type == "CCSplitSwin16HProj" || s.type == "CCSplitSwin16HProjPool")
            return key_of(s.block, 1); // descriptor sources [2, 1]
        if (s.type == "CCVit1DProjection")
            return key_of(s.block, 1); // descriptor sources [3, 1]
        if (s.type == "CCSplitSwin16HFfwdProj" || s.type == "CCVit1DFfnContract") {
            for (const auto& first : plan)
                if (first.block == s.block && first.layer == 0) return x_src(first);
        }
        return s.layer > 1 ? key_of(s.block, s.layer - 2) : blk_out(s.in0);
    };

    // **How many of the 176 inter-dispatch barriers are actually needed?**
    //
    // `nrvk::Runner` emits one after every dispatch unconditionally, and they
    // measure ~10-13 us each - 2.33 ms of a 9.2 ms frame at 1080p, 1.87 of 5.1
    // at 720p. Vulkan lets consecutive dispatches overlap unless a barrier
    // orders them, so every unnecessary one is pure loss. The graph already
    // knows every step's input slots (`x_src`, `r_src`, `in0`, `in1`) and its
    // output (`key_of(block, layer)`), so the question is arithmetic rather
    // than a guess. `--barrier-report` prints it and exits nothing.
    if (flag(argc, argv, "--barrier-report")) {
        auto writes = [&](const Step& s) {
            std::set<int> w{key_of(s.block, s.layer)};
            return w;
        };
        auto reads = [&](const Step& s) {
            std::set<int> r{x_src(s)};
            const Shape sh = shape_of(s);
            if (sh.fam != F_NONE && sh.residual) r.insert(r_src(s));
            for (int in : {s.in0, s.in1})
                if (in >= 0 && in != s.block) r.insert(blk_out(in));
            return r;
        };
        size_t independent = 0;
        for (size_t i = 0; i + 1 < plan.size(); ++i) {
            const std::set<int> wi = writes(plan[i]), wj = writes(plan[i + 1]);
            const std::set<int> ri = reads(plan[i]), rj = reads(plan[i + 1]);
            auto hits = [](const std::set<int>& a, const std::set<int>& b) {
                for (int v : a) if (b.count(v)) return true;
                return false;
            };
            const bool raw = hits(wi, rj), war = hits(wj, ri), waw = hits(wi, wj);
            if (!raw && !war && !waw) {
                ++independent;
                std::cout << "independent pair " << i << " -> " << i + 1 << "  "
                          << plan[i].type << " / " << plan[i + 1].type << '\n';
            }
        }
        std::cout << "adjacent pairs: " << (plan.size() - 1)
                  << "  independent: " << independent << '\n';
    }

    size_t act_total = 0;
    // **Arena reuse, by lifetime.** Every value used to own its bytes for the
    // whole frame: 3.9 GB at 3818x1998, which on Windows is charged twice (VRAM
    // budget and system commit) and took a tester's machine down. A value is
    // only needed from the
    // first dispatch that touches it to the last, and the dispatches are
    // separated by barriers, so two values whose dispatch ranges do not overlap
    // can share memory and nothing about the arithmetic changes.
    //
    // The ranges are not modelled by hand. A probe build (the same code, the
    // same arguments, stopped before the device) records which lowered steps
    // ask `voff` for each slot, maps them through the persistent-run merge onto
    // the final dispatches, and packs the values; this build takes that layout
    // and checks afterwards that it lowered and merged to the same dispatches.
    // Off for the per-layer and scoring tools, which read values by slot after
    // the frame.
    bool reuse = flag(argc, argv, "--reuse") && !flag(argc, argv, "--no-reuse") && !arena_probe;
    if (slot_reading_diagnostic(argc, argv)) reuse = false;
    size_t plain_total = 0;
    for (const auto& kv : vsize) if (kv.second) plain_total += align(kv.second, 256);
    if (reuse) {
        NrSession probe;
        probe.arena_probe = true;
        const nr::LogSink sink = nr::g_log_sink;
        const uint32_t epoch_word = g_chain_epoch_word;
        nr::set_log_sink([](const char*) {});
        int rc = -1;
        try { rc = probe.build(argc, argv, &plan); } catch (...) { nr::set_log_sink(sink); throw; }
        nr::set_log_sink(sink);
        g_chain_epoch_word = epoch_word;
        if (rc != 3) throw std::runtime_error("arena reuse: the layout probe did not finish");
        voff.m = probe.arena_layout;
        act_total = probe.arena_values;
        arena_expect = probe.arena_disp_steps;
        nr::logf("activation arena %.1f MB over %zu values, shared by lifetime (%.1f MB without)",
                 double(act_total) / 1e6, voff.size(), double(plain_total) / 1e6);
    } else {
        // Every value its own bytes, in plan order and then the side buffers.
        //
        // **The side buffers need offsets too, and for a long time they had
        // none.** A first pass allocated only `key_of(block, layer)`; every
        // `skip_key`, `pool_key` and `ups_key` fell through to `std::map`'s
        // default and sat at **offset 0**, aliasing each other and block 0's
        // output. It was visible all along in the arena line - "152 values" is
        // exactly the layer count, with not one extra slot - and in `_ds`'s skip
        // scoring 29% while its main output scored 0.99. Caught by dumping the
        // `_upsample` prologue: the projection buffer came back holding the
        // *blend's* output tiles, which two distinct slots cannot do. (A
        // hand-written liveness beside this loop recycled chain slots and did
        // not model those edges; it was never enabled, and the lifetimes are
        // now recorded rather than modelled - see `reuse` above.)
        // NR_ARENA_GAP=bytes (diagnostic): a guard gap after every value, to
        // catch a kernel writing (NR_ARENA_UNWRITTEN) or reading
        // (NR_POISON_GAPS=all) past its slot.
        const size_t gap = std::getenv("NR_ARENA_GAP") ? size_t(std::atoll(std::getenv("NR_ARENA_GAP"))) : 0;
        for (size_t i = 0; i < plan.size(); ++i) {
            const int k = key_of(plan[i].block, plan[i].layer);
            if (voff.count(k)) continue;
            voff.m[k] = act_total;
            act_total += align(vsize[k], 256);
            if (gap && vsize[k]) { gaps.push_back({k, act_total}); act_total += gap; }
        }
        for (const auto& kv : vsize) {
            if (voff.count(kv.first) || !kv.second) continue;
            voff.m[kv.first] = act_total;
            act_total += align(kv.second, 256);
            if (gap) { gaps.push_back({kv.first, act_total}); act_total += gap; }
        }
        // Through the sink, not stdout: in a game the only reader is
        // dlssnr-amd.log, and a build that throws before its summary line used
        // to leave no record of what it had asked for at all (2026-09-18, a
        // 32-bit game's std::bad_alloc 482 ms into a 4K build).
        nr::logf("activation arena %.1f MB over %zu values",
                 double(act_total) / 1e6, voff.size());
    }

    values_end = act_total;
    // `--vattn-mode`: bit 0 cosine-normalises Q and K per head and applies the
    // per-head temperature, bit 1 permutes the output channels, bit 2 rounds the
    // attention weights to e4m3. Default **5** - bits 0 and 2, not bit 1, which
    // was a permutation this project believed in for a while and the data does
    // not support.
    //
    // They are A/B knobs rather than fallbacks: without bit 0 every logit
    // saturates the exponential's clamp and the layer degenerates to the mean of
    // V, which is a distinct enough failure to be worth reproducing on demand.
    const uint32_t vattn_mode =
        uint32_t(std::atoi(arg(argc, argv, "--vattn-mode", "5")));
    const uint32_t vattn_tokens =
        uint32_t(std::atoi(arg(argc, argv, "--vattn-tokens", "0")));
    // Must match `NR_QT` in the SPV; vit_attn.comp defaults it to the same 32,
    // and its comment carries the sweep that chose it.
    // **32, was 16**; moving one of these three without the other two
    // is a kernel that computes a fraction of the sequence and is fast for it.
    // Windows, measured 10-01: NR_QT=16 (71 VGPRs under LLPC against 114) makes vitattn 1080p
    // -0.009 ms but the ViT projection that reads its output +0.124: kept at 32.
    const uint32_t vattn_qt =
        uint32_t(std::atoi(arg(argc, argv, "--vattn-qt", "32")));

    // ---- the constants the host and the SPVs have to agree on --------------
    // **Three times now an incorrect build has measured as a speedup**, and
    // twice it was this: a shader constant the host also has to know, moved on
    // one side only. `NR_QT` against `--vattn-qt` launches half the workgroups
    // and attends half the sequence; the ViT GEMM tile as a literal here
    // dispatched a grid that stepped further than a workgroup covered and left
    // the rest of every output unwritten, for 4%. Neither is visible in a frame
    // time, in a channel mean or in anything but a `cmp` against a build known
    // to be right - and in the `NR_QT` case the picture metric came out
    // *better* than the shipping build's, which is how it was caught.
    //
    // `--spv-dir` is the hazard: it swaps the pipelines and **not** this
    // binary, so a stale directory pairs old kernels with new launch
    // arithmetic. `shader-constants.txt` is written beside the SPVs by
    // `windows/build/build_network.py` from the same table that produces the
    // `-D`s, and a directory without it is refused rather than guessed at.
    uint32_t qkv_fused_norm = 0, ups_fused_mode = 0, wide_ups_fused_mode = 0;
    uint32_t weight_layout = 0;
    math_profile = 0;
    {
        const std::string path = spv_dir + "/shader-constants.txt";
        std::ifstream f(path);
        if (!f) {
            std::fprintf(stderr, "missing %s; the SPVs and this binary cannot be "
                         "checked against each other. Run windows/build/build_network.py\n",
                         path.c_str());
            return 1;
        }
        // Every entry is a constant compiled into a kernel that this file also
        // has to know. Add one here the moment the host reads a shader's shape.
        const std::pair<const char*, uint32_t> want[] = {
            {"vattn_qt", vattn_qt},
            {"persist_strag", uint32_t(NR_STRAG_PCT)},
            {"persist_one", uint32_t(NR_PERSIST_ONE_MASK)},
            {"noise_field", uint32_t(NR_NOISE_FIELD)},
            {"gemm_wide_mt", uint32_t(NR_GEMM_WIDE_MT)},
            {"gemm_wide_nt", uint32_t(NR_GEMM_WIDE_NT)},
            {"gemm_proj_mt", uint32_t(NR_GEMM_PROJ_MT)},
            {"gemm_proj_nt", uint32_t(NR_GEMM_PROJ_NT)},
            {"gemm_projw_mt", uint32_t(NR_GEMM_PROJW_MT)},
            {"gemm_projw_nt", uint32_t(NR_GEMM_PROJW_NT)},
            {"ffwd_wgw", uint32_t(NR_FFWD_WGW)},
            {"ffwd_gmajor", uint32_t(NR_FFWD_GMAJOR)},
            {"gemm_remap", uint32_t(NR_GEMM_REMAP_HOST)},
            {"ffwd_fm2_min", uint32_t(NR_FFWD_FM2_MIN_TOKENS)},
            {"post_alpha", uint32_t(NR_POST_ALPHA)},
            {"tchain", uint32_t(NR_TCHAIN_TILES)},
            {"upsview_vec", uint32_t(NR_UPSVIEW_VEC)},
            {"repack_vec", uint32_t(NR_REPACK_VEC)},
            {"wide_ups_mask", uint32_t(NR_WIDE_UPS_MASK)},
            {"persist_df", uint32_t(NR_PERSIST_DF)},
            {"ds_fuse", uint32_t(NR_DS_FUSE)},
            {"persist_ds", uint32_t(NR_PERSIST_DS_MASK)},
            {"persist_up", uint32_t(NR_PERSIST_UPS_MASK)},
            {"decups_vec", uint32_t(NR_DECUPS_VEC)},
        };
        std::map<std::string, uint32_t> built;
        try { built = nr::detail::read_shader_manifest(f); }
        catch (const std::exception& e) {
            std::fprintf(stderr,"%s: %s\n",path.c_str(),e.what()); return 1;
        }
        if (built.count("weight_layout")) weight_layout = built.at("weight_layout");
        if (built.count("math_profile")) math_profile = built.at("math_profile");
        if(built.count("wide_ups_fused_mode"))wide_ups_fused_mode=built.at("wide_ups_fused_mode");
        if(wide_ups_fused_mode && (wide_ups_fused_mode>3 || !built.count("ups_fused_mode") || built.at("ups_fused_mode")!=2 ||
           !built.count("qkv_fused_norm") || built.at("qkv_fused_norm")!=1)) {
            std::fprintf(stderr,"wide fusion requires the C32 upsample and QKV fusions\n");return 1;
        }
        if (built.count("ups_fused_mode")) ups_fused_mode=built.at("ups_fused_mode");
        // The fused modes below name the arena policy explicitly: `--no-reuse`,
        // or `--reuse`, whose lifetimes are recorded from these same lowerings
        // (the hand-written reuse they used to exclude modelled none of them).
        if(ups_fused_mode && (ups_fused_mode>2 || accumulation!="fp32" || !host_boundary ||
            !(flag(argc,argv,"--no-reuse") || flag(argc,argv,"--reuse")) || weight_layout!=3 || math_profile!=3 ||
            flag(argc,argv,"--ups-blend") || flag(argc,argv,"--ups-itiles") || flag(argc,argv,"--ups-itiles-gx"))) {
            std::fprintf(stderr,"upsample fusion requires standard FP32 host-boundary layout3/profile3 no-reuse\n");return 1;
        }
        if (built.count("qkv_fused_norm")) qkv_fused_norm = built.at("qkv_fused_norm");
        if (qkv_fused_norm && (qkv_fused_norm != 1 || accumulation != "fp32" ||
            weight_layout != 3 || math_profile != 3 || !host_boundary ||
            !(flag(argc,argv,"--no-reuse") || flag(argc,argv,"--reuse")))) {
            std::fprintf(stderr,"QKV fusion requires FP32 host-boundary layout3/profile3\n"); return 1;
        }
        if(qkv_fused_norm && !std::getenv("NR_DIAG_BARRIERS")) {   // diagnostic override
            for(const char* f : {"--only","--seed-dir","--no-barrier-after","--barrier-stride","--oracle","--no-barrier","--barrier-report","--exec-barrier"})
                if(flag(argc,argv,f)) { std::fprintf(stderr,"QKV fusion requires complete graph with standard barriers; rejects %s\n",f); return 1; }
        }
        // Fast math profiles: fast LUT plus pre-affine position bias.
        if ((math_profile == 2 && !swin_bias_f16) || (math_profile == 3 && swin_bias_f16)) {
            std::fprintf(stderr,"baked exponential bias storage/profile mismatch\n"); return 1;
        }
        if (math_profile && accumulation != "fp32") {
            std::fprintf(stderr,"fast math profile requires fp32 accumulation\n"); return 1;
        }
        if (weight_layout != 0 && accumulation != "fp32") {
            std::fprintf(stderr,"packed shader weights require fp32 accumulation\n"); return 1;
        }
        // Shader directories from before the straggler queue record no key: off.
        if (!built.count("persist_strag")) built["persist_strag"] = 0;
        if (!built.count("persist_one")) built["persist_one"] = 0;
        if (!built.count("noise_field")) built["noise_field"] = 0;
        if (!built.count("ffwd_fm2_min")) built["ffwd_fm2_min"] = 0;
        if (!built.count("post_alpha")) built["post_alpha"] = 0;
        if (!built.count("tchain")) built["tchain"] = 0;
        if (!built.count("ffwd_gmajor")) built["ffwd_gmajor"] = 0;
        if (!built.count("gemm_remap")) built["gemm_remap"] = 0;
        for (const auto& [name, mine] : want) {
            auto it = built.find(name);
            if (it == built.end()) {
                std::fprintf(stderr, "%s does not record %s; run "
                             "windows/build/build_network.py\n", path.c_str(), name);
                return 1;
            }
            if (it->second != mine) {
                std::fprintf(stderr, "%s: the SPVs were built with %s=%u and this "
                             "binary dispatches %u. That launch is wrong and it is "
                             "*faster*, which no timing will tell you. Rebuild, or "
                             "pass the matching flag.\n",
                             path.c_str(), name, it->second, mine);
                return 1;
            }
        }
    }

    // ---- weights -----------------------------------------------------------
    //
    // The weight arena on the host (WeightBlob: 4 MB chunks, zeros free; the arena
    // probe only counts). Logged before and after, so a log that stops between them
    // says the weights were where the build ran out.
    nr::logf("unpacking weights from %s", unp.c_str());
    WeightBlob wblob(arena_probe);
    g_noise_jobs.clear();   // this lowering's own; the arena probe ran its own before
    auto put = [&](const std::vector<uint8_t>& v, size_t a) { return wblob.append(v.data(), v.size(), a); };
    uint32_t act_lut_off = 0;
    if (math_profile >= 1) {
        const auto& table = nr::detail::activation_lut_v1;
        act_lut_off = uint32_t(put(std::vector<uint8_t>(table.begin(), table.end()), 256));
        std::printf("math profile: %u (4096-byte activation table; baked exp bias=%u)\n", math_profile, unsigned(math_profile >= 2));
    } else std::printf("math profile: exact-v1\n");
    auto put_f32 = [&](const std::vector<float>& v) {
        std::vector<uint8_t> b(v.size() * 4);
        std::memcpy(b.data(), v.data(), b.size());
        return put(b, 16);
    };
    auto put_f16 = [&](const std::vector<float>& v) {
        std::vector<uint8_t> b(v.size() * 2);
        for (size_t i = 0; i < v.size(); ++i) {
            const uint16_t h = tin::f_to_f16(v[i]);
            b[2 * i] = uint8_t(h & 0xFF); b[2 * i + 1] = uint8_t(h >> 8);
        }
        return put(b, 16);
    };

    timer.mark("plan");
    if (!arena_probe) {
        nr::g_build_stage = 2;
        if (nr::g_memory_note) nr::logf("planned; %s", nr::g_memory_note().c_str());
    }
    int lowering = 0;
    for (const Step* ps : run) {
        voff.step = lowering++;
        const Step& s = *ps;
        const Shape sh = shape_of(s);
        Disp d{&s};
        char base[256];
        if (sh.fam == F_FSWIN || sh.fam == F_DS || sh.fam == F_UPS) {
            const bool pre = s.type == "CCTinlayoutFusedPreBlockSwin1H";
            const bool post = s.type == "CCTinlayoutFusedPostBlockSwin1H";
            std::snprintf(base, sizeof base, "%s/%s/block%d.layer%d.layer",
                          unp.c_str(),
                          pre ? "unpacked-preblock" : post ? "unpacked-postblock"
                                                           : "unpacked",
                          s.block, s.layer);
            const int C = pre || post ? 32 : s.C;
            const int heads = pre || post ? 1 : s.heads;
            const int H = 4 * C, Hg = H / heads;
            const int Kc = heads > 1 ? C : H;
            PushFSwin p{};
#define NR_HS_FOLDED(on, v) (v)
            p.e_off = uint32_t(put(tin::tile_blocked(
                load_e4m3_requantised(std::string(base) + ".mlp_expand.bin", size_t(H) * C).data(),
                size_t(H), size_t(C)), 256));
            p.ct_off = uint32_t(put(tin::tile_blocked(
                NR_HS_FOLDED(heads == 1, load_e4m3_requantised(std::string(base) + ".mlp_contract.bin", size_t(C) * Kc)).data(),
                size_t(C), size_t(Kc)), 256));
            p.qkv_off = uint32_t(put(tin::tile_blocked(
                load_e4m3_requantised(std::string(base) + ".qkv.bin", size_t(3 * C) * C).data(),
                size_t(3 * C), size_t(C)), 256));
            p.op_off = uint32_t(put(tin::tile_blocked(
                load_e4m3_requantised(std::string(base) + ".attn_out_proj.bin", size_t(C) * C).data(),
                size_t(C), size_t(C)), 256));
            p.mid_off = heads > 1
                ? uint32_t(put(tin::tile_blocked(
                    NR_HS_FOLDED(true, load_e4m3_requantised(std::string(base) + ".mlp_mid.bin", size_t(C) * Hg)).data(),
                    size_t(C), size_t(Hg)), 256))
                : 0u;
            {
                const std::vector<float> raw = load_f16(std::string(base) + ".attn_pos_bias.bin");
                std::vector<float> bias(size_t(heads) * 4096);
                for (int h = 0; h < heads; ++h) {
                    const std::vector<float> one(raw.begin() + size_t(h) * 4096,
                                                 raw.begin() + size_t(h + 1) * 4096);
                    const std::vector<float> dz = deswizzle_bias(one);
                    std::copy(dz.begin(), dz.end(), bias.begin() + size_t(h) * 4096);
                }
                // Fast math profile: distribute exponent affine over the fixed bias.
                // Changes rounding: a*(logit+bias)+b -> fma(a,logit,fma(a,bias,b)).
                if (math_profile == 2 || math_profile == 3)
                    for (float& v : bias) v = std::fma(v, 0.044921875f, 1.30078125f);
                g_audit_bias = (math_profile == 2 || math_profile == 3) ? bias : std::vector<float>();
                // Recovered position bias is already FP16. Widen only when
                // adding it to FP32 logits; no new numerical rounding.
                p.b_off = swin_bias_f16 ? uint32_t(put_f16(bias) / 2)
                                       : uint32_t(put_f32(bias) / 4);
            }
            {
                const std::vector<float> r = load_f16(std::string(base) + ".residual_scale.bin");
                const std::vector<float> a = load_f16(std::string(base) + ".attn_residual_scale.bin");
                // **The eight leading zeros are not always there.** At C=32 the
                // region is 8 zeros, C values, 8 zeros; on an `_upsample` layer
                // at C>=64 it is C values followed by sixteen more that belong
                // to `upsample_gain` - there is no padding at all, and reading
                // from +8 takes the layer's scales eight channels out of step.
                // Detect it rather than assume: the marker is its own check.
                const bool padded = r.size() >= 8 &&
                    std::all_of(r.begin(), r.begin() + 8, [](float v) { return v == 0.0f; });
                // When the region is unpadded it holds C + 16 scalars and the
                // split is the open question: the first C with the extra sixteen
                // going to `upsample_gain`, or the last C with the sixteen at the
                // front. `--rs-last` is the other half of that pair.
                const size_t rbase = padded ? 8u
                    : (flag(argc, argv, "--rs-last") && r.size() >= size_t(C)
                           ? r.size() - size_t(C) : 0u);
                std::vector<float> rs(r.begin() + long(rbase), r.begin() + long(rbase) + C);
                std::vector<float> ars(a.begin(), a.begin() + C);
                const std::vector<uint8_t> sb = slurp(std::string(base) + ".scalars_b.bin");
                std::vector<float> sc(sb.size() / 4);
                if (!sb.empty()) std::memcpy(sc.data(), sb.data(), sb.size());
                const bool residual_half=weight_layout==4 || weight_layout==5;
                p.rs_off = residual_half ? uint32_t(put_f16(rs)/2) : uint32_t(put_f32(rs)/4);
                p.ars_off = residual_half ? uint32_t(put_f16(ars)/2) : uint32_t(put_f32(ars)/4);
                p.s_off = uint32_t(put_f32(sc) / 4);
                if (C == 32 && g_audit_bias.size() == size_t(heads) * 4096 && sc.size() >= size_t(heads)) {
                    // |logit| <= |q^|.|k^| <= 1.139|s| + 0.006 (unit-normalised q, k; e4m3 rounding at
                    // most 1/16 relative plus half a subnormal step; f16 norm within 2^-9); with margin
                    // 1.2|s| + 0.05. The upper clamp is dead when every baked bias + 0.044921875 * that
                    // bound stays at or below it.
                    bool dead = true;
                    for (int h = 0; h < heads && dead; ++h) {
                        const double sa = 0.044921875 * (1.2 * std::fabs(double(sc[size_t(h)])) + 0.05);
                        for (int i = 0; i < 4096 && dead; ++i)
                            if (!(double(g_audit_bias[size_t(h) * 4096 + size_t(i)]) + sa <= 1.5693359375)) dead = false;
                    }
                    if (dead) g_nohi.insert({s.block, s.layer});
                }
                auto diag = [&](const std::vector<float>& v) {
                    std::vector<uint8_t> t(size_t(C / 16) * 256 * 2, 0);
                    for (int n = 0; n < C; ++n) {
                        const uint16_t h = tin::f_to_f16(v[size_t(n)]);
                        const size_t e = size_t(n / 16) * 256 + size_t(n % 16) * 16 + size_t(n % 16);
                        t[2 * e] = uint8_t(h & 0xFF); t[2 * e + 1] = uint8_t(h >> 8);
                    }
                    return t;
                };
                p.rsd_off = uint32_t(put(diag(rs), 16) / 2);
                // Fast-profile kernels reinterpret this unused diagonal offset
                // as a byte offset into the embedded activation table.
                if (math_profile >= 1) p.rsd_off = act_lut_off;
                p.ard_off = uint32_t(put(diag(ars), 16) / 2);
            }
            // An `_upsample` block body reads what its prologue built, not the
            // layer's own input: the input is at half the resolution and twice
            // the width, and the projection, the replication and the skip blend
            // all happen before the block sees anything.
            p.x_off = uint32_t(voff[
                sh.fam == F_UPS || s.type == "CCTinlayoutFusedPostBlockSwin1H"
                    ? ups_key(s.block)
                    : s.type == "CCTinlayoutFusedPreBlockSwin1H"
                        ? lift_key(s.block)
                        : x_src(s)]);
            // A `_ds` block writes its full-resolution result to the skip slot;
            // the pool and the projection below turn that into the block's
            // actual output. A plain block writes straight to it.
            // The pre-block is wired like a `_ds` even though its tail is a
            // bare 2x2 mean: the body writes the full-resolution result to the
            // skip slot and the pool turns it into the block's actual output.
            // That is the same three-slot shape every other downsample has, and
            // it is what stops the pool reading and writing one buffer.
            p.o_off = uint32_t(voff[sh.fam == F_DS ||
                                    s.type == "CCTinlayoutFusedPreBlockSwin1H"
                                        ? skip_key(s.block)
                                        : key_of(s.block, s.layer)]);
            // **The tile row stride is the *unpadded* width, not the grid's.**
            // Probing `cc_tinlayout_fused_swin_1h_32_1_ds_fp8` with a single lit
            // input tile puts the output exactly where a stride of 135 tiles
            // predicts and nowhere near where 136 does: the buffer is allocated
            // 544 x 960 tokens but only 540 x 960 are addressed, leaving 3840
            // tokens of slack (522240 - 518400).
            //
            // The plan's W and H columns are swapped - `+208` is the height and
            // `+212` the width - so the true width is the H column and
            // `gx = ceil(width/8)`. 540/4 = 135 tiles across, 960/4 = 240 down.
            // `_upsample` runs its body one level *up* from the plan's extent -
            // the plan carries the input level and the window grid carries the
            // output level, so the body's tile row stride doubles.
            // **The post block is an `_upsample` too.** It is one of the six
            // U-Net skip edges - `70 <- [69, 0]` - its `div` is 2 while its
            // grid is `ceil(1080/8) x ceil(1920/8)`, the div=1 level, and every
            // other member of that family takes its tile stride at the output
            // level. Taken at the input level, as it was, the row stride is
            // half the row and the picture wraps: a real 1920x1080 frame came
            // out of the frame renderer as horizontal bands of repeated
            // content, which is exactly what a halved stride looks like.
            const bool ups_level = sh.fam == F_UPS ||
                                   s.type == "CCTinlayoutFusedPostBlockSwin1H";
            p.tiles_x = tiles_of(ups_level ? 2 * s.H : s.H);
            p.tiles_y = tiles_of(ups_level ? 2 * s.W : s.W);
            // **`--grid-tiles`: the tile raster is the window grid.** A window
            // is 8 pixels, which is two tiles, so a `gx`-window row is `2*gx`
            // tiles - 136 where `tiles_of(540)` truncates to 135. The `_ds`
            // output's raster is `4*gx` pixels, read out of the kernel's own
            // store address, so the level above it has to be the same grid or
            // the two disagree by a tile per row.
            if (g_grid_tiles) {
                p.tiles_x = uint32_t(s.gx) * (ups_level ? 4u : 2u);
                p.tiles_y = uint32_t(s.gy) * (ups_level ? 4u : 2u);
            }
            if(host_boundary && sh.fam==F_UPS) {
                const auto extent=up_extent(s);p.tiles_x=tiles_of(extent.second);p.tiles_y=tiles_of(extent.first);
            }
            p.shift = s.shift_x == 100 ? (s.shifted ? 1 : 0) : s.shift_x;
            p.shift_y = s.shift_y == 100 ? p.shift : s.shift_y;
            d.kern = "fswin" + std::to_string(C);
            if(host_boundary && pre)d.kern="fswinpre32";
            if(host_boundary && sh.fam==F_UPS && C==32)d.kern="fswinup32";
            // Diagnostic: isolate the existing wide body from persistent runs.
            // Same ordinary SPV, unchanged projection/blend and arithmetic.
            if(host_boundary && sh.fam==F_UPS && C>32 && flag(argc,argv,"--isolate-ups"))
                d.kern="fswinisolatedup"+std::to_string(C);
            if(host_boundary && s.type=="CCTinlayoutFusedPostBlockSwin1H")d.kern="fswinpost32";
            if (host_boundary && (pre || sh.fam==F_DS) && !flag(argc,argv,"--legacy-ds-quant")) {
                p.pool_off=uint32_t(voff[pre ? key_of(s.block,s.layer) : pool_key(s.block)]);
                p.pool_tiles_x=pre ? tiles_of(s.H/2) : uint32_t(pad4((s.H+1)/2)/4);
                p.pool_tiles_y=(p.tiles_y+1u)/2u;
                d.kern=pre ? "fswinpreds32" : "fswinds"+std::to_string(C);
            }
            d.gx = uint32_t(s.gx); d.gy = uint32_t(s.gy); d.gz = 1;
            d.push.resize(sizeof p);
            std::memcpy(d.push.data(), &p, sizeof p);
            // **The pre-block's front and the post-block's tail.** Everything
            // between them is the ordinary C=32 block; these two are the only
            // kernels in the port that touch an image, and they are what turns
            // the graph from a tensor pipeline into something a game can call.
            if (s.type == "CCTinlayoutFusedPreBlockSwin1H") {
                Disp di{&s};
                PushImgIn pi{};
                // FP16[32][16], the one matrix in this record that is not e4m3,
                // because its operand comes out of a texture.
                pi.w_off = uint32_t(put(slurp(std::string(base) + ".input_lift.bin"),
                                        16)) / 2u;
                pi.o_off = uint32_t(voff[lift_key(s.block)]);
                pi.tiles_x = tiles_of(s.H);
                pi.W = uint32_t(s.H); pi.H = uint32_t(s.W);
                pi.source_W = uint32_t(std::atoi(arg(argc, argv, "--source-width", "0")));
                pi.source_H = uint32_t(std::atoi(arg(argc, argv, "--source-height", "0")));
                pi.slot = uint32_t(std::atoi(arg(argc, argv, "--img-slot", "4")));
                // `+196`, DAT_1800b80e8 = 0.0625f, doubled by the kernel. **At
                // zero this whole layer still runs and still fills every output
                // byte with a result that does not depend on the image**, so it
                // is the one parameter never to leave at its default.
                // **0.0625, measured, not the 0.125 this shipped with.** Against
                // a flat-colour capture the response to a pure primary, black
                // subtracted, comes out at 2.07x and 2.05x the reference's for red
                // and green at 0.125, and at **1.043x and 1.024x** at 0.0625 -
                // which is also the `+196` the capture sets. Blue is still 1.72x
                // and is a separate defect.
                pi.gate = float(std::atof(arg(argc, argv, "--img-gate", "0.0625")));
                pi.bias_x = pi.bias_y = 0.0f;
                pi.scale_x = pi.scale_y = pi.post_x = pi.post_y = 1.0f;
                pi.uv_swap = flag(argc, argv, "--img-uv-swap") ? 1u : 0u;
                // Runtime metadata enables Control/History/Style (1,1,1),
                // with three styles. Host: 020e60 -> 021bb0 -> 03f490 -> 061710.
                // Captures nr_host_complete used explicitly zeroed controls.
                const bool capture_controls=flag(argc,argv,"--capture-controls");
                const float tone=std::atof(arg(argc,argv,"--local-tone-strength","1"));
                const float structure=std::atof(arg(argc,argv,"--local-structure-strength","1"));
                const float skin=std::atof(arg(argc,argv,"--skin-structure-strength","-1"));
                const bool mask=auto_mask_text=="1";
                const int style=std::clamp(std::atoi(arg(argc,argv,"--style","0")),0,2);
                const float mask_skin=mask ? (skin<0 ? structure : skin) : -1.0f;
                const float mask_other=mask ? structure : -1.0f;
                const bool any_mask=std::max(mask_skin,mask_other)>=0;
                pi.seed=uint32_t(std::atoi(arg(argc,argv,"--img-seed","0")));
                pi.s434=capture_controls ? 0 : float(style)/128.0f;
                pi.aux0=capture_controls ? 0 : tone;
                pi.aux1=capture_controls ? 0 : any_mask ? 1.0f : structure;
                pi.gate0=capture_controls ? 0 : any_mask ? (mask_skin<0?structure:mask_skin) : -1.0f;
                pi.gate1=capture_controls ? 0 : any_mask ? (mask_other<0?structure:mask_other) : -1.0f;
                // Raw feature overrides are diagnostics, distinct from NGX controls.
                if(const char* v=arg(argc,argv,"--img-s434"))pi.s434=std::atof(v);
                if(const char* v=arg(argc,argv,"--img-aux0"))pi.aux0=std::atof(v);
                if(const char* v=arg(argc,argv,"--img-aux1"))pi.aux1=std::atof(v);
                if(const char* v=arg(argc,argv,"--img-gate0"))pi.gate0=std::atof(v);
                if(const char* v=arg(argc,argv,"--img-gate1"))pi.gate1=std::atof(v);
                std::printf("pre conditioning [style,tone,structure,skin,other]=[%g,%g,%g,%g,%g]%s\n",
                    pi.s434,pi.aux0,pi.aux1,pi.gate0,pi.gate1,capture_controls?" (capture controls)":"");
                std::printf("Automatic Mask: requested=%d, conditioning=%s; external ControlMask absent\n",
                    int(mask),capture_controls?"overridden by capture controls":mask?"enabled":"disabled");
                pi.noise = float(std::atof(arg(argc, argv, "--img-noise", "1")));
                // Feature 3. Measured against a flat-colour capture: at 1.0 the
                // black response - which is what the non-colour features
                // contribute - is 14.7x NVIDIA's while the colour slope is only
                // 1.6x, and no single gain does both. See the shader.
                // **0.125.** Two measurements disagree and this is the better of
                // them: by magnitude against `preflat`'s grey response - which is
                // the pure constant term, since the colour features are zero at
                // 0.5 - the match is near 0.087, but `b000l0`'s r peaks at 0.125
                // (0.8158 against 0.7993 at 0.0866 and 0.7600 at 0.0625). The gap
                // between the two is the structural error that is still open.
                pi.konst = float(std::atof(arg(argc, argv, "--img-const", "1")));
                // **-1, and it contradicts the PTX.** The feature packing says slot 6
                // is `Cz` with no negation and slot 3 a literal 1.0, yet against
                // a capture the entry layer reads 0.795 at those values and
                // **0.9103** at `amp_b = -1, const = 0.125`. Two knobs that each
                // want a clean factor mean the *lift* is wrong in a way they
                // happen to cancel - a row sign and a row scale - and the matrix
                // is where to look next, not these.
                pi.amp_b = float(std::atof(arg(argc, argv, "--img-amp-b", "1")));
                pi.bslot = uint32_t(std::atoi(arg(argc, argv, "--img-bslot", "6")));
                pi.lift_tiled = uint32_t(std::atoi(arg(argc, argv, "--img-lift-tiled", "3")));
                pi.row_shift = uint32_t((std::atoi(arg(argc, argv, "--img-row-shift", "0")) + 32) % 16);
                pi.lift_perm = uint32_t(std::atoi(arg(argc, argv, "--img-lift-perm", "0")));
                // **Transposed and negated, measured.** The lift is 1024 bytes
                // and 32x16 either way, so the unpacker cannot tell; against
                // block 0's gold the four combinations read
                //
                //     plain            r = +0.1915
                //     transposed       r = -0.7414
                //     plain, negated   r = -0.1738
                //     transposed, neg  r = **+0.7351**
                //
                // and that is with a completely different input image, so what
                // is correlating is the layer's structure. `--img-lift-plain`
                // and `--img-lift-pos` take either back.
                pi.lift_t  = flag(argc, argv, "--img-lift-plain") ? 0u : 1u;
                // **The negation is gone, and the four-way table above is
                // retracted.** It was measured against a capture whose pre
                // block had a device pointer in four of its texture slots. Re-run
                // against the same capture, at `--img-slot 4` - the SASS
                // placement - the same four read
                //
                //     transposed, negated   r = -0.7606     (the old default)
                //     transposed, positive  r = **+0.7600**
                //     plain, negated        r = -0.2947
                //     plain, positive       r = +0.2985
                //
                // i.e. the orientation was right, the sign was not, and the
                // magnitude was never the question. `--img-lift-neg` puts it back.
                pi.lift_neg = flag(argc, argv, "--img-lift-neg") ? 1u : 0u;
                di.kern = host_boundary ? "imgin16" : "imgin";
                // The specialized lift keeps its final zero term live so the
                // FP32 accumulator rounds before the FP16 store. It preserves
                // the generic path's complete 1080p/4K images.
                if (host_boundary && pi.slot == 4u && pi.bslot == 6u &&
                    pi.lift_tiled == 3u && pi.lift_perm == 0u && pi.lift_neg == 0u &&
                    !flag(argc, argv, "--generic-image-lift"))
                    di.kern = "imgin16fast";
                // WMMA materializes FP16 features before the projection. Keep
                // the scalar path for partial groups and layout diagnostics.
                if (di.kern == "imgin16fast" && accumulation == "fp32" &&
                    pi.W % 8u == 0u && pi.H % 8u == 0u &&
                    !flag(argc, argv, "--scalar-image-lift"))
                    di.kern = "imgin16wmma";
                di.gx = (pi.W + 7u) / 8u; di.gy = (pi.H + 7u) / 8u; di.gz = 1;
                di.push.resize(sizeof pi);
                std::memcpy(di.push.data(), &pi, sizeof pi);
                if(di.kern=="imgin16wmma" && d.kern=="fswinpreds32" &&
                   p.shift==0 && p.shift_y==0 && p.tiles_x*4u==pi.W && p.tiles_y*4u==pi.H &&
                   d.gx==di.gx && d.gy==di.gy &&
                   pi.gate==0.0625f && pi.amp_b==1.0f && pi.uv_swap==0u &&
                   !score_dir && !arg(argc,argv,"--dump-stage") &&
                   !flag(argc,argv,"--separate-image-input")) {
                    // Decode the native half lift once. No weight arithmetic:
                    // this is exactly img_in.comp's packed operand addressing.
                    const auto packed_lift=slurp(std::string(base)+".input_lift.bin");
                    if(packed_lift.size()!=1024)throw std::runtime_error("invalid native input lift size");
                    std::vector<uint8_t> plain_lift(1024);
                    for(uint32_t ch=0;ch<32;++ch)for(uint32_t k=0;k<16;++k) {
                        const uint32_t off=(ch/16u)*256u+32u*(ch%8u)+8u*((k%8u)/2u)
                            +4u*((ch/8u)%2u)+(k%2u)+2u*(k/8u);
                        std::memcpy(plain_lift.data()+2*(ch*16u+k),packed_lift.data()+2*off,2);
                    }
                    uint32_t noise_off=0;
                    if (NR_NOISE_FIELD) {
                        // A zeroed region the build fills on the GPU after the upload
                        // (see g_noise_jobs); f16 x4 per pixel of the working grid.
                        const uint32_t nw=d.gx*8u, nh=d.gy*8u;
                        noise_off=uint32_t(wblob.append_zeros(size_t(nw)*nh*8u,256)/4);
                        if(!noise_off)throw std::runtime_error("noise field at offset 0");
                        g_noise_jobs.push_back({noise_off,nw,nh,pi.seed,pi.noise});
                    }
                    const PushPreImage ip{uint32_t(put(plain_lift,16)/2),pi.source_W,pi.source_H,
                        pi.seed,pi.s434,pi.aux0,pi.aux1,pi.gate0,pi.gate1,pi.noise,pi.konst,noise_off};
                    d.kern="fswinimagepreds32";d.push.resize(sizeof(PushFSwin)+sizeof ip);
                    std::memcpy(d.push.data()+sizeof(PushFSwin),&ip,sizeof ip);
                } else disp.push_back(std::move(di));   // the lift, first
                disp.push_back(std::move(d));    // then the block

                // **The `_ds` tail here is a 2x2 mean and nothing else.** The
                // record has no resample matrix in it - the extra region is the
                // FP16 lift - and the half-resolution gold is 32 channels wide,
                // the same as the full one. Every other `_ds` in the network
                // doubles C through a learned [2C][C]; this one does not.
                Disp dp{&s};
                PushPool pp{};
                pp.x_off = uint32_t(voff[skip_key(s.block)]);
                pp.o_off = uint32_t(voff[key_of(s.block, s.layer)]);
                pp.tiles_x = g_grid_tiles ? uint32_t(s.gx) * 2u : tiles_of(s.H);
                pp.tiles_y = g_grid_tiles ? uint32_t(s.gy) * 2u : tiles_of(s.W);
                pp.otiles_x = tiles_of(s.H / 2);
                // 540 columns at this level, which is above the 135 where the
                // raster starts being padded, so the two are the same number.
                pp.crow = pp.raster = uint32_t(s.H / 2);
                dp.kern = "pool32";
                dp.gx = uint32_t(s.gx); dp.gy = uint32_t(s.gy); dp.gz = 1;
                dp.push.resize(sizeof pp);
                std::memcpy(dp.push.data(), &pp, sizeof pp);
                if (!host_boundary || flag(argc,argv,"--legacy-ds-quant"))
                    disp.push_back(std::move(dp));
                continue;
            }
            if (s.type == "CCTinlayoutFusedPostBlockSwin1H") {
                // **The prologue this edge never had.** The chain arrives at
                // half resolution and block 0's own body output is the
                // full-resolution skip, which is the `_upsample` shape:
                // replicate 2x2, blend the skip, then the body. Without it the
                // body ran at the doubled grid over an input a quarter its size.
                //
                // Host mode reads separate main and skip gains from the PTX
                // weight offsets. The legacy diagnostic mode retains its flat
                // --post-skip override. This post path still needs a fresh
                // NVIDIA comparison with matching texture/control parameters.
                Disp dbl{&s};
                PushUps pu{};
                pu.p_off = uint32_t(voff[x_src(s)]);
                pu.s_off = uint32_t(voff[skip_key(s.in1 >= 0 ? s.in1 : s.block)]);
                pu.o_off = uint32_t(voff[ups_key(s.block)]);
                pu.g_off = uint32_t(put_f16(std::vector<float>(
                    size_t(C), float(std::atof(arg(argc, argv, "--post-skip", "0"))))) / 2);
                pu.tiles_x = tiles_of(2 * s.H);
                pu.tiles_y = tiles_of(2 * s.W);
                pu.itiles_y = tiles_of(s.W);
                // **The post block's input is block 69's output, and block 69
                // wrote it with `tiles_of(H)`.** The `tokens/16/itiles_y`
                // quotient the `_upsample` family uses gives 136 here where the
                // producer used 135, and a one-tile-per-row shear over 240 tile
                // rows is what put a diagonal streak across the frame and left
                // its right third unwritten.
                pu.itiles_x = flag(argc, argv, "--post-itiles-quotient")
                                  ? (pu.itiles_y ? uint32_t(s.tokens / 16) / pu.itiles_y
                                                 : tiles_of(s.H))
                                  : tiles_of(s.H);
                if(host_boundary) {
                    std::vector<float> gain=load_f16(std::string(base)+".skip_gain.bin");
                    const auto main_gain=load_f16(std::string(base)+".main_gain.bin");
                    gain.insert(gain.end(),main_gain.begin(),main_gain.end());
                    if(gain.size()!=64){std::fprintf(stderr,"regenerate post-block weights\n");return 1;}
                    pu.g_off=uint32_t(put_f16(gain)/2);pu.mode=6;
                    const Step* prev=nullptr;
                    for(const Step& q:plan)if(q.block==s.in0)prev=&q;
                    if(!prev){std::fprintf(stderr,"post block needs preceding extent\n");return 1;}
                    PushUpsView pv{uint32_t(voff[x_src(s)]),uint32_t(voff[pool_key(s.block)]),
                        uint32_t(s.H*s.W),32,uint32_t(prev->H),uint32_t(prev->W),uint32_t(s.H),uint32_t(s.W)};
                    const bool identity = pv.W==pv.RW && pv.H==pv.RH &&
                        pv.W%4u==0u && pv.H%4u==0u && pv.M==pv.W*pv.H;
                    if (identity && !flag(argc,argv,"--force-ups-view")) {
                        pu.p_off=pv.x_off;
                    } else {
                        Disp view{&s};view.kern="upsview";view.gx=(pv.M*(NR_UPSVIEW_VEC?pv.C/16u:pv.C)+63)/64;view.gy=view.gz=1;
                        view.push.resize(sizeof pv);std::memcpy(view.push.data(),&pv,sizeof pv);disp.push_back(std::move(view));
                        pu.p_off=pv.o_off;
                    }
                } else pu.mode = uint32_t(std::atoi(arg(argc, argv, "--ups-blend", "0")));
                dbl.kern = host_boundary ? "upsblendpost32" : "upsblend" + std::to_string(C);
                dbl.gx = pu.tiles_x; dbl.gy = pu.tiles_y; dbl.gz = 1;
                dbl.push.resize(sizeof pu);
                std::memcpy(dbl.push.data(), &pu, sizeof pu);
                // Keep the explicit intermediate for seeded/oracle/stage diagnostics.
                // The fused kernel is validated for the FP32 host-boundary path.
                const bool fused_post = accumulation == "fp32" &&
                    d.kern == "fswinpost32" && pu.mode == 6 &&
                    !score_dir && !arg(argc, argv, "--dump-stage") &&
                    !flag(argc, argv, "--separate-post-blend");
                if (fused_post) {
                    d.kern="fswinblendpost32";
                    static_assert(sizeof(PushFSwin) == 80 && sizeof(PushUps) == 40,
                                  "fused post push layout must match GLSL");
                    d.push.resize(sizeof(PushFSwin)+sizeof pu);
                    std::memcpy(d.push.data()+sizeof(PushFSwin),&pu,sizeof pu);
                } else disp.push_back(std::move(dbl));

                disp.push_back(std::move(d));    // the block, then the surfaces
                Disp du{&s};
                PushImgOut po{};
                po.w_off = uint32_t(put(slurp(std::string(base) + ".out_project.bin"),
                                        16)) / 2u;
                po.x_off = uint32_t(voff[key_of(s.block, s.layer)]);
                po.tiles_x = tiles_of(2 * s.H);
                po.W = uint32_t(2 * s.H); po.H = uint32_t(2 * s.W);
                if (const char* width=arg(argc, argv, "--source-width")) po.W=uint32_t(std::atoi(width));
                if (const char* height=arg(argc, argv, "--source-height")) po.H=uint32_t(std::atoi(height));
                // `+48`, DAT_1800bc9a8 = 0.03125f.
                po.gate = float(std::atof(arg(argc, argv, "--img-out-gate", "0.03125")));
                // **The output is the input plus the network.** Measured on the
                // NVIDIA side: with the post block's own gain at zero it reproduces the
                // input RGB to 1.49e-8. Without this the shader wrote the
                // network's contribution alone, which is why the frame ran to
                // -0.63 where the reference stays inside [0, 1].
                // **Off by default, because no reference we hold shows it.** cap17,
                // cap19 and cap20's final images all correlate with the input at
                // -0.06 to -0.12 and are darker than it (rms 0.24/0.53/0.42
                // against the input's 0.62), so in the configuration those
                // captures ran, the post block's input tap contributes nothing -
                // its own coordinate parameters were never set. A separate
                // measurement on NVIDIA hardware (gain zeroed, corrected parameters)
                // *did* return the input to 1.49e-8, and cap21 -
                // a whole frame captured with the post block's own coordinate
                // parameters set - **correlates with its input at +0.939**. So
                // the shipping network adds the input and this is on by default.
                // cap17/cap19/cap20 do not show it because their post block ran
                // with those parameters at zero; score the final image against
                // cap21, and the per-layer board against cap17.
                po.src = float(std::atof(arg(argc, argv, "--img-out-src", "1")));
                // The network is 1:1, so the input's extent is the output's.
                po.sw = po.W; po.sh = po.H;
                po.base0 = uint32_t(std::atoi(arg(argc, argv, "--img-out-base", "0")));
                po.base1 = uint32_t(std::atoi(arg(argc, argv, "--img-out-base1", "8")));
                po.blend = float(std::atof(arg(argc, argv, "--img-out-blend", "1")));
                po.nr_intensity=nr_intensity;
                if(host_boundary){po.gate*=8.0f;po.blend=-1.0f;}
                std::printf("Overall Intensity: requested=%g, GPU push=%g, mapping=%s\n",
                    nr_intensity,po.nr_intensity,host_boundary?"nr-delta-v1":"legacy diagnostic");
                du.kern = "imgout";
                if(host_boundary){du.kern="imgout16";po.x_off/=2u;}
                du.gx = (po.W + 7u) / 8u; du.gy = (po.H + 7u) / 8u; du.gz = 1;
                du.push.resize(sizeof po);
                std::memcpy(du.push.data(), &po, sizeof po);
                if(fused_post && po.base0==0u && po.src==1.0f && po.gate==0.25f &&
                   !flag(argc,argv,"--separate-image-output")) {
                    // This first-frame specialization keeps the original scalar
                    // projection/strength math. Nondefault diagnostic projection
                    // controls retain the separate output pipeline.
                    Disp& post=disp.back();post.kern="fswinimagepost32";
                    const PushImageTail tail{po.w_off,po.nr_intensity};
                    const size_t offset=sizeof(PushFSwin)+sizeof(PushUps);
                    post.push.resize(offset+sizeof tail);
                    std::memcpy(post.push.data()+offset,&tail,sizeof tail);
                    continue;
                }
                disp.push_back(std::move(du));
                continue;
            }
            if (sh.fam == F_UPS) {
                // C>=64 PTX rounds its half extent to four pixels before
                // reading the view; C=32 uses the unpadded half extent.
                // Outview writes the unpadded raster, so reproduce the byte
                // reinterpretation (including channel-plane crossings) here.
                // Reuse the blend buffer before GEMM; the later blend writes
                // it only after GEMM has finished reading it.
                const bool identity_view = s.H%4==0 && s.W%4==0 && s.tokens==s.H*s.W;
                const bool ups_view = C >= 64 && !flag(argc, argv, "--ups-view-plain") &&
                    (!identity_view || flag(argc,argv,"--force-ups-view"));
                if (ups_view && wide_ups_fused_mode!=3) {
                    Disp dv{&s};
                    PushUpsView pv{uint32_t(voff[x_src(s)]),
                                   uint32_t(voff[ups_key(s.block)]),
                                   uint32_t(s.tokens), uint32_t(2 * C),
                                   uint32_t(s.H), uint32_t(s.W)};
                    // **When the view is a copy, the producer writes it.** With
                    // W and H multiples of four the padded raster is the raster
                    // (rw*rh == W*H), so every destination token below H rows is
                    // the source token of the same index and channel, and the
                    // rest - the window padding past the source's tokens - is
                    // zero. The layer before then stores straight into the view
                    // slot, whose tail stays the zero it was allocated as: it is
                    // pinned for the frame (overread), so no other value shares
                    // those bytes. 1080p and 4K both take this; other extents
                    // keep the adapter.
                    int producer = -1;
                    if (NR_UPSVIEW_FOLD && pv.W % 4u == 0u && pv.H % 4u == 0u && pv.M >= pv.W * pv.H &&
                        !flag(argc, argv, "--no-upsview-fold") && !slot_reading_diagnostic(argc, argv)) {
                        for (int k = int(disp.size()) - 1; k >= 0 && producer < 0; --k) {
                            const Disp& q = disp[size_t(k)];
                            if ((q.kern == "gemmproj" || q.kern == "gemmprojc" || q.kern == "gemmproj5") && q.push.size() == sizeof(PushGemm)) {
                                PushGemm g; std::memcpy(&g, q.push.data(), sizeof g);
                                if (g.o_off == pv.x_off && g.M == pv.W * pv.H) producer = k;
                            } else if (q.kern == "fswin" + std::to_string(2 * C) && q.push.size() == sizeof(PushFSwin)) {
                                PushFSwin f; std::memcpy(&f, q.push.data(), sizeof f);
                                if (f.o_off == pv.x_off) producer = k;
                            }
                        }
                    }
                    if (producer >= 0) {
                        Disp& q = disp[size_t(producer)];
                        if (q.kern == "gemmproj" || q.kern == "gemmprojc" || q.kern == "gemmproj5") {
                            PushGemm g; std::memcpy(&g, q.push.data(), sizeof g);
                            g.o_off = pv.o_off; std::memcpy(q.push.data(), &g, sizeof g);
                        } else {
                            PushFSwin f; std::memcpy(&f, q.push.data(), sizeof f);
                            f.o_off = pv.o_off; std::memcpy(q.push.data(), &f, sizeof f);
                        }
                        const int vk = ups_key(s.block);
                        overread[vk] = std::max(overread[vk], std::max(vsize[vk], size_t(pv.M) * pv.C));
                    } else {
                    dv.kern = "upsview";
                    dv.gx = (pv.M * (NR_UPSVIEW_VEC ? pv.C / 16u : pv.C) + 63u) / 64u; dv.gy = dv.gz = 1;
                    dv.push.resize(sizeof pv);
                    std::memcpy(dv.push.data(), &pv, sizeof pv);
                    disp.push_back(std::move(dv));
                    }
                }
                // **The prologue goes first**, which is the whole difference
                // from `_ds`: project, replicate, blend the skip, then the body.
                //
                // The projection is the mirror of `_ds`'s - `resample.bin` is
                // [C][2C] here where `_ds` has [2C][C], the same 2048 bytes at
                // C=32 read the other way round - and it has no gather in it, so
                // it is an ordinary GEMM at the input's own resolution.
                Disp dg{&s};
                PushGemm pg{};
                const std::vector<uint8_t> rs = slurp(std::string(base) + ".resample.bin");
                if (rs.size() < size_t(2 * C) * size_t(C)) {
                    std::fprintf(stderr, "b%dl%d: resample is %zu B, need %d\n",
                                 s.block, s.layer, rs.size(), 2 * C * C);
                    return 1;
                }
                // `_ds` stores its resample `[2C][C]`; this one is the mirror
                // and should be `[C][2C]`. Square it is not, so the byte count
                // cannot tell them apart - `--ups-transpose` measures it.
                pg.w_off = flag(argc, argv, "--ups-transpose")
                    ? uint32_t(put(tin::tile_blocked(
                          [&] {
                              std::vector<uint8_t> t(size_t(C) * size_t(2 * C));
                              for (size_t n = 0; n < size_t(C); ++n)
                                  for (size_t k = 0; k < size_t(2 * C); ++k)
                                      t[n * size_t(2 * C) + k] = rs[k * size_t(C) + n];
                              return t;
                          }().data(), size_t(C), size_t(2 * C)), 256))
                    : uint32_t(put(tin::tile_blocked(rs.data(), size_t(C),
                                                     size_t(2 * C)), 256));
                pg.x_off = uint32_t(voff[ups_view && wide_ups_fused_mode!=3 ? ups_key(s.block) : x_src(s)]);
                pg.o_off = uint32_t(voff[pool_key(s.block)]);
                pg.M = uint32_t(s.tokens);
                pg.N = uint32_t(C);
                pg.K = uint32_t(2 * C);
                pg.W = uint32_t(s.W);
                // **N is the block's width and can be 32**, which the `_ds`
                // resample's 64-wide N tile does not divide - hence a variant of
                // its own at 16x32 rather than reusing `gemmds`.
                dg.kern = host_boundary ? "gemmups16" : "gemmups";
                if(host_boundary)pg.o_off/=2u;
                dg.gx = pg.M / 16u; dg.gy = pg.N / 32u; dg.gz = 1;
                dg.push.resize(sizeof pg);
                std::memcpy(dg.push.data(), &pg, sizeof pg);
                // Per-width choice. An unfused wide body is a plain fswin{C}
                // and joins the following persistent run; the fused one cannot.
                const uint32_t wide_bit = C==64 ? 1u : C==128 ? 2u : C==256 ? 4u : 0u;
                const uint32_t up_mode=C==32?ups_fused_mode:
                    ((uint32_t(NR_WIDE_UPS_MASK) & wide_bit) ? wide_ups_fused_mode : 0u);
                const bool fused_up=up_mode!=0;
                if(!(fused_up && up_mode>=2))disp.push_back(std::move(dg));

                Disp db{&s};
                PushUps pu{};
                // **The gain region is C f16 only at C=32.** `unpack_swin_family.py`
                // sizes it `2*C` bytes at C=32 and `2*C - 32` above that - a
                // byte-accounting adjustment made to close the whole record, not
                // a measurement of this array - so the unpacked file is sixteen
                // entries short for C=64, 128 and 256. Those channels get a zero
                // gain here, which drops the skip for them rather than inventing
                // a value; b066 is C=32 and unaffected, and it is the one with a
                // gold (`upsample_c32.bin`), so the
                // implementation can be validated while the boundary is open.
                // **The gain's first sixteen entries live at the tail of the
                // `residual_scale` region.** `unpack_swin_family.py` sizes that
                // region `2*C + 32` bytes and this one `2*C - 32`, and the
                // arithmetic closes exactly: 16 + 48 = 64, 16 + 112 = 128,
                // 16 + 240 = 256, while at C=32 the region's extra sixteen are
                // the zero padding and the gain file is already complete. So the
                // boundary is off by sixteen scalars for every width above 32,
                // and b066 - the one width where it is not - scores r=0.9986
                // while the other three sit at 0.16 to 0.70.
                std::vector<float> gn = load_f16(std::string(base) + ".upsample_gain.bin");
                if (gn.size() < size_t(C)) {
                    const std::vector<float> tail = load_f16(std::string(base) +
                                                            ".residual_scale.bin");
                    const size_t need = size_t(C) - gn.size();
                    if (tail.size() >= need) {
                        // Which end the recovered sixteen belong to is a
                        // measurement, not a reading: the region boundary says
                        // only that they are adjacent. b066 is unaffected - its
                        // gain file is already whole - so this is exactly the
                        // untested piece the other three widths depend on.
                        std::vector<float> full;
                        if (flag(argc, argv, "--rs-last")) {
                            // Pairs with the residual taking the *last* C: then
                            // the sixteen at the front are the gain's.
                            full.assign(tail.begin(), tail.begin() + long(need));
                            full.insert(full.end(), gn.begin(), gn.end());
                        } else if (flag(argc, argv, "--gain-tail-last")) {
                            full = gn;
                            full.insert(full.end(), tail.end() - long(need), tail.end());
                        } else {
                            full.assign(tail.end() - long(need), tail.end());
                            full.insert(full.end(), gn.begin(), gn.end());
                        }
                        gn.swap(full);
                    }
                }
                if (gn.size() < size_t(C)) {
                    std::printf("b%dl%d: upsample_gain still short, %zu of %d\n",
                                s.block, s.layer, gn.size(), C);
                    gn.resize(size_t(C), 0.0f);
                }
                pu.p_off = uint32_t(voff[pool_key(s.block)]);
                pu.s_off = uint32_t(voff[skip_key(s.in1 >= 0 ? s.in1 : s.block)]);
                pu.o_off = uint32_t(voff[ups_key(s.block)]);
                pu.g_off = uint32_t(put_f16(std::vector<float>(gn.begin(),
                                                              gn.begin() + C)) / 2);
                // **The output grid is the window grid, `2gx x 2gy`, not
                // `tiles_of`.** Both the skip this kernel reads and the body it
                // writes for are dispatched over `gx x gy` windows of 8x8
                // pixels, so their row stride is `2gx` tiles - the skip scores
                // r=0.9999 against the capture on a *linear* token, which is
                // only possible if our arena holds it in NVIDIA's own grid, and
                // NVIDIA's is `2gx` because `tokens == 64*gx*gy` at every level.
                // `tiles_of` of the output extent is one or two tiles short of
                // that wherever the extent does not divide by 8, and the blend
                // then shears the skip one tile per tile row. The drift rate is
                // the ladder the four `_upsample` layers scored on:
                //
                //     block   out extent   tiles_of   2gx   drift    r was
                //       48       67x120       16       18   2/16    0.1411
                //       56      135x240       33       34   1/33    0.6932
                //       62      270x480       67       68   1/67    0.8205
                //       66      540x960      135      136   1/135   0.9940
                //
                // which is monotone in the drift and in nothing else - not in
                // C, and not in the `upsample_gain` boundary this file used to
                // blame, which measures flat: 0.1411 / 0.1462 / 0.1386 for the
                // three placements of its sixteen recovered scalars.
                //
                // **And it is wrong.** `--ups-otiles-gx` measures 0.1295 /
                // 0.6683 / 0.8166 / **0.7966** - every level the same or worse,
                // and b066 loses 0.20, which is the one level with no doubt
                // about it. So the output grid is `tiles_of` after all and the
                // skip is in that grid too, exactly as the r=0.9999 linear score
                // says. The ladder is real and it is not this. Kept as a knob
                // because the drift table above is the right shape of question.
                const bool otiles_gx = flag(argc, argv, "--ups-otiles-gx");
                pu.tiles_x = otiles_gx ? uint32_t(s.gx) * 2u : tiles_of(2 * s.H);
                pu.tiles_y = otiles_gx ? uint32_t(s.gy) * 2u : tiles_of(2 * s.W);
                // **And the skip is in the output's grid, not its producer's.**
                // Forcing both to `2gx` measured worse, so this knob exists to
                // ask the other half of that question - does the *skip* alone
                // carry `2*gx`? - and the answer is no: b066 0.9940 -> 0.9442,
                // b062 0.8205 -> 0.7640, b056 0.6932 -> 0.6702, b048 0.1411 ->
                // 0.1374. Every level worse, and the one with no doubt about it
                // worst. So the whole `tokens == 64*gx*gy` reading is right
                // about NVIDIA's grid and wrong about ours: our `_ds` block
                // writes its skip at `tiles_of` like every other producer here,
                // and the capture agreeing with it at r=0.9999 means the two
                // *coincide* on the bytes that exist, not that the strides are
                // equal. `--ups-stiles <n>` sets it; 0 is the output's.
                pu.stiles_x = uint32_t(std::atoi(arg(argc, argv, "--ups-stiles", "0")));
                // **The projected input's row stride is the tensor's, not the
                // extent's.** It carries `tokens/16` tiles over `tiles_of(W)`
                // rows, so the tiles per row is that quotient - 68 at level 2,
                // where `tiles_of(270)` truncates to 67 and shears every row by
                // one tile. `--ups-itiles` overrides it to measure the claim.
                pu.itiles_y = tiles_of(s.W);
                // **`gx`, the window grid - the same law the `_ds` seam turned
                // out to obey.** A halved tensor's row stride in tiles is the
                // producer's window grid, and the `tokens/16/tiles_of(W)`
                // quotient this used to compute only *coincides* with it at the
                // shallow levels:
                //
                //     block   quotient   tiles_of(H)   gx     r was
                //       48       10           8        *9*   0.2646
                //       56       18          16       *17*   0.7193
                //       62       34          33        34    0.8322
                //       66       68          67        68    0.9986
                //
                // The two where they differ are exactly the two that failed -
                // **and changing it measures flat**: 0.2604 against 0.2646 and
                // 0.7044 against 0.7193. The standalone harness cannot test this
                // question, because it seeds the input by size and both sides
                // then use our layout. `gx` is kept because it is the law the
                // `_ds` seam turned out to obey and the quotient only coincides
                // with it; the b048/b056 defect is something else, and
                // [[standalone-golds-unblock-the-decoder]] lists what is already
                // falsified for it.
                //
                // **And all three of those readings were the wrong question.**
                // `itiles_x` is not "what stride does the tensor have", it is
                // "what stride did *the dispatch that wrote it* use", and every
                // producer in this program uses `tiles_of`. The plan settles the
                // first question on its own - `tokens == 64 * gx * gy` holds at
                // all six levels, so NVIDIA's tile grid is exactly `2gx x 2gy`
                // and the row stride is `2*gx`:
                //
                //     div     H    tiles_of(H)   2*gx
                //       1  1080       270         270      (they agree)
                //       2   540       135         136
                //       4   270        67          68
                //       8   135        33          34
                //      16    67        16          18
                //      32    33         8          10
                //
                // - but ours is `tiles_of` from `img_in` down, which loses the
                // last 2 pixels of H at each level *consistently*, and a
                // consistent truncation is a crop, not a shear. Reading the
                // input at `gx` while it was written at `tiles_of` is a shear,
                // and it is the one in the picture: at block 66 the producer
                // (b065, div 4) writes 67 tiles a row and this read 68, so the
                // content moves one tile left per tile row and wraps after 67 -
                // **16 full-resolution pixels of drift per 16 rows, period 1072
                // px**, which is exactly the five black diagonals the frame had.
                // `--ups-itiles-gx` puts `s.gx` back.
                pu.itiles_x = flag(argc, argv, "--ups-itiles-gx")
                                  ? uint32_t(s.gx) : tiles_of(s.H);
                if (const char* it = arg(argc, argv, "--ups-itiles"))
                    pu.itiles_x = uint32_t(std::atoi(it));
                if(host_boundary) {
                    const auto extent=up_extent(s);pu.tiles_x=tiles_of(extent.second);pu.tiles_y=tiles_of(extent.first);
                    pu.itiles_x=tiles_of(s.H);pu.itiles_y=tiles_of(s.W);
                    for(const Step& enc:plan)if(enc.block==s.in1){pu.stiles_x=tiles_of(enc.H);break;}
                }
                pu.mode = uint32_t(std::atoi(arg(argc, argv, "--ups-blend", "0")));
                db.kern = (host_boundary ? "upsblend16" : "upsblend") + std::to_string(C);
                db.gx = pu.tiles_x; db.gy = pu.tiles_y; db.gz = 1;
                db.push.resize(sizeof pu);
                std::memcpy(db.push.data(), &pu, sizeof pu);
                if(fused_up) {
                    d.kern="fswinfusedup"+std::to_string(C);
                    if(up_mode>=2)pu.p_off=pg.x_off;
                    // The fused view reuses two otherwise-unused blend fields:
                    // o_off is source raster W, mode is H; zero W means identity.
                    if(up_mode==3){pu.o_off=ups_view?uint32_t(s.H):0;pu.mode=ups_view?uint32_t(s.W):0;}
                    d.push.resize(sizeof(PushFSwin)+sizeof(PushUps)+(up_mode>=2?4:0));
                    std::memcpy(d.push.data()+sizeof(PushFSwin),&pu,sizeof pu);
                    if(up_mode>=2)std::memcpy(d.push.data()+sizeof(PushFSwin)+sizeof(PushUps),&pg.w_off,4);
                } else disp.push_back(std::move(db));

                disp.push_back(std::move(d));     // the body, last
                continue;
            }
            if (sh.fam == F_DS) {
                const size_t ds_body = disp.size();
                disp.push_back(std::move(d));
                // 2x2 mean. One workgroup per window, and a window is exactly
                // one output tile: 8x8 pixels halve to 4x4, which is a tile.
                Disp dp{&s};
                PushPool pp{};
                pp.x_off = uint32_t(voff[skip_key(s.block)]);
                pp.o_off = uint32_t(voff[pool_key(s.block)]);
                pp.tiles_x = g_grid_tiles ? uint32_t(s.gx) * 2u : tiles_of(s.H);
                pp.tiles_y = g_grid_tiles ? uint32_t(s.gy) * 2u : tiles_of(s.W);
                // **The half-resolution row is `pad4` tiles, not `tiles_of`.**
                // 33 columns do not fit in 8 tiles; the ninth is where the
                // capture's smooth norm image says it is.
                // **Ours and theirs are two different numbers.** The capture's
                // raster is `4*gx` (the kernel's own store address); our arena's
                // is whatever we choose, and the scorer maps a pixel through
                // each side's own stride. `--ds-ow <VW>:<tiles>` sets ours.
                pp.otiles_x = flag(argc, argv, "--ds-pixel-stride")
                                  ? tiles_of(s.H / 2)
                                  : uint32_t(ds_tiles(size_t(s.H) / 2, s.gx));
                // The producer's padded raster against the consumer's own
                // extent - see the push block in `pool2x2.comp`. **Measured and
                // off by default**: it is worth +0.03 at b015 and -0.03 at
                // b022, and it costs b014's own view score 0.96 -> 0.44 because
                // the scorer reads the capture at the *producer's* raster. The
                // hypothesis it tests is right in shape - the two deepest
                // levels are exactly where the chain breaks - and wrong in
                // detail. `--ds-reindex` turns it on.
                {
                    size_t Wp, Hp;
                    ds_raster(s.gx, s.gy, size_t(s.H) / 2, size_t(s.W) / 2, Wp, Hp);
                    // The consumer's row **is** the raster now that the plan's
                    // deep levels round up with the kernel - 68 and 36, not 67
                    // and 33 - so these are the same number and the reindex is
                    // the identity. `--ds-reindex` restores the composition
                    // against a consumer that reads its unpadded half extent.
                    // **On by default**: the producer's raster and the
                    // consumer's row are 68 and 67 at level 5, 36 and 33 at
                    // level 6, and `pool2x2.comp` composes them per
                    // 16-channel plane. `--ds-flat` ignores the difference,
                    // which is what shipped while the seam was open.
                    pp.raster = uint32_t(Wp);
                    pp.crow = flag(argc, argv, "--ds-flat")
                                  ? pp.raster : uint32_t(size_t(s.H) / 2);
                }
                if(host_boundary) { pp.otiles_x=uint32_t(pad4((s.H+1)/2)/4);pp.raster=pp.crow=pp.otiles_x*4; }
                dp.kern = "pool" + std::to_string(C);
                dp.gx = uint32_t(s.gx); dp.gy = uint32_t(s.gy); dp.gz = 1;
                dp.push.resize(sizeof pp);
                std::memcpy(dp.push.data(), &pp, sizeof pp);
                if (!host_boundary || flag(argc,argv,"--legacy-ds-quant"))
                    disp.push_back(std::move(dp));
                // The learned [2C][C] resample. A small GEMM, so it takes the
                // narrowest tile - 16x64 - which divides every shape here
                // without a per-width pipeline: M is a multiple of 16 because it
                // is windows x 16, and N = 2C is a multiple of 64 from C >= 32.
                Disp dg{&s};
                PushGemm pg{};
                const std::vector<uint8_t> rs =
                    slurp(std::string(base) + ".resample.bin");
                if (rs.size() < size_t(2 * C) * size_t(C)) {
                    std::fprintf(stderr, "b%dl%d: resample is %zu B, need %d\n",
                                 s.block, s.layer, rs.size(), 2 * C * C);
                    return 1;
                }
                pg.w_off = uint32_t(put(tin::tile_blocked(rs.data(), size_t(2 * C),
                                                          size_t(C)), 256));
                pg.x_off = uint32_t(voff[pool_key(s.block)]);
                pg.o_off = uint32_t(voff[key_of(s.block, s.layer)]);
                pg.M = uint32_t(s.tokens / 4);
                pg.N = uint32_t(2 * C);
                pg.K = uint32_t(C);
                pg.W = uint32_t(s.W);
                {   // The shear the `_inpview` consumer expects - see the store
                    // in `gemm1x1.comp`. `--ds-flat` turns it off.
                    size_t Wp, Hp;
                    ds_raster(s.gx, s.gy, size_t(s.H) / 2, size_t(s.W) / 2, Wp, Hp);
                    pg.otx = uint32_t(ds_tiles(size_t(s.H) / 2, s.gx));
                    pg.rows = uint32_t(tiles_of(s.W) / 2) * 4u;
                    pg.crow = uint32_t(size_t(s.H) / 2);
                    pg.raster = flag(argc, argv, "--ds-flat") ? pg.crow : uint32_t(Wp);
                    pg.ds_raster = flag(argc, argv, "--ds-tok-raster") ? 1u : 0u;
                }
                if(host_boundary) {pg.otx=uint32_t(pad4((s.H+1)/2)/4);pg.crow=pg.raster=pg.otx*4;pg.rows=uint32_t(pad4((s.W+1)/2));
                    if(C==32){
                        pg.writer_rows=uint32_t(s.W/2);pg.raster=uint32_t(s.H/2);
                        // Native image evaluation aligns the working extent
                        // before building this graph. Keep the native view
                        // contract; the provisional spatial change is now
                        // an explicit diagnostic for unaligned standalone plans.
                        if (flag(argc, argv, "--spatial-c32-diagnostic") &&
                            !flag(argc, argv, "--replay-c32-view")) pg.ds_raster=2;
                        if (pg.writer_rows!=pg.rows || pg.raster!=pg.crow)
                            std::printf("C32 DS boundary: %s, actual %ux%u, arena %ux%u\n",
                                pg.ds_raster==2 ? "spatial-v1" : "standalone-view-replay",
                                pg.raster, pg.writer_rows, pg.crow, pg.rows);
                    } else if (!flag(argc, argv, "--keep-ds-padding")) {
                        // Zero the view's padding as the original does; see the
                        // store in gemm1x1.comp. `--keep-ds-padding` restores the
                        // old behaviour for comparison.
                        pg.clear_x = uint32_t((s.H + 1) / 2);
                        pg.clear_y = uint32_t((s.W + 1) / 2);
                    }
                }
                dg.kern = "gemmds";
                dg.gx = pg.M / 16u; dg.gy = pg.N / 64u; dg.gz = 1;
                dg.push.resize(sizeof pg);
                std::memcpy(dg.push.data(), &pg, sizeof pg);
                // The projection runs inside the pooled body instead.
                const uint32_t ds_bit = C == 32 ? 1u : C == 64 ? 2u : C == 128 ? 4u : C == 256 ? 8u : 0u;
                if ((uint32_t(NR_DS_FUSE) & ds_bit) && host_boundary && pg.ds_raster != 1u &&
                    disp[ds_body].kern == "fswinds" + std::to_string(C) &&
                    !flag(argc, argv, "--legacy-ds-quant")) {
                    Disp& body = disp[ds_body];
                    PushDsProj dq{pg.w_off, pg.o_off, pg.otx, pg.raster, pg.crow, pg.rows,
                                  pg.ds_raster, pg.writer_rows, pg.N, pg.clear_x, pg.clear_y};
                    PushFSwin bp{}; std::memcpy(&bp, body.push.data(), sizeof bp);
                    if (bp.pool_tiles_x != pg.otx)
                        throw std::runtime_error("ds fusion: pooled grid != gemmds otx");
                    body.kern = "fswindsp" + std::to_string(C);
                    body.push.resize(sizeof(PushFSwin) + sizeof dq);
                    std::memcpy(body.push.data() + sizeof(PushFSwin), &dq, sizeof dq);
                    continue;
                }
                disp.push_back(std::move(dg));
                continue;
            }
        } else if (sh.fam == F_DECUPS) {
            // Legacy plans use the old same-resolution approximation. A host
            // plan records output dimensions here and takes the input extent
            // from b38; retain FP16 until upsampling and adding the real skip.
            std::snprintf(base, sizeof base, "%s/unpacked-splitswin/block%d.layer%d.layer",
                          unp.c_str(), s.block, s.layer);
            const size_t N = size_t(sh.N), K = size_t(sh.K);
            const std::vector<uint8_t> w = slurp(std::string(base) + ".weight.bin");
            if (w.size() < N * K) {
                std::fprintf(stderr, "b%dl%d %s: weight is %zu B, need %zu\n",
                             s.block, s.layer, s.type.c_str(), w.size(), N * K);
                return 1;
            }
            PushGemm p{};
            p.w_off = uint32_t(put(tin::tile_blocked(w.data(), N, K), 256));
            const std::vector<float> g = load_f16(std::string(base) + ".skip_weight.bin");
            if (g.size() < N) {
                std::fprintf(stderr, "b%dl%d: skip_weight is %zu, need %zu\n",
                             s.block, s.layer, g.size(), N);
                return 1;
            }
            const std::vector<float> gn(g.begin(), g.begin() + long(N));
            p.g_off = uint32_t(put_f16(gn) / 2);
            std::vector<float> gd(N / 16 * 256, 0.0f);
            for (size_t n = 0; n < N; ++n)
                gd[(n / 16) * 256 + (n % 16) * 16 + (n % 16)] = gn[n];
            p.gd_off = uint32_t(put_f16(gd) / 2);
            p.r_off = uint32_t(voff[skip_key(s.in1 >= 0 ? s.in1 : s.block)]);
            p.x_off = uint32_t(voff[x_src(s)]);
            p.o_off = uint32_t(voff[key_of(s.block, s.layer)]);
            p.M = uint32_t(s.tokens);
            p.N = uint32_t(N);
            p.K = uint32_t(K);
            p.W = uint32_t(s.W);
            d.kern = "gemmvproj";
            d.gx = (p.M + NR_GEMM_WIDE_MT - 1u) / NR_GEMM_WIDE_MT;
            d.gy = p.N / NR_GEMM_WIDE_NT; d.gz = 1;
            if (host_boundary) {
                const Step* prev=nullptr;
                for (const Step& candidate : plan)
                    if(candidate.block==s.in0 && candidate.layer==4) prev=&candidate;
                if(!prev) { std::fprintf(stderr,"decoder requires preceding projection dimensions\n");return 1; }
                p.M=uint32_t(prev->W*prev->H);
                p.o_off=uint32_t(voff[pool_key(s.block)]/2);
                d.kern=NR_DECQ_SMALL ? "gemmvqkvs" : "gemmvqkv";
                d.gx=NR_DECQ_SMALL ? (p.M+NR_GEMM_PROJ_MT-1u)/NR_GEMM_PROJ_MT
                                   : (p.M+NR_GEMM_WIDE_MT-1u)/NR_GEMM_WIDE_MT;
                if(NR_DECQ_SMALL) {
                    if(p.N%NR_GEMM_PROJ_NT) { std::fprintf(stderr,"gemmvqkvs: N %u vs tile %u\n",p.N,unsigned(NR_GEMM_PROJ_NT)); return 1; }
                    d.gy=p.N/NR_GEMM_PROJ_NT;
                }
                d.push.resize(sizeof p);std::memcpy(d.push.data(),&p,sizeof p);
                disp.push_back(std::move(d));
                PushDecoderUps pu{p.o_off,uint32_t(voff[key_of(s.in1,3)]),
                    uint32_t(voff[key_of(s.block,s.layer)]),p.g_off,
                    uint32_t(prev->H),uint32_t(s.H),uint32_t(s.W)};
                Disp up{&s};up.kern="decups";// NR_DECUPS_VEC=16: whole 4x4 token tiles take the tile form, anything
                // else the shader's own per-element fallback - the same test on both sides.
                const bool dtile=NR_DECUPS_VEC==16 && pu.OW%4u==0u && pu.OH%4u==0u;
                up.gx=dtile ? (pu.OW*pu.OH/16u*1024u+63u)/64u
                            : NR_DECUPS_VEC==16 ? (pu.OW*pu.OH*512u+63u)/64u
                            : (pu.OW*pu.OH*(512/uint32_t(NR_DECUPS_VEC))+63)/64;
                up.gy=up.gz=1;up.push.resize(sizeof pu);std::memcpy(up.push.data(),&pu,sizeof pu);
                disp.push_back(std::move(up));continue;
            }
            d.push.resize(sizeof p);
            std::memcpy(d.push.data(), &p, sizeof p);
        } else if (sh.fam == F_VATTN) {
            // **The sequence length is `W * H`, not the buffer's token count.**
            // The shipping kernel derives it exactly that way - `IMAD R115,
            // R115, R114, RZ` on the u32 pair at +56/+60 - and the buffer holds
            // 2560 tokens where only 1980 are real, which is why these tensors
            // read 22-38% zero.
            PushVAttn p{};
            p.x_off = uint32_t(voff[x_src(s)]);
            p.o_off = uint32_t(voff[key_of(s.block, s.layer)]);
            p.tokens = uint32_t(s.W) * uint32_t(s.H);
            // `--vattn-tokens` shortens the sequence, which is what running this
            // layer against the 1024-token oracle capture needs: the attention's
            // result depends on *which* keys are in the sum, so padding the
            // oracle out to the frame's 1980 and letting 956 zero keys into the
            // softmax measures a different function.
            if (vattn_tokens) p.tokens = vattn_tokens;
            // **The per-head temperature lives in the producing layer's record,
            // not this one's.** `CCVit1DAttention`'s own weight blob is two
            // bytes; the 32 f32 scales are the first 128 bytes of the
            // `CCVit1DQKV` record one layer up, where NVIDIA folds them into Q
            // in the epilogue and we apply them at the logit instead.
            std::snprintf(base, sizeof base,
                          "%s/inventory/weights/block%d.layer%d.layer.bin",
                          unp.c_str(), s.block, s.layer - 1);
            const std::vector<uint8_t> qrec = slurp(base);
            if (qrec.size() < 128) {
                std::fprintf(stderr, "b%dl%d: QKV record is %zu B, need >= 128\n",
                             s.block, s.layer, qrec.size());
                return 1;
            }
            std::vector<float> hscale(32);
            std::memcpy(hscale.data(), qrec.data(), 128);
            p.s_off = uint32_t(put_f32(hscale) / 4);
            p.mode = vattn_mode & ~1u; // QKV has already normalized and scaled Q/K
            d.kern = "vitattn";
            // One workgroup per (**tile of 32 query tokens**, head). One query
            // per workgroup was correct and read every byte of K and V 1980
            // times, which is 64 GB across the eight layers and was 121 ms of a
            // 132.8 ms frame. The head count is the plan's own column - 4 - not
            // the 32 an earlier reading took from the shipping launch's
            // `gridDim.x`.
            // **Must match `NR_QT` in vit_attn.comp**, which the shader build
            // sets from the same constant. A mismatch is a kernel that quietly computes a
            // fraction of the sequence, so both ends are a knob rather than a
            // constant on one side.
            d.gx = (p.tokens + vattn_qt - 1u) / vattn_qt;
            d.gy = 32u;   // 32 heads of dim 32; the plan's `heads` is blockDim.y
            d.gz = 1;
            d.push.resize(sizeof p);
            std::memcpy(d.push.data(), &p, sizeof p);
        } else if (sh.fam == F_FFWD3) {
            // One record blob, three bit-scattered matrices, eight groups.
            std::snprintf(base, sizeof base, "%s/inventory/weights/block%d.layer%d.layer.bin",
                          unp.c_str(), s.block, s.layer);
            const std::vector<uint8_t> rec = slurp(base);
            if (rec.size() < 524288) {
                std::fprintf(stderr, "b%dl%d: record is %zu B, need 524288\n",
                             s.block, s.layer, rec.size());
                return 1;
            }
            std::vector<uint8_t> A(FF_GROUPS * FF_R * FF_K),
                                 Q0(FF_GROUPS * FF_J * FF_R),
                                 Q2(FF_GROUPS * FF_OUT * FF_J);
            for (size_t g = 0; g < FF_GROUPS; ++g) {
                for (size_t r = 0; r < FF_R; ++r)
                    for (size_t k = 0; k < FF_K; ++k)
                        A[(g * FF_R + r) * FF_K + k] = rec[ff_a(int(g), int(r), int(k))];
                for (size_t j = 0; j < FF_J; ++j)
                    for (size_t r = 0; r < FF_R; ++r)
                        Q0[(g * FF_J + j) * FF_R + r] = rec[ff_q0(int(g), int(j), int(r))];
                for (size_t n = 0; n < FF_OUT; ++n)
                    for (size_t j = 0; j < FF_J; ++j)
                        Q2[(g * FF_OUT + n) * FF_J + j] = rec[ff_q2(int(g), int(n), int(j))];
            }
            PushFfwd3 p{};
            // The kernel indexes each group at a fixed stride, so the three
            // tensors are contiguous per group and the offsets are the bases.
            std::vector<uint8_t> ab, q0b, q2b;
            for (size_t g = 0; g < FF_GROUPS; ++g) {
                const std::vector<uint8_t> ta =
                    tin::tile_blocked(&A[g * FF_R * FF_K], FF_R, FF_K);
                const std::vector<uint8_t> t0 =
                    tin::tile_blocked(&Q0[g * FF_J * FF_R], FF_J, FF_R);
                const std::vector<uint8_t> t2 =
                    tin::tile_blocked(&Q2[g * FF_OUT * FF_J], FF_OUT, FF_J);
                ab.insert(ab.end(), ta.begin(), ta.end());
                q0b.insert(q0b.end(), t0.begin(), t0.end());
                q2b.insert(q2b.end(), t2.begin(), t2.end());
            }
            p.a_off  = uint32_t(put(ab, 256));
            p.q0_off = uint32_t(put(q0b, 256));
            p.q2_off = uint32_t(put(q2b, 256));
            p.x_off = uint32_t(voff[x_src(s)]);
            p.o_off = uint32_t(voff[key_of(s.block, s.layer)]);
            // **The domain is the tile grid, not the window grid.** NVIDIA's
            // store is `tile = row*(W/4) + 2*ctaid.x`, guarded by `row < H/4`
            // and `2*ctaid.x < W/4`, so it writes `tiles_of(H) * tiles_of(W)`
            // tiles - 120 of the slot's 160 at level 6 - and leaves the window
            // grid's padding at zero. Writing all 160 is right values over
            // twice the domain, which scores exactly 1/sqrt(2):
            // [[the-bottleneck-writes-half-its-tokens]].
            const uint32_t vt = flag(argc, argv, "--ffwd-full")
                ? uint32_t(s.tokens)
                : tiles_of(s.H) * tiles_of(s.W) * 16u;
            p.M = vt; p.C = 512u;
            const bool fm2 = NR_FFWD_FM2_MIN_TOKENS != 0 && vt >= uint32_t(NR_FFWD_FM2_MIN_TOKENS);
            d.kern = fm2 ? "ffwd3w" : "ffwd3";
#if NR_FFWD_GMAJOR
            // Group-major workgroups (ffwd3_t.comp NR_FFWD_GMAJOR): eight groups
            // times the token units split NR_FFWD_WGW a workgroup, the last one partial.
            {
                const uint32_t units = (vt / 16u + (fm2 ? 1u : 0u)) / (fm2 ? 2u : 1u);
                d.gx = 8u * ((units + uint32_t(NR_FFWD_WGW) - 1u) / uint32_t(NR_FFWD_WGW));
            }
#else
            d.gx = (vt / 16u + (fm2 ? 1u : 0u)) / (fm2 ? 2u : 1u) * (8u / uint32_t(NR_FFWD_WGW));
#endif
            d.gy = 1; d.gz = 1;
            d.push.resize(sizeof p);
            std::memcpy(d.push.data(), &p, sizeof p);
        } else if (sh.fam == F_ATTN) {
            std::snprintf(base, sizeof base, "%s/unpacked-splitswin/block%d.layer%d.layer",
                          unp.c_str(), s.block, s.layer);
            PushAttn p{};
            const std::vector<uint8_t> qkv = slurp(std::string(base) + ".qkv.bin");
            if (qkv.size() < 1536u * 512u) {
                std::fprintf(stderr, "b%dl%d: qkv is %zu B\n", s.block, s.layer, qkv.size());
                return 1;
            }
            p.w_off = uint32_t(put(tin::tile_blocked(qkv.data(), 1536, 512), 256));
            {
                const std::vector<float> raw = load_f16(std::string(base) + ".attn_pos_bias.bin");
                std::vector<float> bias(16u * 4096u);
                for (int h = 0; h < 16; ++h) {
                    const std::vector<float> one(raw.begin() + size_t(h) * 4096,
                                                 raw.begin() + size_t(h + 1) * 4096);
                    const std::vector<float> dz = deswizzle_bias(one);
                    std::copy(dz.begin(), dz.end(), bias.begin() + size_t(h) * 4096);
                }
                p.b_off = uint32_t(put_f32(bias) / 4);
            }
            {
                const std::vector<uint8_t> tail = slurp(std::string(base) + ".tail.bin");
                std::vector<float> sc(16, 1.0f);
                for (size_t h = 0; h < 16 && 4 * h + 4 <= tail.size(); ++h)
                    std::memcpy(&sc[h], &tail[4 * h], 4);
                p.s_off = uint32_t(put_f32(sc) / 4);
            }
            p.x_off = uint32_t(voff[x_src(s)]);
            p.o_off = uint32_t(voff[key_of(s.block, s.layer)]);
            p.C = 512u;
            p.wins_x = uint32_t(s.gx);
            p.tiles_x = tiles_of(s.H);
            p.tiles_y = tiles_of(s.W);
            p.shift = s.shift_x == 100 ? (s.shifted ? 1 : 0) : s.shift_x;
            p.shift_y = s.shift_y == 100 ? p.shift : s.shift_y;
            d.kern = "attn";
            // gridZ is the head split: NR_HSPLIT=16 head groups, and the kernel
            // strides its head loop by gl_NumWorkGroups.z so any divisor is
            // correct. 16 is 3.33x over one.
            d.gx = uint32_t(s.gx); d.gy = uint32_t(s.gy);
            d.gz = std::getenv("NR_ATTN_GZ") ? uint32_t(std::atoi(std::getenv("NR_ATTN_GZ"))) : 16u;  // diagnostic
            d.push.resize(sizeof p);
            std::memcpy(d.push.data(), &p, sizeof p);
        } else {
            // NR_REPACK_FOLD: FinalHead (gemmnores, NR_OUT_PERM=2) writes the
            // raster copy itself; the repack below only runs when it cannot.
            bool in_folded = false;
            if (NR_REPACK_FOLD && host_boundary && s.block == 31 && s.layer == 0 &&
                s.W % 4 == 0 && s.H % 4 == 0 && !flag(argc, argv, "--no-repack-fold") &&
            !slot_reading_diagnostic(argc, argv)) {
                for (auto it = disp.rbegin(); it != disp.rend(); ++it)
                    if (it->s->block == 30 && it->s->layer == 4 && it->kern == "gemmnores") {
                        PushGemm q{};
                        std::memcpy(&q, it->push.data(), sizeof q);
                        // The slot is written by FinalHead now, so its lifetime
                        // starts at FinalHead's step: record the access there,
                        // or the shared arena may overlap it with FinalHead's
                        // own live output.
                        const int lowering_now = voff.step;
                        for (size_t i = 0; i < run.size(); ++i)
                            if (run[i] == it->s) voff.step = int(i);
                        q.p_off = uint32_t(voff[lift_key(31)]);
                        voff.step = lowering_now;
                        // PushRepack's W/H are passed as s.H/s.W: the repack's raster
                        // is s.H wide and s.W tall, so the fold uses the same.
                        q.W = uint32_t(s.H); q.rows = uint32_t(s.W);
                        std::memcpy(it->push.data(), &q, sizeof q);
                        in_folded = true;
                        break;
                    }
            }
            if (host_boundary && s.block == 31 && s.layer == 0 && !in_folded) {
                Disp rep{&s};
                PushRepack p{uint32_t(voff[blk_out(30)]), uint32_t(voff[lift_key(31)]),
                             uint32_t(s.H), uint32_t(s.W), 1024, 0};
                rep.kern = "repack"; rep.gx = (p.W * p.H * (NR_REPACK_VEC ? p.C / 16u : p.C) + 63) / 64;
                rep.gy = rep.gz = 1; rep.push.resize(sizeof p);
                std::memcpy(rep.push.data(), &p, sizeof p); disp.push_back(std::move(rep));
            }
            const bool vit = s.type.rfind("CCVit1D", 0) == 0;
            std::snprintf(base, sizeof base, "%s/%s/block%d.layer%d.layer",
                          unp.c_str(), vit ? "unpacked-vit" : "unpacked-splitswin",
                          s.block, s.layer);
            PushGemm p{};
            // **The split-Swin family's domain is the tile grid**, not the
            // padded window grid - see the `ffwd3` branch above and
            // [[the-bottleneck-writes-half-its-tokens]]. The ViT and the
            // decoder's own GEMMs keep the plan's token count.
            size_t N = size_t(sh.N), K = size_t(sh.K);
            const size_t M = (s.type.rfind("CCSplitSwin16H", 0) == 0 &&
                              !flag(argc, argv, "--ffwd-full"))
                ? size_t(tiles_of(s.H)) * size_t(tiles_of(s.W)) * 16
                : size_t(s.tokens);
            // `--no-resid B:L` runs this step as a plain projection. A residual
            // that should not be there is not a small error: at b023l3 the
            // residual's rms is 11.7 against a reference output of 4.91, so
            // adding it makes the output larger than the thing it is supposed to
            // be - and a `skip_weight.bin` existing on disk does not prove the
            // kernel spends it on a residual.
            bool resid = sh.residual;
            if (const char* nr = arg(argc, argv, "--no-resid")) {
                const char* colon = std::strchr(nr, ':');
                if (std::atoi(nr) == s.block && colon && std::atoi(colon + 1) == s.layer)
                    resid = false;
            }
            // The ViT QKV is stored as three separate [C][C] tensors rather than
            // one [3C][C] `weight.bin`, so it is concatenated here in the order
            // the fused Swin family also uses - Q, then K, then V, which is what
            // `attn.comp`'s `hrow = h*96` with Q at +0, K at +32, V at +64 says.
            std::vector<uint8_t> w = slurp(std::string(base) + ".weight.bin");
            if (w.empty()) {
                for (const char* part : {".q.bin", ".k.bin", ".v.bin"}) {
                    const std::vector<uint8_t> t = slurp(std::string(base) + part);
                    w.insert(w.end(), t.begin(), t.end());
                }
            }
            if (s.type == "CCVit1DQKV") {
                char record_path[256];
                std::snprintf(record_path, sizeof record_path,
                              "%s/inventory/weights/block%d.layer%d.layer.bin",
                              unp.c_str(), s.block, s.layer);
                const auto record = slurp(record_path);
                if (record.size() < 128 + N*K) {
                    std::fprintf(stderr, "short QKV record %s\n", record_path);
                    return 1;
                }
                w.resize(N*K);
                for (size_t which=0; which<3; ++which)
                    for (size_t row=0; row<1024; ++row)
                        for (size_t k=0; k<1024; ++k)
                            w[(which*1024+row)*1024+k] = record[vit_qkv_weight_byte(which,row,k)];
            }
            if (w.size() < N * K) {
                std::fprintf(stderr, "b%dl%d %s: weight is %zu B, need %zu\n",
                             s.block, s.layer, s.type.c_str(), w.size(), N * K);
                return 1;
            }
            // `--w-transpose B:L` reads this layer's weight as [K][N]. A square
            // weight passes every size check either way round, so orientation is
            // not checkable - only measurable. The per-output-row norm statistic
            // does *not* settle it for this family: it prefers [K][N] for every
            // layer1 and layer3 alike, including the layer1 that demonstrably
            // works, so its premise (normalised output rows) does not hold here.
            if (const char* wt = arg(argc, argv, "--w-transpose")) {
                const char* colon = std::strchr(wt, ':');
                if (std::atoi(wt) == s.block && colon && std::atoi(colon + 1) == s.layer) {
                    std::vector<uint8_t> t(N * K);
                    for (size_t n = 0; n < N; ++n)
                        for (size_t k = 0; k < K; ++k) t[n * K + k] = w[k * N + n];
                    w.swap(t);
                    std::printf("b%dl%d: weight read as [K][N]\n", s.block, s.layer);
                }
            }
            p.w_off = uint32_t(put(tin::tile_blocked(w.data(), N, K), 256));
            if (resid) {
                const std::vector<float> g = load_f16(std::string(base) + ".skip_weight.bin");
                if (g.size() < N) {
                    std::fprintf(stderr, "b%dl%d: skip_weight is %zu, need %zu\n",
                                 s.block, s.layer, g.size(), N);
                    return 1;
                }
                const std::vector<float> gn(g.begin(), g.begin() + long(N));
                p.g_off = uint32_t(put_f16(gn) / 2);
                std::vector<float> gd(N / 16 * 256, 0.0f);
                for (size_t n = 0; n < N; ++n)
                    gd[(n / 16) * 256 + (n % 16) * 16 + (n % 16)] = gn[n];
                p.gd_off = uint32_t(put_f16(gd) / 2);
                p.r_off = uint32_t(voff[r_src(s)]);
            }
            p.x_off = uint32_t(voff[x_src(s)]);
            p.o_off = uint32_t(voff[key_of(s.block, s.layer)]);
            p.M = uint32_t(M); p.N = uint32_t(N); p.K = uint32_t(K);
            p.W = uint32_t(s.W);
            // The wide tile is a **shader** constant - `NR_MTILE = NR_WM *
            // NR_MFRAG * 16` and `NR_NTILE = NR_WN * NR_NFRAG * 16` in
            // gemm1x1.comp - and it was a literal here. A build that changed the
            // shader's tile therefore dispatched the *old* grid: each workgroup
            // covered less than the grid stepped by, and the rest of every
            // output was never written. That is fast and wrong, and the only
            // thing that catches it is the output `cmp` - the frame still looks
            // like a frame. The shader build passes both, and the fallbacks are
            // at the top of the file now because the SPV manifest check also
            // reads them.
            // ViT's residual projections have fewer output channels than its
            // expand/QKV products. Give them more workgroups using the existing
            // 64x128 projection pipeline; --wide-vit-proj retains the A/B path.
            const bool compact_proj = accumulation == "fp32" && resid &&
                                      !flag(argc, argv, "--wide-vit-proj");
            const bool wide = sh.wide && !compact_proj;
            const bool proj_tile = !wide && resid && !sh.act &&
                                   !(host_boundary && s.type == "CCSplitSwin16HProjPool");
            const bool projw = proj_tile && s.type == "CCVit1DFfnContract" &&
                               M >= size_t(NR_PROJW_MIN_TOKENS) && !flag(argc, argv, "--no-projw");
            const bool qkvt = wide && s.type == "CCVit1DQKV" && qkv_fused_norm &&
                              M <= size_t(NR_QKVS_MAX_TOKENS) && !flag(argc, argv, "--no-qkvs") &&
                              std::filesystem::exists(spv_dir + "/g_gemmvqkvnorms.spv");
            const uint32_t mt = qkvt ? uint32_t(NR_GEMM_QKVS_MT) : wide ? uint32_t(NR_GEMM_WIDE_MT) :
                                projw ? uint32_t(NR_GEMM_PROJW_MT) :
                                proj_tile ? uint32_t(NR_GEMM_PROJ_MT) : 64u,
                           nt = qkvt ? uint32_t(NR_GEMM_QKVS_NT) : wide ? uint32_t(NR_GEMM_WIDE_NT) :
                                projw ? uint32_t(NR_GEMM_PROJW_NT) :
                                proj_tile ? uint32_t(NR_GEMM_PROJ_NT) : 128u;
            if (N % nt) {
                std::fprintf(stderr, "b%dl%d %s: %zux%zu does not divide %ux%u\n",
                             s.block, s.layer, s.type.c_str(), M, N, mt, nt);
                return 1;
            }
            d.kern = std::string("gemm") + (wide ? "v" : "") +
                     (sh.act ? "act" : (resid ? "proj" : "nores")) + (projw ? "w" : "");
            // The C=512 projections have their own pipeline: at 1080p their grid
            // is 8.5 waves a SIMD, and the ones given the 9th also start late, so
            // gemmprojc caps residency at 8 (an LDS pad) and runs W.X^T
            // (NR_SWAP_AB). Neither pays on the ViT shapes that share gemmproj.
            // 1080p -8 us, 1440p -24, 4K -35 over the 31 dispatches.
            // Windows: off. Under LLPC gemmprojc takes 169 VGPRs (gemmproj 70) and the 31
            // dispatches cost 0.58 ms at 1080p instead of 0.32 (measured); NR_PROJC=1 for A/B.
            const bool projc = !nr::binary::directory().empty() ||
                (std::getenv("NR_PROJC") && *std::getenv("NR_PROJC") == '1');
            if (projc && d.kern == "gemmproj" && s.type.rfind("CCSplitSwin16H", 0) == 0) d.kern = "gemmprojc";
            // The C=512 projections on their own name, gemmproj5 (gemmproj's code built with the tile
            // counters and the M-major order), so the C=512 chain can take them; the ViT keeps gemmproj.
            // NR_PROJ5=0 keeps gemmproj.
            const bool proj5 = nr::binary::directory().empty() &&
                !(std::getenv("NR_PROJ5") && *std::getenv("NR_PROJ5") == '0');
            if (proj5 && d.kern == "gemmproj" && s.type.rfind("CCSplitSwin16H", 0) == 0 &&
                std::filesystem::exists(spv_dir + "/g_gemmproj5.spv")) d.kern = "gemmproj5";
            if (host_boundary && s.type == "CCSplitSwin16HProjPool") {
                d.kern = "gemmpool";
                p.p_off = uint32_t(voff[pool_key(s.block)]);
                p.W = uint32_t(s.H);
                p.rows = uint32_t(s.W);
            }
            d.gx = uint32_t((M + mt - 1) / mt); d.gy = uint32_t(N / nt); d.gz = 1;
#if NR_GEMM_REMAP_HOST
            // Two token tiles, then every output-channel tile, then the next two
            // (gemm1x1 NR_GEMM_REMAP_PC). Measured per family and extent: the
            // ViT contraction gains at every extent, the ViT projection and
            // expansion at the small ones; the ViT QKV and C=512 products keep
            // the launch order (slower under every order tried).
            if (!flag(argc, argv, "--no-gemm-remap") &&
                (s.type == "CCVit1DFfnContract" ||
                 (s.type == "CCVit1DProjection" && M < size_t(NR_PROJW_MIN_TOKENS)) ||
                 (s.type == "CCVit1DFfnExpand" && M <= 1024)))
                p.remap = 2;
#endif
            d.push.resize(sizeof p);
            std::memcpy(d.push.data(), &p, sizeof p);
        }
        if (s.type == "CCVit1DQKV") {
            PushGemm p{};
            std::memcpy(&p,d.push.data(),sizeof p);
            p.o_off=uint32_t(voff[pool_key(s.block)]/2); // FP16 element offset
            d.kern="gemmvqkv";
            std::memcpy(d.push.data(),&p,sizeof p);
            char record_path[256];
            std::snprintf(record_path,sizeof record_path,"%s/inventory/weights/block%d.layer%d.layer.bin",
                          unp.c_str(),s.block,s.layer);
            const auto rec=slurp(record_path);
            std::vector<float> scales(32);
            std::memcpy(scales.data(),rec.data(),128);
            PushQkvNorm pn{p.o_off,uint32_t(voff[key_of(s.block,s.layer)]),uint32_t(s.tokens),
                           uint32_t(put_f32(scales)/4)};
            if(qkv_fused_norm) {
                if(p.N != 3072 || p.K != 1024) throw std::runtime_error("QKV fusion: unexpected QKV shape");
                p.o_off=pn.dst; p.r_off=pn.scale_off;
                d.kern=(p.M<=uint32_t(NR_QKVS_MAX_TOKENS) && !flag(argc,argv,"--no-qkvs") &&
                        std::filesystem::exists(spv_dir + "/g_gemmvqkvnorms.spv")) ? "gemmvqkvnorms" : "gemmvqkvnorm";
                std::memcpy(d.push.data(),&p,sizeof p);
                disp.push_back(std::move(d));
                continue;
            }
            disp.push_back(std::move(d));
            Disp norm{&s}; norm.kern="qkvnorm";
            // 32 lanes own one (token, head) and a 256-thread workgroup owns
            // eight of them. The old launch was one 32-lane workgroup per
            // (token, head) - 20480 workgroups at 1080p for 5.9 MB of work.
            norm.gx=(pn.tokens*32u+7u)/8u;norm.gy=1;norm.gz=1;
            norm.push.resize(sizeof pn);std::memcpy(norm.push.data(),&pn,sizeof pn);
            disp.push_back(std::move(norm));
            continue;
        }
        // NR_REPACK_FOLD: the last ViT projection stores at the tile
        // positions the inverse repack would move its raster tokens to.
        bool out_folded = false;
        if (NR_REPACK_FOLD && host_boundary && s.block == 38 && s.layer == 4 && d.kern == "gemmproj" &&
            s.W % 4 == 0 && s.H % 4 == 0 && !flag(argc, argv, "--no-repack-fold") &&
            !slot_reading_diagnostic(argc, argv)) {
            PushGemm q{};
            std::memcpy(&q, d.push.data(), sizeof q);
            q.o_off = uint32_t(voff[ups_key(38)]);
            q.W = uint32_t(s.H); q.rows = uint32_t(s.W);   // as the repack (see b031l0)
            std::memcpy(d.push.data(), &q, sizeof q);
            d.kern = "gemmprojt";
            out_folded = true;
        }
        disp.push_back(std::move(d));
        if (host_boundary && s.block == 38 && s.layer == 4 && !out_folded) {
            Disp rep{&s};
            PushRepack p{uint32_t(voff[key_of(38,4)]), uint32_t(voff[ups_key(38)]),
                         uint32_t(s.H), uint32_t(s.W), 1024, 1};
            rep.kern = "repack"; rep.gx = (p.W * p.H * (NR_REPACK_VEC ? p.C / 16u : p.C) + 63) / 64;
            rep.gy = rep.gz = 1; rep.push.resize(sizeof p);
            std::memcpy(rep.push.data(), &p, sizeof p); disp.push_back(std::move(rep));
        }
    }
    voff.step = -1;
    {
        std::map<const Step*, int> position;
        for (size_t i = 0; i < run.size(); ++i) position[run[i]] = int(i);
        for (Disp& d : disp) d.lo = d.hi = position.at(d.s);
    }
    timer.mark("weights-unpack");
    if (!arena_probe && nr::g_memory_note) nr::logf("weights read; %s", nr::g_memory_note().c_str());
    if (!arena_probe)
        nr::logf("weight arena %.1f MB over %zu dispatches", double(wblob.size()) / 1e6, disp.size());

    // ---- one dispatch for a run of consecutive blocks ----------------------
    // Manifest v2 layouts 1/2 pair adjacent output-channel fragments so a lane
    // fetches 16 FP8 weights at once. Original files and K order stay intact.
    if (weight_layout != 0) {
        size_t packed_count=0;
        std::set<uint32_t> binary_packed_attn;
        auto pack_matrix=[&](uint32_t offset,size_t N,size_t K) {
            const size_t bytes=N*K;
            if(N%32 || K%16 || size_t(offset)+bytes>wblob.size())
                throw std::runtime_error("packed weight matrix bounds");
            std::vector<uint8_t> packed(bytes), orig(bytes);
            wblob.read(offset, orig.data(), bytes);
            for(size_t n=0;n<N/16;n+=2) for(size_t k=0;k<K/16;++k)
                for(size_t lane=0;lane<32;++lane) for(size_t j=0;j<2;++j) for(size_t c=0;c<8;++c) {
                    const size_t src=((n+j)*(K/16)+k)*256+(lane%16)*16+(lane/16)*8+c;
                    const size_t dst=((n/2)*(K/16)+k)*512+lane*16+j*8+c;
                    packed[dst]=orig[src];
                }
            wblob.write(offset, packed.data(), bytes); ++packed_count;
        };
        // Layout 2 extends layout 1 with the three grouped FFWD matrices.
        const bool pack_ffwd = weight_layout >= 2;
        for(const auto& pd:disp) {
            // The default Linux ACO attention reads N-pair QKV weights. Native
            // Windows attention retains the original layout, including when
            // selected as a diagnostic shell in an otherwise ACO graph.
            if (!nr::binary::directory().empty() && pd.kern=="attn" &&
                !nr::binary::native_shell("g_attn.spv")) {
                PushAttn p{};
                if(pd.push.size()<sizeof p)throw std::runtime_error("binary attention push mismatch");
                std::memcpy(&p,pd.push.data(),sizeof p);
                if(binary_packed_attn.insert(p.w_off).second)pack_matrix(p.w_off,1536,512);
                continue;
            }
            if(pack_ffwd && (pd.kern=="ffwd3" || pd.kern=="ffwd3w")) {
                PushFfwd3 p{};
                if(pd.push.size()!=sizeof p) throw std::runtime_error("packed FFWD push mismatch");
                std::memcpy(&p,pd.push.data(),sizeof p);
                pack_matrix(p.a_off,FF_GROUPS*FF_R,FF_K);
                pack_matrix(p.q0_off,FF_GROUPS*FF_J,FF_R);
                pack_matrix(p.q2_off,FF_GROUPS*FF_OUT,FF_J);
                continue;
            }
            if(pd.kern=="gemmproj" || pd.kern=="gemmprojc" || pd.kern=="gemmproj5" || pd.kern=="gemmprojt" || pd.kern=="gemmprojw" || pd.kern=="gemmvact" || pd.kern=="gemmvqkv" || pd.kern=="gemmvqkvs" || pd.kern=="gemmvqkvnorm" || pd.kern=="gemmvqkvnorms") {
                PushGemm p{};
                if(pd.push.size()!=sizeof p) throw std::runtime_error("packed GEMM push mismatch");
                std::memcpy(&p,pd.push.data(),sizeof p);
                pack_matrix(p.w_off,p.N,p.K);
                continue;
            }
            uint32_t C=0;
            if((weight_layout==3 || weight_layout==5) && pd.kern.rfind("fswin",0)==0 &&
               pd.kern.size()>=2 && pd.kern.substr(pd.kern.size()-2)=="32") {
                PushFSwin p{};
                if(pd.push.size()<sizeof p) throw std::runtime_error("packed C32 push mismatch");
                std::memcpy(&p,pd.push.data(),sizeof p);
                pack_matrix(p.e_off,128,32);
                pack_matrix(p.ct_off,32,128);
                pack_matrix(p.qkv_off,96,32);
                pack_matrix(p.op_off,32,32);
                continue;
            }
            for(uint32_t c:{64u,128u,256u})
                if(pd.kern=="fswin"+std::to_string(c) || pd.kern=="fswinds"+std::to_string(c) ||
                   pd.kern=="fswindsp"+std::to_string(c) ||
                   pd.kern=="fswinfusedup"+std::to_string(c) || pd.kern=="fswinisolatedup"+std::to_string(c)) C=c;
            if(!C) continue;
            PushFSwin p{};
            if(pd.push.size()<sizeof p) throw std::runtime_error("packed Swin push mismatch");
            std::memcpy(&p,pd.push.data(),sizeof p);
            pack_matrix(p.e_off,4*C,C);
            pack_matrix(p.mid_off,C,128);
            pack_matrix(p.ct_off,C,C);
            pack_matrix(p.qkv_off,3*C,C);
            pack_matrix(p.op_off,C,C);
        }
        std::printf("N-pair FP8 weight layout: %zu matrices\n",packed_count);
    }

    //
    // **Between two fused-Swin layers of one level the barrier enforces more
    // than the data needs.** Layer i+1's window (wx, wy) reads the four tiles
    // it owns, shifted by one tile, which is at most four of layer i's windows
    // - and the full memory barrier drains the whole device. A maximal run of
    // consecutive plain blocks at one level becomes one `fswinp<C>` dispatch
    // whose workgroups pull `(layer, window)` items off one counter and wait on
    // the four producer flags they actually have. The arithmetic is the same
    // arithmetic - the same body, the same order inside a window - so the
    // picture is byte-identical or the merge is wrong.
    //
    // Off for every flag that reasons per step: they dispatch, seed or time
    // individual layers and a merged run has no individual layers left.
    {
        const bool asked_off = flag(argc, argv, "--no-persist") ||
            (std::getenv("NR_NO_PERSIST") && *std::getenv("NR_NO_PERSIST"));
        const char* per_step_flags[] = {"--only", "--seed-dir", "--no-barrier-after",
                                        "--barrier-stride"};
        bool per_step = flag(argc, argv, "--oracle") || flag(argc, argv, "--no-barrier") ||
                        flag(argc, argv, "--barrier-report") || flag(argc, argv, "--exec-barrier");
        for (const char* f : per_step_flags) if (arg(argc, argv, f)) per_step = true;
        if (std::getenv("NR_DIAG_BARRIERS") && !flag(argc, argv, "--only") && !flag(argc, argv, "--seed-dir"))
            per_step = false;   // diagnostic: barrier drops keep the persistent runs
        std::set<int> levels;
        {
            // **64, 128 and 256, measured; C=32 stays per-layer.** The merge
            // removes a barrier per layer and adds, per window, a workgroup
            // barrier, a release, and the claim's round trip through L2. On a
            // 9070 XT the whole frame, normalised to 2810 / 2625 MHz (means of
            // three runs each):
            //
            //     levels             1080p ms   4K ms
            //     none (base)          8.731    30.959
            //     256                  8.245    30.450
            //     64,128,256           8.078    30.328
            //     32,64,128,256        8.066    30.395
            //
            // and the per-width kernel totals behind it, base -> merged:
            //
            //     width   1080p ms        4K ms
            //     C=256   1.397 -> 0.990  3.812 -> 3.350
            //     C=128   0.957 -> 0.850  3.341 -> 3.294
            //     C=64    0.731 -> 0.708  2.773 -> 2.754
            //     C=32    0.895 -> 0.929  3.604 -> 3.753
            //
            // C=32 is the one width whose kernels get slower merged - its
            // windows are cheap and there are 8349 of them at 4K, so the
            // per-item overhead is a larger fraction than the barrier it
            // removes. It buys 0.012 ms at 1080p (inside the run spread) and
            // costs 0.067 at 4K, so it is out. `--persist-levels 32,64,128,256`
            // puts it back.
            const char* env = std::getenv("NR_PERSIST_LEVELS");
            // A benchmark driver may have no argument passthrough, so the set is
            // reachable by environment as well as by flag; the flag wins.
            const std::string t = arg(argc, argv, "--persist-levels",
                                      env && *env ? env : "64,128,256");
            size_t i = 0;
            while (i < t.size()) {
                size_t j = t.find(',', i);
                if (j == std::string::npos) j = t.size();
                if (j > i) levels.insert(std::atoi(t.substr(i, j - i).c_str()));
                i = j + 1;
            }
        }
        const uint32_t wg_override = uint32_t(std::atoi(arg(argc, argv, "--persist-wg", "0")));
        const double strag_pct = std::getenv("NR_STRAG_PCT_ENV") ? std::atof(std::getenv("NR_STRAG_PCT_ENV"))
                                                                 : double(NR_STRAG_PCT);
        // Enough polls that a producer's whole dispatch fits inside one, and
        // few enough that a wiring bug returns rather than hangs the device.
        const uint32_t spin_limit = uint32_t(std::atoi(arg(argc, argv, "--persist-spin", "4194304")));
        if (!asked_off && !per_step) {
            auto fp = [](const Disp& d) {
                PushFSwin p{}; std::memcpy(&p, d.push.data(), sizeof p); return p;
            };
            auto mergeable = [&](const Disp& d, int C) {
                return is_plain_fswin(d.kern) && d.push.size() == sizeof(PushFSwin) &&
                       (C < 0 || d.kern == "fswin" + std::to_string(C));
            };
            std::vector<Disp> out;
            for (size_t i = 0; i < disp.size();) {
                size_t j = i;
                const int C = mergeable(disp[i], -1) ? std::atoi(disp[i].kern.c_str() + 5) : 0;
                // The window grid may differ layer to layer; the *tile* raster
                // may not, because that is the level's own resolution and it
                // is what the producer map is expressed in.
                if (C && levels.count(C))
                    while (j + 1 < disp.size() && mergeable(disp[j + 1], C) &&
                           fp(disp[j + 1]).tiles_x == fp(disp[i]).tiles_x &&
                           fp(disp[j + 1]).tiles_y == fp(disp[i]).tiles_y &&
                           fp(disp[j + 1]).x_off == fp(disp[j]).o_off)
                        ++j;
                if (j == i) { out.push_back(std::move(disp[i])); i = j + 1; continue; }
                // The fused downsample that follows the run joins it as
                // its last layer - same body, same tiles, the DS epilogue behind
                // a wave-uniform branch in the shader.
                const uint32_t ds_bit = C == 64 ? 1u : C == 128 ? 2u : C == 256 ? 4u : 0u;
                const bool ds_fold = NR_PERSIST_DF && (uint32_t(NR_PERSIST_DS_MASK) & ds_bit) &&
                    j + 1 < disp.size() && disp[j + 1].kern == "fswindsp" + std::to_string(C) &&
                    disp[j + 1].push.size() == sizeof(PushFSwin) + sizeof(PushDsProj) &&
                    fp(disp[j + 1]).tiles_x == fp(disp[i]).tiles_x &&
                    fp(disp[j + 1]).tiles_y == fp(disp[i]).tiles_y &&
                    fp(disp[j + 1]).x_off == fp(disp[j]).o_off &&
                    // Every width at every extent. C=256 used to stay standalone where
                    // its grid was several rounds deep (+0.015 ms at 4K then); with
                    // the later run changes the fold wins there too: the standalone
                    // kernel's last round runs at a third of occupancy (4K C=256
                    // DS+UPS folded -34 us, 3440x1440 -58 us).
                    true;
                if (ds_fold) ++j;
                // And the wide fused upsample just before it, as layer 0.
                const bool up_fold = !ds_fold && NR_PERSIST_DF && (uint32_t(NR_PERSIST_UPS_MASK) & ds_bit) &&
                    !out.empty() && out.back().kern == "fswinfusedup" + std::to_string(C) &&
                    out.back().push.size() == sizeof(PushFSwin) + sizeof(PushUps) + 4 &&
                    fp(out.back()).tiles_x == fp(disp[i]).tiles_x &&
                    fp(out.back()).tiles_y == fp(disp[i]).tiles_y &&
                    fp(out.back()).o_off == fp(disp[i]).x_off;
                Disp up_disp{disp[i].s};
                if (up_fold) { up_disp = std::move(out.back()); out.pop_back(); }
                std::vector<const Disp*> lay;
                if (up_fold) lay.push_back(&up_disp);
                for (size_t t = i; t <= j; ++t) lay.push_back(&disp[t]);
                const uint32_t n = uint32_t(lay.size());
                std::vector<PersistRec> rec(n);
                uint32_t windows = 0, most = 0;
                for (uint32_t k = 0; k < n; ++k) {
                    std::memcpy(&rec[k].p, lay[k]->push.data(), sizeof(PushFSwin));
                    rec[k].gx = lay[k]->gx; rec[k].gy = lay[k]->gy;
                    rec[k].windows = lay[k]->gx * lay[k]->gy;
                    rec[k].flag_base = windows;
                    windows += rec[k].windows;
                    most = std::max(most, rec[k].windows);
                }
                std::vector<uint8_t> recs(size_t(n) * sizeof(PersistRec));
                std::memcpy(recs.data(), rec.data(), recs.size());
                // The workgroup count. Items are claimed off a counter now, so
                // this is a throughput choice and no longer a correctness one:
                // a claimed item's owner is resident by construction, whatever
                // else holds the device. It is still the width's occupancy -
                // `spv_stats` gives subgroups per SIMD, four SIMDs a WGP and 32
                // WGPs, capped by the hardware's sixteen barrier-using
                // workgroups a WGP. C=32 is 10/SIMD at two waves a workgroup =
                // 20, capped to 16 -> 512; C=64 8/SIMD at two waves = 16 ->
                // 512; C=128 8/SIMD at four waves = 8 -> 256; C=256 8/SIMD at
                // eight waves = 4 -> 128.
                const uint32_t cap = C == 32 ? 512u : C == 64 ? 512u : C == 128 ? 256u : 128u;
                uint32_t wgo = wg_override;
                // Diagnostic: per-width override NR_PERSIST_WG_<C>.
                if (const char* e = std::getenv(("NR_PERSIST_WG_" + std::to_string(C)).c_str())) wgo = uint32_t(std::atoi(e));
                // C=64 runs used to take 384 workgroups (six a CU) up to 4096
                // windows a layer. The cap is faster at every such extent: Linux
                // 1080p -0.012 ms (with the tile counters), Windows 1080p pds64
                // 0.461 -> 0.432 and pup64 0.445 -> 0.419 ms, 4K tie; byte-identical.
                uint32_t wg = std::min(most, wgo ? wgo : cap);
                // One workgroup per item: a persistent workgroup keeps its launch
                // age for the whole run, and oldest-first wave arbitration runs the
                // youngest on a CU several times slower - whatever it claims, often
                // an item on the run's critical chain. Launched one per item, age
                // follows claim order instead. Pays where a layer has more windows
                // than CUs but not many more: C=256 at 1080p (144 windows) -20..-35
                // us a run, 1280x720 (77) -7, 1440p (252) tie, 60 and below +55
                // (the per-item launch); C=128 at 960x540 (160) -16, 1280x720 (286)
                // -22, 1440x810 (336) -20, 1707x960 (448) and 1080p (558) tie, 4K
                // (2135) +20.
                const bool one = (uint32_t(NR_PERSIST_ONE_MASK) & ds_bit) && (ds_fold || up_fold) &&
                                 most > 64u && most <= uint32_t(NR_PERSIST_ONE_MAX) && !wgo;
                if (one) wg = windows;
                // Launched one per item there are no old workgroups for the straggler
                // queue to prefer: a pipeline that has one (blocking, NR_STRAG=2) gets
                // a threshold of 100% - nothing is routed to it and its claims are the
                // plain queue's.
                const double run_strag = one && strag_pct > 0 ? 100.0 : strag_pct;
                PushPersist pp{};
                pp.layers_off = uint32_t(put(recs, 16) / 4);
                if (ds_fold) {
                    std::vector<uint8_t> db(sizeof(PushDsProj));
                    std::memcpy(db.data(), disp[j].push.data() + sizeof(PushFSwin), db.size());
                    pp.ds_off = uint32_t(put(db, 16) / 4);
                    if (!pp.ds_off) throw std::runtime_error("persist DS: table at offset 0");
                }
                if (up_fold) {
                    std::vector<uint8_t> ub(sizeof(PushUps) + 4);
                    std::memcpy(ub.data(), up_disp.push.data() + sizeof(PushFSwin), ub.size());
                    pp.ds_off = uint32_t(put(ub, 16) / 4);
                    if (!pp.ds_off) throw std::runtime_error("persist UPS: table at offset 0");
                }
                if (NR_PERSIST_DF) {
                    // The shader's own producer map (tile (tx,ty) of layer k
                    // belongs to window floor((tx-shift)/2), floor((ty-shift_y)/2)
                    // of layer k-1), inverted into per-item consumer lists.
                    std::vector<uint32_t> need(windows, 0), cons(size_t(windows) * 4, 0xFFFFFFFFu), init;
                    for (uint32_t k = 0; k < n; ++k)
                        for (uint32_t w = 0; w < rec[k].windows; ++w) {
                            const uint32_t item = rec[k].flag_base + w;
                            if (k == 0) continue;
                            const PersistRec& q = rec[k - 1];
                            const int wx = int(w % rec[k].gx), wy = int(w / rec[k].gx);
                            std::set<uint32_t> prod;
                            for (int t = 0; t < 4; ++t) {
                                const int tx = 2 * wx + rec[k].p.shift + (t & 1);
                                const int ty = 2 * wy + rec[k].p.shift_y + (t >> 1);
                                if (tx < 0 || ty < 0 || tx >= int(rec[k].p.tiles_x) || ty >= int(rec[k].p.tiles_y)) continue;
                                const int ax = tx - q.p.shift, ay = ty - q.p.shift_y;
                                const int px = ax >= 0 ? ax / 2 : -((1 - ax) / 2);
                                const int py = ay >= 0 ? ay / 2 : -((1 - ay) / 2);
                                if (px < 0 || py < 0 || px >= int(q.gx) || py >= int(q.gy)) continue;
                                prod.insert(q.flag_base + uint32_t(py) * q.gx + uint32_t(px));
                            }
                            need[item] = uint32_t(prod.size());
                            for (uint32_t pr : prod) {
                                int slot = 0;
                                while (slot < 4 && cons[size_t(pr) * 4 + slot] != 0xFFFFFFFFu) ++slot;
                                if (slot == 4) throw std::runtime_error("persist DF: more than four consumers");
                                cons[size_t(pr) * 4 + slot] = item;
                            }
                        }
                    for (uint32_t it = 0; it < windows; ++it) if (need[it] == 0) init.push_back(it);
                    std::vector<uint32_t> tab(need);
                    tab.insert(tab.end(), cons.begin(), cons.end());
                    tab.insert(tab.end(), init.begin(), init.end());
                    // Straggler queue tables (pipelines built with NR_STRAG): item -> layer
                    // at 6*TW; (queued items, threshold) per layer at 7*TW; then the two
                    // queues' per-frame totals. An item whose ready-rank in its layer is at
                    // or above the threshold is claimed by the oldest workgroups of a CU,
                    // which oldest-first wave arbitration runs fastest - such items end
                    // their layer late and form the run's critical chain.
                    if (run_strag > 0) {
                        const double f = run_strag / 100.0;
                        tab.resize(size_t(6) * windows, 0u);
                        std::vector<uint32_t> lay(windows, 0u), q(16, 0u);
                        if (n > 16) throw std::runtime_error("straggler queue: more than 16 layers");
                        for (uint32_t k = 0; k < n; ++k)
                            for (uint32_t w = 0; w < rec[k].windows; ++w) {
                                lay[rec[k].flag_base + w] = k;
                                if (need[rec[k].flag_base + w]) ++q[k];
                            }
                        tab.insert(tab.end(), lay.begin(), lay.end());
                        uint32_t ntot = uint32_t(init.size()), ctot = 0;
                        for (uint32_t k = 0; k < 16; ++k) {
                            const uint32_t thr = uint32_t(std::ceil(f * double(q[k])));
                            tab.push_back(q[k]);
                            tab.push_back(thr);
                            ntot += std::min(thr, q[k]); ctot += q[k] - std::min(thr, q[k]);
                        }
                        tab.push_back(ntot); tab.push_back(ctot);
                    }
                    std::vector<uint8_t> tb(tab.size() * 4);
                    std::memcpy(tb.data(), tab.data(), tb.size());
                    pp.df_off = uint32_t(put(tb, 16) / 4);
                    pp.df_n0 = uint32_t(init.size());
                }
                // Inside the activation arena, which is zero-filled once at
                // build - which is what makes epoch 0 mean "nothing done yet".
                // `wg` words past the flags are the per-workgroup broadcast
                // slots the claim is published through.
                act_total = align(act_total, 256);
                pp.sync_off = uint32_t(act_total / 4);
                if (C == 256 && g_chain_epoch_word == 0xFFFFFFFFu) g_chain_epoch_word = pp.sync_off + 2u;
                // DF mode appends the ready queue (one tagged entry per item) and its tail.
                act_total += align(size_t(4 + windows + wg + (NR_PERSIST_DF ? windows + 1 : 0)
                                          + (run_strag > 0 ? 18 + windows + 8 : 0)) * 4, 256);
                pp.n_layers = n; pp.spin_limit = spin_limit;
                pp.total_windows = windows;
                Disp m{disp[i].s};
                m.lo = lay.front()->lo; m.hi = lay.front()->hi;
                for (const Disp* l : lay) { m.lo = std::min(m.lo, l->lo); m.hi = std::max(m.hi, l->hi); }
                // Only a run that ends in the DS layer takes the larger
                // DS-capable pipeline; every other run keeps the plain body.
                m.kern = (ds_fold ? "fswinpds" : up_fold ? "fswinpup" : "fswinp") + std::to_string(C);
                m.gx = wg; m.gy = 1; m.gz = 1;
                m.push.resize(sizeof pp);
                std::memcpy(m.push.data(), &pp, sizeof pp);
                persist_err.push_back(uint32_t((pp.sync_off + 3u) * 4u));
                std::printf("persist: C=%d layers b%03dl%d..b%03dl%d (%u%s) windows %u"
                            " items %u wg %u%s\n", C, disp[i].s->block, disp[i].s->layer,
                            disp[j].s->block, disp[j].s->layer, n,
                            ds_fold ? ", last is DS" : up_fold ? ", first is UPS" : "",
                            most, windows, wg, one ? " (one per item)" : "");
                out.push_back(std::move(m));
                i = j + 1;
            }
            if (persist_err.size())
                std::printf("persist: %zu merged runs, %zu dispatches left\n",
                            persist_err.size(), out.size());
            disp = std::move(out);
        }
    }

    if (arena_probe) {
        // Each value's dispatch range: every final dispatch whose steps meet
        // the steps that asked for it. A merged persistent run executes all its
        // layers at once, so a value any of them touches is live across it.
        std::map<int, std::pair<int, int>> span;
        const int nd = int(disp.size());
        for (const auto& kv : voff.used) {
            int first = nd, last = -1;
            for (int f = 0; f < nd; ++f)
                if (disp[f].lo <= kv.second.second && disp[f].hi >= kv.second.first) {
                    first = std::min(first, f); last = std::max(last, f);
                }
            if (last < 0) { first = 0; last = nd; }   // lowered to nothing: keep it all frame
            span[kv.first] = {first, last};
        }
        // **A slot asked for with no size aliases whatever value the plain
        // layout has at its offset** - a `std::map` default of 0, or a zero-size
        // slot placed at the next value's start. That is what it has always
        // read or written, so it keeps doing so: it follows that value to its
        // new place, and the value's lifetime grows to cover it. (One such slot
        // today: skip_key(30), the unread second output of the block-30 DS.)
        std::map<int, std::pair<int, size_t>> alias;   // slot -> covering value, byte delta
        for (auto& kv : span) {
            if (vsize.count(kv.first) && vsize.at(kv.first)) continue;
            const size_t at = voff.m.count(kv.first) ? voff.m.at(kv.first) : 0;
            int owner = -1;
            for (const auto& v : vsize)
                if (v.second && voff.m.count(v.first) && voff.m.at(v.first) <= at &&
                    at < voff.m.at(v.first) + v.second)
                    owner = v.first;
            if (owner < 0) throw std::runtime_error("arena reuse: an unsized slot aliases no value");
            alias[kv.first] = {owner, at - voff.m.at(owner)};
            auto& o = span[owner];
            o = {std::min(o.first, kv.second.first), std::max(o.second, kv.second.second)};
            std::printf("arena reuse: unsized slot %d (block %d) aliases value %d (block %d layer %d) +%zu\n",
                        kv.first, kv.first / 8, owner, owner / 8, owner % 8, at - voff.m.at(owner));
        }
        // First fit by first use; a block is free again once its last dispatch
        // is behind a barrier, i.e. strictly before the next value's first.
        // Over-read slots: their size grows to what is read, and they live all
        // frame, so the bytes past what is written are never anyone else's.
        for (const auto& o : overread)
            if (span.count(o.first)) span[o.first] = {0, nd};
        auto reserved = [&](int k) {
            const size_t v = vsize.count(k) ? vsize.at(k) : 0;
            return align(std::max(v, overread.count(k) ? overread.at(k) : size_t(0)), 256);
        };
        // **Small slots take cold memory.** Packed tightest, the C=512 blocks at
        // 1080p-class extents cycle three ~1.4 MB slots, and a GEMM whose output
        // lands on memory the dispatch just before it freed - lines still in L2
        // from that dispatch - ran 2-4x slow and varied run to run: the attention
        // projection 7 -> 16..31 us, +0.1 ms a frame at 1080p and 2560x1080, the
        // game path's whole gap to one-value-one-slot. So a slot smaller than
        // NR_ARENA_COLD_MAX only takes a block freed at least NR_ARENA_COLD
        // dispatches before its first. Large slots (4K's C=512 values are ~5.6 MB)
        // never showed it and pack as before; the arena did not grow at any
        // extent measured. `freed_at` keeps, per free block, the last dispatch of
        // what freed it (merged blocks the latest).
        std::map<size_t, int> freed_at;
        std::vector<int> order;
        for (const auto& kv : span)
            if (vsize.count(kv.first) && vsize.at(kv.first)) order.push_back(kv.first);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (span[a].first != span[b].first) return span[a].first < span[b].first;
            if (vsize[a] != vsize[b]) return vsize[a] > vsize[b];
            return a < b;
        });
        // **NR_TCHAIN:** dispatches joined by tile counters overlap, so a value last
        // read inside such a stretch stays reserved to the stretch's end (the next
        // real barrier); otherwise a later dispatch of the stretch could overwrite
        // what a still-running earlier one reads.
        if (tchain_active()) {
            std::vector<int> stretch_end(size_t(nd) + 1);
            for (int f = nd; f >= 0; --f)
                stretch_end[size_t(f)] = (f + 1 < nd && tchain_pair(disp[size_t(f)].kern, disp[size_t(f) + 1].kern))
                                         ? stretch_end[size_t(f) + 1] : f;
            for (auto& kv : span)
                if (kv.second.second >= 0 && kv.second.second < nd)
                    kv.second.second = stretch_end[size_t(kv.second.second)];
        }
        std::vector<std::pair<size_t, size_t>> free_blocks;   // offset, size
        std::vector<std::pair<int, std::pair<size_t, size_t>>> live;   // last dispatch, block
        size_t top = 0, high = 0;
        auto release = [&](size_t off, size_t size, int when) {
            free_blocks.push_back({off, size});
            freed_at[off] = when;
            std::sort(free_blocks.begin(), free_blocks.end());
            std::vector<std::pair<size_t, size_t>> merged;
            for (const auto& b : free_blocks) {
                if (!merged.empty() && merged.back().first + merged.back().second == b.first) {
                    merged.back().second += b.second;
                    freed_at[merged.back().first] = std::max(freed_at[merged.back().first], freed_at[b.first]);
                } else merged.push_back(b);
            }
            free_blocks.swap(merged);
            if (!free_blocks.empty() && free_blocks.back().first + free_blocks.back().second == top) {
                top = free_blocks.back().first;
                free_blocks.pop_back();
            }
        };
        arena_layout.clear();
        for (int k : order) {
            const int first = span[k].first;
            for (size_t l = 0; l < live.size();) {
                if (live[l].first < first) {
                    release(live[l].second.first, live[l].second.second, live[l].first);
                    live.erase(live.begin() + long(l));
                } else ++l;
            }
            const size_t need = reserved(k);
            size_t best = free_blocks.size();
            const bool cold = need < size_t(NR_ARENA_COLD_MAX);
            for (size_t b = 0; b < free_blocks.size(); ++b)
                if (free_blocks[b].second >= need &&
                    !(cold && freed_at[free_blocks[b].first] + int(NR_ARENA_COLD) >= first) &&
                    (best == free_blocks.size() || free_blocks[b].second < free_blocks[best].second))
                    best = b;
            size_t off;
            if (best < free_blocks.size()) {
                off = free_blocks[best].first;
                if (free_blocks[best].second > need) {
                    free_blocks[best] = {off + need, free_blocks[best].second - need};
                    freed_at[off + need] = freed_at[off];
                } else free_blocks.erase(free_blocks.begin() + long(best));
            } else {
                off = top;
                top += need;
            }
            arena_layout[k] = off;
            if (std::getenv("NR_ARENA_DUMP"))
                std::fprintf(stderr, "arena value %d (b%d l%d) disp %d..%d off %zu size %zu\n", k, k / 8, k % 8,
                             span[k].first, span[k].second, off, need);
            live.push_back({span[k].second, {off, need}});
            high = std::max(high, top);
        }
        for (const auto& a : alias) arena_layout[a.first] = arena_layout.at(a.second.first) + a.second.second;
        // Anything else the build ever names without using it keeps offset 0.
        for (const auto& kv : voff.m) if (!arena_layout.count(kv.first)) arena_layout[kv.first] = 0;
        for (const auto& kv : vsize) if (!arena_layout.count(kv.first)) arena_layout[kv.first] = 0;
        arena_values = align(high, 256);
        arena_disp_steps.clear();
        for (const Disp& d : disp) arena_disp_steps.push_back({d.lo, d.hi});
        return 3;
    }
    if (!arena_expect.empty()) {
        bool same = arena_expect.size() == disp.size();
        for (size_t f = 0; same && f < disp.size(); ++f)
            same = arena_expect[f] == std::make_pair(disp[f].lo, disp[f].hi);
        if (!same) throw std::runtime_error("arena reuse: the build lowered differently from its layout probe");
    }

    // NR_CHAIN: NR_CHAIN_KERNELS=k1,k2,... lists kernels whose SPVs were
    // built with -DNR_CHAIN=1. Consecutive dispatches of listed kernels lose
    // their barrier; the second waits on the first's completion counter.
    if (const char* ck = std::getenv("NR_CHAIN_KERNELS")) {
        const std::string cs = ck; size_t p0 = 0;
        while (p0 <= cs.size()) {
            size_t q = cs.find(',', p0); if (q == std::string::npos) q = cs.size();
            if (q > p0) g_chain_kern.insert(cs.substr(p0, q - p0));
            p0 = q + 1;
        }
    }
    g_chain_nobar.assign(disp.size(), 0);
    if (!g_chain_kern.empty()) {
        if (g_chain_epoch_word == 0xFFFFFFFFu)
            throw std::runtime_error("NR_CHAIN needs the C=256 persistent run for its frame epoch");
        act_total = align(act_total, 256);
        g_chain_base = uint32_t(act_total / 4);
        g_chain_n = uint32_t(disp.size());
        act_total += size_t(disp.size()) * 16 * 256 * 4;   // 16 replicas, 256 words apart
        size_t nch = 0;
        for (size_t i = 0; i < disp.size(); ++i) {
            if (!g_chain_kern.count(disp[i].kern)) continue;
            uint32_t tail[4] = {0xFFFFFFFFu, 0u, g_chain_base + uint32_t(4096 * i), g_chain_epoch_word};
            if (i > 0 && g_chain_kern.count(disp[i - 1].kern) && !std::getenv("NR_CHAIN_NOWAIT")) {
                tail[0] = g_chain_base + uint32_t(4096 * (i - 1));
                tail[1] = disp[i - 1].gx * disp[i - 1].gy * disp[i - 1].gz;
                g_chain_nobar[i - 1] = 1; ++nch;
            }
            const size_t o = disp[i].push.size();
            disp[i].push.resize(o + 16);
            std::memcpy(disp[i].push.data() + o, tail, 16);
        }
        std::printf("NR_CHAIN: %zu barriers replaced by completion counters (epoch word %u)\n",
                    nch, g_chain_epoch_word);
    }

    // ---- NR_TCHAIN: tile counters in place of barriers ------------------------
    // A producer counts finished units (attn: a window per workgroup; gemm1x1,
    // ffwd3, vitattn: a token tile per wave); the next dispatch's workgroups wait
    // for the tiles they read and the barrier between the two goes (tchain_pair).
    // Every NR_TCHAIN kernel's push ends with one word, tc_rec: ~0 or the u32
    // index in the weight blob of its record {wait counters, need, tile table
    // (weight blob, ~0 = no wait), signal counters, epoch word, error word};
    // counters and error words live after the activation arena (zero-filled).
    g_tc_nobar.assign(disp.size(), 0);
    g_tc_words.clear();
    g_tc_err.clear();
    tc_expect.clear(); tc_pair.clear(); tc_err.clear(); tc_base = 0; tc_words = 0;
    if (NR_TCHAIN_TILES) {
        const bool on = tchain_active();
        act_total = align(act_total, 256);
        g_tc_base = act_total / 4;
        // The frame counter: ticked by the frame's first fswin32 (a real barrier
        // follows it), read as the epoch by every chained consumer that is not a
        // persistent run (those count their own frames).
        // Windows: that layer runs on fswin32t / fswin32nht, the same kernel built with NR_TCHAIN (the
        // other C=32 layers carry no counter code).
        size_t tick = disp.size();
        for (size_t i = 0; i < disp.size() && tick == disp.size(); ++i)
            if (disp[i].kern == "fswin32" || disp[i].kern == "fswin32nh") {
                if (!nr::binary::directory().empty()) tick = i;
                else if (std::filesystem::exists(spv_dir + "/g_" + disp[i].kern + "t.spv")) {
                    disp[i].kern += "t"; tick = i;
                }
            }
        const uint32_t frame_word = uint32_t(g_tc_base);
        g_tc_words.push_back(0u);
        std::vector<std::array<uint32_t, 7>> rec(disp.size());   // word 6: NR_TC_MAGIC (nr_chain.glsl)
        for (auto& r : rec) r = {0xFFFFFFFFu, 0u, 0xFFFFFFFFu, 0xFFFFFFFFu, frame_word, 0u, 0x54434852u};
        if (tick < disp.size()) rec[tick][1] = 1u;
        auto putw = [&](const void* p, size_t n) { return uint32_t(wblob.append(p, n, 16) / 4); };
        size_t nch = 0;
        bool big_frame = false;
        for (const Disp& d : disp)
            if (is_persist_kern(d.kern)) {
                PushPersist pp{}; std::memcpy(&pp, d.push.data(), sizeof pp);
                big_frame = big_frame || (pp.n_layers && pp.total_windows / pp.n_layers > 4096u);
            }
        // research: NR_TC_BIG=1 keeps the persistent-run chains at 4K too.
        if (std::getenv("NR_TC_BIG") && std::atoi(std::getenv("NR_TC_BIG"))) big_frame = false;
        auto wword = [&](uint32_t i) { uint32_t v; wblob.read(size_t(i) * 4, &v, 4); return v; };
        // The activation a dispatch reads from its predecessor / writes for its
        // successor. A persistent run: its last layer's output, its UPS layer's
        // lower-level input (records in the weight blob); the fused upsample: its
        // lower-level input.
        auto io = [&](const Disp& d, bool out) -> uint32_t {
            uint32_t f[2] = {};
            if (d.kern.rfind("gemm", 0) == 0) { PushGemm g{}; std::memcpy(&g, d.push.data(), sizeof g); f[0] = g.x_off; f[1] = g.o_off; }
            else if (is_persist_kern(d.kern)) {
                PushPersist pp{}; std::memcpy(&pp, d.push.data(), sizeof pp);
                const bool ds = d.kern.rfind("fswinpds", 0) == 0, ups = d.kern.rfind("fswinpup", 0) == 0;
                f[0] = ups && pp.ds_off ? wword(pp.ds_off) : wword(pp.layers_off);
                f[1] = ds && pp.ds_off ? wword(pp.ds_off + 1u)          // the DS projection's output
                     : wword(pp.layers_off + (pp.n_layers - 1u) * uint32_t(sizeof(PersistRec) / 4) + 1u);
            } else if (d.kern == "fswindsp32") {
                PushDsProj dp{}; std::memcpy(&dp, d.push.data() + sizeof(PushFSwin), sizeof dp);
                std::memcpy(f, d.push.data(), 4); f[1] = dp.o_off;
            } else if (d.kern == "fswinfusedup32") {
                PushUps u{}; std::memcpy(&u, d.push.data() + sizeof(PushFSwin), sizeof u);
                f[0] = u.p_off; f[1] = 0xFFFFFFFEu;
            } else std::memcpy(f, d.push.data(), 8);                    // attn, ffwd3, vitattn: x_off, o_off first
            return f[out ? 1 : 0];
        };
        // The consumer's gather grid: a UPS-first run or the fused upsample
        // reads the lower level in its own itiles_x raster (the plain gather:
        // none of these pipelines builds NR_WIDE_UPS_VIEW).
        auto ups_grid = [&](const Disp& d, uint32_t& itx, uint32_t& ity) -> bool {
            PushUps u{};
            if (is_persist_kern(d.kern)) {
                PushPersist pp{}; std::memcpy(&pp, d.push.data(), sizeof pp);
                if (!pp.ds_off) return false;
                for (uint32_t k = 0; k < sizeof u / 4; ++k) { const uint32_t v = wword(pp.ds_off + k); std::memcpy(reinterpret_cast<uint32_t*>(&u) + k, &v, 4); }
            } else std::memcpy(&u, d.push.data() + sizeof(PushFSwin), sizeof u);
            itx = u.itiles_x; ity = u.itiles_y;
            return itx != 0u && ity != 0u;
        };
        // A gemm1x1 producer's M tile (token rows) and waves along N (each wave
        // signals every token tile of the M tile), per pipeline.
        auto gemm_mt_wn = [](const std::string& k) -> std::pair<uint32_t, uint32_t> {
            if (k == "gemmvact" || k == "gemmvqkvnorm") return {NR_GEMM_WIDE_MT, NR_GEMM_WIDE_NT / 64};
            if (k == "gemmvqkvnorms") return {NR_GEMM_QKVS_MT, NR_GEMM_QKVS_NT / 96};
            if (k == "gemmprojw") return {NR_GEMM_PROJW_MT, NR_GEMM_PROJW_NT / 64};
            return {NR_GEMM_PROJ_MT, NR_GEMM_PROJ_NT / 32};   // gemmproj, gemmproj5, gemmprojc, gemmprojt
        };
        // Token tiles a table covers: 4096 holds 4K's C=512 grid (60x34) and every
        // ViT one; the decoder's upsample levels take their own raster (4K C=64: 32640).
        for (size_t i = 0; on && i + 1 < disp.size(); ++i) {
            const Disp& P = disp[i]; Disp& Q = disp[i + 1];
            if (!tchain_pair(P.kern, Q.kern)) continue;
            if (tick > i) continue;   // no frame tick before this pair
            // The persistent runs' boundaries pay where the runs are short: 1080p
            // -0.06 ms, 1440p (C=64 3600 windows a layer) -0.03, but 4K (8160) +0.03 -
            // there the next run's workgroups waiting in their slots cost more than
            // the drain gives back. So none of them once a run has more than 4096
            // windows a layer. The layout probe counts them as chained anyway,
            // which only keeps values a little longer.
            auto run_kern = [](const std::string& k) {
                return is_persist_kern(k) || k == "fswinfusedup32" || k == "fswindsp32";
            };
            if (big_frame && (run_kern(P.kern) || run_kern(Q.kern))) continue;
            if (std::getenv("NR_TCHAIN_DEBUG")) {
                uint32_t a = 0, b = 0; const bool ok = (is_persist_kern(Q.kern) || Q.kern == "fswinfusedup32") ? ups_grid(Q, a, b) : true;
                std::printf("tchain? %s -> %s: out %u in %u grid %u/%u ok %d\n", P.kern.c_str(), Q.kern.c_str(), io(P, true), io(Q, false), a, b, int(ok));
            }
            if (io(Q, false) != io(P, true)) continue;   // not reading its predecessor's output: keep the barrier
            // Consumers that gather through an upsample must read the producer's
            // own tile raster.
            uint32_t itx = 0, ity = 0;
            const bool q_ups = Q.kern.rfind("fswinpup", 0) == 0 || Q.kern == "fswinfusedup32";
            if (q_ups && !ups_grid(Q, itx, ity)) continue;
            uint32_t T = 4096;
            if (is_persist_kern(Q.kern)) {   // the consumer run's own tile raster
                PushPersist qp{}; std::memcpy(&qp, Q.push.data(), sizeof qp);
                T = std::max(T, wword(qp.layers_off + 13u) * wword(qp.layers_off + 14u));
            }
            if (is_persist_kern(P.kern)) {
                PushPersist pp{}; std::memcpy(&pp, P.push.data(), sizeof pp);
                const uint32_t lb = pp.layers_off + (pp.n_layers - 1u) * uint32_t(sizeof(PersistRec) / 4);
                T = std::max(T, wword(lb + 13u) * wword(lb + 14u));
            }
            if (q_ups) T = std::max(T, itx * ity);
            // Table entry: counter | need << 20 (need = producer units a frame).
            std::vector<uint32_t> t(T, 0xFFFFFFFFu);
            uint32_t ncnt = T;
            auto ent = [](uint32_t c, uint32_t need) { return c | (need << 20); };
            if (NR_KBASE(P.kern) == "attn") {                           // tile -> window
                PushAttn pa{}; std::memcpy(&pa, P.push.data(), sizeof pa);
                ncnt = P.gx * P.gy;
                for (uint32_t wy = 0; wy < P.gy; ++wy)
                    for (uint32_t wx = 0; wx < P.gx; ++wx)
                        for (int q = 0; q < 4; ++q) {
                            const int tx = 2 * int(wx) + pa.shift + (q & 1), ty = 2 * int(wy) + pa.shift_y + (q >> 1);
                            if (tx < 0 || ty < 0 || tx >= int(pa.tiles_x) || ty >= int(pa.tiles_y)) continue;
                            t.at(uint32_t(ty) * pa.tiles_x + uint32_t(tx)) = ent(wy * P.gx + wx, P.gz);
                        }
            } else if (P.kern.rfind("gemm", 0) == 0) {                  // every M tile, all N tiles
                const auto mw = gemm_mt_wn(NR_KBASE(P.kern));
                for (uint32_t k = 0; k < P.gx * (mw.first / 16u); ++k) t.at(k) = ent(k, P.gy * mw.second);
            } else if (P.kern == "vitattn") {                            // a workgroup a query tile and head
                PushVAttn pv{}; std::memcpy(&pv, P.push.data(), sizeof pv);
                for (uint32_t k = 0; k < (pv.tokens + 15u) / 16u; ++k) t.at(k) = ent(k, P.gy);
            } else if (P.kern == "fswindsp32" || P.kern.rfind("fswinpds", 0) == 0) {   // DS windows -> next-level tiles
                PushFSwin fs{}; PushDsProj dp{}; uint32_t gx = P.gx, gy = P.gy;
                if (P.kern == "fswindsp32") {
                    std::memcpy(&fs, P.push.data(), sizeof fs);
                    std::memcpy(&dp, P.push.data() + sizeof(PushFSwin), sizeof dp);
                } else {
                    PushPersist pp{}; std::memcpy(&pp, P.push.data(), sizeof pp);
                    if (!pp.ds_off) continue;
                    const uint32_t lb = pp.layers_off + (pp.n_layers - 1u) * uint32_t(sizeof(PersistRec) / 4);
                    for (uint32_t k = 0; k < sizeof fs / 4; ++k) { const uint32_t v = wword(lb + k); std::memcpy(reinterpret_cast<uint32_t*>(&fs) + k, &v, 4); }
                    for (uint32_t k = 0; k < sizeof dp / 4; ++k) { const uint32_t v = wword(pp.ds_off + k); std::memcpy(reinterpret_cast<uint32_t*>(&dp) + k, &v, 4); }
                    gx = wword(lb + 20u); gy = wword(lb + 21u);
                }
                const uint32_t wr = dp.writer_rows ? dp.writer_rows : dp.rows;
                // The identity shear only (the shader's NR_DS_IDENTITY_FAST case).
                if (dp.mode == 2u || dp.raster != dp.crow || wr != dp.rows) continue;
                // The consumer reads the DS output in its own tile raster (a run's first
                // layer), or as canonical C=512 token tiles (ffwd3: rows of otx tiles).
                if (is_persist_kern(Q.kern)) {
                    PushPersist qp{}; std::memcpy(&qp, Q.push.data(), sizeof qp);
                    if (wword(qp.layers_off + 13u) != dp.otx) continue;
                }
                const int lx = std::min(std::min(int(fs.pool_tiles_x * 4u), int(dp.raster)), int(dp.otx * 4u)) - 1;
                const int ly = std::min(int(fs.pool_tiles_y * 4u), int(wr)) - 1;
                std::vector<uint32_t> nd(T, 0u);
                for (uint32_t wy = 0; wy < gy; ++wy)
                    for (uint32_t wx = 0; wx < gx; ++wx) {
                        const int px0 = (2 * int(wx) + fs.shift) * 2, py0 = (2 * int(wy) + fs.shift_y) * 2;
                        const int x0 = std::max(px0, 0), x1 = std::min(px0 + 3, lx), y0 = std::max(py0, 0), y1 = std::min(py0 + 3, ly);
                        if (x0 > x1 || y0 > y1) continue;
                        for (int ty = y0 / 4; ty <= y1 / 4; ++ty)
                            for (int tx = x0 / 4; tx <= x1 / 4; ++tx) ++nd.at(uint32_t(ty) * dp.otx + uint32_t(tx));
                    }
                for (uint32_t k = 0; k < T; ++k) if (nd[k]) t[k] = ent(k, nd[k]);
            } else if (is_persist_kern(P.kern)) {                        // the last layer: a window a tile, once
                PushPersist pp{}; std::memcpy(&pp, P.push.data(), sizeof pp);
                const uint32_t lb = pp.layers_off + (pp.n_layers - 1u) * uint32_t(sizeof(PersistRec) / 4);
                const uint32_t tx = wword(lb + 13u), ty = wword(lb + 14u);
                if (q_ups && itx != tx) continue;
                for (uint32_t k = 0; k < tx * ty; ++k) t.at(k) = ent(k, 1u);
            } else {                                                     // ffwd3: a wave a group and tile
                PushFfwd3 pf{}; std::memcpy(&pf, P.push.data(), sizeof pf);
                for (uint32_t k = 0; k < (pf.M + 15u) / 16u; ++k) t.at(k) = ent(k, 8u);
            }
            const uint32_t err = uint32_t(g_tc_base + g_tc_words.size());
            g_tc_err.push_back(g_tc_words.size());
            g_tc_words.push_back(0u);                                     // error word
            const uint32_t cnt = uint32_t(g_tc_base + g_tc_words.size());
            g_tc_words.insert(g_tc_words.end(), size_t(ncnt), 0u);
            // C=512 projections M-major (every N tile of M tile 0, then M tile 1 ...):
            // their inputs complete in token order.
            if (Q.kern == "gemmprojc" || Q.kern == "gemmproj5") {
                PushGemm pg{}; std::memcpy(&pg, Q.push.data(), sizeof pg);
                pg.remap = 1u;
                std::memcpy(Q.push.data(), &pg, sizeof pg);
            }
            for (uint32_t k = 0; k < uint32_t(t.size()); ++k)
                if (t[k] != 0xFFFFFFFFu)
                    tc_expect.push_back({cnt - uint32_t(g_tc_base) + (t[k] & 0xFFFFFu), t[k] >> 20, uint32_t(tc_pair.size())});
            tc_pair.push_back(P.kern + " -> " + Q.kern);
            rec[i][3] = cnt;
            rec[i + 1][0] = cnt; rec[i + 1][2] = putw(t.data(), t.size() * 4);
            rec[i + 1][5] = err;
            g_tc_nobar[i] = 1; ++nch;
        }
        for (size_t i = 0; i < disp.size(); ++i) {
            if (!tchain_kern(disp[i].kern)) continue;
            const uint32_t r = (rec[i][1] == 0u && rec[i][2] == 0xFFFFFFFFu && rec[i][3] == 0xFFFFFFFFu)
                ? 0xFFFFFFFFu : putw(rec[i].data(), sizeof rec[i]);
            const size_t o = disp[i].push.size();
            disp[i].push.resize(o + 4);
            std::memcpy(disp[i].push.data() + o, &r, 4);
            if (std::getenv("NR_TCHAIN_DEBUG") && (i == tick || i < 3))
                std::printf("tchain rec: disp %zu %s push %zu rec %u [%u %u %u %u %u %u]\n", i, disp[i].kern.c_str(),
                            disp[i].push.size(), r, rec[i][0], rec[i][1], rec[i][2], rec[i][3], rec[i][4], rec[i][5]);
        }
        act_total += align(g_tc_words.size() * 4, 256);
        if (on) nr::logf("tile counters: %zu barriers replaced", nch);
        tc_base = uint32_t(g_tc_base); tc_words = uint32_t(g_tc_words.size());
        for (size_t e : g_tc_err) tc_err.push_back(uint32_t(e));
        // One counter, one count a frame: several tiles may name the same counter.
        std::sort(tc_expect.begin(), tc_expect.end(),
                  [](const TcExpect& a, const TcExpect& b) { return a.word < b.word; });
        size_t kept = 0;
        for (size_t k = 0; k < tc_expect.size(); ++k) {
            if (kept && tc_expect[kept - 1].word == tc_expect[k].word) {
                if (tc_expect[kept - 1].need != tc_expect[k].need)
                    nr::logf("[nr] check: counter %u named with need %u and %u", tc_expect[k].word,
                             tc_expect[kept - 1].need, tc_expect[k].need);
                continue;
            }
            tc_expect[kept++] = tc_expect[k];
        }
        tc_expect.resize(kept);
    }

    // ---- device ------------------------------------------------------------
    if (!ctx.device) ctx.create();
    std::printf("GPU    %s\n", ctx.gpu_name.c_str());
    // Every pipeline this build creates, kept across runs. Empty path: no cache,
    // which is what the kernel benches want.
    if (const char* cache_path = arg(argc, argv, "--pipeline-cache"))
        if (*cache_path) ctx.load_pipeline_cache(cache_path);
    // Announced before it is asked for, not after it is got. These two are the
    // allocations a build dies on, and until now a game's log simply stopped at
    // the line before them with no number in it.
    nr::logf("allocating arenas: activation %.1f MB, weights %.1f MB",
             double(act_total) / 1e6, double(wblob.size()) / 1e6);
    // NR_PROF (diagnostic): a word region after the arena for in-kernel
    // clock records; shaders built with -DNR_PROF_OFF=<printed offset>.
    g_prof_words = std::getenv("NR_PROF") ? size_t(std::atoll(std::getenv("NR_PROF"))) : 0;
    size_t& prof_words = g_prof_words; size_t& prof_off = g_prof_off;
    if (prof_words) {
        act_total = align(act_total, 256);
        prof_off = act_total / 4;
        // A fixed offset (NR_PROF_AT, words) so one SPV set serves every host configuration.
        if (const char* at = std::getenv("NR_PROF_AT")) {
            const size_t want = size_t(std::atoll(at));
            if (want < prof_off) throw std::runtime_error("NR_PROF_AT inside the arena");
            prof_off = want; act_total = want * 4;
        }
        act_total += align(prof_words * 4, 256);
        std::printf("NR_PROF region: word offset %zu, %zu words\n", prof_off, prof_words);
    }
    act = ctx.buffer(act_total, false, /*device_address=*/true); wgt = ctx.buffer(wblob.size());
    // vit_attn.comp NR_VBDA reads V through a buffer_reference: the arena's address goes into
    // every vitattn push block (PushVAttn::act_lo/act_hi).
    nr::logf("[nr] activation arena through its device address: %s",
             act.address ? "yes" : "no (the storage-buffer builds, g_*nobda.spv)");
    if (act.address)
        for (Disp& d : disp) {
            const uint32_t lohi[2] = {uint32_t(act.address), uint32_t(act.address >> 32)};
            if (d.kern == "vitattn" && d.push.size() >= sizeof(PushVAttn))
                std::memcpy(d.push.data() + offsetof(PushVAttn, act_lo), lohi, sizeof lohi);
            else if (gemm_push_kern(d.kern) && d.push.size() >= sizeof(PushGemm))
                std::memcpy(d.push.data() + offsetof(PushGemm, act_lo), lohi, sizeof lohi);
        }
    // **The network's outside edge**: one sampled image in, two storage images
    // out. Sized from the two adapter layers' own plan rows; the extent columns
    // are (H, W) in that order, which is the transposition the gold capture
    // records for +208/+212.
    for (const Step& s2 : plan) {
        if (s2.type == "CCTinlayoutFusedPreBlockSwin1H") { in_w = uint32_t(s2.H); in_h = uint32_t(s2.W); }
        // The post block's plan row carries its *input* level, and it doubles.
        if (s2.type == "CCTinlayoutFusedPostBlockSwin1H") { out_w = uint32_t(2 * s2.H); out_h = uint32_t(2 * s2.W); }
    }
    const int source_w=std::atoi(arg(argc, argv, "--source-width", "0"));
    const int source_h=std::atoi(arg(argc, argv, "--source-height", "0"));
    if (arg(argc,argv,"--source-width") || arg(argc,argv,"--source-height")) {
        if (source_w <= 0 || source_h <= 0 || uint32_t(source_w)>in_w || uint32_t(source_h)>in_h) {
            std::fprintf(stderr,"source extent must be positive and fit inside the working extent\n");
            return 1;
        }
        std::printf("native working extent %ux%u; source/output %dx%d\n",in_w,in_h,source_w,source_h);
        in_w=out_w=uint32_t(source_w);in_h=out_h=uint32_t(source_h);
    }
    // `--tex-in-format N` (the runtime, one pass at the frame's own extent): the
    // input image in the caller's format, filled by a plain copy each frame.
    // Every kernel samples it, and sampling converts exactly as the blit into
    // RGBA32F did, so the network sees the same values for a quarter of the
    // bytes. Nothing writes it as storage on that path.
    const VkFormat tex_in_format = arg(argc, argv, "--tex-in-format")
        ? VkFormat(std::atoi(arg(argc, argv, "--tex-in-format"))) : VK_FORMAT_R32G32B32A32_SFLOAT;
    tex_in =
        ctx.image(std::max(in_w, 1u), std::max(in_h, 1u),
                  tex_in_format, true, true, tex_in_format == VK_FORMAT_R32G32B32A32_SFLOAT);
    // `--out0-format N` (the runtime, native compose, one pass): the post
    // block's answer image in the frame's own format, so the write-back is a
    // copy. Only formats whose store converts exactly as the write-back blit
    // did (nr_runtime.cpp); --out-image cannot read such an image back.
    const VkFormat out0_format = arg(argc, argv, "--out0-format")
        ? VkFormat(std::atoi(arg(argc, argv, "--out0-format"))) : VK_FORMAT_R32G32B32A32_SFLOAT;
    surf0 =
        ctx.image(std::max(out_w, 1u), std::max(out_h, 1u),
                  out0_format, false);
    surf1 =
        ctx.image(std::max(out_w, 1u), std::max(out_h, 1u),
                  VK_FORMAT_R32G32B32A32_SFLOAT, false);
    nr::logf("images in %ux%u, out %ux%u (x2)", in_w, in_h, out_w, out_h);
    timer.mark("arena-alloc");
    nr::g_build_stage = 3;
    // A frame to sample. `--in-image` takes an RGBA32F blob; without one the
    // input is a gradient, which is enough to see the pipeline carry data and
    // is *not* enough to call the result an image - the gate check applies.
    {
        // Written a band of rows at a time (nrvk::Context::image_rows): the
        // frame is never held whole in host or mapped memory. The runtime
        // overwrites this image every frame.
        const uint32_t w = std::max(in_w, 1u), h = std::max(in_h, 1u);
        const std::vector<uint8_t> blob = slurp(arg(argc, argv, "--in-image", ""));
        const size_t floats = size_t(w) * h * 4;
        if (arg(argc, argv, "--in-image") && blob.size() != floats * 4) {
            std::fprintf(stderr, "input image has %zu bytes; plan requires %zu (%ux%u RGBA32F)\n",
                         blob.size(), floats * 4, in_w, in_h);
            return 1;
        }
        // one row of the RGBA32F picture: the blob, or a gradient (enough to
        // see the pipeline carry data, *not* enough to call the result an image)
        auto row_f32 = [&](uint32_t y, float* q) {
            if (!blob.empty()) { std::memcpy(q, blob.data() + size_t(y) * w * 16, size_t(w) * 16); return; }
            for (uint32_t x = 0; x < w; ++x, q += 4) {
                if (y >= in_h || x >= in_w) { q[0] = q[1] = q[2] = q[3] = 0.0f; continue; }
                q[0] = float(x) / float(in_w ? in_w : 1);
                q[1] = float(y) / float(in_h ? in_h : 1);
                q[2] = 0.5f; q[3] = 1.0f;
            }
        };
        if (tex_in_format == VK_FORMAT_R32G32B32A32_SFLOAT)
            ctx.image_rows(tex_in, VkDeviceSize(floats) * 4, true, [&](uint32_t y0, uint32_t n, uint8_t* stage) {
                for (uint32_t y = y0; y < y0 + n; ++y)
                    row_f32(y, reinterpret_cast<float*>(stage + size_t(y - y0) * w * 16));
            });
        else if (tex_in_format == VK_FORMAT_B8G8R8A8_UNORM || tex_in_format == VK_FORMAT_R8G8B8A8_UNORM) {
            // The same picture as an 8-bit frame, for checking the runtime's
            // 8-bit input path from the command line (--in-image on k/255).
            std::vector<float> line(size_t(w) * 4);
            ctx.image_rows(tex_in, VkDeviceSize(floats), true, [&](uint32_t y0, uint32_t n, uint8_t* stage) {
                for (uint32_t y = y0; y < y0 + n; ++y) {
                    row_f32(y, line.data());
                    uint8_t* b8 = stage + size_t(y - y0) * w * 4;
                    for (size_t i = 0; i < line.size(); i += 4)
                        for (int k = 0; k < 4; ++k) {
                            const int src = tex_in_format == VK_FORMAT_B8G8R8A8_UNORM && k < 3 ? 2 - k : k;
                            b8[i + k] = uint8_t(std::lround(std::min(std::max(line[i + src], 0.f), 1.f) * 255.f));
                        }
                }
            });
        }
    }
    timer.mark("input-image");
    nr::logf("uploading weights: %.1f MB through a %.0f MB staging buffer",
             double(wblob.size()) / 1e6, double(std::min<VkDeviceSize>(wblob.size(), nrvk::Context::kStageBytes)) / 1e6);
    ctx.upload_with(wgt, 0, wblob.size(), [&](VkDeviceSize off, VkDeviceSize n, uint8_t* dst) {
        wblob.read(size_t(off), dst, size_t(n));
    });
    // The host copy is not needed again; free it before the runtime allocates.
    wblob.clear();
    timer.mark("weights-upload");
    // The pre-block's noise features (NR_NOISE_FIELD): a fixed function of pixel,
    // seed and gain, filled once here into the zeroed regions the lowering put
    // in the weight blob, by the same GLSL the pre-block would run per frame.
    if (!g_noise_jobs.empty()) {
        nrvk::Kernel nk;
        nk.create(ctx, spv_dir + "/g_noisefield.spv", {wgt.handle}, uint32_t(sizeof(NoiseJob)));
        ctx.one_shot([&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, nk.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, nk.layout, 0, 1, &nk.set, 0, nullptr);
            for (const NoiseJob& j : g_noise_jobs) {
                vkCmdPushConstants(cmd, nk.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof j, &j);
                vkCmdDispatch(cmd, (j.width + 7u) / 8u, (j.height + 7u) / 8u, 1);
            }
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 0, 1, &mb, 0, nullptr, 0, nullptr);
        });
        nk.destroy();
        timer.mark("noise-field");
    }

    // **The arena is not seeded here.** It used to be filled with a plausible
    // activation distribution, one staged 16 MB upload at a time, because an
    // uninitialised arena is e4m3 NaN about one byte in 128 and one of those
    // destroys the next MMA's whole accumulator tile. Then the zero fill below
    // was added - a single vkCmdFillBuffer over the same range, for the reason
    // its own comment gives - and it overwrites every byte of that upload
    // before anything reads it. So the seed was doing nothing except allocating
    // a host-visible staging buffer, submitting and fencing once per 16 MB:
    // 0.10 s at the 309 MB arena of a 960x640 build, and ~1.3 s at the 4.07 GB
    // one a 4K feature asks for. Removing it is bit-identical by construction.
    // ---- does any dispatch read and write the same slot? --------------------
    // **A layer whose output offset is its input offset is a race, not an
    // optimisation.** Every dispatch in this graph is separated from the next by
    // a full barrier, so aliasing *between* layers is ordered - but inside one
    // dispatch nothing orders one workgroup against another, and a fused Swin
    // block reads a whole shifted window, so workgroup A reads tokens workgroup
    // B is writing. The result is a frame that is not reproducible: three runs
    // of `--only 0:0` on an identically seeded arena gave three different
    // outputs, and ~97% of the final image's floats differ by ~0.1% run to run.
    //
    // The check is here rather than in a comment because the allocator recycles
    // slots by last use, so any future plan or allocator change can reintroduce
    // it silently.
    {
        int aliased = 0;
        for (const Disp& d : disp) {
            if (d.push.size() < 16 || is_persist_kern(d.kern)) continue;
            uint32_t f[7] = {};
            std::memcpy(f, d.push.data(), std::min<size_t>(d.push.size(), 28));
            // Field order is per family, so the roles are spelled out rather
            // than guessed: every *activation* input against the output.
            std::vector<uint32_t> in;
            uint32_t oo = 0;
            if (d.kern == "fswinblendpost32" || d.kern == "fswinimagepost32" || d.kern.rfind("fswinfusedup",0)==0) {
                PushUps blend{};
                std::memcpy(&blend, d.push.data() + sizeof(PushFSwin), sizeof blend);
                in = {blend.p_off, blend.s_off}; oo = f[1];
            } else if (d.kern.rfind("gemm", 0) == 0) {          // x, residual -> o
                in = {f[1], f[2]}; oo = f[3];
            } else if (d.kern.rfind("upsblend", 0) == 0) {  // projection, skip -> o
                in = {f[0], f[1]}; oo = f[2];
            } else if (d.kern.rfind("imgin", 0) == 0 || d.kern.rfind("imgout", 0) == 0) {
                continue;                                 // one side is an image
            } else if (d.kern == "attn") {               // x -> o, weights elsewhere
                in = {f[0]}; oo = f[1];
            } else {                                      // fswin, pool, ffwd3, vitattn
                in = {f[0]}; oo = f[1];
            }
            for (uint32_t xo : in)
                if (xo == oo && aliased++ < 8)
                    std::fprintf(stderr,
                                 "  ** b%03dl%d %s (%s) reads and writes offset %u:"
                                 " one dispatch, no ordering between workgroups **\n",
                                 d.s->block, d.s->layer, d.s->type.c_str(),
                                 d.kern.c_str(), xo);
        }
        if (aliased)
            std::fprintf(stderr, "  ** %d dispatches alias their own input **\n", aliased);
    }
    // Audited C=32 steps (g_nohi) take the variant built without the upper exponent
    // clamp they never reach, when the shader directory has it.
    if (!(std::getenv("NR_EXP_NOHI") && !std::atoi(std::getenv("NR_EXP_NOHI")))) {
        int n = 0;
        for (Disp& d : disp)
            if ((d.kern == "fswin32" || d.kern == "fswinimagepreds32" || d.kern == "fswin32t" ||
                 (!nr::binary::directory().empty() && (d.kern == "fswindsp32" || d.kern == "fswinfusedup32"))) && d.s &&
                g_nohi.count({d.s->block, d.s->layer})) {
                // The ACO upsample variant needs a complete, unshifted grid,
                // matching the Linux source's qualification for this binary.
                if (!nr::binary::directory().empty() && d.kern == "fswinfusedup32") {
                    PushFSwin p{}; std::memcpy(&p, d.push.data(), sizeof p);
                    if (p.shift || p.shift_y || p.tiles_x != 2u*d.gx || p.tiles_y != 2u*d.gy) continue;
                }
                // fswin32t (the frame tick's layer, NR_TCHAIN) -> fswin32nht
                const std::string nh = d.kern == "fswin32t" ? std::string("fswin32nht") : d.kern + "nh";
                if (std::filesystem::exists(spv_dir + "/g_" + nh + ".spv")) { d.kern = nh; ++n; }
            }
        if (n) std::printf("exp upper clamp dead (weights): %d C=32 dispatches\n", n);
    }
    // ---- pipelines ---------------------------------------------------------
    // Bindings differ per shader and the buffers do not: every kernel in this
    // project reaches both arenas through push-constant offsets, which is what
    // lets one descriptor set per *shader* serve all of that shader's layers.
    {
        std::set<std::string> kinds;
        for (const Disp& d : disp) if (!kern.count(d.kern)) kinds.insert(d.kern);
        nr::g_build_pipes_total = uint32_t(kinds.size()); nr::g_build_pipes_done = 0; nr::g_build_stage = 4;
    }
    for (const Disp& d : disp) {
        if (kern.count(d.kern)) continue;
        struct Tick { ~Tick() { ++nr::g_build_pipes_done; } } tick;
        std::string p = spv_dir + "/g_" + d.kern + ".spv";
        // The Windows network's vitattn and residual GEMMs read through the arena's device
        // address (NR_VBDA, NR_RBDA); a device without bufferDeviceAddress runs the
        // storage-buffer build of the same kernel, g_<name>nobda.spv.
        if (!act.address) {
            const std::string plain = spv_dir + "/g_" + d.kern + "nobda.spv";
            if (std::ifstream(plain).good()) p = plain;
        }
        if (d.kern.rfind("fswindsp",0)==0)
            kern[d.kern].create(ctx,p,{act.handle,act.handle,wgt.handle,wgt.handle,wgt.handle,act.handle},
                                sizeof(PushFSwin)+sizeof(PushDsProj) + (NR_TCHAIN_TILES && tchain_kern(d.kern) ? 4 : 0));
        else if (d.kern.rfind("fswinimagepreds32",0)==0)
            kern[d.kern].create(ctx,p,{act.handle,act.handle,wgt.handle,wgt.handle,wgt.handle,act.handle},
                                sizeof(PushFSwin)+sizeof(PushPreImage),{&tex_in});
        else if (d.kern=="fswinimagepost32")
            kern[d.kern].create(ctx,p,{act.handle,act.handle,wgt.handle,wgt.handle,wgt.handle},
                                sizeof(PushFSwin)+sizeof(PushUps)+sizeof(PushImageTail),
                                {&surf0,&surf1,&tex_in});
        else if (d.kern.rfind("fswinfusedup",0)==0)
            kern[d.kern].create(ctx,p,{act.handle,act.handle,wgt.handle,wgt.handle,wgt.handle},
                                uint32_t(d.push.size()));
        else if (d.kern=="fswinblendpost32")
            kern[d.kern].create(ctx,p,{act.handle,act.handle,wgt.handle,wgt.handle,wgt.handle},
                                sizeof(PushFSwin)+sizeof(PushUps));
        else if (is_persist_kern(d.kern))
            kern[d.kern].create(ctx, p, {act.handle, act.handle, wgt.handle,
                                         wgt.handle, wgt.handle, act.handle},
                                sizeof(PushPersist) + (NR_TCHAIN_TILES && tchain_kern(d.kern) ? 4 : 0));
        else if (d.kern.rfind("fswin", 0) == 0)
            kern[d.kern].create(ctx, p, {act.handle, act.handle, wgt.handle,
                                         wgt.handle, wgt.handle, act.handle},
                                sizeof(PushFSwin) + (NR_TCHAIN_TILES && tchain_kern(d.kern) ? 4 : 0));
        else if (d.kern.rfind("pool", 0) == 0)
            kern[d.kern].create(ctx, p, {act.handle}, sizeof(PushPool));
        else if (d.kern == "decups")
            kern[d.kern].create(ctx, p, {act.handle, act.handle, wgt.handle}, sizeof(PushDecoderUps));
        else if (d.kern == "qkvnorm")
            kern[d.kern].create(ctx, p, {act.handle, act.handle, wgt.handle}, sizeof(PushQkvNorm));
        else if (d.kern == "repack")
            kern[d.kern].create(ctx, p, {act.handle}, sizeof(PushRepack));
        else if (d.kern == "upsview")
            kern[d.kern].create(ctx, p, {act.handle}, sizeof(PushUpsView));
        else if (d.kern.rfind("imgin", 0) == 0)
            kern[d.kern].create(ctx, p, {act.handle, wgt.handle}, sizeof(PushImgIn),
                                {&tex_in});
        else if (d.kern.rfind("imgout", 0) == 0)
            kern[d.kern].create(ctx, p, {act.handle, wgt.handle}, sizeof(PushImgOut),
                                {&surf0, &surf1, &tex_in});
        else if (d.kern == "vitattn")
            kern[d.kern].create(ctx, p, {act.handle, wgt.handle, act.handle},
                                sizeof(PushVAttn) + (g_chain_kern.count(d.kern) ? 16 : 0) + (NR_TCHAIN_TILES ? 4 : 0));
        else if (d.kern.rfind("upsblend", 0) == 0)
            kern[d.kern].create(ctx, p, {act.handle, wgt.handle}, sizeof(PushUps));
        else if (NR_KBASE(d.kern) == "attn")
            kern[d.kern].create(ctx, p, {act.handle, act.handle, wgt.handle,
                                         wgt.handle, act.handle}, sizeof(PushAttn) + (g_chain_kern.count(d.kern) ? 16 : 0) + (NR_TCHAIN_TILES ? 4 : 0));
        else if (d.kern == "ffwd3" || d.kern == "ffwd3w")
            kern[d.kern].create(ctx, p, {act.handle, act.handle, act.handle,
                                         wgt.handle, wgt.handle, act.handle},
                                sizeof(PushFfwd3) + (g_chain_kern.count(d.kern) ? 16 : 0) + (NR_TCHAIN_TILES ? 4 : 0));
        else
            kern[d.kern].create(ctx, p, {act.handle, act.handle, act.handle, wgt.handle,
                                         wgt.handle, act.handle, wgt.handle, act.handle},
                                sizeof(PushGemm) + (g_chain_kern.count(d.kern) ? 16 : 0) +
                                (NR_TCHAIN_TILES && tchain_kern(d.kern) ? 4 : 0));
    }
    for (const Disp& d : disp)   // a push longer than its pipeline's range reads garbage past the end
        if (kern.count(d.kern) && d.push.size() > kern.at(d.kern).push_range)
            throw std::runtime_error("push constants longer than the pipeline's range: " + d.kern);
    const size_t cache_bytes = ctx.save_pipeline_cache();
    timer.mark("pipelines");
    nr::g_build_stage = 5;
    if (nr::g_memory_note) nr::logf("pipelines created; %s", nr::g_memory_note().c_str());
    // `--wiring` prints what each dispatch reads and writes. A producer and a
    // consumer that disagree about a slot is invisible to the seeded board,
    // because the seeder writes the gold into the producer's slot and the
    // consumer is then the only one that notices.
    if (flag(argc, argv, "--wiring")) {
        std::printf("%-8s %-34s %12s %12s\n", "layer", "kernel", "reads", "writes");
        for (const Disp& d : disp) {
            const Step& s2 = *d.s;
            std::printf("b%03dl%d %-34s %12u %12u\n", s2.block, s2.layer,
                        d.kern.c_str(),
                        d.push.size() >= 8 ? *reinterpret_cast<const uint32_t*>(d.push.data() + 0) : 0u,
                        d.push.size() >= 8 ? *reinterpret_cast<const uint32_t*>(d.push.data() + 4) : 0u);
        }
    }
    runner.create(ctx);
    // **Zero the activation arena, which nothing ever did.** `Runner` has had
    // the field since the arena was introduced and no caller ever set it, so
    // every slot's unwritten tail held whatever the driver last left there.
    // That tail is not small in consequence: a layer's slot is sized for the
    // *padded* token count and the kernels address only the unpadded image, so
    // 3840 of b001's 522240 tokens - 0.74% - were stale, and the next layer's
    // windowed attention mixes them straight into its own output. Seeded, the
    // gold overwrote them and every layer scored 0.99; chained, `b002` fell to
    // **-0.29 from a 0.998-correct input**, and the whole frame followed.
    // NVIDIA's own capture has those 3840 tokens at **exactly zero**.
    // **Once, here, and then off.** `run_graph` records the fill at the top of
    // its command buffer, so leaving it armed would wipe the seeds that
    // `--seed-dir` uploads after `build` returns - which reads as "every layer
    // is exactly zero". Fill now, disarm, and every later submit keeps what the
    // seeder put there.
    runner.zero_buffer = act.handle;
    runner.zero_bytes = VkDeviceSize(act_total);
    // **What arena reuse has to keep zero** (diagnostics, off unless asked for,
    // one-value-one-slot layout). NR_ARENA_UNWRITTEN=1 fills the values with a
    // pattern, runs one frame and lists every slot's bytes that were never
    // written; NR_POISON_KEYS=k1,k2|all starts those slots non-zero, so an
    // output that changes names a slot whose unwritten bytes are read. That is
    // how `overread` was found - rerun both after changing a kernel's padding.
    if (std::getenv("NR_ARENA_UNWRITTEN")) {
        std::vector<nrvk::Runner::Step> all;
        for (const Disp& d : disp) all.push_back({&kern[d.kern], d.gx, d.gy, d.gz, d.push.data(), uint32_t(d.push.size())});
        // Values only: the persistent runs' sync words past them must start at zero.
        runner.poison.push_back({0, VkDeviceSize(values_end)});
        runner.poison_value = 0xA5A5A5A5u;
        runner.run_graph(all, 1);
        runner.poison.clear();
        runner.poison_value = 0x3C3C3C3Cu;
        std::vector<uint32_t> w(act_total / 4);
        ctx.download(act, 0, w.data(), w.size() * 4);
        if (!gaps.empty()) {
            const size_t gap = size_t(std::atoll(std::getenv("NR_ARENA_GAP")));
            for (size_t g = 0; g < gaps.size(); ++g) {
                size_t hit = 0, last = 0;
                for (size_t i = 0; i < gap / 4; ++i)
                    if (w[gaps[g].second / 4 + i] != 0xA5A5A5A5u) { ++hit; last = i * 4 + 3; }
                if (hit) std::printf("gap %zu after key %d (b%d/%d) written: %zu words, up to +%zu\n", g,
                                     gaps[g].first, gaps[g].first / 8, gaps[g].first % 8, hit, last);
            }
        }
        for (const auto& kv : vsize) {
            if (!kv.second || !voff.count(kv.first)) continue;
            const size_t o = voff.m.at(kv.first) / 4, n = kv.second / 4;
            size_t cnt = 0, first = n, last = 0, runs = 0; bool in = false;
            for (size_t i = 0; i < n; ++i) {
                const bool u = w[o + i] == 0xA5A5A5A5u;
                if (u) { ++cnt; first = std::min(first, i); last = i; if (!in) ++runs; }
                in = u;
            }
            if (cnt) std::printf("unwritten key %d (b%d/%d) size %zu: %zu bytes in %zu runs, first %zu last %zu\n",
                                 kv.first, kv.first / 8, kv.first % 8, kv.second, cnt * 4, runs, first * 4, last * 4 + 3);
        }
        std::fflush(stdout);
    }
    if (const char* pg = std::getenv("NR_POISON_GAPS")) {
        const size_t gap = size_t(std::atoll(std::getenv("NR_ARENA_GAP")));
        const std::string sel = std::string(",") + pg + ",";
        for (size_t g = 0; g < gaps.size(); ++g)
            if (sel == ",all," || sel.find("," + std::to_string(g) + ",") != std::string::npos)
                runner.poison.push_back({gaps[g].second, gap});
    }
    if (const char* pk = std::getenv("NR_POISON_KEYS")) {
        std::istringstream is(pk); std::string t;
        if (std::string(pk) == "all") { for (const auto& kv : vsize) if (kv.second && voff.count(kv.first)) runner.poison.push_back({voff.m.at(kv.first), align(kv.second, 4)}); }
        else while (std::getline(is, t, ',')) if (!t.empty()) {
            const int k = std::atoi(t.c_str());
            if (voff.count(k) && vsize.count(k) && vsize.at(k))
                runner.poison.push_back({voff.m.at(k), align(vsize.at(k), 4)});
        }
    }
    runner.run_graph({}, 1);
    runner.poison.clear();
    runner.zero_buffer = VK_NULL_HANDLE;
    runner.zero_bytes = 0;
    // One pass of the graph a submit. A pass is a few milliseconds, which is
    // under a display frame, so the compositor is never blocked for longer than
    // that however many passes are asked for - and the GPU still boosts, because
    // the passes run back to back. Batching passes into one submit is what made
    // this harness starve the desktop.
    runner.chunk = uint32_t(std::atoi(arg(argc, argv, "--chunk", "1")));
    // `--no-barrier` prices the 176 inter-dispatch barriers a frame. It
    // computes the wrong picture on purpose - every layer reads a stale input -
    // and the benchmark's channel-mean guard will refuse the run, which is the
    // correct behaviour. Use it for the number and nothing else.
    runner.no_barrier = flag(argc, argv, "--no-barrier");
    runner.read_barrier = flag(argc, argv, "--read-barrier");
    runner.barrier_stride = uint32_t(std::atoi(arg(argc, argv, "--barrier-stride", "1")));
    runner.exec_barrier = flag(argc, argv, "--exec-barrier");
    // ---- the destination half of the barrier, and nothing else -------------
    // **This is what ships, and it replaced `coherent`.** Measured: no
    // shader-side barrier spelling emits a cache invalidate on this driver, so
    // the activation arena's read side can only be narrowed in the command
    // buffer: `srcAccessMask = 0` asks for the visibility operation - the
    // L0/GL1 invalidate - and not the availability one. The shaders then carry
    // no scope at all, every activation load is free to hit L0, and the
    // boundary costs one packet instead of `scope:SCOPE_DEV` on 935 loads and
    // 966 stores.
    //
    // Sound on gfx1201, and the argument is two steps, each anchored on a
    // configuration this tree already accepts:
    //
    //   the write side  every cache below the device coherence point is
    //                   write-through - LLVM's gfx1201 lowering emits
    //                   `global_wb` only at SCOPE_SYS, never at SCOPE_DEV, so
    //                   an agent-scope release is a store-counter drain and
    //                   nothing more. An unscoped store therefore reaches L2
    //                   exactly when a `scope:SCOPE_DEV` one does, and the
    //                   drain requirement is identical to the one the coherent
    //                   build already depends on with no memory barrier at all.
    //   the read side   `VK_ACCESS_SHADER_READ_BIT` as the destination is the
    //                   same invalidate the full barrier issued, and that is
    //                   what shipped before the arena was made coherent.
    //
    // So it is the coherent build's barrier plus an invalidate, and the
    // pre-coherent build's barrier minus an availability operation that this
    // part has no instruction for. `--no-inv-barrier` / `NR_INV_BARRIERS=0`
    // restores the full barrier.
    runner.inv_barrier = !coherent_act;
    if (const char* env = std::getenv("NR_INV_BARRIERS"))
        if (*env) runner.inv_barrier = std::atoi(env) != 0;
    if (flag(argc, argv, "--inv-barrier")) runner.inv_barrier = true;
    if (flag(argc, argv, "--no-inv-barrier")) runner.inv_barrier = false;
    // `--exec-barrier` above is the raw diagnostic: it drops the memory barrier
    // whatever the shaders were built with, and is unsafe unless they carry
    // device scope. `--coherent-barriers` is the same dependency with the
    // precondition checked - and it is on by default when the SPVs are
    // coherent, so a benchmark driver with no argument passthrough
    // measures what ships. `NR_COHERENT_BARRIERS=0` or
    // `--no-coherent-barriers` puts the full barrier back; the flag wins.
    {
        const char* env = std::getenv("NR_COHERENT_BARRIERS");
        bool want = coherent_act;
        if (env && *env) want = std::atoi(env) != 0;
        if (flag(argc, argv, "--coherent-barriers")) want = true;
        if (flag(argc, argv, "--no-coherent-barriers")) want = false;
        if (want && !coherent_act) {
            std::fprintf(stderr, "--coherent-barriers needs pipelines built with "
                         "NR_COHERENT_ACT=1; %s/coherent-act.txt says otherwise; "
                         "run windows/build/build_network.py\n", spv_dir.c_str());
            return 1;
        }
        if (want) runner.exec_barrier = true;
        // Both answer the same question and only one of them can be measured.
        if (want) runner.inv_barrier = false;
        std::printf("inter-dispatch barrier: %s\n",
                    want ? "execution only (coherent activation arena)"
                         : runner.inv_barrier ? "execution + invalidate (no availability op)"
                         : "execution + memory");
    }
    runner.storage_barrier = flag(argc, argv, "--storage-barrier");
    // **What is the barrier after *this* kernel worth?** `--no-barrier` prices
    // all 176 together and divides, which assigns the same 13 us to a 681 us
    // image adapter and to an 8 us norm. A fusion candidate needs the pair's
    // own number: the recoverable part of a boundary is the overlap between one
    // dispatch's drain and the next one's ramp, and that is a property of the
    // two dispatches, not of the frame. Same contract as `--no-barrier` - the
    // picture it computes is wrong, deliberately.
    if (const char* k = arg(argc, argv, "--no-barrier-after")) {
        runner.no_barrier_after.assign(disp.size(), 0);
        size_t n = 0;
        const std::string ks = std::string(",") + k + ",";   // comma list
        for (size_t i = 0; i < disp.size(); ++i)
            if (ks.find("," + disp[i].kern + ",") != std::string::npos) { runner.no_barrier_after[i] = 1; ++n; }
        std::printf("dropping %zu barriers after kernel %s\n", n, k);
    }
    if (!g_chain_kern.empty()) {
        if (runner.no_barrier_after.size() != disp.size()) runner.no_barrier_after.assign(disp.size(), 0);
        for (size_t i = 0; i < disp.size(); ++i) if (g_chain_nobar[i]) runner.no_barrier_after[i] = 1;
    }
    if (std::find(g_tc_nobar.begin(), g_tc_nobar.end(), 1) != g_tc_nobar.end()) {
        if (runner.no_barrier_after.size() != disp.size()) runner.no_barrier_after.assign(disp.size(), 0);
        for (size_t i = 0; i < disp.size(); ++i) if (g_tc_nobar[i]) runner.no_barrier_after[i] = 1;
    }
    // Diagnostic NR_DUP=k1,k2: run those kernels twice back to back (idempotent
    // layers only), so the second copy shows the kernel with warm caches.
    const std::string dupk = std::getenv("NR_DUP") ? std::string(",") + std::getenv("NR_DUP") + "," : "";
    for (const Disp& d : disp) {
        steps.push_back({&kern[d.kern], d.gx, d.gy, d.gz,
                         d.push.data(), uint32_t(d.push.size())});
        if (!dupk.empty() && dupk.find("," + d.kern + ",") != std::string::npos)
            steps.push_back(steps.back());
    }
    timer.mark("arena-zero");
    nr::logf("[nr] graph build %.2fs: %s(%u pipelines, %.1f MB of pipeline cache)",
             timer.total(), timer.text().c_str(), unsigned(kern.size()),
             double(cache_bytes) / 1e6);
    return 0;
}

// ---- the command line -------------------------------------------------------
// Aliases, so that everything below is the code it always was.
// `nr_runtime.cpp` includes this file to get the whole graph builder, so the
// command-line front end is compiled out for it. One translation unit is how
// every other tool in this port is put together.
#ifndef NR_NO_MAIN
int main(int argc, char** argv) try {
    static NrSession S;
    const int rc = S.build(argc, argv);
    if (rc) return rc == 2 ? 0 : 1;
    auto& ctx = S.ctx; auto& act = S.act; auto& wgt = S.wgt;
    auto& tex_in = S.tex_in; auto& surf0 = S.surf0; auto& surf1 = S.surf1;
    auto& kern = S.kern; auto& disp = S.disp; auto& plan = S.plan;
    auto& voff = S.voff; auto& missing = S.missing; auto& runner = S.runner;
    auto& steps = S.steps; auto& x_src = S.x_src; auto& run = S.run;
    auto& reader_tokens = S.reader_tokens;
    auto& persist_err = S.persist_err;
    const bool host_boundary = S.host_boundary;
    auto& in_w = S.in_w; auto& in_h = S.in_h;
    auto& out_w = S.out_w; auto& out_h = S.out_h;
    const char* seed_path = S.seed_path;
    const char* score_dir = S.score_dir;
    const uint32_t repeats = uint32_t(std::atoi(arg(argc, argv, "--repeats", "1")));
    const uint32_t warm = uint32_t(std::atoi(arg(argc, argv, "--warmup", "0")));
    (void)ctx; (void)act; (void)wgt; (void)tex_in; (void)in_w; (void)in_h;

    // **A bounded spin returns rather than hangs, so the wrong picture is the
    // only symptom left.** Every merged run keeps an error word; anything but
    // zero means a consumer gave up on a producer and read whatever was there.
    auto persist_check = [&]() {
        for (uint32_t i = 0; g_chain_n && i < g_chain_n; ++i) {
            uint32_t e[2] = {0, 0};
            ctx.download(act, VkDeviceSize(g_chain_base + 4096 * i) * 4, e, sizeof e);
            if (e[1]) std::printf("chain error: dispatch %u (%s) timed out waiting\n", i, disp[i].kern.c_str());
        }
        for (uint32_t off : persist_err) {
            uint32_t e = 0;
            ctx.download(act, off, &e, sizeof e);
            if (e) std::printf("persist error: layer %u\n", e - 1u);
        }
    };
    auto write_surface = [&]() {
    if (surf0.format != VK_FORMAT_R32G32B32A32_SFLOAT) {
        std::printf("surface 0: not RGBA32F (--out0-format), not read back\n");
        return;
    }
    if (out_w && out_h) {
        std::vector<float> px(size_t(out_w) * out_h * 4);
        ctx.download(surf0, px.data(), px.size() * 4);
        size_t nz = 0;
        double s2 = 0, lo = 1e30, hi = -1e30;
        for (float v : px) {
            if (v != 0.0f) ++nz;
            s2 += double(v) * v;
            lo = std::min(lo, double(v)); hi = std::max(hi, double(v));
        }
        std::printf("surface 0: %ux%u RGBA32F, %.1f%% non-zero, rms %.4g, "
                    "range [%.4g, %.4g]\n", out_w, out_h,
                    100.0 * double(nz) / double(px.size()),
                    std::sqrt(s2 / double(px.size())), lo, hi);
        if (const char* op = arg(argc, argv, "--out-image")) {
            FILE* f = std::fopen(op, "wb");
            if (!f) throw std::runtime_error(std::string("cannot open output image: ") + op);
            const size_t written = std::fwrite(px.data(), sizeof(float), px.size(), f);
            const int closed = std::fclose(f);
            if (written != px.size() || closed != 0)
                throw std::runtime_error(std::string("cannot write output image: ") + op);
            std::printf("  wrote %s\n", op);
        }
    }
    // `--dump-layer B:L --dump-layer-to FILE`: one layer's output as it sits in
    // the arena (tile-blocked, up to the next value's offset), for comparing two
    // builds or two drivers byte for byte after a normal run.
    if (const char* dl = arg(argc, argv, "--dump-layer")) {
        const int b = std::atoi(dl);
        const char* colon = std::strchr(dl, ':');
        const int key = key_of(b, colon ? std::atoi(colon + 1) : 0);
        auto it = voff.m.find(key);
        if (it == voff.m.end()) throw std::runtime_error(std::string("--dump-layer: no value ") + dl);
        size_t end = act.bytes;
        for (const auto& kv : voff.m) if (kv.second > it->second) end = std::min(end, size_t(kv.second));
        std::vector<uint8_t> bytes(end - it->second);
        ctx.download(act, VkDeviceSize(it->second), bytes.data(), bytes.size());
        const char* to = arg(argc, argv, "--dump-layer-to", "layer.bin");
        FILE* f = std::fopen(to, "wb");
        if (!f || std::fwrite(bytes.data(), 1, bytes.size(), f) != bytes.size())
            throw std::runtime_error(std::string("cannot write ") + to);
        std::fclose(f);
        std::printf("  dumped b%03d value %s: %zu bytes -> %s\n", b, dl, bytes.size(), to);
    }
    // `--dump-all DIR`: every activation value, one file per arena key, in arena
    // order (DIR/NNN-kKEY.bin), to find the first value two runs disagree on.
    if (const char* dir = arg(argc, argv, "--dump-all")) {
        std::vector<std::pair<size_t, int>> order;
        for (const auto& kv : voff.m) order.push_back({kv.second, kv.first});
        std::sort(order.begin(), order.end());
        std::map<int, std::string> label;
        for (const auto& s : plan) {
            char l[96];
            std::snprintf(l, sizeof l, "b%03dl%d %s", s.block, s.layer, s.type.c_str());
            label[key_of(s.block, s.layer)] = l;
        }
        FILE* index = std::fopen((std::string(dir) + "/index.txt").c_str(), "w");
        size_t total = 0;
        for (size_t i = 0; i < order.size(); ++i) {
            if (index) std::fprintf(index, "%03zu-k%d %s\n", i, order[i].second,
                                    label.count(order[i].second) ? label[order[i].second].c_str() : "-");
            const size_t end = i + 1 < order.size() ? order[i + 1].first : size_t(act.bytes);
            if (end <= order[i].first) continue;
            std::vector<uint8_t> bytes(end - order[i].first);
            ctx.download(act, VkDeviceSize(order[i].first), bytes.data(), bytes.size());
            char name[64];
            std::snprintf(name, sizeof name, "/%03zu-k%d.bin", i, order[i].second);
            FILE* f = std::fopen((std::string(dir) + name).c_str(), "wb");
            if (!f || std::fwrite(bytes.data(), 1, bytes.size(), f) != bytes.size())
                throw std::runtime_error(std::string("cannot write ") + dir + name);
            std::fclose(f);
            total += bytes.size();
        }
        if (index) std::fclose(index);
        std::printf("  dumped %zu values, %zu bytes -> %s\n", order.size(), total, dir);
    }
    };

    // ---- correctness first, when asked -------------------------------------
    if (score_dir) {
        // ---- bisecting a `_ds` layer ------------------------------------
        // Legacy diagnostic only: pool a captured FP8 skip and resample it.
        // This does not validate the original half-register pooling path.
        const int ds_only = std::atoi(arg(argc, argv, "--ds-only", "-1"));
        if (ds_only >= 0) {
            const Step* dss = nullptr;
            for (const Step* r : run)
                if (r->block == ds_only && shape_of(*r).fam == F_DS) dss = r;
            if (!dss) { std::fprintf(stderr, "block %d is not a _ds layer\n", ds_only); return 1; }
            char cap[256];
            std::snprintf(cap, sizeof cap, host_boundary ? "%s/b%03dl%d_o1.bin" : "%s/b%03dl%d_skip.bin",
                          score_dir, dss->block, dss->layer);
            const std::vector<uint8_t> sk = slurp(cap);
            const size_t IW = size_t(dss->H), IH = size_t(dss->W), C = size_t(dss->C);
            const size_t TW = size_t(tiles_of(int(IW))) * 4, M = TW * IH;
            if (sk.size() < M * C) {
                std::fprintf(stderr, "%s is %zu B, need %zu\n", cap, sk.size(), M * C);
                return 1;
            }
            std::vector<uint8_t> canon(M * C, 0);
            for (size_t y = 0; y < IH; ++y)
                for (size_t x = 0; x < IW; ++x) {
                    const size_t t = tin::token_of(x, y, TW);
                    for (size_t c = 0; c < C; ++c)
                        canon[t * C + c] = sk[tin::byte_of(t, c, C)];
                }
            const std::vector<uint8_t> tb = tin::tile_blocked(canon.data(), M, C);
            ctx.upload(act, voff[skip_key(dss->block)], tb.data(), tb.size());
            std::vector<nrvk::Runner::Step> two;
            for (size_t i = 0; i < disp.size(); ++i)
                if (disp[i].s == dss &&
                    (disp[i].kern.rfind("pool", 0) == 0 || disp[i].kern.rfind("upsblend", 0) == 0 ||
                     disp[i].kern == "gemmds" || disp[i].kern == "gemmups"))
                    two.push_back(steps[i]);
            std::printf("_ds bisect: block %d, skip seeded from %s, %zu dispatches\n",
                        ds_only, cap, two.size());
            runner.run_graph(two, 1);
            std::snprintf(cap, sizeof cap, "%s/b%03dl%d.bin", score_dir, dss->block, dss->layer);
            const std::vector<uint8_t> ref = slurp(cap);
            const size_t OC = size_t(2 * dss->C), VW = IW / 2, VH = IH / 2;
            const size_t OTW = size_t(tiles_of(int(VW))) * 4;
            std::vector<uint8_t> got(OTW * VH * OC);
            ctx.download(act, voff[key_of(dss->block, dss->layer)], got.data(), got.size());
            const std::vector<uint8_t> ours = tin::un_tile_blocked(got.data(), OTW * VH, OC);
            size_t same = 0, tot = 0;
            for (size_t y = 0; y < VH; ++y)
                for (size_t x = 0; x < VW; ++x) {
                    const size_t t = tin::token_of(x, y, OTW);
                    for (size_t c = 0; c < OC; ++c, ++tot) {
                        const size_t o = tin::view_byte_of(x, y, c, VW, VH);
                        if (o < ref.size() && ours[t * OC + c] == ref[o]) ++same;
                    }
                }
            std::printf("pool+resample alone: %.2f%% of %zu  (host model reaches 60.3%%)\n",
                        100.0 * double(same) / double(tot), tot);
            return 0;
        }

        // Which layer the chain starts at. Seeding block 1 asks the whole graph
        // *and* the undecoded `_ds`-to-`_inpview` view seam at once; seeding
        // block 2 from a plain tinlayout capture asks only the graph, and the
        // single-layer harness already scores 68% on that same pair - so the two
        // runs together say whether the wiring or the seam is at fault.
        const int seed_block = std::atoi(arg(argc, argv, "--seed-block", "-1"));
        // A block's later stages read what its earlier stages wrote, so scoring
        // them off our own chain scores the first stage four times over. Where
        // the capture has every stage's output - block 23 does - seeding at
        // *layer* granularity asks each stage the question on its own.
        const int seed_layer = std::atoi(arg(argc, argv, "--seed-layer", "0"));
        const Step* seed_step = run.front();
        if (seed_block >= 0)
            for (const Step* r : run)
                if (r->block == seed_block && r->layer == seed_layer) { seed_step = r; break; }
        if (seed_path) {
            const Step& first = *seed_step;
            const std::vector<uint8_t> capin = slurp(seed_path);
            const size_t C = size_t(first.C);
            // **The axes.** `+208` is the height and `+212` the width, so the
            // plan's gx is ceil(H/8) and gy is ceil(W/8) - the two were inverted
            // together and cancel in the launch, but anything that has to agree
            // with the image needs them the right way round. In the canonical
            // token order the image is therefore `gx*8` wide by `gy*8` tall, and
            // the kernel's `tiles_x` is that same fast axis.
            const size_t IW = size_t(first.H), IH = size_t(first.W);
            // The token grid is whole 4-pixel tiles - `tiles_of` truncates, and a
            // partial tile at the right edge is not addressed.
            const size_t TW = size_t(tiles_of(int(IW))) * 4;
            // **Which layout the capture is in depends on the consumer.** An
            // `_inpview` block reads the level-boundary "view" format its `_ds`
            // producer wrote - 16-channel planes, raster pixels, a permuted
            // sub-channel - not the tinlayout. Block 1 is such a block, which is
            // why seeding it as tinlayout scored 0.81%.
            const bool view = flag(argc, argv, "--seed-view");
            // **Only a view seed needs an image.** A tinlayout seed is a copy in
            // token order, and the `CCSplitSwin16H*` family has no image shape in
            // the plan at all - its H/W columns are 33x60 where the capture holds
            // 2560 tokens. Deriving M from those columns silently seeds 1920 of
            // them and then walks off the end of the host buffer.
            const size_t TH = size_t(tiles_of(int(IH))) * 4;
            const size_t M = view ? TW * TH : size_t(first.tokens);
            std::printf("seed shape: %zux%zu grid %zu, M=%zu C=%zu, need %zu B of %zu\n",
                        IW, IH, TW, M, C, M * C, capin.size());
            // `tile_blocked` writes 16x16 fragments and walks off the end of its
            // buffer if either extent is not a multiple of 16 - as a *host* heap
            // overrun, which surfaces as `malloc(): invalid size` somewhere else
            // entirely. A plan row whose family has no image shape at all gives
            // C=0 and lands here. Say so instead of corrupting the heap.
            if (C == 0 || M == 0 || M % 16 || C % 16) {
                std::fprintf(stderr, "seeding b%dl%d needs a 16-aligned token grid, "
                             "got M=%zu C=%zu - the plan row has no image shape\n",
                             first.block, first.layer, M, C);
                return 1;
            }
            if (capin.size() < M * C) {
                std::fprintf(stderr, "%s is %zu B, need %zu (%zux%zu x %zu)\n",
                             seed_path, capin.size(), M * C, IW, IH, C);
                return 1;
            }
            // The view layout's raster is `(y * W + x) * 16` and which of the
            // two extents is its W is a separate question from which is the
            // token order's - the plan's W and H were inverted together, so a
            // pair that cancels in the launch need not cancel here.
            const bool swap = flag(argc, argv, "--seed-swap");
            // **The view raster's extent need not be the token grid's.** The
            // token grid is padded to whole 8-pixel windows - 540 becomes 544 -
            // while a `_ds` layer's second output is sized `((H+1)/2 + 3) & ~3`,
            // which pads to whole 4-pixel tiles and stops at 540. A buffer whose
            // *rows* are 540 long read as though they were 544 is scrambled from
            // the second row on, and that is exactly what a 3% score looks like.
            size_t VW = swap ? IH : IW, VH = swap ? IW : IH;
            if (const char* wh = arg(argc, argv, "--seed-view-wh")) {
                VW = size_t(std::atoi(wh));
                const char* comma = std::strchr(wh, ',');
                VH = comma ? size_t(std::atoi(comma + 1)) : VH;
            }
            std::vector<uint8_t> canon(M * C, 0);
            if (!view) {
                for (size_t t = 0; t < M; ++t)
                    for (size_t c = 0; c < C; ++c)
                        canon[t * C + c] = capin[tin::byte_of(t, c, C)];
            } else {
                for (size_t y = 0; y < TH; ++y)
                    for (size_t x = 0; x < TW; ++x) {
                        const size_t t = tin::token_of(x, y, TW);
                        const size_t px = swap ? y : x, py = swap ? x : y;
                        if (px >= VW || py >= VH) continue;
                        for (size_t c = 0; c < C; ++c) {
                            const size_t off = tin::view_byte_of(px, py, c, VW, VH);
                            canon[t * C + c] = off < capin.size() ? capin[off] : 0;
                        }
                    }
            }
            const std::vector<uint8_t> tb = tin::tile_blocked(canon.data(), M, C);
            const size_t soff = voff[x_src(first)];
            ctx.upload(act, soff, tb.data(), tb.size());
            // Read it back. Two seed layouts scoring *identically* is the
            // signature of data that never reached the kernel, and that is
            // cheaper to rule out than to reason about.
            std::vector<uint8_t> back(4096);
            ctx.download(act, soff, back.data(), back.size());
            size_t bad = 0;
            for (size_t i = 0; i < back.size(); ++i) if (back[i] != tb[i]) ++bad;
            uint32_t sum = 0;
            for (size_t i = 0; i < back.size(); ++i) sum = sum * 131u + back[i];
            std::printf("seeded block %d from %s as %s, %zux%zu (grid %zu) x %zu -> offset %zu, "
                        "readback %s, first-4k hash %08x\n",
                        first.block, seed_path, view ? "view" : "tinlayout", IW, IH, TW, C,
                        soff, bad ? "MISMATCH" : "ok", sum);
            // And what the first dispatch will actually read.
            for (const Disp& dd : disp)
                if (dd.s == &first) {
                    PushFSwin q{};
                    std::memcpy(&q, dd.push.data(), std::min(sizeof q, dd.push.size()));
                    std::printf("  block %d dispatch reads x_off=%u  (seed wrote %zu)\n",
                                first.block, q.x_off, soff);
                    break;
                }
        }
        // **Seed everything the capture has, then run one layer.** A chained
        // score answers "is the whole prefix right"; it cannot say which layer
        // broke, and a layer whose *residual* comes from a sibling it did not
        // dispatch is scored against garbage - which is why `CCSplitSwin16HProj`
        // read r=0.28 seeded at layer 2 and says nothing about the Proj at all.
        // With every captured slot filled from NVIDIA's own bytes, `--only`
        // asks one layer the question on its own.
        const int ds_tw = std::atoi(arg(argc, argv, "--ds-tw", "0"));
        const char* seed_dir = arg(argc, argv, "--seed-dir", "");
        if (*seed_dir) {
            size_t n = 0;
            // **A seeder that declines is as bad as one that is wrong, and it is
            // harder to see.** Three separate paths here `continue` on a shape
            // they do not like - the size check, `put_slot`'s divisibility
            // guard, and a missing file - and each one leaves the consumer
            // reading the arena's prefill while the board reports it as an
            // ordinary low score. `--seed-report` prints every decision.
            const bool seed_report = flag(argc, argv, "--seed-report");
            auto put_slot = [&](int key, const std::vector<uint8_t>& canon,
                                size_t M, size_t C) {
                if (!M || !C || M % 16 || C % 16) {
                    if (seed_report)
                        std::printf("       slot %d refused: M=%zu C=%zu\n", key, M, C);
                    return;
                }
                const std::vector<uint8_t> tb = tin::tile_blocked(canon.data(), M, C);
                ctx.upload(act, voff[key], tb.data(), tb.size());
                ++n;
            };
            for (const Disp& d : disp) {
                const Step& s2 = *d.s;
                if (d.kern.rfind("imgin", 0) == 0 || d.kern.rfind("pool", 0) == 0 || d.kern.rfind("upsblend", 0) == 0 ||
                d.kern == "upsview" || d.kern == "decups" || d.kern == "repack" || d.kern == "qkvnorm" || d.kern == "gemmds" || d.kern == "gemmups" || d.kern == "gemmups16") continue;
                char cap[256];
                std::snprintf(cap, sizeof cap, "%s/b%03dl%d.bin", seed_dir, s2.block, s2.layer);
                const std::vector<uint8_t> ref = slurp(cap);
                if (ref.empty()) {
                    if (seed_report)
                        std::printf("       b%03dl%d  no capture at %s\n",
                                    s2.block, s2.layer, cap);
                    continue;
                }
                if (host_boundary && s2.block==38 && s2.layer==4) {
                    const auto side=slurp(std::string(seed_dir)+"/b038l4_2d.bin");
                    const size_t M=size_t(s2.W)*s2.H,C=1024;
                    if(side.size()<M*C) { std::fprintf(stderr,"missing b38 repacked side tensor\n");return 1; }
                    std::vector<uint8_t> canon(M*C);
                    for(size_t t=0;t<M;++t)for(size_t c=0;c<C;++c)canon[t*C+c]=side[tin::byte_of(t,c,C)];
                    put_slot(ups_key(38),canon,M,C);
                }
                if (host_boundary && (s2.type == "CCSplitSwin16HProjPool" ||
                                      (s2.block == 31 && s2.layer == 0))) {
                    char sidepath[256];
                    std::snprintf(sidepath, sizeof sidepath, "%s/b%03dl%d_o1.bin",
                                  seed_dir, s2.block, s2.layer);
                    const auto side = slurp(sidepath);
                    const bool pool = s2.type == "CCSplitSwin16HProjPool";
                    const size_t M = pool ? pad4((s2.W+1)/2)*pad4((s2.H+1)/2) : size_t(s2.tokens);
                    const size_t C = pool ? 512 : 1024;
                    if (side.size() < M*C) {
                        std::fprintf(stderr, "missing host-boundary side tensor %s\n", sidepath);
                        return 1;
                    }
                    std::vector<uint8_t> canon(M*C);
                    for (size_t t=0; t<M; ++t)
                        for (size_t c=0; c<C; ++c) canon[t*C+c]=side[tin::byte_of(t,c,C)];
                    put_slot(pool ? pool_key(s2.block) : lift_key(31), canon, M, C);
                }
                // **`CCVit1DQKV` has three outputs, in three layouts.** Our own
                // GEMM writes one 3C tensor, so seeding this slot means
                // assembling NVIDIA's `_o1` and `_o2` beside the base file and
                // decoding each with its own map. Without this the slot's size
                // check fails silently - "capture is 2621440 B, need 7864320" -
                // the attention runs on the arena's prefill pattern instead, and
                // all eight blocks then score an identical r=+0.0571, which is
                // what a layer reading no input at all looks like.
                if (s2.type == "CCVit1DQKV") {
                    char o1[256], o2[256];
                    std::snprintf(o1, sizeof o1, "%s/b%03dl%d_o1.bin",
                                  seed_dir, s2.block, s2.layer);
                    std::snprintf(o2, sizeof o2, "%s/b%03dl%d_o2.bin",
                                  seed_dir, s2.block, s2.layer);
                    const std::vector<uint8_t> rk = slurp(o1), rv = slurp(o2);
                    const size_t M = size_t(s2.tokens), C1 = 1024;
                    if (rk.size() < M * C1 || rv.size() < M * C1 ||
                        ref.size() < M * C1) {
                        std::fprintf(stderr, "incomplete Q/K/V seed b%03dl%d: Q=%zu K=%zu V=%zu, minimum=%zu bytes each\n",
                                     s2.block, s2.layer, ref.size(), rk.size(), rv.size(), M*C1);
                        return 1;
                    }
                    std::vector<uint8_t> canon(M * 3 * C1, 0);
                    for (size_t t = 0; t < M; ++t)
                        for (size_t c = 0; c < C1; ++c) {
                            const size_t qi=tin::byte_of(t,c,C1), ki=vit_k_byte(t,c), vi=vit_v_byte(t,c);
                            if (qi>=ref.size() || ki>=rk.size() || vi>=rv.size()) {
                                std::fprintf(stderr, "truncated Q/K/V layout seed b%03dl%d at token=%zu channel=%zu\n",
                                             s2.block, s2.layer, t, c);
                                return 1;
                            }
                            canon[t * 3 * C1 + c]            = ref[qi];
                            canon[t * 3 * C1 + C1 + c]       = rk[ki];
                            canon[t * 3 * C1 + 2 * C1 + c]   = rv[vi];
                        }
                    put_slot(key_of(s2.block, s2.layer), canon, M, 3 * C1);
                    continue;
                }
                const Shape sh = shape_of(s2);
                const size_t C = size_t(sh.fam == F_DS ? 2 * s2.C : sh.N);
                // `_upsample`'s plan row is its *input* level; its output is the
            // window grid's, one level up. Scoring it at `s2.tokens` reads a
            // quarter of the layer and calls the rest a mismatch.
            // **The two adapters again**, on the seeding side this time: the
            // pre block halves and the post block doubles, and `shape_of`
            // reports both as plain. Without this the seeder rejected block 0's
            // gold every run - "seed is 16711680 B, need 66355200" - and b001l0,
            // which reads it, has been scoring 0.5471 against an input that was
            // never there.
            const bool pre_ds  = s2.type == "CCTinlayoutFusedPreBlockSwin1H";
            const bool post_up = s2.type == "CCTinlayoutFusedPostBlockSwin1H";
            // **An `_outview` layer's capture is in the view format, and every
            // `_upsample`'s chain input comes from one** - b048 from b047l3,
            // b056 from b055, b062 from b061, b066 from b065. The `--standalone`
            // seeder already knew this (`seed_view` below); the `--seed-dir`
            // one did not, so against cap11 all four `_upsample`
            // layers read a tinlayout decode of a view raster and scored
            // 0.06-0.18 together while their neighbours scored 0.99. Same
            // family-shaped tell as the `CCVit1DQKV` and `_ds` cases above.
            //
            // The map is the scorer's, run backwards: it is the one verified
            // against our own output at 0.9296-1.0000, so the seeder and the
            // scorer cannot disagree about what the arena holds.
            const bool outview =
                s2.kernel.find("_outview") != std::string::npos &&
                !flag(argc, argv, "--seed-outview-plain");
            const size_t M = sh.fam == F_DS || pre_ds
                                 ? (reader_tokens.count(s2.block)
                                        ? reader_tokens[s2.block]
                                        : size_t(s2.tokens) / 4)
                           : sh.fam == F_UPS || post_up
                                 ? (host_boundary && sh.fam==F_UPS && reader_tokens.count(s2.block) ? reader_tokens[s2.block] : size_t(s2.gx)*8*size_t(s2.gy)*8)
                                 : size_t(s2.tokens);
                // **A `_ds` capture is smaller than the slot and that is not a
                // mismatch.** Its half-resolution output is in the *view*
                // format at the **unpadded** half extent, so b022l0's gold is
                // 1013760 B - 60 x 33 x 512 - where the slot is 1105920, the
                // padded window grid. The `M * C` test rejected it, `continue`
                // said nothing, and `CCSplitSwin16HFfwd` at b023l0 has been
                // scoring **r=+0.0138 against the arena's prefill** while its
                // seven identical siblings score 0.9997. Same shape of harness
                // lie as the `CCVit1DQKV` one above, and the same tell: one
                // layer of a family failing where the rest do not.
                const size_t need = sh.fam == F_DS
                    ? (size_t(s2.H) / 2) * (size_t(s2.W) / 2) * C
                  : outview
                    ? size_t(s2.H) * size_t(s2.W) * C
                    : M * C;
                if (ref.size() < need) {
                    std::printf("%-6s b%03dl%d  seed is %zu B, need %zu\n",
                                "", s2.block, s2.layer, ref.size(), need);
                    continue;
                }
                if (seed_report)
                    std::printf("       b%03dl%d %-26s %8zu B  M=%zu C=%zu%s\n",
                                s2.block, s2.layer, s2.type.c_str(), ref.size(),
                                M, C, outview ? "  outview" : "");
                std::vector<uint8_t> canon(M * C, 0);
                // **The pre block's output is a view too.** It is a `_ds` whose
                // tail is a bare 2x2 mean, so `shape_of` calls it plain - and the
                // seeder was decoding its gold as tinlayout. b001l0 reads it.
                if(host_boundary && (sh.fam==F_DS || pre_ds)) {
                    const size_t VW=pad4((s2.H+1)/2),VH=pad4((s2.W+1)/2);
                    for(size_t y=0;y<VH;++y)for(size_t x=0;x<VW;++x)for(size_t c=0;c<C;++c) {
                        size_t t=tin::token_of(x,y,VW),o=tin::view_byte_of(x,y,c,VW,VH);
                        if(o<ref.size() && t*C+c<canon.size())canon[t*C+c]=ref[o];
                    }
                } else if (outview) {
                    // `_outview` changes layout, not resolution, so the extent
                    // is the layer's own and there is no halving anywhere in
                    // this map. `--outview-raster` is deliberately *not* honored
                    // here: the scorer measured the plain extent winning at
                    // every level, and a seeder that took the grid would put the
                    // difference back on the consumer's side of the comparison.
                    const size_t VW = size_t(s2.H), VH = size_t(s2.W);
                    const size_t TW = size_t(tiles_of(int(VW))) * 4;
                    // **Our grid is narrower than the capture's extent and the
                    // overhang must be dropped, not wrapped.** `tiles_of`
                    // truncates, so 33 columns go into a 32-pixel row and 67
                    // into 64; `token_of` of the overhanging x lands in the next
                    // tile row, which corrupts the row it writes into as well as
                    // losing the column. The scorer reads the same min().
                    const size_t XN = std::min(VW, TW);
                    for (size_t y = 0; y < VH; ++y)
                        for (size_t x = 0; x < XN; ++x) {
                            const size_t t = tin::token_of(x, y, TW);
                            for (size_t c = 0; c < C; ++c) {
                                const size_t o = tin::view_byte_of(x, y, c, VW, VH);
                                if (o < ref.size() && t * C + c < canon.size())
                                    canon[t * C + c] = ref[o];
                            }
                        }
                } else if ((sh.fam == F_DS || pre_ds) && !flag(argc, argv, "--ds-plain")) {
                    // The half-resolution output is in the view format, at the
                    // unpadded half extent - the same conversion the scorer does,
                    // run backwards. `--ds-plain` reads it as an ordinary
                    // tinlayout tensor instead, which is what its *size* now
                    // says it is: the consumer's token count, not a view raster.
                    const size_t VW = host_boundary ? pad4((s2.H+1)/2) : size_t(s2.H)/2;
                    const size_t VH = host_boundary ? pad4((s2.W+1)/2) : size_t(s2.W)/2;
                    const size_t PW = pad8(VW), PH = pad8(VH);
                    // **`--ds-tw <tiles>` overrides the token grid**, because
                    // the consumer reads it out. `CCSplitSwin16HFfwd` is a GEMM
                    // over tokens with no 2D structure at all, so b023l0's score
                    // depends on nothing but whether its input's tokens are in
                    // the right order - sweeping this and watching that score is
                    // reading the mapping off the gold instead of deriving it.
                    size_t vrow, vpl, vnx, vny;
                    ds_view_strides(VW, VH, vrow, vpl, vnx, vny);
                    // **The seeder and the scorer do not need the same
                    // raster.** The scorer compares our output to the capture
                    // and wants the capture's true raster; the seeder writes the
                    // capture into our arena for the *consumer*, which reads it
                    // through our own `_inpview` model. Where those two differ
                    // the difference is the `_inpview` seam, not the `_ds` one.
                    { size_t Wp, Hp;
                      const int keep = g_ds_raster;
                      if (flag(argc, argv, "--seed-raster-true")) {}
                      else g_ds_raster = 0;
                      ds_raster(s2.gx, s2.gy, VW, VH, Wp, Hp);
                      g_ds_raster = keep;
                      // **The raster is `4*gx`, the real extent is `(W+1)/2`.**
                      // Both come out of the same store address: the stride
                      // pads to the window grid and the data stops at the
                      // rounded half extent - 34 at block 22, not the 33 a
                      // truncating halve gives. Comparing the padding columns
                      // compares one side's fill against the other's.
                      if (!g_ds_view && !g_ds_grid) {
                          vrow = Wp; vpl = Wp * Hp;
                          vnx = std::min(Wp, (size_t(s2.H) + 1) / 2);
                          vny = std::min(Hp, (size_t(s2.W) + 1) / 2);
                      } }
                    // The three have to move together: the view's row stride,
                    // the raster it iterates, and the token grid our own arena
                    // holds it in. Changing any one alone measures worse,
                    // because the comparison is then inconsistent rather than
                    // wrong - which is how four earlier sweeps read as negatives.
                    // **The seeder's grid has to hold every column the
                    // capture has.** Our pool writes `ds_tiles` tiles, which
                    // follows the consumer blocks; the capture's raster is
                    // wider, so writing it through the narrower grid wraps the
                    // overhang onto the next row. `--ds-seed-floor` restores it.
                    const size_t TW = ds_tw ? size_t(ds_tw) * 4
                                    : g_ds_grid ? PW
                                    : g_ds_view ? size_t(tiles_of(int(VW))) * 4
                                    : flag(argc, argv, "--ds-seed-floor")
                                                ? ds_tiles(VW, s2.gx) * 4
                                                : std::max(ds_tiles(VW, s2.gx) * 4, vrow);
                    // **A `_ds` output feeding a GEMM may be indexed in raster
                    // order, not tile order.** `cc_split_swin_16h_ffwd_inpview`
                    // reads tokens linearly - it has no 2D structure at all - and
                    // the view layout's natural token is `y*W + x`. The fused
                    // Swin `_inpview` blocks re-derive their position from tiles
                    // and do not care; b023l0 would.
                    const bool rast = flag(argc, argv, "--ds-raster-tokens");
                    // **`--ds-window-tokens`: 8x8-window-major, not tile-major.**
                    // `cc_split_swin_16h_ffwd_inpview_512_fp8` is a GEMM, so it
                    // has no `nr_tile_base` to re-derive a position with - its
                    // token index has to *be* the order it wants. Its grid is
                    // 5 x 8, which is `ceil(36/8) x ceil(60/8)` of the view
                    // raster this session pinned down, so a workgroup owns an
                    // 8x8 block of view pixels = 64 tokens, and the tokens are
                    // grouped by window. Our arena's `_ds` output is tile-major,
                    // which the fused-Swin `_inpview` blocks do not care about
                    // and this one would.
                    // **The data raster and the token domain are different
                    // sizes.** 36x60 is 2160 valid positions; the slot is 2560,
                    // which is the 8x8-padded 40x64 domain the grid (5 x 8)
                    // covers. So the token index runs over 40x64 and the *data*
                    // stops at 36x60 - codex's reading of the gather, and the
                    // one combination four sweeps had not tried.
                    const bool wintok = flag(argc, argv, "--ds-window-tokens");
                    const size_t gxw = (vrow + 7) / 8;
                    const size_t dx = vrow, dy = vny;           // where data ends
                    if (wintok) { vnx = gxw * 8; vny = ((vny + 7) / 8) * 8; }
                    for (size_t y = 0; y < vny; ++y)
                        for (size_t x = 0; x < vnx; ++x) {
                            const size_t t =
                                rast ? y * vrow + x
                              : wintok ? ((y / 8) * gxw + x / 8) * 64
                                             + ((y % 8) / 4) * 32 + ((x % 8) / 4) * 16
                                             + 4 * (y % 4) + (x % 4)
                                       : tin::token_of(x, y, TW);
                            if (wintok && (x >= dx || y >= dy)) continue;
                            for (size_t c = 0; c < C; ++c) {
                                // `--ds-no-sub` drops the 16-channel bit
                                // permutation from the decode, which is the only
                                // channel-axis operation in the view map.
                                const size_t o = flag(argc, argv, "--ds-no-sub")
                                    ? (c / 16) * (vpl * 16) + (y * vrow + x) * 16 + c % 16
                                    : view_byte2(x, y, c, vrow, vpl);
                                if (t * C + c < canon.size() && o < ref.size())
                                    canon[t * C + c] = ref[o];
                            }
                        }
                } else {
                    for (size_t t = 0; t < M; ++t)
                        for (size_t c = 0; c < C; ++c)
                            canon[t * C + c] = ref[tin::byte_of(t, c, C)];
                }
                put_slot(key_of(s2.block, s2.layer), canon, M, C);
                // **The pre block has a skip too, and it is the U-Net's longest
                // edge**: `70 <- [69, 0]` reads `skip_key(0)`, so a chain seeded
                // at block 0 and dispatched from block 1 leaves the post block
                // reading the arena's prefill and the final image is noise
                // whatever the 150 layers between them did. Its `s2.C` is the
                // plan's colour count, 3 - the slot's width is the block's, 32.
                if (sh.fam == F_DS || pre_ds) {
                    std::snprintf(cap, sizeof cap, host_boundary ? "%s/b%03dl%d_o1.bin" : "%s/b%03dl%d_skip.bin",
                                  seed_dir, s2.block, s2.layer);
                    std::vector<uint8_t> sk = slurp(cap);
                    // Earlier replays name this same tensor `_skip`, while
                    // host-derived captures name it `_o1`. Never silently
                    // benchmark the post block with an unseeded encoder skip.
                    if (sk.empty() && host_boundary) {
                        std::snprintf(cap, sizeof cap, "%s/b%03dl%d_skip.bin",
                                      seed_dir, s2.block, s2.layer);
                        sk = slurp(cap);
                    }
                    const size_t SM = size_t(s2.tokens),
                                 SC = size_t(pre_ds ? sh.N : s2.C);
                    if (sk.size() < SM * SC) {
                        std::fprintf(stderr, "missing or undersized skip seed %s: %zu B, need %zu\n",
                                     cap, sk.size(), SM * SC);
                        return 1;
                    }
                    if (sk.size() >= SM * SC) {
                        std::vector<uint8_t> sc(SM * SC);
                        for (size_t t = 0; t < SM; ++t)
                            for (size_t c = 0; c < SC; ++c)
                                sc[t * SC + c] = sk[tin::byte_of(t, c, SC)];
                        put_slot(skip_key(s2.block), sc, SM, SC);
                    }
                }
            }
            std::printf("seeded %zu slots from %s\n", n, seed_dir);
        }
        // **`--standalone DIR` seeds each layer's *input* by size**, mirroring
        // the capture tool's `--standalone`. The decoder cannot be reached through
        // the chain - the ViT in front of it is nondeterministic, and three
        // successful runs of `cc_vit_1d_projection_fp8` on one input differ byte
        // for byte - but a per-layer gold only needs the input to be *known*.
        // Bind the same clean tensor on both sides and the comparison is exactly
        // as valid as a chained one.
        //
        // The size rule is the plan's and nothing else: a chain input is
        // `tokens * ci`, a skip is its producer's `tokens * ci`. The U-Net's
        // symmetry makes those the encoder tensors the decoder mirrors.
        const char* sa_dir = arg(argc, argv, "--standalone", "");
        if (*sa_dir) {
            std::map<size_t, std::string> pool;
            for (const auto& e : std::filesystem::directory_iterator(sa_dir)) {
                if (e.path().extension() != ".bin") continue;
                // First in sorted order wins, so the choice is reproducible.
                const size_t n = size_t(e.file_size());
                const std::string p2 = e.path().string();
                auto it = pool.find(n);
                if (it == pool.end() || p2 < it->second) pool[n] = p2;
            }
            std::map<int, std::pair<long, int>> blk;   // block -> (tokens, ci)
            for (const Step& s2 : plan)
                if (s2.ci > 0 && s2.tokens > 0 && !blk.count(s2.block))
                    blk[s2.block] = {s2.tokens, s2.ci};
            // **An `_upsample`'s chain input is in the view layout.** Every one
            // of them is fed by an `_outview` layer - b066 by b065, b062 by
            // b061, b056 by b055, b048 by b047l3 - so NVIDIA's kernel reads that
            // tensor as 16-channel planes over a raster while we were seeding it
            // as tinlayout. The two sides disagreed about the bytes, not about
            // the arithmetic, and all four `_upsample` layers failed together
            // for that one reason.
            auto seed_view = [&](int key, size_t want, const Step& s2) {
                auto it = pool.find(want);
                if (it == pool.end()) return false;
                const std::vector<uint8_t> ref = slurp(it->second);
                const size_t C2 = size_t(s2.ci);
                const size_t VW = size_t(s2.H), VH = size_t(s2.W);
                // **The token grid must be the tensor's, not `tiles_of`'s.** The
                // buffer holds `tokens` tokens over `VH` rows, so it is
                // `tokens / VH` pixels wide - 272 at level 2, where
                // `tiles_of(270) * 4` gives 268 and the blend then reads a grid
                // one tile wider than the seeder wrote. That mismatch is why the
                // first view attempt scored *worse* than tinlayout and made the
                // right hypothesis look wrong.
                const size_t TW = VH ? size_t(s2.tokens) / VH
                                     : size_t(tiles_of(int(VW))) * 4;
                const size_t M2 = TW * VH;
                if (!C2 || M2 % 16 || C2 % 16 || ref.size() < want) return false;
                std::vector<uint8_t> canon(M2 * C2, 0);
                for (size_t y = 0; y < VH; ++y)
                    for (size_t x = 0; x < VW; ++x) {
                        const size_t t = tin::token_of(x, y, TW);
                        for (size_t c = 0; c < C2; ++c) {
                            const size_t o = tin::view_byte_of(x, y, c, VW, VH);
                            if (o < ref.size() && t * C2 + c < canon.size())
                                canon[t * C2 + c] = ref[o];
                        }
                    }
                const std::vector<uint8_t> tb = tin::tile_blocked(canon.data(), M2, C2);
                ctx.upload(act, voff[key], tb.data(), tb.size());
                return true;
            };
            auto seed_slot = [&](int key, size_t want, const char* what, const Step& s2) {
                auto it = pool.find(want);
                if (it == pool.end()) return false;
                const std::vector<uint8_t> ref = slurp(it->second);
                const size_t C2 = size_t(s2.ci), M2 = want / C2;
                if (!C2 || M2 % 16 || C2 % 16 || ref.size() < want) return false;
                std::vector<uint8_t> canon(want);
                for (size_t t = 0; t < M2; ++t)
                    for (size_t c = 0; c < C2; ++c)
                        canon[t * C2 + c] = ref[tin::byte_of(t, c, C2)];
                const std::vector<uint8_t> tb = tin::tile_blocked(canon.data(), M2, C2);
                ctx.upload(act, voff[key], tb.data(), tb.size());
                (void)what;
                return true;
            };
            size_t n = 0;
            for (const Disp& d : disp) {
                const Step& s2 = *d.s;
                const Shape sh = shape_of(s2);
                if (sh.fam == F_NONE || s2.ci <= 0) continue;
                // **Only the cross-block edges.** A block's later layers read
                // their own earlier ones, and the standalone gold keeps that
                // chain intact - the capture tool's `--standalone` only replaces
                // inputs that come from *another* block. Seeding an intra-block
                // input by size hands the layer a tensor the gold never saw, and
                // every `CCSplitSwin16H` stage below the first scored r=0 for
                // exactly that reason while the same kernels score 1.0000 in the
                // encoder. Pair this with `--seed-dir <the standalone golds>`.
                // `--standalone-all` matches the capture tool's flag of the
                // same name: inside the ViT the intra-block chain is worth
                // nothing, so a per-layer gold replaces those inputs too.
                if (s2.layer > 0 && !flag(argc, argv, "--standalone-all")) continue;
                const size_t want = size_t(s2.tokens) * size_t(s2.ci);
                // **The attention's gold has Q = K = V.** The replay binds its
                // three pointers to one tensor, because the generator has a
                // single source for them - while our arena carries Q|K|V as
                // three channel ranges of one 3C buffer. Seeding that buffer
                // with the pool tensor repeated three times makes the two sides
                // compute the same thing; the real chain still gets the genuine
                // QKV output.
                if (sh.fam == F_VATTN) {
                    auto it = pool.find(want);
                    if (it != pool.end()) {
                        const std::vector<uint8_t> ref = slurp(it->second);
                        const size_t C1 = size_t(s2.ci), M2 = want / C1, C3 = 3 * C1;
                        if (C1 && !(M2 % 16) && !(C1 % 16) && ref.size() >= want) {
                            std::vector<uint8_t> canon(M2 * C3);
                            for (size_t t2 = 0; t2 < M2; ++t2)
                                for (size_t c = 0; c < C1; ++c) {
                                    const uint8_t b = ref[tin::byte_of(t2, c, C1)];
                                    canon[t2 * C3 + c] = b;
                                    canon[t2 * C3 + C1 + c] = b;
                                    canon[t2 * C3 + 2 * C1 + c] = b;
                                }
                            const std::vector<uint8_t> tb =
                                tin::tile_blocked(canon.data(), M2, C3);
                            ctx.upload(act, voff[x_src(s2)], tb.data(), tb.size());
                            ++n;
                            continue;
                        }
                    }
                }
                // Measured on b066: as tinlayout r=+0.163, as view r=-0.099.
                // **Neither is right** - the projection side is wrong for some
                // other reason - but tinlayout is the better of the two and the
                // default should be the measured one. `--ups-view` tries the
                // other, so the result stays reproducible.
                // **Probed, not guessed.** Lighting one input tile and reading
                // where the layer responds puts NVIDIA's footprint at x 16..148,
                // y 472..476 - a 34-tile strip. Our lit tile's 1024 contiguous
                // tinlayout bytes, read as *view*, are a contiguous run of 64
                // pixels at input y=237, x=10..74, which doubles to exactly that
                // strip - and b066 then scores **r=0.9986** this way against
                // +0.003 as tinlayout.
                //
                // **It measured *worse* until the aliasing bug was fixed.** With
                // every side buffer at offset 0 the prologue was overwriting its
                // own input, so no input interpretation could win and the right
                // hypothesis looked dead. A confounded measurement does not
                // falsify anything - it just cannot decide.
                const bool as_view = sh.fam == F_UPS && !flag(argc, argv, "--ups-tin");
                if (as_view ? seed_view(x_src(s2), want, s2)
                            : seed_slot(x_src(s2), want, "in", s2)) ++n;
                if (sh.fam == F_UPS && s2.in1 >= 0 && blk.count(s2.in1)) {
                    const auto pr = blk[s2.in1];
                    Step fake = s2; fake.ci = pr.second;
                    seed_slot(skip_key(s2.in1), size_t(pr.first) * size_t(pr.second),
                              "skip", fake);
                }
            }
            std::printf("standalone: seeded %zu layer inputs by size from %s\n", n, sa_dir);
        }
        // **`--dump B:L FILE` writes a layer's output where it can be looked
        // at.** Four hypotheses about `_upsample` have now been killed by
        // scoring whole layers, one build and one remote run each, and the hit
        // rate says to stop guessing: a probe that lights one input tile and
        // reads off where it lands settles a mapping in one shot, the way
        // `fswin_test --find-tile` settled the level-2 stride. That needs the
        // bytes, not a correlation.
        const char* dump_at = arg(argc, argv, "--dump");
        const char* dump_to = arg(argc, argv, "--dump-to", "dump.bin");
        // `--only B:L` dispatches that step and nothing else.
        int only_b = -1, only_l = 0;
        if (const char* o = arg(argc, argv, "--only")) {
            only_b = std::atoi(o);
            const char* colon = std::strchr(o, ':');
            only_l = colon ? std::atoi(colon + 1) : 0;
        }
        // **Dispatch only from the seeded block on.** The seed goes into the slot
        // the chain's first scored layer reads - which is also the *output* slot
        // of the layer before it, so running that layer first overwrites the
        // seed before anything reads it. Block 2 scored 0.53% for exactly this
        // reason while the single-layer harness scored 68% on the same triple.
        // **`--oracle` scores every layer independently in one run.** With every
        // slot seeded, dispatching the graph *backwards* means each layer still
        // reads seeded inputs - the only slots overwritten so far belong to
        // layers further down the chain, which no earlier layer reads. The
        // U-Net skips are safe for the same reason: the `_upsample` consumes a
        // skip slot before the `_ds` that would rewrite it ever runs.
        const bool oracle = flag(argc, argv, "--oracle");
        std::vector<nrvk::Runner::Step> from;
        for (size_t i = 0; i < disp.size(); ++i) {
            const Step& s2 = *disp[i].s;
            const bool keep = oracle ? true
                : only_b >= 0
                ? (s2.block == only_b && s2.layer == only_l)
                : (s2.block > seed_step->block ||
                   (s2.block == seed_step->block && s2.layer >= seed_step->layer));
            if (keep) from.push_back(steps[i]);
        }
        // **Reverse whole layers, not dispatches.** A `_ds` layer is three
        // dispatches - the block, the 2x2 pool, the resample - that must run in
        // that order. Reversing the flat list ran them backwards and every `_ds`
        // row read r=0.03 while the chained run scored the same layer at 0.991,
        // which is the tool lying rather than the kernel failing.
        if (oracle) {
            std::vector<nrvk::Runner::Step> rev;
            for (size_t i = disp.size(); i-- > 0;) {
                size_t j = i;
                while (j > 0 && disp[j - 1].s == disp[i].s) --j;
                for (size_t k = j; k <= i; ++k) rev.push_back(steps[k]);
                i = j;
            }
            from.swap(rev);
        }
        if (only_b >= 0)
            std::printf("dispatching %zu of %zu, only b%03dl%d\n",
                        from.size(), steps.size(), only_b, only_l);
        else
            std::printf("dispatching %zu of %zu, from block %d layer %d\n",
                        from.size(), steps.size(), seed_step->block, seed_step->layer);
        runner.run_graph(from, 1);
        if (flag(argc,argv,"--hash-values")) {
            // Raw slot fingerprints locate the first changing producer across
            // identical single-pass runs, including side outputs and padding.
            for (const auto& [key,bytes] : S.vsize) {
                std::vector<uint8_t> value(bytes);
                ctx.download(act,voff.at(key),value.data(),bytes);
                uint64_t h=14695981039346656037ull;
                for(uint8_t v:value) {h^=v;h*=1099511628211ull;}
                std::printf("value-hash key=%d bytes=%zu fnv1a64=%016llx\n",
                            key,bytes,(unsigned long long)h);
            }
        }
        if (dump_at) {
            const int db = std::atoi(dump_at);
            const char* colon = std::strchr(dump_at, ':');
            const int dl = colon ? std::atoi(colon + 1) : 0;
            for (const Disp& d : disp) {
                const Step& s2 = *d.s;
                if (s2.block != db || s2.layer != dl) continue;
                const Shape sh = shape_of(s2);
                size_t C = size_t(sh.fam == F_DS ? 2 * s2.C : sh.N);
                size_t M = sh.fam == F_DS  ? size_t(s2.tokens) / 4
                         : sh.fam == F_UPS ? (host_boundary && sh.fam==F_UPS && reader_tokens.count(s2.block) ? reader_tokens[s2.block] : size_t(s2.gx)*8*size_t(s2.gy)*8)
                                           : size_t(s2.tokens);
                // **The prologue's own buffers, which is what the `_upsample`
                // question actually needs.** Scoring the layer only shows the
                // block body's view of the prologue; `--dump-stage proj` is the
                // 2C -> C projection before any replication, and `blend` is the
                // replicated-and-blended image the body reads. Five hypotheses
                // died to inference through the body - this looks at the bytes.
                int dkey = key_of(s2.block, s2.layer);
                const std::string stage = arg(argc, argv, "--dump-stage", "out");
                if (sh.fam == F_UPS && stage == "proj") {
                    dkey = pool_key(s2.block); M = size_t(s2.tokens); C = size_t(s2.co);
                } else if (sh.fam == F_UPS && stage == "blend") {
                    dkey = ups_key(s2.block); C = size_t(s2.co);
                } else if (stage == "lift") {
                    // The image adapter's own output, before the block body:
                    // the only place to see whether the 500x the pre block's
                    // output carries is the lift's or the block's.
                    dkey = lift_key(s2.block); M = size_t(s2.tokens); C = 32;
                } else if (sh.fam == F_DS && stage == "pool") {
                    // The 2x2 mean's own output, before the resample mixes the
                    // channels - the only place a positional probe can be read
                    // off directly, which is what settled the `_ds` seam.
                    dkey = pool_key(s2.block); M = size_t(s2.tokens) / 4;
                    C = size_t(s2.C);
                } else if (stage == "skip") {
                    // The body's own output, at the *input* level and the
                    // block's own width - 2C is the resample's, not this.
                    dkey = skip_key(s2.block); M = size_t(s2.tokens);
                    C = size_t(s2.type == "CCTinlayoutFusedPreBlockSwin1H" ? 32 : s2.C);
                }
                const size_t elem_bytes = host_boundary &&
                    ((sh.fam == F_UPS && (stage == "proj" || (stage == "blend" && s2.co == 32))) ||
                     stage == "lift") ? 2 : 1;
                std::vector<uint8_t> got(M * C * elem_bytes);
                ctx.download(act, voff.at(dkey), got.data(), got.size());
                // Canonical [token][channel], which is what a probe wants to
                // read - the arena's tile-blocked form is an implementation
                // detail of the kernels.
                std::vector<uint8_t> canon(got.size());
                for (size_t t = 0; t < M; ++t)
                    for (size_t c = 0; c < C; ++c) {
                        const size_t src = ((t / 16) * (C / 16) + c / 16) * 256
                                         + (t % 16) * 16 + c % 16;
                        std::memcpy(canon.data() + (t*C+c)*elem_bytes,
                                    got.data() + src*elem_bytes, elem_bytes);
                    }
                FILE* f = std::fopen(dump_to, "wb");
                if (f) {
                    std::fwrite(canon.data(), 1, canon.size(), f);
                    std::fclose(f);
                    std::printf("dumped b%03dl%d %s: %zu tokens x %zu ch %s -> %s\n",
                                db, dl, stage.c_str(), M, C, elem_bytes == 2 ? "FP16" : "FP8", dump_to);
                }
                break;
            }
        }
        // **A whole family failing together is a harness lie until proven
        // otherwise.** This rule has been written in three comments in this
        // file - "one layer of a family failing where the rest do not" is the
        // tell that named the `_ds`, `CCVit1DQKV` and `CCSplitSwin16HFfwd`
        // seeding bugs - and it still took four sessions to apply it to
        // `CCVit1DAttention`, whose eight instances all read r=+0.02 while
        // their immediate neighbours read 0.98. The cause was one line of
        // the capture tool binding K and V to Q. So the rule stops living in
        // a comment: collect the scores and say it out loud.
        std::map<std::string, std::vector<double>> by_type;
        std::printf("\n%-6s %-26s %9s %9s  %s\n",
                    "layer", "type", "bytes", "agree", "note");
        for (const Disp& d : disp) {
            const Step& s2 = *d.s;
            if (d.kern.rfind("imgin", 0) == 0 || d.kern.rfind("pool", 0) == 0 || d.kern.rfind("upsblend", 0) == 0 ||
                d.kern == "upsview" || d.kern == "decups" || d.kern == "repack" || d.kern == "qkvnorm" || d.kern == "gemmds" || d.kern == "gemmups" || d.kern == "gemmups16") continue;
            if (!oracle && (only_b >= 0 ? !(s2.block == only_b && s2.layer == only_l)
                                        : s2.block < seed_step->block)) continue;
            char cap[256];
            std::snprintf(cap, sizeof cap, "%s/b%03dl%d.bin",
                          score_dir, s2.block, s2.layer);
            const std::vector<uint8_t> ref = slurp(cap);
            if (ref.empty()) continue;
            // Saturation and repeated bytes are diagnostics, not validity tests.
            // Correct cubin outputs can contain both; never exclude them by default.
            if (flag(argc, argv, "--reference-diagnostics")) {
                size_t sat = 0, big = 0;
                for (uint8_t b3 : ref) {
                    if ((b3 & 0x7F) == 0x7E) ++sat;
                    if ((b3 & 0x7F) >= 0x78) ++big;
                }
                if (ref.size() && 100.0 * double(sat) / double(ref.size()) > 1.0) {
                    std::printf("%-6s %-26s %9zu  REFERENCE DIAGNOSTIC: %.1f%% of "
                                "bytes are exactly +-448, %.1f%% >= 128\n",
                                (std::string("b") + (s2.block < 100 ? "0" : "") +
                                 (s2.block < 10 ? "0" : "") + std::to_string(s2.block) +
                                 "l" + std::to_string(s2.layer)).c_str(),
                                s2.type.c_str(), ref.size(),
                                100.0 * double(sat) / double(ref.size()),
                                100.0 * double(big) / double(ref.size()));
                    if (flag(argc, argv, "--filter-reference")) continue;
                }
            }
            if (flag(argc, argv, "--reference-diagnostics") &&
                s2.kernel.find("_outview") == std::string::npos &&
                s2.kernel.find("_ds") == std::string::npos &&
                s2.type != "CCTinlayoutFusedPreBlockSwin1H" &&
                s2.type != "CCTinlayoutFusedPostBlockSwin1H") {
                const size_t n5 = std::min<size_t>(ref.size(), 1u << 20);
                int dead = 0; char bits[64] = {0}; size_t bl = 0;
                for (size_t m3 : {size_t(1), size_t(2), size_t(8),
                                  size_t(16), size_t(32)}) {
                    size_t same = 0, tot = 0;
                    for (size_t i = 0; i < n5; ++i, ++tot)
                        if (ref[i] == ref[i ^ m3]) ++same;
                    if (tot && double(same) / double(tot) > 0.5) {
                        ++dead;
                        bl += size_t(std::snprintf(bits + bl, sizeof bits - bl,
                                                   "%s+%zu", bl ? "," : "", m3));
                    }
                }
                if (dead) {
                    std::printf("%-6s %-26s %9zu  REFERENCE DIAGNOSTIC: byte "
                                "offsets %s frequently match; repeated values "
                                "alone do not establish missing data\n",
                                (std::string("b") + (s2.block < 100 ? "0" : "") +
                                 (s2.block < 10 ? "0" : "") + std::to_string(s2.block) +
                                 "l" + std::to_string(s2.layer)).c_str(),
                                s2.type.c_str(), ref.size(), bits);
                    if (flag(argc, argv, "--filter-reference")) continue;
                }
            }
            if (s2.type == "CCVit1DQKV") {
                const size_t M=size_t(s2.tokens), C=1024;
                std::vector<uint8_t> got(M*3*C);
                ctx.download(act,voff[key_of(s2.block,s2.layer)],got.data(),got.size());
                const auto canon=tin::un_tile_blocked(got.data(),M,3*C);
                for (int part=0; part<3; ++part) {
                    const auto reference=part==0 ? ref : slurp(std::string(score_dir)+"/b"+
                        (s2.block<100?"0":"")+(s2.block<10?"0":"")+std::to_string(s2.block)+
                        "l"+std::to_string(s2.layer)+"_o"+std::to_string(part)+".bin");
                    double sx=0,sy=0,sxx=0,syy=0,sxy=0;
                    size_t same=0,n=0;
                    for (size_t t=0;t<M;++t) for (size_t c=0;c<C;++c) {
                        const size_t index=part==0?tin::byte_of(t,c,C):part==1?vit_k_byte(t,c):vit_v_byte(t,c);
                        if (index>=reference.size()) continue;
                        const auto a=canon[t*3*C+size_t(part)*C+c],b=reference[index];
                        same+=a==b;
                        const double x=tin::e4m3_to_f(a),y=tin::e4m3_to_f(b);
                        if (!std::isfinite(x)||!std::isfinite(y)) continue;
                        sx+=x;sy+=y;sxx+=x*x;syy+=y*y;sxy+=x*y;++n;
                    }
                    const double nn=double(n?n:1),xx=sxx-sx*sx/nn,yy=syy-sy*sy/nn;
                    const double r=xx>0&&yy>0?(sxy-sx*sy/nn)/std::sqrt(xx*yy):0;
                    std::printf("b%03dl%d CCVit1DQKV/%c %20zu %8.2f%%  r=%+.4f  rms %.3g/%.3g\n",
                                s2.block,s2.layer,"QKV"[part],n,100.0*same/nn,r,std::sqrt(sxx/nn),std::sqrt(syy/nn));
                    by_type[std::string("CCVit1DQKV/")+"QKV"[part]].push_back(r);
                }
                continue;
            }
            const Shape sh = shape_of(s2);
            // **The two adapters are a `_ds` and an `_upsample` that `shape_of`
            // reports as plain blocks.** The pre block halves and does not
            // widen - its gold is 16711680 B where the plain rule asks for
            // 66355200, so the scorer rejected it every run and b001l0, which
            // reads it, has been scoring 0.55 against an input the seeder
            // silently dropped. The post block doubles.
            const bool pre_ds  = s2.type == "CCTinlayoutFusedPreBlockSwin1H";
            const bool post_up = s2.type == "CCTinlayoutFusedPostBlockSwin1H";
            const size_t C = size_t(sh.fam == F_DS ? 2 * s2.C : sh.N);
            // `_upsample`'s plan row is its *input* level; its output is the
            // window grid's, one level up. Scoring it at `s2.tokens` reads a
            // quarter of the layer and calls the rest a mismatch.
            const size_t M = sh.fam == F_DS || pre_ds
                                 ? (reader_tokens.count(s2.block)
                                        ? reader_tokens[s2.block]
                                        : size_t(s2.tokens) / 4)
                           : sh.fam == F_UPS || post_up
                                 ? (host_boundary && sh.fam==F_UPS && reader_tokens.count(s2.block) ? reader_tokens[s2.block] : size_t(s2.gx)*8*size_t(s2.gy)*8)
                                 : size_t(s2.tokens);
            if (ref.size() < M * C) {
                std::printf("%-6s b%03dl%d  capture is %zu B, need %zu\n",
                            "", s2.block, s2.layer, ref.size(), M * C);
                continue;
            }
            std::vector<uint8_t> got(M * C);
            ctx.download(act, voff[key_of(s2.block, s2.layer)], got.data(), got.size());
            const std::vector<uint8_t> ours = tin::un_tile_blocked(got.data(), M, C);
            size_t same = 0;
            // **An `_outview` layer writes the view format at its own extent.**
            // It is the mirror of `_inpview` and the same argument applies: we
            // choose the arena format, so the kernel is the plain block and only
            // the *comparison* converts - exactly as for `_ds`, but without the
            // halving, since `_outview` changes layout and not resolution.
            const bool outview = s2.kernel.find("_outview") != std::string::npos;
            if (outview) {
                const size_t VW = size_t(s2.H), VH = size_t(s2.W);
                // **`_outview` is the mirror of `_ds` and takes the same law.**
                // A `_ds` writes its *halved* output over the window grid, `4*gx`
                // pixels; an `_outview` does not halve, so the same grid is
                // `8*gx`. `--outview-raster 0` forces the plain extent, 1 the
                // grid, -1 (the default) the same threshold the `_ds` table uses.
                // Measured: the plain extent wins at every level -
                // b047l3 1.0000, b055l0 0.9296, b061l0 0.9554, b065l0 0.9726,
                // b069l0 0.9930 against 0.0154 / 0.0262 / 0.2600 / 0.2697 /
                // 0.6705 for the grid. So `_outview` and `_ds` do *not* share
                // the law, and that is worth knowing: the halving is what makes
                // a `_ds` output take the window grid.
                const int orx = std::atoi(arg(argc, argv, "--outview-raster", "0"));
                const bool ogrid = orx == 1;
                const size_t PW = ogrid ? size_t(s2.gx) * 8 : VW;
                const size_t PH = ogrid ? size_t(s2.gy) * 8 : VH;
                const size_t TW = size_t(tiles_of(int(VW))) * 4;
                double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
                size_t n = 0, same2 = 0, tot = 0, near = 0;
                for (size_t y = 0; y < std::min(PH, VH); ++y)
                    for (size_t x = 0; x < std::min(PW, VW); ++x) {
                        const size_t t = tin::token_of(x, y, TW);
                        for (size_t c = 0; c < C; ++c) {
                            const size_t o = tin::view_byte_of(x, y, c, PW, PH);
                            if (o >= ref.size() || t * C + c >= ours.size()) continue;
                            const uint8_t a = ours[t * C + c], b = ref[o];
                            ++tot;
                            if (a == b) ++same2;
                            if (e4m3_codes_apart(a, b) <= 1) ++near;
                            const double u = tin::e4m3_to_f(a), v = tin::e4m3_to_f(b);
                            if (u != u || v != v) continue;
                            sx += u; sy += v; sxx += u * u; syy += v * v; sxy += u * v; ++n;
                        }
                    }
                const double dn = double(n ? n : 1);
                const double cxx = sxx - sx * sx / dn, cyy = syy - sy * sy / dn;
                const double rr = (cxx > 0 && cyy > 0)
                                ? (sxy - sx * sy / dn) / std::sqrt(cxx * cyy) : 0.0;
                char tag[64];
                std::snprintf(tag, sizeof tag, "b%03dl%d", s2.block, s2.layer);
                std::printf("%-6s %-26s %9zu %8.2f%%  r=%+.4f  u1 %5.2f%%  outview %zux%zu\n",
                            tag, s2.type.c_str(), tot,
                            100.0 * double(same2) / double(tot ? tot : 1), rr,
                            100.0 * double(near) / double(tot ? tot : 1), VW, VH);
            } else if (sh.fam == F_DS || pre_ds) {
                // Compare every value the next inpview layer reads. Its PTX
                // uses the logical extent for both row and channel-plane
                // strides, and floor(extent/4) for the output tile grid.
                // Do not invert our producer's store map here: doing that
                // excluded dropped values and hid a broken boundary.
                const size_t VW = host_boundary ? pad4((s2.H+1)/2) : size_t(s2.H)/2;
                    const size_t VH = host_boundary ? pad4((s2.W+1)/2) : size_t(s2.W)/2;
                const size_t TW = size_t(tiles_of(int(VW))) * 4;
                const size_t TH = size_t(tiles_of(int(VH))) * 4;
                double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
                size_t n = 0, near = 0;
                for (size_t y = 0; y < TH; ++y)
                    for (size_t x = 0; x < TW; ++x) {
                        const size_t t = tin::token_of(x, y, TW);
                        for (size_t c = 0; c < C; ++c) {
                            const size_t o = tin::view_byte_of(x, y, c, VW, VH);
                            if (o >= ref.size() || t * C + c >= ours.size()) continue;
                            const uint8_t a = ours[t * C + c], b = ref[o];
                            if (a == b) ++same;
                            if (e4m3_codes_apart(a, b) <= 1) ++near;
                            const double u = tin::e4m3_to_f(a), v = tin::e4m3_to_f(b);
                            if (u != u || v != v) continue;
                            sx += u; sy += v; sxx += u * u; syy += v * v; sxy += u * v; ++n;
                        }
                    }
                const double dn = double(n ? n : 1);
                const double cxx = sxx - sx * sx / dn, cyy = syy - sy * sy / dn;
                const double rr = (cxx > 0 && cyy > 0)
                    ? (sxy - sx * sy / dn) / std::sqrt(cxx * cyy) : 0.0;
                const double vgain = cyy > 0 ? (sxy - sx * sy / dn) / cyy : 0.0;
                const double voffs = sx / dn - vgain * sy / dn;
                char tag[64];
                std::snprintf(tag, sizeof tag, "b%03dl%d", s2.block, s2.layer);
                std::printf("%-6s %-26s %9zu %8.2f%%  r=%+.4f  g=%.3f o=%+.4f  u1 %5.2f%%  consumer-view %zux%zu\n",
                            tag, s2.type.c_str(), n, 100.0 * double(same) / dn, rr, vgain, voffs,
                            100.0 * double(near) / dn, VW, VH);
            } else {
                // **Byte agreement is the wrong end-to-end metric and it took a
                // while to see it.** Our accumulator is f32 where NVIDIA's is a
                // blocked f16, so even a perfect layer disagrees on ~14% of
                // bytes - almost all by one e4m3 ulp - and twenty layers of that
                // compounds to nothing. The *values* are what decides whether
                // the image is right, so report the correlation beside the
                // bytes: 3% bytes at 0.99 correlation is a working chain,
                // 3% bytes at 0.0 is a broken one, and the byte column cannot
                // tell them apart.
                double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
                size_t n = 0, zo = 0, zr = 0, near = 0;
                // **A pooled r cannot tell a wrong kernel from a partly wrong
                // gold, and this project has now been fooled by that twice.**
                // `CCVit1DFfnExpand` scored r=+0.2384 for weeks: 3072 of its
                // 4096 channels match NVIDIA *byte histogram for byte
                // histogram*, and the other 1024 are saturated at 0x7E in the
                // capture, which no weight in the file can explain - the row
                // norms of the two halves have medians 1.000 and 0.997. The
                // same kernel scores 98.21% against the clean single-layer
                // oracle. So the per-channel split goes beside the pooled
                // number, always: `chan` is the fraction of channels whose own
                // correlation clears 0.9, and a layer at r=0.24 / chan 75% is a
                // broken reference, not a broken kernel.
                std::vector<double> px(C, 0), py(C, 0), pxx(C, 0), pyy(C, 0), pxy(C, 0);
                std::vector<size_t> pn(C, 0);
                for (size_t t = 0; t < M; ++t)
                    for (size_t c = 0; c < C; ++c) {
                        const uint8_t a = ours[t * C + c], b = ref[tin::byte_of(t, c, C)];
                        if (a == b) ++same;
                        if (e4m3_codes_apart(a, b) <= 1) ++near;
                        if (!(a & 0x7F)) ++zo;
                        if (!(b & 0x7F)) ++zr;
                        const double u = tin::e4m3_to_f(a), v = tin::e4m3_to_f(b);
                        if (u != u || v != v) continue;
                        sx += u; sy += v; sxx += u * u; syy += v * v; sxy += u * v; ++n;
                        px[c] += u; py[c] += v; pxx[c] += u * u; pyy[c] += v * v;
                        pxy[c] += u * v; ++pn[c];
                    }
                size_t cgood = 0;
                for (size_t c = 0; c < C; ++c) {
                    if (!pn[c]) continue;
                    const double d = double(pn[c]);
                    const double axx = pxx[c] - px[c] * px[c] / d;
                    const double ayy = pyy[c] - py[c] * py[c] / d;
                    if (axx <= 0 || ayy <= 0) continue;
                    if ((pxy[c] - px[c] * py[c] / d) / std::sqrt(axx * ayy) > 0.9) ++cgood;
                }
                const double dn = double(n ? n : 1);
                const double cxx = sxx - sx * sx / dn, cyy = syy - sy * sy / dn;
                const double cxy = sxy - sx * sy / dn;
                const double r = (cxx > 0 && cyy > 0) ? cxy / std::sqrt(cxx * cyy) : 0.0;
                // **A tone error is an affine error, and r cannot see one.** A
                // layer that reproduces every structure but scales it, or adds a
                // constant to it, scores r=1.000 and still moves the picture's
                // tone. Fit `ours = g * reference + o` over the whole tensor and
                // print both: g away from 1 is a gain defect, o away from 0 a
                // DC one, and the first layer in the chain where either leaves
                // its neighbours is where the defect is made.
                const double gain = cyy > 0 ? cxy / cyy : 0.0;
                const double off  = sx / dn - gain * sy / dn;
                char tag[64];
                std::snprintf(tag, sizeof tag, "b%03dl%d", s2.block, s2.layer);
                // **Zeros on both sides agree without meaning anything.** A
                // kernel that writes nothing scores its reference's own zero
                // fraction and reads as a partial result; print both so the two
                // cannot be confused. rms says whether the magnitudes even match.
                std::printf("%-6s %-26s %9zu %8.2f%%  r=%+.4f  u1 %5.2f%%  chan %3.0f%%  "
                            "zero %2.0f%%/%2.0f%%  rms %.3g/%.3g  g=%.3f o=%+.4f\n",
                            tag, s2.type.c_str(),
                            M * C, 100.0 * double(same) / double(M * C), r,
                            100.0 * double(near) / double(M * C),
                            100.0 * double(cgood) / double(C),
                            100.0 * double(zo) / double(M * C),
                            100.0 * double(zr) / double(M * C),
                            std::sqrt(sxx / dn), std::sqrt(syy / dn), gain, off);
                by_type[s2.type].push_back(r);
            }
            // A `_ds` layer has a second capture - its full-resolution output,
            // which the U-Net skip partner reads. Scoring it splits the three
            // dispatches in two: the block itself against `_skip`, and the 2x2
            // mean plus the resample against the main output.
            // **The pre block has a skip too and nothing ever scored it.**
            // `70 <- [69, 0]`: block 0's full-resolution output is the U-Net's
            // longest edge and goes straight into the post block, so an error
            // in it lands in the picture without passing through any of the
            // other 151 layers. `shape_of` reports the pre block as a plain
            // block, so the `F_DS` test below skipped it - the seeder already
            // knew better (it seeds `skip_key(0)` under the same `pre_ds` test).
            if (sh.fam == F_DS || pre_ds) {
                std::snprintf(cap, sizeof cap, host_boundary ? "%s/b%03dl%d_o1.bin" : "%s/b%03dl%d_skip.bin",
                              score_dir, s2.block, s2.layer);
                std::vector<uint8_t> sref = slurp(cap);
                if (sref.empty())
                    sref = slurp(std::string(score_dir) + "/b" +
                                 (s2.block < 100 ? s2.block < 10 ? "00" : "0" : "") +
                                 std::to_string(s2.block) + "l" +
                                 std::to_string(s2.layer) + "_skip.bin");
                const size_t SM = size_t(s2.tokens), SC = size_t(pre_ds ? sh.N : s2.C);
                if (sref.size() >= SM * SC) {
                    std::vector<uint8_t> sg(SM * SC);
                    ctx.download(act, voff[skip_key(s2.block)], sg.data(), sg.size());
                    const std::vector<uint8_t> so = tin::un_tile_blocked(sg.data(), SM, SC);
                    // **A byte agreement on its own is not a score here.**
                    // This tensor is ~19% zero on both sides, and two zeros
                    // agreeing is not the kernel being right: the raw number
                    // read 85.64% at b022l0 while the agreement among *non-zero*
                    // bytes was 2.55% and the correlation 0.058. Same rms
                    // (2.333 both sides), same zero fraction, wrong positions.
                    size_t ss = 0, nz = 0, nzs = 0;
                    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
                    for (size_t t = 0; t < SM; ++t)
                        for (size_t c = 0; c < SC; ++c) {
                            const uint8_t a2 = so[t * SC + c];
                            const uint8_t b2 = sref[tin::byte_of(t, c, SC)];
                            if (a2 == b2) ++ss;
                            if (a2 || b2) { ++nz; if (a2 == b2) ++nzs; }
                            const double x = tin::e4m3_to_f(a2), y = tin::e4m3_to_f(b2);
                            if (x != x || y != y) continue;
                            sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
                        }
                    const double dn2 = double(SM * SC);
                    const double vx = sxx / dn2 - (sx / dn2) * (sx / dn2);
                    const double vy = syy / dn2 - (sy / dn2) * (sy / dn2);
                    const double cv = sxy / dn2 - (sx / dn2) * (sy / dn2);
                    const double sgain = vy > 0 ? cv / vy : 0.0;
                    const double soff = sx / dn2 - sgain * sy / dn2;
                    std::printf("%-6s %-26s %9zu %8.2f%%  r=%+.4f  nonzero %5.2f%%"
                                "  rms %.3g/%.3g  g=%.3f o=%+.4f  the block itself\n",
                                "  skip", "(full resolution)", SM * SC,
                                100.0 * double(ss) / dn2,
                                cv / (std::sqrt(vx * vy) + 1e-30),
                                100.0 * double(nzs) / double(nz ? nz : 1),
                                std::sqrt(sxx / dn2), std::sqrt(syy / dn2), sgain, soff);
                }
            }
        }
        {
            size_t good = 0, total = 0;
            for (const auto& kv : by_type)
                for (double v : kv.second) { ++total; if (v >= 0.9) ++good; }
            // Flag a family-specific discrepancy without assigning its cause.
            if (total >= 20 && double(good) / double(total) >= 0.6)
                for (const auto& kv : by_type) {
                    if (kv.second.size() < 3) continue;
                    double worst = -2.0;
                    for (double v : kv.second) worst = std::max(worst, v);
                    if (worst >= 0.9) continue;
                    std::printf("\nFAMILY DIAGNOSTIC: all %zu %s score below "
                                "0.9 (best %+.4f) while %.0f%% of the frame is "
                                "above it. Check the reference, bindings, launch "
                                "shape, decoding, and shader arithmetic.\n",
                                kv.second.size(), kv.first.c_str(), worst,
                                100.0 * double(good) / double(total));
                }
        }
        persist_check();
        if(arg(argc,argv,"--out-image"))write_surface();
        return 0;
    }
    nrvk::Telemetry tel;
    tel.dir = nrvk::Telemetry::find(0x1002, 0x7550);
    // `--per-layer` times every dispatch inside the real frame, rather than
    // modelling the frame as "each family's kernel, run alone, times its layer
    // count" the way a per-family frame model does. The two disagree wherever the
    // chain matters, and the chain is what ships.
    const bool per_layer = flag(argc, argv, "--per-layer");
    // Warm exactly the measured command pattern, including step queries.
    // Creating its query pool only after warmup introduces an idle transition.
    runner.per_step = per_layer;
    tel.start();
    if (warm) runner.run_graph(steps, warm);
    const double measured_begin = tel.elapsed_ms();
    const double ms = runner.run_graph(steps, repeats);
    const double measured_end = tel.elapsed_ms();
    tel.stop();
    if (g_prof_words) {
        const size_t prof_words = g_prof_words, prof_off = g_prof_off;
        // One more frame with a clean record region, then the region to a file.
        std::vector<uint32_t> pz(prof_words, 0u);
        ctx.upload(act, VkDeviceSize(prof_off) * 4, pz.data(), pz.size() * 4);
        // NR_PROF_FRAMES back-to-back frames so the clock is back at steady
        // state; the profile reader keeps the last one (frames split at idle gaps).
        const uint32_t pf = std::getenv("NR_PROF_FRAMES") ? uint32_t(std::atoi(std::getenv("NR_PROF_FRAMES"))) : 1u;
        runner.run_graph(steps, pf);
        ctx.download(act, VkDeviceSize(prof_off) * 4, pz.data(), pz.size() * 4);
        const char* po = std::getenv("NR_PROF_OUT") ? std::getenv("NR_PROF_OUT") : "prof.bin";
        FILE* f = std::fopen(po, "wb");
        if (f) { std::fwrite(pz.data(), 4, pz.size(), f); std::fclose(f); }
        std::printf("NR_PROF: %u records -> %s\n", pz[0], po);
    }
    if (const char* path=arg(argc,argv,"--telemetry-csv")) {
        std::ofstream trace(path);
        if(!trace)throw std::runtime_error("cannot open telemetry CSV");
        trace << "elapsed_ms,phase,gfx_mhz,power_w,edge_c\n";
        for(size_t i=0;i<tel.elapsed.size();++i)
            trace << tel.elapsed[i] << ','
                  << (tel.elapsed[i]<measured_begin ? "warmup" :
                      tel.elapsed[i]<=measured_end ? "measured" : "after") << ','
                  << tel.mhz[i] << ',' << tel.watts[i] << ',' << tel.celsius[i] << '\n';
    }
    // Keep the reported mean restricted to the measurement interval, while
    // the optional trace retains warmup and its frequency ramp.
    size_t measured_first=0;
    while(measured_first<tel.elapsed.size() && tel.elapsed[measured_first]<measured_begin)
        ++measured_first;
    size_t measured_last=measured_first;
    while(measured_last<tel.elapsed.size() && tel.elapsed[measured_last]<=measured_end)
        ++measured_last;
    for(auto* samples : {&tel.mhz,&tel.watts,&tel.celsius}) {
        samples->erase(samples->begin()+measured_last,samples->end());
        samples->erase(samples->begin(),samples->begin()+measured_first);
    }
    const double mhz = nrvk::Telemetry::mean(tel.mhz);
    std::printf("\n%zu dispatches, one submit: %.3f ms  @ %.0f MHz\n",
                steps.size(), ms, mhz);
    if (!g_tc_err.empty()) {   // NR_TCHAIN: a wait that ran out of its bound set its error word
        std::vector<uint32_t> w(g_tc_words.size());
        ctx.download(act, VkDeviceSize(g_tc_base) * 4, w.data(), w.size() * 4);
        size_t bad = 0;
        for (size_t e : g_tc_err) bad += w[e] != 0u;
        if (std::getenv("NR_TCHAIN_DEBUG")) {
            uint32_t ep = 0; ctx.download(act, VkDeviceSize(g_chain_epoch_word) * 4, &ep, 4);
            std::printf("tile counters: frame word %u, C=256 epoch %u, errors %zu; words 0..8:", w[0], ep, bad);
            for (size_t i = 0; i < 9 && i < w.size(); ++i) std::printf(" %u", w[i]);
            std::printf(" base %zu\n", g_tc_base);
        }
        if (bad) std::printf("tile counters: %zu dispatches ran out of a wait bound (output not trustworthy)\n", bad);
    }
    if (!tel.mhz.empty()) {
        double lo=1e30, hi=0; size_t samples=0;
        for (double v:tel.mhz) if (v>=0) {lo=std::min(lo,v);hi=std::max(hi,v);++samples;}
        if (samples) std::printf("telemetry: %zu samples, gfx %.0f..%.0f MHz, %.1f W, %.1f C edge; warmup %u frames\n",
            samples,lo,hi,nrvk::Telemetry::mean(tel.watts),nrvk::Telemetry::mean(tel.celsius),warm);
    }
    if (!warm) std::printf("timing note: cold run; use matched warmup and measured clocks for performance comparisons\n");
    std::printf("measurement wall %.3f ms, GPU total %.3f ms over %u frames\n",
        measured_end-measured_begin,ms*repeats,repeats);
    if (per_layer && runner.step_ms.size() == disp.size()) {
        // Grouped by (family, width), because a family's cost is not one number:
        // the same fused-Swin kernel is 8 layers at C=32 and 16 at C=256, and
        // those are different problems - occupancy differs with the width.
        //
        // **Keyed by the kernel as well as the family**, because a plan step is
        // not always one dispatch: `CCVit1DQKV` is a GEMM *and* the QKV norm,
        // and grouped by family alone the table averaged 17.5 us over two
        // kernels that are 4x apart - which is exactly the number a fusion
        // round needs split.
        struct Row { int n; double ms; };
        std::map<std::pair<std::string, int>, Row> g;
        double tot = 0;
        for (size_t i = 0; i < disp.size(); ++i) {
            Row& r = g[{disp[i].s->type + "/" + disp[i].kern, disp[i].s->C}];
            r.n += 1;
            r.ms += runner.step_ms[i];
            tot += runner.step_ms[i];
        }
        std::vector<std::pair<double, std::string>> rows;
        for (const auto& kv : g) {
            char b[256];
            std::snprintf(b, sizeof b, "%-44s %5d %5d %8.1f %8.3f %6.1f%%",
                          kv.first.first.c_str(), kv.first.second, kv.second.n,
                          1000.0 * kv.second.ms / kv.second.n, kv.second.ms,
                          100.0 * kv.second.ms / (tot > 0 ? tot : 1));
            rows.push_back({kv.second.ms, b});
        }
        std::sort(rows.begin(), rows.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        std::printf("\n%-44s %5s %5s %8s %8s %7s\n",
                    "family/kernel", "C", "n", "us each", "ms", "share");
        for (const auto& r : rows) std::printf("%s\n", r.second.c_str());
        std::printf("%-44s %5s %5zu %8s %8.3f\n", "sum of the dispatches", "",
                    disp.size(), "", tot);
        // The single dispatches worth looking at next, named by block and layer
        // so `--only B:L` can reproduce one on its own.
        std::vector<size_t> ix(disp.size());
        for (size_t i = 0; i < ix.size(); ++i) ix[i] = i;
        std::sort(ix.begin(), ix.end(), [&](size_t a, size_t b) {
            return runner.step_ms[a] > runner.step_ms[b];
        });
        std::printf("\nthe twelve dearest dispatches\n");
        for (size_t k = 0; k < 12 && k < ix.size(); ++k) {
            const Disp& d = disp[ix[k]];
            std::printf("  b%03dl%d %-28s C=%-4d %5.0f us  %s\n", d.s->block,
                        d.s->layer, d.s->type.c_str(), d.s->C,
                        1000.0 * runner.step_ms[ix[k]], d.kern.c_str());
        }
        // The cost model wants *work per dispatch*, and the only place the
        // launch grid exists is here. `--dispatch-grid` prints one line per
        // dispatch - kernel, grid, and its measured time - which is what turns
        // a per-window cycle estimate from the ISA into a predicted duration.
        // Print only; the graph is unchanged by it.
        if (flag(argc, argv, "--dispatch-grid")) {
            std::printf("\n%-5s %-32s %-34s %5s %7s %7s %5s %9s %9s %6s %6s %9s\n",
                        "i", "kernel", "type", "C", "gx", "gy", "gz",
                        "groups", "us", "ci", "co", "tokens");
            for (size_t i = 0; i < disp.size(); ++i) {
                const Disp& d = disp[i];
                std::printf("%-5zu %-32s %-34s %5d %7u %7u %5u %9llu %9.1f %6d %6d %9ld\n",
                            i, d.kern.c_str(), d.s->type.c_str(), d.s->C,
                            d.gx, d.gy, d.gz,
                            (unsigned long long)d.gx * d.gy * d.gz,
                            1000.0 * runner.step_ms[i], d.s->ci, d.s->co,
                            d.s->tokens);
            }
        }
    }
    // **"Does the output change when the input changes" is the only oracle this
    // edge has.** The pre-block's gate at zero leaves a kernel that runs,
    // returns success and fills every output byte with a result that does not
    // depend on the image at all; the same shape of failure has cost this
    // project a week twice. So the surface comes back and gets described, and
    // `--out-image` writes it for a second run to be compared against.
    persist_check();
    write_surface();
    std::map<std::string, int> by;
    for (const Disp& d : disp) ++by[d.s->type];
    std::printf("\n%-34s %s\n", "family", "dispatches");
    for (const auto& kv : by) std::printf("%-34s %6d\n", kv.first.c_str(), kv.second);
    for (const auto& kv : missing)
        std::printf("%-34s %6d  no kernel\n", kv.first.c_str(), kv.second);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "nr_graph: %s\n", e.what());
    return 1;
}
#endif  // NR_NO_MAIN
