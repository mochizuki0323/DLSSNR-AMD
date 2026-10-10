#pragma once
// The engine-input check (NR_INPUT_CHECK=1): is what the game hands us, and what we hand back,
// what it should be?
//
// Every so often the runtime copies, inside the caller's own command buffer, everything the
// network reads and writes on the engine path into one host-visible buffer:
//
//   frame A   the colour the network is given                       (the caller's frame)
//   frame B   the colour again, the next recording of the same feature
//             the game's motion vectors, full resolution            (EngineFrame::motion)
//             the motion field the network actually samples         (Temporal::flow[0])
//             the depth the network actually samples                (Temporal::depth, or the
//                                                                    caller's R32F depth in place)
//             the frame after the network wrote it back             (what the game gets)
//
// and a marker word after them. No fence and no wait: on the D3D12 path the command buffer is
// the game's and so is the queue, so the marker is polled on later recordings. When it lands the
// buffer is copied out and a detached thread scores it and writes the log lines (and, for the
// first few in which the picture moves, PNGs next to the log):
//
//   motion   the current frame against the previous one warped by the game's vectors, as the
//            network reads them and as the original reads them, against no motion, the
//            opposite sign, half and double the length and one axis flipped. The game's vectors
//            must beat all of those where the picture moves.
//   depth    range, which end is far, whether that agrees with DepthInverted, and whether its
//            edges sit on the picture's edges (per quadrant: a wrong subrect or extent shows as
//            a shift that grows across the frame).
//   output   finite, changed, aligned with the input, no blocks left untouched or black.
//
// Off unless set_input_check(true, ...) was called; then it costs one flag test a recording
// between captures and one ~100 MB copy (1080p) every few seconds.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "nr_log.hpp"
#include "nr_runtime.hpp"
#include "nrvk.hpp"

namespace nr {
namespace incheck {

inline std::atomic<bool> g_on{false};
inline std::string g_folder;                 // where PNGs go; empty: none
inline std::atomic<int> g_dumps_left{4};     // PNG sets, process-wide
inline std::atomic<int> g_serial{0};         // capture number, process-wide

// ---- pixels ------------------------------------------------------------------

inline float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    float v;
    if (e == 0) v = float(m) * (1.0f / 1024.0f) * 6.103515625e-05f;
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(1.0f + float(m) / 1024.0f, int(e) - 15);
    return s ? -v : v;
}
inline float uf(uint32_t v, int mbits) {   // unsigned small float, 5-bit exponent
    const uint32_t e = v >> mbits, m = v & ((1u << mbits) - 1u);
    if (e == 0) return float(m) / float(1u << mbits) * 6.103515625e-05f;
    if (e == 31) return m ? NAN : INFINITY;
    return std::ldexp(1.0f + float(m) / float(1u << mbits), int(e) - 15);
}

inline uint32_t colour_bpp(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: return 8;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return 4;
        default: return 0;
    }
}
inline void decode_rgb(VkFormat f, const uint8_t* p, float o[3]) {
    uint32_t w = 0;
    switch (f) {
        case VK_FORMAT_R16G16B16A16_SFLOAT: {
            uint16_t h[3]; std::memcpy(h, p, 6);
            for (int i = 0; i < 3; ++i) o[i] = h2f(h[i]);
            return;
        }
        case VK_FORMAT_R16G16B16A16_UNORM: {
            uint16_t h[3]; std::memcpy(h, p, 6);
            for (int i = 0; i < 3; ++i) o[i] = h[i] / 65535.0f;
            return;
        }
        case VK_FORMAT_R32G32B32A32_SFLOAT: std::memcpy(o, p, 12); return;
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB:
            for (int i = 0; i < 3; ++i) o[i] = p[i] / 255.0f;
            return;
        case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
            for (int i = 0; i < 3; ++i) o[i] = p[2 - i] / 255.0f;
            return;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
            std::memcpy(&w, p, 4);
            o[0] = (w & 1023u) / 1023.0f; o[1] = ((w >> 10) & 1023u) / 1023.0f; o[2] = ((w >> 20) & 1023u) / 1023.0f;
            return;
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
            std::memcpy(&w, p, 4);
            o[2] = (w & 1023u) / 1023.0f; o[1] = ((w >> 10) & 1023u) / 1023.0f; o[0] = ((w >> 20) & 1023u) / 1023.0f;
            return;
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
            std::memcpy(&w, p, 4);
            o[0] = uf(w & 0x7FFu, 6); o[1] = uf((w >> 11) & 0x7FFu, 6); o[2] = uf((w >> 22) & 0x3FFu, 5);
            return;
        case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: {
            std::memcpy(&w, p, 4);
            const float sc = std::ldexp(1.0f, int(w >> 27) - 15 - 9);
            o[0] = float(w & 511u) * sc; o[1] = float((w >> 9) & 511u) * sc; o[2] = float((w >> 18) & 511u) * sc;
            return;
        }
        default: o[0] = o[1] = o[2] = NAN; return;
    }
}

inline uint32_t motion_bpp(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_SNORM: case VK_FORMAT_R16G16_UNORM: return 4;
        case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
        case VK_FORMAT_R8G8_SNORM: return 2;
        default: return 0;
    }
}
inline void decode_mv(VkFormat f, const uint8_t* p, float o[2]) {
    switch (f) {
        case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16B16A16_SFLOAT: {
            uint16_t h[2]; std::memcpy(h, p, 4); o[0] = h2f(h[0]); o[1] = h2f(h[1]); return;
        }
        case VK_FORMAT_R16G16_SNORM: {
            int16_t h[2]; std::memcpy(h, p, 4);
            o[0] = std::max(-1.0f, h[0] / 32767.0f); o[1] = std::max(-1.0f, h[1] / 32767.0f); return;
        }
        case VK_FORMAT_R16G16_UNORM: {
            uint16_t h[2]; std::memcpy(h, p, 4); o[0] = h[0] / 65535.0f; o[1] = h[1] / 65535.0f; return;
        }
        case VK_FORMAT_R8G8_SNORM: {
            const int8_t a = int8_t(p[0]), b = int8_t(p[1]);
            o[0] = std::max(-1.0f, a / 127.0f); o[1] = std::max(-1.0f, b / 127.0f); return;
        }
        case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32B32A32_SFLOAT: std::memcpy(o, p, 8); return;
        default: o[0] = o[1] = NAN; return;
    }
}

// A W x H float plane.
struct Plane {
    uint32_t w{}, h{};
    std::vector<float> v;
    float at(int x, int y) const {
        x = std::clamp(x, 0, int(w) - 1); y = std::clamp(y, 0, int(h) - 1);
        return v[size_t(y) * w + size_t(x)];
    }
    // Texel-centre convention: (px, py) in texels, (0.5, 0.5) is the first texel's centre.
    float bilinear(float px, float py) const {
        px -= 0.5f; py -= 0.5f;
        const float fx = std::floor(px), fy = std::floor(py);
        const int x = int(fx), y = int(fy);
        const float ax = px - fx, ay = py - fy;
        return (at(x, y) * (1 - ax) + at(x + 1, y) * ax) * (1 - ay) +
               (at(x, y + 1) * (1 - ax) + at(x + 1, y + 1) * ax) * ay;
    }
    float nearest(float u, float vv) const {
        return at(int(std::floor(u * float(w))), int(std::floor(vv * float(h))));
    }
};

inline double percentile(std::vector<float> s, double q) {
    if (s.empty()) return NAN;
    const size_t k = std::min(s.size() - 1, size_t(q * double(s.size() - 1) + 0.5));
    std::nth_element(s.begin(), s.begin() + long(k), s.end());
    return s[k];
}

// ---- PNG (stored deflate: large files, no zlib needed) -------------------------------

inline uint32_t crc32(const uint8_t* p, size_t n, uint32_t c = 0xFFFFFFFFu) {
    static uint32_t table[256];
    static bool ready = false;
    if (!ready) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t r = i;
            for (int k = 0; k < 8; ++k) r = (r & 1u) ? 0xEDB88320u ^ (r >> 1) : r >> 1;
            table[i] = r;
        }
        ready = true;
    }
    for (size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c;
}

inline bool write_png(const std::string& path, uint32_t w, uint32_t h, const std::vector<uint8_t>& rgb) {
    std::vector<uint8_t> raw;
    raw.reserve(size_t(h) * (size_t(w) * 3 + 1));
    for (uint32_t y = 0; y < h; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgb.begin() + long(size_t(y) * w * 3), rgb.begin() + long(size_t(y + 1) * w * 3));
    }
    std::vector<uint8_t> z{0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) { a = (a + c) % 65521u; b = (b + a) % 65521u; }
    for (size_t at = 0; at < raw.size() || at == 0;) {
        const size_t n = std::min<size_t>(65535, raw.size() - at);
        const bool last = at + n >= raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(uint8_t(n)); z.push_back(uint8_t(n >> 8));
        z.push_back(uint8_t(~n)); z.push_back(uint8_t((~n) >> 8));
        z.insert(z.end(), raw.begin() + long(at), raw.begin() + long(at + n));
        at += n;
        if (last) break;
    }
    const uint32_t adler = (b << 16) | a;
    for (int s = 24; s >= 0; s -= 8) z.push_back(uint8_t(adler >> s));
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    auto be32 = [](uint32_t v, uint8_t* o) { o[0] = uint8_t(v >> 24); o[1] = uint8_t(v >> 16); o[2] = uint8_t(v >> 8); o[3] = uint8_t(v); };
    auto chunk = [&](const char* type, const uint8_t* data, size_t n) {
        uint8_t hd[8]; be32(uint32_t(n), hd); std::memcpy(hd + 4, type, 4);
        std::fwrite(hd, 1, 8, fp);
        if (n) std::fwrite(data, 1, n, fp);
        uint32_t c = crc32(hd + 4, 4);
        c = crc32(data, n, c) ^ 0xFFFFFFFFu;
        uint8_t t[4]; be32(c, t); std::fwrite(t, 1, 4, fp);
    };
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::fwrite(sig, 1, 8, fp);
    uint8_t ihdr[13]; be32(w, ihdr); be32(h, ihdr + 4);
    ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = ihdr[11] = ihdr[12] = 0;
    chunk("IHDR", ihdr, 13);
    chunk("IDAT", z.data(), z.size());
    chunk("IEND", nullptr, 0);
    const bool ok = std::ferror(fp) == 0;
    std::fclose(fp);
    return ok;
}

inline uint8_t to8(float v) {
    if (!std::isfinite(v)) return 0;
    return uint8_t(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
}

// ---- one capture, off the render thread ---------------------------------------------

struct Job {
    int serial{};
    bool dump{};
    std::string folder;
    uint64_t feature{}, recording{};
    // Colour: the caller's frame, W x H, its own format.
    uint32_t W{}, H{};
    VkFormat cfmt{};
    std::vector<uint8_t> prev, cur, out;
    // The game's motion vectors as handed over (after the subrect), and the network's field.
    uint32_t mvw{}, mvh{};
    VkFormat mvfmt{};
    std::vector<uint8_t> mv;
    Plane flow_x, flow_y;
    // The depth the network samples, model sized.
    bool have_depth{}, inverted{}, depth_in_place{};
    Plane depth;
    uint32_t mw{}, mh{};
    float sx{}, sy{};         // normalized motion scale (texture units -> uv)
    bool mv_direct{};         // the network samples the game's vectors in place (no quarter field)
    bool gate{}, reset{};
    uint32_t between{};       // recordings of other features between A and B
};

struct Shift {
    int dx{}, dy{};
    double gain{};   // best score over the (0,0) score; 1 = centred
};

// Where B's edges sit relative to A's, over [x0,x1) x [y0,y1). `ea` is the edge strength of
// the guide (its strongest 4% are the probes); `eb` that of the picture.
inline Shift align(const Plane& ea, const Plane& eb, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) {
    constexpr int R = 6;
    std::vector<float> s;
    const uint32_t step = std::max<uint32_t>(1, (x1 - x0) * (y1 - y0) / 200000);
    for (uint32_t y = y0 + R; y + R < y1; y += step)
        for (uint32_t x = x0 + R; x + R < x1; x += step) s.push_back(ea.v[size_t(y) * ea.w + x]);
    if (s.size() < 100) return {0, 0, 0};
    const float thr = float(percentile(s, 0.96));
    if (!(thr > 0)) return {0, 0, 0};
    double score[2 * R + 1][2 * R + 1] = {};
    for (uint32_t y = y0 + R; y + R < y1; y += step)
        for (uint32_t x = x0 + R; x + R < x1; x += step) {
            if (ea.v[size_t(y) * ea.w + x] < thr) continue;
            for (int dy = -R; dy <= R; ++dy)
                for (int dx = -R; dx <= R; ++dx)
                    score[dy + R][dx + R] += eb.v[size_t(int(y) + dy) * eb.w + size_t(int(x) + dx)];
        }
    // Centred unless somewhere else is clearly better: a straight edge scores the same all
    // along itself, and that tie must not read as a shift.
    Shift best{0, 0, 0};
    double top = score[R][R];
    for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx)
            if (score[dy + R][dx + R] > top * 1.10) { top = score[dy + R][dx + R]; best.dx = dx; best.dy = dy; }
    best.gain = score[R][R] > 0 ? top / score[R][R] : 0;
    return best;
}

// |gradient| of a plane, central differences.
inline Plane gradient(const Plane& p) {
    Plane g{p.w, p.h, std::vector<float>(p.v.size(), 0.0f)};
    for (uint32_t y = 1; y + 1 < p.h; ++y)
        for (uint32_t x = 1; x + 1 < p.w; ++x) {
            const float a = std::fabs(p.at(int(x) + 1, int(y)) - p.at(int(x) - 1, int(y)));
            const float b = std::fabs(p.at(int(x), int(y) + 1) - p.at(int(x), int(y) - 1));
            const float v = a + b;
            g.v[size_t(y) * p.w + x] = std::isfinite(v) ? v : 0.0f;
        }
    return g;
}

// The four quadrants' shifts, and a line of text for them.
struct Quadrants {
    Shift q[4];
    bool centred(int tol = 1) const {   // every quadrant within tol pixels of (0,0)
        for (const Shift& s : q)
            if (std::abs(s.dx) > tol || std::abs(s.dy) > tol) return false;
        return true;
    }
    std::string text() const {
        static const char* name[4] = {"TL", "TR", "BL", "BR"};
        std::string t;
        for (int i = 0; i < 4; ++i) {
            char b[48];
            std::snprintf(b, sizeof b, "%s%s %+d,%+d", i ? " " : "", name[i], q[i].dx, q[i].dy);
            t += b;
        }
        return t;
    }
};

inline Quadrants quadrants(const Plane& ea, const Plane& eb) {
    const uint32_t w = std::min(ea.w, eb.w), h = std::min(ea.h, eb.h);
    const uint32_t xs[3] = {0, w / 2, w}, ys[3] = {0, h / 2, h};
    Quadrants r;
    for (int i = 0; i < 4; ++i)
        r.q[i] = align(ea, eb, xs[i & 1], ys[i >> 1], xs[(i & 1) + 1], ys[(i >> 1) + 1]);
    return r;
}

inline void analyse(Job& j) {
    const uint32_t W = j.W, H = j.H;
    const size_t N = size_t(W) * H;
    const uint32_t cb = colour_bpp(j.cfmt);
    char line[1024];
    const int id = j.serial;
    auto logl = [&](const char* fmt, auto... args) {
        std::snprintf(line, sizeof line, fmt, args...);
        nr::logf("[nr] input check %d: %s", id, line);
    };
    logl("feature %llx, recording %llu, colour %ux%u VkFormat %d, model %ux%u, motion %ux%u VkFormat %d, "
         "depth %s%s (%ux%u), DepthInverted %d, MV scale %.6g %.6g (= %.4g %.4g px), reset %d, history used %d, "
         "%u other recordings between the two frames",
         (unsigned long long)j.feature, (unsigned long long)j.recording, W, H, int(j.cfmt), j.mw, j.mh, j.mvw,
         j.mvh, int(j.mvfmt), j.have_depth ? "yes" : "no", j.depth_in_place ? " (sampled in place)" : "",
         j.depth.w, j.depth.h, j.inverted ? 1 : 0, j.sx, j.sy, j.sx * float(j.mw), j.sy * float(j.mh),
         j.reset ? 1 : 0, j.gate ? 1 : 0, j.between);
    if (!cb) { logl("colour format %d is not one this check decodes; stopping here", int(j.cfmt)); return; }

    // ---- colour ----
    Plane Lp{W, H, std::vector<float>(N)}, Lc{W, H, std::vector<float>(N)}, Lo{W, H, std::vector<float>(N)};
    std::vector<float> lum;
    lum.reserve(N / 16 + 1);
    size_t nonfinite_in = 0, nonfinite_out = 0, over1 = 0, black = 0, changed = 0;
    double sum_in = 0, sum_out = 0, maxd = 0;
    for (size_t i = 0; i < N; ++i) {
        float a[3], b[3], c[3];
        decode_rgb(j.cfmt, &j.prev[i * cb], a);
        decode_rgb(j.cfmt, &j.cur[i * cb], b);
        decode_rgb(j.cfmt, &j.out[i * cb], c);
        auto L = [](const float* v) { return 0.2126f * v[0] + 0.7152f * v[1] + 0.0722f * v[2]; };
        Lp.v[i] = L(a); Lc.v[i] = L(b); Lo.v[i] = L(c);
        if (!std::isfinite(Lc.v[i])) { ++nonfinite_in; Lc.v[i] = 0; }
        if (!std::isfinite(Lp.v[i])) Lp.v[i] = 0;
        if (!std::isfinite(Lo.v[i])) { ++nonfinite_out; Lo.v[i] = 0; continue; }
        if (std::max({b[0], b[1], b[2]}) > 1.0f) ++over1;
        if (Lc.v[i] < 1.0f / 255.0f) ++black;
        sum_in += Lc.v[i]; sum_out += Lo.v[i];
        double d = 0;
        for (int k = 0; k < 3; ++k) d = std::max(d, double(std::fabs(c[k] - b[k])));
        maxd = std::max(maxd, d);
        if (d > 1.0 / 255.0) ++changed;
        if ((i & 15) == 0) lum.push_back(Lc.v[i]);
    }
    const double med = percentile(lum, 0.5), p99 = percentile(lum, 0.99);
    logl("colour in: mean luma %.4f, median %.4f, p99 %.4f, above 1.0 %.2f%%, black %.2f%%, NaN/Inf %zu",
         sum_in / double(N), med, p99, 100.0 * double(over1) / double(N), 100.0 * double(black) / double(N),
         nonfinite_in);
    const bool colour_ok = nonfinite_in == 0 && med > 0.01 && black < N * 0.9;
    logl("colour in: %s", colour_ok ? "OK" : nonfinite_in ? "BAD - NaN/Inf in the picture the game hands over"
                                        : "SUSPICIOUS - the picture is (almost) black; wrong exposure or an empty frame");

    // ---- output ----
    // Blocks of 32x32 whose bytes are the input's exactly, and blocks that went black.
    uint32_t untouched = 0, blackened = 0, blocks = 0;
    std::string where;
    for (uint32_t by = 0; by < H; by += 32)
        for (uint32_t bx = 0; bx < W; bx += 32) {
            ++blocks;
            bool same = true, dark = true;
            double in_sum = 0;
            uint32_t in_n = 0;
            for (uint32_t y = by; y < std::min(H, by + 32); ++y) {
                const size_t o = (size_t(y) * W + bx) * cb, n = size_t(std::min(W, bx + 32) - bx) * cb;
                if (same && std::memcmp(&j.cur[o], &j.out[o], n) != 0) same = false;
                for (uint32_t x = bx; x < std::min(W, bx + 32); ++x) {
                    if (Lo.v[size_t(y) * W + x] > 2.0f / 255.0f) dark = false;
                    in_sum += Lc.v[size_t(y) * W + x]; ++in_n;
                }
            }
            if (same) {
                if (untouched < 4) { char b[32]; std::snprintf(b, sizeof b, " (%u,%u)", bx, by); where += b; }
                ++untouched;
            }
            // Turned black: the input block carried a picture (mean above 6/255), the output none.
            // A block that was already near black and got a touch darker is not this.
            if (dark && in_n && in_sum / in_n > 6.0 / 255.0) ++blackened;
        }
    const Plane gi = gradient(Lc), go = gradient(Lo);
    const Quadrants oqs = quadrants(gi, go);
    const std::string oq = oqs.text();
    const bool out_aligned = oqs.centred();
    const double changed_pc = 100.0 * double(changed) / double(N);
    logl("output: mean luma %.4f (in %.4f), changed %.1f%% of pixels, max |d| %.4f, NaN/Inf %zu, 32px blocks "
         "identical to the input %u of %u%s, turned black %u; edges vs input %s",
         sum_out / double(N), sum_in / double(N), changed_pc, maxd, nonfinite_out, untouched, blocks,
         where.c_str(), blackened, oq.c_str());
    const bool out_ok = nonfinite_out == 0 && blackened == 0 && out_aligned &&
                        (changed_pc > 1.0 && (untouched == 0 || changed_pc < 50.0 || untouched * 20 < blocks));
    logl("output: %s", out_ok ? "OK - the frame handed back is the network's, aligned and complete"
                     : nonfinite_out ? "BAD - NaN/Inf in the frame handed back"
                     : blackened ? "BAD - blocks turned black"
                     : !out_aligned ? "BAD - the output is shifted against the input"
                     : changed_pc <= 1.0 ? "SUSPICIOUS - the network changed (almost) nothing; passthrough or intensity 0?"
                                         : "SUSPICIOUS - part of the frame was left untouched");

    // Guides at a lower resolution than the frame (the model after the upscaler): their render-
    // resolution jitter is a display pixel or two against the upscaled picture.
    const bool guides_low = j.mvw && W > j.mvw * 3 / 2;
    const int tol = guides_low ? 2 : 1;
    const char* jitter_note = guides_low ? " (guides at render resolution: up to 2 px of jitter allowed)" : "";

    // ---- depth ----
    const bool depth_used = j.have_depth && !j.depth.v.empty();
    Plane dgrad;
    if (depth_used) {
        size_t zero = 0, one = 0, bad = 0;
        std::vector<float> ds;
        double top = 0, bottom = 0;
        size_t nt = 0, nb = 0;
        const Plane& D = j.depth;
        for (uint32_t y = 0; y < D.h; ++y)
            for (uint32_t x = 0; x < D.w; ++x) {
                const float d = D.v[size_t(y) * D.w + x];
                if (!std::isfinite(d)) { ++bad; continue; }
                if (d <= 0.0f) ++zero;
                if (d >= 1.0f) ++one;
                if (((x ^ y) & 7) == 0) ds.push_back(d);
                if (y < D.h / 4) { top += d; ++nt; }
                if (y >= D.h - D.h / 4) { bottom += d; ++nb; }
            }
        const double n = double(D.w) * D.h;
        const double lo = percentile(ds, 0.01), hi = percentile(ds, 0.99), mid = percentile(ds, 0.5);
        top = nt ? top / double(nt) : 0; bottom = nb ? bottom / double(nb) : 0;
        logl("depth: range p1 %.6g median %.6g p99 %.6g, exactly 0: %.2f%%, >= 1: %.2f%%, NaN/Inf %zu, "
             "mean top quarter %.6g, bottom quarter %.6g",
             lo, mid, hi, 100.0 * double(zero) / n, 100.0 * double(one) / n, bad, top, bottom);
        // Which end is far. Sky and cleared background sit on the far plane: reversed-Z clears
        // to 0, standard to 1. With no sky in view, the top of a frame is usually further than
        // the bottom.
        int far_zero = 0;   // +1: 0 is far (reversed-Z), -1: 1 is far, 0: cannot tell
        const double fz = double(zero) / n, fo = double(one) / n;
        if (fz > 0.005 && fz > 4 * fo) far_zero = 1;
        else if (fo > 0.005 && fo > 4 * fz) far_zero = -1;
        else if (std::fabs(top - bottom) > 0.02 * std::max(1e-6, std::fabs(hi - lo)))
            far_zero = top < bottom ? 1 : -1;
        const char* verdict = far_zero == 0 ? "cannot tell which end is far in this frame"
                              : (far_zero > 0) == j.inverted ? "OK - agrees with DepthInverted"
                              : "BAD - DepthInverted says the opposite of what the depth looks like";
        logl("depth: far looks like %s; %s", far_zero > 0 ? "0 (reversed-Z)" : far_zero < 0 ? "1 (standard)" : "?",
             verdict);
        // Edges: the depth's against the picture's. On log(d) and log(1 - d) both, so a step
        // counts at either end of the range whichever way DepthInverted says it runs.
        Plane dz{W, H, std::vector<float>(N)}, dz1 = dz;
        for (uint32_t y = 0; y < H; ++y)
            for (uint32_t x = 0; x < W; ++x) {
                const float d = D.nearest((x + 0.5f) / float(W), (y + 0.5f) / float(H));
                dz.v[size_t(y) * W + x] = std::log(std::max(d, 1e-7f));
                dz1.v[size_t(y) * W + x] = std::log(std::max(1.0f - d, 1e-7f));
            }
        dgrad = gradient(dz);
        const Plane g1 = gradient(dz1);
        for (size_t i = 0; i < N; ++i) dgrad.v[i] += g1.v[i];
        const Quadrants dq = quadrants(dgrad, gi);
        logl("depth: edges vs the picture %s -> %s%s", dq.text().c_str(),
             dq.centred(tol) ? "OK - aligned"
                             : "SUSPICIOUS - depth does not sit on the picture (subrect, extent or a stale buffer?)",
             jitter_note);
    } else {
        logl("%s", "depth: none (the network reads the motion vector at the pixel itself)");
    }

    // ---- motion ----
    const uint32_t mb = motion_bpp(j.mvfmt);
    if (!mb || j.mv.empty()) { logl("motion: format %d not decodable here; skipped", int(j.mvfmt)); }
    Plane mx{j.mvw, j.mvh, std::vector<float>(size_t(j.mvw) * j.mvh)}, my = mx;
    size_t mv_bad = 0;
    if (mb)
        for (size_t i = 0; i < mx.v.size(); ++i) {
            float v[2];
            decode_mv(j.mvfmt, &j.mv[i * mb], v);
            if (!std::isfinite(v[0]) || !std::isfinite(v[1])) { ++mv_bad; v[0] = v[1] = 0; }
            mx.v[i] = v[0]; my.v[i] = v[1];
        }
    // Hypotheses. Each maps a pixel's uv to where it was in the previous frame.
    enum { kNone, kNet, kNative, kPlain, kNeg, kDouble, kHalf, kFlipX, kFlipY, kCount };
    static const char* hname[kCount] = {"no motion", "as the network reads it", "as the original reads it",
                                        "full-res at the pixel", "opposite sign", "double", "half",
                                        "x flipped", "y flipped"};
    double err[kCount] = {}, err_mov[kCount] = {}, err_edge[kCount] = {};
    size_t used = 0, moving = 0, edge = 0;
    std::vector<float> mags;
    // Moving samples for the scale fit: current luma, uv, the vectors as the original reads them.
    struct Sample { float c, u, v, fx, fy; };
    std::vector<Sample> fit;
    bool still = false;   // the picture barely moves: no pictures, they would show nothing
    const uint32_t stride = N > 2600000 ? 2 : 1;
    Plane warp{W, H, std::vector<float>(N, 0.0f)}, e0{W, H, std::vector<float>(N, 0.0f)},
          en{W, H, std::vector<float>(N, 0.0f)};
    const float mw = float(j.mw), mh = float(j.mh);
    // Depth edges for the silhouette score: the strongest 4% of the depth gradient.
    float dthr = INFINITY;
    if (depth_used) {
        std::vector<float> s;
        for (size_t i = 0; i < N; i += 7) s.push_back(dgrad.v[i]);
        dthr = float(percentile(s, 0.96));
        if (!(dthr > 0)) dthr = INFINITY;
    }
    if (mb) {
        for (uint32_t y = 0; y < H; y += stride)
            for (uint32_t x = 0; x < W; x += stride) {
                const float u = (x + 0.5f) / float(W), v = (y + 0.5f) / float(H);
                // The five-tap nearest-depth choice, as the pre block makes it (model texel steps).
                float mu = u, mvv = v;
                if (depth_used) {
                    float best = j.depth.nearest(u, v);
                    for (int dy = -1; dy <= 1; dy += 2)
                        for (int dx = -1; dx <= 1; dx += 2) {
                            const float au = u + dx / mw, av = v + dy / mh;
                            const float d = j.depth.nearest(au, av);
                            if (j.inverted ? d > best : d < best) { best = d; mu = au; mvv = av; }
                        }
                }
                const float fx = mx.bilinear(mu * mx.w, mvv * mx.h) * j.sx;
                const float fy = my.bilinear(mu * my.w, mvv * my.h) * j.sy;
                const float nx = j.mv_direct ? fx : j.flow_x.bilinear(mu * j.flow_x.w, mvv * j.flow_x.h) * j.sx;
                const float ny = j.mv_direct ? fy : j.flow_y.bilinear(mu * j.flow_y.w, mvv * j.flow_y.h) * j.sy;
                const float px = mx.bilinear(u * mx.w, v * mx.h) * j.sx;
                const float py = my.bilinear(u * my.w, v * my.h) * j.sy;
                const float hu[kCount][2] = {{0, 0}, {nx, ny}, {fx, fy}, {px, py}, {-fx, -fy}, {2 * fx, 2 * fy},
                                             {0.5f * fx, 0.5f * fy}, {-fx, fy}, {fx, -fy}};
                bool inside = true;
                for (int k = 0; k < kCount && inside; ++k) {
                    const float qx = (u + hu[k][0]) * W, qy = (v + hu[k][1]) * H;
                    inside = qx >= 1 && qy >= 1 && qx <= W - 1 && qy <= H - 1;
                }
                const float mag = std::hypot(fx * W, fy * H);
                if (((x ^ y) & 3) == 0) mags.push_back(mag);
                const size_t i = size_t(y) * W + x;
                const float c = Lc.v[i];
                {
                    const float w0 = Lp.bilinear((u + fx) * W, (v + fy) * H);
                    warp.v[i] = w0;
                    e0.v[i] = std::fabs(c - Lp.v[i]);
                    en.v[i] = std::fabs(c - w0);
                }
                if (!inside) continue;
                ++used;
                const bool mov = mag > 0.5f;
                const bool sil = depth_used && dgrad.v[i] >= dthr && mov;
                moving += mov; edge += sil;
                if (mov && ((x * 7 + y * 13) % 5) == 0) fit.push_back({c, u, v, fx, fy});
                for (int k = 0; k < kCount; ++k) {
                    const float e = std::fabs(c - Lp.bilinear((u + hu[k][0]) * W, (v + hu[k][1]) * H));
                    err[k] += e;
                    if (mov) err_mov[k] += e;
                    if (sil) err_edge[k] += e;
                }
            }
        for (int k = 0; k < kCount; ++k) {
            err[k] /= std::max<size_t>(1, used);
            err_mov[k] /= std::max<size_t>(1, moving);
            err_edge[k] /= std::max<size_t>(1, edge);
        }
        const double p50 = percentile(mags, 0.5), p95 = percentile(mags, 0.95), p99m = percentile(mags, 0.99);
        logl("motion: |mv| in pixels median %.2f, p95 %.2f, p99 %.2f; moving (> 0.5 px) %.1f%% of pixels; "
             "NaN/Inf %zu", p50, p95, p99m, 100.0 * double(moving) / double(std::max<size_t>(1, used)), mv_bad);
        std::string t;
        for (int k = 0; k < kCount; ++k) {
            char b[96];
            std::snprintf(b, sizeof b, "%s%s %.5f", k ? ", " : "", hname[k], moving ? err_mov[k] : err[k]);
            t += b;
        }
        logl("motion: reprojection error (mean |luma diff|, %s pixels): %s", moving ? "moving" : "all", t.c_str());
        // The fit: the factors (kx, ky) on the vectors as the original reads them that line the
        // previous frame up best. A 2-D grid over -1.5..3 in steps of 0.25 (textures repeat, so a
        // search along one axis at a time falls into the wrong valley), then 0.05 around the best.
        if (fit.size() > 100000) {
            std::vector<Sample> thin;
            const size_t step = fit.size() / 100000 + 1;
            for (size_t i = 0; i < fit.size(); i += step) thin.push_back(fit[i]);
            fit.swap(thin);
        }
        auto E2 = [&](float ax, float ay) {
            double e = 0;
            for (const Sample& q : fit) e += std::fabs(q.c - Lp.bilinear((q.u + ax * q.fx) * W, (q.v + ay * q.fy) * H));
            return fit.empty() ? 0.0 : e / double(fit.size());
        };
        float kx = 1, ky = 1;
        if (fit.size() >= 1000) {
            double best_e = E2(1, 1);
            for (int iy = -6; iy <= 12; ++iy)
                for (int ix = -6; ix <= 12; ++ix) {
                    const double e = E2(ix * 0.25f, iy * 0.25f);
                    if (e < best_e) { best_e = e; kx = ix * 0.25f; ky = iy * 0.25f; }
                }
            const float cx = kx, cy = ky;
            for (int iy = -5; iy <= 5; ++iy)
                for (int ix = -5; ix <= 5; ++ix) {
                    const float ax = cx + ix * 0.05f, ay = cy + iy * 0.05f;
                    const double e = E2(ax, ay);
                    if (e < best_e) { best_e = e; kx = ax; ky = ay; }
                }
        }
        const double e_fit = E2(kx, ky), e_one = E2(1, 1), e_zero = E2(0, 0);
        if (j.reset)
            logl("%s", "motion: this frame is a reset (camera cut?) - the comparison may not mean anything");
        still = p95 < 0.3;
        const bool tiny_motion = p95 < 2.0;   // sub-pixel jitter of the render is then as large as the motion
        auto close_to = [](float k, float want) { return std::fabs(k - want) <= 0.2f * std::fabs(want) + 0.05f; };
        const char* what = close_to(kx, 1) && close_to(ky, 1) ? "as given"
                           : close_to(kx, 2) && close_to(ky, 2) ? "twice as long as given (the vectors are too short)"
                           : close_to(kx, 0.5f) && close_to(ky, 0.5f) ? "half as long as given (the vectors are too long)"
                           : close_to(kx, -1) && close_to(ky, -1) ? "the opposite sign"
                           : close_to(kx, -1) && close_to(ky, 1) ? "x flipped"
                           : close_to(kx, 1) && close_to(ky, -1) ? "y flipped" : "none of the usual mistakes";
        logl("motion: best fit x %.2f, y %.2f (%s): error %.5f, as given %.5f, no motion %.5f (%zu moving samples)",
             kx, ky, what, e_fit, e_one, e_zero, fit.size());
        if (still) {
            logl("motion: the picture barely moves (p95 %.2f px) - cannot judge; move the camera / turn around", p95);
        } else if (fit.size() < 1000) {
            logl("motion: too few moving pixels (%zu) to judge", fit.size());
        } else if (tiny_motion) {
            logl("motion: motion under 2 px (p95 %.2f) - the render's sub-pixel jitter is as large; fit not conclusive", p95);
        } else if (close_to(kx, 1) && close_to(ky, 1) && e_one < 0.9 * e_zero) {
            logl("motion: OK - the game's vectors line the previous frame up (fit x %.2f y %.2f; error %.1f%% of no-motion's)",
                 kx, ky, 100.0 * e_one / std::max(1e-12, e_zero));
        } else if (!(close_to(kx, 1) && close_to(ky, 1))) {
            logl("motion: BAD - the frames line up best with the vectors x %.2f y %.2f (%s)", kx, ky, what);
        } else {
            logl("motion: SUSPICIOUS - the vectors barely beat no motion (%.1f%%); stale or wrong vectors?",
                 100.0 * e_one / std::max(1e-12, e_zero));
        }
        if (j.mv_direct)
            logl("%s", "motion: the network samples the game's vectors in place at full resolution");
        else if (moving)
            logl("motion: network (1/4-res field) vs full-res: moving %.5f vs %.5f (%+.1f%%), silhouettes (%zu px) "
                 "%.5f vs %.5f (%+.1f%%)", err_mov[kNet], err_mov[kNative],
                 100.0 * (err_mov[kNet] / std::max(1e-12, err_mov[kNative]) - 1.0), edge, err_edge[kNet],
                 err_edge[kNative], 100.0 * (err_edge[kNet] / std::max(1e-12, err_edge[kNative]) - 1.0));
        // Edges of the motion field against the picture.
        Plane mfield{W, H, std::vector<float>(N)};
        for (uint32_t y = 0; y < H; ++y)
            for (uint32_t x = 0; x < W; ++x) {
                const float u = (x + 0.5f) / float(W), v = (y + 0.5f) / float(H);
                mfield.v[size_t(y) * W + x] =
                    std::hypot(mx.nearest(u, v) * j.sx * W, my.nearest(u, v) * j.sy * H);
            }
        const Quadrants mq = quadrants(gradient(mfield), gi);
        logl("motion: edges vs the picture %s%s%s", mq.text().c_str(),
             p95 < 0.3 ? " (little motion: not meaningful)"
                       : mq.centred(tol) ? " -> OK" : " -> SUSPICIOUS (subrect/extent?)", jitter_note);
    }

    // ---- pictures ----
    // Pictures for the first few captures in which something moves (a menu or a loading screen
    // shows nothing), process-wide.
    if (!j.dump || j.folder.empty() || still) return;
    if (g_dumps_left.fetch_sub(1) <= 0) return;
    char base[512];
    std::snprintf(base, sizeof base, "%s/check%02d-%ux%u-f%llx-", j.folder.c_str(), id, W, H,
                  (unsigned long long)j.feature);
    std::vector<uint8_t> rgb(N * 3);
    // Above 2560 wide every other pixel: a 4K set is otherwise 225 MB, and half size shows enough.
    const uint32_t ps = W > 2560 ? 2 : 1, pw = (W + ps - 1) / ps, ph = (H + ps - 1) / ps;
    auto save = [&](const char* name) {
        if (ps == 1) { write_png(std::string(base) + name, W, H, rgb); return; }
        std::vector<uint8_t> tiny_motion(size_t(pw) * ph * 3);
        for (uint32_t y = 0; y < ph; ++y)
            for (uint32_t x = 0; x < pw; ++x)
                std::memcpy(&tiny_motion[(size_t(y) * pw + x) * 3], &rgb[(size_t(y * ps) * W + x * ps) * 3], 3);
        write_png(std::string(base) + name, pw, ph, tiny_motion);
    };
    auto colour_png = [&](const std::vector<uint8_t>& src, const char* name) {
        for (size_t i = 0; i < N; ++i) {
            float c[3];
            decode_rgb(j.cfmt, &src[i * cb], c);
            for (int k = 0; k < 3; ++k) rgb[i * 3 + k] = to8(c[k]);
        }
        save(name);
    };
    colour_png(j.prev, "1-colour-prev.png");
    colour_png(j.cur, "2-colour.png");
    colour_png(j.out, "3-output.png");
    auto grey_png = [&](const Plane& p, float gain, const char* name) {
        for (size_t i = 0; i < N; ++i) rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = to8(p.v[i] * gain);
        save(name);
    };
    if (mb) {
        grey_png(warp, 1.0f, "4-prev-warped-by-mv.png");
        grey_png(e0, 4.0f, "5-diff-no-motion.png");
        grey_png(en, 4.0f, "6-diff-with-mv.png");
        auto flow_png = [&](const Plane& fx, const Plane& fy, float scale_x, float scale_y, const char* name) {
            std::vector<float> m;
            for (size_t i = 0; i < N; i += 13) {
                const uint32_t x = uint32_t(i % W), y = uint32_t(i / W);
                const float u = (x + 0.5f) / W, v = (y + 0.5f) / H;
                m.push_back(std::hypot(fx.bilinear(u * fx.w, v * fx.h) * scale_x * W,
                                       fy.bilinear(u * fy.w, v * fy.h) * scale_y * H));
            }
            const float top = std::max(1.0f, float(percentile(m, 0.99)));
            for (uint32_t y = 0; y < H; ++y)
                for (uint32_t x = 0; x < W; ++x) {
                    const float u = (x + 0.5f) / W, v = (y + 0.5f) / H;
                    const float vx = fx.bilinear(u * fx.w, v * fx.h) * scale_x * W;
                    const float vy = fy.bilinear(u * fy.w, v * fy.h) * scale_y * H;
                    const float val = std::min(1.0f, std::hypot(vx, vy) / top);
                    const float hue = (std::atan2(vy, vx) / 6.2831853f + 0.5f) * 6.0f;
                    const float f = hue - std::floor(hue);
                    const int sector = int(std::floor(hue)) % 6;
                    float r = 0, g = 0, b = 0;
                    switch (sector) {
                        case 0: r = 1; g = f; break;
                        case 1: r = 1 - f; g = 1; break;
                        case 2: g = 1; b = f; break;
                        case 3: g = 1 - f; b = 1; break;
                        case 4: r = f; b = 1; break;
                        default: r = 1; b = 1 - f; break;
                    }
                    const size_t i = (size_t(y) * W + x) * 3;
                    rgb[i] = to8(r * val); rgb[i + 1] = to8(g * val); rgb[i + 2] = to8(b * val);
                }
            save(name);
        };
        flow_png(mx, my, j.sx, j.sy, "7-mv-game.png");
        if (!j.mv_direct) flow_png(j.flow_x, j.flow_y, j.sx, j.sy, "8-mv-network.png");
    }
    if (depth_used) {
        std::vector<float> s;
        for (size_t i = 0; i < j.depth.v.size(); i += 7) {
            const float d = j.depth.v[i];
            if (std::isfinite(d)) s.push_back(j.inverted ? d : 1.0f - d);
        }
        const float lo = float(percentile(s, 0.01)), hi = float(percentile(s, 0.99));
        Plane vis{W, H, std::vector<float>(N)};
        for (uint32_t y = 0; y < H; ++y)
            for (uint32_t x = 0; x < W; ++x) {
                float d = j.depth.nearest((x + 0.5f) / W, (y + 0.5f) / H);
                d = j.inverted ? d : 1.0f - d;   // near = bright
                vis.v[size_t(y) * W + x] = hi > lo ? (d - lo) / (hi - lo) : 0.5f;
            }
        grey_png(vis, 1.0f, "9-depth-near-bright.png");
    }
    nr::logf("[nr] input check %d: pictures written as %s*.png", id, base);
}

// ---- the recording side --------------------------------------------------------------

struct State {
    nrvk::Buffer buf{};
    bool coherent{};
    // Slot offsets and sizes in `buf`.
    VkDeviceSize prev_at{}, cur_at{}, out_at{}, mv_at{}, flow_at{}, depth_at{}, mark_at{}, bytes{};
    // Geometry the slots were laid out for.
    uint32_t W{}, H{}, mvw{}, mvh{}, fw{}, fh{}, dw{}, dh{};
    VkFormat cfmt{}, mvfmt{};
    enum Phase { Idle, HaveA, Waiting } phase{Idle};
    uint64_t feature{}, a_at{}, b_at{}, next_at{180};
    uint32_t between{};
    bool b_now{};                 // begin() took this recording as frame B
    uint32_t seq{}, taken{};
    std::unique_ptr<Job> job;     // frame B's metadata, until the copy lands
    std::shared_ptr<std::atomic<bool>> busy = std::make_shared<std::atomic<bool>>(false);

    static constexpr uint64_t kFirst = 180, kEvery = 600, kGiveUp = 600;
    static constexpr uint32_t kMax = 40;

    void destroy(nrvk::Context& ctx) {
        if (buf.handle) ctx.destroy(buf);
    }

    bool allocate(nrvk::Context& ctx, VkDeviceSize n) {
        if (buf.handle && buf.bytes >= n) return true;
        if (buf.handle) ctx.destroy(buf);
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = n;
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vkCreateBuffer(ctx.device, &info, nullptr, &buf.handle) != VK_SUCCESS) { buf = {}; return false; }
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(ctx.device, buf.handle, &req);
        // Cached host memory: the CPU reads ~100 MB of it, and uncached or write-combined memory
        // reads at a tiny_motion fraction of that speed.
        const VkMemoryPropertyFlags prefs[3] = {
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
        uint32_t type = ~0u;
        for (VkMemoryPropertyFlags want : prefs) {
            for (uint32_t i = 0; i < ctx.mem.memoryTypeCount && type == ~0u; ++i)
                if ((req.memoryTypeBits & (1u << i)) && (ctx.mem.memoryTypes[i].propertyFlags & want) == want)
                    type = i;
            if (type != ~0u) break;
        }
        if (type == ~0u) { vkDestroyBuffer(ctx.device, buf.handle, nullptr); buf = {}; return false; }
        coherent = (ctx.mem.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = req.size;
        alloc.memoryTypeIndex = type;
        if (vkAllocateMemory(ctx.device, &alloc, nullptr, &buf.memory) != VK_SUCCESS ||
            vkBindBufferMemory(ctx.device, buf.handle, buf.memory, 0) != VK_SUCCESS ||
            vkMapMemory(ctx.device, buf.memory, 0, VK_WHOLE_SIZE, 0, &buf.mapped) != VK_SUCCESS) {
            if (buf.memory) vkFreeMemory(ctx.device, buf.memory, nullptr);
            vkDestroyBuffer(ctx.device, buf.handle, nullptr);
            buf = {};
            return false;
        }
        buf.bytes = n;
        return true;
    }

    void invalidate(nrvk::Context& ctx) {
        if (coherent || !buf.memory) return;
        VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        r.memory = buf.memory; r.offset = 0; r.size = VK_WHOLE_SIZE;
        vkInvalidateMappedMemoryRanges(ctx.device, 1, &r);
    }

    // One image region into the buffer, the image left in the layout it came in.
    void copy(VkCommandBuffer cmd, VkImage image, VkImageLayout layout, uint32_t w, uint32_t h, VkDeviceSize at,
              uint32_t ox = 0, uint32_t oy = 0) {
        const bool in_place = layout == VK_IMAGE_LAYOUT_GENERAL || layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        const VkImageLayout use = in_place ? layout : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        auto bar = [&](VkImageLayout from, VkImageLayout to, VkPipelineStageFlags ss, VkAccessFlags sa,
                       VkPipelineStageFlags ds, VkAccessFlags da) {
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.srcAccessMask = sa; b.dstAccessMask = da;
            b.oldLayout = from; b.newLayout = to;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = image; b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(cmd, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
        };
        bar(layout, use, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy c{};
        c.bufferOffset = at;
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageOffset = {int32_t(ox), int32_t(oy), 0};
        c.imageExtent = {w, h, 1};
        vkCmdCopyImageToBuffer(cmd, image, use, buf.handle, 1, &c);
        bar(use, layout, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    }

    static VkDeviceSize up(VkDeviceSize v) { return (v + 255) & ~VkDeviceSize(255); }

    // Before the network touches anything: frame A's or frame B's colour. `recording` is the
    // runtime's recording counter.
    void begin(nrvk::Context& ctx, VkCommandBuffer cmd, const ColourFrame& frame, const EngineFrame& engine,
               uint64_t feature_id, uint64_t recording, uint32_t fw_, uint32_t fh_, uint32_t mw, uint32_t mh) {
        b_now = false;
        if (!g_on.load(std::memory_order_relaxed)) return;
        poll(ctx, recording);
        if (phase == Waiting || busy->load() || taken >= kMax) return;
        const uint32_t cb = colour_bpp(frame.format);
        auto readable = [](VkImageUsageFlags u) { return !u || (u & VK_IMAGE_USAGE_TRANSFER_SRC_BIT); };
        if (phase == HaveA) {
            if (feature_id != feature) {
                if (++between > 16) phase = Idle;   // that feature went away
                return;
            }
            if (frame.width != W || frame.height != H || frame.format != cfmt ||
                engine.motion.width != mvw || engine.motion.height != mvh || engine.motion.format != mvfmt) {
                phase = Idle;
                return;
            }
            ctx_copy_colour(cmd, frame, cur_at);
            b_now = true;
            b_at = recording;
            return;
        }
        if (recording < next_at || !cb || !engine.motion.image || !readable(frame.usage) ||
            !readable(engine.motion.usage) || !motion_bpp(engine.motion.format))
            return;
        // Lay the slots out for this geometry.
        W = frame.width; H = frame.height; cfmt = frame.format;
        mvw = engine.motion.width; mvh = engine.motion.height; mvfmt = engine.motion.format;
        fw = fw_; fh = fh_; dw = mw; dh = mh;
        const VkDeviceSize colour = up(VkDeviceSize(W) * H * cb);
        prev_at = 0; cur_at = colour; out_at = 2 * colour;
        mv_at = 3 * colour;
        flow_at = mv_at + up(VkDeviceSize(mvw) * mvh * motion_bpp(mvfmt));
        depth_at = flow_at + up(VkDeviceSize(fw) * fh * 8);
        mark_at = depth_at + up(VkDeviceSize(dw) * dh * 4);
        bytes = mark_at + 256;
        if (!allocate(ctx, bytes)) {
            nr::logf("[nr] input check: no host-visible buffer of %llu bytes; off", (unsigned long long)bytes);
            g_on = false;
            return;
        }
        ctx_copy_colour(cmd, frame, prev_at);
        phase = HaveA;
        feature = feature_id;
        a_at = recording;
        between = 0;
    }

    void ctx_copy_colour(VkCommandBuffer cmd, const ColourFrame& frame, VkDeviceSize at) {
        copy(cmd, frame.image, frame.before, W, H, at);
    }

    // After the network wrote the frame back (frame B only).
    void end(nrvk::Context& ctx, VkCommandBuffer cmd, const ColourFrame& frame, const EngineFrame& engine,
             const nrvk::Context::Image& flow, const nrvk::Context::Image& depth_img, bool depth_in_place,
             bool have_depth, bool gate, uint32_t mw, uint32_t mh, bool mv_direct) {
        (void)ctx;
        if (!b_now) return;
        b_now = false;
        copy(cmd, frame.image, frame.after, W, H, out_at);
        copy(cmd, engine.motion.image, engine.motion.after, mvw, mvh, mv_at, engine.motion_x, engine.motion_y);
        if (!mv_direct) copy(cmd, flow.handle, VK_IMAGE_LAYOUT_GENERAL, fw, fh, flow_at);
        const bool depth_ok = have_depth && (depth_in_place ? engine.depth.width == dw && engine.depth.height == dh
                                                            : depth_img.handle != VK_NULL_HANDLE);
        if (depth_ok) {
            if (depth_in_place) copy(cmd, engine.depth.image, engine.depth.after, dw, dh, depth_at);
            else copy(cmd, depth_img.handle, VK_IMAGE_LAYOUT_GENERAL, dw, dh, depth_at);
        }
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
        ++seq;
        vkCmdFillBuffer(cmd, buf.handle, mark_at, 4, seq);
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
        job = std::make_unique<Job>();
        Job& j = *job;
        j.feature = feature; j.recording = b_at;
        j.W = W; j.H = H; j.cfmt = cfmt;
        j.mvw = mvw; j.mvh = mvh; j.mvfmt = mvfmt;
        j.flow_x.w = j.flow_y.w = fw; j.flow_x.h = j.flow_y.h = fh;
        j.have_depth = depth_ok; j.inverted = engine.depth_inverted; j.depth_in_place = depth_in_place;
        j.depth.w = dw; j.depth.h = dh;
        j.mw = mw; j.mh = mh;
        j.sx = engine.motion_scale_x; j.sy = engine.motion_scale_y;
        j.gate = gate; j.reset = engine.reset;
        j.mv_direct = mv_direct;
        j.between = between;
        phase = Waiting;
    }

    // On every recording while waiting: has the marker landed?
    void poll(nrvk::Context& ctx, uint64_t recording) {
        if (phase != Waiting || !buf.mapped) return;
        invalidate(ctx);
        const uint8_t* m = static_cast<const uint8_t*>(buf.mapped);
        uint32_t mark = 0;
        std::memcpy(&mark, m + mark_at, 4);
        if (mark != seq) {
            if (recording > b_at + kGiveUp) {
                nr::logf("[nr] input check: the copy never landed after %llu recordings; giving up on this one",
                         (unsigned long long)(recording - b_at));
                phase = Idle;
                next_at = recording + kEvery;
                job.reset();
            }
            return;
        }
        Job& j = *job;
        const uint32_t cb = colour_bpp(cfmt);
        const size_t cn = size_t(W) * H * cb;
        j.prev.assign(m + prev_at, m + prev_at + cn);
        j.cur.assign(m + cur_at, m + cur_at + cn);
        j.out.assign(m + out_at, m + out_at + cn);
        j.mv.assign(m + mv_at, m + mv_at + size_t(mvw) * mvh * motion_bpp(mvfmt));
        if (!j.mv_direct) {
            const float* f = reinterpret_cast<const float*>(m + flow_at);
            j.flow_x.v.resize(size_t(fw) * fh); j.flow_y.v.resize(size_t(fw) * fh);
            for (size_t i = 0; i < size_t(fw) * fh; ++i) { j.flow_x.v[i] = f[2 * i]; j.flow_y.v[i] = f[2 * i + 1]; }
        }
        if (j.have_depth) {
            const float* d = reinterpret_cast<const float*>(m + depth_at);
            j.depth.v.assign(d, d + size_t(dw) * dh);
        }
        j.serial = ++g_serial;
        j.dump = g_dumps_left.load() > 0;
        j.folder = g_folder;
        ++taken;
        phase = Idle;
        next_at = recording + kEvery;
        busy->store(true);
        auto flag = busy;
        std::thread([flag, owned = std::move(job)]() mutable {
            try { analyse(*owned); } catch (...) { nr::logf("[nr] input check: the analysis threw"); }
            flag->store(false);
        }).detach();
    }
};

}  // namespace incheck
}  // namespace nr
