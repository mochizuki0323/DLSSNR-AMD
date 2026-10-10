#pragma once
// Where the runtime's own diagnostics go.
//
// The runtime is a library with two very different callers. A test on the work
// box has a terminal, so `printf` is exactly right. A game has none: the module
// is a DLL loaded by an executable with no console, and every `printf` in
// windows/src/core/nr_graph.cpp has therefore been invisible in a game since the day
// the adapter existed - including the numbers that say where a twelve-second
// build spends its time, which is the one place they are needed.
//
// So the runtime writes through a sink the host may replace. windows/src/pe sets it to
// nr::pe::log, which appends to dlssnr-amd.log next to the module; anything that
// does not set it keeps the terminal behaviour unchanged.
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace nr {

using LogSink = void (*)(const char*);

// Not a static local: this header is included by two translation units that are
// linked into one binary (nr_runtime.cpp, which includes nr_graph.cpp, and the
// standalone nr_graph tool), and an inline variable is one object in both.
inline LogSink g_log_sink = nullptr;

inline void set_log_sink(LogSink sink) { g_log_sink = sink; }

// What the build running now is doing, for a host's UI (builds run one at a time). Stage: 0 none, 1 planning,
// 2 reading the weights, 3 uploading them, 4 creating pipelines (pipes_done of pipes_total), 5 finishing.
inline std::atomic<int> g_build_stage{0};
inline std::atomic<uint32_t> g_build_pipes_done{0}, g_build_pipes_total{0};

// A line about the host's memory (a 32-bit game's free address space), logged at a build's phases; null: none.
using MemoryNote = std::string (*)();
inline MemoryNote g_memory_note = nullptr;

// One line, no trailing newline in `format`; the sink adds it.
inline void logf(const char* format, ...) {
    char line[1024];
    va_list args;
    va_start(args, format);
    std::vsnprintf(line, sizeof line, format, args);
    va_end(args);
    if (g_log_sink) {
        g_log_sink(line);
    } else {
        std::fputs(line, stdout);
        std::fputc('\n', stdout);
        std::fflush(stdout);
    }
}

// Coarse phase timing for a build, logged once per build rather than sampled.
//
// The whole point of the number is attribution: "the network took 12.5 s" is
// not actionable and "weights 4.1 s, upload 3.6 s, pipelines 2.9 s" is. Each
// `mark` closes the phase that was running since the last one.
struct PhaseTimer {
    using Clock = std::chrono::steady_clock;
    Clock::time_point start{Clock::now()}, last{Clock::now()};
    std::vector<std::pair<std::string, double>> phases;

    void mark(const char* name) {
        const auto now = Clock::now();
        phases.emplace_back(name, std::chrono::duration<double>(now - last).count());
        last = now;
    }
    double total() const { return std::chrono::duration<double>(Clock::now() - start).count(); }
    std::string text() const {
        std::string out;
        char one[96];
        for (const auto& p : phases) {
            std::snprintf(one, sizeof one, "%s %.2fs  ", p.first.c_str(), p.second);
            out += one;
        }
        return out;
    }
};

}  // namespace nr
