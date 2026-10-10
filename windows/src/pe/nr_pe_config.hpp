#pragma once
#include "nr_runtime.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

namespace nr::pe {

// Passes, as OptiScaler DLSS-NR: 1..2, or 1..10 with UnlockPasses.
inline constexpr int kMaxPasses = 10;

// A later pass's own settings. Each one that is absent inherits pass 1, except
// LocalTone, which defaults to 0 (OptiScaler DLSS-NR's PassTuning).
struct PassOverride {
    std::optional<int> style;
    std::optional<float> intensity, local_structure, local_tone, skin_structure;
    std::optional<bool> automatic_mask;
};

// [Preprocess] in dlssnr-amd.ini, read by every route: nr::Preprocess and the
// hotkey that flips Enabled for this run. See runtime_prep.comp.
struct PreprocessConfig {
    Preprocess values{};
    std::string hotkey = "Ctrl+F10";   // empty: none
    bool sound = true;                 // a cue when it switches on or off
};
// One key of the section into `p`; false when it is not one of its keys.
bool parse_preprocess_key(PreprocessConfig& p, const std::string& key, const std::string& value);
// The section, with its comments, as the file holds it.
void write_preprocess(FILE* f, const PreprocessConfig& p);
// [Log], with its comments (read by nr_pe_log.cpp; every route's file carries it).
void write_log(FILE* f, bool enabled, bool clear);
// [Network], with its comments: ACO Mode, the network's machine code (nr_pipeline_binary.hpp). Read once per
// process by configure_network; every route's file carries it.
void write_network(FILE* f, bool aco);
// [Network] ACO Mode of dlssnr-amd.ini in `folder`, handed to nr::binary before the first device is made: on
// (and the bundle dlssnr-amd/aco present) the network runs ACO's machine code. Logs what it decided.
void configure_network(const std::string& folder);
// "exposure auto, ExposureBias +0.00 EV, curve none, contrast 1.00, saturation 1.00", for the log.
std::string describe(const Preprocess& p);

// The OptiScaler route's file: the [Preprocess] section alone (a ReShade
// add-on's [DlssNr] in the same file is left to the add-on). Written with the
// defaults, Enabled = 0, when there is no file; re-read when it changes.
class PreprocessFile {
  public:
    // False when the file did not change since the last call.
    bool poll(const std::string& path, PreprocessConfig& out);
  private:
    uint64_t stamp_ = 0;
    bool tried_ = false;
};

// Enabled as a frame sees it: the file's, until the hotkey flips it for this
// run (never written back); a change of Enabled in the file wins again.
class PreprocessSwitch {
  public:
    Preprocess frame(const PreprocessConfig& file);
    bool on() const { return last_on_; }   // what the last frame ran with
    // Seconds since it last switched on or off (hotkey or file); large before the first switch.
    double since_switch() const;
  private:
    std::string key_text_;
    int vk_ = 0, mods_ = 0;
    bool down_ = false, have_override_ = false, override_ = false, file_enabled_ = false;
    bool logged_ = false, last_on_ = false;
    uint64_t switched_ms_ = 0;   // GetTickCount64 at the last switch, 0 before one
    Preprocess last_{};
};

// The ReShade add-on's settings, dlssnr-amd.ini next to it. Section [DlssNr]
// with OptiScaler DLSS-NR's key names (ranges as the file's comments say), plus
// ClassicScaler, History and WhitePoint, which are this project's.
struct Config {
    Controls controls{};          // pass 1, Passes, Enabled, ApplyModel, the apply-edit controls
    bool unlock_passes = false;
    std::array<PassOverride, kMaxPasses - 1> pass{};   // pass[0] is pass 2
    float model_scale = 1.0f;     // WorkingScale, 0.25..1
    float history = 1.0f;         // previous-frame blend in the post block, 0..1
    float white_point = 1.0f;     // linear-light input only
    PreprocessConfig preprocess{};   // [Preprocess]; controls.preprocess is the frame's, see PreprocessSwitch
    bool log_enabled = true;         // [Log] Enabled, read by the log itself at start (nr_pe_log.cpp)
    bool log_clear = true;           // [Log] ClearOnStart
    bool aco = false;                // [Network] ACO Mode, read by configure_network at start (kept on save)

    int pass_limit() const { return unlock_passes ? kMaxPasses : 2; }
    // Fill controls.per_pass from `pass`; call after any change.
    void resolve();

    // Read the file, creating it with defaults when absent and rewriting it in
    // the current format when it still holds the old keys.
    void load(const std::string& path);
    void save(const std::string& path);
    // Re-read if the file changed since the last load/save/reload.
    bool reload(const std::string& path);

  private:
    bool legacy_ = false;
    uint64_t stamp_ = 0;
};

}  // namespace nr::pe
