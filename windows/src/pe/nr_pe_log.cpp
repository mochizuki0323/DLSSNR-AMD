#include "nr_pe_log.hpp"

#include <windows.h>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>

namespace nr::pe {
namespace {

std::mutex log_lock;
HMODULE self_module = nullptr;

// Where this module is, by whichever route works. The loader's own handle first;
// deriving one from an address inside us second, because that is what the code
// used to do and it is right when it works; the executable's own folder last,
// which is where a drop-in module was copied to anyway.
std::string folder_of(HMODULE module) {
    char path[MAX_PATH]{};
    if (!GetModuleFileNameA(module, path, MAX_PATH)) return {};
    std::string s = path;
    const auto slash = s.find_last_of("\\/");
    return slash == std::string::npos ? std::string{} : s.substr(0, slash);
}

std::string resolve_folder() {
    HMODULE self = self_module;
    if (!self)
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(&resolve_folder), &self);
    if (self) {
        std::string folder = folder_of(self);
        if (!folder.empty()) return folder;
    }
    // Last resort: the executable's own directory, which is where a drop-in
    // module was copied to in the first place.
    return folder_of(nullptr);
}

std::string resolve_path() {
    const std::string folder = resolve_folder();
    return folder.empty() ? std::string("dlssnr-amd.log")
                          : folder + "\\dlssnr-amd.log";
}

// [Log] in dlssnr-amd.ini beside this module, read once, before the first line: Enabled (0: no log
// at all) and ClearOnStart (1: the first module of this process to log starts the file afresh).
// Absent file, section or key: both on.
struct LogSettings { bool enabled = true, clear = true; };
LogSettings read_settings() {
    LogSettings s;
    const std::string folder = resolve_folder();
    FILE* f = std::fopen((folder.empty() ? std::string("dlssnr-amd.ini") : folder + "\\dlssnr-amd.ini").c_str(), "r");
    if (!f) return s;
    auto trim = [](std::string t) {
        const auto b = t.find_first_not_of(" \t\r\n"), e = t.find_last_not_of(" \t\r\n");
        return b == std::string::npos ? std::string{} : t.substr(b, e - b + 1);
    };
    auto lower = [](std::string t) { for (char& c : t) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a'); return t; };
    bool in_log = false;
    char line[512];
    while (std::fgets(line, sizeof line, f)) {
        std::string t = line;
        const auto comment = t.find_first_of(";#");
        if (comment != std::string::npos) t = t.substr(0, comment);
        t = trim(t);
        if (t.empty()) continue;
        if (t.front() == '[') { in_log = lower(t) == "[log]"; continue; }
        const auto eq = t.find('=');
        if (!in_log || eq == std::string::npos) continue;
        const std::string key = lower(trim(t.substr(0, eq))), v = lower(trim(t.substr(eq + 1)));
        const bool on = v == "1" || v == "true" || v == "on" || v == "yes";
        const bool off = v == "0" || v == "false" || v == "off" || v == "no";
        if (!on && !off) continue;
        if (key == "enabled") s.enabled = on;
        else if (key == "clearonstart") s.clear = on;
    }
    std::fclose(f);
    return s;
}

FILE* open_log() {
    const LogSettings s = read_settings();
    if (!s.enabled) return nullptr;
    // Several of our modules can live in one process (the NGX forwarder and the core); only the
    // first to log may start the file afresh, or it would erase what the other already wrote.
    char seen[4];
    const bool first = GetEnvironmentVariableA("NR_LOG_OPENED", seen, sizeof seen) == 0;
    if (first) SetEnvironmentVariableA("NR_LOG_OPENED", "1");
    return std::fopen(resolve_path().c_str(), s.clear && first ? "w" : "a");
}

FILE* file() {
    static FILE* handle = open_log();
    return handle;
}

}  // namespace

void set_module(void* module) { self_module = static_cast<HMODULE>(module); }

const char* module_folder() {
    static std::string folder = resolve_folder();
    return folder.c_str();
}

const char* log_path() {
    static std::string path = resolve_path();
    return path.c_str();
}

void log(const char* format, ...) {
    std::lock_guard<std::mutex> guard(log_lock);
    FILE* handle = file();
    if (!handle) return;
    va_list args; va_start(args, format);
    std::vfprintf(handle, format, args);
    va_end(args);
    std::fputc('\n', handle);
    std::fflush(handle);
}

}  // namespace nr::pe
