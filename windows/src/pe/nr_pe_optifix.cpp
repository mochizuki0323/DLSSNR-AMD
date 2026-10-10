#include "nr_pe_optifix.hpp"

#include "nr_pe_log.hpp"

#include <windows.h>

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <string>
#include <vector>

#include <d3d12.h>

#include "MinHook.h"

namespace nr::pe {

#if defined(_WIN64)
namespace {

struct Module {
    const wchar_t* name;
    uint8_t* base;
    size_t image;
};

// OptiScaler is dxgi.dll in both packaged routes; OptiScaler.dll covers a manual install.
bool find_module(Module* out) {
    static const wchar_t* const kModules[] = {L"dxgi.dll", L"OptiScaler.dll"};
    for (const wchar_t* name : kModules) {
        auto* base = reinterpret_cast<uint8_t*>(GetModuleHandleW(name));
        if (!base) continue;
        auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
        if (dos->e_magic != IMAGE_DOS_SIGNATURE) continue;
        auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
        if (nt->Signature != IMAGE_NT_SIGNATURE) continue;
        *out = {name, base, nt->OptionalHeader.SizeOfImage};
        return true;
    }
    return false;
}

bool matches(const uint8_t* p, const uint8_t* pat, const char* mask, size_t n) {
    for (size_t k = 0; k < n; ++k)
        if (mask[k] != '?' && p[k] != pat[k]) return false;
    return true;
}

// Every executable-section position where `pat` matches and `accept` agrees. Only a unique hit is
// ever patched.
template <class Accept>
int find_unique(const Module& m, const uint8_t* pat, const char* mask, size_t n, Accept accept, uint8_t** hit) {
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(m.base + reinterpret_cast<IMAGE_DOS_HEADER*>(m.base)->e_lfanew);
    IMAGE_SECTION_HEADER* sec = IMAGE_FIRST_SECTION(nt);
    int hits = 0;
    for (unsigned s = 0; s < nt->FileHeader.NumberOfSections; ++s, ++sec) {
        if (!(sec->Characteristics & IMAGE_SCN_MEM_EXECUTE)) continue;
        uint8_t* p = m.base + sec->VirtualAddress;
        const size_t len = sec->Misc.VirtualSize;
        for (size_t i = 0; i + n <= len; ++i)
            if (matches(p + i, pat, mask, n) && accept(p + i)) { *hit = p + i; ++hits; }
    }
    return hits;
}

uint8_t* rel32_target(uint8_t* insn_end_minus4) {
    int32_t d;
    std::memcpy(&d, insn_end_minus4, 4);
    return insn_end_minus4 + 4 + d;
}

bool write_code(uint8_t* at, const void* bytes, size_t n) {
    DWORD old = 0;
    if (!VirtualProtect(at, n, PAGE_EXECUTE_READWRITE, &old)) return false;
    std::memcpy(at, bytes, n);
    VirtualProtect(at, n, old, &old);
    FlushInstructionCache(GetCurrentProcess(), at, n);
    return true;
}

// A signature written as hex bytes, "??" for any byte.
struct Sig {
    std::vector<uint8_t> bytes;
    std::string mask;
    explicit Sig(const char* text) {
        for (const char* p = text; *p;) {
            if (*p == ' ') { ++p; continue; }
            if (p[0] == '?') { bytes.push_back(0); mask += '?'; p += 2; continue; }
            bytes.push_back(static_cast<uint8_t>(std::strtoul(std::string(p, 2).c_str(), nullptr, 16)));
            mask += 'x';
            p += 2;
        }
    }
};

template <class Accept>
int find_unique(const Module& m, const Sig& sig, Accept accept, uint8_t** hit) {
    return find_unique(m, sig.bytes.data(), sig.mask.c_str(), sig.bytes.size(), accept, hit);
}

// Code bytes in fresh executable memory; null if it could not be allocated.
uint8_t* place_code(const uint8_t* code, size_t n) {
    auto* p = static_cast<uint8_t*>(VirtualAlloc(nullptr, n, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
    if (!p) return nullptr;
    std::memcpy(p, code, n);
    DWORD old = 0;
    VirtualProtect(p, n, PAGE_EXECUTE_READ, &old);
    FlushInstructionCache(GetCurrentProcess(), p, n);
    return p;
}

// ---- 1. The device-extension query (see the header) ----------------------------------------------
void fix_extension_query(const Module& m) {
    // SupportedDeviceExtensions, OptiScaler-NR v0.8.4 (8802b2b4), MSVC x64:
    //   test rdx,rdx / je        getInstanceProcAddr == nullptr
    //   test r9,r9   / je        physicalDevice == VK_NULL_HANDLE
    //   lea rdx,[rip+name]       "vkEnumerateDeviceExtensionProperties"
    //   mov rcx,r8               instance            <- becomes xor ecx,ecx; nop
    //   call rax / mov rbp,rax / test rax,rax
    static const uint8_t pat[] = {
        0x48, 0x85, 0xD2, 0x0F, 0x84, 0, 0, 0, 0,
        0x4D, 0x85, 0xC9, 0x0F, 0x84, 0, 0, 0, 0,
        0x48, 0x8D, 0x15, 0, 0, 0, 0,
        0x49, 0x8B, 0xC8, 0xFF, 0xD0, 0x48, 0x8B, 0xE8, 0x48, 0x85, 0xC0};
    static const char mask[] = "xxxxx????xxxxx????xxx????xxxxxxxxxxx";
    static_assert(sizeof(pat) == sizeof(mask) - 1, "pattern and mask lengths");
    constexpr size_t kLea = 18, kMov = 25;
    static const char kName[] = "vkEnumerateDeviceExtensionProperties";

    uint8_t* hit = nullptr;
    const int hits = find_unique(m, pat, mask, sizeof(pat), [&](uint8_t* p) {
        const uint8_t* str = rel32_target(p + kLea + 3);
        return str >= m.base && str + sizeof(kName) <= m.base + m.image &&
               std::memcmp(str, kName, sizeof(kName)) == 0;
    }, &hit);
    if (hits != 1) {
        log("[nr] OptiScaler fix (device extensions): %d matching sites in %ls, nothing changed", hits, m.name);
        return;
    }
    static const uint8_t fix[] = {0x33, 0xC9, 0x90};   // xor ecx,ecx ; nop
    if (!write_code(hit + kMov, fix, sizeof(fix))) {
        log("[nr] OptiScaler fix (device extensions): VirtualProtect failed (%lu)", GetLastError());
        return;
    }
    log("[nr] OptiScaler fix: device-extension query no longer uses the stored VkInstance (%ls+0x%llx)",
        m.name, static_cast<unsigned long long>(hit + kMov - m.base));
}

// ---- 2. Finished Picture under DXVK / vkd3d-proton ------------------------------------------------
//
// wrapped_swapchain.cpp LocalPresent returns early on DXVK ("DXVK check, it's here because of
// upscaler time calculations"): it calls the real Present and returns before the native path's
//
//     if (cq && (fg == nullptr || !fg->IsActive() || fg->IsPaused()))
//         DlssNr::ApplyToFinishedPicture(pSwapChain, cq);
//
// so under Proton - every OptiScaler install of this project - [DlssNr] FinishedPicture never runs:
// the slots armed at the upscaler are never consumed and the option sits at "Waiting for the previous
// picture to finish.", with NR off. The fix gives the DXVK path the same call, under the same
// condition, before its Present, and changes nothing else (no D3D overlay, no GPU-time reads, the
// frame counter where it was).
//
// v0.8.4 (8802b2b4), register allocation of LocalPresent at that point: r15 = the swapchain,
// r14d/r13d = SyncInterval/Flags, r12b = willPresent, rdi = State::currentFG, [rbp+0x80] = cq.
// Both the DXVK branch and the native block are matched byte for byte (the native block is where
// ApplyToFinishedPicture's address and the FG field offsets come from), so any other build is left
// alone. v0.8.91 has its own layout below (finished_picture_0891). Each returns its number of matching
// sites; the fix is applied only at exactly one.
int finished_picture_084(const Module& m) {
    // test bl,bl (usesDxvk) / je native / the DXVK branch's first five instructions, then its
    // Present call: mov rax,[r15]; mov r9,[rbp-0x50]; mov r8d,r13d; mov edx,r14d; mov rcx,r15;
    // test r9,r9; jne +5; call [rax+0x40]
    static const uint8_t branch[] = {
        0x84, 0xDB, 0x0F, 0x84, 0, 0, 0, 0,
        0x49, 0x8B, 0x07, 0x4C, 0x8B, 0x4D, 0xB0, 0x45, 0x8B, 0xC5, 0x41, 0x8B, 0xD6, 0x49, 0x8B, 0xCF,
        0x4D, 0x85, 0xC9, 0x75, 0x05, 0xFF, 0x50, 0x40};
    static const char branch_mask[] = "xxxx????xxxxxxxxxxxxxxxxxxxxxxxx";
    static_assert(sizeof(branch) == sizeof(branch_mask) - 1, "pattern and mask lengths");
    // The native block the je lands on: willPresent, TickFrozenCheck, cq, the inlined FG
    // IsActive/IsPaused, mov rcx,r15; call ApplyToFinishedPicture.
    static const uint8_t native[] = {
        0x45, 0x84, 0xE4, 0x0F, 0x84, 0, 0, 0, 0,
        0xE8, 0, 0, 0, 0,
        0x48, 0x8B, 0x88, 0x98, 0x09, 0x00, 0x00, 0x48, 0x85, 0xC9, 0x74, 0x06, 0x48, 0x8B, 0x01, 0xFF, 0x50, 0x40,
        0x48, 0x8B, 0x95, 0x80, 0x00, 0x00, 0x00, 0x48, 0x85, 0xD2, 0x74, 0x45,
        0x48, 0x85, 0xFF, 0x74, 0x33,
        0x48, 0x8B, 0x47, 0x08, 0x48, 0x63, 0x48, 0x04,
        0x80, 0xBC, 0x39, 0x25, 0x02, 0x00, 0x00, 0x00, 0x75, 0x0A,
        0x80, 0xBC, 0x39, 0x18, 0x02, 0x00, 0x00, 0x00, 0x74, 0x17,
        0x48, 0x8B, 0x84, 0x39, 0x28, 0x02, 0x00, 0x00, 0x48, 0x85, 0xC0, 0x74, 0x17,
        0x48, 0x3B, 0x84, 0x39, 0x00, 0x02, 0x00, 0x00, 0x72, 0x0D,
        0x49, 0x8B, 0xCF, 0xE8, 0, 0, 0, 0};
    static const char native_mask[] =
        "xxxxx????"
        "x????"
        "xxxxxxxxxxxxxxxxxx"
        "xxxxxxxxxxxx"
        "xxxxx"
        "xxxxxxxx"
        "xxxxxxxxxx"
        "xxxxxxxxxx"
        "xxxxxxxxxxxxx"
        "xxxxxxxxxx"
        "xxxx????";
    static_assert(sizeof(native) == sizeof(native_mask) - 1, "pattern and mask lengths");

    uint8_t* site = nullptr;
    uint8_t* apply = nullptr;
    const int hits = find_unique(m, branch, branch_mask, sizeof(branch), [&](uint8_t* p) {
        uint8_t* target = rel32_target(p + 4);
        if (target < m.base || target + sizeof(native) > m.base + m.image) return false;
        if (!matches(target, native, native_mask, sizeof(native))) return false;
        apply = rel32_target(target + sizeof(native) - 4);
        return apply >= m.base && apply < m.base + m.image;
    }, &site);
    if (hits != 1) return hits;

    uint8_t* const patch = site + 8;          // the DXVK branch's first instruction
    constexpr size_t kCopied = 16;            // five position-independent instructions
    uint8_t* const back = patch + kCopied;    // test r9,r9

    // The stub: the native block's condition and call, then the five instructions it replaced.
    uint8_t stub[118];
    size_t n = 0;
    auto emit = [&](std::initializer_list<uint8_t> b) { for (uint8_t x : b) stub[n++] = x; };
    auto emit64 = [&](uint64_t v) { std::memcpy(stub + n, &v, 8); n += 8; };
    emit({0x45, 0x84, 0xE4, 0x74, 0x53});                               //  0 test r12b,r12b; jz skip
    emit({0x48, 0x8B, 0x95, 0x80, 0x00, 0x00, 0x00});                   //  5 mov rdx,[rbp+0x80]
    emit({0x48, 0x85, 0xD2, 0x74, 0x47});                               // 12 test rdx,rdx; jz skip
    emit({0x48, 0x85, 0xFF, 0x74, 0x33});                               // 17 test rdi,rdi; jz do
    emit({0x48, 0x8B, 0x47, 0x08, 0x48, 0x63, 0x48, 0x04});             // 22 fg's virtual-base offset
    emit({0x80, 0xBC, 0x39, 0x25, 0x02, 0x00, 0x00, 0x00, 0x75, 0x0A}); // 30 cmp; jne L1
    emit({0x80, 0xBC, 0x39, 0x18, 0x02, 0x00, 0x00, 0x00, 0x74, 0x17}); // 40 cmp; je do
    emit({0x48, 0x8B, 0x84, 0x39, 0x28, 0x02, 0x00, 0x00,               // 50 L1:
          0x48, 0x85, 0xC0, 0x74, 0x19});                               // 58 test rax,rax; jz skip
    emit({0x48, 0x3B, 0x84, 0x39, 0x00, 0x02, 0x00, 0x00, 0x72, 0x0F}); // 63 cmp; jb skip
    emit({0x49, 0x8B, 0xCF, 0x48, 0xB8});                               // 73 do: mov rcx,r15; mov rax,
    emit64(reinterpret_cast<uint64_t>(apply));                          //    ApplyToFinishedPicture
    emit({0xFF, 0xD0});                                                 // 86 call rax
    for (size_t k = 0; k < kCopied; ++k) stub[n++] = patch[k];          // 88 skip: the replaced code
    emit({0xFF, 0x25, 0x00, 0x00, 0x00, 0x00});                         // 104 jmp [rip+0]
    emit64(reinterpret_cast<uint64_t>(back));                           // 110
    if (n != sizeof(stub)) {
        log("[nr] OptiScaler fix (finished picture): stub is %zu bytes, expected %zu; nothing changed", n,
            sizeof(stub));
        return 1;
    }

    auto* code = static_cast<uint8_t*>(VirtualAlloc(nullptr, sizeof(stub), MEM_COMMIT | MEM_RESERVE,
                                                    PAGE_EXECUTE_READWRITE));
    if (!code) {
        log("[nr] OptiScaler fix (finished picture): VirtualAlloc failed (%lu)", GetLastError());
        return 1;
    }
    std::memcpy(code, stub, sizeof(stub));
    DWORD old = 0;
    VirtualProtect(code, sizeof(stub), PAGE_EXECUTE_READ, &old);
    FlushInstructionCache(GetCurrentProcess(), code, sizeof(stub));

    uint8_t jump[kCopied];
    jump[0] = 0xFF; jump[1] = 0x25; std::memset(jump + 2, 0, 4);        // jmp [rip+0]
    const uint64_t to = reinterpret_cast<uint64_t>(code);
    std::memcpy(jump + 6, &to, 8);
    jump[14] = jump[15] = 0x90;
    if (!write_code(patch, jump, sizeof(jump))) {
        log("[nr] OptiScaler fix (finished picture): VirtualProtect failed (%lu)", GetLastError());
        return 1;
    }
    log("[nr] OptiScaler fix: Finished Picture now runs on the DXVK present path (%ls+0x%llx, "
        "ApplyToFinishedPicture %ls+0x%llx)", m.name, static_cast<unsigned long long>(patch - m.base),
        m.name, static_cast<unsigned long long>(apply - m.base));
    return 1;
}

// v0.8.91 (f45ccf3a). The native block now also skips the call when XeFG composes the game's own
// picture (wrapped_swapchain.cpp, xeFgGamePicture):
//
//     if (cq && !xeFgGamePicture && (fg == nullptr || !fg->IsActive() || fg->IsPaused()))
//         DlssNr::ApplyToFinishedPicture(pSwapChain, cq);
//
// and the DXVK branch counts a presented frame itself, after its Present. Registers at the branch:
// r12 = the swapchain, r15d/r13d = SyncInterval/Flags, r14b = willPresent, rsi = State::currentFG,
// [rbp-0x50] = hWnd, [rbp+0x78] = cq. The stub is the native block's D3D12 condition and call,
// writing only rax, rcx and rdx, which the replaced instructions load again.
int finished_picture_0891(const Module& m) {
    // test bl,bl (usesDxvk) / je native / mov rax,[r12]; mov r9,[rbp-0x48]; mov r8d,r13d;
    // mov edx,r15d; mov rcx,r12; test r9,r9; jne +5; call [rax+0x40]
    static const Sig branch("84 DB 0F 84 ?? ?? ?? ?? 49 8B 04 24 4C 8B 4D B8 45 8B C5 41 8B D7 49 8B CC "
                            "4D 85 C9 75 05 FF 50 40");
    // The native block the je lands on: willPresent, TickFrozenCheck, xeFgGamePicture (State::Instance()
    // at +9, +26, +82, +97, +111), cq, the inlined FG IsActive/IsPaused, mov rcx,r12; call
    // ApplyToFinishedPicture.
    static const Sig native(
        "45 84 F6 0F 84 ?? ?? ?? ?? E8 ?? ?? ?? ?? 48 8B 98 98 09 00 00 48 85 DB 74 33 E8 ?? ?? ?? ?? "
        "48 8B 88 A0 09 00 00 4C 8B 03 48 85 C9 74 16 48 8B 41 08 48 63 50 04 8B 94 0A 1C 02 00 00 85 D2 "
        "0F 48 D7 EB 02 8B D7 48 8B CB 41 FF 50 40 48 85 F6 74 41 E8 ?? ?? ?? ?? 48 83 B8 C0 09 00 00 00 "
        "74 32 E8 ?? ?? ?? ?? 83 B8 A4 00 00 00 03 75 24 E8 ?? ?? ?? ?? 83 B8 10 06 00 00 00 75 16 48 8B "
        "06 48 8B CE FF 50 20 48 8B 7D B0 48 3B C7 75 08 B0 01 EB 06 48 8B 7D B0 32 C0 48 8B 55 78 48 85 "
        "D2 74 49 84 C0 75 45 48 85 F6 74 33 48 8B 46 08 48 63 48 04 80 BC 31 25 02 00 00 00 75 0A 80 BC "
        "31 18 02 00 00 00 74 17 48 8B 84 31 28 02 00 00 48 85 C0 74 17 48 3B 84 31 00 02 00 00 72 0D 49 "
        "8B CC E8 ?? ?? ?? ??");
    static const size_t kInstanceCalls[] = {9, 26, 82, 97, 111};

    uint8_t* site = nullptr;
    uint8_t* apply = nullptr;
    uint8_t* instance = nullptr;
    const int hits = find_unique(m, branch, [&](uint8_t* p) {
        uint8_t* target = rel32_target(p + 4);
        if (target < m.base || target + native.bytes.size() > m.base + m.image) return false;
        if (!matches(target, native.bytes.data(), native.mask.c_str(), native.bytes.size())) return false;
        uint8_t* const first = rel32_target(target + kInstanceCalls[0] + 1);
        for (size_t at : kInstanceCalls)
            if (rel32_target(target + at + 1) != first) return false;
        apply = rel32_target(target + native.bytes.size() - 4);
        instance = first;
        return apply >= m.base && apply < m.base + m.image && instance >= m.base && instance < m.base + m.image;
    }, &site);
    if (hits != 1) return hits;

    uint8_t* const patch = site + 8;          // the DXVK branch's first instruction
    constexpr size_t kCopied = 17;            // five position-independent instructions
    uint8_t* const back = patch + kCopied;    // test r9,r9

    // Assembled from the listing on the right; the two movabs immediates at 27 and 145.
    static const uint8_t kStub[] = {
        0x45, 0x84, 0xF6, 0x0F, 0x84, 0x92, 0x00, 0x00, 0x00,              //   0 test r14b,r14b; jz skip
        0x48, 0x83, 0x7D, 0x78, 0x00, 0x0F, 0x84, 0x87, 0x00, 0x00, 0x00,  //   9 cmp [rbp+0x78],0; jz skip
        0x48, 0x85, 0xF6, 0x74, 0x37,                                      //  20 test rsi,rsi; jz fgcheck
        0x48, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xD0,                    //  25 call State::Instance
        0x48, 0x83, 0xB8, 0xC0, 0x09, 0x00, 0x00, 0x00, 0x74, 0x21,        //  37 currentFGSwapchain
        0x83, 0xB8, 0xA4, 0x00, 0x00, 0x00, 0x03, 0x75, 0x18,              //  47 activeFgOutput == XeFG
        0x83, 0xB8, 0x10, 0x06, 0x00, 0x00, 0x00, 0x75, 0x0F,              //  56 no swapchain interop
        0x48, 0x8B, 0x06, 0x48, 0x89, 0xF1, 0xFF, 0x50, 0x20,              //  65 fg->Hwnd()
        0x48, 0x3B, 0x45, 0xB0, 0x74, 0x4B,                                //  74 == hWnd: skip
        0x48, 0x85, 0xF6, 0x74, 0x33,                                      //  80 fgcheck: test rsi,rsi; jz do
        0x48, 0x8B, 0x46, 0x08, 0x48, 0x63, 0x48, 0x04,                    //  85 fg's virtual-base offset
        0x80, 0xBC, 0x31, 0x25, 0x02, 0x00, 0x00, 0x00, 0x75, 0x0A,        //  93 cmp; jne active
        0x80, 0xBC, 0x31, 0x18, 0x02, 0x00, 0x00, 0x00, 0x74, 0x17,        // 103 cmp; je do
        0x48, 0x8B, 0x84, 0x31, 0x28, 0x02, 0x00, 0x00,                    // 113 active:
        0x48, 0x85, 0xC0, 0x74, 0x1D,                                      // 121 test rax,rax; jz skip
        0x48, 0x3B, 0x84, 0x31, 0x00, 0x02, 0x00, 0x00, 0x72, 0x13,        // 126 cmp; jb skip
        0x48, 0x8B, 0x55, 0x78, 0x4C, 0x89, 0xE1,                          // 136 do: rdx = cq; rcx = r12
        0x48, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xD0};                   // 143 call ApplyToFinishedPicture
    static_assert(sizeof(kStub) == 155, "fp0891.s is 155 bytes up to skip");
    uint8_t stub[sizeof(kStub) + kCopied + 14];
    std::memcpy(stub, kStub, sizeof(kStub));
    const uint64_t to_instance = reinterpret_cast<uint64_t>(instance), to_apply = reinterpret_cast<uint64_t>(apply),
                   to_back = reinterpret_cast<uint64_t>(back);
    std::memcpy(stub + 27, &to_instance, 8);
    std::memcpy(stub + 145, &to_apply, 8);
    std::memcpy(stub + sizeof(kStub), patch, kCopied);                  // skip: the replaced code
    uint8_t* const tail = stub + sizeof(kStub) + kCopied;
    tail[0] = 0xFF; tail[1] = 0x25; std::memset(tail + 2, 0, 4);        // jmp [rip+0]
    std::memcpy(tail + 6, &to_back, 8);

    uint8_t* const code = place_code(stub, sizeof(stub));
    if (!code) {
        log("[nr] OptiScaler fix (finished picture): VirtualAlloc failed (%lu)", GetLastError());
        return 1;
    }
    uint8_t jump[kCopied];
    jump[0] = 0xFF; jump[1] = 0x25; std::memset(jump + 2, 0, 4);        // jmp [rip+0]
    const uint64_t to = reinterpret_cast<uint64_t>(code);
    std::memcpy(jump + 6, &to, 8);
    std::memset(jump + 14, 0x90, kCopied - 14);
    if (!write_code(patch, jump, sizeof(jump))) {
        log("[nr] OptiScaler fix (finished picture): VirtualProtect failed (%lu)", GetLastError());
        return 1;
    }
    log("[nr] OptiScaler fix: Finished Picture now runs on the DXVK present path (%ls+0x%llx, "
        "ApplyToFinishedPicture %ls+0x%llx)", m.name, static_cast<unsigned long long>(patch - m.base),
        m.name, static_cast<unsigned long long>(apply - m.base));
    return 1;
}

void fix_finished_picture(const Module& m) {
    const int old_layout = finished_picture_084(m);
    if (old_layout == 1) return;
    const int new_layout = finished_picture_0891(m);
    if (new_layout == 1) return;
    log("[nr] OptiScaler fix (finished picture): %d/%d matching sites (v0.8.4/v0.8.91) in %ls, nothing changed",
        old_layout, new_layout, m.name);
}

// ---- 3. Window-sized swapchains are not overlays ------------------------------------------------
//
// DxgiFactoryHooks::CreateSwapChainForHwnd treats any swapchain under 100 pixels in either direction
// as an overlay's and does not wrap it. DXGI_SWAP_CHAIN_DESC1 allows 0 x 0 ("the size of the
// window"), and a game that asks for that (Helldivers 2) is left unwrapped: LocalPresent never runs,
// the frame counter never advances, and every NR feature waits forever for the frame after the one
// it was created in (DlssNr_Proxy.cpp: ready = submissionEpoch != creationEpoch) - features are
// created and nothing is ever evaluated. CreateSwapChain already resolves 0 x 0 to the window
// before the same test (ResolveWindowSizedSwapchain); ForHwnd does not (still so in v0.8.91).
//
// The two `jb overlay` of the test are redirected to stubs that send a 0 back to the normal path
// and anything from 1 to 99 on to the overlay block as before.
void* alloc_near(uint8_t* base, size_t image, size_t bytes) {
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    const uintptr_t step = si.dwAllocationGranularity ? si.dwAllocationGranularity : 0x10000;
    const uintptr_t lo = reinterpret_cast<uintptr_t>(base), hi = lo + image;
    for (uintptr_t d = step; d < 0x70000000; d += step) {
        const uintptr_t up = ((hi + d) / step) * step;
        if (void* p = VirtualAlloc(reinterpret_cast<void*>(up), bytes, MEM_COMMIT | MEM_RESERVE,
                                   PAGE_EXECUTE_READWRITE)) return p;
        if (lo > d + step) {
            const uintptr_t down = ((lo - d) / step) * step;
            if (void* p = VirtualAlloc(reinterpret_cast<void*>(down), bytes, MEM_COMMIT | MEM_RESERVE,
                                       PAGE_EXECUTE_READWRITE)) return p;
        }
    }
    return nullptr;
}

void fix_window_sized_swapchain(const Module& m) {
    // test r14,r14 / je  (pDesc == nullptr), then the size test on pDesc (r14):
    //   lea r8,[r14+4]; cmp dword [r8],100; jb overlay; cmp dword [r14],100; jb overlay
    static const uint8_t pat[] = {
        0x4D, 0x85, 0xF6, 0x0F, 0x84, 0, 0, 0, 0,
        0x4D, 0x8D, 0x46, 0x04, 0x41, 0x83, 0x38, 0x64, 0x0F, 0x82, 0, 0, 0, 0,
        0x41, 0x83, 0x3E, 0x64, 0x0F, 0x82, 0, 0, 0, 0};
    static const char mask[] = "xxxxx????xxxxxxxxxx????xxxxxx????";
    static_assert(sizeof(pat) == sizeof(mask) - 1, "pattern and mask lengths");
    constexpr size_t kJbHeight = 17, kJbWidth = 27;   // the two 0F 82 rel32
    static const char kLog[] = "DxgiFactoryHooks::CreateSwapChainForHwnd Overlay call!";

    uint8_t* site = nullptr;
    uint8_t* overlay = nullptr;
    const int hits = find_unique(m, pat, mask, sizeof(pat), [&](uint8_t* p) {
        uint8_t* a = rel32_target(p + kJbHeight + 2);
        uint8_t* b = rel32_target(p + kJbWidth + 2);
        // Both lead to the same block, which starts by loading its log line.
        if (a != b || a < m.base || a + 7 > m.base + m.image || a[0] != 0x48 || a[1] != 0x8D || a[2] != 0x05)
            return false;
        const uint8_t* str = rel32_target(a + 3);
        if (str < m.base || str + sizeof(kLog) > m.base + m.image ||
            std::memcmp(str, kLog, sizeof(kLog) - 1) != 0) return false;
        overlay = a;
        return true;
    }, &site);
    if (hits != 1) {
        log("[nr] OptiScaler fix (window-sized swapchain): %d matching sites in %ls, nothing changed", hits,
            m.name);
        return;
    }
    uint8_t* const after_height = site + kJbHeight + 6;   // cmp dword [r14],100
    uint8_t* const after_width = site + kJbWidth + 6;     // the normal path

    // Two stubs, 34 bytes each:  cmp dword [reg],0 / jne +14 / jmp [rip] back / jmp [rip] overlay
    uint8_t stub[68];
    size_t n = 0;
    auto emit = [&](std::initializer_list<uint8_t> b) { for (uint8_t x : b) stub[n++] = x; };
    auto emit64 = [&](const void* v) { const uint64_t x = reinterpret_cast<uint64_t>(v); std::memcpy(stub + n, &x, 8); n += 8; };
    emit({0x41, 0x83, 0x38, 0x00, 0x75, 0x0E, 0xFF, 0x25, 0, 0, 0, 0}); emit64(after_height);   // Height
    emit({0xFF, 0x25, 0, 0, 0, 0}); emit64(overlay);
    emit({0x41, 0x83, 0x3E, 0x00, 0x75, 0x0E, 0xFF, 0x25, 0, 0, 0, 0}); emit64(after_width);    // Width
    emit({0xFF, 0x25, 0, 0, 0, 0}); emit64(overlay);
    if (n != sizeof(stub)) return;

    auto* code = static_cast<uint8_t*>(alloc_near(m.base, m.image, sizeof(stub)));
    auto rel = [](uint8_t* from_end, uint8_t* to, int32_t* out) {
        const int64_t d = to - from_end;
        if (d < INT32_MIN || d > INT32_MAX) return false;
        *out = static_cast<int32_t>(d);
        return true;
    };
    int32_t r_height = 0, r_width = 0;
    if (!code || !rel(site + kJbHeight + 6, code, &r_height) || !rel(site + kJbWidth + 6, code + 34, &r_width)) {
        if (code) VirtualFree(code, 0, MEM_RELEASE);
        log("[nr] OptiScaler fix (window-sized swapchain): no memory within reach of %ls, nothing changed", m.name);
        return;
    }
    std::memcpy(code, stub, sizeof(stub));
    DWORD old = 0;
    VirtualProtect(code, sizeof(stub), PAGE_EXECUTE_READ, &old);
    FlushInstructionCache(GetCurrentProcess(), code, sizeof(stub));
    if (!write_code(site + kJbHeight + 2, &r_height, 4) || !write_code(site + kJbWidth + 2, &r_width, 4)) {
        log("[nr] OptiScaler fix (window-sized swapchain): VirtualProtect failed (%lu)", GetLastError());
        return;
    }
    log("[nr] OptiScaler fix: a 0 x 0 (window-sized) swapchain is wrapped, not taken for an overlay (%ls+0x%llx)",
        m.name, static_cast<unsigned long long>(site - m.base));
}

// ---- 4. A depth/stencil depth is not cloned into a colour texture --------------------------------
//
// DlssNr_Dx12::State::TypedGuideFormat maps a typeless guide to a typed one, and ReadableGuide then
// creates a plain texture in that format (no depth-stencil flag) and CopyResource-s the game's guide
// into it. For a depth/stencil buffer the copy target is a shader-view format that is not a valid
// plain texture in D3D12: R24_UNORM_X8_TYPELESS for R24G8_TYPELESS (D24S8), R32_FLOAT_X8X24_TYPELESS
// for R32G8X24_TYPELESS (D32S8). Under vkd3d-proton that copy can hang the GPU: Helldivers 2 and
// Kingdom Come: Deliverance II (D24S8, "cloned a typeless guide as format 46") and S.T.A.L.K.E.R. 2
// (D32S8, "... as format 21", with our evaluate recording nothing) lost the device on NR's first
// frame. 007 First Light went through the same D32S8 copy without a hang.
//
// The fix returns both formats unchanged, so the depth is not treated as typeless and is handed on
// as it is: no clone, no copy. Our side reads the depth aspect of the game's own buffer
// (nr_pe_interop.cpp depth_stencil_format, runtime_depth.comp); the other guides are untouched.
void fix_depth_stencil_guide(const Module& m) {
    // TypedGuideFormat, v0.8.4: a jump table over (f - 9), eight `mov eax,<typed>; ret` cases and the
    // default `mov eax,edx; ret`. The R24G8_TYPELESS (44) case returns 46 (R24_UNORM_X8_TYPELESS).
    static const uint8_t pat[] = {
        0x8D, 0x42, 0xF7, 0x83, 0xF8, 0x2C, 0x77, 0x4F,
        0x4C, 0x8D, 0x05, 0, 0, 0, 0, 0x48, 0x98,
        0x41, 0x0F, 0xB6, 0x84, 0x00, 0, 0, 0, 0,
        0x41, 0x8B, 0x8C, 0x80, 0, 0, 0, 0, 0x49, 0x03, 0xC8, 0xFF, 0xE1,
        0xB8, 0x29, 0x00, 0x00, 0x00, 0xC3,    // R32_TYPELESS      -> R32_FLOAT
        0xB8, 0x38, 0x00, 0x00, 0x00, 0xC3,    // R16_TYPELESS      -> R16_UNORM
        0xB8, 0x2E, 0x00, 0x00, 0x00, 0xC3,    // R24G8_TYPELESS    -> R24_UNORM_X8_TYPELESS
        0xB8, 0x15, 0x00, 0x00, 0x00, 0xC3,    // R32G8X24_TYPELESS -> R32_FLOAT_X8X24_TYPELESS
        0xB8, 0x10, 0x00, 0x00, 0x00, 0xC3,
        0xB8, 0x22, 0x00, 0x00, 0x00, 0xC3,
        0xB8, 0x1C, 0x00, 0x00, 0x00, 0xC3,
        0xB8, 0x0A, 0x00, 0x00, 0x00, 0xC3,
        0x8B, 0xC2, 0xC3};                      // default: the format itself
    static const char mask[] =
        "xxxxxxxx" "xxx????xx" "xxxxx????" "xxxx????xxxxx"
        "xxxxxx" "xxxxxx" "xxxxxx" "xxxxxx" "xxxxxx" "xxxxxx" "xxxxxx" "xxxxxx" "xxx";
    static_assert(sizeof(pat) == sizeof(mask) - 1, "pattern and mask lengths");
    constexpr size_t kR24 = 39 + 12;            // the `mov eax,0x2e` of the R24G8 case
    constexpr size_t kR32 = 39 + 18;            // the `mov eax,0x15` of the R32G8X24 case

    uint8_t* hit = nullptr;
    const int hits = find_unique(m, pat, mask, sizeof(pat), [](uint8_t*) { return true; }, &hit);
    if (hits != 1) {
        log("[nr] OptiScaler fix (depth/stencil guide): %d matching sites in %ls, nothing changed", hits, m.name);
        return;
    }
    static const uint8_t fix[] = {0x8B, 0xC2, 0x90, 0x90, 0x90};   // mov eax,edx (the format) ; nops
    if (!write_code(hit + kR24, fix, sizeof(fix)) || !write_code(hit + kR32, fix, sizeof(fix))) {
        log("[nr] OptiScaler fix (depth/stencil guide): VirtualProtect failed (%lu)", GetLastError());
        return;
    }
    log("[nr] OptiScaler fix: a D24S8/D32S8 depth guide is passed on as it is, not copied into a colour "
        "texture (%ls+0x%llx)", m.name, static_cast<unsigned long long>(hit + kR24 - m.base));
}

// ---- 5. AutoExposure marks a float colour as linear HDR ----------------------------------------
//
// MakeDlssNrPass decides whether NR's input is linear HDR - and so whether OptiScaler encodes it
// (white point, soft knee, sRGB) and carries the edit back as a ratio - from the game's DLSS create
// flags alone: ColourIsLinearHdr = (flags & IsHDR) != 0, then cleared for non-float formats. Some
// games leave IsHDR clear yet hand over an unexposed scene-linear float buffer and ask the upscaler
// to expose it (AutoExposure): 007 First Light (R11G11B10) and Helldivers 2 (RGBA16, values up to
// ~560 in the readback) - the network, shown that raw, answered with coloured, flickering blocks.
// AutoExposure only has a meaning for HDR input (NVIDIA's DLSS documentation), so the test becomes
// (flags & (IsHDR | AutoExposure)) != 0; the float-format check after it is unchanged, so 8/10-bit
// frames are never affected. Kingdom Come: Deliverance II, which sets IsHDR, already took this path.
void fix_autoexposure_hdr(const Module& m) {
    // mov ecx,[rbp+0x170] (featureFlags); DepthInverted = flags>>3 & 1; MVLowRes = flags>>1 & 1;
    // and cl,1 ; mov [rbp+0x8d],cl  <- ColourIsLinearHdr ; lea rdx,[rip+"DLSS.Output"...]
    static const uint8_t pat[] = {
        0x8B, 0x8D, 0x70, 0x01, 0x00, 0x00, 0x8B, 0xC1, 0xC1, 0xE8, 0x03, 0x24, 0x01,
        0x88, 0x85, 0x81, 0x00, 0x00, 0x00, 0x8B, 0xC1, 0xD1, 0xE8, 0x24, 0x01,
        0x88, 0x85, 0xD4, 0x00, 0x00, 0x00,
        0x80, 0xE1, 0x01, 0x88, 0x8D, 0x8D, 0x00, 0x00, 0x00,
        0x48, 0x8D, 0x15};
    static const char mask[] = "xxxxxxxxxxxxx" "xxxxxxxxxxxx" "xxxxxx" "xxxxxxxxx" "xxx";
    static_assert(sizeof(pat) == sizeof(mask) - 1, "pattern and mask lengths");
    // v0.8.91: the same sequence, flags at [rbp+0x180], ColourIsLinearHdr at [rbp+0x6d]:
    // and cl,1 ; mov [rbp+0x6d],cl ; lea rdx,[rip+"DLSSD.Output"...]
    static const Sig pat_0891("8B 8D 80 01 00 00 8B C1 C1 E8 03 24 01 88 45 61 8B C1 D1 E8 24 01 "
                              "88 85 B4 00 00 00 80 E1 01 88 4D 6D 48 8D 15");
    size_t kAnd = 31, kLen = 9;                 // and cl,1 ; mov [rbp+0x8d],cl

    uint8_t* hit = nullptr;
    const int hits = find_unique(m, pat, mask, sizeof(pat), [](uint8_t*) { return true; }, &hit);
    if (hits != 1) {
        const int hits_0891 = find_unique(m, pat_0891, [](uint8_t*) { return true; }, &hit);
        if (hits_0891 != 1) {
            log("[nr] OptiScaler fix (AutoExposure HDR): %d/%d matching sites (v0.8.4/v0.8.91) in %ls, nothing "
                "changed", hits, hits_0891, m.name);
            return;
        }
        kAnd = 28; kLen = 6;                    // and cl,1 ; mov [rbp+0x6d],cl
    }
    uint8_t* const at = hit + kAnd;
    uint8_t* const back = at + kLen;
    // test cl,IsHDR|AutoExposure ; setne cl ; the site's own store of cl ; jmp back
    uint8_t stub[17] = {0xF6, 0xC1, 0x41, 0x0F, 0x95, 0xC1};
    const size_t store = kLen - 3, size = 6 + store + 5;
    std::memcpy(stub + 6, at + 3, store);
    stub[6 + store] = 0xE9;
    auto* code = static_cast<uint8_t*>(alloc_near(m.base, m.image, size));
    int32_t to_stub = 0, to_back = 0;
    auto rel = [](uint8_t* from_end, uint8_t* to, int32_t* out) {
        const int64_t d = to - from_end;
        if (d < INT32_MIN || d > INT32_MAX) return false;
        *out = static_cast<int32_t>(d);
        return true;
    };
    if (!code || !rel(at + 5, code, &to_stub) || !rel(code + size, back, &to_back)) {
        if (code) VirtualFree(code, 0, MEM_RELEASE);
        log("[nr] OptiScaler fix (AutoExposure HDR): no memory within reach of %ls, nothing changed", m.name);
        return;
    }
    std::memcpy(stub + size - 4, &to_back, 4);
    std::memcpy(code, stub, size);
    DWORD old = 0;
    VirtualProtect(code, size, PAGE_EXECUTE_READ, &old);
    FlushInstructionCache(GetCurrentProcess(), code, size);
    uint8_t jump[9] = {0xE9, 0, 0, 0, 0, 0x90, 0x90, 0x90, 0x90};
    std::memcpy(jump + 1, &to_stub, 4);
    if (!write_code(at, jump, kLen)) {
        log("[nr] OptiScaler fix (AutoExposure HDR): VirtualProtect failed (%lu)", GetLastError());
        return;
    }
    log("[nr] OptiScaler fix: a float colour with AutoExposure is treated as linear HDR (%ls+0x%llx)", m.name,
        static_cast<unsigned long long>(at - m.base));
}

// ---- 6. A released feature outlives the GPU work recorded for it -------------------------------
//
// NVSDK_NGX_D3D12_ReleaseFeature erases the feature from Dx12Contexts at once, and with it the
// feature's shaders and their GpuTime_Dx12 timestamp heaps - whether or not the GPU has executed
// what that feature recorded. S.T.A.L.K.E.R. 2 destroys its first FSR context (ffxDestroyContext ->
// ReleaseFeature) right after presenting the first frame that used it: vkd3d-proton frees the query
// pools, the frame's own list then writes its timestamps into unmapped memory and waits on them
// (CmdCopyQueryPoolResults, WAIT_REG_MEM), and the GPU hangs. RADV_DEBUG=hang reported
// "Potential use-after-free" on a 6-timestamp pool freed in that ReleaseFeature, with the NR model
// recording nothing at all (NR_DEBUG_NOOP=1).
//
// The fix detours the exported EvaluateFeature and ReleaseFeature (the FSR/XeSS inputs call the
// same functions internally). After each evaluate a WriteBufferImmediate at the end of the game's
// list stores a serial in a readback buffer; a ReleaseFeature arriving before the GPU has written
// the feature's last serial is held, and carried out on a later NGX call once it has - or after
// kHoldMs, for a list that is never executed. Shutdown carries out everything still held first.
// The release only reads InHandle->Id, so a held release keeps its own copy of the handle.
namespace release_hold {

struct Handle { unsigned int id; };   // NVSDK_NGX_Handle
using EvaluateFn = int (*)(ID3D12GraphicsCommandList*, const Handle*, const void*, void*);
using ReleaseFn = int (*)(Handle*);
using ShutdownFn = int (*)();
using Shutdown1Fn = int (*)(ID3D12Device*);

constexpr uint32_t kSlots = 1024;          // one per evaluated feature, reused round robin
constexpr DWORD kHoldMs = 5000;

EvaluateFn real_evaluate;
ReleaseFn real_release;
ShutdownFn real_shutdown;
Shutdown1Fn real_shutdown1;

struct Mark { unsigned int id; uint32_t slot, serial; };
struct Held { Handle* handle; uint32_t slot, serial; DWORD since; };

std::mutex lock;
ID3D12Device* device;             // the device the marks live on (not owned)
ID3D12Resource* marks;
volatile uint32_t* written;       // the readback buffer, mapped for the process's life
D3D12_GPU_VIRTUAL_ADDRESS marks_va;
uint32_t serial, next_slot;
std::vector<Mark> live;           // the last mark of each feature that has been evaluated
std::vector<Held> held;
bool logged;

bool reached(uint32_t slot, uint32_t s) { return static_cast<int32_t>(written[slot] - s) >= 0; }

// Releases whose GPU work is done (or that waited long enough); `all` for shutdown.
std::vector<Handle*> take_ready(bool all) {
    std::vector<Handle*> out;
    const DWORD now = GetTickCount();
    for (size_t i = 0; i < held.size();) {
        const Held& h = held[i];
        if (all || reached(h.slot, h.serial) || now - h.since >= kHoldMs) {
            out.push_back(h.handle);
            held.erase(held.begin() + static_cast<long>(i));
        } else {
            ++i;
        }
    }
    return out;
}

void release_all(const std::vector<Handle*>& handles) {
    for (Handle* h : handles) {
        real_release(h);
        delete h;
    }
}

bool ensure_marks(ID3D12GraphicsCommandList* cmd) {
    ID3D12Device* dev = nullptr;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&dev))) || !dev) return false;
    dev->Release();                   // the list keeps its device alive
    if (dev == device) return marks != nullptr;
    // A new device: nothing recorded for the old one can be tracked any more.
    live.clear();
    if (marks) { marks->Release(); marks = nullptr; written = nullptr; }
    device = dev;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = kSlots * sizeof(uint32_t);
    desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                            nullptr, IID_PPV_ARGS(&marks))))
        return false;
    void* p = nullptr;
    if (FAILED(marks->Map(0, nullptr, &p)) || !p) { marks->Release(); marks = nullptr; return false; }
    written = static_cast<volatile uint32_t*>(p);
    for (uint32_t i = 0; i < kSlots; ++i) written[i] = 0;
    marks_va = marks->GetGPUVirtualAddress();
    return true;
}

int evaluate(ID3D12GraphicsCommandList* cmd, const Handle* handle, const void* params, void* callback) {
    std::vector<Handle*> ready;
    {
        std::lock_guard<std::mutex> g(lock);
        ready = take_ready(false);
    }
    release_all(ready);
    const int result = real_evaluate(cmd, handle, params, callback);
    if (!cmd || !handle) return result;
    std::lock_guard<std::mutex> g(lock);
    ID3D12GraphicsCommandList2* list2 = nullptr;
    if (!ensure_marks(cmd) || FAILED(cmd->QueryInterface(IID_PPV_ARGS(&list2))) || !list2) return result;
    Mark* mark = nullptr;
    for (Mark& m : live)
        if (m.id == handle->id) mark = &m;
    if (!mark) {
        live.push_back({handle->id, next_slot++ % kSlots, 0});
        mark = &live.back();
    }
    mark->serial = ++serial;
    const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER p{marks_va + mark->slot * sizeof(uint32_t), mark->serial};
    const D3D12_WRITEBUFFERIMMEDIATE_MODE mode = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
    list2->WriteBufferImmediate(1, &p, &mode);
    list2->Release();
    return result;
}

int release(Handle* handle) {
    std::vector<Handle*> ready;
    {
        std::lock_guard<std::mutex> g(lock);
        ready = take_ready(false);
        if (handle) {
            for (size_t i = 0; i < live.size(); ++i) {
                if (live[i].id != handle->id) continue;
                const Mark m = live[i];
                live.erase(live.begin() + static_cast<long>(i));
                if (written && !reached(m.slot, m.serial)) {
                    held.push_back({new Handle{handle->id}, m.slot, m.serial, GetTickCount()});
                    if (!logged) {
                        logged = true;
                        log("[nr] OptiScaler fix: feature %u released before the GPU finished its work; "
                            "the release waits for it", handle->id);
                    }
                    handle = nullptr;
                }
                break;
            }
        }
    }
    release_all(ready);
    return handle ? real_release(handle) : 1;   // NVSDK_NGX_Result_Success
}

void flush_all() {
    std::vector<Handle*> all;
    {
        std::lock_guard<std::mutex> g(lock);
        all = take_ready(true);
        live.clear();
    }
    release_all(all);
}

int shutdown() { flush_all(); return real_shutdown(); }
int shutdown1(ID3D12Device* d) { flush_all(); return real_shutdown1(d); }

}  // namespace release_hold

void fix_release_hold(const Module& m) {
    auto* module = reinterpret_cast<HMODULE>(m.base);
    struct { const char* name; void* detour; void** original; } entries[] = {
        {"NVSDK_NGX_D3D12_EvaluateFeature", reinterpret_cast<void*>(&release_hold::evaluate),
         reinterpret_cast<void**>(&release_hold::real_evaluate)},
        {"NVSDK_NGX_D3D12_ReleaseFeature", reinterpret_cast<void*>(&release_hold::release),
         reinterpret_cast<void**>(&release_hold::real_release)},
        {"NVSDK_NGX_D3D12_Shutdown", reinterpret_cast<void*>(&release_hold::shutdown),
         reinterpret_cast<void**>(&release_hold::real_shutdown)},
        {"NVSDK_NGX_D3D12_Shutdown1", reinterpret_cast<void*>(&release_hold::shutdown1),
         reinterpret_cast<void**>(&release_hold::real_shutdown1)},
    };
    void* targets[4] = {};
    for (size_t i = 0; i < 4; ++i) {
        targets[i] = reinterpret_cast<void*>(GetProcAddress(module, entries[i].name));
        if (!targets[i]) {
            log("[nr] OptiScaler fix (held release): %ls exports no %s, nothing changed", m.name, entries[i].name);
            return;
        }
    }
    const MH_STATUS init = MH_Initialize();
    if (init != MH_OK && init != MH_ERROR_ALREADY_INITIALIZED) {
        log("[nr] OptiScaler fix (held release): MinHook would not start (%d), nothing changed", int(init));
        return;
    }
    for (size_t i = 0; i < 4; ++i) {
        if (MH_CreateHook(targets[i], entries[i].detour, entries[i].original) != MH_OK) {
            for (size_t k = 0; k < i; ++k) MH_RemoveHook(targets[k]);
            log("[nr] OptiScaler fix (held release): %s could not be hooked, nothing changed", entries[i].name);
            return;
        }
    }
    for (size_t i = 0; i < 4; ++i)
        if (MH_EnableHook(targets[i]) != MH_OK)
            log("[nr] OptiScaler fix (held release): %s could not be enabled", entries[i].name);
    log("[nr] OptiScaler fix: a released feature is kept until the GPU has run its work (%ls+0x%llx)", m.name,
        static_cast<unsigned long long>(static_cast<uint8_t*>(targets[1]) - m.base));
}

// ---- 7. Every float colour format counts as float ----------------------------------------------
//
// Right after the flag test fix 5 changes, MakeDlssNrPass clears ColourIsLinearHdr unless the output
// (or colour) is one of R32G32B32A32 typeless/float, R32G32B32 float, R16G16B16A16 typeless/float or
// R11G11B10 float (DlssNr_Pipeline_Dx12.cpp: a switch, compiled into a jump table over formats 1..26
// whose out-of-range side is the clearing store). R9G9B9E5_SHAREDEXP (67) is float too and is what
// RE Engine hands its upscaler: Resident Evil Requiem creates its FSR context with IsHdr set and a
// format-67 colour, and OptiScaler then logs "the game's DLSS colour space is already tone-mapped",
// sending unexposed scene-linear light to the network. R32G32B32_TYPELESS (5) is the only other
// float member left out. With both added the list is every DXGI float colour format there is.
//
// The range check in front of the table becomes a jump to a stub that sends 67 and 5 to the table's
// own "keep" target and does the original check for everything else; the table is untouched, so the
// answer for every other format is what it was.
void fix_float_formats(const Module& m) {
    // mov ecx,[rax+0x20] (Format) ; dec ecx ; cmp ecx,0x19 ; ja clear ; movsxd rax,ecx ;
    // lea rdx,[rip+image base] ; movzx eax,byte [rdx+rax+index] ; mov ecx,[rdx+rax*4+targets] ;
    // add rcx,rdx ; jmp rcx ; clear: mov byte [rbp+0x8d],0 (ColourIsLinearHdr) ; keep:
    static const uint8_t pat[] = {
        0x8B, 0x48, 0x20, 0xFF, 0xC9, 0x83, 0xF9, 0x19, 0x77, 0x1E,
        0x48, 0x63, 0xC1, 0x48, 0x8D, 0x15, 0, 0, 0, 0,
        0x0F, 0xB6, 0x84, 0x02, 0, 0, 0, 0,
        0x8B, 0x8C, 0x82, 0, 0, 0, 0,
        0x48, 0x03, 0xCA, 0xFF, 0xE1,
        0xC6, 0x85, 0x8D, 0x00, 0x00, 0x00, 0x00};
    static const char mask[] = "xxxxxxxxxx" "xxxxxx????" "xxxx????" "xxx????" "xxxxx" "xxxxxxx";
    static_assert(sizeof(pat) == sizeof(mask) - 1, "pattern and mask lengths");
    // v0.8.91: FormatCanHoldLinearHdr, the same table, now answering in al for
    // `ColourIsLinearHdr &= ...`: ... jmp rcx ; keep: mov al,1 ; jmp store ; clear: xor al,al ;
    // store: and [rbp+0x6d],al
    static const Sig pat_0891("8B 48 20 FF C9 83 F9 19 77 22 48 63 C1 48 8D 15 ?? ?? ?? ?? 0F B6 84 02 ?? ?? ?? ?? "
                              "8B 8C 82 ?? ?? ?? ?? 48 03 CA FF E1 B0 01 EB 02 32 C0 20 45 6D");
    constexpr size_t kCheck = 5, kLen = 5;      // cmp ecx,0x19 ; ja clear
    size_t kClear = 40, kKeep = sizeof(pat);

    uint8_t* hit = nullptr;
    const int hits = find_unique(m, pat, mask, sizeof(pat), [](uint8_t*) { return true; }, &hit);
    if (hits != 1) {
        const int hits_0891 = find_unique(m, pat_0891, [](uint8_t*) { return true; }, &hit);
        if (hits_0891 != 1) {
            log("[nr] OptiScaler fix (float formats): %d/%d matching sites (v0.8.4/v0.8.91) in %ls, nothing "
                "changed", hits, hits_0891, m.name);
            return;
        }
        kClear = 44; kKeep = 40;                // xor al,al / mov al,1
    }
    uint8_t* const at = hit + kCheck;
    uint8_t* const back = at + kLen;
    uint8_t* const clear = hit + kClear;
    uint8_t* const keep = hit + kKeep;
    uint8_t stub[32] = {0x83, 0xF9, 0x42,                        // cmp ecx,67-1 (R9G9B9E5_SHAREDEXP)
                        0x0F, 0x84, 0, 0, 0, 0,                  // je keep
                        0x83, 0xF9, 0x04,                        // cmp ecx,5-1 (R32G32B32_TYPELESS)
                        0x0F, 0x84, 0, 0, 0, 0,                  // je keep
                        0x83, 0xF9, 0x19,                        // cmp ecx,0x19 (the original check)
                        0x0F, 0x87, 0, 0, 0, 0,                  // ja clear
                        0xE9, 0, 0, 0, 0};                       // jmp back (the table)
    auto* code = static_cast<uint8_t*>(alloc_near(m.base, m.image, sizeof(stub)));
    auto rel = [](uint8_t* from_end, uint8_t* to, int32_t* out) {
        const int64_t d = to - from_end;
        if (d < INT32_MIN || d > INT32_MAX) return false;
        *out = static_cast<int32_t>(d);
        return true;
    };
    int32_t to_stub = 0, r[4] = {};
    if (!code || !rel(at + 5, code, &to_stub) || !rel(code + 9, keep, &r[0]) || !rel(code + 18, keep, &r[1]) ||
        !rel(code + 27, clear, &r[2]) || !rel(code + 32, back, &r[3])) {
        if (code) VirtualFree(code, 0, MEM_RELEASE);
        log("[nr] OptiScaler fix (float formats): no memory within reach of %ls, nothing changed", m.name);
        return;
    }
    std::memcpy(stub + 5, &r[0], 4);
    std::memcpy(stub + 14, &r[1], 4);
    std::memcpy(stub + 23, &r[2], 4);
    std::memcpy(stub + 28, &r[3], 4);
    std::memcpy(code, stub, sizeof(stub));
    DWORD old = 0;
    VirtualProtect(code, sizeof(stub), PAGE_EXECUTE_READ, &old);
    FlushInstructionCache(GetCurrentProcess(), code, sizeof(stub));
    uint8_t jump[kLen] = {0xE9, 0, 0, 0, 0};
    std::memcpy(jump + 1, &to_stub, 4);
    if (!write_code(at, jump, sizeof(jump))) {
        log("[nr] OptiScaler fix (float formats): VirtualProtect failed (%lu)", GetLastError());
        return;
    }
    log("[nr] OptiScaler fix: R9G9B9E5 and R32G32B32 typeless colour count as float for linear HDR (%ls+0x%llx)",
        m.name, static_cast<unsigned long long>(at - m.base));
}

}  // namespace
#endif

void fix_optiscaler() {
#if defined(_WIN64)
    if (const char* e = std::getenv("NR_OPTISCALER_FIX"); e && e[0] == '0') {
        log("[nr] OptiScaler fixes: off (NR_OPTISCALER_FIX=0)");
        return;
    }
    Module m{};
    if (!find_module(&m)) {
        log("[nr] OptiScaler fixes: no OptiScaler module found, nothing changed");
        return;
    }
    fix_extension_query(m);
    fix_finished_picture(m);
    fix_window_sized_swapchain(m);
    fix_depth_stencil_guide(m);
    fix_autoexposure_hdr(m);
    fix_release_hold(m);
    fix_float_formats(m);
#endif
}

}  // namespace nr::pe
