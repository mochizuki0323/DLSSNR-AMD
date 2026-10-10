#include "nr_pe_vkdevice.hpp"

#include "nr_pe_bridge.hpp"
#include "nr_pe_log.hpp"
#include "nr_device_features.hpp"
#include "nr_pipeline_binary.hpp"

#include <windows.h>
#include <MinHook.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace nr::pe::vkdevice {
namespace {

using nr::pe::log;

std::mutex lock;
DeviceHandles seen{};

struct Queue { VkDevice device; uint32_t family; VkQueue queue; };
std::vector<Queue> queues;
// The queue families the game asked for in its last vkCreateDevice.
std::vector<uint32_t> requested_families;
// Devices created with the network's features added (network_features_added).
std::vector<VkDevice> augmented_devices;
void (*device_callback)(VkDevice) = nullptr;

using PFN_CreateInstance = VkResult(VKAPI_PTR*)(const VkInstanceCreateInfo*,
                                                const VkAllocationCallbacks*, VkInstance*);
using PFN_CreateDevice = VkResult(VKAPI_PTR*)(VkPhysicalDevice, const VkDeviceCreateInfo*,
                                              const VkAllocationCallbacks*, VkDevice*);
using PFN_GetDeviceQueue = void(VKAPI_PTR*)(VkDevice, uint32_t, uint32_t, VkQueue*);
// The Vulkan 1.1 form. A game that asks for its queues this way left every one
// of them unrecorded, and the present queue's family unknown - which is exactly
// what stood the fallback down in a real game.
using PFN_GetDeviceQueue2 = void(VKAPI_PTR*)(VkDevice, const VkDeviceQueueInfo2*, VkQueue*);
using PFN_EnumeratePhysicalDevices = VkResult(VKAPI_PTR*)(VkInstance, uint32_t*, VkPhysicalDevice*);

PFN_CreateInstance real_create_instance = nullptr;
PFN_CreateDevice real_create_device = nullptr;
PFN_GetDeviceQueue real_get_device_queue = nullptr;
PFN_GetDeviceQueue2 real_get_device_queue2 = nullptr;
PFN_EnumeratePhysicalDevices real_enumerate_physical_devices = nullptr;

// What install() hooked, for uninstall().
std::vector<void*> hooked_targets;

VkResult VKAPI_PTR hooked_create_instance(const VkInstanceCreateInfo* info,
                                          const VkAllocationCallbacks* allocator,
                                          VkInstance* out) {
    const VkResult r = real_create_instance(info, allocator, out);
    if (r == VK_SUCCESS && out && *out && !nr::pe::bridge::creating_device()) {
        std::lock_guard<std::mutex> guard(lock);
        seen.instance = *out;
    }
    return r;
}

// A hook that is installed while the game's vkCreateInstance is still on the
// stack - which is when ReShade's Vulkan layer loads its add-ons - never sees
// that instance. Every app enumerates physical devices right after, and the
// instance is the first argument.
VkResult VKAPI_PTR hooked_enumerate_physical_devices(VkInstance instance, uint32_t* count,
                                                     VkPhysicalDevice* out) {
    const VkResult r = real_enumerate_physical_devices(instance, count, out);
    if (instance && !nr::pe::bridge::creating_device()) {
        std::lock_guard<std::mutex> guard(lock);
        if (!seen.instance) log("[nr] Vulkan instance learned from vkEnumeratePhysicalDevices");
        seen.instance = instance;
    }
    return r;
}

// On Windows the game's device is DXVK's or vkd3d-proton's, created with the features they need.
// The network also needs cooperative matrices, FP8 and the explicit workgroup memory layout;
// vkd3d-proton enables the first two, DXVK none. They are added on the way in when the GPU has all
// three extensions. Anything unexpected in the game's chain (a structure the size table does not
// know) leaves the call exactly as the game made it, and so does a device that then fails to create.
const char* const kNetworkExtensions[] = {
    VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
    VK_EXT_SHADER_FLOAT8_EXTENSION_NAME,
    VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME,
};

struct Augmented {
    std::unique_ptr<nr::DeviceFeatures> features;
    std::vector<const char*> extensions;
    VkDeviceCreateInfo info{};
    nr::binary::Features binary;
};

// Straight from vulkan-1.dll: the DLL's own vk* entry points are resolved later than this runs.
PFN_vkEnumerateDeviceExtensionProperties enumerate_extensions = nullptr;

bool augment(VkPhysicalDevice physical, const VkDeviceCreateInfo* info, Augmented* a) {
    uint32_t n = 0;
    if (!enumerate_extensions || enumerate_extensions(physical, nullptr, &n, nullptr) != VK_SUCCESS) {
        log("[nr] device features left as the game asked: cannot list the GPU's extensions");
        return false;
    }
    std::vector<VkExtensionProperties> have(n);
    if (enumerate_extensions(physical, nullptr, &n, have.data()) != VK_SUCCESS) return false;
    have.resize(n);
    auto listed = [](const char* name, const char* const* list, size_t count) {
        for (size_t i = 0; i < count; ++i)
            if (list[i] && std::strcmp(list[i], name) == 0) return true;
        return false;
    };
    for (const char* e : kNetworkExtensions) {
        bool found = false;
        for (const auto& h : have) found = found || std::strcmp(h.extensionName, e) == 0;
        if (!found) {
            log("[nr] device features left as the game asked: the GPU has no %s", e);
            return false;
        }
    }
    a->extensions.assign(info->ppEnabledExtensionNames,
                         info->ppEnabledExtensionNames + info->enabledExtensionCount);
    for (const char* e : kNetworkExtensions)
        if (!listed(e, a->extensions.data(), a->extensions.size())) a->extensions.push_back(e);
    a->features = std::make_unique<nr::DeviceFeatures>(info->pNext);
    a->features->enable();
    auto* w = a->features->get<VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>(
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR);
    w->workgroupMemoryExplicitLayout = w->workgroupMemoryExplicitLayout8BitAccess =
        w->workgroupMemoryExplicitLayout16BitAccess = VK_TRUE;
    a->info = *info;
    a->info.pNext = a->features->head;
    a->info.enabledExtensionCount = uint32_t(a->extensions.size());
    a->info.ppEnabledExtensionNames = a->extensions.data();
    a->binary.prepare(physical, a->info, a->extensions);
    return true;
}

VkResult VKAPI_PTR hooked_create_device(VkPhysicalDevice physical, const VkDeviceCreateInfo* info,
                                        const VkAllocationCallbacks* allocator, VkDevice* out) {
    // The bridge's own device (nr_pe_bridge.hpp) is not the game's: passed straight through, unrecorded.
    if (nr::pe::bridge::creating_device()) return real_create_device(physical, info, allocator, out);
    // The physical device is known before the call; ReShade's Vulkan layer
    // fires its device events from inside this very call, so anything that
    // runs then must already find it.
    {
        std::lock_guard<std::mutex> guard(lock);
        seen.physical = physical;
        requested_families.clear();
        if (info)
            for (uint32_t i = 0; i < info->queueCreateInfoCount; ++i)
                requested_families.push_back(info->pQueueCreateInfos[i].queueFamilyIndex);
    }
    Augmented augmented;
    bool added = false;
    if (info) {
        try {
            added = augment(physical, info, &augmented);
        } catch (const std::exception& e) {
            log("[nr] device features left as the game asked: %s", e.what());
        }
    }
    VkResult r = real_create_device(physical, added ? &augmented.info : info, allocator, out);
    if (added && r != VK_SUCCESS) {
        log("[nr] device creation with the network's features failed (%d); retrying as the game asked", int(r));
        added = false;
        r = real_create_device(physical, info, allocator, out);
    }
    if (added && r == VK_SUCCESS && augmented.binary.active) nr::binary::mark(*out);
    if (added && r == VK_SUCCESS) log("[nr] network features added to the game's device");
    if (r == VK_SUCCESS && out && *out) {
        {
            std::lock_guard<std::mutex> guard(lock);
            seen.physical = physical;
            seen.device = *out;
            if (added) augmented_devices.push_back(*out);
        }
        log("[nr] Vulkan device created by the game");
        if (device_callback) device_callback(*out);
    }
    return r;
}

void VKAPI_PTR hooked_get_device_queue(VkDevice device, uint32_t family, uint32_t index,
                                       VkQueue* out) {
    real_get_device_queue(device, family, index, out);
    if (!out || !*out) return;
    if (nr::pe::bridge::creating_device() || nr::pe::bridge::is_own_device(device)) return;
    std::lock_guard<std::mutex> guard(lock);
    for (const auto& q : queues)
        if (q.queue == *out) return;
    queues.push_back({device, family, *out});
}

void VKAPI_PTR hooked_get_device_queue2(VkDevice device, const VkDeviceQueueInfo2* info,
                                        VkQueue* out) {
    real_get_device_queue2(device, info, out);
    if (!out || !*out || !info) return;
    if (nr::pe::bridge::creating_device() || nr::pe::bridge::is_own_device(device)) return;
    std::lock_guard<std::mutex> guard(lock);
    for (const auto& q : queues)
        if (q.queue == *out) return;
    queues.push_back({device, info->queueFamilyIndex, *out});
}

}  // namespace

bool install() {
    HMODULE vulkan = GetModuleHandleW(L"vulkan-1.dll");
    if (!vulkan) vulkan = LoadLibraryW(L"vulkan-1.dll");
    if (!vulkan) return false;   // not a Vulkan game; nothing to watch
    enumerate_extensions = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        GetProcAddress(vulkan, "vkEnumerateDeviceExtensionProperties"));
    struct { const char* name; void* detour; void** original; } entries[] = {
        {"vkCreateInstance", reinterpret_cast<void*>(&hooked_create_instance),
         reinterpret_cast<void**>(&real_create_instance)},
        {"vkCreateDevice", reinterpret_cast<void*>(&hooked_create_device),
         reinterpret_cast<void**>(&real_create_device)},
        {"vkGetDeviceQueue", reinterpret_cast<void*>(&hooked_get_device_queue),
         reinterpret_cast<void**>(&real_get_device_queue)},
        {"vkEnumeratePhysicalDevices", reinterpret_cast<void*>(&hooked_enumerate_physical_devices),
         reinterpret_cast<void**>(&real_enumerate_physical_devices)},
        {"vkGetDeviceQueue2", reinterpret_cast<void*>(&hooked_get_device_queue2),
         reinterpret_cast<void**>(&real_get_device_queue2)},
    };
    bool all = true;
    for (const auto& entry : entries) {
        void* target = reinterpret_cast<void*>(GetProcAddress(vulkan, entry.name));
        if (!target || MH_CreateHook(target, entry.detour, entry.original) != MH_OK) {
            all = false;
            continue;
        }
        hooked_targets.push_back(target);
        if (MH_EnableHook(target) != MH_OK) all = false;
    }
    log("[nr] Vulkan device watch %s", all ? "installed" : "incomplete");
    return all;
}

void uninstall() {
    if (hooked_targets.empty()) return;
    for (void* target : hooked_targets) {
        MH_DisableHook(target);
        MH_RemoveHook(target);
    }
    hooked_targets.clear();
    log("[nr] Vulkan device watch removed");
}

DeviceHandles handles() {
    std::lock_guard<std::mutex> guard(lock);
    return seen;
}

void note_handles(VkInstance instance, VkPhysicalDevice physical, VkDevice device) {
    std::lock_guard<std::mutex> guard(lock);
    if (instance) seen.instance = instance;
    if (physical) seen.physical = physical;
    if (device) seen.device = device;
}

bool family_of(VkQueue queue, uint32_t* family) {
    std::lock_guard<std::mutex> guard(lock);
    if (!queue || !family) return false;
    for (const auto& q : queues)
        if (q.queue == queue) { *family = q.family; return true; }
    return false;
}

bool network_features_added(VkDevice device) {
    std::lock_guard<std::mutex> guard(lock);
    for (VkDevice d : augmented_devices)
        if (d == device) return true;
    return false;
}

void set_device_callback(void (*callback)(VkDevice)) {
    std::lock_guard<std::mutex> guard(lock);
    device_callback = callback;
}

bool queue_for(VkDevice device, VkQueue* out, uint32_t* family) {
    if (!device || !out || !family) return false;
    VkPhysicalDevice physical{};
    std::vector<uint32_t> families;
    {
        std::lock_guard<std::mutex> guard(lock);
        physical = seen.physical;
        for (const auto& q : queues) if (q.device == device) families.push_back(q.family);
        if (!physical) return false;
    }
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, props.data());
    auto usable = [&](uint32_t f) {
        return f < count && (props[f].queueFlags & VK_QUEUE_COMPUTE_BIT) && (props[f].queueFlags & VK_QUEUE_GRAPHICS_BIT);
    };
    {
        std::lock_guard<std::mutex> guard(lock);
        for (const auto& q : queues)
            if (q.device == device && usable(q.family)) { *out = q.queue; *family = q.family; return true; }
        families = requested_families;
    }
    for (uint32_t f : families) {
        if (!usable(f)) continue;
        VkQueue q = VK_NULL_HANDLE;
        vkGetDeviceQueue(device, f, 0, &q);   // through the export, so the hook records it
        if (q) { *out = q; *family = f; return true; }
    }
    return false;
}

bool queue(VkQueue* out, uint32_t* family) {
    std::lock_guard<std::mutex> guard(lock);
    if (!out || !family || !seen.device || !seen.physical) return false;
    // The network computes and transfers, and the colour conversion path also
    // needs graphics, so the family has to carry all of it. Asking the physical
    // device rather than assuming family 0 is the difference between working on
    // one driver and working on the machine in front of us.
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(seen.physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(seen.physical, &count, props.data());
    for (const auto& q : queues) {
        if (q.device != seen.device || q.family >= count) continue;
        const VkQueueFlags flags = props[q.family].queueFlags;
        if ((flags & VK_QUEUE_COMPUTE_BIT) && (flags & VK_QUEUE_GRAPHICS_BIT)) {
            *out = q.queue; *family = q.family;
            return true;
        }
    }
    return false;
}

}  // namespace nr::pe::vkdevice
