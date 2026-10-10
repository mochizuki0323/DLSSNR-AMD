#include "nr_pe_bridge.hpp"
#include "nr_pipeline_binary.hpp"

#include "nr_pe_log.hpp"

#include <d3d11_4.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <windows.h>

#include <atomic>
#include <cstring>
#include <iterator>

extern "C" void nr_vk_load(void);

namespace nr::pe::bridge {

struct Job {
    VkCommandBuffer cmd{};
    VkFence fence{};
    enum State { Free, Recording, Ready, Submitted } state{Free};
    uint64_t submitted{};   // submit order, for waiting on the oldest
    uint64_t serial{};      // begin order (every begin of the device), for staleness and recording order
    uint64_t gen{};         // changes at every begin: an old holder's pointer no longer names this use
    const void* owner{};    // what the recording reads and writes besides the images (the network)
    std::function<void()> dropped;   // see Device::on_drop
    std::vector<VkImage> images;   // acquired from the D3D side, released at end()
};

namespace {

// One device per adapter: a process can render on two GPUs, and an imported resource must come from
// the physical device it is imported into.
struct Adapter {
    LUID luid{};
    std::unique_ptr<Device> device;
    std::string why;   // the failure, kept and repeated
};
std::mutex g_device_mutex;
std::vector<Adapter> g_adapters;
uint64_t g_submits{};
uint64_t g_begins{};
// A recorded job older than this many recordings is not run: the runtime recycles its per-recording
// views and descriptor sets after 66 recordings and a released feature's history after 64.
constexpr uint64_t kStaleAfter = 48;
thread_local bool tl_creating = false;
std::atomic<VkDevice> g_own_device{VK_NULL_HANDLE};

const char* const kRequiredExtensions[] = {
    VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
    VK_EXT_SHADER_FLOAT8_EXTENSION_NAME,
    VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
};

template <class T> void release(T*& p) {
    if (p) { p->Release(); p = nullptr; }
}

// D3DCompile from the system's d3dcompiler_47.dll, resolved on first use: a static import would make
// a missing compiler take the whole module down rather than only the depth pass.
using PFN_D3DCompile = HRESULT(WINAPI*)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*, ID3DInclude*, LPCSTR,
                                        LPCSTR, UINT, UINT, ID3DBlob**, ID3DBlob**);

const char kDepthShader[] =
    "Texture2D<float> src : register(t0);\n"
    "RWTexture2D<float> dst : register(u0);\n"
    "[numthreads(8, 8, 1)]\n"
    "void main(uint3 id : SV_DispatchThreadID) {\n"
    "    uint w, h;\n"
    "    dst.GetDimensions(w, h);\n"
    "    if (id.x < w && id.y < h) dst[id.xy] = src.Load(int3(id.xy, 0));\n"
    "}\n";

ID3DBlob* compile_depth_shader(std::string* why) {
    static PFN_D3DCompile compile = [] {
        HMODULE m = LoadLibraryW(L"d3dcompiler_47.dll");
        return m ? reinterpret_cast<PFN_D3DCompile>(reinterpret_cast<void*>(GetProcAddress(m, "D3DCompile")))
                 : nullptr;
    }();
    if (!compile) { *why = "d3dcompiler_47.dll (D3DCompile) is not available"; return nullptr; }
    ID3DBlob* code = nullptr;
    ID3DBlob* errors = nullptr;
    const HRESULT hr = compile(kDepthShader, sizeof kDepthShader - 1, "nr_depth_copy", nullptr, nullptr, "main",
                               "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
    if (FAILED(hr) || !code) {
        *why = "the depth copy shader did not compile";
        if (errors) { *why += ": "; *why += static_cast<const char*>(errors->GetBufferPointer()); }
        release(errors);
        release(code);
        return nullptr;
    }
    release(errors);
    return code;
}

}  // namespace

DXGI_FORMAT shared_format(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_B8G8R8X8_TYPELESS: return DXGI_FORMAT_B8G8R8X8_UNORM;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R32G32B32_TYPELESS: return DXGI_FORMAT_R32G32B32_FLOAT;
    case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R32G32_TYPELESS: return DXGI_FORMAT_R32G32_FLOAT;
    case DXGI_FORMAT_R32_TYPELESS: return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R16_TYPELESS: return DXGI_FORMAT_R16_FLOAT;
    default:
        return vulkan_colour_format_of(format) != VK_FORMAT_UNDEFINED ? format : DXGI_FORMAT_UNKNOWN;
    }
}

DXGI_FORMAT depth_view_format(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D32_FLOAT: case DXGI_FORMAT_R32_FLOAT:
        return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_D24_UNORM_S8_UINT: case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
        return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
        return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_D16_UNORM: case DXGI_FORMAT_R16_UNORM:
        return DXGI_FORMAT_R16_UNORM;
    case DXGI_FORMAT_R16_FLOAT:
        return DXGI_FORMAT_R16_FLOAT;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

// ---- the device ---------------------------------------------------------------------------------

bool creating_device() { return tl_creating; }
bool is_own_device(VkDevice device) { return device && device == g_own_device.load(); }

Device* Device::get(const LUID& luid, std::string* why) {
    std::lock_guard<std::mutex> guard(g_device_mutex);
    for (auto& a : g_adapters) {
        if (std::memcmp(&a.luid, &luid, sizeof luid) != 0) continue;
        if (!a.device) *why = a.why;
        return a.device.get();
    }
    std::unique_ptr<Device> d(new Device());
    tl_creating = true;
    const bool ok = d->create(luid, why);
    tl_creating = false;
    Adapter a;
    a.luid = luid;
    if (ok) {
        // is_own_device: the first one is enough for the hooks that ask (one adapter in practice);
        // a second adapter's device is told apart by tl_creating while it is made.
        VkDevice none = VK_NULL_HANDLE;
        g_own_device.compare_exchange_strong(none, d->device_);
        a.device = std::move(d);
    } else {
        a.why = *why;
        log("[nr] bridge: %s", why->c_str());
    }
    g_adapters.push_back(std::move(a));
    return g_adapters.back().device.get();
}

bool Device::create(const LUID& luid, std::string* why) {
    luid_ = luid;
    nr_vk_load();
    uint32_t api = 0;
    if (vkEnumerateInstanceVersion(&api) != VK_SUCCESS || api < VK_API_VERSION_1_3) {
        *why = "the Vulkan loader is older than 1.3";
        return false;
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "dlssnr-amd";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &instance_) != VK_SUCCESS) {
        *why = "could not create a Vulkan instance";
        return false;
    }
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());
    VkPhysicalDeviceProperties chosen{};
    for (VkPhysicalDevice pd : devices) {
        VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p.pNext = &id;
        vkGetPhysicalDeviceProperties2(pd, &p);
        if (id.deviceLUIDValid && std::memcmp(id.deviceLUID, &luid, sizeof luid) == 0) {
            physical_ = pd;
            chosen = p.properties;
            break;
        }
    }
    if (!physical_) {
        *why = "no Vulkan device has the game's adapter LUID";
        return false;
    }
    if (chosen.apiVersion < VK_API_VERSION_1_3) {
        *why = std::string(chosen.deviceName) + " offers Vulkan below 1.3";
        return false;
    }

    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(physical_, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> have(n);
    vkEnumerateDeviceExtensionProperties(physical_, nullptr, &n, have.data());
    for (const char* e : kRequiredExtensions) {
        bool found = false;
        for (const auto& h : have) found = found || std::strcmp(h.extensionName, e) == 0;
        if (!found) {
            *why = std::string(chosen.deviceName) + " has no " + e;
            return false;
        }
    }
    std::vector<const char*> extensions(std::begin(kRequiredExtensions), std::end(kRequiredExtensions));
    bool robustness_ext = false;
    for (const auto& h : have) {
        if (std::strcmp(h.extensionName, VK_KHR_WIN32_KEYED_MUTEX_EXTENSION_NAME) == 0) keyed_mutex_ = true;
        if (std::strcmp(h.extensionName, VK_EXT_PIPELINE_ROBUSTNESS_EXTENSION_NAME) == 0) robustness_ext = true;
    }
    if (keyed_mutex_) extensions.push_back(VK_KHR_WIN32_KEYED_MUTEX_EXTENSION_NAME);

    // The network's feature set, as nrvk::Context::create enables it for the measurements, plus
    // timeline semaphores (the shared fences) and buffer device addresses (the Windows network reads
    // the activation arena through one).
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceShaderFloat8FeaturesEXT fp8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR wml{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR};
    // nrvk::Context::adopt compiles every pipeline without robust access (VkPipelineRobustnessCreateInfo)
    // when the GPU offers pipeline robustness; on our own device the feature is then enabled with it.
    VkPhysicalDevicePipelineRobustnessFeaturesEXT prf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT};
    VkPhysicalDeviceFeatures2 q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    q.pNext = &coop; coop.pNext = &fp8; fp8.pNext = &f11; f11.pNext = &f12; f12.pNext = &f13; f13.pNext = &wml;
    wml.pNext = &prf;
    vkGetPhysicalDeviceFeatures2(physical_, &q);
    struct { bool have; const char* name; } required[] = {
        {bool(coop.cooperativeMatrix), "cooperativeMatrix"},
        {bool(fp8.shaderFloat8), "shaderFloat8"},
        {bool(fp8.shaderFloat8CooperativeMatrix), "shaderFloat8CooperativeMatrix"},
        {bool(f11.storageBuffer16BitAccess), "storageBuffer16BitAccess"},
        {bool(f12.storageBuffer8BitAccess), "storageBuffer8BitAccess"},
        {bool(f12.shaderFloat16), "shaderFloat16"},
        {bool(f12.shaderInt8), "shaderInt8"},
        {bool(f12.vulkanMemoryModel), "vulkanMemoryModel"},
        {bool(f12.timelineSemaphore), "timelineSemaphore"},
        {bool(f12.bufferDeviceAddress), "bufferDeviceAddress"},
        {bool(f13.subgroupSizeControl), "subgroupSizeControl"},
        {bool(wml.workgroupMemoryExplicitLayout && wml.workgroupMemoryExplicitLayout8BitAccess &&
              wml.workgroupMemoryExplicitLayout16BitAccess), "workgroupMemoryExplicitLayout (8/16-bit)"},
    };
    for (const auto& r : required)
        if (!r.have) { *why = std::string(chosen.deviceName) + " lacks " + r.name; return false; }
    const bool sync2 = f13.synchronization2;

    coop = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    coop.cooperativeMatrix = VK_TRUE;
    fp8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    fp8.shaderFloat8 = fp8.shaderFloat8CooperativeMatrix = VK_TRUE;
    f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    f11.storageBuffer16BitAccess = VK_TRUE;
    f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.storageBuffer8BitAccess = f12.shaderFloat16 = f12.shaderInt8 = f12.vulkanMemoryModel = VK_TRUE;
    f12.timelineSemaphore = f12.bufferDeviceAddress = VK_TRUE;
    f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.subgroupSizeControl = VK_TRUE;
    f13.synchronization2 = sync2 ? VK_TRUE : VK_FALSE;
    wml = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR};
    wml.workgroupMemoryExplicitLayout = wml.workgroupMemoryExplicitLayout8BitAccess =
        wml.workgroupMemoryExplicitLayout16BitAccess = VK_TRUE;
    coop.pNext = &fp8; fp8.pNext = &f11; f11.pNext = &f12; f12.pNext = &f13; f13.pNext = &wml;
    const bool robustness = robustness_ext && prf.pipelineRobustness;
    prf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_ROBUSTNESS_FEATURES_EXT};
    if (robustness) {
        prf.pipelineRobustness = VK_TRUE;
        wml.pNext = &prf;
        extensions.push_back(VK_EXT_PIPELINE_ROBUSTNESS_EXTENSION_NAME);
    }

    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> families(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &qcount, families.data());
    family_ = qcount;
    for (uint32_t i = 0; i < qcount; ++i)
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { family_ = i; break; }
    if (family_ == qcount) { *why = "no compute queue family"; return false; }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = family_; qi.queueCount = 1; qi.pQueuePriorities = &priority;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.pNext = &coop;
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    di.enabledExtensionCount = uint32_t(extensions.size());
    di.ppEnabledExtensionNames = extensions.data();
    nr::binary::Features binary_features;
    binary_features.prepare(physical_, di, extensions);
    const VkResult r = vkCreateDevice(physical_, &di, nullptr, &device_);
    if (r != VK_SUCCESS) {
        *why = "vkCreateDevice failed (" + std::to_string(int(r)) + ")";
        return false;
    }
    if (binary_features.active) nr::binary::mark(device_);
    vkGetDeviceQueue(device_, family_, 0, &queue_);
    vkGetPhysicalDeviceMemoryProperties(physical_, &memory_);
    get_handle_properties_ = reinterpret_cast<PFN_vkGetMemoryWin32HandlePropertiesKHR>(
        vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandlePropertiesKHR"));
    import_semaphore_ = reinterpret_cast<PFN_vkImportSemaphoreWin32HandleKHR>(
        vkGetDeviceProcAddr(device_, "vkImportSemaphoreWin32HandleKHR"));
    if (!get_handle_properties_ || !import_semaphore_) {
        *why = "the driver does not resolve the win32 external memory/semaphore entry points";
        return false;
    }
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.queueFamilyIndex = family_;
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device_, &pi, nullptr, &pool_) != VK_SUCCESS) {
        *why = "could not create the bridge's command pool";
        return false;
    }
    handles_.instance = instance_;
    handles_.physical = physical_;
    handles_.device = device_;
    log("[nr] bridge: own Vulkan device on %s (queue family %u%s), the game's D3D runtime is native",
        chosen.deviceName, family_, keyed_mutex_ ? ", keyed mutexes" : "");
    return true;
}

Device::~Device() {
    // Process exit: the driver may already be gone; only what is certainly safe.
    if (device_) {
        vkDeviceWaitIdle(device_);
        for (auto& j : jobs_) if (j->fence) vkDestroyFence(device_, j->fence, nullptr);
        if (pool_) vkDestroyCommandPool(device_, pool_, nullptr);
        for (auto& t : timelines_) {
            if (t.semaphore) vkDestroySemaphore(device_, t.semaphore, nullptr);
            if (t.fence) t.fence->Release();
            if (t.owner) t.owner->Release();
        }
        nr::binary::forget(device_);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

// ---- shared images ------------------------------------------------------------------------------

bool Device::external_ok(VkFormat format, VkImageUsageFlags usage, VkExternalMemoryHandleTypeFlagBits type,
                         bool* dedicated) {
    VkPhysicalDeviceExternalImageFormatInfo ext{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO};
    ext.handleType = type;
    VkPhysicalDeviceImageFormatInfo2 info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
    info.pNext = &ext;
    info.format = format; info.type = VK_IMAGE_TYPE_2D; info.tiling = VK_IMAGE_TILING_OPTIMAL; info.usage = usage;
    VkExternalImageFormatProperties ep{VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
    VkImageFormatProperties2 props{VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
    props.pNext = &ep;
    if (vkGetPhysicalDeviceImageFormatProperties2(physical_, &info, &props) != VK_SUCCESS) return false;
    const auto f = ep.externalMemoryProperties.externalMemoryFeatures;
    *dedicated = (f & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) != 0;
    return (f & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
}

bool Device::initial_layout(VkImage image, std::string* why) {
    // Once per image, before the D3D side has written anything: the only moment UNDEFINED is true.
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool_; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    std::lock_guard<std::mutex> pool(record_mutex_);
    VkResult r = vkAllocateCommandBuffers(device_, &ai, &cmd);
    if (r != VK_SUCCESS) { *why = "no command buffer for the shared image's layout (" + std::to_string(int(r)) + ")"; return false; }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    r = vkBeginCommandBuffer(cmd, &bi);
    if (r == VK_SUCCESS) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.image = image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &b);
        r = vkEndCommandBuffer(cmd);
    }
    bool submitted = false;
    if (r == VK_SUCCESS) {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        std::lock_guard<std::mutex> guard(queue_mutex_);
        r = vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
        submitted = r == VK_SUCCESS;
        if (submitted) r = vkQueueWaitIdle(queue_);
    }
    // A buffer that may still be pending is left allocated (the pool frees it with the device).
    if (!submitted || r == VK_SUCCESS) vkFreeCommandBuffers(device_, pool_, 1, &cmd);
    if (r != VK_SUCCESS) {
        *why = "the shared image's layout could not be set (" + std::to_string(int(r)) + ")";
        return false;
    }
    return true;
}

bool Device::import_image(HANDLE handle, VkExternalMemoryHandleTypeFlagBits type, uint32_t width, uint32_t height,
                          VkFormat format, bool storage, Image* out, std::string* why) {
    VkExternalMemoryImageCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    ext.handleTypes = type;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.pNext = &ext;
    ci.imageType = VK_IMAGE_TYPE_2D; ci.format = format; ci.extent = {width, height, 1};
    ci.mipLevels = ci.arrayLayers = 1; ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
               (storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage image = VK_NULL_HANDLE;
    if (vkCreateImage(device_, &ci, nullptr, &image) != VK_SUCCESS) {
        *why = "vkCreateImage failed for a shared image";
        return false;
    }
    VkMemoryWin32HandlePropertiesKHR hp{VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
    // The memory types the handle can be imported into: without them there is no valid choice.
    const VkResult pr = get_handle_properties_(device_, type, handle, &hp);
    if (pr != VK_SUCCESS || !hp.memoryTypeBits) {
        vkDestroyImage(device_, image, nullptr);
        *why = "the driver did not say which memory a shared image can be imported into (" + std::to_string(int(pr)) + ")";
        return false;
    }
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device_, image, &req);
    const uint32_t bits = req.memoryTypeBits & hp.memoryTypeBits;
    uint32_t type_index = memory_.memoryTypeCount;
    for (uint32_t i = 0; i < memory_.memoryTypeCount && type_index == memory_.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (memory_.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            type_index = i;
    for (uint32_t i = 0; i < memory_.memoryTypeCount && type_index == memory_.memoryTypeCount; ++i)
        if (bits & (1u << i)) type_index = i;
    if (type_index == memory_.memoryTypeCount) {
        vkDestroyImage(device_, image, nullptr);
        *why = "no memory type can hold the shared image";
        return false;
    }
    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.image = image;
    VkImportMemoryWin32HandleInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
    import.pNext = &dedicated;
    import.handleType = type;
    import.handle = handle;
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.pNext = &import;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type_index;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkResult r = vkAllocateMemory(device_, &alloc, nullptr, &memory);
    if (r == VK_SUCCESS) r = vkBindImageMemory(device_, image, memory, 0);
    if (r != VK_SUCCESS) {
        if (memory) vkFreeMemory(device_, memory, nullptr);
        vkDestroyImage(device_, image, nullptr);
        *why = "importing the shared image failed (" + std::to_string(int(r)) + ")";
        return false;
    }
    if (!initial_layout(image, why)) {
        vkFreeMemory(device_, memory, nullptr);
        vkDestroyImage(device_, image, nullptr);
        return false;
    }
    out->image = image;
    out->memory = memory;
    out->format = format;
    out->usage = ci.usage;
    out->width = width;
    out->height = height;
    out->storage = storage;
    return true;
}

bool Device::create_d3d12(ID3D12Device* device, uint32_t width, uint32_t height, DXGI_FORMAT format, Image* out,
                          std::string* why) {
    destroy(*out);
    const VkFormat vk = vulkan_colour_format_of(format);
    if (vk == VK_FORMAT_UNDEFINED) {
        *why = "DXGI format " + std::to_string(unsigned(format)) + " has no Vulkan counterpart";
        return false;
    }
    D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{format};
    device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof fs);
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(physical_, vk, &fp);
    bool storage = (fs.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) &&
                   (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);
    const auto type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
    const VkImageUsageFlags base =
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    bool dedicated = false;
    if (storage && !external_ok(vk, base | VK_IMAGE_USAGE_STORAGE_BIT, type, &dedicated)) storage = false;
    if (!storage && !external_ok(vk, base, type, &dedicated)) {
        *why = "the driver cannot import a D3D12 texture of VkFormat " + std::to_string(unsigned(vk));
        return false;
    }
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width; desc.Height = height; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
    desc.Format = format; desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = storage ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    ID3D12Resource* resource = nullptr;
    HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                 nullptr, IID_PPV_ARGS(&resource));
    if (FAILED(hr)) {
        char text[96];
        std::snprintf(text, sizeof text, "could not create a shared D3D12 texture (0x%08lX)", (unsigned long)hr);
        *why = text;
        return false;
    }
    HANDLE handle = nullptr;
    hr = device->CreateSharedHandle(resource, nullptr, GENERIC_ALL, nullptr, &handle);
    if (FAILED(hr) || !handle) {
        resource->Release();
        *why = "could not share the D3D12 texture";
        return false;
    }
    const bool ok = import_image(handle, type, width, height, vk, storage, out, why);
    CloseHandle(handle);
    if (!ok) { resource->Release(); return false; }
    out->d3d = resource;
    out->dxgi = format;
    out->device = device;
    return true;
}

bool Device::create_d3d11(ID3D11Device* device, uint32_t width, uint32_t height, DXGI_FORMAT format, Image* out,
                          std::string* why) {
    destroy(*out);
    const VkFormat vk = vulkan_colour_format_of(format);
    if (vk == VK_FORMAT_UNDEFINED) {
        *why = "DXGI format " + std::to_string(unsigned(format)) + " has no Vulkan counterpart";
        return false;
    }
    UINT support = 0;
    device->CheckFormatSupport(format, &support);
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(physical_, vk, &fp);
    bool storage = (support & D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW) &&
                   (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);
    const auto type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
    const VkImageUsageFlags base =
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    bool dedicated = false;
    if (storage && !external_ok(vk, base | VK_IMAGE_USAGE_STORAGE_BIT, type, &dedicated)) storage = false;
    if (!storage && !external_ok(vk, base, type, &dedicated)) {
        *why = "the driver cannot import a D3D11 texture of VkFormat " + std::to_string(unsigned(vk));
        return false;
    }
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width; desc.Height = height; desc.MipLevels = 1; desc.ArraySize = 1;
    desc.Format = format; desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | (storage ? D3D11_BIND_UNORDERED_ACCESS : 0);
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    ID3D11Texture2D* texture = nullptr;
    HRESULT hr = device->CreateTexture2D(&desc, nullptr, &texture);
    if (FAILED(hr)) {
        char text[96];
        std::snprintf(text, sizeof text, "could not create a shared D3D11 texture (0x%08lX)", (unsigned long)hr);
        *why = text;
        return false;
    }
    IDXGIResource1* dxgi = nullptr;
    HANDLE handle = nullptr;
    if (SUCCEEDED(texture->QueryInterface(IID_PPV_ARGS(&dxgi))) && dxgi) {
        hr = dxgi->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr,
                                      &handle);
        dxgi->Release();
    }
    if (FAILED(hr) || !handle) {
        texture->Release();
        *why = "could not share the D3D11 texture";
        return false;
    }
    const bool ok = import_image(handle, type, width, height, vk, storage, out, why);
    CloseHandle(handle);
    if (!ok) { texture->Release(); return false; }
    out->d3d = texture;
    out->dxgi = format;
    out->device = device;
    return true;
}

bool Device::create_d3d10(ID3D10Device* device, uint32_t width, uint32_t height, DXGI_FORMAT format, Image* out,
                          std::string* why) {
    destroy(*out);
    if (!keyed_mutex_) {
        *why = "the driver has no VK_KHR_win32_keyed_mutex (D3D10 has no fences)";
        return false;
    }
    const VkFormat vk = vulkan_colour_format_of(format);
    if (vk == VK_FORMAT_UNDEFINED) {
        *why = "DXGI format " + std::to_string(unsigned(format)) + " has no Vulkan counterpart";
        return false;
    }
    const auto type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT;
    bool dedicated = false;
    if (!external_ok(vk, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                     type, &dedicated)) {
        *why = "the driver cannot import a D3D10 texture of VkFormat " + std::to_string(unsigned(vk));
        return false;
    }
    D3D10_TEXTURE2D_DESC desc{};
    desc.Width = width; desc.Height = height; desc.MipLevels = 1; desc.ArraySize = 1;
    desc.Format = format; desc.SampleDesc.Count = 1;
    desc.Usage = D3D10_USAGE_DEFAULT;
    desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D10_RESOURCE_MISC_SHARED_KEYEDMUTEX;
    ID3D10Texture2D* texture = nullptr;
    HRESULT hr = device->CreateTexture2D(&desc, nullptr, &texture);
    if (FAILED(hr)) {
        char text[96];
        std::snprintf(text, sizeof text, "could not create a shared D3D10 texture (0x%08lX)", (unsigned long)hr);
        *why = text;
        return false;
    }
    IDXGIResource* dxgi = nullptr;
    HANDLE handle = nullptr;
    if (SUCCEEDED(texture->QueryInterface(IID_PPV_ARGS(&dxgi))) && dxgi) {
        hr = dxgi->GetSharedHandle(&handle);   // a legacy (KMT) handle: not closed
        dxgi->Release();
    }
    if (FAILED(hr) || !handle) {
        texture->Release();
        *why = "could not share the D3D10 texture";
        return false;
    }
    if (!import_image(handle, type, width, height, vk, false, out, why)) { texture->Release(); return false; }
    out->d3d = texture;
    out->dxgi = format;
    out->device = device;
    return true;
}

bool Device::submit_keyed(Job* job, uint64_t generation, const std::vector<const Image*>& images,
                          uint64_t acquire_key, uint64_t release_key) {
    std::vector<std::function<void()>> calls;
    {
        std::lock_guard<std::mutex> guard(jobs_mutex_);
        if (job->gen != generation || job->state != Job::Ready) return false;
        if (lost_) { drop_locked(job, calls); }
    }
    if (!calls.empty()) { for (auto& c : calls) c(); return false; }
    std::vector<VkDeviceMemory> memories;
    for (const Image* i : images) if (i && i->memory) memories.push_back(i->memory);
    std::vector<uint64_t> acquire(memories.size(), acquire_key), release(memories.size(), release_key);
    std::vector<uint32_t> timeouts(memories.size(), 1000);
    VkWin32KeyedMutexAcquireReleaseInfoKHR km{VK_STRUCTURE_TYPE_WIN32_KEYED_MUTEX_ACQUIRE_RELEASE_INFO_KHR};
    km.acquireCount = uint32_t(memories.size());
    km.pAcquireSyncs = memories.data();
    km.pAcquireKeys = acquire.data();
    km.pAcquireTimeouts = timeouts.data();
    km.releaseCount = uint32_t(memories.size());
    km.pReleaseSyncs = memories.data();
    km.pReleaseKeys = release.data();
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.pNext = &km;
    si.commandBufferCount = 1; si.pCommandBuffers = &job->cmd;
    VkResult r;
    {
        std::lock_guard<std::mutex> guard(queue_mutex_);
        r = vkQueueSubmit(queue_, 1, &si, job->fence);
    }
    {
        std::lock_guard<std::mutex> guard(jobs_mutex_);
        if (r != VK_SUCCESS) {
            drop_locked(job, calls);
            static bool said = false;
            if (!said) { said = true; log("[nr] bridge: keyed-mutex vkQueueSubmit failed (%d)", int(r)); }
        } else {
            job->state = Job::Submitted;
            job->submitted = ++g_submits;
        }
    }
    for (auto& c : calls) c();
    return r == VK_SUCCESS;
}

void Device::bury(Image& image) {
    if (image.image) vkDestroyImage(device_, image.image, nullptr);
    if (image.memory) vkFreeMemory(device_, image.memory, nullptr);
    ID3D12Resource* r12 = nullptr;
    if (image.d3d && (FAILED(image.d3d->QueryInterface(IID_PPV_ARGS(&r12))) || !r12)) {
        image.d3d->Release();   // D3D10/11 keep a released resource alive for their own pending work
        image.d3d = nullptr;
    }
    if (r12) r12->Release();
    if (image.d3d) {
        std::lock_guard<std::mutex> guard(sync_mutex_);
        uint64_t mark = 0;
        for (auto& t : timelines_) if (t.owner == hand_over_queue_ && t.semaphore) mark = t.value;
        buried_.push_back({image.d3d, mark});
    }
    image = Image{};
}

void Device::sweep_buried_locked(uint64_t completed) {
    for (size_t i = buried_.size(); i-- > 0;)
        if (completed == UINT64_MAX || completed > buried_[i].mark) {
            buried_[i].d3d->Release();
            buried_.erase(buried_.begin() + long(i));
        }
}

void Device::destroy(Image& image) {
    if (image.image) vkDestroyImage(device_, image.image, nullptr);
    if (image.memory) vkFreeMemory(device_, image.memory, nullptr);
    if (image.d3d) image.d3d->Release();
    image = Image{};
}

// ---- recording ------------------------------------------------------------------------------------

Job* Device::begin(std::string* why, const void* owner) {
    record_mutex_.lock();
    Job* job = begin_locked(why, owner);
    if (!job) record_mutex_.unlock();
    return job;
}

bool Device::done(Job* j) {
    return j->state == Job::Submitted && vkGetFenceStatus(device_, j->fence) == VK_SUCCESS;
}

Job* Device::begin_locked(std::string* why, const void* owner) {
    std::lock_guard<std::mutex> guard(jobs_mutex_);
    if (lost_) { *why = "the GPU device was lost"; return nullptr; }
    Job* job = nullptr;
    for (auto& j : jobs_) {
        if (done(j.get())) {
            vkResetFences(device_, 1, &j->fence);
            j->state = Job::Free;
        }
        if (!job && j->state == Job::Free) job = j.get();
    }
    // A job waits in Ready from its frame's evaluate to its command list's execution, so several can
    // be outstanding at once (one per pass, a few frames deep). The pool grows to that; past it the
    // oldest submitted one is waited for.
    if (!job && jobs_.size() < 48) {
        auto j = std::make_unique<Job>();
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool_; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkAllocateCommandBuffers(device_, &ai, &j->cmd) != VK_SUCCESS ||
            vkCreateFence(device_, &fi, nullptr, &j->fence) != VK_SUCCESS) {
            if (j->cmd) vkFreeCommandBuffers(device_, pool_, 1, &j->cmd);
            *why = "could not allocate a bridge command buffer";
            return nullptr;
        }
        jobs_.push_back(std::move(j));
        job = jobs_.back().get();
    }
    if (!job) {
        Job* oldest = nullptr;
        for (auto& j : jobs_)
            if (j->state == Job::Submitted && (!oldest || j->submitted < oldest->submitted)) oldest = j.get();
        if (!oldest || vkWaitForFences(device_, 1, &oldest->fence, VK_TRUE, 1000000000ull) != VK_SUCCESS) {
            *why = oldest ? "a previous neural rendering submit never completed"
                          : "too many neural rendering frames are waiting for their command lists";
            return nullptr;
        }
        vkResetFences(device_, 1, &oldest->fence);
        oldest->state = Job::Free;
        job = oldest;
    }
    if (vkResetCommandBuffer(job->cmd, 0) != VK_SUCCESS) {
        *why = "could not reset a bridge command buffer";
        return nullptr;
    }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(job->cmd, &bi) != VK_SUCCESS) {
        *why = "could not begin a bridge command buffer";
        return nullptr;
    }
    job->state = Job::Recording;
    job->serial = ++g_begins;
    ++job->gen;
    job->owner = owner;
    job->dropped = nullptr;
    job->images.clear();
    return job;
}

VkCommandBuffer Device::command_buffer(Job* job) const { return job ? job->cmd : VK_NULL_HANDLE; }

uint64_t Device::generation(const Job* job) const { return job ? job->gen : 0; }

void Device::on_drop(Job* job, std::function<void()> f) {
    std::lock_guard<std::mutex> guard(jobs_mutex_);
    job->dropped = std::move(f);
}

// jobs_mutex_ held. A recorded job that will not run as recorded: free, and its owner told (outside the lock).
void Device::drop_locked(Job* job, std::vector<std::function<void()>>& calls) {
    const bool recorded = job->state == Job::Ready || job->state == Job::Recording;
    job->state = Job::Free;
    if (recorded && job->dropped) calls.push_back(std::move(job->dropped));
    job->dropped = nullptr;
    job->owner = nullptr;
}

namespace {
// The hand-over of an external image: an acquire from VK_QUEUE_FAMILY_EXTERNAL before our work and a
// release back to it after, both GENERAL -> GENERAL so the contents survive.
void ownership(VkCommandBuffer cmd, const std::vector<VkImage>& images, uint32_t family, bool acquire) {
    if (images.empty()) return;
    std::vector<VkImageMemoryBarrier> b(images.size(), VkImageMemoryBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER});
    for (size_t i = 0; i < images.size(); ++i) {
        b[i].oldLayout = b[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b[i].srcQueueFamilyIndex = acquire ? VK_QUEUE_FAMILY_EXTERNAL : family;
        b[i].dstQueueFamilyIndex = acquire ? family : VK_QUEUE_FAMILY_EXTERNAL;
        b[i].srcAccessMask = acquire ? 0 : VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b[i].dstAccessMask = acquire ? VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT : 0;
        b[i].image = images[i];
        b[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    vkCmdPipelineBarrier(cmd, acquire ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         acquire ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
                         nullptr, 0, nullptr, uint32_t(b.size()), b.data());
}
}  // namespace

void Device::acquire(Job* job, const std::vector<const Image*>& images) {
    for (const Image* i : images)
        if (i && i->image) job->images.push_back(i->image);
    ownership(job->cmd, job->images, family_, true);
}

bool Device::end(Job* job, std::string* why) {
    ownership(job->cmd, job->images, family_, false);
    if (vkEndCommandBuffer(job->cmd) != VK_SUCCESS) {
        *why = "could not close the bridge command buffer";
        abort(job);
        return false;
    }
    {
        std::lock_guard<std::mutex> guard(jobs_mutex_);
        job->state = Job::Ready;
    }
    record_mutex_.unlock();
    return true;
}

void Device::abort(Job* job) {
    std::vector<std::function<void()>> calls;
    {
        std::lock_guard<std::mutex> guard(jobs_mutex_);
        drop_locked(job, calls);
    }
    record_mutex_.unlock();
    for (auto& c : calls) c();
}

void Device::discard(Job* job, uint64_t generation) {
    if (!job) return;
    std::vector<std::function<void()>> calls;
    {
        std::lock_guard<std::mutex> guard(jobs_mutex_);
        if (job->gen != generation || job->state == Job::Submitted || job->state == Job::Free) return;
        drop_locked(job, calls);
    }
    for (auto& c : calls) c();
}

void Device::cancel(const void* owner) {
    std::lock_guard<std::mutex> guard(jobs_mutex_);
    for (auto& j : jobs_)
        if (j->owner == owner && j->state == Job::Ready) { j->state = Job::Free; j->dropped = nullptr; j->owner = nullptr; }
}

bool Device::busy(const void* owner) {
    std::lock_guard<std::mutex> guard(jobs_mutex_);
    for (auto& j : jobs_) {
        if (j->owner != owner || j->state == Job::Free) continue;
        if (j->state == Job::Submitted && vkGetFenceStatus(device_, j->fence) == VK_SUCCESS) continue;
        return true;
    }
    return false;
}

bool Device::busy(VkImage image) {
    std::lock_guard<std::mutex> guard(jobs_mutex_);
    for (auto& j : jobs_) {
        if (j->state == Job::Free) continue;
        if (j->state == Job::Submitted && vkGetFenceStatus(device_, j->fence) == VK_SUCCESS) continue;
        for (VkImage i : j->images) if (i == image) return true;
    }
    return false;
}

void Device::lost(const char* why) {
    std::lock_guard<std::mutex> guard(jobs_mutex_);
    if (!lost_) log("[nr] bridge: %s; nothing more is handed to the network", why);
    lost_ = true;
}

bool Device::submit(Job* job, uint64_t generation, VkSemaphore wait, uint64_t wait_value, VkSemaphore signal,
                    uint64_t signal_value) {
    std::vector<std::function<void()>> calls;
    bool out_of_order = false;
    {
        std::lock_guard<std::mutex> guard(jobs_mutex_);
        if (!job || job->gen != generation || job->state != Job::Ready) return false;
        if (lost_ || g_begins - job->serial > kStaleAfter) {
            // Recorded too long ago: what the runtime recorded it against may have been recycled since.
            static bool said = false;
            if (!lost_ && !said) {
                said = true;
                log("[nr] bridge: a frame recorded %llu recordings ago was not run (its command list ran late)",
                    static_cast<unsigned long long>(g_begins - job->serial));
            }
            drop_locked(job, calls);
        } else {
            // Its owner's history was recorded in an order the GPU will not run it in.
            for (auto& j : jobs_)
                if (j.get() != job && j->owner == job->owner && j->state == Job::Ready && j->serial < job->serial)
                    out_of_order = true;
        }
    }
    if (!calls.empty()) { for (auto& c : calls) c(); return false; }
    VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    ts.waitSemaphoreValueCount = 1; ts.pWaitSemaphoreValues = &wait_value;
    ts.signalSemaphoreValueCount = 1; ts.pSignalSemaphoreValues = &signal_value;
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.pNext = &ts;
    si.waitSemaphoreCount = 1; si.pWaitSemaphores = &wait; si.pWaitDstStageMask = &stage;
    si.commandBufferCount = 1; si.pCommandBuffers = &job->cmd;
    si.signalSemaphoreCount = 1; si.pSignalSemaphores = &signal;
    VkResult r;
    {
        std::lock_guard<std::mutex> guard(queue_mutex_);
        r = vkQueueSubmit(queue_, 1, &si, job->fence);
    }
    {
        std::lock_guard<std::mutex> guard(jobs_mutex_);
        if (r != VK_SUCCESS) {
            drop_locked(job, calls);
            static bool said = false;
            if (!said) { said = true; log("[nr] bridge: vkQueueSubmit failed (%d)", int(r)); }
            if (r == VK_ERROR_DEVICE_LOST && !lost_) { lost_ = true; log("[nr] bridge: our Vulkan device was lost"); }
        } else {
            job->state = Job::Submitted;
            job->submitted = ++g_submits;
            if (out_of_order && job->dropped) {
                static bool said = false;
                if (!said) { said = true; log("[nr] bridge: frames ran in another order than they were recorded; history restarts"); }
                calls.push_back(job->dropped);
            }
        }
    }
    for (auto& c : calls) c();
    return r == VK_SUCCESS;
}


bool Device::wait_idle() {
    std::vector<VkFence> fences;
    {
        std::lock_guard<std::mutex> guard(jobs_mutex_);
        for (auto& j : jobs_) if (j->state == Job::Submitted) fences.push_back(j->fence);
    }
    if (fences.empty()) return true;
    return vkWaitForFences(device_, uint32_t(fences.size()), fences.data(), VK_TRUE, 1000000000ull) == VK_SUCCESS;
}

// ---- fences ---------------------------------------------------------------------------------------

VkSemaphore Device::import_fence(HANDLE handle, std::string* why) {
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkPhysicalDeviceExternalSemaphoreInfo info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO};
    info.pNext = &type;
    info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
    VkExternalSemaphoreProperties props{VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
    vkGetPhysicalDeviceExternalSemaphoreProperties(physical_, &info, &props);
    if (!(props.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT)) {
        *why = "the driver cannot import a D3D fence as a timeline semaphore";
        return VK_NULL_HANDLE;
    }
    VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    ci.pNext = &type;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    if (vkCreateSemaphore(device_, &ci, nullptr, &semaphore) != VK_SUCCESS) {
        *why = "could not create the bridge semaphore";
        return VK_NULL_HANDLE;
    }
    VkImportSemaphoreWin32HandleInfoKHR import{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
    import.semaphore = semaphore;
    import.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
    import.handle = handle;
    const VkResult r = import_semaphore_(device_, &import);
    if (r != VK_SUCCESS) {
        vkDestroySemaphore(device_, semaphore, nullptr);
        *why = "importing the shared fence failed (" + std::to_string(int(r)) + ")";
        return VK_NULL_HANDLE;
    }
    return semaphore;
}

Device::Timeline* Device::timeline_d3d12(ID3D12CommandQueue* queue) {
    for (auto& t : timelines_) if (t.owner == queue) return t.semaphore ? &t : nullptr;
    Timeline t{};
    t.owner = queue;
    queue->AddRef();   // held: a later queue at the same address must not be taken for this one
    std::string why;
    ID3D12Device* device = nullptr;
    ID3D12Fence* fence = nullptr;
    HANDLE handle = nullptr;
    if (FAILED(queue->GetDevice(IID_PPV_ARGS(&device))) || !device) why = "the queue has no device";
    else if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence))) || !fence)
        why = "could not create a shared D3D12 fence";
    else if (FAILED(device->CreateSharedHandle(fence, nullptr, GENERIC_ALL, nullptr, &handle)) || !handle)
        why = "could not share the D3D12 fence";
    else
        t.semaphore = import_fence(handle, &why);
    if (handle) CloseHandle(handle);
    if (device) device->Release();
    if (t.semaphore) t.fence = fence;
    else if (fence) fence->Release();
    if (!t.semaphore) log("[nr] bridge: no GPU hand-over on this queue: %s", why.c_str());
    timelines_.push_back(t);
    return timelines_.back().semaphore ? &timelines_.back() : nullptr;
}

Device::Timeline* Device::timeline_d3d11(ID3D11DeviceContext* context) {
    for (auto& t : timelines_) if (t.owner == context) return t.semaphore ? &t : nullptr;
    Timeline t{};
    t.owner = context;
    context->AddRef();
    std::string why;
    ID3D11Device* device = nullptr;
    ID3D11Device5* device5 = nullptr;
    ID3D11Fence* fence = nullptr;
    HANDLE handle = nullptr;
    context->GetDevice(&device);
    if (!device || FAILED(device->QueryInterface(IID_PPV_ARGS(&device5))) || !device5)
        why = "the D3D11 runtime has no fences (ID3D11Device5)";
    else if (FAILED(device5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence))) || !fence)
        why = "could not create a shared D3D11 fence";
    else if (FAILED(fence->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &handle)) || !handle)
        why = "could not share the D3D11 fence";
    else
        t.semaphore = import_fence(handle, &why);
    if (handle) CloseHandle(handle);
    if (device5) device5->Release();
    if (device) device->Release();
    if (t.semaphore) t.fence = fence;
    else if (fence) fence->Release();
    if (!t.semaphore) log("[nr] bridge: no GPU hand-over on the D3D11 context: %s", why.c_str());
    timelines_.push_back(t);
    return timelines_.back().semaphore ? &timelines_.back() : nullptr;
}

void Device::forget_hand_over_queue() {
    std::lock_guard<std::mutex> guard(sync_mutex_);
    if (hand_over_queue_) hand_over_queue_->Release();
    hand_over_queue_ = nullptr;
}

uint64_t Device::hand_over_mark() {
    std::lock_guard<std::mutex> guard(sync_mutex_);
    for (auto& t : timelines_) if (t.owner == hand_over_queue_ && t.semaphore) return t.value;
    return 0;
}

bool Device::hand_over_passed(uint64_t mark) {
    std::lock_guard<std::mutex> guard(sync_mutex_);
    for (auto& t : timelines_) {
        if (t.owner != hand_over_queue_ || !t.semaphore) continue;
        // `mark` is a frame's `done`; the next frame's `ready` is signalled by the D3D side after that
        // frame's copy out. Nothing later yet: not passed.
        uint64_t completed = 0;
        ID3D12Fence* f12 = nullptr;
        ID3D11Fence* f11 = nullptr;
        if (SUCCEEDED(t.fence->QueryInterface(IID_PPV_ARGS(&f12))) && f12) { completed = f12->GetCompletedValue(); f12->Release(); }
        else if (SUCCEEDED(t.fence->QueryInterface(IID_PPV_ARGS(&f11))) && f11) { completed = f11->GetCompletedValue(); f11->Release(); }
        if (completed == UINT64_MAX) return true;   // the device is gone: nothing of it is running
        return completed > mark;
    }
    return mark == 0;
}

bool Device::run_between(ID3D12CommandQueue* queue, Job* job, uint64_t generation) {
    std::lock_guard<std::mutex> guard(sync_mutex_);
    if (!hand_over_queue_) { hand_over_queue_ = queue; queue->AddRef(); }
    if (queue != hand_over_queue_) {
        static bool said = false;
        if (!said) { said = true; log("[nr] bridge: a frame on a second D3D12 queue passes through (one queue carries the network)"); }
        discard(job, generation);
        return false;
    }
    Timeline* t = timeline_d3d12(queue);
    if (!t) { discard(job, generation); return false; }
    auto* fence = static_cast<ID3D12Fence*>(t->fence);
    const uint64_t ready = t->value + 1, done = t->value + 2;
    HRESULT hr = queue->Signal(fence, ready);
    if (FAILED(hr)) {
        log("[nr] bridge: the D3D12 queue refused a signal (0x%08lX)", (unsigned long)hr);
        if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) lost("the game's D3D12 device was removed");
        discard(job, generation);
        return false;
    }
    t->value = done;
    if (!buried_.empty()) sweep_buried_locked(fence->GetCompletedValue());
    const bool ok = submit(job, generation, t->semaphore, ready, t->semaphore, done);
    if (!ok) {
        // Nothing of ours will signal `done`: the D3D12 side does, so its wait below is met.
        hr = queue->Signal(fence, done);
        if (FAILED(hr)) {
            log("[nr] bridge: the D3D12 queue refused a signal (0x%08lX)", (unsigned long)hr);
            return false;   // no wait: it could never be met
        }
    }
    hr = queue->Wait(fence, done);
    if (FAILED(hr)) {
        // The rest of the queue would not wait for the network: what it reads may be unfinished.
        log("[nr] bridge: the D3D12 queue refused to wait for the network (0x%08lX)", (unsigned long)hr);
        lost("the D3D12 queue does not wait on the shared fence");
    }
    return ok;
}

bool Device::run_between(ID3D11DeviceContext* context, Job* job, uint64_t generation) {
    std::lock_guard<std::mutex> guard(sync_mutex_);
    ID3D11DeviceContext4* c4 = nullptr;
    if (FAILED(context->QueryInterface(IID_PPV_ARGS(&c4))) || !c4) { discard(job, generation); return false; }
    if (!hand_over_queue_) { hand_over_queue_ = context; context->AddRef(); }
    if (context != hand_over_queue_) {
        c4->Release();
        discard(job, generation);
        return false;
    }
    Timeline* t = timeline_d3d11(context);
    if (!t) { c4->Release(); discard(job, generation); return false; }
    auto* fence = static_cast<ID3D11Fence*>(t->fence);
    const uint64_t ready = t->value + 1, done = t->value + 2;
    HRESULT hr = c4->Signal(fence, ready);
    if (FAILED(hr)) {
        log("[nr] bridge: the D3D11 context refused a signal (0x%08lX)", (unsigned long)hr);
        c4->Release();
        discard(job, generation);
        return false;
    }
    t->value = done;
    // The signal has to reach the GPU queue before our queue can see it; D3D11 would otherwise hold it
    // in its batch until the next flush, with our queue waiting on it.
    c4->Flush();
    const bool ok = submit(job, generation, t->semaphore, ready, t->semaphore, done);
    if (!ok && FAILED(hr = c4->Signal(fence, done))) {
        log("[nr] bridge: the D3D11 context refused a signal (0x%08lX)", (unsigned long)hr);
        c4->Release();
        return false;
    }
    hr = c4->Wait(fence, done);
    if (FAILED(hr)) {
        log("[nr] bridge: the D3D11 context refused to wait for the network (0x%08lX)", (unsigned long)hr);
        lost("the D3D11 context does not wait on the shared fence");
    }
    c4->Release();
    return ok;
}


// ---- depth copies ----------------------------------------------------------------------------------

DepthCopyD3D12::~DepthCopyD3D12() {
    release(heap_);
    release(pso_);
    release(root_);
}

bool DepthCopyD3D12::ensure(ID3D12Device* device, std::string* why) {
    if (pso_) return true;
    if (failed_) { *why = "the depth copy is unavailable"; return false; }
    failed_ = true;
    ID3DBlob* code = compile_depth_shader(why);
    if (!code) return false;
    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].OffsetInDescriptorsFromTableStart = 1;
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable.NumDescriptorRanges = 2;
    param.DescriptorTable.pDescriptorRanges = ranges;
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rd{};
    rd.NumParameters = 1;
    rd.pParameters = &param;
    ID3DBlob* blob = nullptr;
    ID3DBlob* errors = nullptr;
    HRESULT hr = D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors);
    if (SUCCEEDED(hr))
        hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root_));
    release(blob);
    release(errors);
    if (SUCCEEDED(hr)) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = root_;
        pd.CS.pShaderBytecode = code->GetBufferPointer();
        pd.CS.BytecodeLength = code->GetBufferSize();
        hr = device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso_));
    }
    release(code);
    if (SUCCEEDED(hr)) {
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = 2 * 256;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr = device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap_));
    }
    if (FAILED(hr)) {
        release(heap_); release(pso_); release(root_);
        *why = "could not create the D3D12 depth copy pipeline";
        return false;
    }
    increment_ = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    failed_ = false;
    return true;
}

bool DepthCopyD3D12::record(ID3D12Device* device, ID3D12GraphicsCommandList* list, ID3D12Resource* src,
                            ID3D12Resource* dst, std::string* why) {
    const D3D12_RESOURCE_DESC sd = src->GetDesc(), dd = dst->GetDesc();
    const DXGI_FORMAT view = depth_view_format(sd.Format);
    if (view == DXGI_FORMAT_UNKNOWN || (sd.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)) {
        *why = "the depth buffer (DXGI format " + std::to_string(unsigned(sd.Format)) + ") cannot be read";
        return false;
    }
    if (!ensure(device, why)) return false;
    // A slot pair per recording, in a ring far deeper than the frames a GPU can be behind: the
    // descriptors are read when the list executes, not when it is recorded.
    const uint32_t slot = (next_++ % 256) * 2;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap_->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = heap_->GetGPUDescriptorHandleForHeapStart();
    cpu.ptr += SIZE_T(slot) * increment_;
    gpu.ptr += UINT64(slot) * increment_;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = view;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(src, &srv, cpu);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R32_FLOAT;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu_uav = cpu;
    cpu_uav.ptr += increment_;
    device->CreateUnorderedAccessView(dst, nullptr, &uav, cpu_uav);
    list->SetDescriptorHeaps(1, &heap_);
    list->SetComputeRootSignature(root_);
    list->SetPipelineState(pso_);
    list->SetComputeRootDescriptorTable(0, gpu);
    list->Dispatch(UINT((dd.Width + 7) / 8), (dd.Height + 7) / 8, 1);
    return true;
}

DepthCopyD3D11::~DepthCopyD3D11() {
    drop_views();
    release(cs_);
}

void DepthCopyD3D11::drop_views() {
    release(srv_);
    release(uav_);
    src_ = nullptr;
    dst_ = nullptr;
}

bool DepthCopyD3D11::run(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Resource* src,
                         ID3D11Texture2D* dst, std::string* why) {
    if (!cs_) {
        if (failed_) { *why = "the depth copy is unavailable"; return false; }
        failed_ = true;
        ID3DBlob* code = compile_depth_shader(why);
        if (!code) return false;
        const HRESULT hr = device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &cs_);
        release(code);
        if (FAILED(hr)) { *why = "could not create the D3D11 depth copy shader"; return false; }
        failed_ = false;
    }
    if (src != src_ || dst != dst_) {
        drop_views();
        ID3D11Texture2D* tex = nullptr;
        D3D11_TEXTURE2D_DESC sd{};
        if (FAILED(src->QueryInterface(IID_PPV_ARGS(&tex))) || !tex) {
            *why = "the depth buffer is not a 2D texture";
            return false;
        }
        tex->GetDesc(&sd);
        tex->Release();
        const DXGI_FORMAT view = depth_view_format(sd.Format);
        if (view == DXGI_FORMAT_UNKNOWN || !(sd.BindFlags & D3D11_BIND_SHADER_RESOURCE)) {
            *why = "the depth buffer (DXGI format " + std::to_string(unsigned(sd.Format)) + ") cannot be read";
            return false;
        }
        D3D11_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = view;
        srv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srv.Texture2D.MipLevels = 1;
        D3D11_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_R32_FLOAT;
        uav.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
        if (FAILED(device->CreateShaderResourceView(src, &srv, &srv_)) ||
            FAILED(device->CreateUnorderedAccessView(dst, &uav, &uav_))) {
            drop_views();
            *why = "could not view the depth buffer";
            return false;
        }
        src_ = src;
        dst_ = dst;
    }
    // The compute state this touches, put back afterwards: on the ReShade route the game's own state
    // has been captured by ReShade, but nothing says the next effect sets all of it.
    ID3D11ComputeShader* old_cs = nullptr;
    ID3D11ClassInstance* old_instances[256] = {};
    UINT old_instance_count = 256;
    ID3D11ShaderResourceView* old_srv = nullptr;
    ID3D11UnorderedAccessView* old_uav = nullptr;
    // A depth buffer still bound for writing cannot be read: D3D11 would quietly null the view. Its
    // depth-stencil binding is taken off for the copy and put back after.
    ID3D11RenderTargetView* rtvs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView* dsv = nullptr;
    context->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtvs, &dsv);
    bool unbound = false;
    // Only the slots in use, so render-target-and-UAV bindings above them are left alone.
    UINT bound_rtvs = 0;
    for (UINT i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i) if (rtvs[i]) bound_rtvs = i + 1;
    if (dsv) {
        ID3D11Resource* bound = nullptr;
        dsv->GetResource(&bound);
        unbound = bound == src;
        if (bound) bound->Release();
        if (unbound) context->OMSetRenderTargets(bound_rtvs, rtvs, nullptr);
    }
    context->CSGetShader(&old_cs, old_instances, &old_instance_count);
    context->CSGetShaderResources(0, 1, &old_srv);
    context->CSGetUnorderedAccessViews(0, 1, &old_uav);
    context->CSSetShader(cs_, nullptr, 0);
    context->CSSetShaderResources(0, 1, &srv_);
    context->CSSetUnorderedAccessViews(0, 1, &uav_, nullptr);
    D3D11_TEXTURE2D_DESC dd{};
    dst->GetDesc(&dd);
    context->Dispatch((dd.Width + 7) / 8, (dd.Height + 7) / 8, 1);
    ID3D11UnorderedAccessView* none = nullptr;
    context->CSSetUnorderedAccessViews(0, 1, old_uav ? &old_uav : &none, nullptr);
    ID3D11ShaderResourceView* none_srv = nullptr;
    context->CSSetShaderResources(0, 1, old_srv ? &old_srv : &none_srv);
    context->CSSetShader(old_cs, old_instances, old_instance_count);
    release(old_cs);
    for (UINT i = 0; i < old_instance_count; ++i) release(old_instances[i]);
    release(old_srv);
    release(old_uav);
    if (unbound) context->OMSetRenderTargets(bound_rtvs, rtvs, dsv);
    for (auto*& r : rtvs) release(r);
    release(dsv);
    return true;
}

}  // namespace nr::pe::bridge
