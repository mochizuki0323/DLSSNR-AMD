#pragma once
#include "nr_pal_binary.hpp"
#include "nr_log.hpp"
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>

namespace nr::binary {
// The ACO network for the whole process: the bundle directory (records/ + shaders/), empty when off.
// NR_ACO_DIR (tests) wins; otherwise the product's switch sets it with configure() before the first
// device is created. disable() turns it off for the rest of the process (a GPU, driver or device that
// cannot take it): every network built afterwards uses the driver's own compiler.
struct Mode {
    std::mutex m;
    bool configured = false, disabled = false;
    std::string dir, why;
    std::vector<std::string> shells;   // template SPIR-V directories, searched in order
    std::string cache;                 // where imported binaries are kept between runs; empty: none
    std::map<std::string,std::string> aliases;   // a template's name -> the native SPIR-V that serves as it
    uint64_t identity = 0;             // the GPU and driver the device was made on (cache key part)
};
inline Mode& mode() { static Mode s; return s; }
inline void configure(const std::string& dir, std::vector<std::string> shells, const std::string& cache,
                      std::map<std::string,std::string> aliases) {
    auto& s=mode();std::lock_guard<std::mutex> l(s.m);
    s.configured=true;s.dir=dir;s.shells=std::move(shells);s.cache=cache;
    s.aliases=std::move(aliases);
}
inline std::string alias_of(const std::string& name) {
    auto& s=mode();std::lock_guard<std::mutex> l(s.m);
    auto it=s.aliases.find(name);return it==s.aliases.end()?std::string():it->second;
}
inline std::string cache_dir() {
    if(const char* p=std::getenv("NR_ACO_CACHE")) if(*p) return p;
    auto& s=mode();std::lock_guard<std::mutex> l(s.m);return s.cache;
}
inline uint64_t identity() { auto& s=mode();std::lock_guard<std::mutex> l(s.m);return s.identity; }
inline std::string directory() {
    auto& s=mode();std::lock_guard<std::mutex> l(s.m);
    if(s.disabled)return "";
    if(const char* p=std::getenv("NR_ACO_DIR")) if(*p) return p;
    return s.configured?s.dir:"";
}
inline void disable(const std::string& why) {
    auto& s=mode();std::lock_guard<std::mutex> l(s.m);
    if(!s.disabled){s.disabled=true;s.why=why;}
}
// Why the ACO network is off although it was asked for; empty when it was not asked for or is on.
inline std::string disabled_why() { auto& s=mode();std::lock_guard<std::mutex> l(s.m);return s.disabled?s.why:""; }
inline std::vector<std::string> shell_dirs() {
    if(const char* p=std::getenv("NR_ACO_SHELL_DIR")) if(*p) return {p};
    auto& s=mode();std::lock_guard<std::mutex> l(s.m);return s.shells;
}
inline std::mutex devices_mutex;
inline std::set<VkDevice> enabled_devices;
inline void mark(VkDevice d) {std::lock_guard<std::mutex> l(devices_mutex);enabled_devices.insert(d);}
inline void forget(VkDevice d) {std::lock_guard<std::mutex> l(devices_mutex);enabled_devices.erase(d);}
inline bool enabled(VkDevice d) {std::lock_guard<std::mutex> l(devices_mutex);return enabled_devices.count(d)!=0;}
inline void check(VkResult r,const char* what) {
    if(r!=VK_SUCCESS)throw std::runtime_error(std::string(what)+": "+std::to_string(int(r)));
}
struct Features {
    VkPhysicalDevicePipelineBinaryFeaturesKHR pb{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR};
    VkPhysicalDeviceMaintenance5FeaturesKHR m5{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR};
    bool active=false;
    // A GPU, driver or device chain that cannot take the ACO network turns it off for the process
    // (disable): the device is created as it would be without it, never refused.
    void prepare(VkPhysicalDevice physical,VkDeviceCreateInfo& ci,std::vector<const char*>& extensions) {
        if(directory().empty())return;
        VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,&drv};
        vkGetPhysicalDeviceProperties2(physical,&props);
        if(props.properties.vendorID!=0x1002 || drv.driverID!=VK_DRIVER_ID_AMD_PROPRIETARY)
            return disable("the ACO network needs AMD's own Windows Vulkan driver");
        uint32_t n=0;
        if(vkEnumerateDeviceExtensionProperties(physical,nullptr,&n,nullptr)!=VK_SUCCESS)return disable("cannot list the device extensions");
        std::vector<VkExtensionProperties> have(n);
        if(vkEnumerateDeviceExtensionProperties(physical,nullptr,&n,have.data())!=VK_SUCCESS)return disable("cannot list the device extensions");
        auto listed=[&](const char* name){for(uint32_t i=0;i<n;++i)if(!std::strcmp(have[i].extensionName,name))return true;return false;};
        if(!listed(VK_KHR_PIPELINE_BINARY_EXTENSION_NAME) || !listed(VK_KHR_MAINTENANCE_5_EXTENSION_NAME))
            return disable("the driver has no VK_KHR_pipeline_binary / VK_KHR_maintenance5");
        pb.pNext=&m5;VkPhysicalDeviceFeatures2 q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,&pb};
        vkGetPhysicalDeviceFeatures2(physical,&q);
        if(!pb.pipelineBinaries || !m5.maintenance5)return disable("the driver offers no pipelineBinaries / maintenance5");
        bool have_pb=false,have_m5=false;
        for(auto* s=static_cast<const VkBaseInStructure*>(ci.pNext);s;s=s->pNext) {
            if(s->sType==pb.sType) {have_pb=true;if(!reinterpret_cast<const VkPhysicalDevicePipelineBinaryFeaturesKHR*>(s)->pipelineBinaries)
                                                     return disable("the device is created with pipelineBinaries off");}
            if(s->sType==m5.sType) {have_m5=true;if(!reinterpret_cast<const VkPhysicalDeviceMaintenance5FeaturesKHR*>(s)->maintenance5)
                                                     return disable("the device is created with maintenance5 off");}
            if(s->sType==VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES) {
                have_m5=true;if(!reinterpret_cast<const VkPhysicalDeviceVulkan14Features*>(s)->maintenance5)
                    return disable("the device is created with maintenance5 off");
            }
        }
        auto add=[&](const char* name){for(auto* e:extensions)if(!std::strcmp(e,name))return;extensions.push_back(name);};
        add(VK_KHR_PIPELINE_BINARY_EXTENSION_NAME);add(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
        if(!have_m5){m5.pNext=const_cast<void*>(ci.pNext);ci.pNext=&m5;}
        if(!have_pb){pb.pNext=const_cast<void*>(ci.pNext);ci.pNext=&pb;}
        ci.enabledExtensionCount=uint32_t(extensions.size());ci.ppEnabledExtensionNames=extensions.data();active=true;
        {
            auto& st=mode();std::lock_guard<std::mutex> l(st.m);
            uint64_t id=pal::hash(props.properties.pipelineCacheUUID,VK_UUID_SIZE);
            id^=(uint64_t(props.properties.driverVersion)<<32)^props.properties.deviceID;
            st.identity=pal::hash(&id,sizeof id);
        }
        nr::logf("[nr] ACO network: pipeline binaries on %s, %s (driverVersion 0x%08x)",props.properties.deviceName,
                 drv.driverInfo,props.properties.driverVersion);
    }
};
struct Functions {
    VkDevice device{};
    PFN_vkCreatePipelineBinariesKHR create{};PFN_vkDestroyPipelineBinaryKHR destroy{};
    PFN_vkGetPipelineBinaryDataKHR data{};PFN_vkReleaseCapturedPipelineDataKHR release{};
    explicit Functions(VkDevice d):device(d) {
        create=reinterpret_cast<PFN_vkCreatePipelineBinariesKHR>(vkGetDeviceProcAddr(d,"vkCreatePipelineBinariesKHR"));
        destroy=reinterpret_cast<PFN_vkDestroyPipelineBinaryKHR>(vkGetDeviceProcAddr(d,"vkDestroyPipelineBinaryKHR"));
        data=reinterpret_cast<PFN_vkGetPipelineBinaryDataKHR>(vkGetDeviceProcAddr(d,"vkGetPipelineBinaryDataKHR"));
        release=reinterpret_cast<PFN_vkReleaseCapturedPipelineDataKHR>(vkGetDeviceProcAddr(d,"vkReleaseCapturedPipelineDataKHR"));
        pal::require(create && destroy && data && release,"pipeline binary functions unavailable");
    }
};
struct Binaries {
    Functions& f;std::vector<VkPipelineBinaryKHR> handles;
    explicit Binaries(Functions& api):f(api) {}
    ~Binaries(){for(auto h:handles)if(h){if(std::getenv("NR_ACO_DUMP")){std::fprintf(stderr,"ACO destroy binary %p\n",(void*)h);std::fflush(stderr);}f.destroy(f.device,h,nullptr);}}
};
struct Captured {
    Functions& f;VkPipeline pipeline{};
    explicit Captured(Functions& api):f(api) {}
    ~Captured(){if(pipeline) {
        VkReleaseCapturedPipelineDataInfoKHR r{VK_STRUCTURE_TYPE_RELEASE_CAPTURED_PIPELINE_DATA_INFO_KHR};r.pipeline=pipeline;
        if(std::getenv("NR_ACO_DUMP")){std::fprintf(stderr,"ACO release captured data\n");std::fflush(stderr);}
        f.release(f.device,&r,nullptr);
        if(std::getenv("NR_ACO_DUMP")){std::fprintf(stderr,"ACO destroy captured pipeline\n");std::fflush(stderr);}
        vkDestroyPipeline(f.device,pipeline,nullptr);
    }}
};
inline std::string keyname(uint64_t h) {char s[24];std::snprintf(s,sizeof s,"%016llx.nrp",static_cast<unsigned long long>(h));return s;}
inline bool network_path(const std::string& path) {
    auto name=path.substr(path.find_last_of("/\\")+1);
    return name.rfind("g_",0)==0 || name.rfind("temporal_pre_fp32",0)==0 || name.rfind("temporal_post_fp32",0)==0;
}
// Diagnostic control: compile selected shell kernels with LLPC while retaining
// the same graph, buffers and push constants used by the ACO run.
inline bool native_shell(const std::string& name) {
    const char* selected=std::getenv("NR_ACO_NATIVE_KERNELS");
    if(!selected || !*selected)return false;
    std::string list(selected);size_t start=0;
    while(start<=list.size()) {
        auto end=list.find(',',start);auto token=list.substr(start,end-start);
        if(token=="*" || token==name || (!token.empty() && token.back()=='*' &&
           name.compare(0,token.size()-1,token,0,token.size()-1)==0))return true;
        if(end==std::string::npos)break;start=end+1;
    }
    return false;
}
// An imported binary kept between runs: the captured template's compile is the slow part of a build
// (the driver compiles the native template only to get its container). File: "NRAC", version 1, the key
// size and key, the data size and data, fnv64 of the data. A file that does not check out is ignored.
inline bool load_cached(const std::string& file,VkPipelineBinaryKeyKHR& key,pal::Bytes& blob) {
    std::ifstream in(file,std::ios::binary);if(!in)return false;
    char magic[4];uint32_t version=0,key_size=0;uint64_t size=0,check=0;
    in.read(magic,4);in.read(reinterpret_cast<char*>(&version),4);in.read(reinterpret_cast<char*>(&key_size),4);
    if(!in || std::memcmp(magic,"NRAC",4)!=0 || version!=1 || !key_size || key_size>VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR)return false;
    key.keySize=key_size;in.read(reinterpret_cast<char*>(key.key),VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR);
    in.read(reinterpret_cast<char*>(&size),8);if(!in || !size || size>64*1024*1024)return false;
    blob.resize(size);in.read(reinterpret_cast<char*>(blob.data()),std::streamsize(size));in.read(reinterpret_cast<char*>(&check),8);
    return bool(in) && check==pal::hash(blob.data(),blob.size());
}
inline void save_cached(const std::string& file,const VkPipelineBinaryKeyKHR& key,const pal::Bytes& blob) {
    const std::string tmp=file+".tmp";
    {
        std::ofstream out(tmp,std::ios::binary|std::ios::trunc);if(!out)return;
        const uint32_t version=1,key_size=key.keySize;const uint64_t size=blob.size(),check=pal::hash(blob.data(),blob.size());
        out.write("NRAC",4);out.write(reinterpret_cast<const char*>(&version),4);out.write(reinterpret_cast<const char*>(&key_size),4);
        out.write(reinterpret_cast<const char*>(key.key),VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR);
        out.write(reinterpret_cast<const char*>(&size),8);out.write(reinterpret_cast<const char*>(blob.data()),std::streamsize(size));
        out.write(reinterpret_cast<const char*>(&check),8);if(!out)return;
    }
    std::remove(file.c_str());std::rename(tmp.c_str(),file.c_str());
}
inline VkPipeline import_pipeline(Functions& f,VkDevice device,const VkComputePipelineCreateInfo& ci,
                                  const VkPipelineBinaryKeyKHR& key_in,const pal::Bytes& blob) {
    VkPipelineBinaryKeyKHR key=key_in;
    VkPipelineBinaryDataKHR bd{blob.size(),const_cast<uint8_t*>(blob.data())};VkPipelineBinaryKeysAndDataKHR kd{1,&key,&bd};
    VkPipelineBinaryCreateInfoKHR bi{VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR};bi.pKeysAndDataInfo=&kd;
    VkPipelineBinaryHandlesInfoKHR hi{VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
    Binaries imported(f);imported.handles.resize(1);hi.pipelineBinaryCount=1;hi.pPipelineBinaries=imported.handles.data();
    check(f.create(device,&bi,nullptr,&hi),"import binary");
    VkPipelineBinaryInfoKHR input{VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR};input.binaryCount=1;input.pPipelineBinaries=imported.handles.data();input.pNext=ci.pNext;
    auto imported_ci=ci;imported_ci.pNext=&input;imported_ci.stage.module=VK_NULL_HANDLE;
    VkPipeline pipeline{};VkResult result=vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&imported_ci,nullptr,&pipeline);
    if(result!=VK_SUCCESS){if(pipeline)vkDestroyPipeline(device,pipeline,nullptr);check(result,"create imported pipeline");}
    return pipeline;
}
inline VkPipeline create(VkDevice device,const VkComputePipelineCreateInfo& ci,const pal::Record& record,uint32_t push_bytes,
                         std::array<uint32_t,3> workgroup,const std::string& cache_file={}) {
    Functions f(device);
    if(!cache_file.empty()) {
        VkPipelineBinaryKeyKHR key{VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};pal::Bytes blob;
        if(load_cached(cache_file,key,blob)) {
            try{return import_pipeline(f,device,ci,key,blob);}catch(const std::exception&){}   // made again below
        }
    }
    Captured captured(f);
    const bool trace=std::getenv("NR_ACO_DUMP")!=nullptr;
    auto step=[&](const char* s){if(trace){std::fprintf(stderr,"ACO %s: %s\n",keyname(record.spv_hash).c_str(),s);std::fflush(stderr);}};
    // The caller's pNext chain is retained. Flags2 and binary input are owned here.
    for(auto* p=static_cast<const VkBaseInStructure*>(ci.pNext);p;p=p->pNext)
        pal::require(p->sType!=VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR &&
                     p->sType!=VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR,"unsupported pipeline input chain");
    pal::require(!ci.stage.pSpecializationInfo && std::strcmp(ci.stage.pName,"main")==0 &&
                 !(ci.flags & VK_PIPELINE_CREATE_DERIVATIVE_BIT),"unsupported pipeline specialization/entry");
    VkPipelineCreateFlags2CreateInfoKHR flags{VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR};
    flags.flags=VkPipelineCreateFlags2KHR(ci.flags)|VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR;flags.pNext=ci.pNext;
    auto capture_ci=ci;capture_ci.pNext=&flags;
    step("capture pipeline");
    check(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&capture_ci,nullptr,&captured.pipeline),"capture pipeline");
    Binaries original(f);VkPipelineBinaryCreateInfoKHR bi{VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR};bi.pipeline=captured.pipeline;
    VkPipelineBinaryHandlesInfoKHR hi{VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
    check(f.create(device,&bi,nullptr,&hi),"count captured binaries");
    pal::require(hi.pipelineBinaryCount==1,"expected one PAL compute binary");original.handles.resize(1);
    hi.pPipelineBinaries=original.handles.data();check(f.create(device,&bi,nullptr,&hi),"capture binaries");
    VkPipelineBinaryDataInfoKHR di{VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR};di.pipelineBinary=original.handles[0];
    VkPipelineBinaryKeyKHR key{VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};size_t bytes=0;
    check(f.data(device,&di,&key,&bytes,nullptr),"size captured binary");
    pal::require(bytes && bytes<=64*1024*1024,"captured binary size limit");pal::Bytes blob(bytes);
    check(f.data(device,&di,&key,&bytes,blob.data()),"read captured binary");pal::require(bytes<=blob.size(),"captured binary grew");blob.resize(bytes);
    if(const char* dump=std::getenv("NR_ACO_DUMP")) if(*dump) {
        std::ofstream out(std::string(dump)+"/"+keyname(record.spv_hash)+".captured.elf",std::ios::binary);
        pal::require(bool(out),"cannot open captured binary dump");
        out.write(reinterpret_cast<const char*>(blob.data()),std::streamsize(blob.size()));
        pal::require(bool(out),"cannot write captured binary dump");
    }
    step("captured");blob=pal::splice(blob,record,push_bytes,workgroup);step("spliced");
    // Domain-separate the imported key and include the complete transformed data.
    pal::require(key.keySize && key.keySize<=VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR,"driver binary key size");
    uint64_t h=pal::hash(blob.data(),blob.size())^record.spv_hash^0x4e5250414c303031ull;
    for(uint32_t i=0;i<key.keySize;++i) {h^=i+1;h*=1099511628211ull;key.key[i]=uint8_t(h>>(8*(i%8)));}
    step("import binary");
    VkPipeline pipeline=import_pipeline(f,device,ci,key,blob);
    step("pipeline created");
    if(!cache_file.empty())save_cached(cache_file,key,blob);
    return pipeline;
}
inline VkResult create_or_default(VkDevice device,VkPipelineCache cache,const VkComputePipelineCreateInfo& ci,
                                 const std::string& path,const uint32_t* code,size_t bytes,VkPipeline* out,uint32_t push_bytes) {
    auto dir=directory();
    if(dir.empty() || !network_path(path))return vkCreateComputePipelines(device,cache,1,&ci,nullptr,out);
    pal::require(enabled(device),"binary mode was not enabled when this device was created");
    auto h=pal::hash(code,bytes);
    const auto record_bytes=pal::load(dir+"/records/"+keyname(h));
    auto record=pal::Record::parse(record_bytes);
    pal::require(record.spv_hash==h,"SPIR-V record mismatch");
    std::array<uint32_t,3> workgroup{};
    for(size_t p=5;p<bytes/4;) {
        auto n=code[p]>>16,op=code[p]&65535u;
        pal::require(n && n<=bytes/4-p,"invalid SPIR-V instruction");
        if(op==16 && n==6 && code[p+2]==17) {
            pal::require(!workgroup[0],"multiple SPIR-V local sizes");
            workgroup={code[p+3],code[p+4],code[p+5]};
        }
        p+=n;
    }
    pal::require(workgroup[0]!=0,"literal SPIR-V local size required");
    // The PAL container comes from the driver's own compile of a template SPIR-V with the same interface:
    // the first shell directory holding the name (NR_ACO_SHELL_DIR, or the product's aco/shell then the
    // native network shaders), else the Linux SPIR-V itself.
    std::string shell_path;
    {
        auto name=path.substr(path.find_last_of("/\\")+1);
        // The name itself, else the native shader that serves as its template (the bundle's alias list).
        for(const auto& n:{name,alias_of(name)}) {
            if(n.empty() || !shell_path.empty())continue;
            for(const auto& d:shell_dirs()) {
                std::ifstream probe(d+"/"+n,std::ios::binary);
                if(probe) {shell_path=d+"/"+n;break;}
            }
        }
    }
    if(!shell_path.empty()) {
        auto name=path.substr(path.find_last_of("/\\")+1);
        auto shell=pal::load(shell_path);
        pal::require(shell.size()%4==0,"invalid shell SPIR-V size");
        std::vector<uint32_t> words(shell.size()/4);std::memcpy(words.data(),shell.data(),shell.size());
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};sm.codeSize=shell.size();sm.pCode=words.data();
        VkShaderModule module{};check(vkCreateShaderModule(device,&sm,nullptr,&module),"create shell module");
        auto shell_ci=ci;shell_ci.stage.module=module;
        try{
            if(native_shell(name)) {
                std::fprintf(stderr,"ACO diagnostic native shell: %s\n",name.c_str());
                check(vkCreateComputePipelines(device,cache,1,&shell_ci,nullptr,out),"create native shell pipeline");
            } else {
                // The cached import is valid for this record, this template, these push/workgroup values
                // and this GPU + driver.
                std::string cache_file;
                const std::string cache=cache_dir();
                if(!cache.empty()) {
                    uint64_t k[6]={pal::hash(record_bytes.data(),record_bytes.size()),pal::hash(shell.data(),shell.size()),
                                   identity(),push_bytes,(uint64_t(workgroup[0])<<32)|workgroup[1],workgroup[2]};
                    char b[40];std::snprintf(b,sizeof b,"%016llx.nrb",static_cast<unsigned long long>(pal::hash(k,sizeof k)));
                    cache_file=cache+"/"+b;
                }
                *out=create(device,shell_ci,record,push_bytes,workgroup,cache_file);
            }
        }catch(...){vkDestroyShaderModule(device,module,nullptr);throw;}
        vkDestroyShaderModule(device,module,nullptr);return VK_SUCCESS;
    }
    *out=create(device,ci,record,push_bytes,workgroup);return VK_SUCCESS;
}
} // namespace nr::binary
