// CPU-only gfx1201 PAL code-object adapter. No Vulkan or driver dependency.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nr::pal {
using Bytes = std::vector<uint8_t>;
inline void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
inline void range(const Bytes& b, size_t p, size_t n) { require(p <= b.size() && n <= b.size()-p, "truncated binary"); }
template<class T> T read(const Bytes& b, size_t p) { range(b,p,sizeof(T)); T t; std::memcpy(&t,b.data()+p,sizeof t); return t; }
template<class T> void write(Bytes& b, size_t p, T t) { range(b,p,sizeof t); std::memcpy(b.data()+p,&t,sizeof t); }
inline void word(Bytes& b, uint32_t v) { size_t p=b.size(); b.resize(p+4); write(b,p,v); }
inline uint64_t hash(const void* data, size_t n) {
    uint64_t h=1469598103934665603ull;
    const auto* p=static_cast<const uint8_t*>(data);
    for(size_t i=0;i<n;++i) { h^=p[i]; h*=1099511628211ull; } return h;
}
inline Bytes load(const std::string& p) {
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    require(bool(f),"cannot open binary"); auto n=f.tellg();
    require(n>0 && n<=64*1024*1024,"binary size outside limit");
    Bytes b(static_cast<size_t>(n)); f.seekg(0); f.read(reinterpret_cast<char*>(b.data()),n);
    require(bool(f),"cannot read binary"); return b;
}

// MessagePack nodes retain unknown scalar encodings. Maps/arrays are rebuilt,
// so fields may grow without overwriting adjacent metadata or ELF sections.
struct Value {
    enum Kind { Raw, UInt, String, Array, Map } kind=Raw;
    uint64_t u=0; std::string s; Bytes raw;
    std::vector<Value> items;
    std::vector<std::pair<std::string,Value>> fields;
    Value()=default;
    explicit Value(uint64_t v):kind(UInt),u(v) {}
    static Value boolean(bool b) { Value v; v.raw={uint8_t(b?0xc3:0xc2)}; return v; }
    Value& at(const std::string& key) {
        require(kind==Map,"expected metadata map");
        for(auto& f:fields) if(f.first==key) return f.second;
        throw std::runtime_error("missing metadata field: "+key);
    }
    const Value& at(const std::string& key) const { return const_cast<Value*>(this)->at(key); }
    void set(const std::string& key, Value v) {
        require(kind==Map,"expected metadata map");
        for(auto& f:fields) if(f.first==key) { f.second=std::move(v); return; }
        fields.emplace_back(key,std::move(v));
    }
    uint64_t number() const { require(kind==UInt,"expected unsigned metadata integer"); return u; }
};
inline uint64_t big(const Bytes& b,size_t& p,size_t n) {
    range(b,p,n); uint64_t v=0; while(n--) v=(v<<8)|b[p++]; return v;
}
inline Value decode(const Bytes& b,size_t& p,unsigned depth=0) {
    require(depth<40,"metadata nesting limit"); range(b,p,1); size_t start=p; unsigned c=b[p++]; Value v;
    size_t n=0; bool array=false,map=false,str=false;
    if(c<=0x7f) return Value(c);
    if(c>=0xa0 && c<=0xbf) {str=true;n=c&31;}
    else if((c&0xf0)==0x90) {array=true;n=c&15;}
    else if((c&0xf0)==0x80) {map=true;n=c&15;}
    else if(c>=0xcc && c<=0xcf) return Value(big(b,p,size_t(1)<<(c-0xcc)));
    else if(c==0xd9 || c==0xda || c==0xdb) {str=true;n=size_t(big(b,p,size_t(1)<<(c-0xd9)));}
    else if(c==0xdc || c==0xdd) {array=true;n=size_t(big(b,p,c==0xdc?2:4));}
    else if(c==0xde || c==0xdf) {map=true;n=size_t(big(b,p,c==0xde?2:4));}
    else {
        if(c==0xca)n=4; else if(c==0xcb)n=8;
        else if(c>=0xd0 && c<=0xd3)n=size_t(1)<<(c-0xd0);
        else if(c==0xc4 || c==0xc5 || c==0xc6)n=size_t(big(b,p,size_t(1)<<(c-0xc4)));
        else if(c>=0xd4 && c<=0xd8)n=(size_t(1)<<(c-0xd4))+1;
        else if(c==0xc7 || c==0xc8 || c==0xc9)n=size_t(big(b,p,size_t(1)<<(c-0xc7)))+1;
        else require(c>=0xe0 || c==0xc0 || c==0xc2 || c==0xc3,"unsupported MessagePack tag");
        range(b,p,n); p+=n; v.raw=Bytes(b.begin()+start,b.begin()+p); return v;
    }
    require(n<=b.size(),"metadata length outside binary");
    if(str) {range(b,p,n); v.kind=Value::String;v.s.assign(reinterpret_cast<const char*>(b.data()+p),n);p+=n;}
    else if(array) {v.kind=Value::Array;for(size_t i=0;i<n;++i)v.items.push_back(decode(b,p,depth+1));}
    else if(map) {
        v.kind=Value::Map;
        for(size_t i=0;i<n;++i) {
            auto k=decode(b,p,depth+1); require(k.kind==Value::String,"non-string metadata key");
            for(const auto& f:v.fields) require(f.first!=k.s,"duplicate metadata key");
            v.fields.emplace_back(std::move(k.s),decode(b,p,depth+1));
        }
    } return v;
}
inline void putbig(Bytes& b,uint64_t n,size_t bytes) { while(bytes) b.push_back(uint8_t(n>>(--bytes*8))); }
inline void encode(Bytes& b,const Value& v) {
    switch(v.kind) {
    case Value::Raw: b.insert(b.end(),v.raw.begin(),v.raw.end()); break;
    case Value::UInt: b.push_back(0xcf);putbig(b,v.u,8);break;
    case Value::String: b.push_back(0xdb);putbig(b,v.s.size(),4);b.insert(b.end(),v.s.begin(),v.s.end());break;
    case Value::Array: b.push_back(0xdd);putbig(b,v.items.size(),4);for(const auto& x:v.items)encode(b,x);break;
    case Value::Map: b.push_back(0xdf);putbig(b,v.fields.size(),4);for(const auto& f:v.fields) {
        Value k;k.kind=Value::String;k.s=f.first;encode(b,k);encode(b,f.second);
    } break;
    }
}
inline Value unpack(const Bytes& b) { size_t p=0;auto v=decode(b,p);require(p==b.size(),"trailing metadata");return v; }
inline Value array(std::initializer_list<uint64_t> xs) { Value v;v.kind=Value::Array;for(auto x:xs)v.items.emplace_back(x);return v; }

struct Record {
    uint64_t spv_hash=0,code_hash=0;
    uint32_t rsrc1=0,rsrc2=0,rsrc3=0,exec=0,wg=0,shared=0,push_bytes=0,user=0;
    int set0=-1,push=-1,grid=-1;
    std::vector<std::pair<int,int>> inlines;
    Bytes code;
    static Record parse(const Bytes& b) {
        range(b,0,96); require(std::memcmp(b.data(),"NRPAL001",8)==0,"bad adapter record version");
        Record r;r.spv_hash=read<uint64_t>(b,8);r.code_hash=read<uint64_t>(b,16);
        r.rsrc1=read<uint32_t>(b,24);r.rsrc2=read<uint32_t>(b,28);r.rsrc3=read<uint32_t>(b,32);
        r.exec=read<uint32_t>(b,36);r.wg=read<uint32_t>(b,40);r.shared=read<uint32_t>(b,44);
        r.push_bytes=read<uint32_t>(b,48);r.user=read<uint32_t>(b,52);
        r.set0=read<int32_t>(b,56);r.push=read<int32_t>(b,60);r.grid=read<int32_t>(b,64);
        auto n=read<uint32_t>(b,68),sz=read<uint32_t>(b,72);
        require(n<=32 && sz && sz%4==0 && uint64_t(96)+8*n+sz==b.size(),"adapter record sizes");
        for(size_t i=76;i<96;++i)require(b[i]==0,"adapter reserved field");
        for(unsigned i=0;i<n;++i)r.inlines.emplace_back(read<int32_t>(b,96+8*i),read<int32_t>(b,100+8*i));
        r.code=Bytes(b.begin()+96+8*n,b.end());
        require(hash(r.code.data(),r.code.size())==r.code_hash,"adapter code checksum");
        require(!(r.rsrc2&~uint32_t(0x00ff9fbe)) && !(r.rsrc3&~uint32_t(0xff0)) &&
                r.user==((r.rsrc2>>1)&31),"unsupported scratch/config");
        require(r.exec && r.exec<=sz && r.exec%4==0 && r.wg && r.wg<=1024 && r.push_bytes<=128,"adapter resource limits");
        // gfx1201 rounds the shader's LDS use up to a 1 KiB allocation.
        require(((r.shared+1023u)&~1023u)==(((r.rsrc2>>15)&511)*512),"adapter LDS allocation mismatch");
        std::vector<bool> used(r.user,false);
        auto take=[&](int reg) {require(reg>=0 && uint32_t(reg)<r.user && !used[reg],"overlapping adapter arguments");used[reg]=true;};
        take(r.set0);if(r.push>=0)take(r.push);
        for(auto [reg,dw]:r.inlines) {take(reg);require(dw>=0 && uint32_t(dw)*4<r.push_bytes,"push index outside block");}
        if(r.grid>=0) for(int j=0;j<3;++j)take(r.grid+j);
        require(std::all_of(used.begin(),used.end(),[](bool x){return x;}),"unhandled adapter argument");
        for(size_t p=r.exec;p<r.code.size();p+=4)require(read<uint32_t>(r.code,p)==0xbf9f0000,"constant data unsupported");
        return r;
    }
};

struct Arguments {
    std::vector<uint64_t> map=std::vector<uint64_t>(32,0xffffffff);
    std::vector<std::pair<int,int>> moves,loads;
    unsigned user=0;int spill=-1,grid=-1;
    explicit Arguments(const Record& r,unsigned descriptor_slot=33) {
        map[0]=0x10000000;map[1]=descriptor_slot;user=2;moves.emplace_back(r.set0,1);
        unsigned extra=(r.grid>=0?2:0)+(r.push>=0?1:0);
        bool spill_needed=r.push>=0 || user+r.inlines.size()+extra>16;
        // A spill slot replaces the push-pointer slot, or one inline slot.
        unsigned limit=16-(r.grid>=0?2:0)-(spill_needed?1:0);
        for(auto [dst,dw]:r.inlines) {
            if(user<limit) {map[user]=dw+1;moves.emplace_back(dst,int(user++));}
            else loads.emplace_back(dst,4*(dw+1));
        }
        if(spill_needed) {spill=int(user);map[user++]=0x10000002;}
        if(r.grid>=0) {grid=int(user);map[user]=0x10000006;user+=2;}
        require(user<=16,"PAL argument limit");
        for(auto [dst,src]:moves)require(dst<=src,"unsupported parallel argument copy");
    }
};
// Incoming pointers are saved before argument copies; the body keeps its
// relative branch offsets and uses m0 for the 32-bit address high word.
inline Bytes prologue(const Record& r,unsigned descriptor_slot=33) {
    Bytes b;Arguments a(r,descriptor_slot);
    auto mov=[&](unsigned dst,unsigned src){word(b,0xbe800000|(dst<<16)|src);};
    if(a.spill>=0)mov(96,a.spill);
    if(a.grid>=0) {mov(100,a.grid);mov(101,a.grid+1);}
    if(r.rsrc2&(1<<10)) mov(102,a.user);
    word(b,0xbe804700|(98<<16)); // s_getpc_b64 s[98:99]
    word(b,0xbf800002); // dependency spacing, also safe before sext/m0
    word(b,0xbe800f00|(97<<16)|99); // s_sext_i32_i16 s97, s99
    word(b,0xbf800002);mov(125,97); // m0 = PAL address32 high
    word(b,0xbf800002);
    auto ld=[&](unsigned dst,unsigned base,unsigned off) {
        word(b,0xf4000000|(dst<<6)|(base>>1));word(b,0xf8000000|off);
    };
    for(auto [dst,src]:a.moves)mov(dst,src);
    for(auto [reg,offset]:a.loads)ld(reg,96,offset);
    if(r.push>=0) {word(b,0x80000000|(uint32_t(r.push)<<16)|(0x84<<8)|96);word(b,0xbf800002);}
    if(r.grid>=0)for(unsigned j=0;j<3;++j)ld(r.grid+j,100,j*4);
    word(b,0xbfc70000);word(b,0xbf800003);
    if(r.rsrc2&(1<<10))mov(r.user,102);
    // Branch over alignment padding rather than executing one nop per dword.
    size_t end=(b.size()+4+255)&~size_t(255);
    word(b,0xbfa00000|uint32_t((end-b.size()-4)/4));
    while(b.size()<end)word(b,0xbf800000);
    return b;
}

struct Section { uint32_t name,type;uint64_t flags,addr,offset,size;uint32_t link,info;uint64_t align,entsize; };
static_assert(sizeof(Section)==64);
struct Elf {
    Bytes header;std::vector<Section> sections;std::vector<Bytes> data;std::vector<std::string> names;
    explicit Elf(const Bytes& b) {
        range(b,0,64);
        require(std::memcmp(b.data(),"\x7f" "ELF",4)==0 && b[4]==2 && b[5]==1 && b[7]==65 &&
                read<uint16_t>(b,16)==1 && read<uint16_t>(b,18)==224 && read<uint32_t>(b,48)==0x4e &&
                read<uint64_t>(b,32)==0 && read<uint16_t>(b,56)==0 && read<uint16_t>(b,58)==64,"unsupported PAL ELF");
        auto count=read<uint16_t>(b,60),str=read<uint16_t>(b,62); auto at=read<uint64_t>(b,40);
        require(count>0 && count<256 && str<count,"ELF section count");range(b,at,size_t(count)*64);
        header=Bytes(b.begin(),b.begin()+64);
        for(unsigned i=0;i<count;++i) {
            auto s=read<Section>(b,at+i*64);require(s.type!=4 && s.type!=9 && s.type!=8,"relocations/NOBITS unsupported");
            require(s.align<=4096 && (!s.align || (s.align&(s.align-1))==0),"ELF section alignment");
            range(b,s.offset,s.size);sections.push_back(s);data.emplace_back(b.begin()+s.offset,b.begin()+s.offset+s.size);
        }
        require(sections[str].type==3,"ELF section strings");
        for(const auto& s:sections) {
            range(data[str],s.name,1);auto it=std::find(data[str].begin()+s.name,data[str].end(),0);
            require(it!=data[str].end(),"unterminated ELF section name");names.emplace_back(data[str].begin()+s.name,it);
        }
    }
    size_t find(const std::string& name) const {
        auto it=std::find(names.begin(),names.end(),name);require(it!=names.end(),"missing ELF section");
        require(std::count(names.begin(),names.end(),name)==1,"duplicate ELF section");return size_t(it-names.begin());
    }
    Bytes build() {
        Bytes out=header;
        for(size_t i=1;i<sections.size();++i) {
            auto& s=sections[i];size_t align=std::max<uint64_t>(s.align,1);
            out.resize((out.size()+align-1)&~(align-1));s.offset=out.size();s.size=data[i].size();
            out.insert(out.end(),data[i].begin(),data[i].end());
        }
        out.resize((out.size()+7)&~size_t(7));write(out,40,uint64_t(out.size()));
        for(const auto& s:sections) {size_t p=out.size();out.resize(p+64);write(out,p,s);}return out;
    }
};
inline Bytes splice(const Bytes& binary,const Record& r,unsigned push_bytes=128,
                    std::array<uint32_t,3> workgroup={0,0,0}) {
    require(push_bytes<=128 && push_bytes%4==0 && r.push_bytes<=push_bytes,"unsupported pipeline push range");
    const unsigned descriptor_slot=push_bytes/4+1;
    Elf e(binary); auto text=e.find(".text"),note=e.find(".note"),sym=e.find(".symtab");
    require(e.sections[text].align==256 && e.sections[text].addr==0,"PAL entry alignment");
    // Exactly one compute entry, no relocations, no secondary functions/data.
    const auto& sy=e.sections[sym];require(sy.entsize==24 && sy.link<e.data.size() && e.data[sym].size()%24==0,"ELF symbols");
    unsigned found=0;
    for(size_t p=0;p<e.data[sym].size();p+=24) {
        auto ix=read<uint16_t>(e.data[sym],p+6);if(ix!=text)continue;
        auto off=read<uint32_t>(e.data[sym],p);range(e.data[sy.link],off,1);
        auto end=std::find(e.data[sy.link].begin()+off,e.data[sy.link].end(),0);
        require(end!=e.data[sy.link].end(),"unterminated symbol");
        std::string name(e.data[sy.link].begin()+off,end);
        require(name=="_amdgpu_cs_main" && (e.data[sym][p+4]&15)==2 && read<uint64_t>(e.data[sym],p+8)==0,
                "unsupported PAL text symbol");
        auto pre=prologue(r,descriptor_slot);write(e.data[sym],p+16,uint64_t(pre.size()+r.exec));++found;
    }
    require(found==1,"PAL compute entry count");
    auto pre=prologue(r,descriptor_slot);e.data[text]=pre;e.data[text].insert(e.data[text].end(),r.code.begin(),r.code.end());
    Bytes notes;size_t pos=0;unsigned metadata=0;
    while(pos<e.data[note].size()) {
        auto& src=e.data[note];size_t start=pos;range(src,pos,12);
        auto ns=read<uint32_t>(src,pos),ds=read<uint32_t>(src,pos+4),type=read<uint32_t>(src,pos+8);
        pos+=12;range(src,pos,(uint64_t(ns)+3)&~uint64_t(3));
        Bytes name(src.begin()+pos,src.begin()+pos+ns);pos+=(uint64_t(ns)+3)&~uint64_t(3);
        range(src,pos,(uint64_t(ds)+3)&~uint64_t(3));size_t desc=pos;pos+=(uint64_t(ds)+3)&~uint64_t(3);
        if(type!=32) {notes.insert(notes.end(),src.begin()+start,src.begin()+pos);continue;}
        require(name==Bytes({'A','M','D','G','P','U',0}),"unexpected PAL note owner");
        auto m=unpack(Bytes(src.begin()+desc,src.begin()+desc+ds));
        auto& ver=m.at("amdpal.version");require(ver.kind==Value::Array && ver.items.size()==2 &&
                ver.items[0].number()==3 && ver.items[1].number()==6,"PAL metadata version");
        auto& ps=m.at("amdpal.pipelines");require(ps.kind==Value::Array && ps.items.size()==1,"PAL pipeline count");
        auto& p=ps.items[0];require(p.at(".user_data_limit").number()==descriptor_slot+1 && p.at(".api").s=="Vulkan" &&
                p.at(".type").s=="Cs","unsupported XGL user-data layout");
        auto& cs=p.at(".hardware_stages").at(".cs");
        require(cs.at(".wavefront_size").number()==32 && cs.at(".entry_point_symbol").s=="_amdgpu_cs_main","PAL compute ABI");
        const auto& dims=cs.at(".threadgroup_dimensions");
        require(dims.kind==Value::Array && dims.items.size()==3,"invalid PAL workgroup dimensions");
        if(workgroup[0]) {
            require(workgroup[1] && workgroup[2] && uint64_t(workgroup[0])*workgroup[1]*workgroup[2]==r.wg,
                    "workgroup size differs from capture");
            cs.set(".threadgroup_dimensions",array({workgroup[0],workgroup[1],workgroup[2]}));
        } else require(dims.items[0].number()*dims.items[1].number()*dims.items[2].number()==r.wg,
                       "workgroup size differs from capture");
        auto& map=cs.at(".user_data_reg_map");require(map.kind==Value::Array && map.items.size()==32 &&
                map.items[0].number()==0x10000000,"PAL global table ABI");
        for(const auto& v:map.items) {auto x=v.number();require(x<=descriptor_slot || x==0xffffffff || x==0x10000000 || x==0x10000002 || x==0x10000006,"unhandled PAL user-data mapping");}
        Arguments a(r,descriptor_slot);map.items.clear();for(auto x:a.map)map.items.emplace_back(x);
        p.set(".spill_threshold",Value(a.spill>=0?1:65535));
        cs.set(".shader_spill_threshold",Value(a.spill>=0?1:65535));
        cs.set(".user_sgprs",Value(a.user));
        cs.set(".sgpr_count",Value(103));
        cs.set(".vgpr_count",Value(((r.rsrc1&63)+1)*8));
        cs.set(".lds_size",Value(((r.rsrc2>>15)&511)*512));cs.set(".float_mode",Value((r.rsrc1>>12)&255));
        cs.set(".wgp_mode",Value::boolean((r.rsrc1>>29)&1));
        cs.set(".mem_ordered",Value::boolean((r.rsrc1>>30)&1));
        cs.set(".forward_progress",Value::boolean((r.rsrc1>>31)&1));
        cs.set(".fp16_overflow",Value::boolean((r.rsrc1>>26)&1));
        cs.set(".scratch_en",Value::boolean(false));cs.set(".scratch_memory_size",Value(0));
        auto& cr=p.at(".compute_registers");
        for(unsigned i=0;i<3;++i)cr.set(i==0?".tgid_x_en":i==1?".tgid_y_en":".tgid_z_en",Value::boolean((r.rsrc2>>(7+i))&1));
        cr.set(".tg_size_en",Value::boolean((r.rsrc2>>10)&1));cr.set(".tidig_comp_cnt",Value((r.rsrc2>>11)&3));
        cr.set(".x_interleave",Value(0));cr.set(".y_interleave",Value(0));
        auto h=hash(e.data[text].data(),e.data[text].size());
        cs.set(".checksum_value",Value(uint32_t(h)));p.set(".internal_pipeline_hash",array({h,r.spv_hash}));
        Bytes md;encode(md,m);word(notes,ns);word(notes,uint32_t(md.size()));word(notes,type);
        notes.insert(notes.end(),name.begin(),name.end());while(notes.size()%4)notes.push_back(0);
        notes.insert(notes.end(),md.begin(),md.end());while(notes.size()%4)notes.push_back(0);++metadata;
    }
    require(metadata==1,"PAL metadata note count");e.data[note]=std::move(notes);return e.build();
}
} // namespace nr::pal
