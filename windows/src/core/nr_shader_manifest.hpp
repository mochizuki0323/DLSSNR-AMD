#pragma once
#include <charconv>
#include <cstdint>
#include <istream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nr::detail {
// A nonnumeric version token deliberately stops the pre-v2 reader before it
// finds any launch constants. Old hosts must reject packed weights, not infer
// their layout from shader filenames that also existed before this format.
inline std::map<std::string, uint32_t> read_shader_manifest(std::istream& input) {
    std::map<std::string, uint32_t> values;
    bool first = true;
    unsigned version = 0;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string key, token, extra;
        if (!(row >> key)) continue;
        if (!(row >> token) || (row >> extra))
            throw std::runtime_error("malformed shader manifest row");
        if (first && key == "nr_shader_manifest") {
            if (token != "v2" && token != "v3") throw std::runtime_error("unsupported shader manifest version");
            version = token == "v3" ? 3 : 2; first = false; continue;
        }
        first = false;
        if (key != "vattn_qt" && key != "gemm_wide_mt" &&
            key != "gemm_wide_nt" && key != "weight_layout" && key != "math_profile" && key != "qkv_fused_norm" && key != "ups_fused_mode" && key != "wide_ups_fused_mode" && key != "gemm_proj_mt" && key != "gemm_proj_nt" && key != "gemm_projw_mt" && key != "gemm_projw_nt" && key != "ffwd_wgw" && key != "ffwd_gmajor" && key != "gemm_remap" && key != "upsview_vec" && key != "repack_vec" && key != "wide_ups_mask" && key != "persist_df" && key != "ds_fuse" && key != "decups_vec" && key != "persist_ds" && key != "persist_up" && key != "persist_strag" && key != "persist_one" && key != "noise_field" && key != "ffwd_fm2_min" && key != "post_alpha" && key != "tchain")
            throw std::runtime_error("unknown shader manifest key: " + key);
        uint64_t number = 0;
        const auto result = std::from_chars(token.data(), token.data() + token.size(), number);
        if (result.ec != std::errc{} || result.ptr != token.data() + token.size() ||
            number > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("invalid shader manifest value: " + key);
        if (!values.emplace(key, uint32_t(number)).second)
            throw std::runtime_error("duplicate shader manifest key: " + key);
    }
    for (const char* key : {"vattn_qt", "gemm_wide_mt", "gemm_wide_nt"})
        if (!values.count(key) || values.at(key) == 0)
            throw std::runtime_error(std::string("missing or zero shader manifest constant: ") + key);
    if (version >= 2 && !values.count("weight_layout"))
        throw std::runtime_error("versioned shader manifest requires weight_layout");
    if (values.count("weight_layout") &&
        (values.at("weight_layout") > 5 || (version == 0 && values.at("weight_layout") != 0)))
        throw std::runtime_error("unsupported shader weight layout");
    if (version == 3 && !values.count("math_profile"))
        throw std::runtime_error("v3 shader manifest requires math_profile");
    if (values.count("math_profile") && (version != 3 || values.at("math_profile") > 3))
        throw std::runtime_error("unsupported shader math profile");
    return values;
}
} // namespace nr::detail
