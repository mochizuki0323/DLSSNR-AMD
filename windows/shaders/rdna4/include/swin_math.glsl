#ifndef NR_SWIN_MATH_GLSL
#define NR_SWIN_MATH_GLSL

// Windows (LLPC) NR_EXP_I16: the exponentials' clamp on the half bit patterns as signed 16-bit
// integers. LLPC turns an f16 pair clamp into a v_med3_num_f16 per half with the unpack and
// repack around it; on bits it is one v_pk_max_i16 + one v_pk_min_i16. For every non-NaN half
// the signed order of the patterns is the order of the values on [lo, hi] (negatives and -0
// sit below any positive lo), so the clamped bits are the same. NaN only differs.
#ifndef NR_EXP_I16
#define NR_EXP_I16 0
#endif
#if NR_EXP_I16
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
// clamp(y, lo, hi) for positive lo < hi, as packed bits.
uint nr_clamp_h2_bits(f16vec2 y, int lo, int hi) {
    return pack32(u16vec2(min(max(float16BitsToInt16(y), i16vec2(int16_t(lo))), i16vec2(int16_t(hi)))));
}
#endif

// Pinned native-FP8 PTX: MpCubicSilu uses two fma.rn.f16x2 instructions
// followed by mul.f16x2. Algebraically expanding the polynomial changes its
// rounding. These elementwise operations are independent of MMA accumulation.
f16vec2 nr_swin_act2(f16vec2 x) {
    const f16vec2 t = clamp(x, f16vec2(-4.0hf), f16vec2(4.0hf));
    const f16vec2 u = fma(f16vec2(-0.055908203125hf), abs(t), f16vec2(0.447265625hf));
    return x * fma(t, u, f16vec2(0.89453125hf));
}

float nr_swin_act(float x) {
    return float(nr_swin_act2(f16vec2(NR_F16(x))).x);
}

// The PTX converts the coefficients to half BEFORE its half FMA, then clamps
// the result. Its packed shl/add implements this scalar half-bit transform.
// Truncating an affine expression evaluated in float is a different function.
float nr_swin_exp(float x) {
    // Spell the already-rounded values: glslang's constant float-to-half
    // conversion truncates 0.044910375, unlike the PTX cvt.rn instruction.
    const NR_F16 y = clamp(fma(NR_F16(x), NR_F16(0.044921875), NR_F16(1.30078125)),
                           NR_F16(1.03125), NR_F16(1.5693359375));
    // `(h & 0x3ff) << 5`, for the reason spelled out beside `nr_swin_exp2`:
    // the `+ 0x8000` is the carry the clamped exponent field puts there, and
    // masking it off before the shift is the same number without the add.
    const uint h = packHalf2x16(vec2(float(y), 0.0)) & 0xffffu;
    return unpackHalf2x16((h & 0x03ffu) << 5u).x;
}

// cc_vit_1d_attention_fp8 uses the same half-FMA construction with its own
// coefficients and four-bit shift. The denominator consumes this half value
// before FP8 probability quantization for the context MMA.
float nr_vit_exp(float x) {
    const NR_F16 y = clamp(fma(NR_F16(x), NR_F16(0.08953857421875), NR_F16(1.708984375)),
                           NR_F16(1.439453125), NR_F16(1.9775390625));
    const uint h = packHalf2x16(vec2(float(y), 0.0)) & 0xffffu;
    return unpackHalf2x16((h & 0x03ffu) << 4u).x;
}

// Packed form of the same half FMA and per-half bit transform.
// Apply the same transform independently to both packed half lanes. Mask
// before shifting so the low lane cannot carry into the high lane. The
// exhaustive GPU probe covers all 63,488 finite half values against the
// recovered PTX oracle.
//
// **The mask can do the sign flip's work, so there is no XOR.** `y` is clamped
// into [1.03125, 1.5693359375], so each half is `0x3C00 + m` with m the 10-bit
// mantissa; `h & 0x7ff` is then `1024 + m`, the shift makes it `32768 + 32m`,
// and the `^ 0x8000` exists only to clear the bit 15 that the `1024` put there.
// Masking that bit off *before* the shift is the same number and one
// instruction fewer: `(h & 0x3ff) << 5` is `32m` directly. It holds for the
// nonfinite classes too - 0x7C00 and 0x7E00 and their negatives all have the
// exponent LSB set, which is the only bit the two spellings disagree about -
// and 32m <= 18656 so the shift still cannot carry into the other half.
// Measured: 32 `v_xor_b32` a wave at C=32, bit-identical at every width.
// `nr_vit_exp2` in vit_attn.comp is written the same way for the same reason.
f16vec2 nr_swin_exp2(vec2 x) {
#if NR_SWIN_EXP_CONTINUOUS
    // Experimental approximation (off by default): the native half transform maps the rounded
    // integer n=46*x+308 to half bits n<<5, clamped to n=[32,583]. For normal
    // positive halves the corresponding FP32 bits are 0x38000000+n*2^18.
    // Interpolate that mantissa construction in FP32 instead of rounding n
    // first. Bounds and every softmax term are retained; values can differ.
    const vec2 bits=clamp(fma(x,vec2(12058624.0),vec2(1020264448.0)),
                          vec2(947912704.0),vec2(1092354048.0));
    return f16vec2(uintBitsToFloat(uvec2(bits)));
#else
#if NR_EXP_I16
    const uint h=nr_clamp_h2_bits(fma(f16vec2(x),f16vec2(0.044921875hf),f16vec2(1.30078125hf)),
                                  0x3C20, 0x3E47);   // 1.03125, 1.5693359375
#else
    const f16vec2 y=clamp(fma(f16vec2(x),f16vec2(0.044921875hf),f16vec2(1.30078125hf)),
                           f16vec2(1.03125hf),f16vec2(1.5693359375hf));
    const uint h=packHalf2x16(vec2(y));
#endif
    const uint bits=(h<<5u)+0x7FF88000u;  // == (h&0x03ff03ff)<<5, see nr_swin_exp_baked
    return f16vec2(unpackHalf2x16(bits));
#endif
}


#ifndef NR_BAKED_EXP_BIAS
#define NR_BAKED_EXP_BIAS 0
#endif
#if NR_BAKED_EXP_BIAS
// Fixed bias already contains fma(original_bias, 0.044921875, 1.30078125).
// Keep exponent construction/bounds but remove the per-frame bias add and
// incoming-logit half rounding. This is a measured approximation, not parity.
// The clamp pins every y to exponent 15 (0x3C00), so masking the
// mantissa and shifting equals shifting and subtracting the shifted exponent
// field, and that is one `v_lshl_add_u32`: (0x3C00+m)<<5 = 0x78000 + 32m per
// half, and the high half's field leaves 0x80000000 after the 32-bit wrap.
// Bit-identical for every clamped value; one instruction instead of two.
#ifndef NR_EXP_NOHI
#define NR_EXP_NOHI 0
#endif
f16vec2 nr_swin_exp_baked(vec2 x, vec2 bias) {
#if defined(NR_ABLATE_EXP) && NR_ABLATE_EXP
    return f16vec2(x + bias);   // diagnostic (wrong output): the exponential priced by leaving it out
#endif
#if NR_EXP_I16
    const uint h=nr_clamp_h2_bits(f16vec2(fma(x,vec2(0.044921875),bias)), 0x3C20, 0x3E47);
#elif NR_EXP_NOHI
    // The host's weight audit (nr_graph.cpp, g_nohi) proved that no logit of this
    // layer reaches the upper bound, so max alone is the same function here.
    const f16vec2 y=max(f16vec2(fma(x,vec2(0.044921875),bias)),f16vec2(1.03125hf));
    const uint h=packFloat2x16(y);
#else
    const f16vec2 y=clamp(f16vec2(fma(x,vec2(0.044921875),bias)),
                         f16vec2(1.03125hf),f16vec2(1.5693359375hf));
    const uint h=packFloat2x16(y);
#endif
    return unpackFloat2x16((h<<5u)+0x7FF88000u);
}
#endif

// The gate is the *paired* quantiser, not mode 4. It read `NR_QUANT_MODE == 4`
// and made **mode 5 unbuildable** - every fswin pipeline failed with "no
// matching overloaded function" here - which is why the packed-f16-clamp mode
// had long gone un-re-priced. This function does not touch the
// quantiser at all; it reduces the probability pairs the packed Swin math
// produces, and both paired modes produce them.
#if (NR_QUANT_MODE == 4 || NR_QUANT_MODE == 5) && NR_ACC_F16 == 0
// A 64-key Swin window: four 16-key fragments, eight values per lane.
// Adjacent values keep the native even/odd reduction chains independent.
NR_F16 nr_swin_probability_sum(f16vec2 probability[4][4]) {
    f16vec2 part[4];
    for (int c=0;c<4;++c) {
        part[c]=f16vec2(0.0hf);
        for (int j=0;j<4;++j) {
            f16vec2 x=probability[j][c];
#if NR_PROB_LATE_SHUFFLE
            part[c]=part[c]+x;
#else
            uint other=subgroupShuffleXor(packHalf2x16(vec2(x)),16u);
            f16vec2 pair=x+f16vec2(unpackHalf2x16(other));
            part[c]=part[c]+pair;
#endif
        }
    }
    f16vec2 total=((part[0]+part[1])+part[2])+part[3];
#if NR_PROB_LATE_SHUFFLE
    // A measured approximation: round local partials first, exchange once.
    // It changes half-add ordering, without dropping any softmax term.
    uint other=subgroupShuffleXor(packFloat2x16(total),16u);
    total=total+unpackFloat2x16(other);
#endif
    NR_F16 sum=NR_F16(total.x+total.y);
    return sum;
}
#endif

#endif
