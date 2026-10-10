#ifndef NR_CHAIN_GLSL
#define NR_CHAIN_GLSL
// NR_CHAIN: the barrier between two dispatches of a chained run is
// replaced by a completion counter, so the next dispatch's workgroups launch
// while the previous one drains and are resident when it finishes.
//
// Semantics are the barrier's: a workgroup does nothing before its predecessor
// dispatch has completed entirely, and since that dispatch did the same, every
// earlier chained dispatch is complete too. Only the launch ramp and the drain
// overlap. The GPU launches a dispatch's workgroups after all of the previous
// dispatch's, so a waiting workgroup never holds a slot its producer needs.
//
// Counters live in the activation arena (binding 0 in every kernel) and are
// never reset: a dispatch of W workgroups adds W per frame, and frame f waits
// for need * f, f being the epoch word of the run before the chain (the C=256
// persistent run's, which counts frames from the zero-filled arena).
// Push tail (after the kernel's own fields): wait counter (u32 index, or
// ~0 for none), need (workgroups of the waited dispatch), own counter, epoch.
#ifndef NR_CHAIN
#define NR_CHAIN 0
#endif
#if NR_CHAIN
#extension GL_KHR_memory_scope_semantics : require
#define NR_CHAIN_FIELDS uint chain_wait, chain_need, chain_sig, chain_epoch;
layout(set = 0, binding = 0, std430) coherent buffer NrChainU { uint nr_chain_u[]; };
// Macros, not functions: they expand inside main, after the push block.
// A dispatch's block is 16 counter replicas NR_CHAIN_STRIDE words apart (word
// 1 of the block is the wait-timeout error word): every workgroup of the next
// dispatch polls, and one hot L2 line for all of them stalls the channel the
// running dispatch's own loads share. Producers add to all 16 (one lane each);
// a consumer polls replica (workgroup id % 16) with a SALU backoff between polls.
#ifndef NR_CHAIN_STRIDE
#define NR_CHAIN_STRIDE 256u
#endif
#ifndef NR_CHAIN_BACKOFF
#define NR_CHAIN_BACKOFF 64
#endif
#define nr_chain_wait() { \
    if (pc.chain_wait != 0xFFFFFFFFu) { \
        if (gl_LocalInvocationIndex == 0u) { \
            const uint nr_ce = atomicLoad(nr_chain_u[pc.chain_epoch], gl_ScopeDevice, \
                                          gl_StorageSemanticsBuffer, gl_SemanticsAcquire); \
            const uint nr_ct = pc.chain_need * nr_ce; \
            const uint nr_ca = pc.chain_wait + ((gl_WorkGroupID.x + gl_WorkGroupID.y * 5u + gl_WorkGroupID.z * 11u) & 15u) * NR_CHAIN_STRIDE; \
            uint nr_cb = 1u << 20, nr_cd = nr_ct; \
            while (atomicLoad(nr_chain_u[nr_ca], gl_ScopeDevice, gl_StorageSemanticsBuffer, \
                              gl_SemanticsAcquire) < nr_ct && nr_cb != 0u) { \
                --nr_cb; \
                for (int nr_k = 0; nr_k < NR_CHAIN_BACKOFF; ++nr_k) nr_cd = nr_cd * 1103515245u + 12345u; \
                if (nr_cd == 0x9e3779b9u) nr_cb = 1u; \
            } \
            if (nr_cb == 0u) atomicMax(nr_chain_u[pc.chain_sig + 1u], 1u); \
        } \
        barrier(); \
        memoryBarrier(gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire); \
    } }
#define nr_chain_signal() { \
    if (pc.chain_sig != 0xFFFFFFFFu) { \
        memoryBarrier(gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease); \
        barrier(); \
        if (gl_LocalInvocationIndex < 16u) \
            atomicAdd(nr_chain_u[pc.chain_sig + gl_LocalInvocationIndex * NR_CHAIN_STRIDE], 1u, \
                      gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease); \
    } }
#else
#define NR_CHAIN_FIELDS
#define nr_chain_wait()
#define nr_chain_signal()
#endif

// NR_TCHAIN: a tile-level dependency in place of the barrier between two
// dispatches (C=512: attention, its projection, the grouped FFN, the FFN
// projection; ViT: expand, contraction, QKV and attention, projection). A
// producer adds 1 to its counter for a unit once the unit's stores are done
// (an attention window per workgroup, a token tile per wave elsewhere); a
// consumer workgroup looks up, for every token tile it reads, the producer
// counter that covers it and waits until that counter reaches need * epoch. The
// counters are never reset: every frame adds `need` per unit, and the epoch is
// the first C=256 persistent run's frame count, final before any of these.
//
// One push word, tc_rec: ~0 or the u32 index, in the weight buffer (binding
// NR_TC_WBIND, read with scalar loads), of the dispatch's record {counters it
// waits on (activation arena), -, tile table (weight buffer, ~0 = no wait;
// entry = counter | need << 20), counters it signals (activation arena), epoch
// word, error word, NR_TC_MAGIC}. Persistent runs use their own epoch instead. Five words
// would push the kernels past the push constants RADV keeps in SGPRs, and then
// every push field is a memory load (measured: +1..3 us a GEMM dispatch).
//
// Why a waiting workgroup cannot starve its producer: each shader engine
// launches workgroups in dispatch order, so by the time a consumer holds a slot
// on an engine, every producer workgroup of that engine has been launched, and
// the producers on other engines only wait for slots held by earlier work.
// (Across engines the order is not global - a consumer can start while another
// engine still launches producers; measured.) The wait is bounded anyway and
// sets the error word, so a protocol bug cannot hang the GPU.
#ifndef NR_TCHAIN
#define NR_TCHAIN 0
#endif
#if NR_TCHAIN
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_ballot : require
#ifndef NR_TC_WBIND
#error "NR_TCHAIN: define NR_TC_WBIND (the weight buffer's binding) before including nr_chain.glsl"
#endif
#define NR_TCHAIN_FIELDS uint tc_rec;
layout(set = 0, binding = 0, std430) coherent buffer NrTChainU { uint nr_tc_u[]; };
layout(set = 0, binding = NR_TC_WBIND, std430) readonly buffer NrTChainW { uint nr_tc_w[]; };
#ifndef NR_TCHAIN_BOUND
#define NR_TCHAIN_BOUND (1u << 16)
#endif
// Every NR_TCHAIN main starts with this: the record's signal word, loaded once
// into an SGPR (a buffer load, so the compiler cannot re-load it at the end).
// A record's word 6 is NR_TC_MAGIC: a push layout that does not line up with
// the host's (a conditional push field, a short pipeline range) reads some other
// word as tc_rec, and then the kernel neither waits nor signals instead of
// counting into a random place.
#define NR_TC_MAGIC 0x54434852u
#define NR_TC_PROLOGUE \
    uint nr_tc_rec = pc.tc_rec; \
    uint nr_tc_sig = 0xFFFFFFFFu; \
    if (nr_tc_rec != 0xFFFFFFFFu && nr_tc_w[nr_tc_rec + 6u] != NR_TC_MAGIC) nr_tc_rec = 0xFFFFFFFFu; \
    if (nr_tc_rec != 0xFFFFFFFFu) nr_tc_sig = nr_tc_w[nr_tc_rec + 3u];
// Consumer: wave 0 waits for every tile named between BEGIN and END, then the
// workgroup barrier and an acquire.
#define NR_TC_WAIT_BEGIN \
    if (nr_tc_rec != 0xFFFFFFFFu && nr_tc_w[nr_tc_rec + 2u] != 0xFFFFFFFFu) { \
        if (gl_SubgroupID == 0u) { \
            const uint nr_twt = nr_tc_w[nr_tc_rec], nr_ttb = nr_tc_w[nr_tc_rec + 2u]; \
            const uint nr_te = subgroupBroadcastFirst(atomicAdd(nr_tc_u[nr_tc_w[nr_tc_rec + 4u]], 0u)); \
            uint nr_tb = NR_TCHAIN_BOUND;
// A table entry: the producer counter (low 20 bits) and how many producer units
// a frame complete it (high 12 bits); ~0 = nothing to wait for.
#define NR_TC_WAIT_TILE(nr_tile_id) { \
            const uint nr_tx = nr_tc_w[nr_ttb + uint(nr_tile_id)]; \
            if (nr_tx != 0xFFFFFFFFu) \
                while (nr_tb != 0u && subgroupBroadcastFirst(atomicAdd(nr_tc_u[nr_twt + (nr_tx & 0xFFFFFu)], 0u)) \
                       < (nr_tx >> 20u) * nr_te) \
                    --nr_tb; }
#define NR_TC_WAIT_END \
            if (nr_tb == 0u && subgroupElect()) atomicMax(nr_tc_u[nr_tc_w[nr_tc_rec + 5u]], 1u); \
        } \
        barrier(); \
        memoryBarrier(gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsAcquire); \
    }
#define nr_tchain_wait(nr_t0, nr_nt) { NR_TC_WAIT_BEGIN \
    for (uint nr_ti = 0u; nr_ti < uint(nr_nt); ++nr_ti) NR_TC_WAIT_TILE(uint(nr_t0) + nr_ti) \
    NR_TC_WAIT_END }
// Producer, per workgroup (a barrier): +1 on counter idx.
#define nr_tchain_signal(nr_idx) { \
    if (nr_tc_sig != 0xFFFFFFFFu) { \
        memoryBarrier(gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease); \
        barrier(); \
        if (gl_LocalInvocationIndex == 0u) \
            atomicAdd(nr_tc_u[nr_tc_sig + uint(nr_idx)], 1u, gl_ScopeDevice, gl_StorageSemanticsBuffer, \
                      gl_SemanticsRelease); \
    } }
// Producer, per wave (no workgroup barrier): once this wave's stores are done,
// +1 on each of the tiles t0 .. t0+nt-1.
#define nr_tchain_signal_wave(nr_t0, nr_nt) { \
    if (nr_tc_sig != 0xFFFFFFFFu) { \
        memoryBarrier(gl_ScopeDevice, gl_StorageSemanticsBuffer, gl_SemanticsRelease); \
        subgroupBarrier(); \
        if (gl_SubgroupInvocationID < uint(nr_nt)) \
            atomicAdd(nr_tc_u[nr_tc_sig + uint(nr_t0) + gl_SubgroupInvocationID], 1u, gl_ScopeDevice, \
                      gl_StorageSemanticsBuffer, gl_SemanticsRelease); \
    } }
#else
#define NR_TCHAIN_FIELDS
#define NR_TC_PROLOGUE
#define NR_TC_WAIT_BEGIN {
#define NR_TC_WAIT_TILE(nr_tile_id)
#define NR_TC_WAIT_END }
#define nr_tchain_wait(nr_t0, nr_nt)
#define nr_tchain_signal(nr_idx)
#define nr_tchain_signal_wave(nr_t0, nr_nt)
#endif
#endif
