// Blocked Bloom prefilter in front of btree_search. Replaces the one-bit
// bitmap of lib/btree_filter.c with the same three-call interface.
//
// WHAT IT DOES. One 64-byte block (a cache line) is chosen per hash, and
// BTF_K bits inside it are set at build time / tested at probe time. Bit
// clear anywhere means the record is definitely absent, so btree_search is
// skipped. All bits set means "maybe", and btree_search then gives the
// definitive answer. Same guarantee as before: the filter is built from
// the heap itself, so a false NEGATIVE is impossible; a false positive only
// costs one btree_search.
//
// WHY BLOCKED. A probe touches exactly ONE cache line, like the old bitmap,
// but the K bits make the false-positive rate fall as (set fraction)^K
// instead of linearly:
//     old bitmap, 21M records, 64 MB:    3.9% false positives
//     this, K = 4, 32 MB:                ~0.9%
//     this, K = 4, 24 MB:                ~2%
// so it needs a half to a third of the memory for the same rate. That is
// what matters on a many-core machine: 24-32 MB stays in the L3 of one
// CCD (EPYC Zen 5c: 32 MB per 16 cores) where 64 MB does not, and the two
// probes per key stop being DRAM traffic. The btree is only reached for
// the false positives, so their traffic falls 2-4x too.
//
// INDEX. Block from bits 96..159 of the hash (h[3], h[4]) through a
// multiplicative mixer and a multiply-high range reduction, so the size
// need not be a power of two. Bit positions from h[1], h[2]. h[0]'s low
// 24 bits are the btree bucket and stay unused here, as before.
//
// MEMORY. Allocated 2 MB aligned before fork(), so every worker shares the
// same pages copy-on-write; on Linux it is madvise(MADV_HUGEPAGE)'d so the
// probes do not also miss the TLB (or set transparent_hugepage to
// "always" system-wide).
//
// ECLOOP_NO_FILTER=1 disables it. ECLOOP_FILTER_MB=N sets the size in MB;
// ECLOOP_FILTER_BITS=N (the old switch, log2 of the bit count) still works.
// Default: 12 bits per record, rounded up to whole 2 MB.

#pragma once

#include "types.hpp"
#include "compat.hpp"   // ECLOOP_ALWAYS_INLINE

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace ecloop {

        // Read-only after btf_build. Exposed so the probes below can inline.
        // BTF_BITS != nullptr also serves the engines as "filter present".
        extern u64* BTF_BITS;      // BTF_NBLOCKS blocks of 8 x u64
        extern u64  BTF_NBLOCKS;
        inline constexpr int BTF_K = 4;

        ECLOOP_ALWAYS_INLINE u64 btf_block(const u32* h) noexcept {
            const u64 v = (static_cast<u64>(h[4]) << 32) | h[3];
            const u64 m = v * 0x9E3779B97F4A7C15ULL;
            return static_cast<u64>((static_cast<unsigned __int128>(m) * BTF_NBLOCKS) >> 64);
        }

        // The BTF_K bit positions (0..511) inside the block.
        ECLOOP_ALWAYS_INLINE void btf_positions(const u32* h, u32 p[BTF_K]) noexcept {
            const u64 b = (static_cast<u64>(h[2]) << 32) | h[1];
            p[0] = static_cast<u32>(b) & 511u;
            p[1] = static_cast<u32>(b >> 9) & 511u;
            p[2] = static_cast<u32>(b >> 18) & 511u;
            p[3] = static_cast<u32>(b >> 27) & 511u;
        }

        // Worth looking in the btree? With no filter, always yes.
        ECLOOP_ALWAYS_INLINE bool btf_maybe(const u32* h) noexcept {
            if (!BTF_BITS) return true;
            const u64* blk = BTF_BITS + 8 * btf_block(h);
            u32 p[BTF_K];
            btf_positions(h, p);
            u64 all = ~0ULL;
            for (int i = 0; i < BTF_K; ++i) all &= blk[p[i] >> 6] >> (p[i] & 63);
            return (all & 1) != 0;
        }

        ECLOOP_ALWAYS_INLINE void btf_prefetch(const u32* h) noexcept {
#if defined(__x86_64__)
            if (!BTF_BITS) return;
            _mm_prefetch(reinterpret_cast<const char*>(BTF_BITS + 8 * btf_block(h)),
                         _MM_HINT_T0);
#else
            (void)h;
#endif
        }

        // Build from an already loaded tree. Call ONCE, right after
        // btree_load and strictly BEFORE fork().
        void btf_build(const u32* tree);

        // Same, sized explicitly (bytes), for tests and benchmarks. Returns
        // false if the allocation failed. Frees any previous filter.
        bool btf_build_sized(const u32* tree, std::size_t bytes);

} // namespace ecloop
