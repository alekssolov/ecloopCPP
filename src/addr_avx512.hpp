// hash160 for 16 points per pass. Port of lib/addr_avx512.c.
//
// Ties two kernels together:
//   sha256_x16_avx512   (s2n/sha256_x16_avx512.S, port of minio/sha256-simd)
//   rmd160_batch16_soa  (rmd160_avx512.cpp)
//
// The 16-wide SHA-256 emits its digest already lane-major, which is exactly
// the layout the 16-wide RIPEMD-160 wants -- nothing is transposed between
// the two hashes.
//
// SELF TEST. addr_avx512_available() runs a known-answer test once per
// process, on the compressed public key of private key 1 (hash160
// 751E76E8...). On mismatch the AVX-512 path stays off, the old path runs
// and the reason goes to stderr. That test catches every way this code has
// been observed to produce silently wrong hashes: object built for the
// wrong ABI, truncated object, bad round-constant table, byte swap done
// twice or not at all.
//
// Set ECLOOP_NO_AVX512_HASH=1 in the environment to force the old path.

#pragma once

#include "types.hpp"
#include "ecc.hpp"
#include "addr.hpp"
#include "compat.hpp"

namespace ecloop {

    // Keys per AVX-512 hashing batch. 16 = one SHA-256 x16 and one
    // RIPEMD-160 x16. 32 (-DECLOOP_A512_BATCH32=1) = two SHA passes and one
    // RIPEMD-160 x32 with the two batches interleaved in the core, for
    // cores whose vector ports the x16 chain leaves idle: measured +10% on
    // the RIPEMD-160 part on Sapphire Rapids, expected more on Zen 5 (four
    // vector pipes); a wash where one 512-bit port is the limit.
#if defined(ECLOOP_A512_BATCH32) && ECLOOP_A512_BATCH32
    inline constexpr std::size_t A512_BATCH = 32;
#else
    inline constexpr std::size_t A512_BATCH = 16;
#endif

    // Probed and self-tested during static initialisation, before main.
    // Namespace-scope rather than a function-local static for the same
    // reason as EC_BATCH8_OK: no guard check on the hot path.
    extern const bool ADDR_AVX512_OK;

    ECLOOP_ALWAYS_INLINE bool addr_avx512_available() noexcept { return ADDR_AVX512_OK; }

    // Exactly 16 (or 32) points each. Only valid when addr_avx512_available().
    void addr33_batch16(h160_t* hashes, const pe* points) noexcept;
    void addr65_batch16(h160_t* hashes, const pe* points) noexcept;
    void addr33_batch32(h160_t* hashes, const pe* points) noexcept;
    void addr65_batch32(h160_t* hashes, const pe* points) noexcept;

    ECLOOP_ALWAYS_INLINE void addr33_batchA(h160_t* hashes, const pe* points) noexcept {
        if constexpr (A512_BATCH == 32) addr33_batch32(hashes, points); else addr33_batch16(hashes, points);
    }
    ECLOOP_ALWAYS_INLINE void addr65_batchA(h160_t* hashes, const pe* points) noexcept {
        if constexpr (A512_BATCH == 32) addr65_batch32(hashes, points); else addr65_batch16(hashes, points);
    }

} // namespace ecloop
