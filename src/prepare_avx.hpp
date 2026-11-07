// Vectorised message preparation for batch-16 hashing.
// Port of lib/prepare_avx.c.
//
// Two costs are removed versus prepare33/prepare65.
//
// 1. The byte swap. The scalar version walks the four limbs, swap64s each
//    and memcpys it into place. The whole 32-byte coordinate becomes its
//    big-endian form with two instructions instead: permute the qwords
//    into reverse order, then byte-reverse inside each qword.
//
// 2. The memset. The batch buffers used to be zeroed before every call --
//    1 KB and 2 KB per batch -- purely to get the padding right. But the
//    padding never changes. So the buffers are written once with their
//    padding and reused, and each call touches only the bytes that differ.
//
// NOTE ON STORAGE. In the C original the reusable buffers are function
// statics, which is safe there because ecloop forks worker processes.
// This project uses threads, so anything holding per-call state must be
// thread_local -- see addr_avx512.cpp.

#pragma once

#include "types.hpp"
#include "ecc.hpp"
#include "compat.hpp"

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define ECLOOP_HAVE_PREPARE_AVX 1
#else
#define ECLOOP_HAVE_PREPARE_AVX 0
#endif

#if ECLOOP_HAVE_PREPARE_AVX

#include <immintrin.h>
#include <cstring>

#pragma GCC push_options
#pragma GCC target("avx2")
namespace ecloop {
    namespace detail_prep {

        // vpshufb only works inside 128-bit lanes, hence the separate
        // permute for the cross-lane part.
        alignas(32) constexpr u8 PREP_BSWAP64[32] = {
            7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
            7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
        };

        // [l0,l1,l2,l3] (l0 least significant) -> BE(l3) BE(l2) BE(l1) BE(l0)
        ECLOOP_ALWAYS_INLINE __m256i prep_fe_be(const u64* limbs) noexcept {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(limbs));
            v = _mm256_permute4x64_epi64(v, _MM_SHUFFLE(0, 1, 2, 3));
            return _mm256_shuffle_epi8(
                v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(PREP_BSWAP64)));
        }

    } // namespace detail_prep

    // Writes bytes 0..32 only. Bytes 33..63 must already hold the padding:
    // 0x80 at 33, zeros, then 0x0108 at 62..63 (message length 264 bits).
    ECLOOP_ALWAYS_INLINE void prepare33_fast(u8 msg[64], const pe* point) noexcept {
        msg[0] = (point->y[0] & 1) ? 0x03 : 0x02;
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(msg + 1),
                            detail_prep::prep_fe_be(point->x));
    }

    // Writes bytes 0..64 only. Bytes 65..127 must already hold the padding:
    // 0x80 at 65, zeros, then 0x0208 at 126..127 (message length 520 bits).
    ECLOOP_ALWAYS_INLINE void prepare65_fast(u8 msg[128], const pe* point) noexcept {
        msg[0] = 0x04;
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(msg + 1),
                            detail_prep::prep_fe_be(point->x));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(msg + 33),
                            detail_prep::prep_fe_be(point->y));
    }

    // One-time padding setup.
    inline void prep_pad33(u8 msg[16][64]) noexcept {
        std::memset(msg, 0, 16 * 64);
        for (int i = 0; i < 16; ++i) {
            msg[i][33] = 0x80;
            msg[i][62] = 0x01;
            msg[i][63] = 0x08;
        }
    }

    inline void prep_pad65(u8 msg[16][128]) noexcept {
        std::memset(msg, 0, 16 * 128);
        for (int i = 0; i < 16; ++i) {
            msg[i][65] = 0x80;
            msg[i][126] = 0x02;
            msg[i][127] = 0x08;
        }
    }

} // namespace ecloop

#pragma GCC pop_options
#endif // ECLOOP_HAVE_PREPARE_AVX
