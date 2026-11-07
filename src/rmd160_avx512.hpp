// RIPEMD-160 over 16 messages per pass (AVX-512).
//
// Port of lib/rmd160_avx512.c. Two entry points:
//
//   rmd160_batch16_soa -- input is already lane-major (word i of all 16
//       messages in one zmm). This is exactly what the 16-wide SHA-256
//       emits, so nothing has to be transposed between the two hashes.
//
//   rmd160_batch16     -- same conventions as rmd160_batch, 16 lanes
//       instead of RMD_LEN. Transposes by gathering columns.
//
// Byte order: the input is the SHA-256 state in host order and the byte
// swap happens here, matching rmd160_batch. Exactly one swap must occur
// between the two hashes -- with none, every hash comes out wrong.
//
// Requires AVX512F and AVX512BW (vpshufb on zmm). When the target has
// neither, the declarations remain but the definitions trap: call
// rmd160_avx512_compiled() first, or go through addr_dispatch.

#pragma once

#include "types.hpp"

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define ECLOOP_HAVE_RMD160_AVX512 1
#else
#define ECLOOP_HAVE_RMD160_AVX512 0
#endif

#if ECLOOP_HAVE_RMD160_AVX512
#include <immintrin.h>
#endif

namespace ecloop {

    // True when this translation unit was built with AVX-512 enabled. Says
    // nothing about the CPU actually running the binary -- see
    // addr_avx512_available() for that.
    inline constexpr bool rmd160_avx512_compiled() noexcept {
        return ECLOOP_HAVE_RMD160_AVX512 != 0;
    }

#if ECLOOP_HAVE_RMD160_AVX512

    // state8[i] holds word i of all 16 messages.
    void rmd160_batch16_soa(u32 r[16][5], const __m512i state8[8]) noexcept;

    // Two SoA batches at once, interleaved in the core (see the .cpp): the
    // dependent round chain of one batch leaves the ports idle, two fill
    // them. r[0..15] from stateA, r[16..31] from stateB.
    void rmd160_batch32_soa(u32 r[32][5], const __m512i stateA[8], const __m512i stateB[8]) noexcept;

#endif

    // Array-of-structs entry point, mirrors rmd160_batch.
    void rmd160_batch16(u32 r[16][5], const u32 x[16][16]) noexcept;

} // namespace ecloop
