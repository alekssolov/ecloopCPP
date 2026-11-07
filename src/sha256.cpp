// SHA-256 with hardware acceleration.
//
//   x86-64 : SHA-NI  (_mm_sha256rnds2_epu32)  -- compile-time if -msha/-march=native,
//            otherwise runtime-dispatched via __builtin_cpu_supports("sha")
//   aarch64: ARMv8 Crypto Extensions (vsha256hq_u32)
//   other  : portable scalar fallback (unrolled)
//
// Port of https://github.com/noloader/SHA-Intrinsics

#include "sha256.hpp"

#include "compat.hpp"

namespace ecloop {
    namespace {

        constexpr u32 SHA256_K[64] = {
            0x428A2F98u, 0x71374491u, 0xB5C0FBCFu, 0xE9B5DBA5u,
            0x3956C25Bu, 0x59F111F1u, 0x923F82A4u, 0xAB1C5ED5u,
            0xD807AA98u, 0x12835B01u, 0x243185BEu, 0x550C7DC3u,
            0x72BE5D74u, 0x80DEB1FEu, 0x9BDC06A7u, 0xC19BF174u,
            0xE49B69C1u, 0xEFBE4786u, 0x0FC19DC6u, 0x240CA1CCu,
            0x2DE92C6Fu, 0x4A7484AAu, 0x5CB0A9DCu, 0x76F988DAu,
            0x983E5152u, 0xA831C66Du, 0xB00327C8u, 0xBF597FC7u,
            0xC6E00BF3u, 0xD5A79147u, 0x06CA6351u, 0x14292967u,
            0x27B70A85u, 0x2E1B2138u, 0x4D2C6DFCu, 0x53380D13u,
            0x650A7354u, 0x766A0ABBu, 0x81C2C92Eu, 0x92722C85u,
            0xA2BFE8A1u, 0xA81A664Bu, 0xC24B8B70u, 0xC76C51A3u,
            0xD192E819u, 0xD6990624u, 0xF40E3585u, 0x106AA070u,
            0x19A4C116u, 0x1E376C08u, 0x2748774Cu, 0x34B0BCB5u,
            0x391C0CB3u, 0x4ED8AA4Au, 0x5B9CCA4Fu, 0x682E6FF3u,
            0x748F82EEu, 0x78A5636Fu, 0x84C87814u, 0x8CC70208u,
            0x90BEFFFAu, 0xA4506CEBu, 0xBEF9A3F7u, 0xC67178F2u,
        };

        alignas(16) constexpr u32 SHA256_IV[8] = {
            0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
            0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
        };

    } // namespace
} // namespace ecloop

// ---------------------------------------------------------------------------
// platform selection
// ---------------------------------------------------------------------------

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#define ECLOOP_SHA256_X86 1
#else
#define ECLOOP_SHA256_X86 0
#endif

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_CRYPTO)
#define ECLOOP_SHA256_ARM 1
#else
#define ECLOOP_SHA256_ARM 0
#endif

// When SHA-NI is not enabled at compile time we still build the SHA-NI kernel
// with a function-level target attribute and pick it at runtime.
#if ECLOOP_SHA256_X86 && !defined(__SHA__) && (defined(__GNUC__) || defined(__clang__))
#define ECLOOP_SHA256_X86_RUNTIME 1
#else
#define ECLOOP_SHA256_X86_RUNTIME 0
#endif

#if ECLOOP_SHA256_X86
#include <immintrin.h>
#endif

#if ECLOOP_SHA256_ARM
#include <arm_neon.h>
#endif

// ===========================================================================
// x86-64 SHA-NI
// ===========================================================================

#if ECLOOP_SHA256_X86

#if ECLOOP_SHA256_X86_RUNTIME
#define ECLOOP_SHANI_TARGET __attribute__((target("sha,sse4.1,ssse3,avx")))
#else
#define ECLOOP_SHANI_TARGET
#endif

namespace ecloop {
    namespace {

        ECLOOP_SHANI_TARGET
            void sha256_shani(u32 state[8], const u8* data, u32 length) noexcept {
            __m128i STATE0, STATE1;
            __m128i MSG, TMP;
            __m128i MSG0, MSG1, MSG2, MSG3;
            __m128i ABEF_SAVE, CDGH_SAVE;

            // The SHA instructions have no VEX form: they are legacy SSE and
            // run with whatever the upper halves of the vector registers
            // hold. After 256/512-bit code (the AVX2 RIPEMD-160, the AVX-512
            // field layer) those halves are "dirty" unless something cleared
            // them, and every SSE instruction then pays a state transition
            // -- measured here as 100x: 12.9 us per key instead of 120 ns.
            // Clearing them once per call costs nothing.
            _mm256_zeroupper();

            const __m128i MASK =
                _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);

            TMP = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&SHA256_IV[0]));
            STATE1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&SHA256_IV[4]));

            TMP = _mm_shuffle_epi32(TMP, 0xB1);          /* CDAB */
            STATE1 = _mm_shuffle_epi32(STATE1, 0x1B);    /* EFGH */
            STATE0 = _mm_alignr_epi8(TMP, STATE1, 8);    /* ABEF */
            STATE1 = _mm_blend_epi16(STATE1, TMP, 0xF0); /* CDGH */

            while (length >= 64) {
                ABEF_SAVE = STATE0;
                CDGH_SAVE = STATE1;

                /* Rounds 0-3 */
                MSG = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 0));
                MSG0 = _mm_shuffle_epi8(MSG, MASK);
                MSG = _mm_add_epi32(MSG0, _mm_set_epi64x(0xE9B5DBA5B5C0FBCFULL, 0x71374491428A2F98ULL));
                STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
                MSG = _mm_shuffle_epi32(MSG, 0x0E);
                STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);

                /* Rounds 4-7 */
                MSG1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 16));
                MSG1 = _mm_shuffle_epi8(MSG1, MASK);
                MSG = _mm_add_epi32(MSG1, _mm_set_epi64x(0xAB1C5ED5923F82A4ULL, 0x59F111F13956C25BULL));
                STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
                MSG = _mm_shuffle_epi32(MSG, 0x0E);
                STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
                MSG0 = _mm_sha256msg1_epu32(MSG0, MSG1);

                /* Rounds 8-11 */
                MSG2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 32));
                MSG2 = _mm_shuffle_epi8(MSG2, MASK);
                MSG = _mm_add_epi32(MSG2, _mm_set_epi64x(0x550C7DC3243185BEULL, 0x12835B01D807AA98ULL));
                STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
                MSG = _mm_shuffle_epi32(MSG, 0x0E);
                STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
                MSG1 = _mm_sha256msg1_epu32(MSG1, MSG2);

                /* Rounds 12-15 */
                MSG3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 48));
                MSG3 = _mm_shuffle_epi8(MSG3, MASK);
                MSG = _mm_add_epi32(MSG3, _mm_set_epi64x(0xC19BF1749BDC06A7ULL, 0x80DEB1FE72BE5D74ULL));
                STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
                TMP = _mm_alignr_epi8(MSG3, MSG2, 4);
                MSG0 = _mm_add_epi32(MSG0, TMP);
                MSG0 = _mm_sha256msg2_epu32(MSG0, MSG3);
                MSG = _mm_shuffle_epi32(MSG, 0x0E);
                STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
                MSG2 = _mm_sha256msg1_epu32(MSG2, MSG3);

                /* Rounds 16-19 */
                MSG = _mm_add_epi32(MSG0, _mm_set_epi64x(0x240CA1CC0FC19DC6ULL, 0xEFBE4786E49B69C1ULL));
                STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
                TMP = _mm_alignr_epi8(MSG0, MSG3, 4);
                MSG1 = _mm_add_epi32(MSG1, TMP);
                MSG1 = _mm_sha256msg2_epu32(MSG1, MSG0);
                MSG = _mm_shuffle_epi32(MSG, 0x0E);
                STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
                MSG3 = _mm_sha256msg1_epu32(MSG3, MSG0);

                /* Rounds 20-23 */
                MSG = _mm_add_epi32(MSG1, _mm_set_epi64x(0x76F988DA5CB0A9DCULL, 0x4A7484AA2DE92C6FULL));
                STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
                TMP = _mm_alignr_epi8(MSG1, MSG0, 4);
                MSG2 = _mm_add_epi32(MSG2, TMP);
                MSG2 = _mm_sha256msg2_epu32(MSG2, MSG1);
                MSG = _mm_shuffle_epi32(MSG, 0x0E);
                STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
                MSG0 = _mm_sha256msg1_epu32(MSG0, MSG1);

                /* Rounds 24-27 */
                MSG = _mm_add_epi32(MSG2, _mm_set_epi64x(0xBF597FC7B00327C8ULL, 0xA831C66D983E5152ULL));
                STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
                TMP = _mm_alignr_epi8(MSG2, MSG1, 4);
                MSG3 = _mm_add_epi32(MSG3, TMP);
                MSG3 = _mm_sha256msg2_epu32(MSG3, MSG2);
                MSG = _mm_shuffle_epi32(MSG, 0x0E);
                STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
                MSG1 = _mm_sha256msg1_epu32(MSG1, MSG2);

                /* Rounds 28-31 */
                MSG = _mm_add_epi32(MSG3, _mm_set_epi64x(0x1429296706CA6351ULL, 0xD5A79147C6E00BF3ULL));
                STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
                TMP = _mm_alignr_epi8(MSG3, MSG2, 4);
                MSG0 = _mm_add_epi32(MSG0, TMP);
                MSG0 = _mm_sha256msg2_epu32(MSG0, MSG3);
                MSG = _mm_shuffle_epi32(MSG, 0x0E);
                STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
                MSG2 = _mm_sha256msg1_epu32(MSG2, MSG3);

                /* Rounds 32-35 */
                MSG = _mm_add_epi32(MSG0, _mm_set_epi64x(0x53380D134D2C6DFCULL, 0x2E1B213827B70A85ULL));
                STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
                TMP = _mm_alignr_epi8(MSG0, MSG3, 4);
                MSG1 = _mm_add_epi32(MSG1, TMP);
                MSG1 = _mm_sha256msg2_epu32(MSG1, MSG0);
                MSG = _mm_shuffle_epi32(MSG, 0x0E);
                STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
                MSG3 = _mm_sha256msg1_epu32(MSG3, MSG0);

                /* Rounds 36-39 */
                MSG = _mm_add_epi32(MSG1, _mm_set_epi64x(0x92722C8581C2C92EULL, 0x766A0ABB650A7354ULL));
                STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
                TMP = _mm_alignr_epi8(MSG1, MSG0, 4);
                MSG2 = _mm_add_epi32(MSG2, TMP);
                MSG2 = _mm_sha256msg2_epu32(MSG2, MSG1);
                MSG = _mm_shuffle_epi32(MSG, 0x0E);
                STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
                MSG0 = _mm_sha256msg1_epu32(MSG0, MSG1);

                /* Rounds 40-43 */
                MSG = _mm_add_epi32(MSG2, _mm_set_epi64x(0xC76C51A3C24B8B70ULL, 0xA81A664BA2BFE8A1ULL));
                STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
                TMP = _mm_alignr_epi8(MSG2, MSG1, 4);
                MSG3 = _mm_add_epi32(MSG3, TMP);
                MSG3 = _mm_sha256msg2_epu32(MSG3, MSG2);
                MSG = _mm_shuffle_epi32(MSG, 0x0E);
                STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
                MSG1 = _mm_sha256msg1_epu32(MSG1, MSG2);

                /* Rounds 44-47 */
                MSG = _mm_add_epi32(MSG3, _mm_set_epi64x(0x106AA070F40E3585ULL, 0xD6990624D192E819ULL));
                STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
                TMP = _mm_alignr_epi8(MSG3, MSG2, 4);
                MSG0 = _mm_add_epi32(MSG0, TMP);
                MSG0 = _mm_sha256msg2_epu32(MSG0, MSG3);
                MSG = _mm_shuffle_epi32(MSG, 0x0E);
                STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
                MSG2 = _mm_sha256msg1_epu32(MSG2, MSG3);

                /* Rounds 48-51 */
                MSG = _mm_add_epi32(MSG0, _mm_set_epi64x(0x34B0BCB52748774CULL, 0x1E376C0819A4C116ULL));
                STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
                TMP = _mm_alignr_epi8(MSG0, MSG3, 4);
                MSG1 = _mm_add_epi32(MSG1, TMP);
                MSG1 = _mm_sha256msg2_epu32(MSG1, MSG0);
                MSG = _mm_shuffle_epi32(MSG, 0x0E);
                STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);
                MSG3 = _mm_sha256msg1_epu32(MSG3, MSG0);

                /* Rounds 52-55 */
                MSG = _mm_add_epi32(MSG1, _mm_set_epi64x(0x682E6FF35B9CCA4FULL, 0x4ED8AA4A391C0CB3ULL));
                STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
                TMP = _mm_alignr_epi8(MSG1, MSG0, 4);
                MSG2 = _mm_add_epi32(MSG2, TMP);
                MSG2 = _mm_sha256msg2_epu32(MSG2, MSG1);
                MSG = _mm_shuffle_epi32(MSG, 0x0E);
                STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);

                /* Rounds 56-59 */
                MSG = _mm_add_epi32(MSG2, _mm_set_epi64x(0x8CC7020884C87814ULL, 0x78A5636F748F82EEULL));
                STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
                TMP = _mm_alignr_epi8(MSG2, MSG1, 4);
                MSG3 = _mm_add_epi32(MSG3, TMP);
                MSG3 = _mm_sha256msg2_epu32(MSG3, MSG2);
                MSG = _mm_shuffle_epi32(MSG, 0x0E);
                STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);

                /* Rounds 60-63 */
                MSG = _mm_add_epi32(MSG3, _mm_set_epi64x(0xC67178F2BEF9A3F7ULL, 0xA4506CEB90BEFFFAULL));
                STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
                MSG = _mm_shuffle_epi32(MSG, 0x0E);
                STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, MSG);

                STATE0 = _mm_add_epi32(STATE0, ABEF_SAVE);
                STATE1 = _mm_add_epi32(STATE1, CDGH_SAVE);

                data += 64;
                length -= 64;
            }

            TMP = _mm_shuffle_epi32(STATE0, 0x1B);       /* FEBA */
            STATE1 = _mm_shuffle_epi32(STATE1, 0xB1);    /* DCHG */
            STATE0 = _mm_blend_epi16(TMP, STATE1, 0xF0); /* DCBA */
            STATE1 = _mm_alignr_epi8(STATE1, TMP, 8);    /* ABEF */

            _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[0]), STATE0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[4]), STATE1);
        }


    } // namespace

    bool sha256_shani_available() noexcept {
#if ECLOOP_SHA256_X86_RUNTIME
        static const bool ok = [] {
            __builtin_cpu_init();
            return __builtin_cpu_supports("sha") && __builtin_cpu_supports("sse4.1");
        }();
        return ok;
#else
        return true;   // compiled in unconditionally (-msha / -march=native on a SHA CPU)
#endif
    }

} // namespace ecloop

#endif // ECLOOP_SHA256_X86

// ===========================================================================
// aarch64 crypto extensions
// ===========================================================================

#if ECLOOP_SHA256_ARM

namespace ecloop {
    namespace {

        void sha256_neon(u32 state[8], const u8* data, u32 length) noexcept {
            uint32x4_t STATE0, STATE1, ABEF_SAVE, CDGH_SAVE;
            uint32x4_t MSG0, MSG1, MSG2, MSG3;
            uint32x4_t TMP0, TMP1, TMP2;

            STATE0 = vld1q_u32(&SHA256_IV[0]);
            STATE1 = vld1q_u32(&SHA256_IV[4]);

            while (length >= 64) {
                ABEF_SAVE = STATE0;
                CDGH_SAVE = STATE1;

                MSG0 = vld1q_u32(reinterpret_cast<const u32*>(data + 0));
                MSG1 = vld1q_u32(reinterpret_cast<const u32*>(data + 16));
                MSG2 = vld1q_u32(reinterpret_cast<const u32*>(data + 32));
                MSG3 = vld1q_u32(reinterpret_cast<const u32*>(data + 48));

                MSG0 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(MSG0)));
                MSG1 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(MSG1)));
                MSG2 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(MSG2)));
                MSG3 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(MSG3)));

                TMP0 = vaddq_u32(MSG0, vld1q_u32(&SHA256_K[0x00]));

                /* Rounds 0-3 */
                MSG0 = vsha256su0q_u32(MSG0, MSG1);
                TMP2 = STATE0;
                TMP1 = vaddq_u32(MSG1, vld1q_u32(&SHA256_K[0x04]));
                STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
                STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
                MSG0 = vsha256su1q_u32(MSG0, MSG2, MSG3);

                /* Rounds 4-7 */
                MSG1 = vsha256su0q_u32(MSG1, MSG2);
                TMP2 = STATE0;
                TMP0 = vaddq_u32(MSG2, vld1q_u32(&SHA256_K[0x08]));
                STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
                STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
                MSG1 = vsha256su1q_u32(MSG1, MSG3, MSG0);

                /* Rounds 8-11 */
                MSG2 = vsha256su0q_u32(MSG2, MSG3);
                TMP2 = STATE0;
                TMP1 = vaddq_u32(MSG3, vld1q_u32(&SHA256_K[0x0c]));
                STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
                STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
                MSG2 = vsha256su1q_u32(MSG2, MSG0, MSG1);

                /* Rounds 12-15 */
                MSG3 = vsha256su0q_u32(MSG3, MSG0);
                TMP2 = STATE0;
                TMP0 = vaddq_u32(MSG0, vld1q_u32(&SHA256_K[0x10]));
                STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
                STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
                MSG3 = vsha256su1q_u32(MSG3, MSG1, MSG2);

                /* Rounds 16-19 */
                MSG0 = vsha256su0q_u32(MSG0, MSG1);
                TMP2 = STATE0;
                TMP1 = vaddq_u32(MSG1, vld1q_u32(&SHA256_K[0x14]));
                STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
                STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
                MSG0 = vsha256su1q_u32(MSG0, MSG2, MSG3);

                /* Rounds 20-23 */
                MSG1 = vsha256su0q_u32(MSG1, MSG2);
                TMP2 = STATE0;
                TMP0 = vaddq_u32(MSG2, vld1q_u32(&SHA256_K[0x18]));
                STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
                STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
                MSG1 = vsha256su1q_u32(MSG1, MSG3, MSG0);

                /* Rounds 24-27 */
                MSG2 = vsha256su0q_u32(MSG2, MSG3);
                TMP2 = STATE0;
                TMP1 = vaddq_u32(MSG3, vld1q_u32(&SHA256_K[0x1c]));
                STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
                STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
                MSG2 = vsha256su1q_u32(MSG2, MSG0, MSG1);

                /* Rounds 28-31 */
                MSG3 = vsha256su0q_u32(MSG3, MSG0);
                TMP2 = STATE0;
                TMP0 = vaddq_u32(MSG0, vld1q_u32(&SHA256_K[0x20]));
                STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
                STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
                MSG3 = vsha256su1q_u32(MSG3, MSG1, MSG2);

                /* Rounds 32-35 */
                MSG0 = vsha256su0q_u32(MSG0, MSG1);
                TMP2 = STATE0;
                TMP1 = vaddq_u32(MSG1, vld1q_u32(&SHA256_K[0x24]));
                STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
                STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
                MSG0 = vsha256su1q_u32(MSG0, MSG2, MSG3);

                /* Rounds 36-39 */
                MSG1 = vsha256su0q_u32(MSG1, MSG2);
                TMP2 = STATE0;
                TMP0 = vaddq_u32(MSG2, vld1q_u32(&SHA256_K[0x28]));
                STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
                STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
                MSG1 = vsha256su1q_u32(MSG1, MSG3, MSG0);

                /* Rounds 40-43 */
                MSG2 = vsha256su0q_u32(MSG2, MSG3);
                TMP2 = STATE0;
                TMP1 = vaddq_u32(MSG3, vld1q_u32(&SHA256_K[0x2c]));
                STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
                STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);
                MSG2 = vsha256su1q_u32(MSG2, MSG0, MSG1);

                /* Rounds 44-47 */
                MSG3 = vsha256su0q_u32(MSG3, MSG0);
                TMP2 = STATE0;
                TMP0 = vaddq_u32(MSG0, vld1q_u32(&SHA256_K[0x30]));
                STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
                STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);
                MSG3 = vsha256su1q_u32(MSG3, MSG1, MSG2);

                /* Rounds 48-51 */
                TMP2 = STATE0;
                TMP1 = vaddq_u32(MSG1, vld1q_u32(&SHA256_K[0x34]));
                STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
                STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);

                /* Rounds 52-55 */
                TMP2 = STATE0;
                TMP0 = vaddq_u32(MSG2, vld1q_u32(&SHA256_K[0x38]));
                STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
                STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);

                /* Rounds 56-59 */
                TMP2 = STATE0;
                TMP1 = vaddq_u32(MSG3, vld1q_u32(&SHA256_K[0x3c]));
                STATE0 = vsha256hq_u32(STATE0, STATE1, TMP0);
                STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP0);

                /* Rounds 60-63 */
                TMP2 = STATE0;
                STATE0 = vsha256hq_u32(STATE0, STATE1, TMP1);
                STATE1 = vsha256h2q_u32(STATE1, TMP2, TMP1);

                STATE0 = vaddq_u32(STATE0, ABEF_SAVE);
                STATE1 = vaddq_u32(STATE1, CDGH_SAVE);

                data += 64;
                length -= 64;
            }

            vst1q_u32(&state[0], STATE0);
            vst1q_u32(&state[4], STATE1);
        }

    } // namespace
} // namespace ecloop

#endif // ECLOOP_SHA256_ARM

// ===========================================================================
// portable scalar fallback
// ===========================================================================

namespace ecloop {
    namespace {

        ECLOOP_ALWAYS_INLINE u32 sha_rotr(u32 x, unsigned n) noexcept {
            return detail::rotr32(x, n);
        }

        ECLOOP_ALWAYS_INLINE u32 sha_maj(u32 a, u32 b, u32 c) noexcept {
            return (a & b) ^ (a & c) ^ (b & c);
        }

        ECLOOP_ALWAYS_INLINE u32 sha_ch(u32 e, u32 f, u32 g) noexcept {
            return (e & f) ^ (~e & g);
        }

        ECLOOP_ALWAYS_INLINE void sha_round(u32 a, u32 b, u32 c, u32& d, u32 e, u32 f,
            u32 g, u32& h, u32 m, u32 k) noexcept {
            const u32 s = sha_ch(e, f, g) +
                (sha_rotr(e, 6) ^ sha_rotr(e, 11) ^ sha_rotr(e, 25)) + k + m;
            d += s + h;
            h += s + sha_maj(a, b, c) +
                (sha_rotr(a, 2) ^ sha_rotr(a, 13) ^ sha_rotr(a, 22));
        }

        // Unused when SHA-NI is enabled at compile time; kept so the file
        // still builds and works on targets without it.
        [[maybe_unused]] void sha256_scalar(u32 state[8], const u8* data, u32 length) noexcept {
            for (int i = 0; i < 8; ++i) state[i] = SHA256_IV[i];

            while (length >= 64) {
                u32 a = state[0], b = state[1], c = state[2], d = state[3];
                u32 e = state[4], f = state[5], g = state[6], h = state[7];

                u32 w[64];
                for (int i = 0; i < 16; ++i) w[i] = detail::load_be32(data + i * 4);

                for (int i = 16; i < 64; ++i) {
                    const u32 x = w[i - 15];
                    const u32 y = w[i - 2];
                    const u32 s0 = sha_rotr(x, 7) ^ sha_rotr(x, 18) ^ (x >> 3);
                    const u32 s1 = sha_rotr(y, 17) ^ sha_rotr(y, 19) ^ (y >> 10);
                    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
                }

                for (int i = 0; i < 64; i += 8) {
                    sha_round(a, b, c, d, e, f, g, h, w[i + 0], SHA256_K[i + 0]);
                    sha_round(h, a, b, c, d, e, f, g, w[i + 1], SHA256_K[i + 1]);
                    sha_round(g, h, a, b, c, d, e, f, w[i + 2], SHA256_K[i + 2]);
                    sha_round(f, g, h, a, b, c, d, e, w[i + 3], SHA256_K[i + 3]);
                    sha_round(e, f, g, h, a, b, c, d, w[i + 4], SHA256_K[i + 4]);
                    sha_round(d, e, f, g, h, a, b, c, w[i + 5], SHA256_K[i + 5]);
                    sha_round(c, d, e, f, g, h, a, b, w[i + 6], SHA256_K[i + 6]);
                    sha_round(b, c, d, e, f, g, h, a, w[i + 7], SHA256_K[i + 7]);
                }

                state[0] += a; state[1] += b; state[2] += c; state[3] += d;
                state[4] += e; state[5] += f; state[6] += g; state[7] += h;

                data += 64;
                length -= 64;
            }
        }

    } // namespace
} // namespace ecloop

// ===========================================================================
// dispatch
// ===========================================================================

namespace ecloop {

#if ECLOOP_SHA256_X86 && !ECLOOP_SHA256_X86_RUNTIME

    // SHA-NI enabled at compile time (-msha / -march=native): call directly.
    void sha256_final(u32 state[8], const u8 data[], u32 length) {
        sha256_shani(state, data, length);
    }

#elif ECLOOP_SHA256_X86_RUNTIME

    // SHA-NI not enabled at compile time: probe the CPU once, then dispatch.
    namespace {
        using sha256_fn = void (*)(u32*, const u8*, u32) noexcept;

        sha256_fn pick_sha256() noexcept {
            __builtin_cpu_init();
            if (__builtin_cpu_supports("sha") && __builtin_cpu_supports("sse4.1")) {
                return &sha256_shani;
            }
            return &sha256_scalar;
        }

        const sha256_fn SHA256_IMPL = pick_sha256();
    } // namespace

    void sha256_final(u32 state[8], const u8 data[], u32 length) {
        SHA256_IMPL(state, data, length);
    }

#elif ECLOOP_SHA256_ARM

    void sha256_final(u32 state[8], const u8 data[], u32 length) {
        sha256_neon(state, data, length);
    }

#else

    void sha256_final(u32 state[8], const u8 data[], u32 length) {
        sha256_scalar(state, data, length);
    }

#endif

#if !ECLOOP_SHA256_X86
    bool sha256_shani_available() noexcept { return false; }
#endif

    // The portable kernel by name, for the forced-scalar path and tests.
    void sha256_final_scalar(u32 state[8], const u8 data[], u32 length) {
        sha256_scalar(state, data, length);
    }

} // namespace ecloop
