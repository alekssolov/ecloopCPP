// Radix-2^52 field arithmetic for secp256k1, 8 lanes per zmm (SoA), built on
// AVX512-IFMA with LAZY reduction.
//
// WHY A SECOND REPRESENTATION. The batch-8 multiply in ec_batch8.cpp keeps
// every value in radix 2^64 and converts to radix 2^52 and back INSIDE the
// multiply, then finishes with a full 256-bit conditional subtraction. Of
// its ~200 vector instructions only 62 are IFMA; the rest is conversion,
// carry chains and the final compare-with-p. Adds and subtracts pay a
// similar price (~30 instructions each) for carry propagation across
// 64-bit limbs plus the conditional correction.
//
// Here the value STAYS in radix 2^52 between operations, and nothing is
// canonical until it has to be:
//
//   * add/sub are 5 or 10 plain vector adds: no carries, no compare.
//   * mul/sqr take limbs < 2^52 and return limbs < 2^52 with the value
//     merely < 2^256 + 2^220 ("N-form"), which is good enough as input for
//     the next multiply. 85 instructions for a multiply (67 of them IFMA),
//     75 for a square (47 IFMA). No radix conversion, no vpmullq.
//   * a value is made canonical (< p, packed to 4 x u64) only when it
//     leaves the layer: to be hashed, or compared.
//
// This is the layout Intel's crypto_mb uses and the one the eng25519 /
// "faster TLS handshake" line of work uses for X25519 (radix 2^51 there,
// because 255-bit); it is adapted here to p = 2^256 - 2^32 - 977.
//
// LIMB LAYOUT.  value = l0 + l1*2^52 + l2*2^104 + l3*2^156 + l4*2^208.
// The top limb NOMINALLY holds 48 bits (4*52 + 48 = 256). That slack is
// what makes normalisation terminate: the bits of l4 above 48 have weight
// 2^256 == C (mod p), C = 2^32 + 977, so they fold into l0 as a product
// below 2^47 and the following carry chain cannot push l4 back over 2^52.
//
// BOUNDS.  The IFMA instructions read only the LOW 52 BITS of each source
// limb, so every limb fed to a multiply must be < 2^52 -- the one hard
// rule of this file. The forms used are:
//
//   N-form   l0..l3 < 2^52, l4 < 2^48 + 2^12.  Output of mul, sqr, from64,
//            normalize. Valid multiply input. Value < 2^256 + 2^220.
//   lazy     anything with limbs < 2^61. Output of add/sub. NOT a valid
//            multiply input: call normalize first.
//
// sub(a, b) computes a + 4p - b limb by limb, which needs b's limbs below
// the limbs of 4p: b0..b3 < 2^54 - 4, b4 < 2^50 - 4. Every N-form value
// and every add or sub of two N-form values satisfies that. Deeper
// chains do not -- normalise in between.

#pragma once

#include <immintrin.h>
#include <cstddef>

#include "types.hpp"
#include "compat.hpp"
#include "ec_batch8.hpp"   // Fe8 (radix-2^64 SoA) for the boundary conversions

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define ECLOOP_HAVE_FE52 1
#else
#define ECLOOP_HAVE_FE52 0
#endif

#if ECLOOP_HAVE_FE52


#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl,avx512ifma,bmi2")
namespace ecloop {

    struct alignas(64) Fe52 {
        __m512i l[5];
    };

    // Stored form (same bytes as Fe52, but a plain POD for arrays/tables).
    struct alignas(64) Fe52s {
        u64 v[5][8];
    };
    static_assert(sizeof(Fe52s) == 320, "Fe52s must be 5 rows of 8 x u64");

    namespace fe52c {
        constexpr u64 M52 = 0x000FFFFFFFFFFFFFull;
        constexpr u64 M48 = 0x0000FFFFFFFFFFFFull;
        constexpr u64 C   = 0x00000001000003D1ull; // 2^32 + 977 == 2^256 (mod p)
        constexpr u64 K   = 16 * C;                // 2^260 (mod p) = 68719492368
        // p in radix 2^52 with a 48-bit top limb
        constexpr u64 P0 = M52 + 1 - C;            // 2^52 - C
        constexpr u64 P1 = M52;
        constexpr u64 P4 = M48;
        // 4p, the bias used by sub
        constexpr u64 P4_0 = 4 * P0;               // 2^54 - 4C
        constexpr u64 P4_1 = 4 * P1;               // 2^54 - 4
        constexpr u64 P4_4 = 4 * P4;               // 2^50 - 4
        static_assert(P0 == 0x000FFFFEFFFFFC2Full, "p0");
    }

    // ── constants as vectors (materialised by the compiler at use sites) ─────
    ECLOOP_ALWAYS_INLINE __m512i fe52_m52() noexcept { return _mm512_set1_epi64((long long)fe52c::M52); }
    ECLOOP_ALWAYS_INLINE __m512i fe52_m48() noexcept { return _mm512_set1_epi64((long long)fe52c::M48); }
    ECLOOP_ALWAYS_INLINE __m512i fe52_vC()  noexcept { return _mm512_set1_epi64((long long)fe52c::C); }
    ECLOOP_ALWAYS_INLINE __m512i fe52_vK()  noexcept { return _mm512_set1_epi64((long long)fe52c::K); }

    // ── load / store ─────────────────────────────────────────────────────────
    ECLOOP_ALWAYS_INLINE Fe52 fe52_load(const Fe52s* s) noexcept {
        Fe52 r;
        for (int i = 0; i < 5; ++i) r.l[i] = _mm512_loadu_si512(s->v[i]);
        return r;
    }
    ECLOOP_ALWAYS_INLINE void fe52_store(Fe52s* s, const Fe52& a) noexcept {
        for (int i = 0; i < 5; ++i) _mm512_storeu_si512(s->v[i], a.l[i]);
    }

    ECLOOP_ALWAYS_INLINE Fe52 fe52_zero() noexcept {
        Fe52 r;
        for (int i = 0; i < 5; ++i) r.l[i] = _mm512_setzero_si512();
        return r;
    }
    ECLOOP_ALWAYS_INLINE Fe52 fe52_one() noexcept {
        Fe52 r = fe52_zero();
        r.l[0] = _mm512_set1_epi64(1);
        return r;
    }

    // ── boundary: radix 2^64 (Fe8 / fe) -> radix 2^52 ────────────────────────
    // Output is N-form (in fact canonical if the input was).
    ECLOOP_ALWAYS_INLINE Fe52 fe52_from64(const __m512i x0, const __m512i x1,
                                          const __m512i x2, const __m512i x3) noexcept {
        const __m512i m = fe52_m52();
        Fe52 r;
        r.l[0] = _mm512_and_si512(x0, m);
        r.l[1] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(x0, 52), _mm512_slli_epi64(x1, 12)), m);
        r.l[2] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(x1, 40), _mm512_slli_epi64(x2, 24)), m);
        r.l[3] = _mm512_and_si512(_mm512_or_si512(_mm512_srli_epi64(x2, 28), _mm512_slli_epi64(x3, 36)), m);
        r.l[4] = _mm512_srli_epi64(x3, 16);
        return r;
    }
    ECLOOP_ALWAYS_INLINE Fe52 fe52_from_fe8(const Fe8* a) noexcept {
        return fe52_from64(_mm512_loadu_si512(a->v[0]), _mm512_loadu_si512(a->v[1]),
                           _mm512_loadu_si512(a->v[2]), _mm512_loadu_si512(a->v[3]));
    }
    // All 8 lanes = the same scalar value.
    ECLOOP_ALWAYS_INLINE Fe52 fe52_bcast(const u64 a[4]) noexcept {
        return fe52_from64(_mm512_set1_epi64((long long)a[0]), _mm512_set1_epi64((long long)a[1]),
                           _mm512_set1_epi64((long long)a[2]), _mm512_set1_epi64((long long)a[3]));
    }

    // ── normalise: lazy -> N-form ────────────────────────────────────────────
    // Input limbs < 2^62. 15 instructions, one of them IFMA.
    ECLOOP_ALWAYS_INLINE Fe52 fe52_normalize(Fe52 a) noexcept {
        const __m512i m = fe52_m52();
        // top bits (weight 2^256 == C) first, so the chain below terminates
        const __m512i t = _mm512_srli_epi64(a.l[4], 48);            // < 2^14
        a.l[4] = _mm512_and_si512(a.l[4], fe52_m48());
        a.l[0] = _mm512_madd52lo_epu64(a.l[0], t, fe52_vC());       // + t*C, < 2^47
        __m512i c;
        c = _mm512_srli_epi64(a.l[0], 52); a.l[0] = _mm512_and_si512(a.l[0], m); a.l[1] = _mm512_add_epi64(a.l[1], c);
        c = _mm512_srli_epi64(a.l[1], 52); a.l[1] = _mm512_and_si512(a.l[1], m); a.l[2] = _mm512_add_epi64(a.l[2], c);
        c = _mm512_srli_epi64(a.l[2], 52); a.l[2] = _mm512_and_si512(a.l[2], m); a.l[3] = _mm512_add_epi64(a.l[3], c);
        c = _mm512_srli_epi64(a.l[3], 52); a.l[3] = _mm512_and_si512(a.l[3], m); a.l[4] = _mm512_add_epi64(a.l[4], c);
        // now l0..l3 < 2^52 and l4 < 2^48 + 2^11
        return a;
    }

    // ── add / sub (lazy) ─────────────────────────────────────────────────────
    ECLOOP_ALWAYS_INLINE Fe52 fe52_add(const Fe52& a, const Fe52& b) noexcept {
        Fe52 r;
        for (int i = 0; i < 5; ++i) r.l[i] = _mm512_add_epi64(a.l[i], b.l[i]);
        return r;
    }
    // a + 4p - b. See the header comment for the bound on b.
    ECLOOP_ALWAYS_INLINE Fe52 fe52_sub(const Fe52& a, const Fe52& b) noexcept {
        const __m512i p0 = _mm512_set1_epi64((long long)fe52c::P4_0);
        const __m512i p1 = _mm512_set1_epi64((long long)fe52c::P4_1);
        const __m512i p4 = _mm512_set1_epi64((long long)fe52c::P4_4);
        Fe52 r;
        r.l[0] = _mm512_sub_epi64(_mm512_add_epi64(a.l[0], p0), b.l[0]);
        r.l[1] = _mm512_sub_epi64(_mm512_add_epi64(a.l[1], p1), b.l[1]);
        r.l[2] = _mm512_sub_epi64(_mm512_add_epi64(a.l[2], p1), b.l[2]);
        r.l[3] = _mm512_sub_epi64(_mm512_add_epi64(a.l[3], p1), b.l[3]);
        r.l[4] = _mm512_sub_epi64(_mm512_add_epi64(a.l[4], p4), b.l[4]);
        return r;
    }
    // a + 16p - b, for a lazier b: b0..b3 < 2^56 - 16, b4 < 2^52 - 16, i.e.
    // any value whose limbs came out of at most two add/sub levels.
    ECLOOP_ALWAYS_INLINE Fe52 fe52_sub16(const Fe52& a, const Fe52& b) noexcept {
        const __m512i p0 = _mm512_set1_epi64((long long)(16 * fe52c::P0));
        const __m512i p1 = _mm512_set1_epi64((long long)(16 * fe52c::P1));
        const __m512i p4 = _mm512_set1_epi64((long long)(16 * fe52c::P4));
        Fe52 r;
        r.l[0] = _mm512_sub_epi64(_mm512_add_epi64(a.l[0], p0), b.l[0]);
        r.l[1] = _mm512_sub_epi64(_mm512_add_epi64(a.l[1], p1), b.l[1]);
        r.l[2] = _mm512_sub_epi64(_mm512_add_epi64(a.l[2], p1), b.l[2]);
        r.l[3] = _mm512_sub_epi64(_mm512_add_epi64(a.l[3], p1), b.l[3]);
        r.l[4] = _mm512_sub_epi64(_mm512_add_epi64(a.l[4], p4), b.l[4]);
        return r;
    }
    // 4p - b: negation with the same bound on b as fe52_sub.
    ECLOOP_ALWAYS_INLINE Fe52 fe52_neg(const Fe52& b) noexcept {
        return fe52_sub(fe52_zero(), b);
    }
    ECLOOP_ALWAYS_INLINE Fe52 fe52_dbl(const Fe52& a) noexcept { return fe52_add(a, a); }

    // ── the reduction shared by mul and sqr ──────────────────────────────────
    // D[0..9]: raw column sums of the schoolbook product, D[k] < 2^56, and
    // D[9] < 2^52 (it only ever receives one term). Returns N-form.
    ECLOOP_ALWAYS_INLINE Fe52 fe52_reduce(const __m512i D[10]) noexcept {
        const __m512i vK = fe52_vK();
        // D[k] for k >= 5 has weight 2^(52k) == K * 2^(52(k-5)). IFMA only
        // sees the low 52 bits of D[k]; the bits above (h < 2^4, weight
        // 2^(52(k+1))) are folded separately as h*K, which is exact in one
        // madd52lo since it is below 2^41. D[9] < 2^52 has no such bits.
        const __m512i h5 = _mm512_srli_epi64(D[5], 52);
        const __m512i h6 = _mm512_srli_epi64(D[6], 52);
        const __m512i h7 = _mm512_srli_epi64(D[7], 52);
        const __m512i h8 = _mm512_srli_epi64(D[8], 52);
        Fe52 E;
        E.l[0] = _mm512_madd52lo_epu64(D[0], D[5], vK);
        E.l[1] = _mm512_madd52lo_epu64(D[1], D[6], vK);
        E.l[2] = _mm512_madd52lo_epu64(D[2], D[7], vK);
        E.l[3] = _mm512_madd52lo_epu64(D[3], D[8], vK);
        E.l[4] = _mm512_madd52lo_epu64(D[4], D[9], vK);
        E.l[1] = _mm512_madd52hi_epu64(E.l[1], D[5], vK);
        E.l[2] = _mm512_madd52hi_epu64(E.l[2], D[6], vK);
        E.l[3] = _mm512_madd52hi_epu64(E.l[3], D[7], vK);
        E.l[4] = _mm512_madd52hi_epu64(E.l[4], D[8], vK);
        __m512i E5 = _mm512_madd52hi_epu64(_mm512_setzero_si512(), D[9], vK); // < 2^37
        E.l[1] = _mm512_madd52lo_epu64(E.l[1], h5, vK);
        E.l[2] = _mm512_madd52lo_epu64(E.l[2], h6, vK);
        E.l[3] = _mm512_madd52lo_epu64(E.l[3], h7, vK);
        E.l[4] = _mm512_madd52lo_epu64(E.l[4], h8, vK);
        // E[k] < 2^56 + 2^52 + 2^52 + 2^41 < 2^57.  E5 has weight 2^260 == K.
        E.l[0] = _mm512_madd52lo_epu64(E.l[0], E5, vK);
        E.l[1] = _mm512_madd52hi_epu64(E.l[1], E5, vK);
        // E[0] < 2^57 + 2^52, then normalize: t < 2^9, all carries < 2^7.
        return fe52_normalize(E);
    }

    // ── multiply: a, b with all limbs < 2^52 (N-form) -> N-form ──────────────
    ECLOOP_ALWAYS_INLINE Fe52 fe52_mul(const Fe52& a, const Fe52& b) noexcept {
        __m512i D[10];
        const __m512i z = _mm512_setzero_si512();
        // Column k collects lo52(a_i b_j) for i+j=k and hi52(a_i b_j) for
        // i+j=k-1: at most 9 terms below 2^52, so D[k] < 2^56.
        D[0] = _mm512_madd52lo_epu64(z, a.l[0], b.l[0]);
        D[1] = _mm512_madd52hi_epu64(z, a.l[0], b.l[0]);
        D[1] = _mm512_madd52lo_epu64(D[1], a.l[0], b.l[1]);
        D[1] = _mm512_madd52lo_epu64(D[1], a.l[1], b.l[0]);
        D[2] = _mm512_madd52hi_epu64(z, a.l[0], b.l[1]);
        D[2] = _mm512_madd52hi_epu64(D[2], a.l[1], b.l[0]);
        D[2] = _mm512_madd52lo_epu64(D[2], a.l[0], b.l[2]);
        D[2] = _mm512_madd52lo_epu64(D[2], a.l[1], b.l[1]);
        D[2] = _mm512_madd52lo_epu64(D[2], a.l[2], b.l[0]);
        D[3] = _mm512_madd52hi_epu64(z, a.l[0], b.l[2]);
        D[3] = _mm512_madd52hi_epu64(D[3], a.l[1], b.l[1]);
        D[3] = _mm512_madd52hi_epu64(D[3], a.l[2], b.l[0]);
        D[3] = _mm512_madd52lo_epu64(D[3], a.l[0], b.l[3]);
        D[3] = _mm512_madd52lo_epu64(D[3], a.l[1], b.l[2]);
        D[3] = _mm512_madd52lo_epu64(D[3], a.l[2], b.l[1]);
        D[3] = _mm512_madd52lo_epu64(D[3], a.l[3], b.l[0]);
        D[4] = _mm512_madd52hi_epu64(z, a.l[0], b.l[3]);
        D[4] = _mm512_madd52hi_epu64(D[4], a.l[1], b.l[2]);
        D[4] = _mm512_madd52hi_epu64(D[4], a.l[2], b.l[1]);
        D[4] = _mm512_madd52hi_epu64(D[4], a.l[3], b.l[0]);
        D[4] = _mm512_madd52lo_epu64(D[4], a.l[0], b.l[4]);
        D[4] = _mm512_madd52lo_epu64(D[4], a.l[1], b.l[3]);
        D[4] = _mm512_madd52lo_epu64(D[4], a.l[2], b.l[2]);
        D[4] = _mm512_madd52lo_epu64(D[4], a.l[3], b.l[1]);
        D[4] = _mm512_madd52lo_epu64(D[4], a.l[4], b.l[0]);
        D[5] = _mm512_madd52hi_epu64(z, a.l[0], b.l[4]);
        D[5] = _mm512_madd52hi_epu64(D[5], a.l[1], b.l[3]);
        D[5] = _mm512_madd52hi_epu64(D[5], a.l[2], b.l[2]);
        D[5] = _mm512_madd52hi_epu64(D[5], a.l[3], b.l[1]);
        D[5] = _mm512_madd52hi_epu64(D[5], a.l[4], b.l[0]);
        D[5] = _mm512_madd52lo_epu64(D[5], a.l[1], b.l[4]);
        D[5] = _mm512_madd52lo_epu64(D[5], a.l[2], b.l[3]);
        D[5] = _mm512_madd52lo_epu64(D[5], a.l[3], b.l[2]);
        D[5] = _mm512_madd52lo_epu64(D[5], a.l[4], b.l[1]);
        D[6] = _mm512_madd52hi_epu64(z, a.l[1], b.l[4]);
        D[6] = _mm512_madd52hi_epu64(D[6], a.l[2], b.l[3]);
        D[6] = _mm512_madd52hi_epu64(D[6], a.l[3], b.l[2]);
        D[6] = _mm512_madd52hi_epu64(D[6], a.l[4], b.l[1]);
        D[6] = _mm512_madd52lo_epu64(D[6], a.l[2], b.l[4]);
        D[6] = _mm512_madd52lo_epu64(D[6], a.l[3], b.l[3]);
        D[6] = _mm512_madd52lo_epu64(D[6], a.l[4], b.l[2]);
        D[7] = _mm512_madd52hi_epu64(z, a.l[2], b.l[4]);
        D[7] = _mm512_madd52hi_epu64(D[7], a.l[3], b.l[3]);
        D[7] = _mm512_madd52hi_epu64(D[7], a.l[4], b.l[2]);
        D[7] = _mm512_madd52lo_epu64(D[7], a.l[3], b.l[4]);
        D[7] = _mm512_madd52lo_epu64(D[7], a.l[4], b.l[3]);
        D[8] = _mm512_madd52hi_epu64(z, a.l[3], b.l[4]);
        D[8] = _mm512_madd52hi_epu64(D[8], a.l[4], b.l[3]);
        D[8] = _mm512_madd52lo_epu64(D[8], a.l[4], b.l[4]);
        D[9] = _mm512_madd52hi_epu64(z, a.l[4], b.l[4]);
        return fe52_reduce(D);
    }

    // ── square: 15 distinct products instead of 25 ───────────────────────────
    ECLOOP_ALWAYS_INLINE Fe52 fe52_sqr(const Fe52& a) noexcept {
        __m512i D[10];
        const __m512i z = _mm512_setzero_si512();
        // off-diagonal half (i < j), to be doubled
        D[1] = _mm512_madd52lo_epu64(z, a.l[0], a.l[1]);
        D[2] = _mm512_madd52hi_epu64(z, a.l[0], a.l[1]);
        D[2] = _mm512_madd52lo_epu64(D[2], a.l[0], a.l[2]);
        D[3] = _mm512_madd52hi_epu64(z, a.l[0], a.l[2]);
        D[3] = _mm512_madd52lo_epu64(D[3], a.l[0], a.l[3]);
        D[3] = _mm512_madd52lo_epu64(D[3], a.l[1], a.l[2]);
        D[4] = _mm512_madd52hi_epu64(z, a.l[0], a.l[3]);
        D[4] = _mm512_madd52hi_epu64(D[4], a.l[1], a.l[2]);
        D[4] = _mm512_madd52lo_epu64(D[4], a.l[0], a.l[4]);
        D[4] = _mm512_madd52lo_epu64(D[4], a.l[1], a.l[3]);
        D[5] = _mm512_madd52hi_epu64(z, a.l[0], a.l[4]);
        D[5] = _mm512_madd52hi_epu64(D[5], a.l[1], a.l[3]);
        D[5] = _mm512_madd52lo_epu64(D[5], a.l[1], a.l[4]);
        D[5] = _mm512_madd52lo_epu64(D[5], a.l[2], a.l[3]);
        D[6] = _mm512_madd52hi_epu64(z, a.l[1], a.l[4]);
        D[6] = _mm512_madd52hi_epu64(D[6], a.l[2], a.l[3]);
        D[6] = _mm512_madd52lo_epu64(D[6], a.l[2], a.l[4]);
        D[7] = _mm512_madd52hi_epu64(z, a.l[2], a.l[4]);
        D[7] = _mm512_madd52lo_epu64(D[7], a.l[3], a.l[4]);
        D[8] = _mm512_madd52hi_epu64(z, a.l[3], a.l[4]);
        // double: at most 4 terms per column so far (< 2^54), doubled < 2^55
        for (int k = 1; k <= 8; ++k) D[k] = _mm512_add_epi64(D[k], D[k]);
        // diagonal
        D[0] = _mm512_madd52lo_epu64(z, a.l[0], a.l[0]);
        D[1] = _mm512_madd52hi_epu64(D[1], a.l[0], a.l[0]);
        D[2] = _mm512_madd52lo_epu64(D[2], a.l[1], a.l[1]);
        D[3] = _mm512_madd52hi_epu64(D[3], a.l[1], a.l[1]);
        D[4] = _mm512_madd52lo_epu64(D[4], a.l[2], a.l[2]);
        D[5] = _mm512_madd52hi_epu64(D[5], a.l[2], a.l[2]);
        D[6] = _mm512_madd52lo_epu64(D[6], a.l[3], a.l[3]);
        D[7] = _mm512_madd52hi_epu64(D[7], a.l[3], a.l[3]);
        D[8] = _mm512_madd52lo_epu64(D[8], a.l[4], a.l[4]);
        D[9] = _mm512_madd52hi_epu64(z, a.l[4], a.l[4]);
        // every column < 2^55 + 2^52 < 2^56, as the reduction assumes
        return fe52_reduce(D);
    }

    // ── zero test on an N-form value: value is 0 or p ────────────────────────
    ECLOOP_ALWAYS_INLINE __mmask8 fe52_iszero_nform(const Fe52& a) noexcept {
        const __m512i z = _mm512_setzero_si512();
        __m512i o = _mm512_or_si512(_mm512_or_si512(a.l[0], a.l[1]),
                                    _mm512_or_si512(_mm512_or_si512(a.l[2], a.l[3]), a.l[4]));
        __mmask8 mz = _mm512_cmpeq_epu64_mask(o, z);
        __mmask8 mp = _mm512_cmpeq_epu64_mask(a.l[0], _mm512_set1_epi64((long long)fe52c::P0));
        mp = _kand_mask8(mp, _mm512_cmpeq_epu64_mask(a.l[1], fe52_m52()));
        mp = _kand_mask8(mp, _mm512_cmpeq_epu64_mask(a.l[2], fe52_m52()));
        mp = _kand_mask8(mp, _mm512_cmpeq_epu64_mask(a.l[3], fe52_m52()));
        mp = _kand_mask8(mp, _mm512_cmpeq_epu64_mask(a.l[4], fe52_m48()));
        return _kor_mask8(mz, mp);
    }
    ECLOOP_ALWAYS_INLINE __mmask8 fe52_iszero(const Fe52& a) noexcept {
        return fe52_iszero_nform(fe52_normalize(a));
    }

    // ── boundary: N-form -> canonical radix 2^64 rows (4 x zmm) ─────────────
    // One conditional subtraction of p suffices: an N-form value is below 2p.
    ECLOOP_ALWAYS_INLINE void fe52_to64_nform(__m512i out[4], const Fe52& a) noexcept {
        const __m512i m = fe52_m52();
        __m512i s0 = _mm512_sub_epi64(a.l[0], _mm512_set1_epi64((long long)fe52c::P0));
        __m512i b = _mm512_srli_epi64(s0, 63); s0 = _mm512_and_si512(s0, m);
        __m512i s1 = _mm512_sub_epi64(_mm512_sub_epi64(a.l[1], fe52_m52()), b);
        b = _mm512_srli_epi64(s1, 63); s1 = _mm512_and_si512(s1, m);
        __m512i s2 = _mm512_sub_epi64(_mm512_sub_epi64(a.l[2], fe52_m52()), b);
        b = _mm512_srli_epi64(s2, 63); s2 = _mm512_and_si512(s2, m);
        __m512i s3 = _mm512_sub_epi64(_mm512_sub_epi64(a.l[3], fe52_m52()), b);
        b = _mm512_srli_epi64(s3, 63); s3 = _mm512_and_si512(s3, m);
        __m512i s4 = _mm512_sub_epi64(_mm512_sub_epi64(a.l[4], fe52_m48()), b);
        // final borrow set  <=>  a < p  <=>  keep a
        const __mmask8 keep = _mm512_cmplt_epi64_mask(s4, _mm512_setzero_si512());
        s0 = _mm512_mask_blend_epi64(keep, s0, a.l[0]);
        s1 = _mm512_mask_blend_epi64(keep, s1, a.l[1]);
        s2 = _mm512_mask_blend_epi64(keep, s2, a.l[2]);
        s3 = _mm512_mask_blend_epi64(keep, s3, a.l[3]);
        s4 = _mm512_mask_blend_epi64(keep, s4, a.l[4]);
        out[0] = _mm512_or_si512(s0, _mm512_slli_epi64(s1, 52));
        out[1] = _mm512_or_si512(_mm512_srli_epi64(s1, 12), _mm512_slli_epi64(s2, 40));
        out[2] = _mm512_or_si512(_mm512_srli_epi64(s2, 24), _mm512_slli_epi64(s3, 28));
        out[3] = _mm512_or_si512(_mm512_srli_epi64(s3, 36), _mm512_slli_epi64(s4, 16));
    }
    ECLOOP_ALWAYS_INLINE void fe52_to64(__m512i out[4], const Fe52& a) noexcept {
        fe52_to64_nform(out, fe52_normalize(a));
    }
    ECLOOP_ALWAYS_INLINE void fe52_to_fe8(Fe8* r, const Fe52& a) noexcept {
        __m512i o[4];
        fe52_to64(o, a);
        for (int i = 0; i < 4; ++i) _mm512_storeu_si512(r->v[i], o[i]);
    }

    // ── eight independent inversions for the price of one ────────────────────
    // The eight lane values are combined with the scalar Montgomery trick,
    // inverted once (safegcd) and expanded back: 21 scalar muls + 1 inversion.
    // Input and output N-form; a zero lane yields zero, like fe_modp_inv.
    ECLOOP_ALWAYS_INLINE Fe52 fe52_invert_lanes(const Fe52& prod) noexcept {
        Fe8 P8;
        fe52_to_fe8(&P8, prod);
        fe lane[8], pre[8], inv, t, invl[8];
        for (int l = 0; l < 8; ++l) fe8_get_lane(lane[l], &P8, l);
        fe_clone(pre[0], lane[0]);
        for (int l = 1; l < 8; ++l) fe_modp_mul(pre[l], pre[l - 1], lane[l]);
        fe_modp_inv(inv, pre[7]);
        for (int l = 7; l >= 1; --l) {
            fe_modp_mul(invl[l], inv, pre[l - 1]);
            fe_modp_mul(t, inv, lane[l]);
            fe_clone(inv, t);
        }
        fe_clone(invl[0], inv);
        Fe8 I8;
        for (int l = 0; l < 8; ++l) fe8_set_lane(&I8, l, invl[l]);
        return fe52_from_fe8(&I8);
    }

    // ── SoA (4 rows) -> AoS: 8 field elements, each 4 x u64 at `stride` bytes ─
    // Writes lane L's four limbs to (u8*)dst + L*stride. 4 unpacks, 4
    // cross-lane permutes and 8 half-vector stores per 8 elements.
    ECLOOP_ALWAYS_INLINE void soa64_to_aos(void* dst, std::ptrdiff_t stride,
                                           const __m512i r[4]) noexcept {
        const __m512i t0 = _mm512_unpacklo_epi64(r[0], r[1]); // lanes 0,2,4,6 of limbs 0,1
        const __m512i t1 = _mm512_unpackhi_epi64(r[0], r[1]); // lanes 1,3,5,7
        const __m512i t2 = _mm512_unpacklo_epi64(r[2], r[3]);
        const __m512i t3 = _mm512_unpackhi_epi64(r[2], r[3]);
        const __m512i iA = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);   // lanes (0,2) / (1,3)
        const __m512i iB = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4); // lanes (4,6) / (5,7)
        const __m512i u02 = _mm512_permutex2var_epi64(t0, iA, t2);
        const __m512i u13 = _mm512_permutex2var_epi64(t1, iA, t3);
        const __m512i u46 = _mm512_permutex2var_epi64(t0, iB, t2);
        const __m512i u57 = _mm512_permutex2var_epi64(t1, iB, t3);
        u8* d = static_cast<u8*>(dst);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + 0 * stride), _mm512_castsi512_si256(u02));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + 2 * stride), _mm512_extracti64x4_epi64(u02, 1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + 1 * stride), _mm512_castsi512_si256(u13));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + 3 * stride), _mm512_extracti64x4_epi64(u13, 1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + 4 * stride), _mm512_castsi512_si256(u46));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + 6 * stride), _mm512_extracti64x4_epi64(u46, 1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + 5 * stride), _mm512_castsi512_si256(u57));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + 7 * stride), _mm512_extracti64x4_epi64(u57, 1));
    }

    // AoS -> SoA: 8 elements of 4 x u64 at `stride` bytes -> 4 rows.
    ECLOOP_ALWAYS_INLINE void aos_to_soa64(__m512i r[4], const void* src,
                                           std::ptrdiff_t stride) noexcept {
        const u8* s = static_cast<const u8*>(src);
        __m512i c[4];
        for (int k = 0; k < 4; ++k) {
            const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + (2 * k) * stride));
            const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + (2 * k + 1) * stride));
            c[k] = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1); // lanes 2k (low), 2k+1 (high)
        }
        // c[k] = [x_{2k}[0..3], x_{2k+1}[0..3]]
        const __m512i t0 = _mm512_unpacklo_epi64(c[0], c[1]); // x0[0],x2[0],x0[2],x2[2],x1[0],x3[0],x1[2],x3[2]
        const __m512i t1 = _mm512_unpackhi_epi64(c[0], c[1]); // x0[1],x2[1],x0[3],x2[3],x1[1],x3[1],x1[3],x3[3]
        const __m512i t2 = _mm512_unpacklo_epi64(c[2], c[3]); // x4[0],x6[0],x4[2],x6[2],x5[0],x7[0],x5[2],x7[2]
        const __m512i t3 = _mm512_unpackhi_epi64(c[2], c[3]);
        // row0 = x0[0] x1[0] x2[0] x3[0] x4[0] x5[0] x6[0] x7[0]
        const __m512i i0 = _mm512_set_epi64(13, 9, 12, 8, 5, 1, 4, 0);
        const __m512i i2 = _mm512_set_epi64(15, 11, 14, 10, 7, 3, 6, 2);
        r[0] = _mm512_permutex2var_epi64(t0, i0, t2);
        r[2] = _mm512_permutex2var_epi64(t0, i2, t2);
        r[1] = _mm512_permutex2var_epi64(t1, i0, t3);
        r[3] = _mm512_permutex2var_epi64(t1, i2, t3);
    }

} // namespace ecloop

#pragma GCC pop_options
#endif // ECLOOP_HAVE_FE52
