// Four-lane AVX2 field layer for secp256k1: the EC rung between the IFMA
// batch-8 layer (fe52.hpp) and the scalar s2n-bignum code, for CPUs with
// AVX2 but no AVX-512 (Haswell..Comet Lake, Alder/Raptor Lake, Zen 1-3).
//
// AVX2 has no 52x52 multiply, only vpmuludq: 32x32 -> 64 in each 64-bit
// lane. So the field element is 9 limbs of 29 bits (weights 2^(29 i),
// i = 0..8; a canonical value has limb 8 below 2^24), a product is 81
// partial products of at most 58 bits, and a column of up to 9 of them
// stays below 2^62 in a 64-bit lane. Reduction uses 2^261 == 2^37 + 31264
// (mod p): the high columns are carried into 29-bit pieces, multiplied by
// 31264 into column j and by 256 into column j+1, and the result is carried
// once more; the multiplies never exceed 48 bits.
//
// Conventions (mirroring fe52.hpp):
//   N-form   every limb below 2^29 (+ a few bits): what mul/sqr produce and
//            what they accept as input. Value may exceed p.
//   lazy     add/sub/neg outputs: limbs below 2^31. Run fe29_normalize
//            before feeding them to mul/sqr; the column bound needs it.
//   sub      a + 64p - b, no borrows as long as b's limbs stay below
//            2^30 - 62528 (an N-form or a sum of two); fe29_sub2 uses
//            128p for b up to 2^31 - 125056 (a sum of four).
// Lanes are independent; fe29_to64 makes each canonical and packs it to
// four 64-bit words.

#pragma once

#include "types.hpp"
#include "compat.hpp"

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define ECLOOP_HAVE_FE29 1
#else
#define ECLOOP_HAVE_FE29 0
#endif

#if ECLOOP_HAVE_FE29


#include <immintrin.h>
#include <cstring>

#include "ecc.hpp"

#pragma GCC push_options
#pragma GCC target("avx2")
namespace ecloop {

    struct Fe29 {
        __m256i l[9];
    };
    struct alignas(32) Fe29s {   // storable
        u64 l[9][4];
    };

    namespace fe29_detail {
        constexpr u64 M29 = (1ULL << 29) - 1;
        constexpr u64 FOLD_LO = 31264;   // 2^261 mod p = 2^37 + 31264 = 256 * 2^29 + 31264
        constexpr u64 FOLD_HI = 256;
        // 64p in radix 2^29, all limbs about 2^30: the borrow-free zero for sub
        constexpr u64 P64_L0 = (1ULL << 30) - 62528, P64_L1 = (1ULL << 30) - 514, P64_L = (1ULL << 30) - 2;

        ECLOOP_ALWAYS_INLINE __m256i set1(u64 v) noexcept { return _mm256_set1_epi64x(static_cast<long long>(v)); }
        ECLOOP_ALWAYS_INLINE __m256i mask29() noexcept { return set1(M29); }
        ECLOOP_ALWAYS_INLINE __m256i mul32(__m256i a, __m256i b) noexcept { return _mm256_mul_epu32(a, b); }
        ECLOOP_ALWAYS_INLINE __m256i mulc(__m256i a, u64 c) noexcept { return _mm256_mul_epu32(a, set1(c)); }
        // vpmuludq reads only the low 32 bits of a lane: for a multiplicand
        // that may reach 2^43 (the carries of the reduction), both halves.
        ECLOOP_ALWAYS_INLINE __m256i mulc_wide(__m256i a, u64 c) noexcept {
            const __m256i lo = _mm256_mul_epu32(a, set1(c));
            const __m256i hi = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), set1(c));
            return _mm256_add_epi64(lo, _mm256_slli_epi64(hi, 32));
        }
    } // namespace fe29_detail

    // ── load / store ────────────────────────────────────────────────────
    ECLOOP_ALWAYS_INLINE Fe29 fe29_load(const Fe29s* s) noexcept {
        Fe29 r;
        for (int i = 0; i < 9; ++i) r.l[i] = _mm256_load_si256(reinterpret_cast<const __m256i*>(s->l[i]));
        return r;
    }
    ECLOOP_ALWAYS_INLINE void fe29_store(Fe29s* d, const Fe29& a) noexcept {
        for (int i = 0; i < 9; ++i) _mm256_store_si256(reinterpret_cast<__m256i*>(d->l[i]), a.l[i]);
    }

    ECLOOP_ALWAYS_INLINE Fe29 fe29_zero() noexcept {
        Fe29 r;
        for (int i = 0; i < 9; ++i) r.l[i] = _mm256_setzero_si256();
        return r;
    }
    ECLOOP_ALWAYS_INLINE Fe29 fe29_one() noexcept {
        Fe29 r = fe29_zero();
        r.l[0] = fe29_detail::set1(1);
        return r;
    }

    // Four radix-2^64 values (w[k] = word k of lanes 0..3) -> 29-bit limbs.
    ECLOOP_ALWAYS_INLINE Fe29 fe29_from64(const __m256i w[4]) noexcept {
        using namespace fe29_detail;
        const __m256i M = mask29();
        Fe29 r;
        r.l[0] = _mm256_and_si256(w[0], M);
        r.l[1] = _mm256_and_si256(_mm256_srli_epi64(w[0], 29), M);
        r.l[2] = _mm256_and_si256(_mm256_or_si256(_mm256_srli_epi64(w[0], 58), _mm256_slli_epi64(w[1], 6)), M);
        r.l[3] = _mm256_and_si256(_mm256_srli_epi64(w[1], 23), M);
        r.l[4] = _mm256_and_si256(_mm256_or_si256(_mm256_srli_epi64(w[1], 52), _mm256_slli_epi64(w[2], 12)), M);
        r.l[5] = _mm256_and_si256(_mm256_srli_epi64(w[2], 17), M);
        r.l[6] = _mm256_and_si256(_mm256_or_si256(_mm256_srli_epi64(w[2], 46), _mm256_slli_epi64(w[3], 18)), M);
        r.l[7] = _mm256_and_si256(_mm256_srli_epi64(w[3], 11), M);
        r.l[8] = _mm256_srli_epi64(w[3], 40);
        return r;
    }
    // One scalar value in all four lanes.
    inline Fe29 fe29_bcast(const fe x) noexcept {
        __m256i w[4];
        for (int k = 0; k < 4; ++k) w[k] = fe29_detail::set1(x[k]);
        return fe29_from64(w);
    }
    // Four scalar values, one per lane.
    inline Fe29 fe29_from_lanes(const fe x0, const fe x1, const fe x2, const fe x3) noexcept {
        __m256i w[4];
        for (int k = 0; k < 4; ++k)
            w[k] = _mm256_set_epi64x(static_cast<long long>(x3[k]), static_cast<long long>(x2[k]),
                                     static_cast<long long>(x1[k]), static_cast<long long>(x0[k]));
        return fe29_from64(w);
    }

    // ── lazy add / sub / neg ────────────────────────────────────────────
    ECLOOP_ALWAYS_INLINE Fe29 fe29_add(const Fe29& a, const Fe29& b) noexcept {
        Fe29 r;
        for (int i = 0; i < 9; ++i) r.l[i] = _mm256_add_epi64(a.l[i], b.l[i]);
        return r;
    }
    ECLOOP_ALWAYS_INLINE Fe29 fe29_sub(const Fe29& a, const Fe29& b) noexcept {   // a + 64p - b
        using namespace fe29_detail;
        Fe29 r;
        r.l[0] = _mm256_sub_epi64(_mm256_add_epi64(a.l[0], set1(P64_L0)), b.l[0]);
        r.l[1] = _mm256_sub_epi64(_mm256_add_epi64(a.l[1], set1(P64_L1)), b.l[1]);
        const __m256i pl = set1(P64_L);
        for (int i = 2; i < 9; ++i) r.l[i] = _mm256_sub_epi64(_mm256_add_epi64(a.l[i], pl), b.l[i]);
        return r;
    }
    ECLOOP_ALWAYS_INLINE Fe29 fe29_sub2(const Fe29& a, const Fe29& b) noexcept {  // a + 128p - b
        using namespace fe29_detail;
        Fe29 r;
        r.l[0] = _mm256_sub_epi64(_mm256_add_epi64(a.l[0], set1(2 * P64_L0)), b.l[0]);
        r.l[1] = _mm256_sub_epi64(_mm256_add_epi64(a.l[1], set1(2 * P64_L1)), b.l[1]);
        const __m256i pl = set1(2 * P64_L);
        for (int i = 2; i < 9; ++i) r.l[i] = _mm256_sub_epi64(_mm256_add_epi64(a.l[i], pl), b.l[i]);
        return r;
    }
    ECLOOP_ALWAYS_INLINE Fe29 fe29_neg(const Fe29& a) noexcept { return fe29_sub(fe29_zero(), a); }

    // ── normalisation: lazy (limbs < 2^32) -> N-form ────────────────────
    // Carry chain over the limbs; the carry out of limb 8 (weight 2^261) is
    // folded as 2^261 == 256 * 2^29 + 31264 and carried through limbs 0..2.
    ECLOOP_ALWAYS_INLINE Fe29 fe29_normalize(const Fe29& a) noexcept {
        using namespace fe29_detail;
        const __m256i M = mask29();
        Fe29 r;
        __m256i c = _mm256_setzero_si256();
        for (int i = 0; i < 9; ++i) {
            const __m256i t = _mm256_add_epi64(a.l[i], c);
            r.l[i] = _mm256_and_si256(t, M);
            c = _mm256_srli_epi64(t, 29);
        }
        // c < 2^(32-29+1) = tiny
        r.l[0] = _mm256_add_epi64(r.l[0], mulc(c, FOLD_LO));
        r.l[1] = _mm256_add_epi64(r.l[1], _mm256_slli_epi64(c, 8));
        for (int i = 0; i < 3; ++i) {
            const __m256i t = r.l[i];
            r.l[i] = _mm256_and_si256(t, M);
            r.l[i + 1] = _mm256_add_epi64(r.l[i + 1], _mm256_srli_epi64(t, 29));
        }
        return r;   // limbs 0..2 < 2^29, 3 < 2^29 + 1, the rest < 2^29
    }

    // ── multiply / square: N-form x N-form -> N-form ────────────────────
    namespace fe29_detail {
        // Reduce the 17 product columns (each < 2^62) to N-form.
        ECLOOP_ALWAYS_INLINE Fe29 reduce_columns(__m256i c[17]) noexcept {
            const __m256i M = mask29();
            // high columns 9..16 -> 29-bit pieces h[0..7] and the carry h8
            __m256i h[9];
            __m256i carry = _mm256_setzero_si256();
            for (int k = 9; k < 17; ++k) {
                const __m256i t = _mm256_add_epi64(c[k], carry);
                h[k - 9] = _mm256_and_si256(t, M);
                carry = _mm256_srli_epi64(t, 29);
            }
            h[8] = carry;   // < 2^34
            // fold: h_j * 2^(29 j) * 2^261 == h_j * (31264 * 2^(29 j) + 256 * 2^(29 (j+1)))
            c[9] = _mm256_setzero_si256();
            for (int j = 0; j < 8; ++j) {
                c[j] = _mm256_add_epi64(c[j], mulc(h[j], FOLD_LO));
                c[j + 1] = _mm256_add_epi64(c[j + 1], _mm256_slli_epi64(h[j], 8));
            }
            c[8] = _mm256_add_epi64(c[8], mulc_wide(h[8], FOLD_LO));   // h[8] may exceed 2^32
            c[9] = _mm256_add_epi64(c[9], _mm256_slli_epi64(h[8], 8));
            // carry columns 0..8 into limbs; c[9] collects the overflow
            Fe29 r;
            carry = _mm256_setzero_si256();
            for (int i = 0; i < 9; ++i) {
                const __m256i t = _mm256_add_epi64(c[i], carry);
                r.l[i] = _mm256_and_si256(t, M);
                carry = _mm256_srli_epi64(t, 29);
            }
            const __m256i c9 = _mm256_add_epi64(c[9], carry);   // < 2^43
            // fold c9 * 2^261 and carry through the low limbs
            r.l[0] = _mm256_add_epi64(r.l[0], mulc_wide(c9, FOLD_LO)); // + < 2^58
            r.l[1] = _mm256_add_epi64(r.l[1], _mm256_slli_epi64(c9, 8)); // + < 2^51
            carry = _mm256_setzero_si256();
            for (int i = 0; i < 9; ++i) {
                const __m256i t = _mm256_add_epi64(r.l[i], carry);
                r.l[i] = _mm256_and_si256(t, M);
                carry = _mm256_srli_epi64(t, 29);
            }
            // carry here is 0 or 1 (value was below 2^261 + 2^80): fold it
            // and let it ripple through the first limbs; a limb may end at
            // 2^29 exactly, which mul's 32-bit inputs and column bound absorb.
            r.l[0] = _mm256_add_epi64(r.l[0], mulc(carry, FOLD_LO));
            r.l[1] = _mm256_add_epi64(r.l[1], _mm256_slli_epi64(carry, 8));
            for (int i = 0; i < 2; ++i) {
                const __m256i t = r.l[i];
                r.l[i] = _mm256_and_si256(t, M);
                r.l[i + 1] = _mm256_add_epi64(r.l[i + 1], _mm256_srli_epi64(t, 29));
            }
            return r;
        }
    } // namespace fe29_detail

    // Row order (i outer, columns accumulated). The column-order variant
    // (one live accumulator, `b` read from memory by vpmuludq) measured
    // slower here; the 17 accumulators spill about one move per multiply
    // with 16 ymm registers, which is the price of 9 limbs on AVX2.
    ECLOOP_ALWAYS_INLINE Fe29 fe29_mul(const Fe29& a, const Fe29& b) noexcept {
        using namespace fe29_detail;
        __m256i c[17];
        for (int k = 0; k < 17; ++k) c[k] = _mm256_setzero_si256();
        for (int i = 0; i < 9; ++i)
            for (int j = 0; j < 9; ++j)
                c[i + j] = _mm256_add_epi64(c[i + j], mul32(a.l[i], b.l[j]));
        return reduce_columns(c);
    }

    ECLOOP_ALWAYS_INLINE Fe29 fe29_sqr(const Fe29& a) noexcept {
        using namespace fe29_detail;
        __m256i c[17];
        for (int k = 0; k < 17; ++k) c[k] = _mm256_setzero_si256();
        __m256i d[9];   // 2 * a_j, still below 2^32
        for (int j = 0; j < 9; ++j) d[j] = _mm256_add_epi64(a.l[j], a.l[j]);
        for (int i = 0; i < 9; ++i) {
            c[2 * i] = _mm256_add_epi64(c[2 * i], mul32(a.l[i], a.l[i]));
            for (int j = i + 1; j < 9; ++j)
                c[i + j] = _mm256_add_epi64(c[i + j], mul32(a.l[i], d[j]));
        }
        return reduce_columns(c);
    }

    // ── canonical form and packing ──────────────────────────────────────
    // N-form -> canonical [0, p), still 29-bit limbs. Two folds of the bits
    // above 2^256 (value < 2^261 -> < 2^256 + small -> < 2^256 + tiny), then
    // one conditional subtraction of p.
    ECLOOP_ALWAYS_INLINE Fe29 fe29_canon(const Fe29& a_in) noexcept {
        using namespace fe29_detail;
        const __m256i M = mask29();
        const __m256i M24 = set1((1ULL << 24) - 1);
        Fe29 a = fe29_normalize(a_in);
        for (int pass = 0; pass < 2; ++pass) {
            // e = bits 24.. of limb 8 (weight 2^256): e * 2^256 == e * 2^32 + e * 977
            const __m256i e = _mm256_srli_epi64(a.l[8], 24);
            a.l[8] = _mm256_and_si256(a.l[8], M24);
            a.l[0] = _mm256_add_epi64(a.l[0], mulc(e, 977));
            a.l[1] = _mm256_add_epi64(a.l[1], _mm256_slli_epi64(e, 3));   // 2^32 = 8 * 2^29
            __m256i c = _mm256_setzero_si256();
            for (int i = 0; i < 9; ++i) {
                const __m256i t = _mm256_add_epi64(a.l[i], c);
                a.l[i] = _mm256_and_si256(t, M);
                c = _mm256_srli_epi64(t, 29);
            }
            // c is folded into limb 8's excess on the next pass; after the
            // second pass the value is below 2^256 + 2^33 and c is 0.
            a.l[8] = _mm256_add_epi64(a.l[8], _mm256_slli_epi64(c, 29));
        }
        // conditional subtraction of p = 2^256 - 2^32 - 977: in limbs
        // p = [2^29 - 977 - 2^32 mod..] -- build p as 29-bit limbs
        // p = 2^256 - 2^32 - 977: 2^256 = 2^24 * 2^(29*8), so
        //   limb8 = 2^24 - 1, limbs 2..7 = M29, limb1 = M29 - 8 (2^32 = 8*2^29), limb0 = 2^29 - 977
        const u64 pl[9] = {(1ULL << 29) - 977, M29 - 8, M29, M29, M29, M29, M29, M29, (1ULL << 24) - 1};
        // a >= p ?  compare from the top limb: lexicographic on 29-bit limbs
        __m256i ge = _mm256_set1_epi64x(-1);   // running "equal so far" -> becomes ge
        __m256i gt = _mm256_setzero_si256();
        for (int i = 8; i >= 0; --i) {
            const __m256i pv = set1(pl[i]);
            // signed compares are fine: values < 2^32
            const __m256i g = _mm256_cmpgt_epi64(a.l[i], pv);
            const __m256i e = _mm256_cmpeq_epi64(a.l[i], pv);
            gt = _mm256_or_si256(gt, _mm256_and_si256(ge, g));
            ge = _mm256_and_si256(ge, e);
        }
        const __m256i sub = _mm256_or_si256(gt, ge);   // a > p or a == p
        // a - p with borrows, masked
        __m256i borrow = _mm256_setzero_si256();
        for (int i = 0; i < 9; ++i) {
            const __m256i pv = _mm256_and_si256(set1(pl[i]), sub);
            const __m256i t = _mm256_sub_epi64(_mm256_sub_epi64(a.l[i], pv), borrow);
            a.l[i] = _mm256_and_si256(t, M);
            borrow = _mm256_srli_epi64(t, 63);   // wrapped -> borrow 1
        }
        return a;
    }

    // Canonical -> four 64-bit words per lane (w[k] = word k of lanes 0..3).
    ECLOOP_ALWAYS_INLINE void fe29_pack64(__m256i w[4], const Fe29& a) noexcept {
        w[0] = _mm256_or_si256(_mm256_or_si256(a.l[0], _mm256_slli_epi64(a.l[1], 29)), _mm256_slli_epi64(a.l[2], 58));
        w[1] = _mm256_or_si256(_mm256_or_si256(_mm256_srli_epi64(a.l[2], 6), _mm256_slli_epi64(a.l[3], 23)),
                               _mm256_slli_epi64(a.l[4], 52));
        w[2] = _mm256_or_si256(_mm256_or_si256(_mm256_srli_epi64(a.l[4], 12), _mm256_slli_epi64(a.l[5], 17)),
                               _mm256_slli_epi64(a.l[6], 46));
        w[3] = _mm256_or_si256(_mm256_or_si256(_mm256_srli_epi64(a.l[6], 18), _mm256_slli_epi64(a.l[7], 11)),
                               _mm256_slli_epi64(a.l[8], 40));
    }
    ECLOOP_ALWAYS_INLINE void fe29_to64(__m256i w[4], const Fe29& a) noexcept {
        fe29_pack64(w, fe29_canon(a));
    }
    // Lane l of packed words -> fe.
    ECLOOP_ALWAYS_INLINE void fe29_lane(fe out, const __m256i w[4], int l) noexcept {
        alignas(32) u64 t[4][4];
        for (int k = 0; k < 4; ++k) _mm256_store_si256(reinterpret_cast<__m256i*>(t[k]), w[k]);
        for (int k = 0; k < 4; ++k) out[k] = t[k][l];
    }

    // Zero test per lane on a canonical value: bit l of the result.
    ECLOOP_ALWAYS_INLINE unsigned fe29_iszero_canon(const Fe29& a) noexcept {
        __m256i acc = a.l[0];
        for (int i = 1; i < 9; ++i) acc = _mm256_or_si256(acc, a.l[i]);
        const __m256i z = _mm256_cmpeq_epi64(acc, _mm256_setzero_si256());
        return static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(z)));
    }

    // Per-lane blend: lanes with bit set take b.
    ECLOOP_ALWAYS_INLINE Fe29 fe29_blend(const Fe29& a, const Fe29& b, unsigned lanes) noexcept {
        const __m256i m = _mm256_set_epi64x((lanes & 8) ? -1 : 0, (lanes & 4) ? -1 : 0,
                                            (lanes & 2) ? -1 : 0, (lanes & 1) ? -1 : 0);
        Fe29 r;
        for (int i = 0; i < 9; ++i) r.l[i] = _mm256_blendv_epi8(a.l[i], b.l[i], m);
        return r;
    }

    // Inverse of every lane: the four lanes are combined into one scalar
    // inversion (Montgomery), scalar s2n-bignum for the multiplies.
    inline Fe29 fe29_invert_lanes(const Fe29& a) noexcept {
        __m256i w[4];
        fe29_to64(w, a);
        fe v[4], inv[4];
        for (int l = 0; l < 4; ++l) fe29_lane(v[l], w, l);
        fe p01, p23, all, i01, i23;
        fe_modp_mul(p01, v[0], v[1]);
        fe_modp_mul(p23, v[2], v[3]);
        fe_modp_mul(all, p01, p23);
        fe_modp_inv(all, all);
        fe_modp_mul(i01, all, p23);
        fe_modp_mul(i23, all, p01);
        fe_modp_mul(inv[0], i01, v[1]);
        fe_modp_mul(inv[1], i01, v[0]);
        fe_modp_mul(inv[2], i23, v[3]);
        fe_modp_mul(inv[3], i23, v[2]);
        return fe29_from_lanes(inv[0], inv[1], inv[2], inv[3]);
    }

    // Four affine points' x (or y) from an array of pe at `stride` bytes ->
    // one Fe29, and back (writes 4 pe coordinates).
    inline Fe29 fe29_gather(const u64* first, std::ptrdiff_t stride) noexcept {
        fe v[4];
        for (int l = 0; l < 4; ++l)
            std::memcpy(v[l], reinterpret_cast<const u8*>(first) + l * stride, 32);
        return fe29_from_lanes(v[0], v[1], v[2], v[3]);
    }
    inline void fe29_scatter(u64* first, std::ptrdiff_t stride, const __m256i w[4]) noexcept {
        alignas(32) u64 t[4][4];
        for (int k = 0; k < 4; ++k) _mm256_store_si256(reinterpret_cast<__m256i*>(t[k]), w[k]);
        for (int l = 0; l < 4; ++l) {
            u64* d = reinterpret_cast<u64*>(reinterpret_cast<u8*>(first) + l * stride);
            for (int k = 0; k < 4; ++k) d[k] = t[k][l];
        }
    }

} // namespace ecloop

#pragma GCC pop_options
#endif // ECLOOP_HAVE_FE29
