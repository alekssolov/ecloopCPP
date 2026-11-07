// Batch-8 SoA field/point layer. Port of lib/ec_batch8.c, with the
// intrinsics kernels from lib/mul_x8_v2.c and lib/fe8_ops_v2.c folded in
// as internal-linkage helpers (in the C tree they were #included).
//
// Build switches, all defaulting to the C original's defaults:
//
//   EC8_MULV2=1   fully vector reduction instead of the scalar phase 5 in
//                 the .S multiply. Longer critical path, higher
//                 throughput, so it wins only with enough ILP.
//   EC8_SQRV2     dedicated batch squaring (defaults to EC8_MULV2). Four
//                 of the eleven multiplies in a mixed addition are
//                 squarings, so it is worth having.
//   EC8_FE8V2=1   add/sub/zero_mask as intrinsics instead of calls into
//                 .S, which lets the compiler keep values in registers
//                 between neighbouring field operations.
//   EC8_INVTREE=1 tree-shaped batch inversion instead of a chain. Needs a
//                 power-of-two n and 4x the scratch -- always size the
//                 buffer with ec_grprdc_x8_scratch().

#include "ec_batch8.hpp"

#include "compat.hpp"
#include "fe52.hpp"

#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef EC8_MULV2
#define EC8_MULV2 0
#endif
#ifndef EC8_FE8V2
#define EC8_FE8V2 0
#endif
#ifndef EC8_INVTREE
#define EC8_INVTREE 0
#endif
#ifndef EC8_SQRV2
#define EC8_SQRV2 EC8_MULV2
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define ECLOOP_HAVE_BATCH8 1
#include <immintrin.h>
#else
#define ECLOOP_HAVE_BATCH8 0
#endif

// The v2 kernels are written in AVX512-IFMA intrinsics, so they can only be
// COMPILED where the target has them -- otherwise every _mm512_madd52lo_epu64
// is a hard error, not a warning. EC8_MULV2/EC8_FE8V2 are therefore requests,
// not commands: they are honoured when the target supports the instructions
// and quietly dropped when it does not. That way EC8=1 can be the default
// without breaking the build on a machine without IFMA, where the .S kernels
// (and, failing those, the scalar fallback) take over anyway.
#if !defined(__x86_64__)
#if EC8_MULV2 || EC8_FE8V2
#undef EC8_MULV2
#undef EC8_FE8V2
#define EC8_MULV2 0
#define EC8_FE8V2 0
#endif
#endif

// s2n-bignum batch primitives. The single-point ones are already declared
// in ecc.cpp; repeating them here keeps this file self-contained.
#include "s2n_dispatch.hpp"

namespace ecloop {
    namespace {

#if ECLOOP_HAVE_BATCH8
// Compiled for AVX-512 IFMA whatever -march says; reached only when
// ec_batch8_available() (cpuid) said yes.
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl,avx512ifma,bmi2")

// ── from lib/mul_x8_v2.c ────────────────────────────────────────────
#if EC8_MULV2
        #define MULV2_M52 0x000FFFFFFFFFFFFFull
        // K = 16 * (2^32 + 977)
        #define MULV2_K   68719492368ull
        // C = 2^32 + 977
        #define MULV2_C   4294968273ull

        // 4 rows of radix-2^64 (SoA) -> 5 rows of radix-2^52
        inline void mulv2_to52(__m512i *a, const u64 x[4][8]) {
          const __m512i m52 = _mm512_set1_epi64(MULV2_M52);
          __m512i x0 = _mm512_loadu_si512(&x[0][0]);
          __m512i x1 = _mm512_loadu_si512(&x[1][0]);
          __m512i x2 = _mm512_loadu_si512(&x[2][0]);
          __m512i x3 = _mm512_loadu_si512(&x[3][0]);

          a[0] = _mm512_and_si512(x0, m52);
          a[1] = _mm512_and_si512(
              _mm512_or_si512(_mm512_srli_epi64(x0, 52), _mm512_slli_epi64(x1, 12)), m52);
          a[2] = _mm512_and_si512(
              _mm512_or_si512(_mm512_srli_epi64(x1, 40), _mm512_slli_epi64(x2, 24)), m52);
          a[3] = _mm512_and_si512(
              _mm512_or_si512(_mm512_srli_epi64(x2, 28), _mm512_slli_epi64(x3, 36)), m52);
          a[4] = _mm512_srli_epi64(x3, 16);
        }

        // Shared tail: take ten radix-2^52 digits D[0..9], reduce mod p and store
        // the result as four radix-2^64 SoA rows. Factored out because the
        // multiply and the squaring differ only in how D[] is produced -- the
        // reduction is bit-for-bit the same, and having one copy means one place
        // to get right.
        inline void
        mulv2_reduce_store(u64 z[4][8], __m512i D[10]) {
          const __m512i m52 = _mm512_set1_epi64(MULV2_M52);
          const __m512i zero = _mm512_setzero_si512();
          const __m512i vK = _mm512_set1_epi64((long long)MULV2_K);

          // ---- carry propagation to canonical form
          for (int k = 0; k < 9; ++k) {
            __m512i c = _mm512_srli_epi64(D[k], 52);
            D[k] = _mm512_and_si512(D[k], m52);
            D[k + 1] = _mm512_add_epi64(D[k + 1], c);
          }

          // ---- reduction step 1: V = Lo + 2^260*Hh, and 2^260 == K (mod p)
          __m512i E[6];
          for (int j = 0; j < 5; ++j) E[j] = D[j];
          E[5] = zero;
          for (int j = 0; j < 5; ++j) {
            E[j] = _mm512_madd52lo_epu64(E[j], D[j + 5], vK);
            E[j + 1] = _mm512_madd52hi_epu64(E[j + 1], D[j + 5], vK);
          }
          for (int j = 0; j < 5; ++j) {
            __m512i c = _mm512_srli_epi64(E[j], 52);
            E[j] = _mm512_and_si512(E[j], m52);
            E[j + 1] = _mm512_add_epi64(E[j + 1], c);
          }

          // ---- reduction step 2: digit E5 carries the same 2^260 weight
          E[0] = _mm512_madd52lo_epu64(E[0], E[5], vK);
          E[1] = _mm512_madd52hi_epu64(E[1], E[5], vK);
          for (int j = 0; j < 4; ++j) {
            __m512i c = _mm512_srli_epi64(E[j], 52);
            E[j] = _mm512_and_si512(E[j], m52);
            E[j + 1] = _mm512_add_epi64(E[j + 1], c);
          }

          // ---- back to radix-2^64; t is whatever spilled above 2^256
          __m512i L0 = _mm512_or_si512(E[0], _mm512_slli_epi64(E[1], 52));
          __m512i L1 = _mm512_or_si512(_mm512_srli_epi64(E[1], 12), _mm512_slli_epi64(E[2], 40));
          __m512i L2 = _mm512_or_si512(_mm512_srli_epi64(E[2], 24), _mm512_slli_epi64(E[3], 28));
          __m512i L3 = _mm512_or_si512(_mm512_srli_epi64(E[3], 36), _mm512_slli_epi64(E[4], 16));
          __m512i t = _mm512_srli_epi64(E[4], 48);

          // ---- V == L + t*C (mod p); t < 2^5, t*C < 2^38
          __m512i tc = _mm512_mullo_epi64(t, _mm512_set1_epi64((long long)MULV2_C));
          __m512i S0 = _mm512_add_epi64(L0, tc);
          __mmask8 cy = _mm512_cmplt_epu64_mask(S0, L0);
          __m512i S1 = _mm512_mask_add_epi64(L1, cy, L1, _mm512_set1_epi64(1));
          __mmask8 c1 = _kand_mask8(cy, _mm512_cmpeq_epu64_mask(S1, zero));
          __m512i S2 = _mm512_mask_add_epi64(L2, c1, L2, _mm512_set1_epi64(1));
          __mmask8 c2 = _kand_mask8(c1, _mm512_cmpeq_epu64_mask(S2, zero));
          __m512i S3 = _mm512_mask_add_epi64(L3, c2, L3, _mm512_set1_epi64(1));
          __mmask8 c3 = _kand_mask8(c2, _mm512_cmpeq_epu64_mask(S3, zero));

          // ---- one conditional correction, same logic as the SoA add.
          // Take S+C if the addition above overflowed 2^256 (c3) OR if S >= p,
          // which is exactly what the carry out of S+C detects.
          const __m512i vC = _mm512_set1_epi64((long long)MULV2_C);
          const __m512i one = _mm512_set1_epi64(1);
          __m512i T0 = _mm512_add_epi64(S0, vC);
          __mmask8 d0 = _mm512_cmplt_epu64_mask(T0, S0);
          __m512i T1 = _mm512_mask_add_epi64(S1, d0, S1, one);
          __mmask8 d1 = _kand_mask8(d0, _mm512_cmpeq_epu64_mask(T1, zero));
          __m512i T2 = _mm512_mask_add_epi64(S2, d1, S2, one);
          __mmask8 d2 = _kand_mask8(d1, _mm512_cmpeq_epu64_mask(T2, zero));
          __m512i T3 = _mm512_mask_add_epi64(S3, d2, S3, one);
          __mmask8 d3 = _kand_mask8(d2, _mm512_cmpeq_epu64_mask(T3, zero));

          __mmask8 sel = _kor_mask8(c3, d3);
          _mm512_storeu_si512(&z[0][0], _mm512_mask_blend_epi64(sel, S0, T0));
          _mm512_storeu_si512(&z[1][0], _mm512_mask_blend_epi64(sel, S1, T1));
          _mm512_storeu_si512(&z[2][0], _mm512_mask_blend_epi64(sel, S2, T2));
          _mm512_storeu_si512(&z[3][0], _mm512_mask_blend_epi64(sel, S3, T3));
        }

        inline void
        bignum_mul_p256k1_x8_soa_v2(u64 z[4][8], const u64 x[4][8],
                                    const u64 y[4][8]) {
          const __m512i zero = _mm512_setzero_si512();
          __m512i A[5], B[5], D[10];
          mulv2_to52(A, x);
          mulv2_to52(B, y);

          // ---- schoolbook 5x5 multiply in radix-2^52, 25 MTERMs = 50 IFMA ops.
          // vpmadd52 accumulates into a 64-bit lane; each D_k collects at most 10
          // terms below 2^52, so it stays under 2^56 and cannot overflow.
          for (int k = 0; k < 10; ++k) D[k] = zero;
          for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
              D[i + j] = _mm512_madd52lo_epu64(D[i + j], A[i], B[j]);
              D[i + j + 1] = _mm512_madd52hi_epu64(D[i + j + 1], A[i], B[j]);
            }
          }

          mulv2_reduce_store(z, D);
        }

        // ---------------------------------------------------------------------------
        // Dedicated squaring. A square is symmetric, so of the 25 partial products
        // only 15 are distinct: 5 on the diagonal and 10 off it, each of the
        // latter appearing twice. Computing the off-diagonal half once and
        // doubling it takes 15 MTERMs = 30 IFMA ops instead of 50, i.e. 40% fewer.
        //
        // Order matters: the off-diagonal terms must be doubled BEFORE the
        // diagonal ones are added, otherwise the diagonal gets doubled too.
        //
        // Overflow check for the accumulator, which is what makes the doubling
        // safe. Off-diagonal pairs (i<j) put their lo at position i+j and their hi
        // at i+j+1, so a position receives at most 4 terms (position 4 gets lo
        // from (0,4) and (1,3), hi from (0,3) and (1,2); position 5 likewise).
        // Four terms below 2^52 give under 2^54, doubling gives under 2^55, and
        // the two diagonal terms add less than 2^53 more -- still well inside 64
        // bits, so no intermediate can wrap.
        // ---------------------------------------------------------------------------
        inline void
        bignum_sqr_p256k1_x8_soa_v2(u64 z[4][8], const u64 x[4][8]) {
          const __m512i zero = _mm512_setzero_si512();
          __m512i A[5], D[10];
          mulv2_to52(A, x);

          // ---- off-diagonal half: 10 MTERMs = 20 IFMA ops
          for (int k = 0; k < 10; ++k) D[k] = zero;
          for (int i = 0; i < 5; ++i) {
            for (int j = i + 1; j < 5; ++j) {
              D[i + j] = _mm512_madd52lo_epu64(D[i + j], A[i], A[j]);
              D[i + j + 1] = _mm512_madd52hi_epu64(D[i + j + 1], A[i], A[j]);
            }
          }

          // ---- double it: every off-diagonal product appears twice in a square
          for (int k = 0; k < 10; ++k) D[k] = _mm512_slli_epi64(D[k], 1);

          // ---- diagonal: 5 MTERMs = 10 IFMA ops
          for (int i = 0; i < 5; ++i) {
            D[2 * i] = _mm512_madd52lo_epu64(D[2 * i], A[i], A[i]);
            D[2 * i + 1] = _mm512_madd52hi_epu64(D[2 * i + 1], A[i], A[i]);
          }

          mulv2_reduce_store(z, D);
        }

#endif // EC8_MULV2

// ── from lib/fe8_ops_v2.c ───────────────────────────────────────────
#if EC8_FE8V2
        #define FE8V2_C 4294968273ull // 2^32 + 977 = 2^256 - p

        // Mask of lanes whose value is zero. Five vector instructions instead of
        // a scalar loop.
        inline u32 fe8_zero_mask_v2(const Fe8 *a) {
          __m512i o = _mm512_or_si512(
              _mm512_or_si512(_mm512_loadu_si512(a->v[0]), _mm512_loadu_si512(a->v[1])),
              _mm512_or_si512(_mm512_loadu_si512(a->v[2]), _mm512_loadu_si512(a->v[3])));
          return (u32)_mm512_testn_epi64_mask(o, o);
        }

        inline void fe8_add_v2(Fe8 *r, const Fe8 *a, const Fe8 *b) {
          const __m512i one = _mm512_set1_epi64(1);
          const __m512i vC = _mm512_set1_epi64((long long)FE8V2_C);
          __m512i a0 = _mm512_loadu_si512(a->v[0]), a1 = _mm512_loadu_si512(a->v[1]);
          __m512i a2 = _mm512_loadu_si512(a->v[2]), a3 = _mm512_loadu_si512(a->v[3]);
          __m512i b0 = _mm512_loadu_si512(b->v[0]), b1 = _mm512_loadu_si512(b->v[1]);
          __m512i b2 = _mm512_loadu_si512(b->v[2]), b3 = _mm512_loadu_si512(b->v[3]);

          // raw addition with carry propagation
          __m512i s0 = _mm512_add_epi64(a0, b0);
          __mmask8 c = _mm512_cmplt_epu64_mask(s0, a0);
          __m512i s1 = _mm512_add_epi64(a1, b1);
          __mmask8 c1 = _mm512_cmplt_epu64_mask(s1, a1);
          s1 = _mm512_mask_add_epi64(s1, c, s1, one);
          c1 = _kor_mask8(c1, _kand_mask8(c, _mm512_cmpeq_epu64_mask(s1, _mm512_setzero_si512())));
          __m512i s2 = _mm512_add_epi64(a2, b2);
          __mmask8 c2 = _mm512_cmplt_epu64_mask(s2, a2);
          s2 = _mm512_mask_add_epi64(s2, c1, s2, one);
          c2 = _kor_mask8(c2, _kand_mask8(c1, _mm512_cmpeq_epu64_mask(s2, _mm512_setzero_si512())));
          __m512i s3 = _mm512_add_epi64(a3, b3);
          __mmask8 c3 = _mm512_cmplt_epu64_mask(s3, a3);
          s3 = _mm512_mask_add_epi64(s3, c2, s3, one);
          c3 = _kor_mask8(c3, _kand_mask8(c2, _mm512_cmpeq_epu64_mask(s3, _mm512_setzero_si512())));

          // corrected value: s + C
          __m512i t0 = _mm512_add_epi64(s0, vC);
          __mmask8 d = _mm512_cmplt_epu64_mask(t0, s0);
          __m512i t1 = _mm512_mask_add_epi64(s1, d, s1, one);
          __mmask8 d1 = _kand_mask8(d, _mm512_cmpeq_epu64_mask(t1, _mm512_setzero_si512()));
          __m512i t2 = _mm512_mask_add_epi64(s2, d1, s2, one);
          __mmask8 d2 = _kand_mask8(d1, _mm512_cmpeq_epu64_mask(t2, _mm512_setzero_si512()));
          __m512i t3 = _mm512_mask_add_epi64(s3, d2, s3, one);
          __mmask8 d3 = _kand_mask8(d2, _mm512_cmpeq_epu64_mask(t3, _mm512_setzero_si512()));

          // take the corrected value if the sum overflowed 2^256 OR was >= p
          __mmask8 sel = _kor_mask8(c3, d3);
          _mm512_storeu_si512(r->v[0], _mm512_mask_blend_epi64(sel, s0, t0));
          _mm512_storeu_si512(r->v[1], _mm512_mask_blend_epi64(sel, s1, t1));
          _mm512_storeu_si512(r->v[2], _mm512_mask_blend_epi64(sel, s2, t2));
          _mm512_storeu_si512(r->v[3], _mm512_mask_blend_epi64(sel, s3, t3));
        }

        inline void fe8_sub_v2(Fe8 *r, const Fe8 *a, const Fe8 *b) {
          const __m512i one = _mm512_set1_epi64(1);
          const __m512i vC = _mm512_set1_epi64((long long)FE8V2_C);
          const __m512i zero = _mm512_setzero_si512();
          __m512i a0 = _mm512_loadu_si512(a->v[0]), a1 = _mm512_loadu_si512(a->v[1]);
          __m512i a2 = _mm512_loadu_si512(a->v[2]), a3 = _mm512_loadu_si512(a->v[3]);
          __m512i b0 = _mm512_loadu_si512(b->v[0]), b1 = _mm512_loadu_si512(b->v[1]);
          __m512i b2 = _mm512_loadu_si512(b->v[2]), b3 = _mm512_loadu_si512(b->v[3]);

          // raw subtraction with borrow propagation
          __m512i s0 = _mm512_sub_epi64(a0, b0);
          __mmask8 c = _mm512_cmpgt_epu64_mask(s0, a0);
          __m512i s1 = _mm512_sub_epi64(a1, b1);
          __mmask8 c1 = _mm512_cmpgt_epu64_mask(s1, a1);
          __mmask8 z1 = _mm512_cmpeq_epu64_mask(s1, zero);
          s1 = _mm512_mask_sub_epi64(s1, c, s1, one);
          c1 = _kor_mask8(c1, _kand_mask8(c, z1));
          __m512i s2 = _mm512_sub_epi64(a2, b2);
          __mmask8 c2 = _mm512_cmpgt_epu64_mask(s2, a2);
          __mmask8 z2 = _mm512_cmpeq_epu64_mask(s2, zero);
          s2 = _mm512_mask_sub_epi64(s2, c1, s2, one);
          c2 = _kor_mask8(c2, _kand_mask8(c1, z2));
          __m512i s3 = _mm512_sub_epi64(a3, b3);
          __mmask8 c3 = _mm512_cmpgt_epu64_mask(s3, a3);
          __mmask8 z3 = _mm512_cmpeq_epu64_mask(s3, zero);
          s3 = _mm512_mask_sub_epi64(s3, c2, s3, one);
          c3 = _kor_mask8(c3, _kand_mask8(c2, z3));

          // corrected value: s - C  (same as s + p mod 2^256)
          __m512i t0 = _mm512_sub_epi64(s0, vC);
          __mmask8 d = _mm512_cmpgt_epu64_mask(t0, s0);
          __m512i t1 = _mm512_mask_sub_epi64(s1, d, s1, one);
          __mmask8 d1 = _kand_mask8(d, _mm512_cmpeq_epu64_mask(s1, zero));
          __m512i t2 = _mm512_mask_sub_epi64(s2, d1, s2, one);
          __mmask8 d2 = _kand_mask8(d1, _mm512_cmpeq_epu64_mask(s2, zero));
          __m512i t3 = _mm512_mask_sub_epi64(s3, d2, s3, one);

          // NOTE: selection uses ONLY the borrow out of the raw x-y (c3). The add
          // above ORs in a second condition and that is correct there -- but for
          // subtraction no second condition exists. This is exactly where the bug
          // in bignum_sub_p256k1_x8_soa.S was: x == y returned p instead of 0.
          _mm512_storeu_si512(r->v[0], _mm512_mask_blend_epi64(c3, s0, t0));
          _mm512_storeu_si512(r->v[1], _mm512_mask_blend_epi64(c3, s1, t1));
          _mm512_storeu_si512(r->v[2], _mm512_mask_blend_epi64(c3, s2, t2));
          _mm512_storeu_si512(r->v[3], _mm512_mask_blend_epi64(c3, s3, t3));
        }

#endif // EC8_FE8V2

#pragma GCC pop_options
#endif // ECLOOP_HAVE_BATCH8

    } // namespace

    // ── runtime dispatch ─────────────────────────────────────────────────────

    const bool EC_BATCH8_OK = []() noexcept {
#if ECLOOP_HAVE_BATCH8
            // ECLOOP_NO_IFMA=1 forces the scalar path -- lets you A/B the
            // batch-8 layer on one binary and makes the fallback testable
            // on IFMA hardware.
            const char* e = std::getenv("ECLOOP_NO_IFMA");
            if (e && e[0] == '1') return false;
            __builtin_cpu_init();
            // __builtin_cpu_supports covers both the CPUID bit and, for the
            // avx512* names, the XGETBV/OS-enabled check.
            if (!__builtin_cpu_supports("avx512f")) return false;
            if (!__builtin_cpu_supports("avx512ifma")) return false;
            return true;
#else
            return false;
#endif
    }();

    // ── AoS <-> SoA ──────────────────────────────────────────────────────────


#if ECLOOP_HAVE_FE52
    // ── the radix-2^52 (fe52.hpp) bodies, compiled for AVX-512 IFMA and
    // called from the dispatching functions below only when the CPU has it.
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl,avx512ifma,bmi2")
    namespace {
        void fe8_mul_ifma(Fe8* r, const Fe8* a, const Fe8* b) noexcept {
            fe52_to_fe8(r, fe52_mul(fe52_from_fe8(a), fe52_from_fe8(b)));
        }
        void fe8_sqr_ifma(Fe8* r, const Fe8* a) noexcept {
            fe52_to_fe8(r, fe52_sqr(fe52_from_fe8(a)));
        }
        void ec_jacobi_add_affine_x8_ifma(Pe8* r, const Pe8* p, const Fe8* qx, const Fe8* qy) noexcept {
        // Whole formula in radix 2^52, values in registers, nothing
        // canonical until X3/Y3/Z3 are stored. Bounds follow fe52.hpp:
        // every multiply input is N-form (a mul/sqr output, or a lazy
        // add/sub run through fe52_normalize); the subtrahend of every
        // fe52_sub is N-form or one add of N-forms; fe52_sub16 is used
        // where it is two levels deep.
        const Fe52 X1 = fe52_from_fe8(&p->x), Y1 = fe52_from_fe8(&p->y);
        const Fe52 Z1 = fe52_from_fe8(&p->z);
        const Fe52 X2 = fe52_from_fe8(qx),    Y2 = fe52_from_fe8(qy);

        const Fe52 Z1Z1 = fe52_sqr(Z1);
        const Fe52 U2   = fe52_mul(X2, Z1Z1);
        const Fe52 S2   = fe52_mul(Y2, fe52_mul(Z1, Z1Z1));
        const Fe52 H    = fe52_normalize(fe52_sub(U2, X1));
        const Fe52 HH   = fe52_sqr(H);
        const Fe52 I    = fe52_normalize(fe52_dbl(fe52_dbl(HH)));
        const Fe52 J    = fe52_mul(H, I);
        const Fe52 R    = fe52_normalize(fe52_dbl(fe52_sub(S2, Y1)));
        const Fe52 V    = fe52_mul(X1, I);
        const Fe52 X3   = fe52_normalize(fe52_sub16(fe52_sub(fe52_sqr(R), J), fe52_dbl(V)));
        const Fe52 Y3   = fe52_sub(fe52_mul(R, fe52_normalize(fe52_sub(V, X3))),
                                   fe52_dbl(fe52_mul(Y1, J)));
        const Fe52 Z3   = fe52_sub(fe52_sub(fe52_sqr(fe52_normalize(fe52_add(Z1, H))), Z1Z1), HH);

        // Degenerate lanes, same classification as below: Z1 == 0 (P at
        // infinity, result is Q), H == 0 and r == 0 (P == Q, needs 2P).
        const u32 zm  = fe52_iszero_nform(Z1);
        const u32 dbl = fe52_iszero_nform(H) & fe52_iszero_nform(R) & ~zm;
        const u32 bad = zm | dbl;

        pe save[8];
        fe qsx[8], qsy[8];
        if (__builtin_expect(bad != 0, 0)) {
            for (int l = 0; l < 8; ++l) {
                if (!((bad >> l) & 1)) continue;
                pe8_get_lane(&save[l], p, l);
                fe8_get_lane(qsx[l], qx, l);
                fe8_get_lane(qsy[l], qy, l);
            }
        }

        fe52_to_fe8(&r->x, X3);
        fe52_to_fe8(&r->y, Y3);
        fe52_to_fe8(&r->z, Z3);

        if (__builtin_expect(bad != 0, 0)) {
            for (int l = 0; l < 8; ++l) {
                if (!((bad >> l) & 1)) continue;
                pe o;
                if ((zm >> l) & 1) {
                    fe_clone(o.x, qsx[l]);
                    fe_clone(o.y, qsy[l]);
                    fe_set64(o.z, 1);
                } else {
                    ec_jacobi_dbl(&o, &save[l]);
                }
                pe8_set_lane(r, l, &o);
            }
        }
        }
        void ec_jacobi_grprdc_x8_ifma(Pe8* pts, std::size_t n, Fe8* scratch, const u32* zmask) noexcept {
        {
            // Running products in radix 2^52. Four interleaved chains when
            // n allows it (n % 4 == 0, n >= 8) so the multiply latency is
            // hidden; one chain otherwise.
            Fe52s* accs = reinterpret_cast<Fe52s*>(scratch);
            const bool four = (n % 4 == 0) && n >= 8;
            Fe52 run;
            if (four) {
                Fe52 a0 = fe52_from_fe8(&pts[0].z), a1 = fe52_from_fe8(&pts[1].z);
                Fe52 a2 = fe52_from_fe8(&pts[2].z), a3 = fe52_from_fe8(&pts[3].z);
                fe52_store(&accs[0], a0); fe52_store(&accs[1], a1);
                fe52_store(&accs[2], a2); fe52_store(&accs[3], a3);
                for (std::size_t i = 4; i < n; i += 4) {
                    a0 = fe52_mul(a0, fe52_from_fe8(&pts[i + 0].z));
                    a1 = fe52_mul(a1, fe52_from_fe8(&pts[i + 1].z));
                    a2 = fe52_mul(a2, fe52_from_fe8(&pts[i + 2].z));
                    a3 = fe52_mul(a3, fe52_from_fe8(&pts[i + 3].z));
                    fe52_store(&accs[i + 0], a0); fe52_store(&accs[i + 1], a1);
                    fe52_store(&accs[i + 2], a2); fe52_store(&accs[i + 3], a3);
                }
                const Fe52 p01 = fe52_mul(a0, a1), p23 = fe52_mul(a2, a3);
                const Fe52 invP = fe52_invert_lanes(fe52_mul(p01, p23));
                const Fe52 inv01 = fe52_mul(invP, p23), inv23 = fe52_mul(invP, p01);
                Fe52 i0 = fe52_mul(inv01, a1), i1 = fe52_mul(inv01, a0);
                Fe52 i2 = fe52_mul(inv23, a3), i3 = fe52_mul(inv23, a2);
                for (std::size_t i = n - 4; i >= 4; i -= 4) {
                    const Fe52 z0 = fe52_from_fe8(&pts[i + 0].z), z1 = fe52_from_fe8(&pts[i + 1].z);
                    const Fe52 z2 = fe52_from_fe8(&pts[i + 2].z), z3 = fe52_from_fe8(&pts[i + 3].z);
                    fe52_store(&accs[i + 0], fe52_mul(i0, fe52_load(&accs[i - 4])));
                    fe52_store(&accs[i + 1], fe52_mul(i1, fe52_load(&accs[i - 3])));
                    fe52_store(&accs[i + 2], fe52_mul(i2, fe52_load(&accs[i - 2])));
                    fe52_store(&accs[i + 3], fe52_mul(i3, fe52_load(&accs[i - 1])));
                    i0 = fe52_mul(i0, z0); i1 = fe52_mul(i1, z1);
                    i2 = fe52_mul(i2, z2); i3 = fe52_mul(i3, z3);
                }
                fe52_store(&accs[0], i0); fe52_store(&accs[1], i1);
                fe52_store(&accs[2], i2); fe52_store(&accs[3], i3);
            } else {
                Fe52 a = fe52_from_fe8(&pts[0].z);
                fe52_store(&accs[0], a);
                for (std::size_t i = 1; i < n; ++i) {
                    a = fe52_mul(a, fe52_from_fe8(&pts[i].z));
                    fe52_store(&accs[i], a);
                }
                run = fe52_invert_lanes(a);
                for (std::size_t i = n - 1; i > 0; --i) {
                    const Fe52 z = fe52_from_fe8(&pts[i].z);
                    fe52_store(&accs[i], fe52_mul(run, fe52_load(&accs[i - 1])));
                    run = fe52_mul(run, z);
                }
                fe52_store(&accs[0], run);
            }
            // accs[i] is now 1/z_i: x = X*zinv^2, y = Y*zinv^3, z = 1
            for (std::size_t i = 0; i < n; ++i) {
                const Fe52 zi = fe52_load(&accs[i]);
                const Fe52 zi2 = fe52_sqr(zi);
                const Fe52 zi3 = fe52_mul(zi2, zi);
                fe52_to_fe8(&pts[i].x, fe52_mul(fe52_from_fe8(&pts[i].x), zi2));
                fe52_to_fe8(&pts[i].y, fe52_mul(fe52_from_fe8(&pts[i].y), zi3));
                fe8_one(&pts[i].z);
                if (__builtin_expect(zmask[i] != 0, 0)) {
                    for (int l = 0; l < 8; ++l)
                        if ((zmask[i] >> l) & 1)
                            for (int k = 0; k < 4; ++k) {
                                pts[i].x.v[k][l] = 0;
                                pts[i].y.v[k][l] = 0;
                            }
                }
            }
        }
        }
    } // namespace
#pragma GCC pop_options
#endif // ECLOOP_HAVE_FE52

    void fe8_set_lane(Fe8* r, int lane, const fe a) noexcept {
        r->v[0][lane] = a[0];
        r->v[1][lane] = a[1];
        r->v[2][lane] = a[2];
        r->v[3][lane] = a[3];
    }

    void fe8_get_lane(fe r, const Fe8* a, int lane) noexcept {
        r[0] = a->v[0][lane];
        r[1] = a->v[1][lane];
        r[2] = a->v[2][lane];
        r[3] = a->v[3][lane];
    }

    void pe8_set_lane(Pe8* r, int lane, const pe* p) noexcept {
        fe8_set_lane(&r->x, lane, p->x);
        fe8_set_lane(&r->y, lane, p->y);
        fe8_set_lane(&r->z, lane, p->z);
    }

    void pe8_get_lane(pe* r, const Pe8* a, int lane) noexcept {
        fe8_get_lane(r->x, &a->x, lane);
        fe8_get_lane(r->y, &a->y, lane);
        fe8_get_lane(r->z, &a->z, lane);
    }

    void fe8_set_all(Fe8* r, const fe a) noexcept {
        for (int l = 0; l < 8; ++l) fe8_set_lane(r, l, a);
    }

    void fe8_zero(Fe8* r) noexcept { std::memset(r, 0, sizeof(Fe8)); }

    void fe8_one(Fe8* r) noexcept {
        std::memset(r, 0, sizeof(Fe8));
        for (int l = 0; l < 8; ++l) r->v[0][l] = 1;
    }

    // Inputs here are always canonical (< p): every s2n primitive returns
    // canonical output, so the "p represents 0" ambiguity cannot arise on
    // values we produce.
    u32 fe8_zero_mask(const Fe8* a) noexcept {
#if EC8_FE8V2 && ECLOOP_HAVE_BATCH8
        return fe8_zero_mask_v2(a);
#else
        u32 m = 0;
        for (int l = 0; l < 8; ++l) {
            const u64 acc = a->v[0][l] | a->v[1][l] | a->v[2][l] | a->v[3][l];
            m |= static_cast<u32>(acc == 0) << l;
        }
        return m;
#endif
    }

    // ── batch-8 field ops ────────────────────────────────────────────────────

    void fe8_add(Fe8* r, const Fe8* a, const Fe8* b) noexcept {
        if (__builtin_expect(ec_batch8_available(), 1)) {
#if EC8_FE8V2 && ECLOOP_HAVE_BATCH8
            fe8_add_v2(r, a, b);
#else
            bignum_add_p256k1_x8_soa(r->v, a->v, b->v);
#endif
        } else {
            for (int l = 0; l < 8; ++l) {
                fe t, u, w;
                fe8_get_lane(t, a, l);
                fe8_get_lane(u, b, l);
                bignum_add_p256k1(w, t, u);
                fe8_set_lane(r, l, w);
            }
        }
    }

    void fe8_sub(Fe8* r, const Fe8* a, const Fe8* b) noexcept {
        if (__builtin_expect(ec_batch8_available(), 1)) {
#if EC8_FE8V2 && ECLOOP_HAVE_BATCH8
            fe8_sub_v2(r, a, b);
#else
            bignum_sub_p256k1_x8_soa(r->v, a->v, b->v);
#endif
        } else {
            for (int l = 0; l < 8; ++l) {
                fe t, u, w;
                fe8_get_lane(t, a, l);
                fe8_get_lane(u, b, l);
                bignum_sub_p256k1(w, t, u);
                fe8_set_lane(r, l, w);
            }
        }
    }

    void fe8_mul(Fe8* r, const Fe8* a, const Fe8* b) noexcept {
        if (__builtin_expect(ec_batch8_available(), 1)) {
#if ECLOOP_HAVE_FE52
            // Radix-2^52 kernel (fe52.hpp): ~85 instructions against ~200 for
            // the v2 kernel, even with the conversions at both ends.
            fe8_mul_ifma(r, a, b);
#elif EC8_MULV2 && ECLOOP_HAVE_BATCH8
            bignum_mul_p256k1_x8_soa_v2(r->v, a->v, b->v);
#else
            bignum_mul_p256k1_x8_soa(r->v, a->v, b->v);
#endif
        } else {
            for (int l = 0; l < 8; ++l) {
                fe t, u, w;
                fe8_get_lane(t, a, l);
                fe8_get_lane(u, b, l);
                bignum_mul_p256k1(w, t, u);
                fe8_set_lane(r, l, w);
            }
        }
    }

    // With EC8_MULV2 there is a dedicated kernel exploiting symmetry: only
    // the 15 distinct partial products (5 on the diagonal, 10 off it, the
    // latter doubled) -- 30 IFMA ops instead of 50. Without it there is no
    // batch squaring and this is just mul(a, a).
    void fe8_sqr(Fe8* r, const Fe8* a) noexcept {
#if ECLOOP_HAVE_FE52
        if (__builtin_expect(ec_batch8_available(), 1)) {
            fe8_sqr_ifma(r, a);
            return;
        }
#elif EC8_MULV2 && EC8_SQRV2 && ECLOOP_HAVE_BATCH8
        if (__builtin_expect(ec_batch8_available(), 1)) {
            bignum_sqr_p256k1_x8_soa_v2(r->v, a->v);
            return;
        }
#endif
        fe8_mul(r, a, a);
    }

    // ── points ───────────────────────────────────────────────────────────────

    // MEASURED in the C tree (tests/t_degen.c, 300 trials each): stock s2n
    // secp256k1_jmixadd handles P == -Q (returns infinity) and P == infinity
    // (returns Q) correctly, but does NOT handle P == Q -- it returns
    // infinity where 2P is required, in 300/300 trials. So that one case,
    // and only that one, needs intercepting.
    //
    // The check is done after the fact: jmixadd yields z == 0 only when the
    // x coordinates coincide, so the 3 extra muls that tell P == Q from
    // P == -Q are paid only in the rare case that needs them.
    void ec_jmixadd_safe(pe* r, const pe* p, const pe* q) noexcept {
        if (__builtin_expect(fe_iszero(p->z), 0)) {
            pe_clone(r, q);
            return;
        }
        // P is saved because r and p may be the SAME object. The s2n point
        // routines tolerate that aliasing, but the recovery branch below
        // needs the ORIGINAL P, and by then r has overwritten it. The C
        // original reads p->z and p->y after the add: correct only while
        // nobody calls it with r == p, and silently wrong the moment
        // somebody does -- a degenerate step then yields infinity instead
        // of 2P, and every later step walks from the wrong point.
        const pe P = *p;

        ec_jacobi_add_affine(r, &P, q);
        if (__builtin_expect(fe_iszero(r->z), 0)) {
            fe z2, z3, s2;
            fe_modp_sqr(z2, P.z);
            fe_modp_mul(z3, z2, P.z);
            fe_modp_mul(s2, q->y, z3); // Y2 * Z1^3
            if (fe_cmp(s2, P.y) == 0) ec_jacobi_dbl(r, &P); // P == Q -> 2P
            // else P == -Q: infinity is the right answer, leave it
        }
    }

    // madd-2007-bl, 7M + 4S. Q is affine so z2 == 1 and never appears.
    //
    //   Z1Z1 = Z1^2 ; U2 = X2*Z1Z1 ; S2 = Y2*Z1*Z1Z1
    //   H = U2-X1 ; HH = H^2 ; I = 4*HH ; J = H*I ; r = 2*(S2-Y1) ; V = X1*I
    //   X3 = r^2 - J - 2V
    //   Y3 = r*(V-X3) - 2*Y1*J
    //   Z3 = (Z1+H)^2 - Z1Z1 - HH     [ = 2*Z1*H ]
    //
    // Degenerate lanes are patched afterwards from the scalar path.
    void ec_jacobi_add_affine_x8(Pe8* r, const Pe8* p, const Fe8* qx,
                                 const Fe8* qy) noexcept {
        if (__builtin_expect(!ec_batch8_available(), 0)) {
            for (int l = 0; l < 8; ++l) {
                pe a, q, o;
                pe8_get_lane(&a, p, l);
                fe8_get_lane(q.x, qx, l);
                fe8_get_lane(q.y, qy, l);
                fe_set64(q.z, 1);
                ec_jmixadd_safe(&o, &a, &q);
                pe8_set_lane(r, l, &o);
            }
            return;
        }

#if ECLOOP_HAVE_FE52
        ec_jacobi_add_affine_x8_ifma(r, p, qx, qy);
#else // radix-2^64 formula on the Fe8 primitives
        Fe8 z1z1, u2, s2, h, hh, i4, j, rr, v, t0, t1, x3, y3, z3;

        fe8_sqr(&z1z1, &p->z);      // Z1Z1 = Z1^2
        fe8_mul(&u2, qx, &z1z1);    // U2   = X2*Z1Z1
        fe8_mul(&t0, &p->z, &z1z1); // Z1^3
        fe8_mul(&s2, qy, &t0);      // S2   = Y2*Z1^3

        fe8_sub(&h, &u2, &p->x); // H = U2 - X1
        fe8_sqr(&hh, &h);        // HH = H^2
        fe8_dbl(&i4, &hh);
        fe8_dbl(&i4, &i4);        // I = 4*HH
        fe8_mul(&j, &h, &i4);     // J = H*I
        fe8_sub(&t1, &s2, &p->y);
        fe8_dbl(&rr, &t1);        // r = 2*(S2-Y1)
        fe8_mul(&v, &p->x, &i4);  // V = X1*I

        fe8_sqr(&x3, &rr);       // r^2
        fe8_sub(&x3, &x3, &j);   // -J
        fe8_dbl(&t0, &v);
        fe8_sub(&x3, &x3, &t0);  // -2V  => X3

        fe8_sub(&t0, &v, &x3);   // V - X3
        fe8_mul(&y3, &rr, &t0);  // r*(V-X3)
        fe8_mul(&t0, &p->y, &j); // Y1*J
        fe8_dbl(&t0, &t0);       // 2*Y1*J
        fe8_sub(&y3, &y3, &t0);  // => Y3

        fe8_add(&z3, &p->z, &h);
        fe8_sqr(&z3, &z3);       // (Z1+H)^2
        fe8_sub(&z3, &z3, &z1z1);
        fe8_sub(&z3, &z3, &hh);  // => Z3

        // Which lanes did the generic formula get wrong?
        //
        //   H == 0            <=> the x coordinates coincide
        //   H == 0, r == 0    <=> P == Q  -> needs doubling, formula gives infinity
        //   H == 0, r != 0    <=> P == -Q -> infinity IS correct, Z3 is already 0
        //   Z1 == 0           <=> P is infinity -> result must be Q. Such a lane
        //                         also has H == 0 and r == 0, so it must be
        //                         excluded from the doubling mask.
        const u32 zm = fe8_zero_mask(&p->z);
        const u32 dbl = fe8_zero_mask(&h) & fe8_zero_mask(&rr) & ~zm;
        const u32 bad = zm | dbl;

        // Save the operands BEFORE overwriting r, so r == p aliasing is safe.
        pe save[8];
        fe qsx[8], qsy[8];
        if (__builtin_expect(bad != 0, 0)) {
            for (int l = 0; l < 8; ++l) {
                if (!((bad >> l) & 1)) continue;
                pe8_get_lane(&save[l], p, l);
                fe8_get_lane(qsx[l], qx, l);
                fe8_get_lane(qsy[l], qy, l);
            }
        }

        r->x = x3;
        r->y = y3;
        r->z = z3;

        if (__builtin_expect(bad != 0, 0)) {
            for (int l = 0; l < 8; ++l) {
                if (!((bad >> l) & 1)) continue;
                pe o;
                if ((zm >> l) & 1) { // P was infinity -> result is Q
                    fe_clone(o.x, qsx[l]);
                    fe_clone(o.y, qsy[l]);
                    fe_set64(o.z, 1);
                } else { // P == Q -> 2P
                    ec_jacobi_dbl(&o, &save[l]);
                }
                pe8_set_lane(r, l, &o);
            }
        }
#endif // ECLOOP_HAVE_FE52
    }

    std::size_t ec_grprdc_x8_scratch(std::size_t n) noexcept {
#if ECLOOP_HAVE_FE52
        // The radix-2^52 chain keeps n running products of 320 bytes each
        // in the scratch, which is counted in 256-byte Fe8 units.
        return (n * sizeof(Fe52s) + sizeof(Fe8) - 1) / sizeof(Fe8) + 1;
#elif EC8_INVTREE
        return 4 * n;
#else
        return n;
#endif
    }

    // n Pe8 = 8n Jacobian points -> affine, ONE field inversion for all.
    //
    // The Montgomery trick is a strictly serial product chain, which looks
    // hostile to batching -- but running EIGHT independent chains (one per
    // lane) is perfectly parallel. That yields 8 lane-products, which a
    // small scalar product tree collapses into a single inversion and
    // expands back.
    //
    // Cost: 7n batched muls + 21 scalar muls + 1 inversion, versus
    // 7*(8n) scalar muls + 1 inversion for _ec_jacobi_grprdc1.
    void ec_jacobi_grprdc_x8(Pe8* pts, std::size_t n, Fe8* scratch) noexcept {
        if (n == 0) return;

        if (__builtin_expect(!ec_batch8_available(), 0)) {
            for (std::size_t i = 0; i < n; ++i) {
                pe tmp[8];
                for (int l = 0; l < 8; ++l) pe8_get_lane(&tmp[l], &pts[i], l);
                ec_jacobi_grprdc(tmp, 8);
                for (int l = 0; l < 8; ++l) pe8_set_lane(&pts[i], l, &tmp[l]);
            }
            return;
        }

        Fe8* acc = scratch; // caller-owned: no malloc in the hot loop

        // z == 0 would poison the whole chain (one zero makes every product
        // 0), exactly as documented in fe_modp_grpinv. Substitute 1 and
        // restore afterwards.
        //
        // std::vector, not alloca: n is caller-controlled and alloca has no
        // failure mode -- a large n silently runs off the stack.
        std::vector<u32> zmask(n);
        for (std::size_t i = 0; i < n; ++i) {
            const u32 m = fe8_zero_mask(&pts[i].z);
            zmask[i] = m;
            if (__builtin_expect(m != 0, 0)) {
                for (int l = 0; l < 8; ++l)
                    if ((m >> l) & 1) {
                        pts[i].z.v[0][l] = 1;
                        pts[i].z.v[1][l] = pts[i].z.v[2][l] = pts[i].z.v[3][l] = 0;
                    }
            }
        }

#if ECLOOP_HAVE_FE52
        ec_jacobi_grprdc_x8_ifma(pts, n, scratch, zmask.data());
        return;
#endif // ECLOOP_HAVE_FE52

#if EC8_INVTREE
        const bool use_tree = n >= 2 && (n & (n - 1)) == 0;
#endif

        fe lane[8], pre[8], inv, t;

#if EC8_INVTREE
        if (use_tree) {
            // Tree form of the Montgomery trick instead of a chain. The
            // chain acc[i] = acc[i-1]*z[i] has depth n-1; a binary tree does
            // the same n-1 products with depth log2(n).
            //
            // Leaves at 0..n-1, internal node i at n+i with children 2i and
            // 2i+1, root at 2n-2. Needs n a power of two and 2n-1 entries.
            for (std::size_t i = 0; i < n; ++i) acc[i] = pts[i].z;
            for (std::size_t i = 0; i + 1 < n; ++i)
                fe8_mul(&acc[n + i], &acc[2 * i], &acc[2 * i + 1]);
        } else {
            acc[0] = pts[0].z;
            for (std::size_t i = 1; i < n; ++i) fe8_mul(&acc[i], &acc[i - 1], &pts[i].z);
        }
        for (int l = 0; l < 8; ++l)
            fe8_get_lane(lane[l], use_tree ? &acc[2 * n - 2] : &acc[n - 1], l);
#else
        // forward: acc[i] = z[0]*...*z[i], per lane
        acc[0] = pts[0].z;
        for (std::size_t i = 1; i < n; ++i) fe8_mul(&acc[i], &acc[i - 1], &pts[i].z);
        for (int l = 0; l < 8; ++l) fe8_get_lane(lane[l], &acc[n - 1], l);
#endif

        // collapse the 8 lane-products to one inversion and expand back
        fe_clone(pre[0], lane[0]);
        for (int l = 1; l < 8; ++l) fe_modp_mul(pre[l], pre[l - 1], lane[l]);

        fe_modp_inv(inv, pre[7]);

        fe invl[8];
        for (int l = 7; l >= 1; --l) {
            fe_modp_mul(invl[l], inv, pre[l - 1]);
            fe_modp_mul(t, inv, lane[l]);
            fe_clone(inv, t);
        }
        fe_clone(invl[0], inv);

        Fe8 run;
        for (int l = 0; l < 8; ++l) fe8_set_lane(&run, l, invl[l]);

#if EC8_INVTREE
        if (use_tree) {
            // Down-sweep: push the inverted root back down. Node i holds
            // 1/(product of its subtree); its children get it multiplied by
            // the sibling. Depth log2(n), and the two products at each node
            // are independent.
            Fe8* iv = acc + 2 * n - 1; // second half of scratch
            iv[2 * n - 2] = run;
            for (std::size_t k = n - 1; k-- > 0;) {
                const Fe8 a = acc[2 * k], b = acc[2 * k + 1];
                fe8_mul(&iv[2 * k], &b, &iv[n + k]);
                fe8_mul(&iv[2 * k + 1], &a, &iv[n + k]);
            }
            for (std::size_t i = 0; i < n; ++i) acc[i] = iv[i];
        } else {
            for (std::size_t i = n - 1; i > 0; --i) {
                Fe8 zi;
                fe8_mul(&zi, &run, &acc[i - 1]);
                fe8_mul(&run, &run, &pts[i].z);
                acc[i] = zi;
            }
            acc[0] = run;
        }
#else
        // backward: turn run into each point's z^-1
        for (std::size_t i = n - 1; i > 0; --i) {
            Fe8 zi;
            fe8_mul(&zi, &run, &acc[i - 1]);
            fe8_mul(&run, &run, &pts[i].z);
            acc[i] = zi; // reuse acc as the z^-1 store
        }
        acc[0] = run;
#endif

        // x = X*zinv^2, y = Y*zinv^3, z = 1
        for (std::size_t i = 0; i < n; ++i) {
            Fe8 zinv2, zinv3;
            fe8_sqr(&zinv2, &acc[i]);
            fe8_mul(&zinv3, &zinv2, &acc[i]);
            fe8_mul(&pts[i].x, &pts[i].x, &zinv2);
            fe8_mul(&pts[i].y, &pts[i].y, &zinv3);
            fe8_one(&pts[i].z);

            // Degenerate (infinity) lanes: x = y = 0, z = 1. This matches
            // the scalar _ec_jacobi_grprdc1, which writes z = 1
            // unconditionally while x,y become zero on their own. Matching
            // it is not cosmetic: prepare33/prepare65 assert z == 1 and the
            // build has no -DNDEBUG, so z = 0 would abort the process.
            if (__builtin_expect(zmask[i] != 0, 0)) {
                for (int l = 0; l < 8; ++l)
                    if ((zmask[i] >> l) & 1)
                        for (int k = 0; k < 4; ++k) {
                            pts[i].x.v[k][l] = 0;
                            pts[i].y.v[k][l] = 0;
                        }
            }
        }
    }

} // namespace ecloop

