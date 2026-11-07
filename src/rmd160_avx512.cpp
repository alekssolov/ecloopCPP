// RIPEMD-160 x16 (AVX-512). Port of lib/rmd160_avx512.c.
//
// The round schedule below is copied verbatim from the C original; only
// the surrounding scaffolding is C++ (namespace, noexcept, constexpr
// tables). The vpternlogd truth tables and the rotate-by-immediate are
// unchanged.

#include "rmd160_avx512.hpp"

#if ECLOOP_HAVE_RMD160_AVX512

#include <cstring>

#include "compat.hpp"

#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl")
namespace ecloop {
    namespace {

        constexpr u32 R5_K1 = 0x67452301u;
        constexpr u32 R5_K2 = 0xEFCDAB89u;
        constexpr u32 R5_K3 = 0x98BADCFEu;
        constexpr u32 R5_K4 = 0x10325476u;
        constexpr u32 R5_K5 = 0xC3D2E1F0u;

        // ── the vector type of the core ──────────────────────────────────
        // The round schedule below is written once, over an abstract vector
        // V: one zmm (16 hashes), or V2, a pair of zmm carrying two
        // independent batches of 16 in lockstep. RIPEMD-160 is a dependent
        // chain of 80 steps per line and the core alone leaves the vector
        // ports half idle (measured: independent batches run no faster
        // than dependent ones); two batches interleaved fill the gaps.
        struct V2 {
            __m512i x, y;
        };

        ECLOOP_ALWAYS_INLINE __m512i r5_add(__m512i a, __m512i b) noexcept { return _mm512_add_epi32(a, b); }
        ECLOOP_ALWAYS_INLINE V2 r5_add(V2 a, V2 b) noexcept { return {_mm512_add_epi32(a.x, b.x), _mm512_add_epi32(a.y, b.y)}; }

        template <int N> ECLOOP_ALWAYS_INLINE __m512i r5_rol(__m512i a) noexcept { return _mm512_rol_epi32(a, N); }
        template <int N> ECLOOP_ALWAYS_INLINE V2 r5_rol(V2 a) noexcept { return {_mm512_rol_epi32(a.x, N), _mm512_rol_epi32(a.y, N)}; }

        template <int IMM> ECLOOP_ALWAYS_INLINE __m512i r5_tern(__m512i a, __m512i b, __m512i c) noexcept {
            return _mm512_ternarylogic_epi32(a, b, c, IMM);
        }
        template <int IMM> ECLOOP_ALWAYS_INLINE V2 r5_tern(V2 a, V2 b, V2 c) noexcept {
            return {_mm512_ternarylogic_epi32(a.x, b.x, c.x, IMM), _mm512_ternarylogic_epi32(a.y, b.y, c.y, IMM)};
        }

        template <class V> ECLOOP_ALWAYS_INLINE V r5_set1(int k) noexcept;
        template <> ECLOOP_ALWAYS_INLINE __m512i r5_set1<__m512i>(int k) noexcept { return _mm512_set1_epi32(k); }
        template <> ECLOOP_ALWAYS_INLINE V2 r5_set1<V2>(int k) noexcept { return {_mm512_set1_epi32(k), _mm512_set1_epi32(k)}; }

        // vpternlogd: the immediate is a per-bit truth table indexed by
        // (a<<2 | b<<1 | c).
        //   f1 = x^y^z         -> 0x96
        //   f2 = (x&y)|(~x&z)  -> 0xCA
        //   f3 = (x|~y)^z      -> 0x59
        //   f4 = (x&z)|(y&~z)  -> 0xE4
        //   f5 = x^(y|~z)      -> 0x2D
#define f1(x, y, z) r5_tern<0x96>(x, y, z)
#define f2(x, y, z) r5_tern<0xCA>(x, y, z)
#define f3(x, y, z) r5_tern<0x59>(x, y, z)
#define f4(x, y, z) r5_tern<0xE4>(x, y, z)
#define f5(x, y, z) r5_tern<0x2D>(x, y, z)

#define R5_ROL(x, n) r5_rol<n>(x)
#define add3(x0, x1, x2) r5_add(r5_add(x0, x1), x2)
#define add4(x0, x1, x2, x3) r5_add(r5_add(x0, x1), r5_add(x2, x3))

#define Round(a, b, c, d, e, f, x, k, r) \
  u = add4(a, f, x, r5_set1<V>(k)); \
  a = r5_add(R5_ROL(u, r), e); \
  c = R5_ROL(c, 10)

#define R11(a, b, c, d, e, x, r) Round(a, b, c, d, e, f1(b, c, d), x, 0, r)
#define R21(a, b, c, d, e, x, r) Round(a, b, c, d, e, f2(b, c, d), x, 0x5A827999ul, r)
#define R31(a, b, c, d, e, x, r) Round(a, b, c, d, e, f3(b, c, d), x, 0x6ED9EBA1ul, r)
#define R41(a, b, c, d, e, x, r) Round(a, b, c, d, e, f4(b, c, d), x, 0x8F1BBCDCul, r)
#define R51(a, b, c, d, e, x, r) Round(a, b, c, d, e, f5(b, c, d), x, 0xA953FD4Eul, r)
#define R12(a, b, c, d, e, x, r) Round(a, b, c, d, e, f5(b, c, d), x, 0x50A28BE6ul, r)
#define R22(a, b, c, d, e, x, r) Round(a, b, c, d, e, f4(b, c, d), x, 0x5C4DD124ul, r)
#define R32(a, b, c, d, e, x, r) Round(a, b, c, d, e, f3(b, c, d), x, 0x6D703EF3ul, r)
#define R42(a, b, c, d, e, x, r) Round(a, b, c, d, e, f2(b, c, d), x, 0x7A6D76E9ul, r)
#define R52(a, b, c, d, e, x, r) Round(a, b, c, d, e, f1(b, c, d), x, 0, r)

        // w[0..15]: message words, already transposed and byte swapped.
        template <class V>
        void rmd160_core(V s[5], const V w[16]) noexcept {
  V a1 = r5_set1<V>((int)R5_K1);
  V b1 = r5_set1<V>((int)R5_K2);
  V c1 = r5_set1<V>((int)R5_K3);
  V d1 = r5_set1<V>((int)R5_K4);
  V e1 = r5_set1<V>((int)R5_K5);
  V a2 = a1, b2 = b1, c2 = c1, d2 = d1, e2 = e1;
  V u;

    // Rounds 0-15
    R11(a1, b1, c1, d1, e1, w[0], 11);
    R12(a2, b2, c2, d2, e2, w[5], 8);
    R11(e1, a1, b1, c1, d1, w[1], 14);
    R12(e2, a2, b2, c2, d2, w[14], 9);
    R11(d1, e1, a1, b1, c1, w[2], 15);
    R12(d2, e2, a2, b2, c2, w[7], 9);
    R11(c1, d1, e1, a1, b1, w[3], 12);
    R12(c2, d2, e2, a2, b2, w[0], 11);
    R11(b1, c1, d1, e1, a1, w[4], 5);
    R12(b2, c2, d2, e2, a2, w[9], 13);
    R11(a1, b1, c1, d1, e1, w[5], 8);
    R12(a2, b2, c2, d2, e2, w[2], 15);
    R11(e1, a1, b1, c1, d1, w[6], 7);
    R12(e2, a2, b2, c2, d2, w[11], 15);
    R11(d1, e1, a1, b1, c1, w[7], 9);
    R12(d2, e2, a2, b2, c2, w[4], 5);
    R11(c1, d1, e1, a1, b1, w[8], 11);
    R12(c2, d2, e2, a2, b2, w[13], 7);
    R11(b1, c1, d1, e1, a1, w[9], 13);
    R12(b2, c2, d2, e2, a2, w[6], 7);
    R11(a1, b1, c1, d1, e1, w[10], 14);
    R12(a2, b2, c2, d2, e2, w[15], 8);
    R11(e1, a1, b1, c1, d1, w[11], 15);
    R12(e2, a2, b2, c2, d2, w[8], 11);
    R11(d1, e1, a1, b1, c1, w[12], 6);
    R12(d2, e2, a2, b2, c2, w[1], 14);
    R11(c1, d1, e1, a1, b1, w[13], 7);
    R12(c2, d2, e2, a2, b2, w[10], 14);
    R11(b1, c1, d1, e1, a1, w[14], 9);
    R12(b2, c2, d2, e2, a2, w[3], 12);
    R11(a1, b1, c1, d1, e1, w[15], 8);
    R12(a2, b2, c2, d2, e2, w[12], 6);

    R21(e1, a1, b1, c1, d1, w[7], 7);
    R22(e2, a2, b2, c2, d2, w[6], 9);
    R21(d1, e1, a1, b1, c1, w[4], 6);
    R22(d2, e2, a2, b2, c2, w[11], 13);
    R21(c1, d1, e1, a1, b1, w[13], 8);
    R22(c2, d2, e2, a2, b2, w[3], 15);
    R21(b1, c1, d1, e1, a1, w[1], 13);
    R22(b2, c2, d2, e2, a2, w[7], 7);
    R21(a1, b1, c1, d1, e1, w[10], 11);
    R22(a2, b2, c2, d2, e2, w[0], 12);
    R21(e1, a1, b1, c1, d1, w[6], 9);
    R22(e2, a2, b2, c2, d2, w[13], 8);
    R21(d1, e1, a1, b1, c1, w[15], 7);
    R22(d2, e2, a2, b2, c2, w[5], 9);
    R21(c1, d1, e1, a1, b1, w[3], 15);
    R22(c2, d2, e2, a2, b2, w[10], 11);
    R21(b1, c1, d1, e1, a1, w[12], 7);
    R22(b2, c2, d2, e2, a2, w[14], 7);
    R21(a1, b1, c1, d1, e1, w[0], 12);
    R22(a2, b2, c2, d2, e2, w[15], 7);
    R21(e1, a1, b1, c1, d1, w[9], 15);
    R22(e2, a2, b2, c2, d2, w[8], 12);
    R21(d1, e1, a1, b1, c1, w[5], 9);
    R22(d2, e2, a2, b2, c2, w[12], 7);
    R21(c1, d1, e1, a1, b1, w[2], 11);
    R22(c2, d2, e2, a2, b2, w[4], 6);
    R21(b1, c1, d1, e1, a1, w[14], 7);
    R22(b2, c2, d2, e2, a2, w[9], 15);
    R21(a1, b1, c1, d1, e1, w[11], 13);
    R22(a2, b2, c2, d2, e2, w[1], 13);
    R21(e1, a1, b1, c1, d1, w[8], 12);
    R22(e2, a2, b2, c2, d2, w[2], 11);

    R31(d1, e1, a1, b1, c1, w[3], 11);
    R32(d2, e2, a2, b2, c2, w[15], 9);
    R31(c1, d1, e1, a1, b1, w[10], 13);
    R32(c2, d2, e2, a2, b2, w[5], 7);
    R31(b1, c1, d1, e1, a1, w[14], 6);
    R32(b2, c2, d2, e2, a2, w[1], 15);
    R31(a1, b1, c1, d1, e1, w[4], 7);
    R32(a2, b2, c2, d2, e2, w[3], 11);
    R31(e1, a1, b1, c1, d1, w[9], 14);
    R32(e2, a2, b2, c2, d2, w[7], 8);
    R31(d1, e1, a1, b1, c1, w[15], 9);
    R32(d2, e2, a2, b2, c2, w[14], 6);
    R31(c1, d1, e1, a1, b1, w[8], 13);
    R32(c2, d2, e2, a2, b2, w[6], 6);
    R31(b1, c1, d1, e1, a1, w[1], 15);
    R32(b2, c2, d2, e2, a2, w[9], 14);
    R31(a1, b1, c1, d1, e1, w[2], 14);
    R32(a2, b2, c2, d2, e2, w[11], 12);
    R31(e1, a1, b1, c1, d1, w[7], 8);
    R32(e2, a2, b2, c2, d2, w[8], 13);
    R31(d1, e1, a1, b1, c1, w[0], 13);
    R32(d2, e2, a2, b2, c2, w[12], 5);
    R31(c1, d1, e1, a1, b1, w[6], 6);
    R32(c2, d2, e2, a2, b2, w[2], 14);
    R31(b1, c1, d1, e1, a1, w[13], 5);
    R32(b2, c2, d2, e2, a2, w[10], 13);
    R31(a1, b1, c1, d1, e1, w[11], 12);
    R32(a2, b2, c2, d2, e2, w[0], 13);
    R31(e1, a1, b1, c1, d1, w[5], 7);
    R32(e2, a2, b2, c2, d2, w[4], 7);
    R31(d1, e1, a1, b1, c1, w[12], 5);
    R32(d2, e2, a2, b2, c2, w[13], 5);

    R41(c1, d1, e1, a1, b1, w[1], 11);
    R42(c2, d2, e2, a2, b2, w[8], 15);
    R41(b1, c1, d1, e1, a1, w[9], 12);
    R42(b2, c2, d2, e2, a2, w[6], 5);
    R41(a1, b1, c1, d1, e1, w[11], 14);
    R42(a2, b2, c2, d2, e2, w[4], 8);
    R41(e1, a1, b1, c1, d1, w[10], 15);
    R42(e2, a2, b2, c2, d2, w[1], 11);
    R41(d1, e1, a1, b1, c1, w[0], 14);
    R42(d2, e2, a2, b2, c2, w[3], 14);
    R41(c1, d1, e1, a1, b1, w[8], 15);
    R42(c2, d2, e2, a2, b2, w[11], 14);
    R41(b1, c1, d1, e1, a1, w[12], 9);
    R42(b2, c2, d2, e2, a2, w[15], 6);
    R41(a1, b1, c1, d1, e1, w[4], 8);
    R42(a2, b2, c2, d2, e2, w[0], 14);
    R41(e1, a1, b1, c1, d1, w[13], 9);
    R42(e2, a2, b2, c2, d2, w[5], 6);
    R41(d1, e1, a1, b1, c1, w[3], 14);
    R42(d2, e2, a2, b2, c2, w[12], 9);
    R41(c1, d1, e1, a1, b1, w[7], 5);
    R42(c2, d2, e2, a2, b2, w[2], 12);
    R41(b1, c1, d1, e1, a1, w[15], 6);
    R42(b2, c2, d2, e2, a2, w[13], 9);
    R41(a1, b1, c1, d1, e1, w[14], 8);
    R42(a2, b2, c2, d2, e2, w[9], 12);
    R41(e1, a1, b1, c1, d1, w[5], 6);
    R42(e2, a2, b2, c2, d2, w[7], 5);
    R41(d1, e1, a1, b1, c1, w[6], 5);
    R42(d2, e2, a2, b2, c2, w[10], 15);
    R41(c1, d1, e1, a1, b1, w[2], 12);
    R42(c2, d2, e2, a2, b2, w[14], 8);

    R51(b1, c1, d1, e1, a1, w[4], 9);
    R52(b2, c2, d2, e2, a2, w[12], 8);
    R51(a1, b1, c1, d1, e1, w[0], 15);
    R52(a2, b2, c2, d2, e2, w[15], 5);
    R51(e1, a1, b1, c1, d1, w[5], 5);
    R52(e2, a2, b2, c2, d2, w[10], 12);
    R51(d1, e1, a1, b1, c1, w[9], 11);
    R52(d2, e2, a2, b2, c2, w[4], 9);
    R51(c1, d1, e1, a1, b1, w[7], 6);
    R52(c2, d2, e2, a2, b2, w[1], 12);
    R51(b1, c1, d1, e1, a1, w[12], 8);
    R52(b2, c2, d2, e2, a2, w[5], 5);
    R51(a1, b1, c1, d1, e1, w[2], 13);
    R52(a2, b2, c2, d2, e2, w[8], 14);
    R51(e1, a1, b1, c1, d1, w[10], 12);
    R52(e2, a2, b2, c2, d2, w[7], 6);
    R51(d1, e1, a1, b1, c1, w[14], 5);
    R52(d2, e2, a2, b2, c2, w[6], 8);
    R51(c1, d1, e1, a1, b1, w[1], 12);
    R52(c2, d2, e2, a2, b2, w[2], 13);
    R51(b1, c1, d1, e1, a1, w[3], 13);
    R52(b2, c2, d2, e2, a2, w[13], 6);
    R51(a1, b1, c1, d1, e1, w[8], 14);
    R52(a2, b2, c2, d2, e2, w[14], 5);
    R51(e1, a1, b1, c1, d1, w[11], 11);
    R52(e2, a2, b2, c2, d2, w[0], 15);
    R51(d1, e1, a1, b1, c1, w[6], 8);
    R52(d2, e2, a2, b2, c2, w[3], 13);
    R51(c1, d1, e1, a1, b1, w[15], 5);
    R52(c2, d2, e2, a2, b2, w[9], 11);
    R51(b1, c1, d1, e1, a1, w[13], 6);
    R52(b2, c2, d2, e2, a2, w[11], 11);
  s[0] = add3(r5_set1<V>((int)R5_K2), c1, d2);
  s[1] = add3(r5_set1<V>((int)R5_K3), d1, e2);
  s[2] = add3(r5_set1<V>((int)R5_K4), e1, a2);
  s[3] = add3(r5_set1<V>((int)R5_K5), a1, b2);
  s[4] = add3(r5_set1<V>((int)R5_K1), b1, c2);
        }

#undef R11
#undef R21
#undef R31
#undef R41
#undef R51
#undef R12
#undef R22
#undef R32
#undef R42
#undef R52
#undef Round
#undef R5_ROL
#undef add4
#undef f1
#undef f2
#undef f3
#undef f4
#undef f5

        // rmd160_batch swaps every input word (RMD_LOAD_SWAP). The same
        // convention is kept here so the two paths are interchangeable.
        alignas(64) constexpr u8 RMD_BSWAP[64] = {
            3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8,  15, 14, 13, 12,
            19, 18, 17, 16, 23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28,
            35, 34, 33, 32, 39, 38, 37, 36, 43, 42, 41, 40, 47, 46, 45, 44,
            51, 50, 49, 48, 55, 54, 53, 52, 59, 58, 57, 56, 63, 62, 61, 60,
        };

        // 5 words x 16 lanes -> 16 hashes of 5 words.
        //
        // THE FINAL SWAP. rmd160_batch ends with RMD_SWAP over the five
        // state words (rmd160s.cpp), and rmd160_final does the same in the
        // scalar path. The C original of this file omits it, so its
        // batch-16 output came out byte-swapped against every other path
        // in the project -- verified directly against lib/: addr33 and
        // addr33_batch agree on 1520f087..., addr33_batch16 returns
        // 87f02015... Its self test does not catch this because the test
        // vector A512_TV_H160 is written in that same swapped order.
        //
        // A search using the AVX-512 path would therefore compare hashes
        // that can never match, and find nothing however long it runs.
        inline void unpack16(u32 r[16][5], const __m512i s[5]) noexcept {
            const __m512i bsw = _mm512_loadu_si512(RMD_BSWAP);
            alignas(64) u32 tmp[5][16];
            for (int i = 0; i < 5; ++i)
                _mm512_store_si512(tmp[i], _mm512_shuffle_epi8(s[i], bsw));
            for (int l = 0; l < 16; ++l)
                for (int i = 0; i < 5; ++i) r[l][i] = tmp[i][l];
        }

    } // namespace

    void rmd160_batch16_soa(u32 r[16][5], const __m512i state8[8]) noexcept {
        const __m512i bsw = _mm512_loadu_si512(RMD_BSWAP);
        __m512i w[16];

#if defined(RMD16_NO_SWAP) && RMD16_NO_SWAP
        for (int i = 0; i < 8; ++i) w[i] = state8[i];
#else
        for (int i = 0; i < 8; ++i) w[i] = _mm512_shuffle_epi8(state8[i], bsw);
#endif
        w[8] = _mm512_set1_epi32(0x00000080); // start of padding
        for (int i = 9; i < 14; ++i) w[i] = _mm512_setzero_si512();
        w[14] = _mm512_set1_epi32(256); // message length in bits
        w[15] = _mm512_setzero_si512();

        __m512i s[5];
        rmd160_core<__m512i>(s, w);
        unpack16(r, s);
    }

    // Two batches of 16 in lockstep: the same core over V2.
    void rmd160_batch32_soa(u32 r[32][5], const __m512i stateA[8], const __m512i stateB[8]) noexcept {
        const __m512i bsw = _mm512_loadu_si512(RMD_BSWAP);
        V2 w[16];
        for (int i = 0; i < 8; ++i) w[i] = {_mm512_shuffle_epi8(stateA[i], bsw), _mm512_shuffle_epi8(stateB[i], bsw)};
        w[8] = r5_set1<V2>(0x00000080);
        for (int i = 9; i < 14; ++i) w[i] = {_mm512_setzero_si512(), _mm512_setzero_si512()};
        w[14] = r5_set1<V2>(256);
        w[15] = {_mm512_setzero_si512(), _mm512_setzero_si512()};

        V2 s[5];
        rmd160_core<V2>(s, w);
        __m512i sa[5], sb[5];
        for (int i = 0; i < 5; ++i) { sa[i] = s[i].x; sb[i] = s[i].y; }
        unpack16(r, sa);
        unpack16(r + 16, sb);
    }

    void rmd160_batch16(u32 r[16][5], const u32 x[16][16]) noexcept {
        const __m512i bsw = _mm512_loadu_si512(RMD_BSWAP);
        const __m512i idx = _mm512_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112,
                                              128, 144, 160, 176, 192, 208, 224, 240);
        __m512i w[16];
        for (int i = 0; i < 16; ++i) {
            const __m512i g =
                _mm512_i32gather_epi32(idx, reinterpret_cast<const int*>(&x[0][i]), 4);
#if defined(RMD16_NO_SWAP) && RMD16_NO_SWAP
            w[i] = g;
#else
            w[i] = _mm512_shuffle_epi8(g, bsw);
#endif
        }

        __m512i s[5];
        rmd160_core<__m512i>(s, w);
        unpack16(r, s);
    }

#undef add3

} // namespace ecloop

#pragma GCC pop_options
#else // !ECLOOP_HAVE_RMD160_AVX512

#include <cstdlib>

namespace ecloop {

    // Built for a target without AVX-512. The symbol exists so the project
    // links, but reaching it means the dispatcher was bypassed.
    void rmd160_batch16(u32 /*r*/[16][5], const u32 /*x*/[16][16]) noexcept {
        __builtin_trap();
    }

} // namespace ecloop

#endif // ECLOOP_HAVE_RMD160_AVX512
