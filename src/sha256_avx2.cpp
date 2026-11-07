// Eight SHA-256 messages at once in AVX2 registers: lane i of every ymm is
// message i. This is the middle rung of the hashing ladder for CPUs that
// have AVX2 but neither AVX-512 (the x16 kernel) nor SHA-NI (sha256.cpp):
// Haswell through Comet Lake on the Intel side, nothing current on AMD.
//
// The round function and the lane layout follow Bitcoin Core's
// crypto/sha256_avx2.cpp (Transform_8way, MIT licence, Copyright (c)
// 2017-present The Bitcoin Core developers); the message flow here is a
// plain one- or two-block SHA-256 with the padding already in the input,
// as sha256_final expects it, rather than Core's double-SHA of 64 bytes.
//
// Built with a function-level target attribute, so it compiles whatever
// -march says and is only called after sha256_avx2_available() said yes.

#include "sha256_avx2.hpp"

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define ECLOOP_HAVE_SHA256_AVX2 1
#else
#define ECLOOP_HAVE_SHA256_AVX2 0
#endif

namespace ecloop {

#if ECLOOP_HAVE_SHA256_AVX2

    namespace {

        constexpr u32 K256[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

#define AVX2_FN __attribute__((target("avx2"), always_inline)) inline

        AVX2_FN __m256i ror(__m256i x, int n) {
            return _mm256_or_si256(_mm256_srli_epi32(x, n), _mm256_slli_epi32(x, 32 - n));
        }
        AVX2_FN __m256i Sigma0(__m256i x) { return _mm256_xor_si256(_mm256_xor_si256(ror(x, 2), ror(x, 13)), ror(x, 22)); }
        AVX2_FN __m256i Sigma1(__m256i x) { return _mm256_xor_si256(_mm256_xor_si256(ror(x, 6), ror(x, 11)), ror(x, 25)); }
        AVX2_FN __m256i sigma0(__m256i x) { return _mm256_xor_si256(_mm256_xor_si256(ror(x, 7), ror(x, 18)), _mm256_srli_epi32(x, 3)); }
        AVX2_FN __m256i sigma1(__m256i x) { return _mm256_xor_si256(_mm256_xor_si256(ror(x, 17), ror(x, 19)), _mm256_srli_epi32(x, 10)); }
        AVX2_FN __m256i Ch(__m256i e, __m256i f, __m256i g) {
            // (e & f) ^ (~e & g) as g ^ (e & (f ^ g)): one op fewer
            return _mm256_xor_si256(g, _mm256_and_si256(e, _mm256_xor_si256(f, g)));
        }
        AVX2_FN __m256i Maj(__m256i a, __m256i b, __m256i c) {
            // (a & b) ^ (a & c) ^ (b & c) as (a & b) | (c & (a | b))
            return _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(c, _mm256_or_si256(a, b)));
        }

        // One round; k already holds K[i] + W[i].
        AVX2_FN void Round(__m256i a, __m256i b, __m256i c, __m256i& d, __m256i e, __m256i f,
                           __m256i g, __m256i& h, __m256i k) {
            const __m256i t1 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(h, Sigma1(e)), Ch(e, f, g)), k);
            const __m256i t2 = _mm256_add_epi32(Sigma0(a), Maj(a, b, c));
            d = _mm256_add_epi32(d, t1);
            h = _mm256_add_epi32(t1, t2);
        }

        // Word `w` (0..15) of block `blk` of all eight messages, big-endian
        // as SHA-256 reads it, one message per lane.
        AVX2_FN __m256i load_word(const u8* const* msg, std::size_t off) {
            auto be = [](const u8* p) -> int {
                return static_cast<int>((u32(p[0]) << 24) | (u32(p[1]) << 16) | (u32(p[2]) << 8) | u32(p[3]));
            };
            return _mm256_set_epi32(be(msg[7] + off), be(msg[6] + off), be(msg[5] + off), be(msg[4] + off),
                                    be(msg[3] + off), be(msg[2] + off), be(msg[1] + off), be(msg[0] + off));
        }

        __attribute__((target("avx2")))
        void transform8(u32 out[8][8], const u8* const msg[8], std::size_t nblocks) {
            __m256i s[8] = {
                _mm256_set1_epi32(0x6a09e667), _mm256_set1_epi32(static_cast<int>(0xbb67ae85)),
                _mm256_set1_epi32(0x3c6ef372), _mm256_set1_epi32(static_cast<int>(0xa54ff53a)),
                _mm256_set1_epi32(0x510e527f), _mm256_set1_epi32(static_cast<int>(0x9b05688c)),
                _mm256_set1_epi32(0x1f83d9ab), _mm256_set1_epi32(0x5be0cd19)};

            for (std::size_t blk = 0; blk < nblocks; ++blk) {
                __m256i W[16];
                for (int i = 0; i < 16; ++i) W[i] = load_word(msg, blk * 64 + 4 * static_cast<std::size_t>(i));

                __m256i a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];

                // Sixteen rounds per iteration with the registers rotating
                // through the eight roles, the schedule updated in place.
#pragma GCC unroll 4
                for (int i = 0; i < 64; i += 16) {
#pragma GCC unroll 16
                    for (int j = 0; j < 16; ++j) {
                        if (i > 0) {
                            const __m256i w = _mm256_add_epi32(
                                _mm256_add_epi32(W[j], sigma0(W[(j + 1) & 15])),
                                _mm256_add_epi32(W[(j + 9) & 15], sigma1(W[(j + 14) & 15])));
                            W[j] = w;
                        }
                        const __m256i k = _mm256_add_epi32(W[j], _mm256_set1_epi32(static_cast<int>(K256[i + j])));
                        switch (j & 7) {
                        case 0: Round(a, b, c, d, e, f, g, h, k); break;
                        case 1: Round(h, a, b, c, d, e, f, g, k); break;
                        case 2: Round(g, h, a, b, c, d, e, f, k); break;
                        case 3: Round(f, g, h, a, b, c, d, e, k); break;
                        case 4: Round(e, f, g, h, a, b, c, d, k); break;
                        case 5: Round(d, e, f, g, h, a, b, c, k); break;
                        case 6: Round(c, d, e, f, g, h, a, b, k); break;
                        default: Round(b, c, d, e, f, g, h, a, k); break;
                        }
                    }
                }
                s[0] = _mm256_add_epi32(s[0], a); s[1] = _mm256_add_epi32(s[1], b);
                s[2] = _mm256_add_epi32(s[2], c); s[3] = _mm256_add_epi32(s[3], d);
                s[4] = _mm256_add_epi32(s[4], e); s[5] = _mm256_add_epi32(s[5], f);
                s[6] = _mm256_add_epi32(s[6], g); s[7] = _mm256_add_epi32(s[7], h);
            }

            // Lane i of s[w] is state word w of message i.
            alignas(32) u32 t[8][8];
            for (int w = 0; w < 8; ++w) _mm256_store_si256(reinterpret_cast<__m256i*>(t[w]), s[w]);
            for (int m = 0; m < 8; ++m)
                for (int w = 0; w < 8; ++w) out[m][w] = t[w][m];
            _mm256_zeroupper();
        }

#undef AVX2_FN

    } // namespace

    bool sha256_avx2_available() noexcept {
        static const bool ok = [] {
            __builtin_cpu_init();
            return __builtin_cpu_supports("avx2") != 0;
        }();
        return ok;
    }

    void sha256_avx2_8x1(u32 out[8][8], const u8 in[8][64]) noexcept {
        const u8* msg[8];
        for (int i = 0; i < 8; ++i) msg[i] = in[i];
        transform8(out, msg, 1);
    }

    void sha256_avx2_8x2(u32 out[8][8], const u8 in[8][128]) noexcept {
        const u8* msg[8];
        for (int i = 0; i < 8; ++i) msg[i] = in[i];
        transform8(out, msg, 2);
    }

#else

    bool sha256_avx2_available() noexcept { return false; }
    void sha256_avx2_8x1(u32[8][8], const u8[8][64]) noexcept {}
    void sha256_avx2_8x2(u32[8][8], const u8[8][128]) noexcept {}

#endif

} // namespace ecloop
