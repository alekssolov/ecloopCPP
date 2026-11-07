#include "rmd160.hpp"

#include "compat.hpp"

namespace ecloop {
    namespace {

        constexpr u8 RL[80] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
            3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
            1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
            4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13,
        };

        constexpr u8 SL[80] = {
            11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
            7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
            11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
            11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
            9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6,
        };

        constexpr u8 RR[80] = {
            5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
            6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
            15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
            8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
            12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11,
        };

        constexpr u8 SR[80] = {
            8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
            9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
            9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
            15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
            8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11,
        };

        ECLOOP_ALWAYS_INLINE u32 f1(u32 x, u32 y, u32 z) noexcept {
            return x ^ y ^ z;
        }

        ECLOOP_ALWAYS_INLINE u32 f2(u32 x, u32 y, u32 z) noexcept {
            return (x & y) | (~x & z);
        }

        ECLOOP_ALWAYS_INLINE u32 f3(u32 x, u32 y, u32 z) noexcept {
            return (x | ~y) ^ z;
        }

        ECLOOP_ALWAYS_INLINE u32 f4(u32 x, u32 y, u32 z) noexcept {
            return (x & z) | (y & ~z);
        }

        ECLOOP_ALWAYS_INLINE u32 f5(u32 x, u32 y, u32 z) noexcept {
            return x ^ (y | ~z);
        }

        template <typename Fn>
        ECLOOP_ALWAYS_INLINE void step(u32& a,
            u32& b,
            u32& c,
            u32& d,
            u32& e,
            Fn fn,
            u32 word,
            u8 rot,
            u32 k) noexcept {
            const u32 alpha = detail::rotl32(a + fn(b, c, d) + word + k, rot) + e;
            const u32 beta = detail::rotl32(c, 10);

            a = e;
            e = d;
            d = beta;
            c = b;
            b = alpha;
        }

    } // namespace

    void rmd160_final(u32 state[5], const u32 x[16]) {
        constexpr u32 H0 = 0x67452301u;
        constexpr u32 H1 = 0xefcdab89u;
        constexpr u32 H2 = 0x98badcfeu;
        constexpr u32 H3 = 0x10325476u;
        constexpr u32 H4 = 0xc3d2e1f0u;

        u32 a1 = H0;
        u32 b1 = H1;
        u32 c1 = H2;
        u32 d1 = H3;
        u32 e1 = H4;

        u32 a2 = H0;
        u32 b2 = H1;
        u32 c2 = H2;
        u32 d2 = H3;
        u32 e2 = H4;

        int i = 0;

        for (; i < 16; ++i) {
            step(a1, b1, c1, d1, e1, f1, x[RL[i]], SL[i], 0x00000000u);
            step(a2, b2, c2, d2, e2, f5, x[RR[i]], SR[i], 0x50a28be6u);
        }

        for (; i < 32; ++i) {
            step(a1, b1, c1, d1, e1, f2, x[RL[i]], SL[i], 0x5a827999u);
            step(a2, b2, c2, d2, e2, f4, x[RR[i]], SR[i], 0x5c4dd124u);
        }

        for (; i < 48; ++i) {
            step(a1, b1, c1, d1, e1, f3, x[RL[i]], SL[i], 0x6ed9eba1u);
            step(a2, b2, c2, d2, e2, f3, x[RR[i]], SR[i], 0x6d703ef3u);
        }

        for (; i < 64; ++i) {
            step(a1, b1, c1, d1, e1, f4, x[RL[i]], SL[i], 0x8f1bbcdcu);
            step(a2, b2, c2, d2, e2, f2, x[RR[i]], SR[i], 0x7a6d76e9u);
        }

        for (; i < 80; ++i) {
            step(a1, b1, c1, d1, e1, f5, x[RL[i]], SL[i], 0xa953fd4eu);
            step(a2, b2, c2, d2, e2, f1, x[RR[i]], SR[i], 0x00000000u);
        }

        state[0] = H1 + c1 + d2;
        state[1] = H2 + d1 + e2;
        state[2] = H3 + e1 + a2;
        state[3] = H4 + a1 + b2;
        state[4] = H0 + b1 + c2;

        for (int j = 0; j < 5; ++j) { state[j] = detail::bswap32(state[j]); }
    }

} // namespace ecloop
