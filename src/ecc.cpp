// Copyright (c) vladkens
// https://github.com/vladkens/ecloop
// Licensed under the MIT License.

#include "ecc.hpp"
#include "shm.hpp"
#include <limits>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <vector>
#include <memory>   // std::unique_ptr

#include "compat.hpp"

// ── bring helper names into scope ───────────────────────────────────────────
using ecloop::detail::addc64;
using ecloop::detail::subc64;


// ─────────────────────────────────────────────────────────────────────────────
// s2n-bignum (AWS, formally verified). PLAIN arithmetic, not Montgomery.
//
// jdouble/jadd/jmixadd operate on a 96-byte point laid out as three
// consecutive 32-byte field elements, which is exactly what pe is -- the
// static_assert below keeps that true. Classic Jacobian: x = X/Z^2,
// y = Y/Z^3. Note the original ecloop used HOMOGENEOUS projective
// coordinates (x = X/Z) despite the ec_jacobi_* names, so the reduction
// functions change too.
// ─────────────────────────────────────────────────────────────────────────────
#include "s2n_dispatch.hpp"
#include "cpu.hpp"

namespace ecloop {
    static_assert(sizeof(pe) == 96, "s2n point routines require pe to be 3 x 32 bytes");
    static_assert(offsetof(pe, x) == 0 && offsetof(pe, y) == 32 && offsetof(pe, z) == 64,
                  "s2n point routines require pe fields in X,Y,Z order with no padding");
}

namespace ecloop {

    // ============================================================================
    // MARK: umul128
    // ============================================================================

    static inline u64 umul128(const u64 a, const u64 b, u64* hi) noexcept {
        // https://stackoverflow.com/a/50958815
        u128 t = static_cast<u128>(a) * b;
        *hi = static_cast<u64>(t >> 64);
        return static_cast<u64>(t);
    }

    // ============================================================================
    // MARK: Constants
    // ============================================================================

    const fe FE_ZERO = { 0, 0, 0, 0 };

    // Secp256k1 prime field (2^256 - 2^32 - 977) and order
    const fe FE_P = { 0xfffffffefffffc2fULL, 0xffffffffffffffffULL,
                     0xffffffffffffffffULL, 0xffffffffffffffffULL };
    const fe FE_N = { 0xbfd25e8cd0364141ULL, 0xbaaedce6af48a03bULL,
                     0xfffffffffffffffeULL, 0xffffffffffffffffULL };

    // endomorphism constants (alpha, alpha^2, beta, beta^2)
    const fe A1 = { 0xdf02967c1b23bd72ULL, 0x122e22ea20816678ULL,
                   0xa5261c028812645aULL, 0x5363ad4cc05c30e0ULL };
    const fe A2 = { 0xe0cfc810b51283ceULL, 0xa880b9fc8ec739c2ULL,
                   0x5ad9e3fd77ed9ba4ULL, 0xac9c52b33fa3cf1fULL };
    const fe B1 = { 0xc1396c28719501eeULL, 0x9cf0497512f58995ULL,
                   0x6e64479eac3434e9ULL, 0x7ae96a2b657c0710ULL };
    const fe B2 = { 0x3ec693d68e6afa40ULL, 0x630fb68aed0a766aULL,
                   0x919bb86153cbcb16ULL, 0x851695d49a83f8efULL };

    const pe G1 = {
        {0x59f2815b16f81798ULL, 0x029bfcdb2dce28d9ULL,
         0x55a06295ce870b07ULL, 0x79be667ef9dcbbacULL},
        {0x9c47d08ffb10d4b8ULL, 0xfd17b448a6855419ULL,
         0x5da4fbfc0e1108a8ULL, 0x483ada7726a3c465ULL},
        {0x1ULL, 0x0ULL, 0x0ULL, 0x0ULL},
    };

    const pe G2 = {
        {0xabac09b95c709ee5ULL, 0x5c778e4b8cef3ca7ULL,
         0x3045406e95c07cd8ULL, 0xc6047f9441ed7d6dULL},
        {0x236431a950cfe52aULL, 0xf7f632653266d0e1ULL,
         0xa3c58419466ceaeeULL, 0x1ae168fea63dc339ULL},
        {0x1ULL, 0x0ULL, 0x0ULL, 0x0ULL},
    };

    // ============================================================================
    // MARK: Field Element helpers
    // ============================================================================

    void fe_print(const char* label, const fe a) {
        printf("%s: %016llx %016llx %016llx %016llx\n",
            label, a[3], a[2], a[1], a[0]);
    }

    bool fe_iszero(const fe r) {
        return r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 0;
    }

    void fe_clone(fe r, const fe a) {
        memcpy(r, a, sizeof(fe));
    }

    void fe_set64(fe r, const u64 a) {
        memset(r, 0, sizeof(fe));
        r[0] = a;
    }

    std::size_t fe_bitlen(const fe a) {
        for (int i = 3; i >= 0; --i) {
            if (a[i]) {
                return static_cast<std::size_t>(64 * i) +
                    static_cast<std::size_t>(64 - __builtin_clzll(a[i]));
            }
        }
        return 0;
    }

    void fe_add64(fe r, const u64 a) {
        u64 c = 0;
        r[0] = addc64(r[0], a, 0, &c);
        r[1] = addc64(r[1], 0ULL, c, &c);
        r[2] = addc64(r[2], 0ULL, c, &c);
        r[3] = addc64(r[3], 0ULL, c, &c);
    }

    int fe_cmp64(const fe a, const u64 b) {
        if (a[3] != 0 || a[2] != 0 || a[1] != 0) return 1;
        if (a[0] != b) return a[0] > b ? 1 : -1;
        return 0;
    }

    int fe_cmp(const fe a, const fe b) {
        if (a[3] != b[3]) return a[3] > b[3] ? 1 : -1;
        if (a[2] != b[2]) return a[2] > b[2] ? 1 : -1;
        if (a[1] != b[1]) return a[1] > b[1] ? 1 : -1;
        if (a[0] != b[0]) return a[0] > b[0] ? 1 : -1;
        return 0;
    }

    void fe_from_hex(fe r, const char* hex) {
        fe_set64(r, 0);

        int cnt = 0;
        int len = static_cast<int>(strlen(hex));
        while (len-- > 0) {
            u64 v = static_cast<u64>(tolower(static_cast<unsigned char>(hex[len])));
            if (v >= '0' && v <= '9')      v = v - '0';
            else if (v >= 'a' && v <= 'f') v = v - 'a' + 10;
            else continue;

            r[cnt / 16] = (v << (cnt * 4 % 64)) | r[cnt / 16];
            cnt += 1;
        }
    }

    void fe_shiftl(fe r, const u8 n) {
        if (n == 0) return;

        const u8 s = n / 64;
        const u8 rem = n % 64;

        for (int i = 3; i >= 0; --i) r[i] = (i >= s) ? r[i - s] : 0;
        if (rem == 0) return;

        u128 carry = 0;
        for (int i = 0; i < 4; ++i) {
            u128 val = static_cast<u128>(r[i]) << rem;
            r[i] = static_cast<u64>(val | carry);
            carry = val >> 64;
        }
    }

    void fe_shiftr64(fe r, const u8 n) {
        assert(n < 64);
        r[0] = (r[0] >> n) | (r[1] << (64 - n));
        r[1] = (r[1] >> n) | (r[2] << (64 - n));
        r[2] = (r[2] >> n) | (r[3] << (64 - n));
        r[3] = r[3] >> n;
    }

    // ============================================================================
    // MARK: 320-bit helpers
    // ============================================================================

    static void fe_mul_scalar(fe320 r, const fe a, const u64 b) {
        u64 h1, h2, c = 0;
        r[0] = umul128(a[0], b, &h1);
        r[1] = addc64(umul128(a[1], b, &h2), h1, c, &c);
        r[2] = addc64(umul128(a[2], b, &h1), h2, c, &c);
        r[3] = addc64(umul128(a[3], b, &h2), h1, c, &c);
        r[4] = addc64(0ULL, h2, c, &c);
    }

    static u64 fe320_addc(fe320 r, const fe320 a, const fe320 b) {
        u64 c = 0;
        r[0] = addc64(a[0], b[0], c, &c);
        r[1] = addc64(a[1], b[1], c, &c);
        r[2] = addc64(a[2], b[2], c, &c);
        r[3] = addc64(a[3], b[3], c, &c);
        r[4] = addc64(a[4], b[4], c, &c);
        return c;
    }

    static u64 fe320_subc(fe320 r, const fe320 a, const fe320 b) {
        u64 c = 0;
        r[0] = subc64(a[0], b[0], c, &c);
        r[1] = subc64(a[1], b[1], c, &c);
        r[2] = subc64(a[2], b[2], c, &c);
        r[3] = subc64(a[3], b[3], c, &c);
        r[4] = subc64(a[4], b[4], c, &c);
        return c;
    }

    static void fe320_add_shift(fe320 r, const fe320 a, const fe320 b, u64 ch) {
        u64 c = 0;
        addc64(a[0], b[0], c, &c); // keep only carry
        r[0] = addc64(a[1], b[1], c, &c);
        r[1] = addc64(a[2], b[2], c, &c);
        r[2] = addc64(a[3], b[3], c, &c);
        r[3] = addc64(a[4], b[4], c, &c);
        r[4] = c + ch;
    }

    // ============================================================================
    // MARK: Modulo N arithmetic
    // ============================================================================

    // clang-format off
    static const fe320 _NN = { 0xbfd25e8cd0364141ULL, 0xbaaedce6af48a03bULL,
                              0xfffffffffffffffeULL, 0xffffffffffffffffULL, 0x0ULL };
    static const fe320 _R2 = { 0x896cf21467d7d140ULL, 0x741496c20e7cf878ULL,
                              0xe697f5e45bcd07c6ULL, 0x9d671cd581c69bc5ULL, 0x0ULL };
    static const u64 _MM64o = 0x4b0dff665588b13fULL; // 64-bit lsb negative inverse of secp256k1 order
    // clang-format on

    void fe_modn_neg(fe r, const fe a) {
        u64 c = 0;
        r[0] = subc64(FE_N[0], a[0], c, &c);
        r[1] = subc64(FE_N[1], a[1], c, &c);
        r[2] = subc64(FE_N[2], a[2], c, &c);
        r[3] = subc64(FE_N[3], a[3], c, &c);
    }

    void fe_modn_add(fe r, const fe a, const fe b) {
        u64 c = 0;
        r[0] = addc64(a[0], b[0], c, &c);
        r[1] = addc64(a[1], b[1], c, &c);
        r[2] = addc64(a[2], b[2], c, &c);
        r[3] = addc64(a[3], b[3], c, &c);

        if (c) {
            r[0] = subc64(r[0], FE_N[0], 0, &c);
            r[1] = subc64(r[1], FE_N[1], c, &c);
            r[2] = subc64(r[2], FE_N[2], c, &c);
            r[3] = subc64(r[3], FE_N[3], c, &c);
        }
    }

    void fe_modn_sub(fe r, const fe a, const fe b) {
        u64 c = 0;
        r[0] = subc64(a[0], b[0], c, &c);
        r[1] = subc64(a[1], b[1], c, &c);
        r[2] = subc64(a[2], b[2], c, &c);
        r[3] = subc64(a[3], b[3], c, &c);

        if (c) {
            r[0] = addc64(r[0], FE_N[0], 0, &c);
            r[1] = addc64(r[1], FE_N[1], c, &c);
            r[2] = addc64(r[2], FE_N[2], c, &c);
            r[3] = addc64(r[3], FE_N[3], c, &c);
        }
    }

    // https://github.com/albertobsd/keyhunt/blob/main/secp256k1/IntMod.cpp#L1109
    void fe_modn_mul(fe __restrict__ r, const fe __restrict__ a, const fe __restrict__ b) {
        fe320 t = { 0 }, pr = { 0 }, p = { 0 }, rr = { 0 };
        u64 ml, c;

        fe_mul_scalar(pr, a, b[0]);
        ml = pr[0] * _MM64o;
        fe_mul_scalar(p, _NN, ml);
        c = fe320_addc(pr, pr, p);
        memcpy(t, pr + 1, 32);
        t[4] = c;

        for (int i = 1; i < 4; ++i) {
            fe_mul_scalar(pr, a, b[i]);
            ml = (pr[0] + t[0]) * _MM64o;
            fe_mul_scalar(p, _NN, ml);
            c = fe320_addc(pr, pr, p);
            fe320_add_shift(t, t, pr, c);
        }

        fe320_subc(p, t, _NN);
        static_cast<int64_t>(p[4]) >= 0 ? memcpy(rr, p, 40) : memcpy(rr, t, 40);

        // normalize
        fe_mul_scalar(pr, _R2, rr[0]);
        ml = pr[0] * _MM64o;
        fe_mul_scalar(p, _NN, ml);
        c = fe320_addc(pr, pr, p);
        memcpy(t, pr + 1, 32);
        t[4] = c;

        for (int i = 1; i < 4; ++i) {
            fe_mul_scalar(pr, _R2, rr[i]);
            ml = (pr[0] + t[0]) * _MM64o;
            fe_mul_scalar(p, _NN, ml);
            c = fe320_addc(pr, pr, p);
            fe320_add_shift(t, t, pr, c);
        }

        fe320_subc(p, t, _NN);
        static_cast<int64_t>(p[4]) >= 0 ? memcpy(rr, p, 40) : memcpy(rr, t, 40);

        fe_clone(r, rr);
    }

    void fe_modn_add_stride(fe r, const fe base, const fe stride, const u64 offset) {
        fe t;
        fe_set64(t, offset);
        fe_modn_mul(t, t, stride);
        fe_modn_add(r, t, base);
    }

    void fe_modn_from_hex(fe r, const char* hex) {
        fe_from_hex(r, hex);
        if (fe_cmp(r, FE_N) >= 0) fe_modn_sub(r, r, FE_N);
    }

    // ============================================================================
    // MARK: Modulo P arithmetic
    // ============================================================================

    void fe_modp_neg(fe r, const fe a) { bignum_neg_p256k1(r, a); }

    void fe_modp_sub(fe r, const fe a, const fe b) { bignum_sub_p256k1(r, a, b); }

    void fe_modp_add(fe r, const fe a, const fe b) { bignum_add_p256k1(r, a, b); }

    void fe_modp_mul(fe __restrict__ r, const fe __restrict__ a, const fe __restrict__ b) { bignum_mul_p256k1(r, a, b); }

    void fe_modp_sqr(fe __restrict__ r, const fe __restrict__ a) { bignum_sqr_p256k1(r, a); }

    // ── inversion ────────────────────────────────────────────────────────────────

    void _fe_modp_inv_binpow(fe r, const fe a) {
        // a^(P-2) = a^-1 (mod P)
        fe q = { 1 }, p, t;
        fe_clone(p, FE_P);
        fe_clone(t, a);
        p[0] -= 2;

        while (p[0] || p[1] || p[2] || p[3]) {
            if ((a[0] & 1) != 0) fe_modp_mul(q, q, t);
            fe_modp_sqr(t, t);
            fe_shiftr64(p, 1);
        }

        fe_clone(r, q);
    }

    void _fe_modp_inv_addchn(fe r, const fe a) {
        // https://briansmith.org/ecc-inversion-addition-chains-01#secp256k1_field_inversion
        fe x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t1;

        fe_modp_sqr(x2, a);
        fe_modp_mul(x2, x2, a);

        fe_clone(x3, x2);
        fe_modp_sqr(x3, x2);
        fe_modp_mul(x3, x3, a);

        fe_clone(x6, x3);
        for (int j = 0; j < 3; ++j) fe_modp_sqr(x6, x6);
        fe_modp_mul(x6, x6, x3);

        fe_clone(x9, x6);
        for (int j = 0; j < 3; ++j) fe_modp_sqr(x9, x9);
        fe_modp_mul(x9, x9, x3);

        fe_clone(x11, x9);
        for (int j = 0; j < 2; ++j) fe_modp_sqr(x11, x11);
        fe_modp_mul(x11, x11, x2);

        fe_clone(x22, x11);
        for (int j = 0; j < 11; ++j) fe_modp_sqr(x22, x22);
        fe_modp_mul(x22, x22, x11);

        fe_clone(x44, x22);
        for (int j = 0; j < 22; ++j) fe_modp_sqr(x44, x44);
        fe_modp_mul(x44, x44, x22);

        fe_clone(x88, x44);
        for (int j = 0; j < 44; ++j) fe_modp_sqr(x88, x88);
        fe_modp_mul(x88, x88, x44);

        fe_clone(x176, x88);
        for (int j = 0; j < 88; ++j) fe_modp_sqr(x176, x176);
        fe_modp_mul(x176, x176, x88);

        fe_clone(x220, x176);
        for (int j = 0; j < 44; ++j) fe_modp_sqr(x220, x220);
        fe_modp_mul(x220, x220, x44);

        fe_clone(x223, x220);
        for (int j = 0; j < 3; ++j) fe_modp_sqr(x223, x223);
        fe_modp_mul(x223, x223, x3);

        fe_clone(t1, x223);
        for (int j = 0; j < 23; ++j) fe_modp_sqr(t1, t1);
        fe_modp_mul(t1, t1, x22);
        for (int j = 0; j < 5; ++j) fe_modp_sqr(t1, t1);
        fe_modp_mul(t1, t1, a);
        for (int j = 0; j < 3; ++j) fe_modp_sqr(t1, t1);
        fe_modp_mul(t1, t1, x2);
        for (int j = 0; j < 2; ++j) fe_modp_sqr(t1, t1);
        fe_modp_mul(r, t1, a);
    }

    namespace {
        // === SafeGCD-инверсия по модулю secp256k1 p, портировано на чистый C99 из
        // === UltrafastSecp256k1/src/cpu/src/field.cpp (fe_inverse_safegcd_impl),
        // === который сам является портом bitcoin-core/secp256k1 modinv64
        // === (ссылки в оригинале: secp256k1_modinv64_update_de_62,
        // === secp256k1_modinv64_var, secp256k1_modinv64_normalize_62).
        // === Только синтаксис изменён под C (ссылки -> указатели, auto/constexpr
        // === убраны) - алгоритм и арифметика побитово идентичны C++-версии,
        // === полностью перепроверено заново после портирования.

        struct SafeGCD_Int { std::int64_t v[5]; };
        struct SafeGCD_Trans { std::int64_t u, v, q, r; };

        const SafeGCD_Int SAFEGCD_P = {{-(std::int64_t)0x1000003D1LL, 0, 0, 0, 256}};
        const std::uint64_t SAFEGCD_P_INV62 = 0x27C7F6E22DDACACFULL;

        ECLOOP_ALWAYS_INLINE int safegcd_ctz64(std::uint64_t x) { return __builtin_ctzll(x); }

        // u64[4] LE -> SafeGCD (5x62 signed)
        SafeGCD_Int fe_to_s62(const u64 d[4]) {
          const std::uint64_t M = (1ULL << 62) - 1;
          SafeGCD_Int r;
          r.v[0] = (std::int64_t)(d[0] & M);
          r.v[1] = (std::int64_t)(((d[0] >> 62) | (d[1] << 2)) & M);
          r.v[2] = (std::int64_t)(((d[1] >> 60) | (d[2] << 4)) & M);
          r.v[3] = (std::int64_t)(((d[2] >> 58) | (d[3] << 6)) & M);
          r.v[4] = (std::int64_t)(d[3] >> 56);
          return r;
        }

        void s62_to_fe(u64 out[4], const SafeGCD_Int *s) {
          out[0] = (std::uint64_t)s->v[0] | ((std::uint64_t)s->v[1] << 62);
          out[1] = ((std::uint64_t)s->v[1] >> 2) | ((std::uint64_t)s->v[2] << 60);
          out[2] = ((std::uint64_t)s->v[2] >> 4) | ((std::uint64_t)s->v[3] << 58);
          out[3] = ((std::uint64_t)s->v[3] >> 6) | ((std::uint64_t)s->v[4] << 56);
        }

        ECLOOP_ALWAYS_INLINE std::int64_t
        safegcd_divsteps_62_var(std::int64_t delta, std::uint64_t f0, std::uint64_t g0, SafeGCD_Trans *t) {
          std::uint64_t u = 1, v = 0, q = 0, r = 1;
          std::uint64_t f = f0, g = g0;
          int i = 62;
          for (;;) {
            int zeros = safegcd_ctz64(g | ((std::uint64_t)1 << i));
            g >>= zeros;
            u <<= zeros;
            v <<= zeros;
            delta += zeros;
            i -= zeros;
            if (i == 0) break;
            if (delta > 0) {
              std::uint64_t tmp;
              delta = -delta;
              tmp = f;
              f = g;
              g = (std::uint64_t)(-(std::int64_t)tmp);
              tmp = u;
              u = q;
              q = (std::uint64_t)(-(std::int64_t)tmp);
              tmp = v;
              v = r;
              r = (std::uint64_t)(-(std::int64_t)tmp);
            }
            g += f;
            q += u;
            r += v;
            g >>= 1;
            u <<= 1;
            v <<= 1;
            ++delta;
            --i;
            if (i == 0) break;
          }
          t->u = (std::int64_t)u;
          t->v = (std::int64_t)v;
          t->q = (std::int64_t)q;
          t->r = (std::int64_t)r;
          return delta;
        }

        ECLOOP_ALWAYS_INLINE void
        safegcd_update_fg(SafeGCD_Int *f, SafeGCD_Int *g, const SafeGCD_Trans *t, int len) {
          const std::int64_t M62 = (std::int64_t)((std::uint64_t)(-1) >> 2);
          __int128 cf, cg;
          int i;
          cf = (__int128)t->u * f->v[0] + (__int128)t->v * g->v[0];
          cg = (__int128)t->q * f->v[0] + (__int128)t->r * g->v[0];
          cf >>= 62;
          cg >>= 62;
          for (i = 1; i < len; ++i) {
            cf += (__int128)t->u * f->v[i] + (__int128)t->v * g->v[i];
            cg += (__int128)t->q * f->v[i] + (__int128)t->r * g->v[i];
            f->v[i - 1] = (std::int64_t)cf & M62;
            g->v[i - 1] = (std::int64_t)cg & M62;
            cf >>= 62;
            cg >>= 62;
          }
          f->v[len - 1] = (std::int64_t)cf;
          g->v[len - 1] = (std::int64_t)cg;
          for (i = len; i < 5; ++i) {
            f->v[i] = 0;
            g->v[i] = 0;
          }
        }

        ECLOOP_ALWAYS_INLINE void safegcd_update_de(SafeGCD_Int *d, SafeGCD_Int *e,
                                                                              const SafeGCD_Trans *t) {
          const std::uint64_t M62 = UINT64_MAX >> 2;
          const std::int64_t d0 = d->v[0], d1 = d->v[1], d2 = d->v[2], d3 = d->v[3], d4 = d->v[4];
          const std::int64_t e0 = e->v[0], e1 = e->v[1], e2 = e->v[2], e3 = e->v[3], e4 = e->v[4];
          const std::int64_t u = t->u, v = t->v, q = t->q, r = t->r;
          std::int64_t md, me, sd, se;
          __int128 cd, ce;

          sd = d4 >> 63;
          se = e4 >> 63;
          md = (u & sd) + (v & se);
          me = (q & sd) + (r & se);

          cd = (__int128)u * d0 + (__int128)v * e0;
          ce = (__int128)q * d0 + (__int128)r * e0;

          md -= (std::int64_t)((SAFEGCD_P_INV62 * (std::uint64_t)cd + (std::uint64_t)md) & M62);
          me -= (std::int64_t)((SAFEGCD_P_INV62 * (std::uint64_t)ce + (std::uint64_t)me) & M62);

          cd += (__int128)SAFEGCD_P.v[0] * md;
          ce += (__int128)SAFEGCD_P.v[0] * me;
          cd >>= 62;
          ce >>= 62;

          cd += (__int128)u * d1 + (__int128)v * e1;
          ce += (__int128)q * d1 + (__int128)r * e1;
          d->v[0] = (std::int64_t)((std::uint64_t)cd & M62);
          cd >>= 62;
          e->v[0] = (std::int64_t)((std::uint64_t)ce & M62);
          ce >>= 62;

          cd += (__int128)u * d2 + (__int128)v * e2;
          ce += (__int128)q * d2 + (__int128)r * e2;
          d->v[1] = (std::int64_t)((std::uint64_t)cd & M62);
          cd >>= 62;
          e->v[1] = (std::int64_t)((std::uint64_t)ce & M62);
          ce >>= 62;

          cd += (__int128)u * d3 + (__int128)v * e3;
          ce += (__int128)q * d3 + (__int128)r * e3;
          d->v[2] = (std::int64_t)((std::uint64_t)cd & M62);
          cd >>= 62;
          e->v[2] = (std::int64_t)((std::uint64_t)ce & M62);
          ce >>= 62;

          cd += (__int128)u * d4 + (__int128)v * e4 + (__int128)SAFEGCD_P.v[4] * md;
          ce += (__int128)q * d4 + (__int128)r * e4 + (__int128)SAFEGCD_P.v[4] * me;
          d->v[3] = (std::int64_t)((std::uint64_t)cd & M62);
          cd >>= 62;
          e->v[3] = (std::int64_t)((std::uint64_t)ce & M62);
          ce >>= 62;

          d->v[4] = (std::int64_t)cd;
          e->v[4] = (std::int64_t)ce;
        }

        ECLOOP_ALWAYS_INLINE void safegcd_reduce_len(int *len, SafeGCD_Int *f,
                                                                               SafeGCD_Int *g) {
          std::int64_t fn = f->v[*len - 1];
          std::int64_t gn = g->v[*len - 1];
          std::int64_t cond = ((std::int64_t)(*len) - 2) >> 63;
          cond |= fn ^ (fn >> 63);
          cond |= gn ^ (gn >> 63);
          if (cond == 0) {
            f->v[*len - 2] |= (std::uint64_t)fn << 62;
            g->v[*len - 2] |= (std::uint64_t)gn << 62;
            --(*len);
          }
        }

        ECLOOP_ALWAYS_INLINE void safegcd_normalize(SafeGCD_Int *r, std::int64_t f_sign) {
          const std::int64_t M62 = (std::int64_t)(UINT64_MAX >> 2);
          std::int64_t r0 = r->v[0], r1 = r->v[1], r2 = r->v[2], r3 = r->v[3], r4 = r->v[4];
          std::int64_t cond_add, cond_negate;

          cond_add = r4 >> 63;
          r0 += SAFEGCD_P.v[0] & cond_add;
          r4 += SAFEGCD_P.v[4] & cond_add;

          cond_negate = f_sign >> 63;
          r0 = (r0 ^ cond_negate) - cond_negate;
          r1 = (r1 ^ cond_negate) - cond_negate;
          r2 = (r2 ^ cond_negate) - cond_negate;
          r3 = (r3 ^ cond_negate) - cond_negate;
          r4 = (r4 ^ cond_negate) - cond_negate;

          r1 += r0 >> 62;
          r0 &= M62;
          r2 += r1 >> 62;
          r1 &= M62;
          r3 += r2 >> 62;
          r2 &= M62;
          r4 += r3 >> 62;
          r3 &= M62;

          cond_add = r4 >> 63;
          r0 += SAFEGCD_P.v[0] & cond_add;
          r4 += SAFEGCD_P.v[4] & cond_add;

          r1 += r0 >> 62;
          r0 &= M62;
          r2 += r1 >> 62;
          r1 &= M62;
          r3 += r2 >> 62;
          r2 &= M62;
          r4 += r3 >> 62;
          r3 &= M62;

          r->v[0] = r0;
          r->v[1] = r1;
          r->v[2] = r2;
          r->v[3] = r3;
          r->v[4] = r4;
        }

        // -- главная точка входа: u64[4] -> u64[4], инверсия по модулю secp256k1 p --
        void fe_inverse_safegcd_impl(u64 out[4], const u64 x[4]) noexcept {
          SafeGCD_Int d = {{0, 0, 0, 0, 0}};
          SafeGCD_Int e = {{1, 0, 0, 0, 0}};
          SafeGCD_Int f = SAFEGCD_P;
          SafeGCD_Int g = fe_to_s62(x);
          std::int64_t delta = 1;
          int len = 5;
          int i;

          for (i = 0; i < 12; ++i) {
            SafeGCD_Trans t;
            delta = safegcd_divsteps_62_var(delta, (std::uint64_t)f.v[0], (std::uint64_t)g.v[0], &t);
            safegcd_update_de(&d, &e, &t);
            safegcd_update_fg(&f, &g, &t, len);
            {
              std::int64_t cond = 0;
              int j;
              for (j = 0; j < len; ++j) cond |= g.v[j];
              if (cond == 0) break;
            }
            if (i < 11) safegcd_reduce_len(&len, &f, &g);
          }

          safegcd_normalize(&d, f.v[len - 1]);
          s62_to_fe(out, &d);
        }

    } // namespace

    void fe_inverse_safegcd(u64 out[4], const u64 x[4]) noexcept {
        fe_inverse_safegcd_impl(out, x);
    }

    void fe_modp_inv(fe r, const fe a) { fe_inverse_safegcd(r, a); }

    // ── group inversion (RAII) ───────────────────────────────────────────────────

    void fe_modp_grpinv(fe r[], u32 n) {
        if (n == 0) return;   // nothing to invert; also what the fortified memcpy warned about
        // RAII: unique_ptr replaces malloc/free
        struct FreeDeleter { void operator()(void* p) const noexcept { std::free(p); } };
        std::unique_ptr<fe[], FreeDeleter> zs(static_cast<fe*>(std::malloc(static_cast<std::size_t>(n) * sizeof(fe))));
        
        // Guard against r[i] == 0. Without it a single zero wipes out the
        // WHOLE group: the running product zs[] becomes 0 from that index on,
        // inverse(0) == 0, and unrolling then corrupts every result, not just
        // the degenerate one. r[i] == 0 means equal or opposite points when
        // subtracting x coordinates (dx == 0) -- rare, but real.
        std::vector<u8> is_zero(n, 0);
        fe one;
        fe_set64(one, 1);
        for (u32 i = 0; i < n; ++i) {
            if (fe_iszero(r[i])) {
                is_zero[i] = 1;
                fe_clone(r[i], one); // safe stand-in for 0
            }
        }

        fe_clone(zs[0], r[0]);
        for (u32 i = 1; i < n; ++i) fe_modp_mul(zs[i], zs[i - 1], r[i]);

        fe t1, t2;
        fe_clone(t1, zs[n - 1]);
        fe_modp_inv(t1, t1);

        for (u32 i = n - 1; i > 0; --i) {
            fe_modp_mul(t2, t1, zs[i - 1]);
            fe_modp_mul(t1, r[i], t1);
            fe_clone(r[i], t2);
        }

        fe_clone(r[0], t1);

        // Put an exact 0 back where the input was 0 -- consistent with
        // fe_modp_inv(0) == 0 for the single-element case. Callers can test
        // fe_iszero(r[i]) and skip that key.
        for (u32 i = 0; i < n; ++i) {
            if (is_zero[i]) fe_set64(r[i], 0);
        }
    }

    // ============================================================================
    // MARK: EC Point helpers
    // ============================================================================


    // Wrapper so std::vector can hold a raw fe (a C array cannot be a vector
    // element type). Layout is identical, hence the reinterpret_cast below.
    struct FeStore { u64 v[4]; };
    static_assert(sizeof(FeStore) == sizeof(fe), "FeStore must alias fe exactly");

    void pe_clone(pe* r, const pe* a) {
        memcpy(r, a, sizeof(pe));
    }

    // ============================================================================
    // MARK: Affine coordinates
    // ============================================================================

    void ec_affine_dbl(pe* r, const pe* p) {
        fe t1, t2, t3;
        fe_modp_sqr(t1, p->x);       // x^2
        fe_modp_add(t2, t1, t1);     // 2*x^2
        fe_modp_add(t2, t2, t1);     // 3*x^2
        fe_modp_add(t1, p->y, p->y); // 2*y
        fe_modp_inv(t1, t1);
        fe_modp_mul(t1, t2, t1);     // λ = (3*x^2)/(2*y)
        fe_modp_sqr(t3, t1);         // λ^2
        fe_modp_sub(t3, t3, p->x);   // λ^2 - x1
        fe_modp_sub(t3, t3, p->x);   // λ^2 - 2*x1
        fe_modp_sub(t2, p->x, t3);   // x1 - x
        fe_modp_mul(t2, t1, t2);     // λ*(x1-x)
        fe_modp_sub(r->y, t2, p->y); // λ*(x1-x) - y1
        fe_clone(r->x, t3);
    }

    void ec_affine_add(pe* r, const pe* p, const pe* q) {
        fe t1, t2, t3, t4;
        fe_modp_sub(t1, p->y, q->y); // y1-y2
        fe_modp_sub(t2, p->x, q->x); // x1-x2
        fe_modp_inv(t2, t2);
        fe_modp_mul(t1, t1, t2);     // λ
        fe_modp_sqr(t3, t1);         // λ^2
        fe_modp_sub(t3, t3, p->x);   // λ^2 - x1
        fe_modp_sub(t3, t3, q->x);   // λ^2 - x1 - x2
        fe_modp_sub(t4, p->x, t3);   // x1 - x
        fe_modp_mul(t4, t1, t4);     // λ*(x1-x)
        fe_modp_sub(r->y, t4, p->y); // λ*(x1-x) - y1
        fe_clone(r->x, t3);
    }

    // ============================================================================
    // MARK: Standard Projective (v1)
    // ============================================================================

    void _ec_jacobi_dbl1(pe* r, const pe* p) {
        // pe matches the 96-byte Jacobian layout s2n expects.
        secp256k1_jdouble(reinterpret_cast<u64*>(r), reinterpret_cast<const u64*>(p));
    }

    void _ec_jacobi_add1(pe* r, const pe* p, const pe* q) {
        secp256k1_jadd(reinterpret_cast<u64*>(r), reinterpret_cast<const u64*>(p),
                       reinterpret_cast<const u64*>(q));
    }

    void _ec_jacobi_rdc1(pe* r, const pe* a) {
        // Jacobian -> affine: x = X/Z^2, y = Y/Z^3.
        fe zinv, zinv2, zinv3;
        fe_clone(zinv, a->z);
        fe_modp_inv(zinv, zinv);
        fe_modp_sqr(zinv2, zinv);
        fe_modp_mul(zinv3, zinv2, zinv);
        fe_modp_mul(r->x, a->x, zinv2);
        fe_modp_mul(r->y, a->y, zinv3);
        fe_set64(r->z, 0x1);
    }

    // RAII variant
    void _ec_jacobi_grprdc1(pe r[], u64 n) {
        std::vector<FeStore> zz(n);
        for (u64 i = 0; i < n; ++i) fe_clone(zz[i].v, r[i].z);
        fe_modp_grpinv(reinterpret_cast<fe*>(zz.data()), static_cast<u32>(n));

        for (u64 i = 0; i < n; ++i) {
            fe zinv2, zinv3;
            fe_modp_sqr(zinv2, zz[i].v);
            fe_modp_mul(zinv3, zinv2, zz[i].v);
            fe_modp_mul(r[i].x, r[i].x, zinv2);
            fe_modp_mul(r[i].y, r[i].y, zinv3);
            fe_set64(r[i].z, 0x1);
        }
    }

    // ============================================================================
    // MARK: Jacobian coordinates (v2)
    // ============================================================================

    void _ec_jacobi_dbl2(pe* r, const pe* p) {
        u64 s[4], m[4], t[4];
        fe_modp_mul(r->z, p->y, p->z);
        fe_modp_add(r->z, r->z, r->z); // 2*Y*Z
        fe_modp_sqr(t, p->y);          // Y^2
        fe_modp_mul(s, p->x, t);
        fe_modp_add(s, s, s);
        fe_modp_add(s, s, s);          // S = 4*X*Y^2
        fe_modp_sqr(t, t);             // Y^4
        fe_modp_add(r->y, t, t);
        fe_modp_add(t, r->y, r->y);
        fe_modp_add(r->y, t, t);       // 8*Y^4
        fe_modp_sqr(t, p->x);
        fe_modp_add(m, t, t);
        fe_modp_add(m, m, t);          // M = 3*X^2
        fe_modp_sqr(r->x, m);
        fe_modp_add(t, s, s);
        fe_modp_sub(r->x, r->x, t);   // X' = M^2 - 2*S
        fe_modp_sub(t, s, r->x);
        fe_modp_mul(t, m, t);
        fe_modp_sub(r->y, t, r->y);   // Y'
    }

    void _ec_jacobi_add2(pe* r, const pe* p, const pe* q) {
        fe u1, u2, s1, s2, tt, ta;
        fe_modp_sqr(tt, q->z);
        fe_modp_mul(u1, p->x, tt);
        fe_modp_mul(ta, tt, q->z);
        fe_modp_mul(s1, p->y, ta);
        fe_modp_sqr(tt, p->z);
        fe_modp_mul(u2, q->x, tt);
        assert(fe_cmp(u1, u2) != 0);
        fe_modp_mul(ta, tt, p->z);
        fe_modp_mul(s2, q->y, ta);
        fe_modp_sub(u2, u2, u1);       // H
        fe_modp_sub(s2, s2, s1);       // R
        fe_modp_sqr(tt, u2);           // H^2
        fe_modp_mul(u1, u1, tt);       // U1H2
        fe_modp_mul(tt, tt, u2);       // H^3
        fe_modp_add(ta, u1, u1);       // 2*U1H2
        fe_modp_sqr(r->x, s2);
        fe_modp_sub(r->x, r->x, tt);
        fe_modp_sub(r->x, r->x, ta);   // nx
        fe_modp_mul(ta, tt, s1);       // S1*H^3
        fe_modp_sub(r->y, u1, r->x);
        fe_modp_mul(r->y, r->y, s2);
        fe_modp_sub(r->y, r->y, ta);   // ny
        fe_modp_mul(r->z, p->z, q->z);
        fe_modp_mul(r->z, r->z, u2);   // nz
    }

    void _ec_jacobi_rdc2(pe* r, const pe* a) {
        fe t;
        fe_clone(r->z, a->z);
        fe_modp_inv(r->z, r->z);
        fe_modp_sqr(t, r->z);
        fe_modp_mul(r->x, a->x, t);
        fe_modp_mul(t, t, r->z);
        fe_modp_mul(r->y, a->y, t);
        fe_set64(r->z, 0x1);
    }

    // RAII variant
    void _ec_jacobi_grprdc2(pe r[], u64 n) {
        assert(n > 0 && n <= static_cast<u64>(std::numeric_limits<std::size_t>::max())); 
        auto zz = std::make_unique<fe[]>(static_cast<std::size_t>(n)); 
        for (u64 i = 0; i < n; ++i) fe_clone(zz[i], r[i].z);
        fe_modp_grpinv(zz.get(), static_cast<u32>(n));

        fe z = { 0 };
        for (u64 i = 0; i < n; ++i) {
            fe_modp_sqr(z, zz[i]);
            fe_modp_mul(r[i].x, r[i].x, z);
            fe_modp_mul(z, z, zz[i]);
            fe_modp_mul(r[i].y, r[i].y, z);
            fe_set64(r[i].z, 0x1);
        }
    }

    // ── dispatch (v1 selected) ───────────────────────────────────────────────────


    void fe_modp_mul2(fe r, const fe a) { bignum_double_p256k1(r, a); }
    void fe_modp_mul3(fe r, const fe a) { bignum_triple_p256k1(r, a); }
    void fe_modp_half(fe r, const fe a) { bignum_half_p256k1(r, a); }
    void fe_modp_mulword(fe r, const fe a, u64 w) { bignum_cmul_p256k1(r, w, a); }
    void fe_modp_reduce(fe r, const fe a) { bignum_mod_p256k1_4(r, a); }
    void fe_modn_reduce(fe r, const fe a) { bignum_mod_n256k1_4(r, a); }

    // q must be affine (z == 1). _gtable[] always is, and jmixadd reads only
    // X,Y from q -- 64 bytes, not 96.
    void _ec_jacobi_add1_affine(pe* r, const pe* p, const pe* q) {
        secp256k1_jmixadd(reinterpret_cast<u64*>(r), reinterpret_cast<const u64*>(p),
                          reinterpret_cast<const u64*>(q));
    }

    void ec_jacobi_add_affine(pe* r, const pe* p, const pe* q) { _ec_jacobi_add1_affine(r, p, q); }

    void ec_jacobi_dbl(pe* r, const pe* p) { _ec_jacobi_dbl1(r, p); }
    void ec_jacobi_add(pe* r, const pe* p, const pe* q) { _ec_jacobi_add1(r, p, q); }
    void ec_jacobi_rdc(pe* r, const pe* a) { _ec_jacobi_rdc1(r, a); }
    void ec_jacobi_grprdc(pe r[], u64 n) { _ec_jacobi_grprdc1(r, n); }

    // ============================================================================
    // MARK: EC scalar multiplication
    // ============================================================================

    void ec_jacobi_mul(pe* r, const pe* p, const fe k) {
        pe t;
        pe_clone(&t, p);
        fe_set64(r->x, 0x0);
        fe_set64(r->y, 0x0);
        fe_set64(r->z, 0x1);

        u32 bits = static_cast<u32>(fe_bitlen(k));
        for (u32 i = 0; i < bits; ++i) {
            if (k[i / 64] & (1ULL << (i % 64))) {
                if (r->x[0] == 0 && r->y[0] == 0) {
                    pe_clone(r, &t);
                }
                else {
                    ec_jacobi_add(r, r, &t);
                }
            }
            ec_jacobi_dbl(&t, &t);
        }
    }

    void ec_jacobi_addrdc(pe* r, const pe* p, const pe* q) {
        ec_jacobi_add(r, p, q);
        ec_jacobi_rdc(r, r);
    }

    void ec_jacobi_mulrdc(pe* r, const pe* p, const fe k) {
        ec_jacobi_mul(r, p, k);
        ec_jacobi_rdc(r, r);
    }

    void ec_jacobi_dblrdc(pe* r, const pe* p) {
        ec_jacobi_dbl(r, p);
        ec_jacobi_rdc(r, r);
    }

    // ============================================================================
    // MARK: EC verify
    // ============================================================================

    bool ec_verify(const pe* p) {
        // y^2 = x^3 + 7
        pe q, g;
        pe_clone(&q, p);
        ec_jacobi_rdc(&q, &q);

        pe_clone(&g, &q);
        fe_modp_sqr(g.y, g.y);       // y^2
        fe_modp_sqr(g.x, g.x);       // x^2
        fe_modp_mul(g.x, g.x, q.x);  // x^3
        fe_modp_sub(g.y, g.y, g.x);  // y^2 - x^3
        return g.y[0] == 7 && g.y[1] == 0 && g.y[2] == 0 && g.y[3] == 0;
    }

    // ============================================================================
    // MARK: EC GTable
    // ============================================================================

    u64  _GTABLE_W = 14;
    static pe* _gtable = nullptr;

    // https://www.sav.sk/journals/uploads/0215094304C459.pdf (Algorithm 3)
    std::size_t ec_gtable_init() {
        const u64 n = 1ULL << _GTABLE_W;
        const u64 d = ((256 - 1) / _GTABLE_W) + 1;
        const u64 s = n * d - d;

        const std::size_t mem_size = s * sizeof(pe);

        // Shared mapping, not new[]: 30 MB that every forked worker would
        // otherwise receive as a private copy under Cygwin (see shm.hpp).
        static ShmBlock gtable_blk;
        shm_free(gtable_blk);
        gtable_blk = shm_alloc(mem_size, 64);
        _gtable = static_cast<pe*>(gtable_blk.ptr);

        pe b, p;
        pe_clone(&b, &G1);
        for (u64 i = 0; i < d; ++i) {
            u64 x = (n - 1) * i;
            pe_clone(&_gtable[x], &b);
            pe_clone(&p, &b);
            for (u64 j = 1; j < n - 1; ++j) {
                j == 1 ? ec_jacobi_dbl(&p, &p) : ec_jacobi_add(&p, &p, &b);
                x = (n - 1) * i + j;
                pe_clone(&_gtable[x], &p);
            }
            ec_jacobi_add(&b, &p, &b);
        }

        ec_jacobi_grprdc(_gtable, s);
        return mem_size;
    }

    void ec_gtable_mul(pe* r, const fe pk) {
        if (_gtable == nullptr) {
            printf("GTable is not initialized\n");
            exit(1);
        }

        const u64 n = 1ULL << _GTABLE_W;
        const u64 d = ((256 - 1) / _GTABLE_W) + 1;

        pe q = {};
        fe k;
        fe_clone(k, pk);

        for (u64 i = 0; i < d; ++i) {
            u64 b = k[0] & (n - 1);
            fe_shiftr64(k, static_cast<u8>(_GTABLE_W));
            if (!b) continue;

            u64 x = (n - 1) * i + b - 1;
            // _gtable[] is always affine (ec_jacobi_grprdc in ec_gtable_init),
            // and mixed addition is cheaper than the general one.
            fe_iszero(q.x) ? pe_clone(&q, &_gtable[x])
                : ec_jacobi_add_affine(&q, &q, &_gtable[x]);
        }

        pe_clone(r, &q);
    }

} // namespace ecloop

namespace ecloop {
#if ECLOOP_UNIVERSAL
    bool EC_ASM_OK = false;
    void ec_dispatch_init() noexcept { EC_ASM_OK = cpu_has_bmi2_adx(); }
#else
    void ec_dispatch_init() noexcept {}
#endif
} // namespace ecloop
