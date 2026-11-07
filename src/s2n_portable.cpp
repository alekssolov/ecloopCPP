// The fourteen s2n-bignum secp256k1 routines the project calls, in plain
// C++ on 64x64->128 multiplies: for a CPU without BMI2/ADX (mulx, adcx,
// adox), where the assembly objects in s2n/ raise SIGILL. Built only with
// -DECLOOP_PORTABLE_S2N=1 (the Makefile's PORTABLE=1), and then s2n/*.o
// must NOT be linked -- the symbols are the same.
//
// Semantics match s2n-bignum: field routines accept any 256-bit input and
// return the fully reduced result; the point routines take Jacobian points
// as 12 words (X, Y, Z, 4 words each, little-endian limbs), jmixadd's
// second point affine as 8 words, and handle the exceptional cases (either
// input at infinity, P1 == P2, P1 == -P2) so callers need no special care.
//
// Speed is a third to a half of the assembly's per operation; on the kind
// of machine this is for (2004-era Pentium 4) that is the least of it.

#if (defined(ECLOOP_PORTABLE_S2N) && ECLOOP_PORTABLE_S2N) || (defined(ECLOOP_UNIVERSAL) && ECLOOP_UNIVERSAL)

#include <cstring>

#include "types.hpp"

extern "C" {

using ecloop::u64;
typedef unsigned __int128 u128;

namespace {

    // p = 2^256 - C, C = 2^32 + 977
    constexpr u64 P_C = 0x1000003D1ULL;
    constexpr u64 P0 = 0xFFFFFFFEFFFFFC2FULL, P1 = 0xFFFFFFFFFFFFFFFFULL,
                  P2 = 0xFFFFFFFFFFFFFFFFULL, P3 = 0xFFFFFFFFFFFFFFFFULL;
    // n = curve order
    constexpr u64 N0 = 0xBFD25E8CD0364141ULL, N1 = 0xBAAEDCE6AF48A03BULL,
                  N2 = 0xFFFFFFFFFFFFFFFEULL, N3 = 0xFFFFFFFFFFFFFFFFULL;

    inline bool ge_p(const u64 x[4]) {
        if (x[3] != P3) return x[3] > P3;
        if (x[2] != P2) return x[2] > P2;
        if (x[1] != P1) return x[1] > P1;
        return x[0] >= P0;
    }
    inline bool ge_n(const u64 x[4]) {
        if (x[3] != N3) return x[3] > N3;
        if (x[2] != N2) return x[2] > N2;
        if (x[1] != N1) return x[1] > N1;
        return x[0] >= N0;
    }
    inline u64 sub4(u64 z[4], const u64 x[4], const u64 y[4]) {   // returns borrow
        u128 b = 0;
        for (int i = 0; i < 4; ++i) {
            const u128 t = static_cast<u128>(x[i]) - y[i] - b;
            z[i] = static_cast<u64>(t);
            b = (t >> 64) & 1;   // borrow if wrapped
        }
        return static_cast<u64>(b);
    }
    inline u64 add4(u64 z[4], const u64 x[4], const u64 y[4]) {   // returns carry
        u128 c = 0;
        for (int i = 0; i < 4; ++i) {
            const u128 t = static_cast<u128>(x[i]) + y[i] + c;
            z[i] = static_cast<u64>(t);
            c = t >> 64;
        }
        return static_cast<u64>(c);
    }
    inline void sub_p_if_ge(u64 z[4]) {
        if (ge_p(z)) { const u64 p[4] = {P0, P1, P2, P3}; sub4(z, z, p); }
    }

    // z = (value of the 5 limbs t[0..4]) mod p, given t < 2^256 + 2^33 * 2^33
    // (a folded product): fold the top limb once more, then subtract p.
    inline void reduce5(u64 z[4], const u64 t[5]) {
        u128 acc = static_cast<u128>(t[4]) * P_C + t[0];
        z[0] = static_cast<u64>(acc); acc >>= 64;
        acc += t[1]; z[1] = static_cast<u64>(acc); acc >>= 64;
        acc += t[2]; z[2] = static_cast<u64>(acc); acc >>= 64;
        acc += t[3]; z[3] = static_cast<u64>(acc); acc >>= 64;
        if (acc) {   // one more wrap of at most 1 * C
            acc = static_cast<u128>(static_cast<u64>(acc)) * P_C + z[0];
            z[0] = static_cast<u64>(acc); acc >>= 64;
            for (int i = 1; i < 4 && acc; ++i) { acc += z[i]; z[i] = static_cast<u64>(acc); acc >>= 64; }
        }
        sub_p_if_ge(z);
    }

    // z = (8-limb product r) mod p
    inline void reduce8(u64 z[4], const u64 r[8]) {
        // low + high * C, high * C < 2^256 * 2^33
        u64 t[5];
        u128 acc = 0, carry = 0;
        for (int i = 0; i < 4; ++i) {
            const u128 m = static_cast<u128>(r[4 + i]) * P_C;
            acc = static_cast<u128>(r[i]) + static_cast<u64>(m) + carry;
            t[i] = static_cast<u64>(acc);
            carry = (acc >> 64) + (m >> 64);
        }
        t[4] = static_cast<u64>(carry);
        reduce5(z, t);
    }

    inline void mul_p(u64 z[4], const u64 x[4], const u64 y[4]) {
        u64 r[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < 4; ++i) {
            u128 c = 0;
            for (int j = 0; j < 4; ++j) {
                const u128 t = static_cast<u128>(x[i]) * y[j] + r[i + j] + c;
                r[i + j] = static_cast<u64>(t);
                c = t >> 64;
            }
            r[i + 4] = static_cast<u64>(c);
        }
        reduce8(z, r);
    }

    inline void reduce_mod_p(u64 z[4], const u64 x[4]) {   // x < 2^256
        u64 t[4] = {x[0], x[1], x[2], x[3]};
        sub_p_if_ge(t);
        std::memcpy(z, t, 32);
    }

    inline bool is_zero4(const u64 x[4]) { return (x[0] | x[1] | x[2] | x[3]) == 0; }
    inline bool eq4(const u64 x[4], const u64 y[4]) { return std::memcmp(x, y, 32) == 0; }

} // namespace

void port_bignum_add_p256k1(u64 z[4], const u64 x[4], const u64 y[4]) {
    u64 t[5];
    t[4] = add4(t, x, y);
    reduce5(z, t);
}

void port_bignum_sub_p256k1(u64 z[4], const u64 x[4], const u64 y[4]) {
    // x - y, plus p if that went negative; inputs may be non-canonical, so
    // the result of the plain subtraction is (x - y) mod 2^256 with a
    // borrow flag, and adding p (== -C mod 2^256) restores it mod p.
    u64 t[4];
    const u64 borrow = sub4(t, x, y);
    if (borrow) {
        const u64 p[4] = {P0, P1, P2, P3};
        add4(t, t, p);
    }
    sub_p_if_ge(t);
    std::memcpy(z, t, 32);
}

void port_bignum_neg_p256k1(u64 z[4], const u64 x[4]) {
    u64 t[4];
    reduce_mod_p(t, x);
    if (is_zero4(t)) { std::memset(z, 0, 32); return; }
    const u64 p[4] = {P0, P1, P2, P3};
    sub4(z, p, t);
}

void port_bignum_mul_p256k1(u64 z[4], const u64 x[4], const u64 y[4]) { mul_p(z, x, y); }
void port_bignum_sqr_p256k1(u64 z[4], const u64 x[4]) { mul_p(z, x, x); }

void port_bignum_double_p256k1(u64 z[4], const u64 x[4]) { port_bignum_add_p256k1(z, x, x); }

void port_bignum_triple_p256k1(u64 z[4], const u64 x[4]) {
    u64 t[4];
    port_bignum_add_p256k1(t, x, x);
    port_bignum_add_p256k1(z, t, x);
}

void port_bignum_cmul_p256k1(u64 z[4], u64 c, const u64 x[4]) {
    u64 t[5];
    u128 carry = 0;
    for (int i = 0; i < 4; ++i) {
        const u128 m = static_cast<u128>(c) * x[i] + carry;
        t[i] = static_cast<u64>(m);
        carry = m >> 64;
    }
    t[4] = static_cast<u64>(carry);
    // t < 2^320: fold the top limb (up to 64 bits) once through reduce8's
    // path by treating it as a partial product
    u64 r[8] = {t[0], t[1], t[2], t[3], t[4], 0, 0, 0};
    reduce8(z, r);
}

void port_bignum_half_p256k1(u64 z[4], const u64 x[4]) {
    // x/2 mod p: if x is odd add p first (p is odd, so x + p is even)
    u64 t[4];
    reduce_mod_p(t, x);
    u64 carry = 0;
    if (t[0] & 1) {
        const u64 p[4] = {P0, P1, P2, P3};
        carry = add4(t, t, p);
    }
    for (int i = 0; i < 4; ++i) {
        const u64 next = (i < 3) ? t[i + 1] : carry;
        t[i] = (t[i] >> 1) | (next << 63);
    }
    std::memcpy(z, t, 32);
}

void port_bignum_mod_p256k1_4(u64 z[4], const u64 x[4]) { reduce_mod_p(z, x); }

void port_bignum_mod_n256k1_4(u64 z[4], const u64 x[4]) {
    u64 t[4] = {x[0], x[1], x[2], x[3]};
    if (ge_n(t)) { const u64 n[4] = {N0, N1, N2, N3}; sub4(t, t, n); }
    std::memcpy(z, t, 32);
}

// ── points ────────────────────────────────────────────────────────────────
namespace {
    inline void fe_copy(u64 z[4], const u64 x[4]) { std::memcpy(z, x, 32); }

    // p3 = 2 * p1, secp256k1 (a = 0). Z == 0 -> infinity stays infinity.
    void jdouble(u64 p3[12], const u64 p1[12]) {
        const u64* X = p1; const u64* Y = p1 + 4; const u64* Z = p1 + 8;
        u64 yy[4], s[4], m[4], t[4], x3[4], y3[4], z3[4];
        mul_p(yy, Y, Y);                 // YY = Y^2
        mul_p(s, X, yy);                 // S = 4*X*YY
        port_bignum_double_p256k1(s, s);
        port_bignum_double_p256k1(s, s);
        mul_p(m, X, X);                  // M = 3*X^2
        port_bignum_triple_p256k1(m, m);
        mul_p(t, m, m);                  // X3 = M^2 - 2S
        port_bignum_sub_p256k1(x3, t, s);
        port_bignum_sub_p256k1(x3, x3, s);
        port_bignum_sub_p256k1(t, s, x3);     // Y3 = M*(S - X3) - 8*YY^2
        mul_p(y3, m, t);
        mul_p(t, yy, yy);
        port_bignum_double_p256k1(t, t);
        port_bignum_double_p256k1(t, t);
        port_bignum_double_p256k1(t, t);
        port_bignum_sub_p256k1(y3, y3, t);
        mul_p(z3, Y, Z);                 // Z3 = 2*Y*Z
        port_bignum_double_p256k1(z3, z3);
        fe_copy(p3, x3); fe_copy(p3 + 4, y3); fe_copy(p3 + 8, z3);
    }

    // p3 = p1 + p2, both Jacobian; p2 may have Z == 1 (mixed) or be given
    // as affine (z2 == nullptr). Exceptional cases handled.
    void jadd_impl(u64 p3[12], const u64 p1[12], const u64 x2[4], const u64 y2[4], const u64* z2) {
        const u64* X1 = p1; const u64* Y1 = p1 + 4; const u64* Z1 = p1 + 8;
        u64 z1c[4]; reduce_mod_p(z1c, Z1);
        if (is_zero4(z1c)) {             // P1 at infinity -> P2
            fe_copy(p3, x2); fe_copy(p3 + 4, y2);
            if (z2) fe_copy(p3 + 8, z2); else { p3[8] = 1; p3[9] = p3[10] = p3[11] = 0; }
            return;
        }
        if (z2) {
            u64 z2c[4]; reduce_mod_p(z2c, z2);
            if (is_zero4(z2c)) { std::memcpy(p3, p1, 96); return; }   // P2 at infinity -> P1
        }
        u64 z1z1[4], z2z2[4], u1[4], u2[4], s1[4], s2[4], t[4];
        mul_p(z1z1, Z1, Z1);
        if (z2) mul_p(z2z2, z2, z2); else { z2z2[0] = 1; z2z2[1] = z2z2[2] = z2z2[3] = 0; }
        mul_p(u1, X1, z2z2);             // U1 = X1*Z2^2
        mul_p(u2, x2, z1z1);             // U2 = X2*Z1^2
        if (z2) { mul_p(t, z2z2, z2); mul_p(s1, Y1, t); }   // S1 = Y1*Z2^3
        else fe_copy(s1, Y1);
        mul_p(t, z1z1, Z1);
        mul_p(s2, y2, t);                // S2 = Y2*Z1^3
        u64 h[4], r[4];
        port_bignum_sub_p256k1(h, u2, u1);    // H = U2 - U1
        port_bignum_sub_p256k1(r, s2, s1);    // R = S2 - S1
        u64 hc[4], rc[4]; reduce_mod_p(hc, h); reduce_mod_p(rc, r);
        if (is_zero4(hc)) {
            if (is_zero4(rc)) { jdouble(p3, p1); return; }   // P1 == P2
            std::memset(p3, 0, 96);                            // P1 == -P2
            return;
        }
        u64 hh[4], hhh[4], v[4], x3[4], y3[4], z3[4];
        mul_p(hh, h, h);
        mul_p(hhh, hh, h);
        mul_p(v, u1, hh);                // V = U1*HH
        mul_p(x3, r, r);                 // X3 = R^2 - HHH - 2V
        port_bignum_sub_p256k1(x3, x3, hhh);
        port_bignum_sub_p256k1(x3, x3, v);
        port_bignum_sub_p256k1(x3, x3, v);
        port_bignum_sub_p256k1(t, v, x3);     // Y3 = R*(V - X3) - S1*HHH
        mul_p(y3, r, t);
        mul_p(t, s1, hhh);
        port_bignum_sub_p256k1(y3, y3, t);
        if (z2) { mul_p(t, Z1, z2); mul_p(z3, t, h); }   // Z3 = Z1*Z2*H
        else mul_p(z3, Z1, h);
        fe_copy(p3, x3); fe_copy(p3 + 4, y3); fe_copy(p3 + 8, z3);
    }
} // namespace

void port_secp256k1_jdouble(u64 p3[12], const u64 p1[12]) {
    u64 t[12];
    jdouble(t, p1);
    std::memcpy(p3, t, 96);
}

void port_secp256k1_jadd(u64 p3[12], const u64 p1[12], const u64 p2[12]) {
    u64 t[12];
    jadd_impl(t, p1, p2, p2 + 4, p2 + 8);
    std::memcpy(p3, t, 96);
}

void port_secp256k1_jmixadd(u64 p3[12], const u64 p1[12], const u64 p2[8]) {
    u64 t[12];
    jadd_impl(t, p1, p2, p2 + 4, nullptr);
    std::memcpy(p3, t, 96);
}

// ── the eight-lane SoA variants ec_batch8.cpp falls back to ───────────────
// z[limb][lane]: one lane at a time through the scalar routines.
namespace {
    typedef void (*fe_op3)(u64*, const u64*, const u64*);
    inline void soa8(u64 z[4][8], const u64 x[4][8], const u64 y[4][8], fe_op3 f) {
        for (int l = 0; l < 8; ++l) {
            u64 a[4], b[4], c[4];
            for (int i = 0; i < 4; ++i) { a[i] = x[i][l]; b[i] = y[i][l]; }
            f(c, a, b);
            for (int i = 0; i < 4; ++i) z[i][l] = c[i];
        }
    }
} // namespace

void port_bignum_add_p256k1_x8_soa(u64 z[4][8], const u64 x[4][8], const u64 y[4][8]) {
    soa8(z, x, y, port_bignum_add_p256k1);
}
void port_bignum_sub_p256k1_x8_soa(u64 z[4][8], const u64 x[4][8], const u64 y[4][8]) {
    soa8(z, x, y, port_bignum_sub_p256k1);
}
void port_bignum_mul_p256k1_x8_soa(u64 z[4][8], const u64 x[4][8], const u64 y[4][8]) {
    soa8(z, x, y, port_bignum_mul_p256k1);
}

} // extern "C"

#endif // ECLOOP_PORTABLE_S2N
