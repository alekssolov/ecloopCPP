// The secp256k1 field and point routines by name, routed to the s2n-bignum
// assembly or to the portable C++ (s2n_portable.cpp):
//   default build      assembly only (compiled for a CPU that has BMI2/ADX)
//   ECLOOP_PORTABLE_S2N portable only (no assembly linked)
//   ECLOOP_UNIVERSAL   both linked; EC_ASM_OK, set by ec_dispatch_init() at
//                      the top of main, picks the assembly when the CPU has
//                      mulx/adcx/adox and the portable code otherwise.
// The wrappers live in namespace ecloop under the assembly's own names, so
// ecc.cpp and ec_batch8.cpp call them unchanged; the assembly symbols are
// reached through asm labels.
#pragma once

#include "types.hpp"

#ifndef ECLOOP_PORTABLE_S2N
#define ECLOOP_PORTABLE_S2N 0
#endif
#ifndef ECLOOP_UNIVERSAL
#define ECLOOP_UNIVERSAL 0
#endif

extern "C" {
#if !ECLOOP_PORTABLE_S2N
    void asm_bignum_add_p256k1(ecloop::u64 z[4], const ecloop::u64 x[4], const ecloop::u64 y[4]) asm("bignum_add_p256k1");
    void asm_bignum_sub_p256k1(ecloop::u64 z[4], const ecloop::u64 x[4], const ecloop::u64 y[4]) asm("bignum_sub_p256k1");
    void asm_bignum_neg_p256k1(ecloop::u64 z[4], const ecloop::u64 x[4]) asm("bignum_neg_p256k1");
    void asm_bignum_mul_p256k1(ecloop::u64 z[4], const ecloop::u64 x[4], const ecloop::u64 y[4]) asm("bignum_mul_p256k1");
    void asm_bignum_sqr_p256k1(ecloop::u64 z[4], const ecloop::u64 x[4]) asm("bignum_sqr_p256k1");
    void asm_bignum_double_p256k1(ecloop::u64 z[4], const ecloop::u64 x[4]) asm("bignum_double_p256k1");
    void asm_bignum_triple_p256k1(ecloop::u64 z[4], const ecloop::u64 x[4]) asm("bignum_triple_p256k1");
    void asm_bignum_half_p256k1(ecloop::u64 z[4], const ecloop::u64 x[4]) asm("bignum_half_p256k1");
    void asm_bignum_cmul_p256k1(ecloop::u64 z[4], ecloop::u64 c, const ecloop::u64 x[4]) asm("bignum_cmul_p256k1");
    void asm_bignum_mod_p256k1_4(ecloop::u64 z[4], const ecloop::u64 x[4]) asm("bignum_mod_p256k1_4");
    void asm_bignum_mod_n256k1_4(ecloop::u64 z[4], const ecloop::u64 x[4]) asm("bignum_mod_n256k1_4");
    void asm_secp256k1_jdouble(ecloop::u64 p3[12], const ecloop::u64 p1[12]) asm("secp256k1_jdouble");
    void asm_secp256k1_jadd(ecloop::u64 p3[12], const ecloop::u64 p1[12], const ecloop::u64 p2[12]) asm("secp256k1_jadd");
    void asm_secp256k1_jmixadd(ecloop::u64 p3[12], const ecloop::u64 p1[12], const ecloop::u64 p2[8]) asm("secp256k1_jmixadd");
    void asm_bignum_add_p256k1_x8_soa(ecloop::u64 z[4][8], const ecloop::u64 x[4][8], const ecloop::u64 y[4][8]) asm("bignum_add_p256k1_x8_soa");
    void asm_bignum_sub_p256k1_x8_soa(ecloop::u64 z[4][8], const ecloop::u64 x[4][8], const ecloop::u64 y[4][8]) asm("bignum_sub_p256k1_x8_soa");
    void asm_bignum_mul_p256k1_x8_soa(ecloop::u64 z[4][8], const ecloop::u64 x[4][8], const ecloop::u64 y[4][8]) asm("bignum_mul_p256k1_x8_soa");
#endif
#if ECLOOP_PORTABLE_S2N || ECLOOP_UNIVERSAL
    void port_bignum_add_p256k1(ecloop::u64 z[4], const ecloop::u64 x[4], const ecloop::u64 y[4]);
    void port_bignum_sub_p256k1(ecloop::u64 z[4], const ecloop::u64 x[4], const ecloop::u64 y[4]);
    void port_bignum_neg_p256k1(ecloop::u64 z[4], const ecloop::u64 x[4]);
    void port_bignum_mul_p256k1(ecloop::u64 z[4], const ecloop::u64 x[4], const ecloop::u64 y[4]);
    void port_bignum_sqr_p256k1(ecloop::u64 z[4], const ecloop::u64 x[4]);
    void port_bignum_double_p256k1(ecloop::u64 z[4], const ecloop::u64 x[4]);
    void port_bignum_triple_p256k1(ecloop::u64 z[4], const ecloop::u64 x[4]);
    void port_bignum_half_p256k1(ecloop::u64 z[4], const ecloop::u64 x[4]);
    void port_bignum_cmul_p256k1(ecloop::u64 z[4], ecloop::u64 c, const ecloop::u64 x[4]);
    void port_bignum_mod_p256k1_4(ecloop::u64 z[4], const ecloop::u64 x[4]);
    void port_bignum_mod_n256k1_4(ecloop::u64 z[4], const ecloop::u64 x[4]);
    void port_secp256k1_jdouble(ecloop::u64 p3[12], const ecloop::u64 p1[12]);
    void port_secp256k1_jadd(ecloop::u64 p3[12], const ecloop::u64 p1[12], const ecloop::u64 p2[12]);
    void port_secp256k1_jmixadd(ecloop::u64 p3[12], const ecloop::u64 p1[12], const ecloop::u64 p2[8]);
    void port_bignum_add_p256k1_x8_soa(ecloop::u64 z[4][8], const ecloop::u64 x[4][8], const ecloop::u64 y[4][8]);
    void port_bignum_sub_p256k1_x8_soa(ecloop::u64 z[4][8], const ecloop::u64 x[4][8], const ecloop::u64 y[4][8]);
    void port_bignum_mul_p256k1_x8_soa(ecloop::u64 z[4][8], const ecloop::u64 x[4][8], const ecloop::u64 y[4][8]);
#endif
}

namespace ecloop {

#if ECLOOP_UNIVERSAL
    extern bool EC_ASM_OK;   // set by ec_dispatch_init(), false until then
#define ECLOOP_S2N(name, ...) do { if (EC_ASM_OK) asm_##name(__VA_ARGS__); else port_##name(__VA_ARGS__); } while (0)
#elif ECLOOP_PORTABLE_S2N
#define ECLOOP_S2N(name, ...) port_##name(__VA_ARGS__)
#else
#define ECLOOP_S2N(name, ...) asm_##name(__VA_ARGS__)
#endif

    inline void bignum_add_p256k1(u64 z[4], const u64 x[4], const u64 y[4]) { ECLOOP_S2N(bignum_add_p256k1, z, x, y); }
    inline void bignum_sub_p256k1(u64 z[4], const u64 x[4], const u64 y[4]) { ECLOOP_S2N(bignum_sub_p256k1, z, x, y); }
    inline void bignum_neg_p256k1(u64 z[4], const u64 x[4]) { ECLOOP_S2N(bignum_neg_p256k1, z, x); }
    inline void bignum_mul_p256k1(u64 z[4], const u64 x[4], const u64 y[4]) { ECLOOP_S2N(bignum_mul_p256k1, z, x, y); }
    inline void bignum_sqr_p256k1(u64 z[4], const u64 x[4]) { ECLOOP_S2N(bignum_sqr_p256k1, z, x); }
    inline void bignum_double_p256k1(u64 z[4], const u64 x[4]) { ECLOOP_S2N(bignum_double_p256k1, z, x); }
    inline void bignum_triple_p256k1(u64 z[4], const u64 x[4]) { ECLOOP_S2N(bignum_triple_p256k1, z, x); }
    inline void bignum_half_p256k1(u64 z[4], const u64 x[4]) { ECLOOP_S2N(bignum_half_p256k1, z, x); }
    inline void bignum_cmul_p256k1(u64 z[4], u64 c, const u64 x[4]) { ECLOOP_S2N(bignum_cmul_p256k1, z, c, x); }
    inline void bignum_mod_p256k1_4(u64 z[4], const u64 x[4]) { ECLOOP_S2N(bignum_mod_p256k1_4, z, x); }
    inline void bignum_mod_n256k1_4(u64 z[4], const u64 x[4]) { ECLOOP_S2N(bignum_mod_n256k1_4, z, x); }
    inline void secp256k1_jdouble(u64 p3[12], const u64 p1[12]) { ECLOOP_S2N(secp256k1_jdouble, p3, p1); }
    inline void secp256k1_jadd(u64 p3[12], const u64 p1[12], const u64 p2[12]) { ECLOOP_S2N(secp256k1_jadd, p3, p1, p2); }
    inline void secp256k1_jmixadd(u64 p3[12], const u64 p1[12], const u64 p2[8]) { ECLOOP_S2N(secp256k1_jmixadd, p3, p1, p2); }
    inline void bignum_add_p256k1_x8_soa(u64 z[4][8], const u64 x[4][8], const u64 y[4][8]) { ECLOOP_S2N(bignum_add_p256k1_x8_soa, z, x, y); }
    inline void bignum_sub_p256k1_x8_soa(u64 z[4][8], const u64 x[4][8], const u64 y[4][8]) { ECLOOP_S2N(bignum_sub_p256k1_x8_soa, z, x, y); }
    inline void bignum_mul_p256k1_x8_soa(u64 z[4][8], const u64 x[4][8], const u64 y[4][8]) { ECLOOP_S2N(bignum_mul_p256k1_x8_soa, z, x, y); }

    // Call first thing in main (and in tests): decides asm vs portable in
    // the universal build; a no-op elsewhere.
    void ec_dispatch_init() noexcept;

} // namespace ecloop
