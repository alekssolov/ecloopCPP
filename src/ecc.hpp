

#pragma once

#include <cstddef>

#include "types.hpp"

namespace ecloop {

    // Chooses the s2n assembly or the portable field code for this CPU in
    // the universal build (no-op elsewhere). Call first thing in main.
    void ec_dispatch_init() noexcept;


    using u128 = unsigned __int128;

    using fe = u64[4];
    using fe320 = u64[5];

    struct pe {
        fe x;
        fe y;
        fe z;
    };

    extern const fe FE_ZERO;
    extern const fe FE_P;
    extern const fe FE_N;

    extern const fe A1;
    extern const fe A2;
    extern const fe B1;
    extern const fe B2;

    extern const pe G1;
    extern const pe G2;

    extern u64 _GTABLE_W;

    // field helpers
    void fe_print(const char* label, const fe a);
    bool fe_iszero(const fe r);
    void fe_clone(fe r, const fe a);
    void fe_set64(fe r, u64 a);
    std::size_t fe_bitlen(const fe a);
    void fe_add64(fe r, u64 a);
    int fe_cmp64(const fe a, u64 b);
    int fe_cmp(const fe a, const fe b);
    void fe_from_hex(fe r, const char* hex);
    void fe_shiftl(fe r, u8 n);
    void fe_shiftr64(fe r, u8 n);

    // modulo N
    void fe_modn_neg(fe r, const fe a);
    void fe_modn_add(fe r, const fe a, const fe b);
    void fe_modn_sub(fe r, const fe a, const fe b);
    void fe_modn_mul(fe r, const fe a, const fe b);
    void fe_modn_from_hex(fe r, const char* hex);
    void fe_modn_add_stride(fe r, const fe base, const fe stride, u64 offset);

    // modulo P
    void fe_modp_neg(fe r, const fe a);
    void fe_modp_add(fe r, const fe a, const fe b);
    void fe_modp_sub(fe r, const fe a, const fe b);
    void fe_modp_mul(fe __restrict__ r, const fe __restrict__ a, const fe __restrict__ b);
    void fe_modp_sqr(fe __restrict__ r, const fe __restrict__ a);
    void fe_modp_inv(fe r, const fe a);
    void fe_modp_grpinv(fe r[], u32 n);

    // EC point helpers
    void pe_clone(pe* r, const pe* a);

    void ec_affine_dbl(pe* r, const pe* p);
    void ec_affine_add(pe* r, const pe* p, const pe* q);

    void _ec_jacobi_dbl1(pe* r, const pe* p);
    void _ec_jacobi_dbl2(pe* r, const pe* p);
    void _ec_jacobi_add1(pe* r, const pe* p, const pe* q);
    void _ec_jacobi_add2(pe* r, const pe* p, const pe* q);
    void _ec_jacobi_rdc1(pe* r, const pe* a);
    void _ec_jacobi_rdc2(pe* r, const pe* a);
    void _ec_jacobi_grprdc1(pe r[], u64 n);
    void _ec_jacobi_grprdc2(pe r[], u64 n);

    void ec_jacobi_dbl(pe* r, const pe* p);
    void ec_jacobi_add(pe* r, const pe* p, const pe* q);
    void ec_jacobi_rdc(pe* r, const pe* a);
    void ec_jacobi_grprdc(pe r[], u64 n);

    void ec_jacobi_mul(pe* r, const pe* p, const fe k);
    void ec_jacobi_addrdc(pe* r, const pe* p, const pe* q);
    void ec_jacobi_mulrdc(pe* r, const pe* p, const fe k);
    void ec_jacobi_dblrdc(pe* r, const pe* p);

    bool ec_verify(const pe* p);

    // precomputed G table
    std::size_t ec_gtable_init();
    void ec_gtable_mul(pe* r, const fe pk);

    // benchmark-visible inversion variants
    void _fe_modp_inv_binpow(fe r, const fe a);
    void _fe_modp_inv_addchn(fe r, const fe a);


    // s2n-bignum helpers
    void fe_modp_mul2(fe r, const fe a);
    void fe_modp_mul3(fe r, const fe a);
    void fe_modp_half(fe r, const fe a);
    void fe_modp_mulword(fe r, const fe a, u64 w);
    void fe_modp_reduce(fe r, const fe a);
    void fe_modn_reduce(fe r, const fe a);
    void fe_inverse_safegcd(u64 out[4], const u64 x[4]) noexcept;

    // Mixed addition: q must be affine (z == 1).
    void _ec_jacobi_add1_affine(pe* r, const pe* p, const pe* q);
    void ec_jacobi_add_affine(pe* r, const pe* p, const pe* q);

} // namespace ecloop
