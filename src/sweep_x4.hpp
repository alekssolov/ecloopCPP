// Four-lane AVX2 version of the sweep's group step (seq_group in
// engine_x8.cpp), on the radix-2^29 field layer in fe29.hpp. Same
// arithmetic as sweep_x8.hpp with half the lanes and no IFMA: it is the
// EC rung for CPUs with AVX2 but no AVX-512. Output is bit-for-bit what
// seq_group produces, degenerate dx == 0 lanes included.

#pragma once

#include "fe29.hpp"

#if ECLOOP_HAVE_FE29

#include <cstddef>
#include <cstring>

#include "ecc.hpp"

#pragma GCC push_options
#pragma GCC target("avx2")
namespace ecloop {

    // (i+1)G, i = 0..n-1, already in radix-2^29 SoA blocks of four:
    // block k lane l holds i = 4k + l.
    struct SweepX4Table {
        Fe29s* gx = nullptr;
        Fe29s* gy = nullptr;
        std::size_t nblk = 0;
    };

    inline void sweep_x4_build(SweepX4Table* t, const pe* sgp, std::size_t n, Fe29s* storage) {
        // storage must hold 2 * n/4 Fe29s
        t->nblk = n / 4;
        t->gx = storage;
        t->gy = storage + t->nblk;
        for (std::size_t k = 0; k < t->nblk; ++k) {
            const pe* p = sgp + 4 * k;
            fe29_store(&t->gx[k], fe29_from_lanes(p[0].x, p[1].x, p[2].x, p[3].x));
            fe29_store(&t->gy[k], fe29_from_lanes(p[0].y, p[1].y, p[2].y, p[3].y));
        }
    }

    // Scratch: 2 * nblk Fe29s (dx and the running products).
    inline std::size_t sweep_x4_scratch(std::size_t nblk) noexcept { return 2 * nblk; }

    namespace sweep4_detail {
        ECLOOP_ALWAYS_INLINE void store_z_one(pe* base, std::ptrdiff_t stride) noexcept {
            u8* d = reinterpret_cast<u8*>(base->z);
            for (int l = 0; l < 4; ++l) {
                u64* z = reinterpret_cast<u64*>(d + l * stride);
                z[0] = 1; z[1] = z[2] = z[3] = 0;
            }
        }
        // One output point of four lanes: x3 = s^2 - cx - gx, y3 = s (cx - x3) - cy
        ECLOOP_ALWAYS_INLINE void emit(pe* base, std::ptrdiff_t stride, const Fe29& s, const Fe29& cx,
                                       const Fe29& ncx, const Fe29& ncy, const Fe29& gx) noexcept {
            const Fe29 x3 = fe29_normalize(fe29_sub(fe29_add(fe29_sqr(s), ncx), gx));
            const Fe29 u  = fe29_normalize(fe29_sub(cx, x3));
            const Fe29 y3 = fe29_add(fe29_mul(u, s), ncy);
            __m256i xr[4], yr[4];
            fe29_to64(xr, x3);
            fe29_to64(yr, y3);
            fe29_scatter(base->x, stride, xr);
            fe29_scatter(base->y, stride, yr);
            store_z_one(base, stride);
        }
    } // namespace sweep4_detail

    // points[SHALF] = centre; points[SHALF-1-i] = centre - (i+1)G;
    // points[SHALF+1+i] = centre + (i+1)G, for i in [0, SHALF); like
    // seq_group_x8 it writes the padding slot points[2*SHALF].
    inline void seq_group_x4(pe* points, const pe* centre, const SweepX4Table& T,
                             Fe29s* scratch) noexcept {
        using namespace sweep4_detail;
        const std::size_t nblk = T.nblk;            // 256 for SHALF = 1024; a multiple of 4, >= 8
        const std::size_t SHALF = nblk * 4;
        Fe29s* dx  = scratch;
        Fe29s* acc = scratch + nblk;

        const Fe29 cx  = fe29_bcast(centre->x);
        const Fe29 cy  = fe29_bcast(centre->y);
        const Fe29 ncx = fe29_neg(cx);              // 64p - cx, lazy
        const Fe29 ncy = fe29_neg(cy);

        // dx = gx - cx, canonical; zero lanes swapped for 1 as the scalar
        // fe_modp_grpinv does (one zero would poison the whole chain).
        unsigned zm[1024];                          // nblk <= 1024, i.e. SHALF <= 4096
        for (std::size_t k = 0; k < nblk; ++k) {
            Fe29 d = fe29_canon(fe29_sub(fe29_load(&T.gx[k]), cx));
            const unsigned z = fe29_iszero_canon(d);
            zm[k] = z;
            if (__builtin_expect(z != 0, 0)) d = fe29_blend(d, fe29_one(), z);
            fe29_store(&dx[k], d);
        }

        // Montgomery trick, four interleaved chains: acc[k] = acc[k-4] * dx[k]
        Fe29 a0 = fe29_load(&dx[0]), a1 = fe29_load(&dx[1]);
        Fe29 a2 = fe29_load(&dx[2]), a3 = fe29_load(&dx[3]);
        fe29_store(&acc[0], a0); fe29_store(&acc[1], a1);
        fe29_store(&acc[2], a2); fe29_store(&acc[3], a3);
        for (std::size_t k = 4; k < nblk; k += 4) {
            a0 = fe29_mul(a0, fe29_load(&dx[k + 0]));
            a1 = fe29_mul(a1, fe29_load(&dx[k + 1]));
            a2 = fe29_mul(a2, fe29_load(&dx[k + 2]));
            a3 = fe29_mul(a3, fe29_load(&dx[k + 3]));
            fe29_store(&acc[k + 0], a0); fe29_store(&acc[k + 1], a1);
            fe29_store(&acc[k + 2], a2); fe29_store(&acc[k + 3], a3);
        }
        const Fe29 p01 = fe29_mul(a0, a1);
        const Fe29 p23 = fe29_mul(a2, a3);
        const Fe29 invP = fe29_invert_lanes(fe29_mul(p01, p23));
        const Fe29 inv01 = fe29_mul(invP, p23);
        const Fe29 inv23 = fe29_mul(invP, p01);
        Fe29 i0 = fe29_mul(inv01, a1), i1 = fe29_mul(inv01, a0);
        Fe29 i2 = fe29_mul(inv23, a3), i3 = fe29_mul(inv23, a2);
        for (std::size_t k = nblk - 4; k >= 4; k -= 4) {
            const Fe29 d0 = fe29_load(&dx[k + 0]), d1 = fe29_load(&dx[k + 1]);
            const Fe29 d2 = fe29_load(&dx[k + 2]), d3 = fe29_load(&dx[k + 3]);
            fe29_store(&dx[k + 0], fe29_mul(i0, fe29_load(&acc[k - 4])));
            fe29_store(&dx[k + 1], fe29_mul(i1, fe29_load(&acc[k - 3])));
            fe29_store(&dx[k + 2], fe29_mul(i2, fe29_load(&acc[k - 2])));
            fe29_store(&dx[k + 3], fe29_mul(i3, fe29_load(&acc[k - 1])));
            i0 = fe29_mul(i0, d0); i1 = fe29_mul(i1, d1);
            i2 = fe29_mul(i2, d2); i3 = fe29_mul(i3, d3);
        }
        fe29_store(&dx[0], i0); fe29_store(&dx[1], i1);
        fe29_store(&dx[2], i2); fe29_store(&dx[3], i3);

        pe_clone(&points[SHALF], centre);

        for (std::size_t k = 0; k < nblk; ++k) {
            Fe29 dinv = fe29_load(&dx[k]);
            if (__builtin_expect(zm[k] != 0, 0)) dinv = fe29_blend(dinv, fe29_zero(), zm[k]);   // 1/0 := 0, as the scalar path
            const Fe29 gx = fe29_load(&T.gx[k]);
            const Fe29 gy = fe29_load(&T.gy[k]);
            {   // centre - (i+1)G: slope from (-gy - cy)
                const Fe29 t = fe29_normalize(fe29_sub2(fe29_zero(), fe29_add(gy, cy)));
                emit(&points[SHALF - 1 - 4 * k], -static_cast<std::ptrdiff_t>(sizeof(pe)),
                     fe29_mul(t, dinv), cx, ncx, ncy, gx);
            }
            {   // centre + (i+1)G: slope from (gy - cy)
                const Fe29 t = fe29_normalize(fe29_add(gy, ncy));
                emit(&points[SHALF + 1 + 4 * k], static_cast<std::ptrdiff_t>(sizeof(pe)),
                     fe29_mul(t, dinv), cx, ncx, ncy, gx);
            }
        }
    }

} // namespace ecloop

#pragma GCC pop_options
#endif // ECLOOP_HAVE_FE29
