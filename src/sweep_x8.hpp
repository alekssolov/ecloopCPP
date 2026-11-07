// Eight-lane IFMA version of the sweep's group step (seq_group in
// engine_x8.cpp), on the radix-2^52 lazy field layer in fe52.hpp.
//
// WHAT THE SCALAR VERSION COSTS. seq_group computes, for each of SHALF
// precomputed points (i+1)G, the two affine sums centre +- (i+1)G with ONE
// inversion for the whole group: 2 muls + 1 square + 7 add/sub per output
// point, plus 3 muls per input for the Montgomery trick. On s2n-bignum
// scalar code that is ~90 ns per key on the machine this was profiled on,
// 44% of the time of the -z / -k modes -- while the AVX-512 hashing next
// to it takes ~100 ns and the IFMA batch-8 layer sits idle, because the
// sweep never calls it.
//
// THIS VERSION does exactly the same arithmetic, eight independent i at a
// time in SoA form, and never leaves radix 2^52 until a coordinate is
// final: it is then made canonical, packed to radix 2^64 and transposed
// into the `pe` array the hasher reads. The output is bit-for-bit what
// seq_group produces, degenerate dx == 0 lanes included (they yield the
// same garbage point the scalar path yields for key 0 and the group seam).
//
// The batch inversion runs as four interleaved chains so the multiply
// latency overlaps: one 8-lane chain would leave half the IFMA throughput
// on the table.

#pragma once

#include "fe52.hpp"

#if ECLOOP_HAVE_FE52

#include <cstddef>
#include <cstring>

#include "ecc.hpp"

#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl,avx512ifma,bmi2")
namespace ecloop {

    // Tables built once from the affine points (i+1)G, i = 0..n-1, in
    // radix-2^64 SoA blocks of eight: block k lane l holds i = 8k + l.
    struct SweepX8Table {
        Fe8* gx = nullptr;
        Fe8* gy = nullptr;
        std::size_t nblk = 0;
    };

    inline void sweep_x8_build(SweepX8Table* t, const pe* sgp, std::size_t n, Fe8* storage) {
        // storage must hold 2 * n/8 Fe8
        t->nblk = n / 8;
        t->gx = storage;
        t->gy = storage + t->nblk;
        for (std::size_t k = 0; k < t->nblk; ++k)
            for (int l = 0; l < 8; ++l) {
                fe8_set_lane(&t->gx[k], l, sgp[8 * k + l].x);
                fe8_set_lane(&t->gy[k], l, sgp[8 * k + l].y);
            }
    }

    // Scratch: 2 * nblk Fe52s (dx and the running products).
    inline std::size_t sweep_x8_scratch(std::size_t nblk) noexcept { return 2 * nblk; }

    namespace sweep_detail {

        // Stores (1,0,0,0) as z of 8 consecutive points at `stride` bytes.
        ECLOOP_ALWAYS_INLINE void store_z_one(pe* base, std::ptrdiff_t stride) noexcept {
            const __m256i one = _mm256_set_epi64x(0, 0, 0, 1);
            u8* d = reinterpret_cast<u8*>(base->z);
            for (int l = 0; l < 8; ++l)
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + l * stride), one);
        }

    } // namespace sweep_detail

    // points[SHALF] = centre; points[SHALF-1-i] = centre - (i+1)G;
    // points[SHALF+1+i] = centre + (i+1)G, for i in [0, SHALF). The last "up"
    // point (i = SHALF-1) lands at index 2*SHALF, so `points` must have room
    // for 2*SHALF + 1 entries (it is a padding slot; the caller ignores it).
    inline void seq_group_x8(pe* points, const pe* centre, const SweepX8Table& T,
                             Fe52s* scratch) noexcept {
        using namespace sweep_detail;
        const std::size_t nblk = T.nblk;            // 128 for SHALF = 1024; must be a multiple of 4, >= 8
        const std::size_t SHALF = nblk * 8;
        Fe52s* dx  = scratch;
        Fe52s* acc = scratch + nblk;

        const Fe52 cx = fe52_bcast(centre->x);
        const Fe52 cy = fe52_bcast(centre->y);
        const Fe52 ncy = fe52_neg(cy);              // 4p - cy, lazy
        const Fe52 ncx = fe52_neg(cx);              // 4p - cx, lazy

        // ── dx = gx - cx, normalised; zero lanes swapped for 1 as the scalar
        // fe_modp_grpinv does (one zero would poison the whole chain).
        __mmask8 zm[512];                            // nblk <= 512, i.e. SHALF <= 4096
        for (std::size_t k = 0; k < nblk; ++k) {
            Fe52 d = fe52_normalize(fe52_sub(fe52_from_fe8(&T.gx[k]), cx));
            const __mmask8 z = fe52_iszero_nform(d);
            zm[k] = z;
            if (__builtin_expect(z != 0, 0)) {
                Fe52 one = fe52_one();
                for (int i = 0; i < 5; ++i) d.l[i] = _mm512_mask_blend_epi64(z, d.l[i], one.l[i]);
            }
            fe52_store(&dx[k], d);
        }

        // ── Montgomery trick, four interleaved chains: acc[k] = acc[k-4]*dx[k]
        Fe52 a0 = fe52_load(&dx[0]), a1 = fe52_load(&dx[1]);
        Fe52 a2 = fe52_load(&dx[2]), a3 = fe52_load(&dx[3]);
        fe52_store(&acc[0], a0); fe52_store(&acc[1], a1);
        fe52_store(&acc[2], a2); fe52_store(&acc[3], a3);
        for (std::size_t k = 4; k < nblk; k += 4) {
            a0 = fe52_mul(a0, fe52_load(&dx[k + 0]));
            a1 = fe52_mul(a1, fe52_load(&dx[k + 1]));
            a2 = fe52_mul(a2, fe52_load(&dx[k + 2]));
            a3 = fe52_mul(a3, fe52_load(&dx[k + 3]));
            fe52_store(&acc[k + 0], a0); fe52_store(&acc[k + 1], a1);
            fe52_store(&acc[k + 2], a2); fe52_store(&acc[k + 3], a3);
        }
        // combine the four chain products, invert the eight lanes once
        const Fe52 p01 = fe52_mul(a0, a1);
        const Fe52 p23 = fe52_mul(a2, a3);
        const Fe52 invP = fe52_invert_lanes(fe52_mul(p01, p23));
        const Fe52 inv01 = fe52_mul(invP, p23);
        const Fe52 inv23 = fe52_mul(invP, p01);
        Fe52 i0 = fe52_mul(inv01, a1), i1 = fe52_mul(inv01, a0);
        Fe52 i2 = fe52_mul(inv23, a3), i3 = fe52_mul(inv23, a2);
        // unroll: dx[k] <- 1/dx[k]
        for (std::size_t k = nblk - 4; k >= 4; k -= 4) {
            const Fe52 d0 = fe52_load(&dx[k + 0]), d1 = fe52_load(&dx[k + 1]);
            const Fe52 d2 = fe52_load(&dx[k + 2]), d3 = fe52_load(&dx[k + 3]);
            fe52_store(&dx[k + 0], fe52_mul(i0, fe52_load(&acc[k - 4])));
            fe52_store(&dx[k + 1], fe52_mul(i1, fe52_load(&acc[k - 3])));
            fe52_store(&dx[k + 2], fe52_mul(i2, fe52_load(&acc[k - 2])));
            fe52_store(&dx[k + 3], fe52_mul(i3, fe52_load(&acc[k - 1])));
            i0 = fe52_mul(i0, d0); i1 = fe52_mul(i1, d1);
            i2 = fe52_mul(i2, d2); i3 = fe52_mul(i3, d3);
        }
        fe52_store(&dx[0], i0); fe52_store(&dx[1], i1);
        fe52_store(&dx[2], i2); fe52_store(&dx[3], i3);

        pe_clone(&points[SHALF], centre);

        // ── the points
        for (std::size_t k = 0; k < nblk; ++k) {
            Fe52 dinv = fe52_load(&dx[k]);
            if (__builtin_expect(zm[k] != 0, 0)) {  // scalar path: inverse of 0 is 0
                const __m512i z = _mm512_setzero_si512();
                for (int i = 0; i < 5; ++i) dinv.l[i] = _mm512_mask_blend_epi64(zm[k], dinv.l[i], z);
            }
            const Fe52 gx = fe52_from_fe8(&T.gx[k]);
            const Fe52 gy = fe52_from_fe8(&T.gy[k]);

            // centre - (i+1)G : slope from (-gy - cy)
            {
                const Fe52 t  = fe52_normalize(fe52_sub16(fe52_zero(), fe52_add(gy, cy)));
                const Fe52 s  = fe52_mul(t, dinv);
                const Fe52 x3 = fe52_normalize(fe52_sub(fe52_add(fe52_sqr(s), ncx), gx));
                const Fe52 u  = fe52_normalize(fe52_sub(cx, x3));
                const Fe52 y3 = fe52_add(fe52_mul(u, s), ncy);
                __m512i xr[4], yr[4];
                fe52_to64_nform(xr, x3);
                fe52_to64(yr, y3);
                pe* base = &points[SHALF - 1 - 8 * k];    // lanes go downwards
                soa64_to_aos(base->x, -static_cast<std::ptrdiff_t>(sizeof(pe)), xr);
                soa64_to_aos(base->y, -static_cast<std::ptrdiff_t>(sizeof(pe)), yr);
                store_z_one(base, -static_cast<std::ptrdiff_t>(sizeof(pe)));
            }
            // centre + (i+1)G : slope from (gy - cy)
            {
                const Fe52 t  = fe52_normalize(fe52_add(gy, ncy));
                const Fe52 s  = fe52_mul(t, dinv);
                const Fe52 x3 = fe52_normalize(fe52_sub(fe52_add(fe52_sqr(s), ncx), gx));
                const Fe52 u  = fe52_normalize(fe52_sub(cx, x3));
                const Fe52 y3 = fe52_add(fe52_mul(u, s), ncy);
                __m512i xr[4], yr[4];
                fe52_to64_nform(xr, x3);
                fe52_to64(yr, y3);
                pe* base = &points[SHALF + 1 + 8 * k];    // lanes go upwards
                soa64_to_aos(base->x, static_cast<std::ptrdiff_t>(sizeof(pe)), xr);
                soa64_to_aos(base->y, static_cast<std::ptrdiff_t>(sizeof(pe)), yr);
                store_z_one(base, static_cast<std::ptrdiff_t>(sizeof(pe)));
            }
        }
    }

} // namespace ecloop

#pragma GCC pop_options
#endif // ECLOOP_HAVE_FE52
