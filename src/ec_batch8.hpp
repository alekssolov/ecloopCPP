// Batch-8 (SoA / AVX512IFMA) field and point layer. Port of lib/ec_batch8.c.
//
// Processes 8 INDEPENDENT curve points per pass so the batch-8 IFMA
// multiply in s2n/bignum_mul_p256k1_x8_soa.S is actually used. The
// single-point path in ecc.cpp is untouched and remains the fallback and
// the reference.
//
// LAYOUT. Everything here is limb-major (SoA):
//
//     Fe8.v[limb][lane]      limb = 0..3 (radix 2^64), lane = 0..7
//
// so v[i] is one full 64-byte zmm row -- exactly what the *_x8_soa
// primitives load and store. Values therefore flow from one batch
// primitive to the next with no transpose at all. A transpose happens only
// at the true API boundary (fe8_set_lane / fe8_get_lane), i.e. when
// talking to the scalar AoS world.
//
// DISPATCH. ec_batch8_available() checks AVX512F + AVX512IFMA once. When
// it is false every entry point falls back to the scalar ecc.cpp routines
// lane by lane, so the program still runs correctly on a CPU without IFMA.
// ECLOOP_NO_IFMA=1 forces that fallback, which makes both paths testable
// from one binary.
//
// NOTE ON CALLERS. Nothing in ecloop calls ec_jacobi_add_affine_x8 yet:
// this layer models eight independent Jacobian walkers, whereas
// batch_add() in main.cpp amortises ONE inversion over 2048 affine points.
// The layer is ported as a complete, tested unit for callers that want it.

#pragma once

#include <cstddef>

#include "types.hpp"
#include "ecc.hpp"
#include "compat.hpp"

namespace ecloop {

    struct alignas(64) Fe8 {
        u64 v[4][8];
    };

    struct alignas(64) Pe8 {
        Fe8 x, y, z;
    };

    static_assert(sizeof(Fe8) == 256, "Fe8 must be 4 x 8 x u64 with no padding");
    static_assert(sizeof(Pe8) == 768, "Pe8 must be three Fe8 with no padding");

    // Resolved once during static initialisation, before main runs.
    //
    // NOT a function-local static: that form emits a guard-variable check on
    // EVERY call, and this is called from each fe8_add/sub/mul. Across a
    // block that is thousands of acquire loads for a value that never
    // changes. As a namespace-scope const with a dynamic initialiser it is
    // ready before main, so reading it is a plain load and the accessor
    // inlines away even without LTO.
    extern const bool EC_BATCH8_OK;

    ECLOOP_ALWAYS_INLINE bool ec_batch8_available() noexcept { return EC_BATCH8_OK; }

    // ── AoS <-> SoA at the API boundary ──────────────────────────────────────
    void fe8_set_lane(Fe8* r, int lane, const fe a) noexcept;
    void fe8_get_lane(fe r, const Fe8* a, int lane) noexcept;
    void pe8_set_lane(Pe8* r, int lane, const pe* p) noexcept;
    void pe8_get_lane(pe* r, const Pe8* a, int lane) noexcept;
    void fe8_set_all(Fe8* r, const fe a) noexcept;
    void fe8_zero(Fe8* r) noexcept;
    void fe8_one(Fe8* r) noexcept;

    // Bit L set when lane L is zero.
    u32 fe8_zero_mask(const Fe8* a) noexcept;

    // ── batch-8 field ops ────────────────────────────────────────────────────
    // Aliasing: r == a and r == b are both safe -- every primitive reads its
    // whole input before writing any of the output.
    void fe8_add(Fe8* r, const Fe8* a, const Fe8* b) noexcept;
    void fe8_sub(Fe8* r, const Fe8* a, const Fe8* b) noexcept;
    void fe8_mul(Fe8* r, const Fe8* a, const Fe8* b) noexcept;
    void fe8_sqr(Fe8* r, const Fe8* a) noexcept;
    inline void fe8_dbl(Fe8* r, const Fe8* a) noexcept { fe8_add(r, a, a); }

    // ── points ───────────────────────────────────────────────────────────────

    // Scalar mixed add that survives the degenerate cases. Used by the
    // non-IFMA fallback and by the lane patch-up in the batch version.
    void ec_jmixadd_safe(pe* r, const pe* p, const pe* q) noexcept;

    // P (Jacobian) += Q (affine), eight lanes at a time. madd-2007-bl.
    void ec_jacobi_add_affine_x8(Pe8* r, const Pe8* p, const Fe8* qx,
                                 const Fe8* qy) noexcept;

    // Scratch size ec_jacobi_grprdc_x8 needs for n entries. Use this rather
    // than n: with the tree inversion the down-sweep writes past a buffer
    // sized n, which is silent heap corruption rather than a crash.
    std::size_t ec_grprdc_x8_scratch(std::size_t n) noexcept;

    // n Pe8 (= 8n Jacobian points) -> affine, with ONE field inversion for
    // the whole set. scratch must hold ec_grprdc_x8_scratch(n) entries.
    void ec_jacobi_grprdc_x8(Pe8* pts, std::size_t n, Fe8* scratch) noexcept;

} // namespace ecloop
