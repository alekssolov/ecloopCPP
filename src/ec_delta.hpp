// Incremental walk over precomputed deltas. Port of lib/ec_delta.c.
//
// A generator picks a random position lev in the hex string and bumps that
// digit by one WITHOUT carry ('F' -> '0'). The difference between adjacent
// scalars is therefore always one of
//
//     dk = +16^j          when the digit was 0..E
//     dk = -15 * 16^j     when the digit was 'F'
//
// with j = ndig-1-lev. That is 2*ndig distinct deltas, at most 128 points
// (8 KB, comfortably in L1). One point addition per key instead of a full
// multiplication.
//
// STATE IS PER THREAD. The C original keeps _delta_k and the delta table in
// file-scope statics, which is safe there because that program forks its
// workers. ecloop uses threads, so the mutable walk position lives in a
// DeltaWalker the caller owns. The delta TABLE itself is immutable after
// delta_init and stays shared.

#pragma once

#include <cstddef>

#include "types.hpp"
#include "ecc.hpp"

namespace ecloop {

    inline constexpr int DELTA_NPOS = 64;

    // Builds the delta table for a key of ndigits hex digits. Call once,
    // before any worker starts, and after ec_gtable_init().
    void delta_init(int ndigits) noexcept;

    // Number of hex digits the table was built for.
    int delta_ndigits() noexcept;

    // The delta point for a bump at position lev: +16^j when the digit
    // was 0..E, -15*16^j when it was 'F'. Affine (z == 1). Immutable
    // after delta_init, so callers in different threads may share it.
    const pe* delta_point(int lev, bool was_F) noexcept;

    // The same table by move kind, for the walk inside a key range
    // (engine_x8.cpp, walk_step): kind 0 is the +16^j bump, kind d in
    // 1..15 is the digit d being set to 0, i.e. -d*16^j. Kind 15 is the
    // 'F' -> '0' case above.
    const pe* delta_point_kind(int lev, int kind) noexcept;

    // Tracks the current scalar while it still fits in a u64 (up to 15
    // digits). Needed to catch the degenerate additions: secp256k1_jadd
    // handles neither P == D (that needs a doubling) nor P == -D (the
    // result is the point at infinity). At 64 digits their probability is
    // 2^-256 and tracking is off; at 2 digits the whole space is 256 keys
    // and nothing works without it.
    struct DeltaWalker {
        u64 k = 0;
        bool track = false;

        // Syncs the tracker with a starting hex string. A no-op for long keys.
        void sync(const char* buff) noexcept;

        // One step: P += delta(lev, whether the digit was 'F').
        void step(pe* P, int lev, bool was_F) noexcept;

        // One step by move kind (walk_common.hpp's walk_step): kind 0 is
        // +16^j, kind d = 1..15 is -d*16^j. step(P, lev, was_F) is
        // step_kind(P, lev, was_F ? 15 : 0).
        void step_kind(pe* P, int lev, int kind) noexcept;
    };

    // Converts a batch to affine with a single inversion for the whole batch.
    inline void delta_affine(pe* pts, std::size_t n) noexcept {
        ec_jacobi_grprdc(pts, static_cast<u64>(n));
    }

} // namespace ecloop
