// Copyright (c) vladkens
// https://github.com/vladkens/ecloop
// Licensed under the MIT License.
//
// SIMD (batched) RIPEMD-160. Lane width depends on the target:
//   AVX2 -> 8 lanes, NEON -> 4 lanes, scalar fallback -> 1 lane.

#pragma once

#include <cstddef>

#include "types.hpp"

#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(NO_SIMD)
#define ECLOOP_RMD_LEN 4
#elif defined(__x86_64__) && !defined(NO_SIMD)
#define ECLOOP_RMD_LEN 8
#else
#define ECLOOP_RMD_LEN 1
#endif

namespace ecloop {

    inline constexpr std::size_t RMD_LEN = ECLOOP_RMD_LEN;

    void rmd160_batch(u32 r[ECLOOP_RMD_LEN][5], const u32 x[ECLOOP_RMD_LEN][16]);

} // namespace ecloop

