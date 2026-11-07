// Runtime selection of the hashing path. Port of lib/addr_dispatch.c.
//
// HASH_BATCH is 16 because both new kernels are 16-wide. When AVX-512 is
// unavailable the same batch of 16 points is split into RMD_LEN chunks and
// handed to the original path (SHA-NI one at a time + AVX2 x8). The
// existing addr33_batch/addr65_batch are untouched and still work.
//
// ECLOOP_NO_AVX512_HASH=1 forces the old path -- useful for checking that
// both paths agree, and for benchmarking them from one binary.

#pragma once

#include <cstddef>

#include "types.hpp"
#include "ecc.hpp"
#include "addr.hpp"
#include "addr_avx512.hpp"

namespace ecloop {

    inline constexpr std::size_t HASH_BATCH = A512_BATCH;

    static_assert(HASH_BATCH % RMD_LEN == 0,
                  "HASH_BATCH must split evenly into RMD_LEN chunks");

    void addr33_batchN(h160_t* hashes, const pe* points, std::size_t count) noexcept;
    void addr65_batchN(h160_t* hashes, const pe* points, std::size_t count) noexcept;

} // namespace ecloop
