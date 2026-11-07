#pragma once

#include "types.hpp"

namespace ecloop {

    // AVX2 eight-way SHA-256 (see sha256_avx2.cpp). The inputs carry the
    // padding already, like sha256_final's; out[i] is message i's state as
    // sha256_final would leave it.
    bool sha256_avx2_available() noexcept;
    void sha256_avx2_8x1(u32 out[8][8], const u8 in[8][64]) noexcept;    // one block each
    void sha256_avx2_8x2(u32 out[8][8], const u8 in[8][128]) noexcept;   // two blocks each

} // namespace ecloop
