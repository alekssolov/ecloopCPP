// secp256k1 endomorphism (GLV). Port of lib/ec_glv.c.
//
// phi(x, y) = (beta*x mod p, y), with phi(P) = lambda*P. One computed point
// therefore yields two more at the cost of a single multiplication each --
// y does not change at all. The keys of the images are lambda*k and
// lambda^2*k modulo n.
//
// beta^3 == 1 mod p and lambda^3 == 1 mod n.
//
// Works for any key size: the extra keys land anywhere in the space, but
// the target set is a shared base of millions of addresses rather than
// something tied to the search range, so checking them is exactly as
// worthwhile as checking any other key.
//
// The constants live here and nowhere else. In the C tree they had been
// copied into engine_x8.c and two test files -- four copies of the same 128
// hex digits, where one wrong digit in one copy lets the tests pass while
// production searches the wrong keys, or the reverse.

#pragma once

#include <cstddef>

#include "types.hpp"
#include "ecc.hpp"

namespace ecloop {

    extern const fe GLV_BETA;
    extern const fe GLV_BETA2;
    extern const fe GLV_LAMBDA;
    extern const fe GLV_LAMBDA2;

    // Turns n points into 3n: [0,n) the originals, [n,2n) phi, [2n,3n) phi^2.
    // pts must have room for 3n entries.
    void glv_expand(pe* pts, std::size_t n) noexcept;

    // Key of the point at idx within the expanded array.
    void glv_key(fe out, const fe base_k, std::size_t idx, std::size_t n) noexcept;

} // namespace ecloop
