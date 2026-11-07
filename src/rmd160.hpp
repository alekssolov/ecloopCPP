// Copyright (c) vladkens
// https://github.com/vladkens/ecloop
// Licensed under the MIT License.
//
// Scalar RIPEMD-160 single-block compression.

#pragma once

#include "types.hpp"

namespace ecloop {

	void rmd160_final(u32 state[5], const u32 x[16]);

} // namespace ecloop
