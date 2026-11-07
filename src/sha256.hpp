#pragma once

#include "types.hpp"

namespace ecloop {

	void sha256_final(u32 state[8], const u8 data[], u32 length);

	// Whether sha256_final runs on SHA-NI (else the portable kernel), and
	// the portable kernel itself, for a forced scalar path and for tests.
	bool sha256_shani_available() noexcept;
	void sha256_final_scalar(u32 state[8], const u8 data[], u32 length);

} // namespace ecloop
