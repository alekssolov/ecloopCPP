#pragma once

#include <cstddef>

#include "types.hpp"
#include "ecc.hpp"
#include "rmd160s.hpp"

namespace ecloop {

	inline constexpr std::size_t HASH_BATCH_SIZE = RMD_LEN;

	using h160_t = u32[5];

	int compare_160(const void* a, const void* b);
	void print_h160(const h160_t h);

	void prepare33(u8 msg[64], const pe* point);
	void prepare65(u8 msg[128], const pe* point);
	void prepare_rmd(u32 rs[16]);

	void addr33(u32 r[5], const pe* point);
	void addr65(u32 r[5], const pe* point);

	void addr33_batch(h160_t* hashes, const pe* points, std::size_t count);
	void addr65_batch(h160_t* hashes, const pe* points, std::size_t count);

	// What the batch functions above run on (the rung below AVX-512).
	const char* addr_fallback_name() noexcept;

} // namespace ecloop
