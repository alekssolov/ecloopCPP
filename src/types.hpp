#pragma once

#include <array>
#include <cstdint>

namespace ecloop {

	using u8 = std::uint8_t;
	using u32 = std::uint32_t;

	// В legacy-коде используются __builtin_addcll/subcll,
	// поэтому unsigned long long безопаснее, чем std::uint64_t на LP64.
	using u64 = unsigned long long;
	using u128 = unsigned __int128;

	using FeWords = std::array<u64, 4>;
	using HashWords160 = std::array<u32, 5>;

} // namespace ecloop
