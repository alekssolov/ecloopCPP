
#pragma once

#include <cstddef>

#include "types.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define ECLOOP_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ECLOOP_ALWAYS_INLINE inline
#endif

#ifndef ECLOOP_HAS_BUILTIN
#if defined(__has_builtin)
#define ECLOOP_HAS_BUILTIN(x) __has_builtin(x)
#else
#define ECLOOP_HAS_BUILTIN(x) 0
#endif
#endif

namespace ecloop::detail {

    ECLOOP_ALWAYS_INLINE u64 addc64(u64 x, u64 y, u64 carryin, u64* carryout) noexcept {
#if ECLOOP_HAS_BUILTIN(__builtin_addcll)
        return __builtin_addcll(x, y, carryin, carryout);
#else
        const u64 r1 = x + y;
        const u64 c1 = r1 < x;

        const u64 r2 = r1 + carryin;
        const u64 c2 = r2 < r1;

        *carryout = c1 | c2;
        return r2;
#endif
    }

    ECLOOP_ALWAYS_INLINE u64 subc64(u64 x, u64 y, u64 carryin, u64* carryout) noexcept {
#if ECLOOP_HAS_BUILTIN(__builtin_subcll)
        return __builtin_subcll(x, y, carryin, carryout);
#else
        const u64 r1 = x - y;
        const u64 b1 = x < y;

        const u64 r2 = r1 - carryin;
        const u64 b2 = r1 < carryin;

        *carryout = b1 | b2;
        return r2;
#endif
    }

    ECLOOP_ALWAYS_INLINE u32 rotl32(u32 x, unsigned n) noexcept {
#if ECLOOP_HAS_BUILTIN(__builtin_rotateleft32)
        return __builtin_rotateleft32(x, n);
#else
        n &= 31u;
        return static_cast<u32>((x << n) | (x >> ((32u - n) & 31u)));
#endif
    }

    ECLOOP_ALWAYS_INLINE u32 rotr32(u32 x, unsigned n) noexcept {
#if ECLOOP_HAS_BUILTIN(__builtin_rotateright32)
        return __builtin_rotateright32(x, n);
#else
        n &= 31u;
        return static_cast<u32>((x >> n) | (x << ((32u - n) & 31u)));
#endif
    }

    ECLOOP_ALWAYS_INLINE u32 bswap32(u32 x) noexcept {
#if ECLOOP_HAS_BUILTIN(__builtin_bswap32) || defined(__GNUC__) || defined(__clang__)
        return static_cast<u32>(__builtin_bswap32(x));
#else
        return static_cast<u32>(((x & 0x000000ffu) << 24) |
            ((x & 0x0000ff00u) << 8) |
            ((x & 0x00ff0000u) >> 8) |
            ((x & 0xff000000u) >> 24));
#endif
    }

    ECLOOP_ALWAYS_INLINE u64 bswap64(u64 x) noexcept {
#if ECLOOP_HAS_BUILTIN(__builtin_bswap64) || defined(__GNUC__) || defined(__clang__)
        return static_cast<u64>(__builtin_bswap64(x));
#else
        return ((x & 0x00000000000000ffULL) << 56) |
            ((x & 0x000000000000ff00ULL) << 40) |
            ((x & 0x0000000000ff0000ULL) << 24) |
            ((x & 0x00000000ff000000ULL) << 8) |
            ((x & 0x000000ff00000000ULL) >> 8) |
            ((x & 0x0000ff0000000000ULL) >> 24) |
            ((x & 0x00ff000000000000ULL) >> 40) |
            ((x & 0xff00000000000000ULL) >> 56);
#endif
    }

    ECLOOP_ALWAYS_INLINE u32 load_be32(const u8* p) noexcept {
        return (static_cast<u32>(p[0]) << 24) |
            (static_cast<u32>(p[1]) << 16) |
            (static_cast<u32>(p[2]) << 8) |
            static_cast<u32>(p[3]);
    }

    ECLOOP_ALWAYS_INLINE u32 load_le32(const u8* p) noexcept {
        return static_cast<u32>(p[0]) |
            (static_cast<u32>(p[1]) << 8) |
            (static_cast<u32>(p[2]) << 16) |
            (static_cast<u32>(p[3]) << 24);
    }

} // namespace ecloop::detail
