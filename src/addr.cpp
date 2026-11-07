// Copyright (c) vladkens
// https://github.com/vladkens/ecloop
// Licensed under the MIT License.

#include "addr.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>

#include "sha256.hpp"
#include "sha256_avx2.hpp"
#include "cpu.hpp"
#include "rmd160.hpp"
#include "rmd160s.hpp"
#include "compat.hpp"
#include <cstdlib>

namespace ecloop {

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    int compare_160(const void* a, const void* b) {
        const u32* ua = static_cast<const u32*>(a);
        const u32* ub = static_cast<const u32*>(b);
        for (int i = 0; i < 5; ++i) {
            if (ua[i] < ub[i]) return -1;
            if (ua[i] > ub[i]) return  1;
        }
        return 0;
    }

    void print_h160(const h160_t h) {
        for (int i = 0; i < 5; ++i) std::printf("%08x", h[i]);
        std::printf("\n");
    }

    // ---------------------------------------------------------------------------
    // Serialisation helpers
    // ---------------------------------------------------------------------------

    void prepare33(u8 msg[64], const pe* point) {
        // Point must be in affine coordinates (z == 1)
        assert(point->z[0] == 1 &&
            point->z[1] == 0 &&
            point->z[2] == 0 &&
            point->z[3] == 0);

        msg[0] = (point->y[0] & 1u) ? 0x03u : 0x02u;

        for (int i = 0; i < 4; ++i) {
            u64 x_be = detail::bswap64(point->x[3 - i]);
            std::memcpy(&msg[1 + i * 8], &x_be, sizeof(u64));
        }

        msg[33] = 0x80u;
        msg[62] = 0x01u;
        msg[63] = 0x08u;
    }

    void prepare65(u8 msg[128], const pe* point) {
        // Point must be in affine coordinates (z == 1)
        assert(point->z[0] == 1 &&
            point->z[1] == 0 &&
            point->z[2] == 0 &&
            point->z[3] == 0);

        msg[0] = 0x04u;

        // X coordinate in big-endian order → msg[1..32]
        for (int i = 0; i < 4; ++i) {
            u64 x_be = detail::bswap64(point->x[3 - i]);
            std::memcpy(&msg[1 + i * 8], &x_be, sizeof(u64));
        }

        // Y coordinate in big-endian order → msg[33..64]
        for (int i = 0; i < 4; ++i) {
            u64 y_be = detail::bswap64(point->y[3 - i]);
            std::memcpy(&msg[33 + i * 8], &y_be, sizeof(u64));
        }

        msg[65] = 0x80u;
        msg[126] = 0x02u;
        msg[127] = 0x08u;
    }

    void prepare_rmd(u32 rs[16]) {
        for (int i = 0; i < 8; ++i) rs[i] = detail::bswap32(rs[i]);
        rs[8] = 0x00000080u;
        rs[14] = 256u;
    }

    // ---------------------------------------------------------------------------
    // Single-point address derivation
    // ---------------------------------------------------------------------------

    void addr33(u32 r[5], const pe* point) {
        u8  msg[64] = {};   // SHA-256 payload
        u32 rs[16] = {};   // SHA-256 output / RMD-160 input

        prepare33(msg, point);
        sha256_final(rs, msg, sizeof(msg));

        prepare_rmd(rs);
        rmd160_final(r, rs);
    }

    void addr65(u32 r[5], const pe* point) {
        u8  msg[128] = {};   // SHA-256 payload
        u32 rs[16] = {};   // SHA-256 output / RMD-160 input

        prepare65(msg, point);
        sha256_final(rs, msg, sizeof(msg));

        prepare_rmd(rs);
        rmd160_final(r, rs);
    }

    // ---------------------------------------------------------------------------
    // Batched (SIMD) address derivation
    // ---------------------------------------------------------------------------

    // SHA-256 for a batch: SHA-NI (one message at a time; the out-of-order
    // core overlaps the independent messages, an interleaved kernel measured
    // no faster), else eight messages at once in AVX2 registers, else the
    // portable kernel. RIPEMD-160 is rmd160s.cpp's, eight wide with AVX2.
    // Picked once; ECLOOP_HASH=shani|avx2|scalar forces a rung.
    namespace {
        enum class ShaPath { Scalar, Avx2, Shani };

        ShaPath pick_sha_path() noexcept {
            const char* h = std::getenv("ECLOOP_HASH");
            if (h && h[0]) {
                if (!std::strcmp(h, "scalar")) return ShaPath::Scalar;
                if (!std::strcmp(h, "avx2")) return sha256_avx2_available() ? ShaPath::Avx2 : ShaPath::Scalar;
                if (!std::strcmp(h, "shani")) return sha256_shani_available() ? ShaPath::Shani : ShaPath::Scalar;
            }
            if (sha256_shani_available()) return ShaPath::Shani;
            if (sha256_avx2_available()) return ShaPath::Avx2;
            return ShaPath::Scalar;
        }

        const ShaPath SHA_PATH = pick_sha_path();

        template <std::size_t MSGLEN>
        ECLOOP_ALWAYS_INLINE void sha_batch(u32 rs[HASH_BATCH_SIZE][16], const u8 msg[HASH_BATCH_SIZE][MSGLEN],
                                            std::size_t count) noexcept {
            if constexpr (HASH_BATCH_SIZE == 8) {
                if (SHA_PATH == ShaPath::Avx2 && count == 8) {
                    u32 out[8][8];
                    if constexpr (MSGLEN == 64) sha256_avx2_8x1(out, msg);
                    else sha256_avx2_8x2(out, msg);
                    for (int i = 0; i < 8; ++i) std::memcpy(rs[i], out[i], 32);
                    return;
                }
            }
            if (SHA_PATH == ShaPath::Scalar)
                for (std::size_t i = 0; i < count; ++i) sha256_final_scalar(rs[i], msg[i], MSGLEN);
            else
                for (std::size_t i = 0; i < count; ++i) sha256_final(rs[i], msg[i], MSGLEN);
        }
    } // namespace

    const char* addr_fallback_name() noexcept {
        const char* rmd = HASH_BATCH_SIZE == 8 ? (cpu_has_avx2() ? "RIPEMD-160 AVX2 x8" : "RIPEMD-160 scalar (8 lanes)")
                        : HASH_BATCH_SIZE == 4 ? "RIPEMD-160 NEON x4" : "RIPEMD-160 scalar";
        static char buf[96];
        std::snprintf(buf, sizeof buf, "%s + %s",
                      SHA_PATH == ShaPath::Shani ? "SHA-256 SHA-NI"
                      : SHA_PATH == ShaPath::Avx2 ? "SHA-256 AVX2 x8" : "SHA-256 scalar", rmd);
        return buf;
    }

    void addr33_batch(h160_t* hashes, const pe* points, std::size_t count) {
        assert(count <= HASH_BATCH_SIZE);
        // Zero-initialised on the stack; HASH_BATCH_SIZE is a small compile-time constant.
        u8  msg[HASH_BATCH_SIZE][64] = {};
        u32 rs[HASH_BATCH_SIZE][16] = {};
        for (std::size_t i = 0; i < count; ++i)
            prepare33(msg[i], points + i);
        sha_batch<64>(rs, msg, count);
        for (std::size_t i = 0; i < count; ++i) {
            rs[i][8] = 0x80000000u;  // 0x80 padding byte in little-endian word
            rs[i][14] = 0x00010000u;  // bit-length 256 in little-endian word
        }
        rmd160_batch(hashes, rs);
    }

    void addr65_batch(h160_t* hashes, const pe* points, std::size_t count) {
        assert(count <= HASH_BATCH_SIZE);
        u8  msg[HASH_BATCH_SIZE][128] = {};
        u32 rs[HASH_BATCH_SIZE][16] = {};
        for (std::size_t i = 0; i < count; ++i)
            prepare65(msg[i], points + i);
        sha_batch<128>(rs, msg, count);
        for (std::size_t i = 0; i < count; ++i) {
            rs[i][8] = 0x80000000u;  // 0x80 padding byte in little-endian word
            rs[i][14] = 0x00010000u;  // bit-length 256 in little-endian word
        }
        rmd160_batch(hashes, rs);
    }

} // namespace ecloop
