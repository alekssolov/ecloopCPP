#include "addr_avx512.hpp"

#include "rmd160_avx512.hpp"
#include "prepare_avx.hpp"

// Per-worker storage: see the note on the message buffers below.
#if defined(ECLOOP_THREADED) && ECLOOP_THREADED
#define ECLOOP_WORKER_STORAGE thread_local
#else
#define ECLOOP_WORKER_STORAGE static
#endif

#if ECLOOP_HAVE_RMD160_AVX512 && ECLOOP_HAVE_PREPARE_AVX
#define ECLOOP_HAVE_ADDR_AVX512 1
#else
#define ECLOOP_HAVE_ADDR_AVX512 0
#endif

#if ECLOOP_HAVE_ADDR_AVX512

#include <cstdio>
#include "cpu.hpp"
#include <cstdlib>
#include <cstring>

// The assembly kernel. mask is declared const void* rather than const u64*
// on purpose: this project defines u64 as unsigned long long, while
// <cstdint> on LP64 makes uint64_t an unsigned long. Same size, different
// types, and pointers to them are incompatible -- on Cygwin that is a hard
// error, not a warning.
extern "C" void sha256_x16_avx512(void* digests, void* scratch, const void* table,
                                  const void* mask, ecloop::u64 mask_len,
                                  const ecloop::u8* const* inputs);

namespace ecloop {
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl,avx2")
    namespace {

        constexpr u32 A512_IV[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        };

        constexpr u32 A512_K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
            0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
            0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
            0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        };

        // The round table in the layout the assembly kernel expects: 64
        // rounds, one zmm each, constant broadcast across all 16 lanes.
        // Written once during the probe and read-only afterwards.
        alignas(64) u32 A512_TABLE[64][16];

        // 16 prepared messages -> 16 hash160 values. nblk is the number of
        // 64-byte blocks per message (1 compressed, 2 uncompressed).
        void a512_hash160_x16(h160_t* out, const u8* const msgs[16], u64 nblk) noexcept {
            alignas(64) u32 dig[8][16];
            alignas(64) u8 scratch[512];
            u64 mask[2] = {0xFFFFull, 0xFFFFull};

            for (int w = 0; w < 8; ++w)
                for (int l = 0; l < 16; ++l) dig[w][l] = A512_IV[w];

            sha256_x16_avx512(dig, scratch, A512_TABLE, mask, nblk, msgs);

            __m512i st[8];
            for (int w = 0; w < 8; ++w) st[w] = _mm512_load_si512(dig[w]);
            rmd160_batch16_soa(reinterpret_cast<u32(*)[5]>(out), st);
        }

        // Compressed public key of private key 1 and its hash160. The
        // address 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH can be checked with
        // any third-party tool and depends on nothing inside this project.
        constexpr u8 A512_TV_PUB[33] = {
            0x02, 0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0,
            0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D,
            0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98,
        };
        // hash160 = 751E76E8199196D454941C45D1B3A323F1433BD6.
        //
        // Stored as u32 WORDS, not bytes. This project keeps h160 so that
        // print_h160 can emit the canonical hex with plain %08x, i.e.
        // h[0] == 0x751E76E8. The C original declares this vector as a
        // byte array and memcmps it, which asserts the opposite byte
        // order -- and so its self test passes on output that disagrees
        // with addr33 and addr33_batch.
        constexpr u32 A512_TV_H160[5] = {
            0x751E76E8u, 0x199196D4u, 0x54941C45u, 0xD1B3A323u, 0xF1433BD6u,
        };

        // Returns 0 on success, otherwise the failing lane + 1.
        int a512_selftest() noexcept {
            alignas(64) u8 msg[16][64];
            const u8* ptr[16];
            h160_t out[16];

            for (int i = 0; i < 16; ++i) {
                std::memset(msg[i], 0, 64);
                std::memcpy(msg[i], A512_TV_PUB, 33);
                msg[i][33] = 0x80;
                msg[i][62] = 0x01; // length 264 bits
                msg[i][63] = 0x08;
                ptr[i] = msg[i];
            }
            a512_hash160_x16(out, ptr, 1);

            for (int i = 0; i < 16; ++i)
                if (std::memcmp(out[i], A512_TV_H160, 20) != 0) return i + 1;
            return 0;
        }

#pragma GCC pop_options
        // baseline code: runs on every CPU, decides whether the kernels may
        bool a512_probe() noexcept {
            const char* e = std::getenv("ECLOOP_NO_AVX512_HASH");
            if (e && e[0] == '1') return false;
            // ECLOOP_HASH=avx512|shani|avx2|scalar forces one rung of the
            // hashing ladder (addr.cpp picks among the lower three).
            const char* h = std::getenv("ECLOOP_HASH");
            if (h && h[0] && std::strcmp(h, "avx512") != 0 && std::strcmp(h, "auto") != 0) return false;

            __builtin_cpu_init();
            if (!__builtin_cpu_supports("avx512f")) return false;
            if (!__builtin_cpu_supports("avx512bw")) return false; // vpshufb on zmm
            if (!__builtin_cpu_supports("avx512dq") || !__builtin_cpu_supports("avx512vl") || !__builtin_cpu_supports("avx2")) return false;

            for (int i = 0; i < 64; ++i)
                for (int l = 0; l < 16; ++l) A512_TABLE[i][l] = A512_K[i];

            const int fail = a512_selftest();
            if (fail) {
                std::fprintf(stderr,
                             "WARNING: AVX-512 hashing self test failed (lane %d).\n"
                             "  Falling back to the old path (SHA-NI + AVX2): results are\n"
                             "  correct but slower. Likely causes:\n"
                             "    - sha256_x16_avx512.o built for a different ABI;\n"
                             "    - it came out truncated (should be ~18 KB, not ~1 KB);\n"
                             "    - the byte swap between SHA-256 and RIPEMD-160 is done\n"
                             "      twice or not at all: try -DRMD16_NO_SWAP=1 (or drop it).\n",
                             fail - 1);
                std::fflush(stderr);
                return false;
            }
            return true;
        }

        // Reusable message buffers.
        //
        // STORAGE CLASS MATTERS HERE. This is the hottest allocation in the
        // program: one access per batch of 16 points, and the collider
        // hashes 1536 points per block. A thread_local with a non-trivial
        // constructor is not a free variable -- the compiler must emit a
        // TLS wrapper plus an initialisation guard, and calls it on EVERY
        // access. On Linux that is a cheap %fs-relative load; on
        // Cygwin/MinGW it is a real function call, and it lands right in
        // the inner loop.
        //
        // Workers here are PROCESSES created by fork(), never threads, so
        // each already has a private copy of everything and plain static is
        // both correct and free -- which is exactly why the C original used
        // static and said so.
        //
        // Build with -DECLOOP_THREADED=1 if these routines are ever called
        // from several threads of one process; a shared buffer would then
        // let one thread overwrite another's message between preparation
        // and hashing.
        struct Msg33Buf {
            alignas(64) u8 msg[16][64];
            const u8* ptr[16];
            Msg33Buf() noexcept {
                prep_pad33(msg);
                for (int i = 0; i < 16; ++i) ptr[i] = msg[i];
            }
        };

        struct Msg65Buf {
            alignas(64) u8 msg[16][128];
            const u8* ptr[16];
            Msg65Buf() noexcept {
                prep_pad65(msg);
                for (int i = 0; i < 16; ++i) ptr[i] = msg[i];
            }
        };

    } // namespace

    const bool ADDR_AVX512_OK = a512_probe();
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vl,avx2")

    void addr33_batch16(h160_t* hashes, const pe* points) noexcept {
        ECLOOP_WORKER_STORAGE Msg33Buf buf;
        for (int i = 0; i < 16; ++i) prepare33_fast(buf.msg[i], points + i);
        a512_hash160_x16(hashes, buf.ptr, 1);
    }

    void addr65_batch16(h160_t* hashes, const pe* points) noexcept {
        ECLOOP_WORKER_STORAGE Msg65Buf buf;
        for (int i = 0; i < 16; ++i) prepare65_fast(buf.msg[i], points + i);
        a512_hash160_x16(hashes, buf.ptr, 2);
    }

    // 32 points: two SHA-256 x16 passes, then one interleaved RIPEMD-160
    // x32 (rmd160_batch32_soa). Used when A512_BATCH is 32
    // (-DECLOOP_A512_BATCH32=1); see addr_avx512.hpp for when that pays.
    namespace {
        void a512_sha_x16(u32 dig[8][16], const u8* const msgs[16], u64 nblk) noexcept {
            alignas(64) u8 scratch[512];
            u64 mask[2] = {0xFFFFull, 0xFFFFull};
            for (int w = 0; w < 8; ++w)
                for (int l = 0; l < 16; ++l) dig[w][l] = A512_IV[w];
            sha256_x16_avx512(dig, scratch, A512_TABLE, mask, nblk, msgs);
        }
    } // namespace

    void addr33_batch32(h160_t* hashes, const pe* points) noexcept {
        ECLOOP_WORKER_STORAGE Msg33Buf bufa, bufb;
        for (int i = 0; i < 16; ++i) { prepare33_fast(bufa.msg[i], points + i); prepare33_fast(bufb.msg[i], points + 16 + i); }
        alignas(64) u32 da[8][16], db[8][16];
        a512_sha_x16(da, bufa.ptr, 1);
        a512_sha_x16(db, bufb.ptr, 1);
        __m512i sa[8], sb[8];
        for (int w = 0; w < 8; ++w) { sa[w] = _mm512_load_si512(da[w]); sb[w] = _mm512_load_si512(db[w]); }
        rmd160_batch32_soa(reinterpret_cast<u32(*)[5]>(hashes), sa, sb);
    }

    void addr65_batch32(h160_t* hashes, const pe* points) noexcept {
        ECLOOP_WORKER_STORAGE Msg65Buf bufa, bufb;
        for (int i = 0; i < 16; ++i) { prepare65_fast(bufa.msg[i], points + i); prepare65_fast(bufb.msg[i], points + 16 + i); }
        alignas(64) u32 da[8][16], db[8][16];
        a512_sha_x16(da, bufa.ptr, 2);
        a512_sha_x16(db, bufb.ptr, 2);
        __m512i sa[8], sb[8];
        for (int w = 0; w < 8; ++w) { sa[w] = _mm512_load_si512(da[w]); sb[w] = _mm512_load_si512(db[w]); }
        rmd160_batch32_soa(reinterpret_cast<u32(*)[5]>(hashes), sa, sb);
    }

} // namespace ecloop

#pragma GCC pop_options
#else // !ECLOOP_HAVE_ADDR_AVX512

namespace ecloop {

    const bool ADDR_AVX512_OK = false;

    void addr33_batch16(h160_t*, const pe*) noexcept { __builtin_trap(); }
    void addr33_batch32(h160_t*, const pe*) noexcept { __builtin_trap(); }
    void addr65_batch32(h160_t*, const pe*) noexcept { __builtin_trap(); }
    void addr65_batch16(h160_t*, const pe*) noexcept { __builtin_trap(); }

} // namespace ecloop

#endif // ECLOOP_HAVE_ADDR_AVX512
