// Random walk over the hex digits of a private key, checked
// against a hash160 database loaded from a flat .bin file.
//
// Workers are PROCESSES, not threads: main forks one per requested worker,
// each inherits the loaded database copy-on-write and reports progress
// through a shared mmap of counters. The parent then only prints the rate.
//
// Usage:  ecloop -b <file.bin> [-t threads] [-s bits] [-v level]
//
//   -b  database of hash160 records, 20 bytes each (see hash_to_bin.py)
//   -t  worker count, default = usable CPUs
//   -s  private key size in bits, default 256
//   -v  0 quiet, 1 rate line, 2 print every key, 5 print hashes too
//
// Environment:
//   ECLOOP_OLD_ENGINE=1     single-walker reference engine
//   ECLOOP_NO_FILTER=1      disable the bitmap prefilter
//   ECLOOP_FILTER_BITS=N    override its size
//   ECLOOP_NO_AVX512_HASH=1 old hashing path (SHA-NI + AVX2 x8)
//   ECLOOP_NO_IFMA=1        scalar fallback instead of batch-8
//
// DROPPED FROM THE C ORIGINAL, all of it unreachable: <openssl/sha.h> and
// <openssl/ripemd.h> (nothing called them -- only local variable names
// contained "SHA256"); <windows.h> and <tchar.h>, whose only real use was
// GetTickCount inside fe_random, itself never called; fe_random,
// fe_random_seeded, fast_pow_u64, add_prefix, get_difficulty,
// set_working_cpu, and the pkey_baseCH / rng / patsize / prefix_cch /
// patch globals, which were assigned but never read. Also the hand-rolled
// CPU_* macros over Cygwin internals: their only consumer was
// get_num_cpus, and set_working_cpu, the reason they were written, is
// never called. Removing all of it makes the program build anywhere, not
// just on Cygwin.

#include <sys/mman.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <thread>
#include <vector>
#include <sstream>
#include "main.hpp"
#include "BigInt.hpp"
#include "addr.hpp"
#include "addr_dispatch.hpp"
#include "btree_filter.hpp"
#include "shm.hpp"
#include "s2n_dispatch.hpp"
#include "compat.hpp"
#include "ec_delta.hpp"
#include "ec_glv.hpp"
#include "ecc.hpp"
#include "fastPRNG.h"
#include "b58.hpp"

namespace ecloop {

    // ── shared state ─────────────────────────────────────────────────────────

    int  pksize = 256;
    int  verb = 0;
    u64* thread_count = nullptr;
    u32* BTREE_HEAP = nullptr;
    u32  BTREE_HEAP_SIZE = 0;

    // ── sequential sweep ─────────────────────────────────────────────────────
    bool  seq_active = false;
    bool  range_active = false;
    bool  glv_walk = false;
    bool  endo_on = true;
    int   walk_ndig = 64;
    fe    seq_start = {0, 0, 0, 0};
    fe    seq_end = {~0ULL, ~0ULL, ~0ULL, ~0ULL};
    int   seq_threads = 1;
    u64*  seq_progress = nullptr;
    int   mix_bits = 0;
    char  seq_ckpt[256] = "progress.txt";
    std::vector<unsigned char> patv;
    int patsize = 0;
    const char* prefix_cch;
    char* patch;
    std::string mode = "";
    std::string prefix = "";
    uint8_t pat[20];
    double difficulty;

    // The expected number of keys per hit of a -m mode, from the prefix
    // difficulty D (1/D = one address carries the prefix). Used by the
    // start-up print and by the manager's probability/ETA alike:
    //   unc/com  one address per key: D
    //   any      either of two independent addresses: ~D/2
    //   unccom   the two hash160s of a key agree on patsize leading bytes:
    //            256^patsize (the content of -p is not used)
    //   triple   unccom AND the prefix on either address. Given the shared
    //            bytes -- 8 bits each, more than the ~5.9 bits a Base58
    //            character pins -- the two addresses carry the prefix
    //            together or not at all, so the factor is D, not D/2
    //            (tests/test_vanity.cpp measures P(a or b | shared) = 1/D).
    // 0 when the mode has no usable difficulty (Bin, or no -p).
    double effective_difficulty(double D) {
        if (mode == "unc" || mode == "com") return D;
        if (mode == "any") return D / 2.0;
        if (mode == "unccom" || mode == "triple") {
            int nb = patsize;
            if (nb < 0) nb = 0;
            if (nb > 20) nb = 20;
            double e = std::pow(256.0, nb);
            if (mode == "triple") e *= D;
            return e;
        }
        return 0.0;
    }

    namespace {
        int sock[2];
    } // namespace

    // Usable CPUs. The C original reimplemented the CPU_* macros by hand
    // against Cygwin's internals, which is fragile and, since
    // set_working_cpu was never actually called, bought nothing beyond this
    // count. hardware_concurrency answers the same question portably.
    int get_num_cpus() {
        const unsigned n = std::thread::hardware_concurrency();
        return n ? static_cast<int>(n) : 1;
    }
            // Called from vanity::report_found (vanity_match.cpp) -- a different
            // translation unit, hence plain ecloop:: scope here, not the anonymous
            // namespace sock itself lives in a few lines up: an anonymous-namespace
            // definition would give this internal linkage and vanity_match.cpp's
            // call would fail to link.
   
    // ── btree ────────────────────────────────────────────────────────────────
    //
    // Layout of the heap, all u32 words:
    //
    //   [0]                  next free slot
    //   [1 .. 1+2^24)        bucket heads, indexed by the low three bytes
    //                        of the hash
    //   [BTREE_HSIZE ..)     records, 6 words each: hash160 (5 words) then
    //                        the index of the next record in the bucket
    //
    // Not a tree despite the name -- separate chaining over 16.7M buckets.
    // Kept as is; renaming would only make this harder to compare against
    // the original.

    u32 btree_search(const u32* hash, const u32* tree) noexcept {
        if (!tree) tree = BTREE_HEAP;
        if (!tree) return 0;

        const u8* h = reinterpret_cast<const u8*>(hash);
        const u32 hpos = 1 + (h[0] + h[1] * 256u + h[2] * 65536u);

        u32 pos_next = tree[hpos];
        while (pos_next > 0) {
            const u32 pos = pos_next;
            pos_next = tree[pos + 5];
            if (hash[0] == tree[pos] && hash[1] == tree[pos + 1] &&
                hash[2] == tree[pos + 2] && hash[3] == tree[pos + 3] &&
                hash[4] == tree[pos + 4]) {
                return pos;
            }
        }
        return 0;
    }

    void notify_found() {
        // A single byte is enough; manager_loop only needs to know that
        // *something* arrived, not what -- the actual key was already
        // printed by report_found() itself. Best-effort: sock[1] is
        // SOCK_DGRAM (connected by socketpair() in main() below), so a lost
        // or failed write just means this one reset is missed, not an
        // error worth acting on.
        const char msg = 1;
        const ssize_t wr = write(sock[1], &msg, 1);
        (void)wr;
    }

    namespace {

        void btree_add(const u32* hash, u32* tree) noexcept {
            if (!tree) tree = BTREE_HEAP;
            if (!tree) return;
            if (btree_search(hash, tree) > 0) return; // deduplicate

            const u8* h = reinterpret_cast<const u8*>(hash);
            const u32 hpos = 1 + (h[0] + h[1] * 256u + h[2] * 65536u);

            // 20 bytes of hash160 plus 4 bytes for the next-record index
            const u32 last = tree[0];
            tree[0] = last + 6;

            tree[last + 0] = hash[0];
            tree[last + 1] = hash[1];
            tree[last + 2] = hash[2];
            tree[last + 3] = hash[3];
            tree[last + 4] = hash[4];
            tree[last + 5] = tree[hpos];
            tree[hpos] = last;
        }

        bool btree_load(const char* filename) {
            struct stat st;
            if (!filename || stat(filename, &st) == -1) {
                std::printf("BIN file [%s] not found.\n", filename ? filename : "(none)");
                return false;
            }

            const std::size_t fsize = static_cast<std::size_t>(st.st_size);
            const u32 bsize = (static_cast<u32>(std::ceil(fsize / 20.0)) + 2) * 6;
            const u32 tsize = bsize + BTREE_HSIZE;

            // Shared, not calloc'd: see shm.hpp -- under Cygwin a private
            // allocation of this size is copied into every forked worker.
            static ShmBlock heap_blk;
            shm_free(heap_blk);
            heap_blk = shm_alloc(static_cast<std::size_t>(tsize) * 4, 1u << 21);
            u32* heap = static_cast<u32*>(heap_blk.ptr);
            if (!heap) {
                std::printf("BIN file [%s]: out of memory (%u words needed).\n",
                            filename, tsize);
                return false;
            }
            heap[0] = BTREE_HSIZE;

            BTREE_HEAP = heap;
            BTREE_HEAP_SIZE = tsize * 4;

            const auto before = std::chrono::steady_clock::now();

            std::FILE* wfd = std::fopen(filename, "rb");
            if (!wfd) {
                std::printf("%s Can not open \n", filename);
                std::free(heap);
                BTREE_HEAP = nullptr;
                BTREE_HEAP_SIZE = 0;
                return false;
            }

            u32 numrec = 0;
            alignas(4) u8 buf[32] = {};
            // Looping on the result of fread rather than testing feof first:
            // feof only reports a read that already hit the end, so the
            // original ran one extra iteration on every file.
            while (std::fread(buf, 1, 20, wfd) == 20) {
                // BYTE ORDER. The file holds hash160 in its canonical form,
                // most significant byte first: the first record of test.bin
                // is 91b24bf9f5288532960ac687abb035127b1d28a5, the hash160
                // of the uncompressed key 1 (1EHNa6Q4Jz2uvNExL497mE43ikXhwF6kZm).
                //
                // addr33/addr65 return h[0] == 0x91b24bf9, so that
                // print_h160 can emit the canonical hex with plain %08x. On
                // a little-endian machine reading those same bytes as u32
                // gives 0xf94bb291 instead -- the reverse. The C original
                // cast the buffer straight to u32*, so every record went in
                // byte-reversed and no lookup could ever match. That is why
                // even 24-bit keys were never found.
                //
                // One swap per word on load, which happens once, instead of
                // touching the hot path.
                u32 rec[5];
                for (int w = 0; w < 5; ++w) {
                    u32 v;
                    std::memcpy(&v, buf + w * 4, 4);
                    rec[w] = detail::bswap32(v);
                }
                btree_add(rec, heap);
                if (btree_search(rec, heap) == 0) std::printf("Bad algorithm\n");
                else ++numrec;
            }
            std::fclose(wfd);

            const double secs =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - before)
                    .count();

            std::printf("Downloading of BIN file %s is completed, ", filename);
            std::printf("loaded %u records, %01.3f sec.\n", numrec, secs);
            std::fflush(stdout);
            return true;
        }

        // ── manager ──────────────────────────────────────────────────────────

        void manager_loop(int threads)
        {
            fd_set readset;
            struct timeval tv = { 1, 0 };
            int ret;
            char* unit;
            BigInt total;
            BigInt totalold;
            // Snapshot of total at the moment of the most recent find, so
            // the probability display measures "since the last find" like
            // vanitygen's last_result, rather than accumulating from
            // program start forever. Stays 0 until the first find.
            BigInt total_at_last_find = 0;
            char* spd;
            char* spd3;
            char* spd4;
            char* spd5;
            char* spd6;
            char* spd7;
            char* spd8;
            char* spd9;
            char* spd10;
            char* spd11;
            char* spd12;
            char* frc;
            
            std::string str, str3, str4, str5, str6, str7, str8, str9, str10, str11, str12, str13, totalstr, GETstr;
                            // ── vanity probability / ETA ─────────────────────────────────────────
                            // Adapted from vanitygen's prob() (its line ~293: prob=(1-exp(count/
                            // -difficulty))*100, then inverting the same formula to find how many
                            // more keys reach each target). get_difficulty() is b58.cpp's exact
                            // figure for the registered -p prefix; the three cases below turn it
                            // (or, for unccom, patsize alone) into the ONE number this formula
                            // needs: the expected key count for THIS mode's actual pass condition,
                            // not just for a single hash160 landing in the prefix's range.
                            //
                            //   unc/com  -- exactly what get_difficulty() already means: one
                            //               hash160 checked per key.
                            //   any      -- two independent hash160 forms checked per key, each
                            //               with get_difficulty()'s odds, so a hit is roughly
                            //               twice as likely per key: P(hit) = 1-(1-p)^2 ~ 2p for
                            //               p = 1/get_difficulty() this small, i.e. difficulty/2.
                            //   unccom   -- content of -p is unused in this mode (only its
                            //               length sets patsize), so get_difficulty() would be
                            //               measuring the wrong thing or, since add_prefix is
                            //               skipped for unccom, would just read 0. The actual
                            //               condition is nbytes leading bytes of two otherwise
                            //               unrelated hash160 values agreeing by chance: 256 per
                            //               byte position, so 256^nbytes.
                            //   triple   -- unccom's byte-agreement AND any's prefix match,
                            //               treated as independent (they test unrelated bit
                            //               ranges of the hash outputs), so the two expected
                            //               counts multiply.
                            //   (empty)  -- BIN search: no equivalent notion of difficulty is
                            //               tracked, so this whole feature stays off, same as
                            //               patsize == 0 always did before this.
                double eff_difficulty = effective_difficulty(get_difficulty());
                            // get_difficulty() returns 0 if add_prefix was never called (no -p,
                            // or -p skipped for unccom) and +inf for an unreachable prefix
                            // (caught earlier, at startup, but guarded again here defensively);
                            // neither is a usable difficulty for this display.
            if (!(eff_difficulty > 0.0) || std::isinf(eff_difficulty)) eff_difficulty = 0.0;
            
            static const int prob_targets[] = { 50, 75, 80, 90, 95 };
            static const long prob_units[] = { 31536000, 86400, 3600, 60, 1 };
            static const char prob_units_str[] = "ydhms";
            constexpr int PROB_NT = sizeof(prob_targets) / sizeof(prob_targets[0]);
            constexpr int PROB_NU = sizeof(prob_units) / sizeof(prob_units[0]);
            

            FD_ZERO(&readset);

            // The end of the FIRST interval, one second out, not "now":
            // with next == now, the "next += 1s" after the first sleep landed
            // in the past, the resync set the timeout to zero and the second
            // tick fired at once, printing a 0.0 Keys/Sec line at start-up.
            auto next = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            // When the counters were last read: the speed figure is the
            // delta divided by the time that actually passed, not by the
            // nominal second (wake-up latency, an early wake on a find, or
            // falling behind would otherwise all show up as speed).
            auto last_tick = std::chrono::steady_clock::now();
            BigInt targ3 = 0, targ4 = 0, targ5 = 0, targ6 = 0, targ7 = 0, targ8 = 0, targ9 = 0, targ10 = 0, targ11 = 0, targ12 = 0, frac = 0, tl;
            while (1)
            {
                // Sleep until the next scheduled tick. A find (notify_found
                // writes a byte to the socketpair) wakes select() early: the
                // counters are folded in at once so the probability baseline
                // is exact, but nothing is printed and the schedule is kept --
                // otherwise an easy prefix, with thousands of finds a second,
                // turns the status line into noise measured over microseconds.
                {
                    auto rem = next - std::chrono::steady_clock::now();
                    if (rem < std::chrono::seconds(0)) rem = std::chrono::seconds(0);
                    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(rem).count();
                    tv.tv_sec = static_cast<long>(us / 1000000);
                    tv.tv_usec = static_cast<long>(us % 1000000);
                }
                // Re-armed every iteration: select() consumes readset, so a
                // stale bit from a previous call cannot linger into this one.
                FD_ZERO(&readset);
                FD_SET(sock[0], &readset);

                if ((ret = select(sock[0] + 1, &readset, NULL, NULL, &tv)) == -1) perror("select");
                bool just_found = false;
                if (ret > 0 && FD_ISSET(sock[0], &readset)) {
                    just_found = true;
                    char drain[64];
                    while (recv(sock[0], drain, sizeof(drain), MSG_DONTWAIT) > 0) {}
                    // Atomic read-and-clear: the worker may add to its counter
                    // between a plain read and the store of 0, and that group
                    // (2048-3072 keys) would be lost.
                    for (int i = 0; i < threads; i++)
                        total += std::to_string(__atomic_exchange_n(&thread_count[i], (u64)0, __ATOMIC_RELAXED));
                    total_at_last_find = total;
                }
                if (std::chrono::steady_clock::now() < next) continue;   // early wake, tick not due

                next += std::chrono::seconds(1);
                if (next < std::chrono::steady_clock::now())   // fell behind: resync
                    next = std::chrono::steady_clock::now() + std::chrono::seconds(1);
                
                const auto tick_now = std::chrono::steady_clock::now();
                const double elapsed = std::chrono::duration<double>(tick_now - last_tick).count();
                last_tick = tick_now;
                for (int i = 0; i < threads; i++)
                    total += std::to_string(__atomic_exchange_n(&thread_count[i], (u64)0, __ATOMIC_RELAXED));
                (void)just_found;
                BigInt targ = total - totalold;
                if (elapsed > 0.0) {
                    // Per-second figure over the measured interval.
                    std::stringstream ds;
                    ds << targ;
                    const double per_sec = std::stod(ds.str()) / elapsed;
                    char buf[40];
                    std::snprintf(buf, sizeof buf, "%.0f", per_sec);
                    targ = BigInt(std::string(buf));
                }
                std::stringstream totalstgstr;
                totalstgstr << targ << std::endl;
                totalstr = totalstgstr.str();
                totalstr.erase(totalstr.find_last_not_of(" \n\r\t") + 1);
                //speed = std::stod(totalcr);
                unit = (char*)"Keys/Sec";
                frac = 0;   // otherwise a tick below 1000 keys shows the previous fraction digit
                if (targ > "1000")
                {
                    unit = (char*)"KiloKeys/Sec";
                    frac = targ % "1000";
                    targ /= "1000";
                    if (targ > "1000")
                    {
                        unit = (char*)"MegaKeys/Sec";
                        frac = targ % "1000";
                        targ /= "1000";
                        if (targ > "1000")
                        {
                            unit = (char*)"GigaKeys/Sec";
                            frac = targ % "1000";
                            targ /= "1000";
                            if (targ > "1000")
                            {
                                unit = (char*)"TeraKeys/Sec";
                                frac = targ % "1000";
                                targ /= "1000";
                                if (targ > "1000")
                                {
                                    unit = (char*)"PetaKeys/Sec";
                                    frac = targ % "1000";
                                    targ /= "1000";
                                    if (targ > "1000")
                                    {
                                        unit = (char*)"ExaKeys/Sec";
                                        frac = targ % "1000";
                                        targ /= "1000";
                                        if (targ > "1000")
                                        {
                                            unit = (char*)"ZetaKeys/Sec";
                                            frac = targ % "1000";
                                            targ /= "1000";
                                            if (targ > "1000")
                                            {
                                                unit = (char*)"YotaKeys/Sec";
                                                frac = targ % "1000";
                                                targ /= "1000";
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (total >= "1000" && total < "1000000")
                {
                    targ7 = 0;
                    targ6 = 0;
                    targ5 = 0;
                    targ4 = 0;
                    targ3 = 0;
                }
                if (total >= "1000000" && total < "1000000000")
                {
                    targ7 = 0;
                    targ6 = 0;
                    targ5 = 0;
                    targ4 = 0;
                    targ3 = total / "1000000";
                }
                if (total >= "1000000000" && total < "1000000000000")
                {
                    targ7 = 0;
                    targ6 = 0;
                    targ5 = 0;
                    targ4 = total / "1000000000";
                    targ3 = (total % "1000000000") / "1000000";
                }
                if (total >= "1000000000000" && total < "1000000000000000")
                {
                    targ7 = 0;
                    targ6 = 0;
                    targ5 = total / "1000000000000";
                    targ4 = (total % "1000000000000") / "1000000000";
                    targ3 = (total % "1000000000") / "1000000";
                }
                if (total >= "1000000000000000" && total < "1000000000000000000")
                {
                    targ7 = 0;
                    targ6 = total / "1000000000000000";
                    targ5 = (total % "1000000000000000") / "1000000000000";
                    targ4 = (total % "1000000000000") / "1000000000";
                    targ3 = (total % "1000000000") / "1000000";
                }
                if (total > "1000000000000000000" && total < "1000000000000000000000")
                {
                    targ7 = total / "1000000000000000000";
                    targ6 = (total % "1000000000000000000") / "1000000000000000";
                    targ5 = (total % "1000000000000000") / "1000000000000";
                    targ4 = (total % "1000000000000") / "1000000000";
                    targ3 = (total % "1000000000") / "1000000";
                }
                if (total > "1000000000000000000000" && total < "1000000000000000000000000")
                {
                    targ8 = total / "1000000000000000000000";
                    targ7 = (total % "1000000000000000000000") / "1000000000000000000";
                    targ6 = (total % "1000000000000000000") / "1000000000000000";
                    targ5 = (total % "1000000000000000") / "1000000000000";
                    targ4 = (total % "1000000000000") / "1000000000";
                    targ3 = (total % "1000000000") / "1000000";
                }
                if (total > "1000000000000000000000000" && total < "1000000000000000000000000000")
                {
                    targ9 = total / "1000000000000000000000000";
                    targ8 = (total % "1000000000000000000000000") / "1000000000000000000000";
                    targ7 = (total % "1000000000000000000000") / "1000000000000000000";
                    targ6 = (total % "1000000000000000000") / "1000000000000000";
                    targ5 = (total % "1000000000000000") / "1000000000000";
                    targ4 = (total % "1000000000000") / "1000000000";
                    targ3 = (total % "1000000000") / "1000000";
                }
                if (total > "1000000000000000000000000000" && total < "1000000000000000000000000000000")
                {
                    targ10 = total / "1000000000000000000000000000";
                    targ9 = (total % "1000000000000000000000000000") / "1000000000000000000000000";
                    targ8 = (total % "1000000000000000000000000") / "1000000000000000000000";
                    targ7 = (total % "1000000000000000000000") / "1000000000000000000";
                    targ6 = (total % "1000000000000000000") / "1000000000000000";
                    targ5 = (total % "1000000000000000") / "1000000000000";
                    targ4 = (total % "1000000000000") / "1000000000";
                    targ3 = (total % "1000000000") / "1000000";
                }
                if (total > "1000000000000000000000000000000" && total < "1000000000000000000000000000000000")
                {
                    targ11 = total / "1000000000000000000000000000000";
                    targ10 = (total % "1000000000000000000000000000000") / "1000000000000000000000000000";
                    targ9 = (total % "1000000000000000000000000000") / "1000000000000000000000000";
                    targ8 = (total % "1000000000000000000000000") / "1000000000000000000000";
                    targ7 = (total % "1000000000000000000000") / "1000000000000000000";
                    targ6 = (total % "1000000000000000000") / "1000000000000000";
                    targ5 = (total % "1000000000000000") / "1000000000000";
                    targ4 = (total % "1000000000000") / "1000000000";
                    targ3 = (total % "1000000000") / "1000000";
                }
                if (total > "1000000000000000000000000000000000" && total < "1000000000000000000000000000000000000")
                {
                    targ12 = total / "1000000000000000000000000000000000";
                    targ11 = (total % "1000000000000000000000000000000000") / "1000000000000000000000000000000";
                    targ10 = (total % "1000000000000000000000000000000") / "1000000000000000000000000000";
                    targ9 = (total % "1000000000000000000000000000") / "1000000000000000000000000";
                    targ8 = (total % "1000000000000000000000000") / "1000000000000000000000";
                    targ7 = (total % "1000000000000000000000") / "1000000000000000000";
                    targ6 = (total % "1000000000000000000") / "1000000000000000";
                    targ5 = (total % "1000000000000000") / "1000000000000";
                    targ4 = (total % "1000000000000") / "1000000000";
                    targ3 = (total % "1000000000") / "1000000";
                }
                std::stringstream buff, buff3, buff4, buff5, buff6, buff7, buff8, buff9, buff10, buff11, buff12, buff13, GetExpTime;
                buff << targ << std::endl;
                buff3 << targ3 << std::endl;
                buff4 << targ4 << std::endl;
                buff5 << targ5 << std::endl;
                buff6 << targ6 << std::endl;
                buff7 << targ7 << std::endl;
                buff8 << targ8 << std::endl;
                buff9 << targ9 << std::endl;
                buff10 << targ10 << std::endl;
                buff11 << targ11 << std::endl;
                buff12 << targ12 << std::endl;
                buff13 << frac << std::endl;
                str = buff.str();
                str3 = buff3.str();
                str4 = buff4.str();
                str5 = buff5.str();
                str6 = buff6.str();
                str7 = buff7.str();
                str8 = buff8.str();
                str9 = buff9.str();
                str10 = buff10.str();
                str11 = buff11.str();
                str12 = buff12.str();
                str13 = buff13.str();
                str.erase(str.find_last_not_of(" \n\r\t") + 1);
                str3.erase(str3.find_last_not_of(" \n\r\t") + 1);
                str4.erase(str4.find_last_not_of(" \n\r\t") + 1);
                str5.erase(str5.find_last_not_of(" \n\r\t") + 1);
                str6.erase(str6.find_last_not_of(" \n\r\t") + 1);
                str7.erase(str7.find_last_not_of(" \n\r\t") + 1);
                str8.erase(str8.find_last_not_of(" \n\r\t") + 1);
                str9.erase(str9.find_last_not_of(" \n\r\t") + 1);
                str10.erase(str10.find_last_not_of(" \n\r\t") + 1);
                str11.erase(str11.find_last_not_of(" \n\r\t") + 1);
                str12.erase(str12.find_last_not_of(" \n\r\t") + 1);
                str13.erase(str13.find_last_not_of(" \n\r\t") + 1);
                str13.erase(str13.begin() + 1, str13.end());
                spd = const_cast<char*>(str.c_str());
                spd3 = const_cast<char*>(str3.c_str());
                spd4 = const_cast<char*>(str4.c_str());
                spd5 = const_cast<char*>(str5.c_str());
                spd6 = const_cast<char*>(str6.c_str());
                spd7 = const_cast<char*>(str7.c_str());
                spd8 = const_cast<char*>(str8.c_str());
                spd9 = const_cast<char*>(str9.c_str());
                spd10 = const_cast<char*>(str10.c_str());
                spd11 = const_cast<char*>(str11.c_str());
                spd12 = const_cast<char*>(str12.c_str());
                frc = const_cast<char*>(str13.c_str());
               
                if (verb == 1)
                {
                    printf("\r%s.%1s %s, %s Dec %s Non %s Oct %s Sep %s Sex %s Qui %s Qua %s Trl %s Mld %s Mln", spd, frc, unit, spd12, spd11, spd10, spd9, spd8, spd7, spd6, spd5, spd4, spd3);
                    if (eff_difficulty > 0.0)
                    {
                        // total: grand total since manager_loop started, for
                              // count in the formula. totalstr (set above from targ)
                              // is this interval's delta, i.e. the current speed.
                        // Keys since the last find (or since start, if
                        // total_at_last_find is still its initial 0) --
                        // resetting this is the whole point of notify_found.
                        // totalstr (set above from targ) is this interval's
                        // delta, i.e. the current speed, and is unaffected
                        // by the reset.
                        const BigInt since_find = total - total_at_last_find;

                        std::stringstream ttlbuf;
                        ttlbuf << since_find;
                        std::string ttlstr = ttlbuf.str();
                        ttlstr.erase(ttlstr.find_last_not_of(" \n\r\t") + 1);

                        double count_dbl = 0.0, avg_dbl = 0.0;
                        try { count_dbl = std::stod(ttlstr); }
                        catch (...) {}
                        try { avg_dbl = std::stod(totalstr); }
                        catch (...) {}

                        const double prob = (1.0 - std::exp(count_dbl / -eff_difficulty)) * 100.0;
                        if (prob < 99.95)
                            printf(", Prob %.1f%%", prob);

                        if (avg_dbl > 0.0 && prob < prob_targets[PROB_NT - 1])
                        {
                            int ti = 0;
                            while (ti < PROB_NT && prob >= prob_targets[ti]) ++ti;
                            if (ti < PROB_NT)
                               {
                                double secs = (-eff_difficulty * std::log(1.0 - prob_targets[ti] / 100.0) - count_dbl) / avg_dbl;
                                int uj = 0;
                                while (uj < PROB_NU - 1 && secs < static_cast<double>(prob_units[uj])) ++uj;
                                secs /= static_cast<double>(prob_units[uj]);
                                if (secs >= 1e8) printf(", %d%% in %e%c", prob_targets[ti], secs, prob_units_str[uj]);
                                else printf(", %d%% in %.1f%c", prob_targets[ti], secs, prob_units_str[uj]);
                               }
                        } // if (avg_dbl > 0.0
                    } // if (eff_difficulty > 0.0)
                } // if (verb == 1)
                fflush(stdout);
                totalold = total;

                // A sequential sweep has an end: each worker publishes ~0 in
                // seq_progress once it has passed -e (after adding its last
                // group to the counter -- x86 keeps those two stores in
                // order for the reader, so the counters read after the
                // check are complete). Nothing else ends, so this is the
                // only exit: the walk and the mixed mode run until killed.
                if (seq_active) {
                    bool all_done = true;
                    for (int i = 0; i < threads && all_done; i++)
                        if (seq_progress[i] != ~0ULL) all_done = false;
                    if (all_done) {
                        for (int i = 0; i < threads; i++)
                            total += std::to_string(__atomic_exchange_n(&thread_count[i], (u64)0, __ATOMIC_RELAXED));
                        std::stringstream ts;
                        ts << total;
                        seq_checkpoint(~0ULL);   // resume point: the first key after -e
                        std::printf("\nSweep finished: %s keys checked. %s now holds the first key after -e.\n",
                                    ts.str().c_str(), seq_ckpt);
                        std::fflush(stdout);
                        return;
                    }
                }
            }

        }

    } // namespace


} // namespace ecloop

// ── entry point ──────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    ecloop::ec_dispatch_init();   // asm or portable field code (universal build)
    using namespace ecloop;

    const int ncpus = get_num_cpus();
    int threads = ncpus;
    const char* bin_file = nullptr;
    const char* start_hex = nullptr;
    const char* end_hex = nullptr;
    int opt;
    char* dif3;
    char* dif4;
    char* dif5;
    char* dif6;
    char* dif7;
    char* dif8;
    char* dif9;
    char* dif10;
    char* dif11;
    char* dif12;
    char* dif13;
    char* dif14;
    char* dif15;
    char* dif16;
    char* dif17;
    BigInt Diff;
    BigInt trg3 = 0, trg4 = 0, trg5 = 0, trg6 = 0, trg7 = 0, trg8 = 0, trg9 = 0, trg10 = 0, trg11 = 0, trg12 = 0, trg13 = 0, trg14 = 0, trg15 = 0, trg16 = 0, trg17 = 0;
    std::stringstream buf, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17;
    std::string st, st3, st4, st5, st6, st7, st8, st9, st10, st11, st12, st13, st14, st15, st16, st17;

    bool s_given = false;
    int endo_mode = -1;   // -g: -1 auto, 0 off, 1 on
    while ((opt = getopt(argc, argv, "a:b:m:t:k:s:p:v:e:z:g:")) != -1) {
        switch (opt) {
        case 'b':
            bin_file = optarg;
            break;
        case 'm':
            mode = std::string(optarg);
            break;
        case 't':
            threads = std::atoi(optarg);
            if (threads <= 0) threads = ncpus;
            break;
        case 's':
            // The original wrote `if (pksize < 1 && pksize > 32)`, which is
            // never true, so the clamp did nothing and -s 0 produced a
            // zero-digit walk. Both bounds are enforced here, and the upper
            // one is 256 because pksize is in BITS: -s 32 would have capped
            // every key at 8 hex digits.
            pksize = std::atoi(optarg);
            if (pksize < 4) pksize = 4;
            if (pksize > 256) pksize = 256;
            pksize -= pksize % 4; // whole hex digits
            s_given = true;
            break;
        case 'p':
            prefix = std::string(optarg);
            if (prefix.data()[0] != '1') prefix.insert(0, (const char*)"1");
            patsize = prefix.size();
            break;
        case 'v':
            verb = std::atoi(optarg);
            break;
        case 'k':   // first key of the sweep; giving it selects the sweep
            start_hex = optarg;
            break;
        case 'e':   // last key of the sweep, inclusive
            end_hex = optarg;
            break;
        case 'z':   // low bits swept in full after every draw of the walker
            mix_bits = std::atoi(optarg);
            break;
        case 'g':   // the walk inside the -k/-e range; value = endomorphism on/off/auto
            glv_walk = true;
            if (!std::strcmp(optarg, "on") || !std::strcmp(optarg, "1")) endo_mode = 1;
            else if (!std::strcmp(optarg, "off") || !std::strcmp(optarg, "0")) endo_mode = 0;
            else if (!std::strcmp(optarg, "auto")) endo_mode = -1;
            else {
                std::fprintf(stderr, "-g: expected on, off or auto, got [%s]\n", optarg);
                return 1;
            }
            break;
        default:
            // getopt has already printed what is wrong (unknown option or
            // a missing value); running on with a half-parsed command line
            // would search something other than what was asked.
            std::fprintf(stderr, "usage: see %s without arguments\n", argv[0]);
            return 1;
        }
    }

    if (!bin_file) {
        std::fprintf(stderr,
                     "usage: %s -b <hash160.bin> [-t threads] [-v level] <EC mode> [-m mode -p prefix]\n"
                     "  -s <bits>            endomorphism walk over keys below 2^bits (-s 256: all)\n"
                     "  -s <bits> -z <low>   mixed: random high part, sweep of the low <low> bits\n"
                     "  -k <hex> -e <hex>    sequential sweep of the range\n"
                     "  -k <hex> -e <hex> -z <low>  mixed mode inside the range\n"
                     "  -k <hex> -e <hex> -g auto|on|off  walk inside the range; on/off: the\n"
                     "                       endomorphism (auto: off, the images fall outside)\n"
                     "  -s <bits> -g on|off  same choice for the walk below 2^bits (auto: on at 254+)\n"
                     "  -s and -k/-e exclude each other, so do -g and -z\n",
                     argv[0]);
        return 1;
    }

    if (!btree_load(bin_file)) return 1;
    btf_build(BTREE_HEAP);

    // Which hashing kernels this run uses, so a CPU without AVX-512 can be
    // seen to land on SHA-NI or AVX2 and not on the portable code.
#if defined(ECLOOP_UNIVERSAL) && ECLOOP_UNIVERSAL
    std::printf("EC field: %s (universal build)\n", EC_ASM_OK ? "s2n-bignum assembly (mulx/adcx)" : "portable C++ (no BMI2/ADX on this CPU)");
#elif defined(ECLOOP_PORTABLE_S2N) && ECLOOP_PORTABLE_S2N
    std::printf("EC field: portable C++ (s2n_portable.cpp, no BMI2/ADX)\n");
#else
    std::printf("EC field: %s\n", addr_avx512_available()
        ? "AVX-512IFMA, 8 lanes per batch."
        : addr_fallback_name());
#endif
    std::printf("Hashing: %s\n", addr_avx512_available()
                                     ? "AVX-512 x16 (SHA-256 + RIPEMD-160), 16 keys per batch."
                                     : addr_fallback_name());

    // Two arrays in one shared mapping: per-worker key counters, then the
    // block index each worker publishes for the sweep checkpoint.
    thread_count = static_cast<u64*>(mmap(nullptr, 2 * threads * sizeof(u64),
                                          PROT_READ | PROT_WRITE,
                                          MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    if (thread_count == MAP_FAILED) {
        std::perror("mmap");
        return 1;
    }

    // ── which EC mode ─────────────────────────────────────────────────────────
    //   -k/-e            sequential sweep of the range
    //   -k/-e -z N       mixed mode inside the range: random high part, sweep of the low N bits
    //   -k/-e -g         endomorphism walk inside the range
    //   -s N             endomorphism walk over keys below 2^N (as before)
    //   -s N -z M        mixed mode over keys below 2^N (as before)
    // -s and -k/-e bound the key space in two different ways, so they are
    // mutually exclusive; -g and -z are two different things to do with a
    // range, so are they. -m/-p (vanity) combine with any of the above.
    if (end_hex && !start_hex) {
        std::fprintf(stderr, "-e needs -k: a range starts with its first key.\n");
        return 1;
    }
    if (start_hex && s_given) {
        std::fprintf(stderr, "-s and -k/-e are mutually exclusive: -s bounds the key by its bit "
                             "length, -k/-e by a range. Use one of them.\n");
        return 1;
    }
    if (glv_walk && mix_bits > 0) {
        std::fprintf(stderr, "-g and -z are mutually exclusive: -g is the endomorphism walk, "
                             "-z the mixed mode. Use one of them.\n");
        return 1;
    }
    // The endomorphism images of a key are (pseudo-)random over the whole
    // group: below 2^N they land with probability 2^N/n each, i.e. never
    // unless N is 256. So they are worth hashing only when every key is a
    // target (a database search over the whole space); inside a bound they
    // cost two hashings per point for a third of the throughput on the
    // keys actually searched. Automatic: on for -s 256 (and -s 255/254,
    // where half/a quarter of the images still land inside), off below
    // that and inside a -k/-e range; -g on/off overrides.
    const bool walk_selected = !(start_hex && mix_bits == 0 && !glv_walk) && mix_bits == 0;
    if (endo_mode == -1) endo_on = !start_hex && pksize >= 254;
    else endo_on = (endo_mode == 1);
    if (walk_selected) {
        if (endo_on && (start_hex || pksize < 256))
            std::printf("Endomorphism: on -- 3 keys per point, but the two images of a key are "
                        "spread over the whole key space, so about %s of the keys checked fall "
                        "outside the bound (fine for a database search, not for covering the "
                        "bound; -g off turns it off).\n", pksize >= 254 && !start_hex ? "half or more" : "2/3");
        else if (!endo_on)
            std::printf("Endomorphism: off -- %s. -g on turns it on.\n",
                        endo_mode == 0 ? "by -g off"
                                       : "the images of a key never fall inside this bound");
        else
            std::printf("Endomorphism: on, 3 keys per point.\n");
    }

    // ── mixed mode checks ────────────────────────────────────────────────────
    // -z is a whole number of hex digits; the walker needs at least one digit
    // of its own above the swept zone. With -s the bound is pksize; with a
    // range it is checked below, once the range is known. Anything wrong here
    // is reported and the mode is dropped rather than guessed at, because a
    // silently narrowed search looks exactly like a working one.
    if (mix_bits > 0) {
        if (mix_bits < 4) {
            std::fprintf(stderr, "WARNING: -z %d is less than one hex digit, there is "
                                 "nothing to sweep. Mixed mode off.\n", mix_bits);
            mix_bits = 0;
        } else if (mix_bits % 4) {
            const int r = mix_bits - mix_bits % 4;
            std::fprintf(stderr, "WARNING: -z %d is not a whole number of hex digits, "
                                 "using %d.\n", mix_bits, r);
            mix_bits = r;
        }
        if (mix_bits > 0 && !start_hex && mix_bits >= pksize - 3) {
            std::fprintf(stderr, "WARNING: -z %d leaves no room under -s %d for the walk "
                                 "to move. Mixed mode off.\n", mix_bits, pksize);
            mix_bits = 0;
        }
        if (mix_bits > 0 && mix_bits < 11)
            std::fprintf(stderr, "NOTE: the sweep works in groups of 2048 keys, so every draw covers "
                                 "2^11 keys even with -z %d.\n", mix_bits);
        if (mix_bits >= 44) {
            std::fprintf(stderr, "NOTE: -z %d means %.3g keys per draw. At 30 M/s one "
                                 "sweep alone takes over a year.\n",
                         mix_bits, std::pow(2.0, mix_bits));
        }
        if (mix_bits > 0 && !start_hex)
            std::printf("Mixed mode: walk over the top %d bits, sweep of the low %d\n",
                        pksize - mix_bits, mix_bits);
    }

    seq_progress = thread_count + threads;
    seq_threads = threads;
    walk_ndig = pksize / 4;

    // A start key gives the range. Everything is settled here, before fork,
    // so every worker inherits the same range.
    if (start_hex) {
        auto to_fe = [](const char* hex, fe out) -> bool {
            const std::size_t n = std::strlen(hex);
            if (n == 0 || n > 64) return false;
            char buf[65];
            std::memset(buf, '0', 64);
            buf[64] = 0;
            std::memcpy(buf + 64 - n, hex, n);
            fe_modn_from_hex(out, buf);
            return true;
        };
        if (!to_fe(start_hex, seq_start)) {
            std::fprintf(stderr, "-k: 1 to 64 hex digits expected\n");
            return 1;
        }
        if (end_hex && !to_fe(end_hex, seq_end)) {
            std::fprintf(stderr, "-e: 1 to 64 hex digits expected\n");
            return 1;
        }
        if (fe_cmp(seq_start, seq_end) > 0) {
            std::fprintf(stderr, "-k is above -e: the range is empty\n");
            return 1;
        }
        range_active = true;
        seq_active = (mix_bits == 0) && !glv_walk;

        // The walk and the mixed mode step through the OFFSET from -k as a
        // hex string: as many digits as the offset of -e has.
        fe span;
        fe_modn_sub(span, seq_end, seq_start);   // exact: seq_end >= seq_start, both < n
        int bits = 0;
        for (int i = 3; i >= 0 && bits == 0; --i)
            if (span[i]) bits = 64 * i + 64 - __builtin_clzll(span[i]);
        int rdig = (bits + 3) / 4;
        if (rdig < 1) rdig = 1;

        if (glv_walk) {
            if (bits < 8) {
                std::fprintf(stderr, "-g: the range holds fewer than 256 keys, too small for a "
                                     "walk; use the sequential sweep.\n");
                return 1;
            }
            walk_ndig = rdig;
            std::printf("Walk in %s .. %s (%d hex digits of offset)\n",
                        start_hex, end_hex ? end_hex : "end of the key space", rdig);
        } else if (mix_bits > 0) {
            // At least one hex digit of offset above the swept zone.
            if (bits <= mix_bits) {
                std::fprintf(stderr, "-z %d: the range %s .. %s is not larger than one swept "
                                     "block of 2^%d keys; sweep it sequentially instead.\n",
                             mix_bits, start_hex, end_hex ? end_hex : "end", mix_bits);
                return 1;
            }
            walk_ndig = rdig;
            std::printf("Mixed mode in %s .. %s: random high part (%d hex digits), sweep of "
                        "the low %d bits\n", start_hex,
                        end_hex ? end_hex : "end of the key space", rdig - mix_bits / 4, mix_bits);
        } else {
            // The progress file is named after the start key, so several ranges
            // can be swept side by side without overwriting each other, and the
            // file belonging to a run is obvious from its name.
            char clean[80];
            std::size_t c = 0;
            for (const char* p = start_hex; *p && c < sizeof(clean) - 1; ++p)
                if (std::isxdigit(static_cast<unsigned char>(*p)))
                    clean[c++] = static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
            clean[c] = 0;
            std::snprintf(seq_ckpt, sizeof(seq_ckpt), "progress_%s.txt", clean);

            std::printf("Sequential sweep %s .. %s, progress in %s\n",
                        start_hex, end_hex ? end_hex : "end of the key space", seq_ckpt);
        }
        std::fflush(stdout);
    }

    if (socketpair(AF_UNIX, SOCK_DGRAM, 0, sock)) {
        std::perror("socketpair");
        return 1;
    }

    std::signal(SIGPIPE, SIG_IGN);
    std::signal(SIGCHLD, SIG_IGN);

    const pid_t parent_pid = getpid();

    ec_gtable_init();       // once per program, before any ec_gtable_mul
    delta_init(walk_ndig);  // deltas depend on the digit count (offset digits in a range)
    if (prefix != "")
    {
        // WITHOUT THIS LINE the pattern set is empty, so get_difficulty()
        // returns 1.0/0.0 == inf, to_string gives "inf", and BigInt Diff =
        // "inf" takes the process down. add_prefix is what actually turns the
        // prefix into the hash160 ranges the difficulty is computed over; it
        // prints its own message when the prefix has a character outside the
        // base58 alphabet.
        if (!add_prefix(prefix.c_str())) return 1;
        // Still infinite with patterns present means no address with this
        // prefix exists at all -- past the 29th character the tail is decided
        // by the four checksum bytes, which a prefix cannot control. Not an
        // overflow, and not something a longer run would fix.
        difficulty = get_difficulty();
        if (std::isinf(difficulty) && mode != "unccom") {
            std::fprintf(stderr, "Prefix [%s]: no such address exists, nothing to search for.\n", prefix.c_str());
            return 1;
        }
        // Printed as the expected number of keys per hit for THIS mode, the
        // same figure the probability display divides by.
        if (!mode.empty()) difficulty = effective_difficulty(difficulty);
        std::string sg = std::to_string(round(ceil(difficulty)));
        sg.erase(sg.find_last_not_of("000000") + 1);
        sg.erase(sg.find_last_not_of(" .\n\r\t") + 1);
        Diff = const_cast<char*>(sg.c_str());

        if (Diff >= "1000" && Diff < "1000000")
        {
            trg7 = 0;
            trg6 = 0;
            trg5 = 0;
            trg4 = 0;
            trg3 = 0;
        }
        if (Diff >= "1000000" && Diff < "1000000000")
        {
            trg7 = 0;
            trg6 = 0;
            trg5 = 0;
            trg4 = 0;
            trg3 = Diff / "1000000";
        }
        if (Diff >= "1000000000" && Diff < "1000000000000")
        {
            trg7 = 0;
            trg6 = 0;
            trg5 = 0;
            trg4 = Diff / "1000000000";
            trg3 = (Diff % "1000000000") / "1000000";
        }
        if (Diff >= "1000000000000" && Diff < "1000000000000000")
        {
            trg7 = 0;
            trg6 = 0;
            trg5 = Diff / "1000000000000";
            trg4 = (Diff % "1000000000000") / "1000000000";
            trg3 = (Diff % "1000000000") / "1000000";
        }
        if (Diff >= "1000000000000000" && Diff < "1000000000000000000")
        {
            trg7 = 0;
            trg6 = Diff / "1000000000000000";
            trg5 = (Diff % "1000000000000000") / "1000000000000";
            trg4 = (Diff % "1000000000000") / "1000000000";
            trg3 = (Diff % "1000000000") / "1000000";
        }
        if (Diff > "1000000000000000000" && Diff < "1000000000000000000000")
        {
            trg7 = Diff / "1000000000000000000";
            trg6 = (Diff % "1000000000000000000") / "1000000000000000";
            trg5 = (Diff % "1000000000000000") / "1000000000000";
            trg4 = (Diff % "1000000000000") / "1000000000";
            trg3 = (Diff % "1000000000") / "1000000";
        }
        if (Diff > "1000000000000000000000" && Diff < "1000000000000000000000000")
        {
            trg8 = Diff / "1000000000000000000000";
            trg7 = (Diff % "1000000000000000000000") / "1000000000000000000";
            trg6 = (Diff % "1000000000000000000") / "1000000000000000";
            trg5 = (Diff % "1000000000000000") / "1000000000000";
            trg4 = (Diff % "1000000000000") / "1000000000";
            trg3 = (Diff % "1000000000") / "1000000";
        }
        if (Diff > "1000000000000000000000000" && Diff < "1000000000000000000000000000")
        {
            trg9 = Diff / "1000000000000000000000000";
            trg8 = (Diff % "1000000000000000000000000") / "1000000000000000000000";
            trg7 = (Diff % "1000000000000000000000") / "1000000000000000000";
            trg6 = (Diff % "1000000000000000000") / "1000000000000000";
            trg5 = (Diff % "1000000000000000") / "1000000000000";
            trg4 = (Diff % "1000000000000") / "1000000000";
            trg3 = (Diff % "1000000000") / "1000000";
        }
        if (Diff > "1000000000000000000000000000" && Diff < "1000000000000000000000000000000")
        {
            trg10 = Diff / "1000000000000000000000000000";
            trg9 = (Diff % "1000000000000000000000000000") / "1000000000000000000000000";
            trg8 = (Diff % "1000000000000000000000000") / "1000000000000000000000";
            trg7 = (Diff % "1000000000000000000000") / "1000000000000000000";
            trg6 = (Diff % "1000000000000000000") / "1000000000000000";
            trg5 = (Diff % "1000000000000000") / "1000000000000";
            trg4 = (Diff % "1000000000000") / "1000000000";
            trg3 = (Diff % "1000000000") / "1000000";
        }////////////
        if (Diff > "1000000000000000000000000000000" && Diff < "1000000000000000000000000000000000")
        {
            trg11 = Diff / "1000000000000000000000000000000";
            trg10 = (Diff % "1000000000000000000000000000000") / "1000000000000000000000000000";
            trg9 = (Diff % "1000000000000000000000000000") / "1000000000000000000000000";
            trg8 = (Diff % "1000000000000000000000000") / "1000000000000000000000";
            trg7 = (Diff % "1000000000000000000000") / "1000000000000000000";
            trg6 = (Diff % "1000000000000000000") / "1000000000000000";
            trg5 = (Diff % "1000000000000000") / "1000000000000";
            trg4 = (Diff % "1000000000000") / "1000000000";
            trg3 = (Diff % "1000000000") / "1000000";
        }
        if (Diff > "1000000000000000000000000000000000" && Diff < "1000000000000000000000000000000000000")
        {
            trg12 = Diff / "1000000000000000000000000000000000";
            trg11 = (Diff % "1000000000000000000000000000000000") / "1000000000000000000000000000000";
            trg10 = (Diff % "1000000000000000000000000000000") / "1000000000000000000000000000";
            trg9 = (Diff % "1000000000000000000000000000") / "1000000000000000000000000";
            trg8 = (Diff % "1000000000000000000000000") / "1000000000000000000000";
            trg7 = (Diff % "1000000000000000000000") / "1000000000000000000";
            trg6 = (Diff % "1000000000000000000") / "1000000000000000";
            trg5 = (Diff % "1000000000000000") / "1000000000000";
            trg4 = (Diff % "1000000000000") / "1000000000";
            trg3 = (Diff % "1000000000") / "1000000";
        }
        if (Diff > "1000000000000000000000000000000000000" && Diff < "1000000000000000000000000000000000000000")
        {
            trg13 = Diff / "1000000000000000000000000000000000000";
            trg12 = (Diff % "1000000000000000000000000000000000000") / "1000000000000000000000000000000000";
            trg11 = (Diff % "1000000000000000000000000000000000") / "1000000000000000000000000000000";
            trg10 = (Diff % "1000000000000000000000000000000") / "1000000000000000000000000000";
            trg9 = (Diff % "1000000000000000000000000000") / "1000000000000000000000000";
            trg8 = (Diff % "1000000000000000000000000") / "1000000000000000000000";
            trg7 = (Diff % "1000000000000000000000") / "1000000000000000000";
            trg6 = (Diff % "1000000000000000000") / "1000000000000000";
            trg5 = (Diff % "1000000000000000") / "1000000000000";
            trg4 = (Diff % "1000000000000") / "1000000000";
            trg3 = (Diff % "1000000000") / "1000000";
        }
        if (Diff > "1000000000000000000000000000000000000000" && Diff < "1000000000000000000000000000000000000000000")
        {
            trg14 = Diff / "1000000000000000000000000000000000000000";
            trg13 = (Diff % "1000000000000000000000000000000000000000") / "1000000000000000000000000000000000000";
            trg12 = (Diff % "1000000000000000000000000000000000000") / "1000000000000000000000000000000000";
            trg11 = (Diff % "1000000000000000000000000000000000") / "1000000000000000000000000000000";
            trg10 = (Diff % "1000000000000000000000000000000") / "1000000000000000000000000000";
            trg9 = (Diff % "1000000000000000000000000000") / "1000000000000000000000000";
            trg8 = (Diff % "1000000000000000000000000") / "1000000000000000000000";
            trg7 = (Diff % "1000000000000000000000") / "1000000000000000000";
            trg6 = (Diff % "1000000000000000000") / "1000000000000000";
            trg5 = (Diff % "1000000000000000") / "1000000000000";
            trg4 = (Diff % "1000000000000") / "1000000000";
            trg3 = (Diff % "1000000000") / "1000000";
        }
        if (Diff > "1000000000000000000000000000000000000000000" && Diff < "1000000000000000000000000000000000000000000000")
        {
            trg15 = Diff / "1000000000000000000000000000000000000000000";
            trg14 = (Diff % "1000000000000000000000000000000000000000000") / "1000000000000000000000000000000000000000";
            trg13 = (Diff % "1000000000000000000000000000000000000000") / "1000000000000000000000000000000000000";
            trg12 = (Diff % "1000000000000000000000000000000000000") / "1000000000000000000000000000000000";
            trg11 = (Diff % "1000000000000000000000000000000000") / "1000000000000000000000000000000";
            trg10 = (Diff % "1000000000000000000000000000000") / "1000000000000000000000000000";
            trg9 = (Diff % "1000000000000000000000000000") / "1000000000000000000000000";
            trg8 = (Diff % "1000000000000000000000000") / "1000000000000000000000";
            trg7 = (Diff % "1000000000000000000000") / "1000000000000000000";
            trg6 = (Diff % "1000000000000000000") / "1000000000000000";
            trg5 = (Diff % "1000000000000000") / "1000000000000";
            trg4 = (Diff % "1000000000000") / "1000000000";
            trg3 = (Diff % "1000000000") / "1000000";
        }
        if (Diff > "1000000000000000000000000000000000000000000000" && Diff < "1000000000000000000000000000000000000000000000000")
        {
            trg16 = Diff / "1000000000000000000000000000000000000000000000";
            trg15 = (Diff % "1000000000000000000000000000000000000000000000") / "1000000000000000000000000000000000000000000";
            trg14 = (Diff % "1000000000000000000000000000000000000000000") / "1000000000000000000000000000000000000000";
            trg13 = (Diff % "1000000000000000000000000000000000000000") / "1000000000000000000000000000000000000";
            trg12 = (Diff % "1000000000000000000000000000000000000") / "1000000000000000000000000000000000";
            trg11 = (Diff % "1000000000000000000000000000000000") / "1000000000000000000000000000000";
            trg10 = (Diff % "1000000000000000000000000000000") / "1000000000000000000000000000";
            trg9 = (Diff % "1000000000000000000000000000") / "1000000000000000000000000";
            trg8 = (Diff % "1000000000000000000000000") / "1000000000000000000000";
            trg7 = (Diff % "1000000000000000000000") / "1000000000000000000";
            trg6 = (Diff % "1000000000000000000") / "1000000000000000";
            trg5 = (Diff % "1000000000000000") / "1000000000000";
            trg4 = (Diff % "1000000000000") / "1000000000";
            trg3 = (Diff % "1000000000") / "1000000";
        }
        if (Diff > "1000000000000000000000000000000000000000000000000" && Diff < "1000000000000000000000000000000000000000000000000000")
        {
            trg17 = Diff / "1000000000000000000000000000000000000000000000000";
            trg16 = (Diff % "1000000000000000000000000000000000000000000000000") / "1000000000000000000000000000000000000000000000";
            trg15 = (Diff % "1000000000000000000000000000000000000000000000") / "1000000000000000000000000000000000000000000";
            trg14 = (Diff % "1000000000000000000000000000000000000000000") / "1000000000000000000000000000000000000000";
            trg13 = (Diff % "1000000000000000000000000000000000000000") / "1000000000000000000000000000000000000";
            trg12 = (Diff % "1000000000000000000000000000000000000") / "1000000000000000000000000000000000";
            trg11 = (Diff % "1000000000000000000000000000000000") / "1000000000000000000000000000000";
            trg10 = (Diff % "1000000000000000000000000000000") / "1000000000000000000000000000";
            trg9 = (Diff % "1000000000000000000000000000") / "1000000000000000000000000";
            trg8 = (Diff % "1000000000000000000000000") / "1000000000000000000000";
            trg7 = (Diff % "1000000000000000000000") / "1000000000000000000";
            trg6 = (Diff % "1000000000000000000") / "1000000000000000";
            trg5 = (Diff % "1000000000000000") / "1000000000000";
            trg4 = (Diff % "1000000000000") / "1000000000";
            trg3 = (Diff % "1000000000") / "1000000";
        }
        buf3 << trg3 << std::endl;
        buf4 << trg4 << std::endl;
        buf5 << trg5 << std::endl;
        buf6 << trg6 << std::endl;
        buf7 << trg7 << std::endl;
        buf8 << trg8 << std::endl;
        buf9 << trg9 << std::endl;
        buf10 << trg10 << std::endl;
        buf11 << trg11 << std::endl;
        buf12 << trg12 << std::endl;
        buf13 << trg13 << std::endl;
        buf14 << trg14 << std::endl;
        buf15 << trg15 << std::endl;
        buf16 << trg16 << std::endl;
        buf17 << trg17 << std::endl;
        st3 = buf3.str();
        st4 = buf4.str();
        st5 = buf5.str();
        st6 = buf6.str();
        st7 = buf7.str();
        st8 = buf8.str();
        st9 = buf9.str();
        st10 = buf10.str();
        st11 = buf11.str();
        st12 = buf12.str();
        st13 = buf13.str();
        st14 = buf14.str();
        st15 = buf15.str();
        st16 = buf16.str();
        st17 = buf17.str();
        st3.erase(st3.find_last_not_of(" \n\r\t") + 1);
        st4.erase(st4.find_last_not_of(" \n\r\t") + 1);
        st5.erase(st5.find_last_not_of(" \n\r\t") + 1);
        st6.erase(st6.find_last_not_of(" \n\r\t") + 1);
        st7.erase(st7.find_last_not_of(" \n\r\t") + 1);
        st8.erase(st8.find_last_not_of(" \n\r\t") + 1);
        st9.erase(st9.find_last_not_of(" \n\r\t") + 1);
        st10.erase(st10.find_last_not_of(" \n\r\t") + 1);
        st11.erase(st11.find_last_not_of(" \n\r\t") + 1);
        st12.erase(st12.find_last_not_of(" \n\r\t") + 1);
        st13.erase(st13.find_last_not_of(" \n\r\t") + 1);
        st14.erase(st14.find_last_not_of(" \n\r\t") + 1);
        st15.erase(st15.find_last_not_of(" \n\r\t") + 1);
        st16.erase(st16.find_last_not_of(" \n\r\t") + 1);
        st17.erase(st17.find_last_not_of(" \n\r\t") + 1);
        dif3 = const_cast<char*>(st3.c_str());
        dif4 = const_cast<char*>(st4.c_str());
        dif5 = const_cast<char*>(st5.c_str());
        dif6 = const_cast<char*>(st6.c_str());
        dif7 = const_cast<char*>(st7.c_str());
        dif8 = const_cast<char*>(st8.c_str());
        dif9 = const_cast<char*>(st9.c_str());
        dif10 = const_cast<char*>(st10.c_str());
        dif11 = const_cast<char*>(st11.c_str());
        dif12 = const_cast<char*>(st12.c_str());
        dif13 = const_cast<char*>(st13.c_str());
        dif14 = const_cast<char*>(st14.c_str());
        dif15 = const_cast<char*>(st15.c_str());
        dif16 = const_cast<char*>(st16.c_str());
        dif17 = const_cast<char*>(st17.c_str());
        if (!mode.empty())
            printf("Prefix search (-m %s) runs on top of the database lookup: every key is checked "
                   "against the .bin AND against the prefix.\n", mode.c_str());
        if (mode.empty())          printf("\rDifficulty (keys per address with prefix %s):\n", prefix.c_str());
        else if (mode == "any")    printf("\rDifficulty (-m any: keys per hit, either address with prefix %s):\n", prefix.c_str());
        else if (mode == "unccom") printf("\rDifficulty (-m unccom: keys per hit, %d shared leading bytes of hash160):\n", patsize > 20 ? 20 : patsize);
        else if (mode == "triple") printf("\rDifficulty (-m triple: keys per hit, %d shared bytes AND prefix %s):\n", patsize > 20 ? 20 : patsize, prefix.c_str());
        else                       printf("\rDifficulty (-m %s: keys per hit, prefix %s):\n", mode.c_str(), prefix.c_str());
        printf("\r%s Qid %s Qad %s Trd %s Dud %s Und %s Dec %s Non %s Oct %s Sep %s Sex %s Qui %s Qua %s Trl %s Mld %s Mln.\n", dif17, dif16, dif15, dif14, dif13, dif12, dif11, dif10, dif9, dif8, dif7, dif6, dif5, dif4, dif3);
        printf("\rMln - million,           10^6\n"
            "Mld - milliard,          10^9\n"
            "Trl - trillion,          10^12\n"
            "Qua - quadrillion,       10^15\n"
            "Qui - quintillion,       10^18\n"
            "Sex - sextillion,        10^21\n"
            "Sep - septillion,        10^24\n"
            "Oct - octillion,         10^27\n"
            "Non - nonillion,         10^30\n"
            "Dec - decillion,         10^33\n"
            "Und - undecillion,       10^36\n"
            "Duo - duodecillion,      10^39\n"
            "Trd - tredecillion,      10^42\n"
            "Qad - quattuordecillion, 10^45\n"
            "Qid - quindecillion,     10^48\n");
    } // if(prefix != "")
    // Everything printed so far leaves the buffer before fork(): with stdout
    // redirected to a pipe or file it is fully buffered, and every worker
    // would otherwise flush its inherited copy at exit, repeating the
    // start-up lines.
    std::fflush(stdout);
    std::fflush(stderr);
    const bool old_engine = std::getenv("ECLOOP_OLD_ENGINE") != nullptr;

    for (int i = 0; i < threads; ++i) {
        const pid_t pid = fork();
        if (pid < 0) {
            std::perror("fork");
            return 1;
        }
        if (pid == 0) {
            close(sock[0]);
            // The parent died between our fork and here: exit rather than
            // linger as an orphan burning a core.
            if (getppid() != parent_pid) return 0;
            if (old_engine) engine(i);
            else engine_x8(i);
            return 0;
        }
    }

    close(sock[1]);
    manager_loop(threads);
    return 0;
}
