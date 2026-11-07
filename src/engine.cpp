// ── reference engine: one walker ─────────────────────────────────────────
    //
    // Kept as the correctness baseline for engine_x8, selected with
    // ECLOOP_OLD_ENGINE=1. Slower, but every step is plain.
    //
    // THE WALK. A random hex digit of the key string is bumped by one
    // WITHOUT carry ('F' -> '0'), so consecutive scalars differ by a value
    // from a small precomputed table -- one point addition per key instead
    // of a full multiplication. See ec_delta.hpp.

#include "main.hpp"

#include "addr.hpp"
#include "addr_dispatch.hpp"
#include "btree_filter.hpp"
#include "compat.hpp"
#include "ec_batch8.hpp"
#include "ec_delta.hpp"
#include "ec_glv.hpp"
#include "ecc.hpp"
#include "sha256.hpp"
#include "fastPRNG.h"
#include "vanity_match.hpp"
#include "walk_common.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>



namespace ecloop {
    
    // ── PRNG ─────────────────────────────────────────────────────────────

        // Lemire reduction rather than a mask. The original used
        // & ((pksize/4)-1), which is only correct when the digit count is a
        // power of two: at pksize = 24 (6 digits) the mask &5 yields
        // positions {0,1,4,5}, so digits 2 and 3 were NEVER bumped and a
        // third of the space was unreachable. A modulo would fix that but
        // costs a hardware division per key.
    ECLOOP_ALWAYS_INLINE u32 rnd_pos(fastPRNG::fastXS64& st, u32 ndig) noexcept {
        return static_cast<u32>(((st.xoshiro256pp() >> 32) *
            static_cast<u64>(ndig)) >> 32);
    }
    
namespace {

    // Mode/resolve_mode/check_mode_prereqs/report_found live in
    // vanity_match.*, shared with engine_x8.cpp, so the two engines can
    // never disagree about what a mode means or what a result looks like.
    using vanity::Mode;
    using vanity::resolve_mode;
    using vanity::check_mode_prereqs;

} // namespace


// Reference sweep, reached with ECLOOP_OLD_ENGINE=1.
//
// Deliberately plain: every point comes from its own generator multiplication
// instead of the shared-inversion group trick. That makes it an independent
// check of the fast sweep rather than a copy -- if the two ever disagree about
// which keys a range contains, the group layout is where to look.
// The plain check of `count` consecutive keys from `base`: every point from
// its own generator multiplication, one inversion per chunk of 256. Used by
// the sequential sweep and by the reference mixed mode. Reports are cut at
// `limit` (inclusive) when it is given.
static void ref_sweep(const fe base, u64 count, int thread, const fe* limit, Mode vmode) {
    constexpr std::size_t CHUNK = 256;
    static_assert(CHUNK % HASH_BATCH == 0, "chunk must divide into hash batches");

    std::vector<pe> pts(CHUNK);
    h160_t h33[HASH_BATCH], h65[HASH_BATCH];
    u64 scanned = 0;

    for (u64 off = 0; off < count; off += CHUNK) {
        for (std::size_t i = 0; i < CHUNK; ++i) {
            fe k, o;
            fe_set64(o, off + i);
            fe_modn_add(k, base, o);
            ec_gtable_mul(&pts[i], k);
        }
        ec_jacobi_grprdc(pts.data(), CHUNK);

        for (std::size_t b = 0; b < CHUNK; b += HASH_BATCH) {
            addr33_batchN(h33, pts.data() + b, HASH_BATCH);
            addr65_batchN(h65, pts.data() + b, HASH_BATCH);
            for (std::size_t q = 0; q < HASH_BATCH; ++q) {
                // Same shape as sweep_from's switch in engine_x8.cpp:
                // Bin is the exact two lines this file always ran here;
                // UncCom/Triple set both flags to the same verdict since
                // report_found always shows both addresses anyway.
                bool c33 = false, c65 = false;
                // The database in every mode, the prefix on top when -m asks.
                if (BTREE_HEAP) {
                    c33 = btf_maybe(h33[q]) && btree_search(h33[q], nullptr) > 0;
                    c65 = btf_maybe(h65[q]) && btree_search(h65[q], nullptr) > 0;
                }
                switch (vmode) {
                case Mode::Bin:
                    break;
                case Mode::Unc:
                    c65 = c65 || vanity::matches_prefix(h65[q]);
                    break;
                case Mode::Com:
                    c33 = c33 || vanity::matches_prefix(h33[q]);
                    break;
                case Mode::Any:
                    c33 = c33 || vanity::matches_prefix(h33[q]);
                    c65 = c65 || vanity::matches_prefix(h65[q]);
                    break;
                case Mode::UncCom:
                    c33 = c65 = c33 || c65 || vanity::shares_prefix_bytes(h65[q], h33[q], patsize);
                    break;
                case Mode::Triple:
                    c33 = c65 = c33 || c65 ||
                                (vanity::shares_prefix_bytes(h65[q], h33[q], patsize) &&
                                 (vanity::matches_prefix(h65[q]) ||
                                  vanity::matches_prefix(h33[q])));
                    break;
                }
                // Accumulated locally and flushed once per chunk: the
                // per-worker counters share a cache line, and touching it
                // on every key makes all workers fight over it.
                ++scanned;
                if (__builtin_expect(!(c33 || c65), 1)) continue;
                fe found, o;
                fe_set64(o, off + b + q);
                fe_modn_add(found, base, o);
                if (limit && fe_cmp(found, *limit) > 0) continue;
                vanity::report_found(found, h65[q], h33[q]);
            }
        }
        thread_count[thread] += scanned;
        scanned = 0;
    }
}

// Reference sweep, reached with ECLOOP_OLD_ENGINE=1.
//
// Deliberately plain: every point comes from its own generator multiplication
// instead of the shared-inversion group trick. That makes it an independent
// check of the fast sweep rather than a copy -- if the two ever disagree about
// which keys a range contains, the group layout is where to look.
static void engine_seq_ref(int thread) {
    const Mode vmode = resolve_mode(thread);
    check_mode_prereqs(vmode, thread);

    const int T = seq_threads > 0 ? seq_threads : 1;
    constexpr u64 BLOCK = 1ULL << 20;

    for (u64 blk = static_cast<u64>(thread);; blk += static_cast<u64>(T)) {
        seq_progress[thread] = blk;

        fe base;
        seq_key_at(base, blk * BLOCK);
        if (fe_cmp(base, seq_end) > 0) break;

        ref_sweep(base, BLOCK, thread, &seq_end, vmode);
    }
    seq_progress[thread] = ~0ULL;
}

// Reference mixed mode: the same draw of the high part as engine_x8_mixed
// (walk_step over the offset digits above the swept zone, bounded in a
// range), then the plain sweep of the block.
static void engine_mixed_ref(int thread) {
    const Mode vmode = resolve_mode(thread);
    check_mode_prereqs(vmode, thread);

    const int ndig = walk_ndig;
    const int zdig = mix_bits / 4;
    const int hdig = ndig - zdig;
    const u64 span = 1ULL << mix_bits;

    char bound_buf[65];
    const char* bound = walk_bound(ndig, bound_buf);

    std::vector<char> buff(static_cast<std::size_t>(ndig) + 1, '0');
    buff[ndig] = 0;

    const auto now = std::chrono::system_clock::now().time_since_epoch();
    fastPRNG::fastXS64 st(
        static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()) +
        static_cast<u64>(thread) * 0x9E3779B97F4A7C15ULL);

    for (int attempt = 0; attempt < 4096; ++attempt) {
        for (int i = 0; i < hdig; ++i) {
            const u32 d = static_cast<u32>(((st.xoshiro256pp() >> 32) * 16ULL) >> 32);
            buff[i] = static_cast<char>(d < 10 ? '0' + d : 'A' + d - 10);
        }
        if (!bound || std::memcmp(buff.data(), bound, static_cast<std::size_t>(hdig)) <= 0) break;
        std::memset(buff.data(), '0', static_cast<std::size_t>(hdig));
    }

    for (;;) {
        fe base;
        walk_key(base, buff.data(), ndig);
        u64 count = span;
        const fe* limit = nullptr;
        if (range_active) {
            limit = &seq_end;
            fe last, t;
            fe_set64(t, span - 1);
            fe_modn_add(last, base, t);
            if (fe_cmp(last, seq_end) > 0) {
                fe rem;
                fe_modn_sub(rem, seq_end, base);
                count = rem[0] + 1;
            }
        }
        ref_sweep(base, count, thread, limit, vmode);

        const u8 lev = static_cast<u8>(((st.xoshiro256pp() >> 32) *
                                        static_cast<u64>(hdig)) >> 32);
        u8 used;
        walk_step(buff.data(), hdig, bound, lev, &used);
    }
}

void engine(int thread) {
    // Same dispatch as engine_x8: the sweep, the mixed mode, or the walk --
    // the latter two inside the -k/-e range when one is set.
    if (seq_active) { engine_seq_ref(thread); return; }
    if (mix_bits > 0) { engine_mixed_ref(thread); return; }

    const Mode vmode = resolve_mode(thread);
    check_mode_prereqs(vmode, thread);

        constexpr int BLK = 256;

        // BLK*3 must divide the hash batch. 3 and 16 are coprime, so this
        // reduces to BLK % 16 == 0 -- with the old batch of 8 the
        // requirement was weaker, and changing BLK could silently leave a
        // tail.
        static_assert((BLK * 3) % 16 == 0,
            "BLK*3 must be divisible by 16 (hash batch size)");

        // Offset digits (main.cpp: pksize/4, or the digits of the range),
        // bounded by the range end; see walk_common.hpp.
        const int ndig = walk_ndig;
        char bound_buf[65];
        const char* bound = walk_bound(ndig, bound_buf);
        const std::size_t NPTS = endo_on ? static_cast<std::size_t>(BLK) * 3
                                         : static_cast<std::size_t>(BLK);

        // Heap, not stack: at pksize = 256 this is ~74 KB of points and
        // 16 KB of strings, all of which the C version put on the stack of a
        // freshly forked process.
        std::vector<pe> pts(BLK * 3); // originals + phi + phi^2
        std::vector<char> keys(static_cast<std::size_t>(BLK) * 65);

        std::vector<char> buff(static_cast<std::size_t>(ndig) + 1, '0');
        buff[ndig] = '\0';

        // Seeded per worker. The C original declared xoshiro256_state on the
        // stack and then seeded with st->s[i] + seed, reading the
        // uninitialised limbs -- undefined behaviour, and the walk order was
        // whatever the stack happened to hold. fastXS64 derives the whole
        // state from the seed through splitMix64, which is what the comment
        // there says was intended.
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        fastPRNG::fastXS64 st(
            static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()) +
            static_cast<u64>(thread) * 0x9E3779B97F4A7C15ULL);

        // Starting point: one full generator multiplication for the current
        // buff, then only delta_step increments. Must come AFTER buff is set
        // and BEFORE the loop. With buff all zeros the scalar is 0 and P is
        // the point at infinity, which delta_step handles.
        pe P;
        DeltaWalker walk;
        {
            fe pk0 = {};
            walk_key(pk0, buff.data(), ndig);   // seq_start + offset in a range
            ec_gtable_mul(&P, pk0);
            walk.sync(buff.data());
            // The degenerate-addition tracker follows the ABSOLUTE key
            // (P == +-D depends on it): in a range that is seq_start +
            // offset, kept only while it fits comfortably in a u64.
            if (range_active && walk.track) {
                if (seq_start[1] == 0 && seq_start[2] == 0 && seq_start[3] == 0 &&
                    seq_start[0] < (1ULL << 60))
                    walk.k += seq_start[0];
                else
                    walk.track = false;
            }
        }

        std::size_t cnt = 0;
        u64 walk_scanned = 0;

        for (;;) {
            // One bounded digit step, the same rule as engine_x8 (walk_step),
            // and the matching point move by kind.
            const u8 lv = static_cast<u8>(rnd_pos(st, static_cast<u32>(ndig)));
            u8 moved;
            const u8 kind = walk_step(buff.data(), ndig, bound, lv, &moved);

            // The C original memcpy'd 65 bytes out of a buffer of
            // (pksize/4)+1. For any pksize < 256 that reads past the end of
            // the object, and that is what was corrupting the printed key.
            std::memcpy(&keys[cnt * 65], buff.data(),
                static_cast<std::size_t>(ndig) + 1);

            walk.step_kind(&P, moved, kind);
            pe_clone(&pts[cnt], &P);

            if (++cnt < BLK) continue;

            delta_affine(pts.data(), BLK);            // one inversion for the block
            if (endo_on) glv_expand(pts.data(), BLK); // three keys per point

            h160_t h65[HASH_BATCH], h33[HASH_BATCH];

            for (std::size_t b = 0; b < NPTS; b += HASH_BATCH) {
                const std::size_t nb = (NPTS - b < HASH_BATCH) ? (NPTS - b) : HASH_BATCH;
                addr65_batchN(h65, pts.data() + b, nb);
                addr33_batchN(h33, pts.data() + b, nb);

                for (std::size_t q = 0; q < nb; ++q) {
                    const std::size_t j = b + q;
                    const char* key_str = &keys[(j % BLK) * 65];

                    if (verb == 2) {
                        if (range_active) {
                            fe ak; char hex[65];
                            walk_key(ak, key_str, ndig);
                            fe_to_hex(hex, ak);
                            std::printf("%s %d\n", hex, thread);
                        } else {
                            std::printf("%s %d\n", key_str, thread);
                        }
                    }
                    if (verb == 5) {
                        u32 t1[5], t3[5];
                        for (int w = 0; w < 5; ++w) t1[w] = detail::bswap32(h65[q][w]);
                        for (int w = 0; w < 5; ++w) t3[w] = detail::bswap32(h33[q][w]);
                        print_h160(t1);
                        print_h160(t3);
                    }

                    // The database in every mode; the prefix on top when -m asks.
                    const bool db_hit = BTREE_HEAP &&
                        ((btf_maybe(h65[q]) && btree_search(h65[q], nullptr) > 0) ||
                         (btf_maybe(h33[q]) && btree_search(h33[q], nullptr) > 0));
                    bool hit = false;
                    switch (vmode) {
                    case Mode::Bin:
                        break;
                    case Mode::Unc:
                        hit = vanity::matches_prefix(h65[q]);
                        break;
                    case Mode::Com:
                        hit = vanity::matches_prefix(h33[q]);
                        break;
                    case Mode::Any:
                        hit = vanity::matches_prefix(h65[q]) ||
                              vanity::matches_prefix(h33[q]);
                        break;
                    case Mode::UncCom:
                        hit = vanity::shares_prefix_bytes(h65[q], h33[q], patsize);
                        break;
                    case Mode::Triple:
                        hit = vanity::shares_prefix_bytes(h65[q], h33[q], patsize) &&
                              (vanity::matches_prefix(h65[q]) ||
                               vanity::matches_prefix(h33[q]));
                        break;
                    }

                    if (__builtin_expect(!(hit || db_hit), 1)) {
                        ++walk_scanned;
                        continue;
                    }

                    // keys[] holds strings for the BLK base points only; the
                    // phi and phi^2 images share that base and get their key
                    // by multiplying by lambda or lambda^2.
                    fe base_k, found;
                    walk_key(base_k, key_str, ndig);
                    glv_key(found, base_k, j, BLK);

                    // Rebuild the point from the recovered key by an
                    // independent route and compare. One generator
                    // multiplication, paid only on a hit, so effectively
                    // never. Catches both key reconstruction errors and
                    // corrupted points. This project has historically
                    // skipped this check for Bin (disabled block below,
                    // kept for that history); the vanity modes below get an
                    // active version of it, because Unc/Com/Any/UncCom/
                    // Triple all still need real confidence that `found`
                    // reproduces the hash that actually triggered the hit.
                    /* {
                        pe V;
                        ec_gtable_mul(&V, found);
                        ec_jacobi_rdc(&V, &V);
                        h160_t v65, v33;
                        addr65(v65, &V);
                        addr33(v33, &V);
                        const bool ok65 = std::memcmp(v65, h65[q], sizeof(h160_t)) == 0;
                        const bool ok33 = std::memcmp(v33, h33[q], sizeof(h160_t)) == 0;
                        if (!ok65 && !ok33) {
                            std::fprintf(stderr,
                                         "FALSE POSITIVE: key does not reproduce the "
                                         "matched hash (idx = %zu, branch = %zu, "
                                         "str = %s)\n",
                                         j, j / BLK, key_str);
                            continue;
                        }
                    }*/

                    // What report_found prints below: the block's own
                    // hashes for Bin (unverified, matching the disabled
                    // check above), or the freshly-verified pair for a
                    // vanity mode, overwritten just below.
                    h160_t r65, r33;
                    std::memcpy(r65, h65[q], sizeof(h160_t));
                    std::memcpy(r33, h33[q], sizeof(h160_t));

                    // A prefix hit is verified by the mode's rule; a database
                    // hit by either hash reproducing (the unverified Bin
                    // report of old is kept only when there is no prefix
                    // mode at all).
                    if (vmode != Mode::Bin) {
                        pe V;
                        ec_gtable_mul(&V, found);
                        ec_jacobi_rdc(&V, &V);
                        h160_t v65, v33;
                        addr65(v65, &V);
                        addr33(v33, &V);
                        const bool ok65 = std::memcmp(v65, h65[q], sizeof(h160_t)) == 0;
                        const bool ok33 = std::memcmp(v33, h33[q], sizeof(h160_t)) == 0;

                        bool ok = false;
                        if (!hit && db_hit) ok = ok65 || ok33;
                        else switch (vmode) {
                        case Mode::Bin:    break; // excluded by the if above
                        case Mode::Unc:    ok = ok65; break;
                        case Mode::Com:    ok = ok33; break;
                        case Mode::Any:    ok = ok65 || ok33; break;
                        case Mode::UncCom: ok = ok65 && ok33; break;
                        case Mode::Triple: ok = ok65 && ok33; break;
                        }
                        if (!ok) {
                            std::fprintf(stderr,
                                         "FALSE POSITIVE: key does not reproduce the "
                                         "matched hash (idx = %zu, branch = %zu, "
                                         "str = %s)\n",
                                         j, j / BLK, key_str);
                            continue;
                        }
                        std::memcpy(r65, v65, sizeof(h160_t));
                        std::memcpy(r33, v33, sizeof(h160_t));
                    }

                    vanity::report_found(found, r65, r33);
                }
            }
            thread_count[thread] += walk_scanned;
            walk_scanned = 0;
            cnt = 0;
        }
    }

 

} // namespace ecloop
