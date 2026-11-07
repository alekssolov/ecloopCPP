// Eight independent walkers driven through the batch-8 IFMA layer. Port of
// lib/engine_x8.c.
//
// Each of the eight lanes walks its own key string. One
// ec_jacobi_add_affine_x8 advances all eight at once, EC8_NSTEP such steps
// accumulate into a block, and the whole block goes affine on a single
// field inversion.
//
// The key of a hit is not stored per point -- that would be 768 strings
// per block. Only the position log is kept, and the string is replayed
// from the block-start snapshot when something actually matches.

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
#include "sweep_x8.hpp"
#include "sweep_x4.hpp"
#include "walk_common.hpp"
#include "cpu.hpp"
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <memory>
#include <vector>

#ifndef EC8_NWALK
#define EC8_NWALK 8 // batch-8 width; fixed, the SoA layout depends on it
#endif

// Batch steps accumulated before converting to affine.
// EC8_NSTEP * 8 = base points per block, one field inversion per block.
#ifndef EC8_NSTEP
#define EC8_NSTEP 32
#endif

#ifndef EC8_RANDOM_START
#define EC8_RANDOM_START 0
#endif

// How far ahead to prefetch filter words.
#ifndef EC8_PFDIST
#define EC8_PFDIST 24
#endif

#define EC8_NBASE (EC8_NSTEP * EC8_NWALK) // base keys per block
#define EC8_NPTS (EC8_NBASE * 3)          // including the endomorphism
#define EC8_MAXDIG 64

namespace ecloop {

            // seq_start + offset. Defined here because the sweep lives here;
    // engine.cpp uses the same one through main.hpp.
    void seq_key_at(fe out, u64 offset) {
        fe off;
        fe_set64(off, offset);
        fe_modn_add(out, seq_start, off);
    }

namespace {

            // EC8_NPTS = 768 at NSTEP = 32, divisible by 16.
            static_assert(EC8_NPTS % 16 == 0, "EC8_NPTS must be divisible by 16");

            ECLOOP_ALWAYS_INLINE u32 rnd_pos(fastPRNG::fastXS64& st, u32 ndig) noexcept {
                return static_cast<u32>(((st.xoshiro256pp() >> 32) *
                                         static_cast<u64>(ndig)) >> 32);
            }

            // Gather eight DIFFERENT deltas into SoA form. Each lane picks its
            // own point, so this is a real gather -- unlike everything else,
            // which flows through SoA with no shuffling.
            ECLOOP_ALWAYS_INLINE void ec8_gather_delta(Fe8* qx, Fe8* qy, const u8* lev,
                                                       const u8* kind) noexcept {
                for (int w = 0; w < EC8_NWALK; ++w) {
                    const pe* d = delta_point_kind(static_cast<int>(lev[w]), kind[w]);
                    qx->v[0][w] = d->x[0]; qx->v[1][w] = d->x[1];
                    qx->v[2][w] = d->x[2]; qx->v[3][w] = d->x[3];
                    qy->v[0][w] = d->y[0]; qy->v[1][w] = d->y[1];
                    qy->v[2][w] = d->y[2]; qy->v[3][w] = d->y[3];
                }
            }

            // Prefetch whatever is touched FIRST. With the filter that is its
            // word: only a few percent of points reach the btree, so
            // prefetching the bucket up front would waste bandwidth.
            ECLOOP_ALWAYS_INLINE void ec8_prefetch_bucket(const u32* tree,
                                                          const u32* hash) noexcept {
#if defined(__x86_64__)
                if (BTF_BITS) { btf_prefetch(hash); return; }
                if (!tree) return;
                const u8* h = reinterpret_cast<const u8*>(hash);
                const u32 hpos = 1 + (h[0] + h[1] * 256u + h[2] * 65536u);
                _mm_prefetch(reinterpret_cast<const char*>(&tree[hpos]), _MM_HINT_T0);
#else
                (void)tree; (void)hash;
#endif
            }

            u64 worker_seed(int thread) noexcept {
                const auto now = std::chrono::system_clock::now().time_since_epoch();
                const u64 ns = static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
                return ns + static_cast<u64>(thread) * 0x9E3779B97F4A7C15ULL;
            }

            // Aligned heap buffers. unique_ptr rather than _mm_malloc with a
            // matching _mm_free: the C version leaked on every early return,
            // which only did not matter because the function never returns.
            template <typename T>
            struct AlignedArray {
                std::unique_ptr<unsigned char[]> raw;
                T* ptr = nullptr;

                explicit AlignedArray(std::size_t n) {
                    const std::size_t bytes = n * sizeof(T) + 64;
                    raw.reset(new unsigned char[bytes]);
                    auto addr = reinterpret_cast<std::uintptr_t>(raw.get());
                    addr = (addr + 63) & ~static_cast<std::uintptr_t>(63);
                    ptr = reinterpret_cast<T*>(addr);
                }
                T& operator[](std::size_t i) noexcept { return ptr[i]; }
                T* get() noexcept { return ptr; }
            };

        
    // Scans `count` consecutive keys starting at `base`, in groups sharing one
    // modular inversion. Both the sequential sweep and the mixed mode run on
    // this: they differ only in where `base` comes from -- a block index in one
    // case, the walker's current key in the other.
    //
    // `limit` is checked per hit rather than per key: the group is generated
    // whole, and testing every key against the end of the range would cost a
    // 256-bit comparison on the hot path for a condition that matters only at
    // the very last group.

} // namespace

        // ─────────────────────────────────────────────────────────────────────────────
// Sequential sweep. Selected by giving a start key; there is no separate flag.
//
// WHY IT REPLACES THE WALK RATHER THAN EXTENDING IT. The random walk revisits
// keys -- covering a space of N takes about N ln N draws and no part of it can
// ever be called finished. A sweep covers N in exactly N, hits the key after
// N/2 on average, and its progress is one number you can write to a file.
//
// THE ENDOMORPHISM IS NOT USED HERE, on purpose. It gives two extra points per
// key for one multiplication each, but their keys are lambda*k and lambda^2*k
// modulo n -- full 256-bit values that fall outside any bounded range. Inside
// a sweep they can never be the target, so hashing them would throw away two
// thirds of the work. The walk keeps them because it is not bounded.
//
// WORK SPLIT. The range is cut into blocks of SEQ_BLOCK keys and worker t
// takes blocks t, t+T, t+2T. Nothing is shared and nothing is negotiated, so
// no key is visited twice and none is skipped -- a property a shared cursor
// makes hard to state, let alone prove.
//
// CHECKPOINT. Each worker publishes the block it is on. Everything below the
// minimum of those is done, so that minimum is the only safe resume point;
// worker 0 writes it hourly to progress_<startkey>.txt. Taking the maximum
// instead would skip whatever the slowest worker had not reached, silently.
// On resume the key from that file goes straight into -k.

namespace {

    // Keys per block. Large enough that the one generator multiplication
    // starting a block disappears into the noise, small enough that a
    // checkpoint never lags far behind the work actually finished.
    constexpr u64 SEQ_BLOCK = 1ULL << 20;

    // Points sharing a single modular inversion. This is what makes a sweep
    // cheap: without it every key would cost an inversion of its own.
    constexpr std::size_t SGRP = 2048;
    constexpr std::size_t SHALF = SGRP / 2;

    static_assert(SGRP % HASH_BATCH == 0, "group must divide into hash batches");
    static_assert(SEQ_BLOCK % SGRP == 0, "block must divide into groups");

    pe   _sgp[SHALF];    // (i+1)*G, affine
    pe   _sstep;         // SGRP*G, carries one group to the next
    bool _sready = false;

#if ECLOOP_HAVE_FE52
    // Eight-lane sweep (sweep_x8.hpp): the same (i+1)*G table in SoA form,
    // plus the scratch its batch inversion needs. Plain statics: workers
    // are processes, so each has its own copy, and unlike a function-local
    // static there is no guard check on every use.
    alignas(64) Fe8   _sgp8[2 * SHALF / 8];
    alignas(64) Fe52s _sweep_scr[2 * SHALF / 8];
    SweepX8Table _stab;
    bool _sweep_x8 = false;
#endif

#if ECLOOP_HAVE_FE29
    // The AVX2 four-lane sweep (sweep_x4.hpp): the EC rung for CPUs with
    // AVX2 but no IFMA, selected automatically when IFMA is absent (see
    // seq_prepare for the switches).
    Fe29s* _sweep4_scr = nullptr;
    SweepX4Table _stab4;
    bool _sweep_x4 = false;
#endif

    // ECLOOP_COMPRESSED_ONLY=1: in database mode (-b) hash and look up only
    // the address of the COMPRESSED public key. The uncompressed one costs
    // two SHA-256 blocks against one, so this drops about 60% of the hashing
    // -- but a key whose database entry came from the uncompressed key will
    // not be found. Off by default; ignored by the -m vanity modes, which
    // already say which address they want. Read once, in engine_x8().
    bool _com_only = false;

    // ── vanity search modes ──────────────────────────────────────────────────
    // Mode/resolve_mode/check_mode_prereqs now live in vanity_match.*, shared
    // with engine.cpp -- brought into scope here so every existing call site
    // below (Mode::Bin, resolve_mode(thread), ...) keeps working unqualified.
    using vanity::Mode;
    using vanity::resolve_mode;
    using vanity::check_mode_prereqs;

    void seq_prepare() {
        if (_sready) return;
        fe k;
        for (std::size_t i = 0; i < SHALF; ++i) {
            fe_set64(k, static_cast<u64>(i) + 1);
            ec_gtable_mul(&_sgp[i], k);
        }
        ec_jacobi_grprdc(_sgp, SHALF);
        fe_set64(k, SGRP);
        ec_gtable_mul(&_sstep, k);
        ec_jacobi_rdc(&_sstep, &_sstep);
#if ECLOOP_HAVE_FE52
        // ECLOOP_NO_IFMA=1 (checked by ec_batch8_available) keeps the
        // scalar group step, so both can be compared from one binary.
        if (ec_batch8_available()) {
            sweep_x8_build(&_stab, _sgp, SHALF, _sgp8);
            _sweep_x8 = true;
        }
#endif
#if ECLOOP_HAVE_FE29
        {
            // Automatic below IFMA (measured 12% faster than the scalar
            // group step on Tiger Lake, parity on Sapphire Rapids), or
            // forced either way: ECLOOP_EC=avx2 (also instead of IFMA, for
            // A/B) / ECLOOP_EC=scalar.
            const char* e = std::getenv("ECLOOP_EC");
            const bool forced4 = e && !std::strcmp(e, "avx2");
            const bool forced_scalar = e && !std::strcmp(e, "scalar");
            bool use4 = forced4;
#if ECLOOP_HAVE_FE52
            if (forced4) _sweep_x8 = false;
            if (!e && !_sweep_x8 && cpu_has_avx2()) use4 = true;
#else
            if (!e && cpu_has_avx2()) use4 = true;
#endif
            if (forced_scalar) use4 = false;
            if (use4) {
                static Fe29s* storage = static_cast<Fe29s*>(std::aligned_alloc(32, 2 * (SHALF / 4) * sizeof(Fe29s)));
                static Fe29s* scr = static_cast<Fe29s*>(std::aligned_alloc(32, sweep_x4_scratch(SHALF / 4) * sizeof(Fe29s)));
                if (storage && scr) {
                    sweep_x4_build(&_stab4, _sgp, SHALF, storage);
                    _sweep4_scr = scr;
                    _sweep_x4 = true;
                }
            }
        }
#endif
        _sready = true;
    }

    // points[idx] receives the point of key (base + idx) for idx in [0, SGRP),
    // given (base + SHALF)*G in `centre`. One inversion of SHALF differences
    // serves the whole group, since the x of +i*G and -i*G are equal and only
    // the y differs.
    //
    // The layout is the part worth reading twice. Putting SHALF points below
    // the centre and SHALF above, and dropping the centre itself, is the
    // obvious arrangement and it is wrong: it skips one key per group and
    // repeats one at the seam with the next group, because the next centre is
    // SGRP further along. Here the centre is kept at index SHALF, SHALF points
    // sit below it and SHALF-1 above, so the group is exactly [base, base+SGRP).
    void seq_group_scalar(pe* points, const pe* centre, fe* dx) {
        for (std::size_t i = 0; i < SHALF; ++i)
            fe_modp_sub(dx[i], _sgp[i].x, centre->x);
        fe_modp_grpinv(dx, static_cast<u32>(SHALF));

        pe_clone(&points[SHALF], centre);

        for (std::size_t i = 0; i < SHALF; ++i) {
            fe s, t, x3;

            // key = centre-(i+1), at index SHALF-1-i
            pe* dn = &points[SHALF - 1 - i];
            fe_modp_neg(t, _sgp[i].y);
            fe_modp_sub(t, t, centre->y);
            fe_modp_mul(s, t, dx[i]);
            fe_modp_sqr(x3, s);
            fe_modp_sub(x3, x3, centre->x);
            fe_modp_sub(x3, x3, _sgp[i].x);
            fe_modp_sub(t, centre->x, x3);
            fe_modp_mul(t, t, s);
            fe_modp_sub(dn->y, t, centre->y);
            fe_clone(dn->x, x3);
            fe_set64(dn->z, 1);

            // key = centre+(i+1), at index SHALF+1+i; the last one would fall
            // outside the group
            if (i + 1 >= SHALF) continue;
            pe* up = &points[SHALF + 1 + i];
            fe_modp_sub(t, _sgp[i].y, centre->y);
            fe_modp_mul(s, t, dx[i]);
            fe_modp_sqr(x3, s);
            fe_modp_sub(x3, x3, centre->x);
            fe_modp_sub(x3, x3, _sgp[i].x);
            fe_modp_sub(t, centre->x, x3);
            fe_modp_mul(t, t, s);
            fe_modp_sub(up->y, t, centre->y);
            fe_clone(up->x, x3);
            fe_set64(up->z, 1);
        }
    }

    // Same contract as seq_group_scalar, eight i at a time on IFMA when the
    // CPU has it (see sweep_x8.hpp). `points` must have SGRP + 8 entries:
    // the vector version writes one padding slot past the group.
    void seq_group(pe* points, const pe* centre, fe* dx) {
#if ECLOOP_HAVE_FE52
        if (__builtin_expect(_sweep_x8, 1)) {
            seq_group_x8(points, centre, _stab, _sweep_scr);
            return;
        }
#endif
#if ECLOOP_HAVE_FE29
        if (_sweep_x4) {
            seq_group_x4(points, centre, _stab4, _sweep4_scr);
            return;
        }
#endif
        seq_group_scalar(points, centre, dx);
    }

    // Written to a temporary and renamed, so a crash during the write leaves
    // the previous checkpoint whole instead of a truncated line.
} // namespace

    // Declared in main.hpp: the manager writes the final checkpoint.
    void seq_checkpoint(u64 block) {
        fe k;
        if (block == ~0ULL) {
            // Every worker is past -e: the resume point is the first key
            // after the range (block * SEQ_BLOCK would wrap around).
            fe one;
            fe_set64(one, 1);
            fe_modn_add(k, seq_end, one);
        } else {
            seq_key_at(k, block * SEQ_BLOCK);
        }
        char hex[65];
        for (int w = 0; w < 4; ++w)
            std::snprintf(hex + w * 16, 17, "%016llX",
                          static_cast<unsigned long long>(k[3 - w]));
        const char* p = hex;
        while (p[1] && *p == '0') ++p;      // trimmed, so it pastes into -k

        std::string tmp = std::string(seq_ckpt) + ".tmp";
        if (std::FILE* f = std::fopen(tmp.c_str(), "w")) {
            std::fprintf(f, "%s\n", p);
            std::fclose(f);
            std::remove(seq_ckpt);
            std::rename(tmp.c_str(), seq_ckpt);
        }
    }

namespace {

    int seq_ckpt_seconds() {
        if (const char* e = std::getenv("ECLOOP_CKPT_SECS")) {
            const int v = std::atoi(e);
            if (v > 0) return v;
        }
        return 3600;
    }

    // Lowest block any worker is still on: everything below it is finished.
    u64 seq_low_block() {
        u64 low = seq_progress[0];
        for (int t = 1; t < seq_threads; ++t)
            if (seq_progress[t] < low) low = seq_progress[t];
        return low;
    }

    // ── deferred btree lookups (Mode::Bin, sweep and walk) ────────────────
    // A filter "maybe" -- about 4% of probes with a 2^29-bit filter, so 8%
    // of keys -- sends a hash into btree_search, which is a chain of
    // dependent DRAM misses: bucket head, first node, further nodes. Done
    // in line that is 300-500 ns per maybe (each array is hundreds of MB,
    // so every step also misses the TLB), 30-40 ns per key on average, and
    // the out-of-order window cannot hide it: it is 1000+ cycles. Measured
    // with a 21M-record database: a quarter of the time of a key.
    //
    // Here a maybe is put in a small pool instead and the lookup advances
    // ONE STEP PER HASHING BATCH (per 64 keys in the walk): each step reads
    // a cell that was prefetched during the previous step (about a
    // microsecond ago -- plenty for a miss), compares one node, and
    // prefetches the next. The chain is walked exactly like btree_search
    // does, just spread over time:
    //   step 0  prefetch the bucket head          (the batch it appeared in)
    //   step 1  read the head; prefetch the first node
    //   step k  compare node k-1; read its next pointer; prefetch node k
    // A hit or an empty next pointer ends the entry. Everything stays in
    // the L1/L2 hit range; the pool is drained in line at the end of a
    // group / block (a handful of entries) and if it ever fills.
    struct SweepPend {
        h160_t h33, h65;    // both hashes: report_found prints both addresses
        u64    idx;         // key = base + idx, points[idx - done] while in the group
        u64    born;        // batch counter at enqueue; not stepped in that batch
        u32    cell33, cell65;   // index of the cell holding the next node index
        u32    node33, node65;   // node whose line was prefetched, 0 at the head step
        u8     kinds;       // 1: h33 still being looked up, 2: h65
    };
    constexpr int SWEEP_POOL = 64;   // ~1.2 arrivals per batch, ~3 live on average
    struct SweepPool {
        SweepPend e[SWEEP_POOL];
        int n = 0;
    };
    ECLOOP_ALWAYS_INLINE u32 sweep_bucket(const u32* h) noexcept {
        const u8* b = reinterpret_cast<const u8*>(h);   // same as btree_search
        return 1 + (b[0] + b[1] * 256u + b[2] * 65536u);
    }
    ECLOOP_ALWAYS_INLINE void sweep_pf(const u32* p) noexcept {
#if defined(__x86_64__)
        _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0);
#else
        (void)p;
#endif
    }
    ECLOOP_ALWAYS_INLINE bool sweep_node_eq(const u32* tree, u32 pos, const u32* h) noexcept {
        return h[0] == tree[pos] && h[1] == tree[pos + 1] && h[2] == tree[pos + 2] &&
               h[3] == tree[pos + 3] && h[4] == tree[pos + 4];
    }
    // One step of one hash's lookup. Returns 1 on a hit, 0 otherwise;
    // clears *kind_bit in `kinds` when this hash is finished.
    ECLOOP_ALWAYS_INLINE int sweep_step_one(const u32* tree, const u32* h, u32& cell, u32& node,
                                            u8& kinds, u8 bit) noexcept {
        if (node != 0) {
            if (sweep_node_eq(tree, node, h)) return 1;
            cell = node + 5;
        }
        const u32 next = tree[cell];
        if (next == 0) { kinds = static_cast<u8>(kinds & ~bit); return 0; }
        node = next;
        sweep_pf(&tree[next]);
        sweep_pf(&tree[next + 5]);
        return 0;
    }
    inline void sweep_report(const SweepPend& e, const fe base, u64 done, const fe* limit,
                             const std::vector<pe>& points, bool com_only) {
        fe found, delta;
        fe_set64(delta, e.idx);
        fe_modn_add(found, base, delta);
        if (limit && fe_cmp(found, *limit) > 0) return;
        if (com_only) {
            // h65 was never computed for this batch; the report prints
            // both addresses, so make it now (a hit is rare).
            h160_t u65;
            addr65(u65, &points[static_cast<std::size_t>(e.idx - done)]);
            vanity::report_found(found, u65, e.h33);
        } else {
            vanity::report_found(found, e.h65, e.h33);
        }
    }
    template <class OnHit> inline void pool_finish_now(SweepPool& pool, int i, OnHit&& on_hit);
    // Add a key's maybe(s): both hashes are copied so the entry does not
    // depend on the hash buffers being kept, and the bucket heads are
    // prefetched (step 0). A full pool finishes its oldest entry in line.
    template <class OnHit>
    ECLOOP_ALWAYS_INLINE void pool_push(SweepPool& pool, const u32* h33, const u32* h65,
                                        bool m33, bool m65, u64 idx, u64 born, OnHit&& on_hit) {
        if (pool.n == SWEEP_POOL) pool_finish_now(pool, 0, on_hit);
        SweepPend& e = pool.e[pool.n++];
        std::memcpy(e.h33, h33, sizeof(h160_t));
        if (h65) std::memcpy(e.h65, h65, sizeof(h160_t));
        e.idx = idx;
        e.born = born;
        e.kinds = static_cast<u8>((m33 ? 1u : 0u) | (m65 ? 2u : 0u));
        e.node33 = e.node65 = 0;
        e.cell33 = m33 ? sweep_bucket(h33) : 0;
        e.cell65 = m65 ? sweep_bucket(h65) : 0;
        if (BTREE_HEAP) {
            if (m33) sweep_pf(&BTREE_HEAP[e.cell33]);
            if (m65) sweep_pf(&BTREE_HEAP[e.cell65]);
        }
    }
    // Finish entry i in line (btree_search from the start) and remove it.
    template <class OnHit>
    inline void pool_finish_now(SweepPool& pool, int i, OnHit&& on_hit) {
        const SweepPend& e = pool.e[i];
        const bool hit = ((e.kinds & 1) && btree_search(e.h33, nullptr) > 0) ||
                         ((e.kinds & 2) && btree_search(e.h65, nullptr) > 0);
        if (hit) on_hit(e);
        pool.e[i] = pool.e[--pool.n];
    }
    template <class OnHit>
    inline void pool_drain(SweepPool& pool, OnHit&& on_hit) {
        while (pool.n > 0) pool_finish_now(pool, pool.n - 1, on_hit);
    }
    // Advance every entry older than `now` by one step.
    template <class OnHit>
    inline void pool_advance(SweepPool& pool, u64 now, OnHit&& on_hit) {
        const u32* tree = BTREE_HEAP;
        for (int i = 0; i < pool.n;) {
            SweepPend& e = pool.e[i];
            if (e.born == now) { ++i; continue; }
            if (!tree) { pool_finish_now(pool, i, on_hit); continue; }
            int hit = 0;
            if (e.kinds & 1) hit |= sweep_step_one(tree, e.h33, e.cell33, e.node33, e.kinds, 1);
            if (!hit && (e.kinds & 2)) hit |= sweep_step_one(tree, e.h65, e.cell65, e.node65, e.kinds, 2);
            if (hit) {
                on_hit(e);
                e.kinds = 0;
            }
            if (e.kinds == 0) pool.e[i] = pool.e[--pool.n];
            else ++i;
        }
    }

    void sweep_from(const fe base, u64 count, int thread, const fe* limit,
                    std::vector<pe>& points, std::vector<fe>& dx, Mode vmode) {
        // Two sets: batch b is hashed while batch b-1 is looked up, see below.
        h160_t h33[2][HASH_BATCH], h65[2][HASH_BATCH];
        u64 scanned = 0;
        const bool com_only = _com_only && vmode == Mode::Bin;
        // The database lookup is always on when a database is loaded; -m
        // adds the prefix test on top of it, it does not replace it.
        const bool db_on = BTREE_HEAP != nullptr;
        // Pending btree lookups, see SweepPend. Drained at the end of every
        // group, so an entry never outlives the points it may still need.
        SweepPool pool;
        u64 batch_no = 0;
        u64 done = 0;   // group base, also read by on_hit below
        const auto on_hit = [&](const SweepPend& e) {
            sweep_report(e, base, done, limit, points, com_only);
        };

        fe ck, off;
        fe_set64(off, SHALF);
        fe_modn_add(ck, base, off);              // centre of the first group
        pe centre;
        ec_gtable_mul(&centre, ck);
        ec_jacobi_rdc(&centre, &centre);

        for (done = 0; done < count; done += SGRP) {
            seq_group(points.data(), &centre, dx.data());

            // Hash batch b, prefetch the filter words of its 32 hashes, and
            // only then look up batch b-1: its words have then had a whole
            // batch of hashing (over a microsecond) to arrive from memory.
            // With a filter of 2^30 bits every probe is a DRAM miss, and
            // without this the lookups wait for it in line. Same idea as
            // EC8_PFDIST in the walk.
            for (std::size_t b = 0; b <= SGRP; b += HASH_BATCH) {
                const unsigned cur = static_cast<unsigned>(b / HASH_BATCH) & 1u;
                if (b < SGRP) {
                    // The 16 prefetches of each hash kind are issued straight
                    // after its kernel, so the two groups are separated by a
                    // whole addr65 (~0.6 us): 16 misses in flight at a time
                    // fit the L1 fill buffers, 32 at once do not.
                    addr33_batchN(h33[cur], points.data() + b, HASH_BATCH);
                    if (db_on)
                        for (std::size_t q = 0; q < HASH_BATCH; ++q) btf_prefetch(h33[cur][q]);
                    if (!com_only) {
                        addr65_batchN(h65[cur], points.data() + b, HASH_BATCH);
                        if (db_on)
                            for (std::size_t q = 0; q < HASH_BATCH; ++q) btf_prefetch(h65[cur][q]);
                    }
                }
                if (b == 0) continue;
                const std::size_t pb = b - HASH_BATCH;   // the batch being checked
                const h160_t* const c33 = h33[cur ^ 1u];
                const h160_t* const c65 = h65[cur ^ 1u];

                for (std::size_t q = 0; q < HASH_BATCH; ++q) {
                    // hit33/hit65 answer "does this q's compressed/uncompressed
                    // hash count as a hit", whatever -m made that mean. Bin is
                    // the exact two lines this project always ran here.
                    // report_found (below, once past the !( hit33 || hit65 )
                    // check) always prints both addresses regardless of which
                    // flag fired, so UncCom/Triple setting both to the same
                    // verdict costs nothing extra.
                    bool hit33 = false, hit65 = false;
                    // The database, in every mode. Deferred, see SweepPend
                    // above: the filter is probed here (its word was
                    // prefetched a batch ago), the btree over the following
                    // batches.
                    if (db_on) {
                        const bool m33 = btf_maybe(c33[q]);
                        const bool m65 = !com_only && btf_maybe(c65[q]);
                        if (__builtin_expect(m33 || m65, 0))
                            pool_push(pool, c33[q], com_only ? nullptr : c65[q], m33, m65,
                                      done + pb + q, batch_no, on_hit);
                    }
                    // And the prefix, if -m asked for one.
                    switch (vmode) {
                    case Mode::Bin:
                        break;
                    case Mode::Unc:
                        hit33 = false;
                        hit65 = vanity::matches_prefix(c65[q]);
                        break;
                    case Mode::Com:
                        hit33 = vanity::matches_prefix(c33[q]);
                        hit65 = false;
                        break;
                    case Mode::Any:
                        hit33 = vanity::matches_prefix(c33[q]);
                        hit65 = vanity::matches_prefix(c65[q]);
                        break;
                    case Mode::UncCom:
                        hit33 = hit65 = vanity::shares_prefix_bytes(c65[q], c33[q], patsize);
                        break;
                    case Mode::Triple:
                        hit33 = hit65 = vanity::shares_prefix_bytes(c65[q], c33[q], patsize) &&
                                    (vanity::matches_prefix(c65[q]) ||
                                     vanity::matches_prefix(c33[q]));
                        break;
                    }
                    // Counted locally, flushed once per group: the per-worker
                    // counters share a cache line and touching it on every key
                    // has all workers invalidating it in each other's cache.
                    ++scanned;

                    // -v2 prints every key, -v5 adds both hashes, the same as
                    // the walk does. The key is rebuilt here rather than
                    // carried along, because outside verbose mode nobody wants
                    // to pay for it: this is the one place in the loop that
                    // runs 28 million times a second.
                    if (__builtin_expect(verb >= 2, 0)) {
                        fe vk, vd;
                        fe_set64(vd, done + pb + q);
                        fe_modn_add(vk, base, vd);
                        char hex[65];
                        for (int w = 0; w < 4; ++w)
                            std::snprintf(hex + w * 16, 17, "%016llX",
                                          static_cast<unsigned long long>(vk[3 - w]));
                        int nd = pksize / 4;
                        if (nd < 1 || nd > 64) nd = 64;
                        std::printf("%s %d\n", hex + (64 - nd), thread);
                        if (verb == 5) {
                            // Printed as they are. The walk byte-swaps first,
                            // which was right before the final swap came back
                            // into rmd160 and now reverses the words on screen.
                            print_h160(c33[q]);
                            if (com_only) { h160_t u; addr65(u, &points[pb + q]); print_h160(u); }
                            else print_h160(c65[q]);
                        }
                    }

                    if (__builtin_expect(!(hit33 || hit65), 1)) continue;

                    // points[idx] carries key base+idx, so no correction term.
                    fe found, delta;
                    fe_set64(delta, done + pb + q);
                    fe_modn_add(found, base, delta);
                    if (limit && fe_cmp(found, *limit) > 0) continue;
                    if (com_only) {
                        // h65 was never computed for this batch; the report
                        // prints both addresses, so make it now (a hit is rare).
                        h160_t u65;
                        addr65(u65, &points[pb + q]);
                        vanity::report_found(found, u65, c33[q]);
                    } else {
                        vanity::report_found(found, c65[q], c33[q]);
                    }
                }

                if (db_on) {
                    pool_advance(pool, batch_no, on_hit);
                    ++batch_no;
                }
            }
            // Drain before the next group overwrites `points`: a few
            // entries, looked up in line.
            if (db_on) pool_drain(pool, on_hit);
            thread_count[thread] += scanned;
            scanned = 0;
            ec_affine_add(&centre, &centre, &_sstep);
        }
    }

} // namespace

// The sweep itself. Reached from engine_x8 whenever a range was given.
static void engine_x8_seq(int thread) {
    const Mode vmode = resolve_mode(thread);
    check_mode_prereqs(vmode, thread);

    seq_prepare();
    const int T = seq_threads > 0 ? seq_threads : 1;
    const int ckpt_secs = seq_ckpt_seconds();

    std::vector<pe> points(SGRP + 8); // +8: seq_group_x8 writes one slot past the group
    std::vector<fe> dx(SHALF);
    auto last = std::chrono::steady_clock::now();

    for (u64 blk = static_cast<u64>(thread);; blk += static_cast<u64>(T)) {
        // Published before the work starts, so the minimum over workers never
        // runs ahead of what is actually finished.
        seq_progress[thread] = blk;

        fe base;
        seq_key_at(base, blk * SEQ_BLOCK);
        if (fe_cmp(base, seq_end) > 0) break;

        sweep_from(base, SEQ_BLOCK, thread, &seq_end, points, dx, vmode);

        const auto now = std::chrono::steady_clock::now();
        if (thread == 0 &&
            std::chrono::duration_cast<std::chrono::seconds>(now - last).count() >= ckpt_secs) {
            last = now;
            seq_checkpoint(seq_low_block());
        }
    }

    // The range ran out: publish a block past the end so the others are not
    // held back, then record where the run reached.
    seq_progress[thread] = ~0ULL;
    if (thread == 0) seq_checkpoint(seq_low_block());
}

// Mixed mode: the walker picks the high part of the key, the low mix_bits are
// then swept in full before the next draw.
//
// This is the arrangement that actually pays off when many people search the
// same puzzle. A pure sweep from the bottom of a range puts you in a queue
// behind everyone else doing the same, and only the fastest wins. A pure walk
// never collides but revisits its own keys forever. Drawing the high part at
// random makes your territory yours, and sweeping the low part means nothing
// inside it is ever checked twice.
//
// The walker only bumps digits ABOVE the swept zone -- otherwise it would land
// inside a block it is about to enumerate anyway.
static void engine_x8_mixed(int thread) {
    const Mode vmode = resolve_mode(thread);
    check_mode_prereqs(vmode, thread);

    seq_prepare();
    // Below -s: the key is ndig digits, the top hdig belong to the walker.
    // In a range (-k/-e): the same over the OFFSET from seq_start, and the
    // walker's digits are bounded by the offset of seq_end (its top hdig
    // digits: a block starting at or below that lies inside the range,
    // except for the tail of the last one, which the limit cuts off).
    const int ndig = walk_ndig;
    const int zdig = mix_bits / 4;          // hex digits taken by the sweep
    const int hdig = ndig - zdig;           // digits the walker may touch
    const u64 span = 1ULL << mix_bits;      // keys per draw

    char bound_buf[EC8_MAXDIG + 1];
    const char* bound = walk_bound(ndig, bound_buf);   // nullptr below -s
    // walk_step compares the first hdig digits only, so the full string
    // serves as the bound of the high part.

    std::vector<pe> points(SGRP + 8); // +8: seq_group_x8 writes one slot past the group
    std::vector<fe> dx(SHALF);

    std::vector<char> buff(static_cast<std::size_t>(ndig) + 1, '0');
    buff[ndig] = 0;

    const auto now = std::chrono::system_clock::now().time_since_epoch();
    fastPRNG::fastXS64 st(
        static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()) +
        static_cast<u64>(thread) * 0x9E3779B97F4A7C15ULL);

    // A random starting position in the high digits, so workers do not all
    // begin at the same place and grind the same blocks. In a range, redrawn
    // until it is at or below the bound.
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

        // Keys per draw: the whole block, or what is left of the range.
        u64 count = span;
        const fe* limit = nullptr;
        if (range_active) {
            limit = &seq_end;
            fe last, t;
            fe_set64(t, span - 1);
            fe_modn_add(last, base, t);
            if (fe_cmp(last, seq_end) > 0) {
                fe rem;
                fe_modn_sub(rem, seq_end, base);   // < span, fits a u64
                count = rem[0] + 1;
            }
        }
        sweep_from(base, count, thread, limit, points, dx, vmode);

        // Advance one hex digit of the high part, without carry, exactly as
        // the walk does (bounded in a range). The low digits stay zero:
        // they are the sweep's.
        const u8 lev = static_cast<u8>(((st.xoshiro256pp() >> 32) *
                                        static_cast<u64>(hdig)) >> 32);
        u8 used;
        walk_step(buff.data(), hdig, bound, lev, &used);
    }
}

void engine_x8(int thread) {
    {
        const char* e = std::getenv("ECLOOP_COMPRESSED_ONLY");
        _com_only = e && e[0] == '1';
        if (_com_only && thread == 0 && verb >= 1)
            std::printf("ECLOOP_COMPRESSED_ONLY=1: compressed addresses only\n");
    }
    // A range on the command line selects a sweep instead of the walk.
    // Both engine_x8_seq (-k/-e) and engine_x8_mixed (-z) resolve and check
    // -m for themselves before calling into sweep_from, which is shared by
    // all three search shapes -- see the Mode block above sweep_from.
    if (seq_active) { engine_x8_seq(thread); return; }
    if (mix_bits > 0) { engine_x8_mixed(thread); return; }
    // The endomorphism walk: over keys below 2^pksize (-s), or, with -k/-e
    // (-g), over offsets from seq_start that stay at or below seq_end. Both
    // are the same walk over an offset string; the range adds a bound (see
    // walk_step) and seq_start to every key.
    const Mode vmode = resolve_mode(thread);
    check_mode_prereqs(vmode, thread);
    const bool com_only = _com_only && vmode == Mode::Bin;
    const bool db_on = BTREE_HEAP != nullptr;   // database lookup in every mode

            const int ndig = walk_ndig;
            char bound_buf[EC8_MAXDIG + 1];
            const char* bound = walk_bound(ndig, bound_buf);
            // Keys per block: the base points alone, or with their two
            // endomorphism images (main.cpp decides, see endo_on there).
            const std::size_t npts = endo_on ? static_cast<std::size_t>(EC8_NPTS)
                                             : static_cast<std::size_t>(EC8_NBASE);

            AlignedArray<Pe8> blk(EC8_NSTEP);
            // Sized through ec_grprdc_x8_scratch, which accounts for the tree
            // form needing four times the space. Sizing by hand would be a
            // silent heap overflow the day EC8_INVTREE is turned on.
            AlignedArray<Fe8> scr(ec_grprdc_x8_scratch(EC8_NSTEP));
            AlignedArray<pe> pts(EC8_NPTS);
            AlignedArray<h160_t> H65(EC8_NPTS);
            AlignedArray<h160_t> H33(EC8_NPTS);

            // Walker state plus the position log used to rebuild a key.
            char wbuf[EC8_NWALK][EC8_MAXDIG + 1];  // current
            char wsnap[EC8_NWALK][EC8_MAXDIG + 1]; // as of block start
            u8 levlog[EC8_NSTEP][EC8_NWALK];   // position drawn, replayed by walk_step
            u8 kindlog[EC8_NSTEP][EC8_NWALK];  // move made, for the delta gather

            fastPRNG::fastXS64 st(worker_seed(thread));

            // Walker 0 starts at all zeros, exactly where the reference
            // engine starts. That is not cosmetic: from key 0 the small keys
            // are a step or two away -- 0x10 is one bump of digit 62 -- and
            // seeding the database with such keys is a reasonable way to
            // smoke-test a build. Giving every walker a random 64-digit start
            // made those keys unreachable in any practical time.
            //
            // Walkers 1..7 do get random starts, otherwise all eight chains
            // would sweep the same region from the same origin.
            //
            // EC8_RANDOM_START=1 randomises walker 0 as well.
            Pe8 P8;
            std::memset(&P8, 0, sizeof(P8));
            for (int w = 0; w < EC8_NWALK; ++w) {
                std::memset(wbuf[w], '0', sizeof(wbuf[w]));
                wbuf[w][ndig] = '\0';
#if !EC8_RANDOM_START
                if (w != 0)
#endif
                {
                    // Random digits; inside a range, redrawn until the offset
                    // is at or below the bound (the top digit decides, so a
                    // handful of draws at most).
                    for (int attempt = 0; attempt < 4096; ++attempt) {
                        for (int i = 0; i < ndig; ++i) {
                            const u32 d = static_cast<u32>(
                                ((st.xoshiro256pp() >> 32) * 16ULL) >> 32);
                            wbuf[w][i] = static_cast<char>(d < 10 ? '0' + d : 'A' + d - 10);
                        }
                        if (!bound || std::memcmp(wbuf[w], bound, static_cast<std::size_t>(ndig)) <= 0) break;
                        std::memset(wbuf[w], '0', static_cast<std::size_t>(ndig));
                    }
                }
                fe pk0;
                pe P0;
                walk_key(pk0, wbuf[w], ndig);
                ec_gtable_mul(&P0, pk0);
                pe8_set_lane(&P8, w, &P0);
            }

            u64 walk_scanned = 0;

            for (;;) {
                std::memcpy(wsnap, wbuf, sizeof(wbuf));

                // ===== phase 1: EC8_NSTEP batches of 8 points =====
                for (int s = 0; s < EC8_NSTEP; ++s) {
                    u8 moved[EC8_NWALK];
                    for (int w = 0; w < EC8_NWALK; ++w) {
                        const u8 lv = static_cast<u8>(rnd_pos(st, static_cast<u32>(ndig)));
                        levlog[s][w] = lv;
                        kindlog[s][w] = walk_step(wbuf[w], ndig, bound, lv, &moved[w]);
                    }

                    Fe8 qx, qy;
                    ec8_gather_delta(&qx, &qy, moved, kindlog[s]);
                    // Straight into blk[s]; the next step reads blk[s-1], no copy.
                    ec_jacobi_add_affine_x8(&blk[s], (s == 0) ? &P8 : &blk[s - 1],
                                            &qx, &qy);
                }
                P8 = blk[EC8_NSTEP - 1];

                // ===== phase 2: one inversion for the whole block =====
                ec_jacobi_grprdc_x8(blk.get(), EC8_NSTEP, scr.get());

                // ===== phase 3: endomorphism, 8 wide =====
                // pts[] layout: idx = branch*EC8_NBASE + s*8 + w, so the first
                // third is base points, then phi, then phi^2. With the
                // endomorphism off only the first third is filled and hashed.
                if (endo_on) {
                    Fe8 beta, beta2;
                    fe8_set_all(&beta, GLV_BETA);
                    fe8_set_all(&beta2, GLV_BETA2);
                    for (int s = 0; s < EC8_NSTEP; ++s) {
                        Fe8 x1, x2;
                        fe8_mul(&x1, &blk[s].x, &beta);
                        fe8_mul(&x2, &blk[s].x, &beta2);
                        for (int w = 0; w < EC8_NWALK; ++w) {
                            const std::size_t base =
                                static_cast<std::size_t>(s) * EC8_NWALK + w;
                            pe* p0 = &pts[base];
                            pe* p1 = &pts[EC8_NBASE + base];
                            pe* p2 = &pts[2 * EC8_NBASE + base];
                            pe8_get_lane(p0, &blk[s], w);
                            fe8_get_lane(p1->x, &x1, w);
                            fe_clone(p1->y, p0->y);
                            fe_set64(p1->z, 1);
                            fe8_get_lane(p2->x, &x2, w);
                            fe_clone(p2->y, p0->y);
                            fe_set64(p2->z, 1);
                        }
                    }
                } else {
                    for (int s = 0; s < EC8_NSTEP; ++s)
                        for (int w = 0; w < EC8_NWALK; ++w)
                            pe8_get_lane(&pts[static_cast<std::size_t>(s) * EC8_NWALK + w], &blk[s], w);
                }

                // ===== phase 4a: all hashes of the block =====
                for (std::size_t b = 0; b < npts; b += HASH_BATCH) {
                    const std::size_t nb =
                        (npts - b < HASH_BATCH)
                            ? (npts - b)
                            : HASH_BATCH;
                    if (!com_only) addr65_batchN(H65.get() + b, pts.get() + b, nb);
                    addr33_batchN(H33.get() + b, pts.get() + b, nb);
                }

                // ===== phase 4b: lookup with prefetch =====
                // Prefetching BTREE_HEAP only helps the Bin path; the other
                // modes never touch that table, so skip the wasted traffic.
                if (vmode == Mode::Bin) {
                    for (std::size_t i = 0; i < static_cast<std::size_t>(EC8_PFDIST) &&
                                            i < npts; ++i) {
                        if (!com_only) ec8_prefetch_bucket(BTREE_HEAP, H65[i]);
                        ec8_prefetch_bucket(BTREE_HEAP, H33[i]);
                    }
                }

                // A confirmed hit: rebuild the key of point j and verify it.
                // `db` says whether it came from the database (either hash
                // may have matched) or from the -m prefix test, whose rule
                // depends on the mode. Called in line for prefix hits and
                // from the lookup pool for database hits.
                const auto walk_hit = [&](std::size_t j, bool db) {
                    const std::size_t branch = j / EC8_NBASE;
                    const std::size_t base = j % EC8_NBASE;
                    const std::size_t s = base / EC8_NWALK, w = base % EC8_NWALK;
                    char key[EC8_MAXDIG + 1];
                    std::memcpy(key, wsnap[w], static_cast<std::size_t>(ndig) + 1);
                    for (std::size_t t = 0; t <= s; ++t) {
                        u8 lv_used;
                        walk_step(key, ndig, bound, levlog[t][w], &lv_used);
                    }

                    fe base_k, found;
                    walk_key(base_k, key, ndig);
                    if (branch == 0) fe_clone(found, base_k);
                    else if (branch == 1) fe_modn_mul(found, base_k, GLV_LAMBDA);
                    else fe_modn_mul(found, base_k, GLV_LAMBDA2);

                    // False positive check: rebuild the point from the
                    // recovered key and compare the hash.
                    pe V;
                    ec_gtable_mul(&V, found);
                    ec_jacobi_rdc(&V, &V);
                    h160_t v65, v33;
                    addr65(v65, &V);
                    addr33(v33, &V);
                    const bool ok65 = std::memcmp(v65, H65[j], sizeof(h160_t)) == 0;
                    const bool ok33 = std::memcmp(v33, H33[j], sizeof(h160_t)) == 0;
                    // Same check as Bin always did (ok65||ok33), extended
                    // to the other modes. Unc/Com/Any ask only for the
                    // hash(es) a hit could come from, same as Bin;
                    // UncCom/Triple test a RELATIONSHIP between the two
                    // hashes, so both must be confirmed reproduced, or a
                    // wrong recovered key could coincidentally satisfy it
                    // for some other, unrelated pair of hashes.
                    bool ok = false;
                    if (db) ok = com_only ? ok33 : (ok65 || ok33);
                    else switch (vmode) {
                    case Mode::Bin:    ok = ok65 || ok33; break;
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
                                     j, branch, key);
                        return;
                    }
                    // v65/v33 rather than H65[j]/H33[j]: identical when ok
                    // is true, but these are the ones just proven to come
                    // from `found` itself.
                    vanity::report_found(found, v65, v33);
                };
                const auto on_hit = [&](const SweepPend& e) { walk_hit(static_cast<std::size_t>(e.idx), true); };
                // Bin lookups go through the pool, one node per step; a step
                // every 64 keys is ~0.5 us, enough for the prefetched line.
                SweepPool pool;
                u64 step_no = 0;

                for (std::size_t j = 0; j < npts; ++j) {
                    if (db_on && j + EC8_PFDIST < npts) {
                        if (!com_only) ec8_prefetch_bucket(BTREE_HEAP, H65[j + EC8_PFDIST]);
                        ec8_prefetch_bucket(BTREE_HEAP, H33[j + EC8_PFDIST]);
                    }

                    // The database, in every mode (see SweepPend).
                    if (db_on) {
                        const bool m65 = !com_only && btf_maybe(H65[j]);
                        const bool m33 = btf_maybe(H33[j]);
                        if (__builtin_expect(m33 || m65, 0))
                            pool_push(pool, H33[j], com_only ? nullptr : H65[j], m33, m65,
                                      j, step_no, on_hit);
                        if ((j & 63) == 63) { pool_advance(pool, step_no, on_hit); ++step_no; }
                    }
                    // And the prefix, if -m asked for one.
                    bool hit = false;
                    switch (vmode) {
                    case Mode::Bin:
                        break;
                    case Mode::Unc:
                        hit = vanity::matches_prefix(H65[j]);
                        break;
                    case Mode::Com:
                        hit = vanity::matches_prefix(H33[j]);
                        break;
                    case Mode::Any:
                        hit = vanity::matches_prefix(H65[j]) || vanity::matches_prefix(H33[j]);
                        break;
                    case Mode::UncCom:
                        hit = vanity::shares_prefix_bytes(H65[j], H33[j], patsize);
                        break;
                    case Mode::Triple:
                        // Same condition as the sweep and engine.cpp: shared
                        // bytes AND the prefix on either address (this used
                        // to say "and both", which the shared bytes make the
                        // same event except at a range boundary).
                        hit = vanity::shares_prefix_bytes(H65[j], H33[j], patsize) &&
                              (vanity::matches_prefix(H65[j]) || vanity::matches_prefix(H33[j]));
                        break;
                    }

                    if (__builtin_expect(!hit, 1)) {
                        ++walk_scanned;
                        if (__builtin_expect(verb < 2, 1)) continue;
                    }

                    // Rare path: a hit, or verbose output. The key string is
                    // rebuilt by replaying this block's steps.
                    if (verb >= 2) {
                        const std::size_t base = j % EC8_NBASE;
                        const std::size_t s = base / EC8_NWALK, w = base % EC8_NWALK;
                        char key[EC8_MAXDIG + 1];
                        std::memcpy(key, wsnap[w], static_cast<std::size_t>(ndig) + 1);
                        for (std::size_t t = 0; t <= s; ++t) {
                            u8 lv_used;
                            walk_step(key, ndig, bound, levlog[t][w], &lv_used);
                        }
                        if (verb == 2) {
                            if (range_active) {
                                fe ak; char hex[65];
                                walk_key(ak, key, ndig);
                                fe_to_hex(hex, ak);
                                std::printf("%s %d\n", hex, thread);
                            } else {
                                std::printf("%s %d\n", key, thread);
                            }
                        }
                        if (verb == 5) {
                            u32 t1[5], t3[5];
                            if (com_only) addr65(H65[j], &pts[j]); // not hashed above
                            for (int k = 0; k < 5; ++k) t1[k] = detail::bswap32(H65[j][k]);
                            for (int k = 0; k < 5; ++k) t3[k] = detail::bswap32(H33[j][k]);
                            print_h160(t1);
                            print_h160(t3);
                        }
                    }

                    if (hit) walk_hit(j, false);
                }
                // The pool entries index this block's H33/H65 and its walker
                // state, so they are finished before the next block.
                if (db_on) pool_drain(pool, on_hit);
                thread_count[thread] += walk_scanned;
                walk_scanned = 0;
            }
        }

} // namespace ecloop
