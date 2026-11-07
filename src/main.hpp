// Declarations of what main.cpp provides to the engine modules.
//
// In the C tree engine_x8.c was #included into main.c, so it simply saw
// everything declared above the include point. Separate translation units
// need those names declared somewhere, and this header is that place --
// the C++ equivalent of "already defined above", nothing more.
//
// ON GLOBALS. These were file-scope statics in main.c and stay globals on
// purpose: workers are separate PROCESSES created by fork(), so each gets
// its own copy of everything except the mmap'd counters and the btree
// heap, which are shared deliberately. Turning them into a context object
// would be churn with no benefit for a fork-based design, and would
// obscure which of them really are shared.
//
// The only thing written from several processes is thread_count, which
// lives in a MAP_SHARED mapping. Each worker touches only its own slot, so
// there is no contention and no need for atomics.

#pragma once

#include <string>

#include "types.hpp"
#include "ecc.hpp"   // fe, for the sweep declarations

namespace ecloop {

    inline constexpr u32 BTREE_HSIZE = 1 + 256 * 256 * 256;

    // Private key size in BITS. The hex string driving the walk is
    // pksize/4 digits long.
    extern int pksize;

    // 0 quiet, 1 rate line, 2 print every key, 5 print hashes too.
    extern int verb;

    // Per-worker key counters, MAP_SHARED so the manager can read them.
    extern u64* thread_count;

    // The hash160 database. Loaded before fork(), so every worker shares
    // the same physical pages copy-on-write.
    extern u32* BTREE_HEAP;
    extern u32  BTREE_HEAP_SIZE;

    // Returns the record position, or 0 when absent. nullptr means
    // BTREE_HEAP.
    u32 btree_search(const u32* hash, const u32* tree) noexcept;

    // Reference engine: one walker, the correctness baseline.
    // ECLOOP_OLD_ENGINE=1 selects it.
    void engine(int thread);

    // Eight independent walkers through the batch-8 IFMA layer.
    void engine_x8(int thread);


    // ── sequential sweep ─────────────────────────────────────────────────────
    // Turned on by giving a start key on the command line; there is no
    // separate switch. All of these are set before fork and read-only
    // afterwards, except seq_progress, which lives in the shared mapping.
    extern bool  seq_active;     // the sequential sweep of the range was selected
    // -k/-e were given: seq_start..seq_end is the key range. Without -z or
    // -g that is the sequential sweep (seq_active); with -z the mixed mode
    // draws inside it; with -g the endomorphism walk moves inside it.
    extern bool  range_active;
    extern bool  glv_walk;       // -g: the walk inside the -k/-e range
    // Whether the walk also checks the two endomorphism images lambda*k
    // and lambda^2*k of every base key. Decided in main.cpp before fork:
    // -g on / off, or automatically -- on only when the walk covers the
    // whole key space (-s 256), because the images of a key are spread
    // over the whole group and never fall inside a smaller -s or a range.
    extern bool  endo_on;
    // Hex digits of the offset from seq_start when range_active (the walk
    // and the mixed mode step through offsets, not absolute keys); equals
    // pksize/4 otherwise. Set by main.cpp before fork, delta_init'd to match.
    extern int   walk_ndig;
    extern fe    seq_start;      // first key, from -k
    extern fe    seq_end;        // last key, from -e; all ones when absent
    extern int   seq_threads;    // worker count, needed for the block stride
    extern u64*  seq_progress;   // block each worker is on, MAP_SHARED
    extern char  seq_ckpt[256];  // progress file, named after the start key

    // seq_start + offset.
    void seq_key_at(fe out, u64 offset);

    // Writes the resume key of block `block` to seq_ckpt; ~0 means the range
    // is finished and writes the first key after seq_end. Defined in
    // engine_x8.cpp, called by the workers and, at the end, by the manager.
    void seq_checkpoint(u64 block);

    // Mixed mode: low mix_bits of the key are swept in full after every
    // draw of the walker. Zero disables it.
    extern int mix_bits;

    // ── search mode ──────────────────────────────────────────────────────────
    // -m: "" (default) is the BIN-database search, untouched by any of this.
    // "unc"/"com"/"any"/"unccom"/"triple" select engine_x8.cpp's vanity
    // matching instead; see vanity_match.hpp. Set before fork, read-only
    // afterwards, same as everything else on this page.
    extern std::string mode;

    // -p, verbatim (leading '1' already added by main.cpp's option parsing).
    // Empty when -p was not given.
    extern std::string prefix;

    // prefix.size(). unc/com/any/triple pass this to vanity::matches_prefix
    // as the number of leading characters of the Base58Check address that
    // must equal prefix. unccom/triple also reuse it, with a different unit:
    // there it is the number of leading BYTES of hash160 that must agree
    // between the compressed and uncompressed address, prefix's own
    // characters going unused -- see vanity_match.hpp.
    extern int patsize;

    // Called from vanity::report_found (vanity_match.cpp), once per
    // reported find, from whichever worker found it: wakes manager_loop's
    // select() early over the same socketpair used to detect a dead parent,
    // so the probability display can reset its baseline right when a find
    // happens instead of up to a second late. Best-effort -- a failed or
    // lost write just means the reset is missed once, not a crash.
    void notify_found();

} // namespace ecloop
