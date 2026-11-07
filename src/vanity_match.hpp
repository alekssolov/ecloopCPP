// Copyright (c) vladkens
// https://github.com/vladkens/ecloop
// Licensed under the MIT License.
//
// Everything the -m search modes need that is not specific to one engine:
// matching hash160 against the -p prefix and against each other, deciding
// what a -m string means, checking its prerequisites, and printing a found
// key. Both engine_x8.cpp (the AVX-512 path) and engine.cpp (the scalar
// ECLOOP_OLD_ENGINE=1 reference) call into this file rather than each
// keeping its own copy, so the two engines can never disagree about what a
// mode means or what a result looks like. The BIN-database matching itself
// (btree_filter.*) is untouched and does not use any of this.

#pragma once

#include "types.hpp"
#include "addr.hpp"   // h160_t, fe

namespace ecloop {
namespace vanity {

    // -m's six possible meanings. Bin means "ignore vanity entirely, this is
    // the plain BIN-database search" -- every engine keeps that path exactly
    // as it always was; the other five select what matches_prefix and
    // shares_prefix_bytes below are asked.
    enum class Mode { Bin, Unc, Com, Any, UncCom, Triple };

    // Reads the global `mode` (main.hpp). Prints an error and calls
    // std::exit(1) if it is set but not one of the five known names.
    // Printing is limited to thread == 0 so `threads` workers do not all
    // print the same line; every worker still exits.
    Mode resolve_mode(int thread);

    // Checked once per worker before a vanity search starts: refuses to run
    // rather than scan forever finding nothing because -p or patsize was
    // never set up for the requested mode. Same thread == 0 printing rule
    // as resolve_mode.
    void check_mode_prereqs(Mode vmode, int thread);

    // True once a usable -p prefix has been read into at least one range
    // (see matches_prefix below); false means unc/com/any/triple have
    // nothing to search for and the caller should refuse to start rather
    // than scan forever finding nothing.
    bool has_ranges() noexcept;

    // True if h's Base58Check address starts with -p's prefix (its first
    // `patsize` characters, main.hpp's global -- same length add_prefix used
    // to build the pattern in the first place).
    //
    // Cheap on almost every call: a numeric range check over h alone,
    // reusing the exact ranges b58.cpp already built for the difficulty
    // print (b58.hpp's prefix_ranges()). The real address is built and
    // compared only for the rare candidate that passes that range -- the
    // same two-step shape btf_maybe() and btree_search() use for the BIN
    // path a few lines away in engine_x8.cpp.
    bool matches_prefix(const h160_t h) noexcept;

    // hash160 -> mainnet Base58Check address, NUL-terminated. out must hold
    // at least 40 bytes. Used by report_found below; exposed here too since
    // it is the one place in the project that turns a hash160 into the
    // address someone would actually paste into a wallet.
    void h160_to_address(const unsigned char h160[20], char out[40]);

    // hash160 (20 bytes, big-endian: out[0] is the most significant byte)
    // from the word form addr33/addr65 produce -- word i, printed with
    // %08x, is bytes [4i, 4i+3] of the digest.
    inline void h160_to_bytes(const h160_t h, unsigned char out[20]) noexcept {
        for (int i = 0; i < 5; ++i) {
            out[4 * i + 0] = static_cast<unsigned char>(h[i] >> 24);
            out[4 * i + 1] = static_cast<unsigned char>(h[i] >> 16);
            out[4 * i + 2] = static_cast<unsigned char>(h[i] >> 8);
            out[4 * i + 3] = static_cast<unsigned char>(h[i]);
        }
    }

    // True if a and b agree on their leading `nbytes` bytes. Clamped to
    // [0, 20]; nbytes <= 0 always matches. Defined here, not in the .cpp, so
    // the compiler can inline it at the one call site that matters -- the
    // per-point test in the search loop, where nbytes (patsize) is
    // loop-invariant.
    inline bool shares_prefix_bytes(const h160_t a, const h160_t b, int nbytes) noexcept {
        if (nbytes <= 0) return true;
        if (nbytes > 20) nbytes = 20;
        const int full = nbytes / 4;
        for (int i = 0; i < full; ++i)
            if (a[i] != b[i]) return false;
        const int rem = nbytes - full * 4;
        if (rem == 0) return true;
        const u32 shift = static_cast<u32>((4 - rem) * 8);
        return (a[full] >> shift) == (b[full] >> shift);
    }

    // The one "KEY FOUND" report every search path uses: private key in
    // uppercase hex, then the Base58Check address for both the uncompressed
    // and compressed forms of the same key, to stdout and appended to
    // key.txt. Both addresses are shown regardless of which hash actually
    // triggered the hit -- knowing the other one too is never wrong, and it
    // means one format instead of the three this project used to have
    // (engine_x8.cpp's seq_report, engine.cpp's seq_report_ref, and the
    // bare-hex printf in each main loop).
    void report_found(const fe key, const h160_t h_unc, const h160_t h_com);

} // namespace vanity
} // namespace ecloop
