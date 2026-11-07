// Helpers of the random walk shared by engine_x8.cpp and the reference
// engine.cpp: the bounded digit step, the range bound as a digit string,
// and offset <-> absolute key. Header-only so both engines run exactly the
// same rule and can be checked against each other.

#pragma once

#include <cstdio>
#include <cstring>

#include "main.hpp"
#include "compat.hpp"
#include "ecc.hpp"

namespace ecloop {

    // fe_modn_from_hex wants a full 64-digit string; the walkers keep a
    // shorter one, so it is padded on the left here.
    inline void fe_modn_from_hex_pad(fe out, const char* digits, int n) {
        char buf[65];
        std::memset(buf, '0', 64);
        buf[64] = 0;
        std::memcpy(buf + 64 - n, digits, static_cast<std::size_t>(n));
        fe_modn_from_hex(out, buf);
    }

    ECLOOP_ALWAYS_INLINE u8 walk_hexval(char c) noexcept {
        return static_cast<u8>(c <= '9' ? c - '0' : c - 'A' + 10);
    }

    ECLOOP_ALWAYS_INLINE bool walk_bump(char* c) noexcept {
        const char v = *c;
        if (v == 'F') { *c = '0'; return true; }
        *c = (v == '9') ? 'A' : static_cast<char>(v + 1);
        return false;
    }

    // One walk step that stays inside a range. `key` is the hex string of
    // the OFFSET from the range start (nd digits, upper case), `bound` the
    // offset of the range end in the same form, or nullptr for no bound
    // (the plain -s walk, where every bump is valid). The digit at position
    // lv is bumped by one without carry ('F' -> '0'); if that would take the
    // offset past the bound, the digit is set to 0 instead (a move of
    // -d*16^j), and if it already is 0 the next lower digit is tried, and
    // so on round the string. A move always exists for bound >= 1: the
    // leftmost non-zero digit can be zeroed, and 0 can be incremented at
    // the bottom.
    //
    // Returns the move kind for delta_point_kind: 0 = +16^j, d = 1..15 =
    // -d*16^j; *lev_out is the position moved. Pure in (key, lv): the hit
    // path replays it from the logged lv. Lexicographic order of
    // same-length upper-case hex strings is numeric order, so memcmp is
    // the range test.
    ECLOOP_ALWAYS_INLINE u8 walk_step(char* key, int nd, const char* bound,
                                      u8 lv, u8* lev_out) noexcept {
        for (int tries = 0; tries < nd; ++tries) {
            char* c = &key[lv];
            const char old = *c;
            *lev_out = lv;
            if (old == 'F') { *c = '0'; return 15; }
            *c = (old == '9') ? 'A' : static_cast<char>(old + 1);
            if (!bound || std::memcmp(key, bound, static_cast<std::size_t>(nd)) <= 0) return 0;
            *c = old;                       // past the bound: undo
            if (old != '0') { *c = '0'; return walk_hexval(old); }
            lv = static_cast<u8>(lv + 1 == nd ? 0 : lv + 1);
        }
        // Unreachable for bound >= 1: bump digit 0 and report it, so the
        // point and the string still move together.
        *lev_out = 0;
        return walk_bump(&key[0]) ? 15 : 0;
    }

    // The offset of seq_end from seq_start as an nd-digit upper-case hex
    // string (the walk's bound), or nullptr when no range is set. Hidden
    // digits above nd are zero by construction of nd.
    inline const char* walk_bound(int nd, char* buf) noexcept {
        if (!range_active) return nullptr;
        fe span;
        fe_modn_sub(span, seq_end, seq_start);
        for (int i = 0; i < nd; ++i) {
            const int bit = 4 * (nd - 1 - i);
            const unsigned d = static_cast<unsigned>((span[bit / 64] >> (bit % 64)) & 15u);
            buf[i] = static_cast<char>(d < 10 ? '0' + d : 'A' + d - 10);
        }
        buf[nd] = 0;
        return buf;
    }

    // Absolute key of an offset string: seq_start + offset when a range is
    // set (offset digits above nd are zero), the offset itself otherwise.
    inline void walk_key(fe out, const char* offset, int nd) noexcept {
        fe_modn_from_hex_pad(out, offset, nd);
        if (range_active) fe_modn_add(out, out, seq_start);
    }

    inline void fe_to_hex(char out[65], const fe k) noexcept {
        for (int w = 0; w < 4; ++w)
            std::snprintf(out + w * 16, 17, "%016llX",
                          static_cast<unsigned long long>(k[3 - w]));
    }

} // namespace ecloop
