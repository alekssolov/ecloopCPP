// Implementation of vanity_match.hpp.
//
// The only thing this file borrows from b58.cpp is the read-only
// prefix_ranges() getter -- the numeric ranges add_prefix already built for
// the difficulty print. Everything else (compiling those ranges into a
// sorted, fast-comparable form; building a real address from a hash160;
// deciding what -m means; printing a found key) is written fresh here, so
// b58.cpp's own logic for turning a prefix string into ranges is used
// exactly once, not duplicated, and neither engine file needs its own copy
// of any of this.

#include "vanity_match.hpp"

#include "b58.hpp"     // b58enc, prefix_ranges()
#include "main.hpp"    // extern std::string mode, prefix; extern int patsize;
#include "sha256.hpp"  // sha256_final

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace ecloop {
namespace vanity {

namespace {

    struct RangeW {
        u32 low[5];
        u32 high[5];
    };

    inline void bytes_to_words(const unsigned char b[20], u32 w[5]) noexcept {
        for (int i = 0; i < 5; ++i)
            w[i] = (static_cast<u32>(b[4 * i]) << 24) |
                   (static_cast<u32>(b[4 * i + 1]) << 16) |
                   (static_cast<u32>(b[4 * i + 2]) << 8) |
                   static_cast<u32>(b[4 * i + 3]);
    }

    inline int cmp5(const u32 a[5], const u32 b[5]) noexcept {
        for (int i = 0; i < 5; ++i) {
            if (a[i] < b[i]) return -1;
            if (a[i] > b[i]) return 1;
        }
        return 0;
    }

    // Same double-SHA256 checksum b58.cpp's own reachability check uses
    // (sha256_padded there); written fresh here rather than exported, so
    // this file's one dependency on b58.cpp stays the plain ranges getter.
    void sha256_once(const unsigned char* data, std::size_t len,
                      unsigned char out[32]) noexcept {
        unsigned char blk[128] = {};
        std::memcpy(blk, data, len);
        blk[len] = 0x80;
        const std::size_t total = (len + 9 <= 64) ? 64 : 128;
        const std::uint64_t bits = static_cast<std::uint64_t>(len) * 8;
        for (int i = 0; i < 8; ++i)
            blk[total - 1 - i] = static_cast<unsigned char>(bits >> (i * 8));

        u32 st[8];
        sha256_final(st, blk, static_cast<u32>(total));
        for (int i = 0; i < 8; ++i) {
            out[i * 4 + 0] = static_cast<unsigned char>(st[i] >> 24);
            out[i * 4 + 1] = static_cast<unsigned char>(st[i] >> 16);
            out[i * 4 + 2] = static_cast<unsigned char>(st[i] >> 8);
            out[i * 4 + 3] = static_cast<unsigned char>(st[i]);
        }
    }

    struct Compiled {
        bool ok = false;
        std::vector<RangeW> ranges; // sorted by low, disjoint
        std::string prefix;
    };

    // Built once per process, on first use. Workers are separate fork()ed
    // processes (main.hpp's note on globals), so this runs again in each of
    // them -- redundant but cheap, a handful of ranges b58.cpp already
    // computed before fork, and it means no extra hook is needed in
    // main.cpp.
    const Compiled& compiled() noexcept {
        static const Compiled c = [] {
            Compiled r;
            r.prefix = ecloop::prefix;
            for (const PrefixRange& pr : prefix_ranges()) {
                RangeW w;
                bytes_to_words(pr.low, w.low);
                bytes_to_words(pr.high, w.high);
                r.ranges.push_back(w);
            }
            std::sort(r.ranges.begin(), r.ranges.end(),
                     [](const RangeW& a, const RangeW& b) { return cmp5(a.low, b.low) < 0; });
            r.ok = !r.ranges.empty();
            return r;
        }();
        return c;
    }

} // namespace

bool has_ranges() noexcept { return compiled().ok; }

void h160_to_address(const unsigned char h160[20], char out[40]) {
    unsigned char body[25];
    body[0] = 0x00; // mainnet P2PKH version byte
    std::memcpy(body + 1, h160, 20);

    unsigned char d1[32], d2[32];
    sha256_once(body, 21, d1);
    sha256_once(d1, 32, d2);
    std::memcpy(body + 21, d2, 4);

    b58enc(out, body, 25);
}

bool matches_prefix(const h160_t h) noexcept {
    const Compiled& c = compiled();
    if (!c.ok) return false;

    // Ranges are disjoint (PatternSet merges overlaps as it builds them) and
    // sorted by low: binary search for the last range whose low is <= h,
    // then one compare against its high settles the reject case.
    std::size_t lo = 0, hi = c.ranges.size();
    while (lo < hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (cmp5(c.ranges[mid].low, h) <= 0) lo = mid + 1;
        else hi = mid;
    }
    if (lo == 0 || cmp5(h, c.ranges[lo - 1].high) > 0) return false;

    // Rare path: build the real address and compare it for real, exactly
    // the way b58.cpp's own reachability check does.
    unsigned char raw[20];
    h160_to_bytes(h, raw);
    char addr[40];
    h160_to_address(raw, addr);
    return std::strncmp(addr, c.prefix.c_str(), static_cast<std::size_t>(patsize)) == 0;
}

Mode resolve_mode(int thread) {
    if (mode.empty())    return Mode::Bin;
    if (mode == "unc")    return Mode::Unc;
    if (mode == "com")    return Mode::Com;
    if (mode == "any")    return Mode::Any;
    if (mode == "unccom") return Mode::UncCom;
    if (mode == "triple") return Mode::Triple;

    if (thread == 0)
        std::fprintf(stderr,
                    "-m [%s]: unknown mode. Expected unc, com, any, unccom, "
                    "triple, or omit -m for the BIN-database search.\n",
                    mode.c_str());
    std::exit(1);
}

void check_mode_prereqs(Mode vmode, int thread) {
    // -p was registered (for the difficulty print in main.cpp) but Bin --
    // reachable here only when -m was never given -- means nothing tells
    // the search to actually use it: the run would just fall through to a
    // plain BIN search, having printed a difficulty it then never checks
    // against.
    if (vmode == Mode::Bin && !prefix.empty()) {
        if (thread == 0)
            std::fprintf(stderr,
                        "-p was given without -m: the difficulty was printed, "
                        "but nothing tells the search to use the prefix -- it "
                        "would just run a plain BIN search instead. Add -m "
                        "unc/com/any/unccom/triple to search for the prefix, "
                        "or drop -p to run the BIN search as before.\n");
        std::exit(1);
    }

    const bool needs_pattern = vmode == Mode::Unc || vmode == Mode::Com ||
                               vmode == Mode::Any || vmode == Mode::Triple;
    if (needs_pattern && !has_ranges()) {
        if (thread == 0)
            std::fprintf(stderr,
                        "-m %s needs at least one -p prefix; none was "
                        "registered.\n", mode.c_str());
        std::exit(1);
    }

    const bool needs_patsize = vmode == Mode::UncCom || vmode == Mode::Triple;
    if (needs_patsize && patsize <= 0) {
        if (thread == 0)
            std::fprintf(stderr,
                        "-m %s needs -p to set how many leading bytes of "
                        "hash160 must agree (patsize); got 0. Pass any -p "
                        "of the desired length, e.g. -p 111111 for 6 "
                        "bytes -- its content is not used in this mode.\n",
                        mode.c_str());
        std::exit(1);
    }
    if (needs_patsize && thread == 0 && patsize > 20)
        std::fprintf(stderr,
                    "NOTE: -m %s: patsize=%d exceeds the 20 bytes of a "
                    "hash160; treated as 20 (exact equality).\n",
                    mode.c_str(), patsize);
}

void report_found(const fe key, const h160_t h_unc, const h160_t h_com) {
    char keyhex[65];
    for (int w = 0; w < 4; ++w)
        std::snprintf(keyhex + w * 16, 17, "%016llX",
                      static_cast<unsigned long long>(key[3 - w]));

    unsigned char raw_unc[20], raw_com[20];
    h160_to_bytes(h_unc, raw_unc);
    h160_to_bytes(h_com, raw_com);
    char addr_unc[40], addr_com[40];
    h160_to_address(raw_unc, addr_unc);
    h160_to_address(raw_com, addr_com);

    std::printf("\n\n*******************************KEY FOUND"
                "*******************************\n\n");
    std::printf("PrivKey(HEX)      : %s\n", keyhex);
    std::printf("Address(uncompr.) : %s\n", addr_unc);
    std::printf("Address(compr.)   : %s\n", addr_com);
    std::fflush(stdout);

    if (std::FILE* f = std::fopen("key.txt", "a+")) {
        std::fprintf(f, "PrivKey(HEX)      : %s\n", keyhex);
        std::fprintf(f, "Address(uncompr.) : %s\n", addr_unc);
        std::fprintf(f, "Address(compr.)   : %s\n\n", addr_com);
        std::fclose(f);
    } else {
        std::fprintf(stderr, "ERROR: could not open result file: %s\n",
                     std::strerror(errno));
    }

    // main.cpp's manager_loop resets the probability display's baseline
    // right when this arrives, instead of accumulating from program start
    // (or the previous find) forever.
    notify_found();
}

} // namespace vanity
} // namespace ecloop
