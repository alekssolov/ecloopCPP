// txt2bin -- turn a text list of hash160 values into the .bin the search
// reads, and back again.
//g++ -std=gnu++17 -O2 tools/txt2bin.cpp -o txt2bin
//   txt2bin [-u] [-o out.bin] in.txt [more.txt ...]
//   txt2bin -d file.bin
//
//   -o FILE  output file. Defaults to the first input with its extension
//            replaced by .bin, so `txt2bin 10.txt` writes 10.bin.
//   -u       drop duplicates. btree_add already ignores repeats on load,
//            so this only makes the file smaller.
//   -d       dump a .bin back to hex on stdout (-a: as addresses). Use it to
//            check a database you did not build yourself: the output should
//            read like the addresses you expect.
//
//   -a       with -d: print Base58Check P2PKH addresses instead of hex.
//
// INPUT LINES may be hash160 hex (40 digits) or addresses, mixed freely:
//   1...        P2PKH Base58Check: decoded, checksum verified
//   bc1q...     P2WPKH bech32 (42 chars): the witness program IS the
//               hash160 of the compressed key, so it is searchable too
//   3... / bc1q (62 chars) / bc1p...  P2SH, P2WSH, taproot: no key hashes
//               to these directly, reported and skipped
//
// THE FORMAT is simply the hex written as bytes: line
// 91b24bf9f5288532960ac687abb035127b1d28a5 becomes those twenty bytes, most
// significant first, and records follow one another with no header and no
// separator. The byte swap the search needs happens in btree_load, not
// here, so a .bin produced by this tool reads back through -d unchanged.
//
// Input is forgiving: blank lines, `#` comments, leading and trailing
// whitespace, CRLF line endings, a UTF-8 BOM and either letter case are all
// accepted. Anything else is reported with its line number and skipped, so
// one bad line in a million-line list does not cost you the whole run.

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

    constexpr std::size_t H160_BYTES = 20;
    constexpr std::size_t H160_CHARS = 40;

    void usage(const char* argv0) {
        std::fprintf(stderr,
                     "usage: %s [-u] [-o out.bin] in.txt [more.txt ...]\n"
                     "       %s -d file.bin\n\n"
                     "  -o FILE  output file (default: first input with .bin)\n"
                     "  -u       drop duplicates\n"
                     "  -d       dump a .bin back to hex on stdout\n"
                     "  -a       with -d: print addresses instead of hex\n"
                     "input lines: 40 hex digits, 1... (P2PKH) or bc1q... (P2WPKH) addresses\n",
                     argv0, argv0);
    }

    int hex_val(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    }

    // Strips a BOM, comments, and whitespace including the CR of a CRLF
    // ending -- text files coming from Windows are the normal case here,
    // and a stray CR would otherwise make every single line invalid.
    void clean(std::string& s, bool first_line) {
        if (first_line && s.size() >= 3 && static_cast<unsigned char>(s[0]) == 0xEF &&
            static_cast<unsigned char>(s[1]) == 0xBB &&
            static_cast<unsigned char>(s[2]) == 0xBF) {
            s.erase(0, 3);
        }
        const std::size_t hash = s.find('#');
        if (hash != std::string::npos) s.erase(hash);
        std::size_t b = 0, e = s.size();
        while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
        while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
        s = s.substr(b, e - b);
    }

    // ── SHA-256, for the Base58Check checksum only ─────────────────────
    void sha256(const unsigned char* msg, std::size_t len, unsigned char out[32]) {
        static const unsigned K[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        unsigned h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
        std::vector<unsigned char> m(msg, msg + len);
        m.push_back(0x80);
        while (m.size() % 64 != 56) m.push_back(0);
        const unsigned long long bits = static_cast<unsigned long long>(len) * 8;
        for (int i = 7; i >= 0; --i) m.push_back(static_cast<unsigned char>(bits >> (8 * i)));
        auto rotr = [](unsigned x, int n) { return (x >> n) | (x << (32 - n)); };
        for (std::size_t off = 0; off < m.size(); off += 64) {
            unsigned w[64];
            for (int i = 0; i < 16; ++i)
                w[i] = (unsigned(m[off+4*i]) << 24) | (unsigned(m[off+4*i+1]) << 16) |
                       (unsigned(m[off+4*i+2]) << 8) | unsigned(m[off+4*i+3]);
            for (int i = 16; i < 64; ++i) {
                const unsigned s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
                const unsigned s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
                w[i] = w[i-16] + s0 + w[i-7] + s1;
            }
            unsigned a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
            for (int i = 0; i < 64; ++i) {
                const unsigned S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
                const unsigned ch = (e & f) ^ (~e & g);
                const unsigned t1 = hh + S1 + ch + K[i] + w[i];
                const unsigned S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
                const unsigned mj = (a & b) ^ (a & c) ^ (b & c);
                const unsigned t2 = S0 + mj;
                hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
            }
            h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
        }
        for (int i = 0; i < 8; ++i) {
            out[4*i] = static_cast<unsigned char>(h[i] >> 24); out[4*i+1] = static_cast<unsigned char>(h[i] >> 16);
            out[4*i+2] = static_cast<unsigned char>(h[i] >> 8); out[4*i+3] = static_cast<unsigned char>(h[i]);
        }
    }

    const char* B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    // Base58Check -> version byte + hash160. Returns an explanation on
    // failure, nullptr on success.
    const char* b58check_decode(const std::string& s, unsigned char* version, unsigned char h160[H160_BYTES]) {
        if (s.size() < 26 || s.size() > 35) return "not an address (length)";
        unsigned char num[25] = {0};   // big number, big-endian in 25 bytes
        for (char c : s) {
            const char* p = std::strchr(B58, c);
            if (!p || c == 0) return "not base58";
            unsigned carry = static_cast<unsigned>(p - B58);
            for (int i = 24; i >= 0; --i) {
                carry += 58u * num[i];
                num[i] = static_cast<unsigned char>(carry & 0xFF);
                carry >>= 8;
            }
            if (carry) return "not an address (overflow)";
        }
        // Leading '1's are leading zero bytes; the number above already
        // has them as long as the value fits 25 bytes, which the overflow
        // check guarantees. So: version, 20 bytes, 4 bytes of checksum.
        unsigned char d1[32], d2[32];
        sha256(num, 21, d1);
        sha256(d1, 32, d2);
        if (std::memcmp(d2, num + 21, 4) != 0) return "bad checksum (typo?)";
        *version = num[0];
        std::memcpy(h160, num + 1, H160_BYTES);
        return nullptr;
    }

    void b58check_encode(unsigned char version, const unsigned char h160[H160_BYTES], char out[40]) {
        unsigned char body[25];
        body[0] = version;
        std::memcpy(body + 1, h160, H160_BYTES);
        unsigned char d1[32], d2[32];
        sha256(body, 21, d1);
        sha256(d1, 32, d2);
        std::memcpy(body + 21, d2, 4);
        char tmp[40];
        int n = 0;
        unsigned char x[25];
        std::memcpy(x, body, 25);
        for (;;) {   // repeated division by 58
            unsigned rem = 0;
            bool zero = true;
            for (int i = 0; i < 25; ++i) {
                const unsigned v = (rem << 8) | x[i];
                x[i] = static_cast<unsigned char>(v / 58);
                rem = v % 58;
                if (x[i]) zero = false;
            }
            tmp[n++] = B58[rem];
            if (zero) break;
        }
        for (int i = 0; i < 25 && body[i] == 0; ++i) tmp[n++] = '1';
        for (int i = 0; i < n; ++i) out[i] = tmp[n - 1 - i];
        out[n] = 0;
    }

    // ── bech32 (BIP173) P2WPKH: bc1q + 20-byte program ─────────────────
    const char* BECH32 = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

    unsigned bech32_polymod(const std::vector<unsigned char>& v) {
        static const unsigned G[5] = {0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3};
        unsigned chk = 1;
        for (unsigned char x : v) {
            const unsigned top = chk >> 25;
            chk = ((chk & 0x1ffffff) << 5) ^ x;
            for (int i = 0; i < 5; ++i) if ((top >> i) & 1) chk ^= G[i];
        }
        return chk;
    }

    // Returns nullptr and the 20-byte program on success, else the reason.
    const char* bech32_decode_p2wpkh(std::string s, unsigned char h160[H160_BYTES]) {
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (s.size() < 8 || s.compare(0, 4, "bc1q") != 0) return "not bech32";
        std::vector<unsigned char> data;
        for (std::size_t i = 3; i < s.size(); ++i) {
            const char* p = std::strchr(BECH32, s[i]);
            if (!p || s[i] == 0) return "not bech32 (character)";
            data.push_back(static_cast<unsigned char>(p - BECH32));
        }
        std::vector<unsigned char> hrp_exp = {'b' >> 5, 'c' >> 5, 0, 'b' & 31, 'c' & 31};
        std::vector<unsigned char> chk(hrp_exp);
        chk.insert(chk.end(), data.begin(), data.end());
        if (bech32_polymod(chk) != 1) return "bad bech32 checksum";   // v0 uses plain bech32
        if (data.size() < 7) return "not bech32 (short)";
        if (data[0] != 0) return "witness version is not 0 (bc1p taproot?)";
        // 5-bit groups -> bytes, without the witness version and the 6 checksum groups
        std::vector<unsigned char> prog;
        unsigned acc = 0; int bits = 0;
        for (std::size_t i = 1; i + 6 < data.size(); ++i) {
            acc = (acc << 5) | data[i]; bits += 5;
            if (bits >= 8) { bits -= 8; prog.push_back(static_cast<unsigned char>((acc >> bits) & 0xFF)); }
        }
        if (prog.size() != H160_BYTES) return "not P2WPKH (program is not 20 bytes: P2WSH?)";
        std::memcpy(h160, prog.data(), H160_BYTES);
        return nullptr;
    }

    // One cleaned input line -> hash160. Returns nullptr on success, or a
    // message to show with the line number.
    const char* parse_line(const std::string& line, unsigned char rec[H160_BYTES]) {
        if (line.size() == H160_CHARS) {
            for (std::size_t i = 0; i < H160_BYTES; ++i) {
                const int hi = hex_val(line[i * 2]), lo = hex_val(line[i * 2 + 1]);
                if (hi < 0 || lo < 0) return "40 characters but not hexadecimal";
                rec[i] = static_cast<unsigned char>((hi << 4) | lo);
            }
            return nullptr;
        }
        if (line[0] == '1') {
            unsigned char version;
            const char* err = b58check_decode(line, &version, rec);
            if (err) return err;
            if (version != 0) return "not a mainnet P2PKH address";
            return nullptr;
        }
        if (line[0] == '3') return "P2SH (3...): a script hash, no private key hashes to it, skipped";
        if (line.size() > 4 && (line.compare(0, 4, "bc1p") == 0 || line.compare(0, 4, "BC1P") == 0))
            return "taproot (bc1p...): a tweaked key, not a hash160, skipped";
        if (line.size() > 3 && (line.compare(0, 3, "bc1") == 0 || line.compare(0, 3, "BC1") == 0))
            return bech32_decode_p2wpkh(line, rec);
        return "expected 40 hex digits, a 1... address or a bc1q... address";
    }

    int dump(const char* path, bool as_addr) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            std::fprintf(stderr, "%s: cannot open\n", path);
            return 1;
        }
        unsigned char rec[H160_BYTES];
        std::size_t n = 0;
        while (in.read(reinterpret_cast<char*>(rec), H160_BYTES)) {
            if (as_addr) {
                char addr[40];
                b58check_encode(0, rec, addr);
                std::printf("%s\n", addr);
            } else {
                char line[H160_CHARS + 2];
                for (std::size_t i = 0; i < H160_BYTES; ++i)
                    std::snprintf(line + i * 2, 3, "%02X", rec[i]);
                std::printf("%s\n", line);
            }
            ++n;
        }
        // A partial record means the file is truncated or is not a hash160
        // database at all. Worth saying out loud: the search itself just
        // stops reading and reports the records it got.
        if (in.gcount() != 0)
            std::fprintf(stderr, "warning: %lld trailing bytes, file is not a "
                                 "whole number of 20-byte records\n",
                         static_cast<long long>(in.gcount()));
        std::fprintf(stderr, "%zu records\n", n);
        return 0;
    }

} // namespace

int main(int argc, char** argv) {
    std::vector<const char*> inputs;
    std::string out_path;
    bool unique = false, do_dump = false, as_addr = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-u") unique = true;
        else if (a == "-d") do_dump = true;
        else if (a == "-a") as_addr = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else if (a == "-o") {
            if (++i >= argc) { usage(argv[0]); return 1; }
            out_path = argv[i];
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        } else {
            inputs.push_back(argv[i]);
        }
    }

    if (inputs.empty()) { usage(argv[0]); return 1; }
    if (do_dump) return dump(inputs[0], as_addr);

    if (out_path.empty()) {
        out_path = inputs[0];
        const std::size_t dot = out_path.find_last_of('.');
        const std::size_t slash = out_path.find_last_of("/\\");
        if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
            out_path.erase(dot);
        out_path += ".bin";
    }

    // Refusing to read and write the same file: with several inputs the
    // output would otherwise be overwritten while still being read.
    for (const char* in : inputs) {
        if (out_path == in) {
            std::fprintf(stderr, "refusing to write output over input %s\n", in);
            return 1;
        }
    }

    std::vector<unsigned char> data;
    std::unordered_set<std::string> seen;
    std::size_t written = 0, dups = 0, bad = 0;

    for (const char* path : inputs) {
        std::ifstream in(path);
        if (!in) {
            std::fprintf(stderr, "%s: cannot open\n", path);
            return 1;
        }

        std::string line;
        std::size_t lineno = 0;
        while (std::getline(in, line)) {
            ++lineno;
            clean(line, lineno == 1);
            if (line.empty()) continue;

            unsigned char rec[H160_BYTES];
            if (const char* err = parse_line(line, rec)) {
                if (++bad <= 10)
                    std::fprintf(stderr, "%s:%zu: %s: %s\n", path, lineno, err, line.c_str());
                continue;
            }

            if (unique) {
                std::string key(reinterpret_cast<const char*>(rec), H160_BYTES);
                if (!seen.insert(std::move(key)).second) { ++dups; continue; }
            }

            data.insert(data.end(), rec, rec + H160_BYTES);
            ++written;
        }
    }

    if (bad > 10) std::fprintf(stderr, "... and %zu more bad lines\n", bad - 10);

    if (written == 0) {
        std::fprintf(stderr, "no valid records, %s not written\n", out_path.c_str());
        return 1;
    }

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "%s: cannot create\n", out_path.c_str());
        return 1;
    }
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
    if (!out) {
        std::fprintf(stderr, "%s: write failed\n", out_path.c_str());
        return 1;
    }

    std::printf("%s: %zu records, %zu bytes", out_path.c_str(), written, data.size());
    if (dups) std::printf(", %zu duplicates dropped", dups);
    if (bad) std::printf(", %zu bad lines skipped", bad);
    std::printf("\n");
    return 0;
}
