#include "ec_delta.hpp"

#include <cstring>

namespace ecloop {
    namespace {

        // [j][0] = (+16^j)*G, [j][d] = (-d*16^j)*G for d = 1..15; the
        // 'F' -> '0' bump of the plain walk is [j][15], the other d are the
        // in-range walk zeroing a digit that cannot be incremented.
        // Immutable once delta_init has run, so sharing it across threads is
        // fine; only the walk position is per-thread. 64 x 16 x 96 bytes.
        pe _delta[DELTA_NPOS][16];
        int _delta_nd = DELTA_NPOS;
        bool _delta_track = false;

    } // namespace

    int delta_ndigits() noexcept { return _delta_nd; }

    const pe* delta_point(int lev, bool was_F) noexcept {
        return &_delta[_delta_nd - 1 - lev][was_F ? 15 : 0];
    }

    const pe* delta_point_kind(int lev, int kind) noexcept {
        return &_delta[_delta_nd - 1 - lev][kind & 15];
    }

    void delta_init(int ndigits) noexcept {
        if (ndigits < 1) ndigits = 1;
        if (ndigits > DELTA_NPOS) ndigits = DELTA_NPOS;
        _delta_nd = ndigits;
        _delta_track = (ndigits <= 15);

        for (int j = 0; j < ndigits; ++j) {
            fe s;
            pe P;

            // s = 16^j = 2^(4j)
            std::memset(s, 0, sizeof(fe));
            s[(4 * j) / 64] = 1ULL << ((4 * j) % 64);
            ec_gtable_mul(&P, s);
            ec_jacobi_rdc(&P, &P);
            pe_clone(&_delta[j][0], &P);

            // s = d*16^j for d = 1..15, then negate the point -> (-d*16^j)*G
            for (int d = 1; d < 16; ++d) {
                std::memset(s, 0, sizeof(fe));
                for (int k = 0; k < 4; ++k) {
                    if (!((d >> k) & 1)) continue;
                    const int b = 4 * j + k;
                    if (b < 256) s[b / 64] |= 1ULL << (b % 64);
                }
                ec_gtable_mul(&P, s);
                ec_jacobi_rdc(&P, &P);
                pe_clone(&_delta[j][d], &P);
                fe_modp_neg(_delta[j][d].y, P.y);
            }
        }
    }

    void DeltaWalker::sync(const char* buff) noexcept {
        k = 0;
        track = _delta_track;
        if (!track) return;
        for (int i = 0; i < _delta_nd; ++i) {
            const char c = buff[i];
            const u64 v = (c >= '0' && c <= '9') ? static_cast<u64>(c - '0')
                        : (c >= 'A' && c <= 'F') ? static_cast<u64>(c - 'A' + 10)
                        : (c >= 'a' && c <= 'f') ? static_cast<u64>(c - 'a' + 10)
                                                 : 0;
            k = k * 16 + v;
        }
    }

    void DeltaWalker::step(pe* P, int lev, bool was_F) noexcept {
        step_kind(P, lev, was_F ? 15 : 0);
    }

    void DeltaWalker::step_kind(pe* P, int lev, int kind) noexcept {
        const int j = _delta_nd - 1 - lev;
        kind &= 15;
        const pe* d = &_delta[j][kind];

        if (track) {
            const u64 w = 1ULL << (4 * j); // 16^j
            if (kind == 0) {
                if (k == w) { // P == D -> needs a doubling
                    ec_jacobi_dbl(P, P);
                    k += w;
                    return;
                }
                k += w;
            } else {
                const u64 dw = static_cast<u64>(kind) * w;
                if (k == dw) { // P == -D -> point at infinity
                    std::memset(P, 0, sizeof(pe));
                    k -= dw;
                    return;
                }
                k -= dw;
            }
        }

        if (fe_cmp(P->z, FE_ZERO) == 0) { // P is infinity: result is D
            pe_clone(P, d);
            return;
        }

        // Mixed addition, not the general one: the delta points are affine
        // (z == 1), and secp256k1_jmixadd is markedly cheaper than jadd.
        ec_jacobi_add_affine(P, P, d);
    }

} // namespace ecloop
