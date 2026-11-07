#include "ec_glv.hpp"

namespace ecloop {

    const fe GLV_BETA = {0xC1396C28719501EEULL, 0x9CF0497512F58995ULL,
                         0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL};
    const fe GLV_BETA2 = {0x3EC693D68E6AFA40ULL, 0x630FB68AED0A766AULL,
                          0x919BB86153CBCB16ULL, 0x851695D49A83F8EFULL};
    const fe GLV_LAMBDA = {0xDF02967C1B23BD72ULL, 0x122E22EA20816678ULL,
                           0xA5261C028812645AULL, 0x5363AD4CC05C30E0ULL};
    const fe GLV_LAMBDA2 = {0xE0CFC810B51283CEULL, 0xA880B9FC8EC739C2ULL,
                            0x5AD9E3FD77ED9BA4ULL, 0xAC9C52B33FA3CF1FULL};

    void glv_expand(pe* pts, std::size_t n) noexcept {
        for (std::size_t i = 0; i < n; ++i) {
            fe_modp_mul(pts[n + i].x, pts[i].x, GLV_BETA);
            fe_clone(pts[n + i].y, pts[i].y);
            fe_set64(pts[n + i].z, 1);

            fe_modp_mul(pts[2 * n + i].x, pts[i].x, GLV_BETA2);
            fe_clone(pts[2 * n + i].y, pts[i].y);
            fe_set64(pts[2 * n + i].z, 1);
        }
    }

    void glv_key(fe out, const fe base_k, std::size_t idx, std::size_t n) noexcept {
        if (idx < n) {
            fe_clone(out, base_k);
            return;
        }
        if (idx < 2 * n) {
            fe_modn_mul(out, base_k, GLV_LAMBDA);
            return;
        }
        fe_modn_mul(out, base_k, GLV_LAMBDA2);
    }

} // namespace ecloop
