// Copyright (c) vladkens
// https://github.com/vladkens/ecloop
// Licensed under the MIT License.

#include "bench.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "types.hpp"
#include "addr.hpp"
#include "ecc.hpp"
#include "utils.hpp"

// tsnow() объявлена в utils.hpp / реализована в utils.cpp
// fe_prand() — тоже оттуда

namespace ecloop {

    // ---------------------------------------------------------------------------
    // helpers
    // ---------------------------------------------------------------------------

    //static u64 tsnow() noexcept; // forward-declaration; реализация — в utils.cpp

    static void print_res(const char* label, u64 stime, std::size_t iters) {
        double dt = static_cast<double>(tsnow() - stime);
        if (dt < 1.0) dt = 1.0;
        dt /= 1000.0; // мс → с
        std::printf("%28s: %.2fM it/s ~ %.2fs\n",
            label,
            static_cast<double>(iters) / dt / 1'000'000.0,
            dt);
    }

    // ---------------------------------------------------------------------------
    // run_bench
    // ---------------------------------------------------------------------------

    void run_bench() {
        ec_gtable_init();

        // note: asserts used to prevent compiler optimization
        std::size_t iters, i;
        u64         stime;
        pe          g;
        fe          f;

        // ── projective & jacobian coordinates ───────────────────────────────────
        iters = 1000ULL * 1000ULL * 6ULL;

        pe_clone(&g, &G2);
        stime = tsnow();
        for (i = 0; i < iters; ++i) _ec_jacobi_add1(&g, &g, &G1);
        print_res("_ec_jacobi_add1", stime, iters);
        assert(fe_cmp(g.x, G1.x) != 0);

        pe_clone(&g, &G2);
        stime = tsnow();
        for (i = 0; i < iters; ++i) _ec_jacobi_add2(&g, &g, &G1);
        print_res("_ec_jacobi_add2", stime, iters);
        assert(fe_cmp(g.x, G1.x) != 0);

        pe_clone(&g, &G2);
        stime = tsnow();
        for (i = 0; i < iters; ++i) _ec_jacobi_dbl1(&g, &g);
        print_res("_ec_jacobi_dbl1", stime, iters);
        assert(fe_cmp(g.x, G1.x) != 0);

        pe_clone(&g, &G2);
        stime = tsnow();
        for (i = 0; i < iters; ++i) _ec_jacobi_dbl2(&g, &g);
        print_res("_ec_jacobi_dbl2", stime, iters);
        assert(fe_cmp(g.x, G1.x) != 0);

        // ── ec multiplication ────────────────────────────────────────────────────
        std::srand(42);
        constexpr std::size_t numSize = 1024ULL * 16ULL;
        fe numbers[numSize];
        for (std::size_t j = 0; j < numSize; ++j) fe_prand(numbers[j]);
        pe_clone(&g, &G2);

        iters = 1000ULL * 10ULL;
        stime = tsnow();
        for (i = 0; i < iters; ++i) ec_jacobi_mul(&g, &G1, numbers[i % numSize]);
        print_res("ec_jacobi_mul", stime, iters);
        assert(fe_cmp(g.x, G1.x) != 0);

        iters = 1000ULL * 500ULL;
        stime = tsnow();
        for (i = 0; i < iters; ++i) ec_gtable_mul(&g, numbers[i % numSize]);
        print_res("ec_gtable_mul", stime, iters);
        assert(fe_cmp(g.x, G1.x) != 0);

        // ── affine coordinates ───────────────────────────────────────────────────
        iters = 1000ULL * 500ULL;

        pe_clone(&g, &G2);
        stime = tsnow();
        for (i = 0; i < iters; ++i) ec_affine_add(&g, &g, &G1);
        print_res("ec_affine_add", stime, iters);
        assert(fe_cmp(g.x, G1.x) != 0);

        pe_clone(&g, &G2);
        stime = tsnow();
        for (i = 0; i < iters; ++i) ec_affine_dbl(&g, &g);
        print_res("ec_affine_dbl", stime, iters);
        assert(fe_cmp(g.x, G1.x) != 0);

        // ── modular inversion ────────────────────────────────────────────────────
        iters = 1000ULL * 100ULL;

        stime = tsnow();
        for (i = 0; i < iters; ++i) _fe_modp_inv_binpow(f, g.x);
        print_res("_fe_modinv_binpow", stime, iters);
        assert(fe_cmp(f, G1.x) != 0);

        stime = tsnow();
        for (i = 0; i < iters; ++i) _fe_modp_inv_addchn(f, g.x);
        print_res("_fe_modinv_addchn", stime, iters);
        assert(fe_cmp(f, G1.x) != 0);

        // ── hash functions ───────────────────────────────────────────────────────
        iters = 1000ULL * 1000ULL * 10ULL;
        h160_t h160{};

        stime = tsnow();
        for (i = 0; i < iters; ++i) addr33(h160, &g);
        print_res("addr33", stime, iters);
        assert(h160[0] != 0);

        stime = tsnow();
        for (i = 0; i < iters; ++i) addr65(h160, &g);
        print_res("addr65", stime, iters);
        assert(h160[0] != 0);
    }

    // ---------------------------------------------------------------------------
    // run_bench_gtable
    // ---------------------------------------------------------------------------

    void run_bench_gtable() {
        std::srand(42);
        constexpr std::size_t numSize = 1024ULL * 16ULL;
        fe numbers[numSize];
        for (std::size_t j = 0; j < numSize; ++j) fe_prand(numbers[j]);

        constexpr std::size_t iters = 1000ULL * 500ULL;
        pe                    g{};

        for (int w = 8; w <= 22; w += 2) {
            _GTABLE_W = static_cast<u64>(w);

            u64    stime = tsnow();
            std::size_t mem_used = ec_gtable_init();
            double gent = static_cast<double>(tsnow() - stime) / 1000.0;

            stime = tsnow();
            for (std::size_t j = 0; j < iters; ++j)
                ec_gtable_mul(&g, numbers[j % numSize]);
            double mult = static_cast<double>(tsnow() - stime) / 1000.0;

            double mem = static_cast<double>(mem_used) / 1024.0 / 1024.0; // MB
            std::printf("w=%02d: %.1fK it/s | gen: %5.2fs | mul: %5.2fs | mem: %8.1fMB\n",
                w,
                static_cast<double>(iters) / mult / 1000.0,
                gent,
                mult,
                mem);
        }
    }

    // ---------------------------------------------------------------------------
    // mult_verify
    // ---------------------------------------------------------------------------

    void mult_verify() {
        ec_gtable_init();

        pe r1, r2;
        fe pk;

        for (int i = 0; i < 1000 * 16; ++i) {
            fe_set64(pk, static_cast<u64>(i + 2));

            ec_jacobi_mulrdc(&r1, &G1, pk);
            ec_verify(&r1);

            ec_gtable_mul(&r2, pk);
            ec_jacobi_rdc(&r2, &r2);
            ec_verify(&r2);

            if (std::memcmp(&r1, &r2, sizeof(pe)) != 0) {
                std::printf("invalid on %d\n", i);
                fe_print("pk", pk);
                fe_print("r1", r1.x);
                fe_print("r2", r2.x);
                std::exit(1);
            }
        }
    }

} // namespace ecloop
