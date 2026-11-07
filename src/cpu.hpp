// Runtime CPU feature queries for the fat ("universal") build, where every
// SIMD tier is compiled in and chosen when the program starts. Cheap
// enough for start-up decisions; the field-op dispatch reads a plain flag
// (EC_ASM_OK, s2n_dispatch.hpp) set once by ec_dispatch_init().
#pragma once

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

namespace ecloop {

    inline bool cpu_has_avx2() noexcept {
#if defined(__x86_64__)
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") != 0;
#else
        return false;
#endif
    }
    // AVX-512 F/BW/DQ/VL + IFMA, OS state included (__builtin_cpu_supports
    // checks XGETBV for the avx512 names).
    inline bool cpu_has_avx512_ifma() noexcept {
#if defined(__x86_64__)
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
               __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl") &&
               __builtin_cpu_supports("avx512ifma");
#else
        return false;
#endif
    }
    // mulx and adcx/adox: what the s2n-bignum assembly needs (Broadwell+, Zen).
    inline bool cpu_has_bmi2_adx() noexcept {
#if defined(__x86_64__)
        unsigned a, b, c, d;
        if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
        return ((b >> 8) & 1u) && ((b >> 19) & 1u);
#else
        return false;
#endif
    }

} // namespace ecloop
