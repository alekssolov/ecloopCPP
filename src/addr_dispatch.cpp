#include "addr_dispatch.hpp"

namespace ecloop {

    void addr33_batchN(h160_t* hashes, const pe* points, std::size_t count) noexcept {
        if (__builtin_expect(count == A512_BATCH && addr_avx512_available(), 1)) {
            addr33_batchA(hashes, points);
            return;
        }
        for (std::size_t i = 0; i < count; i += RMD_LEN) {
            const std::size_t n = count - i < RMD_LEN ? count - i : RMD_LEN;
            addr33_batch(hashes + i, points + i, n);
        }
    }

    void addr65_batchN(h160_t* hashes, const pe* points, std::size_t count) noexcept {
        if (__builtin_expect(count == A512_BATCH && addr_avx512_available(), 1)) {
            addr65_batchA(hashes, points);
            return;
        }
        for (std::size_t i = 0; i < count; i += RMD_LEN) {
            const std::size_t n = count - i < RMD_LEN ? count - i : RMD_LEN;
            addr65_batch(hashes + i, points + i, n);
        }
    }

} // namespace ecloop
