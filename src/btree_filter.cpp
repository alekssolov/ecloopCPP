#include "btree_filter.hpp"

#include "main.hpp"
#include "shm.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace ecloop {

        u64* BTF_BITS = nullptr;
        u64  BTF_NBLOCKS = 0;

        namespace {
            ECLOOP_ALWAYS_INLINE void btf_set(const u32* h) noexcept {
                u64* blk = BTF_BITS + 8 * btf_block(h);
                u32 p[BTF_K];
                btf_positions(h, p);
                for (int i = 0; i < BTF_K; ++i) blk[p[i] >> 6] |= 1ULL << (p[i] & 63);
            }


            // 2 MB-aligned zeroed allocation through shm.hpp: shared with the
            // forked workers under Cygwin (one copy, not one per worker),
            // huge pages under Linux.
            ShmBlock btf_blk;
        } // namespace

        bool btf_build_sized(const u32* tree, std::size_t bytes) {
            shm_free(btf_blk);
            BTF_BITS = nullptr;
            BTF_NBLOCKS = 0;
            if (bytes < 64) bytes = 64;
            btf_blk = shm_alloc(bytes, std::size_t(1) << 21);
            if (!btf_blk.ptr) return false;
            BTF_BITS = static_cast<u64*>(btf_blk.ptr);
            BTF_NBLOCKS = bytes / 64;
            if (tree) {
                const u32 hsize = BTREE_HSIZE;
                for (u32 p = hsize; p + 6 <= tree[0]; p += 6) btf_set(&tree[p]);
            }
            return true;
        }

        namespace {
        } // namespace

        void btf_build(const u32* tree) {
            const u32 hsize = BTREE_HSIZE;
            if (!tree || tree[0] <= hsize) return;

            const char* off = std::getenv("ECLOOP_NO_FILTER");
            if (off && off[0] == '1') {
                std::printf("Pre - filter: disabled (ECLOOP_NO_FILTER=1)\n");
                return;
            }

            // Record count differs from what btree_load prints: btree_add
            // deduplicates, so repeated hash160 entries in the .bin are read
            // but not stored. btree_load reports how many were READ, this is
            // how many are IN the tree, and that is the number that matters.
            const u64 nrec = (tree[0] - hsize) / 6;

            // 12 bits per record, i.e. ~0.9% false positives with K = 4.
            // ECLOOP_FILTER_MB overrides in MB; ECLOOP_FILTER_BITS (log2 of
            // the bit count, the old switch) is still honoured.
            std::size_t bytes = static_cast<std::size_t>(nrec) * 12 / 8;
            if (const char* env = std::getenv("ECLOOP_FILTER_MB")) {
                const long mb = std::atol(env);
                if (mb > 0) bytes = static_cast<std::size_t>(mb) << 20;
            } else if (const char* env2 = std::getenv("ECLOOP_FILTER_BITS")) {
                int bb = std::atoi(env2);
                if (bb < 20) bb = 20;
                if (bb > 36) bb = 36;
                bytes = std::size_t(1) << (bb - 3);
            }
            constexpr std::size_t HUGE = std::size_t(1) << 21;
            bytes = (bytes + HUGE - 1) & ~(HUGE - 1);

            if (!btf_build_sized(tree, bytes)) {
                std::printf("Pre - filter: out of memory, running without it\n");
                return;
            }

            // Expected false-positive rate of a blocked filter: the mean over
            // blocks of (set fraction)^K -- exact for a uniformly hashed
            // probe, and it accounts for the blocks that ended up fuller
            // than average.
            double fp = 0;
            u64 set = 0;
            for (u64 b = 0; b < BTF_NBLOCKS; ++b) {
                unsigned c = 0;
                for (int w = 0; w < 8; ++w) c += static_cast<unsigned>(__builtin_popcountll(BTF_BITS[8 * b + w]));
                set += c;
                double f = c / 512.0, pk = 1;
                for (int i = 0; i < BTF_K; ++i) pk *= f;
                fp += pk;
            }
            fp /= static_cast<double>(BTF_NBLOCKS);

            const double bpr = static_cast<double>(bytes) * 8 / static_cast<double>(nrec);
            std::printf("Pre - filter: blocked Bloom, %.0f MB, %.1f bits/record, K=%d, records in tree: %llu,\n"
                        "occupied: %.1f%%, expected false positives: %.2f%%.\n",
                        static_cast<double>(bytes) / 1048576.0,
                        bpr > 9999 ? 9999.0 : bpr, BTF_K,
                        static_cast<unsigned long long>(nrec),
                        100.0 * static_cast<double>(set) / static_cast<double>(BTF_NBLOCKS * 512),
                        100.0 * fp);
            std::fflush(stdout);
        }

} // namespace ecloop
