// Allocation for the large tables that are built before fork() and only
// read afterwards: the btree heap, the prefilter, the generator table.
//
// Cygwin's fork() has no copy-on-write: it re-creates every private
// anonymous mapping of the parent in the child and COPIES its contents.
// With 8 workers that is eight full copies of ~600 MB, all committed
// against the Windows commit limit (RAM + page file), and once that runs
// short a fork dies with "fixup_mmap_after_fork ... Win32 error 1455"
// (ERROR_COMMITMENT_LIMIT) and "fork: Resource temporarily unavailable".
// A MAP_SHARED anonymous mapping is one section object that every child
// simply maps again: nothing is copied, the tables exist once.
//
// On Linux fork() shares private pages copy-on-write already, so the
// mapping stays private there and only gets the huge-page hint (a table of
// hundreds of MB with random access otherwise misses the TLB on every
// probe). Both cases return zeroed memory.

#pragma once

#include <cstddef>
#include <cstring>

#include <sys/mman.h>

namespace ecloop {

    struct ShmBlock {
        void* base = nullptr;       // what mmap returned
        std::size_t len = 0;        // its length
        void* ptr = nullptr;        // aligned start for the caller
    };

    // `align` must be a power of two; the block is `bytes` long from ptr.
    inline ShmBlock shm_alloc(std::size_t bytes, std::size_t align = 4096) noexcept {
        ShmBlock b;
        if (bytes == 0) bytes = 1;
        const std::size_t len = bytes + align;
#if defined(__CYGWIN__)
        const int flags = MAP_SHARED | MAP_ANONYMOUS;
#else
        const int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#endif
        void* m = mmap(nullptr, len, PROT_READ | PROT_WRITE, flags, -1, 0);
        if (m == MAP_FAILED) return b;
        b.base = m;
        b.len = len;
        const std::size_t a = reinterpret_cast<std::size_t>(m);
        b.ptr = reinterpret_cast<void*>((a + align - 1) & ~(align - 1));
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        madvise(b.ptr, bytes, MADV_HUGEPAGE);
#endif
        return b;
    }

    inline void shm_free(ShmBlock& b) noexcept {
        if (b.base) munmap(b.base, b.len);
        b = ShmBlock{};
    }

} // namespace ecloop
