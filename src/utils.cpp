// Copyright (c) vladkens
// https://github.com/vladkens/ecloop
// Licensed under the MIT License.

#include "utils.hpp"

#include <atomic>

#include "fastPRNG.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cctype>

#include <unistd.h>

#ifdef _WIN32
#  include <windows.h>
#endif

namespace ecloop {

    // ─────────────────────────────────────────────────────────────────────────────
    // MARK: helpers
    // ─────────────────────────────────────────────────────────────────────────────

    [[maybe_unused]]
    static bool strendswith(const char* str, const char* suffix) {
        const std::size_t str_len = std::strlen(str);
        const std::size_t suffix_len = std::strlen(suffix);
        return (str_len >= suffix_len) &&
            (std::strcmp(str + str_len - suffix_len, suffix) == 0);
    }

    static char* strtrim(char* str) {
        if (!str) return nullptr;

        char* since = str;
        while (std::isspace(static_cast<unsigned char>(*since))) ++since;

        char* until = str + std::strlen(str) - 1;
        while (until > since && std::isspace(static_cast<unsigned char>(*until))) --until;

        *(until + 1) = '\0';
        if (since != str)
            std::memmove(str, since, static_cast<std::size_t>(until - since + 2));

        return str;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // MARK: random helpers
    // ─────────────────────────────────────────────────────────────────────────────

    static FILE* _urandom = nullptr;

    static void _close_urandom() {
        if (_urandom) {
            std::fclose(_urandom);
            _urandom = nullptr;
        }
    }

    // xoshiro256++ (fastPRNG.h) instead of std::rand().
    //
    // std::rand() has three problems here. Its state is shared and its
    // implementations are typically not thread-safe, so several search
    // threads drawing at once is undefined behaviour. RAND_MAX is only
    // guaranteed to be 32767, and where that holds the two shifted calls
    // above yield 30 usable bits out of 64, leaving most of a key range
    // unreachable. And the low bits of many libc generators are weak.
    //
    // The state is thread_local, so each thread owns an independent stream
    // and no locking is needed on the hot path.
    //
    // Seeding: with -seed the user wants a reproducible run, so the stream
    // is derived from that value; without it, from defaultSeed64(), which
    // mixes random_device, rdtsc, the clock and a stack address. The thread
    // id is folded in either way -- otherwise every thread would walk the
    // very same sequence.
    static u64 _prng_user_seed = 0;
    static bool _prng_seeded = false;

    static fastPRNG::fastXS64& _prng() {
        static std::atomic<u64> counter{0};
        thread_local fastPRNG::fastXS64 gen(
            (_prng_seeded ? fastPRNG::splitMix64(_prng_user_seed)
                          : fastPRNG::defaultSeed64()) +
            0x9E3779B97F4A7C15ULL * counter.fetch_add(1, std::memory_order_relaxed));
        return gen;
    }

    static u64 _prand64() { return _prng().xoshiro256pp(); }

    static u64 _urand64() {
        if (!_urandom) {
            _urandom = std::fopen("/dev/urandom", "rb");
            if (!_urandom) {
                std::fprintf(stderr, "failed to open /dev/urandom\n");
                std::exit(1);
            }
            std::atexit(_close_urandom);
        }

        u64 r;
        if (std::fread(&r, sizeof(r), 1, _urandom) != 1) {
            std::fprintf(stderr, "failed to read from /dev/urandom\n");
            std::exit(1);
        }
        return r;
    }

    // ─── public API ──────────────────────────────────────────────────────────────

    void prng_seed(u64 seed) {
        _prng_user_seed = seed;
        _prng_seeded = true;
    }

    u64 rand64(bool urandom) {
        return urandom ? _urand64() : _prand64();
    }

    u32 encode_seed(const char* seed) {
        u32 hash = 0;
        while (*seed) {
            const char c = *seed++;
            hash = (hash << 5) - hash + static_cast<unsigned char>(c);
            hash &= 0xFFFFFFFFu;
        }
        return hash;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // MARK: fe_random
    // ─────────────────────────────────────────────────────────────────────────────

    void fe_prand(fe r) {
        for (int i = 0; i < 4; ++i) r[i] = _prand64();
        r[3] &= 0xfffffffefffffc2fULL;
    }

    void fe_urand(fe r) {
        for (int i = 0; i < 4; ++i) r[i] = _urand64();
        r[3] &= 0xfffffffefffffc2fULL;
    }

    void fe_rand_range(fe r, const fe a, const fe b, bool urandom) {
        fe range, x;
        fe_modn_sub(range, b, a);
        fe_add64(range, 1);

        const std::size_t bits = fe_bitlen(range);
        assert(bits > 0 && bits <= 256);

        do {
            urandom ? fe_urand(x) : fe_prand(x);

            const int top = static_cast<int>((bits - 1) / 64);
            for (int i = top + 1; i < 4; ++i) x[i] = 0;

            const int rem = static_cast<int>(bits % 64);
            if (rem) x[top] &= (1ULL << rem) - 1;

        } while (fe_cmp(x, range) >= 0);

        fe_modn_add(x, x, a);
        assert(fe_cmp(x, a) >= 0);
        assert(fe_cmp(x, b) <= 0);
        fe_clone(r, x);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // MARK: CPU count
    // ─────────────────────────────────────────────────────────────────────────────

    int get_cpu_count() {
#ifdef _WIN32
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return static_cast<int>(si.dwNumberOfProcessors);
#else
        const int n = static_cast<int>(::sysconf(_SC_NPROCESSORS_ONLN));
        return n > 1 ? n : 1;
#endif
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // MARK: tsnow — milliseconds since epoch
    // ─────────────────────────────────────────────────────────────────────────────

    size_t tsnow() {
#ifdef _WIN32
        // GetTickCount64 returns milliseconds since system start; for a
        // monotonic ms-stamp that is sufficient for benchmarking purposes.
        return static_cast<size_t>(GetTickCount64());
#else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<size_t>(ts.tv_sec) * 1000u +
            static_cast<size_t>(ts.tv_nsec) / 1000000u;
#endif
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // MARK: terminal helpers
    // ─────────────────────────────────────────────────────────────────────────────

    void term_clear_line() {
        std::fputs("\r\033[K", stdout);
        std::fflush(stdout);
    }

    // tty_init — installs a raw-mode keystroke callback on a background thread.
    // On POSIX we put stdin into raw/non-canonical mode and spin a reader thread.
    // On Windows (including Cygwin with a real console) we use ReadConsoleInput.

#ifndef _WIN32
#  include <termios.h>
#  include <pthread.h>

    struct _tty_ctx {
        void (*cb)(void*, char);
        void* ctx;
    };

    static void* _tty_thread(void* arg) {
        auto* t = static_cast<_tty_ctx*>(arg);

        // Switch stdin to raw mode so we get keystrokes immediately.
        struct termios oldt, newt;
        if (tcgetattr(STDIN_FILENO, &oldt) != 0) {
            delete t;
            return nullptr;
        }
        newt = oldt;
        newt.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);

        int ch;
        while ((ch = ::getchar()) != EOF) {
            t->cb(t->ctx, static_cast<char>(ch));
        }

        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        delete t;
        return nullptr;
    }

    void tty_init(void (*cb)(void*, char), void* ctx) {
        if (!cb) return;
        auto* t = new _tty_ctx{ cb, ctx };
        pthread_t tid;
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
        pthread_create(&tid, &attr, _tty_thread, t);
        pthread_attr_destroy(&attr);
    }

#else // _WIN32

    void tty_init(void (*cb)(void*, char), void* ctx) {
        // Minimal stub for Windows / Cygwin without a POSIX tty.
        // A full implementation would use ReadConsoleInput on a separate thread.
        (void)cb; (void)ctx;
    }

#endif // _WIN32

    // ─────────────────────────────────────────────────────────────────────────────
    // MARK: public args API
    // ─────────────────────────────────────────────────────────────────────────────

    bool args_bool(const args_t* args, const char* name) {
        for (int i = 1; i < args->argc; ++i) {
            if (std::strcmp(args->argv[i], name) == 0) return true;
        }
        return false;
    }

    u64 args_uint(const args_t* args, const char* name, u64 def) {
        for (int i = 1; i < args->argc - 1; ++i) {
            if (std::strcmp(args->argv[i], name) == 0)
                return std::strtoull(args->argv[i + 1], nullptr, 10);
        }
        return def;
    }

    // Heap-allocated copy of the value string; caller is responsible for free().
    char* arg_str(const args_t* args, const char* name) {
        for (int i = 1; i < args->argc - 1; ++i) {
            if (std::strcmp(args->argv[i], name) == 0) {
                const char* val = args->argv[i + 1];
                const std::size_t len = std::strlen(val);
                char* copy = static_cast<char*>(std::malloc(len + 1));
                std::memcpy(copy, val, len + 1);
                return copy;
            }
        }
        return nullptr;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // MARK: bloom filter — internal types & helpers
    // ─────────────────────────────────────────────────────────────────────────────

    // blf_t is declared in utils.hpp — no redefinition here.

#define BLF_MAGIC   0x45434246u  // FourCC: ECBF
#define BLF_VERSION 1u

    static inline void blf_setbit(blf_t* blf, u64 idx) {
        blf->bits[idx % (static_cast<u64>(blf->size) * 64u) / 64u] |=
            (u64)1 << (idx % 64u);
    }

    static inline bool blf_getbit(const blf_t* blf, u64 idx) {
        return (blf->bits[idx % (static_cast<u64>(blf->size) * 64u) / 64u] &
            ((u64)1 << (idx % 64u))) != 0;
    }

    // ── public (declared extern in utils.hpp) ────────────────────────────────────

    void blf_add(blf_t* blf, const h160_t hash) {
        const u64 a1 = (static_cast<u64>(hash[0]) << 32) | hash[1];
        const u64 a2 = (static_cast<u64>(hash[2]) << 32) | hash[3];
        const u64 a3 = (static_cast<u64>(hash[4]) << 32) | hash[0];
        const u64 a4 = (static_cast<u64>(hash[1]) << 32) | hash[2];
        const u64 a5 = (static_cast<u64>(hash[3]) << 32) | hash[4];

        const u8 shifts[4] = { 24, 28, 36, 40 };
        for (std::size_t i = 0; i < 4; ++i) {
            const u8 S = shifts[i];
            blf_setbit(blf, (a1 << S) | (a2 >> S));
            blf_setbit(blf, (a2 << S) | (a3 >> S));
            blf_setbit(blf, (a3 << S) | (a4 >> S));
            blf_setbit(blf, (a4 << S) | (a5 >> S));
            blf_setbit(blf, (a5 << S) | (a1 >> S));
        }
    }

    bool blf_has(const blf_t* blf, const h160_t hash) {
        const u64 a1 = (static_cast<u64>(hash[0]) << 32) | hash[1];
        const u64 a2 = (static_cast<u64>(hash[2]) << 32) | hash[3];
        const u64 a3 = (static_cast<u64>(hash[4]) << 32) | hash[0];
        const u64 a4 = (static_cast<u64>(hash[1]) << 32) | hash[2];
        const u64 a5 = (static_cast<u64>(hash[3]) << 32) | hash[4];

        const u8 shifts[4] = { 24, 28, 36, 40 };
        for (std::size_t i = 0; i < 4; ++i) {
            const u8 S = shifts[i];
            if (!blf_getbit(blf, (a1 << S) | (a2 >> S))) return false;
            if (!blf_getbit(blf, (a2 << S) | (a3 >> S))) return false;
            if (!blf_getbit(blf, (a3 << S) | (a4 >> S))) return false;
            if (!blf_getbit(blf, (a4 << S) | (a5 >> S))) return false;
            if (!blf_getbit(blf, (a5 << S) | (a1 >> S))) return false;
        }
        return true;
    }

    static bool blf_save(const char* filepath, const blf_t* blf) {
        FILE* file = std::fopen(filepath, "wb");
        if (!file) {
            std::fprintf(stderr, "failed to open output file\n");
            std::exit(1);
        }

        const u32 magic = BLF_MAGIC;
        const u32 version = BLF_VERSION;

        if (std::fwrite(&magic, sizeof(magic), 1, file) != 1 ||
            std::fwrite(&version, sizeof(version), 1, file) != 1 ||
            std::fwrite(&blf->size, sizeof(blf->size), 1, file) != 1 ||
            std::fwrite(blf->bits, sizeof(u64), blf->size, file) != blf->size) {
            std::fprintf(stderr, "failed to write bloom filter\n");
            std::fclose(file);
            return false;
        }

        std::fclose(file);
        return true;
    }

    bool blf_load(const char* filepath, blf_t* blf) {
        FILE* file = std::fopen(filepath, "rb");
        if (!file) {
            std::fprintf(stderr, "failed to open input file\n");
            return false;
        }

        u32         magic = 0, version = 0;
        std::size_t size = 0;

        const bool ok =
            std::fread(&magic, sizeof(magic), 1, file) == 1 &&
            std::fread(&version, sizeof(version), 1, file) == 1 &&
            std::fread(&size, sizeof(size), 1, file) == 1;

        if (!ok) {
            std::fprintf(stderr, "failed to read bloom filter header\n");
            std::fclose(file);
            return false;
        }

        if (magic != BLF_MAGIC || version != BLF_VERSION) {
            std::fprintf(stderr,
                "invalid bloom filter version; "
                "create a new filter with blf-gen command\n");
            std::fclose(file);
            return false;
        }

        u64* bits = static_cast<u64*>(std::calloc(size, sizeof(u64)));
        if (std::fread(bits, sizeof(u64), size, file) != size) {
            std::fprintf(stderr, "failed to read bloom filter bits\n");
            std::free(bits);
            std::fclose(file);
            return false;
        }

        std::fclose(file);
        blf->size = size;
        blf->bits = bits;
        return true;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // MARK: private args helpers (used by blf_gen / blf_check)
    // ─────────────────────────────────────────────────────────────────────────────

    [[maybe_unused]]
    static bool _args_bool(const args_t* args, const char* name) {
        for (int i = 1; i < args->argc; ++i) {
            if (std::strcmp(args->argv[i], name) == 0) return true;
        }
        return false;
    }

    static u64 _args_uint(const args_t* args, const char* name, int def) {
        for (int i = 1; i < args->argc - 1; ++i) {
            if (std::strcmp(args->argv[i], name) == 0)
                return std::strtoull(args->argv[i + 1], nullptr, 10);
        }
        return static_cast<u64>(def);
    }

    static const char* _arg_str(const args_t* args, const char* name) {
        for (int i = 1; i < args->argc; ++i) {
            if (std::strcmp(args->argv[i], name) == 0 && i + 1 < args->argc)
                return args->argv[i + 1];
        }
        return nullptr;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // MARK: queue
    // ─────────────────────────────────────────────────────────────────────────────

    void queue_init(queue_t* q, size_t capacity) {
        q->items = static_cast<void**>(std::calloc(capacity, sizeof(void*)));
        q->capacity = capacity;
        q->head = 0;
        q->tail = 0;
        q->count = 0;
        q->done = false;
        pthread_mutex_init(&q->lock, nullptr);
        pthread_cond_init(&q->not_empty, nullptr);
        pthread_cond_init(&q->not_full, nullptr);
    }

    void queue_put(queue_t* q, void* item) {
        pthread_mutex_lock(&q->lock);
        while (q->count == q->capacity)
            pthread_cond_wait(&q->not_full, &q->lock);

        q->items[q->tail] = item;
        q->tail = (q->tail + 1) % q->capacity;
        ++q->count;

        pthread_cond_signal(&q->not_empty);
        pthread_mutex_unlock(&q->lock);
    }

    void* queue_get(queue_t* q) {
        pthread_mutex_lock(&q->lock);
        while (q->count == 0 && !q->done)
            pthread_cond_wait(&q->not_empty, &q->lock);

        if (q->count == 0) {          // done && empty
            pthread_mutex_unlock(&q->lock);
            return nullptr;
        }

        void* item = q->items[q->head];
        q->head = (q->head + 1) % q->capacity;
        --q->count;

        pthread_cond_signal(&q->not_full);
        pthread_mutex_unlock(&q->lock);
        return item;
    }

    void queue_done(queue_t* q) {
        pthread_mutex_lock(&q->lock);
        q->done = true;
        pthread_cond_broadcast(&q->not_empty); // wake all blocked consumers
        pthread_mutex_unlock(&q->lock);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // MARK: blf-gen command
    // ─────────────────────────────────────────────────────────────────────────────

    [[noreturn]] static void _blf_gen_usage(const args_t* args) {
        std::printf("Usage: %s blf-gen -n <count> -o <file>\n", args->argv[0]);
        std::printf("Generate a bloom filter from a list of hex-encoded hash160 values passed to stdin.\n");
        std::printf("\nOptions:\n");
        std::printf("  -n <count>      - Number of hashes to add.\n");
        std::printf("  -o <file>       - File to write bloom filter (must have a .blf extension).\n");
        std::exit(1);
    }

    void blf_gen(args_t* args) {
        const u64 n = _args_uint(args, "-n", 0);
        if (n == 0) {
            std::fprintf(stderr, "[!] missing filter size (-n <number>)\n");
            _blf_gen_usage(args);
        }

        const char* filepath = _arg_str(args, "-o");
        if (!filepath) {
            std::fprintf(stderr, "[!] missing output file (-o <file>)\n");
            _blf_gen_usage(args);
        }

        const u64    r = static_cast<u64>(1e9);
        const double p = 1.0 / static_cast<double>(r);
        const u64    m = static_cast<u64>(static_cast<double>(n) * std::log(p) /
            std::log(1.0 / std::pow(2.0, std::log(2.0))));
        const double mb = static_cast<double>(m) / 8.0 / 1024.0 / 1024.0;
        const std::size_t size = static_cast<std::size_t>((m + 63) / 64);

        blf_t blf{};
        if (::access(filepath, F_OK) == 0) {
            std::printf("file %s already exists; loading...\n", filepath);

            if (!blf_load(filepath, &blf)) {
                std::fprintf(stderr,
                    "[!] failed to load bloom filter: "
                    "delete it or choose a different file\n");
                std::exit(1);
            }

            if (blf.size != size) {
                std::fprintf(stderr,
                    "[!] bloom filter size mismatch (%zu != %zu): "
                    "delete it or choose a different file\n",
                    blf.size, size);
                std::exit(1);
            }

            std::printf("updating bloom filter...\n");
        }
        else {
            std::printf("creating bloom filter...\n");
            blf.size = size;
            blf.bits = static_cast<u64*>(std::calloc(size, sizeof(u64)));
        }

        std::printf(
            "bloom filter params: n = %llu | p = 1:%llu | m = %llu (%.1f MB)\n",
            static_cast<unsigned long long>(n),
            static_cast<unsigned long long>(r),
            static_cast<unsigned long long>(m),
            mb);

        u64   count = 0;
        hex40 line;
        while (std::fgets(line, sizeof(line), stdin)) {
            if (std::strlen(line) != sizeof(line) - 1) continue;

            h160_t hash;
            for (std::size_t j = 0; j < sizeof(line) - 1; j += 8)
                std::sscanf(line + j, "%8x", &hash[j / 8]);

            if (blf_has(&blf, hash)) continue;

            blf_add(&blf, hash);
            ++count;
        }

        std::printf("added %llu new items; saving to %s\n",
            static_cast<unsigned long long>(count), filepath);

        if (!blf_save(filepath, &blf)) {
            std::fprintf(stderr, "[!] failed to save bloom filter\n");
            std::exit(1);
        }

        std::free(blf.bits);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // MARK: blf-check command
    // ─────────────────────────────────────────────────────────────────────────────

    [[noreturn]] static void _blf_check_usage(const args_t* args) {
        std::printf("Usage: %s blf-check -f <file> <hash> [hash...]\n", args->argv[0]);
        std::printf("Check if one or more hex-encoded hash160 values are in the bloom filter.\n");
        std::printf("\nOptions:\n");
        std::printf("  -f <file>       Path to the bloom filter file (required).\n");
        std::printf("\nArguments:\n");
        std::printf("  <hash>          One or more hex-encoded hash160 values to check.\n");
        std::printf("                  If no arguments are provided, stdin will be used as source.\n");
        std::exit(1);
    }

    static bool _blf_check_hex(const blf_t* blf, const char* hex) {
        h160_t h = { 0, 0, 0, 0, 0 };
        for (std::size_t i = 0; i < 40; i += 8)
            std::sscanf(hex + i, "%8x", &h[i / 8]);
        return blf_has(blf, h);
    }

    void blf_check(args_t* args) {
        const char* filepath = _arg_str(args, "-f");
        if (!filepath) {
            std::fprintf(stderr, "[!] missing input file (-f <file>)\n");
            _blf_check_usage(args);
        }

        blf_t blf{};
        if (!blf_load(filepath, &blf)) {
            std::fprintf(stderr, "[!] failed to load bloom filter\n");
            std::exit(1);
        }

        bool has_opts = false;
        for (int i = 1; i < args->argc; ++i) {
            if (std::strlen(args->argv[i]) != 40) continue;

            has_opts = true;
            const bool found = _blf_check_hex(&blf, args->argv[i]);
            std::printf("%s %s\n", args->argv[i], found ? "FOUND" : "NOT FOUND");
        }

        if (has_opts) {
            std::free(blf.bits);
            return;
        }

        char line[128];
        while (std::fgets(line, sizeof(line), stdin)) {
            strtrim(line);
            if (std::strlen(line) != 40) continue;

            const bool found = _blf_check_hex(&blf, line);
            std::printf("%s %s\n", line, found ? "FOUND" : "NOT FOUND");
        }

        std::free(blf.bits);
    }

} // namespace ecloop
