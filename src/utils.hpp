#pragma once

#include <cstddef>
#include <cstdio>
#include <pthread.h>

#include "types.hpp"
#include "ecc.hpp"

namespace ecloop {

    // ─── args ────────────────────────────────────────────────────────────────────

    struct args_t {
        int argc;
        const char** argv;
    };

    bool   args_bool(const args_t* args, const char* name);
    u64    args_uint(const args_t* args, const char* name, u64 def);
    char* arg_str(const args_t* args, const char* name); // heap-allocated, caller frees

    // ─── random ──────────────────────────────────────────────────────────────────

    u64  rand64(bool urandom);
    void prng_seed(u64 seed);   // makes the xoshiro stream reproducible
    u32  encode_seed(const char* seed);

    void fe_prand(fe r);
    void fe_urand(fe r);
    void fe_rand_range(fe r, const fe a, const fe b, bool urandom);

    // ─── system ──────────────────────────────────────────────────────────────────

    int    get_cpu_count();
    size_t tsnow();          // milliseconds since epoch

    // ─── terminal ────────────────────────────────────────────────────────────────

#define COLOR_YELLOW "\033[33m"
#define COLOR_RESET  "\033[0m"

    void term_clear_line();
    void tty_init(void (*cb)(void*, char), void* ctx);

    // ─── bloom filter ────────────────────────────────────────────────────────────

    using h160_t = u32[5];
    using hex40 = char[41];

    struct blf_t {
        size_t size = 0;
        u64* bits = nullptr;
    };

    void blf_add(blf_t* blf, const h160_t hash);
    bool blf_has(const blf_t* blf, const h160_t hash);
    bool blf_load(const char* filepath, blf_t* blf);

    void blf_gen(args_t* args);
    void blf_check(args_t* args);

    // ─── queue ───────────────────────────────────────────────────────────────────

    struct queue_t {
        void** items = nullptr;
        size_t          capacity = 0;
        size_t          head = 0;
        size_t          tail = 0;
        size_t          count = 0;
        bool            done = false;
        pthread_mutex_t lock;
        pthread_cond_t  not_empty;
        pthread_cond_t  not_full;
    };

    void  queue_init(queue_t* q, size_t capacity);
    void  queue_put(queue_t* q, void* item);
    void* queue_get(queue_t* q);          // returns nullptr when done & empty
    void  queue_done(queue_t* q);

    // ─── misc ────────────────────────────────────────────────────────────────────

    int compare_160(const void* a, const void* b);

#ifndef MIN
#  define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
#  define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

} // namespace ecloop
