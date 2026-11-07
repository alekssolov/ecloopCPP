// New file (not part of upstream s2n-bignum): batch-8 AVX512IFMA field
// multiply for secp256k1.
//
// ----------------------------------------------------------------------------
// Multiply 8 INDEPENDENT field element pairs modulo p_256k1 simultaneously.
//
//    extern void bignum_mul_p256k1_x8(uint64_t z[8][4], const uint64_t x[8][4],
//                                      const uint64_t y[8][4]);
//
// z[lane] := (x[lane] * y[lane]) mod p_256k1, for lane = 0..7.
//
// Standard x86-64 ABI: RDI = z, RSI = x, RDX = y
// Microsoft x64 ABI:   RCX = z, RDX = x, R8 = y
//
// ----------------------------------------------------------------------------
// DESIGN NOTE
// ----------------------------------------------------------------------------
// This is the "real" throughput primitive: ecloop's hot loop (see
// batch_add_demo.c) processes large arrays of INDEPENDENT field elements
// (GROUP_INV_SIZE of them at a time) through the same handful of
// operations (sub, mul, sqr, sub, sub, mul, sub for the chord/lambda
// step). bignum_mul_p256k1 (the single-item AVX512IFMA routine) still
// pays AoS<->radix52 conversion overhead per call that isn't amortized;
// batching 8 independent multiplies per call amortizes that conversion
// over 8x the work, AND -- unlike a single big multiply spread across
// lanes -- needs NO cross-lane data movement for the O(n^2) schoolbook
// step at all, since each lane carries a fully independent number (see
// the multiply core below: every vpmadd52 processes "digit i of lane L"
// for all 8 L simultaneously, with i fixed by the (fully unrolled)
// instruction, so there is nothing to shuffle).
//
// The O(n) final reduction (fold 2^256 = 4294968273 (mod p) into the low
// half, then a quotient-estimate-and-correct step) is run once per lane
// using the *exact* proven scalar algorithm from bignum_mul_p256k1.S,
// after transposing the canonical radix-52 SoA digits back into a
// per-lane radix-2^64 buffer. This is a deliberate scope decision: that
// reduction step is cheap relative to the now-8-wide schoolbook core, and
// a genuinely vectorized version of it would need to multiply values
// that don't fit IFMA's 52-bit operand constraint (the repacked 64-bit
// digits), which means splitting each into 52+12-bit pieces and carrying
// the alignment through the fold by hand -- a second, comparably complex
// arithmetic core. Keeping the O(n) tail on the already-proven scalar
// path, 8 times, was judged the better risk/reward trade; a fully
// vectorized reduction is a well-scoped follow-up, not attempted here.
//
// Validated with an extensive differential test (Python ground truth,
// 800+ random lane values across 100 batches) against an
// AVX512F-executable C model of this exact algorithm (real data-movement
// instructions, IFMA's two primitives software-emulated per Intel's
// documented semantics, since this development machine lacks
// AVX512IFMA), then transliterated below; see the accompanying notes.
// ----------------------------------------------------------------------------

#include "_internal_s2n_bignum_x86.h"

        .intel_syntax noprefix
        S2N_BN_SYM_VISIBILITY_DIRECTIVE(bignum_mul_p256k1_x8)
        S2N_BN_FUNCTION_TYPE_DIRECTIVE(bignum_mul_p256k1_x8)
        S2N_BN_SYM_PRIVACY_DIRECTIVE(bignum_mul_p256k1_x8)
        .text
        .balign 4

#if WINDOWS_ABI
#define SHADOW 32
#else
#define SHADOW 0
#endif

// Stack layout (offsets from rsp after the frame is set up), all 64-byte
// rows (one zmm register's worth = 8 lanes x 8 bytes):
//   ABUF : 5 rows -- x's radix52 digits, SoA (row k, lane L -> digit k of x[L])
//   BBUF : 5 rows -- same for y
//   HBUF : 8 rows -- repacked radix-2^64 words, SoA, ready for per-lane reduction
#define ABUF_OFF (SHADOW+0)
#define BBUF_OFF (ABUF_OFF+5*64)
#define HBUF_OFF (BBUF_OFF+5*64)
#define XSAVE_OFF (HBUF_OFF+8*64)
// Windows x64 ABI: only the LOW 128 bits of xmm6..xmm15 are callee-saved
// (the upper bits of zmm6..15, and all of zmm16..31, are volatile). This
// function's working registers span zmm0..zmm11, so xmm6..xmm11 (6
// registers) must be saved -- NOT just xmm6 as an earlier revision did,
// which silently clobbered a caller's xmm7..xmm11 whenever the compiler
// happened to keep something long-lived there across the call (e.g. a
// `double` local in a big, multi-phase caller). zmm16.. are unaffected:
// registers 16 and up are fully volatile under this ABI.
#define NSPACE   (XSAVE_OFF+6*16)

#define MASK52IMM 0xFFFFFFFFFFFFF

S2N_BN_SYMBOL(bignum_mul_p256k1_x8):
        CFI_START
        _CET_ENDBR

#if WINDOWS_ABI
        CFI_PUSH(rdi)
        CFI_PUSH(rsi)
        mov     rdi, rcx
        mov     rsi, rdx
        mov     rdx, r8
#endif
        CFI_PUSH(rbx)
        CFI_PUSH(rbp)
        CFI_PUSH(r12)
        CFI_PUSH(r13)
        CFI_PUSH(r14)
        CFI_PUSH(r15)

        // z, x, y pointers are needed only in phases 1 (x,y) and 5 (z);
        // keep them in callee-saved regs across the vector phases.
        mov     r12, rdi        // z
        mov     r13, rsi        // x
        mov     r14, rdx        // y

        CFI_DEC_RSP(NSPACE)

#if WINDOWS_ABI
        CFI_STACKSAVEU(xmm6, XSAVE_OFF+0*16)
        CFI_STACKSAVEU(xmm7, XSAVE_OFF+1*16)
        CFI_STACKSAVEU(xmm8, XSAVE_OFF+2*16)
        CFI_STACKSAVEU(xmm9, XSAVE_OFF+3*16)
        CFI_STACKSAVEU(xmm10, XSAVE_OFF+4*16)
        CFI_STACKSAVEU(xmm11, XSAVE_OFF+5*16)
#endif

        mov     r15, MASK52IMM

// ----------------------------------------------------------------------
// Phase 1: AoS(8 x u64[4]) -> SoA radix52 (5 rows x 8 lanes), for both
// x (-> ABUF) and y (-> BBUF). One small loop, run twice.
// ----------------------------------------------------------------------

        lea     rbx, [rsp+ABUF_OFF]
        mov     rbp, r13                 // source pointer, advances by lane
        xor     ecx, ecx                 // lane counter
1:
        mov     r8, [rbp]
        mov     r9, [rbp+8]
        mov     r10, [rbp+16]
        mov     r11, [rbp+24]

        mov     rax, r8
        and     rax, r15
        mov     [rbx+0*64+rcx*8], rax

        mov     rax, r8
        shr     rax, 52
        mov     rdx, r9
        shl     rdx, 12
        or      rax, rdx
        and     rax, r15
        mov     [rbx+1*64+rcx*8], rax

        mov     rax, r9
        shr     rax, 40
        mov     rdx, r10
        shl     rdx, 24
        or      rax, rdx
        and     rax, r15
        mov     [rbx+2*64+rcx*8], rax

        mov     rax, r10
        shr     rax, 28
        mov     rdx, r11
        shl     rdx, 36
        or      rax, rdx
        and     rax, r15
        mov     [rbx+3*64+rcx*8], rax

        mov     rax, r11
        shr     rax, 16
        mov     [rbx+4*64+rcx*8], rax

        add     rbp, 32
        inc     ecx
        cmp     ecx, 8
        jl      1b

        lea     rbx, [rsp+BBUF_OFF]
        mov     rbp, r14
        xor     ecx, ecx
2:
        mov     r8, [rbp]
        mov     r9, [rbp+8]
        mov     r10, [rbp+16]
        mov     r11, [rbp+24]

        mov     rax, r8
        and     rax, r15
        mov     [rbx+0*64+rcx*8], rax

        mov     rax, r8
        shr     rax, 52
        mov     rdx, r9
        shl     rdx, 12
        or      rax, rdx
        and     rax, r15
        mov     [rbx+1*64+rcx*8], rax

        mov     rax, r9
        shr     rax, 40
        mov     rdx, r10
        shl     rdx, 24
        or      rax, rdx
        and     rax, r15
        mov     [rbx+2*64+rcx*8], rax

        mov     rax, r10
        shr     rax, 28
        mov     rdx, r11
        shl     rdx, 36
        or      rax, rdx
        and     rax, r15
        mov     [rbx+3*64+rcx*8], rax

        mov     rax, r11
        shr     rax, 16
        mov     [rbx+4*64+rcx*8], rax

        add     rbp, 32
        inc     ecx
        cmp     ecx, 8
        jl      2b

// ----------------------------------------------------------------------
// Phase 2: batch-8 schoolbook multiply. zmm16..zmm20 = A0..A4, zmm21..25
// = B0..B4, zmm0..zmm9 = D0..D9 (accumulators). Pure elementwise: every
// vpmadd52 combines "digit i of every lane" with "digit j of every lane",
// no cross-lane movement at all -- this is the entire reason batching
// independent elements is simpler here than vectorizing one big multiply.
// ----------------------------------------------------------------------

        vmovdqu64 zmm16, [rsp+ABUF_OFF+0*64]
        vmovdqu64 zmm17, [rsp+ABUF_OFF+1*64]
        vmovdqu64 zmm18, [rsp+ABUF_OFF+2*64]
        vmovdqu64 zmm19, [rsp+ABUF_OFF+3*64]
        vmovdqu64 zmm20, [rsp+ABUF_OFF+4*64]
        vmovdqu64 zmm21, [rsp+BBUF_OFF+0*64]
        vmovdqu64 zmm22, [rsp+BBUF_OFF+1*64]
        vmovdqu64 zmm23, [rsp+BBUF_OFF+2*64]
        vmovdqu64 zmm24, [rsp+BBUF_OFF+3*64]
        vmovdqu64 zmm25, [rsp+BBUF_OFF+4*64]

        vpxorq  zmm0, zmm0, zmm0
        vpxorq  zmm1, zmm1, zmm1
        vpxorq  zmm2, zmm2, zmm2
        vpxorq  zmm3, zmm3, zmm3
        vpxorq  zmm4, zmm4, zmm4
        vpxorq  zmm5, zmm5, zmm5
        vpxorq  zmm6, zmm6, zmm6
        vpxorq  zmm7, zmm7, zmm7
        vpxorq  zmm8, zmm8, zmm8
        vpxorq  zmm9, zmm9, zmm9

// D[i+j]   += lo52(A[i]*B[j]) ; D[i+j+1] += hi52(A[i]*B[j])  for i,j=0..4
#define MTERM(AI,BJ,DLO,DHI)               \
        vpmadd52luq DLO, AI, BJ;           \
        vpmadd52huq DHI, AI, BJ

        MTERM(zmm16,zmm21, zmm0,zmm1)   // i=0,j=0 -> D0,D1
        MTERM(zmm16,zmm22, zmm1,zmm2)   // i=0,j=1 -> D1,D2
        MTERM(zmm16,zmm23, zmm2,zmm3)   // i=0,j=2
        MTERM(zmm16,zmm24, zmm3,zmm4)   // i=0,j=3
        MTERM(zmm16,zmm25, zmm4,zmm5)   // i=0,j=4

        MTERM(zmm17,zmm21, zmm1,zmm2)   // i=1,j=0 -> D1,D2
        MTERM(zmm17,zmm22, zmm2,zmm3)   // i=1,j=1
        MTERM(zmm17,zmm23, zmm3,zmm4)   // i=1,j=2
        MTERM(zmm17,zmm24, zmm4,zmm5)   // i=1,j=3
        MTERM(zmm17,zmm25, zmm5,zmm6)   // i=1,j=4

        MTERM(zmm18,zmm21, zmm2,zmm3)   // i=2,j=0
        MTERM(zmm18,zmm22, zmm3,zmm4)   // i=2,j=1
        MTERM(zmm18,zmm23, zmm4,zmm5)   // i=2,j=2
        MTERM(zmm18,zmm24, zmm5,zmm6)   // i=2,j=3
        MTERM(zmm18,zmm25, zmm6,zmm7)   // i=2,j=4

        MTERM(zmm19,zmm21, zmm3,zmm4)   // i=3,j=0
        MTERM(zmm19,zmm22, zmm4,zmm5)   // i=3,j=1
        MTERM(zmm19,zmm23, zmm5,zmm6)   // i=3,j=2
        MTERM(zmm19,zmm24, zmm6,zmm7)   // i=3,j=3
        MTERM(zmm19,zmm25, zmm7,zmm8)   // i=3,j=4

        MTERM(zmm20,zmm21, zmm4,zmm5)   // i=4,j=0
        MTERM(zmm20,zmm22, zmm5,zmm6)   // i=4,j=1
        MTERM(zmm20,zmm23, zmm6,zmm7)   // i=4,j=2
        MTERM(zmm20,zmm24, zmm7,zmm8)   // i=4,j=3
        MTERM(zmm20,zmm25, zmm8,zmm9)   // i=4,j=4

// ----------------------------------------------------------------------
// Phase 3: carry-propagate D0..D9 to canonical (<2^52) form, entirely in
// the vector domain (elementwise across all 8 lanes; the dependency
// chain runs D0->D1->...->D9, same structure as the scalar version, just
// 8-wide).
// ----------------------------------------------------------------------

        vpbroadcastq zmm31, r15          // MASK52, broadcast
        vpxorq  zmm30, zmm30, zmm30      // running carry, starts at 0

#define CARRY_STEP_V(D)                    \
        vpaddq  D, D, zmm30;               \
        vpsrlq  zmm30, D, 52;              \
        vpandq  D, D, zmm31

        CARRY_STEP_V(zmm0)
        CARRY_STEP_V(zmm1)
        CARRY_STEP_V(zmm2)
        CARRY_STEP_V(zmm3)
        CARRY_STEP_V(zmm4)
        CARRY_STEP_V(zmm5)
        CARRY_STEP_V(zmm6)
        CARRY_STEP_V(zmm7)
        CARRY_STEP_V(zmm8)
        CARRY_STEP_V(zmm9)
        // zmm30 (final carry-out) is provably 0 here, same argument as
        // the single-item version (product of two <2^256 values is
        // exactly representable in 10 canonical 52-bit digits).

// ----------------------------------------------------------------------
// Phase 4: repack the 10 canonical SoA digits into 8 SoA radix-2^64
// rows (HBUF), entirely with vector shift/or -- still no cross-lane
// movement, same bit-slicing as the scalar repack, just 8-wide.
// ----------------------------------------------------------------------

        // H0 = D0 | (D1<<52)
        vpsllq  zmm10, zmm1, 52
        vporq   zmm10, zmm0, zmm10
        vmovdqu64 [rsp+HBUF_OFF+0*64], zmm10

        // H1 = (D1>>12) | (D2<<40)
        vpsrlq  zmm10, zmm1, 12
        vpsllq  zmm11, zmm2, 40
        vporq   zmm10, zmm10, zmm11
        vmovdqu64 [rsp+HBUF_OFF+1*64], zmm10

        // H2 = (D2>>24) | (D3<<28)
        vpsrlq  zmm10, zmm2, 24
        vpsllq  zmm11, zmm3, 28
        vporq   zmm10, zmm10, zmm11
        vmovdqu64 [rsp+HBUF_OFF+2*64], zmm10

        // H3 = (D3>>36) | (D4<<16)
        vpsrlq  zmm10, zmm3, 36
        vpsllq  zmm11, zmm4, 16
        vporq   zmm10, zmm10, zmm11
        vmovdqu64 [rsp+HBUF_OFF+3*64], zmm10

        // H4 = (D4>>48) | (D5<<4) | (D6<<56)
        vpsrlq  zmm10, zmm4, 48
        vpsllq  zmm11, zmm5, 4
        vporq   zmm10, zmm10, zmm11
        vpsllq  zmm11, zmm6, 56
        vporq   zmm10, zmm10, zmm11
        vmovdqu64 [rsp+HBUF_OFF+4*64], zmm10

        // H5 = (D6>>8) | (D7<<44)
        vpsrlq  zmm10, zmm6, 8
        vpsllq  zmm11, zmm7, 44
        vporq   zmm10, zmm10, zmm11
        vmovdqu64 [rsp+HBUF_OFF+5*64], zmm10

        // H6 = (D7>>20) | (D8<<32)
        vpsrlq  zmm10, zmm7, 20
        vpsllq  zmm11, zmm8, 32
        vporq   zmm10, zmm10, zmm11
        vmovdqu64 [rsp+HBUF_OFF+6*64], zmm10

        // H7 = (D8>>32) | (D9<<20)
        vpsrlq  zmm10, zmm8, 32
        vpsllq  zmm11, zmm9, 20
        vporq   zmm10, zmm10, zmm11
        vmovdqu64 [rsp+HBUF_OFF+7*64], zmm10

        vzeroupper

// ----------------------------------------------------------------------
// Phase 5: per-lane reduction, 8 repetitions of the exact scalar
// algorithm proven in bignum_mul_p256k1.S, reading lane L's h[0..7] from
// HBUF (row i, lane L) and writing z[L][0..3] via r12 (=z).
// ----------------------------------------------------------------------

#define RED_LANE(L)                                    \
        mov     r8,  [rsp+HBUF_OFF+0*64+8*(L)];        \
        mov     r9,  [rsp+HBUF_OFF+1*64+8*(L)];        \
        mov     r10, [rsp+HBUF_OFF+2*64+8*(L)];        \
        mov     r11, [rsp+HBUF_OFF+3*64+8*(L)];        \
        mov     rax, [rsp+HBUF_OFF+4*64+8*(L)];        \
        mov     rbx, [rsp+HBUF_OFF+5*64+8*(L)];        \
        mov     rbp, [rsp+HBUF_OFF+6*64+8*(L)];        \
        mov     rcx, [rsp+HBUF_OFF+7*64+8*(L)];        \
        mov     r13, rax;                              \
        mov     r14, rbx;                              \
        mov     r15, rbp;                              \
        /* rdx = 4294968273; fold h[4..7] into l[0..3], same as the */  \
        /* scalar tail, using mulx+adcx/adox with r13/r14/r15/rcx as */ \
        /* h and a fresh zero (rbp is dead after the copies above, */   \
        /* reused as zero here) */                                     \
        mov     rdx, 4294968273;                        \
        xor     ebp, ebp;                                \
        mulx    rax, rbx, r13;                           \
        adcx    r8, rbx;                                 \
        adox    r9, rax;                                 \
        mulx    rax, rbx, r14;                            \
        adcx    r9, rbx;                                  \
        adox    r10, rax;                                 \
        mulx    rax, rbx, r15;                             \
        adcx    r10, rbx;                                   \
        adox    r11, rax;                                    \
        mulx    r13, rbx, rcx;                                \
        adcx    r11, rbx;                                      \
        adox    r13, rbp;                                        \
        adcx    r13, rbp;                                          \
        lea     rax, [r13+1];                                       \
        mulx    rbx, rax, rax;                                        \
        add     r8, rax;                                              \
        adc     r9, rbx;                                              \
        adc     r10, rbp;                                             \
        adc     r11, rbp;                                             \
        cmovc   rdx, rbp;                                             \
        sub     r8, rdx;                                              \
        sbb     r9, rbp;                                              \
        sbb     r10, rbp;                                             \
        sbb     r11, rbp;                                             \
        mov     [r12+32*(L)+0], r8;                                   \
        mov     [r12+32*(L)+8], r9;                                   \
        mov     [r12+32*(L)+16], r10;                                 \
        mov     [r12+32*(L)+24], r11

        RED_LANE(0)
        RED_LANE(1)
        RED_LANE(2)
        RED_LANE(3)
        RED_LANE(4)
        RED_LANE(5)
        RED_LANE(6)
        RED_LANE(7)

#if WINDOWS_ABI
        CFI_STACKLOADU(xmm6, XSAVE_OFF+0*16)
        CFI_STACKLOADU(xmm7, XSAVE_OFF+1*16)
        CFI_STACKLOADU(xmm8, XSAVE_OFF+2*16)
        CFI_STACKLOADU(xmm9, XSAVE_OFF+3*16)
        CFI_STACKLOADU(xmm10, XSAVE_OFF+4*16)
        CFI_STACKLOADU(xmm11, XSAVE_OFF+5*16)
#endif

        CFI_INC_RSP(NSPACE)
        CFI_POP(r15)
        CFI_POP(r14)
        CFI_POP(r13)
        CFI_POP(r12)
        CFI_POP(rbp)
        CFI_POP(rbx)

#if WINDOWS_ABI
        CFI_POP(rsi)
        CFI_POP(rdi)
#endif
        CFI_RET

S2N_BN_SIZE_DIRECTIVE(bignum_mul_p256k1_x8)

#if defined(__linux__) && defined(__ELF__)
.section .note.GNU-stack,"",%progbits
#endif
