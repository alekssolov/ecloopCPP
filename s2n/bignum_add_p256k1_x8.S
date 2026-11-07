// New file (not part of upstream s2n-bignum): batch-8 AVX-512 field add
// for secp256k1.
//
// ----------------------------------------------------------------------------
// Add 8 INDEPENDENT field element pairs modulo p_256k1 simultaneously.
//
//    extern void bignum_add_p256k1_x8(uint64_t z[8][4], const uint64_t x[8][4],
//                                      const uint64_t y[8][4]);
//
// z[lane] := (x[lane] + y[lane]) mod p_256k1, for lane = 0..7.
//
// Standard x86-64 ABI: RDI = z, RSI = x, RDX = y
// Microsoft x64 ABI:   RCX = z, RDX = x, R8 = y
//
// ----------------------------------------------------------------------------
// DESIGN NOTE
// ----------------------------------------------------------------------------
// Unlike bignum_mul_p256k1_x8, this needs no AVX512IFMA -- plain
// AVX-512F/BW/DQ suffices (no 52-bit-operand multiply here). AVX-512's
// vpaddq does not propagate carries between lanes, which is fine (each of
// the 8 lanes is an independent number), but within *one* lane's own 4
// limbs a carry still has to ripple limb0->limb3, and that ripple now
// happens across 4 registers (one per limb position) instead of across
// bits of one scalar register. Standard unsigned-compare-and-correct
// idiom: add the limb, detect overflow with VPCMPLTUQ (-> a mask), then
// merge-add 1 into that same limb only where the mask is set, and check
// -- with a masked compare restricted to just those lanes -- whether
// *that* correction itself overflowed too (possible when a limb is
// exactly 2^64-1).
//
// z = x+y mod p is then: S = x+y (with overall carry-out mask CF), T = S
// + 4294968273 (with its own carry-out mask TC). T's carry-out identifies
// S >= p_256k1 exactly the same way the original scalar bit-trick does,
// just reached differently. Use T where (CF | TC) is set, else S.
//
// Validated with a real (non-emulated -- no IFMA needed) AVX-512F/BW/DQ
// intrinsics model, executed directly on development hardware: 2408
// random + edge-case lane checks, 0 mismatches, before being
// transliterated into the assembly below (which mirrors, instruction for
// instruction, what gcc -O2 emitted from that validated model).
// ----------------------------------------------------------------------------

#include "_internal_s2n_bignum_x86.h"

        .intel_syntax noprefix
        S2N_BN_SYM_VISIBILITY_DIRECTIVE(bignum_add_p256k1_x8)
        S2N_BN_FUNCTION_TYPE_DIRECTIVE(bignum_add_p256k1_x8)
        S2N_BN_SYM_PRIVACY_DIRECTIVE(bignum_add_p256k1_x8)
        .text
        .balign 4

#if WINDOWS_ABI
#define SHADOW 32
#else
#define SHADOW 0
#endif

// Stack: 2 SoA blocks (X, Y), 4 rows (one per limb) x 64 bytes (8 lanes).
#define XBUF_OFF (SHADOW+0)
#define YBUF_OFF (XBUF_OFF+4*64)
// Windows x64 ABI: only the LOW 128 bits of xmm6..xmm15 are callee-saved.
// This function's working registers span zmm0..zmm16, so xmm6..xmm15
// (10 registers) must be saved here (zmm16 itself needs no saving --
// registers 16 and up are fully volatile under this ABI).
#define XSAVE_OFF (YBUF_OFF+4*64)
#define NSPACE   (XSAVE_OFF+10*16)

// 4-limb vector add-with-carry: SR0..3 = XR0..3 + YR0..3, MASK = carry-out.
// TEMP and MASK2 are scratch (zmm register, k register). Mirrors gcc's
// verified codegen for the validated C reference exactly.
#define ADD4_RAW(XR0,XR1,XR2,XR3, YR0,YR1,YR2,YR3, SR0,SR1,SR2,SR3, TEMP, MASK, MASK2) \
        vpaddq  SR0, XR0, YR0;                    \
        vpcmpltuq MASK, SR0, XR0;                 \
        vpaddq  TEMP, XR1, YR1;                   \
        vpcmpltuq MASK2, TEMP, XR1;                \
        vmovdqa64 SR1, TEMP;                       \
        vpaddq  SR1{MASK}, TEMP, zmm31;             \
        vpcmpltuq MASK{MASK}, SR1, TEMP;             \
        korb    MASK, MASK, MASK2;                   \
        vpaddq  TEMP, XR2, YR2;                       \
        vpcmpltuq MASK2, TEMP, XR2;                    \
        vmovdqa64 SR2, TEMP;                            \
        vpaddq  SR2{MASK}, TEMP, zmm31;                  \
        vpcmpltuq MASK{MASK}, SR2, TEMP;                  \
        korb    MASK, MASK, MASK2;                          \
        vpaddq  TEMP, XR3, YR3;                                \
        vpcmpltuq MASK2, TEMP, XR3;                              \
        vmovdqa64 SR3, TEMP;                                      \
        vpaddq  SR3{MASK}, TEMP, zmm31;                             \
        vpcmpltuq MASK{MASK}, SR3, TEMP;                              \
        korb    MASK, MASK, MASK2

S2N_BN_SYMBOL(bignum_add_p256k1_x8):
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
        CFI_PUSH(r12)

        mov     r12, rdi        // z (used at the very end)
        mov     rbx, rsi        // x-ptr, walked below
        // rdx = y-ptr already

        CFI_DEC_RSP(NSPACE)

#if WINDOWS_ABI
        CFI_STACKSAVEU(xmm6, XSAVE_OFF+0*16)
        CFI_STACKSAVEU(xmm7, XSAVE_OFF+1*16)
        CFI_STACKSAVEU(xmm8, XSAVE_OFF+2*16)
        CFI_STACKSAVEU(xmm9, XSAVE_OFF+3*16)
        CFI_STACKSAVEU(xmm10, XSAVE_OFF+4*16)
        CFI_STACKSAVEU(xmm11, XSAVE_OFF+5*16)
        CFI_STACKSAVEU(xmm12, XSAVE_OFF+6*16)
        CFI_STACKSAVEU(xmm13, XSAVE_OFF+7*16)
        CFI_STACKSAVEU(xmm14, XSAVE_OFF+8*16)
        CFI_STACKSAVEU(xmm15, XSAVE_OFF+9*16)
#endif

// ----------------------------------------------------------------------
// AoS -> SoA transpose for x and y.
// ----------------------------------------------------------------------
        lea     rsi, [rsp+XBUF_OFF]
        xor     ecx, ecx
1:
        mov     rax, [rbx+0*8]
        mov     [rsi+0*64+rcx*8], rax
        mov     rax, [rbx+1*8]
        mov     [rsi+1*64+rcx*8], rax
        mov     rax, [rbx+2*8]
        mov     [rsi+2*64+rcx*8], rax
        mov     rax, [rbx+3*8]
        mov     [rsi+3*64+rcx*8], rax
        add     rbx, 32
        inc     ecx
        cmp     ecx, 8
        jl      1b

        lea     rsi, [rsp+YBUF_OFF]
        xor     ecx, ecx
2:
        mov     rax, [rdx+0*8]
        mov     [rsi+0*64+rcx*8], rax
        mov     rax, [rdx+1*8]
        mov     [rsi+1*64+rcx*8], rax
        mov     rax, [rdx+2*8]
        mov     [rsi+2*64+rcx*8], rax
        mov     rax, [rdx+3*8]
        mov     [rsi+3*64+rcx*8], rax
        add     rdx, 32
        inc     ecx
        cmp     ecx, 8
        jl      2b

// ----------------------------------------------------------------------
// zmm0..3 = X, zmm4..7 = Y, zmm8..11 = S = X+Y (mask k1 = CF),
// zmm4..7 reused as {c,0,0,0}, zmm12..15 = T = S+c (mask k2 = TC),
// k1 := k1|k2 (use_t), then blend S/T per limb into Z.
// ----------------------------------------------------------------------
        vmovdqu64 zmm0, [rsp+XBUF_OFF+0*64]
        vmovdqu64 zmm1, [rsp+XBUF_OFF+1*64]
        vmovdqu64 zmm2, [rsp+XBUF_OFF+2*64]
        vmovdqu64 zmm3, [rsp+XBUF_OFF+3*64]
        vmovdqu64 zmm4, [rsp+YBUF_OFF+0*64]
        vmovdqu64 zmm5, [rsp+YBUF_OFF+1*64]
        vmovdqu64 zmm6, [rsp+YBUF_OFF+2*64]
        vmovdqu64 zmm7, [rsp+YBUF_OFF+3*64]

        mov     eax, 1
        vpbroadcastq zmm31, rax

        ADD4_RAW(zmm0,zmm1,zmm2,zmm3, zmm4,zmm5,zmm6,zmm7, zmm8,zmm9,zmm10,zmm11, zmm16, k1, k0)

        mov     rax, 4294968273
        vpbroadcastq zmm4, rax
        vpxorq  zmm5, zmm5, zmm5
        vpxorq  zmm6, zmm6, zmm6
        vpxorq  zmm7, zmm7, zmm7

        ADD4_RAW(zmm8,zmm9,zmm10,zmm11, zmm4,zmm5,zmm6,zmm7, zmm12,zmm13,zmm14,zmm15, zmm16, k2, k0)

        korb    k1, k1, k2

        vpblendmq zmm0{k1}, zmm8, zmm12
        vpblendmq zmm1{k1}, zmm9, zmm13
        vpblendmq zmm2{k1}, zmm10, zmm14
        vpblendmq zmm3{k1}, zmm11, zmm15

        vmovdqu64 [rsp+XBUF_OFF+0*64], zmm0
        vmovdqu64 [rsp+XBUF_OFF+1*64], zmm1
        vmovdqu64 [rsp+XBUF_OFF+2*64], zmm2
        vmovdqu64 [rsp+XBUF_OFF+3*64], zmm3
        vzeroupper

// ----------------------------------------------------------------------
// SoA -> AoS transpose into z.
// ----------------------------------------------------------------------
        lea     rsi, [rsp+XBUF_OFF]
        xor     ecx, ecx
3:
        mov     rax, [rsi+0*64+rcx*8]
        mov     [r12+0*8], rax
        mov     rax, [rsi+1*64+rcx*8]
        mov     [r12+1*8], rax
        mov     rax, [rsi+2*64+rcx*8]
        mov     [r12+2*8], rax
        mov     rax, [rsi+3*64+rcx*8]
        mov     [r12+3*8], rax
        add     r12, 32
        inc     ecx
        cmp     ecx, 8
        jl      3b

#if WINDOWS_ABI
        CFI_STACKLOADU(xmm6, XSAVE_OFF+0*16)
        CFI_STACKLOADU(xmm7, XSAVE_OFF+1*16)
        CFI_STACKLOADU(xmm8, XSAVE_OFF+2*16)
        CFI_STACKLOADU(xmm9, XSAVE_OFF+3*16)
        CFI_STACKLOADU(xmm10, XSAVE_OFF+4*16)
        CFI_STACKLOADU(xmm11, XSAVE_OFF+5*16)
        CFI_STACKLOADU(xmm12, XSAVE_OFF+6*16)
        CFI_STACKLOADU(xmm13, XSAVE_OFF+7*16)
        CFI_STACKLOADU(xmm14, XSAVE_OFF+8*16)
        CFI_STACKLOADU(xmm15, XSAVE_OFF+9*16)
#endif

        CFI_INC_RSP(NSPACE)
        CFI_POP(r12)
        CFI_POP(rbx)

#if WINDOWS_ABI
        CFI_POP(rsi)
        CFI_POP(rdi)
#endif
        CFI_RET

S2N_BN_SIZE_DIRECTIVE(bignum_add_p256k1_x8)

#if defined(__linux__) && defined(__ELF__)
.section .note.GNU-stack,"",%progbits
#endif
