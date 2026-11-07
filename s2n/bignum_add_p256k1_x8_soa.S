// New file (not part of upstream s2n-bignum): SoA-native batch-8 AVX-512
// field add for secp256k1 -- no AoS<->SoA transpose.
//
// ----------------------------------------------------------------------------
//
//    extern void bignum_add_p256k1_x8_soa(uint64_t z[4][8], const uint64_t x[4][8],
//                                          const uint64_t y[4][8]);
//
// Unlike bignum_add_p256k1_x8.S, the layout here is LIMB-MAJOR (SoA): z[i]
// is a full 8-lane row (limb i of all 8 independent field elements), not
// z[lane] as in the original. This is the layout an orchestrator (e.g.
// ec_jacobi_add_affine_x8) should use for ALL its persistent/intermediate
// state, so values pass from one batch-8 primitive to the next as direct
// zmm loads -- no scalar transpose loop needed at all, in either this
// function or its caller. Only the true API boundary (turning 8 separate
// per-lane fe's into one SoA blob, and back) ever needs the AoS<->SoA
// transpose -- see design notes in bignum_mul_p256k1_x8.S and the
// ecc.c orchestrators for where that boundary now sits.
//
// (This mirrors, at the level of this codebase's own architecture, the
// pattern in Intel's own crypto_mb AVX512IFMA multi-buffer library:
// transform_8sb_to_mb8 does the one AoS->SoA conversion at the API
// boundary; every internal field/point operation from there on takes and
// returns the SoA form directly, with no further transpose.)
//
// Same validated carry logic as bignum_add_p256k1_x8.S (2408+4000 lane
// checks on that core across two independent test harnesses); only the
// load/store convention differs here.
// ----------------------------------------------------------------------------

#include "_internal_s2n_bignum_x86.h"

        .intel_syntax noprefix
        S2N_BN_SYM_VISIBILITY_DIRECTIVE(bignum_add_p256k1_x8_soa)
        S2N_BN_FUNCTION_TYPE_DIRECTIVE(bignum_add_p256k1_x8_soa)
        S2N_BN_SYM_PRIVACY_DIRECTIVE(bignum_add_p256k1_x8_soa)
        .text
        .balign 4

#if WINDOWS_ABI
#define SHADOW 32
#else
#define SHADOW 0
#endif

#define XSAVE_OFF SHADOW
#define NSPACE     (XSAVE_OFF+10*16)

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

S2N_BN_SYMBOL(bignum_add_p256k1_x8_soa):
        CFI_START
        _CET_ENDBR

#if WINDOWS_ABI
#define ZP rcx
#define XP rdx
#define YP r8
#else
#define ZP rdi
#define XP rsi
#define YP rdx
#endif

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

        vmovdqu64 zmm0, [XP+0*64]
        vmovdqu64 zmm1, [XP+1*64]
        vmovdqu64 zmm2, [XP+2*64]
        vmovdqu64 zmm3, [XP+3*64]
        vmovdqu64 zmm4, [YP+0*64]
        vmovdqu64 zmm5, [YP+1*64]
        vmovdqu64 zmm6, [YP+2*64]
        vmovdqu64 zmm7, [YP+3*64]

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

        vmovdqu64 [ZP+0*64], zmm0
        vmovdqu64 [ZP+1*64], zmm1
        vmovdqu64 [ZP+2*64], zmm2
        vmovdqu64 [ZP+3*64], zmm3
        vzeroupper

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
        CFI_RET

S2N_BN_SIZE_DIRECTIVE(bignum_add_p256k1_x8_soa)

#if defined(__linux__) && defined(__ELF__)
.section .note.GNU-stack,"",%progbits
#endif
