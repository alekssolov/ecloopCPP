.PHONY: default clean build rebuild fmt

TARGET := ecloop
BUILD_DIR := build

CXX ?= g++
CPPFLAGS ?= -Iinclude
CXXFLAGS ?= -std=gnu++17 -O3 -Wall -Wextra -Wno-unused-parameter
LDFLAGS ?=
LDLIBS ?= -pthread

# The CPU to compile for. Default: this machine (-march=native), which
# makes a binary that runs ONLY on CPUs at least as capable -- copied to an
# older machine it dies at start-up with an illegal instruction, silently
# under cmd.exe. To build for another machine, name its level:
#   make MARCH=x86-64-v3 rebuild        Haswell and later (AVX2, BMI2/ADX): fine with the s2n assembly
#   make MARCH=x86-64-v2 PORTABLE=1 rebuild   Nehalem/Sandy Bridge/Core 2: no BMI2, portable arithmetic
#   make MARCH=x86-64 PORTABLE=1 rebuild      anything 64-bit
# or simply build on the target machine.
MARCH ?= native
ifeq ($(shell uname -m),x86_64)
	CXXFLAGS += -march=$(MARCH)
endif

# Batch-8 (AVX512-IFMA) tuning. ON by default -- measured on the mixed
# addition: 9.0 M points/s with the stock .S multiply against 12.4 with
# these on. Harmless on a CPU without IFMA: ec_batch8.cpp drops the v2
# kernels when the target cannot compile them.
#   make build EC8=0     to turn them off.
EC8 ?= 1
ifneq ($(EC8),0)
	CXXFLAGS += -DEC8_MULV2=1 -DEC8_FE8V2=1
endif

# Workers are processes (fork), so the hashing buffers are plain statics.
# Set this only if the batch routines are ever called from several threads
# of one process.
#   make build THREADED=1
ifeq ($(THREADED),1)
	CXXFLAGS += -DECLOOP_THREADED=1
endif

# PORTABLE=1 builds without LTO unless asked otherwise: the CPUs it is for
# gain nothing from it, and on one toolchain the LTO link of that build
# ended with unresolved cross-partition references (matches_prefix,
# rmd160_final, report_found) that a plain link does not have.
ifeq ($(PORTABLE),1)
LTO ?= 0
endif

# Link-time optimisation. ON by default: without it the small field
# operations stay real calls across translation units, where the C tree
# inlined them all into one unity build.
#   make build LTO=0     to turn it off (faster builds, slower binary)
LTO ?= 1
ifneq ($(LTO),0)
	CXXFLAGS += -flto=auto
	LDFLAGS  += -flto=auto
endif

SRCS := $(shell find src -name '*.cpp' | sort)
OBJS := $(SRCS:src/%.cpp=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

# ── the assembly: s2n-bignum secp256k1 routines and the SHA-256 x16 kernel
# Built from s2n/*.S right here, for the platform make runs on:
#   Linux    gcc -c -Is2n -DWINDOWS_ABI=0 f.S
#   Cygwin   the s2n macros expand to ';'-separated instructions on one
#            line, which the Cygwin assembler rejects, so: preprocess with
#            -DWINDOWS_ABI=1, split on ';', assemble (the recipe from
#            s2n/compile.txt).
# sha256_x16_avx512.S detects the ABI itself and must NOT go through the
# split (its comments contain ';'): plain gcc -c on both platforms.
# The .o files are kept next to the sources and rebuilt when a .S changes.
#
# PORTABLE=1: for a CPU without BMI2/ADX (mulx/adcx/adox -- anything older
# than Haswell), where those objects raise SIGILL. The field and point
# routines then come from src/s2n_portable.cpp (plain C++, same names and
# semantics, verified against the assembly) and no assembly is built or
# linked. Every SIMD path selects itself off at compile or run time, so
# the result is the scalar search: correct, and slow.
#   make PORTABLE=1 rebuild
UNAME_S := $(shell uname -s)
ifneq (,$(findstring CYGWIN,$(UNAME_S))$(findstring MINGW,$(UNAME_S))$(findstring MSYS,$(UNAME_S)))
	S2N_WINDOWS_ABI := 1
else
	S2N_WINDOWS_ABI := 0
endif
AS ?= as

S2N_ASM  := $(wildcard s2n/*.S)
# The default build is UNIVERSAL: one binary for every x86-64 CPU.
# Baseline -march=x86-64, every SIMD tier compiled in under its own target
# pragma and picked at run time (cpuid); the s2n assembly and the portable
# field code are both linked and chosen at start-up (s2n_dispatch.hpp).
#   make            (or make rebuild)
# UNIVERSAL=0 gives the old machine-specific build (-march=native or MARCH=)
# in build-native/, for comparison only.
UNIVERSAL ?= 1
ifeq ($(UNIVERSAL),1)
	MARCH := native
	CXXFLAGS += -DECLOOP_UNIVERSAL=0
else
	BUILD_DIR := build-native
	OBJS := $(SRCS:src/%.cpp=$(BUILD_DIR)/%.o)
	DEPS := $(OBJS:.o=.d)
endif
ifeq ($(PORTABLE),1)
	CXXFLAGS += -DECLOOP_PORTABLE_S2N=1
	# Only the SHA-256 x16 kernel stays: it has no BMI2/ADX, detects the
	# ABI itself, is referenced only when the compiler targets AVX-512 and
	# runs only when the CPU has it. The secp256k1 assembly is out.
	S2N_OBJS := $(filter s2n/sha256_x16_avx512.o,$(S2N_ASM:.S=.o))
	# Its own object directory: objects compiled with and without the
	# portable define must never meet in one link.
	BUILD_DIR := build-portable
	OBJS := $(SRCS:src/%.cpp=$(BUILD_DIR)/%.o)
	DEPS := $(OBJS:.o=.d)
else
	S2N_OBJS := $(S2N_ASM:.S=.o)
endif

default: build

build: $(TARGET)

rebuild: clean build

# LDFLAGS_EXTRA: e.g. -static-libstdc++ -static-libgcc (build-all.sh), so a
# Cygwin binary needs only cygwin1.dll from the outside.
$(TARGET): $(OBJS) $(S2N_OBJS)
	$(CXX) $(LDFLAGS) $(LDFLAGS_EXTRA) $(OBJS) $(S2N_OBJS) -o $@ $(LDLIBS)

s2n/sha256_x16_avx512.o: s2n/sha256_x16_avx512.S
	$(CC) -c $< -o $@

ifeq ($(S2N_WINDOWS_ABI),1)
s2n/%.o: s2n/%.S
	$(CC) -E -Is2n -DWINDOWS_ABI=1 -xassembler-with-cpp $< | tr ';' '\n' | $(AS) -o $@ -
else
s2n/%.o: s2n/%.S
	$(CC) -c -Is2n -DWINDOWS_ABI=0 $< -o $@
endif

s2n: $(S2N_OBJS)
.PHONY: s2n clean-s2n

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

fmt:
	@find . \( -name '*.c' -o -name '*.cpp' -o -name '*.hpp' \) | xargs clang-format -i

clean:
	@rm -rf $(TARGET) build build-native build-portable build-universal *.o *.profraw *.profdata

# The assembly objects survive `clean` (they rarely change and take a while
# on Cygwin); `make clean-s2n` or `make rebuild-all` throws them away too.
clean-s2n:
	@rm -f s2n/*.o

rebuild-all: clean clean-s2n build
.PHONY: rebuild-all

-include $(DEPS)
