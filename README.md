# ecloopCPP

![build](https://github.com/alekssolov/ecloopCPP/actions/workflows/build.yml/badge.svg)

> A C++ fork of [**vladkens/ecloop**](https://github.com/vladkens/ecloop).

The author was inspired to write this program by the **"ecloop"** project
<https://github.com/vladkens/ecloop>, which adds a hash‑collision search for the
RIPEMD‑160 hash (hash160) that is the basis for calculating a Bitcoin wallet
address. This assumes that there are at least 2^96 different private keys that
will have the same RIPEMD‑160 hash and, therefore, the same Bitcoin address.

This software was developed as a tool for studying the security aspects of
practical elliptic‑curve applications. The program enumerates private keys,
calculates the public key and, based on it, the RIPEMD‑160 hash, and then
searches the calculated hash against a list of existing Bitcoin address hashes.
If a hash match is found, it means that a private key for one of the specified
Bitcoin addresses has been found.

## How it works

1. First, you'll need a file with a list of RIPEMD‑160 hashes of the Bitcoin
   addresses you want to search for, in **binary** format (20 bytes per hash).
   A file with Bitcoin address hashes (`btc.bin`) is included with this project.

2. Next, run the program, specifying the created hash file as a parameter with
   `-b` (see *Launch parameters* below).

3. If the initial private key is not specified, one is generated randomly. The
   public key is calculated from the private key, and from it, in turn, the
   RIPEMD‑160 hash is calculated (details on calculating a Bitcoin address from
   a private key:
   <https://en.bitcoin.it/wiki/Technical_background_of_version_1_Bitcoin_addresses>).
   The resulting hash is searched for in the list of hashes of interest. If a
   match is found, the private key has been found, information about the find is
   displayed on the screen and written to the `key.txt` file in the program
   folder. The private key is then incremented by one in an arbitrary location
   and the process is repeated.

---

## Launch parameters

```
ecloop -b <hash160.bin> [-t threads] [-v level] <EC mode> [-m mode -p prefix]
```

`-b` is the only **required** option; running `ecloop` with no arguments prints
a short usage reminder.

| Option | Argument | Default | Description |
|--------|----------|---------|-------------|
| `-b`   | `<file.bin>` | — (required) | Database of hash160 records to search against, **20 bytes each**, stored most‑significant byte first (canonical hash160 order). Loaded once at start‑up and shared copy‑on‑write with every worker. |
| `-t`   | `<n>`    | all usable CPUs | Number of parallel workers — the *threads* option, as in upstream ecloop (the variable is even named `threads`). In this fork each worker is a separate **process** (`fork()`), not an OS thread, which is why the database is shared copy-on-write and progress is reported through a shared memory mapping. A value `≤ 0` falls back to the CPU count. |
| `-s`   | `<bits>` | `256`   | Private‑key size in **bits**. Defines an endomorphism random walk over keys below `2^bits` ( `-s 256` = the whole key space). Clamped to the range `4…256` and rounded **down** to a whole number of hex digits (a multiple of 4), so e.g. `-s 71` becomes `68`. Mutually exclusive with `-k`/`-e`. |
| `-v`   | `<level>`| `0`     | Verbosity: `0` quiet, `1` print the rate line, `2` print every key, `5` also print the hashes. |
| `-k`   | `<hex>`  | —       | **First** private key of a sequential sweep (1…64 hex digits). Supplying `-k` selects *range mode*. Mutually exclusive with `-s`. |
| `-e`   | `<hex>`  | end of key space | **Last** private key of the sweep, inclusive (1…64 hex digits). Requires `-k`. |
| `-z`   | `<low>`  | `0` (off) | **Mixed mode**: after each random draw of the walker, the low `<low>` bits are swept in full. Must be a whole number of hex digits (a multiple of 4); other values are rounded or the mode is disabled with a warning. Mutually exclusive with `-g`. The sweep works in groups of 2048 keys, so each draw covers at least `2^11` keys even for small `-z`. |
| `-g`   | `on` \| `off` \| `auto` | `auto` | Enable/disable the **endomorphism (GLV) walk** inside a `-k`/`-e` range, or choose it for the `-s` walk (`1`/`0` are also accepted). `auto` = on for `-s 254…256` (where the endomorphism images still fall inside the bound), off inside a narrower bound or a range. Mutually exclusive with `-z`. |
| `-m`   | `<mode>` | `""` (DB search) | Search mode — see *Search modes* below. The empty default is the plain hash160‑database search. |
| `-p`   | `<prefix>` | —     | Base58 address prefix for the vanity modes. A leading `1` is prepended automatically if missing. |

> **Note on `-a`.** Upstream ecloop used `-a c|u|cu` to pick the address type.
> This fork checks the compressed **and** uncompressed address of every key, so
> `-a` has been removed — passing it just prints the usage reminder.

### EC mode combinations

| Command shape | Meaning |
|---------------|---------|
| `-s <bits>` | Endomorphism random walk over keys below `2^bits` (`-s 256` = all). |
| `-s <bits> -z <low>` | Mixed: random high part, sequential sweep of the low `<low>` bits. |
| `-s <bits> -g on\|off` | Same walk, forcing the endomorphism on or off (`auto` is on at `254+`). |
| `-k <hex> -e <hex>` | Sequential sweep of the range. |
| `-k <hex> -e <hex> -z <low>` | Mixed mode inside the range. |
| `-k <hex> -e <hex> -g auto\|on\|off` | Endomorphism walk inside the range (`auto` = off, the images fall outside). |

Rules: `-s` and `-k`/`-e` are mutually exclusive (they bound the key space in
two different ways); `-g` and `-z` are mutually exclusive (two different things
to do with a range); `-e` requires `-k`. The vanity options `-m`/`-p` combine
with any of the above.

### Search modes (`-m`)

With no `-m`, the program does the pure **hash160‑database search**: every
computed hash160 is looked up in the `-b` file. When `-m` is given together with
`-p`, the prefix search runs **on top of** the database lookup — each key is
checked against the `.bin` *and* against the prefix.

| `-m` value | What must match | Expected keys per hit |
|------------|-----------------|-----------------------|
| `unc`    | the **uncompressed** address carries the `-p` prefix | `D` |
| `com`    | the **compressed** address carries the `-p` prefix | `D` |
| `any`    | **either** address carries the prefix | `≈ D / 2` |
| `unccom` | the compressed and uncompressed hash160 of a key **agree on their leading bytes** (the number of bytes = the length of `-p`; the characters of `-p` themselves are not used) | `256^bytes` |
| `triple` | `unccom` **and** the prefix on either address | `256^bytes × D` |

(`D` is the exact difficulty of the registered `-p` prefix.) At start‑up the
program prints the effective difficulty for the chosen mode, and during the run
it prints a probability/ETA line derived from the same figure.

### Output files

- **`key.txt`** — written in the program folder when a match is found.
- **`progress_<STARTKEY>.txt`** — for a sequential sweep, the resume point is
  checkpointed to a file named after the start key, so several ranges can be
  swept side by side without overwriting one another. (The default name is
  `progress.txt`.)

### Environment variables

These override the automatic engine/CPU‑feature selection — useful for
benchmarking and for forcing a specific code path on a given CPU:

| Variable | Effect |
|----------|--------|
| `ECLOOP_OLD_ENGINE=1` | Use the single‑walker reference engine instead of the batched `engine_x8`. |
| `ECLOOP_HASH=avx512\|shani\|avx2\|scalar` | Force a specific hashing implementation ("rung") instead of auto‑detecting. |
| `ECLOOP_NO_AVX512_HASH=1` | Skip the AVX‑512 hashing path (use SHA‑NI + AVX2 x8). |
| `ECLOOP_NO_IFMA=1` | Use the scalar field path instead of the AVX‑512‑IFMA batch‑8 path. |
| `ECLOOP_NO_FILTER=1` | Disable the bitmap pre‑filter in front of the database. |
| `ECLOOP_FILTER_MB=<n>` | Set the pre‑filter size in megabytes. |
| `ECLOOP_FILTER_BITS=<n>` | Set the pre‑filter size as `log2` of its bit count. |

---

## Examples

```sh
# Full‑space random walk over all keys, using every CPU (endomorphism auto‑on):
ecloop -b btc.bin

# Sequential sweep of a fixed range (e.g. a puzzle range) with 8 workers:
ecloop -b btc.bin -k 400000000000000000 -e 7fffffffffffffffff -t 8

# Mixed mode inside that range: random high part, full sweep of the low 24 bits:
ecloop -b btc.bin -k 400000000000000000 -e 7fffffffffffffffff -z 24 -t 8

# Endomorphism walk inside the range:
ecloop -b btc.bin -k 400000000000000000 -e 7fffffffffffffffff -g auto -t 8

# Vanity search: a compressed address starting with "1Love" (-b is still required):
ecloop -b btc.bin -m com -p 1Love -t 8 -v 1
```

---

## Build

The project builds with `make` (default compiler `g++`, C++17, `-pthread`).
It needs the C++ sources under `src/` **and** the secp256k1 assembly sources
under `s2n/` (`*.S` plus `sha256_x16_avx512.S`). The Makefile auto-detects
Cygwin / MinGW / MSYS and preprocesses the assembly for the Windows ABI, so the
Cygwin build works out of the box.

```sh
make                 # default: a UNIVERSAL binary - baseline x86-64 with every
                     # SIMD tier compiled in and picked at run time via cpuid;
                     # the one binary runs on any x86-64 CPU
make rebuild         # clean + build
make rebuild-all     # also rebuild the s2n/*.o assembly objects (they survive a
                     # plain `make clean`)
```

Other builds:

```sh
# For older CPUs without BMI2/ADX (anything pre-Haswell), where the secp256k1
# assembly would raise an illegal instruction. Uses src/s2n_portable.cpp
# instead - correct, but slow. Output in build-portable/.
make PORTABLE=1 rebuild

# Machine-specific build (for comparison), output in build-native/:
make UNIVERSAL=0 rebuild
# ...optionally targeting a specific CPU level, e.g. Haswell+ (AVX2, BMI2/ADX):
make UNIVERSAL=0 MARCH=x86-64-v3 rebuild
```

Tuning knobs (see the comments at the top of the `Makefile` for the full
matrix): `EC8=0` disables the AVX-512-IFMA batch-8 kernels, `LTO=0` disables
link-time optimisation (faster builds, slower binary), `THREADED=1` is only
needed if the batch routines are ever called from several threads of one
process. `make fmt` runs clang-format; `make clean` / `make clean-s2n` clean up.

---

## Credits & license

This project is a fork of **ecloop** by **vladkens**
(<https://github.com/vladkens/ecloop>). Please keep the upstream `LICENSE`
file in the repository and comply with its terms (ecloop is distributed under
the MIT License); add your own copyright/attribution notice alongside it.

## Disclaimer

This software is provided for research and educational purposes — studying the
security properties of elliptic‑curve cryptography and Bitcoin addresses. Use it
only on addresses and ranges you are authorized to analyze.
