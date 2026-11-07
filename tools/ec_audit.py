#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ec_audit.py — аудит эллиптической кривой на известные структурные слабости.

Кривая в короткой форме Вейерштрасса  y^2 = x^3 + a*x + b  над F_p,
подгруппа <G> простого порядка n, кофактор h.

Апгрейд secp256k1_audit.cpp. К исходным 12 проверкам добавлено:
  * p проверяется на простоту (в оригинале нет)
  * ДОКАЗАТЕЛЬСТВО порядка группы: единственность h в интервале Хассе
    + кросс-чек случайными точками (не только по G)
  * сильная простота: Miller–Rabin со случайными базами (+ BPSW из sympy, если есть)
  * rho-стойкость основной группы и твиста: plain / negation map / автоморфизмы (j=0, j=1728)
  * ТОЧНАЯ степень вложения (через факторизацию n-1) для кривой и для твиста, с таймаутом;
    плюс критерий SafeCurves k >= (n-1)/100
  * CM-дискриминант D (t^2 - 4p = D*v^2); для j=0 — вывод и проверка GLV-констант beta, lambda
  * полная факторизация порядка твиста, оценка утечки через малые подгруппы твиста
  * тестовые векторы: точки малого порядка на твисте для проверки валидации точек
    в ВАШЕЙ реализации (выводятся по умолчанию; --no-twist-vectors чтобы отключить)
  * форма простого p (Crandall / Solinas) — не уязвимость, а признак реализуемости
    быстрой редукции (A4)
  * явная отметка про Weil descent / GHS: неприменимо над простым полем (C5)
  * rigidity-прокси (A5): класс параметров (минимальные / seeded / необъяснённые);
    для seeded-кривых — проверка seed->b по X9.62 / FIPS 186-4 A.3.3. Только INFO:
    прошедший seed доказывает лишь корректность транскрипции стандарта, не честность
    выбора seed (см. BADA55). Rigidity как таковая — свойство процесса, не вычислима.
  * происхождение генератора (B7): не сконструирован ли G из «малой» точки (x(G/k)
    или x(k*G) аномально мал). Известный случай: у secp256k1 x(G/2) — 166 бит
    (шанс 2^-90). Только INFO — в группе простого порядка выбор генератора на
    сложность DLP не влияет, «малый» прообраз публичен.
  * JSON-вывод; пресеты secp256k1 / P-256 / заведомо слабая суперсингулярная кривая

Использование:
  python3 ec_audit.py                       # secp256k1
  python3 ec_audit.py --curve p256
  python3 ec_audit.py --curve weak_ss       # демо: суперсингулярная кривая, k=2 (MOV)
  python3 ec_audit.py --p 0x.. --a .. --b .. --n 0x.. --gx 0x.. --gy 0x.. [--h 1]
  python3 ec_audit.py --json
  --fast              не факторизовать n-1 (только ограниченный поиск степени вложения)
  --factor-timeout S  таймаут одной факторизации, сек (по умолчанию 60; 0 = без лимита)
  --mov-bound K       граница ограниченного поиска степени вложения (по умолчанию 100000)
  --disc-bound B      граница поиска малого |D| (по умолчанию 10^6)
  --no-sympy          принудительно чистый Python (trial division + Pollard–Brent)
  --no-twist-vectors  не выводить тестовые векторы твиста (по умолчанию выводятся)
  --seed HEX          seed 'verifiably random' кривой для проверки seed->b (A5)

Зависимости: стандартная библиотека. sympy (опционально) — быстрая факторизация и BPSW.

Чего НЕ делает:
  * не считает #E для кривой без заявленного n (нужен Schoof/SEA: PARI/GP `ellcard`, Sage);
  * не проверяет "rigidity" (происхождение параметров) — это не вычислимо, см. спецификацию.
"""

import argparse
import hashlib
import json
import math
import random
import signal
import sys
import time

try:
    import sympy  # noqa: F401
    HAVE_SYMPY = True
except ImportError:
    HAVE_SYMPY = False

SYSRNG = random.SystemRandom()

# ============================================================================
# Timeout helper (POSIX only; no-op elsewhere)
# ============================================================================

class Timeout(Exception):
    pass


class time_limit:
    def __init__(self, seconds):
        self.seconds = seconds
        self.active = False

    def __enter__(self):
        self.active = bool(self.seconds) and hasattr(signal, "SIGALRM")
        if self.active:
            self._old = signal.signal(signal.SIGALRM, self._raise)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    @staticmethod
    def _raise(*_):
        raise Timeout()

    def __exit__(self, *exc):
        if self.active:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._old)
        return False


# ============================================================================
# Number theory
# ============================================================================

def sieve(limit):
    s = bytearray([1]) * (limit + 1)
    s[0:2] = b"\x00\x00"
    for i in range(2, math.isqrt(limit) + 1):
        if s[i]:
            s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i, v in enumerate(s) if v]


SMALL_PRIMES = sieve(2000)


def is_probable_prime(n, rounds=40):
    """Miller–Rabin with random bases; error <= 4^-rounds for composite n."""
    if n < 2:
        return False
    for q in SMALL_PRIMES:
        if n == q:
            return True
        if n % q == 0:
            return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(rounds):
        a = SYSRNG.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def is_prime(n):
    """Strong probable-prime test, cross-checked with sympy's BPSW when available."""
    r = is_probable_prime(n)
    if HAVE_SYMPY:
        r = r and bool(sympy.isprime(n))
    return r


def iroot(n, k):
    """floor(n ** (1/k)) for big ints."""
    if n < 2:
        return n
    lo, hi = 1, 1 << (n.bit_length() // k + 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid ** k <= n:
            lo = mid
        else:
            hi = mid - 1
    return lo


def pollard_brent(n, max_iter=3_000_000):
    """Return a nontrivial factor of composite n, or None if budget exhausted."""
    if n % 2 == 0:
        return 2
    y, c, m = SYSRNG.randrange(1, n), SYSRNG.randrange(1, n), 128
    g = r = q = 1
    x = ys = y
    iters = 0
    while g == 1:
        x = y
        for _ in range(r):
            y = (y * y + c) % n
        k = 0
        while k < r and g == 1:
            ys = y
            for _ in range(min(m, r - k)):
                y = (y * y + c) % n
                q = q * abs(x - y) % n
            g = math.gcd(q, n)
            k += m
            iters += m
            if iters > max_iter:
                return None
        r *= 2
    if g == n:
        while True:
            ys = (ys * ys + c) % n
            g = math.gcd(abs(x - ys), n)
            if g > 1:
                break
    return g if g != n else None


def trial_division(n, limit):
    f = {}
    for q in SMALL_PRIMES if limit <= 2000 else sieve(limit):
        if q > limit or q * q > n:
            break
        while n % q == 0:
            f[q] = f.get(q, 0) + 1
            n //= q
    return f, n


def factor(n, timeout=60):
    """
    Factor n. Returns (factors: {prime: exp}, leftover, complete).
    leftover == 1 iff complete; otherwise leftover is an unfactored composite.
    """
    if n <= 1:
        return {}, 1, True
    factors, rem = trial_division(n, 100_000)
    if rem == 1:
        return factors, 1, True
    if is_prime(rem):
        factors[rem] = factors.get(rem, 0) + 1
        return factors, 1, True

    if HAVE_SYMPY:
        try:
            with time_limit(timeout):
                f = sympy.factorint(rem)
            for q, e in f.items():
                factors[int(q)] = factors.get(int(q), 0) + int(e)
            return factors, 1, True
        except Timeout:
            return factors, rem, False

    # pure-Python fallback: perfect powers + Pollard–Brent with a budget
    deadline = time.time() + timeout if timeout else None
    stack, leftover = [rem], 1
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            factors[m] = factors.get(m, 0) + 1
            continue
        pw = None
        for k in range(2, m.bit_length() + 1):
            r = iroot(m, k)
            if r ** k == m:
                pw = (r, k)
                break
        if pw:
            stack += [pw[0]] * pw[1]
            continue
        d = None
        if deadline is None or time.time() < deadline:
            d = pollard_brent(m)
        if d is None:
            leftover *= m
            continue
        stack += [d, m // d]
    return factors, leftover, leftover == 1


def mult_order(g, m, m1_factors):
    """Order of g in (Z/mZ)^*, m prime, given the COMPLETE factorization of m-1."""
    k = m - 1
    for q in m1_factors:
        while k % q == 0 and pow(g, k // q, m) == 1:
            k //= q
    return k


def embedding_degree_bounded(p, r, bound):
    """Smallest k <= bound with p^k == 1 (mod r), or None."""
    x, pm = 1, p % r
    for k in range(1, bound + 1):
        x = x * pm % r
        if x == 1:
            return k
    return None


def rho_log2(l, aut=1):
    """log2 of expected Pollard-rho group operations for a prime-order-l subgroup.
    aut = size of the automorphism group exploited (1 plain, 2 negation map, 4 or 6 with CM)."""
    return 0.5 * (math.log2(math.pi) + math.log2(l) - math.log2(2 * aut))


def naf_weight(n):
    """Number of nonzero digits in the non-adjacent form (NAF) of n.
    A small value means n is a sum/difference of few powers of two, i.e. a
    Solinas / generalized-Mersenne prime (fast reduction by a few shifts and adds)."""
    w = 0
    while n:
        if n & 1:
            n -= 2 - (n & 3)  # subtract the NAF digit (+1 or -1)
            w += 1
        n >>= 1
    return w


def verify_x962_seed(seed_hex, p, a, b):
    """ANSI X9.62 / FIPS 186-4 Appendix A.3.3 check of a 'verifiably random' curve:
    derive r from SEED via SHA-1 and test  r * b^2 == a^3  (mod p).
    Validated against P-256's published seed (True) and against a corrupted seed
    and a wrong b (both False).
    IMPORTANT: a True result proves only that (SEED, a, b) are CONSISTENT -- i.e.
    the standard was transcribed correctly. It says NOTHING about whether SEED was
    chosen honestly: BADA55 (Bernstein et al., 2015) showed a designer can try
    many seeds until one yields a curve with a desired hidden property, and the
    seed will still 'verify'. Hence this is an INFO-level provenance note."""
    seed = bytes.fromhex(seed_hex)
    g = len(seed) * 8
    t = p.bit_length()
    s = (t - 1) // 160
    h = t - 160 * s
    H = int.from_bytes(hashlib.sha1(seed).digest(), "big")
    c0 = H & ((1 << h) - 1)          # rightmost h bits of H
    W0 = c0 & ~(1 << (h - 1))        # leftmost of those h bits set to 0
    seed_int = int.from_bytes(seed, "big")
    W = W0
    for i in range(1, s + 1):
        si = ((seed_int + i) % (1 << g)).to_bytes(len(seed), "big")
        W = (W << 160) | int.from_bytes(hashlib.sha1(si).digest(), "big")
    return (W * b * b - a * a * a) % p == 0


# ============================================================================
# Elliptic-curve arithmetic (affine; audit tool, not constant-time)
# ============================================================================

INF = None


def ec_add(P, Q, p, a):
    if P is INF:
        return Q
    if Q is INF:
        return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2:
        if (y1 + y2) % p == 0:
            return INF
        lam = (3 * x1 * x1 + a) * pow(2 * y1, -1, p) % p
    else:
        lam = (y2 - y1) * pow((x2 - x1) % p, -1, p) % p
    x3 = (lam * lam - x1 - x2) % p
    return (x3, (lam * (x1 - x3) - y1) % p)


def ec_mul(k, P, p, a):
    R = INF
    while k > 0:
        if k & 1:
            R = ec_add(R, P, p, a)
        P = ec_add(P, P, p, a)
        k >>= 1
    return R


def on_curve(P, p, a, b):
    if P is INF:
        return True
    x, y = P
    return (y * y - (x * x * x + a * x + b)) % p == 0


def legendre(x, p):
    return pow(x % p, (p - 1) // 2, p)


def sqrt_mod(n, p):
    """Tonelli–Shanks. Returns a square root of n mod p, or None."""
    n %= p
    if n == 0:
        return 0
    if legendre(n, p) != 1:
        return None
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while legendre(z, p) != p - 1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(n, q, p), pow(n, (q + 1) // 2, p)
    while t != 1:
        i, tt = 0, t
        while tt != 1:
            tt = tt * tt % p
            i += 1
        bb = pow(c, 1 << (m - i - 1), p)
        m, c, t, r = i, bb * bb % p, t * bb * bb % p, r * bb % p
    return r


def random_point(p, a, b):
    while True:
        x = SYSRNG.randrange(p)
        y = sqrt_mod((x * x * x + a * x + b) % p, p)
        if y is not None:
            return (x, y if SYSRNG.random() < 0.5 else (p - y) % p)


def find_non_square(p):
    d = 2
    while legendre(d, p) != p - 1:
        d += 1
    return d


# ============================================================================
# Small-order twist points (test vectors for point-validation, cf. Wycheproof)
# ============================================================================
#
# The quadratic twist E' of E: y^2 = x^3 + a*x + b (twist by a non-square d) is
# isomorphic over F_p to the SHORT-WEIERSTRASS curve
#     W:  Y^2 = X^3 + (a*d^2)*X + (b*d^3)
# via  (X, Y) <-> (x, y) = (X/d, Y/d^2),  which satisfies  d*y^2 = x^3 + a*x + b.
# #W = #E' = p + 1 + t, so W's group has the same small factors as the twist order.
# We build a point of small prime order on W (this is exact and needs no point
# counting) and map it back. The x-coordinate is a genuine twist x on E's x-line
# (i.e. x^3 + a*x + b is a NON-residue).
#
# Why these are useful: an implementation that scalar-multiplies a secret by an
# UNVALIDATED input point/x will, on such input, operate in the small twist
# subgroup and leak (secret mod order) via Pohlig-Hellman + CRT. A correct
# implementation MUST reject the point (fails the on-curve check). For a = 0
# curves (secp256k1) W has the same a = 0, so the full (X, Y) also serves as an
# invalid-curve vector for full-coordinate implementations (the group law ignores
# b). Vectors are for TESTING that your validation rejects them.

def generate_twist_vectors(p, a, b, twist_order, factor_bound=1_000_000, tries=2000, verify=True):
    """Return (vectors, d, aW, bW). Each vector: dict(order, x, y, wx, wy)."""
    a %= p
    b %= p
    d = find_non_square(p)
    aW, bW = a * d * d % p, b * d * d % p * d % p
    dinv = pow(d, -1, p)
    d2inv = dinv * dinv % p

    small_factors, _ = trial_division(twist_order, factor_bound)
    vectors = []
    for ell in sorted(small_factors):
        if ell == 2:
            continue  # order-2 points are the shared 2-torsion (y=0, rhs=0): not twist-distinguishing
        full_power = ell ** small_factors[ell]      # exact max power of ell dividing #E'
        cof = twist_order // full_power             # coprime to ell -> lands in ell-Sylow
        found = None
        for _ in range(tries):
            Q = ec_mul(cof, random_point(p, aW, bW), p, aW)
            if Q is INF:
                continue
            while ec_mul(ell, Q, p, aW) is not INF:  # walk down to exact order ell
                Q = ec_mul(ell, Q, p, aW)
            X, Y = Q
            x_tw, y_tw = X * dinv % p, Y * d2inv % p
            if legendre((x_tw * x_tw * x_tw + a * x_tw + b) % p, p) != p - 1:
                continue                             # degenerate (unexpected for odd ell); retry
            found = (X, Y, x_tw, y_tw)
            break
        if found is None:
            continue
        X, Y, x_tw, y_tw = found
        if verify:
            rhs = (x_tw * x_tw * x_tw + a * x_tw + b) % p
            assert legendre(rhs, p) == p - 1, "x is not a twist x-coordinate"
            assert d * y_tw % p * y_tw % p == rhs, "d*y^2 != x^3+ax+b"
            assert on_curve((X, Y), p, aW, bW), "(X,Y) not on W"
            assert (X, Y) != INF and ec_mul(ell, (X, Y), p, aW) is INF, "order != ell"
        vectors.append(dict(order=ell, x=x_tw, y=y_tw, wx=X, wy=Y))
    return vectors, d, aW, bW


def print_twist_vectors(name, p, a, b, twist_order, factor_bound):
    vecs, d, aW, bW = generate_twist_vectors(p, a, b, twist_order, factor_bound)
    print(f"\n=== Twist small-order test vectors: {name} ===")
    print(f"  non-square d = {d}")
    print(f"  auxiliary curve W (same a): Y^2 = X^3 + {hex(aW)} * X + {hex(bW)}")
    print(f"  twist point form: d*y^2 = x^3 + a*x + b   (x^3+a*x+b is a non-residue mod p)")
    print("  USE: feed x (x-only impls) or (X,Y) on W (full-point impls) to your validation /")
    print("       scalar-mult routine. Correct behaviour = REJECT (point not on the curve).")
    print("       Processing it leaks secret mod <order>. Vectors verified: on-twist + exact order.")
    if not vecs:
        print(f"  (no small twist factors below {factor_bound})")
        return
    for v in vecs:
        print(f"\n  order {v['order']}")
        print(f"    x = {hex(v['x'])}")
        print(f"    y = {hex(v['y'])}   (d*y^2 = x^3 + a*x + b)")
        print(f"    on W: X = {hex(v['wx'])}")
        print(f"          Y = {hex(v['wy'])}")


# ============================================================================
# Presets
# ============================================================================

PRESETS = {
    "secp256k1": dict(
        p=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F,
        a=0, b=7,
        n=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141,
        gx=0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
        gy=0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
        h=1),
    "p256": dict(
        p=0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF,
        a=0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFC,
        b=0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B,
        n=0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551,
        gx=0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296,
        gy=0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5,
        h=1,
        seed="c49d360886e704936a6678e1139d26b7819f7e90"),  # published SEED (FIPS 186-4 D.1.2.3 / SEC 2)
    # Demo of a WEAK curve: y^2 = x^3 + x over p = 3 (mod 4) is supersingular,
    # #E = p+1 = 4*q with q prime. Passes "n prime, n*G = O" but MOV-breakable (k = 2).
    "weak_ss": dict(
        p=0x1000000CA269E0ECB, a=1, b=0,
        n=0x4000003289A783B3,
        gx=0x766365AA250A827E, gy=0x8E4E49E3CDEAB4FF,
        h=4),
}


# ============================================================================
# Report plumbing
# ============================================================================

STATUS_ORDER = {"FAIL": 3, "WARN": 2, "PASS": 1, "INFO": 0}


class Report:
    def __init__(self, name):
        self.name = name
        self.items = []
        self.data = {}

    def add(self, sid, title, status, *lines, **data):
        self.items.append(dict(id=sid, title=title, status=status, lines=[str(l) for l in lines]))
        for k, v in data.items():
            self.data[k] = v

    def worst(self):
        return max((i["status"] for i in self.items), key=lambda s: STATUS_ORDER[s], default="INFO")

    def print(self):
        print(f"=== EC audit: {self.name} ===")
        print(f"    sympy: {'yes' if HAVE_SYMPY else 'no (pure-Python fallback)'}\n")
        section = None
        for it in self.items:
            sec = it["id"][0]
            if sec != section:
                section = sec
                print(SECTION_TITLES.get(sec, ""))
            print(f"  [{it['status']:4s}] {it['id']:>4s}  {it['title']}")
            for l in it["lines"]:
                print(f"               {l}")
        fails = [i for i in self.items if i["status"] == "FAIL"]
        warns = [i for i in self.items if i["status"] == "WARN"]
        print("\n=== Summary ===")
        print(f"  verdict: {self.worst()}   ({len(fails)} FAIL, {len(warns)} WARN)")
        for i in fails + warns:
            print(f"    {i['status']} {i['id']}: {i['title']}")
        print("  Not covered here: rigidity (how the parameters were generated), side channels,")
        print("  nonce generation, point validation in YOUR implementation — those are where real bugs live.")
        print("  (The twist test vectors below are inputs you can feed your point validation to test it.)")

    def to_json(self):
        return json.dumps(dict(curve=self.name, verdict=self.worst(), items=self.items,
                               data={k: (hex(v) if isinstance(v, int) and abs(v) > 2**63 else v)
                                     for k, v in self.data.items()}), indent=2)


SECTION_TITLES = {
    "A": "--- A. Field and equation ---",
    "B": "--- B. Generator, subgroup order, group order ---",
    "C": "--- C. Transfer attacks (Smart/SSSA, MOV/Frey-Rück) ---",
    "D": "--- D. Generic (Pollard rho) security ---",
    "E": "--- E. CM discriminant / special structure ---",
    "F": "--- F. Quadratic twist ---",
}


def fmt_factors(f, leftover=1):
    parts = [f"{q}^{e}" if e > 1 else f"{q}" for q, e in sorted(f.items())]
    if leftover != 1:
        parts.append(f"C{leftover.bit_length()}")
    return " * ".join(parts) if parts else "1"


def fmt_factor_bits(f, leftover=1):
    parts = [f"({q.bit_length()}b)^{e}" if e > 1 else f"({q.bit_length()}b)" for q, e in sorted(f.items())]
    if leftover != 1:
        parts.append(f"[unfactored {leftover.bit_length()}b]")
    return " * ".join(parts) if parts else "1"


# ============================================================================
# The audit
# ============================================================================

def audit(name, p, a, b, n, gx, gy, h_claimed=None, *, fast=False, factor_timeout=60,
          mov_bound=100_000, disc_bound=1_000_000, n_random_points=8, seed=None):
    R = Report(name)
    G = (gx % p, gy % p)
    a %= p
    b %= p

    # ---------------- A. field & equation ----------------
    p_prime = is_prime(p)
    R.add("A1", f"p is prime ({p.bit_length()} bit)", "PASS" if p_prime and p > 3 else "FAIL",
          f"p = {hex(p)}", p=p)
    if not p_prime:
        R.add("A1b", "field is not prime — nothing else is meaningful", "FAIL")
        return R

    disc = (4 * a * a * a + 27 * b * b) % p
    R.add("A2", "non-singular: 4a^3 + 27b^2 != 0 (mod p)", "PASS" if disc else "FAIL",
          f"a = {hex(a)}", f"b = {hex(b)}")
    if not disc:
        return R

    j = 1728 * 4 * a * a * a * pow(disc, -1, p) % p
    aut = 2  # negation map always
    special = None
    if j == 0:
        special = ("j = 0: ordinary with CM by Z[zeta_3] (D = -3) when p = 1 (mod 3) -> extra automorphism "
                   "(x,y)->(beta*x, y); supersingular when p = 2 (mod 3)")
        if p % 3 == 1:
            aut = 6
    elif j == 1728:
        special = ("j = 1728: ordinary with CM by Z[i] (D = -4) when p = 1 (mod 4) -> extra automorphism "
                   "(x,y)->(-x, i*y); supersingular when p = 3 (mod 4)")
        if p % 4 == 1:
            aut = 4
    R.add("A3", f"j-invariant = {hex(j)}", "INFO" if special else "PASS",
          *( [special, f"|Aut(E)| usable by rho = {aut}"] if special else ["generic j; |Aut| = 2 (negation only)"] ),
          j=j, aut=aut)

    # A4: special form of the prime p. NOT a security finding -- no known ECDLP
    # attack exploits the form of p over a prime field -- but it is exactly the
    # structure that makes fast modular reduction implementable (single-word
    # "fold" for Crandall, shift-and-add for Solinas), so it is worth surfacing.
    kbits = p.bit_length()
    c = (1 << kbits) - p
    w = naf_weight(p)
    forms = []
    # "Crandall" only counts if c is small RELATIVE to p (<= half its bits) AND
    # fits one machine word: an absolute 64-bit cap alone wrongly flagged a
    # 65-bit test prime with c ~ p/2 as pseudo-Mersenne (fold reduction would
    # barely shrink anything there). Both conditions are needed.
    is_crandall = c.bit_length() <= min(64, kbits // 2)
    if is_crandall:
        forms.append(f"Crandall / pseudo-Mersenne: p = 2^{kbits} - c,  c = {c} ({c.bit_length()} bit, fits one word)")
        forms.append("  -> reduction = one single-word multiply-and-fold per 2^k overflow")
    if w <= 8:
        forms.append(f"Solinas / generalized Mersenne: NAF weight {w} (p = +/- sum of {w} powers of two)")
        forms.append("  -> reduction = a fixed pattern of shifts and adds, no multiply")
    if forms:
        R.add("A4", "prime p has a special (fast-reduction) form", "INFO", *forms,
              crandall_c=c if is_crandall else None, naf_weight=w)
    else:
        R.add("A4", f"prime p is generic (no Crandall/Solinas form; NAF weight {w})", "INFO",
              "reduction needs a general method (Montgomery / Barrett)", naf_weight=w)

    # A5: rigidity PROXIES. Rigidity proper -- whether the generation process left
    # the designer room to hide a weakness -- is a property of the PROCESS, not of
    # (p, a, b, n, G), and is not computable. What IS computable: how much apparent
    # freedom the parameters show, and (if a seed is given) whether the standard's
    # own seed->b derivation is at least self-consistent. Always INFO, never a
    # verdict: a "seed verifies" result must not read as "curve is trustworthy"
    # (see verify_x962_seed docstring / BADA55).
    a_std = "0" if a == 0 else ("-3" if a == (p - 3) % p else None)
    b_tiny = b.bit_length() <= 32
    lines = []
    if a_std is not None:
        lines.append(f"a = {a_std}: a standard convention (a=0: Koblitz/GLV-friendly, a=-3: fast Jacobian doubling), not a free choice")
    if b_tiny:
        lines.append(f"b = {b}: tiny -- consistent with 'smallest b giving prime order' (no room to hide anything in b)")
    if seed is not None:
        ok = verify_x962_seed(seed, p, a, b)
        lines.append(f"seed = {seed}")
        lines.append(f"X9.62 / FIPS 186-4 A.3.3 derivation r*b^2 == a^3 (mod p): {'CONSISTENT' if ok else 'INCONSISTENT -- seed does not produce this b!'}")
        if ok:
            lines.append("  this proves only that the standard was transcribed correctly, NOT that the seed was chosen")
            lines.append("  honestly: BADA55 (Bernstein et al. 2015) shows 'verifiably random' seeds can be searched for a")
            lines.append("  hidden weakness and still verify. Seed provenance itself is unexplained for the NIST curves.")
    if a_std is not None and b_tiny:
        cls = "minimal"
        head = "rigidity class: MINIMAL parameters (strongest computable proxy: ~0 bits of designer freedom)"
    elif seed is not None:
        cls = "seeded"
        head = "rigidity class: SEEDED ('verifiably random' b) -- see BADA55 caveat below"
    else:
        cls = "unexplained"
        why = []
        if a_std is None:
            why.append(f"a = {a} is neither 0 nor -3 (a genuine free choice)")
        if not b_tiny:
            why.append(f"b is large ({b.bit_length()} bit) with no seed supplied")
        head = "rigidity class: UNEXPLAINED -- " + "; ".join(why) + " -- provenance not assessable from parameters"
        lines.append("if a generation seed exists for this curve, pass it with --seed to check consistency")
    R.add("A5", head, "INFO", *lines, rigidity_class=cls,
          seed_consistent=(verify_x962_seed(seed, p, a, b) if seed is not None else None))

    # ---------------- B. generator & order ----------------
    ok_G = G is not INF and 0 <= G[0] < p and 0 <= G[1] < p and on_curve(G, p, a, b)
    R.add("B1", "G is on the curve and G != O", "PASS" if ok_G else "FAIL",
          f"Gx = {hex(G[0])}", f"Gy = {hex(G[1])}")
    if not ok_G:
        return R

    n_prime = is_prime(n)
    R.add("B2", f"n is prime ({n.bit_length()} bit)  [MR x40 random bases"
          + (" + BPSW]" if HAVE_SYMPY else "]"),
          "PASS" if n_prime else "FAIL", f"n = {hex(n)}", n=n)

    nG = ec_mul(n, G, p, a)
    R.add("B3", "n * G == O", "PASS" if nG is INF else "FAIL",
          "with n prime this proves ord(G) = n exactly" if (nG is INF and n_prime) else "")
    if nG is not INF or not n_prime:
        R.add("B4", "group order cannot be established from these parameters", "FAIL")
        return R

    # Hasse: #E in [p+1-2sqrt(p), p+1+2sqrt(p)], and #E is a multiple of ord(G) = n
    two_sqrt = math.isqrt(4 * p)
    lo, hi = p + 1 - two_sqrt, p + 1 + two_sqrt + 1
    h_lo, h_hi = -(-lo // n), hi // n
    cands = list(range(max(h_lo, 1), h_hi + 1))
    if len(cands) == 1:
        h = cands[0]
        R.add("B4", f"group order PROVEN: #E = h*n with unique h = {h} in the Hasse interval", "PASS",
              f"Hasse interval [p+1-2sqrt(p), p+1+2sqrt(p)] has width 4sqrt(p) ~ 2^{(4*p).bit_length()/2:.0f};",
              f"multiples of n in it: only {h}*n. (Original tool ASSUMED h={h_claimed}.)", h=h)
    else:
        h = h_claimed if h_claimed in cands else None
        R.add("B4", f"group order NOT unique from n alone: candidate h in {cands[:8]}{'...' if len(cands) > 8 else ''}",
              "WARN" if h else "FAIL",
              "n is too small relative to p to pin #E down; need Schoof/SEA point counting (PARI ellcard / Sage)"
              if not h else f"using claimed h = {h} (in candidate set) — NOT proven")
        if not h:
            return R
    if h_claimed is not None and h_claimed != h:
        R.add("B4b", f"claimed cofactor h = {h_claimed} contradicts proven h = {h}", "FAIL")

    order_E = h * n
    t = p + 1 - order_E
    R.data.update(order_E=order_E, trace=t)

    # cross-check with random points: (h*n)*Q must be O for every Q on E
    bad = 0
    for _ in range(n_random_points):
        Q = random_point(p, a, b)
        if ec_mul(order_E, Q, p, a) is not INF:
            bad += 1
    R.add("B5", f"random-point cross-check: (h*n)*Q == O for {n_random_points} random Q", "PASS" if bad == 0 else "FAIL",
          f"{bad} failures" if bad else "consistent with #E = h*n (independent of G)")

    R.add("B6", f"trace t = p + 1 - #E = {hex(t)}", "PASS" if abs(t) <= two_sqrt + 1 else "FAIL",
          f"|t| = 2^{math.log2(abs(t)) if t else 0:.1f}  (Hasse bound 2sqrt(p) = 2^{math.log2(two_sqrt):.1f})",
          f"cofactor h = {h}")

    # B7: generator PROVENANCE (INFO only). A "nothing-up-my-sleeve" generator
    # should look like a random point: x(G/k) and x(k*G) should all be full-size
    # field elements. If some x(G/k) is far smaller than log2(p) bits, G was
    # almost certainly CONSTRUCTED as k*H from a 'small' H -- a fact about how G
    # was chosen, not a weakness: in a prime-order group every non-identity
    # point is a generator, DLP hardness is a property of the GROUP not the base
    # point, and anyone can compute H = G/k themselves (there is no secret).
    # Known instance: secp256k1's G/2 has a 166-bit x (chance: 2^-90); the
    # structure stops there (G/4, G/8 look random). Same spirit as A5 (rigidity
    # of the curve parameters), applied to the base point.
    logp = math.log2(p)
    slack = 40  # flag only if P(chance) <= 2^-40
    anomalies = []
    for k in range(2, 9):
        kinv = pow(k, -1, n)
        for label, Q in ((f"G/{k}", ec_mul(kinv, (gx, gy), p, a)), (f"{k}*G", ec_mul(k, (gx, gy), p, a))):
            if Q is INF:
                continue
            bl = Q[0].bit_length()
            if bl <= logp - slack:
                anomalies.append((label, Q[0], bl))
    gbits = gx.bit_length()
    if gbits <= logp - slack:
        anomalies.insert(0, ("G itself", gx, gbits))
    if anomalies:
        lines = []
        for label, x, bl in anomalies:
            lines.append(f"x({label}) = {hex(x)}  ({bl} bit vs ~{logp:.0f} expected; chance 2^{bl - logp:.0f})")
        lines.append("=> G was evidently CONSTRUCTED, not sampled at random (provenance is not 'nothing-up-my-sleeve').")
        lines.append("   NOT a weakness: any point generates a prime-order group and DLP hardness does not depend on")
        lines.append("   the base point; the 'small' pre-image is public (computed right here). Rigidity note only.")
        R.add("B7", f"generator provenance: {len(anomalies)} small-x relative(s) of G found", "INFO", *lines,
              generator_anomalies=[(l, x, bl) for l, x, bl in anomalies])
    else:
        R.add("B7", "generator provenance: G, G/k, k*G (k<=8) all have full-size x -- consistent with a random point", "INFO",
              "no evidence G was constructed from a small pre-image (checked small multiples and divisions)")

    # ---------------- C. transfer attacks ----------------
    # Note: a separate "n != p" test is NOT needed. If n == p then #E = h*n = h*p,
    # but Hasse gives #E < 2p, forcing h = 1 and #E == p -- i.e. exactly the
    # anomalous case tested here. So n == p  <=>  #E == p; C1 covers both.
    R.add("C1", "anomalous (#E == p) -> Smart/SSSA attack, DLP in F_p^+", "FAIL" if order_E == p else "PASS",
          "trace t = 1" if order_E == p else f"#E != p (t = {'0' if t == 0 else '2^%.1f' % math.log2(abs(t))})",
          f"(also implies n != p: n == p would force h = 1 and #E == p by Hasse)")
    R.add("C2", "supersingular (t == 0 mod p, i.e. t == 0 for p > 3) -> MOV, DLP in F_{p^2}",
          "FAIL" if t % p == 0 else "PASS")

    k_b = embedding_degree_bounded(p, n, mov_bound)
    if k_b is not None:
        R.add("C3", f"embedding degree k = {k_b}  (n | p^k - 1) -> MOV/Frey-Rück transfer to F_{{p^{k_b}}}",
              "FAIL" if k_b <= 20 else "WARN",
              f"DLP in F_{{p^{k_b}}} of size 2^{k_b * p.bit_length()} bits is subexponential (index calculus)")
    else:
        R.add("C3", f"no embedding degree k <= {mov_bound}", "PASS",
              f"MOV/Frey-Rück impractical: would need F_{{p^k}} with k > {mov_bound}")

    if not fast:
        f_n1, left, complete = factor(n - 1, factor_timeout)
        if complete:
            k_exact = mult_order(p, n, f_n1)
            ratio = (n - 1) // k_exact
            R.add("C4", f"exact embedding degree: k = (n-1)/{ratio}  (log2 k = {math.log2(k_exact):.1f})",
                  "PASS" if ratio <= 100 else "WARN",
                  f"n-1 = {fmt_factors(f_n1)}",
                  f"SafeCurves criterion k >= (n-1)/100: {'OK' if ratio <= 100 else 'NOT met'}")
        else:
            R.add("C4", "exact embedding degree: not computed (n-1 not fully factored within timeout)", "INFO",
                  f"n-1 = {fmt_factor_bits(f_n1, left)}", "bounded search in C3 still rules out small k")

    # C5: Weil descent / GHS (Gaudry-Hess-Smart) and related index-calculus
    # via descent apply only to curves over EXTENSION fields F_{p^m}, m > 1
    # (they map the ECDLP to a hyperelliptic-curve DLP over the subfield).
    # This tool audits curves over a prime field F_p, so they do not apply.
    # Stated explicitly so a reader does not assume the check was forgotten.
    R.add("C5", "Weil descent / GHS: not applicable (curve is over a prime field F_p)", "PASS",
          "these attacks need an extension field F_{p^m}, m > 1; for prime fields",
          "no index-calculus method beats generic Pollard rho (section D)")

    # ---------------- D. rho security ----------------
    l = n  # n is prime
    R.add("D1", f"largest prime factor of subgroup order: {l.bit_length()} bit (n itself, prime)", "INFO")
    r_plain, r_neg, r_aut = rho_log2(l, 1), rho_log2(l, 2), rho_log2(l, aut)
    st = "PASS" if r_neg >= 100 else "FAIL"
    R.add("D2", f"Pollard rho cost: 2^{r_neg:.1f} (with negation map)", st,
          f"plain 2^{r_plain:.1f};  negation map 2^{r_neg:.1f}"
          + (f";  with |Aut| = {aut} automorphisms 2^{r_aut:.1f} (speed-up sqrt({aut // 2}))" if aut > 2 else ""),
          "SafeCurves threshold: 2^100" + ("" if st == "PASS" else "  <-- BELOW"),
          rho_negation=r_neg, rho_aut=r_aut)

    # ---------------- E. CM discriminant ----------------
    delta = t * t - 4 * p
    D_found = v_found = None
    if t == 0:
        D_ss = -p if p % 4 == 3 else -4 * p
        R.add("E1", f"supersingular: Frobenius pi^2 = -p, CM discriminant D = {'-p' if p % 4 == 3 else '-4p'} ({D_ss.bit_length()} bit)",
              "INFO", "endomorphism ring over F_p is an order in Q(sqrt(-p)); the real problem is C2/C3 above",
              cm_discriminant=D_ss)
    for D in ([] if t == 0 else range(-3, -disc_bound - 1, -1)):
        if D % 4 not in (0, 1) or delta % D:
            continue
        v2 = delta // D
        v = math.isqrt(v2)
        if v * v == v2:
            D_found, v_found = D, v
            break
    if D_found is not None:
        lines = [f"t^2 - 4p = D * v^2 with D = {D_found}, v = 2^{math.log2(v_found):.1f}",
                 "small |D| => the curve has an efficiently computable endomorphism (GLV); no known attack",
                 "exploits this for well-chosen curves, but SafeCurves flags it (|D| < 2^100) as 'special'/rigidity."]
        R.add("E1", f"CM discriminant D = {D_found} (|D| < {disc_bound})", "WARN", *lines, cm_discriminant=D_found)
        if j == 0 and D_found == -3 and p % 3 == 1 and n % 3 == 1:
            def cube_root_of_unity(m):
                g = 2
                while True:
                    w = pow(g, (m - 1) // 3, m)
                    if w != 1:
                        return w
                    g += 1
            beta = cube_root_of_unity(p)
            lam = cube_root_of_unity(n)
            target = (beta * G[0] % p, G[1])
            if ec_mul(lam, G, p, a) != target:
                lam = lam * lam % n
            ok = ec_mul(lam, G, p, a) == target
            R.add("E2", "GLV endomorphism verified: lambda*G == (beta*Gx, Gy)" if ok else "GLV constants NOT matched",
                  "INFO" if ok else "WARN",
                  f"beta   = {hex(beta)}", f"lambda = {hex(lam)}",
                  "(this is exactly the structure secp256k1 uses to speed up scalar multiplication)",
                  glv_beta=beta, glv_lambda=lam)
    elif t != 0:
        R.add("E1", f"no CM discriminant with |D| <= {disc_bound} found", "PASS",
              "t^2 - 4p has no small fundamental-discriminant factor => no cheap extra endomorphisms")

    # ---------------- F. twist ----------------
    tw = p + 1 + t
    f_tw, left_tw, complete_tw = factor(tw, factor_timeout)
    R.add("F1", f"quadratic twist order #E' = p + 1 + t  ({tw.bit_length()} bit)", "INFO",
          f"#E' = {hex(tw)}",
          f"factorization: {fmt_factors(f_tw, left_tw)}",
          f"bit sizes:     {fmt_factor_bits(f_tw, left_tw)}", twist_order=tw)

    small = {q: e for q, e in f_tw.items() if q.bit_length() <= 60}
    leak = sum(e * math.log2(q) for q, e in small.items())
    if small:
        R.add("F2", f"twist has small subgroups: {fmt_factors(small)}  (~{leak:.1f} bits)", "WARN",
              "invalid-curve / twist attack: if an implementation multiplies its secret by an UNVALIDATED",
              f"point, an attacker feeding twist points of these orders learns the secret mod {fmt_factors(small)}",
              f"(~{leak:.1f} bits via CRT). MITIGATION: always check that received points satisfy y^2 = x^3 + ax + b.")
    else:
        R.add("F2", "twist has no small subgroups (<= 60 bit)", "PASS")

    big = max(f_tw) if f_tw else 1
    if complete_tw:
        tr_neg, tr_aut = rho_log2(big, 2), rho_log2(big, aut)
        st = "PASS" if tr_neg >= 100 else "WARN"
        R.add("F3", f"twist rho security: 2^{tr_neg:.1f} (largest twist prime {big.bit_length()} bit)", st,
              f"negation map 2^{tr_neg:.1f}" + (f";  with |Aut| = {aut}: 2^{tr_aut:.1f}" if aut > 2 else ""),
              f"vs main group 2^{r_neg:.1f}" + ("" if st == "PASS" else "  <-- below 2^100"),
              twist_rho=tr_neg)
        if big > 1 and big.bit_length() > 60:
            kb = embedding_degree_bounded(p, big, mov_bound)
            if kb is not None:
                R.add("F4", f"twist embedding degree k = {kb}", "INFO")
            elif not fast:
                f_b1, lb, cb = factor(big - 1, factor_timeout)
                if cb:
                    ke = mult_order(p, big, f_b1)
                    R.add("F4", f"twist embedding degree: exact k = (r-1)/{(big - 1) // ke} for r = largest twist prime",
                          "INFO", f"r-1 = {fmt_factor_bits(f_b1)}")
                else:
                    R.add("F4", f"twist embedding degree: no k <= {mov_bound} (exact not computed)", "INFO")
            else:
                R.add("F4", f"twist embedding degree: no k <= {mov_bound}", "INFO")
    else:
        R.add("F3", "twist rho security: UNDETERMINED (twist order not fully factored within timeout)", "INFO",
              f"largest KNOWN prime factor {big.bit_length()} bit; unfactored composite of {left_tw.bit_length()} bit remains",
              "increase --factor-timeout or install sympy")

    return R


# ============================================================================
# CLI
# ============================================================================

def parse_int(s):
    return int(s, 0)


def main():
    ap = argparse.ArgumentParser(description="Elliptic-curve security audit (short Weierstrass over F_p)")
    ap.add_argument("--curve", choices=sorted(PRESETS), default="secp256k1")
    for name in ("p", "a", "b", "n", "gx", "gy", "h"):
        ap.add_argument(f"--{name}", type=parse_int)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", default=None,
                    help="hex SEED of an X9.62/FIPS 186-4 'verifiably random' curve; checks seed->b consistency (A5)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fast", action="store_true", help="skip factoring n-1 (bounded embedding degree only)")
    ap.add_argument("--factor-timeout", type=float, default=60.0)
    ap.add_argument("--mov-bound", type=int, default=100_000)
    ap.add_argument("--disc-bound", type=int, default=1_000_000)
    ap.add_argument("--no-sympy", action="store_true")
    ap.add_argument("--no-twist-vectors", dest="twist_vectors", action="store_false",
                    help="omit the small-order twist test vectors (emitted by default)")
    ap.add_argument("--twist-vectors", dest="twist_vectors", action="store_true",
                    help=argparse.SUPPRESS)  # kept for backward compatibility; now the default
    ap.set_defaults(twist_vectors=True)
    ap.add_argument("--twist-factor-bound", type=int, default=1_000_000,
                    help="search twist small factors up to this bound (default 10^6)")
    args = ap.parse_args()

    global HAVE_SYMPY
    if args.no_sympy:
        HAVE_SYMPY = False

    if args.p is not None:
        need = ("p", "a", "b", "n", "gx", "gy")
        missing = [k for k in need if getattr(args, k) is None]
        if missing:
            ap.error("custom curve needs " + ", ".join("--" + m for m in missing))
        params = dict(p=args.p, a=args.a, b=args.b, n=args.n, gx=args.gx, gy=args.gy, h=args.h)
        name = args.name or "custom"
    else:
        params = dict(PRESETS[args.curve])
        name = args.name or args.curve

    t0 = time.time()
    seed = args.seed if args.seed is not None else params.get("seed")
    rep = audit(name, params["p"], params["a"], params["b"], params["n"], params["gx"], params["gy"],
                params.get("h"), fast=args.fast, factor_timeout=args.factor_timeout,
                mov_bound=args.mov_bound, disc_bound=args.disc_bound, seed=seed)
    if args.json:
        out = json.loads(rep.to_json())
        if args.twist_vectors and "twist_order" in rep.data:
            vecs, d, aW, bW = generate_twist_vectors(
                params["p"], params["a"], params["b"], rep.data["twist_order"], args.twist_factor_bound)
            out["twist_vectors"] = dict(
                non_square_d=d, W_a=hex(aW), W_b=hex(bW),
                note="feed x (x-only) or (wx,wy) on W (full-point) to your validation; correct = REJECT",
                vectors=[dict(order=v["order"], x=hex(v["x"]), y=hex(v["y"]),
                              wx=hex(v["wx"]), wy=hex(v["wy"])) for v in vecs])
        print(json.dumps(out, indent=2))
    else:
        rep.print()
        if args.twist_vectors:
            if "twist_order" in rep.data:
                print_twist_vectors(name, params["p"], params["a"], params["b"],
                                    rep.data["twist_order"], args.twist_factor_bound)
            else:
                print("\n(twist order not established; cannot generate twist vectors)")
        print(f"\n  ({time.time() - t0:.1f} s)")
    return 0 if rep.worst() != "FAIL" else 2


if __name__ == "__main__":
    sys.exit(main())
