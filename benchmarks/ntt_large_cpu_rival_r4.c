/* Multi-pass BabyBear NTT on the CPU, **radix-4**: the strong version of the rival.
 *
 * `ntt_large_cpu_rival.c` is the same algorithm at radix 2. Fusing two stages per pass halves the
 * memory traffic and measures 1.48x faster on one thread (48.7 ms against 71.9 at 2^20), which
 * moves the device's `vs 16P` from 2.9-6.2x to 2.9-4.0x. That is the honest denominator.
 *
 * Same algorithm (Stockham DIF, one pass per stage, pre-twiddled for the negacyclic wrap), int64
 * arithmetic with a `% q`, OpenMP over butterflies. Compile:
 *
 *     gcc -O3 -march=native -fopenmp -o ntt_cpu ntt_large_cpu_rival.c
 *     OMP_NUM_THREADS=1  ./ntt_cpu 20
 *     OMP_NUM_THREADS=16 ./ntt_cpu 20
 *
 * This is a *straightforward* implementation, not a tuned one: no blocked layout, no SIMD
 * intrinsics (the compiler may vectorise the inner loop, and on this machine it does not do
 * it well because of the `%`). Treat it the way `poseidon2-cpu-rival.c` is treated: a conservative
 * rival that makes the GPU number meaningful rather than flattering.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define Q 2013265921LL

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static int64_t mulmod(int64_t a, int64_t b) { return (int64_t)((__int128)a * b % Q); }

static int64_t powmod(int64_t b, int64_t e) {
    int64_t r = 1;
    b %= Q;
    while (e) {
        if (e & 1) r = mulmod(r, b);
        b = mulmod(b, b);
        e >>= 1;
    }
    return r;
}

/* One negacyclic transform in place across `x`/`y`: pre-twiddle, radix-4 passes, then a radix-2
 * stage when `log2(n)` is odd. Split out of `main` so repetitions can be timed as a unit. */
static void one_transform(int64_t *x, int64_t *y, long n, int64_t psi, const int64_t *pw) {
#pragma omp parallel for
    for (long j = 0; j < n; j++) x[j] = mulmod(x[j], pw[j]);
    long m = n;
    int64_t *src = x, *dst = y, *tmp;
    while (m >= 4) {
        long quarter = m / 4;
        long blocks = n / m;
        int64_t omega = mulmod(psi, psi);
        int64_t step = powmod(omega, n / m);
        int64_t step2 = powmod(omega, 2 * n / m);
#pragma omp parallel for schedule(static)
        for (long b = 0; b < blocks; b++) {
            int64_t w1a = 1, w1b = powmod(step, quarter), w2 = 1;
            int64_t t1a = step, t1b = step, t2 = step2;
            for (long j = 0; j < quarter; j++) {
                int64_t a0 = src[b * m + j];
                int64_t a1 = src[b * m + quarter + j];
                int64_t a2 = src[b * m + 2 * quarter + j];
                int64_t a3 = src[b * m + 3 * quarter + j];
                int64_t u = a0 + a2; if (u >= Q) u -= Q;
                int64_t d = a0 - a2; if (d < 0) d += Q;
                d = mulmod(d, w1a);
                int64_t u2 = a1 + a3; if (u2 >= Q) u2 -= Q;
                int64_t d2 = a1 - a3; if (d2 < 0) d2 += Q;
                d2 = mulmod(d2, w1b);
                int64_t s0 = u + u2; if (s0 >= Q) s0 -= Q;
                int64_t s1 = u - u2; if (s1 < 0) s1 += Q;
                int64_t s2 = d + d2; if (s2 >= Q) s2 -= Q;
                int64_t s3 = d - d2; if (s3 < 0) s3 += Q;
                dst[b * m + j] = s0;
                dst[b * m + quarter + j] = mulmod(s1, w2);
                dst[b * m + 2 * quarter + j] = s2;
                dst[b * m + 3 * quarter + j] = mulmod(s3, w2);
                w1a = mulmod(w1a, t1a);
                w1b = mulmod(w1b, t1b);
                w2 = mulmod(w2, t2);
            }
        }
        tmp = src; src = dst; dst = tmp;
        m /= 4;
    }
    while (m >= 2) {
        long half = m / 2;
        long blocks = n / m;
        int64_t omega = mulmod(psi, psi);
        int64_t step = powmod(omega, n / m);
#pragma omp parallel for schedule(static)
        for (long b = 0; b < blocks; b++) {
            int64_t w = 1;
            for (long j = 0; j < half; j++) {
                int64_t u = src[b * m + j], v = src[b * m + half + j];
                int64_t sum = u + v; if (sum >= Q) sum -= Q;
                int64_t dif = u - v; if (dif < 0) dif += Q;
                dst[b * m + j] = sum;
                dst[b * m + half + j] = mulmod(dif, w);
                w = mulmod(w, step);
            }
        }
        tmp = src; src = dst; dst = tmp;
        m = half;
    }
}

int main(int argc, char **argv) {
    int logn = argc > 1 ? atoi(argv[1]) : 20;
    long n = 1L << logn;
    int64_t psi = 0;
    for (int g = 2; g < 256; g++) {
        int64_t z = powmod(g, (Q - 1) / (2 * n));
        if (powmod(z, n) == Q - 1) { psi = z; break; }
    }
    if (!psi) { fprintf(stderr, "no root\n"); return 1; }

    int64_t *x = aligned_alloc(64, sizeof(int64_t) * n);
    int64_t *y = aligned_alloc(64, sizeof(int64_t) * n);
    uint64_t s = 12345;
    for (long i = 0; i < n; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        x[i] = (int64_t)((s >> 33) % (uint64_t)Q);
    }
    /* negacyclic pre-twiddle, with the powers table built once outside the timing (the device
     * caches its own table, so charging the CPU for it would be a rigged comparison) */
    int64_t *pw = aligned_alloc(64, sizeof(int64_t) * n);
    pw[0] = 1;
    for (long j = 1; j < n; j++) pw[j] = mulmod(pw[j - 1], psi);

    long reps = argc > 2 ? atol(argv[2]) : 1;
    if (reps > 1) {
        /* One long window for external barrier timing: a caller needs ~0.3 s so that process
         * start-up and cold caches do not dominate, and this binary cannot be asked for a longer
         * window any other way. */
        double t0 = now();
        for (long r = 0; r < reps; r++) one_transform(x, y, n, psi, pw);
        double dt = now() - t0;
        printf("{\"logn\":%d,\"reps\":%ld,\"ms_total\":%.3f,\"gbs\":%.1f}\n",
               logn, reps, dt * 1e3,
               (double)n * 4 * 2 * logn * (double)reps / dt / 1e9);
        free(x);
        free(y);
        free(pw);
        return 0;
    }
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now();
        one_transform(x, y, n, psi, pw);
        double dt = now() - t0;
        if (dt < best) best = dt;
    }
    printf("{\"logn\":%d,\"ms\":%.3f,\"gbs\":%.1f}\n", logn, best * 1e3, (double)n * 4 * 2 * logn / best / 1e9);
    free(x);
    free(y);
    free(pw);
    return 0;
}
