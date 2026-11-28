/* Native C CPU rival driver for SM3, Ascon (SP 800-232 & v1.2), X25519, Ed25519, MAYO, and UOV.
 *
 * Designed for barrier-synchronized 1T and 16P benchmarking from `benchmarks/bench_new_algos.py`.
 * Build:
 *   gcc -O3 -march=native -Ithird_party/liboqs/build-mq/include \
 *       -o benchmarks/new_algos_cpu benchmarks/new_algos_cpu_rival.c \
 *       -Lthird_party/liboqs/build-mq/lib -Wl,-rpath,'$ORIGIN/../third_party/liboqs/build-mq/lib' \
 *       -loqs -lcrypto
 */
#include <openssl/evp.h>
#include <oqs/oqs.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static inline uint64_t rotr64(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

static inline uint64_t load64_le(const uint8_t *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}
static inline void store64_le(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }
static inline uint64_t bswap64(uint64_t x) { return __builtin_bswap64(x); }

static inline void ascon_perm(uint64_t s[5], int rounds) {
    uint64_t x0 = s[0], x1 = s[1], x2 = s[2], x3 = s[3], x4 = s[4];
    for (int r = 12 - rounds; r < 12; r++) {
        x2 ^= (uint64_t)(((0xF - r) << 4) | r);
        x0 ^= x4; x4 ^= x3; x2 ^= x1;
        uint64_t t0 = (~x0) & x1, t1 = (~x1) & x2, t2 = (~x2) & x3, t3 = (~x3) & x4, t4 = (~x4) & x0;
        x0 ^= t1; x1 ^= t2; x2 ^= t3; x3 ^= t4; x4 ^= t0;
        x1 ^= x0; x0 ^= x4; x3 ^= x2; x2 = ~x2;
        x0 ^= rotr64(x0, 19) ^ rotr64(x0, 28);
        x1 ^= rotr64(x1, 61) ^ rotr64(x1, 39);
        x2 ^= rotr64(x2,  1) ^ rotr64(x2,  6);
        x3 ^= rotr64(x3, 10) ^ rotr64(x3, 17);
        x4 ^= rotr64(x4,  7) ^ rotr64(x4, 41);
    }
    s[0] = x0; s[1] = x1; s[2] = x2; s[3] = x3; s[4] = x4;
}

static void ascon_hash256_one(const uint8_t *msg, size_t len, uint8_t out[32]) {
    uint64_t s[5] = {0x0000080100cc0002ULL, 0, 0, 0, 0};
    ascon_perm(s, 12);
    size_t off = 0;
    while (off + 8 <= len) {
        s[0] ^= load64_le(msg + off);
        ascon_perm(s, 12);
        off += 8;
    }
    uint8_t last[8] = {0};
    size_t rem = len - off;
    if (rem) memcpy(last, msg + off, rem);
    last[rem] = 0x01;
    s[0] ^= load64_le(last);
    ascon_perm(s, 12);
    for (int i = 0; i < 4; i++) {
        store64_le(out + 8 * i, s[0]);
        if (i < 3) ascon_perm(s, 12);
    }
}

static void ascon_aead_one(const uint8_t key[16], const uint8_t nonce[16],
                           const uint8_t *pt, size_t len, uint8_t *ct, uint8_t tag[16], int sp800) {
    uint64_t k0 = load64_le(key), k1 = load64_le(key + 8);
    uint64_t n0 = load64_le(nonce), n1 = load64_le(nonce + 8);
    if (!sp800) { k0 = bswap64(k0); k1 = bswap64(k1); n0 = bswap64(n0); n1 = bswap64(n1); }
    uint64_t s[5] = {sp800 ? 0x00001000808c0001ULL : 0x80800c0800000000ULL, k0, k1, n0, n1};
    ascon_perm(s, 12);
    s[3] ^= k0; s[4] ^= k1;
    s[4] ^= sp800 ? (1ULL << 63) : 1ULL;
    size_t off = 0;
    while (off + 16 <= len) {
        uint64_t p0 = load64_le(pt + off), p1 = load64_le(pt + off + 8);
        if (!sp800) { p0 = bswap64(p0); p1 = bswap64(p1); }
        s[0] ^= p0; s[1] ^= p1;
        store64_le(ct + off, sp800 ? s[0] : bswap64(s[0]));
        store64_le(ct + off + 8, sp800 ? s[1] : bswap64(s[1]));
        ascon_perm(s, 8);
        off += 16;
    }
    uint8_t last[16] = {0};
    size_t rem = len - off;
    if (rem) memcpy(last, pt + off, rem);
    last[rem] = sp800 ? 0x01 : 0x80;
    uint64_t p0 = load64_le(last), p1 = load64_le(last + 8);
    if (!sp800) { p0 = bswap64(p0); p1 = bswap64(p1); }
    s[0] ^= p0; s[1] ^= p1;
    if (rem) {
        uint8_t outb[16];
        store64_le(outb, sp800 ? s[0] : bswap64(s[0]));
        store64_le(outb + 8, sp800 ? s[1] : bswap64(s[1]));
        memcpy(ct + off, outb, rem);
    }
    s[2] ^= k0; s[3] ^= k1;
    ascon_perm(s, 12);
    uint64_t t0 = s[3] ^ k0, t1 = s[4] ^ k1;
    store64_le(tag, sp800 ? t0 : bswap64(t0));
    store64_le(tag + 8, sp800 ? t1 : bswap64(t1));
}

static volatile uint64_t sink = 0;

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <algo> <size_or_msglen> <iters>\n", argv[0]);
        return 2;
    }
    const char *algo = argv[1];
    size_t sz = (size_t)atol(argv[2]);
    uint64_t iters = (uint64_t)atoll(argv[3]);
    uint64_t warm = iters / 10 + 8;

    uint8_t *msg = malloc(sz ? sz : 1);
    uint8_t *out = malloc(sz + 64);
    memset(msg, 0x5a, sz ? sz : 1);
    double t0 = 0, t1 = 0;

    if (!strcmp(algo, "sm3")) {
        EVP_MD *md = EVP_MD_fetch(NULL, "SM3", NULL);
        EVP_MD_CTX *ctx = EVP_MD_CTX_new();
        unsigned int olen = 0;
        for (uint64_t i = 0; i < warm; i++) {
            EVP_DigestInit_ex(ctx, md, NULL);
            EVP_DigestUpdate(ctx, msg, sz);
            EVP_DigestFinal_ex(ctx, out, &olen);
        }
        t0 = now();
        for (uint64_t i = 0; i < iters; i++) {
            EVP_DigestInit_ex(ctx, md, NULL);
            EVP_DigestUpdate(ctx, msg, sz);
            EVP_DigestFinal_ex(ctx, out, &olen);
        }
        t1 = now();
        sink ^= out[0];
        EVP_MD_CTX_free(ctx); EVP_MD_free(md);
    } else if (!strcmp(algo, "ascon-hash256")) {
        for (uint64_t i = 0; i < warm; i++) ascon_hash256_one(msg, sz, out);
        t0 = now();
        for (uint64_t i = 0; i < iters; i++) ascon_hash256_one(msg, sz, out);
        t1 = now();
        sink ^= out[0];
    } else if (!strcmp(algo, "ascon-aead128") || !strcmp(algo, "ascon-128a")) {
        int sp800 = !strcmp(algo, "ascon-aead128");
        uint8_t key[16] = {0}, nonce[16] = {1}, tag[16];
        for (uint64_t i = 0; i < warm; i++) ascon_aead_one(key, nonce, msg, sz, out, tag, sp800);
        t0 = now();
        for (uint64_t i = 0; i < iters; i++) ascon_aead_one(key, nonce, msg, sz, out, tag, sp800);
        t1 = now();
        sink ^= tag[0] ^ out[0];
    } else if (!strcmp(algo, "x25519")) {
        uint8_t sk_raw[32] = {7}, pk_raw[32] = {9};
        EVP_PKEY *sk = EVP_PKEY_new_raw_private_key(EVP_PKEY_X25519, NULL, sk_raw, 32);
        EVP_PKEY *peer = EVP_PKEY_new_raw_public_key(EVP_PKEY_X25519, NULL, pk_raw, 32);
        EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new(sk, NULL);
        size_t slen = 32;
        for (uint64_t i = 0; i < warm; i++) {
            EVP_PKEY_derive_init(ctx);
            EVP_PKEY_derive_set_peer(ctx, peer);
            EVP_PKEY_derive(ctx, out, &slen);
        }
        t0 = now();
        for (uint64_t i = 0; i < iters; i++) {
            EVP_PKEY_derive_init(ctx);
            EVP_PKEY_derive_set_peer(ctx, peer);
            EVP_PKEY_derive(ctx, out, &slen);
        }
        t1 = now();
        sink ^= out[0];
        EVP_PKEY_CTX_free(ctx); EVP_PKEY_free(peer); EVP_PKEY_free(sk);
    } else if (!strcmp(algo, "ed25519-verify")) {
        EVP_PKEY *pkey = EVP_PKEY_Q_keygen(NULL, NULL, "ED25519");
        EVP_MD_CTX *mctx = EVP_MD_CTX_new();
        uint8_t sig[64]; size_t siglen = 64;
        EVP_DigestSignInit(mctx, NULL, NULL, NULL, pkey);
        EVP_DigestSign(mctx, sig, &siglen, msg, sz);
        for (uint64_t i = 0; i < warm; i++) {
            EVP_DigestVerifyInit(mctx, NULL, NULL, NULL, pkey);
            if (EVP_DigestVerify(mctx, sig, siglen, msg, sz) != 1) return 3;
        }
        t0 = now();
        for (uint64_t i = 0; i < iters; i++) {
            EVP_DigestVerifyInit(mctx, NULL, NULL, NULL, pkey);
            if (EVP_DigestVerify(mctx, sig, siglen, msg, sz) != 1) return 3;
        }
        t1 = now();
        EVP_MD_CTX_free(mctx); EVP_PKEY_free(pkey);
    } else {
        OQS_SIG *s = OQS_SIG_new(algo);
        if (!s) { fprintf(stderr, "unknown algo %s\n", algo); return 2; }
        uint8_t *pk = malloc(s->length_public_key);
        uint8_t *sk = malloc(s->length_secret_key);
        uint8_t *sig = malloc(s->length_signature);
        size_t siglen = 0;
        if (OQS_SIG_keypair(s, pk, sk) != OQS_SUCCESS ||
            OQS_SIG_sign(s, sig, &siglen, msg, sz, sk) != OQS_SUCCESS) return 3;
        for (uint64_t i = 0; i < warm; i++) {
            if (OQS_SIG_verify(s, msg, sz, sig, siglen, pk) != OQS_SUCCESS) return 3;
        }
        t0 = now();
        for (uint64_t i = 0; i < iters; i++) {
            if (OQS_SIG_verify(s, msg, sz, sig, siglen, pk) != OQS_SUCCESS) return 3;
        }
        t1 = now();
        free(pk); free(sk); free(sig); OQS_SIG_free(s);
    }

    printf("{\"algo\":\"%s\",\"size\":%zu,\"iters\":%llu,\"secs\":%.6f}\n",
           algo, sz, (unsigned long long)iters, t1 - t0);
    free(msg); free(out);
    return 0;
}
