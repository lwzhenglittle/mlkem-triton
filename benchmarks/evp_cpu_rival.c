/* The CPU rival for every primitive whose reference implementation is libcrypto.
 *
 * Why a C driver instead of `openssl speed` or ctypes: the device rows in
 * `bench_primitive.py` feed *whole batches of messages* per kernel launch, so their per-message
 * overhead is zero. The CPU has to be measured the same way and at the same message size, and
 * there are only two honest ways to do that:
 *
 *   - `openssl speed -bytes N` is C, but it reports a rate rather than a unit count, so it
 *     cannot be timed by an external barrier, and its start-up has to be guessed at.
 *   - ctypes from Python is 3.4x too slow at 1 KiB messages (0.97 GB/s against this driver's
 *     3.3), because three FFI calls per message cost more than the hash does.
 *
 * So: same libcrypto, same API shape (`EVP_DigestInit/Update/Final` per message, one
 * `EVP_EncryptUpdate` per buffer), one continuous timing window, exact message count.
 *
 *     gcc -O3 -march=native -o benchmarks/evp_cpu benchmarks/evp_cpu_rival.c -lcrypto
 *     ./evp_cpu sha256 1024 268435456
 *     {"algo":"sha256","msgsize":1024,"messages":262144,"bytes":268435456,"secs":0.0813,"gbs":3.30}
 *
 * Message size is a parameter because it changes the answer by 3x: SHA-256 is 3.30 GB/s over
 * 1 KiB messages and 4.21 over 16 KiB ones on this machine, and the device row hashes 1 KiB.
 */
#include <openssl/evp.h>
#include <openssl/macros.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXOUT 64

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static int is_cipher(const char *algo) {
    return !strcmp(algo, "aes-128-ctr") || !strcmp(algo, "chacha20");
}

static int is_aead(const char *algo) { return !strcmp(algo, "chacha20-poly1305"); }

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <algo> <msgsize> <total_bytes>\n", argv[0]);
        return 2;
    }
    const char *algo = argv[1];
    size_t msgsize = (size_t)atol(argv[2]);
    uint64_t total = (uint64_t)atoll(argv[3]);
    if (msgsize == 0 || total == 0) {
        fprintf(stderr, "msgsize and total must be positive\n");
        return 2;
    }
    uint64_t messages = total / msgsize;
    if (messages == 0) messages = 1;

    unsigned char *msg = malloc(msgsize);
    if (!msg) { fprintf(stderr, "out of memory\n"); return 2; }
    memset(msg, 0x5a, msgsize);
    unsigned char out[MAXOUT];
    unsigned int outlen = 0;
    unsigned char key[32], iv[16];
    for (int i = 0; i < 32; i++) key[i] = (unsigned char)i;
    for (int i = 0; i < 16; i++) iv[i] = (unsigned char)i;

    OPENSSL_init_crypto(OPENSSL_INIT_LOAD_CRYPTO_STRINGS, NULL);
    double t0, t1;
    uint64_t done = 0;
    /* `openssl speed` reports the best of its internal repetitions, so a single cold window comes
     * out ~10% below it on every algorithm. Warm up on a fraction of the work first: the number
     * this driver prints must be comparable with the CLI it replaces, or the correction it exists
     * to make would bake in a fresh bias. */
    uint64_t warm = messages / 8 + 16;

    if (!strcmp(algo, "poly1305")) {
        EVP_MAC *mac = EVP_MAC_fetch(NULL, "POLY1305", NULL);
        if (!mac) { fprintf(stderr, "no POLY1305 MAC\n"); return 2; }
        EVP_MAC_CTX *ctx = EVP_MAC_CTX_new(mac);
        OSSL_PARAM params[2];
        params[0] = OSSL_PARAM_construct_octet_string("key", key, 32);
        params[1] = OSSL_PARAM_construct_end();
        size_t mlen = 0;
#define MAC_ONE()                                                                 \
        do {                                                                      \
            if (!EVP_MAC_init(ctx, NULL, 0, params) ||                             \
                !EVP_MAC_update(ctx, msg, msgsize) ||                              \
                !EVP_MAC_final(ctx, out, &mlen, sizeof(out))) {                    \
                fprintf(stderr, "POLY1305 failed\n");                             \
                return 2;                                                         \
            }                                                                     \
        } while (0)
        for (uint64_t i = 0; i < warm; i++) MAC_ONE();
        t0 = now();
        for (uint64_t i = 0; i < messages; i++) {
            MAC_ONE();
            done += msgsize;
        }
        t1 = now();
#undef MAC_ONE
        EVP_MAC_CTX_free(ctx);
        EVP_MAC_free(mac);
    } else if (is_aead(algo)) {
        /* One seal per message, tag included: the device row is `chacha20_poly1305_torch` over a
         * batch of 1 KiB plaintexts, so per-message init and final are part of the measurement. */
        EVP_CIPHER *cipher = EVP_CIPHER_fetch(NULL, "ChaCha20-Poly1305", NULL);
        if (!cipher) { fprintf(stderr, "no ChaCha20-Poly1305\n"); return 2; }
        EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
        unsigned char *ct = malloc(msgsize + 16);
        int n = 0, fin = 0;
        unsigned char nonce[12], tag[16];
        for (int i = 0; i < 12; i++) nonce[i] = (unsigned char)i;
#define AEAD_ONE()                                                                      \
        do {                                                                            \
            if (!EVP_EncryptInit_ex(ctx, cipher, NULL, NULL, NULL) ||                    \
                !EVP_EncryptInit_ex(ctx, NULL, NULL, key, nonce) ||                      \
                !EVP_EncryptUpdate(ctx, ct, &n, msg, (int)msgsize) ||                    \
                !EVP_EncryptFinal_ex(ctx, ct + n, &fin) ||                               \
                !EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_AEAD_GET_TAG, 16, tag)) {             \
                fprintf(stderr, "AEAD seal failed\n");                                 \
                return 2;                                                               \
            }                                                                           \
        } while (0)
        for (uint64_t i = 0; i < warm; i++) AEAD_ONE();
        t0 = now();
        for (uint64_t i = 0; i < messages; i++) {
            AEAD_ONE();
            done += msgsize;
        }
        t1 = now();
#undef AEAD_ONE
        free(ct);
        EVP_CIPHER_CTX_free(ctx);
        EVP_CIPHER_free(cipher);
    } else if (is_cipher(algo)) {
        EVP_CIPHER *cipher = EVP_CIPHER_fetch(NULL, algo, NULL);
        if (!cipher) { fprintf(stderr, "no cipher %s\n", algo); return 2; }
        EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
        int n = 0;
        if (!EVP_EncryptInit_ex(ctx, cipher, NULL, key, iv)) {
            fprintf(stderr, "EncryptInit failed\n");
            return 2;
        }
        /* One `update` per message-sized chunk, exactly like the device's per-launch buffer. A
         * stream cipher keeps its counter across calls, so this is a single keystream pass. */
        unsigned char *ct = malloc(msgsize);
        for (uint64_t i = 0; i < warm; i++) {
            if (!EVP_EncryptUpdate(ctx, ct, &n, msg, (int)msgsize)) return 2;
        }
        t0 = now();
        for (uint64_t i = 0; i < messages; i++) {
            if (!EVP_EncryptUpdate(ctx, ct, &n, msg, (int)msgsize) || (size_t)n != msgsize) {
                fprintf(stderr, "EncryptUpdate failed\n");
                return 2;
            }
            done += msgsize;
        }
        t1 = now();
        free(ct);
        EVP_CIPHER_CTX_free(ctx);
        EVP_CIPHER_free(cipher);
    } else {
        EVP_MD *md = EVP_MD_fetch(NULL, algo, NULL);
        if (!md) { fprintf(stderr, "no digest %s\n", algo); return 2; }
        int shake = !strncmp(algo, "shake", 5);
        EVP_MD_CTX *ctx = EVP_MD_CTX_new();
        /* Per-message init/final is the point: a 1 KiB row is mostly per-message cost. */
#define DIGEST_ONE()                                                                     \
        do {                                                                             \
            if (!EVP_DigestInit_ex(ctx, md, NULL) ||                                     \
                !EVP_DigestUpdate(ctx, msg, msgsize) ||                                  \
                (shake ? !EVP_DigestFinalXOF(ctx, out, 32)                               \
                       : !EVP_DigestFinal_ex(ctx, out, &outlen))) {                      \
                fprintf(stderr, "%s failed\n", algo);                                   \
                return 2;                                                                \
            }                                                                            \
        } while (0)
        for (uint64_t i = 0; i < warm; i++) DIGEST_ONE();
        t0 = now();
        for (uint64_t i = 0; i < messages; i++) {
            DIGEST_ONE();
            done += msgsize;
        }
        t1 = now();
#undef DIGEST_ONE
        EVP_MD_CTX_free(ctx);
        EVP_MD_free(md);
    }

    double dt = t1 - t0;
    printf("{\"algo\":\"%s\",\"msgsize\":%zu,\"messages\":%llu,\"bytes\":%llu,"
           "\"secs\":%.6f,\"gbs\":%.4f}\n",
           algo, msgsize, (unsigned long long)messages, (unsigned long long)done, dt,
           dt > 0 ? (double)done / dt / 1e9 : 0.0);
    free(msg);
    return 0;
}
