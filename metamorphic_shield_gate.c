/* metamorphic_shield_gate.c — PulsarX as deterministic metamorphic shield.
 *
 * Falsifiable gate for "metamorphism replaces ASLR": the SAME pipeline
 * compiled under N different seeds must —
 *   1. DIVERSITY:      produce pairwise byte-different machine code,
 *   2. CORRECTNESS:    compute bit-identical outputs (== scalar reference),
 *   3. GADGET DEATH:   ROP-gadget byte signatures from blob_0 do not survive
 *                      at the same offsets in the other blobs,
 *   4. DETERMINISM:    re-compiling seed_0 yields byte-identical code
 *                      (no entropy source, reproducible builds).
 *
 * Build: gcc -O2 -mavx512f -mavx512dq -Wall -Wextra -o metamorphic-gate \
 *            metamorphic_shield_gate.c -lpthread -lm
 * Exit 0 = GATE PASS, 1 = FAIL.
 *
 * Host-aware: the gate EXECUTES the JIT code it compiles, so a
 * section whose ISA the host lacks is SKIPPED, never run into an
 * illegal instruction. Skips are reported, never silent.
 * -DPULSAR_GATE_FAKE_IVY / -DPULSAR_GATE_FAKE_HASWELL force the
 * probe, so the skip paths can be tested on an AVX-512 host.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define main pulsarx_main_disabled
#include "pulsarX_ASM.c"
#undef main

#define NSEEDS 16

/* Host capability probe (v5.2): see the header comment. */
static int host_has_avx512(void)
{
#if defined(PULSAR_GATE_FAKE_IVY) || defined(PULSAR_GATE_FAKE_HASWELL)
    return 0;
#else
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512dq");
#endif
}

static int host_has_avx2(void)
{
#ifdef PULSAR_GATE_FAKE_IVY
    return 0;
#elif defined(PULSAR_GATE_FAKE_HASWELL)
    return 1;
#else
    return __builtin_cpu_supports("avx2");
#endif
}

static int host_has_avx(void)
{
#if defined(PULSAR_GATE_FAKE_IVY) || defined(PULSAR_GATE_FAKE_HASWELL)
    return 1;
#else
    return __builtin_cpu_supports("avx");
#endif
}

/* Fixed pipeline exercising every mutated op:
 *   v = ((v+5)*3)^2; v -= 7; v /= 2; v += -11                                    */
static void load_gate_pipeline(engine *e)
{
    e->pipe.count = 6;
    e->pipe.ops[0].type = OP_ADD;    e->pipe.ops[0].constant = 5;
    e->pipe.ops[1].type = OP_MUL;    e->pipe.ops[1].constant = 3;
    e->pipe.ops[2].type = OP_SQUARE; e->pipe.ops[2].constant = 0;
    e->pipe.ops[3].type = OP_SUB;    e->pipe.ops[3].constant = 7;
    e->pipe.ops[4].type = OP_DIV;    e->pipe.ops[4].constant = 2;
    e->pipe.ops[5].type = OP_ADD;    e->pipe.ops[5].constant = -11;
    for (int i = 0; i < 6; i++)
        e->pipe.ops[i].field = 0;
}

/* Scalar reference: matches interpret() semantics (C wraparound-free here:
 * inputs are small, (x+5)*3 stays far from int64 overflow). */
static int64_t ref_run(int64_t v)
{
    v += 5;
    v *= 3;
    v *= v;
    v -= 7;
    v /= 2;      /* trunc toward zero, like idiv */
    v += -11;
    return v;
}

static const int64_t kInputs[] = { -100, -7, -1, 0, 1, 2, 13, 100, 1000 };
#define NINPUTS (sizeof(kInputs)/sizeof(kInputs[0]))

/* Compile pipeline under seed -> snapshot bytes. Engine must be init'ed and
 * carry the gate pipeline. Returns emitted length. */
static size_t compile_snapshot(engine *e, uint64_t seed,
                               uint8_t *out, size_t out_cap)
{
    meta_config cfg = {
        .seed = seed,
        .enable_nop = 1,
        .enable_reg_variation = 1,
        .enable_instruction_swap = 1
    };
    jit_compile_meta_scalar(e, &cfg);
    size_t len = e->jit_code.offset;
    if (len > out_cap) {
        fprintf(stderr, "blob too large\n");
        exit(1);
    }
    memcpy(out, e->jit_code.mem, len);
    return len;
}

/* Vector twin: compile the same pipeline with the EVEX meta JIT
 * (8 packed int64s per call). Returns emitted length, 0 if the vector
 * form does not apply (it always does for this pipeline). */
static size_t compile_snapshot_vec(engine *e, uint64_t seed,
                                   uint8_t *out, size_t out_cap)
{
    meta_config cfg = {
        .seed = seed,
        .enable_nop = 1,
        .enable_reg_variation = 1,
        .enable_instruction_swap = 1
    };
    if(!jit_compile_meta_avx512(e, &cfg))
    {
        fprintf(stderr, "vector meta JIT refused the gate pipeline\n");
        exit(1);
    }
    size_t len = e->jit_code.offset;
    if (len > out_cap) {
        fprintf(stderr, "vector blob too large\n");
        exit(1);
    }
    memcpy(out, e->jit_code.mem, len);
    return len;
}

static size_t hamming(const uint8_t *a, const uint8_t *b, size_t n)
{
    size_t d = 0;
    for (size_t i = 0; i < n; i++)
        d += (a[i] != b[i]);
    return d;
}

int main(void)
{
    int fails = 0;
    int skipped = 0;   /* execution checks skipped: host lacks that ISA */
    engine e;
    engine_init(&e);
    load_gate_pipeline(&e);

    static uint8_t blobs[NSEEDS][65536];
    static size_t  lens[NSEEDS];
    /* Distinct odd seeds: deterministic, no entropy anywhere. */
    static const uint64_t seeds[NSEEDS] = {
        0x123456789ABCDEF1ULL, 0x0FEDCBA987654321ULL,
        0x1111111111111111ULL, 0x2222222222222223ULL,
        0x3333333333333335ULL, 0x4444444444444447ULL,
        0x5555555555555559ULL, 0x6666666666666661ULL,
        0x7777777777777773ULL, 0x8888888888888885ULL,
        0x9999999999999997ULL, 0xAAAAAAAAAAAAAAAAULL | 1,
        0xBBBBBBBBBBBBBBB1ULL, 0xCCCCCCCCCCCCCCD1ULL,
        0xDDDDDDDDDDDDDDE1ULL, 0xEEEEEEEEEEEEEEF1ULL,
    };

    /* ---- compile all variants, snapshotting bytes ---- */
    for (int i = 0; i < NSEEDS; i++)
        lens[i] = compile_snapshot(&e, seeds[i], blobs[i], sizeof(blobs[i]));

    /* ---- 1. DIVERSITY: pairwise byte-different ---- */
    size_t min_hd = (size_t)-1;
    int diverse_pairs = 0, total_pairs = 0;
    for (int i = 0; i < NSEEDS; i++)
        for (int j = i + 1; j < NSEEDS; j++) {
            total_pairs++;
            size_t n = lens[i] < lens[j] ? lens[i] : lens[j];
            size_t hd = hamming(blobs[i], blobs[j], n)
                      + (lens[i] > lens[j] ? lens[i] - lens[j]
                                           : lens[j] - lens[i]);
            if (hd < min_hd) min_hd = hd;
            if (hd > 0) diverse_pairs++;
        }
    printf("[1] diversity: %d/%d pairs differ, min Hamming distance = %zu bytes\n",
           diverse_pairs, total_pairs, min_hd);
    if (diverse_pairs != total_pairs) {
        printf("    FAIL: some seed pair produced identical code\n");
        fails++;
    }

    /* ---- 2. CORRECTNESS: bit-identical outputs == reference ---- */
    /* Re-install each blob and call it directly (scalar ABI: RDI=data). */
    {
        /* jit_fn_t takes (uint8_t*, int); scalar path uses data pointer. */
        int bad = 0;
        for (int i = 0; i < NSEEDS && !bad; i++) {
            /* Re-install blob i into the engine without recompiling:
             * point jit_fn at a private executable copy. */
            /* Simplest faithful route: recompile is deterministic, so just
             * recompile seed i (proven byte-identical in check 4 below is
             * circular here) — instead install the snapshot via mmap. */
            void *wx = mmap(NULL, lens[i], PROT_READ | PROT_WRITE | PROT_EXEC,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (wx == MAP_FAILED) { perror("mmap"); return 1; }
            memcpy(wx, blobs[i], lens[i]);
            jit_fn_t fn = (jit_fn_t)wx;
            for (size_t k = 0; k < NINPUTS; k++) {
                int64_t v = kInputs[k];
                int64_t expect = ref_run(v);
                fn((uint8_t *)&v, 1);
                if (v != expect) {
                    printf("    FAIL: seed %d input %lld -> %lld, want %lld\n",
                           i, (long long)kInputs[k], (long long)v,
                           (long long)expect);
                    bad = 1;
                    break;
                }
            }
            munmap(wx, lens[i]);
        }
        if (!bad)
            printf("[2] correctness: %d blobs x %zu inputs bit-identical to reference\n",
                   NSEEDS, NINPUTS);
        else
            fails++;
    }

    /* ---- 3. GADGET DEATH: byte patterns at fixed offsets die elsewhere ----
     * An exploit hardcoded against variant A embeds addresses; if the bytes
     * at those addresses differ in variant B, the exploit fails. We sample
     * K fixed offsets across blob_0 and check each 4-byte window against
     * every other blob at the SAME offset. */
    {
        int offs[] = { 0, 4, 8, 12, 16, 20, 24, 28 };
        int noff = sizeof(offs) / sizeof(offs[0]);
        int dead = 0, total = 0;
        for (int o = 0; o < noff; o++) {
            size_t off = (size_t)offs[o];
            if (off + 4 > lens[0]) continue;
            for (int i = 1; i < NSEEDS; i++) {
                if (off + 4 > lens[i]) continue;
                total++;
                if (memcmp(blobs[0] + off, blobs[i] + off, 4) != 0) dead++;
            }
        }
        printf("[3] gadget death: %d/%d (offset,blob) 4-byte windows differ "
               "from blob_0\n", dead, total);
        if (total == 0) {
            printf("    FAIL: no comparable windows\n");
            fails++;
        } else if (dead * 2 < total) {
            printf("    FAIL: majority of fixed-offset windows survive "
                   "(need >= half dead)\n");
            fails++;
        }
        /* Classic ret-gadget illustration: 0xC3 offsets from blob_0. */
        int ret_off[256]; int nret = 0;
        for (size_t i = 0; i < lens[0] && nret < 256; i++)
            if (blobs[0][i] == 0xC3) ret_off[nret++] = (int)i;
        int rdead = 0;
        for (int r = 0; r < nret; r++) {
            int survives = 0;
            for (int i = 1; i < NSEEDS; i++) {
                if ((size_t)ret_off[r] < lens[i] &&
                    blobs[i][ret_off[r]] == 0xC3) { survives = 1; break; }
            }
            if (!survives) rdead++;
        }
        printf("    ret-gadget offsets: %d/%d dead across all other blobs\n",
               rdead, nret);
    }

    /* ---- 4. DETERMINISM: same seed -> byte-identical (no entropy) ---- */
    {
        static uint8_t again[65536];
        size_t len2 = compile_snapshot(&e, seeds[0], again, sizeof(again));
        int same = (len2 == lens[0] && memcmp(again, blobs[0], len2) == 0);
        /* And once more, to rule out hidden state. */
        size_t len3 = compile_snapshot(&e, seeds[0], again, sizeof(again));
        same = same && (len3 == lens[0] && memcmp(again, blobs[0], len3) == 0);
        printf("[4] determinism: seed_0 recompiled twice -> %s\n",
               same ? "byte-identical" : "DIFFERENT");
        if (!same) { printf("    FAIL: nondeterministic codegen\n"); fails++; }
    }

    /* ============================================================
       VECTOR METAMORPHIC GATE (§3): same 4 checks on the EVEX
       meta JIT (jit_compile_meta_avx512), 8 packed int64s per call.
       ============================================================ */
    static uint8_t vblobs[NSEEDS][65536];
    static size_t  vlens[NSEEDS];
    for (int i = 0; i < NSEEDS; i++)
        vlens[i] = compile_snapshot_vec(&e, seeds[i], vblobs[i], sizeof(vblobs[i]));

    /* [V1] DIVERSITY */
    {
        size_t min_hd = (size_t)-1;
        int diverse_pairs = 0, total_pairs = 0;
        for (int i = 0; i < NSEEDS; i++)
            for (int j = i + 1; j < NSEEDS; j++) {
                total_pairs++;
                size_t n = vlens[i] < vlens[j] ? vlens[i] : vlens[j];
                size_t hd = hamming(vblobs[i], vblobs[j], n)
                          + (vlens[i] > vlens[j] ? vlens[i] - vlens[j]
                                                 : vlens[j] - vlens[i]);
                if (hd < min_hd) min_hd = hd;
                if (hd > 0) diverse_pairs++;
            }
        printf("[V1] vector diversity: %d/%d pairs differ, min Hamming = %zu bytes\n",
               diverse_pairs, total_pairs, min_hd);
        if (diverse_pairs != total_pairs) {
            printf("    FAIL: some vector seed pair produced identical code\n");
            fails++;
        }
    }

    /* [V2] CORRECTNESS: 8 lanes at once, bit-identical to reference.
       Executes the emitted EVEX code: skip when the host has no
       AVX-512 ([V1]/[V3]/[V4] still verify the emitted bytes). */
    if(!host_has_avx512())
    {
        printf("[V2] vector correctness: SKIP (host has no AVX-512; emitted code still covered by [V1]/[V3]/[V4])\n");
        skipped++;
    }
    else
    {
        int bad = 0;
        for (int i = 0; i < NSEEDS && !bad; i++) {
            void *wx = mmap(NULL, vlens[i], PROT_READ | PROT_WRITE | PROT_EXEC,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (wx == MAP_FAILED) { perror("mmap"); return 1; }
            memcpy(wx, vblobs[i], vlens[i]);
            jit_fn_t fn = (jit_fn_t)wx;
            int64_t v[8];
            for (int k = 0; k < 8; k++)
                v[k] = kInputs[k % NINPUTS];
            int64_t expect[8];
            for (int k = 0; k < 8; k++)
                expect[k] = ref_run(v[k]);
            fn((uint8_t *)v, 8);
            for (int k = 0; k < 8; k++) {
                if (v[k] != expect[k]) {
                    printf("    FAIL: vector seed %d lane %d -> %lld, want %lld\n",
                           i, k, (long long)v[k], (long long)expect[k]);
                    bad = 1;
                    break;
                }
            }
            munmap(wx, vlens[i]);
        }
        if (!bad)
            printf("[V2] vector correctness: %d blobs x 8 lanes bit-identical to reference\n",
                   NSEEDS);
        else
            fails++;
    }

    /* [V3] GADGET DEATH */
    {
        int offs[] = { 0, 4, 8, 12, 16, 20, 24, 28 };
        int noff = sizeof(offs) / sizeof(offs[0]);
        int dead = 0, total = 0;
        for (int o = 0; o < noff; o++) {
            size_t off = (size_t)offs[o];
            if (off + 4 > vlens[0]) continue;
            for (int i = 1; i < NSEEDS; i++) {
                if (off + 4 > vlens[i]) continue;
                total++;
                if (memcmp(vblobs[0] + off, vblobs[i] + off, 4) != 0) dead++;
            }
        }
        printf("[V3] vector gadget death: %d/%d (offset,blob) 4-byte windows differ "
               "from blob_0\n", dead, total);
        if (total == 0) {
            printf("    FAIL: no comparable vector windows\n");
            fails++;
        } else if (dead * 2 < total) {
            printf("    FAIL: majority of fixed-offset vector windows survive "
                   "(need >= half dead)\n");
            fails++;
        }
    }

    /* [V4] DETERMINISM */
    {
        static uint8_t again[65536];
        size_t len2 = compile_snapshot_vec(&e, seeds[0], again, sizeof(again));
        int same = (len2 == vlens[0] && memcmp(again, vblobs[0], len2) == 0);
        size_t len3 = compile_snapshot_vec(&e, seeds[0], again, sizeof(again));
        same = same && (len3 == vlens[0] && memcmp(again, vblobs[0], len3) == 0);
        printf("[V4] vector determinism: seed_0 recompiled twice -> %s\n",
               same ? "byte-identical" : "DIFFERENT");
        if (!same) { printf("    FAIL: nondeterministic vector codegen\n"); fails++; }
    }


    /* ============================================================
       VEX METAMORPHIC GATE (v5 §10): same 4 checks on the VEX-256
       meta JIT (jit_compile_meta_vex, Haswell 4x int64 per call).
       ============================================================ */
    static uint8_t xblobs[NSEEDS][65536];
    static size_t  xlens[NSEEDS];
    for (int i = 0; i < NSEEDS; i++) {
        meta_config cfg = {
            .seed = seeds[i],
            .enable_nop = 1,
            .enable_reg_variation = 1,
            .enable_instruction_swap = 1
        };
        if (!jit_compile_meta_vex(&e, &cfg, 1 /*AVX2 VEX-256*/)) {
            fprintf(stderr, "VEX meta JIT refused gate pipeline\n");
            return 1;
        }
        xlens[i] = e.jit_code.offset;
        memcpy(xblobs[i], e.jit_code.mem, xlens[i]);
    }

    /* [X1] VEX DIVERSITY */
    {
        size_t min_hd = (size_t)-1;
        int diverse_pairs = 0, total_pairs = 0;
        for (int i = 0; i < NSEEDS; i++)
            for (int j = i + 1; j < NSEEDS; j++) {
                total_pairs++;
                size_t n = xlens[i] < xlens[j] ? xlens[i] : xlens[j];
                size_t hd = hamming(xblobs[i], xblobs[j], n)
                          + (xlens[i] > xlens[j] ? xlens[i] - xlens[j]
                                                 : xlens[j] - xlens[i]);
                if (hd < min_hd) min_hd = hd;
                if (hd > 0) diverse_pairs++;
            }
        printf("[X1] VEX-256 diversity: %d/%d pairs differ, min Hamming = %zu bytes\n",
               diverse_pairs, total_pairs, min_hd);
        if (diverse_pairs != total_pairs) fails++;
    }

    /* [X2] VEX CORRECTNESS (4 lanes at once). Executes the emitted
       VEX-256 code: skip when the host has no AVX2 ([X1]/[X3]/[X4]
       still verify the emitted bytes). */
    if(!host_has_avx2())
    {
        printf("[X2] VEX-256 correctness: SKIP (host has no AVX2; emitted code still covered by [X1]/[X3]/[X4])\n");
        skipped++;
    }
    else
    {
        int bad = 0;
        for (int i = 0; i < NSEEDS && !bad; i++) {
            void *wx = mmap(NULL, xlens[i], PROT_READ | PROT_WRITE | PROT_EXEC,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (wx == MAP_FAILED) { perror("mmap"); return 1; }
            memcpy(wx, xblobs[i], xlens[i]);
            jit_fn_t fn = (jit_fn_t)wx;
            for (size_t base = 0; base < NINPUTS; base++) {
                int64_t v[4], expect[4];
                for (int k = 0; k < 4; k++) {
                    v[k] = kInputs[(base + k) % NINPUTS];
                    expect[k] = ref_run(v[k]);
                }
                fn((uint8_t *)v, 4);
                for (int k = 0; k < 4; k++)
                    if (v[k] != expect[k]) { bad = 1; break; }
            }
            munmap(wx, xlens[i]);
        }
        if (!bad)
            printf("[X2] VEX-256 correctness: %d blobs x 4 lanes bit-identical to reference\n",
                   NSEEDS);
        else
            fails++;
    }

    /* [X3] VEX GADGET DEATH */
    {
        int offs[] = { 0, 4, 8, 12, 16, 20, 24, 28 };
        int noff = sizeof(offs) / sizeof(offs[0]);
        int dead = 0, total = 0;
        for (int o = 0; o < noff; o++) {
            size_t off = (size_t)offs[o];
            if (off + 4 > xlens[0]) continue;
            for (int i = 1; i < NSEEDS; i++) {
                if (off + 4 > xlens[i]) continue;
                total++;
                if (memcmp(xblobs[0] + off, xblobs[i] + off, 4) != 0) dead++;
            }
        }
        printf("[X3] VEX-256 gadget death: %d/%d (offset,blob) 4-byte windows differ from blob_0\n",
               dead, total);
        if (total == 0 || dead * 2 < total) fails++;
    }

    /* [X4] VEX DETERMINISM */
    {
        static uint8_t again[65536];
        meta_config cfg = {
            .seed = seeds[0], .enable_nop = 1, .enable_reg_variation = 1, .enable_instruction_swap = 1
        };
        jit_compile_meta_vex(&e, &cfg, 1);
        size_t len2 = e.jit_code.offset;
        memcpy(again, e.jit_code.mem, len2);
        int same = (len2 == xlens[0] && memcmp(again, xblobs[0], len2) == 0);
        printf("[X4] VEX-256 determinism: seed_0 recompiled -> %s\n",
               same ? "byte-identical" : "DIFFERENT");
        if (!same) fails++;
    }

    /* ============================================================
       VEX-128 METAMORPHIC GATE (v5.2): the Ivy Bridge section the
       v5 gate was missing entirely — same 4 checks on the VEX-128
       meta JIT (jit_compile_meta_vex, 2x int64 per call). This is
       the only vector section that executes on an Ivy host.
       ============================================================ */
    static uint8_t yblobs[NSEEDS][65536];
    static size_t  ylens[NSEEDS];
    for (int i = 0; i < NSEEDS; i++) {
        meta_config cfg = {
            .seed = seeds[i],
            .enable_nop = 1,
            .enable_reg_variation = 1,
            .enable_instruction_swap = 1
        };
        if (!jit_compile_meta_vex(&e, &cfg, 0 /*AVX1 VEX-128*/)) {
            fprintf(stderr, "VEX-128 meta JIT refused gate pipeline\n");
            return 1;
        }
        ylens[i] = e.jit_code.offset;
        memcpy(yblobs[i], e.jit_code.mem, ylens[i]);
    }

    /* [Y1] VEX-128 DIVERSITY */
    {
        size_t min_hd = (size_t)-1;
        int diverse_pairs = 0, total_pairs = 0;
        for (int i = 0; i < NSEEDS; i++)
            for (int j = i + 1; j < NSEEDS; j++) {
                total_pairs++;
                size_t n = ylens[i] < ylens[j] ? ylens[i] : ylens[j];
                size_t hd = hamming(yblobs[i], yblobs[j], n)
                          + (ylens[i] > ylens[j] ? ylens[i] - ylens[j]
                                                 : ylens[j] - ylens[i]);
                if (hd < min_hd) min_hd = hd;
                if (hd > 0) diverse_pairs++;
            }
        printf("[Y1] VEX-128 diversity: %d/%d pairs differ, min Hamming = %zu bytes\n",
               diverse_pairs, total_pairs, min_hd);
        if (diverse_pairs != total_pairs) fails++;
    }

    /* [Y2] VEX-128 CORRECTNESS (2 lanes at once). Executes the
       emitted VEX-128 code: skip when the host has no AVX at all. */
    if(!host_has_avx())
    {
        printf("[Y2] VEX-128 correctness: SKIP (host has no AVX; emitted code still covered by [Y1]/[Y3]/[Y4])\n");
        skipped++;
    }
    else
    {
        int bad = 0;
        for (int i = 0; i < NSEEDS && !bad; i++) {
            void *wx = mmap(NULL, ylens[i], PROT_READ | PROT_WRITE | PROT_EXEC,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (wx == MAP_FAILED) { perror("mmap"); return 1; }
            memcpy(wx, yblobs[i], ylens[i]);
            jit_fn_t fn = (jit_fn_t)wx;
            for (size_t base = 0; base < NINPUTS; base++) {
                int64_t v[2], expect[2];
                for (int k = 0; k < 2; k++) {
                    v[k] = kInputs[(base + k) % NINPUTS];
                    expect[k] = ref_run(v[k]);
                }
                fn((uint8_t *)v, 2);
                for (int k = 0; k < 2; k++)
                    if (v[k] != expect[k]) { bad = 1; break; }
            }
            munmap(wx, ylens[i]);
        }
        if (!bad)
            printf("[Y2] VEX-128 correctness: %d blobs x 2 lanes bit-identical to reference\n",
                   NSEEDS);
        else
            fails++;
    }

    /* [Y3] VEX-128 GADGET DEATH */
    {
        int offs[] = { 0, 4, 8, 12, 16, 20, 24, 28 };
        int noff = sizeof(offs) / sizeof(offs[0]);
        int dead = 0, total = 0;
        for (int o = 0; o < noff; o++) {
            size_t off = (size_t)offs[o];
            if (off + 4 > ylens[0]) continue;
            for (int i = 1; i < NSEEDS; i++) {
                if (off + 4 > ylens[i]) continue;
                total++;
                if (memcmp(yblobs[0] + off, yblobs[i] + off, 4) != 0) dead++;
            }
        }
        printf("[Y3] VEX-128 gadget death: %d/%d (offset,blob) 4-byte windows differ from blob_0\n",
               dead, total);
        if (total == 0 || dead * 2 < total) fails++;
    }

    /* [Y4] VEX-128 DETERMINISM */
    {
        static uint8_t again[65536];
        meta_config cfg = {
            .seed = seeds[0], .enable_nop = 1, .enable_reg_variation = 1, .enable_instruction_swap = 1
        };
        jit_compile_meta_vex(&e, &cfg, 0);
        size_t len2 = e.jit_code.offset;
        memcpy(again, e.jit_code.mem, len2);
        int same = (len2 == ylens[0] && memcmp(again, yblobs[0], len2) == 0);
        printf("[Y4] VEX-128 determinism: seed_0 recompiled -> %s\n",
               same ? "byte-identical" : "DIFFERENT");
        if (!same) fails++;
    }

    engine_shutdown(&e);

    if (!fails) {
        printf("=== METAMORPHIC SHIELD GATE: PASS ===\n");
        if (skipped)
            printf("    note: %d execution check(s) skipped — host lacks that ISA (emitted code still verified)\n",
                   skipped);
        return 0;
    }
    printf("=== METAMORPHIC SHIELD GATE: FAIL (%d) ===\n", fails);
    return 1;
}
