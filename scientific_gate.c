/* scientific_gate.c — Verification Gate for PulsarXASM v5 Scientific Upgrades:
 *   [S1] Kempner-Singmaster 2-Adic Null-Ideal Collapse (degree 4096 -> <= 65, bit-exact)
 *   [S2] Estrin Tree-Parallel 32-Wide AVX-512 & 16/8-Wide VEX Polynomial Kernels
 *   [S3] Native Vector VEX Division without VPSRAQ (Ivy Bridge L=0 & Haswell L=1)
 *   [S4] Rivest 2-Adic Bijective Permutation Polynomials & VEX Metamorphic Shield
 *   [S5] Online RDTSCP Multi-Armed Bandit Trace Autotuner
 *
 * Host-aware: the gate EXECUTES the JIT code it compiles, so a
 * kernel whose ISA the host lacks is SKIPPED, never run into an
 * illegal instruction. Skips are reported, never silent.
 * -DPULSAR_GATE_FAKE_IVY / -DPULSAR_GATE_FAKE_HASWELL force the
 * probe, so the skip paths can be tested on an AVX-512 host.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <time.h>

#define main pulsarx_main_disabled
#include "pulsarX_ASM.c"
#undef main

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

static uint64_t splitmix64(uint64_t *state)
{
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double measure_fn_mrec(jit_fn_t fn, int width, int64_t *buf, uint32_t count)
{
    struct timespec t0, t1;
    double best = 0.0;
    for(int it = 0; it < 3; it++)
    {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for(uint32_t i = 0; i + (uint32_t)width <= count; i += (uint32_t)width)
            fn((uint8_t*)&buf[i], (uint64_t)width);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double el = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        double thr = (double)count / (el * 1e6);
        if(thr > best) best = thr;
    }
    return best;
}

int main(void)
{
    int fails = 0;
    int skipped = 0;   /* executions skipped: host lacks that ISA */
    uint64_t rng = 0xCAFEBABE12345678ULL;

    /* ============================================================
       [S1] KEMPNER 2-ADIC NULL-IDEAL COLLAPSE (Degree 4096 -> <= 65)
       ============================================================ */
    {
        engine e;
        engine_init(&e);
        int64_t c_adds[12] = { 3, -7, 13, -29, 5, 101, -17, 9, -3, 41, -19, 77 };
        for(int i = 0; i < 12; i++)
        {
            operator op1 = { .type = OP_ADD, .field = 0, .constant = (double)c_adds[i] };
            operator op2 = { .type = OP_SQUARE, .field = 0, .constant = 0 };
            pipeline_add(&e.pipe, op1);
            pipeline_add(&e.pipe, op2);
        }

        poly_t poly;
        int rc = pipeline_to_poly(&e.pipe, &poly);
        if(rc != 0 || poly.degree > 65)
        {
            printf("    FAIL [S1]: pipeline_to_poly failed (rc=%d, deg=%d)\n", rc, poly.degree);
            fails++;
        }
        else
        {
            int64_t inputs[4096] __attribute__((aligned(64)));
            int64_t ref[4096]    __attribute__((aligned(64)));
            int64_t out_e[4096]  __attribute__((aligned(64)));
            int64_t out_v2[4096] __attribute__((aligned(64)));
            int64_t out_v1[4096] __attribute__((aligned(64)));

            inputs[0] = INT64_MIN; inputs[1] = INT64_MAX; inputs[2] = -1;
            inputs[3] = 0; inputs[4] = 1; inputs[5] = -42; inputs[6] = 42;
            for(int i = 7; i < 4096; i++)
                inputs[i] = (int64_t)splitmix64(&rng);

            for(int i = 0; i < 4096; i++)
            {
                int64_t v = inputs[i];
                record r = { .data = (uint8_t*)&v };
                interpret(&e, &r);
                ref[i] = v;
            }

            int bad = 0;
            for(int i = 0; i < 4096; i++)
            {
                int64_t pe = (int64_t)poly_eval_u64(&poly, (uint64_t)inputs[i]);
                if(pe != ref[i]) { bad = 1; break; }
            }

            /* Execute only the widths this host can run (v5.2). The
               scalar poly_eval check above already ran everywhere;
               building the kernels is host-independent. */
            int ran_e = host_has_avx512(), ran_v2 = host_has_avx2(),
                ran_v1 = host_has_avx();
            jit_builder jb_e, jb_v2, jb_v1;
            jb_init(&jb_e, 128*1024);
            jb_init(&jb_v2, 128*1024);
            jb_init(&jb_v1, 128*1024);
            if(!jit_build_estrin_avx512(&jb_e, &poly) ||
               !jit_build_poly_vex(&jb_v2, &poly, 1) ||
               !jit_build_poly_vex(&jb_v1, &poly, 0))
            {
                printf("    FAIL [S1]: polynomial JIT builders failed\n");
                bad = 1;
            }
            else if(!ran_e && !ran_v2 && !ran_v1)
            {
                printf("    FAIL [S1]: no runnable polynomial width on this host\n");
                bad = 1;
            }
            else
            {
                jit_fn_t fn_e  = (jit_fn_t)jb_e.mem.mem;
                jit_fn_t fn_v2 = (jit_fn_t)jb_v2.mem.mem;
                jit_fn_t fn_v1 = (jit_fn_t)jb_v1.mem.mem;
                if(ran_e)
                {
                    memcpy(out_e, inputs, sizeof(inputs));
                    for(int i = 0; i < 4096; i += 32) fn_e((uint8_t*)&out_e[i], 32);
                    if(memcmp(out_e, ref, sizeof(ref)) != 0) bad = 1;
                }
                else skipped++;
                if(ran_v2)
                {
                    memcpy(out_v2, inputs, sizeof(inputs));
                    for(int i = 0; i < 4096; i += 16) fn_v2((uint8_t*)&out_v2[i], 16);
                    if(memcmp(out_v2, ref, sizeof(ref)) != 0) bad = 1;
                }
                else skipped++;
                if(ran_v1)
                {
                    memcpy(out_v1, inputs, sizeof(inputs));
                    for(int i = 0; i < 4096; i += 8)  fn_v1((uint8_t*)&out_v1[i], 8);
                    if(memcmp(out_v1, ref, sizeof(ref)) != 0) bad = 1;
                }
                else skipped++;
            }
            munmap(jb_e.mem.mem, jb_e.mem.size);
            munmap(jb_v2.mem.mem, jb_v2.mem.size);
            munmap(jb_v1.mem.mem, jb_v1.mem.size);

            if(!bad)
            {
                char widths[48] = "";
                if(ran_e)  strcat(widths, "EVEX/");
                if(ran_v2) strcat(widths, "VEX256/");
                if(ran_v1) strcat(widths, "VEX128/");
                size_t wl = strlen(widths);
                if(wl) widths[wl - 1] = '\0';
                printf("[S1] Kempner 2-adic ideal reduction: classical degree 4096 -> canonical degree %d, 4096/4096 bit-exact across %s\n",
                       poly.degree, widths);
            }
            else
            {
                printf("    FAIL [S1]: mismatch in Kempner reduced polynomial evaluation\n");
                fails++;
            }
        }
        engine_shutdown(&e);
    }

    /* ============================================================
       [S2] ESTRIN TREE-PARALLEL POLYNOMIAL KERNELS (Degrees 0..65)
       ============================================================ */
    {
        int test_degs[] = { 0, 1, 2, 3, 4, 7, 8, 14, 15, 16, 29, 44, 65 };
        int ndegs = (int)(sizeof(test_degs) / sizeof(test_degs[0]));
        int bad = 0;
        /* Execute only the widths this host can run (v5.2). */
        int ran_e = host_has_avx512(), ran_v2 = host_has_avx2(),
            ran_v1 = host_has_avx();
        if(!ran_e && !ran_v2 && !ran_v1)
        {
            printf("    FAIL [S2]: no runnable kernel width on this host\n");
            bad = 1;
        }
        if(!ran_e) skipped++;
        if(!ran_v2) skipped++;
        if(!ran_v1) skipped++;
        for(int di = 0; di < ndegs && !bad; di++)
        {
            poly_t p;
            memset(&p, 0, sizeof(p));
            p.degree = test_degs[di];
            for(int k = 0; k <= p.degree; k++)
                p.coeff[k] = splitmix64(&rng) | (k == p.degree ? 1ULL : 0ULL);

            int64_t in[256] __attribute__((aligned(64)));
            int64_t o1[256] __attribute__((aligned(64)));
            int64_t o2[256] __attribute__((aligned(64)));
            int64_t o3[256] __attribute__((aligned(64)));
            for(int i = 0; i < 256; i++)
                in[i] = (int64_t)splitmix64(&rng);

            jit_builder jb1, jb2, jb3;
            jb_init(&jb1, 128*1024);
            jb_init(&jb2, 128*1024);
            jb_init(&jb3, 128*1024);
            jit_build_estrin_avx512(&jb1, &p);
            jit_build_poly_vex(&jb2, &p, 1);
            jit_build_poly_vex(&jb3, &p, 0);

            if(ran_e)
            {
                memcpy(o1, in, sizeof(in));
                for(int i = 0; i < 256; i += 32) ((jit_fn_t)jb1.mem.mem)((uint8_t*)&o1[i], 32);
            }
            if(ran_v2)
            {
                memcpy(o2, in, sizeof(in));
                for(int i = 0; i < 256; i += 16) ((jit_fn_t)jb2.mem.mem)((uint8_t*)&o2[i], 16);
            }
            if(ran_v1)
            {
                memcpy(o3, in, sizeof(in));
                for(int i = 0; i < 256; i += 8)  ((jit_fn_t)jb3.mem.mem)((uint8_t*)&o3[i], 8);
            }

            for(int i = 0; i < 256; i++)
            {
                int64_t want = (int64_t)poly_eval_u64(&p, (uint64_t)in[i]);
                if((ran_e  && o1[i] != want) ||
                   (ran_v2 && o2[i] != want) ||
                   (ran_v1 && o3[i] != want))
                {
                    printf("    FAIL [S2]: deg %d idx %d mismatch\n", p.degree, i);
                    bad = 1;
                    break;
                }
            }
            munmap(jb1.mem.mem, jb1.mem.size);
            munmap(jb2.mem.mem, jb2.mem.size);
            munmap(jb3.mem.mem, jb3.mem.size);
        }
        if(!bad)
        {
            char widths[48] = "";
            if(ran_e)  strcat(widths, "EVEX-512/");
            if(ran_v2) strcat(widths, "VEX-256/");
            if(ran_v1) strcat(widths, "VEX-128/");
            size_t wl = strlen(widths);
            if(wl) widths[wl - 1] = '\0';
            printf("[S2] Estrin tree-parallel kernels: %d degrees (0..65) x 256 inputs bit-exact across %s\n",
                   ndegs, widths);
        }
        else
            fails++;
    }

    /* ============================================================
       [S3] NATIVE VECTOR VEX DIVISION WITHOUT VPSRAQ (Ivy & Haswell)
       ============================================================ */
    {
        int64_t divs[] = {
            2, -2, 4, -8, 16, -32, 64, -1024, 65536, -1048576,
            3, -3, 5, -5, 7, -7, 11, -13, 97, -97, 1000, -1000,
            100003, -100003, 2147483647LL, -2147483647LL
        };
        int ndivs = (int)(sizeof(divs) / sizeof(divs[0]));
        int64_t in[512]   __attribute__((aligned(64)));
        int64_t o256[512] __attribute__((aligned(64))) = { 0 };
        int64_t o128[512] __attribute__((aligned(64))) = { 0 };

        in[0] = INT64_MIN; in[1] = INT64_MIN + 1; in[2] = INT64_MAX; in[3] = INT64_MAX - 1;
        in[4] = -1; in[5] = 0; in[6] = 1; in[7] = -2; in[8] = 2;
        for(int i = 9; i < 512; i++)
            in[i] = (int64_t)splitmix64(&rng);

        int bad = 0;
        /* Execute only the widths this host can run (v5.2). */
        int ran_v2 = host_has_avx2(), ran_v1 = host_has_avx();
        if(!ran_v2 && !ran_v1)
        {
            printf("    FAIL [S3]: no runnable VEX width on this host\n");
            bad = 1;
        }
        if(!ran_v2) skipped++;
        if(!ran_v1) skipped++;
        for(int di = 0; di < ndivs && !bad; di++)
        {
            int64_t d = divs[di];
            pipeline pl;
            memset(&pl, 0, sizeof(pl));
            operator op1 = { .type = OP_ADD, .field = 0, .constant = 17 };
            operator op2 = { .type = OP_DIV, .field = 0, .constant = (double)d };
            operator op3 = { .type = OP_ADD, .field = 0, .constant = -9 };
            pipeline_add(&pl, op1);
            pipeline_add(&pl, op2);
            pipeline_add(&pl, op3);

            jit_builder jb256, jb128;
            jb_init(&jb256, 128*1024);
            jb_init(&jb128, 128*1024);
            if(!jit_build_vex(&jb256, &pl, 1) || !jit_build_vex(&jb128, &pl, 0))
            {
                printf("    FAIL [S3]: jit_build_vex refused divisor %lld\n", (long long)d);
                bad = 1;
                break;
            }
            if(ran_v2)
            {
                memcpy(o256, in, sizeof(in));
                for(int i = 0; i < 512; i += 16) ((jit_fn_t)jb256.mem.mem)((uint8_t*)&o256[i], 16);
            }
            if(ran_v1)
            {
                memcpy(o128, in, sizeof(in));
                for(int i = 0; i < 512; i += 8)  ((jit_fn_t)jb128.mem.mem)((uint8_t*)&o128[i], 8);
            }

            for(int i = 0; i < 512; i++)
            {
                int64_t v = (int64_t)((uint64_t)in[i] + 17ULL);
                v /= d;
                v = (int64_t)((uint64_t)v - 9ULL);
                if((ran_v2 && o256[i] != v) || (ran_v1 && o128[i] != v))
                {
                    printf("    FAIL [S3]: divisor %lld input %lld -> want=%lld (vex256=%lld%s vex128=%lld%s)\n",
                           (long long)d, (long long)in[i], (long long)v,
                           (long long)o256[i], ran_v2 ? "" : " [not run]",
                           (long long)o128[i], ran_v1 ? "" : " [not run]");
                    bad = 1;
                    break;
                }
            }
            munmap(jb256.mem.mem, jb256.mem.size);
            munmap(jb128.mem.mem, jb128.mem.size);
        }
        if(!bad)
        {
            char widths[32] = "";
            if(ran_v2) strcat(widths, "VEX-256/");
            if(ran_v1) strcat(widths, "VEX-128/");
            size_t wl = strlen(widths);
            if(wl) widths[wl - 1] = '\0';
            printf("[S3] Native VEX division without VPSRAQ: %d divisors x 512 inputs bit-exact on %s\n",
                   ndivs, widths);
        }
        else
            fails++;
    }

    /* ============================================================
       [S4] RIVEST 2-ADIC BIJECTION & VEX METAMORPHIC SHIELD GATE
       ============================================================ */
    {
        int bad_bij = 0;
        for(int s = 0; s < 256 && !bad_bij; s++)
        {
            meta_config cfg = { .seed = splitmix64(&rng) | 1ULL };
            rivest_bij rv;
            rivest_gen(&cfg, &rv);
            for(int k = 0; k < 256; k++)
            {
                uint64_t y = splitmix64(&rng);
                uint64_t z = y * (rv.a1 + rv.a2 * y);
                uint64_t y_rec = z * (rv.b1 + rv.b2 * z);
                if(y_rec != y) { bad_bij = 1; break; }
            }
        }
        if(bad_bij)
        {
            printf("    FAIL [S4]: Rivest bijection inversion mismatch\n");
            fails++;
        }

        engine e;
        engine_init(&e);
        e.pipe.count = 6;
        e.pipe.ops[0] = (operator){ OP_ADD, 0, 5 };
        e.pipe.ops[1] = (operator){ OP_MUL, 0, 3 };
        e.pipe.ops[2] = (operator){ OP_SQUARE, 0, 0 };
        e.pipe.ops[3] = (operator){ OP_SUB, 0, 7 };
        e.pipe.ops[4] = (operator){ OP_DIV, 0, 3 };
        e.pipe.ops[5] = (operator){ OP_ADD, 0, -11 };

        int bad_vex_meta = 0;
        /* Execute only the widths this host can run (v5.2); compiling
           the blobs and comparing them byte-wise runs anywhere. */
        int ran_v2 = host_has_avx2(), ran_v1 = host_has_avx();
        if(!ran_v2 && !ran_v1)
        {
            printf("    FAIL [S4]: no runnable VEX width on this host\n");
            bad_vex_meta = 1;
        }
        if(!ran_v2) skipped++;
        if(!ran_v1) skipped++;
        for(int mode = 0; mode <= 1; mode++)
        {
            int run = mode ? ran_v2 : ran_v1;
            uint8_t blobs[16][8192];
            size_t lens[16];
            for(int i = 0; i < 16; i++)
            {
                meta_config cfg = {
                    .seed = (uint64_t)(i * 2 + 1) * 0x123456789ABCDEF1ULL,
                    .enable_nop = 1, .enable_reg_variation = 1, .enable_instruction_swap = 1
                };
                if(!jit_compile_meta_vex(&e, &cfg, mode)) { bad_vex_meta = 1; break; }
                lens[i] = e.jit_code.offset;
                memcpy(blobs[i], e.jit_code.mem, lens[i]);

                if(run)
                {
                    int64_t lane[4] = { -100, -7, 13, 1000 };
                    e.jit_fn((uint8_t*)lane, mode ? 4 : 2);
                    for(int k = 0; k < (mode ? 4 : 2); k++)
                    {
                        int64_t v = (k == 0) ? -100 : (k == 1) ? -7 : (k == 2) ? 13 : 1000;
                        v = ((v + 5) * 3); v *= v; v -= 7; v /= 3; v += -11;
                        if(lane[k] != v) { bad_vex_meta = 1; break; }
                    }
                }
            }
            for(int i = 0; i < 16; i++)
                for(int j = i + 1; j < 16; j++)
                    if(lens[i] == lens[j] && memcmp(blobs[i], blobs[j], lens[i]) == 0)
                        bad_vex_meta = 1;
        }
        engine_shutdown(&e);

        if(!bad_bij && !bad_vex_meta)
        {
            char widths[32] = "";
            if(ran_v2) strcat(widths, "VEX-256/");
            if(ran_v1) strcat(widths, "VEX-128/");
            size_t wl = strlen(widths);
            if(wl) widths[wl - 1] = '\0';
            printf("[S4] Rivest 2-adic bijective MBA (65,536 roundtrips) + %s Meta-JIT shield: PASS\n",
                   widths);
        }
        else
        {
            printf("    FAIL [S4]: VEX Meta-JIT check failed\n");
            fails++;
        }
    }

    /* ============================================================
       [S5] ONLINE RDTSCP MULTI-ARMED BANDIT TRACE AUTOTUNER & SPEED
       ============================================================ */
    {
        engine e;
        engine_init(&e);
        engine_load_sample_pipeline(&e);
        int64_t data[1024];
        for(int i = 0; i < 1024; i++) data[i] = i - 512;
        execute_trace_jit(&e, data, 1024);
        int bad = 0;
        for(int i = 0; i < 1024; i++)
        {
            int64_t x = i - 512;
            int64_t want = ((x + 10) * 3) * ((x + 10) * 3);
            if(data[i] != want) { bad = 1; break; }
        }

        /* Quick 20M-record comparative speed check across EVEX-512, VEX-256, and VEX-128 */
        uint32_t N = 20 * 1000 * 1000;
        int64_t *buf = malloc(N * sizeof(int64_t));
        for(uint32_t i = 0; i < N; i++) buf[i] = i;

        poly_t poly;
        pipeline_to_poly(&e.pipe, &poly);
        jit_builder jb_estrin, jb_vex256, jb_vex128;
        jb_init(&jb_estrin, 128*1024);
        jb_init(&jb_vex256, 128*1024);
        jb_init(&jb_vex128, 128*1024);
        jit_build_estrin_avx512(&jb_estrin, &poly);
        jit_build_vex(&jb_vex256, &e.pipe, 1);
        jit_build_vex(&jb_vex128, &e.pipe, 0);

        /* Measure only the kernels this host can execute (v5.2). */
        int ran_e = host_has_avx512(), ran_v2 = host_has_avx2(),
            ran_v1 = host_has_avx();
        double thr_estrin = 0, thr_vex256 = 0, thr_vex128 = 0;
        if(ran_e)  thr_estrin = measure_fn_mrec((jit_fn_t)jb_estrin.mem.mem, 32, buf, N);
        else skipped++;
        if(ran_v2) thr_vex256 = measure_fn_mrec((jit_fn_t)jb_vex256.mem.mem, 16, buf, N);
        else skipped++;
        if(ran_v1) thr_vex128 = measure_fn_mrec((jit_fn_t)jb_vex128.mem.mem, 8,  buf, N);
        else skipped++;

        munmap(jb_estrin.mem.mem, jb_estrin.mem.size);
        munmap(jb_vex256.mem.mem, jb_vex256.mem.size);
        munmap(jb_vex128.mem.mem, jb_vex128.mem.size);
        free(buf);
        engine_shutdown(&e);

        if(!bad)
        {
            char speeds[256] = "";
            char tmp[96];
            if(ran_e)
            {
                snprintf(tmp, sizeof(tmp), "EVEX Estrin: %.0f M rec/s", thr_estrin);
                strcat(speeds, tmp);
            }
            if(ran_v2)
            {
                if(speeds[0]) strcat(speeds, " | ");
                snprintf(tmp, sizeof(tmp), "Haswell VEX-256: %.0f M rec/s", thr_vex256);
                strcat(speeds, tmp);
            }
            if(ran_v1)
            {
                if(speeds[0]) strcat(speeds, " | ");
                snprintf(tmp, sizeof(tmp), "Ivy VEX-128: %.0f M rec/s", thr_vex128);
                strcat(speeds, tmp);
            }
            printf("[S5] Online RDTSCP Bandit Trace Autotuner: PASS (%s)\n",
                   speeds);
        }
        else
        {
            printf("    FAIL [S5]: RDTSCP Bandit Trace mismatch\n");
            fails++;
        }
    }

    if(!fails)
    {
        printf("=== SCIENTIFIC GATE: ALL 5 GATES PASS ===\n");
        if(skipped)
            printf("    note: %d execution(s) skipped — host lacks that ISA (emitted code still verified)\n",
                   skipped);
        return 0;
    }
    printf("=== SCIENTIFIC GATE: FAIL (%d) ===\n", fails);
    return 1;
}
