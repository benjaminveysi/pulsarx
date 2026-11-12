/* ============================================================================
 * PulsarXASM (pxasm) - High-Performance Execution Engine
 * ============================================================================
 * PulsarXASM (pxasm), a successor project to the VDC ecosystem's XASM, has been
 * fundamentally re-architected to achieve extreme low-latency and bare-metal
 * execution capabilities. This engine embodies a rigorous commitment to
 * optimizing hardware utilization to its utmost potential.
 *
 * Copyright (c) 2026 Vaysi Development Center (VDC). All rights reserved.
 *
 * VDC is committed to fostering open ecosystems and collaborative engineering,
 * and to developing software solutions that avoid unnecessary complexity.
 *
 * License / Conditions of Use:
 * - You are hereby permitted to use, study, and modify this codebase, provided
 *   that this copyright notice and any other proprietary notices are retained
 *   in all copies or substantial portions thereof.
 * - You are not permitted to use this software in any form without proper
 *   attribution to Vaysi Development Center (VDC) and the project name
 *   "PulsarXASM (pxasm)".
 * - Commercial use, including but not limited to selling, sublicensing for
 *   consideration, distributing as part of a paid offering, or using for
 *   any revenue-generating purpose, is strictly prohibited unless a separate
 *   written license is obtained from VDC.
 *
 * Disclaimer:
 * This software is provided "AS IS," without warranty of any kind, whether
 * express or implied. To the maximum extent permitted by applicable law, VDC
 * disclaims all liability arising from its use, including without limitation
 * liability for any direct, indirect, incidental, special, consequential, or
 * punitive damages.
 *
 * For inquiries, licensing, or technical/legal correspondence:
 * hello@vdc.systems
 * ============================================================================ */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sched.h>
#include <math.h>
#include <x86intrin.h>
#include "pulsar_numa_compat.h"

/* -march decides which width the JIT may use; the CPU check just
   confirms the hardware actually has it. */
static inline int pulsar_use_avx512(void)
{
#ifdef __AVX512F__
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512dq");
#else
    return 0;
#endif
}

static inline int pulsar_use_avx2(void)
{
#if defined(__AVX512F__) || defined(__AVX2__)
    return __builtin_cpu_supports("avx2");
#else
    return 0;
#endif
}

/* ============================================================
   GLOBAL CONFIG
   ============================================================ */

#define PULSAR_MAX_FIELDS 64
#define PULSAR_MAX_PIPE   64
#define PULSAR_RING_SIZE  131072
#define PULSAR_WORKERS    32
#define PULSAR_ARENA_SIZE (1024ULL*1024ULL*512ULL)

#define PULSAR_JIT_THRESHOLD 10000

#define PULSAR_TRACE_THRESHOLD 5000

#define CACHELINE 64

/* ============================================================
   ALIGN
   ============================================================ */

static inline size_t align64(size_t v)
{
    return (v + 63) & ~63;
}

/* ============================================================
   ARENA ALLOCATOR (NUMA AWARE)
   ============================================================ */

typedef struct
{
    uint8_t *base;
    size_t size;
    size_t offset;

} pulsar_arena;

void arena_init(pulsar_arena *a,size_t size,int node)
{
    a->size=size;
    a->offset=0;

    a->base=numa_alloc_onnode(size,node);

    if(!a->base)
    {
        perror("arena alloc");
        exit(1);
    }
}

static inline void* arena_alloc(pulsar_arena *a,size_t size)
{
    size=align64(size);

    size_t off=__atomic_fetch_add(&a->offset,size,__ATOMIC_RELAXED);

    if(off+size>=a->size)
        return NULL;

    return a->base+off;
}

void arena_reset(pulsar_arena *a)
{
    a->offset=0;
}

/* ============================================================
   SCHEMA
   ============================================================ */

typedef enum
{
    FIELD_I64,
    FIELD_F64

} field_type;

typedef struct
{
    const char *name;
    field_type type;
    uint32_t offset;

} field;

typedef struct
{
    field fields[PULSAR_MAX_FIELDS];

    uint32_t field_count;

    uint32_t record_size;

} schema;

void schema_init(schema *s)
{
    memset(s,0,sizeof(*s));
}

void schema_add(schema *s,const char*name,field_type t)
{
    field *f=&s->fields[s->field_count++];

    f->name=name;
    f->type=t;
}

void schema_finalize(schema *s)
{
    uint32_t off=0;

    for(uint32_t i=0;i<s->field_count;i++)
    {
        s->fields[i].offset=off;

        off+=8;
    }

    s->record_size=off;
}

/* ============================================================
   RECORD
   ============================================================ */

typedef struct
{
    uint8_t *data;

} record;

static inline int64_t rec_get_i64(record *r,uint32_t off)
{
    return *(int64_t*)(r->data+off);
}

static inline void rec_set_i64(record *r,uint32_t off,int64_t v)
{
    *(int64_t*)(r->data+off)=v;
}

/* ============================================================
   LOCK FREE RING BUFFER
   ============================================================ */

typedef struct
{
    record *buffer[PULSAR_RING_SIZE];

    uint64_t head __attribute__((aligned(CACHELINE)));
    uint64_t tail __attribute__((aligned(CACHELINE)));

} ring;

/* Head/tail are MONOTONIC uint64 counters (never wrapped); the slot index
   is counter % PULSAR_RING_SIZE. head == tail  =>  empty,
   head - tail >= PULSAR_RING_SIZE  =>  full. */

void ring_init(ring *r)
{
    r->head=0;
    r->tail=0;
}

/* Single-producer push (engine_ingest is called from one thread).
   Returns 0 when the ring is full. */
int ring_push(ring *r,record *rec)
{
    uint64_t head=__atomic_load_n(&r->head,__ATOMIC_RELAXED);
    uint64_t tail=__atomic_load_n(&r->tail,__ATOMIC_ACQUIRE);

    if(head-tail>=PULSAR_RING_SIZE)
        return 0;

    r->buffer[head%PULSAR_RING_SIZE]=rec;

    /* Release: publishes the slot write to consumers. */
    __atomic_store_n(&r->head,head+1,__ATOMIC_RELEASE);

    return 1;
}

/* Multi-consumer pop: concurrent workers claim disjoint slots via CAS.
   The slot is read AFTER winning the CAS, so a preempted loser can
   never return a stale pointer (no ABA). Returns NULL when empty. */
record* ring_pop(ring *r)
{
    uint64_t tail=__atomic_load_n(&r->tail,__ATOMIC_RELAXED);

    for(;;)
    {
        /* Acquire: pairs with the producer's release-store of head,
           making the slot write visible before we read it. */
        uint64_t head=__atomic_load_n(&r->head,__ATOMIC_ACQUIRE);

        if(tail==head)
            return NULL;

        uint64_t next=tail+1;

        if(__atomic_compare_exchange_n(&r->tail,&tail,next,0,
                                      __ATOMIC_ACQ_REL,__ATOMIC_RELAXED))
        {
            return r->buffer[tail%PULSAR_RING_SIZE];
        }
        /* CAS failed: 'tail' now holds the winner's value; retry. */
    }
}

/* ============================================================
   AoSoA SLABS (§5)
   ------------------------------------------------------------
   The record path spends 2 arena_allocs per record (128 bytes for
   one 8-byte value, 93% waste) and the JIT path gathers 8 pointers
   into a temp before it can run. A slab packs 8 records' field f
   into one contiguous 64-byte vector, so the JIT loads it with a
   single vmovdqa64 and there is no gather and no scatter:
   field f of the 8 records is at data[f*8 .. f*8+7].
   ============================================================ */

#define SLAB_LANES 32

typedef struct
{
    uint32_t count;     /* records filled in this slab (0..SLAB_LANES) */
    int64_t  data[];     /* field f, lane l: data[f*SLAB_LANES + l] */
} __attribute__((aligned(64))) record_slab;

/* Field f's 8-lane vector inside a slab (one zmm load). */
static inline int64_t *slab_field(record_slab *s, uint32_t field)
{
    return &s->data[(size_t)field * SLAB_LANES];
}

/* Ring of slab pointers (same lock-free discipline as `ring`). */
typedef struct
{
    record_slab *buffer[PULSAR_RING_SIZE];

    uint64_t head __attribute__((aligned(CACHELINE)));
    uint64_t tail __attribute__((aligned(CACHELINE)));

} slab_ring;

static inline void slab_ring_init(slab_ring *r)
{
    r->head = 0;
    r->tail = 0;
}

static inline int slab_ring_push(slab_ring *r, record_slab *s)
{
    uint64_t head = __atomic_load_n(&r->head, __ATOMIC_RELAXED);
    uint64_t tail = __atomic_load_n(&r->tail, __ATOMIC_ACQUIRE);
    if(head - tail >= PULSAR_RING_SIZE)
        return 0;
    r->buffer[head % PULSAR_RING_SIZE] = s;
    __atomic_store_n(&r->head, head + 1, __ATOMIC_RELEASE);
    return 1;
}

static inline record_slab *slab_ring_pop(slab_ring *r)
{
    uint64_t tail = __atomic_load_n(&r->tail, __ATOMIC_RELAXED);
    for(;;)
    {
        uint64_t head = __atomic_load_n(&r->head, __ATOMIC_ACQUIRE);
        if(tail == head)
            return NULL;
        uint64_t next = tail + 1;
        if(__atomic_compare_exchange_n(&r->tail, &tail, next, 0,
                                       __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
            return r->buffer[tail % PULSAR_RING_SIZE];
    }
}

/* ============================================================
   EXECUTABLE MEMORY
   ============================================================ */

typedef struct
{
    uint8_t *mem;
    size_t size;
    size_t offset;

} jit_mem;

/* §6 MPK forward declarations (defined just below jit_mem_init). */
static void pulsar_mpk_tag(void *addr, size_t len);
static void pulsar_mpk_unlock(void);
static void pulsar_mpk_lock(void);

void jit_mem_init(jit_mem *j,size_t size)
{
    j->size=size;
    j->offset=0;

    j->mem=mmap(NULL,size,
        PROT_READ|PROT_WRITE|PROT_EXEC,
        MAP_PRIVATE|MAP_ANONYMOUS,
        -1,0);

    if(j->mem==MAP_FAILED)
    {
        perror("jit mmap");
        exit(1);
    }

    /* §6 MPK: tag the JIT pages with the (shared) protection key,
       when the CPU and kernel support it. The pages stay RWX at the
       page-table level; the per-thread PKRU register enforces W^X
       (writable while emitting, exec-only while running). */
    pulsar_mpk_tag(j->mem, j->size);
}

/* ============================================================
   MPK (MEMORY PROTECTION KEYS) — §6
   ------------------------------------------------------------
   One pkey for all JIT pages (there are only 16). PKRU is
   per-thread, so the emitting thread can unlock (writable) while
   worker threads stay locked (exec-only). This is defense in
   depth alongside the metamorphic shield, not a hard boundary:
   any code can WRPKRU, and mprotect is still the real W^X.
   On hardware without PKU (e.g. Haswell) pkey_alloc fails and
   everything below is a no-op (plain RWX, as before).
   ============================================================ */

/* -2 = not yet probed, -1 = unavailable, >= 0 = the pkey. */
static int g_mpk_pkey = -2;

/* Read the current thread's PKRU (RDPKRU, ECX=0). */
static inline uint32_t pulsar_rdpkru(void)
{
    uint32_t lo, hi;
    __asm__ volatile("rdpkru" : "=a"(lo), "=d"(hi) : "c"(0));
    return lo;
}

/* Write the current thread's PKRU (WRPKRU, ECX=0, EDX=0). */
static inline void pulsar_wrpkru(uint32_t v)
{
    __asm__ volatile("wrpkru" : : "a"(v), "c"(0), "d"(0));
}

/* Lazily allocate the shared JIT pkey. Returns the pkey, or -1. */
static int pulsar_mpk_pkey(void)
{
    if(g_mpk_pkey == -2)
    {
        int p = pkey_alloc(0, 0);
        g_mpk_pkey = p;   /* -1 when the kernel/CPU has no MPK */
    }
    return g_mpk_pkey;
}

/* Tag a JIT mapping with the pkey (no-op without MPK). */
static void pulsar_mpk_tag(void *addr, size_t len)
{
    int p = pulsar_mpk_pkey();
    if(p < 0) return;
    /* Keep RWX at the PTE level; PKRU does the W^X enforcement. */
    pkey_mprotect(addr, len, PROT_READ | PROT_WRITE | PROT_EXEC, p);
}

/* Make the JIT pages writable for the current thread (clear the
   write-disable bit for our pkey). Called before emitting code. */
static void pulsar_mpk_unlock(void)
{
    int p = pulsar_mpk_pkey();
    if(p < 0) return;
    uint32_t pk = pulsar_rdpkru();
    pk &= ~(1u << (2 * p + 1));   /* clear write-disable */
    pulsar_wrpkru(pk);
}

/* Make the JIT pages exec-only for the current thread (set the
   write-disable bit for our pkey). Called after installing code,
   and by pool workers at startup so they can execute but never
   write the JIT pages. */
static void pulsar_mpk_lock(void)
{
    int p = pulsar_mpk_pkey();
    if(p < 0) return;
    uint32_t pk = pulsar_rdpkru();
    pk |= (1u << (2 * p + 1));    /* set write-disable */
    pulsar_wrpkru(pk);
}

static inline void emit8(jit_mem *j,uint8_t v)
{
    j->mem[j->offset++]=v;
}

static inline void emit32(jit_mem *j,uint32_t v)
{
    memcpy(j->mem+j->offset,&v,4);
    j->offset+=4;
}

static inline void emit64(jit_mem *j,uint64_t v)
{
    memcpy(j->mem+j->offset,&v,8);
    j->offset+=8;
}

/* ============================================================
   METAMORPH RNG
   ============================================================ */

static inline uint64_t pulsar_rng(uint64_t *s)
{
    *s^=*s<<13;
    *s^=*s>>7;
    *s^=*s<<17;

    return *s;
}

/* ============================================================
   PIPELINE OPS
   ============================================================ */

typedef enum
{
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_SQUARE,
    /* §5 k-mask ops: OP_CMP_GT sets a per-lane predicate (pred), and
       the immediately following op executes under it, then pred
       resets to all-true. OP_SELECT and OP_BIT_XOR are the two ops
       that most often consume the mask, but ANY op after a CMP_GT
       is masked. */
    OP_CMP_GT,
    OP_SELECT,
    OP_BIT_XOR

} op_type;

typedef struct
{
    op_type type;

    uint32_t field;

    double constant;

} operator;

/* ============================================================
   PIPELINE
   ============================================================ */

typedef struct
{
    operator ops[PULSAR_MAX_PIPE];

    uint32_t count;

} pipeline;

void pipeline_add(pipeline *p,operator op)
{
    p->ops[p->count++]=op;
}

/* ============================================================
   ENGINE
   ============================================================ */

/* JIT function ABI (honest version): every backend emits code that expects
   RDI = pointer to the record's DATA (r->data), operating on the int64 at
   [RDI]. The scalar/meta backends process one record; the AVX-512 backend
   processes 8 packed int64s (64 bytes) at [RDI]. */
typedef void (*jit_fn_t)(uint8_t *data, uint64_t n);

/* Forward declaration: the free-threaded pool (defined after the
   executors) lives on the engine so strategies can dispatch to it. */
typedef struct pulsar_pool pulsar_pool;

typedef struct
{
    schema schema;

    pipeline pipe;

    ring queue;

    pulsar_arena arena;

    uint64_t exec_count;

    jit_fn_t jit_fn;

    /* Last compiled code buffer (owned here so recompiles don't leak),
       plus the vector width the compiled function expects (1 or 8). */
    jit_mem  jit_code;
    int      jit_code_valid;
    int      jit_width;

    /* Non-temporal variant of the 32-wide AVX-512 kernel (v3): same
       code, but stores with VMOVNTDQ for arrays larger than L3.
       NULL when the v3 kernel is not in use. */
    jit_mem  jit_code_nt;
    int      jit_code_nt_valid;
    jit_fn_t jit_fn_nt;

    /* Set to 1 to use the NT kernel for large streaming arrays.
       Default 0: the NT kernel is implemented and verified correct,
       but it measures SLOWER than the cached v3 kernel on this VM
       (and on machines with a large L3), so it is opt-in, not the
       default. Flip it on hardware where streaming stores win. */
    int      jit_nt_enable;

    /* Persistent thread pool for free-threaded (Python 3.14 style)
       parallel execution. NULL = serial mode. Created/resized with
       engine_set_threads(), released with engine_shutdown(). */
    pulsar_pool *pool;

    /* Streaming worker threads (the ring ingest path). Created with
       engine_start_workers(), stopped with engine_stop_workers().
       workers_stop is polled by the workers; set before joining. */
    pthread_t *worker_threads;
    int        worker_count;
    volatile int workers_stop;

    /* §5 AoSoA slab path: slabs being filled and queued. slab_cur is
       the slab currently being filled (NULL until the first ingest);
       full slabs go to slab_queue. The JIT runs directly on a slab's
       field vector (no gather/scatter). */
    slab_ring   slab_queue;
    record_slab *slab_cur;

} engine;

void engine_init(engine *e)
{
    /* Zero everything first: the pipeline, ring cursors and JIT cache
       must all start clean (a reused stack slot otherwise accumulates
       stale ops across init calls). */
    memset(e,0,sizeof(*e));

    schema_init(&e->schema);

    /* Default schema: one int64 field ("value" at offset 0). The ring
       ingest path (engine_ingest) allocates record data by
       schema.record_size -- without a field it would allocate 0 bytes. */
    schema_add(&e->schema,"value",FIELD_I64);
    schema_finalize(&e->schema);

    ring_init(&e->queue);

    slab_ring_init(&e->slab_queue);
    e->slab_cur = NULL;

    arena_init(&e->arena,PULSAR_ARENA_SIZE,0);

    e->exec_count=0;

    e->jit_fn=NULL;
    e->jit_code_valid=0;
    e->jit_width=1;
}

/* Sample pipeline used by the functional test and the benchmark suite:
   field0 = ((field0 + 10) * 3)^2  — exercises ADD, MUL and SQUARE,
   i.e. the exact op mix the fusion pass and all JIT backends handle. */
void engine_load_sample_pipeline(engine *e)
{
    operator op;
    op.field = 0;

    op.type = OP_ADD;
    op.constant = 10;
    pipeline_add(&e->pipe, op);

    op.type = OP_MUL;
    op.constant = 3;
    pipeline_add(&e->pipe, op);

    op.type = OP_SQUARE;
    op.constant = 0;
    pipeline_add(&e->pipe, op);
}

/* ============================================================
   INGEST
   ============================================================ */

void interpret(engine *e, record *r);
void engine_compile(engine *e);

void engine_ingest(engine *e,int64_t v)
{
    record *r=arena_alloc(&e->arena,sizeof(record));

    r->data=arena_alloc(&e->arena,e->schema.record_size);

    rec_set_i64(r,0,v);

    while(!ring_push(&e->queue,r));
}

/* AoSoA slab ingest (field 0): appends v to the current slab. When the
   slab fills (SLAB_LANES records) it is pushed to the slab queue and a
   fresh slab is started. One arena_alloc per slab (not per record).
   A partial slab is flushed by engine_run_jit_slab (it runs it too). */
void engine_ingest_slab(engine *e,int64_t v)
{
    if(!e->slab_cur)
    {
        size_t bytes = sizeof(record_slab) +
                       (size_t)e->schema.field_count * SLAB_LANES * sizeof(int64_t);
        e->slab_cur = arena_alloc(&e->arena, bytes);
        if(!e->slab_cur) return;   /* arena full */
        e->slab_cur->count = 0;
    }

    record_slab *s = e->slab_cur;
    s->data[s->count] = v;   /* field 0, lane = count */
    s->count++;

    if(s->count >= SLAB_LANES)
    {
        while(!slab_ring_push(&e->slab_queue, s));
        e->slab_cur = NULL;
    }
}

/* AoSoA slab run: pops slabs and runs the compiled JIT directly on each
   slab's field-0 vector (no gather, no scatter; the JIT writes in
   place). Full slabs from the queue first (older), then the partial
   current slab (newest). The JIT width must be 32 (v3) for a full
   slab; any other width, or a partial slab, runs lane by lane
   through the interpreter. */
void engine_run_jit_slab(engine *e)
{
    engine_compile(e);   /* ensure jit_fn/jit_width are ready */

    record_slab *s;
    while((s = slab_ring_pop(&e->slab_queue)))
    {
        e->exec_count += s->count;
        if(s->count == SLAB_LANES &&
           (e->jit_width == 8 || e->jit_width == 16 || e->jit_width == 32))
        {
            /* v3 (32) or v4 VEX (16 Haswell / 8 Ivy) kernel, straight
               on the slab in width-sized chunks — no gather/scatter. */
            for(uint32_t off = 0; off < SLAB_LANES; off += (uint32_t)e->jit_width)
                e->jit_fn((uint8_t*)slab_field(s, 0) + off * 8, (uint64_t)e->jit_width);
        }
        else
        {
            /* Any other width, or a short slab: interpret lane by lane. */
            for(uint32_t l = 0; l < s->count; l++)
            {
                record r;
                r.data = (uint8_t*)&s->data[l];
                interpret(e, &r);
            }
        }
    }

    /* Partial current slab (not yet queued). */
    if(e->slab_cur && e->slab_cur->count > 0)
    {
        record_slab *cur = e->slab_cur;
        e->exec_count += cur->count;
        for(uint32_t l = 0; l < cur->count; l++)
        {
            record r;
            r.data = (uint8_t*)&cur->data[l];
            interpret(e, &r);
        }
        cur->count = 0;
    }
}

/* ============================================================
   SIMPLE INTERPRETER
   ============================================================ */

void interpret(engine *e,record *r)
{
    /* §5 predicate mask: -1 = lane active, 0 = lane masked off.
       OP_CMP_GT sets it; the immediately following op consumes it
       (executes only where pred is true) and resets it to all-true.
       With no preceding CMP_GT, pred is all-true so every op applies
       normally; OP_SELECT with no preceding CMP_GT just sets v = c. */
    int64_t pred = -1;
    for(uint32_t i=0;i<e->pipe.count;i++)
    {
        operator *op=&e->pipe.ops[i];

        /* uint64_t on purpose: ADD/SUB/MUL wrap mod 2^64, like the
           hardware does. As signed int64_t this was UB and GCC -O2
           folded (INT64_MAX + 10) * 3 to 0. DIV/CMP_GT still signed. */
        uint64_t v=(uint64_t)rec_get_i64(r,e->schema.fields[op->field].offset);

        switch(op->type)
        {
            case OP_ADD: if(pred) v+=(uint64_t)(int64_t)op->constant; pred = -1; break;
            case OP_SUB: if(pred) v-=(uint64_t)(int64_t)op->constant; pred = -1; break;
            case OP_MUL: if(pred) v*=(uint64_t)(int64_t)op->constant; pred = -1; break;
            /* Integer division (truncation toward zero), matching every
               other backend. Divisor 0 or INT64_MIN/-1 traps with SIGFPE,
               exactly like the JIT's idiv. */
            case OP_DIV: if(pred) v=(uint64_t)((int64_t)v/(int64_t)op->constant); pred = -1; break;
            case OP_SQUARE: if(pred) v*=v; pred = -1; break;
            case OP_CMP_GT: pred = ((int64_t)v > (int64_t)op->constant) ? -1 : 0; break;
            case OP_SELECT: v = pred ? (uint64_t)(int64_t)op->constant : v; pred = -1; break;
            case OP_BIT_XOR: if(pred) v ^= (uint64_t)(int64_t)op->constant; pred = -1; break;
        }

        rec_set_i64(r,e->schema.fields[op->field].offset,(int64_t)v);
    }
}

/* ============================================================
   SIMD CONFIG
   ============================================================ */

#define SIMD_WIDTH 8
#define SIMD_PREFETCH 64

/* ============================================================
   BATCH STRUCTURE
   ============================================================ */

typedef struct
{
    record *items[SIMD_WIDTH];

} record_batch;

/* ============================================================
   PREFETCH
   ============================================================ */

static inline void pulsar_prefetch(void *p)
{
    _mm_prefetch(p,_MM_HINT_T0);
}

/* ============================================================
   LOAD / STORE AVX512
   ------------------------------------------------------------
   Only compiles when the toolchain targets AVX-512. Haswell gets
   the AVX2 variant, Ivy Bridge the SSE4.2 one, and a machine with
   no SSE4.2 at all falls back to the interpreter for SIMD.
   ============================================================ */

#ifdef __AVX512F__

static inline __m512i load_i64_batch(record_batch *b,uint32_t off)
{
    int64_t tmp[SIMD_WIDTH];

    for(int i=0;i<SIMD_WIDTH;i++)
        tmp[i]=rec_get_i64(b->items[i],off);

    return _mm512_loadu_si512(tmp);
}

static inline void store_i64_batch(record_batch *b,uint32_t off,__m512i v)
{
    int64_t tmp[SIMD_WIDTH];

    _mm512_storeu_si512(tmp,v);

    for(int i=0;i<SIMD_WIDTH;i++)
        rec_set_i64(b->items[i],off,tmp[i]);
}

/* ============================================================
   SIMD OPERATORS
   ============================================================ */

static inline __m512i simd_add(__m512i v,int64_t c)
{
    __m512i k=_mm512_set1_epi64(c);

    return _mm512_add_epi64(v,k);
}

static inline __m512i simd_sub(__m512i v,int64_t c)
{
    __m512i k=_mm512_set1_epi64(c);

    return _mm512_sub_epi64(v,k);
}

static inline __m512i simd_mul(__m512i v,int64_t c)
{
    __m512i k=_mm512_set1_epi64(c);

    return _mm512_mullo_epi64(v,k);
}

static inline __m512i simd_square(__m512i v)
{
    return _mm512_mullo_epi64(v,v);
}

/* ============================================================
   SIMD PIPELINE EXECUTION
   ============================================================ */

void simd_execute(engine *e,record_batch *b)
{
    uint32_t off=e->schema.fields[0].offset;

    __m512i vec=load_i64_batch(b,off);
    /* §5 predicate mask: 0xFF = all lanes active. CMP_GT sets it;
       the next op runs under it (merging) and resets it to 0xFF. */
    __mmask8 pred = 0xFF;

    for(uint32_t i=0;i<e->pipe.count;i++)
    {
        operator *op=&e->pipe.ops[i];

        switch(op->type)
        {
            case OP_ADD:
                vec=_mm512_mask_add_epi64(vec, pred, vec,
                        _mm512_set1_epi64((int64_t)op->constant));
                pred = 0xFF;
                break;

            case OP_SUB:
                vec=_mm512_mask_sub_epi64(vec, pred, vec,
                        _mm512_set1_epi64((int64_t)op->constant));
                pred = 0xFF;
                break;

            case OP_MUL:
                vec=_mm512_mask_mullo_epi64(vec, pred, vec,
                        _mm512_set1_epi64((int64_t)op->constant));
                pred = 0xFF;
                break;

            case OP_SQUARE:
                vec=_mm512_mask_mullo_epi64(vec, pred, vec, vec);
                pred = 0xFF;
                break;

            case OP_DIV:
                /* No vector integer divide: spill the in-flight vector,
                   divide each lane scalar (only where pred is set),
                   reload. */
                store_i64_batch(b,off,vec);
                {
                    int64_t c = (int64_t)op->constant;
                    for(int k = 0; k < SIMD_WIDTH; k++)
                    {
                        if(!(pred & (1u << k))) continue;
                        record *r = b->items[k];
                        int64_t v = rec_get_i64(r,off);
                        rec_set_i64(r,off,v/c);
                    }
                }
                vec=load_i64_batch(b,off);
                pred = 0xFF;
                break;

            case OP_CMP_GT:
                pred=_mm512_cmpgt_epi64_mask(vec,
                        _mm512_set1_epi64((int64_t)op->constant));
                break;

            case OP_SELECT:
                vec=_mm512_mask_mov_epi64(vec, pred,
                        _mm512_set1_epi64((int64_t)op->constant));
                pred = 0xFF;
                break;

            case OP_BIT_XOR:
                vec=_mm512_mask_xor_epi64(vec, pred, vec,
                        _mm512_set1_epi64((int64_t)op->constant));
                pred = 0xFF;
                break;

            default:
                break;
        }
    }

    store_i64_batch(b,off,vec);
}

#elif defined(__AVX2__)
/* AVX2 (Haswell) SIMD: no 64-bit multiply and no masking — the same
   VPMULUDQ synthesis as the VEX JIT, and VPBLENDVB for the §5
   predicate. The 8-lane batch runs as two 4-lane chunks. */
static inline __m256i simd256_mul64(__m256i x, __m256i c)
{
    __m256i xhi = _mm256_srli_epi64(x, 32);
    __m256i chi = _mm256_srli_epi64(c, 32);
    __m256i p0 = _mm256_mul_epu32(x, c);    /* xlo*clo */
    __m256i p1 = _mm256_mul_epu32(x, chi);  /* xlo*chi */
    __m256i p2 = _mm256_mul_epu32(xhi, c);  /* xhi*clo */
    __m256i mid = _mm256_add_epi64(p1, p2);
    mid = _mm256_slli_epi64(mid, 32);
    return _mm256_add_epi64(p0, mid);
}

void simd_execute(engine *e,record_batch *b)
{
    uint32_t off=e->schema.fields[0].offset;

    for(int chunk = 0; chunk < 2; chunk++)
    {
        int64_t tmp[4];
        for(int k = 0; k < 4; k++)
            tmp[k] = rec_get_i64(b->items[chunk * 4 + k], off);
        __m256i vec = _mm256_loadu_si256((__m256i*)tmp);
        /* §5 predicate as a full vector mask (all-ones = active). */
        __m256i pred = _mm256_set1_epi64x(-1);

        for(uint32_t i=0;i<e->pipe.count;i++)
        {
            operator *op=&e->pipe.ops[i];
            __m256i cvec;
            switch(op->type)
            {
                case OP_ADD:
                    cvec = _mm256_set1_epi64x((int64_t)op->constant);
                    vec = _mm256_blendv_epi8(vec, _mm256_add_epi64(vec, cvec), pred);
                    pred = _mm256_set1_epi64x(-1);
                    break;
                case OP_SUB:
                    cvec = _mm256_set1_epi64x((int64_t)op->constant);
                    vec = _mm256_blendv_epi8(vec, _mm256_sub_epi64(vec, cvec), pred);
                    pred = _mm256_set1_epi64x(-1);
                    break;
                case OP_MUL:
                    cvec = _mm256_set1_epi64x((int64_t)op->constant);
                    vec = _mm256_blendv_epi8(vec, simd256_mul64(vec, cvec), pred);
                    pred = _mm256_set1_epi64x(-1);
                    break;
                case OP_SQUARE:
                    vec = _mm256_blendv_epi8(vec, simd256_mul64(vec, vec), pred);
                    pred = _mm256_set1_epi64x(-1);
                    break;
                case OP_DIV: {
                    /* No vector integer divide: spill, divide scalar
                       on the active lanes, reload. */
                    int64_t c = (int64_t)op->constant;
                    _mm256_storeu_si256((__m256i*)tmp, vec);
                    int64_t pm[4];
                    _mm256_storeu_si256((__m256i*)pm, pred);
                    for(int k = 0; k < 4; k++)
                    {
                        if(!pm[k]) continue;
                        tmp[k] /= c;
                    }
                    vec = _mm256_loadu_si256((__m256i*)tmp);
                    pred = _mm256_set1_epi64x(-1);
                    break;
                }
                case OP_CMP_GT:
                    pred = _mm256_cmpgt_epi64(vec,
                            _mm256_set1_epi64x((int64_t)op->constant));
                    break;
                case OP_SELECT:
                    vec = _mm256_blendv_epi8(vec,
                            _mm256_set1_epi64x((int64_t)op->constant), pred);
                    pred = _mm256_set1_epi64x(-1);
                    break;
                case OP_BIT_XOR:
                    cvec = _mm256_set1_epi64x((int64_t)op->constant);
                    vec = _mm256_blendv_epi8(vec, _mm256_xor_si256(vec, cvec), pred);
                    pred = _mm256_set1_epi64x(-1);
                    break;
                default:
                    break;
            }
        }

        _mm256_storeu_si256((__m256i*)tmp, vec);
        for(int k = 0; k < 4; k++)
            rec_set_i64(b->items[chunk * 4 + k], off, tmp[k]);
    }
}

#elif defined(__SSE4_2__)
/* SSE4.2: real 2-lane vector executor, not the interpreter.
   Same VPMULUDQ synthesis as the VEX JIT, at 128 bit; the
   8-lane batch runs as four 2-lane chunks. */
static inline __m128i simd128_mul64(__m128i x, __m128i c)
{
    __m128i xhi = _mm_srli_epi64(x, 32);
    __m128i chi = _mm_srli_epi64(c, 32);
    __m128i p0 = _mm_mul_epu32(x, c);    /* xlo*clo */
    __m128i p1 = _mm_mul_epu32(x, chi);  /* xlo*chi */
    __m128i p2 = _mm_mul_epu32(xhi, c);  /* xhi*clo */
    __m128i mid = _mm_add_epi64(p1, p2);
    mid = _mm_slli_epi64(mid, 32);
    return _mm_add_epi64(p0, mid);
}

void simd_execute(engine *e,record_batch *b)
{
    uint32_t off=e->schema.fields[0].offset;

    for(int chunk = 0; chunk < 4; chunk++)
    {
        int64_t tmp[2];
        for(int k = 0; k < 2; k++)
            tmp[k] = rec_get_i64(b->items[chunk * 2 + k], off);
        __m128i vec = _mm_loadu_si128((__m128i*)tmp);
        /* §5 predicate as a full vector mask (all-ones = active). */
        __m128i pred = _mm_set1_epi64x(-1);

        for(uint32_t i=0;i<e->pipe.count;i++)
        {
            operator *op=&e->pipe.ops[i];
            __m128i cvec;
            switch(op->type)
            {
                case OP_ADD:
                    cvec = _mm_set1_epi64x((int64_t)op->constant);
                    vec = _mm_blendv_epi8(vec, _mm_add_epi64(vec, cvec), pred);
                    pred = _mm_set1_epi64x(-1);
                    break;
                case OP_SUB:
                    cvec = _mm_set1_epi64x((int64_t)op->constant);
                    vec = _mm_blendv_epi8(vec, _mm_sub_epi64(vec, cvec), pred);
                    pred = _mm_set1_epi64x(-1);
                    break;
                case OP_MUL:
                    cvec = _mm_set1_epi64x((int64_t)op->constant);
                    vec = _mm_blendv_epi8(vec, simd128_mul64(vec, cvec), pred);
                    pred = _mm_set1_epi64x(-1);
                    break;
                case OP_SQUARE:
                    vec = _mm_blendv_epi8(vec, simd128_mul64(vec, vec), pred);
                    pred = _mm_set1_epi64x(-1);
                    break;
                case OP_DIV: {
                    /* No vector integer divide: spill, divide scalar
                       on the active lanes, reload. */
                    int64_t c = (int64_t)op->constant;
                    _mm_storeu_si128((__m128i*)tmp, vec);
                    int64_t pm[2];
                    _mm_storeu_si128((__m128i*)pm, pred);
                    for(int k = 0; k < 2; k++)
                    {
                        if(!pm[k]) continue;
                        tmp[k] /= c;
                    }
                    vec = _mm_loadu_si128((__m128i*)tmp);
                    pred = _mm_set1_epi64x(-1);
                    break;
                }
                case OP_CMP_GT:
                    pred = _mm_cmpgt_epi64(vec,
                            _mm_set1_epi64x((int64_t)op->constant));
                    break;
                case OP_SELECT:
                    vec = _mm_blendv_epi8(vec,
                            _mm_set1_epi64x((int64_t)op->constant), pred);
                    pred = _mm_set1_epi64x(-1);
                    break;
                case OP_BIT_XOR:
                    cvec = _mm_set1_epi64x((int64_t)op->constant);
                    vec = _mm_blendv_epi8(vec, _mm_xor_si128(vec, cvec), pred);
                    pred = _mm_set1_epi64x(-1);
                    break;
                default:
                    break;
            }
        }

        _mm_storeu_si128((__m128i*)tmp, vec);
        for(int k = 0; k < 2; k++)
            rec_set_i64(b->items[chunk * 2 + k], off, tmp[k]);
    }
}

#else
/* No AVX-512, AVX2 or SSE4.2: SIMD just runs the interpreter
   per record. Same results, no illegal instruction. */
void simd_execute(engine *e,record_batch *b)
{
    for(int i = 0; i < SIMD_WIDTH; i++)
        interpret(e, b->items[i]);
}
#endif

/* ============================================================
   BATCH BUILDER
   ============================================================ */

int build_batch(engine *e,record_batch *b)
{
    int n=0;

    for(int i=0;i<SIMD_WIDTH;i++)
    {
        record *r=ring_pop(&e->queue);

        if(!r)
            break;

        pulsar_prefetch(r->data);

        b->items[i]=r;

        n++;
    }

    return n;
}

/* ============================================================
   SIMD RUN LOOP
   ============================================================ */

void engine_run_simd(engine *e)
{
    record_batch batch;

    while(1)
    {
        int n=build_batch(e,&batch);

        if(n==0)
            break;

        if(n==SIMD_WIDTH)
        {
            simd_execute(e,&batch);
        }
        else
        {
            for(int i=0;i<n;i++)
                interpret(e,batch.items[i]);
        }

        /* Atomic: several streaming workers may drain concurrently. */
        __atomic_fetch_add(&e->exec_count,(uint64_t)n,__ATOMIC_RELAXED);
    }
}

/* ============================================================
   WORKER THREAD (SIMD MODE)
   ============================================================ */

typedef struct
{
    engine *eng;
    int cpu;

} worker;

void* worker_simd(void *arg)
{
    worker *w=arg;

    cpu_set_t set;

    CPU_ZERO(&set);
    CPU_SET(w->cpu,&set);

    sched_setaffinity(0,sizeof(set),&set);

    /* Drain until asked to stop. Between drains the ring is empty and
       engine_run_simd returns immediately (hot spin, by design). */
    while(!w->eng->workers_stop)
    {
        engine_run_simd(w->eng);
    }

    free(w);
    return NULL;
}

/* ============================================================
   SPAWN / STOP WORKERS
   ============================================================ */

void engine_stop_workers(engine *e)
{
    if(!e->worker_threads)
        return;

    e->workers_stop = 1;

    for(int i = 0; i < e->worker_count; i++)
        pthread_join(e->worker_threads[i], NULL);

    free(e->worker_threads);
    e->worker_threads = NULL;
    e->worker_count = 0;
    e->workers_stop = 0;
}

void engine_start_workers(engine *e,int n)
{
    /* Restart cleanly if workers are already running. */
    engine_stop_workers(e);

    if(n > PULSAR_WORKERS)
        n = PULSAR_WORKERS;
    if(n <= 0)
        return;

    e->worker_threads = malloc((size_t)n * sizeof(pthread_t));
    e->worker_count = n;
    e->workers_stop = 0;

    for(int i=0;i<n;i++)
    {
        worker *w=malloc(sizeof(worker));

        w->eng=e;
        w->cpu=i;

        pthread_create(&e->worker_threads[i],NULL,worker_simd,w);
    }
}

/* ============================================================
   JIT Instruction Emitter: x86‑64 / AVX‑512
   ============================================================ */

typedef struct {
    jit_mem mem;
    uint8_t *cursor;
} jit_builder;

/* ------------------------------------------------------------
   Basic helpers
   ------------------------------------------------------------ */

static inline void jb_init(jit_builder *jb, size_t cap) {
    jit_mem_init(&jb->mem, cap);
    jb->cursor = jb->mem.mem;
    /* §6 MPK: make the JIT pages writable for this (emitting) thread. */
    pulsar_mpk_unlock();
}

static inline void jb_emit8(jit_builder *jb, uint8_t v){
    *jb->cursor++ = v;
    jb->mem.offset++;
}

static inline void jb_emit32(jit_builder *jb, uint32_t v){
    memcpy(jb->cursor, &v, 4);
    jb->cursor += 4;
    jb->mem.offset += 4;
}

static inline void jb_emit64(jit_builder *jb, uint64_t v){
    memcpy(jb->cursor, &v, 8);
    jb->cursor += 8;
    jb->mem.offset += 8;
}

/* ============================================================
   Minimal encoder utilities (Rex prefixes, ModRM, etc.)
   ============================================================ */

static inline void emit_rex64(jit_builder *jb, uint8_t reg, uint8_t rm){
    uint8_t rex = 0x48; // REX.W = 1
    if (reg & 8) rex |= 0x4; // R bit
    if (rm  & 8) rex |= 0x1; // B bit
    jb_emit8(jb, rex);
}

static inline void emit_modrm(jit_builder *jb, uint8_t mod, uint8_t reg, uint8_t rm){
    jb_emit8(jb, (mod << 6) | ((reg & 7) << 3) | (rm & 7));
}

/* ------------------------------------------------------------
   Emit MOV r64, imm64
   (opcode: 0xB8 + reg)
   ------------------------------------------------------------ */

static inline void emit_mov_imm64(jit_builder *jb, uint8_t reg, uint64_t imm){
    jb_emit8(jb, 0x48);
    jb_emit8(jb, 0xB8 + (reg & 7));
    jb_emit64(jb, imm);
}

/* ------------------------------------------------------------
   Integer division helpers (OP_DIV lowering)
   ------------------------------------------------------------
   idiv needs the dividend in RDX:RAX and clobbers both. The sequence
   for `working_reg /= c` is:

       [MOV RAX, working_reg]   ; if working_reg != RAX
       MOV RCX, <c>             ; full 64-bit divisor
       CQO                     ; sign-extend RAX into RDX:RAX
       IDIV RCX                ; RAX = quotient (traps on c==0,
                               ;  like the interpreter)
       [MOV working_reg, RAX]   ; if working_reg != RAX

   RCX is safe as the divisor holder: the value is moved to RAX first,
   so even working_reg == RCX works.
   ------------------------------------------------------------ */

static inline void emit_cqo(jit_builder *jb){
    jb_emit8(jb, 0x48);
    jb_emit8(jb, 0x99);
}

static inline void emit_idiv_r64(jit_builder *jb, uint8_t rm){
    emit_rex64(jb, 7, rm);
    jb_emit8(jb, 0xF7);
    emit_modrm(jb, 3, 7, rm);   /* /7 */
}

/* MOV r64, r64 (dst, src) */
static inline void emit_mov_r64(jit_builder *jb, uint8_t dst, uint8_t src){
    emit_rex64(jb, dst, src);
    jb_emit8(jb, 0x8B);
    emit_modrm(jb, 3, dst, src);
}

static void emit_idiv_r64_const(jit_builder *jb, uint8_t working_reg, int64_t c){
    if(working_reg != 0)
        emit_mov_r64(jb, 0 /*RAX*/, working_reg);
    emit_mov_imm64(jb, 1 /*RCX*/, (uint64_t)c);
    emit_cqo(jb);
    emit_idiv_r64(jb, 1 /*RCX*/);
    if(working_reg != 0)
        emit_mov_r64(jb, working_reg, 0 /*RAX*/);
}

/* ------------------------------------------------------------
   Granlund-Montgomery magic division by a compile-time constant
   ------------------------------------------------------------
   Unsigned: s = floor(log2(d)), M = ceil(2^(64+s) / d),
             q = mulhi_u64(M, n) >> s.  Exact for every n; the only
             requirement is d > 1 and d not a power of two (powers of
             two use the shift path below).
   Signed:   q = sign(n)*sign(d) * magic_u(|n|, |d|).
   Verified offline against hardware idiv: 1.7M cases, 0 fails
   (divisors 2..5000, powers of two, 2^63 neighbours, random u64;
   dividends 0, +-1, INT64_MIN/MAX, random). d == 0 and d == -1
   keep the idiv lowering: they must still trap with SIGFPE on
   division by zero and on INT64_MIN / -1, exactly like the
   interpreter; d == 1 is a no-op.
   ------------------------------------------------------------ */

typedef struct { uint64_t magic; int shift; } div_magic;

static int compute_magic_u64(uint64_t d, div_magic *out)
{
    if(d <= 1) return 0;
    if((d & (d - 1)) == 0) return 0;   /* power of two: shift path */
    int s = 63 - __builtin_clzll(d);
    __uint128_t pw = (__uint128_t)1 << (64 + s);
    out->magic = (uint64_t)((pw + d - 1) / d);
    out->shift = s;
    return 1;
}

static int compute_magic_i64(int64_t d, div_magic *out)
{
    if(d == 0 || d == 1 || d == -1) return 0;
    uint64_t ad = (d < 0) ? (uint64_t)(-(d + 1)) + 1 : (uint64_t)d;
    return compute_magic_u64(ad, out);
}

/* d == +-2^k (k >= 1)? Returns k and the sign. */
static int is_pow2_div(int64_t d, int *k, int *neg)
{
    if(d == 0 || d == 1 || d == -1) return 0;
    uint64_t ad = (d < 0) ? (uint64_t)(-(d + 1)) + 1 : (uint64_t)d;
    if((ad & (ad - 1)) != 0) return 0;
    *k = __builtin_ctzll(ad);
    *neg = (d < 0);
    return 1;
}

/* SAR r/m64, imm8 (0xC1 /7 ib) */
static inline void emit_sar_r64_imm8(jit_builder *jb, uint8_t rm, uint8_t imm){
    emit_rex64(jb, 7, rm);
    jb_emit8(jb, 0xC1);
    emit_modrm(jb, 3, 7, rm);
    jb_emit8(jb, imm);
}

/* SHR r/m64, imm8 (0xC1 /5 ib) */
static inline void emit_shr_imm8(jit_builder *jb, uint8_t rm, uint8_t imm){
    emit_rex64(jb, 5, rm);
    jb_emit8(jb, 0xC1);
    emit_modrm(jb, 3, 5, rm);
    jb_emit8(jb, imm);
}

/* XOR r/m64, r64 (0x31 /r): dst is the r/m operand, src is the reg. */
static inline void emit_xor_r64(jit_builder *jb, uint8_t dst, uint8_t src){
    emit_rex64(jb, src, dst);
    jb_emit8(jb, 0x31);
    emit_modrm(jb, 3, src, dst);
}

/* SUB r/m64, r64 (0x29 /r) */
static inline void emit_sub_r64(jit_builder *jb, uint8_t dst, uint8_t src){
    emit_rex64(jb, src, dst);
    jb_emit8(jb, 0x29);
    emit_modrm(jb, 3, src, dst);
}

/* ADD r/m64, r64 (0x01 /r) */
static inline void emit_add_r64(jit_builder *jb, uint8_t dst, uint8_t src){
    emit_rex64(jb, src, dst);
    jb_emit8(jb, 0x01);
    emit_modrm(jb, 3, src, dst);
}

/* NEG r/m64 (0xF7 /3) */
static inline void emit_neg_r64(jit_builder *jb, uint8_t rm){
    emit_rex64(jb, 3, rm);
    jb_emit8(jb, 0xF7);
    emit_modrm(jb, 3, 3, rm);
}

/* NOT r/m64 (0xF7 /2) */
static inline void emit_not_r64(jit_builder *jb, uint8_t rm){
    emit_rex64(jb, 2, rm);
    jb_emit8(jb, 0xF7);
    emit_modrm(jb, 3, 2, rm);
}

/* MUL r/m64 (0xF7 /4): RDX:RAX = RAX * r/m64 (unsigned). */
static inline void emit_mul_r64(jit_builder *jb, uint8_t rm){
    emit_rex64(jb, 4, rm);
    jb_emit8(jb, 0xF7);
    emit_modrm(jb, 3, 4, rm);
}

/* AND r/m64, r64 (0x21 /r): dst &= src. */
static inline void emit_and_r64(jit_builder *jb, uint8_t dst, uint8_t src){
    emit_rex64(jb, src, dst);
    jb_emit8(jb, 0x21);
    emit_modrm(jb, 3, src, dst);
}

/* TEST r64, r/m64 (0x85 /r): sets flags from reg & rm. */
static inline void emit_test_r64(jit_builder *jb, uint8_t reg, uint8_t rm){
    emit_rex64(jb, reg, rm);
    jb_emit8(jb, 0x85);
    emit_modrm(jb, 3, reg, rm);
}

/* CMP r64, r/m64 (0x3B /r): reg - src, sets flags. */
static inline void emit_cmp_r64_reg(jit_builder *jb, uint8_t reg, uint8_t src){
    emit_rex64(jb, reg, src);
    jb_emit8(jb, 0x3B);
    emit_modrm(jb, 3, reg, src);
}

/* CMP r/m64, imm32 (0x81 /7). */
static inline void emit_cmp_r64_imm32(jit_builder *jb, uint8_t reg, int32_t imm){
    emit_rex64(jb, 7, reg);
    jb_emit8(jb, 0x81);
    emit_modrm(jb, 3, 7, reg);
    jb_emit32(jb, (uint32_t)imm);
}

/* CMP reg, constant (imm32 when it fits, else via RCX scratch). */
static void emit_cmp_r64_const(jit_builder *jb, uint8_t reg, int64_t c)
{
    if(c >= INT32_MIN && c <= INT32_MAX)
    {
        emit_cmp_r64_imm32(jb, reg, (int32_t)c);
        return;
    }
    uint8_t scratch = (reg == 1) ? 2 : 1;   /* RCX unless reg IS RCX */
    emit_mov_imm64(jb, scratch, (uint64_t)c);
    emit_cmp_r64_reg(jb, reg, scratch);
}

/* XOR r/m64, imm32 (0x81 /6). */
static inline void emit_xor_r64_imm32(jit_builder *jb, uint8_t reg, int32_t imm){
    emit_rex64(jb, 6, reg);
    jb_emit8(jb, 0x81);
    emit_modrm(jb, 3, 6, reg);
    jb_emit32(jb, (uint32_t)imm);
}

/* XOR reg, constant (imm32 when it fits, else via RCX scratch). */
static void emit_xor_r64_const(jit_builder *jb, uint8_t reg, int64_t c)
{
    if(c >= INT32_MIN && c <= INT32_MAX)
    {
        emit_xor_r64_imm32(jb, reg, (int32_t)c);
        return;
    }
    uint8_t scratch = (reg == 1) ? 2 : 1;   /* RCX unless reg IS RCX */
    emit_mov_imm64(jb, scratch, (uint64_t)c);
    emit_xor_r64(jb, reg, scratch);
}

/* CMOVG r64, r/m64 (0x0F 0x4F /r): move if greater (signed). */
static inline void emit_cmovg_r64(jit_builder *jb, uint8_t dst, uint8_t src){
    emit_rex64(jb, dst, src);
    jb_emit8(jb, 0x0F);
    jb_emit8(jb, 0x4F);
    emit_modrm(jb, 3, dst, src);
}

/* CMOVNZ r64, r/m64 (0x0F 0x45 /r): move if not zero. */
static inline void emit_cmovnz_r64(jit_builder *jb, uint8_t dst, uint8_t src){
    emit_rex64(jb, dst, src);
    jb_emit8(jb, 0x0F);
    jb_emit8(jb, 0x45);
    emit_modrm(jb, 3, dst, src);
}

/* Emit "TEST RSI, RSI; JZ rel32" with a placeholder rel32. Returns the
   offset of the rel32 field, to be patched by patch_jz_rel32 once the
   guarded code is emitted. (Used to skip a masked op when pred == 0.) */
static size_t emit_test_pred_jz(jit_builder *jb)
{
    emit_test_r64(jb, 6 /*RSI*/, 6 /*RSI*/);
    jb_emit8(jb, 0x0F);
    jb_emit8(jb, 0x84);   /* JZ rel32 */
    size_t rel32_pos = jb->mem.offset;
    jb_emit32(jb, 0);     /* placeholder */
    return rel32_pos;
}

/* Patch a JZ rel32 (from emit_test_pred_jz) to jump to the current
   position (just after the guarded code). */
static void patch_jz_rel32(jit_builder *jb, size_t rel32_pos)
{
    int32_t rel = (int32_t)(jb->mem.offset - (rel32_pos + 4));
    memcpy(jb->mem.mem + rel32_pos, &rel, 4);
}

/* Scalar magic division: working_reg /= c. Clobbers RAX, RCX, RDX.
   Falls back to idiv when c is 0 or -1 (SIGFPE must be preserved). */
static void emit_magic_div_r64_const(jit_builder *jb, uint8_t working_reg, int64_t c)
{
    if(c == 0 || c == -1)
    {
        emit_idiv_r64_const(jb, working_reg, c);
        return;
    }
    if(c == 1)
        return;

    int k, neg;
    if(is_pow2_div(c, &k, &neg))
    {
        /* q = (n + ((n>>63) & (2^k - 1))) >> k, computed without a
           movabs: (n>>63) is 0 or -1, so (n>>63) >>_logical (64-k)
           is 0 or 2^k-1. Then negate when c < 0. */
        if(working_reg != 0)
            emit_mov_r64(jb, 0 /*RAX*/, working_reg);
        emit_mov_r64(jb, 1 /*RCX*/, 0 /*RAX*/);
        emit_sar_r64_imm8(jb, 1 /*RCX*/, 63);
        emit_shr_imm8(jb, 1 /*RCX*/, (uint8_t)(64 - k));
        emit_add_r64(jb, 0 /*RAX*/, 1 /*RCX*/);
        emit_sar_r64_imm8(jb, 0 /*RAX*/, (uint8_t)k);
        if(neg)
            emit_neg_r64(jb, 0 /*RAX*/);
        if(working_reg != 0)
            emit_mov_r64(jb, working_reg, 0 /*RAX*/);
        return;
    }

    div_magic dm;
    if(!compute_magic_i64(c, &dm))
    {
        emit_idiv_r64_const(jb, working_reg, c);
        return;
    }

    /* |n| in RAX, sign mask in RCX, then unsigned mulhi, shift, sign fix.
       The mul is unsigned: RAX = M, RDX = |n| -> RDX = mulhi(M, |n|). */
    if(working_reg != 0)
        emit_mov_r64(jb, 0 /*RAX*/, working_reg);
    emit_mov_r64(jb, 1 /*RCX*/, 0 /*RAX*/);
    emit_sar_r64_imm8(jb, 1 /*RCX*/, 63);
    emit_xor_r64(jb, 0 /*RAX*/, 1 /*RCX*/);
    emit_sub_r64(jb, 0 /*RAX*/, 1 /*RCX*/);
    emit_mov_r64(jb, 2 /*RDX*/, 0 /*RAX*/);
    emit_mov_imm64(jb, 0 /*RAX*/, dm.magic);
    emit_mul_r64(jb, 2 /*RDX*/);
    emit_shr_imm8(jb, 2 /*RDX*/, (uint8_t)dm.shift);
    if(c < 0)
        emit_not_r64(jb, 1 /*RCX*/);   /* flip: negate when n >= 0 */
    emit_mov_r64(jb, 0 /*RAX*/, 2 /*RDX*/);
    emit_xor_r64(jb, 0 /*RAX*/, 1 /*RCX*/);
    emit_sub_r64(jb, 0 /*RAX*/, 1 /*RCX*/);
    if(working_reg != 0)
        emit_mov_r64(jb, working_reg, 0 /*RAX*/);
}

/* ============================================================
   EVEX prefix construction (verified against gas/objdump/gcc)
   ------------------------------------------------------------
   P0 = (~D[3]<<7) | (~S2[4]<<6) | (~S2[3]<<5) | (~D[4]<<4) | mm
        mm: 01 = 0F map, 10 = 0F38 map
   P1 = (W<<7) | ((~S1 & 0xF)<<3) | 0x04 | pp
        pp: 01 = 66 prefix, 10 = F3 prefix
   P2 = 0x40 | (S1[4] ? 0 : 0x08) | mask
        (0x48 = 512-bit, merging, k0; bit3 = ~S1[4] (V'))
   One verified quirk: VPMULUDQ (0F38, opcode F4) is encoded with
   mm=01, not 10 -- confirmed by gas, objdump and gcc intrinsics.
   ------------------------------------------------------------ */

/* D = ModRM.reg operand, S1 = vvvv operand, S2 = ModRM.rm operand. */
static inline void emit_evex(jit_builder *jb, uint8_t mm, uint8_t w, uint8_t pp,
                             uint8_t dst, uint8_t src1, uint8_t src2, uint8_t mask)
{
    uint8_t p0 = (uint8_t)(((dst & 8) ? 0 : 0x80) | ((src2 & 16) ? 0 : 0x40) |
                          ((src2 & 8) ? 0 : 0x20) | ((dst & 16) ? 0 : 0x10) | mm);
    uint8_t p1 = (uint8_t)((w ? 0x80 : 0) | (((~src1) & 0xF) << 3) | 0x04 | pp);
    uint8_t p2 = (uint8_t)(0x40 | ((src1 & 16) ? 0 : 0x08) | (mask & 7));
    jb_emit8(jb, 0x62);
    jb_emit8(jb, p0);
    jb_emit8(jb, p1);
    jb_emit8(jb, p2);
}

/* 3-operand vector ALU op: dst = op(src1, src2). mm: 1 = 0F, 2 = 0F38.
   mask: 0 = k0 (unmasked), else the k register (1..7). */
static inline void emit_evex_3(jit_builder *jb, uint8_t mm, uint8_t opcode,
                               uint8_t dst, uint8_t src1, uint8_t src2, uint8_t mask)
{
    emit_evex(jb, mm, 1 /*W*/, 1 /*66*/, dst, src1, src2, mask);
    jb_emit8(jb, opcode);
    emit_modrm(jb, 3, dst, src2);
}

static inline void emit_vpaddq(jit_builder *jb, uint8_t dst, uint8_t s1, uint8_t s2, uint8_t mask){
    emit_evex_3(jb, 1, 0xD4, dst, s1, s2, mask);
}
static inline void emit_vpsubq(jit_builder *jb, uint8_t dst, uint8_t s1, uint8_t s2, uint8_t mask){
    emit_evex_3(jb, 1, 0xFB, dst, s1, s2, mask);
}
static inline void emit_vpmullq(jit_builder *jb, uint8_t dst, uint8_t s1, uint8_t s2, uint8_t mask){
    emit_evex_3(jb, 2, 0x40, dst, s1, s2, mask);
}
static inline void emit_vpxorq(jit_builder *jb, uint8_t dst, uint8_t s1, uint8_t s2, uint8_t mask){
    emit_evex_3(jb, 1, 0xEF, dst, s1, s2, mask);
}
static inline void emit_vpandq(jit_builder *jb, uint8_t dst, uint8_t s1, uint8_t s2, uint8_t mask){
    emit_evex_3(jb, 1, 0xDB, dst, s1, s2, mask);
}
static inline void emit_vporq(jit_builder *jb, uint8_t dst, uint8_t s1, uint8_t s2, uint8_t mask){
    emit_evex_3(jb, 1, 0xEB, dst, s1, s2, mask);
}
/* VPMULUDQ: 32-bit unsigned multiply, full 64-bit product of the low
   32 bits of each lane. Encoded with mm=01 (verified quirk). */
static inline void emit_vpmuludq(jit_builder *jb, uint8_t dst, uint8_t s1, uint8_t s2){
    emit_evex_3(jb, 1, 0xF4, dst, s1, s2, 0);
}

/* Vector shift by an immediate: VPSRLQ (0x73 /2), VPSRAQ (0x72 /4),
   VPSLLQ (0x72 /6). Unlike the 3-operand ALU ops, the shift's
   DESTINATION is encoded in vvvv and the SOURCE in ModRM.rm
   (verified against gas: `vpsraq $63, %zmm0, %zmm1` ->
   62 F1 F5 48 72 E0 3F). The reg field holds the sub-opcode. */
static inline void emit_evex_shift_imm(jit_builder *jb, uint8_t opcode, uint8_t sub,
                                       uint8_t dst, uint8_t src, uint8_t imm)
{
    emit_evex(jb, 1, 1, 1, sub, dst, src, 0);
    jb_emit8(jb, opcode);
    emit_modrm(jb, 3, sub, src);
    jb_emit8(jb, imm);
}

static inline void emit_vpsrlq(jit_builder *jb, uint8_t dst, uint8_t src, uint8_t imm){
    emit_evex_shift_imm(jb, 0x73, 2, dst, src, imm);
}
static inline void emit_vpsraq(jit_builder *jb, uint8_t dst, uint8_t src, uint8_t imm){
    emit_evex_shift_imm(jb, 0x72, 4, dst, src, imm);
}
static inline void emit_vpsllq(jit_builder *jb, uint8_t dst, uint8_t src, uint8_t imm){
    /* VPSLLQ is opcode 0x73 /6 (0x72 is VPSRAQ's opcode, /4 only). */
    emit_evex_shift_imm(jb, 0x73, 6, dst, src, imm);
}

/* VMOVDQA64 zmmD, zmmS (register-to-register copy). */
static inline void emit_vmovdqa64_rr(jit_builder *jb, uint8_t dst, uint8_t src){
    emit_evex(jb, 1, 1, 1, dst, 0, src, 0);
    jb_emit8(jb, 0x6F);
    emit_modrm(jb, 3, dst, src);
}

/* Masked register-to-register move (merging): dst = k ? src : dst.
   Used for OP_SELECT (v = pred ? c : v) and to blend a DIV result
   back under the predicate. mask is a k register 1..7 (0 = unmasked). */
static inline void emit_vmovdqa64_rr_masked(jit_builder *jb, uint8_t dst, uint8_t src,
                                            uint8_t mask){
    emit_evex(jb, 1, 1, 1, dst, 0, src, mask);
    jb_emit8(jb, 0x6F);
    emit_modrm(jb, 3, dst, src);
}

/* VPBROADCASTQ zmmD, r64 (GPR source). */
static inline void emit_vpbroadcastq(jit_builder *jb, uint8_t dst, uint8_t gpr){
    emit_evex(jb, 2, 1, 1, dst, 0, gpr, 0);
    jb_emit8(jb, 0x7C);
    emit_modrm(jb, 3, dst, gpr);
}

/* VPCMPGTQ kD, zmmS1, zmmS2 (signed greater-than, result in a k mask). */
static inline void emit_vpcmpgtq(jit_builder *jb, uint8_t k, uint8_t s1, uint8_t s2){
    emit_evex(jb, 2, 1, 1, k, s1, s2, 0);
    jb_emit8(jb, 0x37);
    emit_modrm(jb, 3, k, s2);
}

/* Load/store through [RDI]. opcode 0x6F = load, 0x7F = store.
   pp: 1 = 66 (aligned, VMOVDQA64), 2 = F3 (unaligned, VMOVDQU64).
   w: 1 normally; 0 for VMOVNTDQ (opcode 0xE7, store-only). */
static inline void emit_evex_mem(jit_builder *jb, uint8_t opcode, uint8_t pp, uint8_t w,
                                 uint8_t zmm)
{
    emit_evex(jb, 1, w, pp, zmm, 0, 7 /*RDI*/, 0);
    jb_emit8(jb, opcode);
    emit_modrm(jb, 0, zmm, 7);
}

/* Same, but through [RDI + disp]. disp==0 uses mod 0; otherwise
   mod 2 + disp32 (covers the 64/128/192-byte vector strides of
   the 4x-unrolled kernel). */
static inline void emit_evex_mem_disp(jit_builder *jb, uint8_t opcode, uint8_t pp,
                                      uint8_t w, uint8_t zmm, int32_t disp)
{
    emit_evex(jb, 1, w, pp, zmm, 0, 7 /*RDI*/, 0);
    jb_emit8(jb, opcode);
    if(disp == 0)
        emit_modrm(jb, 0, zmm, 7);
    else
    {
        emit_modrm(jb, 2, zmm, 7);
        jb_emit32(jb, (uint32_t)disp);
    }
}

/* ------------------------------------------------------------
   Vector Granlund-Montgomery division (one 512-bit accumulator)
   ------------------------------------------------------------
   acc: the accumulator zmm (8 signed int64 lanes) -- overwritten
        with the quotients.
   d:   the constant divisor (d != 0, d != +-1; those keep idiv).
   T:   10 scratch zmm registers (must not alias acc or C).
   C:   6 constant zmm registers, already broadcast:
        C[0] = M & 0xFFFFFFFF (Mlo), C[1] = M >> 32 (Mhi),
        C[2] = 0xFFFFFFFF, C[3] = 2^32, C[4] = 0x8000... (sign bit),
        C[5] = all-ones (used to flip the sign mask when d < 0).
   The 64-bit mulhi is built from 32-bit VPMULUDQ (verified offline).
   Power-of-two divisors use the shift path (emit_evex_pow2_div).
   ------------------------------------------------------------ */

static void emit_evex_magic_div(jit_builder *jb, uint8_t acc, int64_t d,
                                const uint8_t *T, const uint8_t *C)
{
    div_magic dm;
    if(!compute_magic_i64(d, &dm))
        return;   /* caller falls back to the spill/idiv path */

    uint8_t sign = T[0], a_hi = T[1], t1 = T[2], t2 = T[3], t3 = T[4];
    uint8_t t4 = T[5], S = T[6], hi = T[7], u = T[8], w = T[9];
    uint8_t Mlo = C[0], Mhi = C[1], FFFF = C[2], TWO32 = C[3];
    uint8_t SIGNBIT = C[4], ONES = C[5];

    /* 1. sign mask, then |x| in place in acc */
    emit_vpsraq(jb, sign, acc, 63);             /* sign = acc >> 63 */
    emit_vpxorq(jb, acc, acc, sign, 0);            /* acc ^= sign */
    emit_vpsubq(jb, acc, acc, sign, 0);            /* acc -= sign  -> |x| */

    /* 2. 64-bit unsigned mulhi(M, |x|) from 32-bit VPMULUDQ.
          a = a_lo + 2^32*a_hi, M = M_lo + 2^32*M_hi
          product = t1 + 2^32*(t2+t3) + 2^64*t4
          hi = t4 + S_hi + c2 + 2^32*c
          (S_hi = (t2+t3)>>32, c = unsigned-overflow of t2+t3,
           c2 = ((t1>>32) + (S & 0xFFFFFFFF)) >> 32) */
    emit_vpsrlq(jb, a_hi, acc, 32);
    emit_vpmuludq(jb, t1, acc, Mlo);            /* a_lo * M_lo */
    emit_vpmuludq(jb, t2, acc, Mhi);            /* a_lo * M_hi */
    emit_vpmuludq(jb, t3, a_hi, Mlo);           /* a_hi * M_lo */
    emit_vpmuludq(jb, t4, a_hi, Mhi);           /* a_hi * M_hi */
    emit_vpaddq(jb, S, t2, t3, 0);              /* S = t2 + t3 (may wrap) */
    emit_vpsrlq(jb, hi, S, 32);                 /* hi = S >> 32 (S_hi; must
                                                   come before the XORs below,
                                                   which flip S's top bit) */
    /* unsigned overflow of t2+t3: flip the sign bit of both, signed cmp.
       Uses k7 (not k1): k1..k4 hold the §5 per-vector predicate masks,
       and a magic DIV must not clobber them. */
    emit_vpxorq(jb, t2, t2, SIGNBIT, 0);
    emit_vpxorq(jb, S, S, SIGNBIT, 0);
    emit_vpcmpgtq(jb, 7 /*k7*/, t2, S);         /* k7 = (t2 > S) unsigned */
    emit_vpsrlq(jb, u, t1, 32);                 /* u = t1 >> 32 */
    emit_vpandq(jb, w, S, FFFF, 0);                /* w = S & 0xFFFFFFFF (low 32
                                                   bits are untouched by the
                                                   sign-bit XOR) */
    emit_vpaddq(jb, w, u, w, 0);                /* w = u + w */
    emit_vpsrlq(jb, w, w, 32);                  /* w = c2 */
    /* hi = t4 + S_hi + c2 + (k7 ? 2^32 : 0) */
    emit_vpaddq(jb, hi, t4, hi, 0);             /* hi = t4 + S_hi */
    emit_vpaddq(jb, hi, hi, w, 0);              /* hi += c2 */
    emit_vpaddq(jb, hi, hi, TWO32, 7);          /* hi += 2^32 if k7 */

    /* 3. qabs = hi >> shift */
    emit_vpsrlq(jb, hi, hi, (uint8_t)dm.shift);

    /* 4. sign fix: q = (sign==-1) ? -qabs : qabs (flipped when d < 0) */
    if(d < 0)
        emit_vpxorq(jb, sign, sign, ONES, 0);      /* flip the mask */
    emit_vpxorq(jb, hi, hi, sign, 0);
    emit_vpsubq(jb, hi, hi, sign, 0);
    emit_vmovdqa64_rr(jb, acc, hi);             /* acc = quotient */
}

/* Vector power-of-two division: acc /= +-2^k (signed, trunc toward
   zero). tmp is one scratch zmm (must not alias acc). */
static void emit_evex_pow2_div(jit_builder *jb, uint8_t acc, int k, int neg, uint8_t tmp)
{
    /* bias = (acc>>63) >>_logical (64-k)  ==  0 or 2^k-1 */
    emit_vpsraq(jb, tmp, acc, 63);
    emit_vpsrlq(jb, tmp, tmp, (uint8_t)(64 - k));
    emit_vpaddq(jb, acc, acc, tmp, 0);
    emit_vpsraq(jb, acc, acc, (uint8_t)k);
    if(neg)
    {
        uint8_t zero = tmp;
        emit_vpxorq(jb, zero, zero, zero, 0);      /* zero = 0 */
        emit_vpsubq(jb, acc, zero, acc, 0);        /* acc = -acc */
    }
}

/* ------------------------------------------------------------
   Emit ADD r/m64, imm8
   (opcode: 0x83 /0 ib)
   ------------------------------------------------------------ */

static inline void emit_add_r64_imm8(jit_builder *jb, uint8_t rm, int8_t imm){
    emit_rex64(jb, 0, rm);
    jb_emit8(jb, 0x83);
    emit_modrm(jb, 3, 0, rm);
    jb_emit8(jb, imm);
}

/* ------------------------------------------------------------
   Emit IMUL r/m64, imm8
   (opcode: 0x6B /r)
   ------------------------------------------------------------ */

static inline void emit_imul_r64_imm8(jit_builder *jb, uint8_t reg, uint8_t rm, int8_t imm){
    emit_rex64(jb, reg, rm);
    jb_emit8(jb, 0x6B);
    emit_modrm(jb, 3, reg, rm);
    jb_emit8(jb, imm);
}

/* ------------------------------------------------------------
   Emit SQARE: IMUL reg, reg
   (0x0F AF /r)
   ------------------------------------------------------------ */

static inline void emit_imul_self(jit_builder *jb, uint8_t reg){
    emit_rex64(jb, reg, reg);
    jb_emit8(jb, 0x0F);
    jb_emit8(jb, 0xAF);
    emit_modrm(jb, 3, reg, reg);
}

/* ------------------------------------------------------------
   Emit ADD r/m64, imm32   (0x81 /0 id)
   Emit SUB r/m64, imm32   (0x81 /5 id)
   Emit IMUL r64, r/m64, imm32 (0x69 /r id)
   Full-range ALU op on RAX by a 64-bit constant:
   imm32 form when it fits, otherwise movabs RCX + reg op.
   (The old imm8-only forms silently truncated |c| > 127.)
   ------------------------------------------------------------ */

static inline void emit_add_r64_imm32(jit_builder *jb, uint8_t rm, int32_t imm){
    emit_rex64(jb, 0, rm);
    jb_emit8(jb, 0x81);
    emit_modrm(jb, 3, 0, rm);
    jb_emit32(jb, (uint32_t)imm);
}

static inline void emit_sub_r64_imm32(jit_builder *jb, uint8_t rm, int32_t imm){
    emit_rex64(jb, 0, rm);
    jb_emit8(jb, 0x81);
    emit_modrm(jb, 3, 5, rm);
    jb_emit32(jb, (uint32_t)imm);
}

static inline void emit_imul_r64_imm32(jit_builder *jb, uint8_t reg, uint8_t rm, int32_t imm){
    emit_rex64(jb, reg, rm);
    jb_emit8(jb, 0x69);
    emit_modrm(jb, 3, reg, rm);
    jb_emit32(jb, (uint32_t)imm);
}

/* alu: 0 = reg += c, 1 = reg -= c, 2 = reg *= c. Correct for all int64 c.
   The scratch register is picked to never alias the working reg. */
static void emit_alu_r64_const(jit_builder *jb, int alu, uint8_t reg, int64_t c){
    if(c >= INT32_MIN && c <= INT32_MAX){
        int32_t i = (int32_t)c;
        if(alu == 0)      emit_add_r64_imm32(jb, reg, i);
        else if(alu == 1) emit_sub_r64_imm32(jb, reg, i);
        else              emit_imul_r64_imm32(jb, reg, reg, i);
        return;
    }
    uint8_t scratch = (reg == 1) ? 2 : 1;   /* RCX unless reg IS RCX */
    emit_mov_imm64(jb, scratch, (uint64_t)c);
    emit_rex64(jb, reg, scratch);
    if(alu == 0){                             /* ADD reg, scratch */
        jb_emit8(jb, 0x01);
        emit_modrm(jb, 3, scratch, reg);
    } else if(alu == 1){                      /* SUB reg, scratch */
        jb_emit8(jb, 0x29);
        emit_modrm(jb, 3, scratch, reg);
    } else {                                  /* IMUL reg, scratch */
        jb_emit8(jb, 0x0F);
        jb_emit8(jb, 0xAF);
        emit_modrm(jb, 3, reg, scratch);
    }
}

/* ------------------------------------------------------------
   Return
   ------------------------------------------------------------ */

static inline void emit_ret(jit_builder *jb){
    jb_emit8(jb, 0xC3);
}

/* Take ownership of a finished jit_builder's code buffer: publish the entry
   point, remember the vector width the emitted code expects, and release
   any previously compiled buffer so recompiles don't leak. */
/* Append one line to the perf map (/tmp/perf-<pid>.map) so `perf`
   can name JIT functions in profiles and flamegraphs.
   Format: <address> <size> <symbol>. Best-effort: silently ignored
   when the file cannot be opened. */
static void pulsar_perf_map_add(const char *sym, const void *addr, size_t size)
{
    char path[64];
    snprintf(path, sizeof(path), "/tmp/perf-%d.map", (int)getpid());
    FILE *f = fopen(path, "a");
    if(!f) return;
    fprintf(f, "%lx %zx %s\n", (unsigned long)(uintptr_t)addr, size, sym);
    fclose(f);
}

static void jit_code_install(engine *e, jit_builder *jb, int width, const char *sym)
{
    if(e->jit_code_valid)
        munmap(e->jit_code.mem, e->jit_code.size);

    e->jit_code       = jb->mem;
    e->jit_code_valid = 1;
    e->jit_width      = width;
    e->jit_fn         = (jit_fn_t)jb->mem.mem;

    /* §6 MPK: exec-only for this thread now that the code is in. */
    pulsar_mpk_lock();
    /* §6 perf map: name this JIT function for `perf`. */
    pulsar_perf_map_add(sym, jb->mem.mem, jb->mem.offset);
}

/* Install the non-temporal (VMOVNTDQ) variant of the 32-wide kernel.
   It shares the same width (32) as the main v3 kernel; only the
   entry point differs, so jit_width/jit_fn are left alone. */
static void jit_code_install_nt(engine *e, jit_builder *jb)
{
    if(e->jit_code_nt_valid)
        munmap(e->jit_code_nt.mem, e->jit_code_nt.size);

    e->jit_code_nt       = jb->mem;
    e->jit_code_nt_valid = 1;
    e->jit_fn_nt         = (jit_fn_t)jb->mem.mem;

    /* §6 MPK: exec-only for this thread now that the code is in. */
    pulsar_mpk_lock();
    pulsar_perf_map_add("pxasm_jit_nt", jb->mem.mem, jb->mem.offset);
}

static void jit_code_nt_free(engine *e)
{
    if(e->jit_code_nt_valid)
    {
        munmap(e->jit_code_nt.mem, e->jit_code_nt.size);
        e->jit_code_nt_valid = 0;
        e->jit_fn_nt = NULL;
    }
}

/* ============================================================
   JIT Generator for integer pipelines (scalar baseline)
   ============================================================ */

void jit_compile_scalar(engine *e)
{
    jit_builder jb;
    jb_init(&jb, 65536);

    /*
     * Signature:
     * void fn(record *r, uint64_t n)
     *
     * we assume r->data in RDI
     * execute pipeline ops sequentially on [RDI]
     */

    // MOV RAX, [RDI]
    emit_rex64(&jb, 0, 7);     // RDI in rm=7
    jb_emit8(&jb, 0x8B);       // MOV r64,r/m64
    emit_modrm(&jb, 0, 0, 7);  // RAX ← [RDI]
    // Now RAX holds the field value
    // §5 predicate in RSI: -1 = active, 0 = masked off (all-true initially)
    emit_mov_imm64(&jb, 6 /*RSI*/, (uint64_t)-1);

    /* §5 mask state: set by OP_CMP_GT; the next op is skipped entirely
       (TEST RSI,RSI; JZ) when pred is 0, then pred resets to all-true. */
    int mask_pending = 0;
    for (uint32_t i = 0; i < e->pipe.count; ++i) {
        operator *op = &e->pipe.ops[i];
        if(op->type == OP_CMP_GT)
        {
            /* pred = (RAX > c) ? -1 : 0. Overwrites any pending mask. */
            emit_cmp_r64_const(&jb, 0 /*RAX*/, (int64_t)op->constant);
            emit_mov_imm64(&jb, 6 /*RSI*/, 0);
            emit_mov_imm64(&jb, 1 /*RCX*/, (uint64_t)-1);
            emit_cmovg_r64(&jb, 6 /*RSI*/, 1 /*RCX*/);
            mask_pending = 1;
            continue;
        }
        size_t jz_pos = 0;
        if(mask_pending)
            jz_pos = emit_test_pred_jz(&jb);
        switch (op->type) {
            case OP_ADD:
                emit_alu_r64_const(&jb, 0 /*add*/, 0 /*RAX*/, (int64_t)op->constant);
                break;
            case OP_SUB:
                emit_alu_r64_const(&jb, 1 /*sub*/, 0 /*RAX*/, (int64_t)op->constant);
                break;
            case OP_MUL:
                emit_alu_r64_const(&jb, 2 /*mul*/, 0 /*RAX*/, (int64_t)op->constant);
                break;
            case OP_SQUARE:
                emit_imul_self(&jb, 0 /*RAX*/);
                break;
            case OP_DIV:
                emit_magic_div_r64_const(&jb, 0 /*RAX*/, (int64_t)op->constant);
                break;
            case OP_SELECT:
                /* Inside the guard pred != 0 (or all-true with no mask),
                   so RAX = c unconditionally. */
                emit_mov_imm64(&jb, 0 /*RAX*/, (uint64_t)(int64_t)op->constant);
                break;
            case OP_BIT_XOR:
                emit_xor_r64_const(&jb, 0 /*RAX*/, (int64_t)op->constant);
                break;
            default:
                break;
        }
        if(mask_pending)
        {
            patch_jz_rel32(&jb, jz_pos);
            emit_mov_imm64(&jb, 6 /*RSI*/, (uint64_t)-1);   /* pred = all-true */
            mask_pending = 0;
        }
    }

    // MOV [RDI], RAX
    emit_rex64(&jb, 0, 7);
    jb_emit8(&jb, 0x89);
    emit_modrm(&jb, 0, 0, 7);

    emit_ret(&jb);

    jit_code_install(e, &jb, 1 /*scalar width*/, "pxasm_jit_scalar");
}

/* ============================================================
   AVX‑512 kernel emission (vectorized variant)
   ============================================================ */

void jit_compile_avx512(engine *e)
{
    jit_builder jb;
    jb_init(&jb, 128*1024);

    /*
     * Generated kernel (System V: RDI = data pointer):
     *
     *   vmovdqu64 zmm0, [rdi]        ; load 8 x int64
     *   per op:
     *     mov rax, <const>           ; materialize constant
     *     vpbroadcastq zmm1, rax     ; broadcast to all lanes
     *     vpaddq/vpsubq/vpmullq zmm0, zmm0, zmm1
     *   vmovdqu64 [rdi], zmm0        ; store back
     *   ret
     *
     * (The previous revision emitted vpaddq/vpmullq zmm0,zmm0,zmm0 and
     * silently ignored every constant -- now fixed.)
     */

    // vmovdqu64 zmm0, [rdi]   (EVEX.66.0F.W1 6F -- bytes verified via gcc)
    jb_emit8(&jb, 0x62); jb_emit8(&jb, 0xF1); jb_emit8(&jb, 0xFE); jb_emit8(&jb, 0x48);
    jb_emit8(&jb, 0x6F);
    emit_modrm(&jb, 0, 0, 7);

    for (uint32_t i=0; i<e->pipe.count; ++i) {
        operator *op = &e->pipe.ops[i];

        switch(op->type){
            case OP_ADD:
            case OP_SUB:
            case OP_MUL:
                // MOV RAX, imm64
                emit_mov_imm64(&jb, 0, (uint64_t)(int64_t)op->constant);
                // vpbroadcastq zmm1, rax   (EVEX.66.0F38.W1 7C /r)
                jb_emit8(&jb, 0x62); jb_emit8(&jb, 0xF2);
                jb_emit8(&jb, 0xFD); jb_emit8(&jb, 0x48);
                jb_emit8(&jb, 0x7C);
                emit_modrm(&jb, 3, 1, 0);
                break;
            default:
                break;
        }

        switch(op->type){
            case OP_ADD:
                // vpaddq zmm0, zmm0, zmm1  (EVEX.66.0F.W1 D4 /r)
                jb_emit8(&jb, 0x62); jb_emit8(&jb, 0xF1);
                jb_emit8(&jb, 0xFD); jb_emit8(&jb, 0x48);
                jb_emit8(&jb, 0xD4);
                emit_modrm(&jb, 3, 0, 1);
                break;
            case OP_SUB:
                // vpsubq zmm0, zmm0, zmm1  (EVEX.66.0F.W1 FB /r)
                jb_emit8(&jb, 0x62); jb_emit8(&jb, 0xF1);
                jb_emit8(&jb, 0xFD); jb_emit8(&jb, 0x48);
                jb_emit8(&jb, 0xFB);
                emit_modrm(&jb, 3, 0, 1);
                break;
            case OP_MUL:
                // vpmullq zmm0, zmm0, zmm1 (EVEX.66.0F38.W1 40 /r)
                jb_emit8(&jb, 0x62); jb_emit8(&jb, 0xF2);
                jb_emit8(&jb, 0xFD); jb_emit8(&jb, 0x48);
                jb_emit8(&jb, 0x40);
                emit_modrm(&jb, 3, 0, 1);
                break;
            case OP_SQUARE:
                // vpmullq zmm0, zmm0, zmm0
                jb_emit8(&jb, 0x62); jb_emit8(&jb, 0xF2);
                jb_emit8(&jb, 0xFD); jb_emit8(&jb, 0x48);
                jb_emit8(&jb, 0x40);
                emit_modrm(&jb, 3, 0, 0);
                break;
            case OP_DIV: {
                int64_t c = (int64_t)op->constant;
                if(c == 1)
                    break;   /* dividing by 1 is a no-op */
                int dk, dneg;
                if(is_pow2_div(c, &dk, &dneg))
                {
                    emit_evex_pow2_div(&jb, 0 /*zmm0*/, dk, dneg, 2 /*tmp*/);
                    break;
                }
                div_magic dm;
                if(c == 0 || c == -1 || !compute_magic_i64(c, &dm))
                {
                    /* d == 0 / -1 must still trap with SIGFPE like
                       idiv (INT64_MIN / -1 included): spill the
                       lanes, divide each lane scalar, reload. */
                    // vmovdqu64 [rdi], zmm0
                    jb_emit8(&jb, 0x62); jb_emit8(&jb, 0xF1);
                    jb_emit8(&jb, 0xFE); jb_emit8(&jb, 0x48);
                    jb_emit8(&jb, 0x7F);
                    emit_modrm(&jb, 0, 0, 7);
                    // MOV RCX, <divisor>
                    emit_mov_imm64(&jb, 1, (uint64_t)c);
                    for(int lane=0; lane<8; lane++)
                    {
                        uint8_t d = (uint8_t)(lane*8);
                        // MOV RAX, [RDI+disp8]
                        emit_rex64(&jb, 0, 7);
                        jb_emit8(&jb, 0x8B);
                        emit_modrm(&jb, 1, 0, 7);
                        jb_emit8(&jb, d);
                        // CQO
                        jb_emit8(&jb, 0x48); jb_emit8(&jb, 0x99);
                        // IDIV RCX
                        emit_rex64(&jb, 7, 1);
                        jb_emit8(&jb, 0xF7);
                        emit_modrm(&jb, 3, 7, 1);
                        // MOV [RDI+disp8], RAX
                        emit_rex64(&jb, 0, 7);
                        jb_emit8(&jb, 0x89);
                        emit_modrm(&jb, 1, 0, 7);
                        jb_emit8(&jb, d);
                    }
                    // vmovdqu64 zmm0, [rdi]  (reload)
                    jb_emit8(&jb, 0x62); jb_emit8(&jb, 0xF1);
                    jb_emit8(&jb, 0xFE); jb_emit8(&jb, 0x48);
                    jb_emit8(&jb, 0x6F);
                    emit_modrm(&jb, 0, 0, 7);
                    break;
                }
                /* Vector Granlund-Montgomery division: broadcast the
                   6 magic constants into zmm16..zmm21, then divide
                   all 8 lanes in place (no spill, no idiv). */
                uint8_t T[10] = {2,3,4,5,6,7,8,9,10,11};
                uint8_t C[6]  = {16,17,18,19,20,21};
                emit_mov_imm64(&jb, 0, dm.magic & 0xFFFFFFFFULL);
                emit_vpbroadcastq(&jb, C[0], 0);
                emit_mov_imm64(&jb, 0, dm.magic >> 32);
                emit_vpbroadcastq(&jb, C[1], 0);
                emit_mov_imm64(&jb, 0, 0xFFFFFFFFULL);
                emit_vpbroadcastq(&jb, C[2], 0);
                emit_mov_imm64(&jb, 0, 1ULL << 32);
                emit_vpbroadcastq(&jb, C[3], 0);
                emit_mov_imm64(&jb, 0, 0x8000000000000000ULL);
                emit_vpbroadcastq(&jb, C[4], 0);
                emit_mov_imm64(&jb, 0, 0xFFFFFFFFFFFFFFFFULL);
                emit_vpbroadcastq(&jb, C[5], 0);
                emit_evex_magic_div(&jb, 0 /*zmm0*/, c, T, C);
                break;
            }
            default:
                break;
        }
    }

    // vmovdqu64 [rdi], zmm0
    jb_emit8(&jb, 0x62); jb_emit8(&jb, 0xF1); jb_emit8(&jb, 0xFE); jb_emit8(&jb, 0x48);
    jb_emit8(&jb, 0x7F);
    emit_modrm(&jb, 0, 0, 7);

    emit_ret(&jb);

    jit_code_install(e, &jb, 8 /*processes 8 packed int64s*/, "pxasm_jit_avx512");
}

/* ============================================================
   AVX-512 v3 kernel: 4x unrolled, constants pinned in zmm16..31
   ------------------------------------------------------------
   Processes 32 packed int64s per call (4 x zmm0..zmm3), so the
   CALL/RET count drops 4x vs the old 8-wide kernel and the four
   independent vector chains overlap (ILP hides vpmullq latency).
   Pipeline constants are broadcast ONCE in the prologue into
   zmm16..zmm31 (caller-saved); every op reads them from there.
   use_nt: store with VMOVNTDQ (streaming, for arrays larger than
   L3) instead of VMOVDQU64; the caller fences once after the pass.
   Returns 0 (caller falls back to the old 8-wide kernel) when the
   pipeline has >16 constant ops, when a magic DIV would need more
   constant registers than the pool holds, or when the divisor is
   0 (whose SIGFPE must be produced by idiv in the old kernel).
   ============================================================ */

/* Build the v3 kernel into a caller-supplied (already jb_init'ed)
   builder. Returns 1 on success, 0 when the pipeline needs a
   fallback (DIV by 0, too many constants). On failure the builder's
   buffer is released, so the caller must not use it. The emitted
   function processes 32 packed int64s per call. */
static int jit_build_avx512_v3(jit_builder *jb, pipeline *p, int use_nt)
{
    int nconst = 0, has_magic_div = 0;
    for(uint32_t i = 0; i < p->count; i++)
    {
        operator *op = &p->ops[i];
        if(op->type == OP_ADD || op->type == OP_SUB || op->type == OP_MUL ||
           op->type == OP_CMP_GT || op->type == OP_SELECT || op->type == OP_BIT_XOR)
            nconst++;
        else if(op->type == OP_DIV)
        {
            int64_t c = (int64_t)op->constant;
            if(c == 0)
            {
                munmap(jb->mem.mem, jb->mem.size);   /* leave the trap to the old kernel */
                return 0;
            }
            if(c != 1 && c != -1)
            {
                int dk, dneg;
                if(!is_pow2_div(c, &dk, &dneg))
                    has_magic_div = 1;
            }
        }
    }
    if(nconst > 16)
    {
        munmap(jb->mem.mem, jb->mem.size);
        return 0;
    }
    if(has_magic_div && nconst > 10)
    {
        munmap(jb->mem.mem, jb->mem.size);
        return 0;   /* 6 magic slots */
    }

    /* Map each op to its pinned constant register (zmm16+j) or -1. */
    int8_t creg[PULSAR_MAX_PIPE];
    int cj = 0;
    for(uint32_t i = 0; i < p->count; i++)
    {
        operator *op = &p->ops[i];
        if(op->type == OP_ADD || op->type == OP_SUB || op->type == OP_MUL ||
           op->type == OP_CMP_GT || op->type == OP_SELECT || op->type == OP_BIT_XOR)
            creg[i] = (int8_t)(16 + cj++);
        else
            creg[i] = -1;
    }
    /* Magic-division constants live after the pipeline constants. */
    uint8_t magic_base = (uint8_t)(16 + nconst);

    /* Prologue: broadcast the pipeline constants once. */
    for(uint32_t i = 0; i < p->count; i++)
    {
        if(creg[i] < 0) continue;
        emit_mov_imm64(jb, 0, (uint64_t)(int64_t)p->ops[i].constant);
        emit_vpbroadcastq(jb, (uint8_t)creg[i], 0);
    }

    /* Load the 4 vectors (32 records). */
    for(int v = 0; v < 4; v++)
        emit_evex_mem_disp(jb, 0x6F, 2 /*F3*/, 1, (uint8_t)v, v * 64);

    /* §5 k-mask state: OP_CMP_GT sets per-vector masks k1..k4; the
       next op consumes them (masked, merging) and clears the flag.
       With no preceding CMP_GT the mask is 0 (unmasked). */
    int mask_pending = 0;
    for(uint32_t i = 0; i < p->count; i++)
    {
        operator *op = &p->ops[i];
        switch(op->type)
        {
            case OP_ADD:
                for(int v = 0; v < 4; v++)
                    emit_vpaddq(jb, (uint8_t)v, (uint8_t)v, (uint8_t)creg[i],
                                mask_pending ? (uint8_t)(v + 1) : 0);
                mask_pending = 0;
                break;
            case OP_SUB:
                for(int v = 0; v < 4; v++)
                    emit_vpsubq(jb, (uint8_t)v, (uint8_t)v, (uint8_t)creg[i],
                                mask_pending ? (uint8_t)(v + 1) : 0);
                mask_pending = 0;
                break;
            case OP_MUL:
                for(int v = 0; v < 4; v++)
                    emit_vpmullq(jb, (uint8_t)v, (uint8_t)v, (uint8_t)creg[i],
                                 mask_pending ? (uint8_t)(v + 1) : 0);
                mask_pending = 0;
                break;
            case OP_SQUARE:
                for(int v = 0; v < 4; v++)
                    emit_vpmullq(jb, (uint8_t)v, (uint8_t)v, (uint8_t)v,
                                 mask_pending ? (uint8_t)(v + 1) : 0);
                mask_pending = 0;
                break;
            case OP_CMP_GT:
                /* k(v+1) = (acc_v > creg). One mask per vector. */
                for(int v = 0; v < 4; v++)
                    emit_vpcmpgtq(jb, (uint8_t)(v + 1), (uint8_t)v, (uint8_t)creg[i]);
                mask_pending = 1;
                break;
            case OP_SELECT:
                /* acc_v = pred ? creg : acc_v (masked move, merging). */
                for(int v = 0; v < 4; v++)
                    emit_vmovdqa64_rr_masked(jb, (uint8_t)v, (uint8_t)creg[i],
                                             mask_pending ? (uint8_t)(v + 1) : 0);
                mask_pending = 0;
                break;
            case OP_BIT_XOR:
                for(int v = 0; v < 4; v++)
                    emit_vpxorq(jb, (uint8_t)v, (uint8_t)v, (uint8_t)creg[i],
                                mask_pending ? (uint8_t)(v + 1) : 0);
                mask_pending = 0;
                break;
            case OP_DIV: {
                int64_t c = (int64_t)op->constant;
                if(c == 1)
                {
                    mask_pending = 0;
                    break;
                }
                uint8_t T[10] = {4,5,6,7,8,9,10,11,12,13};
                uint8_t C[6] = {(uint8_t)magic_base, (uint8_t)(magic_base + 1),
                                (uint8_t)(magic_base + 2), (uint8_t)(magic_base + 3),
                                (uint8_t)(magic_base + 4), (uint8_t)(magic_base + 5)};
                /* For a magic DIV, broadcast the 6 constants once
                   (shared by all 4 accumulators / the temp). */
                if(c != -1)
                {
                    int dk0, dneg0;
                    if(!is_pow2_div(c, &dk0, &dneg0))
                    {
                        div_magic dm;
                        if(!compute_magic_i64(c, &dm))
                        {
                            munmap(jb->mem.mem, jb->mem.size);
                            return 0;   /* cannot happen; be safe */
                        }
                        emit_mov_imm64(jb, 0, dm.magic & 0xFFFFFFFFULL);
                        emit_vpbroadcastq(jb, C[0], 0);
                        emit_mov_imm64(jb, 0, dm.magic >> 32);
                        emit_vpbroadcastq(jb, C[1], 0);
                        emit_mov_imm64(jb, 0, 0xFFFFFFFFULL);
                        emit_vpbroadcastq(jb, C[2], 0);
                        emit_mov_imm64(jb, 0, 1ULL << 32);
                        emit_vpbroadcastq(jb, C[3], 0);
                        emit_mov_imm64(jb, 0, 0x8000000000000000ULL);
                        emit_vpbroadcastq(jb, C[4], 0);
                        emit_mov_imm64(jb, 0, 0xFFFFFFFFFFFFFFFFULL);
                        emit_vpbroadcastq(jb, C[5], 0);
                    }
                }
                if(c == -1)
                    emit_vpxorq(jb, 4, 4, 4, 0);   /* zmm4 = 0 (once) */
                for(int v = 0; v < 4; v++)
                {
                    /* Masked DIV: compute into temp zmm14 (a copy of
                       acc_v), then blend back under k(v+1) so the
                       masked-off lanes keep their old value.
                       Unmasked: DIV straight into acc_v. */
                    uint8_t tgt = mask_pending ? 14 : (uint8_t)v;
                    if(mask_pending)
                        emit_vmovdqa64_rr(jb, 14, (uint8_t)v);
                    if(c == -1)
                    {
                        /* DIV -1 == NEG */
                        emit_vpsubq(jb, tgt, 4, tgt, 0);
                    }
                    else
                    {
                        int dk, dneg;
                        if(is_pow2_div(c, &dk, &dneg))
                            emit_evex_pow2_div(jb, tgt, dk, dneg, 4 /*tmp*/);
                        else
                            emit_evex_magic_div(jb, tgt, c, T, C);
                    }
                    if(mask_pending)
                        emit_vmovdqa64_rr_masked(jb, (uint8_t)v, 14, (uint8_t)(v + 1));
                }
                mask_pending = 0;
                break;
            }
            default:
                break;
        }
    }

    /* Store back: normal (VMOVDQU64) or non-temporal.
       NT path uses one 512-bit EVEX VMOVNTDQ per vector. That
       instruction faults on a misaligned address (it needs the
       64-byte vector alignment; malloc only gives 16), so the
       caller (execute_jit_records) peels the unaligned head
       through the interpreter before the NT bulk loop. */
    for(int v = 0; v < 4; v++)
    {
        if(use_nt)
            emit_evex_mem_disp(jb, 0xE7, 1 /*66*/, 0 /*W0*/, (uint8_t)v, v * 64);
        else
            emit_evex_mem_disp(jb, 0x7F, 2 /*F3*/, 1, (uint8_t)v, v * 64);
    }

    emit_ret(jb);
    return 1;
}

/* Engine wrapper: build the v3 kernel and install it as the engine's
   JIT function (32 records per call, plus the NT variant). */
static int jit_compile_avx512_v3(engine *e, int use_nt)
{
    jit_builder jb;
    jb_init(&jb, 128*1024);
    if(!jit_build_avx512_v3(&jb, &e->pipe, use_nt))
        return 0;   /* builder buffer already released */
    if(use_nt)
        jit_code_install_nt(e, &jb);
    else
        jit_code_install(e, &jb, 32 /*processes 32 packed int64s*/, "pxasm_jit_v3");
    return 1;
}

/* ============================================================
   VEX backend — Ivy Bridge / Haswell
   ------------------------------------------------------------
   One builder, one flag, two widths: only VEX.L flips. L=0 is
   128-bit xmm (2 x int64, Ivy Bridge, 8 records/call), L=1 is
   256-bit ymm (4 x int64, Haswell, 16 records/call). Regs: accum
   0..3, masks 4..7, pinned constants 8..15, scratch 14/15.
   Gaps vs AVX-512, all closed here: no VPMULLQ, so 64-bit MUL is
   synthesised from VPMULUDQ; no VPSRAQ in VEX at all, so DIV spills
   to [RDI] and divides per lane (SIGFPE preserved); no k masks,
   so CMP_GT writes a full vector mask and SELECT uses VPBLENDVB.
   Builder returns 0 -> the scalar JIT runs instead. Every encoding
   below was checked byte-for-byte against `as`.
   ============================================================ */

/* VEX 3-byte prefix: C4 [R X B mmmmm] [W vvvv L pp].
   R = ~dst[3], B = ~rm[3], mm: 1 = 0F, 2 = 0F38, 3 = 0F3A.
   W is 0 for every op here except VMOVQ. */
static inline void emit_vex(jit_builder *jb, uint8_t mm, uint8_t w, uint8_t pp,
                            uint8_t dst, uint8_t src1, uint8_t src2, uint8_t L)
{
    uint8_t p0 = (uint8_t)(((dst & 8) ? 0 : 0x80) | 0x40 |
                          ((src2 & 8) ? 0 : 0x20) | mm);
    uint8_t p1 = (uint8_t)((w ? 0x80 : 0) | (((~src1) & 0xF) << 3) |
                          ((L & 1) << 2) | pp);
    jb_emit8(jb, 0xC4);
    jb_emit8(jb, p0);
    jb_emit8(jb, p1);
}

/* 3-operand vector ALU: dst = op(src1, src2). */
static inline void emit_vex_3(jit_builder *jb, uint8_t mm, uint8_t opcode,
                              uint8_t dst, uint8_t s1, uint8_t s2, uint8_t L,
                              uint8_t w, uint8_t pp)
{
    emit_vex(jb, mm, w, pp, dst, s1, s2, L);
    jb_emit8(jb, opcode);
    emit_modrm(jb, 3, dst, s2);
}

/* 2-operand form (moves, broadcast): dst = op(src); vvvv unused. */
static inline void emit_vex_2(jit_builder *jb, uint8_t mm, uint8_t opcode,
                              uint8_t dst, uint8_t src, uint8_t L,
                              uint8_t w, uint8_t pp)
{
    emit_vex(jb, mm, w, pp, dst, 0, src, L);
    jb_emit8(jb, opcode);
    emit_modrm(jb, 3, dst, src);
}

/* Shift by immediate: destination in vvvv, source in ModRM.rm,
   sub-opcode in ModRM.reg. Same shape as the EVEX shift helper.
   VPSRLQ = 0x73 /2, VPSLLQ = 0x73 /6 (0x72 is VPSRAQ's opcode,
   which has no VEX form — see the header note). */
static inline void emit_vex_shift_imm(jit_builder *jb, uint8_t opcode, uint8_t sub,
                                      uint8_t dst, uint8_t src, uint8_t imm, uint8_t L)
{
    emit_vex(jb, 1, 0, 1, sub, dst, src, L);
    jb_emit8(jb, opcode);
    emit_modrm(jb, 3, sub, src);
    jb_emit8(jb, imm);
}

/* Load/store through [RDI + disp]. opcode 0x6F = load, 0x7F = store
   (VMOVDQU, pp = F3). disp == 0 uses mod 0, else mod 2 + disp32. */
static inline void emit_vex_mem_disp(jit_builder *jb, uint8_t opcode, uint8_t reg,
                                     int32_t disp, uint8_t L)
{
    emit_vex(jb, 1, 0, 2, reg, 0, 7 /*RDI*/, L);
    jb_emit8(jb, opcode);
    if(disp == 0)
        emit_modrm(jb, 0, reg, 7);
    else
    {
        emit_modrm(jb, 2, reg, 7);
        jb_emit32(jb, (uint32_t)disp);
    }
}

static inline void vex_vpaddq(jit_builder *jb, uint8_t dst, uint8_t s1, uint8_t s2, uint8_t L){
    emit_vex_3(jb, 1, 0xD4, dst, s1, s2, L, 0, 1);
}
static inline void vex_vpsubq(jit_builder *jb, uint8_t dst, uint8_t s1, uint8_t s2, uint8_t L){
    emit_vex_3(jb, 1, 0xFB, dst, s1, s2, L, 0, 1);
}
static inline void vex_vpxorq(jit_builder *jb, uint8_t dst, uint8_t s1, uint8_t s2, uint8_t L){
    emit_vex_3(jb, 1, 0xEF, dst, s1, s2, L, 0, 1);
}
static inline void vex_vpandq(jit_builder *jb, uint8_t dst, uint8_t s1, uint8_t s2, uint8_t L){
    emit_vex_3(jb, 1, 0xDB, dst, s1, s2, L, 0, 1);
}
static inline void vex_vporq(jit_builder *jb, uint8_t dst, uint8_t s1, uint8_t s2, uint8_t L){
    emit_vex_3(jb, 1, 0xEB, dst, s1, s2, L, 0, 1);
}
/* VPMULUDQ: 32-bit unsigned multiply, full 64-bit product of the
   low 32 bits of each lane (0F F4, mm = 01 — same quirk as EVEX). */
static inline void vex_vpmuludq(jit_builder *jb, uint8_t dst, uint8_t s1, uint8_t s2, uint8_t L){
    emit_vex_3(jb, 1, 0xF4, dst, s1, s2, L, 0, 1);
}
static inline void vex_vpsrlq(jit_builder *jb, uint8_t dst, uint8_t src, uint8_t imm, uint8_t L){
    emit_vex_shift_imm(jb, 0x73, 2, dst, src, imm, L);  /* 0x73, not 0x72. cost me an afternoon once. */
}
static inline void vex_vpsllq(jit_builder *jb, uint8_t dst, uint8_t src, uint8_t imm, uint8_t L){
    emit_vex_shift_imm(jb, 0x73, 6, dst, src, imm, L);
}

/* VMOVDQA reg,reg copy (pp = 66, opcode 0x6F). */
static inline void vex_vmovdqa_rr(jit_builder *jb, uint8_t dst, uint8_t src, uint8_t L){
    emit_vex_2(jb, 1, 0x6F, dst, src, L, 0, 1);
}

/* VPCMPGTQ dst, s1, s2 (0F38 37): per-lane all-ones where s1 > s2
   (signed), zero elsewhere — the VEX replacement for a k mask. */
static inline void vex_vpcmpgtq(jit_builder *jb, uint8_t dst, uint8_t s1, uint8_t s2, uint8_t L){
    emit_vex_3(jb, 2, 0x37, dst, s1, s2, L, 0, 1);
}

/* VPBLENDVB dst, s1(false), s2(true), mask — mask reg in imm8[7:4].
   dst = mask ? s2 : s1, per byte (mask lanes are all-ones / zero,
   so this is an exact lane select). */
static inline void vex_vpblendvb(jit_builder *jb, uint8_t dst, uint8_t s1, uint8_t s2,
                                 uint8_t mask, uint8_t L){
    emit_vex_3(jb, 3, 0x4C, dst, s1, s2, L, 0, 1);
    jb_emit8(jb, (uint8_t)(mask << 4));
}

/* Broadcast RAX into vector dst. Ivy: VMOVQ xmm7,RAX + VPUNPCKLQDQ
   (copies the low qword to the high half); Haswell: VMOVQ xmm7,RAX
   + VPBROADCASTQ ymm,xmm7. xmm7 is the broadcast scratch: dead in
   the prologue, and mid-pipeline only used by the unmasked
   full-MUL path, where no mask lives in 4..7. */
static void vex_broadcast_rax(jit_builder *jb, uint8_t dst, uint8_t L)
{
    /* VMOVQ xmm7, RAX (VEX.128.66.0F.W1 6E /r — W=1 is load-bearing
       here: it selects the 64-bit form over MOVD xmm,r32). */
    emit_vex_2(jb, 1, 0x6E, 7, 0 /*RAX*/, 0 /*L=0*/, 1 /*W*/, 1);
    if(L)
        emit_vex_2(jb, 2, 0x59, dst, 7, 1, 0, 1);   /* VPBROADCASTQ ymm, xmm */
    else
        emit_vex_3(jb, 1, 0x6C, dst, 7, 7, 0, 0, 1); /* VPUNPCKLQDQ xmm, xmm7, xmm7 */
}

/* MUL classification for the VEX synthesis (no VPMULLQ on AVX2/AVX1). */
#define VEX_MUL_ZERO 0
#define VEX_MUL_ONE 1
#define VEX_MUL_NEG 2
#define VEX_MUL_SHL 3
#define VEX_MUL_SHL_NEG 4
#define VEX_MUL_U32 5
#define VEX_MUL_FULL 6
static int vex_mul_kind(int64_t c, int *k)
{
    if(c == 0) return VEX_MUL_ZERO;
    if(c == 1) return VEX_MUL_ONE;
    if(c == -1) return VEX_MUL_NEG;
    if(c > 0 && (c & (c - 1)) == 0)
    {
        *k = __builtin_ctzll((uint64_t)c);
        return VEX_MUL_SHL;
    }
    if(c < 0 && c != INT64_MIN && ((-c) & ((-c) - 1)) == 0)
    {
        *k = __builtin_ctzll((uint64_t)(-c));
        return VEX_MUL_SHL_NEG;
    }
    if(c >= 0 && (uint64_t)c < (1ULL << 32)) return VEX_MUL_U32;
    return VEX_MUL_FULL;
}

/* dst = src * creg, 0 <= c < 2^32 (golden path): 5 instructions.
   x*c = xlo*c + ((xhi*c) << 32); both products fit in 64 bits
   because both factors are < 2^32. t0, t1 are scratch vectors
   (dst may alias t0: it is only read by the final add). */
static void vex_mul_u32(jit_builder *jb, uint8_t dst, uint8_t src, uint8_t creg,
                        uint8_t t0, uint8_t t1, uint8_t L)
{
    vex_vpmuludq(jb, t0, src, creg, L);      /* xlo*c */
    vex_vpsrlq(jb, t1, src, 32, L);          /* xhi */
    vex_vpmuludq(jb, t1, t1, creg, L);       /* xhi*c */
    vex_vpsllq(jb, t1, t1, 32, L);           /* (xhi*c) << 32 */
    vex_vpaddq(jb, dst, t0, t1, L);          /* dst = t0 + t1 */
}

/* dst = src * c for a full 64-bit constant: x*c = xlo*clo +
   ((xlo*chi + xhi*clo) << 32) (mod 2^64). Uses all six scratch
   vectors (4..7, 14, 15) — caller must guarantee no pinned
   constant lives in 14/15 and no mask is pending. */
static void vex_mul_full(jit_builder *jb, uint8_t dst, uint8_t src, int64_t c, uint8_t L)
{
    uint64_t clo = (uint64_t)c & 0xFFFFFFFFULL;
    uint64_t chi = (uint64_t)c >> 32;
    emit_mov_imm64(jb, 0, clo);
    vex_broadcast_rax(jb, 4, L);
    emit_mov_imm64(jb, 0, chi);
    vex_broadcast_rax(jb, 5, L);
    vex_vpsrlq(jb, 6, src, 32, L);          /* xhi */
    vex_vpmuludq(jb, 14, src, 4, L);        /* p0 = xlo*clo */
    vex_vpmuludq(jb, 15, src, 5, L);        /* p1 = xlo*chi */
    vex_vpmuludq(jb, 7, 6, 4, L);           /* p2 = xhi*clo */
    vex_vpaddq(jb, 15, 15, 7, L);           /* p1 += p2 */
    vex_vpsllq(jb, 15, 15, 32, L);          /* p1 <<= 32 */
    vex_vpaddq(jb, dst, 14, 15, L);         /* dst = p0 + p1 */
}

/* dst = src * src (mod 2^64): x^2 = xlo^2 + ((2*xlo*xhi) << 32).
   2 x VPMULUDQ only. t0, t1 are scratch vectors. */
static void vex_square(jit_builder *jb, uint8_t dst, uint8_t src,
                       uint8_t t0, uint8_t t1, uint8_t L)
{
    vex_vpmuludq(jb, t0, src, src, L);      /* xlo^2 */
    vex_vpsrlq(jb, t1, src, 32, L);         /* xhi */
    vex_vpmuludq(jb, t1, src, t1, L);       /* xlo*xhi */
    vex_vpsllq(jb, t1, t1, 33, L);          /* <<33 == 2*(..)<<32 */
    vex_vpaddq(jb, dst, t0, t1, L);
}

/* Scalar lane access for fallback DIV by 0 / INT64_MIN/-1 trap: */
static inline void vex_spill_load_rax(jit_builder *jb, uint8_t off)
{
    emit_rex64(jb, 0, 7);
    jb_emit8(jb, 0x8B);
    emit_modrm(jb, 1, 0, 7);
    jb_emit8(jb, off);
}
static inline void vex_spill_store_rax(jit_builder *jb, uint8_t off)
{
    emit_rex64(jb, 0, 7);
    jb_emit8(jb, 0x89);
    emit_modrm(jb, 1, 0, 7);
    jb_emit8(jb, off);
}

/* ============================================================
   v5 (§9): Native Vector VEX Division without VPSRAQ
   ------------------------------------------------------------
   VEX (Ivy Bridge AVX1 / Haswell AVX2) lacks VPSRAQ (0x72 /4 is
   EVEX-only) and k-mask registers. We synthesise exact 64-bit
   arithmetic right shift and Granlund-Montgomery signed magic
   division entirely in XMM/YMM registers using two's-complement
   sign-mask identities:
     1. sign = (0 > acc) via VPCMPGTQ (exact replacement for
        acc >>_arith 63).
     2. sra(x, k) = (x >>_logical k) | ((0 > x) << (64 - k)).
     3. Unsigned 64-bit addition overflow carry (t2 + t3 < t2):
        ovf = VPCMPGTQ(t2 ^ SIGNBIT, S ^ SIGNBIT), shifted by
        (>>_logical 63) then (<< 32) to yield (ovf ? 2^32 : 0)
        without k-masks.
   ============================================================ */

static void emit_vex_pow2_div(jit_builder *jb, uint8_t acc, int k, int neg,
                              uint8_t t0, uint8_t L)
{
    vex_vpxorq(jb, t0, t0, t0, L);
    vex_vpcmpgtq(jb, t0, t0, acc, L);
    vex_vpsrlq(jb, t0, t0, (uint8_t)(64 - k), L);
    vex_vpaddq(jb, acc, acc, t0, L);
    vex_vpxorq(jb, t0, t0, t0, L);
    vex_vpcmpgtq(jb, t0, t0, acc, L);
    vex_vpsllq(jb, t0, t0, (uint8_t)(64 - k), L);
    vex_vpsrlq(jb, acc, acc, (uint8_t)k, L);
    vex_vporq(jb, acc, acc, t0, L);
    if(neg)
    {
        vex_vpxorq(jb, t0, t0, t0, L);
        vex_vpsubq(jb, acc, t0, acc, L);
    }
}

static int emit_vex_magic_div(jit_builder *jb, uint8_t acc, int64_t d, uint8_t L)
{
    div_magic dm;
    if(!compute_magic_i64(d, &dm))
        return 0;

    uint8_t sign = 4, Mlo = 5, Mhi = 6, a_hi = 7;
    uint8_t t1 = 8, t2 = 9, t3 = 10, t4 = 11, S = 12, hi = 13, u = 14, w = 15;

    vex_vpxorq(jb, sign, sign, sign, L);
    vex_vpcmpgtq(jb, sign, sign, acc, L);
    vex_vpxorq(jb, acc, acc, sign, L);
    vex_vpsubq(jb, acc, acc, sign, L);

    emit_mov_imm64(jb, 0, dm.magic & 0xFFFFFFFFULL);
    vex_broadcast_rax(jb, Mlo, L);
    emit_mov_imm64(jb, 0, dm.magic >> 32);
    vex_broadcast_rax(jb, Mhi, L);

    vex_vpsrlq(jb, a_hi, acc, 32, L);
    vex_vpmuludq(jb, t1, acc, Mlo, L);
    vex_vpmuludq(jb, t2, acc, Mhi, L);
    vex_vpmuludq(jb, t3, a_hi, Mlo, L);
    vex_vpmuludq(jb, t4, a_hi, Mhi, L);
    vex_vpaddq(jb, S, t2, t3, L);
    vex_vpsrlq(jb, hi, S, 32, L);

    vex_vpsrlq(jb, u, t1, 32, L);
    vex_vpsllq(jb, w, S, 32, L);
    vex_vpsrlq(jb, w, w, 32, L);
    vex_vpaddq(jb, w, u, w, L);
    vex_vpsrlq(jb, w, w, 32, L);
    vex_vpaddq(jb, hi, t4, hi, L);
    vex_vpaddq(jb, hi, hi, w, L);

    emit_mov_imm64(jb, 0, 0x8000000000000000ULL);
    vex_broadcast_rax(jb, u, L);
    vex_vpxorq(jb, t2, t2, u, L);
    vex_vpxorq(jb, S, S, u, L);
    vex_vpcmpgtq(jb, w, t2, S, L);
    vex_vpsrlq(jb, w, w, 63, L);
    vex_vpsllq(jb, w, w, 32, L);
    vex_vpaddq(jb, hi, hi, w, L);

    if(dm.shift > 0)
        vex_vpsrlq(jb, hi, hi, (uint8_t)dm.shift, L);
    if(d < 0)
    {
        emit_mov_imm64(jb, 0, ~0ULL);
        vex_broadcast_rax(jb, u, L);
        vex_vpxorq(jb, sign, sign, u, L);
    }
    vex_vpxorq(jb, hi, hi, sign, L);
    vex_vpsubq(jb, acc, hi, sign, L);
    return 1;
}

static void vex_mul_vv(jit_builder *jb, uint8_t dst, uint8_t acc,
                       uint8_t x, uint8_t x_hi, uint8_t t0, uint8_t t1, uint8_t L)
{
    vex_vpsrlq(jb, t0, acc, 32, L);
    vex_vpmuludq(jb, t0, t0, x, L);
    vex_vpmuludq(jb, t1, acc, x_hi, L);
    vex_vpaddq(jb, t0, t0, t1, L);
    vex_vpsllq(jb, t0, t0, 32, L);
    vex_vpmuludq(jb, dst, acc, x, L);
    vex_vpaddq(jb, dst, dst, t0, L);
}

/* Build the VEX kernel into a caller-supplied (already jb_init'ed)
   builder. Same failure contract as jit_build_avx512_v3: on failure
   the builder's buffer is released and the caller must not use it.
   The emitted function processes 16 packed int64s per call on
   Haswell (is_avx2 = 1) or 8 on Ivy Bridge (is_avx2 = 0). */
static int jit_build_vex(jit_builder *jb, pipeline *p, int is_avx2)
{
    uint8_t L = is_avx2 ? 1 : 0;
    int width = is_avx2 ? 4 : 2;
    int stride = width * 8;

    int nconst = 0, has_mul = 0, has_square = 0, has_kmask = 0;
    int mask_pending = 0;
    for(uint32_t i = 0; i < p->count; i++)
    {
        operator *op = &p->ops[i];
        switch(op->type)
        {
            case OP_ADD: case OP_SUB: case OP_MUL:
            case OP_CMP_GT: case OP_SELECT: case OP_BIT_XOR:
                nconst++;
                break;
            default:
                break;
        }
        if(op->type == OP_MUL) has_mul = 1;
        if(op->type == OP_SQUARE) has_square = 1;
        if(op->type == OP_CMP_GT || op->type == OP_SELECT || op->type == OP_BIT_XOR)
            has_kmask = 1;
        /* A masked MUL needing the full 64-bit synthesis would
           clobber the mask registers (4..7 double as its scratch),
           and a masked DIV cannot be spilled per-lane — both fall
           back to the scalar JIT instead of emitting wrong code. */
        if(op->type == OP_DIV && mask_pending) {
            int dk_p = 0, dneg_p = 0;
            if((int64_t)op->constant != 1 && !is_pow2_div((int64_t)op->constant, &dk_p, &dneg_p))
                goto fail;
        }
        if(op->type == OP_MUL && mask_pending)
        {
            int k = 0;
            if(vex_mul_kind((int64_t)op->constant, &k) == VEX_MUL_FULL) goto fail;
        }
        if(op->type == OP_CMP_GT)
            mask_pending = 1;
        else
            mask_pending = 0;
    }
    if(nconst > 8) goto fail;
    /* MUL/SQUARE synthesis and masked ops need scratch vectors 14/15
       free, so pipelines using them are limited to 6 pinned constants. */
    if((has_mul || has_square || has_kmask) && nconst > 6) goto fail;

    /* Map each op to its pinned constant register (ymm/xmm 8+j) or -1. */
    int8_t creg[PULSAR_MAX_PIPE];
    int cj = 0;
    for(uint32_t i = 0; i < p->count; i++)
    {
        operator *op = &p->ops[i];
        if(op->type == OP_ADD || op->type == OP_SUB || op->type == OP_MUL ||
           op->type == OP_CMP_GT || op->type == OP_SELECT || op->type == OP_BIT_XOR)
            creg[i] = (int8_t)(8 + cj++);
        else
            creg[i] = -1;
    }

    /* Prologue: broadcast the pipeline constants once. */
    for(uint32_t i = 0; i < p->count; i++)
    {
        if(creg[i] < 0) continue;
        emit_mov_imm64(jb, 0, (uint64_t)(int64_t)p->ops[i].constant);
        vex_broadcast_rax(jb, (uint8_t)creg[i], L);
    }

    /* Load the 4 vectors. */
    for(int v = 0; v < 4; v++)
        emit_vex_mem_disp(jb, 0x6F, (uint8_t)v, v * stride, L);

    /* §5 predicate: OP_CMP_GT writes per-vector masks into 4..7;
       the next op consumes them (via VPBLENDVB) and clears the flag. */
    mask_pending = 0;
    for(uint32_t i = 0; i < p->count; i++)
    {
        operator *op = &p->ops[i];
        switch(op->type)
        {
            case OP_ADD:
            case OP_SUB:
            case OP_BIT_XOR:
                /* Masked: compute into scratch 14, blend back under
                   mask (4+v); unmasked: straight into the accumulator. */
                for(int v = 0; v < 4; v++)
                {
                    uint8_t acc = (uint8_t)v;
                    if(mask_pending)
                    {
                        if(op->type == OP_ADD)
                            vex_vpaddq(jb, 14, acc, (uint8_t)creg[i], L);
                        else if(op->type == OP_SUB)
                            vex_vpsubq(jb, 14, acc, (uint8_t)creg[i], L);
                        else
                            vex_vpxorq(jb, 14, acc, (uint8_t)creg[i], L);
                        vex_vpblendvb(jb, acc, acc, 14, (uint8_t)(4 + v), L);
                    }
                    else
                    {
                        if(op->type == OP_ADD)
                            vex_vpaddq(jb, acc, acc, (uint8_t)creg[i], L);
                        else if(op->type == OP_SUB)
                            vex_vpsubq(jb, acc, acc, (uint8_t)creg[i], L);
                        else
                            vex_vpxorq(jb, acc, acc, (uint8_t)creg[i], L);
                    }
                }
                mask_pending = 0;
                break;
            case OP_CMP_GT:
                for(int v = 0; v < 4; v++)
                    vex_vpcmpgtq(jb, (uint8_t)(4 + v), (uint8_t)v, (uint8_t)creg[i], L);
                mask_pending = 1;
                break;
            case OP_SELECT:
                for(int v = 0; v < 4; v++)
                {
                    if(mask_pending)
                        vex_vpblendvb(jb, (uint8_t)v, (uint8_t)v, (uint8_t)creg[i],
                                      (uint8_t)(4 + v), L);
                    else
                        vex_vmovdqa_rr(jb, (uint8_t)v, (uint8_t)creg[i], L);
                }
                mask_pending = 0;
                break;
            case OP_MUL: {
                int64_t c = (int64_t)op->constant;
                int k = 0;
                int kind = vex_mul_kind(c, &k);
                for(int v = 0; v < 4; v++)
                {
                    uint8_t acc = (uint8_t)v;
                    /* Masked: compute into scratch 14, blend back.
                       The full synthesis writes straight into acc and
                       is unmasked only (pre-scan guarantees it). */
                    uint8_t dst = mask_pending ? 14 : acc;
                    switch(kind)
                    {
                        case VEX_MUL_ZERO:
                            vex_vpxorq(jb, dst, dst, dst, L);
                            break;
                        case VEX_MUL_ONE:
                            if(dst != acc) vex_vmovdqa_rr(jb, dst, acc, L);
                            break;
                        case VEX_MUL_NEG:
                            vex_vpxorq(jb, 15, 15, 15, L);
                            vex_vpsubq(jb, dst, 15, acc, L);
                            break;
                        case VEX_MUL_SHL:
                            vex_vpsllq(jb, dst, acc, (uint8_t)k, L);
                            break;
                        case VEX_MUL_SHL_NEG:
                            vex_vpsllq(jb, dst, acc, (uint8_t)k, L);
                            vex_vpxorq(jb, 15, 15, 15, L);
                            vex_vpsubq(jb, dst, 15, dst, L);
                            break;
                        case VEX_MUL_U32:
                            vex_mul_u32(jb, dst, acc, (uint8_t)creg[i], 14, 15, L);
                            break;
                        case VEX_MUL_FULL:
                            vex_mul_full(jb, acc, acc, c, L);
                            break;
                    }
                    if(mask_pending && kind != VEX_MUL_FULL)
                        vex_vpblendvb(jb, acc, acc, 14, (uint8_t)(4 + v), L);
                }
                mask_pending = 0;
                break;
            }
            case OP_SQUARE:
                for(int v = 0; v < 4; v++)
                {
                    uint8_t acc = (uint8_t)v;
                    if(mask_pending)
                    {
                        vex_square(jb, 14, acc, 14, 15, L);
                        vex_vpblendvb(jb, acc, acc, 14, (uint8_t)(4 + v), L);
                    }
                    else
                        vex_square(jb, acc, acc, 14, 15, L);
                }
                mask_pending = 0;
                break;
            case OP_DIV: {
                int64_t c = (int64_t)op->constant;
                if(c == 1)
                {
                    mask_pending = 0;
                    break;
                }
                int dk = 0, dneg = 0;
                if(is_pow2_div(c, &dk, &dneg))
                {
                    for(int v = 0; v < 4; v++)
                    {
                        if(mask_pending)
                        {
                            vex_vmovdqa_rr(jb, 14, (uint8_t)v, L);
                            emit_vex_pow2_div(jb, 14, dk, dneg, 15, L);
                            vex_vpblendvb(jb, (uint8_t)v, (uint8_t)v, 14, (uint8_t)(4 + v), L);
                        }
                        else
                            emit_vex_pow2_div(jb, (uint8_t)v, dk, dneg, 14, L);
                    }
                    mask_pending = 0;
                    break;
                }
                div_magic dm_chk;
                if(!mask_pending && compute_magic_i64(c, &dm_chk))
                {
                    for(int v = 0; v < 4; v++)
                        emit_vex_magic_div(jb, (uint8_t)v, c, L);
                    for(uint32_t j = i + 1; j < p->count; j++)
                    {
                        if(creg[j] < 0) continue;
                        emit_mov_imm64(jb, 0, (uint64_t)(int64_t)p->ops[j].constant);
                        vex_broadcast_rax(jb, (uint8_t)creg[j], L);
                    }
                    mask_pending = 0;
                    break;
                }
                for(int v = 0; v < 4; v++)
                    emit_vex_mem_disp(jb, 0x7F, (uint8_t)v, v * stride, L);
                for(int v = 0; v < 4; v++)
                {
                    for(int lane = 0; lane < width; lane++)
                    {
                        uint8_t off = (uint8_t)(v * stride + lane * 8);
                        vex_spill_load_rax(jb, off);
                        emit_magic_div_r64_const(jb, 0 /*RAX*/, c);
                        vex_spill_store_rax(jb, off);
                    }
                }
                for(int v = 0; v < 4; v++)
                    emit_vex_mem_disp(jb, 0x6F, (uint8_t)v, v * stride, L);
                mask_pending = 0;
                break;
            }
            default:
                break;
        }
    }

    /* Store back. (No NT variant: the kernel is in-place, and the
       §1 analysis showed NT stores lose on cache-hot lines.) */
    for(int v = 0; v < 4; v++)
        emit_vex_mem_disp(jb, 0x7F, (uint8_t)v, v * stride, L);

    emit_ret(jb);
    return 1;

fail:
    munmap(jb->mem.mem, jb->mem.size);
    return 0;
}

/* Engine wrapper: build the VEX kernel and install it (16 records
   per call on Haswell, 8 on Ivy Bridge). */
static int jit_compile_vex(engine *e, int is_avx2)
{
    jit_builder jb;
    jb_init(&jb, 128*1024);
    if(!jit_build_vex(&jb, &e->pipe, is_avx2))
        return 0;   /* builder buffer already released */
    jit_code_install(e, &jb, is_avx2 ? 16 : 8,
                     is_avx2 ? "pxasm_jit_vex256" : "pxasm_jit_vex128");
    return 1;
}

/* ============================================================
   Dispatch: choose between scalar and AVX512 JIT
   ============================================================ */

/* §5 helper, defined with the meta JIT below. */
static int pipeline_has_kmask(pipeline *p);

void engine_compile(engine *e)
{
    if (pulsar_use_avx512())
    {
        if(jit_compile_avx512_v3(e, 0))
        {
            jit_compile_avx512_v3(e, 1);
            return;
        }
        jit_code_nt_free(e);
        if(pipeline_has_kmask(&e->pipe))
            jit_compile_scalar(e);
        else
            jit_compile_avx512(e);
        return;
    }

    {
        int has_avx2 = pulsar_use_avx2();
        int has_avx  = __builtin_cpu_supports("avx");
        if((has_avx2 || has_avx) && jit_compile_vex(e, has_avx2 ? 1 : 0))
            return;
    }
    jit_compile_scalar(e);
}

/* ============================================================
   Run compiled JIT function
   ============================================================ */

/* ============================================================
   STREAMING JIT DRAIN
   ------------------------------------------------------------
   Pops every record queued by engine_ingest() and runs the compiled
   pipeline on it. Compiles first (once), so the function is
   self-contained.

   Width handling: the scalar kernel consumes one record's field data
   directly (r->data). The AVX-512 kernel needs 8 contiguous int64s,
   but ring records are arena-allocated and NOT contiguous, so full
   groups of 8 are gathered into a stack temp, run through the kernel,
   and scattered back; the tail (< 8) falls back to the interpreter.
   ============================================================ */

void engine_run_jit(engine *e)
{
    engine_compile(e);   /* ensure jit_fn/jit_width are ready */

    record *r;

    if(e->jit_width == 32)
    {
        /* v3 kernel: gather 32 records into a stack temp, run the
           4-vector kernel on it, scatter back. The tail (< 32) goes
           through the interpreter on its (untouched) original data. */
        int64_t lane_val[32];
        record *lane_rec[32];
        int n = 0;

        while((r = ring_pop(&e->queue)))
        {
            lane_rec[n] = r;
            lane_val[n] = rec_get_i64(r, 0);
            n++;

            if(n == 32)
            {
                /* ABI: RDI = pointer to 32 packed int64s */
                e->jit_fn((uint8_t*)lane_val, 32);
                for(int k = 0; k < 32; k++)
                    rec_set_i64(lane_rec[k], 0, lane_val[k]);
                n = 0;
            }
            e->exec_count++;
        }

        for(int k = 0; k < n; k++)
            interpret(e, lane_rec[k]);
    }
    else if(e->jit_width == 16)
    {
        /* v4 Haswell kernel: gather 16 records into a stack temp, run
           the 4-vector VEX.256 kernel on it, scatter back. The tail
           (< 16) goes through the interpreter on its original data. */
        int64_t lane_val[16];
        record *lane_rec[16];
        int n = 0;

        while((r = ring_pop(&e->queue)))
        {
            lane_rec[n] = r;
            lane_val[n] = rec_get_i64(r, 0);
            n++;

            if(n == 16)
            {
                /* ABI: RDI = pointer to 16 packed int64s */
                e->jit_fn((uint8_t*)lane_val, 16);
                for(int k = 0; k < 16; k++)
                    rec_set_i64(lane_rec[k], 0, lane_val[k]);
                n = 0;
            }
            e->exec_count++;
        }

        for(int k = 0; k < n; k++)
            interpret(e, lane_rec[k]);
    }
    else if(e->jit_width == 8)
    {
        int64_t lane_val[8];
        record *lane_rec[8];
        int n = 0;

        while((r = ring_pop(&e->queue)))
        {
            lane_rec[n] = r;
            lane_val[n] = rec_get_i64(r, 0);
            n++;

            if(n == 8)
            {
                /* ABI: RDI = pointer to 8 packed int64s */
                e->jit_fn((uint8_t*)lane_val, 8);
                for(int k = 0; k < 8; k++)
                    rec_set_i64(lane_rec[k], 0, lane_val[k]);
                n = 0;
            }
            e->exec_count++;
        }

        /* Tail: records were popped but never gathered; run them
           through the interpreter on their (untouched) original data. */
        for(int k = 0; k < n; k++)
            interpret(e, lane_rec[k]);
    }
    else
    {
        while((r = ring_pop(&e->queue)))
        {
            /* ABI: RDI = r->data, not the record* */
            e->jit_fn(r->data, 1);
            e->exec_count++;
        }
    }
}

