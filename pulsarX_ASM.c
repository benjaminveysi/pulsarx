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

