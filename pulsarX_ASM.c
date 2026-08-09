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

