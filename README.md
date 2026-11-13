# PulsarX: A Single-File Native Execution Engine

[![Language: C](https://img.shields.io/badge/language-C-blue.svg)]()
[![Version: v5.2](https://img.shields.io/badge/version-v5.2-green.svg)]()

A single-file C engine that runs pipelines of arithmetic operators over streams of records — and compiles them to native machine code at runtime. One file, `pulsarX_ASM.c` (~6,100 lines), no dependencies beyond libc, pthreads and libm.

Designed for the machine it finds itself on: it picks the fastest way to run your pipeline, from a 2012 Ivy Bridge to an AVX-512 server, and never crashes on an ISA it doesn't have.

## Why PulsarX?

- **Five ways to run, one dispatch:** Scalar, SIMD, JIT, Metamorphic JIT and Trace JIT. The engine probes the CPU and picks the fastest kernel — AVX-512, then VEX-256, then VEX-128, then scalar — and degrades gracefully instead of crashing.
- **Math that shrinks the work:** Kempner 2-adic reduction collapses a degree-4096 polynomial to degree 64, bit-exact. Estrin tree kernels replace the serial Horner chain. VEX division is synthesized natively, without VPSRAQ.
- **Metamorphism instead of ASLR:** The same pipeline re-randomized per compile (register allocation, NOPs, instruction order) — **120/120 gadget death** on scalar, EVEX-512 and VEX-256 alike.
- **It tunes itself:** An online RDTSCP bandit autotuner probes the candidate kernels on the host and locks in the physically fastest one.

## Verified numbers

**1. Throughput (AVX-512 VM, 200M records, pipeline `((x+10)*3)²`)**
- Scalar **~70 M rec/s**; SIMD **~153**; JIT **~2810**; Metamorphic JIT **~1560**; Trace JIT **~2610**.

**2. Ivy Bridge i5 (2012, real hardware)**
- SIMD **162.81 M rec/s** vs Scalar **94.15** (1.73×). Autotuner locks onto VEX-128 at **~737 M rec/s**. Both gates pass.

**3. Gates (both exit 0 on pass)**
- Metamorphic shield: **120/120** gadget death, 120/120 pairwise-different code, bit-identical outputs.
- Scientific: Kempner **4096 → 64, 4096/4096 bit-identical**; VEX division bit-exact across **26,624** combinations; **65,536** Rivest roundtrips.

## Quickstart

```bash
git clone https://github.com/benjaminveysi/pulsarx.git
cd pulsarx
make              # AVX-512 binary, ./pulsarx
make ivybridge    # Ivy Bridge (AVX1, VEX-128)
make haswell      # Haswell (AVX2, VEX-256)
make gates        # run both verification gates
```

License: free to use, study and modify with attribution to Vaysi Development Center (VDC); commercial use requires a separate written license. See the header comment of `pulsarX_ASM.c`.
