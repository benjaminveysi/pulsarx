CC      ?= gcc
CFLAGS  ?= -O2 -mavx512f -mavx512dq -Wall -Wextra
LDFLAGS ?= -lpthread -lm

# -lnuma only if it actually links. Some boxes have numa.h but no
# libnuma — linking it blindly breaks the build there.
NUMA_LDFLAGS := $(shell printf 'int main(void){return 0;}\n' | $(CC) -x c - -lnuma -o /dev/null 2>/dev/null && echo -lnuma)
LDFLAGS += $(NUMA_LDFLAGS)

TARGET = pulsarx
SRC    = pulsarX_ASM.c

all: $(TARGET)

$(TARGET): $(SRC) pulsar_numa_compat.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

# Per-arch builds:
#   make ivybridge -> Ivy Bridge (AVX1, VEX.128 JIT, 8 rec/call)
#   make haswell   -> Haswell (AVX2, VEX.256 JIT, 16 rec/call)
CFLAGS_IVYBRIDGE  ?= -O2 -march=ivybridge -Wall -Wextra
CFLAGS_HASWELL ?= -O2 -march=haswell -Wall -Wextra

ivybridge: $(SRC) pulsar_numa_compat.h
	$(CC) $(CFLAGS_IVYBRIDGE) -o pulsarx-ivybridge $(SRC) $(LDFLAGS)

haswell: $(SRC) pulsar_numa_compat.h
	$(CC) $(CFLAGS_HASWELL) -o pulsarx-haswell $(SRC) $(LDFLAGS)

gates: metamorphic-gate scientific-gate
	./metamorphic-gate
	./scientific-gate

metamorphic-gate: metamorphic_shield_gate.c $(SRC) pulsar_numa_compat.h
	$(CC) $(CFLAGS) -o metamorphic-gate metamorphic_shield_gate.c $(LDFLAGS)

scientific-gate: scientific_gate.c $(SRC) pulsar_numa_compat.h
	$(CC) $(CFLAGS) -o scientific-gate scientific_gate.c $(LDFLAGS)

# The gates probe the host and skip execution the CPU can't run,
# so they never crash with an illegal instruction on older machines.
# These per-arch builds also keep AVX-512 out of the gate binaries.
#   make gates-ivybridge -> VEX-128 executes
#   make gates-haswell   -> VEX-256 + VEX-128 execute
gates-ivybridge: metamorphic-gate-ivybridge scientific-gate-ivybridge
	./metamorphic-gate-ivybridge
	./scientific-gate-ivybridge

metamorphic-gate-ivybridge: metamorphic_shield_gate.c $(SRC) pulsar_numa_compat.h
	$(CC) $(CFLAGS_IVYBRIDGE) -o metamorphic-gate-ivybridge metamorphic_shield_gate.c $(LDFLAGS)

scientific-gate-ivybridge: scientific_gate.c $(SRC) pulsar_numa_compat.h
	$(CC) $(CFLAGS_IVYBRIDGE) -o scientific-gate-ivybridge scientific_gate.c $(LDFLAGS)

gates-haswell: metamorphic-gate-haswell scientific-gate-haswell
	./metamorphic-gate-haswell
	./scientific-gate-haswell

metamorphic-gate-haswell: metamorphic_shield_gate.c $(SRC) pulsar_numa_compat.h
	$(CC) $(CFLAGS_HASWELL) -o metamorphic-gate-haswell metamorphic_shield_gate.c $(LDFLAGS)

scientific-gate-haswell: scientific_gate.c $(SRC) pulsar_numa_compat.h
	$(CC) $(CFLAGS_HASWELL) -o scientific-gate-haswell scientific_gate.c $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) pulsarx-ivybridge pulsarx-haswell metamorphic-gate scientific-gate metamorphic-gate-ivybridge scientific-gate-ivybridge metamorphic-gate-haswell scientific-gate-haswell

.PHONY: all run clean ivybridge haswell gates gates-ivybridge gates-haswell
