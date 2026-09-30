#pragma once
/* Portable NUMA header: uses libnuma when available, otherwise falls back
   to plain malloc. PulsarX only needs numa_alloc_onnode for its arenas. */
#include <stdlib.h>

#if defined(__has_include)
#  if __has_include(<numa.h>)
#    include <numa.h>
#  else
static inline void *numa_alloc_onnode(size_t size, int node)
{
    (void)node;
    return malloc(size);
}
#  endif
#else
#  include <numa.h>
#endif
