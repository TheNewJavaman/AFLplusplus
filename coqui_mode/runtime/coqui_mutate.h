/*
 * coqui_mutate.h --- GPU-side havoc mutation runtime.
 *
 * Declares the single public entry point for AFL's havoc stage running on
 * the GPU. All state (seed pool, extras, mutation weight table, PRNG base)
 * is bound by the host via module-level extern globals — see
 * coqui_runtime.h for the extern decls.
 *
 * Spec: docs/superpowers/specs/2026-04-21-gpu-havoc-mutations-design.md §4.2
 */

#ifndef _COQUI_MUTATE_H
#define _COQUI_MUTATE_H

#include "coqui_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mutate `buf` in place using AFL havoc semantics. Returns the new length.
 * `buf` must have at least `max_len` bytes of headroom (may grow).
 * `self_idx` is this thread's seed pool index — splice ops exclude it to
 * avoid self-splice. `prng` is the per-thread splitmix64 state. */
u32 __coqui_havoc_mutate(u8 *buf, u32 len, u32 max_len,
                          u32 self_idx, u64 *prng);

#ifdef __cplusplus
}
#endif

#endif /* _COQUI_MUTATE_H */
