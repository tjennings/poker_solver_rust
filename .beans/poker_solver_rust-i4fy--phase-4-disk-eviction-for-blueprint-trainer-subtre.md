---
# poker_solver_rust-i4fy
title: 'Phase 4: disk eviction for blueprint trainer subtrees'
status: draft
type: feature
priority: high
created_at: 2026-06-03T18:10:21Z
updated_at: 2026-06-03T18:10:49Z
parent: poker_solver_rust-34kn
blocked_by:
    - poker_solver_rust-bgbz
---

Phase 4 of the blueprint trainer tree roadmap.

Scope:
- Add lossless disk eviction for resident tree memory pressure after in-memory pruning is working.
- Define durable subtree IDs independent of whether nodes are resident.
- Introduce a storage format for evicted subtrees, including versioning, dirty tracking, checksum/integrity validation, atomic temp-file writes/renames, manifest updates, and deterministic reload behavior.
- Start with max resident nodes as the memory pressure trigger.
- Rank eviction by coldness, subtree size, dirty state/reload cost, and negative-performance/pruning signals; do not rely only on worst regret.
- Add metrics: evictions, reloads, bytes written/read, dirty flushes, cache hit/miss rate, resident node high-water mark.
- Update docs/training.md for config and docs/architecture.md for storage lifecycle.

Acceptance criteria:
- With eviction disabled, behavior matches Phase 3.
- A subtree round-trip test realizes, mutates CFR state, evicts, reloads, continues, and matches a never-evicted run.
- With a low max-node threshold, subtrees spill and reload deterministically in a small trainer run.
- Disk-backed runs preserve strategy/regret state, action order, child realization state, and discount epoch across eviction/reload.
- Corrupt/incompatible subtree files fail loudly with useful errors.
- Runtime impact is measured and documented.

Blocked by Phase 3 pruning and resident subtree identity design.
