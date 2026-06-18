---
# poker_solver_rust-mt3l
title: Store action identity in lazy sparse rows at realization
status: todo
type: feature
priority: high
created_at: 2026-06-10T19:06:22Z
updated_at: 2026-06-18T15:05:52Z
parent: poker_solver_rust-osss
blocking:
    - poker_solver_rust-9p59
---

Lazy sparse rows currently store only num_actions; universal exports carry Opaque action descriptors because kinds/amounts are unrecoverable (>32-action histories are hash-only). Since backwards compatibility of the sparse snapshot schema is NOT required (only Tauri Explorer compatibility matters, per the 2026-06-10 goal recap), store action identity at realization time — e.g. per-row action schema fingerprint plus a bundle-level action table, or inline compact descriptors — so lazy universal exports carry real actions. Prerequisite for meaningful Explorer browsing of lazy-export bundles.

## Note (2026-06-18)
No longer marked as blocking Phase 6 (9p59). Phase 6 delivers read-only MP bundle info + row lookup without needing real action labels. mt3l gates the FUTURE full MP-lazy browsing UI (rich rendering with real action descriptors), which is a follow-on to Phase 6.
