---
# poker_solver_rust-cnjm
title: Defer negative-action subtree purge to DCFR discount boundary
status: in-progress
type: task
priority: high
created_at: 2026-05-14T16:49:04Z
updated_at: 2026-05-14T16:50:23Z
---

Change MP lazy sparse negative-action purge so physical subtree deletion only happens immediately after DCFR discounting is applied. Existing blocked-edge masking should still apply after prune_after_iterations, but new subtree drops should be batched at the discount boundary rather than happening during ordinary traversal transitions. The purge decision should use the post-discount regret state, so DCFR has the first chance to reactivate or soften negative edges before any physical descendant storage is dropped.
