---
# poker_solver_rust-cnjm
title: Defer negative-action subtree purge to DCFR discount boundary
status: completed
type: task
priority: high
created_at: 2026-05-14T16:49:04Z
updated_at: 2026-05-14T16:59:04Z
---

Change MP lazy sparse negative-action purge so physical subtree deletion only happens immediately after DCFR discounting is applied. Existing blocked-edge masking should still apply after prune_after_iterations, but new subtree drops should be batched at the discount boundary rather than happening during ordinary traversal transitions. The purge decision should use the post-discount regret state, so DCFR has the first chance to reactivate or soften negative edges before any physical descendant storage is dropped.

## Implementation Checklist

- [x] Split traversal gate transitions from physical sparse subtree purge
- [x] Add discount-boundary purge/reactivation sweep using post-discount regrets
- [x] Call boundary sweep immediately after lazy DCFR discounting
- [x] Update focused tests and docs wording
- [x] Run focused validation

## Summary of Changes

- Traversal-time negative-action gates now only update logical blocked-edge state.
- Sparse storage keeps child history prefixes for blocked edges and purges remaining blocked subtrees at the post-DCFR discount boundary.
- Lazy training invokes the boundary sweep immediately after discounting and accounts for it in lazy discount timing.
- Focused storage/lazy traversal tests and training/architecture docs were updated for the new contract.
