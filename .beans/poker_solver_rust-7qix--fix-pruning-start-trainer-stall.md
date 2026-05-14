---
# poker_solver_rust-7qix
title: Fix pruning-start trainer stall
status: completed
type: bug
priority: high
created_at: 2026-05-14T17:34:59Z
updated_at: 2026-05-14T17:39:07Z
---

Trainer throughput drops to zero with near-zero CPU after pruning/negative-action purge starts. Investigate the lazy sparse pruning and post-DCFR negative-action purge path for deadlock or pathological maintenance work, then fix the stall without touching user config edits.

## Summary of Changes

- Replaced per-blocked-edge post-DCFR subtree purge scans with a batched boundary purge that indexes remaining blocked child prefixes and scans sparse storage once per discount boundary.
- Added a regression test proving multiple blocked edges produce one purge scan while preserving sibling histories.
- Updated training and architecture docs to describe the batched boundary sweep.

## Validation

- cargo fmt --check
- cargo test -p poker-solver-core negative_action -- --nocapture
- git diff --check
