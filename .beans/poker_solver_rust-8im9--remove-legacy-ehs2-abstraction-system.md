---
# poker_solver_rust-8im9
title: Remove legacy EHS2 abstraction system
status: todo
type: task
created_at: 2026-03-29T23:08:47Z
updated_at: 2026-03-29T23:08:47Z
---

The legacy EHS2 abstraction system in crates/core/src/abstraction/ is fully superseded by the blueprint_v2 potential-aware EMD clustering pipeline. Remove the dead code.

## Context

The legacy system lives in `crates/core/src/abstraction/` and includes:
- `mod.rs` — CardAbstraction, suit canonicalization, bucket lookup
- `buckets.rs` — BucketAssigner (percentile-based binary search)
- `hand_strength.rs` — EHS/EHS2/PPot/NPot calculator

The blueprint_v2 system (`crates/core/src/blueprint_v2/`) has fully replaced it with:
- EMD k-means clustering on equity histograms
- Pre-computed bucket files with O(1) lookup
- Equity cache for delta features

## Tasks

- [ ] Identify all references to `crates/core/src/abstraction/` from other crates
- [ ] Identify any types/functions from the legacy system still used by blueprint_v2 (e.g. HandStrengthCalculator used in cluster_pipeline)
- [ ] Remove or inline anything still needed, delete the rest
- [ ] Remove the `abstraction` module declaration from `crates/core/src/lib.rs`
- [ ] Ensure `cargo build` and `cargo test` pass across the full workspace
- [ ] Clean up any orphaned config fields or CLI flags that referenced the old system
