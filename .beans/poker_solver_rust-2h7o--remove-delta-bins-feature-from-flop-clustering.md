---
# poker_solver_rust-2h7o
title: Remove delta bins feature from flop clustering
status: todo
type: task
created_at: 2026-03-29T23:12:11Z
updated_at: 2026-03-29T23:12:11Z
---

Remove the `delta_bins` config option that splits flop combos by expected equity change to the next street. This adds complexity to the clustering pipeline for unclear benefit.

## What to remove

- `delta_bins` config field in `config.rs` (e.g. `delta_bins: [0.0, 0.1]`)
- Delta bin partitioning logic in `cluster_pipeline.rs` that splits combos into sub-groups before clustering
- Any histogram augmentation that incorporates delta bin membership

## Tasks

- [ ] Grep for all delta_bins references across the workspace
- [ ] Remove config field and deserialization
- [ ] Remove delta bin partitioning logic from flop clustering
- [ ] Simplify flop clustering to use a single pass without delta splits
- [ ] Ensure `cargo build` and `cargo test` pass
