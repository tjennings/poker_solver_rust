---
# poker_solver_rust-fxpk
title: Remove per-flop clustering feature
status: todo
type: task
created_at: 2026-03-29T23:12:07Z
updated_at: 2026-03-29T23:12:07Z
---

Remove the per-flop clustering option that generates separate turn/river bucket files per flop texture. This creates massive file counts and complexity for marginal accuracy gains.

## What to remove

- Per-flop bucket file loading/caching in `mccfr.rs` (`per_flop_cache: RwLock<FxHashMap>`, `PerFlopBucketFile`)
- Per-flop clustering logic in `cluster_pipeline.rs`
- `flop_NNNN.buckets` file generation and directory handling in `bucket_file.rs`
- Config fields: `per_flop.turn_buckets`, `per_flop.river_buckets` in `config.rs`
- Any YAML sample configs referencing `per_flop`

## Tasks

- [ ] Grep for all per-flop references across the workspace
- [ ] Remove per-flop config fields and deserialization
- [ ] Remove per-flop clustering pipeline code
- [ ] Remove per-flop bucket file loading and caching from MCCFR
- [ ] Simplify bucket lookup to always use global bucket files
- [ ] Ensure `cargo build` and `cargo test` pass
