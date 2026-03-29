---
# poker_solver_rust-iy1d
title: Remove expected delta / equity cache feature
status: todo
type: task
created_at: 2026-03-29T23:12:16Z
updated_at: 2026-03-29T23:12:16Z
---

Remove the `expected_delta` option and its backing `EquityDeltaCache`. This provides deterministic delta computation but is 30× slower and adds significant code complexity.

## What to remove

- `expected_delta` config field in `config.rs`
- `equity_cache.rs` — the entire `EquityDeltaCache` struct and generation logic
- `equity_cache_path` config field and file I/O
- Any code in `cluster_pipeline.rs` that branches on `expected_delta` to use cached vs sampled deltas
- References in sample YAML configs

## Tasks

- [ ] Grep for all expected_delta and equity_cache references across the workspace
- [ ] Remove `EquityDeltaCache` and `equity_cache.rs`
- [ ] Remove config fields (`expected_delta`, `equity_cache_path`)
- [ ] Remove delta computation branching in cluster pipeline
- [ ] Remove the module declaration from `mod.rs` / `lib.rs`
- [ ] Ensure `cargo build` and `cargo test` pass
