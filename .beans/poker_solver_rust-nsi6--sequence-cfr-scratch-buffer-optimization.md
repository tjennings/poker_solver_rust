---
# poker_solver_rust-nsi6
title: Sequence-CFR scratch buffer optimization
status: completed
type: feature
priority: normal
created_at: 2026-02-27T04:20:57Z
updated_at: 2026-02-27T04:28:27Z
---

Eliminate per-deal and per-node heap allocations in SequenceCfrSolver's hot loop by reusing scratch buffers, and add a Criterion benchmark harness.

## Tasks
- [x] Task 1: Add Criterion benchmark harness (baseline)
- [x] Task 2: Add regret_match_into (in-place regret matching)
- [x] Task 3: Add scratch buffers to SequenceCfrSolver and thread through passes
- [x] Task 4: Run benchmark and verify improvement

## Summary of Changes

Eliminated per-deal heap allocations in SequenceCfrSolver by adding 5 scratch buffer fields (reach_p1, reach_p2, utility, strategy, traversal) that are allocated once at construction and reused via std::mem::take.

### Results
- 100 iterations: 234µs → ~52% faster
- 1000 iterations: 2.36ms → ~52% faster
- 5000 iterations: 11.6ms → ~53% faster

### Files Changed
- `crates/core/src/cfr/regret.rs` — added `regret_match_into`
- `crates/core/src/cfr/sequence_cfr.rs` — scratch buffers, `current_strategy_into`, updated passes
- `crates/core/benches/sequence_cfr_bench.rs` — new Criterion benchmark
- `crates/core/Cargo.toml` — criterion dev-dependency
