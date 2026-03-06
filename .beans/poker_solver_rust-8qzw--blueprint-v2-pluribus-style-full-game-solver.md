---
# poker_solver_rust-8qzw
title: 'Blueprint V2: Pluribus-style full-game solver'
status: completed
type: epic
priority: normal
created_at: 2026-03-06T08:21:33Z
updated_at: 2026-03-06T15:55:52Z
---

Full-game MCCFR solver training all 4 streets with potential-aware clustering.

Design: docs/plans/2026-03-06-blueprint-v2-design.md
Plan: docs/plans/2026-03-06-blueprint-v2-plan.md

## Tasks (18 total)

### Phase 1: Config & I/O
- [x] Task 1: Config types
- [x] Task 2: Bucket file format
- [x] Task 3: k-means clustering (EMD + 1-D)

### Phase 2: Clustering Pipeline
- [x] Task 4: River clustering
- [x] Task 5: Turn clustering
- [x] Task 6: Flop clustering
- [x] Task 7: Preflop clustering
- [x] Task 8: Cluster CLI
- [x] Task 9: Cluster diagnostics

### Phase 3: Game Tree
- [x] Task 10: Game tree builder

### Phase 4: MCCFR Engine
- [x] Task 11: Strategy/regret storage
- [x] Task 12: MCCFR traversal
- [x] Task 13: Training loop

### Phase 5: Output & Integration
- [x] Task 14: Bundle format
- [x] Task 15: Explorer integration
- [x] Task 16: train-blueprint CLI
- [x] Task 17: Sample configs

### Phase 6: E2E
- [x] Task 18: E2E test


## Summary of Changes

### Implementation (18 tasks)
- **Config & I/O**: BlueprintV2Config types, BucketFile binary format, k-means clustering with EMD
- **Clustering Pipeline**: Potential-aware card abstraction for all 4 streets (river → turn → flop → preflop)
- **Game Tree**: Arena-allocated game tree covering preflop through river
- **MCCFR Engine**: External-sampling MCCFR with LCFR weighting, flat-buffer storage
- **Bundle & Explorer**: Strategy extraction, snapshot I/O, explorer integration
- **CLI**: train-blueprint, cluster, diag-clusters commands with sample configs
- **E2E Test**: Full pipeline integration test

### Review Fixes (6 commits)
- O(1) get_action_probs with precomputed offsets
- Trainer save_snapshot now includes strategy.bin
- Hot-path heap allocations eliminated (current_strategy_into, stack arrays, deck field)
- AllBuckets wired to load bucket files from cluster_path
- DRY cluster_pipeline (-91 lines)
- Slow tests marked #[ignore] (4.68s test suite)

### Review Results (2 rounds × 3 reviewers)
- software-architect: APPROVED
- rust-perf-reviewer: APPROVED  
- idiomatic-rust-enforcer: APPROVED (false positives on already-fixed code)
