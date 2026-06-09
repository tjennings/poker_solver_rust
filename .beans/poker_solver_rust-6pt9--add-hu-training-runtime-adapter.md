---
# poker_solver_rust-6pt9
title: Add HU training runtime adapter
status: in-progress
type: task
priority: high
created_at: 2026-06-09T15:29:54Z
updated_at: 2026-06-09T15:45:12Z
parent: poker_solver_rust-tzv5
---

Add a heads-up BlueprintTrainer adapter for the shared training runtime. Scope: create a small adapter that implements TrainingRuntimeBackend for the existing HU BlueprintTrainer/control surface, preserves the arena/lazy tree and current MCCFR traversal unchanged, seeds RuntimeCounters from restored iteration state, uses BatchBudget to cap finite run batches, and adds focused adapter tests. Non-goals: do not modify HU traversal, game_tree arena storage, MP trainers, CLI command wiring, or TUI shells in this slice.



Review findings before completion:
- P1: runtime adapter must preserve legacy startup validation: initial baseline validation and postflop bucket-file validation must run before batches, matching BlueprintTrainer::train().
- P1: runtime adapter must preserve target_strategy_delta stopping semantics from BlueprintTrainer::should_stop(); shared runtime currently only sees iteration/time limits.
- Non-blocker: batch extraction preserved MCCFR traversal body and arena/lazy storage semantics; controls/counters/budget plumbing looked sound.
