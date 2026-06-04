---
# poker_solver_rust-nobl
title: 'Phase 2 slice: trainer and TUI baseline convergence'
status: in-progress
type: task
priority: high
created_at: 2026-06-04T03:48:24Z
updated_at: 2026-06-04T03:48:24Z
parent: poker_solver_rust-l6r9
---

Trainer/TUI integration slice for Phase 2 baseline validation.

Scope:
- Add config for `training.baseline_validation`: enabled, baseline_path, interval_iterations and/or interval_minutes, top_n_spots, top_n_combos_per_spot. Defaults must leave validation disabled.
- Wire `BlueprintTrainer` to load the baseline when enabled and compute validation reports against `active_storage()` without dense projection.
- Fill `BaselineGamePreconditions` from the actual original `GameConfig` values: stack_depth, small_blind, big_blind, allow_preflop_limp, and preflop bucket count from storage/config. Do not fabricate pinned values in integration.
- Report convergence in no-TUI progress/log output: aggregate TV, root TV, first-response TV, worst spot TV, coverage, skipped zero-mass rows, invalid rows, unsupported spots/actions, and top 5 worst spots with worst combo rows.
- Extend TUI metrics/rendering to show baseline convergence and top 5 worst spots plus diagnostic data, in the existing dense/professional style.
- Add a reproducible sample config for the 20bb cEV baseline: stack_depth 40, blinds 1/2, limp disabled, preflop buckets 169, preflop actions `2.5bb` then `5bb`, pruning/eviction disabled for validation.
- Update `docs/architecture.md` and `docs/training.md` for config, CLI/runbook, metrics, limitations, and non-EV strategy-frequency validation.
- Add tests for config defaults, enabled validation path, trusted preconditions sourced from actual config, no-TUI/TUI metric formatting as practical, and unsupported wrong config rejection.

Non-goals: no range-solver validation, no EV pass/fail, no pruning or disk eviction, no sparse on-disk changes.
