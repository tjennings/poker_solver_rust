---
# poker_solver_rust-l6r9
title: 'Phase 2: validate lazy tree trainer against known-good HU game'
status: in-progress
type: task
priority: high
created_at: 2026-06-03T18:09:40Z
updated_at: 2026-06-04T02:58:00Z
parent: poker_solver_rust-34kn
---

Phase 2 of the blueprint trainer lazy/sparse storage roadmap.

Scope:
- Use the smallest supplied heads-up cEV baseline: `local_data/baselines/cash_hu_20bb_cev.json` (`game.stackDepthBb = 20`, heads-up cash cEV, 2.5x open size).
- Train the smallest stack-size-equivalent HU blueprint configuration available in this repo, with pruning and disk eviction disabled.
- Extend the blueprint trainer validation path to compare learned preflop strategy against the supplied baseline's preflop spots.
- Extend the blueprint trainer TUI/progress surface to show convergence toward the baseline during training.
- Display the top 5 worst baseline mismatches, with enough data for each spot to diagnose the mismatch: spot key/path/history, position to act, actions, aggregate action deltas, per-combo worst deltas, learned frequencies, baseline frequencies, and a scalar mismatch score.
- Preserve Phase 1 dense export/resume compatibility and support the new sparse/lazy storage backend as an opt-in backend where practical.
- Update `docs/architecture.md` and `docs/training.md` if new commands/config/TUI behavior are added.

Acceptance criteria:
- A reproducible CLI/TUI validation run can train the smallest stack-size-equivalent HU blueprint and report convergence toward `cash_hu_20bb_cev.json`.
- Preflop strategy comparison uses documented action mapping/tolerances between trainer actions and baseline action labels.
- The TUI/progress output includes a baseline convergence metric and top 5 worst spots with diagnostic detail.
- Any mismatch report is actionable: spot/history, player/position, legal actions, baseline strategy, learned strategy, and score are visible.
- Full test suite passes in under 1 minute, or any runtime violation is fixed/beaned before proceeding.
- No Phase 3 pruning or Phase 4 disk eviction logic is enabled during validation.

Implementation must be delegated to rust-developer/worker agents; manager does not write Rust directly.
