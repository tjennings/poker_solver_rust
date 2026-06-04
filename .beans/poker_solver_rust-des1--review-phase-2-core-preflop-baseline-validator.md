---
# poker_solver_rust-des1
title: Review Phase 2 core preflop baseline validator
status: in-progress
type: task
priority: high
created_at: 2026-06-04T03:17:30Z
updated_at: 2026-06-04T03:17:30Z
parent: poker_solver_rust-l6r9
---

Independent review of implementation commit `d243086c Add blueprint baseline validation core` for Phase 2 slice `poker_solver_rust-28c2`.

Review focus:
- Baseline JSON parser matches `local_data/baselines/cash_hu_20bb_cev.json` and handles missing/unknown fields sensibly.
- Spot resolver correctly maps the six preflop baseline paths under the exact 20bb-equivalent tree config (`stack_depth: 40`, no limp, `2.5bb` then `5bb`).
- Context-aware action mapping is correct, especially `C` vs all-in-call and `RAI` vs aggressive all-in.
- Metrics compute combo-weighted total variation distance correctly and skip/report zero-mass baseline rows without hiding meaningful mismatches.
- Unsupported/unmapped actions and candidate mass are reported explicitly.
- API shape is suitable for trainer/TUI integration over `BlueprintCfrStorage::average_strategy` without dense projection.
- Tests cover the intended claims and stay cheap.

Reviewer should report blocking findings with file/line references and recommend whether the trainer/TUI integration slice can proceed.
