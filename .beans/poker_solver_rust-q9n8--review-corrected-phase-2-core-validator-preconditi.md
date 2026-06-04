---
# poker_solver_rust-q9n8
title: Review corrected Phase 2 core validator preconditions
status: in-progress
type: task
priority: high
created_at: 2026-06-04T03:30:20Z
updated_at: 2026-06-04T03:30:20Z
parent: poker_solver_rust-l6r9
---

Independent review of corrective commit `b1c98416 Harden baseline validation preconditions`, following blockers found in review bean `poker_solver_rust-des1`.

Review focus:
- Validator now refuses non-169 preflop bucket providers before scoring.
- Validator now refuses non-20bb-equivalent trees/baselines before scoring, including wrong stack/all-in semantics, wrong big blind, wrong baseline metadata, and wrong action schema.
- Malformed/unparsable baseline hand rows are reported through counters/details rather than silently dropped.
- Prior correct behavior remains intact: six target spots resolve, `C` all-in-call mapping works, aggressive `RAI` mapping works, zero-mass rows are skipped/reported, and API remains suitable for trainer/TUI integration over `average_strategy`.

Reviewer should report blocking findings with file/line references and recommend whether trainer/TUI integration can proceed.
