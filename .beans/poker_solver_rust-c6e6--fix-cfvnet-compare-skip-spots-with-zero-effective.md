---
# poker_solver_rust-c6e6
title: 'Fix cfvnet compare: skip spots with zero effective stack'
status: in-progress
type: bug
created_at: 2026-03-12T03:51:56Z
updated_at: 2026-03-12T03:51:56Z
---

sample_situation can generate effective_stack=0, which causes solve_situation to fail with 'effective_stack must be positive'. Two fixes needed: (1) sample_situation should use gen_range(1..=max_stack) and (2) run_comparison should skip unsolvable spots instead of failing the whole run.
