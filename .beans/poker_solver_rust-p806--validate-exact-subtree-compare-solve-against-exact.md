---
# poker_solver_rust-p806
title: Validate exact_subtree compare-solve against exact
status: completed
type: task
priority: normal
created_at: 2026-05-04T00:00:04Z
updated_at: 2026-05-04T00:03:08Z
---

Run trainer compare-solve for the canonical JhTh9h 4-bet spot with --river-boundary exact_subtree and compare the boundary Subgame result against the full-depth Exact solution.\n\n- [x] Build poker-solver-trainer release binary\n- [x] Run compare-solve with exact_subtree boundary\n- [x] Summarize exact_exp, subgame_exp, and worst_delta

## Summary of Changes\n\nRan the release trainer build and compare-solve for the canonical JhTh9h7d spot with --river-boundary exact_subtree, --iters 200, and --tolerance 1.0. The run completed successfully but showed the boundary Subgame was far from exact: exact_exp 23.06 mbb/hand, subgame_exp 1063.89 mbb/hand, exploitability delta +1040.83 mbb/hand worse, mean mass moved per hand 0.438, max mass moved 1.000.
