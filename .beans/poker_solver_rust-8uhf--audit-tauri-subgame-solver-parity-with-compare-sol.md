---
# poker_solver_rust-8uhf
title: Audit Tauri subgame solver parity with compare-solve
status: completed
type: task
priority: high
created_at: 2026-05-04T06:25:29Z
updated_at: 2026-05-04T06:27:42Z
parent: poker_solver_rust-e90m
---

Audit the Tauri subgame solve path against trainer compare-solve to identify any behavioral mismatches in game construction, boundary selection, evaluator wiring, gadget handling, seeding, iterations, and reported strategy output.\n\n## Scope\n\n- Compare Tauri solve entrypoints against crates/trainer/src/compare_solve.rs.\n- Focus on exact_subtree and cfvnet boundary behavior for subgame solves.\n- Report findings with file/line references and recommend follow-up fixes if needed.

## Audit Findings

1. Tauri frontend sends a default range_clamp_threshold of 0.05, while compare-solve does not clamp ranges. This can make Tauri solve a different game/range even when the spot, boundary config, and iteration count match.
2. Gadget tree seeding differs: compare-solve uses seed_start=4 in A2 gadget mode, while Tauri uses seed_start=0 because A2 keeps the real subgame root at game.root(). This can make gadget-enabled runs diverge before CFR starts.
3. Tauri UI matrix aggregation is reach-weighted, while compare-solve tolerance aggregation is uniform across combos. This does not change the solver, but can make reported hand-class probabilities look different.

## Parity Checks That Match

- Both paths share build_solve_game/build_solve_game_parts for card config, bet sizes, depth limit, rake, all-in thresholds, and tree construction.
- Both paths use the same resolve_street_boundary and validate_cfvnet_boundary_cut helpers.
- Both paths wire cfvnet and exact_subtree per-boundary evaluators with the same board/private-card ordering in non-gadget mode.
- Both paths clear boundary CFV caches each iteration, set the same DCFR boundary discount parameters, run solve_step with the same iteration index, finalize, cache normalized weights, and compute exploitability with lazy evaluators temporarily removed.

## Recommendation

Fix the frontend/config default so parity runs can send range_clamp_threshold=0.0, fix compare-solve A2 seed_start to 0, and add a small parity regression harness that builds Tauri-style and compare-style subgames from the same GameSession and asserts same root actions, boundary count, private-card count, and seeded root strategy before CFR.
