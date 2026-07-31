---
# poker_solver_rust-yfkr
title: Implement range-solver reference audit harness
status: completed
type: task
priority: normal
created_at: 2026-05-19T13:52:48Z
updated_at: 2026-05-19T14:28:43Z
---

Implements the staged audit harness planned in poker_solver_rust-ixkg, starting by repairing the postflop-solver reference path/configuration and promoting deterministic fast river parity checks.\n\n- [x] Commit this implementation bean\n- [x] Run the full test suite baseline and confirm it completes under 1 minute\n- [x] Dispatch research/brainstorming on parity dimensions and staged scope\n- [x] Dispatch Rust implementation in a worktree\n- [x] Dispatch review of the implementation\n- [x] Integrate accepted changes into this branch\n- [x] Run the full test suite after code changes and confirm it completes under 1 minute\n- [x] Update docs if CLI, config, or workflow behavior changes

## Summary of Changes\n\nImplemented the first reference audit harness slice from poker_solver_rust-ixkg. Rewired range-solver-compare to the sibling ../../../postflop-solver checkout, added a deterministic river smoke parity suite with structural, exploitability, strategy, EV, and equity comparisons, preserved/unignored the fast 50-river identity test, and generated the standalone compare-crate Cargo.lock. Documented the sibling reference checkout and scoped RUSTFLAGS workaround in the compare crate manifest. Final verification passed, including the compare smoke and full cargo test in 54.47s.
