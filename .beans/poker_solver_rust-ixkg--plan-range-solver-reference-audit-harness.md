---
# poker_solver_rust-ixkg
title: Plan range-solver reference audit harness
status: completed
type: task
priority: normal
created_at: 2026-05-19T13:26:37Z
updated_at: 2026-05-19T13:31:34Z
---

Inspect existing range-solver/postflop-solver comparison tooling and produce an implementation plan for an audit harness that validates behavioral identity, starting with quick river spots, then turn spots, then low-SPR 4-bet flop spots.\n\n- [x] Inspect existing comparison crates, scripts, tests, and docs\n- [x] Inspect sibling ../postflop-solver shape and available APIs/CLI\n- [x] Identify parity dimensions and fast validation spot suite\n- [x] Produce staged audit-tool implementation plan

## Summary of Changes\n\nInspected the existing range-solver reference comparison path. Found that crates/range-solver-compare already links range-solver and postflop-solver and compares exploitability, root strategy, EVs, and equities, but it is excluded from the workspace and currently points to missing external/postflop-solver while the available reference checkout is ../postflop-solver. Confirmed the sibling postflop-solver is a library-first reference with examples but no parameterized CLI. Planned a staged audit harness: repair reference path/configuration, promote deterministic river suites, add turn suites, add low-SPR 4-bet flop suites, deepen comparisons across actions/private-hand ordering/current EV/chance navigation/selected histories, and keep local range-solver extensions separate from upstream parity.
