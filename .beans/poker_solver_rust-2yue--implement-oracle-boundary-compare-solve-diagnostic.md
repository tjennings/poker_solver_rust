---
# poker_solver_rust-2yue
title: Implement oracle-boundary compare-solve diagnostic
status: completed
type: feature
priority: normal
created_at: 2026-05-04T00:10:05Z
updated_at: 2026-05-04T00:54:06Z
---

Add a diagnostic compare-solve boundary mode that feeds depth-limited subgame boundaries with oracle continuation values extracted from the full exact solve, so we can isolate whether exact_subtree divergence comes from the evaluator or boundary integration.\n\n- [x] Create feature branch\n- [x] Research boundary evaluator interfaces and exact-game traversal\n- [x] Implement diagnostic oracle-boundary mode\n- [x] Run focused tests and compare-solve validation\n- [x] Summarize results

## Research Notes 2026-05-04

The existing exact_subtree path installs one evaluator per depth boundary before subgame solving. A true oracle boundary should instead solve the full exact game first, then attach per-boundary evaluators that evaluate the solved exact continuation at the matching subgame boundary history using the subgame solver's live boundary reaches. Range-solver has the recursive CFV machinery for this, but it needs a small public helper that starts from a supplied action history rather than always from root.

## Summary of Changes

Added a compare-solve exact_oracle boundary diagnostic that solves the full exact game first, then feeds depth-limited boundaries from the solved exact continuation using live subgame reaches. Added range-solver CFV extraction by play history, CLI/docs coverage, focused tests, and verified the full cargo test gate in 51.705s warm.
