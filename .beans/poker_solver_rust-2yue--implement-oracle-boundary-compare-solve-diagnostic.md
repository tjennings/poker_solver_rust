---
# poker_solver_rust-2yue
title: Implement oracle-boundary compare-solve diagnostic
status: in-progress
type: feature
priority: normal
created_at: 2026-05-04T00:10:05Z
updated_at: 2026-05-04T00:37:45Z
---

Add a diagnostic compare-solve boundary mode that feeds depth-limited subgame boundaries with oracle continuation values extracted from the full exact solve, so we can isolate whether exact_subtree divergence comes from the evaluator or boundary integration.\n\n- [x] Create feature branch\n- [x] Research boundary evaluator interfaces and exact-game traversal\n- [ ] Implement diagnostic oracle-boundary mode\n- [ ] Run focused tests and compare-solve validation\n- [ ] Summarize results

## Research Notes 2026-05-04

The existing exact_subtree path installs one evaluator per depth boundary before subgame solving. A true oracle boundary should instead solve the full exact game first, then attach per-boundary evaluators that evaluate the solved exact continuation at the matching subgame boundary history using the subgame solver's live boundary reaches. Range-solver has the recursive CFV machinery for this, but it needs a small public helper that starts from a supplied action history rather than always from root.
