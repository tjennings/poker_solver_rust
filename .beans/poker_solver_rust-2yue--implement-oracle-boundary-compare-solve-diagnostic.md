---
# poker_solver_rust-2yue
title: Implement oracle-boundary compare-solve diagnostic
status: in-progress
type: feature
priority: normal
created_at: 2026-05-04T00:10:05Z
updated_at: 2026-05-04T00:10:33Z
---

Add a diagnostic compare-solve boundary mode that feeds depth-limited subgame boundaries with oracle continuation values extracted from the full exact solve, so we can isolate whether exact_subtree divergence comes from the evaluator or boundary integration.\n\n- [x] Create feature branch\n- [ ] Research boundary evaluator interfaces and exact-game traversal\n- [ ] Implement diagnostic oracle-boundary mode\n- [ ] Run focused tests and compare-solve validation\n- [ ] Summarize results
