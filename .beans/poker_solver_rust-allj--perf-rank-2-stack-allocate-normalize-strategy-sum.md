---
# poker_solver_rust-allj
title: 'Perf Rank 2: Stack-allocate normalize_strategy_sum'
status: completed
type: task
priority: normal
created_at: 2026-02-28T22:40:15Z
updated_at: 2026-02-28T22:40:15Z
---

Replace Vec allocation in normalize_strategy_sum hot path with stack-allocated output buffer. ~30% speedup on exploitability checks in best_response_ev. Added normalize_strategy_sum_into() in postflop_abstraction.rs. Implemented in commit 8f44fb3.
