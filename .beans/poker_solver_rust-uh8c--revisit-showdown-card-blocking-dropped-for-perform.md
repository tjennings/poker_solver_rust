---
# poker_solver_rust-uh8c
title: Revisit showdown card blocking (dropped for performance)
status: todo
type: task
priority: normal
tags:
    - gpu
    - perf
created_at: 2026-03-15T19:30:56Z
updated_at: 2026-03-15T19:30:56Z
parent: poker_solver_rust-twez
---

The fast showdown kernel (showdown_eval_fast.cu) and 3-kernel O(n) showdown
(showdown_eval_sorted_batch.cu) both drop card blocking — they don't exclude
opponent hands that share cards with the traverser's hand. This is an
approximation that affects ~8% of matchups.

For training data, the noise is acceptable (neural nets are robust). But for
exact solving or production use, card blocking must be restored.

**Options:**
1. Add card blocking back to the 3-kernel prefix-sum approach using per-card
   ascending/descending prefix sums (matching CPU range-solver's algorithm)
2. Use a correction pass after the approximate showdown that adjusts only
   the ~8% of affected hands (O(k) where k ≈ 6 per hand)

**Why it was dropped:** The O(n²) shared-memory kernel with card blocking
took 30.7ms/iter. Without blocking: 4.5ms. The 3-kernel O(n) approach: 0.45ms.
Card blocking was the performance bottleneck due to thread divergence and
global memory reads for card comparison.

**How to apply:** Revisit after Phase 5 when the solver is used for
real-time resolving where exact strategies matter more than training data.
