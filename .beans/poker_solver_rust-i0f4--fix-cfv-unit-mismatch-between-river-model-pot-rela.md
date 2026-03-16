---
# poker_solver_rust-i0f4
title: Fix CFV unit mismatch between river model (pot-relative) and GPU solver (raw)
status: todo
type: bug
priority: high
tags:
    - gpu
    - correctness
created_at: 2026-03-16T03:40:34Z
updated_at: 2026-03-16T03:40:34Z
parent: poker_solver_rust-twez
---

The river model (trained by cfvnet) outputs pot-relative EVs.
The GPU solver's terminal payoffs use raw units (half_pot / num_combinations).
When the turn solver uses the river model at leaf boundaries, the leaf values
are pot-relative but the terminal payoffs are raw. This creates a mixed-unit
CFV at root that neither matches pot-relative nor raw.

Needs investigation:
- What units does CudaNetInference output? (pot-relative, from the river model)
- What units are fold terminal payoffs? (raw: half_pot / num_combinations)
- Are these consistent? No — they need to be aligned.

Options:
1. Convert river model output to raw units inside the leaf eval kernel
2. Convert fold payoffs to pot-relative units
3. Train turn/flop/preflop models in raw units and convert at inference time

The first run without scaling showed loss decreasing (912M -> 3.9M), suggesting
the model CAN learn from the mixed units, just at a different scale.
