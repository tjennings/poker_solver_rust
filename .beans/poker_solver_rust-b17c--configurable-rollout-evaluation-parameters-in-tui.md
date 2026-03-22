---
# poker_solver_rust-b17c
title: Configurable rollout evaluation parameters in TUI settings
status: in-progress
type: task
created_at: 2026-03-22T02:14:31Z
updated_at: 2026-03-22T02:14:31Z
---

Add bias_factor (default 10.0), num_rollouts (default 3), num_opponent_samples (default 8) to Settings UI. Follow solve_iterations pattern: GlobalConfig type, Settings.tsx UI, PostflopExplorer reads at solve time, passes to backend.
