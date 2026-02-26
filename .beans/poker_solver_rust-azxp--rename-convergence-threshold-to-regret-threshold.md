---
# poker_solver_rust-azxp
title: Rename convergence_threshold to regret_threshold
status: completed
type: task
priority: normal
created_at: 2026-02-25T16:48:53Z
updated_at: 2026-02-25T17:41:20Z
---

Rename config field in TrainingConfig and PreflopTrainingConfig. Add serde alias for backward compat. Update sample configs.

## Todo
- [ ] Rename field in TrainingConfig with alias
- [ ] Rename field in PreflopTrainingConfig with alias
- [ ] Update all references in main.rs
- [ ] Update 5 sample YAML configs
- [ ] Verify compilation
