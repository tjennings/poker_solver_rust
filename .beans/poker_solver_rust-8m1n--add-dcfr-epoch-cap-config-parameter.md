---
# poker_solver_rust-8m1n
title: Add dcfr_epoch_cap config parameter
status: completed
type: feature
priority: normal
created_at: 2026-03-22T23:06:18Z
updated_at: 2026-03-22T23:12:38Z
---

Add dcfr_epoch_cap: Option<u64> to TrainingConfig to cap the DCFR epoch counter t, preventing discount factors from approaching 1.0 during indefinite training.

## Todo
- [ ] Add dcfr_epoch_cap field to TrainingConfig in config.rs
- [ ] Apply the cap in apply_lcfr_discount in trainer.rs
- [ ] Update test configs that construct TrainingConfig
- [ ] Add test verifying capping behavior
- [ ] Update sample YAML configs if applicable
- [ ] Run full test suite
