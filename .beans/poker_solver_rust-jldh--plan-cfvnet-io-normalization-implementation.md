---
# poker_solver_rust-jldh
title: Plan CFVNet IO normalization implementation
status: completed
type: task
priority: high
created_at: 2026-05-14T01:04:39Z
updated_at: 2026-05-14T01:05:34Z
---

Create an implementation plan to normalize CFVNet boundary evaluator IO around the training-native model contract.\n\n- [x] Map required code changes by file\n- [x] Define canonical model/storage/consumer units\n- [x] Identify tests and docs to update\n- [x] Summarize rollout order and risk controls

## Summary of Changes\n\nCreated an implementation plan for normalizing CFVNet boundary IO. Canonical model contract: normalized 1326-combo ranges after blockers, current 2720-feature BoundaryNet input layout, and model output chip_cfv_over_pot_plus_effective_stack. Dataset pot-relative CFVs remain storage-only. Runtime should prefer raw chip-CFV BoundaryEvaluator integration, with legacy half-pot conversion isolated if still needed. Key edit sites: cfvnet eval boundary evaluator, range-solver boundary evaluator contract/tests, trainer model-kind defaults, docs architecture/training, and CFVNet unit tests.
