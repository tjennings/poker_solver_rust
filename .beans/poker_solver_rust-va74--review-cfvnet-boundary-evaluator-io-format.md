---
# poker_solver_rust-va74
title: Review CFVNet boundary evaluator IO format
status: completed
type: task
priority: high
created_at: 2026-05-14T00:55:45Z
updated_at: 2026-05-14T00:59:33Z
---

Review the CFVNet boundary evaluator input contract and the response translation path. Compare formats used by inference and training, identify unnecessary translation, and recommend the canonical format best suited for CFVNet training.\n\n- [x] Dispatch research/design review agents\n- [x] Inspect evaluator input construction and model response decoding\n- [x] Inspect CFVNet training data schema and loader expectations\n- [x] Compare formats and identify translation points\n- [x] Recommend canonical normalized format\n- [x] Summarize findings for user

## Summary of Changes\n\nReviewed delegated findings for CFVNet boundary evaluator input construction, response decoding, training record schema, target normalization, and range-solver boundary unit expectations. Recommendation: standardize model IO on 1326-combo canonical ranges with blocked combos zeroed, ranges normalized after blockers, feature layout oop_range/ip_range/board/rank/pot_ratio/stack_ratio/player, and model output in chip_cfv_over_pot_plus_effective_stack. Keep pot-relative CFVs only as a dataset storage detail or convert at loader boundaries. Identified mismatches: runtime inference ranges may not be normalized like training data; ONNX and Burn decode model outputs differently; ONNX legacy boundary scaling appears half-sized for range-solver half-pot units; default river-enumerated inference mode is unsafe for direct turn-boundary models.
