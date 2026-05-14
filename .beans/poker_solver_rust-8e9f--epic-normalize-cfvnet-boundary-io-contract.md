---
# poker_solver_rust-8e9f
title: 'Epic: normalize CFVNet boundary IO contract'
status: in-progress
type: epic
priority: high
created_at: 2026-05-14T01:10:26Z
updated_at: 2026-05-14T01:10:26Z
---

Normalize CFVNet boundary model IO around the training-native BoundaryNet contract.\n\nCanonical model contract:\n- Input: 1326 canonical OOP range + 1326 canonical IP range + board one-hot + rank presence + pot/(pot+stack) + stack/(pot+stack) + player.\n- Ranges: board/river blockers zeroed, non-negative, finite, normalized to sum 1 after blockers.\n- Output: chip_cfv / (pot + effective_stack), one 1326-vector in canonical combo order.\n- Dataset pot-relative CFVs are storage-only.\n- Range-solver half-pot boundary units are legacy adapter-only.\n\nSubtasks cover contract helpers, inference input normalization, output units/raw CFV integration, inference-mode defaults, tests, docs, and the existing full-suite runtime blocker.
