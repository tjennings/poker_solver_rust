---
# poker_solver_rust-gv8m
title: 'CFVNet IO: update architecture and training docs'
status: completed
type: task
priority: normal
created_at: 2026-05-14T01:11:04Z
updated_at: 2026-05-14T01:37:32Z
parent: poker_solver_rust-8e9f
---

Document the normalized CFVNet boundary IO contract and runtime adapters.\n\n- [x] Update docs/architecture.md BoundaryNet section\n- [x] Update docs/training.md model-kind/direct-vs-river-enumerated guidance\n- [x] Clarify dataset storage units vs model output units\n- [x] Clarify range-solver legacy half-pot units are adapter-only\n\nPrimary files: docs/architecture.md, docs/training.md

## Summary of Changes

Updated docs/architecture.md and docs/training.md to document the canonical BoundaryNet model IO contract, range normalization after blockers, dataset storage units versus model target units, raw chip-CFV integration, legacy half-pot adapter units, and direct versus river_enumerated_turn model-kind guidance. Verified markdown diff with git diff --check.
