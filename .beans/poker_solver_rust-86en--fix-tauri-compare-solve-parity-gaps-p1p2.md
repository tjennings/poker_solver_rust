---
# poker_solver_rust-86en
title: Fix Tauri compare-solve parity gaps P1/P2
status: completed
type: bug
priority: high
created_at: 2026-05-04T06:30:38Z
updated_at: 2026-05-04T06:35:03Z
parent: poker_solver_rust-e90m
---

Fix the two high-priority parity gaps from the Tauri vs compare-solve audit.\n\n## Tasks\n\n- [x] Disable default frontend range clamping so Tauri solves the same ranges as compare-solve unless the user explicitly sets a clamp.\n- [x] Align compare-solve A2 gadget blueprint seeding with Tauri by seeding from the real subgame root.\n- [x] Update focused tests for the new defaults/seed helper.\n- [x] Run full validation.

## Summary of Changes\n\n- Changed frontend solve parameter defaults so rangeClampThreshold defaults to 0.0 instead of 0.05.\n- Changed the global config and Settings UI fallback for range_clamp_threshold to 0.0.\n- Changed compare-solve A2 gadget seeding to seed from the real subgame root, matching Tauri.\n- Added a compare-solve regression test for A2 seed start behavior and updated frontend tests for the zero clamp default.\n\n## Validation\n\n- cargo test (pre-change)\n- cargo test -p poker-solver-trainer compare_solve\n- ./node_modules/.bin/vitest run strategy-tabs Settings\n- npm run build\n- cargo test (post-change)
