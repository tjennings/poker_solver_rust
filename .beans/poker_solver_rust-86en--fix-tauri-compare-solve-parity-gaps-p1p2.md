---
# poker_solver_rust-86en
title: Fix Tauri compare-solve parity gaps P1/P2
status: in-progress
type: bug
priority: high
created_at: 2026-05-04T06:30:38Z
updated_at: 2026-05-04T06:30:38Z
parent: poker_solver_rust-e90m
---

Fix the two high-priority parity gaps from the Tauri vs compare-solve audit.\n\n## Tasks\n\n- [ ] Disable default frontend range clamping so Tauri solves the same ranges as compare-solve unless the user explicitly sets a clamp.\n- [ ] Align compare-solve A2 gadget blueprint seeding with Tauri by seeding from the real subgame root.\n- [ ] Update focused tests for the new defaults/seed helper.\n- [ ] Run full validation.
