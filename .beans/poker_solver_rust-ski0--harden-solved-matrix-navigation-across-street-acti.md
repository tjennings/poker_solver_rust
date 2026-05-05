---
# poker_solver_rust-ski0
title: Harden solved matrix navigation across street actions
status: in-progress
type: bug
priority: critical
created_at: 2026-05-05T20:36:53Z
updated_at: 2026-05-05T20:36:53Z
---

Ensure solved subgame/exact representative matrices stay attached for every action navigation node within the solved street subtree, not just the first child transition.\n\n- [ ] Add broad regression coverage for navigating multiple actions after a street-root solve\n- [ ] Fix any cache/path miss found by the regression\n- [ ] Verify focused Tauri tests on main
