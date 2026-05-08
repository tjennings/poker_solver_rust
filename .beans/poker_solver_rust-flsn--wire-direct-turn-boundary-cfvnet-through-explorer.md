---
# poker_solver_rust-flsn
title: Wire direct turn-boundary CFVNet through Explorer solve UI
status: in-progress
type: task
created_at: 2026-05-08T19:16:36Z
updated_at: 2026-05-08T19:16:36Z
parent: poker_solver_rust-fp06
---

Expose turn-boundary CFVNet model path and model kind in the frontend solve configuration, defaulting to the new direct turn-boundary model mode while preserving legacy river_enumerated_turn behavior.\n\n- [ ] Research current frontend and Tauri solve config wiring\n- [ ] Add frontend fields and UI controls for turn model path and kind\n- [ ] Pass model path/kind through solve config without changing local model files\n- [ ] Verify with targeted tests or builds\n- [ ] Review changes and update docs if behavior changes
