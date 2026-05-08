---
# poker_solver_rust-flsn
title: Wire direct turn-boundary CFVNet through Explorer solve UI
status: completed
type: task
priority: normal
created_at: 2026-05-08T19:16:36Z
updated_at: 2026-05-08T19:21:04Z
parent: poker_solver_rust-fp06
---

Expose turn-boundary CFVNet model path and model kind in the frontend solve configuration, defaulting to the new direct turn-boundary model mode while preserving legacy river_enumerated_turn behavior.\n\n- [x] Research current frontend and Tauri solve config wiring\n- [x] Add frontend fields and UI controls for turn model path and kind\n- [x] Pass model path/kind through solve config without changing local model files\n- [x] Verify with targeted tests or builds\n- [x] Review changes and update docs if behavior changes

## Summary of Changes

Added Explorer frontend support for turn-boundary CFVNet model kind selection. Turn CFVNet now defaults to direct inference, river CFVNet preserves the legacy river-enumerated default, solve params pass inference_mode through to Tauri, and Explorer docs describe the new direct checkpoint path and legacy mode.
