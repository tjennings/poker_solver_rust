---
# poker_solver_rust-ap2s
title: Log boundary solver model and inference mode
status: done
type: task
created_at: 2026-05-08T19:46:16Z
updated_at: 2026-05-08T19:49:43Z
parent: poker_solver_rust-fp06
---

Add backend solve logging so Explorer solves print which boundary evaluator, inference mode, and model path are actually being used, including gadget-tree CFVNet solves where setup_neural_boundaries logging is bypassed.

- [x] Locate boundary resolution and gadget-tree logging path
- [x] Add concise solve log for boundary evaluator and model
- [x] Add/update tests if practical
- [x] Verify with targeted Rust tests and full cargo test
- [x] Commit bean and code

Added a resolved `[solve] solver... boundary evaluator... inference_mode... model...` line before solve game construction so it covers both normal and gadget-tree paths. Covered Direct CFVNet and legacy CFVNet formatter output with unit tests.
