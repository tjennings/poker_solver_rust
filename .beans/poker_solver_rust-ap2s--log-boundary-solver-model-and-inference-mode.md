---
# poker_solver_rust-ap2s
title: Log boundary solver model and inference mode
status: in-progress
type: task
created_at: 2026-05-08T19:46:16Z
updated_at: 2026-05-08T19:46:16Z
parent: poker_solver_rust-fp06
---

Add backend solve logging so Explorer solves print which boundary evaluator, inference mode, and model path are actually being used, including gadget-tree CFVNet solves where setup_neural_boundaries logging is bypassed.\n\n- [ ] Locate boundary resolution and gadget-tree logging path\n- [ ] Add concise solve log for boundary evaluator and model\n- [ ] Add/update tests if practical\n- [ ] Verify with targeted Rust tests and full cargo test\n- [ ] Commit bean and code
