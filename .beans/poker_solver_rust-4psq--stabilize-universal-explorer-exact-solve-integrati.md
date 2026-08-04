---
# poker_solver_rust-4psq
title: Stabilize universal Explorer exact-solve integration tests
status: in-progress
type: bug
priority: high
created_at: 2026-08-04T14:03:15Z
updated_at: 2026-08-04T14:13:41Z
---

The complete cargo test run passes unit crates but fails three tests in crates/tauri-app/tests/universal_explorer_integration.rs because exact-solve workers do not complete under full-suite parallel load: asymmetric flop snapshot, configured big blind root actions, and unrepresentable fractional action. Diagnose isolation/concurrency/timeout behavior and make the integration suite deterministic.

- [x] Research isolated versus full-suite behavior and test synchronization
- [x] Brainstorm the minimal deterministic test design
- [ ] Plan and dispatch implementation in an isolated worktree
- [ ] Review the repair
- [ ] Confirm targeted and complete suites pass
- [ ] Summarize the outcome
