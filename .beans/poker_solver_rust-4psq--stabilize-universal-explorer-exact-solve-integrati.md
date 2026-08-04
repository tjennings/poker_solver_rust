---
# poker_solver_rust-4psq
title: Stabilize universal Explorer exact-solve integration tests
status: completed
type: bug
priority: high
created_at: 2026-08-04T14:03:15Z
updated_at: 2026-08-04T14:32:49Z
---

The complete cargo test run passes unit crates but fails three tests in crates/tauri-app/tests/universal_explorer_integration.rs because exact-solve workers do not complete under full-suite parallel load: asymmetric flop snapshot, configured big blind root actions, and unrepresentable fractional action. Diagnose isolation/concurrency/timeout behavior and make the integration suite deterministic.

- [x] Research isolated versus full-suite behavior and test synchronization
- [x] Brainstorm the minimal deterministic test design
- [x] Plan and dispatch implementation in an isolated worktree
- [x] Review the repair
- [x] Confirm targeted and complete suites pass
- [x] Summarize the outcome

## Summary of Changes

Serialized the four memory-heavy detached exact-solve integration workers with a scoped test mutex and replaced expensive polling with generation-aware atomic lifecycle polling, bounded timeout handling, and cancellation acknowledgement. The formerly failing four-test group passed repeatedly under default parallelism, the universal Explorer integration binary passed 32/32, and the complete workspace suite passed in 563.90 seconds (runtime limit waived by the user).
