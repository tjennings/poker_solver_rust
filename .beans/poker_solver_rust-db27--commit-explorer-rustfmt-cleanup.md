---
# poker_solver_rust-db27
title: Commit Explorer rustfmt cleanup
status: completed
type: task
priority: normal
created_at: 2026-06-22T19:12:18Z
updated_at: 2026-06-22T19:12:23Z
---

Commit the rustfmt-only cleanup in Explorer universal-bundle code and tests. Scope: crates/tauri-app/src/exploration.rs and crates/tauri-app/tests/universal_explorer_integration.rs. No behavior changes intended.

## Summary of Changes

- Committed rustfmt-only cleanup in `crates/tauri-app/src/exploration.rs`.
- Committed rustfmt-only cleanup in `crates/tauri-app/tests/universal_explorer_integration.rs`.
- Verified the scoped diff with `git diff --check`.
