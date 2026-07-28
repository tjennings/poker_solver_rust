---
# poker_solver_rust-j5a6
title: Harden MP exact solve navigation invalidation and cache rewind
status: completed
type: bug
priority: high
created_at: 2026-07-28T20:37:17Z
updated_at: 2026-07-28T20:51:09Z
parent: poker_solver_rust-ja8p
---

Fix the final review findings for Universal MP lazy turn/river navigation.

- [x] Preserve completed exact-solve caches when back-navigation stays within the same street/state family.
- [x] Capture solve generation atomically with solve inputs and guard all worker completion/error state updates against stale generations.
- [x] Add deterministic regressions for same-street rewind and navigation during solve setup where feasible.
- [x] Update docs/architecture.md to remove stale turn/river unsupported wording.
- [x] Run focused Tauri/core tests and diff checks.

## Summary of Changes

- Preserved completed MP exact caches for replayable same-street back navigation and invalidated stale paths after street, board, or history changes.
- Serialized solve input capture with navigation and guarded stale worker snapshots, errors, and completion.
- Added same-street integration coverage and stale-generation publication assertions.
- Documented the Universal MP browser versus exact-solve boundary.

## Verification

- `cargo test -p poker-solver-tauri --lib`: 382 passed.
- `rustfmt --edition 2021` and `--check` passed for both owned Rust files.
- `git diff --check` passed.
