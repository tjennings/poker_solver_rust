---
# poker_solver_rust-yrlo
title: Harden mmap lazy reader compatibility and publication safety
status: in-progress
type: bug
priority: high
created_at: 2026-07-28T14:16:36Z
updated_at: 2026-07-28T14:16:36Z
parent: poker_solver_rust-osss
---

Review follow-up for the UniversalMpLazy mmap reader.

- Define and enforce the immutable bundle publication assumption so post-map truncation/modification cannot cause an unsafe SIGBUS path; prefer validation/fallback behavior that preserves a safe error.
- Align mmap header/payload length handling with the existing BundleReader trailing-byte compatibility contract, or document and test an intentional format change.
- Preserve the public query API for MP lazy callers where feasible; avoid exposing an unnecessary incompatible view type or provide a compatible adapter.
- Re-run focused core/Tauri tests and document the final contract.
