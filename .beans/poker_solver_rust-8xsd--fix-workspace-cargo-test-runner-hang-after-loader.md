---
# poker_solver_rust-8xsd
title: Fix workspace cargo test runner hang after loader changes
status: todo
type: bug
priority: high
created_at: 2026-06-24T20:35:07Z
updated_at: 2026-06-24T20:35:07Z
---

After the MP lazy sparse loader patch, focused tests pass, but broad workspace verification is not producing a valid full-suite result in this shell. `cargo test --workspace --quiet` and `cargo test --workspace --quiet --lib --tests` repeatedly advance through many test binaries, including core loader tests, then sit with only the cargo process visible and no active rustc/test child. This prevents satisfying the AGENTS.md under-one-minute full-suite gate. Scope: identify whether Cargo is waiting on a specific test binary, doctest/libtest interaction, process cleanup, or environment issue; make the standard workspace verification finish reliably under 60s again. Acceptance: hot `cargo test --workspace --quiet` passes and exits under 60s, or AGENTS.md/test docs are updated to the repo-approved equivalent command.
