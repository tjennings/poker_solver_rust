---
# poker_solver_rust-3yat
title: Bring full workspace tests under one minute
status: in-progress
type: bug
priority: high
created_at: 2026-06-24T18:35:44Z
updated_at: 2026-06-24T18:35:44Z
---

The required preflight for poker_solver_rust-n8tv passed but missed the project timing gate: cold cargo test --workspace --quiet took 278.13s and hot rerun took 70.99s. AGENTS.md requires the entire test suite to complete in less than 1 minute before development work proceeds. Scope: identify the slow workspace test targets/orchestration overhead and make the standard full-suite command complete under 60s without weakening meaningful coverage, or document and implement the repo-approved replacement command if the intended suite differs from cargo test --workspace. Acceptance: hot full-suite preflight completes in <60s and passes.
