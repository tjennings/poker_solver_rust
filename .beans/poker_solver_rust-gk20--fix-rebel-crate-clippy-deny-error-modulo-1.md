---
# poker_solver_rust-gk20
title: Fix rebel crate clippy deny error (modulo-1)
status: todo
type: bug
priority: low
created_at: 2026-06-10T18:48:07Z
updated_at: 2026-06-10T18:48:07Z
---

cargo clippy -p rebel fails with a deny-level 'any number modulo 1 will be 0' error (plus 16 warnings), which also breaks cargo clippy -p poker-solver-trainer since trainer depends on rebel. Pre-existing, unrelated to the universal format work; discovered during Phase 3 verification on 2026-06-10. Fix the modulo-1 expression (likely a configurable interval defaulting to 1) and clean the warnings.
