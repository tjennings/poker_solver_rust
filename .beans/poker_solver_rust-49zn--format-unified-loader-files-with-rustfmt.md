---
# poker_solver_rust-49zn
title: Format unified loader files with rustfmt
status: completed
type: task
priority: normal
created_at: 2026-07-28T13:19:24Z
updated_at: 2026-07-28T13:19:43Z
---

Format only crates/core/src/blueprint_universal/loader.rs and crates/core/tests/loader_unified.rs with rustfmt edition 2021. Do not alter sample_configurations/blueprint_mp_hu_500f_100t_100r.yaml, logic, or run Cargo builds/tests. Verify rustfmt --check and commit the formatting changes with the bean file.

## Summary of Changes\n\nApplied rustfmt with edition 2021 to the two requested loader files and verified both with rustfmt --check. No Cargo builds or tests were run. The pre-existing sample configuration modification was left untouched and excluded from the commit.
