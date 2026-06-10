---
# poker_solver_rust-q95i
title: Review final Phase 2 core validator guards
status: completed
type: task
priority: high
created_at: 2026-06-04T03:43:30Z
updated_at: 2026-06-04T03:48:06Z
parent: poker_solver_rust-l6r9
---

Independent review of second corrective commit `041f8982 Require trusted baseline game preconditions`, following remaining blockers in review bean `poker_solver_rust-q9n8`.

Review focus:
- Validator now requires trusted game preconditions carrying original stack/blinds/limp policy and refuses scoring when missing or mismatched.
- Wrong-blind same-chip-action trees are rejected.
- Exact pinned baseline spot set is enforced; extra spot keys are rejected.
- Previous guards remain intact: non-169 preflop providers rejected, invalid hand rows reported, metadata checked, all-in-call/RAI mapping preserved, zero-mass rows skipped/reported.
- API remains suitable for trainer/TUI integration, where the integration layer can fill trusted preconditions from the original config.

Reviewer should report blockers with file/line references and recommend whether trainer/TUI integration can proceed.

## Summary of Review

Final core-validator review completed for `041f8982 Require trusted baseline game preconditions`.

Findings: no blocking findings. The reviewer confirmed:

- Validation precondition failures are collected and returned before any spot scoring.
- Trusted `BaselineGamePreconditions` are required and checked for starting stack, small blind, big blind, and limp policy.
- Wrong-blind same-chip-action tree counterexample is rejected by trusted SB/BB validation.
- Extra baseline spot keys are rejected by exact pinned schema enforcement.
- Previous guards remain intact: non-169 provider rejection, malformed hand reporting, all-in-call to `C`, aggressive all-in to `RAI`, and zero-mass row skip/reporting.
- API is suitable for trainer/TUI integration via public `BaselineGamePreconditions` and provider `preflop_bucket_count`.

Review tests passed:

- `cargo test -p poker-solver-core blueprint_v2::baseline_validation --quiet`
- `cargo test -p poker-solver-core blueprint_v2 --quiet`
- `/usr/bin/time -p cargo test --quiet` in `real 43.77`.

Residual integration risk: the trainer/TUI layer must fill trusted preconditions from the actual original `GameConfig`, not fabricated pinned values. That must be covered in the integration slice.

Recommendation: proceed to trainer/TUI integration.
