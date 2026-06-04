---
# poker_solver_rust-q95i
title: Review final Phase 2 core validator guards
status: in-progress
type: task
priority: high
created_at: 2026-06-04T03:43:30Z
updated_at: 2026-06-04T03:43:30Z
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
