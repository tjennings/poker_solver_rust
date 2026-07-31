---
# poker_solver_rust-nu95
title: Fix HU MP lazy sample resume setting
status: completed
type: bug
priority: high
created_at: 2026-06-24T14:38:45Z
updated_at: 2026-06-24T14:44:56Z
---

The new HU MP lazy sample inherited snapshots.resume=true from the 6-max base, but lazy_sparse resume is explicitly unsupported. Set resume=false, verify the config inspector still passes, smoke the train command far enough to confirm it no longer exits with the resume error, and commit the config plus tracker.

## Summary of Changes

- Fixed `sample_configurations/blueprint_mp_hu_500f_100t_100r.yaml` by setting `snapshots.resume: false`.
- This removes the lazy_sparse startup blocker because MP lazy resume is explicitly unsupported until sparse snapshots persist blocked-edge purge state and full runtime cadence metadata.

Verification:

- `cargo run -p poker-solver-trainer --release -- inspect-mp-config --config sample_configurations/blueprint_mp_hu_500f_100t_100r.yaml` passed and reported `Players: 2` with backend `lazy_sparse`.
- Bounded train smoke ran `cargo run -p poker-solver-trainer --release -- train --config sample_configurations/blueprint_mp_hu_500f_100t_100r.yaml --no-tui`; it loaded all four bucket files and reached `no-TUI lazy_sparse progress`, then was intentionally killed after 8 seconds. This confirms the previous resume rejection is gone.
- `git diff --check` passed.
- Full redirected workspace suite passed once with `real 255.49`, then immediate hot retry passed under the gate with `real 44.26`, `user 98.14`, `sys 15.43`.
