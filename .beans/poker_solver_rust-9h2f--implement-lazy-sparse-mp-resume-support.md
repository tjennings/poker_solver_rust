---
# poker_solver_rust-9h2f
title: Implement lazy_sparse MP resume support
status: todo
type: feature
priority: high
created_at: 2026-06-24T14:45:56Z
updated_at: 2026-06-30T19:58:22Z
parent: poker_solver_rust-osss
blocking:
    - poker_solver_rust-hoq8
---

Implement true resume for the Blueprint MP lazy_sparse backend. Today train-blueprint-mp rejects snapshots.resume=true for lazy_sparse because sparse snapshots do not persist blocked-edge purge state or full runtime cadence metadata. Scope: design/version the sparse snapshot schema; persist and reload SparseMpStorage rows including action identity, lazy action-limit / blocked-edge / negative-action purge state, completed meta-iterations, DCFR discount/prune/purge cadence, snapshot metadata, and enough trainer runtime state to continue without corrupting strategy sums or reactivating purged branches incorrectly. Wire both no-TUI and TUI lazy_sparse paths to resume from the latest compatible snapshot, preserve universal export compatibility, and fail loudly on incompatible/incomplete snapshots. Acceptance: lazy_sparse config with snapshots.resume=true starts from a previous snapshot; resumed run advances from the saved meta-iteration count; sparse row/action identity counts match before/after reload; negative-action purge/blocked-edge state is preserved; snapshot cadence remains stable after resume; focused tests cover no-TUI and TUI resume setup; docs/training.md and docs/architecture.md are updated.

## Current Symptom

Training can write sparse snapshots, but resumption is not reliable for lazy_sparse MP runs. The fix must cover both explicit snapshot paths and automatic latest-snapshot discovery under the configured `snapshots.output_dir`, including TUI and no-TUI startup paths. It should report the loaded snapshot iteration/config/schema in startup output so a user can tell immediately whether training resumed or started fresh.

## Additional Acceptance Criteria

- With `snapshots.resume: true`, trainer locates the latest compatible snapshot for the configured output directory without requiring manual path copying.
- Startup logs/TUI state clearly indicate `fresh` vs `resumed`, snapshot path, and resumed meta-iteration.
- A resumed lazy_sparse MP run continues snapshot cadence from the saved iteration rather than resetting interval accounting.
- Resume rejection errors distinguish missing snapshots, incompatible schema/config, and incomplete/corrupt snapshot data.
- Regression coverage includes at least one small lazy_sparse MP train-save-resume smoke test.
