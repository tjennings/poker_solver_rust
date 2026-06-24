---
# poker_solver_rust-9h2f
title: Implement lazy_sparse MP resume support
status: todo
type: feature
priority: high
created_at: 2026-06-24T14:45:56Z
updated_at: 2026-06-24T14:46:04Z
parent: poker_solver_rust-osss
blocking:
    - poker_solver_rust-hoq8
---

Implement true resume for the Blueprint MP lazy_sparse backend. Today train-blueprint-mp rejects snapshots.resume=true for lazy_sparse because sparse snapshots do not persist blocked-edge purge state or full runtime cadence metadata. Scope: design/version the sparse snapshot schema; persist and reload SparseMpStorage rows including action identity, lazy action-limit / blocked-edge / negative-action purge state, completed meta-iterations, DCFR discount/prune/purge cadence, snapshot metadata, and enough trainer runtime state to continue without corrupting strategy sums or reactivating purged branches incorrectly. Wire both no-TUI and TUI lazy_sparse paths to resume from the latest compatible snapshot, preserve universal export compatibility, and fail loudly on incompatible/incomplete snapshots. Acceptance: lazy_sparse config with snapshots.resume=true starts from a previous snapshot; resumed run advances from the saved meta-iteration count; sparse row/action identity counts match before/after reload; negative-action purge/blocked-edge state is preserved; snapshot cadence remains stable after resume; focused tests cover no-TUI and TUI resume setup; docs/training.md and docs/architecture.md are updated.
