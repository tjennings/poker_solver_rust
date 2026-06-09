---
# poker_solver_rust-vh99
title: Wire MP lazy trainer and TUI through shared runtime
status: in-progress
type: task
priority: high
created_at: 2026-06-09T17:47:52Z
updated_at: 2026-06-09T17:53:17Z
parent: poker_solver_rust-tzv5
---

Route the existing train-blueprint-mp lazy_sparse CLI/TUI execution through the shared training runtime and LazySparseMpTrainingRuntimeAdapter while preserving current lazy sparse snapshot format, TUI controls, telemetry, and no-TUI heartbeat behavior. Scope includes designing trainer-side snapshot/resume hooks for sparse_entries.bin metadata if required by runtime integration, wiring pause/quit/snapshot/refresh requests through RuntimeControls, updating docs/training.md for any behavior changes, and adding focused tests. Non-goals: do not merge HU and MP traversal algorithms, do not change sparse storage identity, do not implement strategy pruning or disk eviction yet.

## Research And Architecture Notes

Completed research plus architecture brainstorming for routing lazy MP CLI/TUI through the shared runtime.

Current behavior to preserve:

• Lazy no-TUI currently spawns `run_lazy_training`, polls `ctx.iterations`, and prints sparse heartbeat every 60s plus one final heartbeat.
• Lazy no-TUI has no snapshot cadence today; manual sparse snapshots exist only through the TUI bridge.
• Lazy TUI currently resolves lazy scenario spots, runs `run_lazy_training`, polls metrics every 50ms, refreshes sparse telemetry/scenario grids/probes every 10s, and writes manual sparse snapshots from trainer code.
• TUI quit is real today via `ctx.quit`; TUI pause is currently only visual because lazy training has no pause bridge.
• Lazy sparse snapshots write `snapshot_NNNN/sparse_entries.bin` as bincode `Vec<SparseSnapshotEntry>` plus `metadata.json` with kind, index, iterations, elapsed, entries, slots, and approx bytes.
• Lazy sparse disk resume is not implemented today. In-memory sparse entry restore exists, but snapshots do not persist negative-action blocked-edge state, cadence state, or full runtime metadata.

Implementation decisions:

• Keep core adapter responsible only for backend execution, limits, counters, and batch stepping.
• Keep trainer responsible for snapshot paths, TUI metrics, heartbeat formatting, stderr messages, scenario grids, and strategy probes.
• Add optional hook support to `LazySparseMpTrainingRuntimeAdapter` for snapshot/refresh/reload side effects; default unsupported/no-op behavior remains for hooks that are not installed.
• Route lazy no-TUI and lazy TUI training threads through `run_until_stopped` with the MP lazy adapter.
• Preserve no-TUI heartbeat by polling cloned `ctx.iterations` and sparse storage from the trainer thread.
• Bridge TUI controls into `RuntimeControls`: quit -> request_quit, pause -> request_pause/resume, snapshot -> request_snapshot. Periodic TUI telemetry remains bridge-owned to avoid blocking training batches.
• Implement sparse snapshot save hook only; keep `snapshots.resume = true` explicitly rejected before spawning the runtime thread.
• Document user-visible behavior changes: lazy sparse now honors runtime time limits and TUI pause actually pauses between batches.

Risks to guard:

• Runtime counters must not double-count `ctx.iterations`.
• Snapshot failures should preserve current TUI behavior: mark failed/log warning without killing training unless the user explicitly asked for fatal snapshot failures.
• Runtime-thread telemetry scans must not reduce IPS.
• Resume must not be half-implemented.
• Snapshot requests made after target completion may not be serviced unless the trainer explicitly handles final snapshots.

Test plan:

• Core adapter hook tests for snapshot, refresh, pause, quit, resume rejection, and no double-counting.
• Trainer tests around sparse snapshot format/status and lazy runtime path where practical.
• Focused trainer/core test runs plus full `cargo test --quiet` under one minute.
