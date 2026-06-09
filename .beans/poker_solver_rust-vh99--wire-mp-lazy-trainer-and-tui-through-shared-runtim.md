---
# poker_solver_rust-vh99
title: Wire MP lazy trainer and TUI through shared runtime
status: completed
type: task
priority: high
created_at: 2026-06-09T17:47:52Z
updated_at: 2026-06-09T18:11:04Z
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

## Implementation Notes

Implemented lazy MP CLI/TUI routing through the shared training runtime.

Changes:

• Added optional trainer-owned hooks to `LazySparseMpTrainingRuntimeAdapter` for snapshot, telemetry refresh, and reload requests while keeping default snapshot/reload behavior explicitly unsupported without hooks.
• Routed lazy sparse no-TUI training through `LazySparseMpTrainingRuntimeAdapter` plus `run_until_stopped`, preserving sparse heartbeat polling and final meta-iteration reporting.
• Routed lazy sparse TUI training through the runtime adapter and bridged TUI controls into `RuntimeControls`: quit requests stop the runtime, pause now pauses between lazy batches, and runtime-level snapshot hooks remain available.
• Preserved manual TUI snapshot reliability by saving queued `s` snapshots synchronously in the trainer bridge before checking finished/quit state. This fixes the review-found race where a runtime stop could otherwise drop a queued snapshot request.
• Preserved lazy sparse snapshot format: `snapshot_NNNN/sparse_entries.bin` plus `metadata.json` with the existing fields.
• Kept lazy sparse resume explicitly unsupported before spawning runtime threads, because current sparse snapshots do not persist blocked-edge purge state or full runtime/cadence metadata.
• Updated `docs/training.md` for lazy sparse time-limit support, real TUI pause behavior, and unsupported lazy sparse resume.

Review:

• Review found one P1 snapshot-stop race; corrected by moving manual TUI snapshot execution back to the trainer bridge while retaining adapter hooks for runtime-level requests.
• No evidence of runtime counter double-counting after review.

Verification:

• `cargo fmt --all --check`
• `cargo test -p poker-solver-core blueprint_mp::training_runtime_adapter --quiet`
• `cargo test -p poker-solver-trainer lazy_mp --quiet`
• `git diff --check`
• `cargo test --quiet` passed in 44.780s on warmed rerun, satisfying the under-one-minute project gate.
