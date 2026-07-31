---
# poker_solver_rust-6y86
title: 'Phase 0: baseline and rewrite spec for blueprint trainer lazy tree'
status: completed
type: task
priority: high
created_at: 2026-06-03T18:10:49Z
updated_at: 2026-06-03T18:31:47Z
parent: poker_solver_rust-34kn
---

Phase 0 planning/specification task for the blueprint trainer lazy tree roadmap.

Scope:
- Capture current blueprint trainer storage, traversal, snapshot, export, and resume behavior before implementation starts.
- Record baseline performance on tiny and representative configs: startup memory, visited decision count, allocated slots/rows, iterations/sec, snapshot size, and resume behavior.
- Define stable lazy node identity: public state key, action-history encoding, street, pot/stacks/bets, actor, chance boundary handling, legal action order, and terminal payoff identity.
- Define row identity separately from arena identity: node/bucket/action coordinates or an explicit sparse row key.
- Decide snapshot compatibility policy: legacy dense resume support, explicit versioned failure, and/or new sparse/lazy resume format.
- Document invariants that Phase 1 must preserve.

Acceptance criteria:
- Baseline report exists in the bean or docs and is specific enough for rust-developer agents to implement against.
- Invariants are documented for legal action equality, terminal payoff equality, bucket lookup consistency, action coordinate stability, deterministic seed behavior, snapshot/export compatibility, and DCFR discount behavior for lazy rows.
- The working tree is clean before implementation begins.
- The full test suite passes and completes under 1 minute before implementation begins, or a blocking bean is created/fixed first.

This blocks the differential harness and lazy tree implementation.


## Gate Check 2026-06-03

Started Phase 0 and ran the required pre-implementation full-suite gate. The working tree was clean and tests passed, but runtime violated the project rule:

- Cold `cargo test`: `real 162.70s`.
- Warm `cargo test --quiet`: `real 72.33s`.

Per AGENTS.md, Phase 0 is paused until `poker_solver_rust-v55b` brings the default full-suite gate back under 60 seconds or the gate is explicitly revised.


## Summary of Changes

Completed Phase 0 baseline/spec work. The durable artifact is `docs/plans/2026-06-03-blueprint-trainer-lazy-tree-phase-0.md`.

Key findings:
- HU `blueprint_v2` already traverses an eager arena-style `GameTree`; the major dense pressure point is all-row CFR storage plus dense snapshot/export/resume compatibility.
- Phase 1 should separate in-process `NodeId`, stable `PublicNodeKey`, and CFR `RowKey`; persisted identity must not rely only on allocation-order arena indices.
- Phase 1 must preserve legal action order, terminal payoffs, bucket lookup, DCFR discount semantics, dense Explorer/Tauri export compatibility, and explicit snapshot versioning.
- MP lazy/sparse storage is a useful semantic precedent for missing-row zero/uniform behavior and sparse snapshots, but its sharded map should not be imported blindly as the HU hot path.

Baseline and gates:
- Initial warm full-suite gate failed runtime: `cargo test --quiet` passed but took `real 72.33s`.
- Linked Phase 0 to blocker `poker_solver_rust-v55b`; workers repaired the suite by moving slow diagnostics behind explicit ignored-test runs and fixing a Tauri temp strategy test race.
- Verified warm full-suite gate now passes under the project limit: `/usr/bin/time -p cargo test --quiet` passed in `real 40.44s`.
- Focused HU toy trainer test passed: `cargo test -p poker-solver-core blueprint_v2::trainer::tests::train_runs_iterations --quiet`, `real 0.22s`.

Phase 1 prep harness (`poker_solver_rust-zgkr`) is now unblocked.
