---
# poker_solver_rust-hrq7
title: Profile and reduce blueprint load latency
status: completed
type: bug
priority: high
created_at: 2026-07-28T01:45:42Z
updated_at: 2026-07-28T02:34:53Z
parent: poker_solver_rust-osss
---

Blueprint loading in the Tauri Explorer can take several minutes while the user sees little CPU or memory activity. Trace the complete load workflow, including universal bundle detection, manifest/payload/index construction, legacy HU reconstruction, MP lazy session setup, initial game state, and initial strategy matrix/range computation. Identify the dominant wait and whether work is being serialized, repeated, or performed before progress is visible. Implement the smallest safe fix after profiling, retaining arena/lazy storage semantics.

Acceptance:
- A representative HU and MP lazy blueprint load path has phase-level timing or equivalent evidence identifying the expensive/waiting phase.
- The load path does not eagerly perform work that can safely remain lazy for the first view.
- Tauri/devserver behavior remains correct for legacy HU, universal HU, and universal MP lazy bundles.
- Add regression coverage for the fixed behavior and update explorer documentation if the load lifecycle or progress reporting changes.
- Focused tests pass; the pre-existing dirty sample configuration remains untouched.


## Work Plan

- [ ] Research current loading boundaries and test seams
- [ ] Add phase-level timing logs
- [ ] Make lazy MP session bucket loading demand-driven
- [ ] Add focused regression coverage
- [ ] Update explorer docs if lifecycle behavior changed
- [ ] Run formatting and focused tests
- [ ] Review, commit, and report results


Scope note: this turn is limited to the MP lazy bucket initialization fix, focused regression tests, formatting, narrow verification, and a commit; timing instrumentation and docs are deferred.


## Verification

- Universal bundle load tracing now reports command, reader/index, HU reconstruction, MP source setup, and initial matrix phase timings.
- Universal MP lazy session construction no longer eagerly loads the file-backed all-street bucket corpus; bucket data is loaded and cached on first flop access.
- Focused verification passed: `cargo test -p poker-solver-tauri --lib` (376 passed, 6 ignored) and `cargo test -p poker-solver-tauri --test universal_explorer_integration` (20 passed).
- Changed Tauri files pass targeted rustfmt and `git diff --check`.
- Repository-wide formatter check still reports unrelated pre-existing formatting drift outside this change; the pre-existing sample configuration edit remains untouched.
