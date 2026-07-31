---
# poker_solver_rust-ev6p
title: 'Phase 2 preflight: full suite runtime exceeds one minute'
status: completed
type: bug
priority: high
created_at: 2026-06-04T03:00:03Z
updated_at: 2026-06-10T20:39:58Z
---

Phase 2 implementation is blocked by the required pre-development gate.

Observed on 2026-06-04 before starting Phase 2 implementation:

- `/usr/bin/time -p cargo test --quiet` passed functionally but measured `real 86.81`, over the required one-minute limit.

Scope:
- Determine whether this is a reproducible runtime regression, cold/rebuild artifact, or environmental contention.
- Restore the default full-suite gate to under one minute before Phase 2 implementation proceeds.
- If a specific slow/default test is responsible, move it out of the default suite or optimize it without weakening meaningful coverage.

Acceptance criteria:
- Full `cargo test --quiet` passes in under one minute in a warm manager run.
- Any test-tier changes are documented in the bean and committed.
- Phase 2 implementation remains paused until this gate is green.

## Resolution

The initial `real 86.81` run was not reproduced. A warm confirmation run immediately afterward passed under the project gate:

- `/usr/bin/time -p cargo test --quiet` passed in `real 44.47`.

No Rust/test-tier changes were required. Treat the first result as a transient cold/environmental timing outlier. Phase 2 implementation may proceed from the warm green gate.
