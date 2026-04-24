---
# poker_solver_rust-wcc4
title: 'Pre-existing test failure: blueprint_mp::mccfr::traverse_updates_strategy_sums'
status: todo
type: bug
priority: low
created_at: 2026-04-24T04:53:25Z
updated_at: 2026-04-24T04:53:25Z
---

Detected 2026-04-24 during Option A baseline check (bean 02wj).

Failure:
    crates/core/src/blueprint_mp/mccfr.rs:539 panics with
    'at least one strategy sum should be non-zero after traversal'

Observed: cargo test -p poker-solver-core --lib → 970 passed, 1 failed.

Also failing on main (unrelated to Option A gadget work):
- gpu-range-solver --lib
- poker-solver-core --test blueprint_mp_validation
- poker-solver-trainer --bin poker-solver-trainer

Unblocks nothing for Option A since the failing modules (blueprint_mp,
gpu-range-solver, trainer binary) are structurally separate from
Option A's change footprint (crates/range-solver, crates/tauri-app,
crates/core/src/blueprint_v2/cbv*).

Flaky or deterministic? TBD — the panic may be RNG-dependent (MCCFR
uses random sampling). Re-run 5x and categorize.
