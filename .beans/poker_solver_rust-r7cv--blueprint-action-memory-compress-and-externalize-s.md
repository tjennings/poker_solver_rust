---
# poker_solver_rust-r7cv
title: 'Blueprint action memory: compress and externalize strategy sums'
status: todo
type: task
priority: high
created_at: 2026-05-05T15:01:37Z
updated_at: 2026-05-05T15:01:56Z
parent: poker_solver_rust-ohyt
blocked_by:
    - poker_solver_rust-zu7v
---

Order 4. Reduce the resident footprint of average-policy bookkeeping. Related older bean: poker_solver_rust-dm26, but current code already uses i16 regrets and i32 strategy sums, so this task is specifically about the next strategy-sum step.

Implementation notes:
- Evaluate alternatives: u16 with per-row scale, sparse sampled-action-only average sums, periodic disk/checkpoint accumulation, or storing only final/export regions hot.
- Preserve average_strategy semantics for TUI, snapshots, and bundle export.
- Make precision/overflow behavior explicit and measurable.
- Integrate with touched-row storage if implemented first.

Acceptance criteria:
- Strategy sum storage no longer consumes 4 hot bytes per slot in the default high-memory mode, or there is a documented reason to keep i32.
- Average strategy comparisons against current i32 baseline stay within an agreed tolerance on small deterministic tests.
- Snapshot/load/export compatibility is covered.
- docs/architecture.md documents precision and storage tradeoffs.
