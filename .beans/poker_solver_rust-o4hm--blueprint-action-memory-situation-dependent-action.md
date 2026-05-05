---
# poker_solver_rust-o4hm
title: 'Blueprint action memory: situation-dependent action abstraction'
status: todo
type: task
priority: high
created_at: 2026-05-05T15:01:21Z
updated_at: 2026-05-05T15:01:50Z
parent: poker_solver_rust-ohyt
blocked_by:
    - poker_solver_rust-ljo7
---

Order 1. Implement richer-but-targeted action abstractions so we add useful actions without exploding the whole 6-max tree.

Implementation notes:
- Extend blueprint_mp action config beyond flat per-street lead/raise lists.
- Support conditions such as street, heads-up vs multiway, preflop position/spot, facing bet vs lead, SPR/effective stack bands, raise depth, and possibly pot-size bands.
- Keep current configs valid as the default schema.
- Produce at least one 128GB-target richer 6-max preset.

Acceptance criteria:
- Existing 6-max config produces the same tree/action behavior.
- New richer preset adds more strategically useful sizes while passing the sizing budget task.
- Tests cover action selection for HU vs multiway, low-SPR collapse, and preflop/postflop depth behavior.
- docs/training.md documents the extended action config.
