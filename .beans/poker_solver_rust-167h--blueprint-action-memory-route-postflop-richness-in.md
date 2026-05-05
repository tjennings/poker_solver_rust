---
# poker_solver_rust-167h
title: 'Blueprint action memory: route postflop richness into subgame solving'
status: todo
type: task
priority: high
created_at: 2026-05-05T15:01:42Z
updated_at: 2026-05-05T15:01:59Z
parent: poker_solver_rust-ohyt
blocked_by:
    - poker_solver_rust-o4hm
---

Order 5. Keep the global blueprint coarse where that is the best RAM tradeoff, and add richer action handling through real-time or cached postflop solving.

Implementation notes:
- Define when blueprint_mp should rely on table lookup vs invoke richer subgame solving.
- Support richer postflop action sets in subgame solves without requiring them in the full global blueprint.
- Consider cached/batch solved study spots for product use, not only live play.
- Integrate with existing compare-solve/subgame/cfvnet/gadget machinery where appropriate.

Acceptance criteria:
- A selected postflop spot can expose richer actions than the global blueprint action abstraction.
- The interface clearly reports when strategy comes from blueprint vs subgame solve/cache.
- Validation compares blueprint-only vs richer subgame strategy on representative spots.
- docs/architecture.md, docs/training.md, and docs/explorer.md are updated where behavior changes.
