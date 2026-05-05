---
# poker_solver_rust-ohyt
title: 'Epic: fit richer blueprint_mp actions in 128GB RAM'
status: in-progress
type: epic
priority: high
created_at: 2026-05-05T15:01:07Z
updated_at: 2026-05-05T15:06:52Z
---

Increase the number of useful blueprint_mp actions that can be trained and served within a 128GB RAM target. Scope covers the first five implementation ideas from the memory research pass.

Suggested delivery order:
1. Add measurement and sizing guardrails so every later change has hard numbers.
2. Implement situation-dependent action abstraction to reduce tree growth before adding richer sizes.
3. Implement true touched-row storage and lazy/logical DCFR discounts to keep unvisited rows free.
4. Implement total regret-based dormant-action pruning to reduce traversal and eventually resident storage pressure.
5. Compress/externalize strategy sums so average-policy bookkeeping stops dominating slot bytes.
6. Keep blueprint coarse where appropriate and route postflop richness into real-time or cached subgame solving.

Success criteria:
- A richer-than-current 6-max action abstraction can run under 128GB resident memory.
- Trainer startup and docs report expected tree slots, virtual bytes, and estimated resident/touched bytes.
- Existing 6-max 200-bucket config remains supported.
- docs/architecture.md and docs/training.md stay current.
