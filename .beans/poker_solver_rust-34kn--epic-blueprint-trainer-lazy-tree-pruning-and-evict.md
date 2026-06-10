---
# poker_solver_rust-34kn
title: 'Epic: blueprint trainer lazy arena, validation, pruning, and eviction rewrite'
status: todo
type: epic
priority: high
created_at: 2026-06-03T18:08:59Z
updated_at: 2026-06-04T04:11:20Z
---

Plan and deliver a staged rewrite of the blueprint trainer storage/traversal model.

Goal: move the trainer away from hot-path map-based node lookup toward a lazily realized in-memory tree model, then validate equivalence before layering in pruning and disk-backed memory pressure handling.

Phases:
- [ ] Phase 1: Implement the lazy in-memory tree model only; no pruning or disk eviction.
- [x] Phase 2: Validate against a small known-good heads-up game variant supplied by the user.
- [ ] Phase 3: Add in-memory strategy pruning for persistently negative lines.
- [ ] Phase 4: Add disk eviction for selected subtrees when resident node pressure exceeds configured limits.

Non-negotiables:
- Preserve CFR correctness before optimizing memory policy.
- Keep map/interner use out of the traversal hot path unless strictly required for canonicalization.
- Add instrumentation before using memory pressure as a policy input.
- Keep docs/architecture.md and docs/training.md current for trainer/storage/config changes.
- Use agent workflow for implementation and review per AGENTS.md.
