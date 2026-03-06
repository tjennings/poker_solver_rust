---
# poker_solver_rust-60h7
title: 'Blueprint V2: Pluribus-style full-game solver'
status: in-progress
type: epic
priority: normal
created_at: 2026-03-06T07:31:28Z
updated_at: 2026-03-06T07:46:54Z
---

Implement a new full-game MCCFR solver pipeline with potential-aware clustering, configurable action abstraction, and explorer-compatible snapshots. Design: docs/plans/2026-03-06-blueprint-v2-design.md, Plan: docs/plans/2026-03-06-blueprint-v2-plan.md

## Tasks
- [x] Task 1: Config types
- [x] Task 2: Bucket file I/O
- [x] Task 3: EMD & k-means
- [ ] Task 4: River clustering
- [ ] Task 5: Turn clustering
- [ ] Task 6: Flop clustering
- [ ] Task 7: Preflop clustering
- [ ] Task 8: Cluster CLI + pipeline
- [ ] Task 9: Cluster diagnostics
- [x] Task 10: Game tree builder
- [ ] Task 11: Strategy/regret storage
- [ ] Task 12: MCCFR traversal
- [ ] Task 13: Training loop
- [ ] Task 14: Bundle format
- [ ] Task 15: Explorer integration
- [ ] Task 16: train-blueprint CLI
- [ ] Task 17: Sample configs
- [ ] Task 18: E2E test
