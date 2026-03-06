---
# poker_solver_rust-8qzw
title: 'Blueprint V2: Pluribus-style full-game solver'
status: in-progress
type: epic
priority: normal
created_at: 2026-03-06T08:21:33Z
updated_at: 2026-03-06T08:31:48Z
---

Full-game MCCFR solver training all 4 streets with potential-aware clustering.

Design: docs/plans/2026-03-06-blueprint-v2-design.md
Plan: docs/plans/2026-03-06-blueprint-v2-plan.md

## Tasks (18 total)

### Phase 1: Config & I/O
- [x] Task 1: Config types
- [x] Task 2: Bucket file format
- [x] Task 3: k-means clustering (EMD + 1-D)

### Phase 2: Clustering Pipeline
- [x] Task 4: River clustering
- [ ] Task 5: Turn clustering
- [ ] Task 6: Flop clustering
- [ ] Task 7: Preflop clustering
- [ ] Task 8: Cluster CLI
- [ ] Task 9: Cluster diagnostics

### Phase 3: Game Tree
- [x] Task 10: Game tree builder

### Phase 4: MCCFR Engine
- [x] Task 11: Strategy/regret storage
- [x] Task 12: MCCFR traversal
- [ ] Task 13: Training loop

### Phase 5: Output & Integration
- [ ] Task 14: Bundle format
- [ ] Task 15: Explorer integration
- [ ] Task 16: train-blueprint CLI
- [ ] Task 17: Sample configs

### Phase 6: E2E
- [ ] Task 18: E2E test
