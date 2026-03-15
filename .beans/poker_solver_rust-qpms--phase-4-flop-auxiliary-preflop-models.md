---
# poker_solver_rust-qpms
title: 'Phase 4: Flop + Auxiliary Preflop Models'
status: completed
type: feature
priority: normal
tags:
    - gpu
    - cfvnet
created_at: 2026-03-15T04:14:40Z
updated_at: 2026-03-15T21:50:16Z
parent: poker_solver_rust-twez
---

Complete the neural network stack with flop and auxiliary preflop models.

Tasks:
- [ ] Flop datagen: solve random flop subgames with turn model at leaves
- [ ] Flop CFVNet training on GPU
- [ ] Auxiliary preflop network: trained on preflop with flop model at leaves
- [ ] Full model stack: preflop-aux → flop → turn → river
- [ ] CLI: gpu-datagen flop/preflop, gpu-train flop/preflop
- [ ] Validate each model layer against exact solving on small trees
