---
# poker_solver_rust-ieak
title: Compare DeepStack/Supremus betting abstraction to river datagen config
status: completed
type: task
priority: normal
created_at: 2026-05-04T14:57:46Z
updated_at: 2026-05-04T14:59:45Z
---

Read docs/papers DeepStack and Supremus papers to extract what bet sizes/actions were used for CFV/value-network data generation, then compare them with sample_configurations/boundary_net_river_datagen.yaml.\n\n## Tasks\n\n- [ ] Extract DeepStack CFV/value-network bet abstraction details.\n- [ ] Extract Supremus CFV/value-network bet abstraction details.\n- [ ] Inspect boundary_net_river_datagen.yaml.\n- [x] Compare gaps and implications for our CFVNet training data.


## Summary of Findings

DeepStack paper: target CFV/value-network training situations were solved with only fold, call, pot-sized bet, and all-in. The implementation tables also discuss richer sparse lookahead choices by round, but the network-target solve passage is explicitly F/C/P/A.

Supremus paper: random subgames were generated in the same manner as DeepStack, solved with 4,000 DCFR+ iterations per player, and used a wider action abstraction. Table 2 lists Supremus actions as first action F/C/0.33/0.5/0.75/1.0/1.25/2.0/A, second action F/C/0.25/0.5/1.0/A, third action F/C/0.25/A, remaining F/C/1.0/A.

boundary_net_river_datagen.yaml currently uses river exact datagen, 5M samples, 200 solver iterations, first bet sizes 25/50/100/all-in, raise sizes 25/75/all-in, and 20% fuzz. In code, all-in is skipped during percentage parsing because the game tree builder adds all-in automatically. Effective numeric support is roughly first bet 20-120% pot and raises 20-90% pot, plus all-in.

Implication: our config is broader than DeepStack target solves on small sizes but lacks DeepStack/Supremus 2x pot overbets (and DeepStack table-ground-truth experiments went as high as 3x/10x in analysis). It also uses far fewer solve iterations than the papers: 200 versus DeepStack 1,000 CFR+ for targets and Supremus 4,000 DCFR+ per player. For our observed large-bet/all-in CFVNet failures, the missing 2x pot overbet support and low target-solve iteration count are stronger suspects than the presence of 25/50/75% sizes.
