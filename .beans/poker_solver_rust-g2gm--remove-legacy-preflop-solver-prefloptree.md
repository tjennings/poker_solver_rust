---
# poker_solver_rust-g2gm
title: Remove legacy preflop solver & PreflopTree
status: completed
type: task
priority: normal
created_at: 2026-03-11T18:10:08Z
updated_at: 2026-03-11T19:54:32Z
---

PreflopTree and PreflopSolver are dead code now that V2 blueprints handle preflop via V2GameTree. Remove: crates/core/src/preflop/ (tree.rs, solver.rs, config.rs, bundle.rs, exploitability.rs, equity.rs), PreflopSolve variant in exploration.rs, lhe_viz.rs references, trainer CLI preflop-only commands. The explorer's PreflopSolve path should be replaced with V2 bundle loading.
