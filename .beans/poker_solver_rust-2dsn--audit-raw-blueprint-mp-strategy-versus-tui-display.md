---
# poker_solver_rust-2dsn
title: Audit raw Blueprint MP strategy versus TUI display
status: in-progress
type: bug
priority: high
created_at: 2026-05-20T13:27:27Z
updated_at: 2026-05-20T13:27:59Z
parent: poker_solver_rust-kiqt
---

Determine whether suspect preflop folds are present in raw sparse storage or introduced by TUI scenario resolution/rendering.

## Subtasks

- [ ] Pick suspect hands: A2s-A5s, ATs-AQs, K9s, 22, 72o
- [ ] Pick suspect spots: UTG root, BTN unopened, BTN versus CO open, BB versus SB open
- [ ] Dump average strategy, current regret-matched strategy, regrets, and strategy sums for each hand/spot
- [ ] Compare raw action probabilities to TUI matrix colors and labels
- [ ] Classify failure as storage/training bug, path-resolution bug, or rendering/display bug
- [ ] Add a regression test or diagnostic command if raw/TUI disagree
