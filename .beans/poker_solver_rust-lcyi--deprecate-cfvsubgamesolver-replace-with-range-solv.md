---
# poker_solver_rust-lcyi
title: Deprecate CfvSubgameSolver — replace with range-solver + depth boundary
status: in-progress
type: task
priority: normal
created_at: 2026-03-26T02:55:19Z
updated_at: 2026-03-26T02:56:44Z
---

CfvSubgameSolver uses bucket abstraction for subgame solving, which limits hand resolution. The range-solver PostFlopGame with depth_limit supports exact 1326-combo solving with boundary CFVs (Libratus/Pluribus approach). 

## TODO
- [x] Add #[deprecated] attribute to CfvSubgameSolver with clear warning message
- [ ] Remove CfvSubgameSolver usage from game_session solve path
- [ ] Wire PostFlopGame with depth_limit into GameSession::solve instead
- [ ] Eventually remove CfvSubgameSolver entirely once all callers migrated
