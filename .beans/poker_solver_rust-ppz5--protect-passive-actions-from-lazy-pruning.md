---
# poker_solver_rust-ppz5
title: Protect passive actions from lazy pruning
status: in-progress
type: bug
priority: high
created_at: 2026-05-14T18:56:08Z
updated_at: 2026-05-14T18:56:08Z
---

Lazy sparse traversal pruning currently skips any action whose regret is below prune_threshold, unlike eager MCCFR which protects terminal fold children. In MP, passive actions route to high-leverage terminal/showdown/later-player states, and pruning them may make open strategies degenerate after pruning starts. Restrict lazy traversal pruning to aggressive actions and add focused tests.
