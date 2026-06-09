---
# poker_solver_rust-g54j
title: 'Phase 5: unified bundle detection and loader'
status: todo
type: task
priority: high
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-09T18:41:22Z
parent: poker_solver_rust-a29s
---

Add one loader detection path for legacy HU bundles and universal HU/MP bundles. Acceptance: loader distinguishes old config.yaml+strategy.bin, universal HU, universal MP eager, universal MP lazy; errors explain missing/incompatible files; old HU discovery remains compatible.
