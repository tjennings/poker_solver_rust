---
# poker_solver_rust-7g9f
title: Plan universal N-player dense blueprint format
status: in-progress
type: task
priority: high
created_at: 2026-06-09T18:35:50Z
updated_at: 2026-06-09T18:35:50Z
parent: poker_solver_rust-tzv5
---

Design a new dense blueprint strategy/snapshot format that can represent both heads-up blueprint_v2 and N-player blueprint_mp strategies. The plan must cover schema/versioning, player/seat metadata, game/action abstraction metadata, bucket metadata, row/action layout, dense eager and lazy sparse export, loader compatibility, Explorer/TUI/API integration, migration/compatibility, validation tests, and phased implementation beans. Non-goals for this planning bean: do not implement the format yet and do not change traversal/storage semantics.
