---
# poker_solver_rust-70km
title: Add turn-boundary training config and dataset loader
status: completed
type: task
priority: high
created_at: 2026-05-05T02:57:12Z
updated_at: 2026-05-05T04:24:53Z
parent: poker_solver_rust-bvcw
---

Implement a training entry point and loader for turn-boundary records, including schema/version checks and normalization compatible with runtime inference.

Completed the turn-boundary training entry slice. Python TrainConfig now carries datagen.street and game.board_size, lazy/GPU loaders enforce manifest and raw-file board-size compatibility, and sample_configurations/turn_boundary_cfvnet.yaml documents the training command for turn-boundary datasets.
