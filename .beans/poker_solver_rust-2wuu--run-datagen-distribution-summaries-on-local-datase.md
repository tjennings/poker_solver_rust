---
# poker_solver_rust-2wuu
title: Run datagen distribution summaries on local datasets
status: completed
type: task
priority: normal
created_at: 2026-05-06T02:38:37Z
updated_at: 2026-05-06T02:48:34Z
---

Run cfvnet datagen-eval on representative local generated datasets and summarize coverage/distribution signals for training data planning.\n\n- [x] Locate representative local cfvnet datasets\n- [x] Run datagen-eval on selected datasets\n- [x] Summarize distribution findings

## Summary of Changes\n\nRan cfvnet datagen-eval on the full symlinked turn_boundary dataset, representative turn_boundary shards/partials, a small GPU turn sample, and one legacy bucketed_river_10m shard. Found the full turn_boundary dataset is structurally valid but has dense unnormalized all-combo input ranges, while the GPU turn sample has board-filtered normalized ranges. Legacy bucketed river files are not compatible with the current TrainingRecord reader and produce invalid values/panic.
