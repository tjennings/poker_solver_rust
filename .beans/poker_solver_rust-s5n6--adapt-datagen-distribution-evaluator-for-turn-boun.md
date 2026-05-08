---
# poker_solver_rust-s5n6
title: Adapt datagen distribution evaluator for turn-boundary records
status: completed
type: task
priority: high
created_at: 2026-05-06T02:20:33Z
updated_at: 2026-05-06T02:30:13Z
---

Ensure the existing cfvnet data distribution diagnostics work for the current turn_boundary datagen output, including SPR/pot coverage and boundary-specific metadata where available.\n\n- [x] Audit existing distribution/evaluator command and current turn_boundary record format\n- [x] Identify whether current command already works on new datagen output\n- [x] Implement compatibility/reporting fixes if needed\n- [x] Verify with tests and/or a small generated dataset

## Summary of Changes\n\nMade cfvnet datagen-eval manifest-aware for turn-boundary dataset directories: it now reads manifest.yaml shard metadata instead of parsing every file, skips obvious metadata in legacy directories, and prints deterministic manifest coverage summaries before raw histograms. Verified with focused datagen_eval tests and the full cfvnet crate tests.
