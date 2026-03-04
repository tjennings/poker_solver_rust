---
# poker_solver_rust-wch9
title: Add dump-tree diagnostic subcommand
status: completed
type: feature
priority: normal
created_at: 2026-03-04T03:54:58Z
updated_at: 2026-03-04T04:00:17Z
---

Add a dump-tree CLI subcommand to the trainer that renders the PostflopTree as indented text (box-drawing chars) or Graphviz DOT for visual debugging.

## Tasks
- [x] Create crates/trainer/src/tree_dump.rs with text + DOT rendering
- [x] Wire into main.rs (mod, DumpTree variant, match arm, pub(crate) PostflopSolveConfig)
- [x] Verify: cargo test, cargo clippy, manual smoke test

## Summary of Changes
Added `dump-tree` subcommand to the trainer CLI. Requires `--config` (same YAML as solve-postflop). Renders the PostflopTree as indented box-drawing text (default) or Graphviz DOT (`--dot`). Supports `--spr` and `--pot-type` flags.
