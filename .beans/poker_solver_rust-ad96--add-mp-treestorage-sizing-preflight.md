---
# poker_solver_rust-ad96
title: Add MP tree/storage sizing preflight
status: in-progress
type: task
priority: high
created_at: 2026-05-07T13:18:39Z
updated_at: 2026-05-07T13:45:38Z
parent: poker_solver_rust-5kvv
---

Add a lightweight sizing/preflight mode for Blueprint MP configs that reports stack depth, action-depth rows, estimated/eager tree nodes, storage slots, and virtual bytes before committing huge mmap/storage. Use it to fail fast with a useful diagnostic when eager mode is selected for a 100bb-scale tree.

## Work Started

Starting implementation of MP config sizing/preflight so 100bb eager-mode configs report tree/storage scale and can fail fast before dense allocation.
