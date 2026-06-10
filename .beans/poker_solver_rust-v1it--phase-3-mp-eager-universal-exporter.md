---
# poker_solver_rust-v1it
title: 'Phase 3: MP eager universal exporter'
status: todo
type: task
priority: high
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-09T18:41:22Z
parent: poker_solver_rust-a29s
---

Export MP eager dense snapshots into the universal format with explicit seat/action metadata. Acceptance: row descriptors include acting seat, street, bucket, arena node, ordered actions, and fingerprints; probabilities match MpStorage::average_strategy for known nodes; existing MP snapshot artifacts are preserved during migration.
