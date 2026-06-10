---
# poker_solver_rust-v1it
title: 'Phase 3: MP eager universal exporter'
status: todo
type: task
priority: high
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-10T17:49:11Z
parent: poker_solver_rust-a29s
---

Export MP eager dense snapshots into the universal format with explicit seat/action metadata. Acceptance: row descriptors include acting seat, street, bucket, arena node, ordered actions, and fingerprints; probabilities match MpStorage::average_strategy for known nodes; existing MP snapshot artifacts are preserved during migration.

## Note from Phase 2 (2026-06-10)

When implementing the MP eager exporter, consider introducing the spec's StrategyRowSource trait (now that there are 2+ export sources), harden action_abstraction_fingerprint away from Debug-format hashing, and add config-vs-storage bucket count cross-checks. See Phase 2 bean (le5g) deferred list.
