---
# poker_solver_rust-2juu
title: Add schema compatibility tests for turn-boundary records
status: completed
type: task
priority: normal
created_at: 2026-05-05T02:54:42Z
updated_at: 2026-05-05T03:10:15Z
parent: poker_solver_rust-ewjj
---

Add tests that encode/decode turn-boundary records and assert stable shape, units, metadata preservation, and compatibility with existing BoundaryNet training infrastructure where reused.



Completed with schema compatibility coverage in Rust and Python. Tests assert board_size=4, record_size(4), 2720 input shape, 1326 output shape, normalized target contract, YAML round trip, and rejection of river-sized manifests.



Follow-up: Python dataset directory loading now honors manifest.yaml shard order and rejects missing declared shards before training starts.
