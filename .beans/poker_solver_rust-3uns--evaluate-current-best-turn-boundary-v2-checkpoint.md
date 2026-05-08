---
# poker_solver_rust-3uns
title: Evaluate current best turn-boundary v2 checkpoint
status: completed
type: task
priority: high
created_at: 2026-05-08T02:13:45Z
updated_at: 2026-05-08T02:15:40Z
---

Evaluate local_data/models/turn_boundary_cfvnet_v2/best.pt on a representative v2 shard and record the metrics.\n\n- [x] Locate current best checkpoint\n- [x] Export/verify ONNX from best.pt\n- [x] Evaluate on representative v2 shard\n- [x] Record metrics

## Summary of Changes\n\nEvaluated local_data/models/turn_boundary_cfvnet_v2/best.pt, updated best.onnx, and verified Python and Rust eval-boundary agree on local_data/cfvnet/turn_boundary/v2/a_BVZnf_00001.bin. Overall MAE mean=0.003262, p95=0.0060, p99=0.0108, max=0.0255 across 10,000 records.
