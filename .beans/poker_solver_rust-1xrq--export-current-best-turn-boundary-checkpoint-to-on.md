---
# poker_solver_rust-1xrq
title: Export current best turn-boundary checkpoint to ONNX
status: completed
type: task
priority: high
created_at: 2026-05-08T02:18:17Z
updated_at: 2026-05-08T02:19:47Z
---

Refresh local_data/models/turn_boundary_cfvnet_v2/best.onnx from best.pt and verify it is usable for copying to another machine.\n\n- [x] Locate best checkpoint\n- [x] Export ONNX\n- [x] Verify exported model file

## Summary of Changes\n\nRefreshed local_data/models/turn_boundary_cfvnet_v2/best.onnx from best.pt. The export produced an external-data pair: best.onnx (6,102 bytes, sha256 afd43193ae0048aea682ead4095f52d34facada2e15dd45773076ce00491441b) plus best.onnx.data (26,705,920 bytes, sha256 856f4abe103b3f46df4a91063fb77e1645e00dac9b833ae4cb75d6a5985c5407). Verified with Python ONNX export self-check and Rust eval-boundary on 10,000 v2 records; p99 MAE remained 0.0108.
