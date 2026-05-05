---
# poker_solver_rust-dn7d
title: Train and export first turn-boundary ONNX model
status: in-progress
type: task
priority: high
created_at: 2026-05-05T02:57:15Z
updated_at: 2026-05-05T04:27:06Z
parent: poker_solver_rust-bvcw
---

Train the initial turn-boundary CFVNet, export ONNX, and record dataset version, config, validation metrics, and model artifact checksum.

Partial progress: added automatic model_artifact.yaml generation after Python BoundaryNet training/export. The manifest records checkpoint and ONNX checksums, external ONNX data checksums, config checksum, dataset manifest summary, validation split summary, final training-log row, and git commit. A real first turn-boundary training run is still blocked locally because local_data/cfvnet/turn_boundary is absent.
