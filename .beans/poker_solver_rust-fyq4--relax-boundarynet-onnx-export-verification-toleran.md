---
# poker_solver_rust-fyq4
title: Relax BoundaryNet ONNX export verification tolerance
status: in-progress
type: bug
priority: high
created_at: 2026-05-08T18:09:33Z
updated_at: 2026-05-08T18:09:33Z
---

Training completed but best-checkpoint ONNX export failed verification on tiny numerical drift: max abs diff 0.000155 against 0.0001 tolerance. Make export verification robust while preserving useful failure diagnostics, then verify the exported best model loads in Rust.\n\n- [ ] Reproduce or inspect tolerance failure\n- [ ] Adjust export verification\n- [ ] Run Python export tests\n- [ ] Export and verify current best ONNX\n- [ ] Record results
