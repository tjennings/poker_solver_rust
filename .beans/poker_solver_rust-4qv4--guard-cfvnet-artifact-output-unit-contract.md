---
# poker_solver_rust-4qv4
title: Guard CFVNet artifact output-unit contract
status: completed
type: bug
priority: high
created_at: 2026-05-14T04:57:36Z
updated_at: 2026-05-14T04:57:36Z
parent: poker_solver_rust-lnpl
---

Add artifact/runtime checks so Python-trained legacy-normalized CFVNet exports cannot be evaluated as solver-native direct models by compare-solve.

## Summary of Changes

Python model artifact manifests now record `model.output_unit` as `bcfv_scaled_by_pot_over_total_stake` and `model.recommended_model_kind` as `direct_normalized_legacy`, matching the current Python data encoder.

`compare-solve` now looks for `model_artifact.yaml` beside CFVNet ONNX files and validates the requested `--*-model-kind` against the artifact output unit when the manifest is present. A Python legacy-scaled checkpoint evaluated with `direct` now fails early with an error telling the caller to use `direct_normalized_legacy`.

Also updated the Python encoder docstring and training docs so the legacy-scaled contract is explicit.

Verification passed:
- `cargo fmt`
- `cargo test -p poker-solver-trainer cfvnet_artifact_guard`
- `crates/cfvnet/python/.venv/bin/python -m pytest crates/cfvnet/python/tests/test_artifact.py crates/cfvnet/python/tests/test_encoding.py`
- `git diff --check`
