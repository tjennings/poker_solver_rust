---
# poker_solver_rust-f3ic
title: Fix Python cfvnet bugs found in pre-turn-datagen review
status: completed
type: bug
priority: high
created_at: 2026-04-19T20:03:47Z
updated_at: 2026-04-19T20:21:15Z
---

Code review (2026-04-19) of `crates/cfvnet/python/` found four bugs that must be fixed before turn-net training can begin. Two are critical (silently corrupt or skip turn data), one breaks ONNX dynamic batch export, one breaks checkpoint resume past epoch 9.

## Findings

### Critical — silently break turn datagen

**1. `cfvnet/data.py:302-304` — `_decode_single_record` hardcodes 5-card board.**
Reads `raw[base+1:base+6]` regardless of the `board_size` prefix byte. For 4-card turn records every subsequent field (pot, stack, player, ranges, mask) is parsed at the wrong offset. Affects all production decode paths (`_decode_file_records`, `_encode_raw_to_tensors`, `_fill_chunk`).

Fix:
```python
board_size = int(raw[base])
board = raw[base + 1: base + 1 + board_size].astype(np.int32)
pos = base + 1 + board_size
```

**2. `cfvnet/data.py:191-196` and `cfvnet/gpu_buffer.py:67,280` — record size hardcoded as `RECORD_SIZE_RIVER`.**
Turn files (4-card records, 17256 B) fail `file_size % 17257 != 0` check → `_count_records_in_file` returns -1 → file silently skipped. Trainer indexes zero records when turn datagen runs. Also: `fh.read(RECORD_SIZE_RIVER)` and `i * RECORD_SIZE_RIVER` byte offsets are wrong for non-river records.

Fix: introduce `record_size(board_size: int) -> int` mirroring the Rust function, peek at the first byte in `_count_records_in_file` to derive board_size, replace all hot-path uses of `RECORD_SIZE_RIVER`.

### Important

**3. `cfvnet/export.py:26-35` — `dynamic_shapes` silently ignored without `dynamo=True`.**
ONNX graph likely bakes batch dim to 1; `tract` (Rust runtime, stricter than onnxruntime) may reject non-1 batch inference. The `test_onnx_supports_dynamic_batch` test passes only because BatchNorm1d in eval mode accidentally tolerates other batch sizes.

Fix: switch to legacy `dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}` parameter, drop the `torch.export.Dim` call.

**4. `cfvnet/train.py:485` — checkpoint resume sorts alphabetically.**
`epoch9 > epoch10 > epoch100` lexicographically. Past epoch 9, resume loads the wrong checkpoint, corrupting LR schedule and optimizer state. Same bug in `scripts/eval_boundary.py:72` and `scripts/train_boundary.py:84`.

Fix: `sorted(..., key=lambda p: int(p.stem.replace("checkpoint_epoch", "")))`

## Todo

- [x] Fix 1: variable board_size in `_decode_single_record`
- [x] Fix 2: variable record_size everywhere (`_count_records_in_file`, `gpu_buffer.py` reads/seeks)
- [x] Add a `record_size(board_size)` helper that mirrors the Rust side
- [x] Fix 3: ONNX export dynamic batch — use `dynamic_axes`, drop `torch.export.Dim`
- [x] Fix 4: numeric checkpoint sort in `train.py`, `eval_boundary.py`, `train_boundary.py`
- [x] Add a Python test that decodes a turn-format record (board_size=4) end-to-end and asserts every field matches what the Rust producer wrote
- [x] Add a Python test for `record_size` against known river/turn/flop sizes
- [x] Add a test that asserts the ONNX export accepts batch=1 and batch=N
- [x] Add a test for checkpoint sort ordering past epoch 9
- [x] Run full Python test suite (`pytest crates/cfvnet/python/`) — all pass
- [ ] Manual smoke (deferred): no turn shard exists yet — will validate with first real turn datagen run after the OOM blocker is fixed

## Out of scope

- Finding 5 (val sampled from training buffer) — design issue, separate bean if needed.
- Pre-existing failing Rust tests `mp_tui_scenarios::tests::resolve_empty_returns_root`, `tests::mp_6player_tui_section_parses` — separate beans.



## Summary of Changes

Cherry-picked from worktree `agent-ac865e53` to main as 4 commits:

| SHA | Commit |
|-|-|
| `d91894f` | fix(cfvnet): variable board_size in Python record decoder (fixes 1+2) |
| `c1f5fd5` | fix(cfvnet): use dynamic_axes for ONNX batch dimension (fix 3) |
| `6c0e8a1` | fix(cfvnet): sort checkpoints by numeric epoch, not alphabetically (fix 4) |
| `6812547` | test(cfvnet): fix off-by-one in record_size comment |

**Verified:**
- `record_size(b) = 17252 + b` matches Rust `cfvnet::datagen::storage::record_size` exactly. River=17257, turn=17256, flop=17255.
- All 36 Python tests pass in 5.20s (10 newly added).
- Code-reviewer agent confirmed all 4 fixes correct, tests grounded in the Rust binary contract (not tautological encode/decode round-trips), production code paths exercised.

**Files changed:** `cfvnet/{constants,data,gpu_buffer,export,train}.py`, `scripts/{eval,train}_boundary.py`, `tests/{test_data,test_export,test_train}.py`. No Rust changes.

**Deferred:** manual smoke test on a real turn shard — to run as part of first real turn datagen run after the OOM blocker is fixed.
