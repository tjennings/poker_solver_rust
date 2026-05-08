# Flop-Boundary CFVNet Contract

Project epic: `poker_solver_rust-obst`

## Goal

Train a direct flop-boundary CFVNet that can be used one street earlier than the
current direct turn-boundary model. The intended runtime use is a
preflop/flop boundary evaluator: a solver may cut at flop-boundary nodes, run
one direct ONNX evaluation on a 3-card flop board, and receive per-combo CFVs
for both players.

The teacher for this model is the current direct turn-boundary CFVNet. Data
generation samples flop states, solves flop subgames to turn boundary leaves,
evaluates those turn leaves with the direct turn-boundary ONNX model, and
stores the resulting flop-root counterfactual values.

## Existing Contract We Should Reuse

The current binary `TrainingRecord` layout is already variable-board-size:

- `board_size: u8`
- `board: [u8; board_size]`
- `pot: f32`
- `effective_stack: f32`
- `player: u8`
- `game_value: f32`
- `oop_range: [f32; 1326]`
- `ip_range: [f32; 1326]`
- `cfvs: [f32; 1326]`
- `valid_mask: [u8; 1326]`

This means direct flop-boundary records do not need a new physical binary
record format. A flop-boundary record should be the same `TrainingRecord`
layout with `board_size=3`.

The model input size also stays fixed at 2720:

- OOP range: 1326
- IP range: 1326
- Board one-hot: 52
- Rank presence: 13
- Pot fraction: 1
- Effective-stack fraction: 1
- Player indicator: 1

Because the board is encoded as deck one-hot plus rank-presence, 3-card, 4-card,
and 5-card boards all share the same neural input shape.

## New Dataset Manifest Contract

The manifest layer is currently turn-specific. Flop-boundary support should
generalize it without changing existing turn datasets.

Recommended schema additions:

- Add `DatasetStreet::FlopBoundary`.
- Add `FLOP_BOUNDARY_BOARD_SIZE = 3`.
- Keep `schema_version = 1` unless the on-disk `TrainingRecord` bytes change.
- Add `RecordSchema::flop_boundary()` with:
  - `format = "cfvnet_training_record_v1"`
  - `board_size = 3`
  - `record_size_bytes = record_size(3)`
  - `input_size = 2720`
  - `output_size = 1326`
  - `normalization = chip_cfv_over_pot_plus_stack`
- Add generic validation helpers or a `validate_flop_boundary()` sibling.
- Add `add_flop_boundary_shard()` or generalize shard insertion by street.

Target source should distinguish learned turn-net labels from exact labels.
Recommended enum additions:

- `TurnNet`
- `ExactTurn`

For this project's main production path, `target_source = turn_net`.

## Range Contract

Every generated flop-boundary record must use the same canonical range contract
as the validated turn-boundary v2 data:

- `oop_range` and `ip_range` are 1326-combo canonical vectors.
- Combos blocked by the 3-card flop board must be exactly zero.
- All values must be finite and non-negative.
- Each player's unblocked range must normalize to total mass 1.0, within the
  existing evaluator tolerance.
- `valid_mask[i] = 0` for board-blocked combos and `1` for legal combos.
- `cfvs[i] = 0` for invalid/blocked combos.

The datagen evaluator must reject raw blueprint-style unnormalized ranges for
flop-boundary datasets, just as it now does for turn-boundary v2 data.

## Value Contract

Stored `cfvs` remain pot-relative values:

```text
stored_cfv = chip_cfv / pot
```

Training converts stored values to the neural target:

```text
target = chip_cfv / (pot + effective_stack)
       = stored_cfv * pot / (pot + effective_stack)
```

Runtime converts neural outputs back to chip values:

```text
chip_cfv = output * (pot + effective_stack)
```

This is the same direct BoundaryNet value contract as the current
turn-boundary model.

`game_value` in the record should be recomputed from the canonical player range
and stored pot-relative CFVs:

```text
game_value = sum_i player_range[i] * stored_cfv[i]
```

The trainer may continue recomputing normalized game value from normalized
targets, which is already the safer behavior.

## Model Contract

The first flop-boundary model should reuse the current `BoundaryNet` architecture:

- Input: `[batch, 2720]`
- Output: `[batch, 1326]`
- Board size: exactly 3 for direct flop-boundary training data.
- Inference mode: `BoundaryInferenceMode::Direct`.
- Runtime call pattern: batch OOP and IP rows together for each boundary cache
  fill, same as the direct turn-boundary evaluator.

The runtime evaluator must allow direct mode on 3-card boards:

```rust
(BoundaryInferenceMode::Direct, 3 | 4 | 5)
```

`RiverEnumeratedTurn` must remain unchanged:

- 5-card board: direct river-net eval.
- 4-card board: enumerate rivers and average.
- 3-card board: unsupported.

This prevents accidental use of a river model as a flop model.

## Flop-To-Turn Oracle Contract

For one sampled flop situation:

1. Sample or construct a 3-card flop board, pot, effective stack, and canonical
   OOP/IP ranges.
2. Build a flop subgame whose depth-boundary leaves occur at turn states.
3. For each turn boundary leaf:
   - Use the boundary board `[flop cards..., turn card]`.
   - Use the leaf OOP/IP reaches.
   - Evaluate with the current direct turn-boundary ONNX model using
     `BoundaryInferenceMode::Direct`.
   - Convert returned chip CFVs into the range-solver boundary units expected by
     the solve.
4. Solve the flop subgame.
5. Extract root CFVs for OOP and IP.
6. Write two `TrainingRecord`s, one per player, using the flop board and the
   root canonical ranges.

This gives the flop model the same recursive role as DeepStack/Supremus-style
depth-limited solving: the learned value function approximates the next street,
and the current street is solved against that value function.

## Sampling Policy

Carry forward the lessons from turn-boundary v2:

- Do not trust uniform random pots/SIR alone.
- Stratify by pot bucket and SPR bucket.
- Oversample high-action-depth / 4-bet-plus pots.
- Oversample low-SPR all-in-pressure pots.
- Preserve tiny-pot high-SPR coverage because these were difficult for the
  previous net and are common in early-street solving.
- Track board texture, range entropy, boundary ordinal, and raise depth in the
  manifest.

Initial recommended output path:

```text
local_data/cfvnet/flop_boundary/v1
```

Initial recommended model path:

```text
local_data/models/flop_boundary_cfvnet_v1
```

## Validation Requirements

Before large-scale generation:

- Generate a small pilot shard.
- Run `datagen-eval` with flop-boundary support.
- Verify range contract: finite, non-negative, board-blocked combos zero,
  player range sums near 1.0.
- Verify target contract: finite, blocked targets zero, target scale plausible
  by pot/SPR buckets.
- Compare a small number of pilot spots to exact or slower baselines if
  feasible.
- Record teacher model hashes for `best.onnx` and `best.onnx.data`.

Before training:

- Confirm manifest coverage includes all intended pot/SPR/action strata.
- Confirm validation split is frozen and representative.
- Confirm old turn-boundary datasets still validate.

Before runtime use:

- Export `best.pt` to ONNX plus external data.
- Verify Python ONNX and Rust `eval-boundary` agree on a held-out shard.
- Verify `compare-solve` accepts:

```text
--flop-boundary cfvnet --flop-model <path> --flop-model-kind direct
```

## Implementation Slices

1. Generalize manifest/data validation for `flop_boundary`.
2. Let direct ONNX/Burn inference accept 3-card boards.
3. Add a flop-boundary oracle generator using turn-boundary ONNX leaves.
4. Extend `datagen-eval` for flop-boundary manifests and summaries.
5. Add sample datagen/training configs.
6. Run pilot generation and evaluate distribution.
7. Train/export/evaluate the first model.
8. Wire the model through compare-solve and Tauri preflop/flop boundary config.
