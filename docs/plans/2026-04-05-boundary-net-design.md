# BoundaryNet — River Boundary Evaluator for Range-Solver

## Problem

The range-solver currently solves river subgames to showdown (full depth). For turn-level solving, this is expensive — each turn DCFR iteration must traverse the full river tree for all 48 possible river cards.

A trained neural network can approximate river subgame values at the boundary, eliminating the need to solve through the river. This enables depth-bounded turn solving: solve the turn game tree with the network providing leaf values at river entry points.

### Mapping Problem

Using a CFVNet as a boundary evaluator introduces a representation mismatch:

- **Bet tree is baked in**: The network's predictions assume the specific bet sizes used during datagen. Different river bet structures require retraining.
- **Non-linear range dependence**: Unlike showdown leaves (linear in opponent range), the network output for each combo depends on both players' full range distributions.
- **Training vs inference distribution**: The network trains on random/blueprint ranges but evaluates ranges produced by the turn solver's DCFR iterations, which may be out-of-distribution in early iterations.
- **Iteration coupling**: The turn solver's ranges change every DCFR iteration, requiring re-evaluation of all river boundary nodes each iteration.

These are inherent to any depth-bounded approach with learned value functions. The design minimizes their impact through careful normalization and integration.

## Design Decisions

### Output: Normalized EV (not pot-relative, not chip EV)

The network outputs **normalized EV** per combo: chip EV divided by `(pot + effective_stack)`.

- **Bounded output**: values range from `-eff_stack/(pot+eff_stack)` (max loss) to `+1.0` (max win)
- **Interpretable**: "fraction of total money at stake that each hand wins/loses"
- **Recoverable**: multiply by `(pot + effective_stack)` to get chip EVs for the solver
- **No magic constants**: normalization is derived from game state, not hardcoded

### Input encoding: normalized pot/stack (not raw, not log)

Pot and effective stack are encoded as their fractions of total money at stake:

```
pot_feature = pot / (pot + effective_stack)
stack_feature = effective_stack / (pot + effective_stack)
```

Both in [0, 1], sum to 1. The ratio encodes SPR directly. The same normalization constant ties inputs to outputs — the network "knows" what 1.0 means because the pot/stack split in the input defines the output scale.

### Architecture: same as CfvNet

No equity skip connection. The MLP architecture is proven and the training infrastructure is shared. A new `BoundaryNet` struct reuses `HiddenBlock` from `CfvNet`.

### Datagen: no changes

Existing river datagen produces `TrainingRecord` with pot-relative CFVs. The encoding function converts to normalized targets at training time:

```
target[h] = cfv_pot_relative[h] * pot / (pot + effective_stack)
```

## Architecture

### Model

```
Input(2720) → [Linear → BatchNorm → PReLU] × N → Linear(1326)
```

Default: 7 hidden layers × 500 units (~2.9M parameters).

#### Input features (2720 floats)

| Feature              | Dim  | Encoding                          |
|----------------------|------|-----------------------------------|
| OOP range            | 1326 | Probability distribution          |
| IP range             | 1326 | Probability distribution          |
| Board one-hot        | 52   | 5 cards set to 1.0                |
| Rank presence        | 13   | Binary flags for board ranks      |
| pot / (pot+stack)    | 1    | [0, 1]                            |
| stack / (pot+stack)  | 1    | [0, 1]                            |
| Player               | 1    | 0.0 = OOP, 1.0 = IP              |

#### Output (1326 floats)

Normalized EV per combo. Board-blocked combos masked to zero during training.

At inference: `chip_ev[h] = normalized_ev[h] * (pot + effective_stack)`

### Loss Function

```
L = Huber(pred, target, mask, delta) + λ * (Σ range[h] * pred[h] - game_value_normalized)²
```

Where `game_value_normalized = game_value * pot / (pot + effective_stack)`.

Same structure as existing `cfvnet_loss`, operating in normalized space.

### Training Pipeline

Reuses existing infrastructure entirely:

- Streaming dataloader with shuffle buffer
- Cosine LR annealing with Adam optimizer
- Checkpoint/resume support
- Validation split

Only the `encode_record` function changes to produce normalized targets.

## Validation & Convergence

### Training metrics (per epoch)

- **Training loss**: Huber + aux on training batches
- **Validation loss**: same on held-out split (existing)
- **Normalized MAE**: mean absolute error on raw network output — "on average, how far off is each combo as a fraction of total money at stake?" (e.g., 0.02 = 2% error)
- **Per-SPR-bucket MAE**: normalized MAE broken out by SPR range (<1, 1-3, 3-10, 10+)

### Post-training evaluation (new CLI command)

- Load trained model + separate held-out test data (different datagen seed)
- Report: overall normalized MAE, per-SPR MAE, worst-case error
- **Exploitability comparison**: solve N reference river situations with full-depth range-solver AND with boundary evaluator, compare exploitability. This is the ground truth test — if boundary evaluation adds < X mbb/g exploitability vs full-depth, the model is good enough.

## Integration

### Boundary evaluator module

New `cfvnet::eval::boundary_evaluator` adapts `BoundaryNet` for the range-solver's existing depth-boundary code path in `evaluate_internal()`.

At a boundary node:

```
1. Normalize cfreach into ranges (zero blocked combos, renormalize)
2. Encode inputs (ranges, board, pot/(pot+stack), stack/(pot+stack), player)
3. Forward pass → normalized_ev[h]
4. chip_ev[h] = normalized_ev[h] * (pot + effective_stack)
5. result[h] = chip_ev[h] * cfreach_adjusted[h] / num_combinations
```

Step 5 matches the existing boundary formula — the evaluator provides `chip_ev[h]`, the solver handles reach weighting.

For turn solving: evaluate all 48 possible river cards, batch into a single forward pass (same pattern as existing `river_net_evaluator`).

### Tauri integration

Wire boundary evaluator into the explorer for turn solving:

- Config option to specify trained model path
- Range-solver's `solve` call accepts an optional boundary evaluator
- Tauri exploration commands pass it through when a model is loaded
- Users can solve turn spots with CFVNet at river boundaries instead of full-depth

## Scope

### In scope

- `BoundaryNet` model struct (reuses `HiddenBlock`, new encoding)
- Normalized EV encode/decode functions
- Boundary evaluator integration with range-solver
- Validation metrics (normalized MAE, per-SPR breakdown)
- Post-training evaluation command
- Tauri wiring for turn solving

### Out of scope

- Turn/flop boundary networks (future — depends on river net validation)
- Equity skip connection (considered and removed for simplicity)
- New datagen pipeline (reusing existing)
- Changes to existing `CfvNet` (untouched)

## File Changes

| File | Change |
|------|--------|
| `cfvnet/src/model/boundary_net.rs` | New — `BoundaryNet` struct, reuses `HiddenBlock` |
| `cfvnet/src/model/boundary_dataset.rs` | New — normalized EV encoding from `TrainingRecord` |
| `cfvnet/src/model/boundary_loss.rs` | New or extend `loss.rs` — same loss, normalized space |
| `cfvnet/src/model/boundary_training.rs` | New — training loop adapted for `BoundaryNet` |
| `cfvnet/src/eval/boundary_evaluator.rs` | New — range-solver leaf evaluator |
| `cfvnet/src/eval/mod.rs` | Export new module |
| `cfvnet/src/model/mod.rs` | Export new modules |
| `range-solver/src/game/evaluation.rs` | Wire boundary evaluator into depth-boundary path |
| `trainer/src/main.rs` | New CLI commands: `train-boundary`, `eval-boundary` |
| `tauri-app/src/exploration.rs` | Accept optional boundary evaluator for turn solving |
