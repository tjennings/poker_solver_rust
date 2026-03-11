# River CFVnet Design

## Overview

Train a Deep Counterfactual Value Network (CFVnet) for the river street. Given a river game state (5-card board, pot, stacks, both players' ranges), the network predicts the Nash counterfactual value for each of the 1326 hole card combos. This replaces solving full river subtrees at runtime, enabling real-time depth-limited solving.

Based on the DeepStack/Supremus architecture (Moravcik et al. 2017, Zarick et al. 2020).

## Architecture Decisions

- **Framework**: burn (pure Rust, wgpu backend for Mac/Metal, CUDA backend for workstation)
- **Crate**: `crates/cfvnet` — new workspace member, depends on `range-solver`
- **Structure**: Single crate with modules: `datagen`, `model`, `eval`, `cli`
- **Training data**: Materialized to disk (solve once, iterate on hyperparameters)
- **Starting scale**: 1M subgames (~30GB, few hours on 8 cores), designed to scale to 50M

## Data Generation Pipeline

### Situation Sampling

Each training sample is a random river situation:

- **Board**: 5 cards sampled uniformly from all C(52,5) boards
- **Pot**: Sampled from stratified intervals: {[4,20), [20,80), [80,200), [200,400)} in SB units, uniform probability per interval, uniform within interval
- **Effective stack**: Given pot, sample remaining stack from [0, 200 - pot/2] (for 100bb game)
- **Ranges**: DeepStack's R(S,p) procedure — recursive random split correlated with hand strength. Generate independently for each player. Zero out board-conflicting combos, renormalize.

### R(S, p) Range Generation

```
R(S, p):
  if |S| = 1: assign probability p to that hand
  else:
    p1 ~ Uniform(0, p)
    p2 = p - p1
    S1 = stronger half of S (by hand strength on this board)
    S2 = weaker half
    R(S1, p1), R(S2, p2)
```

Produces ranges correlated with hand strength but with high variance, covering the space of ranges CFR might encounter during re-solving.

### Solving

For each situation:
1. Construct `CardConfig` with board + both ranges
2. Construct `TreeConfig` with `initial_state: BoardState::River`, pot, stack, bet sizes `[0.25, 0.5, 1.0, all-in]`
3. Build `PostFlopGame`, allocate, solve with DCFR (1000 iterations, target exploitability 0.5% of pot)
4. Extract `expected_values(0)` and `expected_values(1)` — 1326 floats each, normalized by pot

### Serialization Format

Each sample written to a flat binary file:
- Header: board (5 × u8), pot (f32), stack (f32), player (u8)
- OOP range: 1326 × f32
- IP range: 1326 × f32
- Labels (CFVs): 1326 × f32
- Valid mask: 1326 × u8 (which combos are board-compatible)

One record per player perspective (both OOP and IP extracted per solve).

### Validation

1. **R(S,p) range generator**: Output sums to 1.0, all values ≥ 0, board-blocked combos are zero, distribution is correlated with hand strength (strong hands have statistically higher mean reach)
2. **Situation sampler**: Pot/stack/board distributions match configured intervals, no invalid card overlaps
3. **Solve wrapper**: Solve a known river spot (e.g., nuts vs air on a dry board), verify CFVs match hand-calculated expected values. Verify exploitability < threshold after solving.
4. **Round-trip serialization**: Write a sample, read it back, verify exact equality

## Network Architecture

### Model

```
Input(2660) → [Linear(500) → BN → PReLU] × 7 → Linear(1326)
```

~3.5M parameters.

### Input Encoding (2660 floats)

| Feature | Dimensions | Encoding |
|-|-|-|
| OOP range | 1326 | Reach probabilities |
| IP range | 1326 | Reach probabilities |
| Board | 5 | Card ID / 51.0 (normalized to [0,1]) |
| Pot | 1 | pot / 400 (normalized by max stack) |
| Effective stack | 1 | stack / 400 |
| Player indicator | 1 | 0.0 (OOP) or 1.0 (IP) |

### Output

1326 floats — pot-relative CFVs. Board-conflicting combos masked in loss.

### Loss Function

Combined loss: `L = L_huber + λ * L_aux`

- **L_huber**: Huber loss (delta=1.0) on valid (non-board-conflicting) combos only
- **L_aux**: `(Σ range[i] * cfv_pred[i] - game_value)²` — enforces that weighted sum of predicted CFVs equals the known game value
- **λ**: 1.0 (configurable)

### Training Configuration

| Parameter | Default |
|-|-|
| Hidden layers | 7 |
| Hidden size | 500 |
| Optimizer | Adam |
| Learning rate | 1e-3 → 1e-5 (cosine decay) |
| Batch size | 2048 |
| Epochs | 2 |
| Huber delta | 1.0 |
| Aux loss weight | 1.0 |
| Validation split | 5% |
| Checkpoint interval | Every 1000 batches |

### Checkpointing

Save model weights every N batches using burn's built-in model record serialization. Track and save the best model by validation loss.

### Validation

1. **Overfit test**: Train on a single batch of ~100 samples. Loss must converge to near-zero.
2. **Shape test**: Verify input/output dimensions at construction time with dummy tensors
3. **Aux loss test**: For a synthetic sample where game value is known, verify the auxiliary loss term computes correctly
4. **Masking test**: Board-conflicting combos contribute zero to the loss regardless of predicted values
5. **Checkpoint round-trip**: Save model, reload, verify identical predictions on a fixed input

## Evaluation Pipeline

### Level 1: Prediction Accuracy (network in isolation)

On held-out validation set (5% of generated data):
- Mean absolute error in mbb/hand
- Max absolute error
- Error by hand category (nuts, strong, medium, weak, air)
- Correlation: predicted CFV vs true CFV

Target: Huber validation loss ≤ 0.015 (matching Supremus river network).

### Level 2: End-to-End Solver Quality

~100 specific river spots with known exact solutions:
1. Solve using full range-solver (ground truth)
2. Query CFVnet for predicted CFVs
3. Compare predictions against exact CFVs
4. Build turn subgames using CFVnet at river leaves, compare resulting turn strategies against full turn+river range-solver solves

### Level 3: Regression Suite

Fixed set of ~20 canonical river spots (saved as fixtures) covering:
- Dry boards (rainbow, unpaired)
- Wet boards (flush possible, paired)
- Polarized vs merged ranges
- Small pot / big pot
- Deep / shallow stacks

Run as a quick smoke test after any model change.

### Validation

1. **Metric correctness**: Compute MAE by hand on 3-4 synthetic samples, verify evaluation code matches
2. **Comparison harness**: For a spot with perfect predictions, verify reported error is zero

## CLI and Workflow

### Subcommands

```bash
# Generate training data
cargo run -p cfvnet --release -- generate \
  --config river_config.yaml \
  --output data/river_training.bin \
  --num-samples 1000000 \
  --threads 8

# Train the network
cargo run -p cfvnet --release -- train \
  --config river_config.yaml \
  --data data/river_training.bin \
  --output models/river_v1 \
  --backend wgpu

# Evaluate on held-out data
cargo run -p cfvnet --release -- evaluate \
  --model models/river_v1 \
  --data data/river_validation.bin

# Compare against exact solves
cargo run -p cfvnet --release -- compare \
  --model models/river_v1 \
  --num-spots 100 \
  --threads 8
```

### Config File (river_config.yaml)

```yaml
game:
  initial_stack: 200
  bet_sizes: ["25%", "50%", "100%", "a"]
  add_allin_threshold: 1.5
  force_allin_threshold: 0.15

datagen:
  num_samples: 1_000_000
  pot_intervals: [[4,20], [20,80], [80,200], [200,400]]
  solver_iterations: 1000
  target_exploitability: 0.005
  threads: 8
  seed: 42

training:
  hidden_layers: 7
  hidden_size: 500
  batch_size: 2048
  epochs: 2
  learning_rate: 0.001
  lr_min: 0.00001
  huber_delta: 1.0
  aux_loss_weight: 1.0
  validation_split: 0.05
  checkpoint_every_n_epochs: 1000

evaluation:
  regression_spots: fixtures/river_spots.json
```

### Progress Reporting

- `generate`: Progress bar with samples/sec, running mean exploitability, ETA
- `train`: Epoch, batch, train loss, validation loss, learning rate per interval
- `evaluate`: Summary table of metrics

### Validation

1. **Config parsing**: YAML loads correctly, defaults fill in, invalid values rejected with clear errors
2. **Reproducibility**: Same seed + config → identical training data (bitwise). Test with 100-sample run.
3. **Interrupt/resume**: `generate` can append to existing output file. Verify by generating 50, stopping, generating 50 more, comparing against single 100-sample run.

## Future Work (not in scope)

- Turn/flop/preflop CFVnet training (same pipeline, different street)
- CFR-D continual resolving gadget integration
- Real-time single-street solver using CFVnet at leaves
- Multiplayer extensions (N-player range inputs)
