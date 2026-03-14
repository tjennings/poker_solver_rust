# CFVnet Training Pipeline

Bottom-up: River -> Turn -> Flop. Each street's model becomes the leaf evaluator for the street above it.

All commands use `--release`. CUDA training adds `--features cuda`.

## Training Data Targets (from Supremus, Zarick et al. 2020)

| Network | Supremus | DeepStack | Our Target |
|---|---|---|---|
| River | 50M subgames | N/A | 10M (config default) |
| Turn | 20M subgames | 10M | 5M (pipeline proof), scale to 20M+ |
| Flop | 5M subgames | 1M | TBD |
| Preflop aux | 10M situations | 10M | TBD |

---

## Phase 1: River

### 1a. Datagen

Solve random 5-card board subgames with exact river evaluation. Each solve produces CFVs for all 1326 hand combos.

```bash
cargo run -p cfvnet --release -- generate \
  -c sample_configurations/river_cfvnet.yaml \
  -o data/river/river.bin \
  --per-file 1000000
```

- `--num-samples N` overrides config (default: 10M)
- `--threads N` overrides config (default: 8)
- `--per-file` splits output into multiple 1M-record files

### 1b. Train

```bash
cargo run -p cfvnet --release --features cuda -- train \
  -c sample_configurations/river_cfvnet.yaml \
  -d data/river/ \
  -o models/river_v1 \
  --backend cuda
```

- `-d` accepts a file or directory of split files
- Checkpoints saved every 10 epochs as `checkpoint_epochN.mpk.gz`
- Resume from checkpoint by pointing `-o` at existing model dir

### 1c. Evaluate (compare against exact solver)

```bash
cargo run -p cfvnet --release -- compare \
  -m models/river_v1 \
  --num-spots 1000 \
  --threads 8
```

- Generates random spots, solves exactly, compares against model predictions
- Reports MAE, mBB, best/worst spots, SPR breakdown, board texture breakdown

### 1d. Inspect training data distribution

```bash
cargo run -p cfvnet --release -- datagen-eval \
  -d data/river/
```

---

## Phase 2: Turn

Requires a trained river model. Turn datagen uses the river net as leaf evaluator instead of solving all 46 river runouts exactly.

**Config**: `sample_configurations/turn_cfvnet.yaml`
- `game.board_size: 4`
- `datagen.street: "turn"`
- `game.river_model_path: "local_data/models/river_v7/checkpoint_epoch340.mpk.gz"`

### 2a. Datagen

```bash
cargo run -p cfvnet --release -- generate \
  -c sample_configurations/turn_cfvnet.yaml \
  -o data/turn/turn.bin \
  --per-file 1000000
```

### 2b. Train

```bash
cargo run -p cfvnet --release --features cuda -- train \
  -c sample_configurations/turn_cfvnet.yaml \
  -d data/turn/ \
  -o models/turn_v1 \
  --backend cuda
```

### 2c. Evaluate (fast — river net at leaves)

```bash
cargo run -p cfvnet --release -- compare-net \
  -m models/turn_v1 \
  --river-model models/river_v1 \
  --num-spots 1000
```

### 2d. Evaluate (slow — exact river solver, ground truth)

```bash
cargo run -p cfvnet --release -- compare-exact \
  -m models/turn_v1 \
  --num-spots 20
```

### 2e. Inspect training data

```bash
cargo run -p cfvnet --release -- datagen-eval \
  -d data/turn/
```

---

## Phase 3: Flop

Requires a trained turn model (which itself requires the river model). Flop datagen uses the turn net as leaf evaluator.

**Config**: `sample_configurations/flop_cfvnet.yaml` (to be created)
- `game.board_size: 3`
- `datagen.street: "flop"`
- `game.turn_model_path` (TBD — not yet implemented)

### 3a-3e. Same pattern as turn

Commands follow the same structure. Flop support may require additional implementation for turn model leaf evaluation.

---

## Notes

- **Checkpoint path**: datagen needs the specific checkpoint file (e.g. `checkpoint_epoch340.mpk.gz`), not the directory
- **Update river_model_path** before turn datagen if river training has progressed past epoch 340
- **Adding more data**: generate additional files into the same directory, then retrain from checkpoint — the model doesn't care about dataset changes
- **Backend options**: `cuda` (NVIDIA GPU), `wgpu` (Metal/Vulkan), `ndarray` (CPU)
