# ReBeL Training Runbook

Step-by-step procedure for a full ReBeL training run, from a trained blueprint to a working value network.

## Prerequisites

- Trained blueprint V2: `strategy.bin` + `config.yaml` in blueprint dir
- Bucket files: `preflop.buckets`, `flop.buckets`, `turn.buckets`, `river.buckets` in cluster dir
- Hardware: RTX 6000 Ada (48GB VRAM) + 16 CPU threads minimum
- Build: `cargo build -p poker-solver-trainer --release`

## Directory Layout

```
/data/rebel/
├── config.yaml          # ReBeL config (copy from sample, edit paths)
├── rebel_buffer.bin     # disk-backed training buffer (auto-created)
├── training_data/       # exported cfvnet-format training files
├── models/              # trained value net checkpoints
└── validation/          # held-out validation sets
```

## Step 0: Create Config

Copy and edit `sample_configurations/rebel_river_seed.yaml`:

```yaml
blueprint_path: "/data/blueprints/500bkt_200bb"
cluster_dir: "/data/blueprints/500bkt_200bb/clusters"
output_dir: "/data/rebel"

game:
  initial_stack: 400      # 200bb × 2 chips per bb
  small_blind: 1
  big_blind: 2

seed:
  num_hands: 2000000      # 2M hands → ~4-8M PBS snapshots across streets
  seed: 42
  threads: 16
  solver_iterations: 1024
  target_exploitability: 0.005
  bet_sizes:
    flop: [[0.33, 0.67, 1.0], [0.33, 0.67, 1.0]]
    turn: [[0.5, 0.75, 1.0], [0.5, 0.75, 1.0]]
    river: [[0.5, 0.75, 1.0], [0.5, 0.75, 1.0]]

training:
  hidden_layers: 7
  hidden_size: 500
  batch_size: 4096
  epochs: 200
  learning_rate: 0.0003

buffer:
  max_records: 12000000
  path: "rebel_buffer.bin"
```

Key tuning knobs:
- `num_hands`: more = better coverage, slower. Start with 500K for smoke test.
- `solver_iterations`: higher = more accurate solves, slower. 256 for smoke test, 1024 for production.
- `epochs`: more = better fit but risk of overfitting. Watch validation MSE.

## Step 1: Smoke Test (5 minutes)

Verify the pipeline works end-to-end with tiny parameters before committing hours:

```bash
# Create a minimal config (override num_hands in a temp file or edit)
# Use num_hands=1000, solver_iterations=100, epochs=5

cargo run -p poker-solver-trainer --release -- rebel-seed \
  --config /data/rebel/config_smoke.yaml
```

Expected output:
- "Generated X PBS snapshots"
- "Solved Y/Z records"
- "Exported N training records"

Check the output dir for `rebel_buffer.bin` and training data files.

## Step 2: Generate Held-Out Validation Set

Generate BEFORE training so it's independent of training data:

```bash
cargo run -p poker-solver-trainer --release -- rebel-validate \
  --config /data/rebel/config.yaml \
  --num-examples 500 \
  --output /data/rebel/validation/val_set.bin
```

This solves 500 random river subgames exactly. Takes ~5-15 minutes depending on solver iterations. Save this file — you'll use it to track MSE throughout training.

## Step 3: Offline Seeding (Full Pipeline)

This is the main training run. The `rebel-train` command runs bottom-up seeding:

```bash
cargo run -p poker-solver-trainer --release -- rebel-train \
  --config /data/rebel/config.yaml \
  --offline-only
```

### What happens internally

1. **River seeding** (~hours)
   - Generates PBSs from blueprint play
   - Solves river subgames exactly (no depth limit, no value net needed)
   - Trains value net on river data
   - Expected: ~2-5M river training records

2. **Turn seeding** (~hours)
   - Solves turn subgames with depth_limit=0
   - River value net evaluates at river boundary leaves
   - Retrains value net on river + turn data

3. **Flop seeding** (~hours)
   - Same pattern, value net at turn boundaries
   - Retrains on river + turn + flop data

4. **Preflop seeding** (~hours)
   - Same pattern, value net at flop boundaries
   - Final retrain on all accumulated data

### Monitoring

Watch stderr for per-street progress:
```
=== Offline seeding: River ===
Generated 1234567 PBSs for River
Solved 800000/1234567 records
River training MSE: 0.004523

=== Offline seeding: Turn ===
...
```

### Expected Timelines (RTX 6000 Ada, 16 threads)

| Phase | Records | Solve Time | Train Time | Total |
|-------|---------|------------|------------|-------|
| River | 2-4M | 2-4h | 30min | ~4h |
| Turn | 1-3M | 3-6h | 30min | ~6h |
| Flop | 1-2M | 4-8h | 30min | ~8h |
| Preflop | 0.5-1M | 2-4h | 30min | ~4h |
| **Total** | **5-10M** | | | **~22h** |

These are rough estimates. River is fastest (exact solving, small trees). Flop is slowest (largest trees, most boundary nodes).

## Step 4: Live Self-Play (Optional, Experimental)

After offline seeding produces a working value net, optionally refine with self-play:

```bash
cargo run -p poker-solver-trainer --release -- rebel-train \
  --config /data/rebel/config.yaml \
  --model /data/rebel/models/checkpoint_epoch200.mpk.gz
```

This skips offline seeding and starts the live self-play loop (Algorithm 1):
- Plays hands via subgame solving at every decision
- Records training examples
- Periodically retrains the value net

Self-play is significantly slower than offline seeding (every decision requires a full subgame solve). Budget days, not hours.

## Step 5: Evaluate

### MSE on held-out set

```bash
cargo run -p poker-solver-trainer --release -- rebel-eval \
  --config /data/rebel/config.yaml \
  --model /data/rebel/models/checkpoint_epoch200.mpk.gz \
  --mode mse
```

Good MSE targets (pot-relative):
- River: < 0.005
- Turn: < 0.01
- Flop: < 0.02
- Overall: < 0.01

### Head-to-head vs blueprint (future)

```bash
cargo run -p poker-solver-trainer --release -- rebel-eval \
  --config /data/rebel/config.yaml \
  --model /data/rebel/models/checkpoint_epoch200.mpk.gz \
  --mode h2h \
  --num-hands 100000
```

Target: positive win rate vs blueprint in mbb/hand.

## Troubleshooting

### "Failed to load blueprint"
- Check `blueprint_path` points to directory containing `strategy.bin` and `config.yaml`
- Verify the blueprint was trained with `train-blueprint`

### "Failed to load buckets"
- Check `cluster_dir` contains `.buckets` files
- Must match the blueprint's clustering (same bucket counts)

### Solver timeout / very slow
- Reduce `solver_iterations` (try 256)
- Increase `target_exploitability` (try 0.01)
- Check thread count matches available cores

### Out of disk space
- Buffer at 12M records × 17KB = ~200GB
- Reduce `max_records` or use smaller `num_hands`
- Training data export is separate from buffer

### Training MSE not decreasing
- Check learning rate (try 1e-4 or 1e-3)
- Increase epochs
- Verify training data has non-zero CFVs (check with rebel-validate)
- Ensure pot-relative scale (values should be in [-2, 2] range, not hundreds)

### GPU OOM during training
- Reduce `batch_size` (try 2048 or 1024)
- The CfvNet model itself is small (~20MB), batch size is the main memory driver

## Config Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed.num_hands` | — | Number of hands to play under blueprint |
| `seed.seed` | 42 | RNG seed for reproducibility |
| `seed.threads` | 16 | Parallel solving threads |
| `seed.solver_iterations` | 1024 | DCFR iterations per subgame |
| `seed.target_exploitability` | 0.005 | Early stopping threshold |
| `training.hidden_layers` | 7 | Value net depth |
| `training.hidden_size` | 500 | Value net width |
| `training.batch_size` | 4096 | Training batch size |
| `training.epochs` | 200 | Training epochs per street |
| `training.learning_rate` | 3e-4 | Adam learning rate |
| `training.huber_delta` | 1.0 | Huber loss delta |
| `buffer.max_records` | 12M | Max records in reservoir buffer |
