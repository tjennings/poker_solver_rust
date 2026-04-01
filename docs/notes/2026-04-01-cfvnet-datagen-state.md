# CFVnet Datagen — State of Play (2026-04-01)

## What We Built

### Pipeline Architecture
3-stage producer-consumer for exact turn+river datagen:
```
[Stage 1: Deal Gen] → channel(16) → [Stage 3: Solve] → channel(50) → [Stage 4: Storage Writer] → disk
```
- Stage 1: samples situations, builds game trees
- Stage 3: solves with DCFR (early exit via `solve()`), extracts turn + river records
- Stage 4: streams writes to numbered files (turn and river separately)
- No buffering in storage — writes immediately via BufWriter, rotates at `per_file` threshold

### Key Features
- **Exact mode** (`mode: "exact"`): solves turn+river to showdown, no neural net needed
- **Per-depth bet sizes**: `bet_sizes: [["33%", "75%", "a"], ["75%", "a"]]` — first depth = bets, remaining = raises
- **Bet size fuzzing**: `bet_size_fuzz: 0.10` — ±10% random perturbation per deal
- **River record extraction**: one solve produces turn record + ~46 river records for free
- **Compressed storage**: 16-bit game trees (~4x less memory)
- **48-card batched GPU inference**: river model forward pass batched into single `[48, INPUT_SIZE]` tensor
- **Two GPU threads** for model-mode pipeline
- **Blueprint range generation**: loads blueprint bundle, walks preflop+flop strategy to produce realistic ranges

### Blueprint Range Precomputation
- `precompute-ranges` CLI command enumerates all preflop action paths
- Produces exact reach-weighted ranges per path (e.g., SRP: ~1000 combos, 3bet: ~400, 4bet: ~200)
- Wired into datagen sampler — samples paths weighted by frequency
- **Missing**: flop action propagation (currently preflop-only), all-in paths

### Performance
- **Exact mode throughput**: ~5/s with reduced bet sizes, early exit at target exploitability
- **Exploitability quality**: ~0.05 chips at convergence (~25 mbb/h) — excellent
- **Model mode throughput**: ~30/s (GPU-bound with 2 threads)
- **Model mode quality**: 14,000+ mbb/h — river model boundary values are bad

### Key Bottleneck: Tree Size
- Full turn+river tree with wide RSP ranges: **5.9 GB per game** (1000+ combos × 2 streets × raises)
- With blueprint ranges (3bet pot ~400 combos): ~200 MB — much more manageable
- With 4bet pot (~200 combos): even smaller

## What Works Well
1. Exact turn+river solving produces high-quality training data (25 mbb/h)
2. River records extracted for free from each turn solve (~46 per sample)
3. Blueprint preflop range enumeration correctly identifies SRP/3bet/4bet paths
4. Pipeline architecture with backpressure prevents OOM
5. Bet size fuzzing for model robustness

## Open Problems

### 1. Range Width → Tree Size
RSP ranges (~1000 combos) produce 5.9 GB trees. Need blueprint-filtered ranges to bring this down.
Currently preflop-only propagation; need flop propagation too for tight turn ranges.

### 2. River Model Quality
River model (river_v6, 7×768) produces 14,000 mbb/h exploitability at turn boundaries.
Exact solve produces 25 mbb/h. 560x quality gap. Root cause unknown — possibly:
- Input encoding mismatch between training and inference
- Model architecture too large for wgpu inference efficiency
- Training data quality issues

### 3. Why Supremus Uses Wide Ranges
Supremus trains on ALL combos (no blueprint filtering) because:
- CFVnets are the strategy (no separate blueprint)
- Self-play requires generalization to any range
- GPU solver is 1000x faster than CPU — wide ranges aren't a bottleneck
Our blueprint-filtered approach trades generalization for tractability.

### 4. Flop Path Enumeration
Need to extend preflop path enumeration through flop actions for all 1,755 canonical flops.
This requires bucket lookups (already loaded in BlueprintRangeGenerator).
~42K entries × ~10KB = ~420MB precomputed table.

### 5. Missing All-In Paths
Preflop enumeration misses all-in + call paths. These produce very tight ranges
that are important for training.

## Alternative Paths Considered
1. **Per-deal blueprint propagation** (`sample_turn_ranges`): walks one random path per deal. Fast but doesn't precompute.
2. **SPR bucketing**: average ranges per SPR bucket. Didn't work — averaging washes out range differences.
3. **MCCFR range accumulation**: collect ranges during blueprint training. Complex, not implemented.
4. **ReBeL batching pattern**: shared GPU queue for micro-batching across subgames. Good for model mode, irrelevant for exact mode.

## Config Reference
```yaml
game:
  initial_stack: 200
  bet_sizes:
    - ["33%", "75%", "a"]
    - ["75%", "a"]
  board_size: 4
  river_model_path: "local_data/models/river_v6/model"
  blueprint_path: "local_data/blueprints/1k_100bb_brdcfr_v2"

datagen:
  street: "turn"
  mode: "exact"
  num_samples: 2000000
  per_file: 10000
  bet_size_fuzz: 0.10
  solver_iterations: 600
  target_exploitability: 0.05
  threads: 10
  river_output: "local_data/cfvnet/river_from_turn.bin"
  blueprint_path: "local_data/blueprints/1k_100bb_brdcfr_v2"
```

## Files Modified (this session)
- `crates/cfvnet/src/datagen/turn_generate.rs` — 3-stage pipeline, exact mode, river extraction, storage stage
- `crates/cfvnet/src/datagen/blueprint_ranges.rs` — BlueprintRangeGenerator, sample_turn_ranges
- `crates/cfvnet/src/datagen/precompute_ranges.rs` — preflop path enumeration, PrecomputedRanges
- `crates/cfvnet/src/datagen/sampler.rs` — sample_situation_with_blueprint
- `crates/cfvnet/src/config.rs` — BetSizeConfig, mode, bet_size_fuzz, per_file, blueprint_path, river_output
- `crates/cfvnet/src/eval/river_net_evaluator.rs` — batched 48-card forward pass
- `crates/cfvnet/src/main.rs` — precompute-ranges CLI command
