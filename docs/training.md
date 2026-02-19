# Training Reference

The `poker-solver-trainer` CLI runs MCCFR iterations and saves a strategy bundle. This document covers all configuration options.

## Config Structure

Training configs are YAML files with two top-level sections: `game` and `training`, plus an optional `abstraction` section for EHS2 mode.

## Game Settings

```yaml
game:
  stack_depth: 100                        # Effective stack in big blinds
  bet_sizes: [0.33, 0.67, 1.0, 2.0, 3.0] # Pot-fraction bet sizes (all-in always included)
  max_raises_per_street: 3                # Cap on bets/raises per street (default: 3)
```

- `bet_sizes` are pot fractions. `[0.5, 1.0, 1.5]` means half-pot, pot, and 1.5x pot. All-in is always available regardless of this list.
- `max_raises_per_street` keeps the game tree tractable. After this many bets on a street, only fold/call/check remain.

## Training Settings

```yaml
training:
  iterations: 5000          # Total MCCFR iterations (or use convergence_threshold)
  seed: 42                  # RNG seed for reproducibility
  output_dir: "./my_strategy"
  mccfr_samples: 5000       # Deals sampled per iteration (default: 500)
  deal_count: 50000          # Pre-generated deal pool size (default: 50000)
```

## Abstraction Modes

Choose one via `abstraction_mode`:

### `ehs2` (default)

EHS2 bucketing with Monte Carlo equity estimation. Fine-grained but expensive to compute. Requires an `abstraction` section:

```yaml
training:
  abstraction_mode: ehs2

abstraction:
  flop_buckets: 200          # EHS2 buckets on flop (production: 5000)
  turn_buckets: 200          # EHS2 buckets on turn (production: 5000)
  river_buckets: 500         # EHS2 buckets on river (production: 20000)
  samples_per_street: 5000   # Monte Carlo samples for bucket boundaries
```

### `hand_class`

Categorical hand classification (19 classes: Flush, Set, Overpair, etc.). O(1) per hand, interpretable, no `abstraction` section needed:

```yaml
training:
  abstraction_mode: hand_class
```

### `hand_class_v2`

Hand class + intra-class strength + showdown equity + draw flags. Finer resolution than `hand_class` while remaining interpretable:

```yaml
training:
  abstraction_mode: hand_class_v2
  strength_bits: 4           # Intra-class strength resolution, 0-4 bits (default: 4)
  equity_bits: 4             # Showdown equity bin resolution, 0-4 bits (default: 4)
```

The `strength_bits` and `equity_bits` control how finely hands within the same class are distinguished. 4 bits = 16 levels, 0 = dimension omitted. Higher values produce more info sets (larger strategy tables) but capture more nuance.

## Stratified Deal Generation

For `hand_class` and `hand_class_v2` modes, you can ensure rare hand classes (e.g., quads, straight flushes) appear in the deal pool:

```yaml
training:
  min_deals_per_class: 50       # Minimum deals per hand class (default: 0 = disabled)
  max_rejections_per_class: 500000  # Max rejection-sample attempts per deficit class
```

## Regret-Based Pruning

Pruning skips actions with low cumulative regret, speeding up training by avoiding clearly bad lines. Probe iterations periodically explore all actions to prevent permanent pruning.

```yaml
training:
  pruning: true                  # Enable pruning (default: false)
  pruning_threshold: -5.0        # Regret threshold for pruning (default: 0.0)
  pruning_warmup_fraction: 0.30  # Fraction of iterations before pruning starts (default: 0.2)
  pruning_probe_interval: 20     # Full exploration every N iterations (default: 20)
```

- `pruning_threshold` controls how negative a regret must be before the action is pruned. With DCFR, negative regrets decay asymptotically but never reach zero — a threshold of 0 would prune actions permanently. A negative threshold (e.g., -5.0) allows DCFR's decay to bring regrets back above the threshold between probes, so actions can recover if the strategy shifts.
- `pruning_warmup_fraction` delays pruning until the strategy has partially converged. At 0.30, the first 30% of iterations explore everything.
- `pruning_probe_interval` runs a full un-pruned iteration every N iterations to discover if pruned actions have become viable.

## Convergence-Based Stopping

Instead of a fixed iteration count, train until the strategy stabilizes:

```yaml
training:
  convergence_threshold: 0.001   # Stop when mean L1 delta < this value
  convergence_check_interval: 100  # Check every N iterations (default: 100)
```

When `convergence_threshold` is set, `iterations` is ignored. The trainer runs in a loop: train for `convergence_check_interval` iterations, snapshot the strategy, compute the mean L1 distance from the previous snapshot, and stop if it falls below the threshold. Each check also saves a numbered checkpoint bundle.

The strategy delta is the average per-info-set sum of absolute differences in action probabilities between consecutive snapshots. A value of 0.001 means action probabilities are changing by less than 0.1% on average per info set.

## Exhaustive Abstract Deals

For `hand_class_v2` mode, you can enumerate **all** abstract deal trajectories instead of sampling randomly. This eliminates Monte Carlo variance entirely — every iteration is a complete traversal of the finite abstract game.

The enumerator walks all hole card pairs x 1,755 canonical flops x all turn/river completions, encodes per-street hand bits, determines showdown winners, and deduplicates into weighted abstract deals. With `strength_bits=0, equity_bits=0`, billions of concrete deals compress to ~1M abstract deals (~100x compression).

```yaml
training:
  abstraction_mode: hand_class_v2
  strength_bits: 0
  equity_bits: 0
  exhaustive: true            # generate abstract deals in-memory
```

Or pre-generate deals to disk and reference them:

```yaml
training:
  abstract_deals_dir: ./my_deals/   # load pre-generated deals
```

Pre-generate with the `generate-deals` command (see [CLI Reference](cli.md#generating-abstract-deals)). Use `exhaustive` with the `sequence` or `gpu` solver — MCCFR ignores it.

Higher bit configs produce more unique trajectories and less compression. At 4/4 bits there is essentially no compression, so `exhaustive` is only practical for low-bit configs (0/0 or 1/1).

## Solver Backends

Select with `--solver`:

### MCCFR (default)

Samples random deals per iteration. Best for large games with many info sets. Low memory, handles any abstraction mode.

```bash
cargo run -p poker-solver-trainer --release -- train -c config.yaml
cargo run -p poker-solver-trainer --release -- train -c config.yaml -t 4  # limit threads
```

MCCFR uses Discounted CFR (DCFR):
- DCFR discounting with alpha=1.5, beta=0.5, gamma=2.0
- Configurable regret-based pruning with negative threshold support
- Parallel training via Rayon (frozen-snapshot accumulation pattern)
- Average strategy skips first 50% of iterations

### Sequence-form CFR

Materializes the game tree as a flat graph and runs level-by-level CFR over all deals every iteration. No sampling variance — each iteration is a complete traversal. Best for `hand_class` mode where the tree is small enough to materialize (~200-300K info sets).

```bash
cargo run -p poker-solver-trainer --release -- train -c config.yaml --solver sequence
```

Trade-offs vs MCCFR:
- Each iteration is more expensive (traverses all deals, not a sample)
- But each iteration makes more progress (no sampling noise)
- No `mccfr_samples` parameter needed
- Memory scales with tree size x deal count

### GPU-accelerated CFR

Same algorithm as sequence-form, but the inner loop runs on GPU compute shaders via wgpu. Cross-platform: Metal (macOS), Vulkan (Linux/Windows), DX12 (Windows).

```bash
cargo run -p poker-solver-trainer --features gpu --release -- train -c config.yaml --solver gpu
```

~7.7x faster than CPU sequence-form on Apple Silicon for 25BB hand_class configs.

### When to use each solver

| Solver | Best for | Deal handling |
|-|-|
| `mccfr` | Large games, `ehs2`/`hand_class_v2`, production training | Samples per iteration |
| `sequence` | Small-medium games, exact convergence, `exhaustive` mode | Full traversal |
| `gpu` | Same as `sequence` but faster, when GPU is available | Full traversal on GPU |

With `exhaustive: true`, `sequence` and `gpu` solvers use weighted abstract deals instead of random concrete deals.

## Training Output

Training saves to the output directory:

```
my_strategy/
├── config.yaml       # Game and abstraction settings (human-readable)
├── blueprint.bin     # Trained strategy (bincode, FxHashMap<u64, Vec<f32>>)
└── boundaries.bin    # EHS2 bucket boundaries (only for ehs2 mode)
```

## Progress Output

Training prints progress at 10 checkpoints with exploitability, sample strategies, and ETA:

```
=== Checkpoint 3/10 (300/1000 iterations) ===
Exploitability: 2.4531 (down from 3.1204)
Time: 12.3s elapsed, ~28.7s remaining

SB Opening Strategy (preflop, facing BB):
Hand  |  Fold  Call   R50  R9950
------|------------------------
AA    |  0.00  0.12  0.72  0.16
AKs   |  0.02  0.35  0.58  0.05
72o   |  0.85  0.10  0.05  0.00
```

## Complete Example

```yaml
game:
  stack_depth: 25
  bet_sizes: [0.5, 1.0, 1.5]

training:
  iterations: 5000
  seed: 42
  output_dir: "./handclass_25bb_v2"
  mccfr_samples: 5000
  deal_count: 50000
  abstraction_mode: hand_class_v2
  strength_bits: 4
  equity_bits: 4
  min_deals_per_class: 50
  max_rejections_per_class: 500000
  pruning: true
  pruning_threshold: -5.0
  pruning_warmup_fraction: 0.30
  # convergence_threshold: 0.001     # uncomment to train until converged
  # convergence_check_interval: 100
  # exhaustive: true                 # uncomment for exhaustive abstract deals (low-bit only)
```
