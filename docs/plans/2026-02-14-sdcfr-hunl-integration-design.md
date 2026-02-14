# SD-CFR HUNL Integration Design

**Date:** 2026-02-14
**Status:** Approved
**Goal:** Run Single Deep CFR on full HUNL poker as an alternative solver alongside MCCFR, with checkpointing for mid-training evaluation via Tauri.

## Architecture

```
training_sdcfr.yaml
    |
poker-solver-trainer (sd-cfr subcommand)
    |
HunlPostflop::initial_states() --> Vec<PostflopState> (generated once, stratified)
    |
HunlStateEncoder (new)
    |
SdCfrSolver::train() (modified: external deal pool + checkpoint callback)
    |  (every N iterations)
ExplicitPolicy --> BlueprintStrategy --> StrategyBundle (checkpoint)
    |
Tauri frontend loads latest checkpoint (no changes needed)
```

## Components

### 1. HunlStateEncoder

**File:** `crates/deep-cfr/src/hunl_encoder.rs` (~50 lines)

Implements `StateEncoder<PostflopState>`:

- Extracts hole cards from `state.p1_holding` / `state.p2_holding` based on `player`
- Extracts board from `state.board` (0-5 cards)
- Calls existing `canonicalize(hole, board)` for suit-isomorphic card encoding
- Iterates `state.history: ArrayVec<(Street, Action), 40>` to build `Vec<BetAction>`
  - For `Action::Bet(idx)` / `Action::Raise(idx)`: looks up `config.bet_sizes[idx]` for pot fraction
  - For `Action::Call`: pot fraction = to_call / pot (or 0.0 as sentinel)
  - For `Action::Check` / `Action::Fold`: pot fraction = 0.0
- Calls existing `encode_bets(action_history)` for 48-dim bet features
- Returns `InfoSetFeatures { cards, bets }`

**Note:** The encoder needs access to `PostflopConfig.bet_sizes` to resolve bet indices to pot fractions. Store a reference to the config in the encoder struct.

### 2. SdCfrSolver Modifications

**File:** `crates/deep-cfr/src/solver.rs`

Changes:
- `train()` accepts an optional pre-generated deal pool `Option<Vec<G::State>>`
  - If provided, samples from this pool each iteration instead of calling `game.initial_states()`
  - If None, falls back to current behavior (call `initial_states()` each iteration)
- `train()` accepts a checkpoint callback `Option<Box<dyn FnMut(u32, &TrainedSdCfr) -> Result<(), SdCfrError>>>`
  - Called every `checkpoint_interval` iterations with current iteration + trained state
  - The callback is responsible for strategy extraction and saving

### 3. Trainer sd-cfr Subcommand

**File:** `crates/trainer/src/main.rs` (new subcommand handler)

Steps:
1. Parse `training_sdcfr.yaml` into `SdCfrTrainingConfig`
2. Construct `HunlPostflop` with stratification:
   ```rust
   let game = HunlPostflop::new(postflop_config, None, deal_count)
       .with_stratification(min_deals_per_class, max_rejections);
   ```
3. Generate deal pool once: `let deals = game.initial_states();`
4. Construct `HunlStateEncoder` with `postflop_config.bet_sizes`
5. Build `SdCfrConfig` from YAML
6. Run `SdCfrSolver::train()` with deal pool + checkpoint callback
7. Checkpoint callback:
   - Load all nets via `ExplicitPolicy::from_buffer()`
   - Walk game tree to extract strategy as `FxHashMap<u64, Vec<f64>>`
   - Convert to `BlueprintStrategy::from_strategies()`
   - Save as `StrategyBundle` to `output_dir/checkpoint_iter_{N}/`
   - Update `output_dir/latest/` symlink

### 4. SdCfrConfig Extensions

**File:** `crates/deep-cfr/src/config.rs`

New fields:
```rust
pub checkpoint_interval: u32,  // Save every N iterations (0 = disabled)
```

### 5. YAML Config

```yaml
game:
  stack_depth: 25
  bet_sizes: [0.33, 0.67, 1.0, 2.0, 3.0]
  max_raises_per_street: 3

deals:
  count: 50000
  min_per_class: 100
  max_rejections: 500000
  seed: 42

training:
  solver: sd-cfr
  iterations: 1500
  traversals_per_iter: 1000
  seed: 42
  output_dir: "./sdcfr_25bb"

network:
  hidden_dim: 256
  num_actions: 8

sgd:
  steps: 500
  batch_size: 4096
  learning_rate: 0.001
  grad_clip_norm: 1.0

memory:
  advantage_cap: 5000000

checkpoint:
  interval: 100
```

## What's Reused (No Changes)

- `HunlPostflop` + `Game` trait + `PostflopState`
- `HunlPostflop::initial_states()` with stratification
- `card_features::canonicalize()` + `encode_bets()`
- `AdvantageNet`, `ReservoirBuffer`, `ModelBuffer`
- `ExplicitPolicy` for strategy extraction
- `BlueprintStrategy` + `StrategyBundle` for persistence
- Tauri exploration + simulation (loads bundle as-is)

## New Code

| Component | Location | ~Lines |
|-----------|----------|--------|
| `HunlStateEncoder` | `crates/deep-cfr/src/hunl_encoder.rs` | 50 |
| Trainer subcommand | `crates/trainer/src/main.rs` | 100 |
| YAML config parsing | `crates/trainer/src/main.rs` | 50 |
| Checkpoint callback | `crates/trainer/src/main.rs` | 40 |
| Config extensions | `crates/deep-cfr/src/config.rs` | 10 |

## Modified Code

| Component | Change |
|-----------|--------|
| `SdCfrSolver::train()` | Accept deal pool + checkpoint callback |
| `SdCfrConfig` | Add `checkpoint_interval` field |

## Strategy Extraction for Checkpoints

The checkpoint callback must walk the HUNL game tree to build the strategy map. This is expensive for full HUNL (millions of info sets), but only runs every N iterations:

1. Build `ExplicitPolicy` for each player from current `ModelBuffer`
2. For each deal in a representative sample of the deal pool:
   - Walk the game tree recursively
   - At each non-terminal node, encode features and query the appropriate player's policy
   - Insert `info_set_key -> action_probabilities` into the strategy map
3. Convert to `BlueprintStrategy` and save

## Open Questions

- **num_actions = 8**: With 5 bet sizes + fold/check/call + all-in, the max action count is 8 (`ArrayVec<Action, 8>`). The network outputs 8 values, sliced to the actual legal action count at each node. Verify this matches across all game states.
- **Action history encoding**: `encode_bets()` uses pot fractions, but `Action::Call` doesn't store a pot fraction. The encoder needs to compute `to_call / pot` from the state at the time of the call, which requires replaying the action sequence. Alternative: just use the action code (like the Kuhn encoder) and let the network learn the mapping.

## Testing

- Unit tests for `HunlStateEncoder` (encode known states, verify features)
- Integration test: train 10 iterations on small HUNL config, verify checkpoint is valid bundle
- Comparison test: load checkpoint in same code path Tauri uses, verify strategy lookup works
