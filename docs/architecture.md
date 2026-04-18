# WarpGTO — Solver Architecture

## Overview

The solver finds a Nash equilibrium for heads-up no-limit Texas Hold'em using **blueprint_v2**, an MCCFR-based solver that operates over the full game tree with hand abstraction (HandClassV2 or EHS2 buckets). The solver samples random deals, traverses preflop through river, and accumulates regrets and strategy sums at each information set. A shared DCFR module (`cfr/dcfr.rs`) handles iteration weighting and regret discounting.

For single-spot analysis, a separate **range solver** provides exact (no-abstraction) DCFR solving with full 1326-combo granularity.

For depth-limited solving, the **cfvnet** pipeline trains neural networks to approximate river counterfactual values, enabling efficient subgame re-solving without full river traversal.

```
GameConfig → HunlPostflop (Game trait) → MccfrSolver → BlueprintV2Strategy
                    ↓
              random deals (full_board)
              HandClassV2 / EHS2 abstraction
```

## Project Structure

```
poker_solver_rust/
├── crates/
│   ├── core/                  # Core solver library
│   │   └── src/
│   │       ├── game/          # Action/Player types, HunlPostflop game
│   │       ├── cfr/           # CFR utilities (regret matching, shared DCFR/LCFR logic)
│   │       ├── blueprint_v2/  # MCCFR blueprint trainer, strategy storage, config
│   │       ├── blueprint_mp/  # N-player (2-8) MCCFR blueprint trainer
│   │       ├── abstraction/   # Card abstraction (isomorphism, EHS2, HandClassV2)
│   │       ├── hand_class.rs  # 19-variant hand classification system
│   │       ├── info_key.rs    # Info set key encoding (64-bit packed)
│   │       ├── agent.rs       # Agent system (TOML-configured play styles)
│   │       ├── simulation.rs  # Arena-based agent simulation
│   ├── range-solver/          # Exact postflop range solver (DCFR, PioSOLVER-compatible)
│   ├── range-solver-compare/  # Comparison harness vs b-inary/postflop-solver
│   ├── trainer/               # CLI for training, diagnostics
│   ├── tauri-app/             # Desktop strategy explorer (Tauri)
│   ├── devserver/             # HTTP mirror of Tauri API for browser debugging
│   ├── cfvnet/                # Deep CFV network: river datagen, training, evaluation
│   └── test-macros/           # #[timed_test] proc macro
├── frontend/                  # React/TypeScript explorer UI
│   └── src/
│       ├── Explorer.tsx       # Strategy browsing interface
│       ├── Simulator.tsx      # Agent simulation interface
│       └── invoke.ts          # Tauri/fetch abstraction layer
├── agents/                    # Agent personality configs (TOML)
├── sample_configurations/     # Training config presets (YAML)
├── docs/                      # Architecture, training, explorer, cloud docs
```

### Component Summary

**`core`** -- The heart of the solver. Contains the blueprint_v2 MCCFR solver (full game tree traversal with hand abstraction), shared CFR utilities (regret matching, DCFR/LCFR iteration weighting via `cfr/dcfr.rs`), card abstractions (HandClassV2, EHS2), hand classification, info set key encoding, and the agent simulation framework.

**`trainer`** -- CLI entry point that orchestrates blueprint training (`train-blueprint`), range solving, and diagnostics. Parses YAML configs and drives the core library.

**`tauri-app`** -- Desktop application for exploring solved strategies. Loads blueprint_v2 bundles, navigates the game tree, queries strategy/EV at any node, and runs agent simulations.

**`devserver`** -- Lightweight HTTP server that mirrors all Tauri exploration commands as POST endpoints, enabling browser-based UI development without the full Tauri build cycle.

**`frontend`** -- React/TypeScript UI shared between the Tauri app and dev server. Provides the strategy explorer (tree browsing, action frequencies, EV display) and agent simulator. Auto-detects Tauri vs browser environment.

**`range-solver`** -- Self-contained postflop solver that takes hero/villain ranges, bet sizes, and board cards, then solves to Nash equilibrium using Discounted CFR. Output-identical (exact f32 equality) to b-inary/postflop-solver. Supports PioSOLVER-compatible range syntax and bet size notation (pot-relative, previous-bet-relative, geometric, additive, all-in). Handles suit isomorphism for turn/river to skip redundant chance nodes.

**`range-solver-compare`** -- Test harness that generates random game configurations and verifies exact output identity between our range-solver and the original postflop-solver. Includes fast default tests (1000 river configs in ~21s) and slow soak tests for overnight validation.

**`cfvnet`** -- Neural network pipeline for learning river counterfactual values. Generates training data by solving random river situations with range-solver DCFR, trains a 7-layer MLP, and evaluates against exact solves. Enables depth-limited solving at river leaves.

**`agents/`** -- TOML files defining agent play styles (tight-aggressive, loose-aggressive, etc.) that map hand classes to action frequency distributions for simulation.

## Unit Convention

All internal values (pot, stacks, bet sizes, EVs) are in **chips**.

- 1 BB = 2 chips
- Config files use chips: `small_blind: 1`, `big_blind: 2`, `stack_depth: 200` (for 100 BB)
- Preflop action sizes use chip amounts with a `bb` suffix: `"5bb"` means raise to 5 chips (2.5 BB)
- Pot-fraction sizes (e.g. `0.67`) and multiplier sizes (e.g. `"3.0x"`) are unitless and unchanged
- Display to users converts chips to BB by dividing by 2, at the UI/CLI boundary only
- The range-solver (`TreeConfig`) also uses chips for `starting_pot` and `effective_stack`

## Blueprint V2 MCCFR Solver

**Algorithm:** External-sampling MCCFR with DCFR discounting. Samples random deals (hole cards + full board), traverses preflop through river, accumulates regrets at each information set. DCFR logic (iteration weighting, regret discounting, strategy discounting) is delegated to the shared `DcfrParams` module in `cfr/dcfr.rs`.

**Key types:**
- `GameConfig` -- game structure: blinds, stacks, bet sizes, abstraction mode, DCFR params
- `HunlPostflop` -- implements the `Game` trait; manages game tree traversal with pre-dealt boards
- `MccfrSolver` -- external-sampling MCCFR; flat buffer layout for regrets and strategy sums
- `BlueprintV2Strategy` -- serialized strategy extracted from solver; maps info set keys to action distributions

**Flow:**
1. Build `GameConfig` from YAML
2. Initialize `HunlPostflop` game with deal pool
3. Run MCCFR iterations with parallel batch processing
4. Extract `BlueprintV2Strategy` for exploration

**Abstractions:**
- `HandClassV2` -- 19-class hand classification with intra-class strength and equity binning (28-bit hand field)
- `PotentialAwareEmd` -- True Pluribus-style potential-aware bucket abstraction (see below)
- Info set keys encode: hand (28 bits) | street (2) | SPR (5) | reserved (5) | actions (24)

**Key files:**
- Game: `crates/core/src/game/hunl_postflop.rs`
- Config: `crates/core/src/blueprint_v2/config.rs`
- MCCFR: `crates/core/src/blueprint_v2/mccfr.rs`
- Storage: `crates/core/src/blueprint_v2/storage.rs`
- Trainer: `crates/core/src/blueprint_v2/trainer.rs`

### Potential-Aware Clustering Pipeline

The clustering pipeline computes card abstractions by running bottom-up from river to preflop. At each street, the feature vector for a (board, combo) pair is a distribution over the *next* street's bucket IDs — true potential-aware abstraction as described by Brown & Sandholm (Pluribus).

**Pipeline flow (all in memory, files written at the end):**

```
1. cluster_river()     → equity-based 1-D k-means           → river BucketFile
2. cluster_turn()      → histogram over river bucket IDs     → turn BucketFile
3. cluster_flop()      → histogram over turn bucket IDs      → flop BucketFile
4. cluster_preflop()   → histogram over flop bucket IDs      → preflop BucketFile
5. Write all 4 BucketFiles to disk
```

**Histogram construction (`build_bucket_histogram_u8`):** For each possible next-street card, extends the board, canonicalizes it via `canonical_key()`, looks up the board index in the previous street's `BucketFile` via `board_index_map()`, and increments the count for that combo's bucket ID. Returns raw u8 counts.

**Clustering:** Turn, flop, and preflop use weighted EMD (Earth Mover's Distance) k-means over these bucket-ID histograms (`kmeans_emd_weighted_u8`). River uses equity-based 1-D k-means.

**Variants:** Each street has three clustering variants:
- **Canonical** (`cluster_*_canonical`): exhaustive enumeration of isomorphic boards with combinatorial weights
- **Sampling** (`cluster_*`): samples from canonical boards with weights
- **With-boards** (`cluster_*_with_boards`): raw random board sampling for testing

All variants store canonical `PackedBoard` entries in the `BucketFile.boards` field (version 2 format) for downstream lookup.

**Diagnostics** (`cluster_diagnostics.rs`):
- `cross_street_transition_matrix` -- counts (board, combo) transitions between adjacent streets
- `centroid_emd_report` -- pairwise EMD between reconstructed bucket centroids
- `sample_hands_for_bucket` -- sample hands from a specific bucket for inspection

**Key files:**
- Pipeline: `crates/core/src/blueprint_v2/cluster_pipeline.rs`
- BucketFile: `crates/core/src/blueprint_v2/bucket_file.rs`
- K-means: `crates/core/src/blueprint_v2/clustering.rs`
- Diagnostics: `crates/core/src/blueprint_v2/cluster_diagnostics.rs`
- Config: `crates/core/src/blueprint_v2/config.rs` (`ClusteringConfig`)

### Heuristic V3 Bucketing

An alternative card abstraction that uses two deterministic axes instead of EMD clustering:

- **Nut distance** (configurable, default 6 bits / 64 bins): fraction of possible opponent holdings that beat the hero's hand on the current board. 0 = absolute nuts, 63 = pure air.
- **Equity delta** (configurable, default 4 bits / 16 bins): expected change in equity from current street to next. Positive = draws improving, negative = vulnerable made hands. Zero on river (no future cards).

Default configuration produces **1,024 buckets per street** (64 on river where delta collapses to midpoint). Buckets are precomputed per-flop and stored in the existing `PerFlopBucketFile` format.

Key advantages over EMD clustering:
- **Deterministic**: same inputs always produce same buckets (no k-means convergence)
- **Fast precomputation**: direct bin assignment, no iterative clustering
- **Interpretable**: each bucket maps to a (nut_distance, equity_delta) pair

Select via config:
```yaml
clustering:
  algorithm: heuristic_v3
  nut_distance_bits: 6
  equity_delta_bits: 4
```

## N-Player Blueprint (`blueprint_mp`)

A clean-room N-player (2-8) MCCFR solver module alongside the existing `blueprint_v2`. Uses strong domain types and supports configurable blind/ante structures.

### Module Structure

```
crates/core/src/blueprint_mp/
├── types.rs            # Domain types: Seat, PlayerSet, Chips, Bucket, Street, Deal
├── config.rs           # BlueprintMpConfig with lead/raise split, ForcedBet blinds
├── game_tree.rs        # N-player game tree with fold-continuation
├── info_key.rs         # 128-bit InfoKey (seat + bucket + street + SPR + 22 actions)
├── terminal.rs         # Side pot resolution, showdown, fold payoffs
├── storage.rs          # Flat-buffer atomic regret/strategy storage
├── mccfr.rs            # External-sampling MCCFR traversal (Pluribus-style)
├── trainer.rs          # Training loop with per-seat traverser cycling, DCFR
└── exploitability.rs   # Per-seat best-response diagnostic
```

### Key Design Decisions

- **2-8 players** with `MAX_PLAYERS = 8`
- **Configurable blinds**: SB, BB, ante, BB-ante, straddle via per-seat config
- **Lead/raise split**: Separate bet sizes for opening bets vs raises
- **Full side pot resolution** at showdown terminals
- **128-bit info set keys** with 22 action slots (panics on overflow)
- **Pre-allocated storage** (lazy allocation planned as future optimization)
- **Pluribus-style strategy averaging** (simple, biased for N>2 but empirically sufficient)
- Shares `abstraction/`, `cfr/`, and `hand_eval` with `blueprint_v2`

## Range Solver (Exact Postflop Solver)

A self-contained postflop solver that computes Nash equilibrium strategies for specific hero/villain ranges on a given board. Unlike the blueprint solver (which uses hand abstraction), the range solver works with concrete hand combinations and produces exact strategies.

**Algorithm:** Discounted CFR (DCFR) with a=1.5, b=0.5, g=3.0. Strategy resets at power-of-4 iterations (4, 16, 64, ...).

**Key features:**
- PioSOLVER-compatible range syntax (AA, AKs, QQ-88, TT+, weights)
- Bet size notation: pot-relative (50%), previous-bet-relative (2.5x), geometric (2e), additive (100c), all-in (a)
- Suit isomorphism detection on turn/river to skip redundant chance nodes
- Arena-allocated game tree with `MutexLike` for lock-free interior mutability
- Two-pass O(n) terminal evaluation using sorted hand strength arrays

**CLI:** `cargo run -p poker-solver-trainer --release -- range-solve` -- see `docs/training.md` for full usage.

**Files:**
- `crates/range-solver/src/` -- solver, action tree, game tree, evaluation, isomorphism, hand evaluator
- `crates/range-solver-compare/` -- comparison harness and identity tests

## CFVnet (Deep Counterfactual Value Network)

A neural network pipeline for learning river-street counterfactual values, enabling depth-limited solving without computing full river subtrees at runtime.

**Crate:** `crates/cfvnet`

### Pipeline

```
generate -> train -> evaluate -> compare
```

1. **Generate** (`datagen`): Sample random river situations (board, pot, stack, ranges via DeepStack R(S,p)), solve each with range-solver DCFR, extract pot-relative CFVs for both players. Output: binary file of training records.

2. **Train** (`model`): Train a 7-layer MLP (2720->500->...->1326) using Huber loss + auxiliary game-value consistency loss. Framework: burn (wgpu/ndarray backends).

3. **Evaluate** (`eval`): Compute MAE, max error, and mbb/hand metrics on held-out data.

4. **Compare** (`eval`): Generate fresh river spots, solve exactly, compare network predictions against ground truth.

### Network Architecture

Two model variants share the same MLP architecture (`HiddenBlock`: Linear -> BatchNorm -> PReLU):

**CfvNet** (pot-relative output):
```
Input(2720) -> [Linear(500) -> BatchNorm -> PReLU] x 7 -> Linear(1326)
```
- Input: OOP range (1326) + IP range (1326) + board one-hot (52) + rank presence (13) + pot/400 + stack/400 + player
- Output: 1326 pot-relative counterfactual values

**BoundaryNet** (normalized EV output):
```
Input(2720) -> [Linear(500) -> BatchNorm -> PReLU] x 7 -> Linear(1326)
```
- Input: same layout as CfvNet, but pot and stack encoded as `pot/(pot+stack)` and `stack/(pot+stack)`
- Output: 1326 normalized EVs (`chip_ev / (pot + effective_stack)`)
- At inference: `chip_ev[h] = normalized_ev[h] * (pot + effective_stack)`

Both use: Huber loss (masked for board-blocked combos) + lambda x auxiliary game-value constraint. ~2.9M parameters (default 7x500).

### Integration Point

**CfvNet** provides standalone river value predictions for evaluation and comparison.

**BoundaryNet** is wired into the range-solver as a depth-boundary evaluator via `NeuralBoundaryEvaluator`. When solving turn subgames, the solver queries BoundaryNet at river boundary nodes instead of full-depth DCFR, enabling fast turn solving. The Tauri explorer supports loading a trained BoundaryNet model via `boundary_model_path` config.

### Key Files

- Config: `crates/cfvnet/src/config.rs`
- Range generator: `crates/cfvnet/src/datagen/range_gen.rs`
- Situation sampler: `crates/cfvnet/src/datagen/sampler.rs`
- Solve wrapper: `crates/cfvnet/src/datagen/solver.rs`
- CfvNet model: `crates/cfvnet/src/model/network.rs`
- BoundaryNet model: `crates/cfvnet/src/model/boundary_net.rs`
- BoundaryNet dataset encoding: `crates/cfvnet/src/model/boundary_dataset.rs`
- BoundaryNet training: `crates/cfvnet/src/model/boundary_training.rs`
- Loss functions: `crates/cfvnet/src/model/loss.rs`
- Training loop: `crates/cfvnet/src/model/training.rs`
- Boundary evaluator (range-solver integration): `crates/cfvnet/src/eval/boundary_evaluator.rs`
- CLI: `crates/cfvnet/src/main.rs`
- Sample config: `sample_configurations/river_cfvnet.yaml`

## Sampled Rollout Evaluator

The default boundary evaluator for subgame re-solving is a **depth-gated MCCFR sampling rollout**. When the range-solver hits a depth boundary (e.g., at river during a turn solve), it queries the blueprint strategy to estimate continuation values for each hero combo.

**Hybrid algorithm:** The evaluator uses exhaustive enumeration at shallow decision depths and Monte Carlo sampling at deeper ones. At decision depth < `enumerate_decision_depth` (default 2), all children are enumerated exactly weighted by the biased blueprint strategy. At decision depth >= the threshold, a single action is sampled from the biased strategy distribution and recursed into. Chance nodes always sample `num_rollouts` random cards (with a 3x sample boost at the first two chance levels for variance reduction).

This follows the approach described in **Modicum** (Brown, Sandholm & Amos, NeurIPS 2018): the first 1-2 decision levels carry the most entropy and have low branching cost, so exhaustive enumeration there preserves accuracy; deeper levels contribute geometrically less to the final value, making sampling sufficient. The stochastic noise at deeper levels is absorbed by DCFR's across-iteration averaging in the outer solver -- the same convergence property that Libratus, Pluribus, and Modicum rely on.

**Configurable knobs** (tunable via Tauri settings or CLI flags):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rollout_enumerate_depth` | 2 | Decision levels to enumerate before sampling. Set to 255 for fully exhaustive rollouts (old behavior). |
| `rollout_opponent_samples` | 8 | Opponent hands sampled per hero combo. Higher = less variance, slower. |
| `rollout_num_samples` | 3 | Chance-node samples (random runout cards) per evaluation. |

**Performance:** The sampling pivot yields ~100-200x speedup over exhaustive enumeration (e.g., 50ms vs 8.2s per evaluator call on a 1176-combo flop scenario) with < 1 mbb/hand mean error, validated by the `validate-rollout` CLI harness.

**Key files:**
- Rollout logic: `crates/core/src/blueprint_v2/continuation.rs`
- Evaluator construction: `crates/tauri-app/src/postflop.rs` (`build_rollout_evaluator`)
- Bench/validate CLI: `crates/trainer/src/bench_rollout.rs`, `crates/trainer/src/validate_rollout.rs`

## Known Limitations

- **No real-time subgame solving yet:** The blueprint is a static strategy. Pluribus-style real-time search is planned but not implemented.
