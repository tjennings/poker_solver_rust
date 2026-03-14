# Solver Architecture

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

2. **Train** (`model`): Train a 7-layer MLP (2660->500->...->1326) using Huber loss + auxiliary game-value consistency loss. Framework: burn (wgpu/ndarray backends).

3. **Evaluate** (`eval`): Compute MAE, max error, and mbb/hand metrics on held-out data.

4. **Compare** (`eval`): Generate fresh river spots, solve exactly, compare network predictions against ground truth.

### Network Architecture

```
Input(2660) -> [Linear(500) -> BatchNorm -> PReLU] x 7 -> Linear(1326)
```

- Input: OOP range (1326) + IP range (1326) + board (5) + pot (1) + stack (1) + player (1)
- Output: 1326 pot-relative counterfactual values
- Loss: Huber (masked for board-blocked combos) + lambda x auxiliary game-value constraint
- ~3.5M parameters

### Integration Point

The CFVnet replaces exact river solving at the leaves of turn subgames during depth-limited re-solving. Given a turn decision point, the solver builds a turn subtree and queries the CFVnet at each river leaf instead of running full DCFR.

### Key Files

- Config: `crates/cfvnet/src/config.rs`
- Range generator: `crates/cfvnet/src/datagen/range_gen.rs`
- Situation sampler: `crates/cfvnet/src/datagen/sampler.rs`
- Solve wrapper: `crates/cfvnet/src/datagen/solver.rs`
- Network model: `crates/cfvnet/src/model/network.rs`
- Loss functions: `crates/cfvnet/src/model/loss.rs`
- Training loop: `crates/cfvnet/src/model/training.rs`
- CLI: `crates/cfvnet/src/main.rs`
- Sample config: `sample_configurations/river_cfvnet.yaml`

## Known Limitations

- **No real-time subgame solving yet:** The blueprint is a static strategy. Pluribus-style real-time search is planned but not implemented.
- **CFVnet not yet integrated:** The network is trainable but not yet wired into depth-limited solving at runtime.
