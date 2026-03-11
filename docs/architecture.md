# Solver Architecture

## Overview

The solver finds a Nash equilibrium for heads-up no-limit Texas Hold'em by splitting the game into two phases: a **preflop solver** that uses Linear CFR (LCFR) over 169 canonical hand matchups, and an **abstracted postflop model** that provides expected values at preflop showdown boundaries. The postflop model uses direct 169-hand indexing per flop -- each canonical hand is its own info set, with no bucket clustering. Two backends are available: **MCCFR** (sampled concrete hands with real showdown eval) and **Exhaustive** (pre-computed equity tables with configurable CFR variant). Both solvers share a common DCFR module (`cfr/dcfr.rs`) for iteration weighting and regret discounting.

```
PreflopConfig -> PreflopTree -> PreflopSolver
                                      |
                                      v
                              Vec<PostflopAbstraction>  (one per configured SPR)
                              +----------------------------+
                              | ComboMap (per flop)         |  <-- 169 hands -> concrete card pairs
                              | PostflopTree (shared)       |  <-- single tree at this model's SPR
                              | PostflopValues (per-flop)   |  <-- solved EV table per flop
                              +----------------------------+

Per SPR in postflop_sprs:
  Per canonical flop (in parallel):
    combo map -> MCCFR solve (or exhaustive CFR) -> extract values -> DROP intermediate data
```

## Project Structure

```
poker_solver_rust/
├── crates/
│   ├── core/                  # Core solver library
│   │   └── src/
│   │       ├── game/          # Action/Player types, PostflopConfig
│   │       ├── cfr/           # CFR utilities (regret matching, shared DCFR/LCFR logic)
│   │       ├── blueprint/     # Blueprint strategy, subgame solving, bundling
│   │       ├── preflop/       # Preflop LCFR solver & postflop abstraction pipeline
│   │       ├── abstraction/   # Card abstraction (isomorphism)
│   │       ├── hand_class.rs  # 19-variant hand classification system
│   │       ├── info_key.rs    # Info set key encoding (64-bit packed)
│   │       ├── agent.rs       # Agent system (TOML-configured play styles)
│   │       ├── simulation.rs  # Arena-based agent simulation
│   ├── range-solver/          # Exact postflop range solver (DCFR, PioSOLVER-compatible)
│   ├── range-solver-compare/  # Comparison harness vs b-inary/postflop-solver
│   ├── trainer/               # CLI for training, diagnostics
│   ├── tauri-app/             # Desktop strategy explorer (Tauri)
│   ├── devserver/             # HTTP mirror of Tauri API for browser debugging
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

**`core`** -- The heart of the solver. Contains the preflop LCFR solver with its postflop abstraction pipeline (169-hand direct indexing, combo maps, tree building, solving), shared CFR utilities (regret matching, DCFR/LCFR iteration weighting and discounting via `cfr/dcfr.rs`), the blueprint strategy system with subgame refinement, card abstractions, hand classification, and the agent simulation framework.

**`trainer`** -- CLI entry point that orchestrates preflop/postflop solving, diagnostics, and hand tracing. Parses YAML configs and drives the core library.

**`tauri-app`** -- Desktop application for exploring solved strategies. Loads blueprint bundles, navigates the game tree, queries strategy/EV at any node, and runs agent simulations.

**`devserver`** -- Lightweight HTTP server that mirrors all Tauri exploration commands as POST endpoints, enabling browser-based UI development without the full Tauri build cycle.

**`frontend`** -- React/TypeScript UI shared between the Tauri app and dev server. Provides the strategy explorer (tree browsing, action frequencies, EV display) and agent simulator. Auto-detects Tauri vs browser environment.

**`range-solver`** -- Self-contained postflop range solver that takes hero/villain ranges, bet sizes, and board cards, then solves to Nash equilibrium using Discounted CFR. Output-identical (exact f32 equality) to b-inary/postflop-solver. Supports PioSOLVER-compatible range syntax and bet size notation (pot-relative, previous-bet-relative, geometric, additive, all-in). Handles suit isomorphism for turn/river to skip redundant chance nodes.

**`range-solver-compare`** -- Test harness that generates random game configurations and verifies exact output identity between our range-solver and the original postflop-solver. Includes fast default tests (1000 river configs in ~21s) and slow soak tests for overnight validation.

**`agents/`** -- TOML files defining agent play styles (tight-aggressive, loose-aggressive, etc.) that map hand classes to action frequency distributions for simulation.

## Preflop Solver

**Algorithm:** Linear CFR (LCFR) with DCFR discounting. Both regret and strategy sums are weighted by iteration number. Simultaneous updates -- both players traverse every iteration. DCFR logic (iteration weighting, regret discounting, strategy discounting, CFR+ flooring) is delegated to the shared `DcfrParams` module in `cfr/dcfr.rs`.

**Key types:**
- `PreflopConfig` -- game structure: blinds, stacks, raise sizes, DCFR params (alpha/beta/gamma), exploration epsilon
- `PreflopTree` -- arena-allocated game tree; nodes are Decision or Terminal (Fold/Showdown)
- `PreflopSolver` -- flat buffer layout: `regret_sum[node_idx * 169 + hand_idx]`, `strategy_sum[...]`; holds a `DcfrParams` instance for all discounting operations

**Flow:**
1. Build `PreflopTree` from config (raise sizes, raise cap)
2. Optionally build one `PostflopAbstraction` per configured SPR (see below)
3. Run LCFR iterations with epsilon-greedy exploration
4. At showdown terminals: if postflop models exist, select the closest SPR model and query it for EV; otherwise use raw preflop equity

**Exploration:** epsilon-greedy -- `intended` strategy (for strategy sums) vs `traversal` strategy (explored) ensure sufficient visits to all actions.

**Files:**
- Config: `crates/core/src/preflop/config.rs`
- Tree: `crates/core/src/preflop/tree.rs`
- Solver: `crates/core/src/preflop/solver.rs`
- Equity: `crates/core/src/preflop/equity.rs`

## Postflop Abstraction Pipeline

The postflop model replaces raw equity evaluation at preflop showdown terminals with solved postflop EVs. It uses direct 169-hand indexing -- each canonical hand is its own info set per flop, with no bucket clustering.

### 169-Hand Direct Indexing

For each canonical flop, a **combo map** expands the 169 canonical hands into concrete (hero, opponent) card pairs that are compatible with the flop. Each hand index (0-168) corresponds to a canonical hand (AA, AKs, AKo, ..., 32s, 32o). The combo map handles card removal (dead cards blocked by the flop) and tracks the count of concrete combos per canonical hand for proper weighting.

This replaces the prior EHS bucket-based approach. Instead of clustering hands into buckets via k-means on equity histograms, each of the 169 hands is its own "bucket". This eliminates information loss from clustering and simplifies the pipeline.

### Streaming Architecture

The pipeline streams per-flop: for each canonical flop in parallel, it builds the combo map, runs CFR to convergence, extracts the value row, then drops all intermediate data. Only the per-flop value rows persist.

```
For each canonical flop (parallel):
  1. Build combo map: 169 hands -> concrete card pairs
  2. Build shared postflop tree at fixed SPR
  3. CFR solve (MCCFR or Exhaustive) using 169-hand indexing
  4. Extract value row: values[hero_pos][hero_hand][opp_hand] -> EV
  5. DROP: regret/strategy buffers, combo map
```

**Output retained at runtime:** `PostflopAbstraction` contains the shared tree and `PostflopValues`.

### MCCFR Backend

The default backend (`solve_type: mccfr`). For each canonical flop:
- Builds a combo map expanding 169 canonical hands to concrete card pairs
- Samples random (hero_hand, opp_hand, turn, river) deals per iteration
- Evaluates showdowns using `rank_hand()` on the actual 5-card board
- Uses `mccfr_sample_pct` to control sampling density (default 1% of deal space)
- Post-convergence: extracts per-hand EVs via Monte Carlo sampling (`value_extraction_samples`)

Embarrassingly parallel: one independent solve per canonical flop.

### Exhaustive Backend

Alternative backend (`solve_type: exhaustive`). For each canonical flop:
- Pre-computes pairwise equity tables for all 169x169 hand pairs per street
- Runs CFR with configurable variant (default: LCFR linear iteration weighting) using the shared `DcfrParams` module
- Supports all four variants: Vanilla, DCFR, CFR+, Linear (LCFR)
- Slower per iteration but converges with fewer iterations and no sampling variance

### Postflop Tree

Each `PostflopAbstraction` has its own tree template at a fixed SPR. All per-flop solves within that model share the same tree structure.

Each tree has three streets of Decision nodes (OOP=0, IP=1), Chance nodes at street transitions, and Terminals (Fold/Showdown). Bet sizes are pot fractions.

`PostflopLayout` maps `(node_idx, hand_idx)` to flat buffer offsets for regret/strategy storage.

### Multi-SPR Model Selection

When multiple SPRs are configured (e.g. `postflop_sprs: [2, 6, 20]`), a separate `PostflopAbstraction` is built for each. At runtime, the preflop solver selects the closest model by absolute SPR distance. No interpolation between models -- the existing within-model interpolation (`ratio = actual_spr / model_spr`) handles gaps between the actual and selected model SPR.

Training time scales linearly with the number of SPR values (each builds its own tree and solves all flops independently).

### Runtime Integration

At each preflop showdown terminal:
1. Compute `actual_spr = effective_remaining / pot`
2. Select the closest `PostflopAbstraction` by `|model.spr - actual_spr|`
3. Loop over all canonical flops
4. Per flop: look up hero and opponent hand indices (0-168)
5. Lookup: `PostflopValues::get_by_flop(flop_idx, hero_pos, hero_hand, opp_hand) -> EV fraction`
6. Average across flops, then scale: `EV_fraction * actual_pot + (pot/2 - hero_investment)`

**Special cases:** Limped pots (raise_count=0) fall back to raw equity instead of the postflop model.

**File:** `crates/core/src/preflop/postflop_abstraction.rs`

## Key Control Parameters

### Preflop (`PreflopConfig`)

| Parameter | Default | Description |
|-|-|-|
| `stack_depth` | 100 | Stack size in BB |
| `open_raise` | 2.5 | Open raise multiplier |
| `three_bet` | 3.0 | 3-bet multiplier |
| `raise_cap` | 4 | Max raises per street |
| `alpha` | 1.5 | DCFR positive regret weight |
| `beta` | 0.5 | DCFR negative regret weight |
| `gamma` | 2.0 | DCFR strategy weight |
| `exploration` | 0.05 | Epsilon-greedy exploration rate |
| `iterations` | 10000 | LCFR iterations |

### Postflop (`PostflopModelConfig`)

| Parameter | Default | Description |
|-|-|-|
| `solve_type` | mccfr | Backend: `mccfr` (sampled concrete hands) or `exhaustive` (equity tables + configurable CFR) |
| `bet_sizes` | [0.5, 1.0] | Pot-fraction bet sizes |
| `max_raises_per_street` | 1 | Raise cap per postflop street |
| `postflop_sprs` | [3.5] | SPR(s) for shared postflop tree (scalar or list) |
| `postflop_solve_iterations` | 500 | CFR/MCCFR iterations per flop |
| `max_flop_boards` | 0 | Max canonical flops; 0 = all ~1,755 |
| `fixed_flops` | none | Explicit flop boards (overrides `max_flop_boards`) |
| `cfr_convergence_threshold` | 0.01 | Per-flop early stopping threshold |
| `mccfr_sample_pct` | 0.01 | Fraction of deal space per MCCFR iteration (MCCFR only) |
| `value_extraction_samples` | 10,000 | Monte Carlo samples for EV extraction (MCCFR only) |
| `cfr_variant` | linear | CFR variant for postflop solver: `vanilla`, `dcfr`, `cfrplus`, `linear` |
| `ev_convergence_threshold` | 0.001 | Early-stop EV estimation threshold (MCCFR only) |

## Caching & Bundles

| Cache | Key | Stores | Status |
|-|-|-|-|
| `PostflopBundle` | directory path | `PostflopModelConfig` + per-SPR `PostflopValues` + flops | Active |

**Files:**
- `crates/core/src/preflop/postflop_bundle.rs`

**Postflop bundles:** Build a postflop abstraction once with `solve-postflop`, then reference the bundle directory via `postflop_model_path` in training configs to skip the expensive rebuild.

Multi-SPR bundles store one subdirectory per SPR (e.g. `spr_2.0/`, `spr_6.0/`, `spr_20.0/`). Legacy single-SPR bundles (flat `solve.bin`) are loaded with backward compatibility.

## Range Solver (Exact Postflop Solver)

A self-contained postflop solver that computes Nash equilibrium strategies for specific hero/villain ranges on a given board. Unlike the abstracted postflop model above (which uses 169-hand indexing for the preflop solver), the range solver works with concrete hand combinations and produces exact strategies.

**Algorithm:** Discounted CFR (DCFR) with α=1.5, β=0.5, γ=3.0. Strategy resets at power-of-4 iterations (4, 16, 64, ...).

**Key features:**
- PioSOLVER-compatible range syntax (AA, AKs, QQ-88, TT+, weights)
- Bet size notation: pot-relative (50%), previous-bet-relative (2.5x), geometric (2e), additive (100c), all-in (a)
- Suit isomorphism detection on turn/river to skip redundant chance nodes
- Arena-allocated game tree with `MutexLike` for lock-free interior mutability
- Two-pass O(n) terminal evaluation using sorted hand strength arrays

**CLI:** `cargo run -p poker-solver-trainer --release -- range-solve` — see `docs/training.md` for full usage.

**Files:**
- `crates/range-solver/src/` — solver, action tree, game tree, evaluation, isomorphism, hand evaluator
- `crates/range-solver-compare/` — comparison harness and identity tests

## Known Limitations

- **Preflop-only model limitation:** The preflop solver with postflop equity reference finds a limp-trap equilibrium (AA ~30% raise) rather than full-game GTO (AA 100% raise). This is inherent to the preflop-only model, not a bug.
- **No real-time subgame solving yet:** The postflop model is a static blueprint. Pluribus-style real-time search is planned but not implemented.

## CFVnet (Deep Counterfactual Value Network)

A neural network pipeline for learning river-street counterfactual values, enabling depth-limited solving without computing full river subtrees at runtime.

**Crate:** `crates/cfvnet`

### Pipeline

```
generate → train → evaluate → compare
```

1. **Generate** (`datagen`): Sample random river situations (board, pot, stack, ranges via DeepStack R(S,p)), solve each with range-solver DCFR, extract pot-relative CFVs for both players. Output: binary file of training records.

2. **Train** (`model`): Train a 7-layer MLP (2660→500→...→1326) using Huber loss + auxiliary game-value consistency loss. Framework: burn (wgpu/ndarray backends).

3. **Evaluate** (`eval`): Compute MAE, max error, and mbb/hand metrics on held-out data.

4. **Compare** (`eval`): Generate fresh river spots, solve exactly, compare network predictions against ground truth.

### Network Architecture

```
Input(2660) → [Linear(500) → BatchNorm → PReLU] × 7 → Linear(1326)
```

- Input: OOP range (1326) + IP range (1326) + board (5) + pot (1) + stack (1) + player (1)
- Output: 1326 pot-relative counterfactual values
- Loss: Huber (masked for board-blocked combos) + λ × auxiliary game-value constraint
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
