# Solver Architecture

## Overview

The solver finds a Nash equilibrium for heads-up no-limit Texas Hold'em by splitting the game into two phases: a **preflop solver** that uses Linear CFR (LCFR) over 169 canonical hand matchups, and an **abstracted postflop model** that provides expected values at preflop showdown boundaries. The postflop model uses imperfect-recall card abstraction (Pluribus-style) with per-flop hand bucketing and a shared tree template at a fixed SPR.

```
PreflopConfig ─► PreflopTree ─► PreflopSolver
                                      │
                                      ▼
                              PostflopAbstraction
                              ┌───────────────────────────┐
                              │ StreetBuckets.flop         │  ◄── flop bucket assignments (retained)
                              │ PostflopTree (shared)      │  ◄── single tree at fixed SPR
                              │ PostflopValues (per-flop)  │  ◄── solved EV table per flop
                              └───────────────────────────┘

Streaming pipeline (per canonical flop, in parallel):
  flop histograms → flop buckets → turn histograms → turn buckets
  → river equities → river buckets → equity tables → transitions
  → CFR solve → extract values → DROP intermediate data
```

## Project Structure

```
poker_solver_rust/
├── crates/
│   ├── core/                  # Core solver library
│   │   └── src/
│   │       ├── game/          # Game implementations (HUNL, Kuhn, LHE)
│   │       ├── cfr/           # CFR variants (vanilla, MCCFR, sequence-form)
│   │       ├── blueprint/     # Blueprint strategy, subgame solving, bundling
│   │       ├── preflop/       # Preflop LCFR solver & postflop abstraction pipeline
│   │       ├── abstraction/   # Card abstraction (EHS buckets, isomorphism)
│   │       ├── hand_class.rs  # 19-variant hand classification system
│   │       ├── info_key.rs    # Info set key encoding (64-bit packed)
│   │       ├── agent.rs       # Agent system (TOML-configured play styles)
│   │       ├── simulation.rs  # Arena-based agent simulation
│   │       └── abstract_game.rs  # Exhaustive deal enumeration
│   ├── trainer/               # CLI for training, diagnostics, deal generation
│   ├── deep-cfr/              # Neural network SD-CFR (candle backend)
│   ├── gpu-cfr/               # GPU sequence-form CFR (wgpu)
│   │   └── src/shaders/       # WGSL compute shaders
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

**`core`** — The heart of the solver. Contains the `Game` trait and its implementations (heads-up no-limit, Kuhn poker, limit hold'em), multiple CFR solver backends (vanilla, MCCFR, sequence-form), the preflop LCFR solver with its full postflop abstraction pipeline (hand bucketing, equity tables, tree building, solving), the blueprint strategy system with subgame refinement, card abstractions, hand classification, and the agent simulation framework.

**`trainer`** — CLI entry point that orchestrates training runs, bucket diagnostics, hand tracing, and deal generation. Parses YAML configs and drives the core library.

**`deep-cfr`** — Neural network approach to CFR using the candle ML framework. Encodes game states as card features, trains advantage networks via reservoir sampling, and supports both HUNL and limit hold'em.

**`gpu-cfr`** — GPU-accelerated sequence-form CFR using wgpu. Includes tabular and tiled solver variants with WGSL compute shaders for forward pass, backward pass, regret matching, and DCFR discounting.

**`tauri-app`** — Desktop application for exploring solved strategies. Loads blueprint bundles, navigates the game tree, queries strategy/EV at any node, and runs agent simulations.

**`devserver`** — Lightweight HTTP server that mirrors all Tauri exploration commands as POST endpoints, enabling browser-based UI development without the full Tauri build cycle.

**`frontend`** — React/TypeScript UI shared between the Tauri app and dev server. Provides the strategy explorer (tree browsing, action frequencies, EV display) and agent simulator. Auto-detects Tauri vs browser environment.

**`agents/`** — TOML files defining agent play styles (tight-aggressive, loose-aggressive, etc.) that map hand classes to action frequency distributions for simulation.

## Preflop Solver

**Algorithm:** Linear CFR (LCFR) with DCFR discounting. Both regret and strategy sums are weighted by iteration number. Simultaneous updates — both players traverse every iteration.

**Key types:**
- `PreflopConfig` — game structure: blinds, stacks, raise sizes, DCFR params (alpha/beta/gamma), exploration epsilon
- `PreflopTree` — arena-allocated game tree; nodes are Decision or Terminal (Fold/Showdown)
- `PreflopSolver` — flat buffer layout: `regret_sum[node_idx * 169 + hand_idx]`, `strategy_sum[...]`

**Flow:**
1. Build `PreflopTree` from config (raise sizes, raise cap)
2. Optionally build `PostflopAbstraction` (see below)
3. Run LCFR iterations with epsilon-greedy exploration
4. At showdown terminals: if postflop model exists, query it for EV; otherwise use raw preflop equity

**Exploration:** epsilon-greedy — `intended` strategy (for strategy sums) vs `traversal` strategy (explored) ensure sufficient visits to all actions.

**Files:**
- Config: `crates/core/src/preflop/config.rs`
- Tree: `crates/core/src/preflop/tree.rs`
- Solver: `crates/core/src/preflop/solver.rs`
- Equity: `crates/core/src/preflop/equity.rs`

## Postflop Abstraction Pipeline

The postflop model replaces raw equity evaluation at preflop showdown terminals with solved postflop EVs. It uses imperfect recall — each street clusters independently, and the player "forgets" prior-street bucket identity.

### Streaming Architecture

The pipeline streams per-flop: for each canonical flop in parallel, it computes all data (buckets, equity tables, transition matrices), runs CFR to convergence, extracts the value row, then drops all intermediate data. Only flop bucket assignments and the per-flop value row persist. This reduces peak memory from ~6 GB (materializing all flops simultaneously) to ~30 MB.

```
For each canonical flop (parallel):
  1. Compute flop histograms → k-means → flop_buckets
  2. Compute flop pairwise equity
  3. Enumerate 47 turn cards → turn histograms → k-means → turn_buckets
  4. Compute turn pairwise equity
  5. Sample 10 turns → enumerate river cards → river equities → k-means → river_buckets
  6. Compute river pairwise equity
  7. Compute flop→turn and turn→river transition matrices
  8. CFR solve using shared tree template + per-flop equity/transitions
  9. Extract value row: values[hero_pos][hero_bucket][opp_bucket] → EV
  10. DROP: turn/river buckets, equity tables, transitions, regret/strategy buffers
```

**Entry point:** `process_single_flop()` (steps 1-7) + `stream_solve_and_extract_one_flop()` (steps 8-10)

**Output retained at runtime:** `PostflopAbstraction` contains only flop bucket assignments (`StreetBuckets.flop`), the shared tree, and `PostflopValues`.

### Per-Flop Bucketing

Independent per-street k-means clustering on equity histogram CDF features:

**Flop:** For each canonical flop, cluster the 169 hands independently into N buckets. Each (hand, flop) pair's feature vector is the 10-bin equity CDF computed by enumerating 47 turn cards and computing equity vs uniform opponent. This gives each flop its own bucket assignments — "top pair on K72r" and "top pair on 987ss" get separate bucket IDs.

**Turn:** Enumerate 47 live turn cards. For each of 169 hands × 47 turns, compute equity histogram CDF over 46 river cards → k-means. Independent clustering per flop.

**River:** Sample 10 evenly-spaced turn cards, enumerate ~46 river cards each → ~460 five-card boards. Compute scalar equity per hand × board → k-means. Independent clustering per flop.

The CDF representation means L2 distance equals Earth Mover's Distance for 1D distributions.

**Files:**
- Histogram CDFs & canonical boards: `crates/core/src/preflop/ehs.rs`
- K-means & clustering pipeline: `crates/core/src/preflop/hand_buckets.rs`
- Single-flop streaming: `process_single_flop()` in `hand_buckets.rs`

### Equity Tables & Transition Matrices

Per-street `BucketEquity` tables: 2D array `[bucket_a][bucket_b] → f32` storing average equity when bucket A faces bucket B. Computed via true pairwise hand-vs-hand evaluation. Used as leaf-node estimates during postflop CFR.

Transition matrices connect streets: `flop_to_turn[flop_bucket][turn_bucket] → P(turn_bucket | flop_bucket)` and `turn_to_river[turn_bucket][river_bucket] → P(river_bucket | turn_bucket)`. Computed by counting (hand, board) assignments across adjacent streets and normalizing rows.

In streaming mode, equity tables and transition matrices are computed per-flop, used for the solve, then dropped. They are not retained in the final `PostflopAbstraction` struct.

### Postflop Tree

A single shared tree template at a fixed SPR (`postflop_spr`, default 5.0). All per-flop solves share this tree structure.

Each tree has three streets of Decision nodes (OOP=0, IP=1), Chance nodes at street transitions, and Terminals (Fold/Showdown). Bet sizes are pot fractions.

`PostflopLayout` maps `(node_idx, bucket)` to flat buffer offsets for regret/strategy storage.

### Postflop Solve

MCCFR per-flop with bucket-level abstraction:
- Embarrassingly parallel: one independent solve per canonical flop
- Each solve uses the shared tree template but its own per-flop equity and transition matrices
- Imperfect recall: each street's decision nodes use that street's independent per-flop bucket ID
- At chance nodes (street transitions): iterate over all new-street bucket pairs, weighted by transition probabilities `P(new_bucket | old_bucket)`. Reach probabilities are multiplied by transition weights for correct regret weighting
- Output: `PostflopValues` — flat 4D array `[flop_idx][hero_pos][hero_bucket][opp_bucket] → EV`

**File:** `crates/core/src/preflop/postflop_abstraction.rs`

### EV Rebucketing (Optional)

When `rebucket_rounds > 1`, an outer loop refines flop bucket assignments using strategy-dependent EV features instead of raw equity:

1. **Round 1 (EHS):** Standard streaming pipeline — cluster on equity histograms, solve all flops
2. **Rounds 2+:** Extract per-hand EV histograms from the converged strategy (distribution of EVs across opponent buckets), re-cluster flop hands on EV features, then stream-solve again (turn/river recomputed fresh per flop)
3. **Final round:** Use values from the last converged strategy

This captures hand value nuances that equity alone misses. Nut hands and second-nut hands have similar equity (~95%+) but divergent EV profiles — nuts extract value from strong-but-second-best hands, while second-best pays off in those spots. EV rebucketing separates them.

Each per-flop CFR solve uses **early stopping**: when the max absolute strategy delta between consecutive iterations drops below `cfr_delta_threshold` (default 0.001), the solve stops early. Otherwise it runs to `postflop_solve_iterations`.

With `rebucket_rounds: 1` (default), behavior is identical to the standard EHS-only pipeline.

**File:** `crates/core/src/preflop/postflop_abstraction.rs`, `crates/core/src/preflop/hand_buckets.rs`

### Runtime Integration

At each preflop showdown terminal:
1. Loop over all canonical flops
2. Per flop: look up hero and opponent bucket IDs via `StreetBuckets::flop_bucket_for_hand(hand_idx, flop_idx)`
3. Lookup: `PostflopValues::get_by_flop(flop_idx, hero_pos, hero_bucket, opp_bucket) → EV fraction`
4. Average across flops, then scale: `EV_fraction * actual_pot + (pot/2 - hero_investment)`

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

| Parameter | Default | Presets (fast/med/std/acc) | Description |
|-|-|-|-|
| `solve_type` | bucketed | — | Postflop backend: `bucketed` or `mccfr` |
| `num_hand_buckets_flop` | 20 | 10/15/20/30 | Per-flop k-means clusters (169 hands per flop) |
| `num_hand_buckets_turn` | 500 | 50/200/500/1000 | Per-flop turn k-means clusters (bucketed only) |
| `num_hand_buckets_river` | 500 | 50/200/500/1000 | Per-flop river k-means clusters (bucketed only) |
| `bet_sizes` | [0.5, 1.0] | — | Pot-fraction bet sizes |
| `max_raises_per_street` | 1 | — | Raise cap per postflop street |
| `postflop_sprs` | [3.5] | — | SPR(s) for shared postflop tree (scalar or list) |
| `postflop_solve_iterations` | 200 | — | CFR/MCCFR iterations per flop |
| `postflop_solve_samples` | 0 | — | Bucket pairs per iteration; 0 = all (bucketed only) |
| `max_flop_boards` | 0 | 10/200/0/0 | Max canonical flops for EHS features; 0 = all ~1,755 |
| `fixed_flops` | none | — | Explicit flop boards (overrides `max_flop_boards`) |
| `equity_rollout_fraction` | 1.0 | — | Fraction of runouts per hand pair; 1.0 = exact (bucketed only) |
| `rebucket_rounds` | 1 | — | EV rebucketing rounds (1 = EHS only, 2+ = EV rebucketing) |
| `cfr_delta_threshold` | 0.001 | — | Max strategy delta for early CFR stopping |
| `mccfr_sample_pct` | 0.01 | — | Fraction of deal space per MCCFR iteration (MCCFR only) |
| `value_extraction_samples` | 10,000 | — | Monte Carlo samples for EV extraction (MCCFR only) |

## Caching

| Cache | Key | Stores | Status |
|-|-|-|-|
| `solve_cache` | config hash + equity flag | `PostflopValues` | Active |
| `abstraction_cache` | config hash + equity flag | `StreetBuckets` + `StreetEquity` + `TransitionMatrices` | Disabled (pending StreetBuckets format migration) |

**Files:**
- `crates/core/src/preflop/abstraction_cache.rs`
- `crates/core/src/preflop/solve_cache.rs`


### MCCFR Postflop Backend

An alternative postflop solve backend selectable via `solve_type: mccfr` in config. Instead of abstract bucket transitions between streets, the MCCFR backend:

- Uses concrete card pairs (expanded from canonical hands) assigned to flop buckets
- Samples random (hero_hand, opp_hand, turn, river) deals per iteration
- Evaluates showdowns using `rank_hand()` on the actual 5-card board
- Uses only flop buckets (no turn/river bucket abstraction needed)

Key config fields:
- `solve_type`: `bucketed` (default) or `mccfr`
- `mccfr_sample_pct`: fraction of deal space sampled per iteration (default 0.01)
- `value_extraction_samples`: Monte Carlo samples for post-convergence EV extraction (default 10,000)

## Known Limitations

- **Preflop-only model limitation:** The preflop solver with postflop equity reference finds a limp-trap equilibrium (AA ~30% raise) rather than full-game GTO (AA 100% raise). This is inherent to the preflop-only model, not a bug.
- **No real-time subgame solving yet:** The postflop model is a static blueprint. Pluribus-style real-time search is planned but not implemented.
