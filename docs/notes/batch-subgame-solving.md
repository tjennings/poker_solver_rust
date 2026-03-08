# Batch Subgame Solving for Training Tool

**Date:** 2026-03-07
**Context:** Architecture note for pre-computing fine-grained postflop strategies using the existing range-solver, fed by blueprint data.

## Motivation

We're building a solver for humans to train with, not a poker bot. Users need to see full 13x13 hand matrices for every spot. There's no "solve it when you get there" — the blueprint IS the product.

Blueprint bucket count scales linearly with convergence time (2x buckets = 2x iterations). At 100 buckets / 11.4M iterations, strategy delta is ~0.0011. Going to 400 buckets would need ~45.6M iterations for the same convergence — expensive for diminishing returns on hand resolution.

Better approach: moderate blueprint (100-200 buckets) + batch offline subgame solving at full precision.

## Architecture

```
Blueprint (coarse, 100-200 buckets, MCCFR)
    |
    v
Batch subgame solver (exact, no abstraction, DCFR)
    |
    v
Per-board strategy tables cached to disk
    |
    v
13x13 hand grids rendered in UI
```

This is how GTO Wizard and similar tools work — pre-solve thousands of spots, serve cached results.

## Integration with range-solver

The existing `range-solver` crate already has everything needed:

- **`initial_weights(player)`** — opponent's range at subgame root (from blueprint reach probabilities)
- **`CardConfig { range: [oop, ip], flop, turn, river }`** — specific board + starting ranges
- **`evaluate()`** — exact equity on the specific board (no abstraction)
- **`ActionTree`** — configurable bet sizes (can use finer sizes than blueprint)
- **DCFR solver** — production quality with alternating updates, regret matching, discounting, compression, suit isomorphism

### Pipeline per subgame

```
For each canonical board (or sampled board):
  1. Walk blueprint to get opponent ranges at the subgame root
  2. Build CardConfig with specific board + blueprint ranges
  3. Build ActionTree with fine bet sizes
  4. PostFlopGame::with_config(card_config, tree)
  5. solve(&mut game, 1000, target_expl, false)
  6. Extract strategy -> 13x13 grid per decision node
  7. Cache to disk
```

### Key properties

- Each subgame is small (~10^3-10^5 info sets depending on street)
- ~1000 DCFR iterations to converge (tree fits in L2/L3 cache)
- River subgame: milliseconds. Turn: <1 second. Flop: 1-10 seconds.
- Subgames are embarrassingly parallel — solve all boards concurrently

## What the blueprint provides

Two critical inputs for each subgame:

1. **Opponent ranges at the subgame root** — "what hands does my opponent arrive here with, and with what probability?" Computed by walking the blueprint strategy down the action history, accumulating reach probabilities.

2. **Leaf values (for depth-limited solving)** — if solving only one street ahead (e.g., flop only), the blueprint provides continuation values for what happens after the betting round ends. Not needed if solving all the way to showdown (river).

## Integration work needed

1. **Range extraction from blueprint** — walk blueprint strategy tree, compute P(hand | action_history) for both players. This is the main new code.
2. **Strategy-to-grid mapping** — extract solved strategies from PostFlopGame nodes, map back to 13x13 hand grid format for display.
3. **Batch orchestration** — enumerate canonical boards/spots, dispatch solves in parallel, cache results.
4. **Storage format** — serialize per-spot solutions to disk for fast lookup.

## Convergence scaling reference

| Buckets | Iterations for delta ~0.0011 | Wall time multiplier |
|-|-|-|
| 100 | 11.4M | 1x |
| 200 | ~22.8M | ~2x |
| 400 | ~45.6M | ~4x |

Scaling is linear because: 2x buckets = 2x info sets, each visited half as often per iteration, need 2x iterations to compensate. Per-iteration cost stays ~constant.

## Pluribus comparison

Pluribus used depth-limited real-time subgame solving (1-4 actions ahead, all opponents assumed to play blueprint). We don't need real-time — we can solve deeper (full street or multi-street) and cache everything offline. The tradeoff shifts from "fast enough for real-time" to "pre-compute as much as possible."

## Key files

- Range solver: `crates/range-solver/src/solver.rs`
- Game interface: `crates/range-solver/src/interface.rs`
- Action tree: `crates/range-solver/src/action_tree.rs`
- Card config: `crates/range-solver/src/card.rs`
- Blueprint strategy: `crates/core/src/blueprint_v2/storage.rs`
- Blueprint trainer: `crates/core/src/blueprint_v2/trainer.rs`
- Subgame architecture reference: `docs/plans/2026-02-12-blueprint-subgame-architecture.md`
