# Preflop Solver Design

**Status:** Draft / Discussion
**Date:** 2026-02-12
**Goal:** Fast preflop solver for HU hold'em, extensible to 6-max/9-max. Near-GTO on consumer hardware.

## Why a separate preflop solver?

The existing MCCFR/sequence-form infrastructure solves full games (preflop through river). A dedicated preflop solver exploits the problem's special structure:

- **169 canonical hands** (suit isomorphism, no board)
- **Small game tree** (~1K-100K nodes depending on bet sizing granularity)
- **No street transitions** — single betting round
- **Equity is precomputable** — 169x169 matrix for HU, enumerable for multiway

This means **exact solutions in seconds** for HU, **minutes** for 6-max on consumer hardware.

## State of the art

### Commercial tools

- **PioSolver/GTO+** — HU preflop in <1s via CFR+, but 2-player only
- **MonkerSolver** — multiway via CFR, takes minutes-hours for 6-max preflop
- **SimplePreflop** — dedicated HU preflop tool, very fast

### Algorithmic landscape

- **CFR+** (Tammelin 2014) — fastest practical convergence for 2-player zero-sum games. Non-negative regrets accumulate faster than vanilla CFR.
- **DCFR** (Brown & Sandholm 2019) — discounted CFR, good general-purpose. Already used in this codebase.
- **Sequence-form LP** — gives *exact* Nash for 2-player in one shot (no iteration). Doesn't extend to 3+ players.
- **CFR for N>2 players** — converges to coarse correlated equilibrium (CCE), not Nash (Nash is PPAD-hard for 3+ players). In practice, CCE strategies are very close to Nash and much better than human play. This is what MonkerSolver does.

## Recommended algorithm: CFR+ with full enumeration

### Why CFR+ over sequence-form LP

- LP gives exact Nash for HU, but doesn't generalize to 3+ players
- CFR+ gives near-exact Nash for HU (epsilon < 0.001 BB in seconds) and naturally extends to N players
- Existing DCFR infrastructure to build on

### Why full enumeration over sampling

- Preflop game tree is small enough to traverse completely every iteration
- No sampling variance = faster convergence
- For HU: 169 hands × ~1K tree nodes = trivial
- For 6-max: 169 hands × 6 positions × ~100K tree nodes = still tractable

## Key design decisions

### 1. Game tree representation

```
PreflopTree {
    nodes: Vec<PreflopNode>,  // arena-allocated
    positions: u8,            // 2 for HU, 6 for 6-max, 9 for 9-max
    raise_cap: u8,            // max raises per round (typically 4-5)
    bet_sizes: Vec<Vec<f64>>, // sizes per raise depth
}

PreflopNode {
    kind: Decision { position, actions } | Terminal { payoffs },
    parent: u32,
    children: Vec<u32>,
}
```

The tree is built once from the blind/ante structure and bet sizing config. For HU with standard sizes (2.5x open, 3x 3bet, 2.5x 4bet, all-in), this is ~200 nodes. For 6-max it's ~10K-100K depending on sizing granularity.

### 2. Hand representation and card removal

For HU: straightforward 169x169 matchup matrix. Precompute all equities once (~28K entries).

For multiway: this is the main complexity jump. Card removal matters — if UTG has AKs, the probability of later players holding AA decreases. Two approaches:

- **Full combo enumeration**: iterate all C(52,2)=1326 combos per player, applying card removal. Exact but O(1326^N) per terminal node — only feasible for N<=3.
- **Canonical-hand weighting with card removal adjustments**: use 169 classes but weight by remaining combos. For example, if player 1 holds one Ace, AA for player 2 drops from 6 combos to 3. This is the standard approach — O(169^N) with correction factors.

### 3. Equity computation

For HU: precompute the 169x169 equity matrix via full enumeration of all boards (C(50,5) = 2.1M boards per matchup, but using precomputed lookup tables this takes ~1 minute total and is done once).

For multiway: need N-way equity (probability each player wins/ties). Can precompute for 3-way matchups (~169^3 = 4.8M entries), or sample for larger N. But preflop equities change subtly with card removal, so the correction factor approach is better.

### 4. Blind/ante structure

```rust
struct PreflopConfig {
    positions: Vec<Position>,      // ordered by action: UTG..BB
    blinds: Vec<(Position, u32)>,  // (position, amount)
    antes: Vec<(Position, u32)>,   // per-position antes
    stack_depth: u32,              // in BB
    raise_sizes: Vec<Vec<f64>>,    // sizes[raise_depth] = [2.5, 3.0, ...]
    raise_cap: u8,                 // max raises (typically 4-5)
}
```

This naturally handles HU (SB, BB), 6-max (UTG, HJ, CO, BTN, SB, BB), 9-max, and exotic structures like antes/straddles.

### 5. Info set key

Much simpler than the postflop key — no street, no SPR changes mid-hand:

```
position(4) | hand_index(8) | action_history(variable)
```

For 169 hands x ~1K action histories x 6 positions = ~1M info sets for 6-max. Tiny.

## Architecture

```
PreflopConfig -> PreflopTree (build once)
                     |
               CfrSolver<PreflopTree>
                     |
               PreflopStrategy { position -> hand -> action_history -> probs }
                     |
               RangeChart (169-grid display)
```

### Integration with existing code

A **new `PreflopGame` type** that implements the existing `Game` trait (or a simplified version of it), rather than reusing `HunlPostflop`. Reasons:

- Preflop game tree is static (no board-dependent branching)
- No need for `full_board`, street transitions, hand classification
- Can reuse the existing `SequenceCfrSolver` or write a simpler dedicated CFR+ loop
- The `DealInfo` concept maps cleanly (169x169 matchups with equity and combo-count weights for HU)

## Performance estimates

| Scenario | Info sets | Convergence (0.1% Nash distance) | Time estimate |
|----------|-----------|----------------------------------|---------------|
| HU, 3 bet sizes | ~5K | ~1K iterations | <1 second |
| HU, 5 bet sizes | ~15K | ~2K iterations | ~2 seconds |
| 6-max, 3 bet sizes | ~500K | ~10K iterations | ~1-5 minutes |
| 6-max, 5 bet sizes | ~2M | ~20K iterations | ~10-30 minutes |
| 9-max, 3 bet sizes | ~5M | ~50K iterations | ~1-3 hours |

All on a single consumer machine (8 cores, 32GB RAM). Memory is never the bottleneck — even 5M info sets x 5 actions x 8 bytes = 200 MB.

## Proposed implementation order

1. **HU preflop game + tree builder** — `PreflopGame` struct, configurable bet sizes, build tree
2. **Equity matrix** — precompute 169x169 HU equity table
3. **Wire to existing CFR** — use `SequenceCfrSolver` or write a simple CFR+ loop
4. **Range chart output** — display the familiar 13x13 grid per decision point
5. **Multiway extension** — generalize to N players, card removal weights, position-aware tree

Steps 1-4 are probably a day of work given the existing infrastructure. Step 5 is the real engineering challenge.

## Open questions

1. **Bet sizing granularity** — configurable sizes per raise depth, or fixed geometric scaling?
2. **Limp/complete option** — should SB be able to limp in HU? (Some solvers include this, it changes strategy significantly.)
3. **Multiway equity approach** — full combo enumeration (exact, slower) vs canonical-hand weighting with card removal (fast, ~0.1% approximation)?
4. **Output format** — just range grids, or also EV/frequency tables per node?
5. **Reuse `SequenceCfrSolver`** vs dedicated preflop CFR loop? The existing solver works but carries overhead from the general game tree structure.
