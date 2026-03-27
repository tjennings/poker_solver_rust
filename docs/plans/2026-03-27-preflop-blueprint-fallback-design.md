# Preflop Blueprint Fallback Design

**Date:** 2026-03-27
**Status:** Approved

## Problem

The range-solver is postflop only (`PostFlopGame` handles flop/turn/river). When the self-play loop or orchestration tries to solve a preflop PBS (board.len() == 0), `solve_depth_limited_pbs()` errors. Preflop needs special handling.

## Decision

Use the blueprint's preflop strategy instead of subgame solving. The blueprint is already exact at 169 preflop buckets (no abstraction loss), so there's no quality gap. Preflop action influences postflop play through the reach probabilities it produces — this is already captured because we play preflop under blueprint policy before entering flop.

## Design

### New function: `play_preflop_under_blueprint()`

Location: `crates/rebel/src/blueprint_sampler.rs`

Extracts the preflop portion of the existing `traverse()` logic. Walks the blueprint game tree from root through preflop Decision nodes, sampling actions and updating reach probabilities for all 1326 combos. Stops at the first Chance node (preflop→flop transition).

```rust
pub struct PreFlopResult {
    pub reach_probs: Box<[[f32; 1326]; 2]>,
    pub pot: i32,
    pub effective_stack: i32,
}

pub fn play_preflop_under_blueprint<R: Rng>(
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    buckets: &AllBuckets,
    deal: &Deal,
    initial_stack: i32,
    small_blind: i32,
    big_blind: i32,
    rng: &mut R,
) -> PreFlopResult
```

### Self-play changes

`play_self_play_hand()` and `self_play_training_loop()` gain `strategy`, `tree`, `buckets` parameters. The hand loop:

1. Call `play_preflop_under_blueprint()` — get reach/pot/stack entering flop
2. Loop over `[Flop, Turn, River]` only (remove Preflop from STREET_ORDER)
3. Use preflop-updated reach/pot/stack as initial state

No preflop training examples are recorded.

### Orchestration changes

Remove `Preflop` from `STREET_ORDER` in `orchestration.rs`. The offline seeding pipeline becomes: River → Turn → Flop (3 streets, not 4).

### No changes needed

- `solver.rs` — already handles flop/turn/river correctly
- `leaf_evaluator.rs` — unchanged
- `data_buffer.rs` — unchanged
- `generate.rs` — unchanged (blueprint sampler already produces flop/turn/river PBSs)
