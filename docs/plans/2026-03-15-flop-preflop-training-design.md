# Phase 4: Flop + Preflop Model Training — Design

## Overview

Two sub-phases completing the full Supremus neural network stack:

**4A: Flop CFVNet** — solve flop subgames with turn model at leaves, same pattern as Phase 3.

**4B: Preflop Auxiliary Network** — NO CFR solving. Average flop model predictions over all 22,100 possible flops. Pure inference pipeline.

After Phase 4, we have the complete model chain: preflop-aux → flop → turn → river.

## 4A: Flop CFVNet Training

Identical architecture to turn training (Phase 3):

```
GPU-Resident Loop:
  Sample random flop situations (3-card boards)
  Build flop subtree (one street of betting → depth boundaries at turn)
  Solve with DCFR+ (4000 iterations):
    Each iteration:
      regret_match → init_reach → forward_pass
      For each traverser:
        Gather reach at depth boundaries
        Encode inputs: (boundary × 45 turn cards × spots) → [N, 2720]
        CUDA-native forward pass through turn model
        Average CFVs across turn cards per combo
        Fold terminal eval
        backward_cfv → update_regrets
  Extract root CFVs → reservoir insert → train flop model
```

**Differences from Phase 3:**
- 3-card boards (flop) instead of 4 (turn)
- ~45 possible turn cards (52 - 3 board - ~4 blocked) instead of 48 river cards
- Turn model as leaf evaluator instead of river model
- Board state: `BoardState::Flop` with `depth_limit: Some(0)`

**No new code required** — the existing `TurnBatchSolverCuda` and `train_turn_cfvnet_cuda` generalize to any street with a different board size and leaf model. Just parameterize by street.

## 4B: Preflop Auxiliary Network Training

Fundamentally different — no CFR solving:

```
GPU-Resident Loop:
  Sample random preflop situations (ranges + pot + stack, NO board)
  For each situation:
    Enumerate all C(52,3) = 22,100 possible flops
    Batch forward pass: flop model on [22100, 2720] inputs
    Average outputs across flops per combo (handling card conflicts)
    Result: 1326-dim CFV target
  Insert into reservoir → train preflop model
```

**Key characteristics:**
- No game tree, no solver, no DCFR+ iterations
- Each sample = one batched inference call (22,100 inputs)
- Training target = averaged flop model predictions
- Input to preflop model: ranges + pot + stack (NO board cards — input size reduces)
- Output: 1326 CFVs
- Very fast per sample (~seconds), very low loss (Supremus: 0.000070)

**Input encoding for preflop model:**
- Same 2720-dim format but board one-hot and rank presence are all zeros
- Or: use a smaller input (2720 - 52 - 13 = 2655) without board features
- Simplest: keep 2720-dim with zeros for board, model learns to ignore them

**Batched flop enumeration on GPU:**
- Pre-enumerate all 22,100 3-card flops as a static lookup table
- For each sample, build 22,100 inputs: same ranges/pot/stack, different board one-hot
- One CUDA-native forward pass on [22,100, 2720]
- Average across 22,100 outputs per combo (with card conflict masking)

## Implementation Strategy

Since flop training reuses the Phase 3 infrastructure (just different street), the main new work is:

1. **Generalize turn pipeline to any street** — parameterize by board_size, leaf model, num_possible_cards
2. **Preflop inference-only pipeline** — new, simpler pipeline
3. **CLI commands** — `gpu-train-flop` and `gpu-train-preflop`
