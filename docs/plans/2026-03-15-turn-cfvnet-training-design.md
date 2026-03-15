# GPU-Resident Turn CFVNet Training — Design

## Overview

Train a turn counterfactual value network on GPU using the Supremus approach. Turn subgames use the Phase 2 river CFVNet as a leaf evaluator at depth boundaries. The entire pipeline — sample, solve, insert, train — runs GPU-resident.

## Architecture

Same pipeline structure as Phase 2 (river training), with one addition: **neural leaf evaluation** during the DCFR+ solve loop.

```
GPU-Resident Loop:
  Sample random turn situations (4-card boards, GPU)
  Build turn subtree (one street of betting → depth boundaries)
  Solve with DCFR+ (4000 iterations, GPU):
    Each iteration:
      regret_match → init_reach → forward_pass
      For each traverser:
        Gather reach at all depth boundaries
        Encode inputs: (boundary × 48 river cards × spots) → [N, 2720]
        ONE batched burn-cuda forward pass → [N, 1326] CFVs
        Average across river cards per combo, scatter to cfvalues
        Terminal fold eval (for non-boundary terminals)
        backward_cfv → update_regrets
  Extract root CFVs → reservoir insert → train turn model
```

## Leaf Evaluation (New in Phase 3)

### Per-iteration flow

At each DCFR+ iteration, for each traverser:

1. **Gather reach** at depth-boundary nodes — the forward pass already propagated reach to all nodes. Read `reach_oop[boundary_node * num_hands + h]` and `reach_ip[boundary_node * num_hands + h]` for all boundary nodes.

2. **Enumerate river cards** — for each of the ~48 possible river cards (52 minus 4 board cards), construct a 5-card board.

3. **Encode inputs** — for each (boundary, river_card, spot), build a 2720-dim feature vector:
   - OOP range at this boundary (from reach, filtered for river card conflicts)
   - IP range at this boundary (same)
   - 5-card board one-hot (turn board + this river card)
   - Rank presence
   - Pot at boundary node (normalized)
   - Stack at boundary node (normalized)
   - Player indicator

4. **One batched forward pass** — all inputs concatenated into one tensor `[num_boundaries × 48 × num_spots, 2720]`. For a typical turn tree with ~4 boundaries, batch_size=1000: `4 × 48 × 1000 = 192,000` inputs per traverser. One burn-cuda call.

5. **Average and scatter** — for each (boundary, spot, combo), average the 48 river-card outputs (skipping rivers that conflict with the combo's hole cards). Write the averaged CFVs to `cfvalues[boundary_node * num_hands + hand]`.

### Batching math

- Per iteration: `num_boundaries × 48 × num_spots × 2` (both traversers)
- For 4 boundaries × 48 rivers × 1000 spots × 2 traversers = 384,000 forward pass inputs
- Network: 2720 → 7×500 → 1326 = ~4M parameters
- Forward pass on 384K inputs ≈ 1-5ms on RTX 6000 Ada

### GPU memory for inference

- Input tensor: 384K × 2720 × 4 bytes = 4.2 GB
- Output tensor: 384K × 1326 × 4 bytes = 2.0 GB
- Total inference buffers: ~6.2 GB (fits in 48GB VRAM)

## Turn Tree Structure

The turn tree has the same shape as a river tree — one street of betting with the same action abstraction. The difference:

- **Depth-boundary leaves** replace showdown terminals at the street boundary
- **No chance nodes** — the river card enumeration is handled by the leaf evaluator, not by expanding the tree
- **Fold terminals** still exist (player folds during turn betting)
- **No showdown terminals** — if both players check through or call, the hand reaches a depth boundary (not showdown, since the river hasn't been dealt)

Wait — actually, if the action tree allows check-check on the turn, that's a depth boundary (proceed to river). If someone bets and gets called, that's also a depth boundary (proceed to river). Fold terminals are when someone folds. There are no showdown terminals in a turn tree.

So the terminal evaluation simplifies: fold terminals only, all non-fold leaves are depth boundaries evaluated by the river model.

## Turn Situation Sampling

Same as Phase 2 but with 4-card boards:
- Sample 4 unique cards for the board (flop + turn)
- Generate random ranges (1326 combos, zero out board-blocked)
- Random pot and stack

## Training Target

Root CFVs from the turn solve — same format as Phase 2. The turn model learns to predict counterfactual values at the start of turn betting.

## Turn CFVNet Architecture

Same as river model: 7 hidden layers × 500 units, input=2720, output=1326.

Input encoding is identical — the board one-hot has 4 cards set for turn situations (instead of 5 for river). The model learns from context that it's evaluating turn positions.

Actually — should the turn model input include the number of board cards? The river model always sees 5-card boards. The turn model sees 4-card boards. If we use the same 2720-dim encoding, the model can distinguish by counting set bits in the board one-hot. This is fine.

## Validation

Same dual validation as Phase 2:
- Hold-out loss from reservoir
- Ground-truth RMSE: pre-solve 100 turn positions at high iterations using the same river model as leaf evaluator

## Dependencies

- Phase 2 river model (trained, loaded at startup)
- burn-cuda for both inference (leaf eval) and training (turn model)
- Shared CUDA context between cudarc and burn-cuda

## Key Design Decisions

1. **One batched inference call per iteration** — all boundaries × rivers × spots combined into one tensor
2. **48 river card enumeration** — explicit, matching Supremus. Network always sees 5-card boards.
3. **No chance nodes in tree** — depth boundaries replace street transitions
4. **Concrete 1326 hands** — no bucketing, same as Phase 2
5. **All GPU-resident** — reach gathering, input encoding, inference, averaging, scatter all on GPU
6. **Call leaf eval every iteration** (option A) — if too slow, add interval caching later
