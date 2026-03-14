# Rank-Presence Vector for CFV Network

## Problem

Straight/connectedness detection requires 2-3 layers of nonlinearity from the 52-element
one-hot board encoding. The network must learn to collapse each 4-card suit group into a
rank-present signal, then detect adjacency — an expensive AND operation across non-contiguous
feature positions. Wheel straights (A-2-3-4-5) are especially hard since Ace and Deuce
occupy opposite ends of the feature vector.

## Solution

Add a 13-element binary rank-presence vector immediately after the board one-hot.
Element `r` is `1.0` if any card of rank `r` (2=0, 3=1, ..., A=12) is on the board.
This makes straight detection a single-layer linear sum, mirroring what the one-hot
did for flush detection.

## Input Layout

```
[0..1326)     OOP range
[1326..2652)  IP range
[2652..2704)  52-element one-hot board
[2704..2717)  13-element rank-presence vector
[2717]        pot / 400
[2718]        effective_stack / 400
[2719]        player indicator
Total: 2720
```

Previously: 2707 (with one-hot only).

## Design Decisions

- **Binary (0/1), not count (0/1/2/3)**: Paired board detection is already a 1-2 layer
  problem from the one-hot. The rank-presence vector targets connectedness only.
- **Placed after one-hot, before pot/stack/player**: Keeps all board-derived features
  contiguous.
- **Derived from the same board cards**: Uses `card / 4` to extract rank from card ID.

## Files Modified

Same 4 encoding sites plus constants, identical to the one-hot change:

| File | Change |
|------|--------|
| `network.rs` | `INPUT_SIZE` 2707→2720, `POT_INDEX` 2704→2717, add `NUM_RANKS = 13` |
| `dataset.rs` | Both encoding functions add rank-presence after one-hot |
| `river_net_evaluator.rs` | `build_input` adds rank-presence |
| `compare_turn.rs` | `predict_with_model` adds rank-presence |
| Tests | Update sizes and offsets |

## What Doesn't Change

- Binary training data format
- Network architecture (just different `in_size`)
- Loss functions, training loop, config schema
- Datagen pipeline
- Existing one-hot encoding (rank-presence is appended after it)
