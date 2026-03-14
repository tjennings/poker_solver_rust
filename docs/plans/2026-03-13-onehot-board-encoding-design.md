# One-Hot Board Encoding for CFV Network

## Problem

The current board encoding represents each card as a single scalar (`card_id / 51.0`),
forcing the network to learn modular arithmetic (`card_id % 4`) to determine suit membership.
MLPs are notoriously bad at this, limiting flush resolution.

## Solution

Replace the variable-length scalar board encoding with a fixed 52-element binary one-hot
vector. Position `i` is `1.0` if card `i` is on the board, `0.0` otherwise. This makes
suit detection a trivial linear operation — the network can sum 13 features per suit.

## Input Layout (all streets)

```
[0..1326)     OOP range
[1326..2652)  IP range
[2652..2704)  52-element one-hot board vector
[2704]        pot / 400.0
[2705]        effective_stack / 400.0
[2706]        player indicator
Total: 2707
```

Previously: 2658 (flop), 2659 (turn), 2660 (river) — variable by street.
Now: 2707 constant regardless of street.

## Design Decisions

- **No derived features** (suit counts, flush flags, etc.) in this change. Can layer on
  in a follow-up with clean before/after metrics.
- **No backward compatibility** with existing trained models — retraining required.
  Existing generated training data (binary records with raw `u8` card IDs) works as-is.
- **`input_size` becomes a constant** `INPUT_SIZE = 2707`. The `board_cards` parameter
  is dropped from the encoding functions since the one-hot is always 52 elements.

## Files Modified

| File | Change |
|------|--------|
| `network.rs` | `input_size()` → constant `INPUT_SIZE = 2707`, drop `board_cards` param |
| `dataset.rs` | `encode_record` and `encode_situation_for_inference` write 52-element one-hot; drop `board_cards` from `encode_record` |
| `river_net_evaluator.rs` | `build_input` writes one-hot; update offsets |
| `compare_turn.rs` | `predict_with_model` writes one-hot |
| `main.rs` | Update pot offset to constant `2704`; update `input_size()` call sites |
| `training.rs` | Update encoding call sites |
| Tests | Update expected sizes and offset-based assertions |

## What Doesn't Change

- Binary training data format (records still store `u8` card IDs)
- Network architecture (MLP with configurable layers/width)
- Loss functions, training loop, config schema
- Datagen pipeline
