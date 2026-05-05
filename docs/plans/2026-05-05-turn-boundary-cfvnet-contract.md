# Turn-Boundary CFVNet Contract

## Goal

Build a direct turn-boundary counterfactual value network for GPU turn datagen.
The model predicts river-averaged boundary CFVs from the 4-card turn public
state, both players' ranges, pot, effective stack, and player perspective.

This replaces the current runtime expansion:

```text
batch * boundary_nodes * players * river_cards
```

with:

```text
batch * boundary_nodes * players
```

For the current GPU turn topology, that changes `128 * 39 * 2 * 48 = 479232`
network rows per boundary sweep into `128 * 39 * 2 = 9984` rows.

## Model Scope

The turn-boundary model is not a full turn root model. It evaluates depth-limit
boundary states reached inside a turn re-solve, where the next public chance
event would be the river card and the remainder of the game is represented by
an oracle target.

The model must be usable in two places:

- Offline dataset/training evaluation against the slow oracle.
- GPU turn datagen boundary evaluation during CFR iterations.

The existing river-enumeration path remains the oracle and debug fallback.

## Input

The input shape stays `2720` `f32` values so the runtime can reuse the existing
BoundaryNet infrastructure.

| Offset | Size | Field | Encoding |
| --- | ---: | --- | --- |
| 0 | 1326 | OOP range | Reach/range probabilities for each two-card combo. |
| 1326 | 1326 | IP range | Reach/range probabilities for each two-card combo. |
| 2652 | 52 | Board one-hot | Four turn board cards set to `1.0`; all other cards `0.0`. |
| 2704 | 13 | Rank presence | Ranks present on the four-card board set to `1.0`. |
| 2717 | 1 | Pot fraction | `pot / (pot + effective_stack)`. |
| 2718 | 1 | Stack fraction | `effective_stack / (pot + effective_stack)`. |
| 2719 | 1 | Player | `0.0 = OOP`, `1.0 = IP`. |

The board one-hot intentionally has five-card capacity even though only four
cards are set. This preserves the current `2720` contract used by Rust, Python,
and exported ONNX models.

## Range Semantics

Ranges must be legal for the four-card board:

- Board-blocked hands have zero probability.
- Impossible self-colliding combos are masked in downstream consumers.
- Ranges should be normalized for model input when the total mass is positive.
- Empty or near-empty ranges are invalid training/evaluation records unless a
  caller explicitly records them as skipped oracle failures.

At GPU runtime, boundary reach vectors are converted to model ranges using the
same semantics as the slow oracle. The output conversion must preserve the
solver's expected counterfactual-reach orientation.

## Output

The output shape is `1326` `f32` values for the requested player perspective.

Each output is normalized EV for the corresponding private combo:

```text
target[h] = chip_cfv[h] / (pot + effective_stack)
```

For records produced by the existing pot-relative storage format:

```text
chip_cfv[h] = stored_cfv[h] * pot
target[h] = stored_cfv[h] * pot / (pot + effective_stack)
```

At inference:

```text
chip_cfv[h] = output[h] * (pot + effective_stack)
```

The GPU solver then applies its normal reach weighting and sign/orientation
rules. The network itself predicts per-hand values, not already-weighted EV
contributions.

## Target Oracle

The first production target source is the current slow river-enumeration path:

1. For a sampled turn boundary state, enumerate legal river cards.
2. Evaluate each resulting 5-card river state with the current river CFVNet
   path or an exact river solver when available.
3. Average legal river-card CFVs back onto each turn combo, using only runouts
   where the combo is not board-blocked.
4. Store one record per player perspective.

Every shard must record its target source:

- `river_net`: generated from the current river CFVNet oracle path.
- `exact_river`: generated from exact river solves.
- `mixed`: generated from both, with per-record or manifest-level counts.

The first scaled dataset can use `river_net`; validation shards should include
`exact_river` samples wherever feasible.

## Dataset Record Schema

The binary `TrainingRecord` remains the physical row format:

```text
board_size: u8
board: [u8; board_size]
pot: f32
effective_stack: f32
player: u8
game_value: f32
oop_range: [f32; 1326]
ip_range: [f32; 1326]
cfvs: [f32; 1326]
valid_mask: [u8; 1326]
```

Turn-boundary rows set `board_size = 4`.

The manifest is the schema boundary for metadata that does not fit in the row:

- `schema_version`
- `street = "turn_boundary"`
- `target_source`
- source model path/checksum
- generator commit/config hash
- action depth or boundary node label
- pot, stack, and SPR buckets
- raise-depth bucket, including 3-bet, 4-bet, and 5-bet-plus coverage
- sampled exploitability and oracle-parity summary

## Validation Gates

A model is not ready for GPU turn datagen replacement until it passes all gates:

- Direct oracle parity on frozen validation records.
- Range-weighted CFV error by pot, stack, SPR, board texture, and raise-depth
  strata.
- Downstream GPU turn exploitability against frozen sampled spots.
- Sampled parity against the slow river-enumeration path during runtime.

Aggregate Huber loss is useful, but not sufficient. The current river-net issue
appears to be sparse-strata weakness in deep-raise and tiny-pot/high-SPR states,
so those buckets must have explicit gates.

## Paper Alignment

This contract follows the DeepStack and Supremus framing: a learned value
function at a public street boundary takes public state plus both players'
ranges and returns counterfactual values for private hands. ReBeL's public
belief state formulation points the same way: the learned object is a vector of
infostate values conditioned on belief/ranges, not a scalar board value.

The Brown/Sandholm/Amos multi-valued-state approach remains a useful fallback
or comparison point, but the direct CFVNet contract is the path that removes the
online 48-river expansion.
