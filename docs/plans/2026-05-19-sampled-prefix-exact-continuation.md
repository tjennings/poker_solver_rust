# Sampled-Prefix Exact-Continuation Training

## Problem

The current lazy MP blueprint trainer samples one complete private/public deal per
meta-iteration. For each sampled full board, it runs external-sampling MCCFR:

- traverser decision nodes visit all legal actions
- opponent decision nodes sample one legal action from the current strategy
- chance is already resolved because the full 5-card board was sampled up front

This is cheap per iteration, but noisy. A single sampled turn/river can make a
postflop line look much better or worse than its expectation over runouts. That
noise can slow convergence, destabilize visual strategy snapshots, and interact
badly with pruning.

The proposed alternative is to sample a deal prefix, then fully evaluate the
continuation below that prefix.

## Goal

Add an opt-in training mode that samples a chance prefix and performs exact
continuation traversal below that prefix.

The design must preserve the full-game objective. A sampled prefix is acceptable
only if the continuation update is an unbiased estimate of the update that would
have been produced by averaging over the omitted chance events.

## Definitions

### Sampled Full Deal

Current behavior. Sample hole cards plus all five board cards:

```text
hole cards + flop + turn + river
```

Then traverse with that fixed board.

### Sampled Turn, Exact River

Sample hole cards, flop, and turn:

```text
hole cards + flop + turn
```

When traversal crosses from turn to river, enumerate every legal river card and
average the child values.

For 6-max after hole cards plus flop plus turn, legal rivers are roughly:

```text
52 - 12 hole cards - 4 board cards = 36 river cards
```

### Sampled Flop, Exact Turn/River

Sample hole cards and flop:

```text
hole cards + flop
```

When traversal crosses from flop to turn, enumerate every legal turn card. When a
turn subtree crosses to river, enumerate every legal river card.

For 6-max after hole cards plus flop, legal ordered turn/river runouts are
roughly:

```text
(52 - 12 - 3) * (52 - 12 - 4) = 37 * 36 = 1,332
```

### Exact Continuation

There are two separate meanings. They should be implemented as separate config
axes.

Chance exactness:

- sampled: current full-board sampling
- exact river: enumerate river cards below sampled turn
- exact turn/river: enumerate turn and river cards below sampled flop

Action exactness:

- external: current MCCFR action behavior; traverser nodes enumerate actions,
  opponent nodes sample one action
- expectation: below the sampled-prefix boundary, opponent nodes also enumerate
  all actions and return the strategy-weighted expectation

The first experiment should enable exact chance continuation while keeping
external action sampling. The stronger "fully traversed spot" experiment should
then enable expectation-mode opponent traversal below the selected prefix.

## Recommended Config Shape

```yaml
training:
  chance_continuation_mode: sampled_full_deal
  exact_continuation_actions: false
```

Allowed `chance_continuation_mode` values:

```text
sampled_full_deal
sampled_turn_exact_river
sampled_flop_exact_turn_river
```

`exact_continuation_actions: true` means opponent nodes below the exact chance
boundary are expectation nodes, not sampled nodes. The default must remain
`false`, because expectation traversal below a flop can multiply action work by a
large factor.

## Estimator Contract

At a chance boundary, exact continuation returns the arithmetic mean of legal
child values:

```text
V(chance_node) = sum_child V(child) / num_legal_children
```

For sampled-turn exact-river:

```text
V(turn_to_river) = mean_river V(river)
```

For sampled-flop exact-turn/river:

```text
V(flop_to_turn) = mean_turn mean_river_given_turn V(river)
```

This is unbiased for the chance events not sampled in the prefix, assuming the
prefix itself is sampled uniformly from legal deals.

Important: regret deltas should be based on the averaged continuation values, not
on the sum. Do not multiply regrets by the number of river or turn/river
runouts. Summing would distort DCFR thresholds, regret scaling, pruning
thresholds, and average-strategy accumulation.

## Traversal Semantics

Current lazy traversal has only:

```rust
enum LazyNode {
    Decision(LazyPublicState),
    Terminal { ... },
}
```

The exact-continuation design needs chance nodes or equivalent street-transition
hooks:

```rust
enum LazyNode {
    Decision(LazyPublicState),
    Chance {
        state: LazyPublicState,
        next_street: Street,
    },
    Terminal { ... },
}
```

`showdown_or_next_street` should return a chance node when the next street is not
already present in the sampled prefix. The chance node enumerates legal next
cards, extends the deal prefix, computes only the newly needed buckets, and
recurses into `Decision(new_street_state(...))`.

For all-in runouts, this mode should still enumerate the missing public cards
when showdown equity needs them. It may skip intermediate betting decisions when
only one active non-all-in player remains, but terminal hand evaluation must use a
complete board.

## Data Model

The current traversal passes `DealWithBuckets`, which assumes a full board and
bucket assignments for every street:

```rust
pub struct DealWithBuckets {
    pub deal: Deal,
    pub buckets: [[Bucket; 4]; MAX_PLAYERS],
}
```

Exact continuation needs a prefix-aware structure:

```rust
struct DealPrefixWithBuckets {
    hole_cards: [[Card; 2]; MAX_PLAYERS],
    board: [Option<Card>; 5],
    num_players: u8,
    buckets: [[Option<Bucket>; 4]; MAX_PLAYERS],
}
```

Required operations:

- create sampled full deal, sampled turn, or sampled flop prefix
- enumerate legal next public cards
- extend prefix with one card
- compute buckets only for the newly completed street
- materialize a full `DealWithBuckets` only at terminal showdown or for legacy
  code paths that still require a complete board

The bucket lookup path must be cached. Exact runout enumeration will otherwise
turn bucket files into the bottleneck.

## Action Traversal Modes

### External Action Sampling

This is the lower-risk MVP:

- traverser nodes: enumerate all actions and update regrets
- opponent nodes: sample one action and update average strategy
- chance nodes: enumerate configured unsampled runout cards

This reduces card variance while preserving the current MCCFR action estimator.
It is the right first benchmark because it isolates the value of exact chance
continuation.

### Expectation Continuation

This is the true "sample a spot, fully traverse continuation" mode:

- traverser nodes: enumerate all actions and update regrets
- opponent nodes below the prefix boundary: enumerate all actions and return the
  strategy-weighted value
- chance nodes below the prefix boundary: enumerate all legal runouts

This is closer to full-width CFR inside a sampled public chance slice. It should
produce much smoother local updates, but the action-tree multiplier may be large.

Implementation detail: strategy sums at expectation opponent nodes should be
weighted by the reach of that node within the exact continuation. Blindly adding
one unit of strategy sum to every enumerated opponent node will overcount
branches that had low opponent reach.

## Phased Implementation

### Phase 1: Prefix Deal Infrastructure

Add prefix deal sampling and extension helpers:

- `sample_deal_prefix(num_players, prefix_street, rng)`
- `legal_next_public_cards(prefix)`
- `extend_public_card(prefix, card)`
- `compute_new_street_buckets(prefix, all_buckets)`

Keep current full-deal sampling as the default path.

### Phase 2: Exact River Continuation

Implement `sampled_turn_exact_river`.

This is the smallest useful experiment. It adds at most about 36 river children
at turn-to-river boundaries and proves the chance-node contract without the
1,332x flop runout multiplier.

### Phase 3: Exact Turn/River Continuation

Implement `sampled_flop_exact_turn_river`.

This is the main experiment for reducing postflop card variance. It should be
guarded by config and benchmarked carefully before being used for long training
runs.

### Phase 4: Expectation Action Continuation

Add `exact_continuation_actions`.

Below the exact-continuation boundary, opponent nodes switch from `sample_action`
to a weighted sum over all actions. This phase requires reach-aware strategy-sum
updates and dedicated tests for average-strategy accounting.

## Validation

Add toy-game tests:

1. Exact river continuation equals explicit average over all river-card sampled
   traversals for a fixed turn prefix.
2. Exact turn/river continuation equals explicit average over all ordered
   turn/river sampled traversals for a fixed flop prefix.
3. Regret deltas do not scale with the number of enumerated runouts.
4. Missing-card all-in showdowns still evaluate with complete boards.
5. `sampled_full_deal` behavior is unchanged when the new config keys are left at
   defaults.

Add trainer benchmarks:

```text
sampled_full_deal
sampled_turn_exact_river
sampled_flop_exact_turn_river
sampled_turn_exact_river + exact_continuation_actions
sampled_flop_exact_turn_river + exact_continuation_actions
```

Record:

- iterations per second
- traversals per second
- terminal evaluations per second
- bucket lookups per second
- sparse entries allocated per million iterations
- regret max/min/average telemetry
- strategy snapshot stability
- prune percentage

The success metric is exploitability proxy or strategy quality per wall-clock
hour, not iterations per second.

## Risks

The main correctness risk is biased weighting. If exact continuation sums child
values instead of averaging them, regrets will explode. If expectation-action
mode updates strategy sums without reach weights, average strategies will be
biased.

The main performance risk is bucket lookup pressure. Enumerating 1,332 runouts
under a flop can do a large number of repeated postflop bucket lookups. Add a
small per-job cache keyed by `(street, board, hole_cards)` or precompute all seat
buckets for each runout in a tight loop before traversing the action subtree.

The main memory risk is exact-action continuation. Chance exactness alone should
reuse the same abstract infoset keys across runouts where buckets collide.
Expectation-action traversal can visit much more of the betting tree per sampled
prefix and may accelerate sparse storage growth.

## Recommendation

Implement this in order:

1. `sampled_turn_exact_river` with external action sampling.
2. Benchmark against `sampled_full_deal`.
3. If wall-clock quality improves, add `sampled_flop_exact_turn_river`.
4. Only after that, add `exact_continuation_actions`.

This keeps the first experiment small and interpretable. If exact river
continuation does not help, full flop exact continuation is unlikely to justify
its much larger multiplier.
