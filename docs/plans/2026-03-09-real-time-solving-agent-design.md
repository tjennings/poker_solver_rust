# Real-Time Solving Agent Design

## Overview

A Pluribus-inspired play system for HUNL that combines blueprint strategy with real-time subgame solving. A new `RealTimeSolvingAgent` uses the blueprint for preflop play and opponent range tracking, then solves subgames at each postflop decision — either depth-limited (1 street ahead with blueprint CBVs at leaves) or full-depth (via the existing range-solver crate) depending on range size.

## Decisions

| Decision | Choice |
|-|-|
| Depth-limited solver | `SubgameCfrSolver` in core |
| Full-depth solver | `range-solver` crate (as-is) |
| CBV computation | Precompute during blueprint training, save with snapshot |
| Depth boundary | 1 street ahead (river always full-depth to showdown) |
| Bet tree | Same as blueprint (no action translation) |
| Hybrid switch | Configurable per-street combo thresholds; river always full-depth |
| Evaluation | Real-time agent vs blueprint agent, full game, mbb/h |
| Agent interface | Implements existing `rs_poker::arena::Agent` trait |
| Simulation | Uses existing simulation infrastructure, no changes needed |

## Architecture

```
RealTimeSolvingAgent
├── blueprint: BlueprintV2Strategy    (loaded bundle for range narrowing + preflop play)
├── cbv_table: CbvTable               (precomputed continuation values per node × bucket)
├── solver_config: SolverConfig        (bet sizes, thresholds, iteration limits)
└── range_state: RangeNarrower         (opponent 1326-combo reach weights)

Decision flow:
  act() called
    → if preflop: blueprint lookup, sample action
    → if postflop:
        1. Update opponent range (multiply by blueprint action prob for their last action)
        2. Apply card removal (zero combos conflicting with board + hero hand)
        3. Count live combos
        4. If combos < street threshold OR river:
             → Build PostFlopGame with Range, solve full-depth via range-solver
        5. Else:
             → Build SubgameTree (1 street deep)
             → Load CBVs for leaf nodes
             → Solve via SubgameCfrSolver
        6. Extract strategy for hero's hand, sample action
```

### Data Flow

```
Blueprint action history → Range narrowing (1326 combo weights)
                              ↓
                    Card removal (zero blocked combos)
                              ↓
                    Combo count check (per-street threshold)
                     ↙                    ↘
           Full-depth solve          Depth-limited solve
          (range-solver crate)      (SubgameCfrSolver + CBVs)
                     ↘                    ↙
                    Strategy for hero hand
                              ↓
                        Sample action
```

## Component 1: Range Narrowing

The opponent's range starts at all 1326 combos with weight 1.0. At each opponent action, we multiply each combo's weight by the blueprint's probability of that action for that combo.

**Bridge problem:** The blueprint uses abstract buckets (169 canonical hands or EHS2 buckets), but the range-solver needs 1326-combo weights. Required:

1. `expand_canonical_to_combos(bucket_reach, board, hero_hand) -> Vec<f64>` — maps each of 1326 combos to its bucket, copies the bucket's reach weight, zeros combos that conflict with board or hero cards.

2. At each opponent decision point:
   - Look up blueprint action probs for the opponent's bucket at this node
   - Multiply the opponent's reach weight for each combo by the probability of the action they took
   - No normalization needed (solvers handle unnormalized weights)

3. Card removal applied **after** all action multiplications, right before feeding into the solver, since new board cards are revealed between streets.

**Location:** `RangeNarrower` struct in `crates/core/src/blueprint/`, holds reference to blueprint, tracks 1326-weight vector, exposes `update(opponent_action, node) -> &[f64]`.

## Component 2: CBV Precomputation

During blueprint training, after each snapshot save, compute counterfactual values for every (decision_node, bucket) pair at street boundaries.

**What gets computed:** For each node at a street boundary, for each bucket, the expected value of continuing play under the blueprint strategy to showdown:
- Terminal nodes: use the payoff
- Chance nodes: average over all next cards (weighted by card probabilities)
- Decision nodes: sum over actions weighted by blueprint action probs

**Storage:** Flat `Vec<f32>` indexed by `(boundary_node_idx, bucket)`, saved as `cbv.bin` alongside the blueprint.

```rust
struct CbvTable {
    values: Vec<f32>,
    node_offsets: Vec<usize>,
    buckets_per_node: Vec<u16>,
}

impl CbvTable {
    fn lookup(&self, boundary_node: usize, bucket: usize) -> f32;
}
```

**Integration:** When `SubgameCfrSolver` hits a `DepthBoundary` node, it calls `cbv_table.lookup(node, bucket)` instead of the current stub returning 0.0.

## Component 3: Hybrid Solver Dispatch

```rust
struct SolverConfig {
    flop_combo_threshold: usize,   // e.g., 200
    turn_combo_threshold: usize,   // e.g., 300
    depth_limited_iterations: u32, // e.g., 200
    full_solve_iterations: u32,    // e.g., 1000
    target_exploitability: f64,    // e.g., 0.5% pot
}
```

**Dispatch logic:**
```
match street {
    River => full_solve(range_solver),
    Turn  => if live_combos < turn_combo_threshold { full_solve } else { depth_limited },
    Flop  => if live_combos < flop_combo_threshold { full_solve } else { depth_limited },
}
```

**Full-depth path:** Build `PostFlopGame` from current board/pot/stacks/ranges, call `solve()`, extract strategy for hero's hand.

**Depth-limited path:** Build `SubgameTree` for current street only, load CBVs for boundary nodes at the next street, feed opponent reach weights, run `SubgameCfrSolver`, extract strategy for hero's hand.

**Action extraction:** Both paths produce per-combo action probabilities. Index into hero's specific combo, get probability distribution over actions, sample.

## Component 4: Agent Integration

`RealTimeSolvingAgent` implements `rs_poker::arena::Agent` and plugs into existing simulation.

**State management:** Like `BlueprintAgent`, reads thread-local `ACTION_LOG` to reconstruct full action history. On each `act()`:
1. Diff action log against internal state to detect new opponent actions
2. For each new opponent action, call `range_narrower.update(action, node)`
3. Dispatch to appropriate solver
4. Return sampled `AgentAction`

**Reset between hands:** `AgentGenerator` creates fresh agent per hand (same pattern as `BlueprintAgent`). Range narrower resets to uniform 1326 weights.

**Configuration:**
```yaml
player1:
  type: real_time_solver
  blueprint_path: /path/to/bundle
  solver:
    flop_combo_threshold: 200
    turn_combo_threshold: 300
    depth_limited_iterations: 200
    full_solve_iterations: 1000
```

**CLI:** Extend existing `simulate` subcommand to accept the new agent type. Same output format (mbb/h, equity curve, solve time stats).

## Component 5: Testing

**All tests use minimal trees** — 2-3 bet sizes, small boards, restricted ranges (10-20 combos per player). No test exceeds 10 seconds. Goal is correctness verification, not realistic solving.

**Unit tests:**
- Range narrowing: 5-combo range, 2-action sequence, verify weights match hand computation; verify card removal zeros blocked combos
- CBV table: 3-node tree with 4 buckets, verify values match manual forward pass
- Solver dispatch: mock solvers, verify threshold logic per street

**Integration tests:**
- Full-depth: river spot, 10 combos per side, 2 bet sizes. Verify nuts bets, air folds.
- Depth-limited: flop spot, 20 combos, 2 bet sizes, 2-bucket CBVs. Verify completes and returns valid probs.
- Agent round-trip: single hand through `act()`, tiny blueprint, verify no panics.

**Evaluation (manual, not CI):**
- Large-scale sims (10K+ hands) run manually, not in test suite.
- Log solve time distribution during evaluation runs.

## Files Modified/Created

| File | Change |
|-|-|
| `crates/core/src/blueprint/range_narrower.rs` | New — 1326-combo range tracking with blueprint action prob updates |
| `crates/core/src/blueprint/cbv.rs` | New — CBV precomputation and lookup table |
| `crates/core/src/blueprint/subgame_cfr.rs` | Modify — integrate CBV lookup at depth boundaries |
| `crates/core/src/agent.rs` | Add `RealTimeSolvingAgent` implementing `Agent` trait |
| `crates/core/src/simulation.rs` | Add agent config variant for real-time solver |
| `crates/trainer/src/main.rs` | Add CBV computation to snapshot save path |
| `crates/trainer/src/commands/simulate.rs` | Support new agent type in CLI |
