# Preflop Solver + Subgame Solving Architecture

**Status:** Approved design
**Date:** 2026-02-19
**Goal:** Three-layer solving architecture: exact preflop solver, coarse postflop blueprint, real-time subgame solving. N-player extensible (up to 9), any stack depth.

## Overview

Modern poker AI (Libratus, Pluribus) uses a two-phase approach: an imprecise blueprint covering the full game, refined by real-time subgame solving at play-time. We extend this with a dedicated preflop layer that exploits preflop's small size for exact solutions.

```
Layer 1: PREFLOP SOLVER (Linear CFR, full enumeration)
  - 169 canonical hands, exact equities
  - Solves in seconds (HU) to minutes (6-max)
  - Produces: preflop strategy + reaching ranges at flop root

Layer 2: POSTFLOP BLUEPRINT (MCCFR/LCFR, coarse abstraction)
  - ~50-150 hand buckets per street, 3 bet sizes
  - Cloud burst: ~100-300 core-hours
  - Produces: coarse strategy table + leaf values + opponent ranges

Layer 3: SUBGAME SOLVER (CFR+, real-time)
  - Specific board, no hand abstraction
  - 5-7 bet sizes, depth-limited (1 street)
  - Solves in 1-10 seconds per decision
  - Produces: precise strategy for the exact situation
```

## Design Principles

- **N-player from the start.** All core types use `Vec<>` for positions/stacks, never hardcoded pairs. `to_act` is always a position index. HU is `num_players == 2`.
- **Any stack depth.** Per-position stacks, not a single `stack_depth`. Supports asymmetric stacks (tournaments).
- **Existing infrastructure reuse.** `SequenceCfrSolver` already supports LCFR via config params. `StrategyBundle` format extends cleanly. Explorer pattern (`StrategySource` enum, async background computation) is proven.

## Layer 1: Preflop Solver

### Algorithm: Linear CFR with Full Enumeration

Linear CFR weights iteration `t`'s contributions linearly by `t`:

```
regret_sum[i]   += t * instantaneous_regret[i]
strategy_sum[i] += t * current_strategy[i] * reach
```

Convergence is faster than CFR+ in practice for poker because early (noisy) iterations are heavily discounted. Used by Pluribus for its blueprint.

### Core Types

```rust
struct PreflopConfig {
    positions: Vec<PositionInfo>,   // UTG..BB, ordered by action
    blinds: Vec<(usize, u32)>,     // (position_idx, amount)
    antes: Vec<(usize, u32)>,      // per-position antes
    stacks: Vec<u32>,              // per-position stacks
    raise_sizes: Vec<Vec<f64>>,    // sizes[raise_depth] = [2.5, 3.0, ...]
    raise_cap: u8,                 // max raises per round (typically 4-5)
}

struct PreflopTree {
    nodes: Vec<PreflopNode>,
    root: u32,
}

enum PreflopNode {
    Decision {
        position: u8,
        actions: Vec<u32>,              // child node indices
        action_labels: Vec<PreflopAction>,
    },
    Terminal {
        payoff_type: TerminalType,      // Fold(who) | Showdown
    },
}

enum PreflopAction {
    Fold,
    Call,
    Raise(f64),     // size as multiplier
    AllIn,
}
```

### Solver

```rust
struct PreflopSolver {
    tree: PreflopTree,
    config: PreflopConfig,
    equity_table: EquityTable,
    regret_sum: Vec<Vec<f64>>,      // [node_id * 169] * num_actions
    strategy_sum: Vec<Vec<f64>>,
    iteration: u64,
}

impl PreflopSolver {
    fn new(config: PreflopConfig) -> Self;
    fn train(&mut self, iterations: u64);
    fn strategy(&self) -> PreflopStrategy;
}
```

### Equity Table

- **HU:** Precompute 169x169 matchup equities using existing `showdown_equity` infrastructure. One-time cost ~10 seconds.
- **3-way:** 169^3 = 4.8M entries. Precomputable in minutes.
- **4+ way:** Canonical-hand weighting with card removal corrections. Sample for N>4.

### Tree Size Estimates

| Scenario | Nodes | Info sets | Solve time |
|-|-|-|-|
| HU, standard sizing | ~200 | ~5K | <1 second |
| HU, 5 raise sizes | ~500 | ~15K | ~2 seconds |
| 6-max, 3 sizes | ~10K-50K | ~500K | ~1-5 minutes |
| 9-max, 3 sizes | ~50K-200K | ~5M | ~1-3 hours |

### Persistence

```
preflop_solve/
  config.yaml        # PreflopConfig
  strategy.bin        # PreflopStrategy (bincode)
  equity_table.bin    # EquityTable (optional, can recompute)
```

## Layer 2: Postflop Blueprint

### Purpose

The blueprint provides two things the subgame solver needs:

1. **Opponent reaching probabilities** — what range does the opponent arrive with at any game node?
2. **Leaf continuation values** — what's a hand worth at depth boundaries where the subgame solver stops?

It does NOT need to be precise. Subgame solving corrects errors.

### Coarse Abstraction

Aggressively simplified compared to current training runs:

| Dimension | Current | Coarse blueprint |
|-|-|-|
| Hand buckets (flop) | HandClassV2 (~500+) | ~50-80 |
| Hand buckets (turn) | HandClassV2 (~500+) | ~80-120 |
| Hand buckets (river) | HandClassV2 (~500+) | ~100-150 |
| Bet sizes | 5 | 3 (`[0.5, 1.0, 2.0]`) |

Achieved via HandClassV2 with `strength_bits=2, equity_bits=2`. No new bucketing infrastructure needed — just a config change.

### Training Algorithm: Linear CFR

The existing `SequenceCfrConfig` already supports LCFR via parameterization:

```rust
impl SequenceCfrConfig {
    fn linear_cfr() -> Self {
        Self {
            dcfr_alpha: 1.0,
            dcfr_beta: 1.0,
            dcfr_gamma: 1.0,
        }
    }
}
```

The DCFR formula `t^alpha / (t^alpha + 1)` with `alpha=1` gives `t / (t + 1)`, which is exactly Linear CFR. No solver code changes needed.

### Training Budget (Cloud Burst)

| Stack depth | Abstract deals | Info sets (est.) | Iterations | Core-hours |
|-|-|-|-|-|
| 25 BB | ~50K | ~2M | 5,000 | ~50 |
| 100 BB | ~50K | ~5M | 10,000 | ~150 |
| 200 BB | ~50K | ~8M | 15,000 | ~300 |

### Blueprint Queries

**Opponent reach computation.** Walk the action history, multiply blueprint action probabilities along the opponent's path:

```rust
impl BlueprintStrategy {
    fn opponent_reach(
        &self,
        hand_bits: u32,
        action_history: &[(Street, Action)],
        pot: u32,
        stacks: &[u32],
    ) -> f64;
}
```

Returns a per-bucket (or per-169-hand for preflop) vector of reach probabilities.

**Leaf value computation.** At depth-limited boundaries, compute expected value under blueprint play:

```rust
impl BlueprintStrategy {
    fn continuation_value(
        &self,
        hand_bits: u32,
        node_state: &NodeState,
        opponent_reach: &[f64],
    ) -> f64;
}
```

One forward pass through the blueprint subtree below the boundary, weighted by opponent reach. Computed once per subgame solve, not per CFR iteration.

### Persistence

Uses existing `StrategyBundle` format unchanged:

```
postflop_blueprint/
  config.yaml       # BundleConfig (with coarse abstraction settings)
  blueprint.bin      # BlueprintStrategy
  boundaries.bin     # (only if EHS2 mode)
```

New fields on `BundleConfig`:

```rust
struct BundleConfig {
    // ... existing fields ...
    #[serde(default = "default_num_players")]
    pub num_players: u8,
    #[serde(default)]
    pub is_subgame_base: bool,
}
```

## Layer 3: Real-Time Subgame Solver

### Subgame Tree

Unlike the blueprint's abstract tree, subgame trees are concrete — specific board, no hand bucketing:

```rust
struct SubgameTree {
    nodes: Vec<SubgameNode>,
    board: Vec<Card>,
    depth_limit: Street,
    bet_sizes: Vec<f32>,          // finer than blueprint (5-7 sizes)
    num_players: u8,
}

enum SubgameNode {
    Decision {
        position: u8,
        actions: Vec<SubgameAction>,
        children: Vec<u32>,
    },
    Terminal {
        payoff_type: TerminalType,
        pot: u32,
    },
    DepthBoundary {
        /// Continuation values per combo, from blueprint
        values: Vec<f64>,
    },
}
```

`DepthBoundary` is the key concept. When traversal hits a depth boundary (e.g., flop betting ends), it reads precomputed continuation values instead of recursing into the next street.

### Hand Representation

No bucketing. Every specific combo not blocked by the board:

```rust
struct SubgameHands {
    combos: Vec<[Card; 2]>,        // typically 990-1081 combos
    equity_matrix: Vec<Vec<f64>>,   // [combo_i][combo_j] = P(i beats j)
    opponent_reach: Vec<f64>,       // from blueprint
}
```

River equity is binary (no draws). Flop/turn equity via enumeration over remaining cards.

### CFR+ Solver

```rust
struct SubgameCfrSolver {
    tree: SubgameTree,
    hands: SubgameHands,
    regret_sum: Vec<Vec<f64>>,      // [node_id * combo_idx] * num_actions
    strategy_sum: Vec<Vec<f64>>,
    iteration: u32,
    config: SubgameConfig,
}

struct SubgameConfig {
    num_players: u8,
    depth_limit: usize,             // streets ahead to solve
    time_budget_ms: u64,
    max_iterations: u32,
    bet_sizes: Vec<f32>,
}

impl SubgameCfrSolver {
    fn solve(
        board: &[Card],
        action_history: &[(Street, Action)],
        pot: u32,
        stacks: &[u32],
        opponent_reach: &[f64],
        leaf_values: &LeafValues,
        config: &SubgameConfig,
    ) -> SubgameStrategy;
}
```

### Performance Targets

| Subgame scope | Combos | Nodes (5 sizes) | Iterations | Time |
|-|-|-|-|-|
| River (1 street) | ~990 | ~50 | 500 | <100ms |
| Turn (depth-limited) | ~1,000 | ~200 | 1,000 | ~500ms |
| Flop (depth-limited) | ~1,081 | ~500 | 2,000 | ~2-5s |
| Flop through river | ~1,081 | ~50,000 | 5,000 | ~30-60s |

Depth-limited (1 street) is the default.

### N-Player Subgame Solving

For 3+ players, the Pluribus approach: solve hero's strategy against opponents who play the blueprint (fixed, not updated). No Nash equilibrium guarantees for 3+ players, but practically strong.

```rust
impl SubgameCfrSolver {
    fn solve_multiplayer(
        hero_position: u8,
        opponent_strategies: &[&BlueprintStrategy],
        // ... same inputs ...
    ) -> SubgameStrategy;
}
```

## Explorer Integration

### Strategy Source Types

```rust
enum StrategySource {
    // Existing
    Bundle { config: BundleConfig, blueprint: BlueprintStrategy },
    Agent(AgentConfig),

    // New
    PreflopSolve {
        config: PreflopConfig,
        strategy: PreflopStrategy,
    },
    SubgameSolve {
        blueprint: Arc<BlueprintStrategy>,
        blueprint_config: BundleConfig,
        subgame_config: SubgameConfig,
    },
}
```

### Generalized Position State

```rust
struct ExplorationPosition {
    board: Vec<String>,
    history: Vec<String>,
    pot: u32,
    stacks: Vec<u32>,              // per-position (not just p1/p2)
    to_act: u8,                    // position index
    num_players: u8,
    active_players: Vec<bool>,     // who hasn't folded
}
```

The current `stack_p1`/`stack_p2` fields become `stacks: Vec<u32>`.

### New Tauri Commands

```
load_preflop_solve(path)     Load a saved preflop solution
solve_preflop(config)        Solve on-the-fly (fast enough for HU)
load_subgame_source(         Load blueprint + configure subgame solver
    blueprint_path,
    subgame_config
)
get_subgame_status()         Progress of in-flight subgame solve
```

`get_strategy_matrix` dispatches on source type:
- `PreflopSolve` → lookup from preflop strategy
- `SubgameSolve` → trigger real-time CFR+ (async, with progress events)
- `Bundle`/`Agent` → existing behavior (unchanged)

### Async Subgame Flow

```
User navigates to postflop node
  |
  +-- Cache HIT  --> display immediately
  |
  +-- Cache MISS --> emit "subgame-solving" event
                     Spawn background thread:
                       1. Compute opponent reach from blueprint
                       2. Compute leaf values from blueprint
                       3. Build subgame tree
                       4. Run CFR+ (progress events every 100 iters)
                       5. Extract average strategy
                       6. Cache result
                       7. Emit "subgame-solved" event
                     Display refined strategy
```

Mirrors existing `start_bucket_computation` pattern.

### UI Additions

| Component | Change |
|-|-|
| `StrategyMatrix` / `MatrixCell` | No change (already generic) |
| Position selector | New dropdown for N>2: "Viewing as: UTG / HJ / CO / BTN / SB / BB" |
| Subgame progress | Spinner + iteration count during real-time solve |
| Depth toggle | "Quick solve (1 street)" vs "Deep solve (full)" option |
| Source picker | Dropdown: Preflop / Blueprint / Subgame / Agent |

## End-to-End Flow

A complete hand using all three layers:

```
1. PREFLOP: User at preflop decision
   -> Explorer loads PreflopSolve
   -> Displays 13x13 grid from preflop strategy
   -> User selects action (e.g., "Raise 2.5x")

2. FLOP DEALT: Ah Kd 7c
   -> Explorer switches to SubgameSolve mode
   -> Computes opponent reach: walk preflop strategy to get
      villain's range at this node
   -> Triggers subgame solve:
      - Builds flop subtree (no abstraction, 5 bet sizes)
      - Gets leaf values from blueprint (continuation into turn)
      - Runs CFR+ (~2000 iterations, ~3 seconds)
   -> Displays precise flop strategy

3. FLOP ACTION COMPLETE: Both players act
   -> Update opponent reach based on actions taken
   -> Cache flop subgame result

4. TURN DEALT: 4s
   -> Trigger turn subgame solve:
      - Updated opponent reach (narrowed by flop actions)
      - Leaf values from blueprint (continuation into river)
      - CFR+ (~1000 iterations, ~500ms)
   -> Display turn strategy

5. RIVER DEALT: Tc
   -> Trigger river subgame solve:
      - No depth limit needed (final street)
      - Binary equity (no draws)
      - CFR+ (~500 iterations, <100ms)
   -> Display river strategy
```

## Implementation Order

1. **Preflop tree builder + LCFR solver** — new `PreflopSolver` struct
2. **Preflop equity table** — 169x169 HU, extensible to N-way
3. **Preflop persistence** — save/load in bundle-like format
4. **Explorer: PreflopSolve source** — load + display preflop strategies
5. **Blueprint coarsening** — config for LCFR + coarse HandClassV2
6. **Blueprint reach/value queries** — `opponent_reach()`, `continuation_value()`
7. **Subgame tree builder** — concrete tree for specific board
8. **Subgame CFR+ solver** — depth-limited with boundary values
9. **Explorer: SubgameSolve source** — async solve + cache + progress
10. **N-player generalization** — `ExplorationPosition` with `Vec<u32>` stacks, multi-position solver
11. **Multiplayer subgame** — Pluribus-style fixed-opponent solving

Steps 1-4 are independent of 5-9. Steps 10-11 extend the foundation.

## Key References

- Brown & Sandholm (2019) — "Superhuman AI for multiplayer poker" (Pluribus, Linear CFR)
- Brown & Sandholm (2018) — "Depth-Limited Solving for Imperfect-Information Games" (Libratus)
- Brown & Sandholm (2017) — "Safe and Nested Subgame Solving"
- Tammelin (2014) — CFR+
- Brown & Sandholm (2019) — "Solving Imperfect-Information Games via Discounted Regret Minimization" (DCFR)
