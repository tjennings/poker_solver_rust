# Plan: N-Player Poker Solver (2-6 Players)

## Context

The solver currently only supports heads-up (2-player) poker. The `Player` enum has exactly 2 variants, reach weights are explicit `p1_reach`/`p2_reach` parameters, utilities are zero-sum with `p2 = -p1`, and all state arrays are sized for 2. This plan extends the solver to support 2-6 player games with full TDD, using MCCFR (standard for multi-way solvers), equal stacks (no side pots), and one-player-at-a-time UX.

## Phase 1: Player Newtype Foundation

**Goal**: Replace `Player` enum with `Player(u8)` newtype. All 432+ existing tests pass.

**Files to modify** (120 occurrences across 8 core files):
- `crates/core/src/game/mod.rs` — Player definition, remove `opponent()`
- `crates/core/src/game/kuhn.rs` — 25 occurrences
- `crates/core/src/game/hunl_postflop.rs` — 46 occurrences
- `crates/core/src/cfr/mccfr.rs` — 18 occurrences
- `crates/core/src/cfr/vanilla.rs` — 8 occurrences
- `crates/core/src/cfr/exploitability.rs` — 2 occurrences
- `crates/core/src/cfr/sequence_cfr.rs` — 8 occurrences
- `crates/core/src/cfr/game_tree.rs` — 8 occurrences
- `crates/core/src/tree.rs` — 5 occurrences
- `crates/core/src/simulation.rs` — uses Player
- `crates/tauri-app/src/exploration.rs` — uses Player
- `crates/tauri-app/src/simulation.rs` — uses Player

### TDD

**Red** — Tests for the new Player type:
```rust
fn player_stores_index()           // Player(0).index() == 0
fn player_equality()               // Player(0) == Player(0), != Player(1)
fn player_constants_compatible()   // Player::PLAYER1 == Player(0)
fn player_is_copy_clone_hash()     // Derive traits work
```

**Green** — Implementation:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Player(u8);

impl Player {
    pub const PLAYER1: Player = Player(0);
    pub const PLAYER2: Player = Player(1);
    pub const fn new(index: u8) -> Self { Self(index) }
    pub const fn index(self) -> u8 { self.0 }
}
```

**Refactor** — Mechanical replacements:
- `Player::Player1` → `Player::PLAYER1` (or `Player(0)`)
- `Player::Player2` → `Player::PLAYER2` (or `Player(1)`)
- `match player { Player1 => x, Player2 => y }` → `[x, y][player.index() as usize]`
- Remove `Player::opponent()` — callers use `Player(1 - p.index())` for 2-player or explicit logic
- `TerminalType::Fold(Player)` stays the same (Player is still the type)

**Verification**: `cargo test` — all existing tests pass. `cargo clippy` clean.

---

## Phase 2: Game Trait Extension

**Goal**: Add `num_players()` to the `Game` trait with default `2`.

**File**: `crates/core/src/game/mod.rs`

### TDD

**Red**:
```rust
fn kuhn_is_two_player()     // KuhnPoker.num_players() == 2
fn hunl_is_two_player()     // HunlPostflop.num_players() == 2
```

**Green** — Add default method to trait:
```rust
pub trait Game: Send + Sync {
    // ... existing methods ...
    fn num_players(&self) -> u8 { 2 }
}
```

Both `KuhnPoker` and `HunlPostflop` inherit the default.

---

## Phase 3: 3-Player Kuhn Poker Test Fixture

**Goal**: Create a minimal 3-player game to validate the CFR generalization before tackling full poker.

**New file**: `crates/core/src/game/kuhn3.rs`

3-player Kuhn poker: 4 cards (J, Q, K, A), 3 players each get 1 card, simple check/bet/fold/call. Known properties: player with Ace should never fold, player with Jack should never bet.

### TDD

**Red**:
```rust
fn kuhn3_has_three_players()
fn kuhn3_initial_states_count()              // C(4,3) * 3! = 24 deals
fn kuhn3_first_to_act_is_player0()
fn kuhn3_all_check_goes_to_showdown()
fn kuhn3_fold_reduces_active_players()
fn kuhn3_two_folds_is_terminal()
fn kuhn3_utility_sums_to_zero()              // Zero-sum property
fn kuhn3_best_hand_wins_at_showdown()
```

**Green**: Implement `Kuhn3` struct implementing `Game` trait with `num_players() -> 3`.

---

## Phase 4: Generalize CFR Solvers

**Goal**: Replace 2-player reach weight params with N-player reach vectors.

**Files**:
- `crates/core/src/cfr/mccfr.rs` — `cfr_traverse`, `cfr_traverse_pure`, `traversing_player`
- `crates/core/src/cfr/vanilla.rs` — `cfr`, `train`

### Key Changes

**Reach weights**: `p1_reach: f64, p2_reach: f64` → `reaches: &[f64]` (length = num_players)

**Traversal rotation**: `iterations % 2` → `iterations % num_players`

**Opponent reach product** (new pure helper):
```rust
fn opponent_reach_product(reaches: &[f64], traverser: Player) -> f64 {
    reaches.iter().enumerate()
        .filter(|(i, _)| *i != traverser.index() as usize)
        .map(|(_, &r)| r)
        .product()
}
```

**My reach**: `reaches[current_player.index() as usize]`

**Reach update in recursion**:
```rust
let mut new_reaches = reaches.to_vec();  // or SmallVec
new_reaches[current_player.index() as usize] *= strategy[i];
```

### TDD

**Red** — Unit tests for helpers:
```rust
fn opponent_reach_product_two_player()       // [0.5, 0.8], P0 → 0.8
fn opponent_reach_product_three_player()     // [0.5, 0.8, 0.3], P0 → 0.24
fn traversing_player_rotates_three()         // iter%3 → P0, P1, P2, P0...
```

**Red** — Integration tests:
```rust
fn mccfr_kuhn_2p_still_converges()           // Existing 2-player Kuhn Nash test
fn vanilla_cfr_kuhn_2p_still_converges()     // Existing 2-player convergence
fn mccfr_kuhn3_produces_valid_strategies()   // 3-player: all probs in [0,1], sum ~1
fn mccfr_kuhn3_ace_never_folds()             // Known 3-player Kuhn property
fn vanilla_cfr_kuhn3_populates_info_sets()
```

**Green**: Modify both solvers to use reach vectors. For `cfr_traverse_pure` (parallel), `TraversalAccumulator` unchanged — it just stores deltas by info set key.

**Verification**: All existing 2-player tests still pass. New 3-player Kuhn tests pass.

---

## Phase 5: N-Player Postflop Game

**Goal**: Create `NPlayerPostflop` implementing `Game` for 2-6 players.

**New file**: `crates/core/src/game/n_player_postflop.rs`
**Modify**: `crates/core/src/game/mod.rs` (re-export)

### State Design

```rust
pub struct NPlayerConfig {
    pub num_players: u8,        // 2-6
    pub stack_depth: u32,       // BB
    pub bet_sizes: Vec<f32>,    // Pot fractions
    pub max_raises_per_street: u8,
}

pub struct NPlayerState {
    pub board: ArrayVec<Card, 5>,
    pub holdings: SmallVec<[[Card; 2]; 6]>,
    pub full_board: Option<[Card; 5]>,
    pub street: Street,
    pub pot: u32,                           // Internal units (SB=1, BB=2)
    pub stacks: SmallVec<[u32; 6]>,
    pub folded: SmallVec<[bool; 6]>,
    pub to_act: Option<Player>,
    pub history: ArrayVec<(Street, Action), 60>,  // Larger for more players
    pub terminal: Option<NTerminalType>,
    pub street_bets: u8,
    pub caches: SmallVec<[PlayerCache; 6]>,
    pub dealer_position: u8,
    pub num_players: u8,
}
```

### Position Rotation

- **Preflop**: SB=dealer+1, BB=dealer+2, UTG=dealer+3, ..., action starts at UTG
- **Postflop**: First active player clockwise from dealer
- `next_active_player(state, after) -> Player` — skips folded players
- **2-player special case**: SB=dealer=BTN (heads-up), BB acts first postflop

### N-Way Showdown (equal stacks, no side pots)

```rust
fn utility_showdown(state, player, starting_stack) -> f64 {
    let active = (0..num_players).filter(|i| !folded[i]);
    let best_rank = active.map(|i| rank(i)).max();
    let winners = active.filter(|i| rank(i) == best_rank);
    let share = pot / winners.count();
    let invested = starting_stack - stacks[player];
    if is_winner { (share - invested) / 2.0 } else { -invested / 2.0 }
}
```

### Fold Logic

- Player folds → `folded[i] = true`
- If only 1 active player remains → terminal (last player wins pot)
- Otherwise → next active player acts

### TDD

**Red**:
```rust
fn three_player_correct_blinds()              // pot=3, stacks=[199,198,200]
fn three_player_utg_acts_first_preflop()      // Player(2) in 3-player
fn six_player_initial_state()                 // 6 holdings, 6 stacks
fn fold_with_three_not_terminal()             // 2 remain → not terminal
fn fold_to_one_is_terminal()                  // 1 remains → terminal
fn three_player_showdown_best_wins()
fn three_player_showdown_tie_splits()
fn utility_sums_to_zero_three_player()        // Proptest-worthy
fn postflop_first_active_after_dealer()
fn six_player_position_rotation()
fn two_player_nplayer_matches_hunl()          // Cross-validate with HunlPostflop
```

**Green**: Full implementation reusing existing patterns from `hunl_postflop.rs`:
- Reuse `PlayerCache`, `Street`, `Action`, `TerminalType` (extended)
- Reuse `resolve_bet_amount()`, `spr_bucket()`, `depth_bucket()`
- Reuse `classify()`, `compute_equity()`, `intra_class_strength()`

---

## Phase 6: Info Set Keys for N-Player

**Goal**: Handle longer action sequences from multi-player games.

**File**: `crates/core/src/info_key.rs`

### Problem

Current: 6 action slots x 4 bits = 24 bits. In 6-max, a single preflop betting round can have 6+ actions (UTG raise, MP fold, CO call, BTN 3bet, SB fold, BB call = 6 actions for just one round).

### Solution

- For sequences with <=6 actions per street: use existing packed encoding (backward compatible)
- For longer sequences: FNV-1a hash truncated to 24 bits (same approach as `HunlPreflop`)

### TDD

**Red**:
```rust
fn action_hash_deterministic()
fn action_hash_order_sensitive()              // [call, raise] != [raise, call]
fn short_sequence_matches_packed()            // <=6 actions same as before
fn long_sequence_produces_24bit_hash()
fn info_key_with_hashed_actions()
```

**Green**: New function `encode_actions_for_key(history, current_street) -> u32` that selects packed vs hashed encoding based on length.

---

## Phase 7: Training Pipeline

**Goal**: Training config supports N-player, trainer creates `NPlayerPostflop`.

**Files**:
- `crates/trainer/src/main.rs` — new config fields, game construction
- New config: `nplayer_training.yaml`

### Config Addition

```yaml
game:
  num_players: 6        # NEW — defaults to 2 for backward compat
  stack_depth: 25
  bet_sizes: [0.33, 0.67, 1.0, 2.0, 3.0]
training:
  iterations: 100000
  mccfr_samples: 500
  deal_count: 50000
  abstraction_mode: hand_class_v2
```

### TDD

**Red**:
```rust
fn trainer_config_parses_num_players()
fn trainer_creates_nplayer_game()
fn nplayer_training_produces_strategies()     // Small 3-player, 100 iters
fn two_player_config_uses_hunl()             // Backward compat
```

**Green**: When `num_players > 2`, construct `NPlayerPostflop`; otherwise use `HunlPostflop`.

---

## Phase 8: N-Player Simulation

**Goal**: Extend agent-vs-agent simulation for 2-6 players.

**Files**:
- `crates/core/src/simulation.rs` — N-player agent arena
- `crates/tauri-app/src/simulation.rs` — N player slots

### Key Changes

- `run_simulation` accepts `Vec<(Box<dyn AgentGenerator>, Vec<f32>)>` (N agents)
- `RotatingDealerGenerator` rotates across N positions
- `ACTION_LOG: Vec<(Round, u8, u8)>` — add player index to each entry
- `SimResult` gains per-player profit array: `profits_bb: Vec<f64>`
- `BlueprintAgent` builds info key from ACTION_LOG with player index awareness

### TDD

**Red**:
```rust
fn three_player_simulation_completes()
fn six_player_simulation_runs()
fn n_player_profits_sum_to_zero()
fn action_log_records_player_index()
fn blueprint_agent_reads_n_player_log()
```

**Green**: Generalize simulation to N agents. `rs_poker::arena::HoldemCompetition` already supports N players.

---

## Phase 9: Frontend UX

**Goal**: Explorer and Simulator support N-player games.

**Files**:
- `frontend/src/types.ts` — Add N-player types
- `frontend/src/Explorer.tsx` — Player switcher, position labels
- `frontend/src/Simulator.tsx` — N player slots
- `crates/tauri-app/src/exploration.rs` — N-player position tracking
- `crates/tauri-app/src/simulation.rs` — N-player sim commands

### Explorer Changes

- Replace fixed SB/BB labels with position names based on player count:
  - 2: SB, BB
  - 3: BTN/SB, BB, UTG
  - 6: BTN, SB, BB, UTG, MP, CO
- Player selector dropdown/tabs to switch viewed player
- `ExplorationPosition` gains `stacks: number[]` (array of N) and `num_players: number`
- `StrategyMatrix` gains `stacks: number[]` replacing `stack_p1`/`stack_p2`

### Simulator Changes

- Dynamic player slot list (2-6 players)
- "Add Player" / "Remove Player" buttons
- Per-player agent selection dropdown
- Results show per-player profit table + equity curves
- `SimulationProgress` and `SimulationResult` gain per-player arrays

### TDD (Rust backend)

```rust
fn exploration_position_n_player()
fn strategy_matrix_n_stacks()
fn simulation_n_player_sources()
```

Frontend changes verified manually via the Tauri app.

---

## Verification Checklist

After each phase:
1. `cargo test` — all tests pass (existing + new)
2. `cargo clippy` — no warnings
3. `cargo fmt` — formatted

End-to-end validation:
1. Train a small 3-player model: `cargo run -p poker-solver-trainer -- train -c nplayer_3p.yaml`
2. Load in explorer, navigate preflop → flop with 3-player position rotation
3. Run 3-player simulation in simulator tab
4. Verify existing 2-player bundles still load and work

## Phase Dependencies

```
Phase 1 (Player newtype) → Phase 2 (Game trait) → Phase 3 (Kuhn3 fixture)
                                                         ↓
                                                   Phase 4 (CFR solvers)
                                                         ↓
                                           Phase 5 (NPlayerPostflop) + Phase 6 (Info keys)
                                                         ↓
                                                   Phase 7 (Training)
                                                         ↓
                                                   Phase 8 (Simulation)
                                                         ↓
                                                   Phase 9 (Frontend)
```
