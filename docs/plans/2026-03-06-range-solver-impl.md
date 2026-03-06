# Range Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Output-identical reimplementation of b-inary/postflop-solver as `crates/range-solver`, with `range-solve` CLI command and 1000-game comparison harness.

**Architecture:** Self-contained workspace crate, zero dependency on `core`. Mirrors original's module structure for audit clarity. DCFR solver with exact same discount formulas, tree construction, terminal evaluation, and isomorphism handling.

**Tech Stack:** Rust, rayon (parallelism), regex (range parsing), once_cell (lazy statics)

**Reference:** Original source at `/Users/ltj/Documents/code/postflop-solver/src/`. Read the corresponding original file before implementing each task.

---

## Task 1: Scaffold Crate + Workspace Integration

**Files:**
- Create: `crates/range-solver/Cargo.toml`
- Create: `crates/range-solver/src/lib.rs`
- Modify: `Cargo.toml` (workspace members)

**Step 1: Create Cargo.toml for range-solver**

```toml
[package]
name = "range-solver"
version = "0.1.0"
edition = "2021"

[dependencies]
once_cell = "1.18"
rayon = "1.8"
regex = "1.9"

[dev-dependencies]
rand = "0.8"
```

**Step 2: Create minimal lib.rs**

```rust
pub mod card;
pub mod range;
pub mod bet_size;
pub mod action_tree;
pub mod interface;
pub mod mutex_like;
pub mod game;
pub mod solver;

// Re-export main public API
pub use card::{Card, CardConfig, NOT_DEALT};
pub use range::Range;
pub use bet_size::{BetSizeOptions, DonkSizeOptions};
pub use action_tree::{ActionTree, TreeConfig, BoardState};
pub use game::PostFlopGame;
pub use solver::{solve, solve_step, compute_exploitability};
```

**Step 3: Add to workspace**

Add `"crates/range-solver"` to workspace members in root `Cargo.toml`.

**Step 4: Verify it compiles**

Run: `cargo check -p range-solver`
Expected: Success (with dead code warnings)

**Step 5: Commit**

```
feat(range-solver): scaffold empty crate in workspace
```

---

## Task 2: Card Types + Encoding

**Files:**
- Create: `crates/range-solver/src/card.rs`
- Reference: `/Users/ltj/Documents/code/postflop-solver/src/card.rs`

**What to implement:**
- `pub type Card = u8`
- `pub const NOT_DEALT: Card = u8::MAX`
- Card encoding: `card_id = 4 * rank + suit` (rank 0=2..12=A, suit: c=0,d=1,h=2,s=3)
- `CardConfig { range: [Range; 2], flop: [Card; 3], turn: Card, river: Card }`
- `card_pair_to_index(card1, card2) -> usize` — canonical index 0..1325
- `index_to_card_pair(index) -> (Card, Card)` — inverse
- `card_to_string`, `hole_to_string`, `holes_to_strings`
- `card_from_str`, `card_from_chars`, `flop_from_str`

**Test:** Unit tests for round-trip card encoding, pair indexing, string parsing.

```rust
#[test]
fn test_card_encoding() {
    assert_eq!(card_from_str("2c").unwrap(), 0);
    assert_eq!(card_from_str("As").unwrap(), 51);
    assert_eq!(card_to_string(0).unwrap(), "2c");
    assert_eq!(card_to_string(51).unwrap(), "As");
}

#[test]
fn test_pair_index_roundtrip() {
    for i in 0..1326 {
        let (c1, c2) = index_to_card_pair(i);
        assert_eq!(card_pair_to_index(c1, c2), i);
    }
}
```

Run: `cargo test -p range-solver card`
Expected: PASS

**Commit:** `feat(range-solver): card types, encoding, and string parsing`

---

## Task 3: MutexLike + AtomicFloat

**Files:**
- Create: `crates/range-solver/src/mutex_like.rs`
- Create: `crates/range-solver/src/atomic_float.rs` (if needed, can be in mutex_like.rs)
- Reference: `/Users/ltj/Documents/code/postflop-solver/src/mutex_like.rs`, `atomic_float.rs`

**What to implement:**

MutexLike — lock-free interior mutability wrapper:
```rust
pub struct MutexLike<T: ?Sized> { data: UnsafeCell<T> }
pub struct MutexGuardLike<'a, T: ?Sized> { mutex: &'a MutexLike<T> }
```
- `new(val)`, `lock() -> MutexGuardLike`
- `Deref`/`DerefMut` on guard via unsafe `*self.mutex.data.get()`
- `unsafe impl<T: Send> Send + Sync for MutexLike<T>`

AtomicFloat:
```rust
pub(crate) struct AtomicF32(AtomicU32);
pub(crate) struct AtomicF64(AtomicU64);
```
- `new`, `load` (Relaxed), `store` (Relaxed)
- `add` on AtomicF64: CAS loop via `fetch_update`

**Test:** Basic lock/deref test, atomic add correctness.

Run: `cargo test -p range-solver mutex`
Expected: PASS

**Commit:** `feat(range-solver): MutexLike wrapper and AtomicFloat`

---

## Task 4: Range Parsing

**Files:**
- Create: `crates/range-solver/src/range.rs`
- Reference: `/Users/ltj/Documents/code/postflop-solver/src/range.rs` (1167 lines)

**What to implement:**

`Range { data: [f32; 1326] }` — weight per hole pair.

Parsing (PioSOLVER format):
- Pairs: "AA", "QQ-88", "TT+"
- Suited: "AKs", "ATs+", "A9s-A6s"
- Offsuit: "AKo", "ATo+", "KJo-K9o"
- Specific: "AsAh", "KsQh"
- Weights: "AA:0.5", "AKs:0.75"
- Comma-separated combinations

Key methods:
- `FromStr` trait — main parser
- `from_sanitized_str(s: &str) -> Result<Self, String>` — pre-cleaned
- `get_hands_weights(&self, dead_cards_mask: u64) -> (Vec<(Card, Card)>, Vec<f32>)`
- `is_empty`, `clear`, `invert`
- Weight getters/setters per pair/suited/offsuit
- `ToString` — compact range string output

**Test:** Parse standard ranges, verify weights, roundtrip.

```rust
#[test]
fn test_parse_range() {
    let range: Range = "AA,AKs,AKo".parse().unwrap();
    assert_eq!(range.get_weight_pair(12), 1.0); // AA
    assert_eq!(range.get_weight_suited(12, 11), 1.0); // AKs
    assert_eq!(range.get_weight_offsuit(12, 11), 1.0); // AKo
    assert_eq!(range.get_weight_suited(12, 10), 0.0); // AQs not in range
}
```

Run: `cargo test -p range-solver range`
Expected: PASS

**Commit:** `feat(range-solver): PioSOLVER-format range parsing`

---

## Task 5: Bet Size Parsing

**Files:**
- Create: `crates/range-solver/src/bet_size.rs`
- Reference: `/Users/ltj/Documents/code/postflop-solver/src/bet_size.rs`

**What to implement:**

```rust
pub enum BetSize {
    PotRelative(f64),      // "50%"
    PrevBetRelative(f64),  // "2.5x"
    Additive(i32, i32),    // "100c" or "20c3r"
    Geometric(i32, f64),   // "2e" or "3e200%"
    AllIn,                 // "a"
}

pub struct BetSizeOptions { pub bet: Vec<BetSize>, pub raise: Vec<BetSize> }
pub struct DonkSizeOptions { pub donk: Vec<BetSize> }
```

Parsing via `TryFrom<(&str, &str)>` for BetSizeOptions, `TryFrom<&str>` for DonkSizeOptions.

**Test:** Parse all bet size formats.

```rust
#[test]
fn test_parse_bet_sizes() {
    let opts = BetSizeOptions::try_from(("50%,100%,a", "60%,2.5x")).unwrap();
    assert_eq!(opts.bet.len(), 3);
    assert_eq!(opts.raise.len(), 2);
}
```

Run: `cargo test -p range-solver bet_size`
Expected: PASS

**Commit:** `feat(range-solver): bet size parsing (pot%, geometric, all-in)`

---

## Task 6: Action Enum + TreeConfig

**Files:**
- Create: `crates/range-solver/src/action_tree.rs` (part 1 — types only)
- Reference: `/Users/ltj/Documents/code/postflop-solver/src/action_tree.rs`

**What to implement:**

```rust
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Action {
    None, Fold, Check, Call,
    Bet(i32), Raise(i32), AllIn(i32),
    Chance(Card),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BoardState { Flop, Turn, River }

pub struct TreeConfig {
    pub initial_state: BoardState,
    pub starting_pot: i32,
    pub effective_stack: i32,
    pub rake_rate: f64,
    pub rake_cap: f64,
    pub flop_bet_sizes: [BetSizeOptions; 2],    // [OOP, IP]
    pub turn_bet_sizes: [BetSizeOptions; 2],
    pub river_bet_sizes: [BetSizeOptions; 2],
    pub turn_donk_sizes: Option<DonkSizeOptions>,
    pub river_donk_sizes: Option<DonkSizeOptions>,
    pub add_allin_threshold: f64,      // default 1.5
    pub force_allin_threshold: f64,    // default 0.15
    pub merging_threshold: f64,        // default 0.12
}
```

Player constants: `PLAYER_OOP=0`, `PLAYER_IP=1`, `PLAYER_CHANCE=2`, `PLAYER_CHANCE_FLAG=4`, `PLAYER_TERMINAL_FLAG=8`, `PLAYER_FOLD_FLAG=24`.

**Test:** Construct TreeConfig, verify defaults.

Run: `cargo test -p range-solver action_tree`
Expected: PASS

**Commit:** `feat(range-solver): Action enum, BoardState, TreeConfig`

---

## Task 7: Action Tree Building

**Files:**
- Modify: `crates/range-solver/src/action_tree.rs` (part 2 — tree logic)
- Reference: `/Users/ltj/Documents/code/postflop-solver/src/action_tree.rs`

**What to implement:**

```rust
pub(crate) struct ActionTreeNode {
    pub player: u8,
    pub board_state: BoardState,
    pub amount: i32,
    pub actions: Vec<Action>,
    pub children: Vec<MutexLike<ActionTreeNode>>,
}

pub struct ActionTree {
    config: TreeConfig,
    added_lines: Vec<Vec<Action>>,
    removed_lines: Vec<Vec<Action>>,
    root: Box<MutexLike<ActionTreeNode>>,
    history: Vec<Action>,
}
```

Key methods:
- `ActionTree::new(config: TreeConfig) -> Result<Self, String>` — builds entire betting tree
- `build_tree_recursive(node, last_action, is_check_possible, ...)` — recursive tree construction
- `push_actions(node, ...)` — generates legal actions based on bet sizes, thresholds
- `push_bet_actions(...)` — converts BetSize to concrete amounts
- Navigation: `play(action)`, `undo()`, `available_actions()`, `is_terminal_node()`, `is_chance_node()`
- `add_line(line)`, `remove_line(line)` — post-construction modification

Critical for output identity: action ordering and bet size rounding must match exactly. The original sorts actions and applies `add_allin_threshold`/`force_allin_threshold`/`merging_threshold` to deduplicate/force all-ins.

**Test:** Build a simple tree, verify action counts and structure.

```rust
#[test]
fn test_simple_tree() {
    let config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 100,
        effective_stack: 100,
        // ... simple bet sizes: 50%, all-in for both
    };
    let tree = ActionTree::new(config).unwrap();
    let actions = tree.available_actions();
    // OOP acts first: Check, Bet(50), AllIn(100)
    assert!(actions.len() >= 2);
}
```

Run: `cargo test -p range-solver action_tree`
Expected: PASS

**Commit:** `feat(range-solver): action tree construction with bet sizing`

---

## Task 8: Game/GameNode Traits (Interface)

**Files:**
- Create: `crates/range-solver/src/interface.rs`
- Reference: `/Users/ltj/Documents/code/postflop-solver/src/interface.rs`

**What to implement:**

```rust
pub trait Game: Send + Sync {
    type Node: GameNode;
    fn root(&self) -> MutexGuardLike<Self::Node>;
    fn num_private_hands(&self, player: usize) -> usize;
    fn initial_weights(&self, player: usize) -> &[f32];
    fn evaluate(
        &self, result: &mut [MaybeUninit<f32>],
        node: &Self::Node, player: usize, cfreach: &[f32],
    );
    fn chance_factor(&self, node: &Self::Node) -> usize;
    fn is_solved(&self) -> bool;
    fn set_solved(&mut self);
    fn is_ready(&self) -> bool { true }
    fn is_raked(&self) -> bool { false }
    fn is_compression_enabled(&self) -> bool { false }
    fn isomorphic_chances(&self, node: &Self::Node) -> (Vec<u8>, bool);
    fn isomorphic_swap(&self, node: &Self::Node, index: usize) -> &[Vec<(u16, u16)>; 2];
    fn locking_strategy(&self, node: &Self::Node) -> &[f32];
}

pub trait GameNode: Send + Sync {
    fn is_terminal(&self) -> bool;
    fn is_chance(&self) -> bool;
    fn player(&self) -> usize;
    fn num_actions(&self) -> usize;
    fn play(&self, action: usize) -> MutexGuardLike<Self>;
    // Strategy, regrets, cfvalues accessors...
    // Compressed variants...
    // Scale factors...
}
```

Port ALL method signatures from original exactly. This is the contract the solver depends on.

**Test:** Trait compiles (no impl yet).

Run: `cargo check -p range-solver`
Expected: Success

**Commit:** `feat(range-solver): Game and GameNode trait definitions`

---

## Task 9: PostFlopNode + PostFlopGame Structs

**Files:**
- Create: `crates/range-solver/src/game.rs` (or `src/game/mod.rs` if submodules needed)
- Reference: `/Users/ltj/Documents/code/postflop-solver/src/game/mod.rs`

**What to implement:**

PostFlopNode (`#[repr(C)]`):
```rust
#[repr(C)]
pub struct PostFlopNode {
    pub(crate) prev_action: Action,
    pub(crate) player: u8,
    pub(crate) turn: Card,
    pub(crate) river: Card,
    pub(crate) is_locked: bool,
    pub(crate) amount: i32,
    pub(crate) children_offset: u32,
    pub(crate) num_children: u16,
    pub(crate) num_elements_ip: u16,
    pub(crate) num_elements: u32,
    pub(crate) scale1: f32,
    pub(crate) scale2: f32,
    pub(crate) scale3: f32,
    pub(crate) storage1: *mut u8,
    pub(crate) storage2: *mut u8,
    pub(crate) storage3: *mut u8,
}
```

PostFlopGame — all fields from original (card_config, tree_config, action_root, private_cards, initial_weights, same_hand_index, valid_indices, hand_strength, isomorphism data, bunching data, node_arena, storage buffers, interpreter state, etc.).

Implement `GameNode` for `PostFlopNode`:
- `is_terminal()` — check `PLAYER_TERMINAL_FLAG` bit
- `is_chance()` — check `PLAYER_CHANCE_FLAG` bit
- `player()` — extract lower bits
- `num_actions()` — return `num_children`
- `play(action)` — offset-based child lookup via unsafe pointer arithmetic
- Strategy/regret/cfvalue accessors via `slice::from_raw_parts` on storage pointers
- Compressed variants returning `&[u16]`/`&[i16]`
- Scale factor getters/setters

**Test:** Create a PostFlopNode, verify field layout.

```rust
#[test]
fn test_node_size() {
    // Verify the node is the expected size (matching original's 36-byte target)
    assert!(std::mem::size_of::<PostFlopNode>() <= 64);
}
```

Run: `cargo test -p range-solver game`
Expected: PASS

**Commit:** `feat(range-solver): PostFlopNode and PostFlopGame struct definitions`

---

## Task 10: Hand Evaluator

**Files:**
- Create: `crates/range-solver/src/hand_table.rs` (lookup tables)
- Modify: `crates/range-solver/src/card.rs` or create `evaluation.rs` stub
- Reference: `/Users/ltj/Documents/code/postflop-solver/src/hand_table.rs` (if exists) or evaluation code

**What to implement:**

The original uses a pre-computed hand evaluation. Port the exact evaluator used for 7-card hand ranking. This produces `u32` strength values where higher = better hand.

The evaluator must produce **identical relative ordering** to the original. Check how `hand_strength` arrays are populated in the original's `game/interpreter.rs` or `game/evaluation.rs`.

Key: `StrengthItem { strength: u32, index: u16 }` — sorted arrays of hand strength per board runout, used for O(n) showdown evaluation.

**Test:** Evaluate known hands, verify ordering.

```rust
#[test]
fn test_hand_evaluation() {
    // Royal flush > straight flush > four of a kind > ...
    let royal = evaluate_hand([As, Ks, Qs, Js, Ts, 2c, 3d]);
    let quads = evaluate_hand([Ac, Ad, Ah, As, 2c, 3d, 4h]);
    assert!(royal > quads);
}
```

Run: `cargo test -p range-solver eval`
Expected: PASS

**Commit:** `feat(range-solver): 7-card hand evaluator`

---

## Task 11: Game Tree Building (Interpreter)

**Files:**
- Modify: `crates/range-solver/src/game.rs`
- Reference: `/Users/ltj/Documents/code/postflop-solver/src/game/interpreter.rs` (~1000 lines)

**What to implement:**

`PostFlopGame::with_config(card_config, action_tree) -> Result<Self, String>`:
1. Validate card config (no duplicate cards, valid ranges)
2. Compute `private_cards` — valid hole pairs per player (excluding board cards)
3. Compute `initial_weights` — range weights filtered by dead cards
4. Compute `same_hand_index` — map identical hands between players
5. Build `node_arena` — recursively create PostFlopNode for every game tree node
6. Compute `valid_indices` per street — hands not blocked by board cards
7. Compute `hand_strength` — per-runout sorted strength arrays
8. Compute isomorphism data (delegate to Task 13)

This is the largest single task. The node arena construction must match the original's traversal order exactly (DFS through action tree, expanding chance nodes for each possible deal).

`allocate_memory(compressed: bool)`:
- Compute total storage needed
- Allocate `storage1`, `storage2`, `storage_ip`, `storage_chance` buffers
- Assign raw pointers from each node to its slice of the storage buffers
- Same allocation order as original (critical for output identity)

`memory_usage() -> (u64, u64)` — returns (uncompressed, compressed) byte counts.

**Test:** Build a small game, verify node count and memory.

```rust
#[test]
fn test_build_small_game() {
    // River spot, simple bet sizes
    let card_config = CardConfig { /* Qs Jh 2c 8d 3s, ranges */ };
    let tree_config = TreeConfig { /* pot=100, stack=100, 50% bet */ };
    let tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
    let (mem, _) = game.memory_usage();
    assert!(mem > 0);
    game.allocate_memory(false);
}
```

Run: `cargo test -p range-solver game`
Expected: PASS

**Commit:** `feat(range-solver): game tree construction and memory allocation`

---

## Task 12: Terminal Evaluation

**Files:**
- Modify: `crates/range-solver/src/game.rs` (or `evaluation.rs` submodule)
- Reference: `/Users/ltj/Documents/code/postflop-solver/src/game/evaluation.rs` (~900 lines)

**What to implement:**

`evaluate_internal(result, node, player, cfreach)`:
- Compute `pot = starting_pot + 2 * node.amount`
- `half_pot = 0.5 * pot`, `rake = min(pot * rake_rate, rake_cap)`
- `amount_win = (half_pot - rake) / num_combinations`
- `amount_lose = -half_pot / num_combinations`

**Fold case:**
- Identify fold player via `PLAYER_FOLD_FLAG`
- Iterate opponent hands, accumulate cfvalue using inclusion-exclusion for card blocking:
  `cfreach = cfreach_sum + cfreach_same - cfreach_minus[c1] - cfreach_minus[c2]`

**Showdown (no rake):**
- Two-pass ascending/descending over sorted `hand_strength` array
- Winner gets `amount_win * cfreach`, loser gets `amount_lose * cfreach`
- Inclusion-exclusion for blocking

**Showdown (raked):**
- Three-pass with separate `cfreach_sum_win`, `cfreach_sum_tie`
- Accounts for ties splitting pot minus rake

**Critical:** Use `f64` for intermediate summations, cast to `f32` for storage. Match the original's iteration order over sorted strength arrays exactly.

Implement `Game::evaluate()` for PostFlopGame delegating to `evaluate_internal`.

**Test:** Set up a river spot with known hands, verify payoff values.

Run: `cargo test -p range-solver eval`
Expected: PASS

**Commit:** `feat(range-solver): terminal evaluation (fold + showdown + rake)`

---

## Task 13: Isomorphism

**Files:**
- Create: `crates/range-solver/src/isomorphism.rs`
- Reference: `/Users/ltj/Documents/code/postflop-solver/src/isomorphism.rs`

**What to implement:**

Suit isomorphism detection:
- Build `suit_isomorphism[4]` — map suits to canonical representatives
- Check: both ranges have identical rank distribution per suit AND suit-isomorphic

Turn isomorphism:
- `isomorphism_ref_turn: Vec<u8>` — reference indices
- `isomorphism_card_turn: Vec<Card>` — cards to skip
- For each suit pair: if flop rankset identical & suit_isomorphism matches, mark as isomorphic

River isomorphism:
- `isomorphism_ref_river: Vec<Vec<u8>>` — per-turn reference indices
- `isomorphism_card_river: [Vec<Card>; 4]` — per-suit skip lists

Swap table generation:
- `isomorphism_swap_internal(suit1, suit2, private_cards)` — generates `(old_idx, new_idx)` pairs for hands that swap when substituting suits
- Card replacement: `(card - suit1 + suit2)` if `card & 3 == suit1`, vice versa

Implement `Game::isomorphic_chances()` and `Game::isomorphic_swap()` for PostFlopGame.

**Test:** Monotone flop should have 3 isomorphic suits for turn.

```rust
#[test]
fn test_monotone_flop_isomorphism() {
    // Flop: Ah Kh Qh — all hearts
    // Turn: any non-heart suit should be isomorphic to each other
    // Verify ref indices group 3 suits together
}
```

Run: `cargo test -p range-solver isomorphism`
Expected: PASS

**Commit:** `feat(range-solver): suit isomorphism detection and swap tables`

---

## Task 14: DCFR Solver

**Files:**
- Create: `crates/range-solver/src/solver.rs`
- Reference: `/Users/ltj/Documents/code/postflop-solver/src/solver.rs` (~370 lines)

**What to implement:**

**Exact discount formulas:**
```rust
let nearest_lower_power_of_4 = match current_iteration {
    0 => 0,
    x => 1 << ((x.leading_zeros() ^ 31) & !1),
};
let t_alpha = (current_iteration as i32 - 1).max(0) as f64;
let t_gamma = (current_iteration - nearest_lower_power_of_4) as f64;
let pow_alpha = t_alpha * t_alpha.sqrt(); // t^1.5
let pow_gamma = (t_gamma / (t_gamma + 1.0)).powi(3); // (t/(t+1))^3

alpha_t = (pow_alpha / (pow_alpha + 1.0)) as f32;
beta_t = 0.5_f32; // constant
gamma_t = pow_gamma as f32;
```

**Core functions:**
- `solve(game, max_iterations, target_exploitability, print_progress) -> f32`
- `solve_step(game, current_iteration)` — single iteration, alternates players
- `solve_recursive(game, node, player, cfreach, result, ...)` — recursive CFR traversal
- `compute_exploitability(game) -> f32`
- `regret_matching(regret, num_actions) -> Vec<f32>` — RMP: `max(r, 0)`, normalize per-hand
- `regret_matching_compressed(regret: &[i16], num_actions) -> Vec<f32>`
- `finalize(game)` — normalize cumulative strategy to final probabilities

**Solve recursive logic:**
1. Terminal → `game.evaluate()`
2. Chance → sum over isomorphic children, apply swap tables, divide cfreach by `chance_factor`
3. Decision (current player) → regret matching, recurse all actions, update regrets with discounting
4. Decision (opponent) → apply strategy to cfreach, recurse all actions, sum cfvalues

**Regret update (exact):**
```rust
// After recursing all actions:
for i in 0..num_hands {
    for a in 0..num_actions {
        let regret = cfv_actions[a][i] - result[i]; // action value minus node value
        let cum = cum_regrets[a * num_hands + i];
        cum_regrets[a * num_hands + i] = if cum > 0.0 {
            alpha_t * cum + regret
        } else {
            beta_t * cum + regret
        };
    }
}
// Strategy update:
for a in 0..num_actions {
    cum_strategy[a * num_hands + i] = gamma_t * cum_strategy[a * num_hands + i] + strategy[a * num_hands + i];
}
```

**Parallelization:** Use rayon to parallelize over root actions when `enable_parallelization()` returns true.

**Test:** Solve a trivial river spot, verify exploitability decreases.

```rust
#[test]
fn test_solver_convergence() {
    // Build a small river game
    // Run 100 iterations
    // Verify exploitability < 5.0
    let expl = solve(&mut game, 100, 0.0, false);
    assert!(expl < 5.0);
}
```

Run: `cargo test -p range-solver solver`
Expected: PASS

**Commit:** `feat(range-solver): DCFR solver with exact discount formulas`

---

## Task 15: Public Query API

**Files:**
- Modify: `crates/range-solver/src/game.rs`
- Reference: `/Users/ltj/Documents/code/postflop-solver/src/game/base.rs`

**What to implement:**

Implement `Game` trait for PostFlopGame, plus these public query methods:

- `private_cards(player) -> &[(Card, Card)]`
- `strategy() -> &[f32]` — action probabilities at current node
- `expected_values(player) -> &[f32]` — per-hand EV
- `equity(player) -> &[f32]` — per-hand equity [0,1]
- `normalized_weights(player) -> &[f32]` — reach probabilities
- `cache_normalized_weights()` — compute weights for EV/equity queries
- `available_actions() -> &[Action]`
- `is_chance_node() -> bool`
- `possible_cards() -> u64` — bitmask of dealable cards
- `play(action_index)` — navigate into child
- `back_to_root()`
- `total_bet_amount() -> [i32; 2]`

**Test:** Solve a game, navigate tree, query strategies.

Run: `cargo test -p range-solver game`
Expected: PASS

**Commit:** `feat(range-solver): public query API (strategy, EV, equity)`

---

## Task 16: End-to-End Validation Against Original Example

**Files:**
- Create: `crates/range-solver/examples/basic.rs`
- Reference: `/Users/ltj/Documents/code/postflop-solver/examples/basic.rs`

**What to implement:**

Port the original's `basic.rs` example exactly. Same ranges, same board, same bet sizes, same iterations. Print same output format.

Run both and diff output manually:
```bash
cd /Users/ltj/Documents/code/postflop-solver && cargo run --release --example basic 2>&1 > /tmp/original.txt
cd /Users/ltj/Documents/code/poker_solver_rust && cargo run -p range-solver --release --example basic 2>&1 > /tmp/ours.txt
diff /tmp/original.txt /tmp/ours.txt
```

Expected: Identical output.

**Commit:** `feat(range-solver): basic example matching original output`

---

## Task 17: CLI Subcommand

**Files:**
- Modify: `crates/trainer/Cargo.toml` (add range-solver dep)
- Modify: `crates/trainer/src/main.rs` (add RangeSolve variant)
- Reference: Design doc CLI interface section

**What to implement:**

Add `RangeSolve` to the `Commands` enum in trainer:

```rust
/// Solve a postflop spot with exact (no abstraction) DCFR
RangeSolve {
    #[arg(long)] oop_range: String,
    #[arg(long)] ip_range: String,
    #[arg(long)] flop: String,
    #[arg(long)] turn: Option<String>,
    #[arg(long)] river: Option<String>,
    #[arg(long, default_value = "100")] pot: i32,
    #[arg(long, default_value = "100")] effective_stack: i32,
    #[arg(long, default_value = "1000")] iterations: u32,
    #[arg(long, default_value = "0.5")] target_exploitability: f32,
    #[arg(long)] config: Option<PathBuf>,
    #[arg(long)] compressed: bool,
},
```

Handler: parse args → build CardConfig + TreeConfig → ActionTree::new() → PostFlopGame::with_config() → allocate_memory() → solve() → print results.

**Test:** Run CLI with simple args.

```bash
cargo run -p poker-solver-trainer --release -- range-solve \
  --oop-range "QQ+,AKs" --ip-range "22+,A2s+" \
  --flop "Qs Jh 2c" --pot 100 --effective-stack 200 --iterations 100
```

Expected: Prints iterations + exploitability + strategy.

**Commit:** `feat(trainer): range-solve CLI subcommand`

---

## Task 18: Comparison Crate + Identity Test

**Files:**
- Create: `crates/range-solver-compare/Cargo.toml`
- Create: `crates/range-solver-compare/src/lib.rs`
- Create: `crates/range-solver-compare/tests/identity.rs`

**Cargo.toml:**
```toml
[package]
name = "range-solver-compare"
version = "0.1.0"
edition = "2021"

[dependencies]
range-solver = { path = "../range-solver" }
postflop-solver = { path = "../../external/postflop-solver" }
rand = "0.8"
rand_chacha = "0.3"
```

Note: symlink `/Users/ltj/Documents/code/postflop-solver` to `external/postflop-solver` in workspace root (or use absolute path). Easy to remove later.

**src/lib.rs — Config generation + adapter layer:**

```rust
pub struct TestConfig {
    pub oop_range: String,
    pub ip_range: String,
    pub flop: [u8; 3],
    pub turn: Option<u8>,
    pub river: Option<u8>,
    pub pot: i32,
    pub stack: i32,
    pub bet_pct: Vec<f64>,
    pub raise_pct: Vec<f64>,
}

pub fn generate_configs(n: usize, seed: u64) -> Vec<TestConfig> { ... }

pub fn run_ours(config: &TestConfig, iterations: u32) -> SolveResult { ... }
pub fn run_original(config: &TestConfig, iterations: u32) -> SolveResult { ... }

pub struct SolveResult {
    pub exploitability: f32,
    pub root_strategy: Vec<f32>,
    pub ev_oop: Vec<f32>,
    pub ev_ip: Vec<f32>,
    pub equity_oop: Vec<f32>,
    pub equity_ip: Vec<f32>,
}
```

**tests/identity.rs:**

```rust
#[test]
fn test_1000_configs_identical() {
    let configs = generate_configs(1000, 42);
    let mut failures = Vec::new();

    for (i, config) in configs.iter().enumerate() {
        let ours = run_ours(config, 200);
        let original = run_original(config, 200);

        if ours.exploitability != original.exploitability
            || ours.root_strategy != original.root_strategy
            || ours.ev_oop != original.ev_oop
        {
            failures.push((i, config.clone(), ours, original));
        }
    }

    if !failures.is_empty() {
        for (i, config, ours, orig) in &failures {
            eprintln!("MISMATCH config #{i}: expl ours={} orig={}", ours.exploitability, orig.exploitability);
        }
        panic!("{} / 1000 configs mismatched", failures.len());
    }
}
```

**Config generation rules:**
- Seed: 42 (deterministic via `rand_chacha::ChaCha8Rng`)
- Ranges: pick from ~20 common range strings (tight to wide)
- Flops: random 3 cards from deck
- Turn: 60% chance of random non-conflicting card
- River: 50% chance if turn exists
- Pot: uniform 20-500
- Stack: uniform 100-2000
- Bet sizes: sample from [0.33, 0.5, 0.67, 1.0, 1.5]

Run: `cargo test -p range-solver-compare --release -- --test-threads=1 test_1000_configs_identical`
Expected: PASS (0 mismatches)
Target: Under 5 minutes in release mode.

**Commit:** `feat(range-solver-compare): 1000-game identity test harness`

---

## Task 19: Performance Benchmark

**Files:**
- Create: `crates/range-solver-compare/benches/perf.rs` or add to `tests/`

**What to implement:**

10 representative configs (mix of flop/turn/river spots, small and large ranges), each solved for 1000 iterations. Time both solvers, compare.

```rust
#[test]
fn test_performance_parity() {
    let configs = generate_configs(10, 99); // different seed, diverse spots

    for config in &configs {
        let t1 = Instant::now();
        run_ours(config, 1000);
        let ours_ms = t1.elapsed().as_millis();

        let t2 = Instant::now();
        run_original(config, 1000);
        let orig_ms = t2.elapsed().as_millis();

        let ratio = ours_ms as f64 / orig_ms.max(1) as f64;
        eprintln!("Config: ours={ours_ms}ms orig={orig_ms}ms ratio={ratio:.2}x");
        assert!(ratio < 1.5, "Performance regression: {ratio:.2}x slower");
    }
}
```

Run: `cargo test -p range-solver-compare --release test_performance_parity`
Expected: All configs within 1.5x of original.

**Commit:** `feat(range-solver-compare): performance parity benchmark`

---

## Execution Order & Dependencies

```
Task 1  (scaffold)
  ├─ Task 2  (card)
  ├─ Task 3  (mutex/atomic)
  │
  ├─ Task 4  (range) ← depends on 2
  ├─ Task 5  (bet_size)
  ├─ Task 6  (action types) ← depends on 2, 5
  │
  ├─ Task 7  (action tree) ← depends on 3, 6
  ├─ Task 8  (interface traits) ← depends on 3
  ├─ Task 10 (hand evaluator) ← depends on 2
  │
  ├─ Task 9  (node/game structs) ← depends on 2, 3, 6, 8
  ├─ Task 11 (tree building) ← depends on 7, 9, 10
  ├─ Task 13 (isomorphism) ← depends on 2, 9
  ├─ Task 12 (terminal eval) ← depends on 9, 10
  │
  ├─ Task 14 (solver) ← depends on 8, 9, 12, 13
  ├─ Task 15 (query API) ← depends on 9, 14
  │
  ├─ Task 16 (e2e validation) ← depends on all above
  ├─ Task 17 (CLI) ← depends on 15
  └─ Task 18-19 (comparison) ← depends on 16
```

Parallelizable groups:
- **Group A** (no deps): Tasks 2, 3, 5
- **Group B** (after A): Tasks 4, 6, 8, 10
- **Group C** (after B): Tasks 7, 9, 13
- **Group D** (after C): Tasks 11, 12
- **Group E** (after D): Tasks 14, 15
- **Group F** (after E): Tasks 16, 17, 18, 19
