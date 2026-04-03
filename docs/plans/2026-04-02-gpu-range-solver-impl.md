# GPU Range Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** New `gpu-range-solve` CLI command implementing GPUGT-style level-synchronous DCFR on GPU via burn, matching the existing `range-solve` interface.

**Architecture:** Reuse `range-solver`'s `PostFlopGame` for tree construction on CPU, extract topology into burn tensors, run DCFR iterations entirely on GPU with per-street batching (batch dim = board runouts). Edge-based tensor layout avoids variable-action padding.

**Tech Stack:** Rust, burn 0.16 (wgpu + optional cuda-jit), range-solver crate for tree building

**Design doc:** `docs/plans/2026-04-02-gpu-range-solver-design.md`

---

## Task 1: Add Public Node Accessors to range-solver

The GPU solver needs to inspect `PostFlopNode` internals that are currently `pub(crate)`. Add targeted public accessors rather than changing visibility of raw fields.

**Files:**
- Modify: `crates/range-solver/src/game/node.rs` (after line 43, inside `impl PostFlopNode`)
- Modify: `crates/range-solver/src/game/mod.rs` (add accessors to `PostFlopGame`, before `#[cfg(test)]` block at line 236)
- Modify: `crates/range-solver/src/action_tree.rs` (make flag constants `pub`)
- Test: `crates/range-solver/src/game/node.rs` (extend existing test module)

**Step 1: Write the failing tests**

In `crates/range-solver/src/game/node.rs`, add to the existing `#[cfg(test)] mod tests` block:

```rust
#[test]
fn node_public_accessors() {
    let mut node = PostFlopNode::default();
    node.player = PLAYER_FOLD_FLAG | PLAYER_OOP;
    node.amount = 50;
    node.turn = 10;
    node.river = 20;
    node.prev_action = Action::Fold;

    assert!(node.is_fold());
    assert!(!node.is_showdown());
    assert_eq!(node.bet_amount(), 50);
    assert_eq!(node.turn_card(), 10);
    assert_eq!(node.river_card(), 20);
    assert_eq!(node.prev_action(), Action::Fold);
    assert_eq!(node.acting_player(), 0);  // PLAYER_OOP from PLAYER_MASK

    // Showdown node
    let mut sd_node = PostFlopNode::default();
    sd_node.player = PLAYER_TERMINAL_FLAG;
    sd_node.river = 20;
    assert!(!sd_node.is_fold());
    assert!(sd_node.is_showdown());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p range-solver -- node_public_accessors`
Expected: Compilation error — `is_fold`, `is_showdown`, `bet_amount`, etc. don't exist

**Step 3: Implement the accessors**

In `crates/range-solver/src/action_tree.rs`, change lines 11-17 from `pub(crate)` to `pub`:

```rust
pub const PLAYER_OOP: u8 = 0;
pub const PLAYER_IP: u8 = 1;
pub const PLAYER_CHANCE: u8 = 2;
pub const PLAYER_MASK: u8 = 3;
pub const PLAYER_CHANCE_FLAG: u8 = 4;
pub const PLAYER_TERMINAL_FLAG: u8 = 8;
pub const PLAYER_FOLD_FLAG: u8 = 24;
```

In `crates/range-solver/src/game/node.rs`, add after the `is_depth_boundary` method (after line 43):

```rust
/// Returns true if this is a fold terminal.
#[inline]
pub fn is_fold(&self) -> bool {
    self.player & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG
}

/// Returns true if this is a showdown terminal (river reached, not fold).
#[inline]
pub fn is_showdown(&self) -> bool {
    self.is_terminal() && !self.is_fold() && !self.is_depth_boundary()
}

/// Returns the acting player (0=OOP, 1=IP) extracted from the player flags.
#[inline]
pub fn acting_player(&self) -> usize {
    (self.player & PLAYER_MASK) as usize
}

/// Returns the previous action that led to this node.
#[inline]
pub fn prev_action(&self) -> Action {
    self.prev_action
}

/// Returns the total bet amount at this node.
#[inline]
pub fn bet_amount(&self) -> i32 {
    self.amount
}

/// Returns the turn card at this node (or `NOT_DEALT`).
#[inline]
pub fn turn_card(&self) -> Card {
    self.turn
}

/// Returns the river card at this node (or `NOT_DEALT`).
#[inline]
pub fn river_card(&self) -> Card {
    self.river
}

/// Returns the raw player byte (for flag inspection).
#[inline]
pub fn player_flags(&self) -> u8 {
    self.player
}
```

In `crates/range-solver/src/game/mod.rs`, add before the `#[cfg(test)]` block (before line 236):

```rust
impl PostFlopGame {
    /// Returns the number of nodes in the tree arena.
    pub fn num_nodes(&self) -> usize {
        self.node_arena.len()
    }

    /// Locks and returns a reference to the node at `index`.
    pub fn node(&self, index: usize) -> crate::mutex_like::MutexGuardLike<'_, PostFlopNode> {
        self.node_arena[index].lock()
    }

    /// Returns the arena index of a node (pointer arithmetic from arena base).
    pub fn node_index(&self, node: &PostFlopNode) -> usize {
        let base = self.node_arena.as_ptr() as usize;
        let node_ptr = node as *const PostFlopNode as usize;
        let stride = std::mem::size_of::<crate::mutex_like::MutexLike<PostFlopNode>>();
        (node_ptr - base) / stride
    }

    /// Returns the same-hand index mapping for `player`.
    /// For hand `i`, `same_hand_index[i]` is the index of the same hand in
    /// the opponent's private cards, or `u16::MAX` if not present.
    pub fn same_hand_index(&self, player: usize) -> &[u16] {
        &self.same_hand_index[player]
    }

    /// Returns the hand strength ordering for a given board.
    /// The outer vec is indexed by `card_pair_to_index(turn, river)`.
    /// Each inner array has two vecs (one per player) of `StrengthItem`
    /// sorted in ascending strength order, with sentinels at both ends.
    pub fn hand_strength(&self) -> &Vec<[Vec<StrengthItem>; 2]> {
        &self.hand_strength
    }

    /// Returns valid hand indices for the flop board.
    pub fn valid_indices_flop(&self) -> &[Vec<u16>; 2] {
        &self.valid_indices_flop
    }

    /// Returns valid hand indices indexed by turn card.
    pub fn valid_indices_turn(&self) -> &Vec<[Vec<u16>; 2]> {
        &self.valid_indices_turn
    }

    /// Returns valid hand indices indexed by card_pair_to_index(turn, river).
    pub fn valid_indices_river(&self) -> &Vec<[Vec<u16>; 2]> {
        &self.valid_indices_river
    }

    /// Returns the number of valid card combinations.
    pub fn num_combinations_f64(&self) -> f64 {
        self.num_combinations
    }
}
```

Also make `StrengthItem` public — in `crates/range-solver/src/game/mod.rs`, line 53:
```rust
#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct StrengthItem {
    pub strength: u16,
    pub index: u16,
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p range-solver`
Expected: All tests pass including the new `node_public_accessors` test

**Step 5: Commit**

```bash
git add crates/range-solver/
git commit -m "feat(range-solver): add public accessors for GPU solver integration"
```

---

## Task 2: Scaffold gpu-range-solver Crate

**Files:**
- Create: `crates/gpu-range-solver/Cargo.toml`
- Create: `crates/gpu-range-solver/src/lib.rs`
- Modify: `Cargo.toml` (workspace members)

**Step 1: Create Cargo.toml**

```toml
[package]
name = "gpu-range-solver"
version.workspace = true
edition.workspace = true

[dependencies]
range-solver = { path = "../range-solver" }
burn = { version = "0.16", features = ["wgpu"] }

[features]
default = []
cuda = ["burn/cuda-jit"]

[dev-dependencies]
rand = "0.8"
```

**Step 2: Create lib.rs**

```rust
pub mod extract;
pub mod tensors;
pub mod solver;
pub mod terminal;

use burn::prelude::*;

/// Configuration for the GPU range solver.
pub struct GpuSolverConfig {
    pub max_iterations: u32,
    pub target_exploitability: f32,
    pub print_progress: bool,
}

/// Result of a GPU solve.
pub struct GpuSolveResult {
    /// Final exploitability achieved.
    pub exploitability: f32,
    /// Number of iterations run.
    pub iterations_run: u32,
    /// Average strategy at root: `[num_actions × num_hands]` in row-major order.
    pub root_strategy: Vec<f32>,
}
```

**Step 3: Create empty module files**

Create `crates/gpu-range-solver/src/extract.rs`:
```rust
//! Extracts PostFlopGame tree topology into flat arrays for GPU tensor creation.
```

Create `crates/gpu-range-solver/src/tensors.rs`:
```rust
//! GPU tensor layout definitions for the level-synchronous DCFR solver.
```

Create `crates/gpu-range-solver/src/solver.rs`:
```rust
//! Level-synchronous DCFR iteration loop on GPU.
```

Create `crates/gpu-range-solver/src/terminal.rs`:
```rust
//! Fold and showdown terminal evaluation as GPU tensor operations.
```

**Step 4: Add to workspace**

In root `Cargo.toml`, line 3, add `"crates/gpu-range-solver"` to members:
```toml
members = ["crates/core", "crates/tauri-app", "crates/trainer", "crates/test-macros", "crates/devserver", "crates/range-solver", "crates/cfvnet", "crates/convergence-harness", "crates/rebel", "crates/gpu-range-solver"]
```

**Step 5: Verify it compiles**

Run: `cargo check -p gpu-range-solver`
Expected: Compiles with no errors

**Step 6: Commit**

```bash
git add crates/gpu-range-solver/ Cargo.toml
git commit -m "feat: scaffold gpu-range-solver crate with burn dependency"
```

---

## Task 3: Tree Extraction — Topology

Walk the `PostFlopGame` node arena via the public accessors from Task 1. Produce a `TreeTopology` struct containing flat arrays of node types, edge parent/child indices, and level groupings.

**Files:**
- Modify: `crates/gpu-range-solver/src/extract.rs`
- Test: inline `#[cfg(test)]` in extract.rs

**Step 1: Write the failing test**

In `crates/gpu-range-solver/src/extract.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::CardConfig;
    use range_solver::PostFlopGame;

    fn make_river_game() -> PostFlopGame {
        let oop_range = "AA".parse().unwrap();
        let ip_range = "KK".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: [48, 36, 24], // Ac Kd Qh
            turn: 12,           // Jc
            river: 0,           // 2c
        };
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 100,
            river_bet_sizes: [
                BetSizeOptions::try_from(("100%", "")).unwrap(),
                BetSizeOptions::try_from(("100%", "")).unwrap(),
            ],
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.1,
            ..Default::default()
        };
        let action_tree = ActionTree::new(tree_config).unwrap();
        PostFlopGame::with_config(card_config, action_tree).unwrap()
    }

    #[test]
    fn extract_river_topology() {
        let game = make_river_game();
        let topo = extract_topology(&game);

        // Must have nodes and edges
        assert!(topo.num_nodes > 0);
        assert!(topo.num_edges > 0);

        // Root is always node 0
        assert!(topo.num_nodes > 1);

        // Every edge has valid parent/child
        for e in 0..topo.num_edges {
            assert!(topo.edge_parent[e] < topo.num_nodes);
            assert!(topo.edge_child[e] < topo.num_nodes);
        }

        // Level grouping covers all nodes
        let total_in_levels: usize = topo.level_nodes.iter().map(|l| l.len()).sum();
        assert_eq!(total_in_levels, topo.num_nodes);

        // Root is at depth 0
        assert!(topo.level_nodes[0].contains(&0));

        // For a river game there should be no chance nodes
        assert!(topo.chance_nodes.is_empty());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p gpu-range-solver -- extract_river_topology`
Expected: Compilation error — `extract_topology`, `TreeTopology` don't exist

**Step 3: Implement tree extraction**

In `crates/gpu-range-solver/src/extract.rs`:

```rust
use range_solver::action_tree::{PLAYER_FOLD_FLAG, PLAYER_TERMINAL_FLAG, PLAYER_CHANCE_FLAG, PLAYER_MASK};
use range_solver::PostFlopGame;
use range_solver::card::NOT_DEALT;
use std::collections::VecDeque;

/// Classification of a node in the game tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    /// Fold terminal — one player folded.
    Fold { folded_player: usize },
    /// Showdown terminal — both players reached showdown.
    Showdown,
    /// Chance node — deals a card (turn or river).
    Chance,
    /// Decision node — a player acts.
    Player { player: usize },
}

/// Flat topology extracted from a PostFlopGame tree.
#[derive(Debug)]
pub struct TreeTopology {
    pub num_nodes: usize,
    pub num_edges: usize,

    // Per-node data (indexed by node_id 0..num_nodes)
    pub node_type: Vec<NodeType>,
    pub node_depth: Vec<usize>,
    pub node_arena_index: Vec<usize>,  // maps our node_id → PostFlopGame arena index
    pub node_amount: Vec<i32>,         // pot contribution at this node
    pub node_turn: Vec<u8>,            // turn card at this node (or NOT_DEALT)
    pub node_river: Vec<u8>,           // river card at this node (or NOT_DEALT)
    pub node_num_actions: Vec<usize>,  // number of child actions (0 for terminals)

    // Per-edge data (indexed by edge_id 0..num_edges)
    pub edge_parent: Vec<usize>,       // node_id of parent
    pub edge_child: Vec<usize>,        // node_id of child
    pub edge_action_index: Vec<usize>, // action index within parent (0..num_actions-1)

    // Level groupings
    pub max_depth: usize,
    pub level_nodes: Vec<Vec<usize>>,  // level_nodes[depth] = list of node_ids
    pub level_edges: Vec<Vec<usize>>,  // level_edges[depth] = edges whose CHILD is at this depth

    // Classified node lists for fast iteration
    pub fold_nodes: Vec<usize>,
    pub showdown_nodes: Vec<usize>,
    pub chance_nodes: Vec<usize>,
    pub player_nodes: [Vec<usize>; 2], // player_nodes[p] = decision nodes for player p
}

/// Extracts flat topology from a PostFlopGame via BFS.
pub fn extract_topology(game: &PostFlopGame) -> TreeTopology {
    let arena_len = game.num_nodes();

    // Map from arena index → our sequential node_id
    let mut arena_to_node: Vec<Option<usize>> = vec![None; arena_len];

    let mut node_type = Vec::new();
    let mut node_depth = Vec::new();
    let mut node_arena_index = Vec::new();
    let mut node_amount = Vec::new();
    let mut node_turn = Vec::new();
    let mut node_river = Vec::new();
    let mut node_num_actions = Vec::new();

    let mut edge_parent = Vec::new();
    let mut edge_child = Vec::new();
    let mut edge_action_index = Vec::new();

    let mut fold_nodes = Vec::new();
    let mut showdown_nodes = Vec::new();
    let mut chance_nodes = Vec::new();
    let mut player_nodes: [Vec<usize>; 2] = [Vec::new(), Vec::new()];

    // BFS queue: (arena_index, depth)
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    queue.push_back((0, 0));

    let mut max_depth: usize = 0;

    while let Some((arena_idx, depth)) = queue.pop_front() {
        if arena_to_node[arena_idx].is_some() {
            continue; // already visited
        }

        let node_id = node_type.len();
        arena_to_node[arena_idx] = Some(node_id);

        let node = game.node(arena_idx);
        let flags = node.player_flags();
        let is_terminal = flags & PLAYER_TERMINAL_FLAG != 0;
        let is_chance = flags & PLAYER_CHANCE_FLAG != 0;
        let is_fold = flags & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG;

        let ntype = if is_fold {
            let folded = (flags & PLAYER_MASK) as usize;
            NodeType::Fold { folded_player: folded }
        } else if is_terminal {
            NodeType::Showdown
        } else if is_chance {
            NodeType::Chance
        } else {
            let p = (flags & PLAYER_MASK) as usize;
            NodeType::Player { player: p }
        };

        let n_actions = if is_terminal { 0 } else { node.num_actions() };
        let amount = node.bet_amount();
        let turn = node.turn_card();
        let river = node.river_card();
        drop(node); // release lock before accessing children

        node_type.push(ntype);
        node_depth.push(depth);
        node_arena_index.push(arena_idx);
        node_amount.push(amount);
        node_turn.push(turn);
        node_river.push(river);
        node_num_actions.push(n_actions);
        max_depth = max_depth.max(depth);

        // Classify
        match ntype {
            NodeType::Fold { .. } => fold_nodes.push(node_id),
            NodeType::Showdown => showdown_nodes.push(node_id),
            NodeType::Chance => chance_nodes.push(node_id),
            NodeType::Player { player } => player_nodes[player].push(node_id),
        }

        // Enqueue children and create edges
        if n_actions > 0 {
            let children = game.child_indices(arena_idx);
            for (action_idx, &child_arena_idx) in children.iter().enumerate() {
                let child_id = node_type.len() + queue.len(); // approximate, fixed below
                edge_parent.push(node_id);
                edge_child.push(0); // placeholder, filled after BFS
                edge_action_index.push(action_idx);
                queue.push_back((child_arena_idx, depth + 1));
            }
        }
    }

    // Fix edge_child: re-scan edges and map arena indices to node_ids
    // We need to re-derive child arena indices
    let mut edge_idx = 0;
    for node_id in 0..node_type.len() {
        let n_actions = node_num_actions[node_id];
        if n_actions > 0 {
            let arena_idx = node_arena_index[node_id];
            let children = game.child_indices(arena_idx);
            for &child_arena in &children {
                edge_child[edge_idx] = arena_to_node[child_arena]
                    .expect("child must have been visited");
                edge_idx += 1;
            }
        }
    }

    let num_nodes = node_type.len();
    let num_edges = edge_parent.len();

    // Build level groupings
    let mut level_nodes = vec![Vec::new(); max_depth + 1];
    for (id, &d) in node_depth.iter().enumerate() {
        level_nodes[d].push(id);
    }

    let mut level_edges = vec![Vec::new(); max_depth + 1];
    for (eid, &child) in edge_child.iter().enumerate() {
        level_edges[node_depth[child]].push(eid);
    }

    TreeTopology {
        num_nodes,
        num_edges,
        node_type,
        node_depth,
        node_arena_index,
        node_amount,
        node_turn,
        node_river,
        node_num_actions,
        edge_parent,
        edge_child,
        edge_action_index,
        max_depth,
        level_nodes,
        level_edges,
        fold_nodes,
        showdown_nodes,
        chance_nodes,
        player_nodes,
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p gpu-range-solver -- extract_river_topology`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/gpu-range-solver/src/extract.rs
git commit -m "feat(gpu-range-solver): tree topology extraction via BFS"
```

---

## Task 4: Terminal Data Extraction (Fold + Showdown)

Pre-compute per-deal terminal evaluation data: fold payoffs (constant part) and showdown outcome matrices.

**Files:**
- Modify: `crates/gpu-range-solver/src/extract.rs` (add terminal data extraction)
- Test: inline `#[cfg(test)]`

**Step 1: Write the failing test**

```rust
#[test]
fn extract_terminal_data_river() {
    let game = make_river_game();
    game.allocate_memory(false);
    let topo = extract_topology(&game);
    let term = extract_terminal_data(&game, &topo);

    let num_hands_oop = game.private_cards(0).len();
    let num_hands_ip = game.private_cards(1).len();

    // Should have fold data for each fold node
    assert_eq!(term.fold_payoffs.len(), topo.fold_nodes.len());

    // Each fold payoff has a pot_win and pot_lose amount
    for fp in &term.fold_payoffs {
        assert!(fp.amount_win > 0.0);
        assert!(fp.amount_lose < 0.0);
    }

    // Showdown outcome matrices should exist for each showdown node
    assert_eq!(term.showdown_outcomes.len(), topo.showdown_nodes.len());

    // Card-index mappings should cover all hands
    assert_eq!(term.hand_cards[0].len(), num_hands_oop);
    assert_eq!(term.hand_cards[1].len(), num_hands_ip);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p gpu-range-solver -- extract_terminal_data_river`
Expected: Compilation error

**Step 3: Implement**

Add to `crates/gpu-range-solver/src/extract.rs`:

```rust
/// Pre-computed fold terminal data.
pub struct FoldData {
    pub folded_player: usize,
    pub amount_win: f64,   // payoff for winner per combo
    pub amount_lose: f64,  // payoff for loser per combo
}

/// Pre-computed showdown outcome matrix for one board.
/// `outcomes[h_player * num_opp + h_opp]` = +1 (win), -1 (loss), 0 (tie or blocked).
pub struct ShowdownData {
    pub num_player_hands: [usize; 2],
    /// For player 0 (OOP) perspective: [num_oop × num_ip] outcome values.
    pub outcome_matrix_p0: Vec<f64>,
    /// Pot payoff amounts: win, tie, lose (per combo, same for both players except sign).
    pub amount_win: f64,
    pub amount_tie: f64,
    pub amount_lose: f64,
}

/// All terminal evaluation data needed by the GPU solver.
pub struct TerminalData {
    /// Fold data for each fold node (same order as topo.fold_nodes).
    pub fold_payoffs: Vec<FoldData>,
    /// Showdown data for each showdown node (same order as topo.showdown_nodes).
    pub showdown_outcomes: Vec<ShowdownData>,
    /// Per-player: hand index → (card1, card2).
    pub hand_cards: [Vec<(u8, u8)>; 2],
    /// Per-player: same_hand_index[i] = opponent index or u16::MAX.
    pub same_hand_index: [Vec<u16>; 2],
    /// Number of valid card combos (for payoff normalization).
    pub num_combinations: f64,
}

/// Extracts terminal evaluation data from the game.
pub fn extract_terminal_data(game: &PostFlopGame, topo: &TreeTopology) -> TerminalData {
    let tree_config = game.tree_config();
    let num_combinations = game.num_combinations_f64();

    // Hand card mappings
    let hand_cards: [Vec<(u8, u8)>; 2] = [
        game.private_cards(0).to_vec(),
        game.private_cards(1).to_vec(),
    ];
    let same_hand_index: [Vec<u16>; 2] = [
        game.same_hand_index(0).to_vec(),
        game.same_hand_index(1).to_vec(),
    ];

    // Fold data
    let fold_payoffs: Vec<FoldData> = topo.fold_nodes.iter().map(|&node_id| {
        let amount = topo.node_amount[node_id];
        let pot = (tree_config.starting_pot + 2 * amount) as f64;
        let half_pot = 0.5 * pot;
        let rake = (pot * tree_config.rake_rate).min(tree_config.rake_cap);
        let folded_player = match topo.node_type[node_id] {
            NodeType::Fold { folded_player } => folded_player,
            _ => unreachable!(),
        };
        FoldData {
            folded_player,
            amount_win: (half_pot - rake) / num_combinations,
            amount_lose: -half_pot / num_combinations,
        }
    }).collect();

    // Showdown data
    let hand_strength = game.hand_strength();
    let showdown_outcomes: Vec<ShowdownData> = topo.showdown_nodes.iter().map(|&node_id| {
        let amount = topo.node_amount[node_id];
        let pot = (tree_config.starting_pot + 2 * amount) as f64;
        let half_pot = 0.5 * pot;
        let rake = (pot * tree_config.rake_rate).min(tree_config.rake_cap);

        let turn = topo.node_turn[node_id];
        let river = topo.node_river[node_id];
        let pair_idx = range_solver::card::card_pair_to_index(turn, river);

        let num_oop = game.private_cards(0).len();
        let num_ip = game.private_cards(1).len();

        // Build outcome matrix for player 0 perspective
        // outcome[h_oop * num_ip + h_ip] = +1 win, -1 loss, 0 tie/blocked
        let mut outcome = vec![0.0f64; num_oop * num_ip];
        let oop_cards = game.private_cards(0);
        let ip_cards = game.private_cards(1);
        let strengths = &hand_strength[pair_idx];

        // Build hand-index-to-strength lookup for fast comparison
        // (strengths[player] is sorted by strength with sentinels, but we need
        //  lookup by hand index)
        let mut oop_strength = vec![0u16; num_oop];
        let mut ip_strength = vec![0u16; num_ip];
        for item in &strengths[0] {
            if (item.index as usize) < num_oop {
                oop_strength[item.index as usize] = item.strength;
            }
        }
        for item in &strengths[1] {
            if (item.index as usize) < num_ip {
                ip_strength[item.index as usize] = item.strength;
            }
        }

        // Build valid hand sets for this board
        let board_mask: u64 = (1u64 << turn) | (1u64 << river)
            | (1u64 << game.card_config().flop[0])
            | (1u64 << game.card_config().flop[1])
            | (1u64 << game.card_config().flop[2]);

        for h_oop in 0..num_oop {
            let (c1, c2) = oop_cards[h_oop];
            let oop_mask = (1u64 << c1) | (1u64 << c2);
            if oop_mask & board_mask != 0 { continue; }  // blocked by board
            let s_oop = oop_strength[h_oop];
            if s_oop == 0 { continue; } // sentinel/invalid

            for h_ip in 0..num_ip {
                let (c3, c4) = ip_cards[h_ip];
                let ip_mask = (1u64 << c3) | (1u64 << c4);
                if ip_mask & board_mask != 0 { continue; }
                if oop_mask & ip_mask != 0 { continue; } // card overlap
                let s_ip = ip_strength[h_ip];
                if s_ip == 0 { continue; }

                outcome[h_oop * num_ip + h_ip] = if s_oop > s_ip {
                    1.0
                } else if s_oop < s_ip {
                    -1.0
                } else {
                    0.0 // tie
                };
            }
        }

        ShowdownData {
            num_player_hands: [num_oop, num_ip],
            outcome_matrix_p0: outcome,
            amount_win: (half_pot - rake) / num_combinations,
            amount_tie: -0.5 * rake / num_combinations,
            amount_lose: -half_pot / num_combinations,
        }
    }).collect();

    TerminalData {
        fold_payoffs,
        showdown_outcomes,
        hand_cards,
        same_hand_index,
        num_combinations,
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p gpu-range-solver -- extract_terminal_data_river`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/gpu-range-solver/src/extract.rs
git commit -m "feat(gpu-range-solver): terminal data extraction (fold payoffs + showdown matrices)"
```

---

## Task 5: GPU Tensor Layout (StreetSolver)

Define the `StreetSolver` struct that holds burn tensors for a single street's batched solve. Implement conversion from `TreeTopology` + `TerminalData` → GPU tensors.

**Files:**
- Modify: `crates/gpu-range-solver/src/tensors.rs`
- Test: inline `#[cfg(test)]`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::{extract_topology, extract_terminal_data};
    use burn::backend::Wgpu;

    // Reuse the make_river_game helper (import or duplicate from extract tests)

    type TestBackend = Wgpu;

    #[test]
    fn create_street_solver_river() {
        let game = make_river_game();
        game.allocate_memory(false);
        let topo = extract_topology(&game);
        let term = extract_terminal_data(&game, &topo);
        let device = Default::default();
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());

        let solver = StreetSolver::<TestBackend>::new(&topo, &term, 1, num_hands, &device);

        // Regrets and strategy_sum should be zeroed
        let regrets_data = solver.regrets.to_data();
        assert!(regrets_data.as_slice::<f32>().unwrap().iter().all(|&x| x == 0.0));

        // Shapes should match: [batch, num_edges, num_hands]
        let shape = solver.regrets.shape();
        assert_eq!(shape.dims[0], 1);  // batch = 1 for river
        assert_eq!(shape.dims[1], topo.num_edges);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p gpu-range-solver -- create_street_solver_river`
Expected: Compilation error — `StreetSolver` doesn't exist

**Step 3: Implement StreetSolver**

In `crates/gpu-range-solver/src/tensors.rs`:

```rust
use burn::prelude::*;
use crate::extract::{TreeTopology, TerminalData, NodeType, FoldData, ShowdownData};

/// GPU tensor state for solving one street's action subtree across a batch of deals.
pub struct StreetSolver<B: Backend> {
    // -- Topology (shared across batch, constant) --
    pub num_nodes: usize,
    pub num_edges: usize,
    pub num_hands: usize,
    pub batch_size: usize,

    /// Per-edge: parent node index. Shape: [num_edges]
    pub edge_parent: Tensor<B, 1, Int>,
    /// Per-edge: child node index. Shape: [num_edges]
    pub edge_child: Tensor<B, 1, Int>,

    /// Nodes at each depth level (CPU-side, used to drive kernel dispatch)
    pub level_nodes: Vec<Vec<usize>>,
    /// Edges whose child is at each depth level
    pub level_edges: Vec<Vec<usize>>,
    pub max_depth: usize,

    /// Node classifications (CPU-side)
    pub node_type: Vec<NodeType>,
    /// Per-node: number of actions
    pub node_num_actions: Vec<usize>,
    /// Per-node: first edge index (cumulative sum of num_actions)
    pub node_first_edge: Vec<usize>,

    // -- Mutable solver state --
    /// Cumulative regrets. Shape: [batch, num_edges, num_hands]
    pub regrets: Tensor<B, 3>,
    /// Cumulative strategy sum. Shape: [batch, num_edges, num_hands]
    pub strategy_sum: Tensor<B, 3>,
    /// Counterfactual reach probabilities. Shape: [batch, num_nodes, num_hands]
    pub reach: Tensor<B, 3>,
    /// Counterfactual values. Shape: [batch, num_nodes, num_hands]
    pub cfv: Tensor<B, 3>,

    // -- Terminal data on GPU --
    // (stored as CPU-side data; transferred to GPU per-evaluation)
    // Fold and showdown data kept as CPU structs, converted to tensors on demand
    pub fold_data: Vec<FoldData>,
    pub showdown_data: Vec<ShowdownData>,
    pub fold_node_ids: Vec<usize>,
    pub showdown_node_ids: Vec<usize>,
    pub player_node_ids: [Vec<usize>; 2],

    // -- Card data --
    /// Per-player hand → (card1, card2)
    pub hand_cards: [Vec<(u8, u8)>; 2],
    /// Per-player same-hand-index
    pub same_hand_index: [Vec<u16>; 2],
    pub num_combinations: f64,
}

impl<B: Backend> StreetSolver<B> {
    pub fn new(
        topo: &TreeTopology,
        term: &TerminalData,
        batch_size: usize,
        num_hands: usize,
        device: &B::Device,
    ) -> Self {
        let num_nodes = topo.num_nodes;
        let num_edges = topo.num_edges;

        // Build edge index tensors
        let edge_parent = Tensor::from_ints(
            &topo.edge_parent.iter().map(|&x| x as i32).collect::<Vec<_>>()[..],
            device,
        );
        let edge_child = Tensor::from_ints(
            &topo.edge_child.iter().map(|&x| x as i32).collect::<Vec<_>>()[..],
            device,
        );

        // Compute node_first_edge (cumulative sum of num_actions)
        let mut node_first_edge = vec![0usize; num_nodes];
        let mut edge_counter = 0;
        for n in 0..num_nodes {
            node_first_edge[n] = edge_counter;
            edge_counter += topo.node_num_actions[n];
        }

        // Allocate solver state tensors (zeroed)
        let regrets = Tensor::zeros([batch_size, num_edges, num_hands], device);
        let strategy_sum = Tensor::zeros([batch_size, num_edges, num_hands], device);
        let reach = Tensor::zeros([batch_size, num_nodes, num_hands], device);
        let cfv = Tensor::zeros([batch_size, num_nodes, num_hands], device);

        StreetSolver {
            num_nodes,
            num_edges,
            num_hands,
            batch_size,
            edge_parent,
            edge_child,
            level_nodes: topo.level_nodes.clone(),
            level_edges: topo.level_edges.clone(),
            max_depth: topo.max_depth,
            node_type: topo.node_type.clone(),
            node_num_actions: topo.node_num_actions.clone(),
            node_first_edge,
            regrets,
            strategy_sum,
            reach,
            cfv,
            fold_data: term.fold_payoffs.clone(),
            showdown_data: term.showdown_outcomes.clone(),
            fold_node_ids: topo.fold_nodes.clone(),
            showdown_node_ids: topo.showdown_nodes.clone(),
            player_node_ids: topo.player_nodes.clone(),
            hand_cards: term.hand_cards.clone(),
            same_hand_index: term.same_hand_index.clone(),
            num_combinations: term.num_combinations,
        }
    }
}
```

Note: `FoldData`, `ShowdownData` need `Clone` derived. Add `#[derive(Clone)]` to both structs in `extract.rs`.

**Step 4: Run tests**

Run: `cargo test -p gpu-range-solver -- create_street_solver_river`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/gpu-range-solver/src/tensors.rs crates/gpu-range-solver/src/extract.rs
git commit -m "feat(gpu-range-solver): StreetSolver GPU tensor layout"
```

---

## Task 6: Regret Matching + DCFR Discount

Implement regret matching (clamp + normalize) and DCFR discounting as burn tensor operations. These are the simplest GPU operations and will validate the tensor pipeline.

**Files:**
- Modify: `crates/gpu-range-solver/src/solver.rs`
- Test: inline `#[cfg(test)]`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    type B = Wgpu;

    #[test]
    fn regret_matching_uniform_when_all_negative() {
        let device = Default::default();
        // 1 batch, 3 actions, 4 hands → regrets shape [1, 3, 4]
        // All negative regrets → uniform strategy 1/3
        let regrets = Tensor::<B, 3>::from_floats(
            [[[-1.0, -2.0, -3.0, -4.0],
              [-5.0, -6.0, -7.0, -8.0],
              [-9.0, -10.0, -11.0, -12.0]]],
            &device,
        );
        let strategy = regret_match::<B>(regrets, 3);
        let data: Vec<f32> = strategy.to_data().to_vec().unwrap();

        // All should be ~1/3
        for &v in &data {
            assert!((v - 1.0 / 3.0).abs() < 1e-5, "expected ~0.333, got {v}");
        }
    }

    #[test]
    fn regret_matching_proportional() {
        let device = Default::default();
        // 1 batch, 2 actions, 2 hands
        // Regrets: action0=[3,0], action1=[1,0]
        // Expected: action0=[0.75, 0.5], action1=[0.25, 0.5]
        let regrets = Tensor::<B, 3>::from_floats(
            [[[3.0, 0.0],
              [1.0, 0.0]]],
            &device,
        );
        let strategy = regret_match::<B>(regrets, 2);
        let data: Vec<f32> = strategy.to_data().to_vec().unwrap();

        assert!((data[0] - 0.75).abs() < 1e-5); // action0, hand0
        assert!((data[1] - 0.5).abs() < 1e-5);  // action0, hand1 (uniform fallback)
        assert!((data[2] - 0.25).abs() < 1e-5); // action1, hand0
        assert!((data[3] - 0.5).abs() < 1e-5);  // action1, hand1
    }

    #[test]
    fn dcfr_discount_scales_correctly() {
        let device = Default::default();
        let mut regrets = Tensor::<B, 3>::from_floats(
            [[[10.0, -5.0],
              [-3.0, 8.0]]],
            &device,
        );
        let alpha = 0.9f32;
        let beta = 0.5f32;
        regrets = dcfr_discount_regrets::<B>(regrets, alpha, beta);
        let data: Vec<f32> = regrets.to_data().to_vec().unwrap();

        assert!((data[0] - 9.0).abs() < 1e-5);  // 10 * 0.9
        assert!((data[1] - -2.5).abs() < 1e-5); // -5 * 0.5
        assert!((data[2] - -1.5).abs() < 1e-5); // -3 * 0.5
        assert!((data[3] - 7.2).abs() < 1e-5);  // 8 * 0.9
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p gpu-range-solver -- regret_matching`
Expected: Compilation error

**Step 3: Implement**

In `crates/gpu-range-solver/src/solver.rs`:

```rust
use burn::prelude::*;

/// DCFR discount parameters, matching range-solver's DiscountParams.
pub struct DiscountParams {
    pub alpha_t: f32,
    pub beta_t: f32,
    pub gamma_t: f32,
}

impl DiscountParams {
    pub fn new(iteration: u32) -> Self {
        let nearest_lower_power_of_4 = match iteration {
            0 => 0,
            x => 1 << ((x.leading_zeros() ^ 31) & !1),
        };
        let t_alpha = (iteration as i32 - 1).max(0) as f64;
        let t_gamma = (iteration - nearest_lower_power_of_4) as f64;
        let pow_alpha = t_alpha * t_alpha.sqrt();
        let pow_gamma = (t_gamma / (t_gamma + 1.0)).powi(3);
        Self {
            alpha_t: (pow_alpha / (pow_alpha + 1.0)) as f32,
            beta_t: 0.5,
            gamma_t: pow_gamma as f32,
        }
    }
}

/// Regret matching: converts cumulative regrets into a strategy.
///
/// Input: regrets `[batch, num_edges, num_hands]` where edges for one node
/// are contiguous (e.g., edges 0..3 belong to node 0 with 3 actions).
///
/// For each node's action group: clip regrets to ≥0, normalize per hand.
/// If all regrets for a hand are ≤0, use uniform 1/num_actions.
///
/// This version operates on a SINGLE node's edges (caller slices by node).
/// `num_actions`: number of actions for this node.
/// Input shape: `[batch, num_actions, num_hands]`
/// Output shape: `[batch, num_actions, num_hands]`
pub fn regret_match<B: Backend>(regrets: Tensor<B, 3>, num_actions: usize) -> Tensor<B, 3> {
    // Clip negative regrets to zero
    let positive = regrets.clamp_min(0.0);

    // Sum over action dimension: [batch, 1, num_hands]
    let denom = positive.clone().sum_dim(1);

    // Where denom > 0, normalize; otherwise uniform
    let uniform_val = 1.0 / num_actions as f32;
    let is_zero = denom.clone().lower_equal_elem(0.0);

    // Normalize: positive / denom (broadcast denom over action dim)
    let normalized = positive / denom.clamp_min(1e-30);

    // Replace with uniform where denom was zero
    let uniform = Tensor::full(normalized.shape(), uniform_val, &normalized.device());
    normalized.mask_where(is_zero.expand(normalized.shape()), uniform)
}

/// Apply DCFR discount to cumulative regrets.
/// Positive regrets scaled by alpha_t, negative by beta_t.
pub fn dcfr_discount_regrets<B: Backend>(
    regrets: Tensor<B, 3>,
    alpha: f32,
    beta: f32,
) -> Tensor<B, 3> {
    let is_positive = regrets.clone().greater_equal_elem(0.0).float();
    let is_negative = regrets.clone().lower_elem(0.0).float();
    let scale = is_positive * alpha + is_negative * beta;
    regrets * scale
}

/// Apply DCFR discount to cumulative strategy sum.
pub fn dcfr_discount_strategy<B: Backend>(
    strategy_sum: Tensor<B, 3>,
    gamma: f32,
) -> Tensor<B, 3> {
    strategy_sum * gamma
}
```

**Step 4: Run tests**

Run: `cargo test -p gpu-range-solver -- solver::tests`
Expected: All 3 tests pass

**Step 5: Commit**

```bash
git add crates/gpu-range-solver/src/solver.rs
git commit -m "feat(gpu-range-solver): regret matching + DCFR discount on GPU"
```

---

## Task 7: Fold Terminal Evaluation on GPU

Implement fold evaluation using card-index gather/scatter: `cfv[h] = payoff * (total_reach - blocking_reach[h])`.

**Files:**
- Modify: `crates/gpu-range-solver/src/terminal.rs`
- Test: inline `#[cfg(test)]`

**Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    type B = Wgpu;

    #[test]
    fn fold_eval_basic() {
        let device = Default::default();
        // 2 hands for player, 2 hands for opponent
        // Hand 0: cards (0, 1), Hand 1: cards (2, 3)
        // Opponent hand 0: cards (4, 5), hand 1: cards (0, 6) ← shares card 0 with player hand 0
        let opp_reach = Tensor::<B, 2>::from_floats(
            [[1.0, 1.0]], // batch=1, 2 opponent hands
            &device,
        );

        let player_cards = vec![(0u8, 1u8), (2, 3)];
        let opp_cards = vec![(4u8, 5u8), (0, 6)];
        let same_hand = vec![u16::MAX, u16::MAX]; // no identical hands

        let cfv = evaluate_fold::<B>(
            &opp_reach,
            &player_cards,
            &opp_cards,
            &same_hand,
            1.0, // payoff per combo
            &device,
        );

        let data: Vec<f32> = cfv.to_data().to_vec().unwrap();
        // Hand 0 (cards 0,1): total_reach=2, blocking = opp_hand1_reach (shares card 0) = 1
        // cfv[0] = 1.0 * (2.0 - 1.0) = 1.0
        assert!((data[0] - 1.0).abs() < 1e-5, "hand 0: got {}", data[0]);
        // Hand 1 (cards 2,3): no blocking
        // cfv[1] = 1.0 * (2.0 - 0.0) = 2.0
        assert!((data[1] - 2.0).abs() < 1e-5, "hand 1: got {}", data[1]);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p gpu-range-solver -- terminal::tests::fold_eval_basic`
Expected: Compilation error

**Step 3: Implement**

In `crates/gpu-range-solver/src/terminal.rs`:

```rust
use burn::prelude::*;

/// Evaluate fold terminal for one player's hands against opponent's reach.
///
/// Returns CFV per hand: `payoff * (total_opp_reach - blocking_reach[h])`.
///
/// `opp_reach`: `[batch, num_opp_hands]`
/// Returns: `[batch, num_player_hands]`
pub fn evaluate_fold<B: Backend>(
    opp_reach: &Tensor<B, 2>,
    player_cards: &[(u8, u8)],
    opp_cards: &[(u8, u8)],
    same_hand_index: &[u16],
    payoff: f64,
    device: &B::Device,
) -> Tensor<B, 2> {
    let batch_size = opp_reach.dims()[0];
    let num_opp = opp_reach.dims()[1];
    let num_player = player_cards.len();

    // 1. Total opponent reach: sum over all opponent hands → [batch]
    let total_reach = opp_reach.clone().sum_dim(1).squeeze(1); // [batch]

    // 2. Per-card reach: for each of 52 cards, sum reach of opponent hands containing that card
    // Build card_to_opp_hands mapping on CPU
    let mut card_opp_indices: Vec<Vec<usize>> = vec![Vec::new(); 52];
    for (h, &(c1, c2)) in opp_cards.iter().enumerate() {
        card_opp_indices[c1 as usize].push(h);
        card_opp_indices[c2 as usize].push(h);
    }

    // Compute card_reach on CPU from opp_reach data
    // Transfer opp_reach to CPU
    let opp_reach_data: Vec<f32> = opp_reach.to_data().to_vec().unwrap();

    // Build per-player-hand blocking reach on CPU
    let mut blocking_data = vec![0.0f32; batch_size * num_player];

    for b in 0..batch_size {
        let opp_slice = &opp_reach_data[b * num_opp..(b + 1) * num_opp];

        // Precompute card reach
        let mut card_reach = [0.0f64; 52];
        for card in 0..52 {
            for &h in &card_opp_indices[card] {
                card_reach[card] += opp_slice[h] as f64;
            }
        }

        // For each player hand, compute blocking reach
        for (h, &(c1, c2)) in player_cards.iter().enumerate() {
            let mut blocking = card_reach[c1 as usize] + card_reach[c2 as usize];

            // Subtract same-hand correction (double-counted if opponent has identical hand)
            let same_idx = same_hand_index[h];
            if same_idx != u16::MAX {
                blocking -= opp_slice[same_idx as usize] as f64;
            }

            blocking_data[b * num_player + h] = blocking as f32;
        }
    }

    let blocking = Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(blocking_data, [batch_size, num_player]),
        device,
    );

    // 3. CFV = payoff * (total_reach - blocking)
    let total_expanded = total_reach.unsqueeze_dim(1).expand([batch_size, num_player]);
    (total_expanded - blocking) * (payoff as f32)
}

/// Evaluate showdown terminal using outcome matrix × opponent reach.
///
/// `opp_reach`: `[batch, num_opp_hands]`
/// `outcome`: `[num_player_hands, num_opp_hands]` (pre-computed +1/0/-1 matrix)
/// Returns: `[batch, num_player_hands]`
pub fn evaluate_showdown<B: Backend>(
    opp_reach: &Tensor<B, 2>,
    outcome: &Tensor<B, 2>,
    amount_win: f64,
    amount_lose: f64,
    amount_tie: f64,
    device: &B::Device,
) -> Tensor<B, 2> {
    let batch_size = opp_reach.dims()[0];
    let num_player = outcome.dims()[0];
    let num_opp = outcome.dims()[1];

    // Separate outcomes into win/loss/tie components
    let win_mask = outcome.clone().greater_elem(0.0).float();   // +1 entries
    let loss_mask = outcome.clone().lower_elem(0.0).float();     // -1 entries
    let tie_mask = outcome.clone().equal_elem(0.0).float()
        * outcome.clone().abs().lower_elem(0.5).float(); // exact 0 entries (ties, not blocked)
    // Actually, ties are 0.0 in outcome matrix along with blocked hands.
    // We need a separate tie indicator. For now, use the simpler formula:
    // cfv = outcome_matrix @ opp_reach, then scale by appropriate amounts.
    //
    // Actually, the outcome matrix has +1 (win), -1 (loss), 0 (tie or blocked).
    // The formula: cfv[h] = sum_opp(outcome[h,opp] * opp_reach[opp])
    // gives: (reach_of_weaker_opponents - reach_of_stronger_opponents)
    // Then scale: win_reach * amount_win + loss_reach * amount_lose + tie_reach * amount_tie
    //
    // For simplicity and correctness, use separate win/lose/tie masks:

    // Reach of opponents we beat: [batch, num_player]
    let opp_reach_expanded = opp_reach.clone().unsqueeze_dim(1); // [batch, 1, num_opp]
    let win_reach = (win_mask.clone().unsqueeze_dim(0).expand([batch_size, num_player, num_opp])
        * opp_reach_expanded.clone().expand([batch_size, num_player, num_opp]))
        .sum_dim(2).squeeze(2);

    let lose_reach = (loss_mask.clone().unsqueeze_dim(0).expand([batch_size, num_player, num_opp])
        * opp_reach_expanded.clone().expand([batch_size, num_player, num_opp]))
        .sum_dim(2).squeeze(2);

    // CFV = win_reach * amount_win + lose_reach * amount_lose
    // (tie component adds amount_tie * tie_reach, but ties are 0 in outcome so skip for v1)
    win_reach * (amount_win as f32) + lose_reach * (amount_lose as f32)
}
```

**Step 4: Run tests**

Run: `cargo test -p gpu-range-solver -- terminal::tests`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/gpu-range-solver/src/terminal.rs
git commit -m "feat(gpu-range-solver): fold + showdown terminal evaluation on GPU"
```

---

## Task 8: Forward + Backward Pass

Implement the level-synchronous forward pass (reach propagation) and backward pass (CFV accumulation) that form the core of each DCFR iteration.

**Files:**
- Modify: `crates/gpu-range-solver/src/solver.rs`
- Test: inline

**This is the largest task.** The key operations per level:

**Forward (top-down):** For each edge at this depth, `reach[child] = reach[parent] * strategy[edge]` (opponent nodes) or `reach[child] = reach[parent]` (traverser nodes).

**Backward (bottom-up):** For each decision node at this depth:
- Traverser: `cfv[node] = sum_a(strategy[a] * cfv[child_a])`, then update regrets
- Opponent: `cfv[node] = sum_a(cfv[child_a])`

The implementation iterates levels using CPU-side loops but dispatches burn tensor operations for the actual computation. Each level's operations are batched across all nodes at that level AND across the batch dimension (board runouts).

**Step 1: Write the failing test**

```rust
#[test]
fn single_iteration_converges_regrets() {
    // Build a tiny river game, run one DCFR iteration, verify regrets are non-zero
    let game = make_river_game();
    game.allocate_memory(false);
    let topo = extract_topology(&game);
    let term = extract_terminal_data(&game, &topo);
    let device = Default::default();
    let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());
    let mut solver = StreetSolver::<Wgpu>::new(&topo, &term, 1, num_hands, &device);

    // Initialize reach with initial weights
    initialize_reach(&mut solver, &game, 0, &device); // player 0

    // Run backward pass
    backward_pass(&mut solver, 0, &device);

    // CFV at root should be non-zero
    let root_cfv = solver.cfv.clone().slice([0..1, 0..1, 0..num_hands]);
    let data: Vec<f32> = root_cfv.to_data().to_vec().unwrap();
    assert!(data.iter().any(|&x| x != 0.0), "root CFV should be non-zero");
}
```

**Step 2: Run test to verify it fails**

**Step 3: Implement forward/backward pass functions**

This is substantial — implement `initialize_reach()`, `forward_pass()`, `backward_pass()`, and `run_iteration()` functions in `solver.rs`. The backward pass calls into `terminal.rs` for fold/showdown evaluation at terminal nodes.

Key pattern for the level-synchronous loop:

```rust
/// Run one backward pass (bottom-up CFV computation) for the given traversing player.
pub fn backward_pass<B: Backend>(
    solver: &mut StreetSolver<B>,
    player: usize,
    device: &B::Device,
) {
    // Process levels bottom-up
    for depth in (0..=solver.max_depth).rev() {
        // 1. Evaluate terminal nodes at this depth
        for &node_id in &solver.level_nodes[depth] {
            match solver.node_type[node_id] {
                NodeType::Fold { .. } => {
                    evaluate_fold_node(solver, node_id, player, device);
                }
                NodeType::Showdown => {
                    evaluate_showdown_node(solver, node_id, player, device);
                }
                _ => {}
            }
        }

        // 2. Process decision nodes: gather child CFVs, compute node CFV
        for &node_id in &solver.level_nodes[depth] {
            match solver.node_type[node_id] {
                NodeType::Player { player: node_player } => {
                    let n_actions = solver.node_num_actions[node_id];
                    if n_actions == 0 { continue; }
                    let first_edge = solver.node_first_edge[node_id];

                    // Get strategy for this node's edges via regret matching
                    let regrets_slice = solver.regrets.clone().slice([
                        0..solver.batch_size,
                        first_edge..first_edge + n_actions,
                        0..solver.num_hands,
                    ]);
                    let strategy = regret_match::<B>(regrets_slice, n_actions);

                    // Gather child CFVs: [batch, n_actions, num_hands]
                    // Each child is edge_child[first_edge + a]
                    // ... (gather from solver.cfv using index tensors)

                    if node_player == player {
                        // Traverser: cfv = sum(strategy * child_cfvs)
                        // Update regrets: regret[a] += child_cfv[a] - cfv
                        // Update strategy_sum
                    } else {
                        // Opponent: cfv = sum(child_cfvs)
                    }
                }
                _ => {}
            }
        }
    }
}
```

The full implementation of gather/scatter for child CFVs and the regret update loop is the core complexity. Each node's operations are expressed as tensor slicing + element-wise ops.

**Step 4: Run tests**

Run: `cargo test -p gpu-range-solver -- single_iteration`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/gpu-range-solver/src/solver.rs
git commit -m "feat(gpu-range-solver): forward + backward pass with regret updates"
```

---

## Task 9: Full Solve Loop + Exploitability

Wire up the complete DCFR iteration loop: alternating player updates, DCFR discounting, exploitability checks, and early stopping.

**Files:**
- Modify: `crates/gpu-range-solver/src/solver.rs` (add `gpu_solve` function)
- Modify: `crates/gpu-range-solver/src/lib.rs` (expose `gpu_solve`)
- Test: integration test comparing against CPU solver

**Step 1: Write the failing integration test**

Create `crates/gpu-range-solver/tests/vs_cpu.rs`:

```rust
use gpu_range_solver::{GpuSolverConfig, GpuSolveResult};
use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
use range_solver::bet_size::BetSizeOptions;
use range_solver::card::CardConfig;
use range_solver::PostFlopGame;

fn make_river_game() -> PostFlopGame {
    let oop_range = "QQ+,AKs".parse().unwrap();
    let ip_range = "JJ-99,AQs".parse().unwrap();
    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [44, 37, 8],  // Qs Jh 2c
        turn: 28,            // 8d
        river: 0,            // 2s → wait, 2c is already on board. Use 3s = 3
    };
    // ... (construct the same game as the CLI test)
    todo!()
}

#[test]
fn gpu_matches_cpu_river_exploitability() {
    let mut cpu_game = make_river_game();
    cpu_game.allocate_memory(false);
    let cpu_exploit = range_solver::solve(&mut cpu_game, 500, 0.5, false);

    let gpu_game = make_river_game();
    let result = gpu_range_solver::gpu_solve_game(
        &gpu_game,
        GpuSolverConfig {
            max_iterations: 500,
            target_exploitability: 0.5,
            print_progress: false,
        },
    );

    // Exploitability should be within 20% relative error
    let ratio = result.exploitability / cpu_exploit;
    assert!(
        ratio > 0.5 && ratio < 2.0,
        "GPU exploit {:.4} vs CPU {:.4} — ratio {:.2} out of range",
        result.exploitability, cpu_exploit, ratio
    );
}
```

**Step 2: Implement `gpu_solve_game`**

In `crates/gpu-range-solver/src/lib.rs`, implement:

```rust
pub fn gpu_solve_game(
    game: &PostFlopGame,
    config: GpuSolverConfig,
) -> GpuSolveResult {
    // 1. Extract topology
    let topo = extract::extract_topology(game);
    let term = extract::extract_terminal_data(game, &topo);

    // 2. Determine batch size and street decomposition
    // (for river: batch=1, single StreetSolver)
    // (for turn: batch=num_river_cards, one StreetSolver per street)
    // (for flop: batch=num_turn*num_river, two StreetSolvers)

    // 3. Create StreetSolver
    // 4. Run iteration loop
    // 5. Extract final strategy
    // 6. Return result
    todo!()
}
```

The full solve loop calls `run_iteration()` for each iteration (which does forward+backward for both players), then checks exploitability every 5 iterations.

**Exploitability** is computed by a best-response backward pass: same as the regular backward pass but using `max` over actions instead of strategy-weighted sum at the traverser's nodes.

**Step 3-5: Implement, test, commit**

```bash
git commit -m "feat(gpu-range-solver): full DCFR solve loop with exploitability"
```

---

## Task 10: Cross-Street Aggregation (Turn/Flop Support)

For turn solves (4 board cards), the GPU solver must:
1. Solve all 44 river subtrees in parallel (batch=44)
2. Average the river root CFVs to get turn leaf values
3. Solve the turn action tree using those leaf values

For flop solves, add the additional turn aggregation layer.

**Files:**
- Modify: `crates/gpu-range-solver/src/lib.rs` (extend `gpu_solve_game`)
- Create: `crates/gpu-range-solver/src/street.rs` (street decomposition logic)
- Test: integration test with turn solve

**Key implementation:** Decompose the tree at chance nodes. The turn action subtree and river action subtree have the same topology across all board runouts (only terminal values and hand validity differ). Build one `StreetSolver` per street, solve bottom-up.

The chance node aggregation: `parent_cfv = mean(child_cfvs across deals) * hand_mask`.

```bash
git commit -m "feat(gpu-range-solver): cross-street aggregation for turn + flop"
```

---

## Task 11: CLI Command in Trainer

Add the `GpuRangeSolve` CLI subcommand to `crates/trainer/src/main.rs`, mirroring `RangeSolve` exactly.

**Files:**
- Modify: `crates/trainer/src/main.rs`
- Modify: `crates/trainer/Cargo.toml` (add gpu-range-solver dependency)

**Step 1: Add dependency**

In `crates/trainer/Cargo.toml`, add:
```toml
gpu-range-solver = { path = "../gpu-range-solver" }
```

**Step 2: Add CLI subcommand**

In `crates/trainer/src/main.rs`, add a new enum variant after `RangeSolve` (around line 136):

```rust
/// Solve a postflop spot using GPU-accelerated DCFR
GpuRangeSolve {
    #[arg(long)]
    oop_range: String,
    #[arg(long)]
    ip_range: String,
    #[arg(long)]
    flop: String,
    #[arg(long)]
    turn: Option<String>,
    #[arg(long)]
    river: Option<String>,
    #[arg(long, default_value = "100")]
    pot: i32,
    #[arg(long, default_value = "100")]
    effective_stack: i32,
    #[arg(long, default_value = "1000")]
    iterations: u32,
    #[arg(long, default_value = "0.5")]
    target_exploitability: f32,
    #[arg(long, default_value = "50%,100%")]
    oop_bet_sizes: String,
    #[arg(long, default_value = "60%,100%")]
    oop_raise_sizes: String,
    #[arg(long, default_value = "50%,100%")]
    ip_bet_sizes: String,
    #[arg(long, default_value = "60%,100%")]
    ip_raise_sizes: String,
},
```

**Step 3: Add match arm + handler**

Add a match arm in the main dispatch (around line 931) and implement `run_gpu_range_solve()` following the same pattern as `run_range_solve()` (lines 1595-1755) but calling `gpu_range_solver::gpu_solve_game()` instead of `range_solver::solve()`.

The output format is identical: header → iteration progress → root strategy table.

**Step 4: Test from CLI**

Run: `cargo run -p poker-solver-trainer --release -- gpu-range-solve --oop-range "QQ+,AKs" --ip-range "JJ-99,AQs" --flop "Qs Jh 2c" --turn "8d" --river "3s" --pot 100 --effective-stack 100 --iterations 500`

Expected: Same output format as `range-solve`, similar exploitability.

**Step 5: Commit**

```bash
git commit -m "feat(trainer): add gpu-range-solve CLI command"
```

---

## Task 12: Final Integration Tests + Cleanup

Comprehensive comparison of GPU vs CPU solver across river, turn, and flop inputs.

**Files:**
- Create: `crates/gpu-range-solver/tests/integration.rs`

**Tests:**

1. **River**: Run both solvers with 500 iterations, verify exploitability within 50% relative.
2. **Turn**: Run both with 200 iterations, verify exploitability within 50% relative.
3. **Flop**: Run both with 100 iterations, verify exploitability within 100% relative (flop is harder).
4. **Verify output format**: Parse root strategy from GPU solver, verify probabilities sum to 1.0 per hand.

**Cleanup:**
- Remove any `todo!()` stubs
- Run `cargo clippy -p gpu-range-solver`
- Run `cargo test` (full suite) — must complete in <1 minute

```bash
git commit -m "test(gpu-range-solver): integration tests comparing GPU vs CPU"
```

---

## Summary

| Task | Component | Key Deliverable |
|------|-----------|-----------------|
| 1 | range-solver accessors | Public methods for node/game inspection |
| 2 | Crate scaffold | Cargo.toml, lib.rs, workspace integration |
| 3 | Tree extraction | BFS walk → `TreeTopology` flat arrays |
| 4 | Terminal data | Fold payoffs + showdown outcome matrices |
| 5 | GPU tensors | `StreetSolver` struct with burn tensors |
| 6 | Regret matching | `regret_match()` + `dcfr_discount_*()` |
| 7 | Terminal eval GPU | `evaluate_fold()` + `evaluate_showdown()` |
| 8 | Forward + backward | Level-synchronous reach/CFV passes |
| 9 | Solve loop | Full DCFR iteration + exploitability |
| 10 | Cross-street | Turn/flop aggregation at chance nodes |
| 11 | CLI command | `gpu-range-solve` in trainer |
| 12 | Integration tests | GPU vs CPU validation |

**Critical path:** Tasks 1-9 are sequential (each builds on the last). Tasks 10-12 can partially overlap once Task 9 works for river-only.
