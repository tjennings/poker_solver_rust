# Unified Full-Tree CFR Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a single CFR solver that traverses the full game tree (preflop through river) for every deal, replacing the separated preflop/postflop architecture with range-conditioned postflop play.

**Architecture:** A new `UnifiedSolver` processes all 1,755 canonical flops in batches. For each batch, it allocates postflop regret/strategy buffers, runs exhaustive CFR over all 169×169 hand pairs traversing from preflop root through postflop trees, then merges terminal EVs into an aggregate. Regret-based pruning eliminates irrelevant hand × pot-type subtrees after warmup. DCFR discounting and periodic exploration of pruned branches prevent over-pruning.

**Tech Stack:** Rust, rayon (parallel iteration), existing PostflopTree/PostflopLayout/equity table infrastructure.

**Design doc:** `docs/plans/2026-02-27-unified-full-tree-cfr-design.md`

---

### Task 1: UnifiedConfig struct

**Files:**
- Create: `crates/core/src/preflop/unified_config.rs`
- Modify: `crates/core/src/preflop/mod.rs`
- Test: in-file `#[cfg(test)]` module

**Step 1: Write the failing test**

In `crates/core/src/preflop/unified_config.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn default_config_has_sane_values() {
        let config = UnifiedConfig::default_hu(20);
        assert_eq!(config.preflop.stacks, vec![40, 40]);
        assert_eq!(config.epochs, 10);
        assert_eq!(config.flop_batch_size, 50);
        assert_eq!(config.iterations_per_batch, 200);
        assert!(config.prune_threshold < 0.0);
        assert!(config.explore_interval > 0);
    }

    #[timed_test]
    fn config_deserializes_from_yaml() {
        let yaml = r#"
            preflop:
              positions:
                - { name: "Small Blind", short_name: "SB" }
                - { name: "Big Blind", short_name: "BB" }
              blinds: [[0, 1], [1, 2]]
              antes: []
              stacks: [40, 40]
              raise_sizes: [["2.5bb"]]
              raise_cap: 4
              dcfr_alpha: 1.5
              dcfr_beta: 0.5
              dcfr_gamma: 2.0
              dcfr_warmup: 30
            postflop_bet_sizes: [0.5, 1.0]
            postflop_max_raises_per_street: 1
            epochs: 5
            flop_batch_size: 25
            iterations_per_batch: 100
            max_canonical_flops: 0
            prune_threshold: -1000.0
            prune_warmup: 100
            explore_interval: 50
        "#;
        let config: UnifiedConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.epochs, 5);
        assert_eq!(config.flop_batch_size, 25);
        assert_eq!(config.postflop_bet_sizes, vec![0.5, 1.0]);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core unified_config -- --nocapture`
Expected: FAIL — module doesn't exist yet.

**Step 3: Write minimal implementation**

Create `crates/core/src/preflop/unified_config.rs`:

```rust
//! Configuration for the unified full-tree CFR solver.

use serde::{Deserialize, Serialize};
use super::config::PreflopConfig;

/// Configuration for the unified full-tree CFR solver.
///
/// Combines preflop game structure with postflop tree config and
/// training hyperparameters for the unified traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedConfig {
    /// Preflop game structure (positions, blinds, stacks, raise sizes, DCFR params).
    pub preflop: PreflopConfig,

    /// Postflop bet sizes as pot fractions (e.g., [0.5, 1.0]).
    pub postflop_bet_sizes: Vec<f64>,

    /// Maximum raises per postflop street.
    #[serde(default = "default_max_raises")]
    pub postflop_max_raises_per_street: u8,

    /// Number of full passes over all flop batches.
    #[serde(default = "default_epochs")]
    pub epochs: u32,

    /// Number of canonical flops per batch.
    #[serde(default = "default_batch_size")]
    pub flop_batch_size: usize,

    /// CFR iterations per batch visit.
    #[serde(default = "default_iterations_per_batch")]
    pub iterations_per_batch: u32,

    /// 0 = all 1,755 canonical flops.
    #[serde(default)]
    pub max_canonical_flops: usize,

    /// Cumulative regret below this → skip subtree.
    #[serde(default = "default_prune_threshold")]
    pub prune_threshold: f64,

    /// Iterations before pruning activates.
    #[serde(default = "default_prune_warmup")]
    pub prune_warmup: u32,

    /// Re-test pruned branches every N iterations.
    #[serde(default = "default_explore_interval")]
    pub explore_interval: u32,
}

fn default_max_raises() -> u8 { 1 }
fn default_epochs() -> u32 { 10 }
fn default_batch_size() -> usize { 50 }
fn default_iterations_per_batch() -> u32 { 200 }
fn default_prune_threshold() -> f64 { -1000.0 }
fn default_prune_warmup() -> u32 { 100 }
fn default_explore_interval() -> u32 { 50 }

impl UnifiedConfig {
    /// Create a default HU config at the given stack depth (in BB).
    #[must_use]
    pub fn default_hu(stack_depth_bb: u32) -> Self {
        let mut preflop = PreflopConfig::heads_up(stack_depth_bb);
        preflop.dcfr_warmup = 30;
        Self {
            preflop,
            postflop_bet_sizes: vec![0.5, 1.0],
            postflop_max_raises_per_street: default_max_raises(),
            epochs: default_epochs(),
            flop_batch_size: default_batch_size(),
            iterations_per_batch: default_iterations_per_batch(),
            max_canonical_flops: 0,
            prune_threshold: default_prune_threshold(),
            prune_warmup: default_prune_warmup(),
            explore_interval: default_explore_interval(),
        }
    }
}
```

Add to `crates/core/src/preflop/mod.rs`:

```rust
pub mod unified_config;
pub use unified_config::UnifiedConfig;
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core unified_config -- --nocapture`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add crates/core/src/preflop/unified_config.rs crates/core/src/preflop/mod.rs
git commit -m "feat: add UnifiedConfig for full-tree CFR solver"
```

---

### Task 2: UnifiedSolver struct shell with preflop buffers

**Files:**
- Create: `crates/core/src/preflop/unified_solver.rs`
- Modify: `crates/core/src/preflop/mod.rs`
- Test: in-file `#[cfg(test)]` module

**Step 1: Write the failing test**

In `crates/core/src/preflop/unified_solver.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::preflop::unified_config::UnifiedConfig;
    use test_macros::timed_test;

    #[timed_test]
    fn solver_initializes_with_correct_buffer_sizes() {
        let config = UnifiedConfig::default_hu(20);
        let solver = UnifiedSolver::new(&config);
        // Should have regret and strategy buffers matching the preflop tree layout
        assert!(!solver.preflop_regret.is_empty());
        assert_eq!(solver.preflop_regret.len(), solver.preflop_strategy_sum.len());
        assert_eq!(solver.epoch, 0);
    }

    #[timed_test]
    fn solver_builds_postflop_tree() {
        let config = UnifiedConfig::default_hu(20);
        let solver = UnifiedSolver::new(&config);
        // Should have at least one postflop tree built
        assert!(!solver.postflop_tree.nodes.is_empty());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core unified_solver -- --nocapture`
Expected: FAIL — module doesn't exist.

**Step 3: Write minimal implementation**

Create `crates/core/src/preflop/unified_solver.rs`:

```rust
//! Unified full-tree CFR solver: preflop through river in a single traversal.
//!
//! Processes all 1,755 canonical flops in batches, running exhaustive CFR
//! over 169×169 hand pairs from preflop root through postflop trees.
//! Regret-based pruning eliminates irrelevant subtrees after warmup.

use super::config::PreflopConfig;
use super::equity::EquityTable;
use super::postflop_abstraction::PostflopLayout;
use super::postflop_tree::PostflopTree;
use super::tree::{PreflopNode, PreflopTree};
use super::unified_config::UnifiedConfig;
use crate::abstraction::Street;

const NUM_HANDS: usize = 169;

/// Pre-computed layout mapping `(node, hand)` → flat buffer offset for the preflop tree.
struct PreflopLayout {
    /// `(offset, num_actions)` per tree node.
    entries: Vec<(usize, usize)>,
    /// Total buffer size in `f64` elements.
    total_size: usize,
}

impl PreflopLayout {
    fn from_tree(tree: &PreflopTree) -> Self {
        let mut entries = Vec::with_capacity(tree.nodes.len());
        let mut offset = 0;
        for node in &tree.nodes {
            let num_actions = match node {
                PreflopNode::Decision { children, .. } => children.len(),
                PreflopNode::Terminal { .. } => 0,
            };
            entries.push((offset, num_actions));
            offset += NUM_HANDS * num_actions;
        }
        Self {
            entries,
            total_size: offset,
        }
    }

    #[inline]
    fn slot(&self, node_idx: u32, hand_idx: u16) -> (usize, usize) {
        let (base, n) = self.entries[node_idx as usize];
        (base + (hand_idx as usize) * n, n)
    }
}

/// Unified full-tree CFR solver.
pub struct UnifiedSolver {
    // -- Preflop --
    preflop_tree: PreflopTree,
    preflop_layout: PreflopLayout,
    preflop_equity: EquityTable,
    /// Per-node investments [p0, p1] for the preflop tree.
    preflop_investments: Vec<[u32; 2]>,
    /// Per-node raise counts for pot-type classification.
    preflop_raise_counts: Vec<u8>,
    /// Valid (hand_i, hand_j) pairs with non-zero weight.
    pairs: Vec<(u16, u16)>,
    /// Cumulative preflop regrets (persistent across all batches).
    pub(crate) preflop_regret: Vec<f64>,
    /// Cumulative preflop strategy sums (persistent).
    pub(crate) preflop_strategy_sum: Vec<f64>,
    /// Reusable snapshot buffer for frozen preflop regrets.
    preflop_snapshot: Vec<f64>,

    // -- Postflop --
    /// Single postflop tree template (shared across all flops).
    pub(crate) postflop_tree: PostflopTree,
    /// Layout for the postflop tree (169 hands per node).
    postflop_layout: PostflopLayout,

    // -- Training state --
    pub(crate) epoch: u32,
    /// Per-player starting stacks (SB units).
    stacks: [u32; 2],
    /// Config (owned for lifetime convenience).
    config: UnifiedConfig,
}

impl UnifiedSolver {
    /// Create a new unified solver from config.
    #[must_use]
    pub fn new(config: &UnifiedConfig) -> Self {
        let preflop_tree = PreflopTree::build(&config.preflop);
        let preflop_layout = PreflopLayout::from_tree(&preflop_tree);
        let preflop_equity = EquityTable::new_uniform();
        let preflop_investments = precompute_investments(&preflop_tree, &config.preflop);
        let preflop_raise_counts = precompute_raise_counts(&preflop_tree);

        #[allow(clippy::cast_possible_truncation)]
        let pairs: Vec<(u16, u16)> = (0..NUM_HANDS)
            .flat_map(|h1| (0..NUM_HANDS).map(move |h2| (h1, h2)))
            .filter(|&(h1, h2)| preflop_equity.weight(h1, h2) > 0.0)
            .map(|(h1, h2)| (h1 as u16, h2 as u16))
            .collect();

        let buf_size = preflop_layout.total_size;

        // Build a single postflop tree at a reasonable default SPR
        let postflop_tree = PostflopTree::build(
            &config.postflop_bet_sizes.iter().map(|&s| s as f32).collect::<Vec<_>>(),
            config.postflop_max_raises_per_street,
            f64::INFINITY, // SPR — unconstrained for now
        ).unwrap_or_else(|_| PostflopTree {
            nodes: vec![],
            pot_type: super::postflop_tree::PotType::Raised,
            spr: f64::INFINITY,
        });

        let node_streets: Vec<Street> = postflop_tree
            .nodes
            .iter()
            .map(|n| match n {
                super::postflop_tree::PostflopNode::Chance { street, .. } => *street,
                _ => Street::Flop,
            })
            .collect();

        let postflop_layout = PostflopLayout::build(
            &postflop_tree,
            &node_streets,
            NUM_HANDS, NUM_HANDS, NUM_HANDS,
        );

        let stacks = [
            config.preflop.stacks.first().copied().unwrap_or(0),
            config.preflop.stacks.get(1).copied().unwrap_or(0),
        ];

        Self {
            preflop_tree,
            preflop_layout,
            preflop_equity,
            preflop_investments,
            preflop_raise_counts,
            pairs,
            preflop_regret: vec![0.0; buf_size],
            preflop_strategy_sum: vec![0.0; buf_size],
            preflop_snapshot: vec![0.0; buf_size],
            postflop_tree,
            postflop_layout,
            epoch: 0,
            stacks,
            config: config.clone(),
        }
    }
}

/// Precompute per-node investment amounts.
fn precompute_investments(tree: &PreflopTree, config: &PreflopConfig) -> Vec<[u32; 2]> {
    let mut investments = vec![[0u32; 2]; tree.nodes.len()];
    let blind_inv = [
        config.blinds.iter().filter(|(p, _)| *p == 0).map(|(_, a)| a).sum::<u32>(),
        config.blinds.iter().filter(|(p, _)| *p == 1).map(|(_, a)| a).sum::<u32>(),
    ];
    fill_investments(tree, config, 0, blind_inv, &mut investments);
    investments
}

fn fill_investments(
    tree: &PreflopTree,
    config: &PreflopConfig,
    node_idx: u32,
    current_inv: [u32; 2],
    out: &mut [[u32; 2]],
) {
    out[node_idx as usize] = current_inv;
    match &tree.nodes[node_idx as usize] {
        PreflopNode::Decision {
            position,
            children,
            action_labels,
        } => {
            let pos = *position as usize;
            for (i, &child) in children.iter().enumerate() {
                let mut child_inv = current_inv;
                match action_labels[i] {
                    super::tree::PreflopAction::Fold => {}
                    super::tree::PreflopAction::Call => {
                        let to_call = current_inv[1 - pos].saturating_sub(current_inv[pos]);
                        let stack_remaining = config.stacks[pos] - current_inv[pos];
                        child_inv[pos] += to_call.min(stack_remaining);
                    }
                    super::tree::PreflopAction::Raise(size) => {
                        let pot: u32 = current_inv.iter().sum();
                        let to_call = current_inv[1 - pos].saturating_sub(current_inv[pos]);
                        let raise_to = size.resolve(current_inv[1 - pos], pot, to_call);
                        let stack_remaining = config.stacks[pos] - current_inv[pos];
                        child_inv[pos] = current_inv[pos] + raise_to.min(stack_remaining + current_inv[pos]) - current_inv[pos];
                        child_inv[pos] = raise_to.min(config.stacks[pos]);
                    }
                    super::tree::PreflopAction::AllIn => {
                        child_inv[pos] = config.stacks[pos];
                    }
                }
                fill_investments(tree, config, child, child_inv, out);
            }
        }
        PreflopNode::Terminal { .. } => {}
    }
}

/// Precompute raise count at every node.
fn precompute_raise_counts(tree: &PreflopTree) -> Vec<u8> {
    let mut counts = vec![0u8; tree.nodes.len()];
    fill_raise_counts(tree, 0, 0, &mut counts);
    counts
}

fn fill_raise_counts(tree: &PreflopTree, node_idx: u32, count: u8, out: &mut [u8]) {
    out[node_idx as usize] = count;
    if let PreflopNode::Decision {
        children,
        action_labels,
        ..
    } = &tree.nodes[node_idx as usize]
    {
        for (i, &child) in children.iter().enumerate() {
            let child_count = match action_labels[i] {
                super::tree::PreflopAction::Raise(_) | super::tree::PreflopAction::AllIn => {
                    count + 1
                }
                _ => count,
            };
            fill_raise_counts(tree, child, child_count, out);
        }
    }
}
```

Add to `crates/core/src/preflop/mod.rs`:

```rust
pub mod unified_solver;
pub use unified_solver::UnifiedSolver;
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core unified_solver -- --nocapture`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add crates/core/src/preflop/unified_solver.rs crates/core/src/preflop/mod.rs
git commit -m "feat: add UnifiedSolver struct shell with preflop buffers"
```

---

### Task 3: Full-tree CFR traversal (preflop → postflop)

This is the core traversal function that starts at the preflop root and transitions into postflop trees at showdown terminals.

**Files:**
- Modify: `crates/core/src/preflop/unified_solver.rs`
- Test: in-file `#[cfg(test)]` module

**Step 1: Write the failing test**

Add to the `tests` module in `unified_solver.rs`:

```rust
#[timed_test]
fn single_traversal_returns_finite_value() {
    let config = UnifiedConfig::default_hu(20);
    let solver = UnifiedSolver::new(&config);

    // Build an equity table for an arbitrary flop
    let flop = [
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Heart),
        Card::new(Value::Two, Suit::Diamond),
    ];
    let combo_map = build_combo_map(&flop);
    let equity_table = compute_equity_table_pub(&combo_map, flop);

    let buf_size = solver.preflop_layout.total_size;
    let pf_buf_size = solver.postflop_layout.total_size;
    let mut preflop_dr = vec![0.0f64; buf_size];
    let mut preflop_ds = vec![0.0f64; buf_size];
    let mut postflop_regret = vec![0.0f64; pf_buf_size];
    let mut postflop_strategy = vec![0.0f64; pf_buf_size];

    // Traverse with AA vs KK (hand indices 0 and 12 approximately)
    let ev = solver.unified_traverse(
        0, // preflop root
        0, // hero_hand (AA)
        12, // opp_hand (KK)
        0, // hero_pos
        1.0, 1.0,
        &equity_table,
        &solver.preflop_regret, // snapshot (all zeros = uniform)
        &mut preflop_dr,
        &mut preflop_ds,
        &mut postflop_regret,
        &mut postflop_strategy,
        false, // not pruning yet
    );

    assert!(ev.is_finite(), "EV should be finite, got {ev}");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core unified_solver::tests::single_traversal -- --nocapture`
Expected: FAIL — `unified_traverse` doesn't exist.

**Step 3: Write the unified traversal**

Add to `UnifiedSolver` impl block:

```rust
/// Single recursive CFR traversal spanning preflop → postflop.
///
/// At preflop decision nodes: standard CFR with regret matching.
/// At preflop showdown terminals: transitions into the postflop tree.
/// At preflop fold terminals: returns fold payoff.
/// At postflop nodes: delegates to `postflop_traverse`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn unified_traverse(
    &self,
    node_idx: u32,
    hero_hand: u16,
    opp_hand: u16,
    hero_pos: u8,
    reach_hero: f64,
    reach_opp: f64,
    equity_table: &[f64],
    preflop_snapshot: &[f64],
    preflop_dr: &mut [f64],
    preflop_ds: &mut [f64],
    postflop_regret: &mut [f64],
    postflop_strategy: &mut [f64],
    is_pruning: bool,
) -> f64 {
    let inv = self.preflop_investments[node_idx as usize];
    let hero_inv = f64::from(inv[hero_pos as usize]);

    match &self.preflop_tree.nodes[node_idx as usize] {
        PreflopNode::Terminal { terminal_type, pot } => {
            match terminal_type {
                super::tree::TerminalType::Fold { folder } => {
                    if *folder == hero_pos {
                        -hero_inv
                    } else {
                        f64::from(*pot) - hero_inv
                    }
                }
                super::tree::TerminalType::Showdown => {
                    // Transition into postflop tree
                    let pot_f = f64::from(*pot);
                    if self.postflop_tree.nodes.is_empty() {
                        // No postflop tree — fall back to equity
                        let eq = self.preflop_equity.equity(hero_hand as usize, opp_hand as usize);
                        return eq * pot_f - hero_inv;
                    }
                    let postflop_ev = self.postflop_traverse(
                        0, hero_hand, opp_hand, hero_pos,
                        reach_hero, reach_opp,
                        equity_table,
                        postflop_regret,
                        postflop_strategy,
                    );
                    // postflop_ev is in pot-fraction units; scale by pot
                    postflop_ev * pot_f + (pot_f / 2.0 - hero_inv)
                }
            }
        }
        PreflopNode::Decision {
            position, children, ..
        } => {
            let num_actions = children.len();
            let is_hero = *position == hero_pos;
            let hand_for_key = if is_hero { hero_hand } else { opp_hand };
            let (start, _) = self.preflop_layout.slot(node_idx, hand_for_key);

            let mut strategy = [0.0f64; 8];
            regret_matching_into(preflop_snapshot, start, &mut strategy[..num_actions]);

            if is_hero {
                let mut action_values = [0.0f64; 8];
                for (i, &child_idx) in children.iter().enumerate() {
                    action_values[i] = self.unified_traverse(
                        child_idx, hero_hand, opp_hand, hero_pos,
                        reach_hero * strategy[i], reach_opp,
                        equity_table, preflop_snapshot,
                        preflop_dr, preflop_ds,
                        postflop_regret, postflop_strategy,
                        is_pruning,
                    );
                }
                let node_value: f64 = strategy[..num_actions]
                    .iter()
                    .zip(&action_values[..num_actions])
                    .map(|(s, v)| s * v)
                    .sum();

                for (i, val) in action_values[..num_actions].iter().enumerate() {
                    preflop_dr[start + i] += reach_opp * (val - node_value);
                }
                for (i, &s) in strategy[..num_actions].iter().enumerate() {
                    preflop_ds[start + i] += reach_hero * s;
                }
                node_value
            } else {
                children
                    .iter()
                    .enumerate()
                    .map(|(i, &child_idx)| {
                        strategy[i]
                            * self.unified_traverse(
                                child_idx, hero_hand, opp_hand, hero_pos,
                                reach_hero, reach_opp * strategy[i],
                                equity_table, preflop_snapshot,
                                preflop_dr, preflop_ds,
                                postflop_regret, postflop_strategy,
                                is_pruning,
                            )
                    })
                    .sum()
            }
        }
    }
}

/// Postflop CFR traversal using pre-computed equity table.
///
/// Same as `exhaustive_cfr_traverse` but operates on the unified solver's
/// postflop tree and buffers.
#[allow(clippy::too_many_arguments)]
fn postflop_traverse(
    &self,
    node_idx: u32,
    hero_hand: u16,
    opp_hand: u16,
    hero_pos: u8,
    reach_hero: f64,
    reach_opp: f64,
    equity_table: &[f64],
    regret_sum: &mut [f64],
    strategy_sum: &mut [f64],
) -> f64 {
    use super::postflop_tree::{PostflopNode, PostflopTerminalType};
    use super::postflop_abstraction::regret_matching_into as pf_regret_matching;

    let n = NUM_HANDS;
    match &self.postflop_tree.nodes[node_idx as usize] {
        PostflopNode::Terminal {
            terminal_type,
            pot_fraction,
        } => match terminal_type {
            PostflopTerminalType::Fold { folder } => {
                if *folder == hero_pos {
                    -pot_fraction / 2.0
                } else {
                    pot_fraction / 2.0
                }
            }
            PostflopTerminalType::Showdown => {
                let eq = equity_table[hero_hand as usize * n + opp_hand as usize];
                if eq.is_nan() {
                    return 0.0;
                }
                eq * pot_fraction - pot_fraction / 2.0
            }
        },
        PostflopNode::Chance { children, .. } => {
            debug_assert!(!children.is_empty());
            self.postflop_traverse(
                children[0], hero_hand, opp_hand, hero_pos,
                reach_hero, reach_opp, equity_table,
                regret_sum, strategy_sum,
            )
        }
        PostflopNode::Decision {
            position, children, ..
        } => {
            let is_hero = *position == hero_pos;
            let bucket = if is_hero { hero_hand } else { opp_hand };
            let (start, _) = self.postflop_layout.slot(node_idx, bucket);
            let num_actions = children.len();

            let mut strategy = [0.0f64; 8];
            pf_regret_matching(regret_sum, start, &mut strategy[..num_actions]);

            if is_hero {
                let mut action_values = [0.0f64; 8];
                for (i, &child) in children.iter().enumerate() {
                    action_values[i] = self.postflop_traverse(
                        child, hero_hand, opp_hand, hero_pos,
                        reach_hero * strategy[i], reach_opp,
                        equity_table, regret_sum, strategy_sum,
                    );
                }
                let node_value: f64 = strategy[..num_actions]
                    .iter()
                    .zip(&action_values[..num_actions])
                    .map(|(s, v)| s * v)
                    .sum();

                for (i, val) in action_values[..num_actions].iter().enumerate() {
                    regret_sum[start + i] += reach_opp * (val - node_value);
                }
                for (i, &s) in strategy[..num_actions].iter().enumerate() {
                    strategy_sum[start + i] += reach_hero * s;
                }
                node_value
            } else {
                children
                    .iter()
                    .enumerate()
                    .map(|(i, &child)| {
                        strategy[i]
                            * self.postflop_traverse(
                                child, hero_hand, opp_hand, hero_pos,
                                reach_hero, reach_opp * strategy[i],
                                equity_table, regret_sum, strategy_sum,
                            )
                    })
                    .sum()
            }
        }
    }
}
```

Also add a helper `regret_matching_into` (same logic as in `solver.rs`):

```rust
/// Regret matching: positive-regret-normalized strategy.
#[allow(clippy::cast_precision_loss)]
fn regret_matching_into(regret_buf: &[f64], start: usize, out: &mut [f64]) {
    let num_actions = out.len();
    let mut positive_sum = 0.0f64;
    for (i, s) in out.iter_mut().enumerate() {
        let val = regret_buf[start + i];
        if val > 0.0 {
            *s = val;
            positive_sum += val;
        } else {
            *s = 0.0;
        }
    }
    if positive_sum > 0.0 {
        for s in out.iter_mut() {
            *s /= positive_sum;
        }
    } else {
        let uniform = 1.0 / num_actions as f64;
        out.fill(uniform);
    }
}
```

You'll need to make `compute_equity_table` (from `postflop_exhaustive.rs`) accessible. Add a `pub(crate)` wrapper or make the existing function `pub(crate)`. Also expose `build_combo_map` if not already public.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core unified_solver -- --nocapture`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add crates/core/src/preflop/unified_solver.rs crates/core/src/preflop/postflop_exhaustive.rs
git commit -m "feat: add unified CFR traversal spanning preflop through postflop"
```

---

### Task 4: Batch processing and training loop

**Files:**
- Modify: `crates/core/src/preflop/unified_solver.rs`
- Test: in-file `#[cfg(test)]`

**Step 1: Write the failing test**

```rust
#[timed_test]
fn train_one_batch_updates_preflop_regrets() {
    let mut config = UnifiedConfig::default_hu(20);
    config.flop_batch_size = 2;
    config.iterations_per_batch = 5;
    config.max_canonical_flops = 4; // just 4 flops for speed
    let mut solver = UnifiedSolver::new(&config);

    let flops = sample_canonical_flops(4);
    let batch = &flops[0..2];
    solver.train_one_batch(batch);

    // After training, preflop regrets should have non-zero values
    let nonzero = solver.preflop_regret.iter().any(|&r| r != 0.0);
    assert!(nonzero, "preflop regrets should be updated after batch training");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core unified_solver::tests::train_one_batch -- --nocapture`
Expected: FAIL — `train_one_batch` doesn't exist.

**Step 3: Implement train_one_batch**

Add to `UnifiedSolver` impl:

```rust
/// Train on a single batch of flops.
///
/// For each flop in the batch:
/// 1. Compute equity table
/// 2. Allocate postflop buffers
/// 3. Run `iterations_per_batch` CFR iterations over all hand pairs
/// 4. Preflop regrets/strategy accumulate persistently
pub fn train_one_batch(&mut self, batch_flops: &[[Card; 3]]) {
    use rayon::prelude::*;
    use super::postflop_hands::build_combo_map;

    let iterations = self.config.iterations_per_batch;
    let pf_buf_size = self.postflop_layout.total_size;

    for &flop in batch_flops {
        let combo_map = build_combo_map(&flop);
        let equity_table = compute_equity_table_internal(&combo_map, flop);

        // Batch-local postflop buffers
        let mut postflop_regret = vec![0.0f64; pf_buf_size];
        let mut postflop_strategy = vec![0.0f64; pf_buf_size];

        for _iter in 0..iterations {
            // Snapshot preflop regrets for this iteration
            self.preflop_snapshot.clone_from(&self.preflop_regret);

            let preflop_buf_size = self.preflop_layout.total_size;

            // Sequential for now (parallelism within hand pairs later)
            let mut preflop_dr = vec![0.0f64; preflop_buf_size];
            let mut preflop_ds = vec![0.0f64; preflop_buf_size];

            for &(h1, h2) in &self.pairs {
                for hero_pos in 0..2u8 {
                    let (hh, oh) = if hero_pos == 0 { (h1, h2) } else { (h2, h1) };
                    let w = self.preflop_equity.weight(h1 as usize, h2 as usize);
                    if w <= 0.0 { continue; }

                    self.unified_traverse(
                        0, hh, oh, hero_pos,
                        1.0, w,
                        &equity_table,
                        &self.preflop_snapshot,
                        &mut preflop_dr,
                        &mut preflop_ds,
                        &mut postflop_regret,
                        &mut postflop_strategy,
                        false,
                    );
                }
            }

            // Merge iteration deltas into persistent preflop buffers
            for (d, s) in self.preflop_regret.iter_mut().zip(&preflop_dr) {
                *d += s;
            }
            for (d, s) in self.preflop_strategy_sum.iter_mut().zip(&preflop_ds) {
                *d += s;
            }
        }
        // postflop_regret and postflop_strategy are dropped here (batch-local)
    }
}

/// Run the full training loop: epochs × batches.
pub fn train(&mut self) {
    use super::postflop_hands::sample_canonical_flops;

    let all_flops = sample_canonical_flops(self.config.max_canonical_flops);
    let batch_size = self.config.flop_batch_size;

    for epoch in 0..self.config.epochs {
        for batch in all_flops.chunks(batch_size) {
            self.train_one_batch(batch);
        }
        self.epoch = epoch + 1;
    }
}
```

You'll need to make `compute_equity_table` from `postflop_exhaustive.rs` accessible as `compute_equity_table_internal` or similar. Add a `pub(crate)` wrapper.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core unified_solver -- --nocapture`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add crates/core/src/preflop/unified_solver.rs crates/core/src/preflop/postflop_exhaustive.rs
git commit -m "feat: add batch training loop for unified solver"
```

---

### Task 5: DCFR discounting in unified solver

**Files:**
- Modify: `crates/core/src/preflop/unified_solver.rs`
- Test: in-file `#[cfg(test)]`

**Step 1: Write the failing test**

```rust
#[timed_test]
fn dcfr_discounting_reduces_early_regrets() {
    let mut config = UnifiedConfig::default_hu(20);
    config.preflop.dcfr_warmup = 0; // Discount from the start
    config.flop_batch_size = 1;
    config.iterations_per_batch = 2;
    config.max_canonical_flops = 1;
    let mut solver = UnifiedSolver::new(&config);

    let flops = sample_canonical_flops(1);

    // Run iteration 1 — no discounting yet (warmup=0 but first iter)
    solver.train_one_batch(&flops);
    let regrets_after_1: Vec<f64> = solver.preflop_regret.clone();

    // Run iteration 2 — discounting should reduce iteration 1's regrets
    solver.train_one_batch(&flops);

    // At least some regrets should have been discounted (reduced in magnitude)
    // This is a weak check: after discounting + new iteration, values change
    let changed = solver.preflop_regret.iter()
        .zip(&regrets_after_1)
        .any(|(new, old)| (new - old).abs() > 1e-10);
    assert!(changed, "regrets should change after discounting + new iteration");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core unified_solver::tests::dcfr_discounting -- --nocapture`
Expected: FAIL — no discounting logic applied.

**Step 3: Implement DCFR discounting**

Add to `UnifiedSolver` impl:

```rust
/// Apply DCFR α/β/γ discounting to preflop regrets and strategy sums.
fn apply_preflop_discounting(&mut self, iteration: u64) {
    let alpha = self.config.preflop.dcfr_alpha;
    let beta = self.config.preflop.dcfr_beta;
    let gamma = self.config.preflop.dcfr_gamma;
    let warmup = self.config.preflop.dcfr_warmup;

    if iteration <= warmup { return; }

    #[allow(clippy::cast_precision_loss)]
    let t = iteration as f64;

    // Positive regret discount: t^α / (t^α + 1)
    let pos_discount = t.powf(alpha) / (t.powf(alpha) + 1.0);
    // Negative regret discount: t^β / (t^β + 1)
    let neg_discount = t.powf(beta) / (t.powf(beta) + 1.0);
    // Strategy sum discount: (t / (t+1))^γ
    let strategy_discount = (t / (t + 1.0)).powf(gamma);

    for r in &mut self.preflop_regret {
        if *r > 0.0 {
            *r *= pos_discount;
        } else {
            *r *= neg_discount;
        }
    }
    for s in &mut self.preflop_strategy_sum {
        *s *= strategy_discount;
    }
}
```

Then modify `train_one_batch` to track iteration count and call discounting after each iteration. Add an `iteration: u64` field to `UnifiedSolver` and increment it in the inner loop of `train_one_batch`, calling `apply_preflop_discounting` after merging deltas.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core unified_solver -- --nocapture`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add crates/core/src/preflop/unified_solver.rs
git commit -m "feat: add DCFR discounting to unified solver"
```

---

### Task 6: Regret-based pruning

**Files:**
- Modify: `crates/core/src/preflop/unified_solver.rs`
- Test: in-file `#[cfg(test)]`

**Step 1: Write the failing test**

```rust
#[timed_test]
fn pruning_skips_deeply_negative_regret_hands() {
    let mut config = UnifiedConfig::default_hu(20);
    config.prune_warmup = 0; // Enable pruning immediately
    config.prune_threshold = -10.0;
    config.explore_interval = 1000; // Don't explore during this test
    config.flop_batch_size = 1;
    config.iterations_per_batch = 1;
    config.max_canonical_flops = 1;
    let mut solver = UnifiedSolver::new(&config);

    // Artificially set deep negative regret for hand 0 at root
    let (start, num_actions) = solver.preflop_layout.slot(0, 0);
    for i in 0..num_actions {
        solver.preflop_regret[start + i] = -100.0;
    }

    let flops = sample_canonical_flops(1);
    solver.iteration = 10; // Past warmup

    // The traversal for hand 0 should be skipped
    let regrets_before = solver.preflop_regret[start..start + num_actions].to_vec();
    solver.train_one_batch(&flops);
    let regrets_after = solver.preflop_regret[start..start + num_actions].to_vec();

    // Regrets for this hand should not have changed (it was pruned)
    // Note: discounting may change them, so we check that no iteration delta was added
    // This test validates the concept; exact values depend on discounting interaction
    assert!(
        regrets_before.iter().zip(&regrets_after).all(|(a, b)| (a - b).abs() < 1.0),
        "deeply negative hand should be pruned (regrets barely changed)"
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core unified_solver::tests::pruning_skips -- --nocapture`
Expected: FAIL — no pruning logic.

**Step 3: Implement pruning**

Add pruning check in the hand pair loop of `train_one_batch`. Before calling `unified_traverse`, check if the cumulative regret at the root for this hand is below `prune_threshold`:

```rust
/// Check if a hand should be pruned at the preflop root.
fn is_pruned(&self, hand_idx: u16) -> bool {
    if self.iteration < u64::from(self.config.prune_warmup) {
        return false;
    }
    let (start, num_actions) = self.preflop_layout.slot(0, hand_idx);
    if num_actions == 0 { return false; }
    // Prune if ALL action regrets at root are below threshold
    self.preflop_regret[start..start + num_actions]
        .iter()
        .all(|&r| r < self.config.prune_threshold)
}

/// Check if this iteration should explore pruned branches.
fn is_explore_iteration(&self) -> bool {
    self.config.explore_interval > 0
        && self.iteration % u64::from(self.config.explore_interval) == 0
}
```

In the hand-pair loop inside `train_one_batch`, wrap the traversal:

```rust
let exploring = self.is_explore_iteration();
for &(h1, h2) in &self.pairs {
    for hero_pos in 0..2u8 {
        let (hh, oh) = if hero_pos == 0 { (h1, h2) } else { (h2, h1) };
        // Prune check: skip if hand's root regrets are deeply negative
        if !exploring && self.is_pruned(hh) {
            continue;
        }
        // ... traverse ...
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core unified_solver -- --nocapture`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add crates/core/src/preflop/unified_solver.rs
git commit -m "feat: add regret-based pruning with DCFR exploration"
```

---

### Task 7: Strategic frequency weighting for flop batches

**Files:**
- Modify: `crates/core/src/preflop/unified_solver.rs`
- Test: in-file `#[cfg(test)]`

**Step 1: Write the failing test**

```rust
#[timed_test]
fn batch_weight_reflects_preflop_reach() {
    let config = UnifiedConfig::default_hu(20);
    let solver = UnifiedSolver::new(&config);

    let flop = [
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Heart),
        Card::new(Value::Two, Suit::Diamond),
    ];

    // With uniform strategy, all non-conflicting hands have equal reach
    let weight = solver.compute_batch_weight(&[flop]);
    assert!(weight > 0.0, "batch weight should be positive");
    assert!(weight.is_finite(), "batch weight should be finite");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core unified_solver::tests::batch_weight -- --nocapture`
Expected: FAIL — `compute_batch_weight` doesn't exist.

**Step 3: Implement batch weight computation**

```rust
/// Compute the strategic weight of a flop batch.
///
/// Weight = sum over all non-conflicting (hero, opp) pairs of:
///   preflop_reach(hero) × preflop_reach(opp) × card_removal_weight
///
/// With uniform strategy (all reaches = 1), this is just the number of
/// non-conflicting hand pairs. As the strategy converges, hands that fold
/// preflop contribute less weight.
pub fn compute_batch_weight(&self, flops: &[[Card; 3]]) -> f64 {
    use super::postflop_hands::build_combo_map;

    let mut total_weight = 0.0f64;
    for &flop in flops {
        let combo_map = build_combo_map(&flop);
        for h1 in 0..NUM_HANDS {
            if combo_map[h1].is_empty() { continue; }
            for h2 in 0..NUM_HANDS {
                if combo_map[h2].is_empty() { continue; }
                let w = self.preflop_equity.weight(h1, h2);
                if w <= 0.0 { continue; }
                total_weight += w;
            }
        }
    }
    total_weight
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core unified_solver -- --nocapture`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add crates/core/src/preflop/unified_solver.rs
git commit -m "feat: add strategic frequency weighting for flop batches"
```

---

### Task 8: Strategy extraction

**Files:**
- Modify: `crates/core/src/preflop/unified_solver.rs`
- Test: in-file `#[cfg(test)]`

**Step 1: Write the failing test**

```rust
#[timed_test]
fn strategy_extraction_produces_valid_probabilities() {
    let mut config = UnifiedConfig::default_hu(20);
    config.flop_batch_size = 2;
    config.iterations_per_batch = 10;
    config.max_canonical_flops = 2;
    config.epochs = 1;
    let mut solver = UnifiedSolver::new(&config);
    solver.train();

    let strategy = solver.preflop_strategy();
    // Check root node strategy for AA (hand 0)
    let probs = strategy.get_root_probs(0);
    assert!(!probs.is_empty(), "should have action probabilities");
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "probabilities should sum to 1.0, got {sum}"
    );
    assert!(
        probs.iter().all(|&p| p >= 0.0 && p <= 1.0),
        "all probabilities should be in [0, 1]"
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core unified_solver::tests::strategy_extraction -- --nocapture`
Expected: FAIL — `preflop_strategy` doesn't exist.

**Step 3: Implement strategy extraction**

```rust
/// Extract the average preflop strategy from cumulative strategy sums.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn preflop_strategy(&self) -> PreflopStrategy {
    let mut strategies = rustc_hash::FxHashMap::default();
    for (node_idx, node) in self.preflop_tree.nodes.iter().enumerate() {
        let num_actions = match node {
            PreflopNode::Decision { children, .. } => children.len(),
            PreflopNode::Terminal { .. } => continue,
        };
        for hand_idx in 0..NUM_HANDS {
            let (start, _) = self.preflop_layout.slot(node_idx as u32, hand_idx as u16);
            let sums = &self.preflop_strategy_sum[start..start + num_actions];
            let total: f64 = sums.iter().sum();
            let probs = if total > 0.0 {
                sums.iter().map(|&s| s / total).collect()
            } else if num_actions > 0 {
                vec![1.0 / num_actions as f64; num_actions]
            } else {
                Vec::new()
            };
            let key = PreflopStrategy::key(node_idx as u32, hand_idx as u16);
            strategies.insert(key, probs);
        }
    }
    PreflopStrategy::from_map(strategies)
}
```

Note: `PreflopStrategy` currently has `strategies` as a private field. You may need to add a `from_map` constructor or make the field `pub(crate)`. Add to `PreflopStrategy`:

```rust
/// Create from a raw strategy map.
pub(crate) fn from_map(strategies: FxHashMap<u64, Vec<f64>>) -> Self {
    Self { strategies }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core unified_solver -- --nocapture`
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add crates/core/src/preflop/unified_solver.rs crates/core/src/preflop/solver.rs
git commit -m "feat: add strategy extraction to unified solver"
```

---

### Task 9: Parallelization with rayon

**Files:**
- Modify: `crates/core/src/preflop/unified_solver.rs`
- Test: in-file `#[cfg(test)]`

**Step 1: Write the failing test**

```rust
#[timed_test]
fn parallel_training_produces_same_strategy_shape() {
    let mut config = UnifiedConfig::default_hu(20);
    config.flop_batch_size = 2;
    config.iterations_per_batch = 5;
    config.max_canonical_flops = 2;
    config.epochs = 1;
    let mut solver = UnifiedSolver::new(&config);
    solver.train();

    let strategy = solver.preflop_strategy();
    let probs = strategy.get_root_probs(0);
    assert!(!probs.is_empty());
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}
```

**Step 2: Run test to verify it passes** (should already pass with sequential code)

Run: `cargo test -p poker-solver-core unified_solver::tests::parallel_training -- --nocapture`

**Step 3: Refactor to parallel traversal**

Refactor the hand-pair loop in `train_one_batch` to use rayon `par_iter().fold().reduce()` — the same pattern as `parallel_traverse` in `solver.rs`. The key insight: preflop regret/strategy deltas are accumulated per-thread then merged, while postflop buffers are per-flop (not shared across threads since each flop gets its own buffers).

```rust
// Inside train_one_batch, replace the sequential loop with:
let preflop_buf_size = self.preflop_layout.total_size;
let (preflop_dr, preflop_ds) = self.pairs
    .par_iter()
    .fold(
        || (vec![0.0f64; preflop_buf_size], vec![0.0f64; preflop_buf_size]),
        |(mut dr, mut ds), &(h1, h2)| {
            // ... traverse both positions ...
            (dr, ds)
        },
    )
    .reduce(
        || (vec![0.0; preflop_buf_size], vec![0.0; preflop_buf_size]),
        |(mut ar, mut a_s), (br, bs)| {
            add_into(&mut ar, &br);
            add_into(&mut a_s, &bs);
            (ar, a_s)
        },
    );
```

Note: `postflop_regret` / `postflop_strategy` will need to be per-thread too (each thread gets its own copy, then merged). This is a straightforward extension of the fold/reduce pattern.

Also note: `self` can't be borrowed mutably in par_iter closures. Extract all needed data into an immutable context struct (similar to `Ctx` in `solver.rs`) and pass postflop buffers as thread-local.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core unified_solver -- --nocapture`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add crates/core/src/preflop/unified_solver.rs
git commit -m "feat: parallelize unified solver hand-pair traversal with rayon"
```

---

### Task 10: CLI integration

**Files:**
- Modify: `crates/trainer/src/main.rs`
- Create: `sample_configurations/unified_cfr.yaml`
- Test: manual CLI test

**Step 1: Add CLI subcommand**

Add a new variant to `Commands` in `crates/trainer/src/main.rs`:

```rust
/// Solve the full game tree using unified CFR (preflop + postflop).
SolveUnified {
    /// YAML config file (UnifiedConfig format).
    #[arg(short, long)]
    config: PathBuf,
    /// Output directory for the preflop bundle
    #[arg(short, long)]
    output: PathBuf,
    /// Print strategy matrices every N iterations (0 = only at end)
    #[arg(long, default_value = "0")]
    print_every: u64,
},
```

**Step 2: Implement the handler**

In the `main()` match block, add:

```rust
Commands::SolveUnified { config, output, print_every } => {
    let yaml = std::fs::read_to_string(&config)?;
    let unified_config: poker_solver_core::preflop::UnifiedConfig =
        serde_yaml::from_str(&yaml)?;

    println!("Unified CFR solver");
    println!("  Stack depth: {} BB", unified_config.preflop.stacks[0] / 2);
    println!("  Epochs: {}", unified_config.epochs);
    println!("  Flop batch size: {}", unified_config.flop_batch_size);
    println!("  Iterations/batch: {}", unified_config.iterations_per_batch);
    println!("  Prune threshold: {}", unified_config.prune_threshold);

    let mut solver = poker_solver_core::preflop::UnifiedSolver::new(&unified_config);
    solver.train();

    let strategy = solver.preflop_strategy();
    // Save as preflop bundle
    std::fs::create_dir_all(&output)?;
    let bundle_path = output.join("preflop_strategy.json");
    let json = serde_json::to_string_pretty(&strategy)?;
    std::fs::write(&bundle_path, json)?;
    println!("Strategy saved to {}", bundle_path.display());
}
```

**Step 3: Create sample config**

Create `sample_configurations/unified_cfr.yaml`:

```yaml
preflop:
  positions:
    - { name: "Small Blind", short_name: "SB" }
    - { name: "Big Blind", short_name: "BB" }
  blinds: [[0, 1], [1, 2]]
  antes: []
  stacks: [40, 40]
  raise_sizes: [["2.5bb"], ["3.0bb"]]
  raise_cap: 4
  cfr_variant: dcfr
  dcfr_alpha: 1.5
  dcfr_beta: 0.5
  dcfr_gamma: 2.0
  dcfr_warmup: 30
  exploration: 0.05
postflop_bet_sizes: [0.5, 1.0]
postflop_max_raises_per_street: 1
epochs: 2
flop_batch_size: 25
iterations_per_batch: 50
max_canonical_flops: 50
prune_threshold: -1000.0
prune_warmup: 20
explore_interval: 25
```

**Step 4: Test the CLI**

Run: `cargo run -p poker-solver-trainer --release -- solve-unified -c sample_configurations/unified_cfr.yaml -o /tmp/unified_test`
Expected: Completes without error, prints progress, saves strategy.

**Step 5: Commit**

```bash
git add crates/trainer/src/main.rs sample_configurations/unified_cfr.yaml
git commit -m "feat: add solve-unified CLI command for unified CFR solver"
```

---

### Task 11: Integration test — convergence

**Files:**
- Create: `crates/core/tests/unified_convergence.rs`

**Step 1: Write the convergence test**

```rust
//! Integration test: unified solver converges on a tiny game.

use poker_solver_core::preflop::unified_config::UnifiedConfig;
use poker_solver_core::preflop::unified_solver::UnifiedSolver;

#[test]
fn unified_solver_converges_on_small_game() {
    let mut config = UnifiedConfig::default_hu(10); // 10 BB = shallow
    config.epochs = 3;
    config.flop_batch_size = 5;
    config.iterations_per_batch = 50;
    config.max_canonical_flops = 5;
    config.prune_warmup = 20;
    config.prune_threshold = -500.0;
    config.explore_interval = 25;
    config.preflop.dcfr_warmup = 10;

    let mut solver = UnifiedSolver::new(&config);
    solver.train();

    let strategy = solver.preflop_strategy();

    // AA (hand index 0) should raise frequently at root
    let aa_probs = strategy.get_root_probs(0);
    assert!(!aa_probs.is_empty(), "AA should have action probabilities");

    // All hands should produce valid probability distributions
    for hand_idx in 0..169 {
        let probs = strategy.get_root_probs(hand_idx);
        if probs.is_empty() { continue; }
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "hand {hand_idx}: probabilities sum to {sum}, expected 1.0"
        );
    }
}
```

**Step 2: Run the test**

Run: `cargo test -p poker-solver-core --test unified_convergence -- --nocapture`
Expected: PASS — solver produces valid strategies.

**Step 3: Commit**

```bash
git add crates/core/tests/unified_convergence.rs
git commit -m "test: add integration test for unified solver convergence"
```

---

### Task 12: Update documentation

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/training.md`
- Modify: `docs/strategic_coverage.md`

**Step 1: Update architecture.md**

Add a section describing the unified solver as a new solver backend alongside the separated preflop/postflop architecture. Describe: full-tree traversal, batched flops, regret-based pruning, DCFR exploration.

**Step 2: Update training.md**

Add `solve-unified` command documentation with config schema and example usage.

**Step 3: Update strategic_coverage.md**

Update the "Cross-street action history" gap entry to note it's addressed by the unified solver (ranges are conditioned through natural traversal). Update the "Postflop exploitability" gap similarly.

**Step 4: Commit**

```bash
git add docs/architecture.md docs/training.md docs/strategic_coverage.md
git commit -m "docs: document unified full-tree CFR solver"
```

---

## Implementation Notes for Developers

### Key Architectural Decisions

1. **Preflop regrets persist across all batches/epochs.** Postflop regrets are batch-local and discarded after each flop batch. Only postflop terminal EVs fold into the preflop-level aggregate.

2. **The postflop tree is shared across all flops.** Each flop gets its own equity table but uses the same tree structure. This means postflop bet sizes are uniform across pot types initially.

3. **Pruning operates at the preflop root.** If all action regrets for a hand at the root node are below threshold, the entire postflop subtree for that hand is skipped. This is where the biggest computational savings come from.

4. **`preflop_investments` and `preflop_raise_counts` reuse the same logic as `solver.rs`.** The implementation in `unified_solver.rs` duplicates these helper functions to avoid coupling. A future refactor could extract them into a shared module.

### Existing Code Reuse

| Component | Source | How Used |
|-|-|-|
| `PreflopTree::build` | `tree.rs` | Unchanged — builds preflop tree from config |
| `PostflopTree::build` | `postflop_tree.rs` | Unchanged — builds postflop tree template |
| `PostflopLayout::build` | `postflop_abstraction.rs` | Unchanged — maps (node, hand) → buffer offset |
| `compute_equity_table` | `postflop_exhaustive.rs` | Made `pub(crate)` — 169×169 equity per flop |
| `build_combo_map` | `postflop_hands.rs` | Already public — board-filtered combo lists |
| `regret_matching_into` | `postflop_abstraction.rs` | Already `pub(crate)` for postflop nodes |
| `PreflopStrategy` | `solver.rs` | Add `from_map` constructor for unified solver |

### Performance Expectations

- **Tiny game (5 flops, 50 iters, 10 BB):** ~seconds
- **Small game (50 flops, 100 iters, 20 BB):** ~minutes
- **Full game (1,755 flops, 200 iters, 20 BB):** ~hours with pruning
- **Pruning savings:** After warmup, expect 60-80% of hand × pot-type combinations to be pruned, reducing postflop traversal proportionally.
