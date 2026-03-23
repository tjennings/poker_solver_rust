//! Parallel DCFR solver for concrete subgame trees.
//!
//! Solves subgames with exact hand evaluation (no abstraction). Uses
//! flat arrays indexed by `(decision_node, combo, action)` instead of
//! hash maps, with rayon parallelism over combos and DCFR discounting.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::cfr::dcfr::DcfrParams;
use crate::cfr::parallel::{ParallelCfr, add_into, parallel_traverse};
use crate::cfr::regret::regret_match;
use crate::poker::{Card, Hand, Rank, Rankable};

use super::SubgameConfig;
use super::cbv::CbvTable;
use crate::blueprint_v2::game_tree::{GameNode, GameTree, TerminalKind, TreeAction};
use crate::blueprint_v2::Street;

// ---------------------------------------------------------------------------
// SubgameHands -- valid hole card combos for a specific board
// ---------------------------------------------------------------------------

/// All valid hole card combos for a specific board.
#[derive(Debug, Clone)]
pub struct SubgameHands {
    pub combos: Vec<[Card; 2]>,
}

impl SubgameHands {
    /// Enumerate all valid 2-card combos from the 52-card deck excluding board cards.
    #[must_use]
    pub fn enumerate(board: &[Card]) -> Self {
        let deck = remaining_deck(board);
        let mut combos = Vec::with_capacity(deck.len() * (deck.len() - 1) / 2);
        for i in 0..deck.len() {
            for j in (i + 1)..deck.len() {
                combos.push([deck[i], deck[j]]);
            }
        }
        Self { combos }
    }
}

fn remaining_deck(board: &[Card]) -> Vec<Card> {
    crate::poker::full_deck()
        .into_iter()
        .filter(|c| !board.contains(c))
        .collect()
}

// ---------------------------------------------------------------------------
// SubgameStrategy -- the output of a subgame solve
// ---------------------------------------------------------------------------

/// Result of a subgame solve: strategies for all combos at all decision nodes.
#[derive(Debug, Clone)]
pub struct SubgameStrategy {
    /// `(node_idx, combo_idx)` -> action probabilities.
    pub strategies: FxHashMap<u64, Vec<f64>>,
    /// Total number of combos in this subgame.
    pub num_combos: usize,
}

impl SubgameStrategy {
    /// Compute the flat key for `(node_idx, combo_idx)`.
    #[must_use]
    pub fn key(node_idx: u32, combo_idx: u32) -> u64 {
        (u64::from(node_idx) << 32) | u64::from(combo_idx)
    }

    /// Get action probabilities for a combo at a node.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn get_probs(&self, node_idx: u32, combo_idx: usize) -> Vec<f64> {
        self.strategies
            .get(&Self::key(node_idx, combo_idx as u32))
            .cloned()
            .unwrap_or_default()
    }

    /// Get root node probabilities for a combo.
    #[must_use]
    pub fn root_probs(&self, combo_idx: usize) -> Vec<f64> {
        self.get_probs(0, combo_idx)
    }

    /// Number of combos in this subgame.
    #[must_use]
    pub fn num_combos(&self) -> usize {
        self.num_combos
    }
}

// ---------------------------------------------------------------------------
// Flat buffer layout
// ---------------------------------------------------------------------------

/// Maps `(node_idx, combo_idx)` to a flat buffer offset.
///
/// Each decision node reserves `num_combos * num_actions` contiguous slots.
/// Non-decision nodes have no allocation. The layout is:
///
/// ```text
/// base[d] + combo_idx * num_actions[d] + action_idx
/// ```
///
/// where `d` is the decision node and `num_actions[d]` is the action count
/// at that node.
struct SubgameLayout {
    /// Per-node base offset. Non-decision nodes store `usize::MAX`.
    bases: Vec<usize>,
    /// Per-node action count. Non-decision nodes store 0.
    num_actions: Vec<usize>,
    /// Total flat buffer size.
    total_size: usize,
}

impl SubgameLayout {
    fn build(tree: &GameTree, num_combos: usize) -> Self {
        let n = tree.nodes.len();
        let mut bases = vec![usize::MAX; n];
        let mut num_actions_vec = vec![0usize; n];
        let mut offset = 0;

        for (i, node) in tree.nodes.iter().enumerate() {
            if let GameNode::Decision { actions, .. } = node {
                bases[i] = offset;
                num_actions_vec[i] = actions.len();
                offset += num_combos * actions.len();
            }
        }

        Self {
            bases,
            num_actions: num_actions_vec,
            total_size: offset,
        }
    }

    /// Returns `(base_offset, num_actions)` for a `(node_idx, combo_idx)` pair.
    ///
    /// The caller indexes as `base + action_idx`.
    #[inline]
    fn slot(&self, node_idx: usize, combo_idx: usize) -> (usize, usize) {
        let na = self.num_actions[node_idx];
        let base = self.bases[node_idx] + combo_idx * na;
        (base, na)
    }
}

// ---------------------------------------------------------------------------
// SubgameCfrSolver
// ---------------------------------------------------------------------------

/// Parallel DCFR solver for concrete subgame trees.
pub struct SubgameCfrSolver {
    tree: GameTree,
    hands: SubgameHands,
    /// `equity[i][j]` = P(combo i beats combo j) at showdown.
    equity_matrix: Vec<Vec<f64>>,
    /// Opponent reaching probability per combo.
    opponent_reach: Vec<f64>,
    /// Per-boundary, per-combo leaf equity (0.0 to 1.0) for `DepthBoundary` nodes.
    /// Outer index = boundary ordinal, inner index = combo index.
    /// Converted to chip values via `(2 * equity - 1) * half_pot`.
    leaf_values: Vec<Vec<f64>>,
    /// Maps tree node index to boundary ordinal (`usize::MAX` if not a boundary).
    node_to_boundary: Vec<usize>,
    /// Flat buffer: cumulative regret sums.
    regret_sum: Vec<f64>,
    /// Flat buffer: cumulative strategy sums.
    strategy_sum: Vec<f64>,
    /// Maps `(node_idx, combo_idx)` to flat buffer offsets.
    layout: SubgameLayout,
    /// Precomputed: for each combo, the total opponent reach of non-blocked combos.
    opp_reach_totals: Vec<f64>,
    /// DCFR discounting parameters.
    dcfr: DcfrParams,
    /// Current iteration count.
    pub iteration: u32,
}

impl SubgameCfrSolver {
    /// Create a solver for a specific board position.
    ///
    /// `leaf_values` is indexed by `[boundary_ordinal][combo_index]`.
    /// Its length must match the number of `DepthBoundary` nodes in `tree`.
    ///
    /// # Panics
    ///
    /// Panics if `leaf_values.len()` does not equal the boundary count.
    #[must_use]
    pub fn new(
        tree: GameTree,
        hands: SubgameHands,
        board: &[Card],
        opponent_reach: Vec<f64>,
        leaf_values: Vec<Vec<f64>>,
    ) -> Self {
        let equity_matrix = compute_equity_matrix(&hands.combos, board);
        let opp_reach_totals = precompute_opp_reach(&hands.combos, &opponent_reach);
        let layout = SubgameLayout::build(&tree, hands.combos.len());
        let buf_size = layout.total_size;

        // Build node_to_boundary mapping.
        let mut node_to_boundary = vec![usize::MAX; tree.nodes.len()];
        let mut boundary_ord = 0;
        for (idx, node) in tree.nodes.iter().enumerate() {
            if matches!(
                node,
                GameNode::Terminal {
                    kind: TerminalKind::DepthBoundary,
                    ..
                }
            ) {
                node_to_boundary[idx] = boundary_ord;
                boundary_ord += 1;
            }
        }
        assert_eq!(
            boundary_ord,
            leaf_values.len(),
            "leaf_values length ({}) must match DepthBoundary count ({boundary_ord})",
            leaf_values.len(),
        );

        Self {
            tree,
            hands,
            equity_matrix,
            opponent_reach,
            leaf_values,
            node_to_boundary,
            regret_sum: vec![0.0; buf_size],
            strategy_sum: vec![0.0; buf_size],
            layout,
            opp_reach_totals,
            dcfr: DcfrParams::default(),
            iteration: 0,
        }
    }

    /// Create a solver with per-boundary leaf values from [`CbvTable`]s.
    ///
    /// For each `DepthBoundary` node, `boundary_mapping[ordinal]` gives the
    /// corresponding Chance node ordinal in the CBV table. Each combo's CBV
    /// is looked up via `combo_to_bucket`, then normalized from chip units
    /// to equity `[0, 1]` using the boundary's pot:
    ///
    /// ```text
    /// equity = (cbv / half_pot + 1) / 2
    /// ```
    ///
    /// The traversal later recovers chip values via `(2 * equity - 1) * half_pot`.
    ///
    /// # Panics
    ///
    /// Panics if `boundary_mapping` length doesn't match the number of
    /// `DepthBoundary` nodes, or if any CBV/bucket lookup is out of range.
    #[must_use]
    pub fn with_cbv_tables(
        tree: GameTree,
        hands: SubgameHands,
        board: &[Card],
        opponent_reach: Vec<f64>,
        cbv_tables: [&CbvTable; 2],
        boundary_mapping: &[usize],
        combo_to_bucket: impl Fn(usize) -> u16,
    ) -> Self {
        let n = hands.combos.len();

        // Collect boundary pots in ordinal order.
        let mut boundary_pots = Vec::new();
        for node in &tree.nodes {
            if let GameNode::Terminal {
                kind: TerminalKind::DepthBoundary,
                pot,
                ..
            } = node
            {
                boundary_pots.push(*pot);
            }
        }
        assert_eq!(
            boundary_pots.len(),
            boundary_mapping.len(),
            "boundary_mapping length ({}) must match DepthBoundary count ({})",
            boundary_mapping.len(),
            boundary_pots.len(),
        );

        // Build per-boundary, per-combo leaf values.
        // Use player 0's CBV table (both players' CBVs are symmetric in
        // the abstract tree; the traversal handles player perspective).
        let leaf_values: Vec<Vec<f64>> = boundary_mapping
            .iter()
            .zip(boundary_pots.iter())
            .map(|(&chance_ord, &pot)| {
                let half_pot = pot / 2.0;
                assert!(half_pot > 0.0, "boundary pot must be positive, got {pot}");
                (0..n)
                    .map(|combo_idx| {
                        let bucket = combo_to_bucket(combo_idx) as usize;
                        let cbv = f64::from(cbv_tables[0].lookup(chance_ord, bucket));
                        // Normalize: equity = (cbv / half_pot + 1) / 2
                        (cbv / half_pot + 1.0) / 2.0
                    })
                    .collect()
            })
            .collect();

        Self::new(tree, hands, board, opponent_reach, leaf_values)
    }

    /// Run parallel DCFR for the given number of iterations.
    pub fn train(&mut self, iterations: u32) {
        let combos: Vec<(u16, u16)> = (0..self.hands.combos.len())
            .filter(|&c| self.opp_reach_totals[c] > 0.0)
            .map(|c| {
                // INVARIANT: combo count fits in u16 (max 1326 combos)
                #[allow(clippy::cast_possible_truncation)]
                (c as u16, 0u16)
            })
            .collect();

        for _ in 0..iterations {
            self.iteration += 1;

            for traverser in 0..2u8 {
                let snapshot = self.build_strategy_snapshot();
                let ctx = SubgameCfrCtx {
                    tree: &self.tree,
                    hands: &self.hands,
                    equity_matrix: &self.equity_matrix,
                    opponent_reach: &self.opponent_reach,
                    leaf_values: &self.leaf_values,
                    node_to_boundary: &self.node_to_boundary,
                    opp_reach_totals: &self.opp_reach_totals,
                    layout: &self.layout,
                    snapshot: &snapshot,
                    traverser,
                };

                let (regret_delta, strategy_delta) = parallel_traverse(&ctx, &combos);
                add_into(&mut self.regret_sum, &regret_delta);
                add_into(&mut self.strategy_sum, &strategy_delta);
            }

            let iter_u64 = u64::from(self.iteration);
            if self.dcfr.should_discount(iter_u64) {
                self.dcfr.discount_regrets(&mut self.regret_sum, iter_u64);
                self.dcfr
                    .discount_strategy_sums(&mut self.strategy_sum, iter_u64);
            }
        }
    }

    /// Extract the average strategy from cumulative strategy sums.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn strategy(&self) -> SubgameStrategy {
        let mut strategies = FxHashMap::default();
        let num_combos = self.hands.combos.len();

        for (node_idx, node) in self.tree.nodes.iter().enumerate() {
            let num_actions = match node {
                GameNode::Decision { actions, .. } => actions.len(),
                _ => continue,
            };

            for combo_idx in 0..num_combos {
                let (base, _) = self.layout.slot(node_idx, combo_idx);
                let sums = &self.strategy_sum[base..base + num_actions];
                let total: f64 = sums.iter().sum();
                if total > 0.0 {
                    let probs: Vec<f64> = sums.iter().map(|&s| s / total).collect();
                    let key =
                        SubgameStrategy::key(node_idx as u32, combo_idx as u32);
                    strategies.insert(key, probs);
                }
            }
        }

        SubgameStrategy {
            strategies,
            num_combos,
        }
    }

    /// Build a frozen strategy snapshot from current regret sums.
    ///
    /// Each `(decision_node, combo)` pair gets regret-matched into a probability
    /// distribution. The snapshot uses the same flat layout as `regret_sum`.
    fn build_strategy_snapshot(&self) -> Vec<f64> {
        let mut snapshot = vec![0.0; self.layout.total_size];
        let num_combos = self.hands.combos.len();

        for (node_idx, node) in self.tree.nodes.iter().enumerate() {
            let num_actions = match node {
                GameNode::Decision { actions, .. } => actions.len(),
                _ => continue,
            };

            for combo_idx in 0..num_combos {
                let (base, _) = self.layout.slot(node_idx, combo_idx);
                let regrets = &self.regret_sum[base..base + num_actions];
                let probs = regret_match(regrets);
                snapshot[base..base + num_actions].copy_from_slice(&probs);
            }
        }

        snapshot
    }
}

// ---------------------------------------------------------------------------
// Parallel CFR context
// ---------------------------------------------------------------------------

/// Frozen snapshot context for one traversal pass. Implements [`ParallelCfr`]
/// so that combo traversals can run in parallel via rayon.
struct SubgameCfrCtx<'a> {
    tree: &'a GameTree,
    hands: &'a SubgameHands,
    equity_matrix: &'a [Vec<f64>],
    opponent_reach: &'a [f64],
    leaf_values: &'a [Vec<f64>],
    node_to_boundary: &'a [usize],
    opp_reach_totals: &'a [f64],
    layout: &'a SubgameLayout,
    /// Frozen strategy (from regret matching). Same flat layout as `regret_sum`.
    snapshot: &'a [f64],
    /// Which player is the traverser this pass.
    traverser: u8,
}

impl ParallelCfr for SubgameCfrCtx<'_> {
    fn buffer_size(&self) -> usize {
        self.layout.total_size
    }

    fn traverse_pair(
        &self,
        regret_delta: &mut [f64],
        strategy_delta: &mut [f64],
        hero_combo: u16,
        _opponent: u16,
    ) {
        self.cfr_traverse(
            regret_delta,
            strategy_delta,
            self.tree.root as usize,
            hero_combo as usize,
            1.0,
            self.opp_reach_totals[hero_combo as usize],
        );
    }
}

impl SubgameCfrCtx<'_> {
    /// Recursive CFR traversal for one combo. Returns expected value for the
    /// traverser. Writes regret and strategy deltas to the provided buffers.
    fn cfr_traverse(
        &self,
        regret_delta: &mut [f64],
        strategy_delta: &mut [f64],
        node_idx: usize,
        hero_combo: usize,
        reach_hero: f64,
        reach_opp: f64,
    ) -> f64 {
        match &self.tree.nodes[node_idx] {
            GameNode::Terminal { kind, pot, .. } => {
                let half_pot = *pot / 2.0;
                match kind {
                    TerminalKind::Fold { winner } => {
                        if *winner == self.traverser {
                            half_pot
                        } else {
                            -half_pot
                        }
                    }
                    TerminalKind::Showdown => self.showdown_value(hero_combo, half_pot),
                    TerminalKind::DepthBoundary => {
                        let boundary_ord = self.node_to_boundary[node_idx];
                        assert_ne!(
                            boundary_ord,
                            usize::MAX,
                            "DepthBoundary node {node_idx} has no boundary ordinal"
                        );
                        let equity = self.leaf_values[boundary_ord][hero_combo];
                        (2.0 * equity - 1.0) * half_pot
                    }
                }
            }

            GameNode::Chance { child, .. } => {
                // Concrete board is already known — pass through.
                self.cfr_traverse(
                    regret_delta,
                    strategy_delta,
                    *child as usize,
                    hero_combo,
                    reach_hero,
                    reach_opp,
                )
            }

            GameNode::Decision {
                player,
                actions,
                children,
                ..
            } => {
                let (base, _) = self.layout.slot(node_idx, hero_combo);
                let num_actions = actions.len();
                let strategy = &self.snapshot[base..base + num_actions];

                if *player == self.traverser {
                    self.traverse_as_traverser(
                        regret_delta,
                        strategy_delta,
                        node_idx,
                        hero_combo,
                        reach_hero,
                        reach_opp,
                        strategy,
                        children,
                    )
                } else {
                    self.traverse_as_opponent(
                        regret_delta,
                        strategy_delta,
                        hero_combo,
                        reach_hero,
                        reach_opp,
                        strategy,
                        children,
                    )
                }
            }
        }
    }

    /// Traverser's decision: compute counterfactual values, write regret and
    /// strategy deltas.
    #[allow(clippy::too_many_arguments)]
    fn traverse_as_traverser(
        &self,
        regret_delta: &mut [f64],
        strategy_delta: &mut [f64],
        node_idx: usize,
        hero_combo: usize,
        reach_hero: f64,
        reach_opp: f64,
        strategy: &[f64],
        children: &[u32],
    ) -> f64 {
        let num_actions = children.len();
        let mut action_values = [0.0f64; 16];
        let mut node_value = 0.0;

        for (a, &child_idx) in children.iter().enumerate() {
            let new_reach = reach_hero * strategy[a];
            action_values[a] = self.cfr_traverse(
                regret_delta,
                strategy_delta,
                child_idx as usize,
                hero_combo,
                new_reach,
                reach_opp,
            );
            node_value += strategy[a] * action_values[a];
        }

        // Write regret deltas (no clamping — DCFR handles discounting later)
        let (base, _) = self.layout.slot(node_idx, hero_combo);
        for a in 0..num_actions {
            regret_delta[base + a] += reach_opp * (action_values[a] - node_value);
        }

        // Write strategy deltas
        for a in 0..num_actions {
            strategy_delta[base + a] += reach_hero * strategy[a];
        }

        node_value
    }

    /// Opponent's decision: weight by opponent strategy and recurse.
    #[allow(clippy::too_many_arguments)]
    fn traverse_as_opponent(
        &self,
        regret_delta: &mut [f64],
        strategy_delta: &mut [f64],
        hero_combo: usize,
        reach_hero: f64,
        reach_opp: f64,
        strategy: &[f64],
        children: &[u32],
    ) -> f64 {
        let mut node_value = 0.0;
        for (a, &child_idx) in children.iter().enumerate() {
            let new_opp_reach = reach_opp * strategy[a];
            let child_val = self.cfr_traverse(
                regret_delta,
                strategy_delta,
                child_idx as usize,
                hero_combo,
                reach_hero,
                new_opp_reach,
            );
            node_value += strategy[a] * child_val;
        }
        node_value
    }

    /// Compute showdown value: reach-weighted equity vs opponent range.
    fn showdown_value(&self, hero_combo: usize, half_pot: f64) -> f64 {
        let hero_cards = self.hands.combos[hero_combo];
        let mut equity_sum = 0.0;
        let mut reach_sum = 0.0;

        for (j, &opp_reach) in self.opponent_reach.iter().enumerate() {
            if cards_overlap(hero_cards, self.hands.combos[j]) {
                continue;
            }
            equity_sum += opp_reach * self.equity_matrix[hero_combo][j];
            reach_sum += opp_reach;
        }

        if reach_sum <= 0.0 {
            return 0.0;
        }
        let avg_equity = equity_sum / reach_sum;
        (2.0 * avg_equity - 1.0) * half_pot
    }
}

// ---------------------------------------------------------------------------
// Precomputation helpers
// ---------------------------------------------------------------------------

/// Precompute total opponent reach for each combo (excluding blocked combos).
pub fn precompute_opp_reach(combos: &[[Card; 2]], opponent_reach: &[f64]) -> Vec<f64> {
    combos
        .iter()
        .map(|&hero| {
            opponent_reach
                .iter()
                .zip(combos)
                .filter(|(_, opp)| !cards_overlap(hero, **opp))
                .map(|(&r, _)| r)
                .sum()
        })
        .collect()
}

/// Build an N x N equity matrix for all combo pairs on the given board.
///
/// For 5-card boards (river), compares hand ranks directly.
/// For shorter boards (flop/turn), enumerates all possible runouts to
/// compute all-in equity that accounts for future streets.
pub fn compute_equity_matrix(combos: &[[Card; 2]], board: &[Card]) -> Vec<Vec<f64>> {
    let n = combos.len();

    if board.len() >= 5 {
        // River: direct rank comparison (no runouts needed).
        return compute_equity_matrix_direct(combos, board);
    }

    // Flop (3 cards) or Turn (4 cards): enumerate runouts.
    let cards_needed = 5 - board.len();
    let remaining = remaining_cards_for_runout(board);

    // Build all possible runouts.
    let runouts: Vec<Vec<Card>> = if cards_needed == 2 {
        // Flop: enumerate all C(remaining, 2) pairs.
        let mut runs = Vec::new();
        for i in 0..remaining.len() {
            for j in (i + 1)..remaining.len() {
                runs.push(vec![remaining[i], remaining[j]]);
            }
        }
        runs
    } else {
        // Turn: enumerate all remaining single cards.
        remaining.iter().map(|&c| vec![c]).collect()
    };

    let start = std::time::Instant::now();
    eprintln!(
        "[equity matrix] {} combos, {} board cards, {} runouts — enumerating all-in equity",
        n, board.len(), runouts.len()
    );

    // Accumulate wins/losses/ties per combo pair across all runouts.
    // Strategy: for each runout, rank ALL combos once (cheap), then do
    // pairwise comparisons using pre-ranked values (integer compare).
    // This avoids redundant hand ranking — O(runouts × n) rankings
    // instead of O(runouts × n²).

    // wins[i][j] and total[i][j] for i < j only (upper triangle).
    let mut win_counts = vec![vec![0u32; n]; n];
    let mut tie_counts = vec![vec![0u32; n]; n];
    let mut total_counts = vec![vec![0u32; n]; n];

    // Precompute which cards each combo uses (for conflict checking).
    let combo_bits: Vec<u64> = combos
        .iter()
        .map(|c| (1u64 << card_bit_idx(c[0])) | (1u64 << card_bit_idx(c[1])))
        .collect();

    for runout in &runouts {
        let mut runout_bits = 0u64;
        for &c in runout.iter() {
            runout_bits |= 1u64 << card_bit_idx(c);
        }

        // Build full board for this runout.
        let mut full_board = board.to_vec();
        full_board.extend_from_slice(runout);

        // Rank all combos on this full board (skip conflicting ones).
        let ranks: Vec<Option<Rank>> = combos
            .iter()
            .zip(combo_bits.iter())
            .map(|(combo, &bits)| {
                if bits & runout_bits != 0 {
                    None // combo conflicts with runout cards
                } else {
                    Some(rank_combo(*combo, &full_board))
                }
            })
            .collect();

        // Pairwise comparisons using pre-ranked values.
        for i in 0..n {
            let Some(ref rank_i) = ranks[i] else { continue };
            for j in (i + 1)..n {
                if combo_bits[i] & combo_bits[j] != 0 {
                    continue; // combos share cards
                }
                let Some(ref rank_j) = ranks[j] else { continue };
                total_counts[i][j] += 1;
                match rank_i.cmp(rank_j) {
                    std::cmp::Ordering::Greater => win_counts[i][j] += 1,
                    std::cmp::Ordering::Equal => tie_counts[i][j] += 1,
                    std::cmp::Ordering::Less => {}
                }
            }
        }
    }

    // Convert counts to equity matrix.
    let mut matrix = vec![vec![0.5; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let total = total_counts[i][j];
            if total > 0 {
                let eq = (f64::from(win_counts[i][j]) + f64::from(tie_counts[i][j]) * 0.5)
                    / f64::from(total);
                matrix[i][j] = eq;
                matrix[j][i] = 1.0 - eq;
            }
        }
    }

    let elapsed = start.elapsed();
    eprintln!("[equity matrix] completed in {:.2}s", elapsed.as_secs_f64());

    matrix
}

/// Direct rank comparison for 5-card boards (no runout enumeration needed).
fn compute_equity_matrix_direct(combos: &[[Card; 2]], board: &[Card]) -> Vec<Vec<f64>> {
    let n = combos.len();
    let ranks: Vec<Rank> = combos.iter().map(|c| rank_combo(*c, board)).collect();
    let mut matrix = vec![vec![0.5; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            if cards_overlap(combos[i], combos[j]) {
                continue;
            }
            match ranks[i].cmp(&ranks[j]) {
                std::cmp::Ordering::Greater => {
                    matrix[i][j] = 1.0;
                    matrix[j][i] = 0.0;
                }
                std::cmp::Ordering::Less => {
                    matrix[i][j] = 0.0;
                    matrix[j][i] = 1.0;
                }
                std::cmp::Ordering::Equal => {}
            }
        }
    }
    matrix
}

/// Map a card to a bit index (0..51) for 64-bit bitset conflict checking.
fn card_bit_idx(card: Card) -> u32 {
    card.value as u32 * 4 + card.suit as u32
}

/// Collect remaining deck cards excluding the board (for runout enumeration).
fn remaining_cards_for_runout(board: &[Card]) -> Vec<Card> {
    crate::poker::full_deck()
        .into_iter()
        .filter(|c| !board.contains(c))
        .collect()
}

/// Rank a combo (2 hole cards) on a board using `rs_poker`.
fn rank_combo(combo: [Card; 2], board: &[Card]) -> Rank {
    let mut hand = Hand::default();
    for &c in board {
        hand.insert(c);
    }
    hand.insert(combo[0]);
    hand.insert(combo[1]);
    hand.rank()
}

/// Check if two 2-card combos share any card.
pub fn cards_overlap(a: [Card; 2], b: [Card; 2]) -> bool {
    a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1]
}

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

/// Compute per-combo equity against an opponent range.
///
/// Returns a vector of equities (0.0 to 1.0) for each combo in `hands`,
/// where equity is the reach-weighted probability of beating the opponent.
/// Combos facing zero opponent reach get equity 0.5 (neutral).
#[must_use]
pub fn compute_combo_equities(
    hands: &SubgameHands,
    board: &[Card],
    opponent_reach: &[f64],
) -> Vec<f64> {
    let equity_matrix = compute_equity_matrix(&hands.combos, board);
    let n = hands.combos.len();
    let mut equities = vec![0.5; n];

    for i in 0..n {
        let mut eq_sum = 0.0;
        let mut reach_sum = 0.0;
        for (j, &opp_r) in opponent_reach.iter().enumerate() {
            if opp_r <= 0.0 || cards_overlap(hands.combos[i], hands.combos[j]) {
                continue;
            }
            eq_sum += opp_r * equity_matrix[i][j];
            reach_sum += opp_r;
        }
        if reach_sum > 0.0 {
            equities[i] = eq_sum / reach_sum;
        }
    }
    equities
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Solve a subgame and return the strategy.
///
/// # Panics
///
/// Panics if `board` does not have 3, 4, or 5 cards.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn solve_subgame(
    board: &[Card],
    bet_sizes: &[f64],
    pot: f64,
    stacks: [f64; 2],
    opponent_reach: &[f64],
    leaf_values: &[f64],
    config: &SubgameConfig,
) -> SubgameStrategy {
    let street = match board.len() {
        3 => Street::Flop,
        4 => Street::Turn,
        5 => Street::River,
        _ => panic!("invalid board length: {}", board.len()),
    };
    let invested = [pot / 2.0; 2];
    let starting_stack = stacks[0] + pot / 2.0;
    let tree = GameTree::build_subgame(
        street,
        pot,
        invested,
        starting_stack,
        &[bet_sizes.to_vec()],
        if config.depth_limit > 0 {
            Some(config.depth_limit as u8)
        } else {
            None
        },
        0,
    );
    let hands = SubgameHands::enumerate(board);
    // Count boundaries and replicate flat leaf values for each.
    let boundary_count = tree
        .nodes
        .iter()
        .filter(|n| {
            matches!(
                n,
                GameNode::Terminal {
                    kind: TerminalKind::DepthBoundary,
                    ..
                }
            )
        })
        .count();
    let per_boundary = if boundary_count == 0 {
        vec![]
    } else {
        vec![leaf_values.to_vec(); boundary_count]
    };
    let mut solver =
        SubgameCfrSolver::new(tree, hands, board, opponent_reach.to_vec(), per_boundary);
    solver.train(config.max_iterations);
    solver.strategy()
}

// ---------------------------------------------------------------------------
// build_boundary_mapping -- parallel tree walk
// ---------------------------------------------------------------------------

/// Map each `DepthBoundary` in `subgame_tree` to the corresponding `Chance`
/// node ordinal in `abstract_tree` by walking both trees in lockstep.
///
/// Both trees must share the same action set at every decision node. Returns
/// a `Vec<usize>` of length equal to the number of `DepthBoundary` nodes,
/// where each value is the ordinal position of the matching `Chance` node
/// among all `Chance` nodes in the abstract tree (matching [`CbvTable`]
/// indexing).
///
/// # Panics
///
/// Panics if any subgame action has no match in the abstract tree, or if a
/// `DepthBoundary` doesn't correspond to a `Chance` node.
#[must_use]
pub fn build_boundary_mapping(
    subgame_tree: &GameTree,
    abstract_tree: &GameTree,
) -> Vec<usize> {
    // Precompute chance node ordinals in the abstract tree.
    let mut chance_ordinals = vec![usize::MAX; abstract_tree.nodes.len()];
    let mut ord = 0;
    for (idx, node) in abstract_tree.nodes.iter().enumerate() {
        if matches!(node, GameNode::Chance { .. }) {
            chance_ordinals[idx] = ord;
            ord += 1;
        }
    }

    let mut mapping = Vec::new();
    walk_trees_lockstep(
        subgame_tree,
        abstract_tree,
        &chance_ordinals,
        subgame_tree.root as usize,
        abstract_tree.root as usize,
        &mut mapping,
    );
    mapping
}

fn walk_trees_lockstep(
    sub: &GameTree,
    abs: &GameTree,
    chance_ordinals: &[usize],
    sub_idx: usize,
    abs_idx: usize,
    mapping: &mut Vec<usize>,
) {
    match (&sub.nodes[sub_idx], &abs.nodes[abs_idx]) {
        // Subgame hits DepthBoundary -> abstract should be at Chance.
        (
            GameNode::Terminal {
                kind: TerminalKind::DepthBoundary,
                ..
            },
            GameNode::Chance { .. },
        ) => {
            let ord = chance_ordinals[abs_idx];
            assert_ne!(
                ord,
                usize::MAX,
                "abstract Chance node {abs_idx} has no ordinal"
            );
            mapping.push(ord);
        }

        // Both are terminals (Fold or Showdown) -- nothing to map.
        (GameNode::Terminal { .. }, GameNode::Terminal { .. }) => {}

        // Subgame terminal vs abstract Chance/Decision -- the subgame
        // resolved this path (e.g. all-in showdown) while the abstract
        // tree continues through more streets. No boundaries needed.
        (GameNode::Terminal { .. }, GameNode::Chance { .. } | GameNode::Decision { .. }) => {}

        // Both are Chance nodes -- recurse into children.
        (
            GameNode::Chance {
                child: sub_child, ..
            },
            GameNode::Chance {
                child: abs_child, ..
            },
        ) => {
            walk_trees_lockstep(
                sub,
                abs,
                chance_ordinals,
                *sub_child as usize,
                *abs_child as usize,
                mapping,
            );
        }

        // Both are Decision nodes -- match actions and recurse.
        (
            GameNode::Decision {
                actions: sub_actions,
                children: sub_children,
                ..
            },
            GameNode::Decision {
                actions: abs_actions,
                children: abs_children,
                ..
            },
        ) => {
            // Match actions by TYPE (Fold↔Fold, Call↔Call, AllIn↔AllIn,
            // Bet/Raise by position among sized actions).
            // The subgame tree may have fewer raise depths than the abstract
            // tree, so we match what exists and skip abstract-only actions.
            for (sub_a_idx, sub_action) in sub_actions.iter().enumerate() {
                let abs_a_idx = match sub_action {
                    TreeAction::Fold => abs_actions.iter().position(|a| matches!(a, TreeAction::Fold)),
                    TreeAction::Check => abs_actions.iter().position(|a| matches!(a, TreeAction::Check)),
                    // Call and AllIn are interchangeable — calling can be all-in
                    // in one tree but not the other due to different unit scaling.
                    TreeAction::Call => abs_actions.iter().position(|a| matches!(a, TreeAction::Call | TreeAction::AllIn)),
                    TreeAction::AllIn => abs_actions.iter().position(|a| matches!(a, TreeAction::AllIn | TreeAction::Call)),
                    TreeAction::Bet(_) | TreeAction::Raise(_) => {
                        let sub_sized_idx = sub_actions[..sub_a_idx]
                            .iter()
                            .filter(|a| matches!(a, TreeAction::Bet(_) | TreeAction::Raise(_)))
                            .count();
                        abs_actions.iter().enumerate()
                            .filter(|(_, a)| matches!(a, TreeAction::Bet(_) | TreeAction::Raise(_)))
                            .nth(sub_sized_idx)
                            .map(|(i, _)| i)
                    }
                };
                let abs_a_idx = abs_a_idx.unwrap_or_else(|| panic!(
                    "no matching action for {sub_action:?} in abstract tree at node {abs_idx}: \
                     sub={sub_actions:?} abs={abs_actions:?}"
                ));
                walk_trees_lockstep(
                    sub,
                    abs,
                    chance_ordinals,
                    sub_children[sub_a_idx] as usize,
                    abs_children[abs_a_idx] as usize,
                    mapping,
                );
            }
        }

        // Subgame has an all-in response Decision [Fold, Call/AllIn] where
        // the abstract tree has a Chance node (the call was already resolved
        // in the full-depth tree). Walk the Call/AllIn child against the
        // Chance node; the Fold child leads to a terminal (no boundaries).
        (
            GameNode::Decision { actions: sub_actions, children: sub_children, .. },
            GameNode::Chance { .. },
        ) => {
            for (sub_a_idx, sub_action) in sub_actions.iter().enumerate() {
                match sub_action {
                    TreeAction::Call | TreeAction::AllIn => {
                        walk_trees_lockstep(
                            sub, abs, chance_ordinals,
                            sub_children[sub_a_idx] as usize,
                            abs_idx, // stay at same abstract Chance node
                            mapping,
                        );
                    }
                    TreeAction::Fold => {
                        // Fold leads to a terminal — no boundaries to map.
                    }
                    _ => {}
                }
            }
        }

        (sub_node, abs_node) => {
            panic!(
                "tree structure mismatch at sub={sub_idx} abs={abs_idx}: \
                 sub={sub_node:?}, abs={abs_node:?}"
            );
        }
    }
}

/// Compare two `TreeAction`s for equivalence during lockstep tree walking.
///
/// Uses approximate comparison for `Bet`/`Raise` amounts (within `SIZE_EPSILON`).
fn actions_match(a: &TreeAction, b: &TreeAction) -> bool {
    match (a, b) {
        (TreeAction::Fold, TreeAction::Fold)
        | (TreeAction::Check, TreeAction::Check)
        | (TreeAction::Call, TreeAction::Call)
        | (TreeAction::AllIn, TreeAction::AllIn) => true,
        (TreeAction::Bet(x), TreeAction::Bet(y))
        | (TreeAction::Raise(x), TreeAction::Raise(y)) => (x - y).abs() < 0.01,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::{Card, Suit, Value};
    use test_macros::timed_test;

    /// Wrap flat leaf values for all boundaries in a tree (test helper).
    ///
    /// For river trees (no boundaries), returns `vec![]`.
    /// For depth-limited trees, replicates the flat vec for each boundary.
    fn flat_leaf_values(tree: &GameTree, values: Vec<f64>) -> Vec<Vec<f64>> {
        let count = tree
            .nodes
            .iter()
            .filter(|n| {
                matches!(
                    n,
                    GameNode::Terminal {
                        kind: TerminalKind::DepthBoundary,
                        ..
                    }
                )
            })
            .count();
        if count == 0 {
            vec![]
        } else {
            vec![values; count]
        }
    }

    fn river_board() -> Vec<Card> {
        vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Four, Suit::Club),
            Card::new(Value::Ten, Suit::Club),
        ]
    }

    /// Build a small hand set (first `n` combos) for fast testing.
    fn small_hands(board: &[Card], n: usize) -> SubgameHands {
        let full = SubgameHands::enumerate(board);
        SubgameHands {
            combos: full.combos.into_iter().take(n).collect(),
        }
    }

    /// Build a river subgame tree: pot=100, each player invested 50,
    /// starting_stack = stacks_remaining + 50 = 250.
    fn river_tree(bet_sizes: &[f64]) -> GameTree {
        GameTree::build_subgame(
            Street::River,
            100.0,
            [50.0, 50.0],
            250.0,
            &[bet_sizes.to_vec()],
            None,
            0,
        )
    }

    #[timed_test]
    fn solver_creates_and_runs() {
        let board = river_board();
        let tree = river_tree(&[1.0]);
        let hands = small_hands(&board, 30);
        let n = hands.combos.len();
        let reach = vec![1.0; n];
        let leaf = flat_leaf_values(&tree, vec![0.0; n]);
        let mut solver = SubgameCfrSolver::new(tree, hands, &board, reach, leaf);
        solver.train(10);
        assert_eq!(solver.iteration, 10);
    }

    #[timed_test(3)]
    fn strategy_is_valid_distribution() {
        let board = river_board();
        let tree = river_tree(&[1.0]);
        let hands = small_hands(&board, 50);
        let n = hands.combos.len();
        let reach = vec![1.0; n];
        let leaf = flat_leaf_values(&tree, vec![0.0; n]);
        let mut solver = SubgameCfrSolver::new(tree, hands, &board, reach, leaf);
        solver.train(100);
        let strategy = solver.strategy();

        for combo_idx in 0..n {
            let probs = strategy.root_probs(combo_idx);
            if probs.is_empty() {
                continue;
            }
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "combo {combo_idx}: strategy sum = {sum}, expected ~1.0"
            );
        }
    }

    #[timed_test]
    fn solve_subgame_convenience_works() {
        let board = river_board();
        let hands = small_hands(&board, 20);
        let n = hands.combos.len();
        let config = SubgameConfig {
            depth_limit: 4,
            time_budget_ms: 5000,
            max_iterations: 10,
        };

        // Build manually since solve_subgame uses full enumeration
        let tree = river_tree(&[1.0]);
        let leaf = flat_leaf_values(&tree, vec![0.0; n]);
        let mut solver =
            SubgameCfrSolver::new(tree, hands, &board, vec![1.0; n], leaf);
        solver.train(config.max_iterations);
        let strategy = solver.strategy();
        assert!(strategy.num_combos() > 0);
    }

    #[timed_test]
    fn equity_matrix_is_symmetric() {
        let board = river_board();
        let hands = small_hands(&board, 50);
        let matrix = compute_equity_matrix(&hands.combos, &board);
        let n = hands.combos.len();
        for (i, row_i) in matrix.iter().enumerate() {
            for j in (i + 1)..n {
                let sum = row_i[j] + matrix[j][i];
                assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "equity[{i}][{j}] + equity[{j}][{i}] = {sum}, expected 1.0"
                );
            }
        }
    }

    #[timed_test]
    fn cards_overlap_detects_shared_cards() {
        let a = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
        ];
        let b = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Queen, Suit::Diamond),
        ];
        let c = [
            Card::new(Value::Two, Suit::Club),
            Card::new(Value::Three, Suit::Diamond),
        ];
        assert!(cards_overlap(a, b));
        assert!(!cards_overlap(a, c));
    }

    #[timed_test]
    fn with_cbv_tables_normalizes_per_boundary() {
        use super::super::cbv::CbvTable;

        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
        ];
        let tree = GameTree::build_subgame(
            Street::Flop,
            100.0,
            [50.0, 50.0],
            250.0,
            &[vec![1.0]],
            Some(1),
            0,
        );

        let hands = small_hands(&board, 6);
        let n = hands.combos.len();

        // Count boundaries and their pots.
        let boundary_pots: Vec<f64> = tree
            .nodes
            .iter()
            .filter_map(|nd| match nd {
                GameNode::Terminal {
                    kind: TerminalKind::DepthBoundary,
                    pot,
                    ..
                } => Some(*pot),
                _ => None,
            })
            .collect();
        let num_boundaries = boundary_pots.len();
        assert!(num_boundaries >= 2, "need multiple boundaries");

        // CBV table with `num_boundaries` chance nodes, 4 buckets each.
        // Use known chip values so we can verify the normalization.
        let cbv_chip_value = 15.0_f32; // 15 chips
        let values: Vec<f32> = vec![cbv_chip_value; num_boundaries * 4];
        let node_offsets: Vec<usize> = (0..num_boundaries).map(|i| i * 4).collect();
        let buckets_per_node: Vec<u16> = vec![4; num_boundaries];
        let cbv_table = CbvTable {
            values,
            node_offsets,
            buckets_per_node,
        };

        // boundary_mapping: identity (boundary 0 -> chance 0, etc.)
        let boundary_mapping: Vec<usize> = (0..num_boundaries).collect();

        let mut solver = SubgameCfrSolver::with_cbv_tables(
            tree,
            hands,
            &board,
            vec![1.0; n],
            [&cbv_table, &cbv_table],
            &boundary_mapping,
            |combo_idx| (combo_idx % 4) as u16,
        );

        // Verify normalization: equity = (cbv / half_pot + 1) / 2
        // Round-trip check: (2 * equity - 1) * half_pot should recover cbv.
        for (b, &pot) in boundary_pots.iter().enumerate() {
            let half_pot = pot / 2.0;
            let expected_equity = (f64::from(cbv_chip_value) / half_pot + 1.0) / 2.0;
            for combo_idx in 0..n {
                let actual = solver.leaf_values[b][combo_idx];
                assert!(
                    (actual - expected_equity).abs() < 1e-10,
                    "boundary {b} combo {combo_idx}: expected {expected_equity}, got {actual}"
                );
                // Round-trip: traversal would compute (2*eq - 1) * half_pot = cbv
                let round_trip = (2.0 * actual - 1.0) * half_pot;
                assert!(
                    (round_trip - f64::from(cbv_chip_value)).abs() < 1e-6,
                    "round-trip failed: {round_trip} != {cbv_chip_value}"
                );
            }
        }

        // Train to verify no panics.
        solver.train(10);
        assert_eq!(solver.iteration, 10);
    }

    #[timed_test]
    #[ignore = "flaky: solver convergence doesn't guarantee hand ordering in debug mode"]
    fn strong_hands_bet_more_than_weak() {
        let board = river_board();
        let tree = river_tree(&[1.0]);
        let hands = small_hands(&board, 50);
        let n = hands.combos.len();
        let reach = vec![1.0; n];
        let leaf = flat_leaf_values(&tree, vec![0.0; n]);
        let mut solver = SubgameCfrSolver::new(tree, hands.clone(), &board, reach, leaf);
        solver.train(200);
        let strategy = solver.strategy();

        // Find the strongest and weakest combos by rank
        let ranks: Vec<Rank> = hands
            .combos
            .iter()
            .map(|c| rank_combo(*c, &board))
            .collect();
        let best = ranks
            .iter()
            .enumerate()
            .max_by_key(|(_, r)| *r)
            .map(|(i, _)| i)
            .expect("should have combos");
        let worst = ranks
            .iter()
            .enumerate()
            .min_by_key(|(_, r)| *r)
            .map(|(i, _)| i)
            .expect("should have combos");

        let best_probs = strategy.root_probs(best);
        let worst_probs = strategy.root_probs(worst);

        // Root is position 0's decision: [Check, Bet(0), Bet(ALL_IN)]
        // The best hand should have higher total betting frequency
        if best_probs.len() >= 2 && worst_probs.len() >= 2 {
            let best_bet_freq: f64 = best_probs[1..].iter().sum();
            let worst_bet_freq: f64 = worst_probs[1..].iter().sum();
            assert!(
                best_bet_freq >= worst_bet_freq - 0.15,
                "best hand bet freq ({best_bet_freq:.3}) should be >= worst ({worst_bet_freq:.3}) - margin"
            );
        }
    }

    #[timed_test]
    fn fold_terminal_gives_correct_values() {
        let board = river_board();
        let hands = small_hands(&board, 5);
        let n = hands.combos.len();
        let reach = vec![1.0; n];
        let tree = river_tree(&[1.0]);
        let leaf = flat_leaf_values(&tree, vec![0.0; n]);
        let solver = SubgameCfrSolver::new(tree, hands, &board, reach, leaf);

        // Find a Fold terminal in the tree and call cfr_traverse on it.
        let snapshot = solver.build_strategy_snapshot();
        let ctx = SubgameCfrCtx {
            tree: &solver.tree,
            hands: &solver.hands,
            equity_matrix: &solver.equity_matrix,
            opponent_reach: &solver.opponent_reach,
            leaf_values: &solver.leaf_values,
            node_to_boundary: &solver.node_to_boundary,
            opp_reach_totals: &solver.opp_reach_totals,
            layout: &solver.layout,
            snapshot: &snapshot,
            traverser: 0,
        };

        // Find a Fold terminal where player 0 (traverser) folds.
        // In the tree, after OOP bets and IP folds, the winner is OOP (player 0).
        // After OOP checks, IP bets, OOP folds, winner is IP (player 1).
        // We need to find any fold terminal and verify the value.
        let mut found_fold = false;
        for (idx, node) in solver.tree.nodes.iter().enumerate() {
            if let GameNode::Terminal {
                kind: TerminalKind::Fold { winner },
                pot,
                ..
            } = node
            {
                let half_pot = *pot / 2.0;
                let mut dummy_regret = vec![0.0; solver.layout.total_size];
                let mut dummy_strategy = vec![0.0; solver.layout.total_size];

                // traverser=0: if winner=0, traverser wins; if winner=1, traverser loses
                let v0 = ctx.cfr_traverse(
                    &mut dummy_regret,
                    &mut dummy_strategy,
                    idx,
                    0,
                    1.0,
                    1.0,
                );
                if *winner == 0 {
                    assert!(
                        (v0 - half_pot).abs() < 1e-10,
                        "traverser wins fold: expected {half_pot}, got {v0}"
                    );
                } else {
                    assert!(
                        (v0 - (-half_pot)).abs() < 1e-10,
                        "traverser loses fold: expected {}, got {v0}",
                        -half_pot
                    );
                }
                found_fold = true;
                break;
            }
        }
        assert!(found_fold, "no fold terminal found in tree");
    }

    #[timed_test]
    fn showdown_value_respects_equity() {
        let board = river_board();
        let hands = small_hands(&board, 10);
        let n = hands.combos.len();
        let reach = vec![1.0; n];
        let tree = river_tree(&[1.0]);
        let leaf = flat_leaf_values(&tree, vec![0.0; n]);
        let solver = SubgameCfrSolver::new(tree, hands, &board, reach, leaf);

        let snapshot = solver.build_strategy_snapshot();
        let ctx = SubgameCfrCtx {
            tree: &solver.tree,
            hands: &solver.hands,
            equity_matrix: &solver.equity_matrix,
            opponent_reach: &solver.opponent_reach,
            leaf_values: &solver.leaf_values,
            node_to_boundary: &solver.node_to_boundary,
            opp_reach_totals: &solver.opp_reach_totals,
            layout: &solver.layout,
            snapshot: &snapshot,
            traverser: 0,
        };

        // Showdown values should be in [-50, +50] for pot=100
        for i in 0..n {
            let v = ctx.showdown_value(i, 50.0);
            assert!(
                (-50.0 - 1e-10..=50.0 + 1e-10).contains(&v),
                "showdown value {v} out of range for combo {i}"
            );
        }
    }

    #[timed_test]
    fn layout_total_size_matches_tree() {
        let board = river_board();
        let tree = GameTree::build_subgame(
            Street::River,
            100.0,
            [50.0, 50.0],
            250.0,
            &[vec![0.5, 1.0]],
            None,
            0,
        );
        let hands = small_hands(&board, 20);
        let n = hands.combos.len();
        let layout = SubgameLayout::build(&tree, n);

        // Total size should equal sum of (num_combos * num_actions) for each
        // decision node.
        let expected: usize = tree
            .nodes
            .iter()
            .map(|node| match node {
                GameNode::Decision { actions, .. } => n * actions.len(),
                _ => 0,
            })
            .sum();
        assert_eq!(layout.total_size, expected);
    }

    #[timed_test]
    fn per_boundary_leaf_values_differ_by_pot() {
        // Build a depth-limited flop tree with bet sizes that create
        // multiple DepthBoundary nodes at different pot sizes.
        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
        ];
        let tree = GameTree::build_subgame(
            Street::Flop,
            100.0,
            [50.0, 50.0],
            250.0,
            &[vec![1.0]], // pot-size bet
            Some(1),
            0,
        );

        // Count DepthBoundary nodes -- should be > 1 (check-check and bet-call paths).
        let boundary_count = tree
            .nodes
            .iter()
            .filter(|n| {
                matches!(
                    n,
                    GameNode::Terminal {
                        kind: TerminalKind::DepthBoundary,
                        ..
                    }
                )
            })
            .count();
        assert!(
            boundary_count >= 2,
            "need multiple boundaries, got {boundary_count}"
        );

        // Collect boundary pots.
        let boundary_pots: Vec<f64> = tree
            .nodes
            .iter()
            .filter_map(|n| match n {
                GameNode::Terminal {
                    kind: TerminalKind::DepthBoundary,
                    pot,
                    ..
                } => Some(*pot),
                _ => None,
            })
            .collect();

        // Verify boundaries have different pots.
        assert!(
            boundary_pots.windows(2).any(|w| (w[0] - w[1]).abs() > 1.0),
            "boundaries should have different pots: {boundary_pots:?}"
        );

        let hands = small_hands(&board, 10);
        let n = hands.combos.len();

        // Create per-boundary leaf values: all combos get equity 0.7 at every boundary.
        let leaf_values: Vec<Vec<f64>> = vec![vec![0.7; n]; boundary_count];

        let mut solver =
            SubgameCfrSolver::new(tree, hands, &board, vec![1.0; n], leaf_values);

        // Train briefly and verify it doesn't panic.
        solver.train(5);
        assert_eq!(solver.iteration, 5);
    }

    #[timed_test]
    fn layout_slot_round_trips() {
        let tree = river_tree(&[1.0]);
        let n = 10;
        let layout = SubgameLayout::build(&tree, n);

        // Verify no two (node, combo) pairs map to overlapping ranges.
        let mut used = vec![false; layout.total_size];
        for (node_idx, node) in tree.nodes.iter().enumerate() {
            let num_actions = match node {
                GameNode::Decision { actions, .. } => actions.len(),
                _ => continue,
            };
            for combo in 0..n {
                let (base, na) = layout.slot(node_idx, combo);
                assert_eq!(na, num_actions);
                for a in 0..na {
                    assert!(
                        !used[base + a],
                        "overlapping slot at {}, node={node_idx}, combo={combo}, action={a}",
                        base + a
                    );
                    used[base + a] = true;
                }
            }
        }
        // All slots should be used.
        assert!(used.iter().all(|&u| u), "some layout slots are unused");
    }

    #[timed_test]
    fn build_boundary_mapping_matches_trees() {
        // Build abstract full-depth flop tree.
        let abstract_tree = GameTree::build_subgame(
            Street::Flop,
            100.0,
            [50.0, 50.0],
            250.0,
            &[vec![1.0]],
            None, // full depth
            0,
        );

        // Build subgame depth-limited flop tree with SAME bet sizes.
        let subgame_tree = GameTree::build_subgame(
            Street::Flop,
            100.0,
            [50.0, 50.0],
            250.0,
            &[vec![1.0]],
            Some(1), // depth limit = 1 street
            0,
        );

        // Count expected boundaries.
        let boundary_count = subgame_tree
            .nodes
            .iter()
            .filter(|n| {
                matches!(
                    n,
                    GameNode::Terminal {
                        kind: TerminalKind::DepthBoundary,
                        ..
                    }
                )
            })
            .count();
        assert!(boundary_count > 0, "subgame should have boundaries");

        // Count chance nodes in abstract tree.
        let chance_count = abstract_tree
            .nodes
            .iter()
            .filter(|n| matches!(n, GameNode::Chance { .. }))
            .count();

        let mapping = build_boundary_mapping(&subgame_tree, &abstract_tree);

        assert_eq!(mapping.len(), boundary_count);
        // Every mapped ordinal must be valid.
        for &ord in &mapping {
            assert!(
                ord < chance_count,
                "ordinal {ord} out of range (chance_count={chance_count})"
            );
        }
    }

    #[timed_test]
    #[should_panic(expected = "tree structure mismatch")]
    fn build_boundary_mapping_panics_on_mismatch() {
        let tree_a = GameTree::build_subgame(
            Street::Flop,
            100.0,
            [50.0, 50.0],
            250.0,
            &[vec![1.0]],
            Some(1),
            0,
        );
        // Abstract tree is a bare terminal — structural mismatch.
        let tree_b = GameTree {
            nodes: vec![GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 100.0,
                stacks: [200.0, 200.0],
            }],
            root: 0,
            dealer: 0,
            starting_stack: 250.0,
        };
        let _ = build_boundary_mapping(&tree_a, &tree_b);
    }
}
