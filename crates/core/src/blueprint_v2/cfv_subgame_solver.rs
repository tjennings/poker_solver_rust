//! CFV-based subgame solver for depth-limited solving.
//!
//! Trains a subgame tree using counterfactual value (CFV) estimates at
//! depth boundaries. Unlike [`SubgameCfrSolver`], which uses a single
//! per-combo leaf value, this solver:
//!
//! 1. Propagates both players' ranges through the tree each iteration.
//! 2. Calls a [`LeafEvaluator`] at each depth boundary with the current
//!    opponent range, producing per-combo CFVs.
//! 3. Uses those CFVs during CFR traversal.

use crate::blueprint::{
    cards_overlap, compute_equity_matrix, precompute_opp_reach, SubgameHands, SubgameStrategy,
};
use crate::blueprint_v2::game_tree::{GameNode, GameTree, TerminalKind};
use crate::cfr::dcfr::DcfrParams;
use crate::cfr::parallel::{add_into, parallel_traverse, ParallelCfr};
use crate::cfr::regret::regret_match;
use crate::poker::Card;

// ---------------------------------------------------------------------------
// LeafEvaluator trait
// ---------------------------------------------------------------------------

/// Evaluates combos at a depth boundary, producing per-combo CFVs.
///
/// Implementations may solve sub-trees exactly (e.g. all river runouts)
/// or use a neural network approximation.
pub trait LeafEvaluator: Send + Sync {
    /// Evaluate counterfactual values for a set of combos at a boundary.
    ///
    /// Returns a `Vec<f64>` of length `combos.len()`, where each entry is
    /// the expected value (in pot fractions) for the traversing player.
    ///
    /// # Arguments
    ///
    /// * `combos` - hole card combos active at this boundary
    /// * `board` - current board cards (3-5 cards)
    /// * `pot` - total pot at the boundary
    /// * `effective_stack` - remaining stack for each player
    /// * `oop_range` - OOP per-combo reach probabilities
    /// * `ip_range` - IP per-combo reach probabilities
    /// * `traverser` - which player (0=OOP, 1=IP) is the traverser
    fn evaluate(
        &self,
        combos: &[[Card; 2]],
        board: &[Card],
        pot: f64,
        effective_stack: f64,
        oop_range: &[f64],
        ip_range: &[f64],
        traverser: u8,
    ) -> Vec<f64>;
}

// ---------------------------------------------------------------------------
// BoundaryInfo — data about each depth boundary node
// ---------------------------------------------------------------------------

/// Information about depth boundary nodes in the tree.
///
/// Each entry records: `(node_index, pot, invested)`.
#[derive(Debug, Clone)]
pub struct BoundaryInfo {
    pub boundaries: Vec<(usize, f64, [f64; 2])>,
}

// ---------------------------------------------------------------------------
// CfvLayout — flat buffer layout (mirrors SubgameLayout)
// ---------------------------------------------------------------------------

/// Maps `(node_idx, combo_idx)` to flat buffer offsets for decision nodes.
struct CfvLayout {
    /// Per-node base offset. Non-decision nodes store `usize::MAX`.
    bases: Vec<usize>,
    /// Per-node action count. Non-decision nodes store 0.
    num_actions: Vec<usize>,
    /// Total flat buffer size.
    total_size: usize,
}

impl CfvLayout {
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
    #[inline]
    fn slot(&self, node_idx: usize, combo_idx: usize) -> (usize, usize) {
        let na = self.num_actions[node_idx];
        let base = self.bases[node_idx] + combo_idx * na;
        (base, na)
    }
}

// ---------------------------------------------------------------------------
// CfvSubgameSolver
// ---------------------------------------------------------------------------

/// CFV-based parallel DCFR solver for depth-limited subgame trees.
///
/// At each iteration, propagates ranges to depth boundaries, calls the
/// [`LeafEvaluator`] to compute per-combo CFVs, then runs CFR traversal
/// using those values at boundary nodes.
pub struct CfvSubgameSolver {
    tree: GameTree,
    hands: SubgameHands,
    board: Vec<Card>,
    equity_matrix: Vec<Vec<f64>>,
    regret_sum: Vec<f64>,
    strategy_sum: Vec<f64>,
    layout: CfvLayout,
    /// Per-boundary, per-combo leaf CFVs. Indexed by `[boundary_ordinal][combo]`.
    leaf_cfvs: Vec<Vec<f64>>,
    boundary_info: BoundaryInfo,
    /// Maps tree node index to boundary ordinal (or `usize::MAX` if not a boundary).
    node_to_boundary: Vec<usize>,
    evaluator: Box<dyn LeafEvaluator>,
    dcfr: DcfrParams,
    pub iteration: u32,
    starting_stack: f64,
    /// Precomputed: for each combo, the total opponent reach of non-blocked combos.
    opp_reach_totals: Vec<f64>,
}

impl CfvSubgameSolver {
    /// Create a new CFV subgame solver.
    ///
    /// # Arguments
    ///
    /// * `tree` - the game tree (may contain `DepthBoundary` terminals)
    /// * `hands` - valid hole card combos for the board
    /// * `board` - the board cards (3-5 cards)
    /// * `evaluator` - leaf evaluator for depth boundaries
    /// * `starting_stack` - the starting stack for each player
    #[must_use]
    pub fn new(
        tree: GameTree,
        hands: SubgameHands,
        board: &[Card],
        evaluator: Box<dyn LeafEvaluator>,
        starting_stack: f64,
    ) -> Self {
        let equity_matrix = compute_equity_matrix(&hands.combos, board);
        let n = hands.combos.len();

        // Uniform initial reach for opp_reach_totals precomputation.
        let uniform_reach = vec![1.0; n];
        let opp_reach_totals = precompute_opp_reach(&hands.combos, &uniform_reach);

        let layout = CfvLayout::build(&tree, n);
        let buf_size = layout.total_size;

        // Discover depth boundary nodes.
        let mut boundaries = Vec::new();
        let mut node_to_boundary = vec![usize::MAX; tree.nodes.len()];
        for (idx, node) in tree.nodes.iter().enumerate() {
            if let GameNode::Terminal {
                kind: TerminalKind::DepthBoundary,
                pot,
                invested,
            } = node
            {
                let ordinal = boundaries.len();
                boundaries.push((idx, *pot, *invested));
                node_to_boundary[idx] = ordinal;
            }
        }

        let num_boundaries = boundaries.len();
        let leaf_cfvs = vec![vec![0.0; n]; num_boundaries];

        Self {
            tree,
            hands,
            board: board.to_vec(),
            equity_matrix,
            regret_sum: vec![0.0; buf_size],
            strategy_sum: vec![0.0; buf_size],
            layout,
            leaf_cfvs,
            boundary_info: BoundaryInfo { boundaries },
            node_to_boundary,
            evaluator,
            dcfr: DcfrParams::default(),
            iteration: 0,
            starting_stack,
            opp_reach_totals,
        }
    }

    /// Build a frozen strategy snapshot from current regret sums.
    ///
    /// Each `(decision_node, combo)` pair gets regret-matched into a
    /// probability distribution. Uses the same flat layout as `regret_sum`.
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

    /// Propagate both players' ranges through the tree using the given
    /// strategy snapshot.
    ///
    /// Returns `(oop_ranges, ip_ranges)` where each is a `Vec<Vec<f64>>`
    /// indexed by `[boundary_ordinal][combo_idx]`.
    fn propagate_ranges(
        &self,
        snapshot: &[f64],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = self.hands.combos.len();
        let num_boundaries = self.boundary_info.boundaries.len();

        let mut oop_ranges = vec![vec![0.0; n]; num_boundaries];
        let mut ip_ranges = vec![vec![0.0; n]; num_boundaries];

        for combo_idx in 0..n {
            self.propagate_combo(
                snapshot,
                self.tree.root as usize,
                combo_idx,
                1.0, // OOP starts with reach 1.0
                1.0, // IP starts with reach 1.0
                &mut oop_ranges,
                &mut ip_ranges,
            );
        }

        (oop_ranges, ip_ranges)
    }

    /// Recursive range propagation for a single combo.
    ///
    /// Walks the tree, multiplying reaches by strategy probabilities at
    /// decision nodes, and recording reaches at depth boundary nodes.
    #[allow(clippy::too_many_arguments)]
    fn propagate_combo(
        &self,
        snapshot: &[f64],
        node_idx: usize,
        combo_idx: usize,
        oop_reach: f64,
        ip_reach: f64,
        oop_ranges: &mut [Vec<f64>],
        ip_ranges: &mut [Vec<f64>],
    ) {
        match &self.tree.nodes[node_idx] {
            GameNode::Terminal { kind, .. } => {
                if *kind == TerminalKind::DepthBoundary {
                    let ordinal = self.node_to_boundary[node_idx];
                    if ordinal != usize::MAX {
                        oop_ranges[ordinal][combo_idx] += oop_reach;
                        ip_ranges[ordinal][combo_idx] += ip_reach;
                    }
                }
                // Fold / Showdown: no propagation needed.
            }

            GameNode::Chance { child, .. } => {
                self.propagate_combo(
                    snapshot,
                    *child as usize,
                    combo_idx,
                    oop_reach,
                    ip_reach,
                    oop_ranges,
                    ip_ranges,
                );
            }

            GameNode::Decision {
                player,
                actions,
                children,
                ..
            } => {
                let (base, _) = self.layout.slot(node_idx, combo_idx);
                let num_actions = actions.len();
                let strategy = &snapshot[base..base + num_actions];

                for (a, &child_idx) in children.iter().enumerate() {
                    let (new_oop, new_ip) = if *player == 0 {
                        (oop_reach * strategy[a], ip_reach)
                    } else {
                        (oop_reach, ip_reach * strategy[a])
                    };
                    self.propagate_combo(
                        snapshot,
                        child_idx as usize,
                        combo_idx,
                        new_oop,
                        new_ip,
                        oop_ranges,
                        ip_ranges,
                    );
                }
            }
        }
    }

    /// Run parallel DCFR for the given number of iterations.
    ///
    /// Each iteration:
    /// 1. Builds a strategy snapshot from current regret sums.
    /// 2. Propagates ranges to depth boundaries.
    /// 3. Evaluates leaf CFVs at each boundary using the [`LeafEvaluator`].
    /// 4. Runs CFR traversal for both traverser positions.
    /// 5. Applies DCFR discounting.
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

            let snapshot = self.build_strategy_snapshot();

            // Propagate ranges to boundaries and evaluate leaf CFVs.
            if !self.boundary_info.boundaries.is_empty() {
                let (oop_ranges, ip_ranges) = self.propagate_ranges(&snapshot);

                for (b_idx, &(_, pot, invested)) in
                    self.boundary_info.boundaries.iter().enumerate()
                {
                    let eff_stack =
                        self.starting_stack - invested[0].max(invested[1]);

                    // Evaluate for each traverser and store the results.
                    // We'll use traverser=0 for the iteration (OOP perspective).
                    // The actual per-traverser leaf CFVs are handled below.
                    self.leaf_cfvs[b_idx] = self.evaluator.evaluate(
                        &self.hands.combos,
                        &self.board,
                        pot,
                        eff_stack,
                        &oop_ranges[b_idx],
                        &ip_ranges[b_idx],
                        0, // evaluated from OOP perspective initially
                    );
                }
            }

            for traverser in 0..2u8 {
                // Re-evaluate boundaries from current traverser's perspective
                // if there are boundaries.
                if !self.boundary_info.boundaries.is_empty() && traverser == 1 {
                    let (oop_ranges, ip_ranges) =
                        self.propagate_ranges(&snapshot);
                    for (b_idx, &(_, pot, invested)) in
                        self.boundary_info.boundaries.iter().enumerate()
                    {
                        let eff_stack = self.starting_stack
                            - invested[0].max(invested[1]);
                        self.leaf_cfvs[b_idx] = self.evaluator.evaluate(
                            &self.hands.combos,
                            &self.board,
                            pot,
                            eff_stack,
                            &oop_ranges[b_idx],
                            &ip_ranges[b_idx],
                            traverser,
                        );
                    }
                }

                let ctx = CfvCfrCtx {
                    tree: &self.tree,
                    hands: &self.hands,
                    equity_matrix: &self.equity_matrix,
                    opp_reach_totals: &self.opp_reach_totals,
                    layout: &self.layout,
                    snapshot: &snapshot,
                    leaf_cfvs: &self.leaf_cfvs,
                    node_to_boundary: &self.node_to_boundary,
                    traverser,
                };

                let (regret_delta, strategy_delta) =
                    parallel_traverse(&ctx, &combos);
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
        let mut strategies =
            rustc_hash::FxHashMap::default();
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
                    let probs: Vec<f64> =
                        sums.iter().map(|&s| s / total).collect();
                    let key = SubgameStrategy::key(
                        node_idx as u32,
                        combo_idx as u32,
                    );
                    strategies.insert(key, probs);
                }
            }
        }

        SubgameStrategy {
            strategies,
            num_combos,
        }
    }
}

// ---------------------------------------------------------------------------
// CfvCfrCtx — parallel CFR context with per-boundary leaf CFVs
// ---------------------------------------------------------------------------

/// Frozen snapshot context for one CFR traversal pass. Like `SubgameCfrCtx`
/// but uses per-boundary leaf CFV vectors instead of a single leaf_values
/// vector.
struct CfvCfrCtx<'a> {
    tree: &'a GameTree,
    hands: &'a SubgameHands,
    equity_matrix: &'a [Vec<f64>],
    opp_reach_totals: &'a [f64],
    layout: &'a CfvLayout,
    snapshot: &'a [f64],
    /// Per-boundary leaf CFVs: `[boundary_ordinal][combo_idx]`.
    leaf_cfvs: &'a [Vec<f64>],
    /// Maps node index to boundary ordinal.
    node_to_boundary: &'a [usize],
    traverser: u8,
}

impl ParallelCfr for CfvCfrCtx<'_> {
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

impl CfvCfrCtx<'_> {
    /// Recursive CFR traversal for one combo. Returns expected value for
    /// the traverser.
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
                    TerminalKind::Showdown => {
                        self.showdown_value(hero_combo, half_pot)
                    }
                    TerminalKind::DepthBoundary => {
                        let ordinal = self.node_to_boundary[node_idx];
                        if ordinal != usize::MAX {
                            // leaf_cfvs are in pot-fraction units; scale by
                            // half_pot to get chip values.
                            self.leaf_cfvs[ordinal]
                                .get(hero_combo)
                                .copied()
                                .unwrap_or(0.0)
                                * half_pot
                        } else {
                            0.0
                        }
                    }
                }
            }

            GameNode::Chance { child, .. } => self.cfr_traverse(
                regret_delta,
                strategy_delta,
                *child as usize,
                hero_combo,
                reach_hero,
                reach_opp,
            ),

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

    /// Traverser's decision: compute counterfactual values, write regret
    /// and strategy deltas.
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

        let (base, _) = self.layout.slot(node_idx, hero_combo);
        for a in 0..num_actions {
            regret_delta[base + a] +=
                reach_opp * (action_values[a] - node_value);
        }
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
        let opp_reach_total = self.opp_reach_totals[hero_combo];
        if opp_reach_total <= 0.0 {
            return 0.0;
        }

        let mut equity_sum = 0.0;
        // Use uniform opponent reach (1.0 per combo) since opp_reach_totals
        // was computed from uniform reach. The reach weighting is already
        // baked into the CFR traversal structure.
        for (j, eq_row) in self.equity_matrix[hero_combo].iter().enumerate() {
            if cards_overlap(hero_cards, self.hands.combos[j]) {
                continue;
            }
            equity_sum += eq_row;
        }

        let avg_equity = equity_sum / opp_reach_total;
        (2.0 * avg_equity - 1.0) * half_pot
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_v2::game_tree::GameTree;
    use crate::blueprint_v2::Street;
    use crate::poker::{Card, Suit, Value};
    use test_macros::timed_test;

    /// A trivial leaf evaluator that returns a constant value for all combos.
    struct ConstantEvaluator(f64);

    impl LeafEvaluator for ConstantEvaluator {
        fn evaluate(
            &self,
            combos: &[[Card; 2]],
            _board: &[Card],
            _pot: f64,
            _effective_stack: f64,
            _oop_range: &[f64],
            _ip_range: &[f64],
            _traverser: u8,
        ) -> Vec<f64> {
            vec![self.0; combos.len()]
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

    fn turn_board() -> Vec<Card> {
        vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Four, Suit::Club),
        ]
    }

    fn small_hands(board: &[Card], n: usize) -> SubgameHands {
        let full = SubgameHands::enumerate(board);
        SubgameHands {
            combos: full.combos.into_iter().take(n).collect(),
        }
    }

    fn river_tree(bet_sizes: &[f64]) -> GameTree {
        GameTree::build_subgame(
            Street::River,
            100.0,
            [50.0, 50.0],
            250.0,
            &[bet_sizes.to_vec()],
            None,
        )
    }

    fn turn_tree_depth_limited(bet_sizes: &[f64], depth_limit: u8) -> GameTree {
        GameTree::build_subgame(
            Street::Turn,
            100.0,
            [50.0, 50.0],
            250.0,
            &[bet_sizes.to_vec()],
            Some(depth_limit),
        )
    }

    #[timed_test(5)]
    fn solver_constructs_without_panic() {
        let board = river_board();
        let tree = river_tree(&[1.0]);
        let hands = small_hands(&board, 30);
        let evaluator = Box::new(ConstantEvaluator(0.5));
        let solver = CfvSubgameSolver::new(tree, hands, &board, evaluator, 250.0);

        assert_eq!(solver.iteration, 0);
        assert!(!solver.hands.combos.is_empty());
        // River tree has no depth boundaries.
        assert!(solver.boundary_info.boundaries.is_empty());
    }

    #[timed_test(5)]
    fn propagate_ranges_sums_correctly() {
        let board = turn_board();
        let tree = turn_tree_depth_limited(&[1.0], 1);
        let hands = small_hands(&board, 30);
        let n = hands.combos.len();
        let evaluator = Box::new(ConstantEvaluator(0.5));
        let solver = CfvSubgameSolver::new(tree, hands, &board, evaluator, 250.0);

        // With no training, strategy is uniform. Propagation should still
        // produce non-negative ranges that sum to at most 1.0 per combo.
        let snapshot = solver.build_strategy_snapshot();
        let (oop_ranges, ip_ranges) = solver.propagate_ranges(&snapshot);

        assert!(
            !oop_ranges.is_empty(),
            "turn tree with depth_limit=1 should have boundaries"
        );

        // All range values must be non-negative.
        for boundary_ranges in oop_ranges.iter().chain(ip_ranges.iter()) {
            for &r in boundary_ranges {
                assert!(r >= 0.0, "negative range value: {r}");
            }
        }

        // Per-combo total reach across all boundaries must be <= 1.0.
        // (Reach can be less than 1.0 because fold/showdown terminals
        // consume some probability mass.)
        for combo_idx in 0..n {
            let oop_total: f64 = oop_ranges.iter().map(|r| r[combo_idx]).sum();
            let ip_total: f64 = ip_ranges.iter().map(|r| r[combo_idx]).sum();
            assert!(
                oop_total <= 1.0 + 1e-10,
                "combo {combo_idx}: OOP total reach {oop_total} > 1.0"
            );
            assert!(
                ip_total <= 1.0 + 1e-10,
                "combo {combo_idx}: IP total reach {ip_total} > 1.0"
            );
        }
    }

    #[timed_test(30)]
    fn solver_trains_and_produces_strategy() {
        let board = river_board();
        let tree = river_tree(&[1.0]);
        let hands = small_hands(&board, 30);
        let evaluator = Box::new(ConstantEvaluator(0.5));
        let mut solver =
            CfvSubgameSolver::new(tree, hands, &board, evaluator, 250.0);

        solver.train(100);
        assert_eq!(solver.iteration, 100);

        let strategy = solver.strategy();
        let n = strategy.num_combos();
        assert!(n > 0);

        // Verify valid probability distributions at the root.
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
            for (a, &p) in probs.iter().enumerate() {
                assert!(
                    p >= -1e-10,
                    "combo {combo_idx} action {a}: negative prob {p}"
                );
            }
        }
    }

    #[timed_test(30)]
    fn solver_with_depth_boundary_converges() {
        let board = turn_board();
        let tree = turn_tree_depth_limited(&[1.0], 1);
        let hands = small_hands(&board, 30);

        // Verify the tree actually has DepthBoundary nodes.
        let has_boundary = tree.nodes.iter().any(|n| {
            matches!(
                n,
                GameNode::Terminal {
                    kind: TerminalKind::DepthBoundary,
                    ..
                }
            )
        });
        assert!(
            has_boundary,
            "turn tree with depth_limit=1 should have DepthBoundary nodes"
        );

        let evaluator = Box::new(ConstantEvaluator(0.5));
        let mut solver =
            CfvSubgameSolver::new(tree, hands, &board, evaluator, 250.0);

        assert!(
            !solver.boundary_info.boundaries.is_empty(),
            "solver should detect boundaries"
        );

        solver.train(200);
        assert_eq!(solver.iteration, 200);

        let strategy = solver.strategy();
        let n = strategy.num_combos();
        assert!(n > 0);

        // Verify all strategy distributions are valid.
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
            for (a, &p) in probs.iter().enumerate() {
                assert!(
                    p >= -1e-10,
                    "combo {combo_idx} action {a}: negative prob {p}"
                );
            }
        }
    }

    #[timed_test(30)]
    fn solver_trains_no_boundaries() {
        let board = river_board();
        let tree = river_tree(&[1.0]);
        let hands = small_hands(&board, 30);

        // Verify no boundaries exist in a river tree.
        let has_boundary = tree.nodes.iter().any(|n| {
            matches!(
                n,
                GameNode::Terminal {
                    kind: TerminalKind::DepthBoundary,
                    ..
                }
            )
        });
        assert!(
            !has_boundary,
            "river tree should have no DepthBoundary nodes"
        );

        let evaluator = Box::new(ConstantEvaluator(0.5));
        let mut solver =
            CfvSubgameSolver::new(tree, hands, &board, evaluator, 250.0);

        assert!(
            solver.boundary_info.boundaries.is_empty(),
            "solver should have no boundaries for river tree"
        );

        solver.train(100);
        assert_eq!(solver.iteration, 100);

        let strategy = solver.strategy();
        assert!(strategy.num_combos() > 0);
    }
}
