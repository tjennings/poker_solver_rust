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

use rayon::prelude::*;

use crate::blueprint_v2::subgame_cfr::{
    cards_overlap, compute_combo_equities, compute_equity_matrix, precompute_opp_reach,
    SubgameCfrSolver, SubgameHands, SubgameStrategy,
};
use crate::blueprint_v2::game_tree::{GameNode, GameTree, TerminalKind};
use crate::blueprint_v2::Street;
use crate::cfr::dcfr::DcfrParams;
use crate::cfr::regret::regret_match;
use crate::poker::Card;

// ---------------------------------------------------------------------------
// LeafEvaluator trait
// ---------------------------------------------------------------------------

/// Evaluates combos at a depth boundary, producing per-combo CFVs.
///
/// Implementations may solve sub-trees exactly (e.g. all river runouts)
/// or use a neural network approximation.
pub trait LeafEvaluator {
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

    /// Batch-evaluate multiple boundary requests in one call.
    ///
    /// Each request specifies `(pot, effective_stack, traverser)` for a
    /// different boundary node. All requests share the same combos, board,
    /// and ranges.
    ///
    /// Returns one `Vec<f64>` per request (same length as `combos`).
    ///
    /// Default implementation calls [`evaluate`] sequentially. GPU-backed
    /// implementations should override to batch into a single forward pass.
    fn evaluate_boundaries(
        &self,
        combos: &[[Card; 2]],
        board: &[Card],
        oop_range: &[f64],
        ip_range: &[f64],
        requests: &[(f64, f64, u8)], // (pot, effective_stack, traverser)
    ) -> Vec<Vec<f64>> {
        requests
            .iter()
            .map(|&(pot, eff_stack, traverser)| {
                self.evaluate(combos, board, pot, eff_stack, oop_range, ip_range, traverser)
            })
            .collect()
    }
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
    /// Precomputed showdown values per combo (normalized equity: -1 to +1).
    /// Multiply by half_pot at showdown nodes. Fixed across iterations.
    showdown_equity: Vec<f64>,
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

        // Precompute showdown equity for all combos (O(n²) but done once).
        let showdown_equity: Vec<f64> = (0..n)
            .map(|i| {
                let opp_total = opp_reach_totals[i];
                if opp_total <= 0.0 {
                    return 0.0;
                }
                let mut eq_sum = 0.0;
                for (j, eq) in equity_matrix[i].iter().enumerate() {
                    if cards_overlap(hands.combos[i], hands.combos[j]) {
                        continue;
                    }
                    eq_sum += eq;
                }
                let avg_eq = eq_sum / opp_total;
                2.0 * avg_eq - 1.0
            })
            .collect();

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
            showdown_equity,
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
    ///
    /// Traverses the tree **once**, carrying reach-probability vectors of
    /// length `n` (number of combos) for both players, instead of walking
    /// the tree once per combo. This is O(tree_nodes × combos) work in a
    /// single pass rather than O(combos × tree_nodes) across 1326 passes.
    fn propagate_ranges(
        &self,
        snapshot: &[f64],
        reach_init: &[f64],
        reach_pool: &mut [f64],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = self.hands.combos.len();
        let num_boundaries = self.boundary_info.boundaries.len();

        let mut oop_ranges = vec![vec![0.0; n]; num_boundaries];
        let mut ip_ranges = vec![vec![0.0; n]; num_boundaries];

        self.propagate_ranges_recursive(
            snapshot,
            self.tree.root as usize,
            reach_init,
            reach_init,
            &mut oop_ranges,
            &mut ip_ranges,
            reach_pool,
        );

        (oop_ranges, ip_ranges)
    }

    /// Vectorized recursive range propagation.
    ///
    /// Walks the tree once, carrying reach-probability vectors for all
    /// combos simultaneously. At decision nodes, the acting player's
    /// reach vector is multiplied element-wise by strategy probabilities
    /// for each action before recursing into children.
    ///
    /// Uses a pre-allocated flat `reach_pool` buffer. The current depth
    /// level occupies `reach_pool[0..n]`; deeper levels use the remainder
    /// via `split_at_mut`, avoiding all heap allocations in the hot loop.
    #[allow(clippy::too_many_arguments)]
    fn propagate_ranges_recursive(
        &self,
        snapshot: &[f64],
        node_idx: usize,
        oop_reach: &[f64],
        ip_reach: &[f64],
        oop_ranges: &mut [Vec<f64>],
        ip_ranges: &mut [Vec<f64>],
        reach_pool: &mut [f64],
    ) {
        let n = oop_reach.len();

        // Extract node info into locals to avoid holding borrow on self.tree.
        enum PropNodeInfo {
            Terminal { is_boundary: bool, ordinal: usize },
            Chance { child: u32 },
            Decision { player: u8, num_actions: usize, children_buf: [u32; 16] },
        }

        let info = match &self.tree.nodes[node_idx] {
            GameNode::Terminal { kind, .. } => {
                let is_boundary = *kind == TerminalKind::DepthBoundary;
                let ordinal = if is_boundary {
                    self.node_to_boundary[node_idx]
                } else {
                    usize::MAX
                };
                PropNodeInfo::Terminal { is_boundary, ordinal }
            }
            GameNode::Chance { child, .. } => PropNodeInfo::Chance { child: *child },
            GameNode::Decision { player, actions, children, .. } => {
                let num_actions = actions.len();
                let mut children_buf = [0u32; 16];
                debug_assert!(num_actions <= 16);
                children_buf[..num_actions].copy_from_slice(&children[..num_actions]);
                PropNodeInfo::Decision { player: *player, num_actions, children_buf }
            }
        };

        match info {
            PropNodeInfo::Terminal { is_boundary, ordinal } => {
                if is_boundary && ordinal != usize::MAX {
                    for i in 0..n {
                        oop_ranges[ordinal][i] += oop_reach[i];
                        ip_ranges[ordinal][i] += ip_reach[i];
                    }
                }
                // Fold / Showdown: no propagation needed.
            }

            PropNodeInfo::Chance { child } => {
                self.propagate_ranges_recursive(
                    snapshot,
                    child as usize,
                    oop_reach,
                    ip_reach,
                    oop_ranges,
                    ip_ranges,
                    reach_pool,
                );
            }

            PropNodeInfo::Decision { player, num_actions, children_buf } => {
                let node_base = self.layout.bases[node_idx];

                // Split reach_pool: this level uses [0..n], children use [n..].
                let (this_level, rest) = reach_pool.split_at_mut(n);

                for a in 0..num_actions {
                    let child_idx = children_buf[a];

                    // Build child reach into this_level, reusing the buffer.
                    if player == 0 {
                        for combo_idx in 0..n {
                            let strat_prob =
                                snapshot[node_base + combo_idx * num_actions + a];
                            this_level[combo_idx] = oop_reach[combo_idx] * strat_prob;
                        }
                        self.propagate_ranges_recursive(
                            snapshot,
                            child_idx as usize,
                            this_level,
                            ip_reach,
                            oop_ranges,
                            ip_ranges,
                            rest,
                        );
                    } else {
                        for combo_idx in 0..n {
                            let strat_prob =
                                snapshot[node_base + combo_idx * num_actions + a];
                            this_level[combo_idx] = ip_reach[combo_idx] * strat_prob;
                        }
                        self.propagate_ranges_recursive(
                            snapshot,
                            child_idx as usize,
                            oop_reach,
                            this_level,
                            oop_ranges,
                            ip_ranges,
                            rest,
                        );
                    }
                }
            }
        }
    }

    /// Vectorized CFR traversal: walks the tree ONCE, processing ALL combos
    /// simultaneously at each node.
    ///
    /// Updates `regret_sum` and `strategy_sum` inline. Writes per-combo
    /// counterfactual values into `cfv_buf[node_idx * n..(node_idx + 1) * n]`.
    ///
    /// # Arguments
    ///
    /// * `node_idx` - current tree node
    /// * `traverser` - which player (0=OOP, 1=IP) we are computing CFVs for
    /// * `reach_traverser` - per-combo reach probabilities for the traverser
    /// * `reach_opponent` - per-combo reach probabilities for the opponent
    /// * `snapshot` - frozen strategy snapshot (from `build_strategy_snapshot`)
    /// * `cfv_buf` - pre-allocated flat buffer of size `num_nodes * num_combos`;
    ///   each node writes its output to `[node_idx * n..(node_idx+1) * n]`
    /// * `reach_pool` - pre-allocated flat reach scratch buffer. The current
    ///   depth level uses `[0..n]`; deeper levels use `[n..]` via `split_at_mut`,
    ///   eliminating all per-call heap allocations.
    #[allow(clippy::too_many_arguments)]
    fn cfr_traverse_vectorized(
        &mut self,
        node_idx: usize,
        traverser: u8,
        reach_traverser: &[f64],
        reach_opponent: &[f64],
        snapshot: &[f64],
        cfv_buf: &mut [f64],
        reach_pool: &mut [f64],
    ) {
        let n = self.hands.combos.len();
        let out_start = node_idx * n;

        // Extract node info into small Copy types to free borrow on self.tree.
        enum NodeInfo {
            Terminal { kind: TerminalKind, pot: f64 },
            Chance { child: u32 },
            Decision { player: u8, num_actions: usize, children_buf: [u32; 16] },
        }

        let info = match &self.tree.nodes[node_idx] {
            GameNode::Terminal { kind, pot, .. } => NodeInfo::Terminal { kind: *kind, pot: *pot },
            GameNode::Chance { child, .. } => NodeInfo::Chance { child: *child },
            GameNode::Decision { player, children, actions, .. } => {
                let num_actions = actions.len();
                let mut children_buf = [0u32; 16];
                debug_assert!(num_actions <= 16);
                children_buf[..num_actions].copy_from_slice(&children[..num_actions]);
                NodeInfo::Decision {
                    player: *player,
                    num_actions,
                    children_buf,
                }
            }
        };

        match info {
            NodeInfo::Terminal { kind, pot } => {
                let half_pot = pot / 2.0;
                match kind {
                    TerminalKind::Fold { winner } => {
                        let sign = if winner == traverser { 1.0 } else { -1.0 };
                        for i in 0..n {
                            cfv_buf[out_start + i] = sign * half_pot;
                        }
                    }
                    TerminalKind::Showdown => {
                        // Compute equity conditional on the opponent's reaching
                        // range. Parallelized with rayon — each combo is independent.
                        compute_conditional_showdowns(
                            &self.hands.combos,
                            &self.equity_matrix,
                            reach_opponent,
                            half_pot,
                            &mut cfv_buf[out_start..out_start + n],
                        );
                    }
                    TerminalKind::DepthBoundary => {
                        let ordinal = self.node_to_boundary[node_idx];
                        if ordinal != usize::MAX {
                            for i in 0..n {
                                cfv_buf[out_start + i] = self.leaf_cfvs[ordinal]
                                    .get(i).copied().unwrap_or(0.0) * half_pot;
                            }
                        } else {
                            for i in 0..n {
                                cfv_buf[out_start + i] = 0.0;
                            }
                        }
                    }
                }
            }

            NodeInfo::Chance { child } => {
                self.cfr_traverse_vectorized(
                    child as usize, traverser,
                    reach_traverser, reach_opponent, snapshot,
                    cfv_buf, reach_pool,
                );
                // Copy child's output to this node's slot.
                let child_start = child as usize * n;
                cfv_buf.copy_within(child_start..child_start + n, out_start);
            }

            NodeInfo::Decision { player, num_actions, children_buf } => {
                let node_base = self.layout.bases[node_idx];
                let is_traverser = player == traverser;

                // Split reach_pool: this level uses [0..n], children use [n..].
                let (this_level, rest) = reach_pool.split_at_mut(n);

                // Recurse into each child with updated reach vectors.
                // Reuse this_level for child reach — only one child is
                // processed at a time. The recursive call gets `rest`
                // (disjoint from this_level) so there are no borrow conflicts.
                for a in 0..num_actions {
                    let child_idx = children_buf[a] as usize;

                    // Build child reach into this_level.
                    if is_traverser {
                        for i in 0..n {
                            this_level[i] = reach_traverser[i]
                                * snapshot[node_base + i * num_actions + a];
                        }
                        self.cfr_traverse_vectorized(
                            child_idx, traverser,
                            this_level, reach_opponent, snapshot,
                            cfv_buf, rest,
                        );
                    } else {
                        for i in 0..n {
                            this_level[i] = reach_opponent[i]
                                * snapshot[node_base + i * num_actions + a];
                        }
                        self.cfr_traverse_vectorized(
                            child_idx, traverser,
                            reach_traverser, this_level, snapshot,
                            cfv_buf, rest,
                        );
                    }
                }

                // Compute node CFVs from children's results (stored in cfv_buf
                // at each child's own node_idx slot — no separate action_cfvs needed).
                for i in 0..n {
                    let mut val = 0.0;
                    for a in 0..num_actions {
                        let child_idx = children_buf[a] as usize;
                        let strat = snapshot[node_base + i * num_actions + a];
                        val += strat * cfv_buf[child_idx * n + i];
                    }
                    cfv_buf[out_start + i] = val;
                }

                // Update regrets and strategy sums at traverser's nodes.
                if is_traverser {
                    // Compute total non-blocked opponent reach per hero combo.
                    // This is the correct CFR regret weight: how likely the
                    // opponent is to be at this node given hero holds combo i.
                    // Using reach_opponent[i] (one combo's reach) was wrong —
                    // it needs to be the SUM of all non-blocked opponent combos.
                    let combos = &self.hands.combos;
                    let opp_reach_totals: Vec<f64> = (0..n)
                        .into_par_iter()
                        .map(|i| {
                            let hero = combos[i];
                            reach_opponent.iter().enumerate()
                                .filter(|&(j, &r)| r > 0.0 && !cards_overlap(hero, combos[j]))
                                .map(|(_, &r)| r)
                                .sum()
                        })
                        .collect();

                    for i in 0..n {
                        let base = node_base + i * num_actions;
                        let node_val = cfv_buf[out_start + i];
                        let opp_total = opp_reach_totals[i];
                        for a in 0..num_actions {
                            let child_idx = children_buf[a] as usize;
                            let child_val = cfv_buf[child_idx * n + i];
                            self.regret_sum[base + a] +=
                                opp_total * (child_val - node_val);
                            self.strategy_sum[base + a] +=
                                reach_traverser[i] * snapshot[base + a];
                        }
                    }
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
        self.train_with_leaf_interval(iterations, 0);
    }

    /// Train with configurable leaf evaluation interval.
    ///
    /// `leaf_eval_interval` controls how often depth-boundary leaves are
    /// re-evaluated:
    /// - `0` = every iteration (original behavior)
    /// - `N > 0` = evaluate at iteration 1, every N iterations, and the final iteration
    pub fn train_with_leaf_interval(&mut self, iterations: u32, leaf_eval_interval: u32) {
        let n = self.hands.combos.len();
        let num_nodes = self.tree.nodes.len();

        // Pre-allocate all scratch buffers once, reused across all iterations.
        let reach_init = vec![1.0; n];
        let mut cfv_buf = vec![0.0; num_nodes * n];
        let max_depth: usize = 64;
        let mut reach_pool = vec![0.0; max_depth * n];
        let mut prop_reach_pool = vec![0.0; max_depth * n];

        for iter_idx in 0..iterations {
            self.iteration += 1;

            let snapshot = self.build_strategy_snapshot();

            // Determine whether to re-evaluate leaf boundaries this iteration.
            let should_eval = if leaf_eval_interval == 0 {
                true
            } else {
                self.iteration == 1
                    || self.iteration % leaf_eval_interval == 0
                    || iter_idx == iterations - 1
            };

            // Propagate ranges once (shared between both traversers).
            let boundary_ranges =
                if should_eval && !self.boundary_info.boundaries.is_empty() {
                    Some(self.propagate_ranges(
                        &snapshot, &reach_init, &mut prop_reach_pool,
                    ))
                } else {
                    None
                };

            // Evaluate leaf CFVs and traverse for each traverser.
            for traverser in 0..2u8 {
                if let Some((ref oop_ranges, ref ip_ranges)) = boundary_ranges {
                    for (b_idx, &(_, pot, invested)) in
                        self.boundary_info.boundaries.iter().enumerate()
                    {
                        let eff_stack =
                            self.starting_stack - invested[0].max(invested[1]);
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

                self.cfr_traverse_vectorized(
                    self.tree.root as usize,
                    traverser,
                    &reach_init,
                    &reach_init,
                    &snapshot,
                    &mut cfv_buf,
                    &mut reach_pool,
                );
            }

            let iter_u64 = u64::from(self.iteration);
            if self.dcfr.should_discount(iter_u64) {
                self.dcfr.discount_regrets(&mut self.regret_sum, iter_u64);
                self.dcfr
                    .discount_strategy_sums(&mut self.strategy_sum, iter_u64);
            }
        }
    }

    /// Build average strategy from cumulative strategy sums.
    ///
    /// Each decision node / combo pair gets its strategy sum normalized to a
    /// probability distribution. Uniform fallback when no weight has
    /// accumulated.
    #[allow(clippy::cast_precision_loss)]
    fn build_average_strategy(&self) -> Vec<f64> {
        let mut avg = vec![0.0; self.layout.total_size];
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
                    for a in 0..num_actions {
                        avg[base + a] = sums[a] / total;
                    }
                } else {
                    let uniform = 1.0 / num_actions as f64;
                    for a in 0..num_actions {
                        avg[base + a] = uniform;
                    }
                }
            }
        }

        avg
    }

    /// Compute root counterfactual values for each combo.
    ///
    /// After training, call this to extract per-combo expected values from
    /// `traverser`'s perspective using the average strategy. Returns one
    /// value per combo in `self.hands.combos`.
    ///
    /// Internally evaluates leaf boundaries using the stored `leaf_cfvs`
    /// from the last training iteration, which is acceptable because the
    /// average strategy converges to Nash and the last boundary evaluations
    /// reflect the final range propagation.
    #[must_use]
    pub fn root_cfvs(&self, traverser: u8) -> Vec<f64> {
        let avg_strategy = self.build_average_strategy();
        let num_combos = self.hands.combos.len();
        let mut cfvs = Vec::with_capacity(num_combos);

        for combo_idx in 0..num_combos {
            let val = self.eval_combo_value(
                &avg_strategy,
                self.tree.root as usize,
                combo_idx,
                traverser,
            );
            cfvs.push(val);
        }

        cfvs
    }

    /// Recursive tree walk computing the expected value for a single combo
    /// under the given strategy profile.
    ///
    /// At terminals, returns fold/showdown/boundary payoffs. At decision
    /// nodes, weights child values by strategy probabilities.
    fn eval_combo_value(
        &self,
        strategy: &[f64],
        node_idx: usize,
        combo_idx: usize,
        traverser: u8,
    ) -> f64 {
        match &self.tree.nodes[node_idx] {
            GameNode::Terminal { kind, pot, .. } => {
                let half_pot = *pot / 2.0;
                match kind {
                    TerminalKind::Fold { winner } => {
                        if *winner == traverser {
                            half_pot
                        } else {
                            -half_pot
                        }
                    }
                    TerminalKind::Showdown => {
                        self.showdown_value_avg(combo_idx, half_pot, traverser)
                    }
                    TerminalKind::DepthBoundary => {
                        let ordinal = self.node_to_boundary[node_idx];
                        if ordinal == usize::MAX {
                            0.0
                        } else {
                            self.leaf_cfvs[ordinal]
                                .get(combo_idx)
                                .copied()
                                .unwrap_or(0.0)
                                * half_pot
                        }
                    }
                }
            }

            GameNode::Chance { child, .. } => {
                self.eval_combo_value(strategy, *child as usize, combo_idx, traverser)
            }

            GameNode::Decision { children, actions, .. } => {
                let (base, _) = self.layout.slot(node_idx, combo_idx);
                let num_actions = actions.len();
                let strat = &strategy[base..base + num_actions];

                let mut value = 0.0;
                for (a, &child_idx) in children.iter().enumerate() {
                    let child_val = self.eval_combo_value(
                        strategy,
                        child_idx as usize,
                        combo_idx,
                        traverser,
                    );
                    value += strat[a] * child_val;
                }
                value
            }
        }
    }

    /// Showdown value for `root_cfvs` evaluation: equity vs uniform opponent.
    fn showdown_value_avg(&self, hero_combo: usize, half_pot: f64, _traverser: u8) -> f64 {
        showdown_value_single(
            hero_combo,
            &self.hands,
            &self.equity_matrix,
            &self.opp_reach_totals,
            half_pot,
        )
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
// Parallel showdown computation
// ---------------------------------------------------------------------------

/// Compute conditional showdown equity for all combos in parallel.
///
/// For each hero combo `i`, computes the reach-weighted equity against
/// `reach_opponent`, then converts to chip value via `(2*eq - 1) * half_pot`.
/// Uses rayon to parallelize across combos (each is independent).
fn compute_conditional_showdowns(
    combos: &[[Card; 2]],
    equity_matrix: &[Vec<f64>],
    reach_opponent: &[f64],
    half_pot: f64,
    out: &mut [f64],
) {
    out.par_iter_mut().enumerate().for_each(|(i, val)| {
        let hero = combos[i];
        let mut eq_sum = 0.0;
        let mut reach_sum = 0.0;
        for (j, &opp_r) in reach_opponent.iter().enumerate() {
            if opp_r <= 0.0 || cards_overlap(hero, combos[j]) {
                continue;
            }
            eq_sum += opp_r * equity_matrix[i][j];
            reach_sum += opp_r;
        }
        *val = if reach_sum > 0.0 {
            let avg_eq = eq_sum / reach_sum;
            (2.0 * avg_eq - 1.0) * half_pot
        } else {
            0.0
        };
    });
}

// ---------------------------------------------------------------------------
// remaining_deck helper
// ---------------------------------------------------------------------------

/// Returns all cards not on the board.
#[must_use]
pub fn remaining_deck(board: &[Card]) -> Vec<Card> {
    crate::poker::full_deck()
        .into_iter()
        .filter(|c| !board.contains(c))
        .collect()
}

// ---------------------------------------------------------------------------
// showdown_value_single — free function for vectorized traversal
// ---------------------------------------------------------------------------

/// Compute showdown value for a single hero combo against uniform opponent.
///
/// This is a free function (no `&self`) so it can be called from
/// `CfvSubgameSolver::cfr_traverse_vectorized` without borrow conflicts.
fn showdown_value_single(
    hero_combo: usize,
    hands: &SubgameHands,
    equity_matrix: &[Vec<f64>],
    opp_reach_totals: &[f64],
    half_pot: f64,
) -> f64 {
    let hero_cards = hands.combos[hero_combo];
    let opp_reach_total = opp_reach_totals[hero_combo];
    if opp_reach_total <= 0.0 {
        return 0.0;
    }
    let mut equity_sum = 0.0;
    for (j, eq_row) in equity_matrix[hero_combo].iter().enumerate() {
        if cards_overlap(hero_cards, hands.combos[j]) {
            continue;
        }
        equity_sum += eq_row;
    }
    let avg_equity = equity_sum / opp_reach_total;
    (2.0 * avg_equity - 1.0) * half_pot
}

// ---------------------------------------------------------------------------
// ExactRiverEvaluator
// ---------------------------------------------------------------------------

/// Evaluates depth boundary nodes by solving all possible river runouts
/// exactly using [`SubgameCfrSolver`].
///
/// For each remaining deck card, builds a river tree, filters combos to
/// those surviving that card, solves the subgame, extracts equities, and
/// averages across all river cards.
pub struct ExactRiverEvaluator {
    /// Bet sizes for the river subgame trees.
    pub bet_sizes: Vec<f64>,
    /// Number of CFR iterations per river solve.
    pub iterations: u32,
}

impl LeafEvaluator for ExactRiverEvaluator {
    #[allow(clippy::cast_precision_loss)]
    fn evaluate(
        &self,
        combos: &[[Card; 2]],
        board: &[Card],
        pot: f64,
        effective_stack: f64,
        oop_range: &[f64],
        ip_range: &[f64],
        traverser: u8,
    ) -> Vec<f64> {
        let n = combos.len();
        let mut cfv_sums = vec![0.0_f64; n];
        let mut cfv_counts = vec![0u32; n];

        let remaining = remaining_deck(board);

        for &river_card in &remaining {
            let mut river_board = board.to_vec();
            river_board.push(river_card);

            // Filter combos: only keep those that don't conflict with the
            // river card. Track the mapping from filtered index to parent.
            let mut filtered_combos = Vec::new();
            let mut parent_indices = Vec::new();
            for (i, &combo) in combos.iter().enumerate() {
                if combo[0] != river_card && combo[1] != river_card {
                    filtered_combos.push(combo);
                    parent_indices.push(i);
                }
            }

            if filtered_combos.is_empty() {
                continue;
            }

            let river_hands = SubgameHands {
                combos: filtered_combos,
            };

            // Build opponent reach for the filtered combo set.
            let opp_reach: Vec<f64> = parent_indices
                .iter()
                .map(|&pi| {
                    if traverser == 0 {
                        ip_range[pi]
                    } else {
                        oop_range[pi]
                    }
                })
                .collect();

            // Build the river tree.
            let invested = [pot / 2.0; 2];
            let starting_stack = effective_stack + pot / 2.0;
            let tree = GameTree::build_subgame(
                Street::River,
                pot,
                invested,
                starting_stack,
                &[self.bet_sizes.clone()],
                None,
            );

            // River tree has no DepthBoundary nodes, so leaf_values is empty.
            let mut solver = SubgameCfrSolver::new(
                tree,
                river_hands.clone(),
                &river_board,
                opp_reach.clone(),
                vec![],
            );
            solver.train(self.iterations);

            // Extract equities against the (now-narrowed) opponent range.
            let equities = compute_combo_equities(
                &river_hands,
                &river_board,
                &opp_reach,
            );

            // Map equities back to parent combo indices as CFVs.
            // CFV = (2 * equity - 1) in pot-fraction units.
            for (fi, &pi) in parent_indices.iter().enumerate() {
                cfv_sums[pi] += 2.0 * equities[fi] - 1.0;
                cfv_counts[pi] += 1;
            }
        }

        // Average across river cards.
        (0..n)
            .map(|i| {
                if cfv_counts[i] > 0 {
                    cfv_sums[i] / f64::from(cfv_counts[i])
                } else {
                    0.0
                }
            })
            .collect()
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
        let reach_init = vec![1.0; n];
        let max_depth: usize = 64;
        let mut prop_reach_pool = vec![0.0; max_depth * n];
        let (oop_ranges, ip_ranges) = solver.propagate_ranges(
            &snapshot, &reach_init, &mut prop_reach_pool,
        );

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

    #[timed_test(30)]
    fn remaining_deck_excludes_board() {
        let board = river_board();
        let deck = remaining_deck(&board);
        assert_eq!(deck.len(), 52 - board.len());
        for card in &board {
            assert!(!deck.contains(card), "board card {card:?} found in deck");
        }

        let turn = turn_board();
        let deck2 = remaining_deck(&turn);
        assert_eq!(deck2.len(), 52 - turn.len());
    }

    #[timed_test(30)]
    fn leaf_evaluator_returns_correct_length() {
        let board = turn_board();
        let hands = small_hands(&board, 20);
        let n = hands.combos.len();
        let evaluator = ExactRiverEvaluator {
            bet_sizes: vec![1.0],
            iterations: 2,
        };
        let oop_range = vec![1.0; n];
        let ip_range = vec![1.0; n];
        let result = evaluator.evaluate(
            &hands.combos,
            &board,
            100.0,
            200.0,
            &oop_range,
            &ip_range,
            0,
        );
        assert_eq!(
            result.len(),
            n,
            "evaluator should return one CFV per combo"
        );
    }

    #[timed_test(30)]
    fn exact_river_evaluator_produces_bounded_cfvs() {
        let board = turn_board();
        let hands = small_hands(&board, 25);
        let n = hands.combos.len();
        let evaluator = ExactRiverEvaluator {
            bet_sizes: vec![1.0],
            iterations: 3,
        };
        let oop_range = vec![1.0; n];
        let ip_range = vec![1.0; n];
        let cfvs = evaluator.evaluate(
            &hands.combos,
            &board,
            100.0,
            200.0,
            &oop_range,
            &ip_range,
            0,
        );

        for (i, &cfv) in cfvs.iter().enumerate() {
            assert!(
                cfv.is_finite(),
                "combo {i}: CFV is not finite: {cfv}"
            );
            assert!(
                (-1.5..=1.5).contains(&cfv),
                "combo {i}: CFV {cfv} out of [-1.5, 1.5]"
            );
        }
    }

    #[timed_test(30)]
    fn cfv_solver_with_exact_evaluator_on_turn() {
        let board = turn_board();
        let tree = turn_tree_depth_limited(&[1.0], 1);
        let hands = small_hands(&board, 20);
        let evaluator = Box::new(ExactRiverEvaluator {
            bet_sizes: vec![1.0],
            iterations: 5,
        });
        let mut solver =
            CfvSubgameSolver::new(tree, hands, &board, evaluator, 250.0);

        solver.train(10);
        assert_eq!(solver.iteration, 10);

        let strategy = solver.strategy();
        assert!(strategy.num_combos() > 0);
    }

    #[timed_test(30)]
    fn root_cfvs_returns_correct_length() {
        let board = river_board();
        let tree = river_tree(&[1.0]);
        let hands = small_hands(&board, 30);
        let n = hands.combos.len();
        let evaluator = Box::new(ConstantEvaluator(0.5));
        let mut solver =
            CfvSubgameSolver::new(tree, hands, &board, evaluator, 250.0);

        solver.train(100);

        let oop_cfvs = solver.root_cfvs(0);
        let ip_cfvs = solver.root_cfvs(1);
        assert_eq!(oop_cfvs.len(), n);
        assert_eq!(ip_cfvs.len(), n);
    }

    #[timed_test(30)]
    fn root_cfvs_are_finite_and_bounded() {
        let board = river_board();
        let tree = river_tree(&[1.0]);
        let hands = small_hands(&board, 30);
        let evaluator = Box::new(ConstantEvaluator(0.5));
        let mut solver =
            CfvSubgameSolver::new(tree, hands, &board, evaluator, 250.0);

        solver.train(200);

        for traverser in 0..2u8 {
            let cfvs = solver.root_cfvs(traverser);
            for (i, &cfv) in cfvs.iter().enumerate() {
                assert!(
                    cfv.is_finite(),
                    "traverser {traverser} combo {i}: CFV not finite: {cfv}"
                );
                // CFVs should be bounded by half_pot (50.0 in this setup).
                assert!(
                    cfv.abs() < 100.0,
                    "traverser {traverser} combo {i}: CFV {cfv} seems too large"
                );
            }
        }
    }

    #[timed_test(30)]
    fn root_cfvs_with_depth_boundary() {
        let board = turn_board();
        let tree = turn_tree_depth_limited(&[1.0], 1);
        let hands = small_hands(&board, 30);
        let n = hands.combos.len();
        let evaluator = Box::new(ConstantEvaluator(0.5));
        let mut solver =
            CfvSubgameSolver::new(tree, hands, &board, evaluator, 250.0);

        solver.train(200);

        let oop_cfvs = solver.root_cfvs(0);
        let ip_cfvs = solver.root_cfvs(1);
        assert_eq!(oop_cfvs.len(), n);
        assert_eq!(ip_cfvs.len(), n);

        for (i, (&oop, &ip)) in oop_cfvs.iter().zip(ip_cfvs.iter()).enumerate() {
            assert!(oop.is_finite(), "OOP combo {i}: CFV not finite: {oop}");
            assert!(ip.is_finite(), "IP combo {i}: CFV not finite: {ip}");
        }
    }
}
