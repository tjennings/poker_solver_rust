//! Libratus-style safe re-solving gadget.
//!
//! Clamps opponent per-hand CFV upward to a pre-computed opt-out floor
//! (typically blueprint CBVs). Makes subgame boundary evaluators "safe"
//! in the sense that the reported opponent CFV is never worse than the
//! blueprint would guarantee. See `docs/plans/2026-04-23-deepstack-gadget.md`
//! for the full design and the distinction from DeepStack-proper
//! (which requires a cfvnet retrain; bean poker_solver_rust-akg3).

use poker_solver_core::blueprint_v2::Street;
use std::sync::Arc;

/// Per-hand opt-out value provider at a boundary.
///
/// Opt-out values are in the SAME units as the boundary evaluator's
/// `compute_cfvs` output (pot-normalised bcfv: 1.0 = one half-pot).
pub trait OptOutProvider: Send + Sync {
    /// Returns per-hand opt-out CFVs for the OPPONENT at this boundary.
    ///
    /// Vec length must equal `opponent_private_cards.len()`.
    fn opt_out_cfvs(
        &self,
        boundary_ordinal: usize,
        opponent: usize,
        pot: i32,
        effective_stack: i32,
        board: &[u8],
        opponent_private_cards: &[(u8, u8)],
    ) -> Vec<f32>;
}

/// Convert a per-hand CBV in raw chip-pot units to pot-normalised bcfv.
///
/// `CbvTable` stores values where `pot` chips = "won the full pot"
/// (see `cbv_compute.rs` `Fold`/`Showdown` terminal math). bcfv convention:
/// `+1.0` = "won one half-pot beyond start", `-1.0` = "lost one half-pot".
/// The conversion subtracts the break-even baseline before normalising,
/// matching the cfvnet training target in `cfvnet/datagen/domain/pipeline.rs:371`.
pub fn chip_cfv_to_bcfv(chip_cfv: f32, half_pot_chips: f32) -> f32 {
    assert!(half_pot_chips > 0.0, "half_pot must be positive");
    (chip_cfv - half_pot_chips) / half_pot_chips
}

/// Compute per-hand opt-out values for a Chance boundary node.
///
/// Looks up the CBV ordinal for the Chance node, retrieves per-bucket
/// values from the table, and converts to bcfv units using the
/// boundary's own half-pot.
fn compute_chance_boundary_optout(
    cbv_context: &crate::postflop::CbvContext,
    ordinal_map: &std::collections::HashMap<u32, usize>,
    chance_arena_idx: u32,
    street: poker_solver_core::blueprint_v2::Street,
    rs_board: &[poker_solver_core::poker::Card],
    private_cards: &[Vec<(u8, u8)>; 2],
) -> [Vec<f32>; 2] {
    use poker_solver_core::blueprint_v2::cbv::CbvTable;

    let cbv_ordinal =
        CbvTable::require_ordinal(ordinal_map, chance_arena_idx);
    let pot_at_chance =
        cbv_context.abstract_tree.pot_at_node(chance_arena_idx);
    let half_pot = (pot_at_chance / 2.0) as f32;
    assert!(
        half_pot > 0.0,
        "chance node {chance_arena_idx} has non-positive pot \
         ({pot_at_chance})"
    );

    let mut per_hand: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
    for player in 0..2 {
        let hands = &private_cards[player];
        per_hand[player].reserve(hands.len());
        for &(c1, c2) in hands {
            let rs_c1 = crate::exploration::range_solver_to_rs_card(c1);
            let rs_c2 = crate::exploration::range_solver_to_rs_card(c2);
            let bucket = cbv_context.all_buckets.get_bucket(
                street, [rs_c1, rs_c2], rs_board,
            );
            let chip_cbv = cbv_context.cbv_table.lookup(
                cbv_ordinal, bucket as usize,
            );
            per_hand[player].push(chip_cfv_to_bcfv(chip_cbv, half_pot));
        }
    }
    per_hand
}

/// Constant opt-out provider for testing.
///
/// Returns the same CFV for every hand at every boundary.
pub struct ConstantOptOut(pub f32);

impl OptOutProvider for ConstantOptOut {
    fn opt_out_cfvs(
        &self,
        _boundary_ordinal: usize,
        _opponent: usize,
        _pot: i32,
        _effective_stack: i32,
        _board: &[u8],
        opponent_private_cards: &[(u8, u8)],
    ) -> Vec<f32> {
        vec![self.0; opponent_private_cards.len()]
    }
}

/// Opt-out provider that pulls per-hand CFVs from a blueprint's `CbvTable`.
///
/// Stores pre-computed opt-out values for every boundary (chance node
/// descendant) and both players. At runtime, `opt_out_cfvs` selects the
/// correct boundary by ordinal and returns a pure vec clone.
pub struct BlueprintCbvOptOut {
    /// Per-boundary, per-player, per-hand pre-computed bcfv opt-out values.
    /// Index: `per_boundary_cbv[boundary_ordinal][player][hand_idx]`.
    per_boundary_cbv: Vec<[Vec<f32>; 2]>,
}

impl BlueprintCbvOptOut {
    /// Test-only constructor that builds zero-valued opt-out vectors
    /// for a single boundary.
    /// Production callers use `BlueprintCbvOptOut::from_cbv_context`.
    #[cfg(test)]
    pub(crate) fn new_for_test(
        cbv_table: Arc<poker_solver_core::blueprint_v2::cbv::CbvTable>,
        num_oop: usize,
        num_ip: usize,
    ) -> Self {
        assert!(
            !cbv_table.values.is_empty(),
            "CbvTable has no values; cannot construct BlueprintCbvOptOut"
        );
        Self {
            per_boundary_cbv: vec![[
                vec![0.0; num_oop],
                vec![0.0; num_ip],
            ]],
        }
    }

    /// Production constructor. Finds all chance node descendants of
    /// `abstract_root` in the abstract tree (DFS order), maps each to
    /// its dense CBV ordinal, and pre-computes per-hand opt-out values
    /// in bcfv units for both players at every boundary.
    ///
    /// Each boundary's CBV chip values are normalised by that boundary's
    /// own half-pot (derived from `GameTree::pot_at_node`), not a single
    /// global half-pot. This is critical when boundaries sit at different
    /// points in the action tree with different accumulated pots.
    ///
    /// `abstract_root` is the abstract tree arena index of the decision
    /// node where the subgame starts (e.g., the turn decision node).
    ///
    /// # Panics
    ///
    /// - If `cbv_context.cbv_table.values` is empty.
    /// - If `board.len()` is not 3, 4, or 5.
    /// - If no chance nodes are found below `abstract_root`.
    /// - If any chance node has a non-positive pot.
    pub fn from_cbv_context(
        cbv_context: &crate::postflop::CbvContext,
        abstract_root: u32,
        board: &[u8],
        private_cards: &[Vec<(u8, u8)>; 2],
    ) -> Self {
        use poker_solver_core::blueprint_v2::cbv::CbvTable;
        use poker_solver_core::blueprint_v2::game_tree::GameNode;

        assert!(
            !cbv_context.cbv_table.values.is_empty(),
            "CbvTable has no values; cannot construct BlueprintCbvOptOut"
        );

        let ordinal_map =
            CbvTable::build_node_to_ordinal_map(&cbv_context.abstract_tree);
        // Use boundary_descendants (not chance_descendants) to include
        // both Chance nodes AND all-in Showdown terminals. The concrete
        // range-solver tree creates depth boundaries for both.
        let boundary_nodes =
            cbv_context.abstract_tree.boundary_descendants(abstract_root);

        assert!(
            !boundary_nodes.is_empty(),
            "no boundary nodes found below abstract tree node \
             {abstract_root}; cannot construct BlueprintCbvOptOut"
        );

        let street = match board.len() {
            3 => Street::Flop,
            4 => Street::Turn,
            5 => Street::River,
            n => panic!("unexpected board length {n}; expected 3, 4, or 5"),
        };

        let rs_board: Vec<poker_solver_core::poker::Card> = board
            .iter()
            .map(|&id| crate::exploration::range_solver_to_rs_card(id))
            .collect();

        let mut per_boundary = Vec::with_capacity(boundary_nodes.len());
        for &arena_idx in &boundary_nodes {
            let node = &cbv_context.abstract_tree.nodes[arena_idx as usize];
            match node {
                GameNode::Chance { .. } => {
                    // Normal street transition: look up CBV opt-out.
                    let per_hand = self::compute_chance_boundary_optout(
                        cbv_context, &ordinal_map, arena_idx,
                        street, &rs_board, private_cards,
                    );
                    per_boundary.push(per_hand);
                }
                GameNode::Terminal { .. } => {
                    // All-in Showdown on the starting street. Both
                    // players already committed all chips, so the
                    // opponent cannot opt out. Use -1e9 (always
                    // dominated) instead of NEG_INFINITY because
                    // NEG_INFINITY * 0 = NaN in solver arithmetic
                    // (bcfvs[h] * payoff_scale * cfreach_adj), which
                    // propagates through compute_mes_ev ->
                    // compute_exploitability, breaking the BR pass.
                    let n_oop = private_cards[0].len();
                    let n_ip = private_cards[1].len();
                    per_boundary.push([
                        vec![-1e9_f32; n_oop],
                        vec![-1e9_f32; n_ip],
                    ]);
                }
                GameNode::Decision { .. } => {
                    panic!(
                        "boundary_descendants returned Decision node \
                         at arena index {arena_idx}; this is a bug"
                    );
                }
            }
        }
        Self { per_boundary_cbv: per_boundary }
    }

    /// Number of boundaries this provider was constructed with.
    #[must_use]
    pub fn num_boundaries(&self) -> usize {
        self.per_boundary_cbv.len()
    }

    /// Returns the per-boundary, per-player opt-out values.
    ///
    /// Index: `[boundary_ordinal][player][hand_idx]`.
    /// Used by `make_per_boundary_gadget_game` to build
    /// `GadgetConfigPerBoundary`.
    #[must_use]
    pub fn per_boundary_opt_outs(&self) -> &Vec<[Vec<f32>; 2]> {
        &self.per_boundary_cbv
    }
}

impl OptOutProvider for BlueprintCbvOptOut {
    fn opt_out_cfvs(
        &self,
        boundary_ordinal: usize,
        opponent: usize,
        _pot: i32,
        _effective_stack: i32,
        _board: &[u8],
        opponent_private_cards: &[(u8, u8)],
    ) -> Vec<f32> {
        let entry = &self.per_boundary_cbv[boundary_ordinal
            .min(self.per_boundary_cbv.len() - 1)];
        assert_eq!(
            opponent_private_cards.len(),
            entry[opponent].len(),
            "opt_out_cfvs called with hand list of length {} but \
             constructor registered {} hands for player {opponent}",
            opponent_private_cards.len(),
            entry[opponent].len(),
        );
        entry[opponent].clone()
    }
}

/// Boundary evaluator wrapper that applies the Libratus range gadget.
///
/// Delegates to an inner `BoundaryEvaluator`, then clamps each opponent
/// hand's CFV upward to the opt-out value. This ensures the opponent
/// never does worse than their blueprint counterfactual best-response.
pub struct GadgetEvaluator {
    inner: Arc<dyn range_solver::game::BoundaryEvaluator>,
    opt_out: Arc<dyn OptOutProvider>,
    /// Which boundary this evaluator serves (passed to `opt_out_cfvs`).
    boundary_ordinal: usize,
    board: Vec<u8>,
    private_cards: [Vec<(u8, u8)>; 2],
}

impl GadgetEvaluator {
    pub fn new(
        inner: Arc<dyn range_solver::game::BoundaryEvaluator>,
        opt_out: Arc<dyn OptOutProvider>,
        boundary_ordinal: usize,
        board: Vec<u8>,
        private_cards: [Vec<(u8, u8)>; 2],
    ) -> Self {
        Self { inner, opt_out, boundary_ordinal, board, private_cards }
    }
}

/// Stats from a single clamp pass: how many hands were pushed up by the
/// opt-out floor, and by how much. Diagnostic-only.
#[derive(Debug, Default, Clone, Copy)]
pub struct ClampStats {
    pub hands_clamped: usize,
    pub hands_total: usize,
    pub max_delta: f32,
    pub mean_delta: f32,
}

/// Apply opt-out clamping: for each opponent hand, clamp CFV upward
/// to the opt-out value. Returns the adjusted (player_cfvs, opp_cfvs)
/// plus stats on how many hands were actually clamped.
fn apply_gadget_clamp(
    player_cfvs: &[f32],
    opp_cfvs: &[f32],
    opt_out_cfvs: &[f32],
) -> (Vec<f32>, Vec<f32>, ClampStats) {
    let mut hands_clamped = 0usize;
    let mut max_delta = 0.0f32;
    let mut total_delta = 0.0f64;

    let clamped_opp: Vec<f32> = opp_cfvs
        .iter()
        .zip(opt_out_cfvs.iter())
        .map(|(&inner, &opt)| {
            if opt > inner {
                hands_clamped += 1;
                let d = opt - inner;
                total_delta += d as f64;
                if d > max_delta {
                    max_delta = d;
                }
                opt
            } else {
                inner
            }
        })
        .collect();

    let mean_delta = if hands_clamped > 0 {
        (total_delta / hands_clamped as f64) as f32
    } else {
        0.0
    };
    let stats = ClampStats {
        hands_clamped,
        hands_total: opp_cfvs.len(),
        max_delta,
        mean_delta,
    };

    // Player CFVs stay the same -- gadget only constrains opponent.
    (player_cfvs.to_vec(), clamped_opp, stats)
}

// ---------------------------------------------------------------------------
// Diagnostic logging
// ---------------------------------------------------------------------------

/// Summary stats over a per-hand CFV vector (for diagnostic logging).
fn summary(cfvs: &[f32]) -> (f32, f32, f32) {
    if cfvs.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let (mut mn, mut mx, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
    for &v in cfvs {
        if v < mn { mn = v; }
        if v > mx { mx = v; }
        sum += v as f64;
    }
    (mn, (sum / cfvs.len() as f64) as f32, mx)
}

/// Controls gadget diagnostic logging. `stride` = log every Nth call per
/// boundary (stride=1 → every call; stride=0 → disabled). Per-boundary
/// atomic counters produce the call index shown in the log.
static GADGET_DIAG_STRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(20);
static GADGET_DIAG_COUNTS: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<usize, u64>>,
> = std::sync::OnceLock::new();

/// Returns `Some(call_idx)` if this call should be logged, else `None`.
/// `call_idx` is the 1-based count of compute_cfvs_both calls for this
/// boundary_ordinal.
fn diag_fire(boundary_ordinal: usize) -> Option<u64> {
    let stride = GADGET_DIAG_STRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if stride == 0 {
        return None;
    }
    let counts = GADGET_DIAG_COUNTS
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut guard = counts.lock().expect("diag counts poisoned");
    let entry = guard.entry(boundary_ordinal).or_insert(0);
    *entry += 1;
    let idx = *entry;
    // Always log the first call + every strideth thereafter.
    if idx == 1 || idx % stride as u64 == 0 {
        Some(idx)
    } else {
        None
    }
}

/// Set the diagnostic log stride. `0` disables. `1` logs every call.
/// Default is `20`.
pub fn set_gadget_diag_stride(stride: usize) {
    GADGET_DIAG_STRIDE.store(stride, std::sync::atomic::Ordering::Relaxed);
}

/// Disable gadget diagnostic prints entirely. Kept for test compatibility.
pub fn set_gadget_diag_enabled(enabled: bool) {
    set_gadget_diag_stride(if enabled { 20 } else { 0 });
}

impl range_solver::game::BoundaryEvaluator for GadgetEvaluator {
    fn compute_cfvs(
        &self,
        player: usize,
        pot: i32,
        remaining_stack: f64,
        opponent_reach: &[f32],
        num_hands: usize,
        continuation_index: usize,
    ) -> Vec<f32> {
        let opp = player ^ 1;
        let opp_num = self.private_cards[opp].len();
        let player_reach = vec![1.0f32; num_hands];
        let (oop_reach, ip_reach) = if player == 0 {
            (player_reach.as_slice(), opponent_reach)
        } else {
            (opponent_reach, player_reach.as_slice())
        };
        let (num_oop, num_ip) = if player == 0 {
            (num_hands, opp_num)
        } else {
            (opp_num, num_hands)
        };
        let (oop_cfvs, ip_cfvs) = self.compute_cfvs_both(
            pot, remaining_stack, oop_reach, ip_reach,
            num_oop, num_ip, continuation_index,
        );
        if player == 0 { oop_cfvs } else { ip_cfvs }
    }

    fn compute_cfvs_both(
        &self,
        pot: i32,
        remaining_stack: f64,
        oop_reach: &[f32],
        ip_reach: &[f32],
        num_oop: usize,
        num_ip: usize,
        continuation_index: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let (oop_inner, ip_inner) = self.inner.compute_cfvs_both(
            pot, remaining_stack, oop_reach, ip_reach,
            num_oop, num_ip, continuation_index,
        );
        let eff_stack = (pot / 2) + remaining_stack.round() as i32;
        // Get opt-out values for each player (as opponent)
        let oop_opt_out = self.opt_out.opt_out_cfvs(
            self.boundary_ordinal, 0, pot, eff_stack,
            &self.board, &self.private_cards[0],
        );
        let ip_opt_out = self.opt_out.opt_out_cfvs(
            self.boundary_ordinal, 1, pot, eff_stack,
            &self.board, &self.private_cards[1],
        );
        // When computing OOP cfvs, IP is opponent => clamp IP upward
        let (_oop_adj_for_oop, ip_clamped, ip_clamp_stats) = apply_gadget_clamp(
            &oop_inner, &ip_inner, &ip_opt_out,
        );
        // When computing IP cfvs, OOP is opponent => clamp OOP upward
        let (_ip_adj_for_ip, oop_clamped, oop_clamp_stats) = apply_gadget_clamp(
            &ip_inner, &oop_inner, &oop_opt_out,
        );

        // Per-N-calls diagnostic: prove ranges and CFVs are evolving across
        // iters. Logs first call + every `stride`-th thereafter per boundary.
        if let Some(call_idx) = diag_fire(self.boundary_ordinal) {
            let b = self.boundary_ordinal;
            // Reach summaries — shows the parent DCFR's evolving ranges.
            let (oor_min, oor_mean, oor_max) = summary(oop_reach);
            let oor_nonzero = oop_reach.iter().filter(|&&w| w > 1e-6).count();
            let (ipr_min, ipr_mean, ipr_max) = summary(ip_reach);
            let ipr_nonzero = ip_reach.iter().filter(|&&w| w > 1e-6).count();
            // CFV summaries.
            let (oi_min, oi_mean, oi_max) = summary(&oop_inner);
            let (ii_min, ii_mean, ii_max) = summary(&ip_inner);
            eprintln!(
                "[gadget] b={b} call={call_idx} pot={pot} \
                 reach_OOP(nz={oor_nonzero}, min/mean/max={oor_min:+.3}/{oor_mean:+.3}/{oor_max:+.3}) \
                 reach_IP(nz={ipr_nonzero}, min/mean/max={ipr_min:+.3}/{ipr_mean:+.3}/{ipr_max:+.3})\n  \
                 inner_OOP(min/mean/max={oi_min:+.3}/{oi_mean:+.3}/{oi_max:+.3})  \
                 clamped_IP={ip_cn}/{ip_ct} max_Δ={ip_mx:+.3}\n  \
                 inner_IP (min/mean/max={ii_min:+.3}/{ii_mean:+.3}/{ii_max:+.3})  \
                 clamped_OOP={op_cn}/{op_ct} max_Δ={op_mx:+.3}",
                ip_cn = ip_clamp_stats.hands_clamped,
                ip_ct = ip_clamp_stats.hands_total,
                ip_mx = ip_clamp_stats.max_delta,
                op_cn = oop_clamp_stats.hands_clamped,
                op_ct = oop_clamp_stats.hands_total,
                op_mx = oop_clamp_stats.max_delta,
            );
        }

        (oop_clamped, ip_clamped)
    }
}

/// Compose a per-boundary-gadget-enabled `PostFlopGame`.
///
/// Builds the game via `with_config`, then invokes `inject_per_boundary_gadgets`
/// with per-boundary opt-outs computed from blueprint CBV (existing
/// `BlueprintCbvOptOut::from_cbv_context`). Allocates memory. Returns
/// the solve-ready game.
///
/// Wires boundary evaluator: the caller's inner evaluator (cfvnet or
/// exact-subtree) handles the ORIGINAL N cfvnet boundaries (ordinals
/// remain 0..N post-injection per Phase A's guarantee). The 2N new
/// gadget Terminate ordinals (N..3N) are already pre-populated in
/// `boundary_cfvs` by `inject_per_boundary_gadgets` with the per-boundary
/// opt-out values; no per-boundary evaluator needed for them.
pub fn make_per_boundary_gadget_game(
    card_config: range_solver::card::CardConfig,
    action_tree: range_solver::action_tree::ActionTree,
    cbv_context: &Arc<crate::postflop::CbvContext>,
    abstract_root: u32,
    board: &[u8],
    inner_evaluator: Arc<dyn range_solver::game::BoundaryEvaluator>,
) -> Result<range_solver::game::PostFlopGame, String> {
    use range_solver::interface::Game;

    let mut game = range_solver::game::PostFlopGame::with_config(card_config, action_tree)
        .map_err(|e| format!("Failed to build game: {e}"))?;

    let private_cards: [Vec<(u8, u8)>; 2] = [
        game.private_cards(0).to_vec(),
        game.private_cards(1).to_vec(),
    ];

    let provider = BlueprintCbvOptOut::from_cbv_context(
        cbv_context, abstract_root, board, &private_cards,
    );

    let concrete_boundaries = game.num_boundary_nodes();
    let abstract_boundaries = provider.num_boundaries();

    let per_boundary_opt_outs = if concrete_boundaries == abstract_boundaries {
        provider.per_boundary_opt_outs().clone()
    } else {
        eprintln!(
            "[gadget] boundary count mismatch: abstract={abstract_boundaries}, \
             concrete={concrete_boundaries}; this may indicate a bet-size \
             config difference between blueprint and subgame. \
             Broadcasting averaged opt-outs to all concrete boundaries."
        );
        broadcast_opt_outs(provider.per_boundary_opt_outs(), concrete_boundaries)
    };

    let config = range_solver::game::gadget::GadgetConfigPerBoundary {
        per_boundary_opt_outs,
    };

    range_solver::game::gadget::inject_per_boundary_gadgets(&mut game, &config);
    game.allocate_memory(false);
    game.boundary_evaluator = Some(inner_evaluator);

    Ok(game)
}

/// Average abstract per-boundary opt-outs into a single per-hand vector,
/// then replicate to `target_count` concrete boundaries.
///
/// Used when the concrete action tree has a different number of boundaries
/// than the abstract tree (e.g., wider bet sizes in the explorer config).
/// Averaging across abstract boundaries preserves the overall opt-out level
/// while avoiding the panic from a length mismatch.
fn broadcast_opt_outs(
    abstract_opt_outs: &[[Vec<f32>; 2]],
    target_count: usize,
) -> Vec<[Vec<f32>; 2]> {
    assert!(
        !abstract_opt_outs.is_empty(),
        "broadcast_opt_outs: abstract_opt_outs must not be empty"
    );
    let n_abs = abstract_opt_outs.len();
    let n_oop = abstract_opt_outs[0][0].len();
    let n_ip = abstract_opt_outs[0][1].len();

    let mut avg: [Vec<f64>; 2] = [vec![0.0; n_oop], vec![0.0; n_ip]];
    for entry in abstract_opt_outs {
        for (h, &v) in entry[0].iter().enumerate() {
            avg[0][h] += v as f64;
        }
        for (h, &v) in entry[1].iter().enumerate() {
            avg[1][h] += v as f64;
        }
    }
    let inv = 1.0 / n_abs as f64;
    let oop_avg: Vec<f32> = avg[0].iter().map(|&s| (s * inv) as f32).collect();
    let ip_avg: Vec<f32> = avg[1].iter().map(|&s| (s * inv) as f32).collect();

    (0..target_count)
        .map(|_| [oop_avg.clone(), ip_avg.clone()])
        .collect()
}

/// Compose a per-boundary-gadget-enabled `PostFlopGame` with constant
/// opt-out values (for `--gadget-provider constant`).
///
/// Same as `make_per_boundary_gadget_game` but uses a uniform constant
/// opt-out value for every hand at every boundary, instead of blueprint CBVs.
pub fn make_per_boundary_gadget_game_constant(
    card_config: range_solver::card::CardConfig,
    action_tree: range_solver::action_tree::ActionTree,
    constant_opt_out: f32,
    inner_evaluator: Arc<dyn range_solver::game::BoundaryEvaluator>,
) -> Result<range_solver::game::PostFlopGame, String> {
    use range_solver::interface::Game;

    let mut game = range_solver::game::PostFlopGame::with_config(card_config, action_tree)
        .map_err(|e| format!("Failed to build game: {e}"))?;

    let n_boundaries = game.num_boundary_nodes();
    let n_oop = game.num_private_hands(0);
    let n_ip = game.num_private_hands(1);

    let per_boundary_opt_outs = (0..n_boundaries)
        .map(|_| [vec![constant_opt_out; n_oop], vec![constant_opt_out; n_ip]])
        .collect();

    let config = range_solver::game::gadget::GadgetConfigPerBoundary {
        per_boundary_opt_outs,
    };

    range_solver::game::gadget::inject_per_boundary_gadgets(&mut game, &config);
    game.allocate_memory(false);
    game.boundary_evaluator = Some(inner_evaluator);

    Ok(game)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use range_solver::game::BoundaryEvaluator;

    // ---------------------------------------------------------------
    // chip_cfv_to_bcfv tests
    // ---------------------------------------------------------------

    #[test]
    fn chip_cfv_to_bcfv_converts_correctly() {
        // CbvTable stores raw chip-pot values where `pot` = "won the full pot".
        // half_pot = 73 chips (i.e. full pot = 146 chips).
        // Break-even = half_pot = 73 chips.
        //
        // chip_cfv of 146 = won full pot => bcfv = (146-73)/73 = +1.0
        assert!((chip_cfv_to_bcfv(146.0, 73.0) - 1.0).abs() < 1e-6);
        // chip_cfv of 73 = break-even => bcfv = (73-73)/73 = 0.0
        assert!((chip_cfv_to_bcfv(73.0, 73.0) - 0.0).abs() < 1e-6);
        // chip_cfv of 0 = lost everything => bcfv = (0-73)/73 = -1.0
        assert!((chip_cfv_to_bcfv(0.0, 73.0) - (-1.0)).abs() < 1e-6);
        // chip_cfv of 109.5 = won half-pot above break-even => bcfv = (109.5-73)/73 = +0.5
        assert!((chip_cfv_to_bcfv(109.5, 73.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn chip_cfv_to_bcfv_matches_cfvnet_training_target() {
        // cfvnet target: (ev_chips - half_pot) / half_pot (pipeline.rs:371)
        let half_pot = 73.0_f32;
        for ev_chips in [0.0, 36.5, 73.0, 109.5, 146.0] {
            let target = (ev_chips - half_pot) / half_pot;
            assert_eq!(chip_cfv_to_bcfv(ev_chips, half_pot), target);
        }
    }

    #[test]
    #[should_panic(expected = "half_pot must be positive")]
    fn chip_cfv_to_bcfv_zero_half_pot_panics() {
        chip_cfv_to_bcfv(10.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "half_pot must be positive")]
    fn chip_cfv_to_bcfv_negative_half_pot_panics() {
        chip_cfv_to_bcfv(10.0, -5.0);
    }

    // ---------------------------------------------------------------
    // BlueprintCbvOptOut tests
    // ---------------------------------------------------------------

    #[test]
    #[should_panic(expected = "CbvTable has no values")]
    fn blueprint_cbv_construct_panics_on_empty_table() {
        use poker_solver_core::blueprint_v2::cbv::CbvTable;
        let empty_table = CbvTable {
            values: vec![],
            node_offsets: vec![],
            buckets_per_node: vec![],
        };
        let _ = BlueprintCbvOptOut::new_for_test(
            Arc::new(empty_table),
            1,
            1,
        );
    }

    /// Build a minimal `CbvContext` + board + private_cards fixture for gadget tests.
    /// Uses a 2-bucket CbvTable with known values and equity-fallback bucketing.
    ///
    /// Returns `(context, abstract_root, board, private_cards)`.
    /// `abstract_root` is the decision node where the subgame starts.
    fn make_cbv_test_context() -> (crate::postflop::CbvContext, u32, Vec<u8>, [Vec<(u8, u8)>; 2]) {
        use crate::postflop::CbvContext;
        use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
        use poker_solver_core::blueprint_v2::cbv::CbvTable;
        use poker_solver_core::blueprint_v2::game_tree::{
            GameNode, GameTree, TerminalKind, TreeAction,
        };
        use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
        use poker_solver_core::blueprint_v2::Street;
        use range_solver::card::flop_from_str;

        // 1 boundary node, 2 buckets: bucket 0 -> 50.0 chips, bucket 1 -> -30.0 chips.
        let cbv_table = CbvTable {
            values: vec![50.0, -30.0],
            node_offsets: vec![0],
            buckets_per_node: vec![2],
        };

        // Hand-built tree with a single chance node at arena index 2
        // (not 0) so the ordinal mapping is exercised.
        //   0: Decision(P0, Turn, [Check])
        //   1: Decision(P1, Turn, [Check])
        //   2: Chance(next=River, child=3) -- ordinal 0
        //   3: Terminal(Showdown, pot=100)
        let nodes = vec![
            GameNode::Decision {
                player: 0,
                street: Street::Turn,
                actions: vec![TreeAction::Check],
                children: vec![1],
                blueprint_decision_idx: None,
            },
            GameNode::Decision {
                player: 1,
                street: Street::Turn,
                actions: vec![TreeAction::Check],
                children: vec![2],
                blueprint_decision_idx: None,
            },
            GameNode::Chance {
                next_street: Street::River,
                child: 3,
            },
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 100.0,
                stacks: [50.0, 50.0],
            },
        ];
        let tree = GameTree {
            nodes,
            root: 0,
            dealer: 0,
            starting_stack: 100.0,
        };
        // Root decision node (arena 0) -- callers pass this to from_cbv_context.
        // The chance node is at arena index 2 (ordinal 0).
        let abstract_root: u32 = 0;

        let mut ab = AllBuckets::new([2, 2, 2, 2], [None, None, None, None]);
        ab.equity_fallback = true;
        let all_buckets = Arc::new(ab);
        let strategy = Arc::new(BlueprintV2Strategy::empty());

        let ctx = CbvContext {
            cbv_table,
            abstract_tree: tree,
            all_buckets,
            strategy,
        };

        let flop = flop_from_str("7h 5d 2c").unwrap();
        let turn_card: u8 = 7;  // 3s
        let river_card: u8 = 30; // 9h
        let board = vec![flop[0], flop[1], flop[2], turn_card, river_card];

        let oop_hands = vec![(48u8, 49u8)]; // Ac, Ad
        let ip_hands = vec![(4u8, 5u8)];    // 3c, 3d
        let private_cards = [oop_hands, ip_hands];

        (ctx, abstract_root, board, private_cards)
    }

    #[test]
    fn blueprint_cbv_opt_out_from_cbv_context_returns_correct_bcfv() {
        let (ctx, root, board, private_cards) = make_cbv_test_context();

        let provider = BlueprintCbvOptOut::from_cbv_context(
            &ctx, root, &board, &private_cards,
        );

        // Verify opt_out_cfvs returns the correct number of hands
        let oop_cfvs = provider.opt_out_cfvs(0, 0, 100, 200, &board, &private_cards[0]);
        let ip_cfvs = provider.opt_out_cfvs(0, 1, 100, 200, &board, &private_cards[1]);
        assert_eq!(oop_cfvs.len(), 1);
        assert_eq!(ip_cfvs.len(), 1);

        // Verify the values are (chip_cbv - half_pot) / half_pot.
        // The chance node's child terminal has pot=100, so half_pot=50.
        // With equity fallback and 2 buckets, the exact bucket depends on
        // equity calculation, but the value must be either
        // (50.0-50.0)/50.0 = 0.0  or  (-30.0-50.0)/50.0 = -1.6.
        for &v in oop_cfvs.iter().chain(ip_cfvs.iter()) {
            assert!(
                (v - 0.0).abs() < 1e-6 || (v - (-1.6)).abs() < 1e-6,
                "bcfv value {v} should be 0.0 or -1.6"
            );
        }
    }

    /// Build a multi-node CbvContext with a connected tree rooted at node 0.
    /// The tree has 3 action paths leading to 3 different chance nodes at
    /// sparse arena indices, with distinct CBV values at each.
    ///
    /// Tree structure (root=0):
    ///   0: Decision(P0, Turn, [Check, Bet, AllIn])
    ///     1: Decision(P1, Turn, [Check])
    ///       2: Chance(River, child=3) -- ordinal 0, CBV=[10, -5]
    ///         3: Terminal(Showdown)
    ///     4: Decision(P1, Turn, [Fold, Call])
    ///       5: Terminal(Fold)
    ///       6: Chance(River, child=7) -- ordinal 1, CBV=[30, -15]
    ///         7: Terminal(Showdown)
    ///     8: Decision(P1, Turn, [Fold, Call])
    ///       9: Terminal(Fold)
    ///       10: Chance(River, child=11) -- ordinal 2, CBV=[60, -30]
    ///         11: Terminal(Showdown)
    fn make_multi_node_cbv_context() -> (crate::postflop::CbvContext, Vec<u8>, [Vec<(u8, u8)>; 2]) {
        use crate::postflop::CbvContext;
        use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
        use poker_solver_core::blueprint_v2::cbv::CbvTable;
        use poker_solver_core::blueprint_v2::game_tree::{
            GameNode, GameTree, TerminalKind, TreeAction,
        };
        use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
        use poker_solver_core::blueprint_v2::Street;
        use range_solver::card::flop_from_str;

        let nodes = vec![
            // 0: Root decision (3 actions)
            GameNode::Decision {
                player: 0,
                street: Street::Turn,
                actions: vec![TreeAction::Check, TreeAction::Bet(2.0), TreeAction::AllIn],
                children: vec![1, 4, 8],
                blueprint_decision_idx: None,
            },
            // 1: P1 after check
            GameNode::Decision {
                player: 1,
                street: Street::Turn,
                actions: vec![TreeAction::Check],
                children: vec![2],
                blueprint_decision_idx: None,
            },
            // 2: Chance (check-check) -- ordinal 0
            GameNode::Chance { next_street: Street::River, child: 3 },
            // 3: Terminal
            GameNode::Terminal {
                kind: TerminalKind::Showdown, pot: 10.0, stacks: [50.0, 50.0],
            },
            // 4: P1 after bet
            GameNode::Decision {
                player: 1,
                street: Street::Turn,
                actions: vec![TreeAction::Fold, TreeAction::Call],
                children: vec![5, 6],
                blueprint_decision_idx: None,
            },
            // 5: Fold terminal
            GameNode::Terminal {
                kind: TerminalKind::Fold { winner: 0 }, pot: 10.0, stacks: [50.0, 50.0],
            },
            // 6: Chance (bet-call) -- ordinal 1
            GameNode::Chance { next_street: Street::River, child: 7 },
            // 7: Terminal
            GameNode::Terminal {
                kind: TerminalKind::Showdown, pot: 30.0, stacks: [45.0, 45.0],
            },
            // 8: P1 after all-in
            GameNode::Decision {
                player: 1,
                street: Street::Turn,
                actions: vec![TreeAction::Fold, TreeAction::Call],
                children: vec![9, 10],
                blueprint_decision_idx: None,
            },
            // 9: Fold terminal
            GameNode::Terminal {
                kind: TerminalKind::Fold { winner: 0 }, pot: 10.0, stacks: [50.0, 50.0],
            },
            // 10: Chance (allin-call) -- ordinal 2
            GameNode::Chance { next_street: Street::River, child: 11 },
            // 11: Terminal
            GameNode::Terminal {
                kind: TerminalKind::Showdown, pot: 60.0, stacks: [20.0, 20.0],
            },
        ];

        // 3 boundary nodes with distinct CBV values.
        // ordinal 0 (arena 2):  bucket 0 = 10.0, bucket 1 = -5.0
        // ordinal 1 (arena 6):  bucket 0 = 30.0, bucket 1 = -15.0
        // ordinal 2 (arena 10): bucket 0 = 60.0, bucket 1 = -30.0
        let cbv_table = CbvTable {
            values: vec![10.0, -5.0, 30.0, -15.0, 60.0, -30.0],
            node_offsets: vec![0, 2, 4],
            buckets_per_node: vec![2, 2, 2],
        };

        let mut ab = AllBuckets::new([2, 2, 2, 2], [None, None, None, None]);
        ab.equity_fallback = true;
        let all_buckets = Arc::new(ab);
        let strategy = Arc::new(BlueprintV2Strategy::empty());
        let tree = GameTree {
            nodes,
            root: 0,
            dealer: 0,
            starting_stack: 100.0,
        };

        let ctx = CbvContext {
            cbv_table,
            abstract_tree: tree,
            all_buckets,
            strategy,
        };

        let flop = flop_from_str("7h 5d 2c").unwrap();
        let board = vec![flop[0], flop[1], flop[2], 7u8, 30u8];

        let oop_hands = vec![(48u8, 49u8)]; // Ac, Ad
        let ip_hands = vec![(4u8, 5u8)];    // 3c, 3d
        let private_cards = [oop_hands, ip_hands];

        (ctx, board, private_cards)
    }

    #[test]
    fn from_cbv_context_multi_node_finds_all_boundaries() {
        let (ctx, board, private_cards) = make_multi_node_cbv_context();

        // Root node 0 has 3 chance descendants
        let provider = BlueprintCbvOptOut::from_cbv_context(
            &ctx, 0, &board, &private_cards,
        );

        assert_eq!(provider.num_boundaries(), 3);
    }

    #[test]
    fn from_cbv_context_multi_node_each_boundary_has_correct_count() {
        let (ctx, board, private_cards) = make_multi_node_cbv_context();

        let provider = BlueprintCbvOptOut::from_cbv_context(
            &ctx, 0, &board, &private_cards,
        );

        // Each boundary should have the correct number of hands per player.
        for b in 0..3 {
            let oop = provider.opt_out_cfvs(b, 0, 100, 200, &board, &private_cards[0]);
            let ip = provider.opt_out_cfvs(b, 1, 100, 200, &board, &private_cards[1]);
            assert_eq!(oop.len(), private_cards[0].len());
            assert_eq!(ip.len(), private_cards[1].len());
        }
    }

    #[test]
    fn from_cbv_context_uses_per_boundary_pot_for_normalization() {
        // The multi-node tree has 3 chance nodes with distinct pots:
        //   ordinal 0 (arena 2):  pot=10,  half_pot=5
        //   ordinal 1 (arena 6):  pot=30,  half_pot=15
        //   ordinal 2 (arena 10): pot=60,  half_pot=30
        //
        // With per-boundary pot, CBV bucket 0 values normalize as:
        //   ordinal 0: chip_cbv=10 => (10-5)/5 = +1.0
        //   ordinal 1: chip_cbv=30 => (30-15)/15 = +1.0
        //   ordinal 2: chip_cbv=60 => (60-30)/30 = +1.0
        //
        // With a WRONG single half_pot=50 (the old bug), we'd get:
        //   ordinal 0: (10-50)/50 = -0.8
        //   ordinal 2: (60-50)/50 = +0.2
        //
        // This test catches the bug: with the fix, all three boundaries
        // should produce the SAME bcfv for bucket 0 hands.
        let (ctx, board, private_cards) = make_multi_node_cbv_context();
        let provider = BlueprintCbvOptOut::from_cbv_context(
            &ctx, 0, &board, &private_cards,
        );

        let oop_0 = provider.opt_out_cfvs(0, 0, 100, 200, &board, &private_cards[0]);
        let oop_1 = provider.opt_out_cfvs(1, 0, 100, 200, &board, &private_cards[0]);
        let oop_2 = provider.opt_out_cfvs(2, 0, 100, 200, &board, &private_cards[0]);

        // All three should be equal (same bucket maps to proportionally
        // equal CBVs when normalised by that boundary's own pot).
        assert!(
            (oop_0[0] - oop_1[0]).abs() < 0.01,
            "boundary 0 ({}) and boundary 1 ({}) should have same bcfv \
             when each is normalised by its own half-pot",
            oop_0[0], oop_1[0],
        );
        assert!(
            (oop_1[0] - oop_2[0]).abs() < 0.01,
            "boundary 1 ({}) and boundary 2 ({}) should have same bcfv \
             when each is normalised by its own half-pot",
            oop_1[0], oop_2[0],
        );
    }

    #[test]
    #[should_panic(expected = "hand list of length")]
    fn blueprint_cbv_opt_out_provider_length_mismatch_panics() {
        let (ctx, root, board, private_cards) = make_cbv_test_context();

        let provider = BlueprintCbvOptOut::from_cbv_context(
            &ctx, root, &board, &private_cards,
        );

        // Call with wrong hand count -- should panic
        provider.opt_out_cfvs(0, 0, 100, 200, &board, &[(0u8, 1u8), (2, 3)]);
    }

    // ---------------------------------------------------------------
    // OptOutProvider tests
    // ---------------------------------------------------------------

    #[test]
    fn constant_opt_out_returns_uniform_values() {
        let provider = ConstantOptOut(0.5);
        let cards = vec![(0u8, 1u8), (2, 3), (4, 5)];
        let result = provider.opt_out_cfvs(0, 0, 100, 200, &[10, 20, 30], &cards);
        assert_eq!(result, vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn constant_opt_out_negative_value() {
        let provider = ConstantOptOut(-1000.0);
        let cards = vec![(0u8, 1u8)];
        let result = provider.opt_out_cfvs(0, 1, 50, 100, &[], &cards);
        assert_eq!(result, vec![-1000.0]);
    }

    #[test]
    fn constant_opt_out_empty_cards() {
        let provider = ConstantOptOut(42.0);
        let result = provider.opt_out_cfvs(0, 0, 100, 200, &[], &[]);
        assert!(result.is_empty());
    }

    // ---------------------------------------------------------------
    // GadgetEvaluator tests
    // ---------------------------------------------------------------

    /// Stub boundary evaluator that returns fixed bcfv values.
    struct StubEvaluator {
        oop_cfvs: Vec<f32>,
        ip_cfvs: Vec<f32>,
    }

    impl range_solver::game::BoundaryEvaluator for StubEvaluator {
        fn compute_cfvs(
            &self,
            player: usize,
            _pot: i32,
            _remaining_stack: f64,
            _opponent_reach: &[f32],
            _num_hands: usize,
            _continuation_index: usize,
        ) -> Vec<f32> {
            if player == 0 {
                self.oop_cfvs.clone()
            } else {
                self.ip_cfvs.clone()
            }
        }

        fn compute_cfvs_both(
            &self,
            _pot: i32,
            _remaining_stack: f64,
            _oop_reach: &[f32],
            _ip_reach: &[f32],
            _num_oop: usize,
            _num_ip: usize,
            _continuation_index: usize,
        ) -> (Vec<f32>, Vec<f32>) {
            (self.oop_cfvs.clone(), self.ip_cfvs.clone())
        }
    }

    #[test]
    fn gadget_huge_opt_out_clamps_opponent_via_both() {
        // Inner evaluator returns moderate values
        let inner = Arc::new(StubEvaluator {
            oop_cfvs: vec![0.3, -0.2, 0.1],
            ip_cfvs: vec![-0.1, 0.4, -0.3],
        });
        let opt_out = Arc::new(ConstantOptOut(1000.0));
        let board = vec![10u8, 20, 30, 40, 50];
        let private_cards = [
            vec![(0u8, 1), (2, 3), (4, 5)],
            vec![(6u8, 7), (8, 9), (11, 12)],
        ];

        let gadget = GadgetEvaluator::new(inner, opt_out, 0, board, private_cards);

        let (oop_cfvs, ip_cfvs) = gadget.compute_cfvs_both(
            100, 150.0, &[1.0; 3], &[1.0; 3], 3, 3, 0,
        );

        // Both players' values should be clamped to at least 1000
        // (since both are "opponent" when the other traverses).
        // OOP inner = [0.3, -0.2, 0.1], opt_out = 1000 => all clamp to 1000
        assert_eq!(oop_cfvs, vec![1000.0, 1000.0, 1000.0]);
        // IP inner = [-0.1, 0.4, -0.3], opt_out = 1000 => all clamp to 1000
        assert_eq!(ip_cfvs, vec![1000.0, 1000.0, 1000.0]);
    }

    #[test]
    fn gadget_very_negative_opt_out_matches_inner() {
        // Opt-out is so bad that opponent always enters subtree
        let inner = Arc::new(StubEvaluator {
            oop_cfvs: vec![0.3, -0.2, 0.1],
            ip_cfvs: vec![-0.1, 0.4, -0.3],
        });
        let opt_out = Arc::new(ConstantOptOut(-1000.0));
        let board = vec![10u8, 20, 30, 40, 50];
        let private_cards = [
            vec![(0u8, 1), (2, 3), (4, 5)],
            vec![(6u8, 7), (8, 9), (11, 12)],
        ];

        let gadget = GadgetEvaluator::new(inner, opt_out, 0, board, private_cards);

        let (oop_cfvs, ip_cfvs) = gadget.compute_cfvs_both(
            100, 150.0, &[1.0; 3], &[1.0; 3], 3, 3, 0,
        );

        // With very negative opt-out, no clamping occurs.
        // CFVs should match inner evaluator exactly.
        assert_eq!(oop_cfvs, vec![0.3, -0.2, 0.1], "OOP cfvs should match inner");
        assert_eq!(ip_cfvs, vec![-0.1, 0.4, -0.3], "IP cfvs should match inner");
    }

    #[test]
    fn gadget_compute_cfvs_both_clamps_opponent() {
        let inner = Arc::new(StubEvaluator {
            oop_cfvs: vec![0.5, -0.5],
            ip_cfvs: vec![-0.3, 0.7],
        });
        // Moderate opt-out: should clamp some hands but not others
        let opt_out = Arc::new(ConstantOptOut(0.0));
        let board = vec![10u8, 20, 30, 40, 50];
        let private_cards = [
            vec![(0u8, 1), (2, 3)],
            vec![(4u8, 5), (6, 7)],
        ];

        let gadget = GadgetEvaluator::new(inner, opt_out, 0, board, private_cards);

        let (oop_cfvs, ip_cfvs) = gadget.compute_cfvs_both(
            100, 150.0, &[1.0; 2], &[1.0; 2], 2, 2, 0,
        );

        // IP is opponent when computing OOP cfvs.
        // IP inner = [-0.3, 0.7]. Opt-out = 0.0.
        // IP hand 0: inner=-0.3 < opt_out=0.0 => clamps to 0.0 (opponent improves)
        // IP hand 1: inner=0.7 > opt_out=0.0 => stays 0.7 (opponent already better)
        // So IP cfvs should be [0.0, 0.7]
        assert_eq!(ip_cfvs, vec![0.0, 0.7], "IP hand 0 should clamp up to 0.0");

        // OOP is opponent when computing IP cfvs.
        // OOP inner = [0.5, -0.5]. Opt-out = 0.0.
        // OOP hand 0: inner=0.5 > opt_out=0.0 => stays 0.5
        // OOP hand 1: inner=-0.5 < opt_out=0.0 => clamps to 0.0
        // So OOP cfvs should be [0.5, 0.0]
        assert_eq!(oop_cfvs, vec![0.5, 0.0], "OOP hand 1 should clamp up to 0.0");
    }

    // ---------------------------------------------------------------
    // Integration test with real SubtreeExactEvaluator
    // ---------------------------------------------------------------

    /// Build a `SubtreeExactEvaluator` for a river-boundary spot:
    /// AA,KK,QQ vs TT,99,88 on 7h 5d 2c 3s 9h.
    fn make_test_evaluator(
        iters: u32,
    ) -> (
        crate::exact_subtree::SubtreeExactEvaluator,
        [Vec<(u8, u8)>; 2],
        Vec<u8>,
    ) {
        use range_solver::action_tree::TreeConfig;
        use range_solver::bet_size::BetSizeOptions;
        use range_solver::card::flop_from_str;
        use range_solver::range::Range;
        use range_solver::BoardState;

        let flop = flop_from_str("7h 5d 2c").unwrap();
        let turn_card: u8 = 7; // 3s
        let river_card: u8 = 30; // 9h
        let board = vec![flop[0], flop[1], flop[2], turn_card, river_card];

        let oop_range: Range = "AA,KK,QQ".parse().unwrap();
        let ip_range: Range = "TT,99,88".parse().unwrap();
        let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1 << c));
        let (oop_hands, oop_weights) = oop_range.get_hands_weights(board_mask);
        let (ip_hands, ip_weights) = ip_range.get_hands_weights(board_mask);

        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 200,
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };

        let eval = crate::exact_subtree::SubtreeExactEvaluator::new(
            board.clone(),
            [oop_hands.clone(), ip_hands.clone()],
            [oop_weights, ip_weights],
            tree_config,
        )
        .with_solve_iters(iters);

        let private_cards = [oop_hands, ip_hands];
        (eval, private_cards, board)
    }

    #[test]
    fn gadget_integration_opponent_cfv_geq_opt_out() {
        let (eval, private_cards, board) = make_test_evaluator(200);
        let num_oop = private_cards[0].len();
        let num_ip = private_cards[1].len();

        // First get baseline CFVs without gadget
        let oop_reach = vec![1.0f32; num_oop];
        let ip_reach = vec![1.0f32; num_ip];
        let (baseline_oop, baseline_ip) = eval.compute_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach, num_oop, num_ip, 0,
        );

        // Use a moderate opt-out: 0.0 (break-even)
        let opt_out = Arc::new(ConstantOptOut(0.0));
        let gadget = GadgetEvaluator::new(
            Arc::new(eval),
            opt_out,
            0,
            board,
            private_cards.clone(),
        );

        let (gadget_oop, gadget_ip) = gadget.compute_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach, num_oop, num_ip, 0,
        );

        // Every opponent hand's gadget CFV should be >= opt-out (0.0)
        for (i, &v) in gadget_oop.iter().enumerate() {
            assert!(
                v >= -0.001,
                "OOP hand {i} gadget CFV {v} should be >= opt-out 0.0"
            );
        }
        for (i, &v) in gadget_ip.iter().enumerate() {
            assert!(
                v >= -0.001,
                "IP hand {i} gadget CFV {v} should be >= opt-out 0.0"
            );
        }

        // Gadget should have modified at least some values (any hand below 0
        // in baseline should be clamped up)
        let oop_changed = baseline_oop.iter().zip(gadget_oop.iter())
            .filter(|(&b, &g)| (b - g).abs() > 0.001)
            .count();
        let ip_changed = baseline_ip.iter().zip(gadget_ip.iter())
            .filter(|(&b, &g)| (b - g).abs() > 0.001)
            .count();
        // At least one player should have some hands clamped
        assert!(
            oop_changed > 0 || ip_changed > 0,
            "Gadget should have clamped at least one hand (oop_changed={oop_changed}, ip_changed={ip_changed})"
        );
    }

    #[test]
    fn gadget_integration_very_negative_matches_baseline() {
        let (eval, private_cards, board) = make_test_evaluator(100);
        let num_oop = private_cards[0].len();
        let num_ip = private_cards[1].len();

        let oop_reach = vec![1.0f32; num_oop];
        let ip_reach = vec![1.0f32; num_ip];
        let (baseline_oop, baseline_ip) = eval.compute_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach, num_oop, num_ip, 0,
        );

        // Very negative opt-out should be dominated -- no clamping
        let opt_out = Arc::new(ConstantOptOut(-1000.0));
        let gadget = GadgetEvaluator::new(
            Arc::new(eval),
            opt_out,
            0,
            board,
            private_cards.clone(),
        );

        let (gadget_oop, gadget_ip) = gadget.compute_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach, num_oop, num_ip, 0,
        );

        // Should match baseline exactly
        assert_eq!(gadget_oop, baseline_oop, "OOP should match baseline");
        assert_eq!(gadget_ip, baseline_ip, "IP should match baseline");
    }

    // ---------------------------------------------------------------
    // Abstract vs concrete boundary mismatch tests
    // ---------------------------------------------------------------

    /// Regression test: make_per_boundary_gadget_game must not panic when
    /// the blueprint's abstract action tree has fewer boundaries than the
    /// concrete subgame's action tree (e.g., blueprint built with narrow
    /// bet sizing, explorer using wider bet sizes).
    ///
    /// Reproduces the panic:
    ///   assertion `left == right` failed: per_boundary_opt_outs length (3)
    ///   must match boundary count (N)
    #[test]
    fn make_per_boundary_gadget_game_handles_abstract_concrete_mismatch() {
        use range_solver::action_tree::{ActionTree, TreeConfig};
        use range_solver::bet_size::BetSizeOptions;
        use range_solver::card::{card_from_str, flop_from_str, CardConfig, NOT_DEALT};
        use range_solver::interface::Game;
        use range_solver::range::Range;
        use range_solver::BoardState;

        // Abstract tree has 3 chance descendants (from make_multi_node_cbv_context).
        let (ctx, board, _private_cards) = make_multi_node_cbv_context();
        let cbv_context = Arc::new(ctx);
        let abstract_root: u32 = 0;
        let abstract_boundaries = cbv_context
            .abstract_tree.chance_descendants(abstract_root).len();
        assert_eq!(abstract_boundaries, 3, "test setup: abstract has 3 boundaries");

        // Helper to build a concrete turn-start depth-limited config with wide
        // bet sizes that produce more boundaries than the abstract tree's 3.
        let make_concrete = || -> (CardConfig, ActionTree, usize) {
            let oop: Range = "AA,KK".parse().unwrap();
            let ip: Range = "QQ,JJ".parse().unwrap();
            let cc = CardConfig {
                range: [oop, ip],
                flop: flop_from_str("7h 5d 2c").unwrap(),
                turn: card_from_str("3s").unwrap(),
                river: NOT_DEALT,
            };
            let sizes = BetSizeOptions::try_from(("33%, 50%, 75%, 100%, a", "")).unwrap();
            let tc = TreeConfig {
                initial_state: BoardState::Turn,
                starting_pot: 100,
                effective_stack: 200,
                turn_bet_sizes: [sizes.clone(), sizes.clone()],
                river_bet_sizes: [sizes.clone(), sizes],
                depth_limit: Some(0),
                ..Default::default()
            };
            let at = ActionTree::new(tc).unwrap();
            let tmp = range_solver::game::PostFlopGame::with_config(
                cc.clone(), at,
            ).unwrap();
            let n = tmp.num_boundary_nodes();
            // Rebuild since ActionTree is consumed.
            let sizes2 = BetSizeOptions::try_from(("33%, 50%, 75%, 100%, a", "")).unwrap();
            let tc2 = TreeConfig {
                initial_state: BoardState::Turn,
                starting_pot: 100,
                effective_stack: 200,
                turn_bet_sizes: [sizes2.clone(), sizes2.clone()],
                river_bet_sizes: [sizes2.clone(), sizes2],
                depth_limit: Some(0),
                ..Default::default()
            };
            let at2 = ActionTree::new(tc2).unwrap();
            (cc, at2, n)
        };

        let (cc, at, concrete_boundaries) = make_concrete();
        assert!(
            concrete_boundaries > abstract_boundaries,
            "test setup: concrete ({concrete_boundaries}) should have \
             more boundaries than abstract ({abstract_boundaries})"
        );

        // Stub boundary evaluator that can handle any hand count.
        // We need to know hand counts, so build a throwaway game.
        let (cc2, at2, _) = make_concrete();
        let probe = range_solver::game::PostFlopGame::with_config(cc2, at2).unwrap();
        let n_oop = probe.num_private_hands(0);
        let n_ip = probe.num_private_hands(1);
        let inner: Arc<dyn range_solver::game::BoundaryEvaluator> =
            Arc::new(StubEvaluator {
                oop_cfvs: vec![0.0; n_oop],
                ip_cfvs: vec![0.0; n_ip],
            });

        // This must not panic (previously panicked with length mismatch).
        let game = make_per_boundary_gadget_game(
            cc, at, &cbv_context, abstract_root, &board, inner,
        ).expect("make_per_boundary_gadget_game should not fail");

        // After gadget injection: 3x concrete boundaries.
        assert_eq!(
            game.num_boundary_nodes(),
            3 * concrete_boundaries,
            "gadget injection should triple the concrete boundary count"
        );
    }

    // ---------------------------------------------------------------
    // broadcast_opt_outs tests
    // ---------------------------------------------------------------

    #[test]
    fn broadcast_opt_outs_single_boundary_replicates_unchanged() {
        let abstract_opt_outs = vec![
            [vec![0.5, -0.3], vec![0.1, 0.2, 0.3]],
        ];
        let result = broadcast_opt_outs(&abstract_opt_outs, 4);
        assert_eq!(result.len(), 4);
        for entry in &result {
            assert_eq!(entry[0], vec![0.5, -0.3]);
            assert_eq!(entry[1], vec![0.1, 0.2, 0.3]);
        }
    }

    #[test]
    fn broadcast_opt_outs_averages_across_boundaries() {
        let abstract_opt_outs = vec![
            [vec![1.0, 2.0], vec![3.0]],
            [vec![3.0, 4.0], vec![5.0]],
        ];
        let result = broadcast_opt_outs(&abstract_opt_outs, 3);
        assert_eq!(result.len(), 3);
        // OOP hand 0: avg(1.0, 3.0) = 2.0
        // OOP hand 1: avg(2.0, 4.0) = 3.0
        // IP hand 0: avg(3.0, 5.0) = 4.0
        for entry in &result {
            assert!((entry[0][0] - 2.0).abs() < 1e-6);
            assert!((entry[0][1] - 3.0).abs() < 1e-6);
            assert!((entry[1][0] - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    #[should_panic(expected = "abstract_opt_outs must not be empty")]
    fn broadcast_opt_outs_panics_on_empty() {
        broadcast_opt_outs(&[], 5);
    }

    #[test]
    fn broadcast_opt_outs_target_one() {
        let abstract_opt_outs = vec![
            [vec![1.0], vec![2.0]],
            [vec![3.0], vec![4.0]],
        ];
        let result = broadcast_opt_outs(&abstract_opt_outs, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0][0][0] - 2.0).abs() < 1e-6);
        assert!((result[0][1][0] - 3.0).abs() < 1e-6);
    }

    /// Build a CbvContext whose abstract tree has BOTH Chance nodes AND
    /// all-in Showdown terminals on the starting street. This mirrors
    /// what a real blueprint tree looks like: all-in-call paths produce
    /// Showdown terminals (not Chance nodes) because there are no further
    /// decisions. The concrete range-solver tree, however, creates depth
    /// boundaries for ALL of these paths.
    ///
    /// Tree structure (root=0, Turn):
    ///   0: Decision(P0, Turn, [Check, AllIn])
    ///     1: Decision(P1, Turn, [Check])
    ///       2: Chance(River, child=3) -- boundary 0 (normal)
    ///         3: Terminal(Showdown, pot=10)
    ///     4: Decision(P1, Turn, [Fold, Call])
    ///       5: Terminal(Fold)
    ///       6: Terminal(Showdown, pot=100, stacks=[0,0]) -- boundary 1 (all-in)
    ///
    /// `chance_descendants(0)` returns [2] (1 node).
    /// `boundary_descendants(0)` returns [2, 6] (2 nodes).
    /// A concrete range-solver tree with identical bet sizes and
    /// depth_limit=0 would have 2 depth boundaries.
    fn make_allin_showdown_cbv_context()
        -> (crate::postflop::CbvContext, Vec<u8>, [Vec<(u8, u8)>; 2])
    {
        use crate::postflop::CbvContext;
        use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
        use poker_solver_core::blueprint_v2::cbv::CbvTable;
        use poker_solver_core::blueprint_v2::game_tree::{
            GameNode, GameTree, TerminalKind, TreeAction,
        };
        use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
        use poker_solver_core::blueprint_v2::Street;
        use range_solver::card::flop_from_str;

        let nodes = vec![
            // 0: Root decision (2 actions: Check, AllIn)
            GameNode::Decision {
                player: 0,
                street: Street::Turn,
                actions: vec![TreeAction::Check, TreeAction::AllIn],
                children: vec![1, 4],
                blueprint_decision_idx: None,
            },
            // 1: P1 after check
            GameNode::Decision {
                player: 1,
                street: Street::Turn,
                actions: vec![TreeAction::Check],
                children: vec![2],
                blueprint_decision_idx: None,
            },
            // 2: Chance (check-check) => river transition
            GameNode::Chance {
                next_street: Street::River,
                child: 3,
            },
            // 3: Terminal after river (placeholder)
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 10.0,
                stacks: [45.0, 45.0],
            },
            // 4: P1 after all-in
            GameNode::Decision {
                player: 1,
                street: Street::Turn,
                actions: vec![TreeAction::Fold, TreeAction::Call],
                children: vec![5, 6],
                blueprint_decision_idx: None,
            },
            // 5: Fold terminal
            GameNode::Terminal {
                kind: TerminalKind::Fold { winner: 0 },
                pot: 10.0,
                stacks: [45.0, 45.0],
            },
            // 6: All-in call => Showdown (NOT a Chance node!)
            // Both stacks are 0 because both players are all-in.
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 100.0,
                stacks: [0.0, 0.0],
            },
        ];

        // 1 CBV entry for the single Chance node (ordinal 0).
        // The all-in Showdown at node 6 has no CBV entry because
        // the blueprint tree doesn't have a Chance node there.
        let cbv_table = CbvTable {
            values: vec![10.0, -5.0],
            node_offsets: vec![0],
            buckets_per_node: vec![2],
        };

        let mut ab = AllBuckets::new([2, 2, 2, 2], [None, None, None, None]);
        ab.equity_fallback = true;
        let all_buckets = Arc::new(ab);
        let strategy = Arc::new(BlueprintV2Strategy::empty());
        let tree = GameTree {
            nodes,
            root: 0,
            dealer: 0,
            starting_stack: 50.0,
        };

        let ctx = CbvContext {
            cbv_table,
            abstract_tree: tree,
            all_buckets,
            strategy,
        };

        let flop = flop_from_str("7h 5d 2c").unwrap();
        let board = vec![flop[0], flop[1], flop[2], 7u8, 30u8];
        let oop_hands = vec![(48u8, 49u8)];
        let ip_hands = vec![(4u8, 5u8)];
        let private_cards = [oop_hands, ip_hands];

        (ctx, board, private_cards)
    }

    /// The abstract tree has 1 Chance + 1 all-in Showdown, but
    /// `from_cbv_context` currently only counts Chance nodes (1).
    /// The concrete tree would have 2 depth boundaries.
    /// `from_cbv_context` must return 2 boundaries so counts match.
    #[test]
    fn from_cbv_context_counts_allin_showdown_as_boundary() {
        let (ctx, board, private_cards) = make_allin_showdown_cbv_context();

        let provider = BlueprintCbvOptOut::from_cbv_context(
            &ctx, 0, &board, &private_cards,
        );

        // Must find 2 boundaries: Chance node + all-in Showdown.
        assert_eq!(
            provider.num_boundaries(), 2,
            "from_cbv_context should count both Chance and all-in \
             Showdown boundaries; got {}",
            provider.num_boundaries(),
        );
    }

    /// For all-in Showdown boundaries (no CBV entry), opt-out must be
    /// finite and very negative so the gadget never clamps -- the
    /// opponent already committed all chips and cannot opt out.
    ///
    /// Must NOT be NEG_INFINITY: solver arithmetic
    /// `bcfvs[h] * payoff_scale * cfreach_adj` produces NaN when
    /// bcfvs[h] = -inf and any factor is 0. NaN propagates through
    /// compute_mes_ev -> compute_exploitability, breaking the BR pass.
    #[test]
    fn from_cbv_context_allin_boundary_opt_out_is_finite_and_very_negative() {
        let (ctx, board, private_cards) = make_allin_showdown_cbv_context();

        let provider = BlueprintCbvOptOut::from_cbv_context(
            &ctx, 0, &board, &private_cards,
        );

        // Boundary 0 is the Chance node => normal CBV opt-out.
        // Boundary 1 is the all-in Showdown => must be finite + dominated.
        let oop_cfvs = provider.opt_out_cfvs(
            1, 0, 100, 200, &board, &private_cards[0],
        );
        let ip_cfvs = provider.opt_out_cfvs(
            1, 1, 100, 200, &board, &private_cards[1],
        );
        for &v in oop_cfvs.iter().chain(ip_cfvs.iter()) {
            assert!(
                v.is_finite(),
                "all-in boundary opt-out must be finite (not NEG_INFINITY), got {v}"
            );
            assert!(
                v < -1e8,
                "all-in boundary opt-out must be very negative (dominated), got {v}"
            );
        }
    }

    #[test]
    fn gadget_partial_clamp_some_hands_clamped_others_not() {
        let inner = Arc::new(StubEvaluator {
            oop_cfvs: vec![0.8, -0.3, 0.1],
            ip_cfvs: vec![0.5, -0.7, 0.2],
        });
        // Opt-out at 0.0: hands below 0 get clamped
        let opt_out = Arc::new(ConstantOptOut(0.0));
        let board = vec![10u8, 20, 30, 40, 50];
        let private_cards = [
            vec![(0u8, 1), (2, 3), (4, 5)],
            vec![(6u8, 7), (8, 9), (11, 12)],
        ];

        let gadget = GadgetEvaluator::new(inner, opt_out, 0, board, private_cards);

        let (oop_cfvs, ip_cfvs) = gadget.compute_cfvs_both(
            100, 150.0, &[1.0; 3], &[1.0; 3], 3, 3, 0,
        );

        // IP (opponent when computing OOP cfvs):
        // inner = [0.5, -0.7, 0.2], opt_out = 0.0
        // => [0.5, 0.0, 0.2] (hand 1 clamped from -0.7 to 0.0)
        assert_eq!(ip_cfvs, vec![0.5, 0.0, 0.2]);

        // OOP (opponent when computing IP cfvs):
        // inner = [0.8, -0.3, 0.1], opt_out = 0.0
        // => [0.8, 0.0, 0.1] (hand 1 clamped from -0.3 to 0.0)
        assert_eq!(oop_cfvs, vec![0.8, 0.0, 0.1]);
    }

}
