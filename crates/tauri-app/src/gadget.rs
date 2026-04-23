//! Libratus-style safe re-solving gadget.
//!
//! Clamps opponent per-hand CFV upward to a pre-computed opt-out floor
//! (typically blueprint CBVs). Makes subgame boundary evaluators "safe"
//! in the sense that the reported opponent CFV is never worse than the
//! blueprint would guarantee. See `docs/plans/2026-04-23-deepstack-gadget.md`
//! for the full design and the distinction from DeepStack-proper
//! (which requires a cfvnet retrain; bean poker_solver_rust-akg3).

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

/// Boundary evaluator wrapper that applies the Libratus range gadget.
///
/// Delegates to an inner `BoundaryEvaluator`, then clamps each opponent
/// hand's CFV upward to the opt-out value. This ensures the opponent
/// never does worse than their blueprint counterfactual best-response.
pub struct GadgetEvaluator {
    inner: Arc<dyn range_solver::game::BoundaryEvaluator>,
    opt_out: Arc<dyn OptOutProvider>,
    board: Vec<u8>,
    private_cards: [Vec<(u8, u8)>; 2],
}

impl GadgetEvaluator {
    pub fn new(
        inner: Arc<dyn range_solver::game::BoundaryEvaluator>,
        opt_out: Arc<dyn OptOutProvider>,
        board: Vec<u8>,
        private_cards: [Vec<(u8, u8)>; 2],
    ) -> Self {
        Self { inner, opt_out, board, private_cards }
    }
}

/// Apply opt-out clamping: for each opponent hand, clamp CFV upward
/// to the opt-out value. Returns the adjusted (player_cfvs, opp_cfvs).
fn apply_gadget_clamp(
    player_cfvs: &[f32],
    opp_cfvs: &[f32],
    opt_out_cfvs: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    // Clamp opponent CFVs upward to opt-out
    let clamped_opp: Vec<f32> = opp_cfvs.iter()
        .zip(opt_out_cfvs.iter())
        .map(|(&inner, &opt)| inner.max(opt))
        .collect();
    // Player CFVs stay the same -- the gadget only constrains the opponent.
    // In the bcfv interface, player and opponent values are independent
    // per-hand scalars (not directly zero-sum per-combo).
    (player_cfvs.to_vec(), clamped_opp)
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
            0, 0, pot, eff_stack, &self.board, &self.private_cards[0],
        );
        let ip_opt_out = self.opt_out.opt_out_cfvs(
            0, 1, pot, eff_stack, &self.board, &self.private_cards[1],
        );
        // When computing OOP cfvs, IP is opponent => clamp IP upward
        let (_oop_adj_for_oop, ip_clamped) = apply_gadget_clamp(
            &oop_inner, &ip_inner, &ip_opt_out,
        );
        // When computing IP cfvs, OOP is opponent => clamp OOP upward
        let (_ip_adj_for_ip, oop_clamped) = apply_gadget_clamp(
            &ip_inner, &oop_inner, &oop_opt_out,
        );
        (oop_clamped, ip_clamped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use range_solver::game::BoundaryEvaluator;

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

        let gadget = GadgetEvaluator::new(inner, opt_out, board, private_cards);

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

        let gadget = GadgetEvaluator::new(inner, opt_out, board, private_cards);

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

        let gadget = GadgetEvaluator::new(inner, opt_out, board, private_cards);

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

        let gadget = GadgetEvaluator::new(inner, opt_out, board, private_cards);

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
