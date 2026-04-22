//! Exact subtree boundary evaluator.
//!
//! At each boundary, builds a fresh PostFlopGame for the downstream subtree,
//! runs full DCFR to convergence, and returns per-hand CFVs. This is a
//! diagnostic/eval tool — performance is not a concern, correctness is paramount.

use std::collections::HashMap;
use std::sync::Mutex;

use range_solver::action_tree::{ActionTree, TreeConfig};
use range_solver::card::{CardConfig, NOT_DEALT};
use range_solver::range::Range;
use range_solver::{solve_step, finalize, root_cfvalues_with_reach, BoardState, PostFlopGame};
use range_solver::interface::Game;

/// Default number of DCFR iterations for the exact subtree solve.
const DEFAULT_SOLVE_ITERS: u32 = 500;

/// Boundary evaluator that solves the downstream subtree exactly via DCFR.
pub struct SubtreeExactEvaluator {
    /// Board cards at this boundary (3, 4, or 5 cards).
    board: Vec<u8>,
    /// Private card lists per player, aligned with parent game ordering.
    private_cards: [Vec<(u8, u8)>; 2],
    /// Parent game's initial weights per player (same ordering as
    /// `private_cards`). Used to build the subtree CardConfig so that
    /// `num_combinations` matches the parent game exactly.
    parent_initial_weights: [Vec<f32>; 2],
    /// Tree config inherited from parent game (bet sizes, pot, stack).
    parent_tree_config: TreeConfig,
    /// Number of DCFR iterations to run.
    solve_iters: u32,
    /// Cache keyed by rounded reach digest.
    cache: Mutex<HashMap<u64, (Vec<f32>, Vec<f32>)>>,
}

impl SubtreeExactEvaluator {
    /// Create a new evaluator for the given boundary.
    pub fn new(
        board: Vec<u8>,
        private_cards: [Vec<(u8, u8)>; 2],
        parent_initial_weights: [Vec<f32>; 2],
        parent_tree_config: TreeConfig,
    ) -> Self {
        Self {
            board,
            private_cards,
            parent_initial_weights,
            parent_tree_config,
            solve_iters: DEFAULT_SOLVE_ITERS,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Set the number of DCFR iterations (for testing).
    pub fn with_solve_iters(mut self, iters: u32) -> Self {
        self.solve_iters = iters;
        self
    }
}

/// Compute a cache key from two reach vectors by rounding to 3 decimals
/// and hashing the resulting byte representation.
fn reach_cache_key(oop_reach: &[f32], ip_reach: &[f32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for &v in oop_reach {
        let rounded = (v * 1000.0).round() as i32;
        rounded.hash(&mut hasher);
    }
    for &v in ip_reach {
        let rounded = (v * 1000.0).round() as i32;
        rounded.hash(&mut hasher);
    }
    hasher.finish()
}

/// Determine the `BoardState` (initial_state) from board length.
fn board_state_from_len(n: usize) -> BoardState {
    match n {
        3 => BoardState::Flop,
        4 => BoardState::Turn,
        5 => BoardState::River,
        _ => panic!("invalid board length for subtree: {n}"),
    }
}

/// Build a CardConfig using the parent game's initial weights (not
/// boundary reaches). This ensures `num_combinations` in the subtree
/// matches the parent, which is required for correct CFV scaling.
fn build_card_config(
    board: &[u8],
    private_cards: &[Vec<(u8, u8)>; 2],
    parent_weights: &[Vec<f32>; 2],
) -> CardConfig {
    let flop = [board[0], board[1], board[2]];
    let turn = if board.len() > 3 { board[3] } else { NOT_DEALT };
    let river = if board.len() > 4 { board[4] } else { NOT_DEALT };

    let mut ranges = [Range::new(), Range::new()];
    for player in 0..2 {
        for (i, &(c1, c2)) in private_cards[player].iter().enumerate() {
            let w = parent_weights[player].get(i).copied().unwrap_or(0.0);
            // Clamp to [0, 1] for Range validity.
            let w = w.max(0.0).min(1.0);
            if w > 0.0 {
                ranges[player].set_weight_by_cards(c1, c2, w);
            }
        }
    }

    CardConfig {
        range: ranges,
        flop,
        turn,
        river,
    }
}

/// True when every element is (near-)zero.
fn reach_is_all_zero(reach: &[f32]) -> bool {
    reach.iter().all(|&v| v.abs() < 1e-9)
}

/// Build the subtree game, run DCFR, finalize, and extract boundary CFVs.
///
/// The subtree is built with the PARENT's initial weights (so that
/// `num_combinations` matches the parent exactly). The equilibrium
/// strategies do not depend on initial weights (two-player zero-sum).
///
/// After solving, per-hand cfvalues are computed using the ACTUAL boundary
/// reach via `root_cfvalues_with_reach`, then converted to the boundary
/// evaluator format (bcfv) by dividing out the `half_pot / N * cfreach_adj`
/// factor that `evaluate_boundary_single` will multiply back in.
fn solve_subtree(
    board: &[u8],
    private_cards: &[Vec<(u8, u8)>; 2],
    parent_weights: &[Vec<f32>; 2],
    parent_tree_config: &TreeConfig,
    pot: i32,
    remaining_stack: f64,
    oop_reach: &[f32],
    ip_reach: &[f32],
    solve_iters: u32,
) -> (Vec<f32>, Vec<f32>) {
    if reach_is_all_zero(oop_reach) || reach_is_all_zero(ip_reach) {
        return (vec![0.0; oop_reach.len()], vec![0.0; ip_reach.len()]);
    }

    let card_config = build_card_config(board, private_cards, parent_weights);
    let initial_state = board_state_from_len(board.len());
    let effective_stack =
        (pot / 2).saturating_add(remaining_stack.round() as i32);

    // The subtree represents the DOWNSTREAM game starting from the boundary.
    // For a turn→river boundary the initial_state is Turn (required by
    // PostFlopGame when river is NOT_DEALT), but the turn action layer must
    // be empty (forced check-check) because the parent already handled turn
    // decisions.  Only river_bet_sizes matter for the actual subtree play.
    let tree_config = TreeConfig {
        initial_state,
        starting_pot: pot,
        effective_stack,
        river_bet_sizes: parent_tree_config.river_bet_sizes.clone(),
        depth_limit: None,
        ..Default::default()
    };

    let tree = ActionTree::new(tree_config)
        .unwrap_or_else(|e| panic!("subtree ActionTree failed: {e}"));
    let mut game = PostFlopGame::with_config(card_config, tree)
        .unwrap_or_else(|e| panic!("subtree PostFlopGame failed: {e}"));
    game.allocate_memory(false);

    run_dcfr(&mut game, solve_iters);
    finalize(&mut game);

    // Compute cfvalues at the root using the ACTUAL boundary reach rather
    // than the game's initial_weights. This is critical: `evaluate_boundary_single`
    // in the parent solver will multiply bcfv by `half_pot / N * cfreach_adj`,
    // and the cfreach_adj comes from the actual boundary reach.
    let sub_oop_reach = remap_reach_to_subtree(
        oop_reach, private_cards, game.private_cards(0), 0,
    );
    let sub_ip_reach = remap_reach_to_subtree(
        ip_reach, private_cards, game.private_cards(1), 1,
    );

    let half_pot = pot as f64 / 2.0;
    let num_combos = game.num_combinations();

    // OOP cfvalue uses IP reach as opponent cfreach.
    let oop_cfv = root_cfvalues_with_reach(&game, 0, &sub_ip_reach);
    // IP cfvalue uses OOP reach as opponent cfreach.
    let ip_cfv = root_cfvalues_with_reach(&game, 1, &sub_oop_reach);

    // Convert cfvalues to bcfv format.
    // evaluate_boundary_single does: result = bcfv * (half_pot / N) * cfreach_adj
    // We want result = cfv, so: bcfv = cfv / (half_pot / N * cfreach_adj)
    //                              = cfv * N / (half_pot * cfreach_adj)
    let oop_bcfv = cfv_to_bcfv(
        &oop_cfv,
        game.private_cards(0),
        game.private_cards(1),
        &sub_ip_reach,
        half_pot,
        num_combos,
    );
    let ip_bcfv = cfv_to_bcfv(
        &ip_cfv,
        game.private_cards(1),
        game.private_cards(0),
        &sub_oop_reach,
        half_pot,
        num_combos,
    );

    // Remap from subtree ordering to parent ordering.
    let oop_out = remap_cfvs_to_parent(
        &oop_bcfv, game.private_cards(0), &private_cards[0],
    );
    let ip_out = remap_cfvs_to_parent(
        &ip_bcfv, game.private_cards(1), &private_cards[1],
    );
    (oop_out, ip_out)
}

/// Convert raw cfvalues to bcfv format by dividing out the scaling factor
/// that `evaluate_boundary_single` will multiply back in.
///
/// For hand h: `bcfv[h] = cfv[h] * N / (half_pot * cfreach_adj[h])`
/// where cfreach_adj is the blocker-adjusted opponent reach sum.
fn cfv_to_bcfv(
    cfv: &[f32],
    hero_cards: &[(u8, u8)],
    opp_cards: &[(u8, u8)],
    opp_reach: &[f32],
    half_pot: f64,
    num_combos: f64,
) -> Vec<f32> {
    let denom_base = half_pot / num_combos;
    cfv.iter()
        .enumerate()
        .map(|(h, &c)| {
            if c == 0.0 {
                return 0.0;
            }
            let (h1, h2) = hero_cards[h];
            let adj = cfreach_adj(h1, h2, opp_cards, opp_reach);
            if adj <= 0.0 {
                return 0.0;
            }
            (c as f64 / (denom_base * adj)) as f32
        })
        .collect()
}

/// Compute the blocker-adjusted opponent reach sum for a hero hand.
/// This mirrors the cfreach accumulation in `evaluate_boundary_single`:
/// sum of opponent reach for hands that don't share any card with the
/// hero hand.
fn cfreach_adj(h1: u8, h2: u8, opp_cards: &[(u8, u8)], opp_reach: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for (j, &(o1, o2)) in opp_cards.iter().enumerate() {
        if o1 == h1 || o1 == h2 || o2 == h1 || o2 == h2 {
            continue;
        }
        let r = opp_reach.get(j).copied().unwrap_or(0.0);
        if r > 0.0 {
            sum += r as f64;
        }
    }
    sum
}

/// Remap a parent-ordering reach vector to the subtree's private_cards
/// ordering. Hands in the subtree that don't appear in the parent get
/// reach 0.
fn remap_reach_to_subtree(
    parent_reach: &[f32],
    parent_private_cards: &[Vec<(u8, u8)>; 2],
    subtree_cards: &[(u8, u8)],
    player: usize,
) -> Vec<f32> {
    let parent_cards = &parent_private_cards[player];
    // Build map: canonical card pair → reach
    let mut map: HashMap<(u8, u8), f32> =
        HashMap::with_capacity(parent_cards.len());
    for (i, &(c1, c2)) in parent_cards.iter().enumerate() {
        let key = if c1 <= c2 { (c1, c2) } else { (c2, c1) };
        let r = parent_reach.get(i).copied().unwrap_or(0.0);
        map.insert(key, r);
    }
    subtree_cards
        .iter()
        .map(|&(c1, c2)| {
            let key = if c1 <= c2 { (c1, c2) } else { (c2, c1) };
            map.get(&key).copied().unwrap_or(0.0)
        })
        .collect()
}

/// Remap CFVs from subtree hand ordering to parent hand ordering.
fn remap_cfvs_to_parent(
    subtree_cfvs: &[f32],
    subtree_cards: &[(u8, u8)],
    parent_cards: &[(u8, u8)],
) -> Vec<f32> {
    let mut map: HashMap<(u8, u8), f32> =
        HashMap::with_capacity(subtree_cards.len());
    for (i, &(c1, c2)) in subtree_cards.iter().enumerate() {
        let key = if c1 <= c2 { (c1, c2) } else { (c2, c1) };
        let cfv = subtree_cfvs.get(i).copied().unwrap_or(0.0);
        map.insert(key, cfv);
    }
    parent_cards
        .iter()
        .map(|&(c1, c2)| {
            let key = if c1 <= c2 { (c1, c2) } else { (c2, c1) };
            map.get(&key).copied().unwrap_or(0.0)
        })
        .collect()
}

/// Run DCFR for the given number of iterations.
fn run_dcfr(game: &mut PostFlopGame, iters: u32) {
    for t in 0..iters {
        solve_step(game, t);
    }
}

impl range_solver::game::BoundaryEvaluator for SubtreeExactEvaluator {
    fn compute_cfvs(
        &self,
        player: usize,
        pot: i32,
        remaining_stack: f64,
        opponent_reach: &[f32],
        num_hands: usize,
        _continuation_index: usize,
    ) -> Vec<f32> {
        let (oop_reach, ip_reach) = if player == 0 {
            let oop_dummy = vec![1.0f32; num_hands];
            (oop_dummy, opponent_reach.to_vec())
        } else {
            let ip_dummy = vec![1.0f32; num_hands];
            (opponent_reach.to_vec(), ip_dummy)
        };
        let num_opp = opponent_reach.len();
        let (num_oop, num_ip) = if player == 0 {
            (num_hands, num_opp)
        } else {
            (num_opp, num_hands)
        };
        let (oop_cfvs, ip_cfvs) = self.compute_cfvs_both(
            pot, remaining_stack, &oop_reach, &ip_reach,
            num_oop, num_ip, 0,
        );
        if player == 0 { oop_cfvs } else { ip_cfvs }
    }

    fn compute_cfvs_both(
        &self,
        pot: i32,
        remaining_stack: f64,
        oop_reach: &[f32],
        ip_reach: &[f32],
        _num_oop: usize,
        _num_ip: usize,
        _continuation_index: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let key = reach_cache_key(oop_reach, ip_reach);
        {
            let cache = self.cache.lock().expect("cache lock poisoned");
            if let Some(cached) = cache.get(&key) {
                return cached.clone();
            }
        }

        let result = solve_subtree(
            &self.board,
            &self.private_cards,
            &self.parent_initial_weights,
            &self.parent_tree_config,
            pot,
            remaining_stack,
            oop_reach,
            ip_reach,
            self.solve_iters,
        );

        let mut cache = self.cache.lock().expect("cache lock poisoned");
        cache.insert(key, result.clone());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::flop_from_str;
    use range_solver::game::BoundaryEvaluator;

    /// Helper: build a SubtreeExactEvaluator for a river-boundary spot.
    /// AA,KK,QQ vs TT,99,88 on 7h 5d 2c 3s 9h (balanced equities).
    fn make_river_evaluator(solve_iters: u32) -> SubtreeExactEvaluator {
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

        SubtreeExactEvaluator::new(
            board,
            [oop_hands.clone(), ip_hands.clone()],
            [oop_weights, ip_weights],
            tree_config,
        ).with_solve_iters(solve_iters)
    }

    #[test]
    fn evaluator_returns_correct_length_cfvs() {
        let eval = make_river_evaluator(50);
        let num_oop = eval.private_cards[0].len();
        let num_ip = eval.private_cards[1].len();
        assert!(num_oop > 0, "should have OOP hands");
        assert!(num_ip > 0, "should have IP hands");

        let oop_reach = vec![1.0f32; num_oop];
        let ip_reach = vec![1.0f32; num_ip];
        let (oop_cfvs, ip_cfvs) = eval.compute_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach, num_oop, num_ip, 0,
        );
        assert_eq!(oop_cfvs.len(), num_oop);
        assert_eq!(ip_cfvs.len(), num_ip);
    }

    #[test]
    fn evaluator_cfvs_are_finite() {
        let eval = make_river_evaluator(100);
        let num_oop = eval.private_cards[0].len();
        let num_ip = eval.private_cards[1].len();
        let oop_reach = vec![1.0f32; num_oop];
        let ip_reach = vec![1.0f32; num_ip];
        let (oop_cfvs, ip_cfvs) = eval.compute_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach, num_oop, num_ip, 0,
        );
        for &v in &oop_cfvs {
            assert!(v.is_finite(), "OOP CFV should be finite, got {v}");
        }
        for &v in &ip_cfvs {
            assert!(v.is_finite(), "IP CFV should be finite, got {v}");
        }
    }

    #[test]
    fn evaluator_cfvs_have_sensible_values() {
        let eval = make_river_evaluator(200);
        let num_oop = eval.private_cards[0].len();
        let num_ip = eval.private_cards[1].len();
        let oop_reach = vec![1.0f32; num_oop];
        let ip_reach = vec![1.0f32; num_ip];
        let (oop_cfvs, ip_cfvs) = eval.compute_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach, num_oop, num_ip, 0,
        );

        let oop_positive = oop_cfvs.iter().filter(|&&v| v > 0.0).count();
        assert!(
            oop_positive > num_oop / 2,
            "majority of OOP hands should have positive CFV, got {oop_positive}/{num_oop}"
        );

        let ip_positive = ip_cfvs.iter().filter(|&&v| v > 0.0).count();
        let ip_negative = ip_cfvs.iter().filter(|&&v| v < -0.5).count();
        assert!(ip_positive > 0, "some IP hands (99) should be positive");
        assert!(ip_negative > 0, "some IP hands (TT,88) should be negative");
    }

    #[test]
    fn evaluator_cache_returns_same_values() {
        let eval = make_river_evaluator(50);
        let num_oop = eval.private_cards[0].len();
        let num_ip = eval.private_cards[1].len();
        let oop_reach = vec![1.0f32; num_oop];
        let ip_reach = vec![1.0f32; num_ip];

        let (oop1, ip1) = eval.compute_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach, num_oop, num_ip, 0,
        );
        let (oop2, ip2) = eval.compute_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach, num_oop, num_ip, 0,
        );
        assert_eq!(oop1, oop2, "cached OOP CFVs should match");
        assert_eq!(ip1, ip2, "cached IP CFVs should match");
    }

    #[test]
    fn evaluator_different_reaches_give_different_cfvs() {
        let eval = make_river_evaluator(50);
        let num_oop = eval.private_cards[0].len();
        let num_ip = eval.private_cards[1].len();

        let oop_reach_full = vec![1.0f32; num_oop];
        let ip_reach_full = vec![1.0f32; num_ip];

        let mut ip_reach_half = ip_reach_full.clone();
        for w in ip_reach_half.iter_mut().take(num_ip / 2) {
            *w = 0.0;
        }

        let (oop1, _) = eval.compute_cfvs_both(
            100, 150.0, &oop_reach_full, &ip_reach_full, num_oop, num_ip, 0,
        );
        let (oop2, _) = eval.compute_cfvs_both(
            100, 150.0, &oop_reach_full, &ip_reach_half, num_oop, num_ip, 0,
        );
        assert_ne!(oop1, oop2, "different reaches should produce different CFVs");
    }

    #[test]
    fn board_state_from_len_correct() {
        assert_eq!(board_state_from_len(3), BoardState::Flop);
        assert_eq!(board_state_from_len(4), BoardState::Turn);
        assert_eq!(board_state_from_len(5), BoardState::River);
    }

    #[test]
    #[should_panic(expected = "invalid board length")]
    fn board_state_from_len_panics_on_invalid() {
        board_state_from_len(2);
    }

    #[test]
    fn reach_cache_key_deterministic() {
        let a = vec![1.0f32, 0.5, 0.333];
        let b = vec![0.0f32, 1.0, 0.667];
        let k1 = reach_cache_key(&a, &b);
        let k2 = reach_cache_key(&a, &b);
        assert_eq!(k1, k2);
    }

    #[test]
    fn reach_cache_key_differs_for_different_reach() {
        let a = vec![1.0f32, 0.5, 0.333];
        let b = vec![0.0f32, 1.0, 0.667];
        let c = vec![0.0f32, 1.0, 0.668];
        let k1 = reach_cache_key(&a, &b);
        let k2 = reach_cache_key(&a, &c);
        assert_ne!(k1, k2);
    }

    #[test]
    fn compute_cfvs_single_player_matches_both() {
        let eval = make_river_evaluator(50);
        let num_oop = eval.private_cards[0].len();
        let num_ip = eval.private_cards[1].len();
        let oop_reach = vec![1.0f32; num_oop];
        let ip_reach = vec![1.0f32; num_ip];

        let oop_single = eval.compute_cfvs(
            0, 100, 150.0, &ip_reach, num_oop, 0,
        );
        let (oop_both, _) = eval.compute_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach, num_oop, num_ip, 0,
        );
        assert_eq!(oop_single, oop_both, "single-player should match both");
    }
}
