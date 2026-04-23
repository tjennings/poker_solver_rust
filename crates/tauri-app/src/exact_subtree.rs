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
/// `num_combinations` matches the parent exactly). After solving, per-hand
/// cfvalues are computed using the ACTUAL boundary reach and converted to
/// bcfv format by dividing out the `half_pot / N * cfreach_adj` factor.
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

    // Remap parent reach to subtree hand ordering
    let sub_oop_reach = remap_reach_to_subtree(
        oop_reach, private_cards, game.private_cards(0), 0,
    );
    let sub_ip_reach = remap_reach_to_subtree(
        ip_reach, private_cards, game.private_cards(1), 1,
    );

    let half_pot = pot as f64 / 2.0;
    let num_combos = game.num_combinations();

    // Compute cfvalues using actual boundary reach
    let oop_cfv = root_cfvalues_with_reach(&game, 0, &sub_ip_reach);
    let ip_cfv = root_cfvalues_with_reach(&game, 1, &sub_oop_reach);

    // Convert cfvalues to bcfv format:
    // bcfv[h] = cfv[h] * N / (half_pot * cfreach_adj[h])
    let oop_bcfv = cfv_to_bcfv(
        &oop_cfv, game.private_cards(0),
        game.private_cards(1), &sub_ip_reach,
        half_pot, num_combos,
    );
    let ip_bcfv = cfv_to_bcfv(
        &ip_cfv, game.private_cards(1),
        game.private_cards(0), &sub_oop_reach,
        half_pot, num_combos,
    );

    // Remap from subtree ordering to parent ordering
    let oop_out = remap_cfvs_to_parent(
        &oop_bcfv, game.private_cards(0), &private_cards[0],
    );
    let ip_out = remap_cfvs_to_parent(
        &ip_bcfv, game.private_cards(1), &private_cards[1],
    );
    (oop_out, ip_out)
}

/// Floor for `cfreach_adj` before division. Below this we treat the hand as
/// effectively unreachable (returns 0 bcfv); avoids divide-by-zero blowups
/// for blocker-heavy hands.
const MIN_ADJ: f64 = 1e-6;
/// Clamp bound for the returned bcfv magnitude. Pot-normalised bcfv should
/// live in `[-1, 1]`; anything larger is a numerical outlier from small
/// `cfreach_adj`. Clamping prevents outliers from poisoning regret updates.
const MAX_BCFV: f32 = 10.0;

/// Convert raw cfvalues to bcfv format.
/// `bcfv[h] = cfv[h] * N / (half_pot * cfreach_adj[h])`
fn cfv_to_bcfv(
    cfv: &[f32],
    hero_cards: &[(u8, u8)],
    opp_cards: &[(u8, u8)],
    opp_reach: &[f32],
    half_pot: f64,
    num_combos: f64,
) -> Vec<f32> {
    let scale = num_combos / half_pot;
    cfv.iter()
        .enumerate()
        .map(|(h, &c)| {
            if c == 0.0 {
                return 0.0;
            }
            let (h1, h2) = hero_cards[h];
            let adj = cfreach_adj(h1, h2, opp_cards, opp_reach);
            if adj <= MIN_ADJ {
                return 0.0;
            }
            let raw = (c as f64 * scale / adj) as f32;
            raw.clamp(-MAX_BCFV, MAX_BCFV)
        })
        .collect()
}

/// Blocker-adjusted opponent reach sum for a hero hand.
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

/// Remap a parent-ordering reach vector to subtree ordering.
fn remap_reach_to_subtree(
    parent_reach: &[f32],
    parent_private_cards: &[Vec<(u8, u8)>; 2],
    subtree_cards: &[(u8, u8)],
    player: usize,
) -> Vec<f32> {
    let parent_cards = &parent_private_cards[player];
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
    use std::sync::Arc;
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::flop_from_str;
    use range_solver::game::BoundaryEvaluator;
    use range_solver::interface::Game;

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

    // =====================================================================
    // Diagnostic tests for sign-flip hypothesis
    // =====================================================================

    /// Helper: build a PostFlopGame for the same river spot used by
    /// `make_river_evaluator`, solve it, finalize, and return it.
    fn build_and_solve_river_game(
        iters: u32,
    ) -> (PostFlopGame, Vec<u8>, Vec<(u8, u8)>, Vec<(u8, u8)>) {
        let flop = flop_from_str("7h 5d 2c").unwrap();
        let turn_card: u8 = 7; // 3s
        let river_card: u8 = 30; // 9h
        let board = vec![flop[0], flop[1], flop[2], turn_card, river_card];

        let oop_range: Range = "AA,KK,QQ".parse().unwrap();
        let ip_range: Range = "TT,99,88".parse().unwrap();
        let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1 << c));
        let (oop_hands, oop_weights) = oop_range.get_hands_weights(board_mask);
        let (ip_hands, ip_weights) = ip_range.get_hands_weights(board_mask);

        let card_config = build_card_config(&board, &[oop_hands.clone(), ip_hands.clone()], &[oop_weights, ip_weights]);

        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 200,
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
        game.allocate_memory(false);

        run_dcfr(&mut game, iters);
        finalize(&mut game);

        (game, board, oop_hands, ip_hands)
    }

    /// Test A: Zero-sum invariant for raw cfvalues from the subtree solve.
    ///
    /// The raw cfvalues (before cfv_to_bcfv) should satisfy:
    ///   Σ_h cfv_oop[h] * oop_reach[h] + Σ_h cfv_ip[h] * ip_reach[h] ≈ 0
    ///
    /// If this fails, the underlying solve is broken (not cfv_to_bcfv).
    #[test]
    fn diagnostic_zero_sum_raw_cfvalues() {
        let (game, _board, _oop_hands, _ip_hands) = build_and_solve_river_game(500);

        let oop_reach: Vec<f32> = vec![1.0; game.num_private_hands(0)];
        let ip_reach: Vec<f32> = vec![1.0; game.num_private_hands(1)];
        assert!(oop_reach.len() > 0 && ip_reach.len() > 0);

        let oop_cfv = root_cfvalues_with_reach(&game, 0, &ip_reach);
        let ip_cfv = root_cfvalues_with_reach(&game, 1, &oop_reach);

        let oop_ev: f64 = oop_cfv.iter().zip(oop_reach.iter())
            .map(|(&c, &r)| c as f64 * r as f64)
            .sum();
        let ip_ev: f64 = ip_cfv.iter().zip(ip_reach.iter())
            .map(|(&c, &r)| c as f64 * r as f64)
            .sum();

        let sum = oop_ev + ip_ev;

        eprintln!("=== Test A: Raw CFV zero-sum ===");
        eprintln!("  OOP EV (raw cfv sum): {oop_ev:.6}");
        eprintln!("  IP  EV (raw cfv sum): {ip_ev:.6}");
        eprintln!("  Sum (should be ~0):   {sum:.6}");

        // Also check via the library's compute_current_ev
        let [ev0, ev1] = range_solver::compute_current_ev(&game);
        eprintln!("  compute_current_ev:   [{ev0:.6}, {ev1:.6}], sum={:.6}", ev0 as f64 + ev1 as f64);

        assert!(
            sum.abs() < 0.1,
            "Raw cfvalues should be zero-sum, got sum={sum:.6} (oop={oop_ev:.6}, ip={ip_ev:.6})"
        );
    }

    /// Test A2: Zero-sum invariant after cfv_to_bcfv conversion.
    ///
    /// bcfv[h] = cfv[h] * N / (half_pot * cfreach_adj[h])
    ///
    /// The reconstituted value is: cfv_recon[h] = bcfv[h] * half_pot / N * cfreach_adj[h]
    /// This should equal the original cfv[h], meaning the zero-sum property is preserved.
    ///
    /// CRITICAL CHECK: We also verify the SIGN of bcfv values.
    /// OOP has AA,KK,QQ (all overpairs to board 7h5d2c3s9h). These should
    /// all have POSITIVE bcfv (they beat TT,99,88 at showdown most of the time).
    /// IP has TT,99,88 — TT and 88 lose, 99 makes a set on the river.
    #[test]
    fn diagnostic_zero_sum_bcfv_and_sign_check() {
        let (game, _board, _oop_hands, _ip_hands) = build_and_solve_river_game(500);

        let oop_cards = game.private_cards(0).to_vec();
        let ip_cards = game.private_cards(1).to_vec();

        let oop_reach: Vec<f32> = vec![1.0; oop_cards.len()];
        let ip_reach: Vec<f32> = vec![1.0; ip_cards.len()];

        let oop_cfv = root_cfvalues_with_reach(&game, 0, &ip_reach);
        let ip_cfv = root_cfvalues_with_reach(&game, 1, &oop_reach);

        let half_pot = 100.0 / 2.0; // starting_pot = 100
        let num_combos = game.num_combinations();

        eprintln!("=== Test A2: bcfv sign check ===");
        eprintln!("  half_pot={half_pot}, num_combos={num_combos}");

        // Convert to bcfv
        let oop_bcfv = cfv_to_bcfv(
            &oop_cfv, &oop_cards, &ip_cards, &ip_reach,
            half_pot, num_combos,
        );
        let ip_bcfv = cfv_to_bcfv(
            &ip_cfv, &ip_cards, &oop_cards, &oop_reach,
            half_pot, num_combos,
        );

        // Print per-hand details
        eprintln!("  OOP hands (AA,KK,QQ — should be POSITIVE bcfv):");
        for (i, &(c1, c2)) in oop_cards.iter().enumerate() {
            let adj = cfreach_adj(c1, c2, &ip_cards, &ip_reach);
            eprintln!("    hand {i} ({c1},{c2}): cfv={:.6}, bcfv={:.6}, adj={adj:.4}",
                oop_cfv[i], oop_bcfv[i]);
        }
        eprintln!("  IP hands (TT,99,88):");
        for (i, &(c1, c2)) in ip_cards.iter().enumerate() {
            let adj = cfreach_adj(c1, c2, &oop_cards, &oop_reach);
            eprintln!("    hand {i} ({c1},{c2}): cfv={:.6}, bcfv={:.6}, adj={adj:.4}",
                ip_cfv[i], ip_bcfv[i]);
        }

        // Check sign: AA,KK,QQ should have positive bcfv (they dominate TT,88 and
        // only lose to 99 which made a set). Most OOP hands should be positive.
        let oop_positive = oop_bcfv.iter().filter(|&&v| v > 0.0).count();
        let oop_negative = oop_bcfv.iter().filter(|&&v| v < 0.0).count();
        eprintln!("  OOP bcfv: {oop_positive} positive, {oop_negative} negative (of {})", oop_cards.len());

        // Reconstruct cfvalues from bcfv and check round-trip
        let scale = half_pot / num_combos;
        let mut oop_recon_ev = 0.0f64;
        for (h, &(c1, c2)) in oop_cards.iter().enumerate() {
            let adj = cfreach_adj(c1, c2, &ip_cards, &ip_reach);
            let recon = oop_bcfv[h] as f64 * scale * adj;
            let orig = oop_cfv[h] as f64;
            let diff = (recon - orig).abs();
            if diff > 0.01 {
                eprintln!("  WARNING: OOP hand {h} round-trip diff: orig={orig:.6}, recon={recon:.6}, diff={diff:.6}");
            }
            oop_recon_ev += recon * oop_reach[h] as f64;
        }

        let mut ip_recon_ev = 0.0f64;
        for (h, &(c1, c2)) in ip_cards.iter().enumerate() {
            let adj = cfreach_adj(c1, c2, &oop_cards, &oop_reach);
            let recon = ip_bcfv[h] as f64 * scale * adj;
            let orig = ip_cfv[h] as f64;
            let diff = (recon - orig).abs();
            if diff > 0.01 {
                eprintln!("  WARNING: IP hand {h} round-trip diff: orig={orig:.6}, recon={recon:.6}, diff={diff:.6}");
            }
            ip_recon_ev += recon * ip_reach[h] as f64;
        }

        eprintln!("  Reconstructed EVs: oop={oop_recon_ev:.6}, ip={ip_recon_ev:.6}, sum={:.6}", oop_recon_ev + ip_recon_ev);

        // SIGN CHECK: If most OOP bcfv values are negative, we have a sign flip
        assert!(
            oop_positive > oop_negative,
            "SIGN FLIP DETECTED: OOP (AA,KK,QQ overpairs) should have mostly positive bcfv, \
             got {oop_positive} positive vs {oop_negative} negative"
        );
    }

    /// Test A3: Verify that cfv_to_bcfv uses the right cfreach_adj formula.
    ///
    /// Build a trivial case where we know the exact cfv and can verify
    /// the bcfv value algebraically.
    #[test]
    fn diagnostic_cfv_to_bcfv_unit() {
        // Fabricate known values:
        // hero_cards = [(0,1)], opp_cards = [(2,3), (4,5)]
        // opp_reach = [1.0, 0.5]
        // cfreach_adj for (0,1) = 1.0 + 0.5 = 1.5 (no blockers)
        // cfv = [2.0]
        // half_pot = 50.0, num_combos = 10.0
        // Expected: bcfv = cfv * N / (half_pot * adj) = 2.0 * 10.0 / (50.0 * 1.5) = 20/75 = 0.2667

        let hero_cards = vec![(0u8, 1u8)];
        let opp_cards = vec![(2u8, 3u8), (4u8, 5u8)];
        let opp_reach = vec![1.0f32, 0.5f32];
        let cfv = vec![2.0f32];
        let half_pot = 50.0;
        let num_combos = 10.0;

        let bcfv = cfv_to_bcfv(&cfv, &hero_cards, &opp_cards, &opp_reach, half_pot, num_combos);

        let expected = 2.0 * 10.0 / (50.0 * 1.5);
        eprintln!("=== Test A3: cfv_to_bcfv unit ===");
        eprintln!("  cfv=2.0, expected bcfv={expected:.6}, got bcfv={:.6}", bcfv[0]);

        assert!(
            (bcfv[0] as f64 - expected).abs() < 1e-4,
            "cfv_to_bcfv arithmetic wrong: expected {expected:.6}, got {:.6}", bcfv[0]
        );

        // Now verify the round-trip:
        // result = bcfv * half_pot / N * adj = 0.2667 * 50 / 10 * 1.5 = 0.2667 * 7.5 = 2.0
        let adj = cfreach_adj(0, 1, &opp_cards, &opp_reach);
        let roundtrip = bcfv[0] as f64 * half_pot / num_combos * adj;
        eprintln!("  Round-trip: {roundtrip:.6} (should be 2.0)");
        assert!(
            (roundtrip - 2.0).abs() < 1e-4,
            "Round-trip failed: expected 2.0, got {roundtrip:.6}"
        );
    }

    /// Test B: Compare per-hand cfv sign between direct solve and bcfv round-trip.
    ///
    /// For each OOP hand, the cfv from `root_cfvalues_with_reach` should have
    /// the same sign as the bcfv from `cfv_to_bcfv`. If they disagree, the
    /// conversion introduced a sign flip.
    #[test]
    fn diagnostic_per_hand_sign_parity() {
        let (game, _board, _oop_hands, _ip_hands) = build_and_solve_river_game(500);

        let oop_cards = game.private_cards(0).to_vec();
        let ip_cards = game.private_cards(1).to_vec();

        let oop_reach: Vec<f32> = vec![1.0; oop_cards.len()];
        let ip_reach: Vec<f32> = vec![1.0; ip_cards.len()];

        let oop_cfv = root_cfvalues_with_reach(&game, 0, &ip_reach);
        let ip_cfv = root_cfvalues_with_reach(&game, 1, &oop_reach);

        let half_pot = 100.0 / 2.0;
        let num_combos = game.num_combinations();

        let oop_bcfv = cfv_to_bcfv(
            &oop_cfv, &oop_cards, &ip_cards, &ip_reach,
            half_pot, num_combos,
        );
        let ip_bcfv = cfv_to_bcfv(
            &ip_cfv, &ip_cards, &oop_cards, &oop_reach,
            half_pot, num_combos,
        );

        eprintln!("=== Test B: Per-hand sign parity ===");

        let mut sign_mismatches = 0;
        let mut total_checked = 0;

        eprintln!("  OOP hands:");
        for (h, &(c1, c2)) in oop_cards.iter().enumerate() {
            let cfv = oop_cfv[h];
            let bcfv = oop_bcfv[h];
            let sign_match = (cfv >= 0.0) == (bcfv >= 0.0);
            if !sign_match && cfv.abs() > 0.001 && bcfv.abs() > 0.001 {
                sign_mismatches += 1;
                eprintln!("    MISMATCH hand {h} ({c1},{c2}): cfv={cfv:.6}, bcfv={bcfv:.6}");
            } else {
                eprintln!("    OK hand {h} ({c1},{c2}): cfv={cfv:.6}, bcfv={bcfv:.6}");
            }
            if cfv.abs() > 0.001 {
                total_checked += 1;
            }
        }

        eprintln!("  IP hands:");
        for (h, &(c1, c2)) in ip_cards.iter().enumerate() {
            let cfv = ip_cfv[h];
            let bcfv = ip_bcfv[h];
            let sign_match = (cfv >= 0.0) == (bcfv >= 0.0);
            if !sign_match && cfv.abs() > 0.001 && bcfv.abs() > 0.001 {
                sign_mismatches += 1;
                eprintln!("    MISMATCH hand {h} ({c1},{c2}): cfv={cfv:.6}, bcfv={bcfv:.6}");
            } else {
                eprintln!("    OK hand {h} ({c1},{c2}): cfv={cfv:.6}, bcfv={bcfv:.6}");
            }
            if cfv.abs() > 0.001 {
                total_checked += 1;
            }
        }

        eprintln!("  Sign mismatches: {sign_mismatches} / {total_checked}");

        // Also compare with the evaluator's output (which goes through
        // solve_subtree + remap)
        let eval = make_river_evaluator(500);
        let eval_oop_reach = vec![1.0f32; eval.private_cards[0].len()];
        let eval_ip_reach = vec![1.0f32; eval.private_cards[1].len()];
        let (eval_oop_bcfv, eval_ip_bcfv) = eval.compute_cfvs_both(
            100, 150.0, &eval_oop_reach, &eval_ip_reach,
            eval.private_cards[0].len(), eval.private_cards[1].len(), 0,
        );

        eprintln!("  Evaluator bcfv vs direct bcfv (OOP):");
        for (i, &(c1, c2)) in eval.private_cards[0].iter().enumerate() {
            eprintln!("    hand {i} ({c1},{c2}): eval_bcfv={:.6}", eval_oop_bcfv[i]);
        }

        // Overall EV comparison
        let [ev0, ev1] = range_solver::compute_current_ev(&game);
        let oop_bcfv_sum: f64 = oop_bcfv.iter().map(|&v| v as f64).sum();
        let ip_bcfv_sum: f64 = ip_bcfv.iter().map(|&v| v as f64).sum();
        eprintln!("  Full solve EV: OOP={ev0:.6}, IP={ev1:.6}");
        eprintln!("  bcfv sums: OOP={oop_bcfv_sum:.6}, IP={ip_bcfv_sum:.6}");

        assert_eq!(
            sign_mismatches, 0,
            "SIGN FLIP: {sign_mismatches} / {total_checked} hands have cfv/bcfv sign mismatch"
        );
    }

    /// Test C: Check num_combinations mismatch between subtree and parent.
    ///
    /// In cfv_to_bcfv, `scale = num_combos / half_pot` uses the SUBTREE's
    /// num_combinations. But evaluate_boundary_single uses the PARENT's
    /// `self.num_combinations`. If these differ, the bcfv is scaled wrong.
    #[test]
    fn diagnostic_num_combinations_mismatch() {
        let flop = flop_from_str("7h 5d 2c").unwrap();
        let turn_card: u8 = 7; // 3s
        let river_card: u8 = 30; // 9h
        let board = vec![flop[0], flop[1], flop[2], turn_card, river_card];

        let oop_range: Range = "AA,KK,QQ".parse().unwrap();
        let ip_range: Range = "TT,99,88".parse().unwrap();
        let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1 << c));
        let (oop_hands, oop_weights) = oop_range.get_hands_weights(board_mask);
        let (ip_hands, ip_weights) = ip_range.get_hands_weights(board_mask);

        // Build the subtree game (as solve_subtree does)
        let card_config = build_card_config(
            &board,
            &[oop_hands.clone(), ip_hands.clone()],
            &[oop_weights.clone(), ip_weights.clone()],
        );

        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 200,
            river_bet_sizes: [sizes.clone(), sizes.clone()],
            ..Default::default()
        };
        let tree = ActionTree::new(tree_config.clone()).unwrap();
        let game = PostFlopGame::with_config(card_config, tree).unwrap();

        let subtree_num_combos = game.num_combinations();
        let subtree_num_hands = [game.num_private_hands(0), game.num_private_hands(1)];

        // Build a "parent" game with the same config (simulating what the
        // parent solver would see)
        let parent_card_config = build_card_config(
            &board,
            &[oop_hands.clone(), ip_hands.clone()],
            &[oop_weights.clone(), ip_weights.clone()],
        );
        let parent_tree = ActionTree::new(tree_config).unwrap();
        let parent_game = PostFlopGame::with_config(parent_card_config, parent_tree).unwrap();
        let parent_num_combos = parent_game.num_combinations();
        let parent_num_hands = [parent_game.num_private_hands(0), parent_game.num_private_hands(1)];

        eprintln!("=== Test C: num_combinations mismatch ===");
        eprintln!("  Subtree: num_combos={subtree_num_combos}, hands=[{}, {}]",
            subtree_num_hands[0], subtree_num_hands[1]);
        eprintln!("  Parent:  num_combos={parent_num_combos}, hands=[{}, {}]",
            parent_num_hands[0], parent_num_hands[1]);

        let ratio = subtree_num_combos / parent_num_combos;
        eprintln!("  Ratio (subtree/parent): {ratio:.6}");

        // In a real scenario, the parent may have a WIDER range than the
        // subtree (more hands), leading to different num_combinations.
        // The bcfv formula divides by subtree's N, but parent multiplies by
        // parent's half_pot / parent_N. If subtree_N != parent_N, there's a
        // scaling error.
        //
        // For this test, both are built identically so they should match.
        assert!(
            (ratio - 1.0).abs() < 1e-6,
            "num_combinations mismatch: subtree={subtree_num_combos}, parent={parent_num_combos}, ratio={ratio}"
        );
    }

    /// Test D: Turn-boundary spot — the actual scenario where the bug was observed.
    ///
    /// Build a turn-start game with depth_limit=1 (boundary at river start),
    /// and a full turn+river game without boundary. Compare the root cfvalues.
    ///
    /// Uses a wider range to be more realistic: OOP=AA-TT, IP=99-22.
    #[test]
    fn diagnostic_turn_boundary_vs_full_solve() {
        let flop = flop_from_str("7h 5d 2c").unwrap();
        let turn_card: u8 = 7; // 3s
        let board_turn = vec![flop[0], flop[1], flop[2], turn_card];

        let oop_range: Range = "AA,KK,QQ,JJ,TT".parse().unwrap();
        let ip_range: Range = "99,88,77,66,55".parse().unwrap();
        let board_mask: u64 = board_turn.iter().fold(0u64, |m, &c| m | (1 << c));
        let (oop_hands, oop_weights) = oop_range.get_hands_weights(board_mask);
        let (ip_hands, ip_weights) = ip_range.get_hands_weights(board_mask);

        // Build full turn+river game (no boundary)
        let card_config_full = build_card_config(
            &board_turn,
            &[oop_hands.clone(), ip_hands.clone()],
            &[oop_weights.clone(), ip_weights.clone()],
        );
        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let tree_config_full = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 200,
            river_bet_sizes: [sizes.clone(), sizes.clone()],
            depth_limit: None,
            ..Default::default()
        };
        let tree_full = ActionTree::new(tree_config_full).unwrap();
        let mut game_full = PostFlopGame::with_config(card_config_full, tree_full).unwrap();
        game_full.allocate_memory(false);

        run_dcfr(&mut game_full, 500);
        finalize(&mut game_full);

        let full_oop_reach = vec![1.0f32; game_full.num_private_hands(0)];
        let full_ip_reach = vec![1.0f32; game_full.num_private_hands(1)];
        let full_oop_cfv = root_cfvalues_with_reach(&game_full, 0, &full_ip_reach);
        let [full_ev0, full_ev1] = range_solver::compute_current_ev(&game_full);

        eprintln!("=== Test D: Turn boundary vs full solve ===");
        eprintln!("  Full solve EV: OOP={full_ev0:.6}, IP={full_ev1:.6}");
        eprintln!("  Full solve per-hand OOP cfv:");
        let full_oop_cards = game_full.private_cards(0).to_vec();
        for (h, &(c1, c2)) in full_oop_cards.iter().enumerate() {
            eprintln!("    hand {h} ({c1},{c2}): cfv={:.6}", full_oop_cfv[h]);
        }

        // Now build depth-limited game with SubtreeExactEvaluator at river boundary
        let card_config_lim = build_card_config(
            &board_turn,
            &[oop_hands.clone(), ip_hands.clone()],
            &[oop_weights.clone(), ip_weights.clone()],
        );
        let tree_config_lim = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 200,
            river_bet_sizes: [sizes.clone(), sizes.clone()],
            depth_limit: Some(0), // Block turn->river transition: boundary at river
            ..Default::default()
        };
        let tree_lim = ActionTree::new(tree_config_lim.clone()).unwrap();
        let mut game_lim = PostFlopGame::with_config(card_config_lim, tree_lim).unwrap();
        game_lim.allocate_memory(false);

        // Attach per-boundary SubtreeExactEvaluators
        let boundary_count = game_lim.num_boundary_nodes();
        let boundary_boards = game_lim.boundary_boards();
        eprintln!("  Depth-limited game has {boundary_count} boundary nodes");

        // For each boundary, build an evaluator
        let mut per_boundary: Vec<Arc<dyn range_solver::game::BoundaryEvaluator>> = Vec::new();
        for b in 0..boundary_count {
            let eval = SubtreeExactEvaluator::new(
                boundary_boards[b].clone(),
                [oop_hands.clone(), ip_hands.clone()],
                [oop_weights.clone(), ip_weights.clone()],
                tree_config_lim.clone(),
            ).with_solve_iters(500);
            per_boundary.push(Arc::new(eval));
        }
        game_lim.per_boundary_evaluators = per_boundary;

        run_dcfr(&mut game_lim, 500);
        finalize(&mut game_lim);

        let lim_oop_reach = vec![1.0f32; game_lim.num_private_hands(0)];
        let lim_ip_reach = vec![1.0f32; game_lim.num_private_hands(1)];
        let lim_oop_cfv = root_cfvalues_with_reach(&game_lim, 0, &lim_ip_reach);
        let [lim_ev0, lim_ev1] = range_solver::compute_current_ev(&game_lim);

        eprintln!("  Depth-limited solve EV: OOP={lim_ev0:.6}, IP={lim_ev1:.6}");
        eprintln!("  Depth-limited per-hand OOP cfv:");
        let lim_oop_cards = game_lim.private_cards(0).to_vec();
        for (h, &(c1, c2)) in lim_oop_cards.iter().enumerate() {
            eprintln!("    hand {h} ({c1},{c2}): cfv={:.6}", lim_oop_cfv[h]);
        }

        // Compare sign agreement between full and limited
        let mut sign_flips = 0;
        let mut total = 0;
        for h in 0..full_oop_cards.len().min(lim_oop_cards.len()) {
            let fc = full_oop_cfv[h];
            let lc = lim_oop_cfv[h];
            if fc.abs() > 0.01 && lc.abs() > 0.01 {
                total += 1;
                if (fc > 0.0) != (lc > 0.0) {
                    sign_flips += 1;
                    let (c1, c2) = full_oop_cards[h];
                    eprintln!("    SIGN FLIP hand {h} ({c1},{c2}): full={fc:.6}, lim={lc:.6}");
                }
            }
        }
        eprintln!("  Sign flips: {sign_flips} / {total}");
        eprintln!("  EV diff: oop={:.6}, ip={:.6}", (lim_ev0 - full_ev0).abs(), (lim_ev1 - full_ev1).abs());

        // The EVs should be reasonably close (same game, same solver)
        // Tolerance is generous because depth-limited may converge differently
        assert!(
            sign_flips == 0,
            "SIGN FLIP in turn-boundary solve: {sign_flips}/{total} hands have inverted cfv sign"
        );
    }

    /// Test E: num_combinations mismatch between parent turn game and subtree
    /// river games built per-boundary.
    ///
    /// When the parent is a turn game and the subtree is a river game,
    /// the subtree has a specific river card which blocks some hand combos.
    /// This causes N_subtree != N_parent, creating a systematic scaling error.
    #[test]
    fn diagnostic_num_combos_parent_vs_subtree_river() {
        let flop = flop_from_str("7h 5d 2c").unwrap();
        let turn_card: u8 = 7; // 3s
        let board_turn = vec![flop[0], flop[1], flop[2], turn_card];

        // Use a wider range where blockers matter more
        let oop_range: Range = "AA,KK,QQ,JJ,TT".parse().unwrap();
        let ip_range: Range = "99,88,77,66,55".parse().unwrap();
        let board_mask: u64 = board_turn.iter().fold(0u64, |m, &c| m | (1 << c));
        let (oop_hands, oop_weights) = oop_range.get_hands_weights(board_mask);
        let (ip_hands, ip_weights) = ip_range.get_hands_weights(board_mask);

        // Build parent turn game
        let parent_cc = build_card_config(
            &board_turn,
            &[oop_hands.clone(), ip_hands.clone()],
            &[oop_weights.clone(), ip_weights.clone()],
        );
        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let parent_tc = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 200,
            river_bet_sizes: [sizes.clone(), sizes.clone()],
            depth_limit: Some(0),
            ..Default::default()
        };
        let parent_tree = ActionTree::new(parent_tc.clone()).unwrap();
        let parent_game = PostFlopGame::with_config(parent_cc, parent_tree).unwrap();
        let parent_n = parent_game.num_combinations();

        eprintln!("=== Test E: N_parent vs N_subtree ===");
        eprintln!("  Parent num_combinations: {parent_n}");

        // Build subtree river games for a few river cards
        let river_cards: Vec<u8> = (0u8..52)
            .filter(|c| !board_turn.contains(c))
            .take(10)
            .collect();

        for &rc in &river_cards {
            let mut river_board = board_turn.clone();
            river_board.push(rc);
            let subtree_cc = build_card_config(
                &river_board,
                &[oop_hands.clone(), ip_hands.clone()],
                &[oop_weights.clone(), ip_weights.clone()],
            );
            let subtree_tc = TreeConfig {
                initial_state: BoardState::River,
                starting_pot: 100,
                effective_stack: 200,
                river_bet_sizes: [sizes.clone(), sizes.clone()],
                ..Default::default()
            };
            let subtree_tree = ActionTree::new(subtree_tc).unwrap();
            let subtree_game = PostFlopGame::with_config(subtree_cc, subtree_tree).unwrap();
            let subtree_n = subtree_game.num_combinations();
            let ratio = subtree_n / parent_n;
            let hands_oop = subtree_game.num_private_hands(0);
            let hands_ip = subtree_game.num_private_hands(1);
            eprintln!("  River card {rc:2}: N_sub={subtree_n:8.1}, ratio={ratio:.4}, hands=[{hands_oop}, {hands_ip}]");
        }

        // The point: if N_subtree != N_parent, then bcfv * (half_pot/N_parent) * adj
        // != cfv, introducing a scaling error. This is NOT a sign flip.
        // The test always passes — it's purely diagnostic output.

        // Now test with a wider range that WILL be blocked by some river cards.
        let oop_range2: Range = "AA-22".parse().unwrap();
        let ip_range2: Range = "AA-22".parse().unwrap();
        let (oop_hands2, oop_weights2) = oop_range2.get_hands_weights(board_mask);
        let (ip_hands2, ip_weights2) = ip_range2.get_hands_weights(board_mask);

        let parent_cc2 = build_card_config(
            &board_turn,
            &[oop_hands2.clone(), ip_hands2.clone()],
            &[oop_weights2.clone(), ip_weights2.clone()],
        );
        let parent_tc2 = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 200,
            river_bet_sizes: [sizes.clone(), sizes.clone()],
            depth_limit: Some(0),
            ..Default::default()
        };
        let parent_tree2 = ActionTree::new(parent_tc2.clone()).unwrap();
        let parent_game2 = PostFlopGame::with_config(parent_cc2, parent_tree2).unwrap();
        let parent_n2 = parent_game2.num_combinations();
        eprintln!("\n  WIDE RANGE (AA-22 vs AA-22):");
        eprintln!("  Parent num_combinations: {parent_n2}");

        for &rc in &river_cards {
            let mut river_board = board_turn.clone();
            river_board.push(rc);
            let subtree_cc = build_card_config(
                &river_board,
                &[oop_hands2.clone(), ip_hands2.clone()],
                &[oop_weights2.clone(), ip_weights2.clone()],
            );
            let subtree_tc = TreeConfig {
                initial_state: BoardState::River,
                starting_pot: 100,
                effective_stack: 200,
                river_bet_sizes: [sizes.clone(), sizes.clone()],
                ..Default::default()
            };
            let subtree_tree = ActionTree::new(subtree_tc).unwrap();
            let subtree_game = PostFlopGame::with_config(subtree_cc, subtree_tree).unwrap();
            let subtree_n = subtree_game.num_combinations();
            let ratio = subtree_n / parent_n2;
            let hands_oop = subtree_game.num_private_hands(0);
            let hands_ip = subtree_game.num_private_hands(1);
            eprintln!("  River card {rc:2}: N_sub={subtree_n:8.1}, ratio={ratio:.4}, hands=[{hands_oop}, {hands_ip}]");
        }
    }
}
