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
use range_solver::{solve_step, finalize, root_cfvalues, BoardState, PostFlopGame};
use range_solver::interface::Game;

/// Default number of DCFR iterations for the exact subtree solve.
const DEFAULT_SOLVE_ITERS: u32 = 500;

/// Boundary evaluator that solves the downstream subtree exactly via DCFR.
pub struct SubtreeExactEvaluator {
    /// Board cards at this boundary (3, 4, or 5 cards).
    board: Vec<u8>,
    /// Private card lists per player, aligned with parent game ordering.
    private_cards: [Vec<(u8, u8)>; 2],
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
        parent_tree_config: TreeConfig,
    ) -> Self {
        Self {
            board,
            private_cards,
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

/// Normalise a reach vector into `[0, 1]`.
///
/// The caller's reach may have values > 1.0 because counterfactual-reach in
/// the parent solver is divided by chance factors (e.g. 1/48 for a river
/// card) which inflates magnitudes above 1. `Range::is_valid` rejects
/// anything outside `[0, 1]`, so we rescale by max. NaN / negative entries
/// collapse to 0. If the max is <= 0 the vector is zero-filled (caller's
/// zero-reach short-circuit should have caught this upstream).
fn normalise_reach(reach: &[f32]) -> Vec<f32> {
    let max = reach
        .iter()
        .filter(|v| v.is_finite())
        .fold(0.0f32, |acc, &v| acc.max(v.max(0.0)));
    if max <= 0.0 {
        return vec![0.0; reach.len()];
    }
    reach
        .iter()
        .map(|&v| {
            if !v.is_finite() || v <= 0.0 {
                0.0
            } else {
                (v / max).min(1.0)
            }
        })
        .collect()
}

/// Build a CardConfig with per-hand reach weights as initial weights.
fn build_card_config(
    board: &[u8],
    private_cards: &[Vec<(u8, u8)>; 2],
    oop_reach: &[f32],
    ip_reach: &[f32],
) -> CardConfig {
    let flop = [board[0], board[1], board[2]];
    let turn = if board.len() > 3 { board[3] } else { NOT_DEALT };
    let river = if board.len() > 4 { board[4] } else { NOT_DEALT };

    let oop_norm = normalise_reach(oop_reach);
    let ip_norm = normalise_reach(ip_reach);

    // Build Range objects with per-combo weights matching normalised reach.
    let mut ranges = [Range::new(), Range::new()];
    for (player, reach) in [&oop_norm, &ip_norm].iter().enumerate() {
        for (i, &(c1, c2)) in private_cards[player].iter().enumerate() {
            let w = if i < reach.len() { reach[i] } else { 0.0 };
            ranges[player].set_weight_by_cards(c1, c2, w);
        }
    }

    CardConfig {
        range: ranges,
        flop,
        turn,
        river,
    }
}

/// True when every element is (near-)zero. Used to short-circuit unreachable
/// boundaries — if a player has zero reach across all combos, the boundary
/// isn't reachable and we can safely return zero CFVs without solving.
fn reach_is_all_zero(reach: &[f32]) -> bool {
    reach.iter().all(|&v| v.abs() < 1e-9)
}

/// Build the subtree game, run DCFR, finalize, and extract root CFVs.
///
/// `pot` and `remaining_stack` come from the boundary caller — these are the
/// pot and remaining stack AT the boundary, which is what the subtree needs
/// as its `starting_pot` / `effective_stack` (NOT the parent's values, which
/// are pre-action).
///
/// Returns CFVs in the PARENT's `private_cards` ordering.  The subtree game
/// may have fewer hands (zero-reach combos are dropped from the Range), so
/// we remap back after extraction.
fn solve_subtree(
    board: &[u8],
    private_cards: &[Vec<(u8, u8)>; 2],
    parent_tree_config: &TreeConfig,
    pot: i32,
    remaining_stack: f64,
    oop_reach: &[f32],
    ip_reach: &[f32],
    solve_iters: u32,
) -> (Vec<f32>, Vec<f32>) {
    // Short-circuit: if EITHER side has zero reach everywhere, this boundary
    // is unreachable under the current strategy. Return zero CFVs — they get
    // multiplied by the zero reach downstream anyway. Avoids "range is empty"
    // panic when Range::new() gets nothing but zero weights.
    if reach_is_all_zero(oop_reach) || reach_is_all_zero(ip_reach) {
        return (vec![0.0; oop_reach.len()], vec![0.0; ip_reach.len()]);
    }

    let card_config = build_card_config(board, private_cards, oop_reach, ip_reach);
    let initial_state = board_state_from_len(board.len());

    // effective_stack = pot/2 (chips each player has committed) + remaining
    // stack. This is the convention used by the parent solver (see
    // evaluation.rs where boundary pot + 2 * remaining = original stack).
    let effective_stack = (pot / 2).saturating_add(remaining_stack.round() as i32);

    // The subtree represents the DOWNSTREAM game starting from the boundary.
    // For a turn→river boundary the initial_state is Turn (required by
    // PostFlopGame when river is NOT_DEALT), but the turn action layer must
    // be empty (forced check-check) because the parent already handled turn
    // decisions.  Only river_bet_sizes matter for the actual subtree play.
    // Flop sizes are similarly irrelevant for turn or river subtrees.
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

    let (sub_oop, sub_ip) = extract_root_cfvs(&mut game);

    // Remap from subtree ordering to parent ordering. The subtree may have
    // excluded hands with zero reach, producing shorter vectors.
    let oop_out = remap_cfvs_to_parent(
        &sub_oop, game.private_cards(0), &private_cards[0],
    );
    let ip_out = remap_cfvs_to_parent(
        &sub_ip, game.private_cards(1), &private_cards[1],
    );
    (oop_out, ip_out)
}

/// Remap CFVs from subtree hand ordering to parent hand ordering.
///
/// The subtree may have a subset of the parent's hands (zero-reach combos
/// excluded).  Build a card-pair lookup from subtree → CFV, then iterate
/// over parent hands to produce the output vector.  Parent hands absent
/// from the subtree get CFV = 0.
fn remap_cfvs_to_parent(
    subtree_cfvs: &[f32],
    subtree_cards: &[(u8, u8)],
    parent_cards: &[(u8, u8)],
) -> Vec<f32> {
    // Build map: (min_card, max_card) → cfv
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

/// Extract per-hand CFVs at the root for both players in pot-normalised
/// units (1.0 = win one half-pot).
///
/// Derivation: `root_cfvalues(player)` returns per-hand CFVs in the solver's
/// internal "per-combination" form. `expected_values_detail` shows the exact
/// scaling to chip EV for the current player:
///
///     chip_ev[h] = cfv[h] * normalizer * (w_raw/w_normalized) + half_pot
///
/// where `normalizer = num_combinations * chance_factor` (chance_factor = 1
/// at root). Pot-normalising via `(chip_ev - half_pot) / half_pot`:
///
///     pot_norm_cfv[h] = cfv[h] * num_combinations * (w_raw/w_normalized)
///                       / half_pot
///
/// We apply this formula for BOTH players. Unlike `expected_values(player)`
/// which reads from `cfvalues_cache` for the non-current player (empty at
/// root without navigation), `root_cfvalues` re-runs `compute_cfvalue_recursive`
/// from the root on demand — always correct.
fn extract_root_cfvs(game: &mut PostFlopGame) -> (Vec<f32>, Vec<f32>) {
    game.back_to_root();
    game.cache_normalized_weights();
    let half_pot = game.tree_config().starting_pot as f32 / 2.0;
    assert!(half_pot > 0.0, "subtree starting_pot must be positive");
    let num_combos = game.num_combinations() as f32;

    let scale_player = |player: usize, cfv_per_combo: Vec<f32>| -> Vec<f32> {
        let w_raw = game.weights(player);
        let w_norm = game.normalized_weights(player);
        cfv_per_combo
            .iter()
            .enumerate()
            .map(|(h, &c)| {
                let r = w_raw.get(h).copied().unwrap_or(0.0);
                let n = w_norm.get(h).copied().unwrap_or(0.0);
                // Blocker-zeroed or unreachable hands → report 0 CFV.
                if n <= 0.0 || r <= 0.0 {
                    return 0.0;
                }
                // Match expected_values_detail: cfv * normalizer * (w_raw/w_norm).
                // pot-normalise: divide by half_pot.
                let chip_cfv_centered = c * num_combos * (r / n);
                chip_cfv_centered / half_pot
            })
            .collect()
    };

    let oop_raw = root_cfvalues(game, 0);
    let ip_raw = root_cfvalues(game, 1);
    (scale_player(0, oop_raw), scale_player(1, ip_raw))
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
        // Delegate to compute_cfvs_both and return the requested player.
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
        // 3s = rank 1, suit 3 => card index = 1*4 + 3 = 7
        let turn_card: u8 = 7; // 3s
        // 9h = rank 7, suit 2 => card index = 7*4 + 2 = 30
        let river_card: u8 = 30; // 9h
        let board = vec![flop[0], flop[1], flop[2], turn_card, river_card];

        let oop_range: Range = "AA,KK,QQ".parse().unwrap();
        let ip_range: Range = "TT,99,88".parse().unwrap();
        let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1 << c));
        let (oop_hands, _) = oop_range.get_hands_weights(board_mask);
        let (ip_hands, _) = ip_range.get_hands_weights(board_mask);

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
            [oop_hands, ip_hands],
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
        // OOP has overpairs (AA,KK,QQ) vs IP underpairs+set (TT,99,88).
        // On 7h5d2c3s9h: OOP overpairs beat TT,88; 99 has a set and beats OOP.
        // Expected: most OOP hands > 0 (they're ahead of most IP range),
        // most IP hands < 0 (they fold), but 99 hands are strongly positive.
        let eval = make_river_evaluator(200);
        let num_oop = eval.private_cards[0].len();
        let num_ip = eval.private_cards[1].len();
        let oop_reach = vec![1.0f32; num_oop];
        let ip_reach = vec![1.0f32; num_ip];
        let (oop_cfvs, ip_cfvs) = eval.compute_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach, num_oop, num_ip, 0,
        );

        // OOP should have positive EVs (ahead of most IP hands)
        let oop_positive = oop_cfvs.iter().filter(|&&v| v > 0.0).count();
        assert!(
            oop_positive > num_oop / 2,
            "majority of OOP hands should have positive CFV, got {oop_positive}/{num_oop}"
        );

        // IP should have some negative EVs (losing hands) and some positive (99 = set)
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

        // Halve one player's reach
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
        let c = vec![0.0f32, 1.0, 0.668]; // differs in 3rd decimal
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
        // compute_cfvs for OOP uses ip_reach as opponent_reach
        // and builds dummy oop_reach of all 1.0 — which matches
        // our full oop_reach. So results should match compute_cfvs_both.
        let (oop_both, _) = eval.compute_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach, num_oop, num_ip, 0,
        );
        assert_eq!(oop_single, oop_both, "single-player should match both");
    }
}
