//! Exact subtree boundary evaluator.
//!
//! At each boundary, builds a fresh PostFlopGame for the downstream subtree,
//! runs full DCFR to convergence, and returns per-hand CFVs. This is a
//! diagnostic/eval tool — performance is not a concern, correctness is paramount.

use std::collections::HashMap;
use std::sync::Mutex;

use range_solver::action_tree::{ActionTree, TreeConfig};
use range_solver::card::{CardConfig, NOT_DEALT};
use range_solver::interface::Game;
use range_solver::range::Range;
use range_solver::{solve_step, finalize, BoardState, PostFlopGame};

/// Default number of DCFR iterations for the exact subtree solve.
const DEFAULT_SOLVE_ITERS: u32 = 500;

/// Boundary evaluator that solves the downstream subtree exactly via DCFR.
/// A solved river subtree game, ready for cfvalue queries with any reach.
struct SolvedRiverGame {
    /// The solved PostFlopGame (strategies frozen after finalize).
    game: PostFlopGame,
    /// The 5-card board for this river runout.
    board_5: [u8; 5],
}

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
    /// Cache keyed by rounded reach digest (legacy bcfv path).
    cache: Mutex<HashMap<u64, (Vec<f32>, Vec<f32>)>>,
    /// Lazily-built solved subtree games for the raw CFV path.
    /// For 5-card boards: one game. For 4-card boards: up to 48 games.
    solved_games: Mutex<Vec<SolvedRiverGame>>,
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
            solved_games: Mutex::new(Vec::new()),
        }
    }

    /// Set the number of DCFR iterations (for testing).
    pub fn with_solve_iters(mut self, iters: u32) -> Self {
        self.solve_iters = iters;
        self
    }

    /// Lazily build and cache the solved subtree games.
    fn ensure_solved_games(&self, pot: i32, remaining_stack: f64) {
        let mut games = self.solved_games.lock().expect("solved_games lock");
        if !games.is_empty() {
            return;
        }
        match self.board.len() {
            5 => {
                let board_5 = [
                    self.board[0], self.board[1], self.board[2],
                    self.board[3], self.board[4],
                ];
                let sg = build_solved_river_game(
                    &board_5, &self.private_cards,
                    &self.parent_initial_weights,
                    &self.parent_tree_config,
                    pot, remaining_stack, self.solve_iters,
                );
                games.push(sg);
            }
            4 => {
                for river in 0..52u8 {
                    if card_on_board(river, &self.board) { continue; }
                    let board_5 = [
                        self.board[0], self.board[1], self.board[2],
                        self.board[3], river,
                    ];
                    // Zero out blocked hands in weights.
                    let mut weights = self.parent_initial_weights.clone();
                    for player in 0..2 {
                        for (i, &(c1, c2)) in
                            self.private_cards[player].iter().enumerate()
                        {
                            if card_blocks_hand(river, c1, c2) {
                                weights[player][i] = 0.0;
                            }
                        }
                    }
                    let oop_live: f32 = weights[0].iter().sum();
                    let ip_live: f32 = weights[1].iter().sum();
                    if oop_live <= 0.0 || ip_live <= 0.0 { continue; }

                    let sg = build_solved_river_game(
                        &board_5, &self.private_cards, &weights,
                        &self.parent_tree_config,
                        pot, remaining_stack, self.solve_iters,
                    );
                    games.push(sg);
                }
                eprintln!(
                    "[exact_subtree] built {} solved river games for turn boundary",
                    games.len()
                );
            }
            _ => panic!("raw CFV path not supported for board len {}", self.board.len()),
        }
    }

    /// Compute raw per-hand chip CFVs using solved subtree games.
    ///
    /// For each solved river game, calls `root_cfvalues_with_reach` with the
    /// given opponent reach (adjusted for blockers), then aggregates across
    /// river runouts. Returns values in the same per-combo units that
    /// `evaluate_internal` writes to its result array.
    fn compute_raw_from_solved_games(
        &self,
        oop_reach: &[f32],
        ip_reach: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let num_oop = self.private_cards[0].len();
        let num_ip = self.private_cards[1].len();
        let mut oop_sum = vec![0.0f64; num_oop];
        let mut ip_sum = vec![0.0f64; num_ip];
        let mut oop_cnt = vec![0u32; num_oop];
        let mut ip_cnt = vec![0u32; num_ip];

        let games = self.solved_games.lock().expect("solved_games lock");
        for sg in games.iter() {
            let river_card = sg.board_5[4];
            let sub_oop_cards = sg.game.private_cards(0);
            let sub_ip_cards = sg.game.private_cards(1);

            // Build reach for this river: zero out blocked hands, remap
            // from parent ordering to subtree ordering.
            let ip_reach_sub = remap_reach_to_subtree(
                ip_reach, &self.private_cards[1], sub_ip_cards, river_card,
            );
            let oop_reach_sub = remap_reach_to_subtree(
                oop_reach, &self.private_cards[0], sub_oop_cards, river_card,
            );

            // Get per-hand cfvalues with the given opponent reach.
            let oop_cfv = range_solver::root_cfvalues_with_reach(
                &sg.game, 0, &ip_reach_sub,
            );
            let ip_cfv = range_solver::root_cfvalues_with_reach(
                &sg.game, 1, &oop_reach_sub,
            );

            // Remap back to parent ordering and accumulate.
            let oop_parent = remap_cfvs_to_parent(
                &oop_cfv, sub_oop_cards, &self.private_cards[0],
            );
            let ip_parent = remap_cfvs_to_parent(
                &ip_cfv, sub_ip_cards, &self.private_cards[1],
            );

            for (h, &(c1, c2)) in self.private_cards[0].iter().enumerate() {
                if !card_blocks_hand(river_card, c1, c2) {
                    oop_sum[h] += oop_parent[h] as f64;
                    oop_cnt[h] += 1;
                }
            }
            for (h, &(c1, c2)) in self.private_cards[1].iter().enumerate() {
                if !card_blocks_hand(river_card, c1, c2) {
                    ip_sum[h] += ip_parent[h] as f64;
                    ip_cnt[h] += 1;
                }
            }
        }

        let oop_raw: Vec<f32> = oop_sum.iter().zip(oop_cnt.iter())
            .map(|(&s, &c)| if c > 0 { (s / c as f64) as f32 } else { 0.0 })
            .collect();
        let ip_raw: Vec<f32> = ip_sum.iter().zip(ip_cnt.iter())
            .map(|(&s, &c)| if c > 0 { (s / c as f64) as f32 } else { 0.0 })
            .collect();

        (oop_raw, ip_raw)
    }
}

/// Build and solve a single 5-card River game, returning the solved game
/// object (strategies frozen). Used by the raw CFV path.
fn build_solved_river_game(
    board_5: &[u8; 5],
    private_cards: &[Vec<(u8, u8)>; 2],
    weights: &[Vec<f32>; 2],
    parent_tree_config: &TreeConfig,
    pot: i32,
    remaining_stack: f64,
    solve_iters: u32,
) -> SolvedRiverGame {
    let card_config = build_card_config(board_5, private_cards, weights);
    let effective_stack =
        (pot / 2).saturating_add(remaining_stack.round() as i32);

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
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

    SolvedRiverGame { game, board_5: *board_5 }
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

/// Blocker-adjusted opponent reach sum for a hero hand.
fn cfreach_adj(
    h1: u8, h2: u8, opp_cards: &[(u8, u8)], opp_reach: &[f32],
) -> f64 {
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

/// Solve a single 5-card River game and return per-hand pot-normalised bcfv,
/// remapped to parent hand ordering.
///
/// Uses `root_cfvalues` to extract counterfactual values, then converts to
/// bcfv via: `bcfv[h] = cfv[h] * N_sub / (half_pot * cfreach_adj[h])`.
/// This mirrors the SPR=0 formula: `bcfv = (weighted_eq - 0.5) * 2`.
fn solve_river_game(
    board_5: &[u8; 5],
    private_cards: &[Vec<(u8, u8)>; 2],
    weights: &[Vec<f32>; 2],
    parent_tree_config: &TreeConfig,
    pot: i32,
    remaining_stack: f64,
    solve_iters: u32,
) -> (Vec<f32>, Vec<f32>) {
    let card_config = build_card_config(
        board_5, private_cards, weights,
    );
    let effective_stack =
        (pot / 2).saturating_add(remaining_stack.round() as i32);

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
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

    let half_pot = pot as f64 / 2.0;
    let n_sub = game.num_combinations();
    let scale = n_sub / half_pot;

    // Compute per-hand bcfv using root_cfvalues and cfv_to_bcfv formula.
    let oop_cfv = range_solver::root_cfvalues(&game, 0);
    let ip_cfv = range_solver::root_cfvalues(&game, 1);

    let sub_oop_cards = game.private_cards(0);
    let sub_ip_cards = game.private_cards(1);
    let sub_oop_weights = game.initial_weights(0);
    let sub_ip_weights = game.initial_weights(1);

    let oop_sub: Vec<f32> = oop_cfv
        .iter()
        .enumerate()
        .map(|(h, &c)| {
            if c == 0.0 { return 0.0; }
            let (h1, h2) = sub_oop_cards[h];
            let adj = cfreach_adj(h1, h2, sub_ip_cards, sub_ip_weights);
            if adj <= 1e-9 { return 0.0; }
            (c as f64 * scale / adj) as f32
        })
        .collect();

    let ip_sub: Vec<f32> = ip_cfv
        .iter()
        .enumerate()
        .map(|(h, &c)| {
            if c == 0.0 { return 0.0; }
            let (h1, h2) = sub_ip_cards[h];
            let adj = cfreach_adj(h1, h2, sub_oop_cards, sub_oop_weights);
            if adj <= 1e-9 { return 0.0; }
            (c as f64 * scale / adj) as f32
        })
        .collect();

    // Remap from subtree ordering to parent ordering.
    let oop_bcfv = remap_cfvs_to_parent(
        &oop_sub, sub_oop_cards, &private_cards[0],
    );
    let ip_bcfv = remap_cfvs_to_parent(
        &ip_sub, sub_ip_cards, &private_cards[1],
    );

    (oop_bcfv, ip_bcfv)
}

/// Check whether card `c` conflicts with any card in `board`.
fn card_on_board(c: u8, board: &[u8]) -> bool {
    board.iter().any(|&b| b == c)
}

/// Check whether card `c` conflicts with hand `(h1, h2)`.
fn card_blocks_hand(c: u8, h1: u8, h2: u8) -> bool {
    c == h1 || c == h2
}

/// Enumerate valid river cards for a turn boundary and aggregate per-hand
/// bcfv across all river runouts.
///
/// For each valid river card, builds a separate 5-card River game, solves it,
/// and extracts per-hand pot-normalised bcfv. The final bcfv for each hand is
/// the mean across all rivers that do not block that hand.
fn solve_turn_via_river_enum(
    board_4: &[u8],
    private_cards: &[Vec<(u8, u8)>; 2],
    parent_weights: &[Vec<f32>; 2],
    parent_tree_config: &TreeConfig,
    pot: i32,
    remaining_stack: f64,
    solve_iters: u32,
) -> (Vec<f32>, Vec<f32>) {
    let num_oop = private_cards[0].len();
    let num_ip = private_cards[1].len();
    let mut oop_sum = vec![0.0f64; num_oop];
    let mut ip_sum = vec![0.0f64; num_ip];
    let mut oop_cnt = vec![0u32; num_oop];
    let mut ip_cnt = vec![0u32; num_ip];

    let mut rivers_solved = 0u32;
    for river in 0..52u8 {
        if card_on_board(river, board_4) {
            continue;
        }
        let board_5: [u8; 5] = [
            board_4[0], board_4[1], board_4[2], board_4[3], river,
        ];

        // Build weights that zero out hands blocked by the river card.
        let mut river_weights = parent_weights.clone();
        for player in 0..2 {
            for (i, &(c1, c2)) in private_cards[player].iter().enumerate() {
                if card_blocks_hand(river, c1, c2) {
                    river_weights[player][i] = 0.0;
                }
            }
        }

        // Skip if either player has no remaining hands.
        let oop_live: f32 = river_weights[0].iter().sum();
        let ip_live: f32 = river_weights[1].iter().sum();
        if oop_live <= 0.0 || ip_live <= 0.0 {
            continue;
        }

        let (oop_bcfv, ip_bcfv) = solve_river_game(
            &board_5,
            private_cards,
            &river_weights,
            parent_tree_config,
            pot,
            remaining_stack,
            solve_iters,
        );

        // Accumulate — only for hands not blocked by this river card.
        // solve_river_game already remapped to parent ordering.
        for (h, &(c1, c2)) in private_cards[0].iter().enumerate() {
            if !card_blocks_hand(river, c1, c2) {
                oop_sum[h] += oop_bcfv[h] as f64;
                oop_cnt[h] += 1;
            }
        }
        for (h, &(c1, c2)) in private_cards[1].iter().enumerate() {
            if !card_blocks_hand(river, c1, c2) {
                ip_sum[h] += ip_bcfv[h] as f64;
                ip_cnt[h] += 1;
            }
        }
        rivers_solved += 1;
    }

    eprintln!(
        "[exact_subtree] turn boundary: solved {rivers_solved} river subgames"
    );

    let oop_bcfv: Vec<f32> = oop_sum
        .iter()
        .zip(oop_cnt.iter())
        .map(|(&s, &c)| if c > 0 { (s / c as f64) as f32 } else { 0.0 })
        .collect();
    let ip_bcfv: Vec<f32> = ip_sum
        .iter()
        .zip(ip_cnt.iter())
        .map(|(&s, &c)| if c > 0 { (s / c as f64) as f32 } else { 0.0 })
        .collect();

    (oop_bcfv, ip_bcfv)
}

/// Solve the downstream subtree and return per-hand pot-normalised bcfv.
///
/// For 5-card boards (river boundary): solves a single River game.
/// For 4-card boards (turn boundary): enumerates each valid river card,
/// solves each as a separate River game, and averages bcfv across runouts.
/// For 3-card boards (flop boundary): enumerates turn+river combos (slow).
///
/// Uses `oop_reach` / `ip_reach` (the actual boundary reach from the parent
/// solver) as initial weights for the per-river games. This ensures
/// `expected_values` computes EVs against the actual opponent distribution,
/// matching the SPR=0 evaluator's reach-weighted equity approach.
fn solve_subtree(
    board: &[u8],
    private_cards: &[Vec<(u8, u8)>; 2],
    _parent_weights: &[Vec<f32>; 2],
    parent_tree_config: &TreeConfig,
    pot: i32,
    remaining_stack: f64,
    oop_reach: &[f32],
    ip_reach: &[f32],
    solve_iters: u32,
) -> (Vec<f32>, Vec<f32>) {
    // Use boundary reach as initial weights so root_cfvalues computes
    // cfvalues weighted by the actual opponent distribution at the boundary.
    let weights = [oop_reach.to_vec(), ip_reach.to_vec()];

    match board.len() {
        5 => {
            let board_5: [u8; 5] = [
                board[0], board[1], board[2], board[3], board[4],
            ];
            solve_river_game(
                &board_5, private_cards, &weights,
                parent_tree_config, pot, remaining_stack, solve_iters,
            )
        }
        4 => solve_turn_via_river_enum(
            board, private_cards, &weights,
            parent_tree_config, pot, remaining_stack, solve_iters,
        ),
        3 => {
            eprintln!(
                "[exact_subtree] WARNING: flop boundary requires ~2256 \
                 subgames — this will be very slow"
            );
            let num_oop = private_cards[0].len();
            let num_ip = private_cards[1].len();
            let mut oop_sum = vec![0.0f64; num_oop];
            let mut ip_sum = vec![0.0f64; num_ip];
            let mut oop_cnt = vec![0u32; num_oop];
            let mut ip_cnt = vec![0u32; num_ip];

            for turn in 0..52u8 {
                if card_on_board(turn, board) { continue; }
                let board_4 = [board[0], board[1], board[2], turn];
                let (oop_turn, ip_turn) = solve_turn_via_river_enum(
                    &board_4, private_cards, &weights,
                    parent_tree_config, pot, remaining_stack, solve_iters,
                );
                for (h, &(c1, c2)) in private_cards[0].iter().enumerate() {
                    if !card_blocks_hand(turn, c1, c2) {
                        oop_sum[h] += oop_turn[h] as f64;
                        oop_cnt[h] += 1;
                    }
                }
                for (h, &(c1, c2)) in private_cards[1].iter().enumerate() {
                    if !card_blocks_hand(turn, c1, c2) {
                        ip_sum[h] += ip_turn[h] as f64;
                        ip_cnt[h] += 1;
                    }
                }
            }

            let oop_bcfv: Vec<f32> = oop_sum.iter().zip(oop_cnt.iter())
                .map(|(&s, &c)| if c > 0 { (s / c as f64) as f32 } else { 0.0 })
                .collect();
            let ip_bcfv: Vec<f32> = ip_sum.iter().zip(ip_cnt.iter())
                .map(|(&s, &c)| if c > 0 { (s / c as f64) as f32 } else { 0.0 })
                .collect();

            (oop_bcfv, ip_bcfv)
        }
        _ => panic!("invalid board length for subtree: {}", board.len()),
    }
}

/// Remap reach from parent hand ordering to subtree hand ordering,
/// zeroing out hands blocked by a river card.
fn remap_reach_to_subtree(
    parent_reach: &[f32],
    parent_cards: &[(u8, u8)],
    subtree_cards: &[(u8, u8)],
    river_card: u8,
) -> Vec<f32> {
    let mut map: HashMap<(u8, u8), f32> = HashMap::with_capacity(parent_cards.len());
    for (i, &(c1, c2)) in parent_cards.iter().enumerate() {
        let key = if c1 <= c2 { (c1, c2) } else { (c2, c1) };
        let r = parent_reach.get(i).copied().unwrap_or(0.0);
        if card_blocks_hand(river_card, c1, c2) {
            map.insert(key, 0.0);
        } else {
            map.insert(key, r);
        }
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

    fn compute_raw_cfvs_both(
        &self,
        pot: i32,
        remaining_stack: f64,
        oop_reach: &[f32],
        ip_reach: &[f32],
        _num_oop: usize,
        _num_ip: usize,
        _continuation_index: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        // Only support river and turn boundaries.
        if self.board.len() < 4 {
            return None;
        }
        self.ensure_solved_games(pot, remaining_stack);
        Some(self.compute_raw_from_solved_games(oop_reach, ip_reach))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::flop_from_str;
    use range_solver::game::BoundaryEvaluator;
    use range_solver::interface::Game;

    /// Determine the `BoardState` from board length (test-only helper).
    fn board_state_from_len(n: usize) -> BoardState {
        match n {
            3 => BoardState::Flop,
            4 => BoardState::Turn,
            5 => BoardState::River,
            _ => panic!("invalid board length for subtree: {n}"),
        }
    }

    /// Blocker-adjusted opponent reach sum for a hero hand (test-only helper).
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
        // With boundary reach as initial weights, bcfv depends on the
        // opponent reach distribution. This matches SPR=0's behavior
        // where equity is weighted by opponent reach.
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

    /// Helper: build a SubtreeExactEvaluator for a TURN-boundary spot.
    /// AA,KK,QQ vs TT,99,88 on 7h 5d 2c 3s (4-card board, no river yet).
    fn make_turn_evaluator(solve_iters: u32) -> SubtreeExactEvaluator {
        let flop = flop_from_str("7h 5d 2c").unwrap();
        let turn_card: u8 = 7; // 3s
        let board = vec![flop[0], flop[1], flop[2], turn_card];

        let oop_range: Range = "AA,KK,QQ".parse().unwrap();
        let ip_range: Range = "TT,99,88".parse().unwrap();
        let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1 << c));
        let (oop_hands, oop_weights) = oop_range.get_hands_weights(board_mask);
        let (ip_hands, ip_weights) = ip_range.get_hands_weights(board_mask);

        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 200,
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };

        SubtreeExactEvaluator::new(
            board,
            [oop_hands, ip_hands],
            [oop_weights.clone(), ip_weights.clone()],
            tree_config,
        ).with_solve_iters(solve_iters)
    }

    #[test]
    fn turn_evaluator_returns_correct_length_cfvs() {
        let eval = make_turn_evaluator(50);
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
    fn turn_evaluator_cfvs_are_finite_and_bounded() {
        let eval = make_turn_evaluator(100);
        let num_oop = eval.private_cards[0].len();
        let num_ip = eval.private_cards[1].len();
        let oop_reach = vec![1.0f32; num_oop];
        let ip_reach = vec![1.0f32; num_ip];
        let (oop_cfvs, ip_cfvs) = eval.compute_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach, num_oop, num_ip, 0,
        );
        for &v in &oop_cfvs {
            assert!(v.is_finite(), "OOP CFV should be finite, got {v}");
            assert!(
                v.abs() <= 5.0,
                "OOP bcfv should be bounded (pot-normalised), got {v}"
            );
        }
        for &v in &ip_cfvs {
            assert!(v.is_finite(), "IP CFV should be finite, got {v}");
            assert!(
                v.abs() <= 5.0,
                "IP bcfv should be bounded (pot-normalised), got {v}"
            );
        }
    }

    #[test]
    fn turn_evaluator_overpairs_have_positive_bcfv() {
        // AA,KK,QQ are all overpairs on 7h5d2c3s. They should average
        // positive bcfv against TT,99,88 across all river runouts.
        let eval = make_turn_evaluator(200);
        let num_oop = eval.private_cards[0].len();
        let num_ip = eval.private_cards[1].len();
        let oop_reach = vec![1.0f32; num_oop];
        let ip_reach = vec![1.0f32; num_ip];
        let (oop_cfvs, _ip_cfvs) = eval.compute_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach, num_oop, num_ip, 0,
        );
        let oop_positive = oop_cfvs.iter().filter(|&&v| v > 0.0).count();
        assert!(
            oop_positive > num_oop / 2,
            "majority of OOP overpairs should have positive bcfv, \
             got {oop_positive}/{num_oop}, values: {:?}",
            &oop_cfvs[..5.min(oop_cfvs.len())]
        );
    }

    /// Diagnostic: compare per-river bcfv reconstruction against full
    /// Turn+River solve to understand scaling. Print ratios.
    #[test]
    fn turn_evaluator_bcfv_scaling_diagnostic() {
        let flop = flop_from_str("7h 5d 2c").unwrap();
        let turn_card: u8 = 7; // 3s
        let board = vec![flop[0], flop[1], flop[2], turn_card];

        let oop_range: Range = "AA,KK,QQ".parse().unwrap();
        let ip_range: Range = "TT,99,88".parse().unwrap();
        let board_mask: u64 = board.iter().fold(0u64, |m, &c| m | (1 << c));
        let (oop_hands, oop_weights) = oop_range.get_hands_weights(board_mask);
        let (ip_hands, ip_weights) = ip_range.get_hands_weights(board_mask);

        let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 200,
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };

        // Build a full Turn+River game as reference
        let card_config = build_card_config(
            &board,
            &[oop_hands.clone(), ip_hands.clone()],
            &[oop_weights.clone(), ip_weights.clone()],
        );
        let ref_tree = ActionTree::new(tree_config.clone()).unwrap();
        let mut ref_game =
            PostFlopGame::with_config(card_config, ref_tree).unwrap();
        ref_game.allocate_memory(false);
        run_dcfr(&mut ref_game, 500);
        finalize(&mut ref_game);
        let ref_oop_cfv = range_solver::root_cfvalues(&ref_game, 0);

        // Build per-river evaluator
        let eval = SubtreeExactEvaluator::new(
            board,
            [oop_hands.clone(), ip_hands.clone()],
            [oop_weights.clone(), ip_weights.clone()],
            tree_config,
        ).with_solve_iters(500);

        let num_oop = oop_hands.len();
        let num_ip = ip_hands.len();
        let oop_reach = vec![1.0f32; num_oop];
        let ip_reach = vec![1.0f32; num_ip];
        let (oop_bcfv, _) = eval.compute_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach, num_oop, num_ip, 0,
        );

        let half_pot = 50.0f64;
        let num_combinations = ref_game.num_combinations();
        let payoff_scale = half_pot / num_combinations;

        eprintln!("--- Scaling diagnostic ---");
        eprintln!("half_pot={half_pot}, N={num_combinations}, payoff_scale={payoff_scale:.8}");

        let mut ratios = Vec::new();
        for (h, &bcfv) in oop_bcfv.iter().enumerate().take(5) {
            let (h1, h2) = oop_hands[h];
            let adj = cfreach_adj(h1, h2, &ip_hands, &ip_reach);
            let reconstructed = bcfv as f64 * payoff_scale * adj;
            let reference = ref_oop_cfv[h] as f64;
            let ratio = if reference.abs() > 1e-9 {
                reconstructed / reference
            } else {
                f64::NAN
            };
            ratios.push(ratio);
            eprintln!(
                "  h={h} ({h1},{h2}): bcfv={bcfv:.6} adj={adj:.2} \
                 recon={reconstructed:.6} ref={reference:.6} ratio={ratio:.4}"
            );
        }

        // The test passes if bcfv values are finite; the diagnostic output
        // shows us the scaling we need to correct.
        for &bcfv in &oop_bcfv {
            assert!(bcfv.is_finite(), "bcfv must be finite, got {bcfv}");
        }
    }

    #[test]
    fn compute_raw_cfvs_both_returns_some() {
        let eval = make_river_evaluator(50);
        let num_oop = eval.private_cards[0].len();
        let num_ip = eval.private_cards[1].len();
        let oop_reach = vec![1.0f32; num_oop];
        let ip_reach = vec![1.0f32; num_ip];
        let result = eval.compute_raw_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach, num_oop, num_ip, 0,
        );
        assert!(result.is_some(), "SubtreeExactEvaluator should support raw CFV path");
        let (oop_raw, ip_raw) = result.unwrap();
        assert_eq!(oop_raw.len(), num_oop);
        assert_eq!(ip_raw.len(), num_ip);
        for &v in &oop_raw {
            assert!(v.is_finite(), "raw OOP CFV should be finite, got {v}");
        }
        for &v in &ip_raw {
            assert!(v.is_finite(), "raw IP CFV should be finite, got {v}");
        }
    }

    #[test]
    fn raw_cfvs_match_reference_full_tree_river() {
        // Build a reference full-tree River game and compare raw CFVs.
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

        // Build the reference game directly
        let card_config = build_card_config(
            &board,
            &[oop_hands.clone(), ip_hands.clone()],
            &[oop_weights.clone(), ip_weights.clone()],
        );
        let ref_tree = ActionTree::new(tree_config.clone()).unwrap();
        let mut ref_game = PostFlopGame::with_config(card_config, ref_tree).unwrap();
        ref_game.allocate_memory(false);
        run_dcfr(&mut ref_game, 500);
        finalize(&mut ref_game);
        // root_cfvalues returns per-combo cfvalues (the same units as evaluate_internal result)
        let ref_oop = range_solver::root_cfvalues(&ref_game, 0);

        // Build the evaluator
        let eval = SubtreeExactEvaluator::new(
            board,
            [oop_hands.clone(), ip_hands.clone()],
            [oop_weights.clone(), ip_weights.clone()],
            tree_config,
        ).with_solve_iters(500);

        let num_oop = oop_hands.len();
        let num_ip = ip_hands.len();
        let oop_reach = vec![1.0f32; num_oop];
        let ip_reach = vec![1.0f32; num_ip];
        let (oop_raw, _) = eval.compute_raw_cfvs_both(
            100, 150.0, &oop_reach, &ip_reach, num_oop, num_ip, 0,
        ).expect("should return Some");

        // Raw CFVs should match reference root_cfvalues within DCFR
        // convergence tolerance. root_cfvalues are in per-combo chip units.
        let mut max_err = 0.0f64;
        for (h, (&raw, &reference)) in oop_raw.iter().zip(ref_oop.iter()).enumerate() {
            let err = (raw as f64 - reference as f64).abs();
            if err > max_err { max_err = err; }
            let tol = 0.1 * reference.abs().max(0.01) as f64; // 10% relative tolerance
            assert!(
                err <= tol,
                "hand {h}: raw={raw:.6} ref={reference:.6} err={err:.6} tol={tol:.6}"
            );
        }
        eprintln!("[raw_cfvs_match_reference] max_err={max_err:.6}");
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
