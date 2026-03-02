//! Heads-Up No-Limit Texas Hold'em Postflop game implementation.
//!
//! Models the complete postflop game tree for heads-up no-limit Texas Hold'em.
//! This is a more complex implementation than `HunlPreflop` that handles
//! flop, turn, and river streets with configurable bet sizing.

use std::sync::Arc;

use arrayvec::ArrayVec;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::abstraction::{CardAbstraction, Street};
use crate::card_utils::hand_rank;
use crate::hand_class::{HandClass, HandClassification, classify, intra_class_strength};
use crate::poker::{Card, Rank, Suit, Value};
use crate::showdown_equity;

use super::{ALL_IN, Action, Actions, Game, Player};

/// Selects which card abstraction to use for postflop info-set keys.
#[derive(Debug, Clone)]
pub enum AbstractionMode {
    /// EHS2-based bucketing (expensive Monte Carlo, fine-grained).
    Ehs2(Arc<CardAbstraction>),
    /// Hand-class V2: class ID + intra-class strength + equity bin + draw flags.
    ///
    /// `strength_bits` and `equity_bits` control quantization (0-4 each).
    /// 0 means that dimension is omitted. With both set to 0, this is equivalent
    /// to the old `HandClass` mode (class ID only).
    HandClassV2 {
        /// Number of bits for intra-class strength (0-4).
        strength_bits: u8,
        /// Number of bits for equity bin (0-4).
        equity_bits: u8,
    },
}

/// Configuration for the postflop game.
///
/// Controls stack depth and bet sizing options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostflopConfig {
    /// Stack depth in big blinds
    pub stack_depth: u32,
    /// Available bet sizes as fractions of pot (e.g., 0.5 = half pot)
    pub bet_sizes: Vec<f32>,
    /// Maximum bets/raises allowed per street (default: 3).
    /// After this many bets on a street, only fold/call/check are available.
    /// This keeps the game tree tractable for CFR traversal.
    #[serde(default = "default_max_raises")]
    pub max_raises_per_street: u8,
}

impl Default for PostflopConfig {
    fn default() -> Self {
        Self {
            stack_depth: 100,
            bet_sizes: vec![0.33, 0.5, 0.75, 1.0],
            max_raises_per_street: 3,
        }
    }
}

fn default_max_raises() -> u8 {
    3
}

/// Type of terminal state in the game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalType {
    /// Player folded, opponent wins the pot
    Fold(Player),
    /// Showdown - hands are compared to determine winner
    Showdown,
}

/// Cached per-player evaluation data, precomputed at deal time.
///
/// Groups rank, classification, equity, and strength caches that were
/// previously 6 separate fields on `PostflopState`.
#[derive(Debug, Clone, Default)]
pub struct PlayerCache {
    /// 7-card hand rank (pre-computed when `full_board` is set).
    pub rank: Option<Rank>,
    /// Hand classification (updated on street advance).
    pub class: Option<HandClassification>,
    /// Equity bins: [flop, turn, river]. Precomputed when `full_board` is set.
    pub equity: [u8; 3],
    /// Intra-class strength: [flop, turn, river]. Precomputed when `full_board` is set.
    pub strength: [u8; 3],
}

/// State of a HUNL Postflop hand.
///
/// Tracks the board, holdings, pot, stacks, and action history.
/// Internal units: SB=1, BB=2 (one unit = 0.5 BB).
/// `stack_depth` is in BB; stacks start at `stack_depth * 2` internal units.
///
/// Uses stack-allocated `ArrayVec` for `board` and `history` to avoid
/// heap allocations on clone (the hot path during tree building).
#[derive(Debug, Clone)]
pub struct PostflopState {
    /// Community cards (0-5 cards)
    pub board: ArrayVec<Card, 5>,
    /// Player 1's (SB) hole cards
    pub p1_holding: [Card; 2],
    /// Player 2's (BB) hole cards
    pub p2_holding: [Card; 2],
    /// Pre-dealt 5-card board. When Some, `advance_street` reveals cards from here.
    /// When None, board stays empty (used by `all_canonical_hands` for preflop-only analysis).
    pub full_board: Option<[Card; 5]>,
    /// Current street
    pub street: Street,
    /// Pot size (SB=1, BB=2, initial pot=3)
    pub pot: u32,
    /// Remaining stacks [SB stack, BB stack]
    pub stacks: [u32; 2],
    /// Amount to call
    pub to_call: u32,
    /// Player to act (None if terminal)
    pub to_act: Option<Player>,
    /// Action history with street context
    pub history: ArrayVec<(Street, super::Action), 40>,
    /// Terminal state type (None if not terminal)
    pub terminal: Option<TerminalType>,
    /// Number of bets/raises on the current street
    pub street_bets: u8,
    /// Cached evaluation data for Player 1 (SB).
    pub p1_cache: PlayerCache,
    /// Cached evaluation data for Player 2 (BB).
    pub p2_cache: PlayerCache,
}

impl PostflopState {
    /// Create a new preflop state with blinds posted.
    ///
    /// SB posts 1, BB posts 2.  SB acts first preflop.
    ///
    /// # Arguments
    /// * `p1_holding` - Player 1 (SB) hole cards
    /// * `p2_holding` - Player 2 (BB) hole cards
    /// * `stack_depth` - Stack depth in big blinds
    #[must_use]
    pub fn new_preflop(p1_holding: [Card; 2], p2_holding: [Card; 2], stack_depth: u32) -> Self {
        // Internal units: SB=1, BB=2.  1 BB = 2 internal units.
        let stack = stack_depth * 2;
        Self {
            board: ArrayVec::new(),
            p1_holding,
            p2_holding,
            full_board: None,
            street: Street::Preflop,
            pot: 3,                         // SB (1) + BB (2) in internal units
            stacks: [stack - 1, stack - 2], // SB posted 1, BB posted 2
            to_call: 1,                     // SB needs 1 more to match BB
            to_act: Some(Player::Player1),  // SB acts first preflop
            history: ArrayVec::new(),
            terminal: None,
            street_bets: 1, // BB's post counts as first bet
            p1_cache: PlayerCache::default(),
            p2_cache: PlayerCache::default(),
        }
    }

    /// The player to act, assuming the state is non-terminal.
    ///
    /// Panics in debug builds if `to_act` is `None` (terminal state).
    /// In release builds, defaults to `Player1` to avoid UB.
    #[must_use]
    pub fn active_player(&self) -> Player {
        debug_assert!(
            self.to_act.is_some(),
            "active_player called on terminal state"
        );
        // Infallible on non-terminal states; to_act is always Some when
        // terminal is None (invariant maintained by apply_fold/advance_street).
        self.to_act.unwrap_or(Player::Player1)
    }

    /// Get the current player's remaining stack.
    #[must_use]
    pub fn current_stack(&self) -> u32 {
        match self.to_act {
            Some(Player::Player1) => self.stacks[0],
            Some(Player::Player2) => self.stacks[1],
            None => 0,
        }
    }

    /// Get the opponent's remaining stack.
    #[must_use]
    pub fn opponent_stack(&self) -> u32 {
        match self.to_act {
            Some(Player::Player1) => self.stacks[1],
            Some(Player::Player2) => self.stacks[0],
            None => 0,
        }
    }

    /// Get the current player's holding.
    #[must_use]
    pub fn current_holding(&self) -> [Card; 2] {
        match self.to_act {
            Some(Player::Player2) => self.p2_holding,
            Some(Player::Player1) | None => self.p1_holding, // Default to P1 for terminal states
        }
    }

    /// Create a new preflop state with a pre-dealt 5-card board.
    ///
    /// The board cards are revealed progressively as streets advance:
    /// flop reveals cards 0-2, turn reveals card 3, river reveals card 4.
    #[must_use]
    pub fn new_preflop_with_board(
        p1_holding: [Card; 2],
        p2_holding: [Card; 2],
        full_board: [Card; 5],
        stack_depth_bb: u32,
    ) -> Self {
        let mut state = Self::new_preflop(p1_holding, p2_holding, stack_depth_bb);
        state.full_board = Some(full_board);

        // Pre-compute final hand ranks (7 cards = 2 hole + 5 board)
        state.p1_cache.rank = Some(hand_rank(p1_holding, &full_board));
        state.p2_cache.rank = Some(hand_rank(p2_holding, &full_board));

        state
    }

    /// Precompute equity bins and intra-class strength for all 3 postflop streets.
    ///
    /// Called once per deal during `initial_states()` when using `HandClassV2`.
    /// This is O(~3K) rank evaluations per deal (enumeration over opponent combos).
    pub fn precompute_v2_caches(&mut self) {
        let Some(full_board) = self.full_board else {
            return;
        };
        let board_slices: [&[Card]; 3] = [&full_board[..3], &full_board[..4], &full_board[..5]];
        for (i, board) in board_slices.iter().enumerate() {
            self.p1_cache.equity[i] = precompute_equity_bin(self.p1_holding, board);
            self.p2_cache.equity[i] = precompute_equity_bin(self.p2_holding, board);
            self.p1_cache.strength[i] = precompute_strength(self.p1_holding, board);
            self.p2_cache.strength[i] = precompute_strength(self.p2_holding, board);
        }
    }
}

/// Map a street to a cache index (flop=0, turn=1, river=2).
/// Returns 0 for preflop (unused in practice).
/// Convert a `Street` to the 2-bit info set key code.
fn street_to_info_code(street: Street) -> u8 {
    match street {
        Street::Preflop => 0,
        Street::Flop => 1,
        Street::Turn => 2,
        Street::River => 3,
    }
}

/// Encode the actions on the current street from the game history.
fn encode_current_street_actions(
    history: &[(Street, Action)],
    current_street: Street,
) -> arrayvec::ArrayVec<u8, 6> {
    use crate::info_key::encode_action;
    let mut codes = arrayvec::ArrayVec::<u8, 6>::new();
    for (street, a) in history {
        if *street == current_street && !codes.is_full() {
            codes.push(encode_action(*a));
        }
    }
    codes
}

/// Apply a fold action to the game state.
fn apply_fold(state: &PostflopState, new_state: &mut PostflopState) {
    let folder = state.active_player();
    new_state.terminal = Some(TerminalType::Fold(folder));
    new_state.to_act = None;
}

/// Convert internal chip units to BB (1 BB = 2 internal units).
fn to_bb(chips: u32) -> f64 {
    f64::from(chips) / 2.0
}

/// Compute P1's payoff when a player folds.
fn compute_fold_payoff(folder: Player, p1_invested: u32, p2_invested: u32) -> f64 {
    if folder == Player::Player1 {
        -to_bb(p1_invested)
    } else {
        to_bb(p2_invested)
    }
}

/// Compute P1's payoff at showdown.
fn compute_showdown_payoff(state: &PostflopState, p1_invested: u32, p2_invested: u32) -> f64 {
    use std::cmp::Ordering;

    let p1_rank = state
        .p1_cache
        .rank
        .unwrap_or_else(|| hand_rank(state.p1_holding, &state.board));
    let p2_rank = state
        .p2_cache
        .rank
        .unwrap_or_else(|| hand_rank(state.p2_holding, &state.board));

    let pot_bb = to_bb(p1_invested + p2_invested);

    match p1_rank.cmp(&p2_rank) {
        Ordering::Greater => pot_bb - to_bb(p1_invested),
        Ordering::Less => -to_bb(p1_invested),
        Ordering::Equal => pot_bb / 2.0 - to_bb(p1_invested),
    }
}

/// Compute the 7-card hand rank from board + hole cards.
fn street_to_cache_idx(street: Street) -> usize {
    match street {
        Street::Preflop | Street::Flop => 0,
        Street::Turn => 1,
        Street::River => 2,
    }
}

/// Compute the equity bin (0-15) for a holding on a given board slice.
///
/// Uses 16 bins as the maximum resolution. The actual number of bits
/// used is determined by the `AbstractionMode::HandClassV2` config.
fn precompute_equity_bin(holding: [Card; 2], board: &[Card]) -> u8 {
    let eq = showdown_equity::compute_equity(holding, board);
    showdown_equity::equity_bin(eq, 16)
}

/// Compute the intra-class strength (1-14) for a holding on a given board.
fn precompute_strength(holding: [Card; 2], board: &[Card]) -> u8 {
    let Ok(classification) = classify(holding, board) else {
        return 1;
    };
    let made_id = classification.strongest_made_id();
    if !HandClass::is_made_hand_id(made_id) {
        return 1; // draw-only — no made-hand differentiation
    }
    // Safe: made_id < NUM_MADE always maps to a valid HandClass discriminant
    let class = HandClass::ALL[made_id as usize];
    intra_class_strength(holding, board, class)
}

/// Build a 7-card hand and rank it.
/// Full HUNL game including postflop streets.
///
/// Implements the [`Game`] trait for use with CFR solvers.
#[derive(Debug)]
pub struct HunlPostflop {
    config: PostflopConfig,
    abstraction: Option<AbstractionMode>,
    /// Number of random deals to generate for the deal pool.
    deal_count: usize,
    rng_seed: u64,
    /// Minimum deals per hand class for stratified generation (0 = disabled).
    min_deals_per_class: usize,
    /// Maximum rejection-sample attempts per deficit class.
    max_rejections_per_class: usize,
    /// Use exhaustive uniform deal enumeration for hand-class modes.
    ///
    /// When true, enumerates all (flop, P1 hole) combos instead of
    /// weighted random sampling. Produces ~2M base deals (1,755 × C(49,2)).
    uniform_deals: bool,
}

impl HunlPostflop {
    /// Create a new HUNL postflop game.
    ///
    /// # Arguments
    /// * `config` - Game configuration (stack depth, bet sizes, etc.)
    /// * `abstraction` - Optional abstraction mode for bucketing
    /// * `deal_count` - Number of random deals to generate for the deal pool
    #[must_use]
    pub fn new(
        config: PostflopConfig,
        abstraction: Option<AbstractionMode>,
        deal_count: usize,
    ) -> Self {
        Self {
            config,
            abstraction,
            deal_count,
            rng_seed: 42,
            min_deals_per_class: 0,
            max_rejections_per_class: 500_000,
            uniform_deals: false,
        }
    }

    /// Set RNG seed for reproducible sampling.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng_seed = seed;
    }

    /// Enable stratified deal generation to ensure rare hand classes are represented.
    ///
    /// After generating the base deal pool, rejection sampling tops up any class
    /// with fewer than `min_per_class` observations. At most `max_rejections`
    /// consecutive failures are attempted per deficit class before giving up.
    #[must_use]
    pub fn with_stratification(mut self, min_per_class: usize, max_rejections: usize) -> Self {
        self.min_deals_per_class = min_per_class;
        self.max_rejections_per_class = max_rejections;
        self
    }

    /// Enable exhaustive uniform deal enumeration.
    ///
    /// Enumerates all (flop, P1 hole) combos instead of weighted
    /// random sampling. Produces ~2M base deals.
    #[must_use]
    pub fn with_uniform_deals(mut self) -> Self {
        self.uniform_deals = true;
        self
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &PostflopConfig {
        &self.config
    }

    /// Generate all 169 canonical preflop hands.
    ///
    /// Returns states for all pairs (13), suited hands (78), and offsuit hands (78).
    #[must_use]
    pub fn all_canonical_hands(stack_depth: u32) -> Vec<PostflopState> {
        let mut states = Vec::with_capacity(169);

        // Descending order (Ace first) for canonical hand display
        let mut values = crate::poker::ALL_VALUES;
        values.reverse();

        // Default opponent hand (avoids most collisions)
        let default_p2 = [
            Card::new(Value::Two, Suit::Club),
            Card::new(Value::Three, Suit::Diamond),
        ];
        let alt_p2 = [
            Card::new(Value::Four, Suit::Club),
            Card::new(Value::Five, Suit::Diamond),
        ];

        let get_opponent = |v1: Value, v2: Value| {
            if v1 == Value::Two || v1 == Value::Three || v2 == Value::Two || v2 == Value::Three {
                alt_p2
            } else {
                default_p2
            }
        };

        for (i, &v1) in values.iter().enumerate() {
            for &v2 in &values[i..] {
                if v1 == v2 {
                    // Pair
                    let p1 = [Card::new(v1, Suit::Spade), Card::new(v2, Suit::Heart)];
                    states.push(PostflopState::new_preflop(
                        p1,
                        get_opponent(v1, v2),
                        stack_depth,
                    ));
                } else {
                    // Suited
                    let p1_suited = [Card::new(v1, Suit::Spade), Card::new(v2, Suit::Spade)];
                    states.push(PostflopState::new_preflop(
                        p1_suited,
                        get_opponent(v1, v2),
                        stack_depth,
                    ));

                    // Offsuit
                    let p1_offsuit = [Card::new(v1, Suit::Spade), Card::new(v2, Suit::Heart)];
                    states.push(PostflopState::new_preflop(
                        p1_offsuit,
                        get_opponent(v1, v2),
                        stack_depth,
                    ));
                }
            }
        }

        states
    }

    /// Generate all 52 cards of a standard deck.
    fn full_deck() -> Vec<Card> {
        crate::poker::full_deck()
    }

    /// Generate random deals with pre-dealt boards for MCCFR sampling.
    ///
    /// Each deal consists of 2 hole cards per player and 5 board cards,
    /// all drawn without replacement from a shuffled deck.
    fn generate_random_deals(&self, seed: u64, count: usize) -> Vec<PostflopState> {
        use rand::SeedableRng;
        use rand::prelude::SliceRandom;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(seed);
        let deck = Self::full_deck();
        let mut deals = Vec::with_capacity(count);

        tracing::info!(count, "generating random deals");
        let gen_start = std::time::Instant::now();
        for _ in 0..count {
            let mut shuffled = deck.clone();
            shuffled.shuffle(&mut rng);

            let p1 = [shuffled[0], shuffled[1]];
            let p2 = [shuffled[2], shuffled[3]];
            let board = [
                shuffled[4],
                shuffled[5],
                shuffled[6],
                shuffled[7],
                shuffled[8],
            ];
            deals.push(PostflopState::new_preflop_with_board(
                p1,
                p2,
                board,
                self.config.stack_depth,
            ));
        }
        tracing::info!(count, elapsed = ?gen_start.elapsed(), "deals generated");
        deals
    }

    /// Generate deals from the 1,755 canonical flops, weighted by frequency.
    ///
    /// For each deal: pick a canonical flop (weighted), then deal random
    /// turn + river + 2+2 hole cards from the remaining 49 cards.
    fn generate_flop_deals(&self, seed: u64, count: usize) -> Vec<PostflopState> {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        tracing::info!(count, "generating flop-weighted deals");
        let gen_start = std::time::Instant::now();

        let mut rng = StdRng::seed_from_u64(seed);
        let sampler = FlopSampler::new();

        let deals: Vec<_> = (0..count)
            .map(|_| sampler.sample_deal(&mut rng, self.config.stack_depth))
            .collect();
        tracing::info!(count, elapsed = ?gen_start.elapsed(), "deals generated");
        deals
    }

    /// Generate exhaustive uniform deals: every (flop, P1 hole) combination.
    ///
    /// For each of the 1,755 canonical flops and each of the C(49,2)=1,176 P1 hole
    /// card pairs, generates one random completion (P2 hole + turn + river).
    /// Total output: 1,755 × 1,176 = 2,064,480 deals.
    fn generate_uniform_deals(&self, seed: u64) -> Vec<PostflopState> {
        use crate::flops;

        let canonical_flops = flops::all_flops();
        let deck = Self::full_deck();
        let total = canonical_flops.len() * C49_2;

        tracing::info!(flops = canonical_flops.len(), pairs = C49_2, total, "generating uniform deals");

        let stack_depth = self.config.stack_depth;

        let deals: Vec<PostflopState> = canonical_flops
            .par_iter()
            .enumerate()
            .flat_map_iter(|(flop_idx, flop)| {
                generate_deals_for_flop(flop, &deck, seed, flop_idx, stack_depth)
            })
            .collect();

        tracing::info!(deals = deals.len(), "deals generated");
        deals
    }

    /// Generate a stratified deal pool that ensures minimum representation per class.
    ///
    /// Phase 1: generate the base pool via `generate_flop_deals`.
    /// Phase 2+3: top up deficit classes via `stratify_pool`.
    #[cfg(test)]
    fn generate_stratified_deals(
        &self,
        seed: u64,
        count: usize,
        min_per_class: usize,
        max_rejections: usize,
    ) -> Vec<PostflopState> {
        let deals = self.generate_flop_deals(seed, count);
        self.stratify_pool(deals, seed, min_per_class, max_rejections)
    }

    /// Top up a deal pool with rejection-sampled deals for rare hand classes.
    ///
    /// Counts per-class coverage, identifies classes below `min_per_class`,
    /// and rejection-samples new deals until each deficit class is covered
    /// or `max_rejections` consecutive failures are hit.
    fn stratify_pool(
        &self,
        mut deals: Vec<PostflopState>,
        seed: u64,
        min_per_class: usize,
        max_rejections: usize,
    ) -> Vec<PostflopState> {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut coverage = count_class_coverage(&deals);
        let deficit_classes = find_deficit_classes(&coverage, min_per_class);

        if deficit_classes.is_empty() {
            return deals;
        }

        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(0x5742_4154));
        let sampler = FlopSampler::new();
        let base_count = deals.len();
        let mut total_added = 0usize;

        for &(_, class_idx) in &deficit_classes {
            let added = fill_deficit_class(
                &sampler,
                &mut rng,
                &mut coverage,
                &mut deals,
                class_idx,
                min_per_class,
                max_rejections,
                self.config.stack_depth,
            );
            total_added += added;
        }

        tracing::info!(deals = deals.len(), base_count, total_added, "stratified pool generated");
        print_coverage_summary(&coverage);

        deals
    }

    /// Resolve a bet-size index to an absolute cent amount.
    ///
    /// [`ALL_IN`] maps to the full effective stack. Any other index looks up
    /// the corresponding pot fraction in `config.bet_sizes` and rounds to
    /// the nearest cent, capped at the effective stack.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn resolve_bet_amount(&self, idx: u32, pot: u32, effective_stack: u32) -> u32 {
        if idx == ALL_IN {
            effective_stack
        } else {
            let fraction = self.config.bet_sizes[idx as usize];
            let size = (f64::from(pot) * f64::from(fraction)).round() as u32;
            size.min(effective_stack)
        }
    }

    /// Advance the game to the next street or showdown.
    ///
    /// When the state has a `full_board`, board cards are revealed progressively:
    /// preflop→flop reveals 3 cards, flop→turn reveals 1, turn→river reveals 1.
    #[allow(clippy::unused_self)]
    fn advance_street(&self, state: &mut PostflopState) {
        match state.street {
            Street::Preflop => {
                state.street = Street::Flop;
                if let Some(fb) = state.full_board {
                    state.board.push(fb[0]);
                    state.board.push(fb[1]);
                    state.board.push(fb[2]);
                }
                state.street_bets = 0;
                state.to_call = 0;
                // Postflop: P1 (SB) is out of position, acts first
                state.to_act = Some(Player::Player1);
            }
            Street::Flop => {
                state.street = Street::Turn;
                if let Some(fb) = state.full_board {
                    state.board.push(fb[3]);
                }
                state.street_bets = 0;
                state.to_call = 0;
                state.to_act = Some(Player::Player1);
            }
            Street::Turn => {
                state.street = Street::River;
                if let Some(fb) = state.full_board {
                    state.board.push(fb[4]);
                }
                state.street_bets = 0;
                state.to_call = 0;
                state.to_act = Some(Player::Player1);
            }
            Street::River => {
                // River checked through or called - showdown
                state.terminal = Some(TerminalType::Showdown);
                state.to_act = None;
            }
        }

        // Cache hand classifications after board cards change (used in info_set_key)
        if !state.board.is_empty() && state.terminal.is_none() {
            state.p1_cache.class = classify(state.p1_holding, &state.board).ok();
            state.p2_cache.class = classify(state.p2_holding, &state.board).ok();
        }
    }

    fn apply_check(&self, state: &PostflopState, new_state: &mut PostflopState) {
        let is_p1 = state.to_act == Some(Player::Player1);
        if !is_p1 && state.street == Street::Preflop && state.to_call == 0 {
            self.advance_street(new_state);
            return;
        }
        let last_action_on_street = state
            .history
            .iter()
            .rev()
            .find(|(s, _)| *s == state.street)
            .map(|(_, a)| a);

        if matches!(last_action_on_street, Some(Action::Check)) {
            self.advance_street(new_state);
        } else {
            new_state.to_act = Some(state.active_player().opponent());
        }
    }

    fn apply_call(&self, state: &PostflopState, new_state: &mut PostflopState) {
        let is_p1 = state.to_act == Some(Player::Player1);
        let player_idx = usize::from(!is_p1);
        let call_amount = state.to_call.min(state.current_stack());
        new_state.stacks[player_idx] -= call_amount;
        new_state.pot += call_amount;
        new_state.to_call = 0;

        if is_p1 && state.street == Street::Preflop && state.street_bets == 1 {
            new_state.to_act = Some(Player::Player2);
        } else {
            self.advance_street(new_state);
        }
    }

    fn apply_bet_or_raise(&self, state: &PostflopState, new_state: &mut PostflopState, idx: u32) {
        let is_p1 = state.to_act == Some(Player::Player1);
        let player_idx = usize::from(!is_p1);
        let effective_stack = state.current_stack().saturating_sub(state.to_call);
        let bet_portion = self.resolve_bet_amount(idx, state.pot, effective_stack);
        let total = state.to_call + bet_portion;
        let actual = total.min(state.current_stack());
        new_state.stacks[player_idx] -= actual;
        new_state.pot += actual;
        new_state.to_call = actual.saturating_sub(state.to_call);
        new_state.street_bets += 1;
        new_state.to_act = Some(state.active_player().opponent());

        if new_state.opponent_stack() == 0 {
            new_state.terminal = Some(TerminalType::Showdown);
            new_state.to_act = None;
        }
    }
}

/// Pre-built sampler for generating deals from the canonical flop distribution.
///
/// Caches the canonical flops, their weighted distribution, and the full deck
/// so callers can generate deals without redundant setup.
struct FlopSampler {
    canonical_flops: Vec<crate::flops::CanonicalFlop>,
    dist: rand::distr::weighted::WeightedIndex<u16>,
    deck: Vec<Card>,
}

impl FlopSampler {
    fn new() -> Self {
        use crate::flops;
        use rand::distr::weighted::WeightedIndex;

        let canonical_flops = flops::all_flops();
        let weights: Vec<u16> = canonical_flops
            .iter()
            .map(flops::CanonicalFlop::weight)
            .collect();
        // Infallible: canonical flops are non-empty with positive weights.
        let dist = WeightedIndex::new(&weights)
            .expect("canonical flops are non-empty with positive weights");
        let deck = crate::poker::full_deck();
        Self {
            canonical_flops,
            dist,
            deck,
        }
    }

    fn sample_deal(&self, rng: &mut rand::rngs::StdRng, stack_depth: u32) -> PostflopState {
        use rand::prelude::Distribution;
        use rand::prelude::SliceRandom;

        let flop = &self.canonical_flops[self.dist.sample(rng)];
        let flop_cards = *flop.cards();

        let mut remaining: Vec<Card> = self
            .deck
            .iter()
            .filter(|c| !flop_cards.contains(c))
            .copied()
            .collect();
        remaining.shuffle(rng);

        let p1 = [remaining[0], remaining[1]];
        let p2 = [remaining[2], remaining[3]];
        let board = [
            flop_cards[0],
            flop_cards[1],
            flop_cards[2],
            remaining[4],
            remaining[5],
        ];

        PostflopState::new_preflop_with_board(p1, p2, board, stack_depth)
    }
}

/// Print a compact summary of per-class deal coverage.
/// Identify hand classes that fall below the minimum coverage threshold.
///
/// Returns `(deficit, class_index)` pairs sorted by largest deficit first.
#[allow(clippy::cast_possible_truncation)]
fn find_deficit_classes(
    coverage: &[usize; HandClass::COUNT],
    min_per_class: usize,
) -> Vec<(usize, u8)> {
    let mut deficit_classes: Vec<(usize, u8)> = (0u8..(HandClass::COUNT as u8))
        .filter(|&i| coverage[i as usize] < min_per_class)
        .map(|i| (min_per_class - coverage[i as usize], i))
        .collect();
    deficit_classes.sort_by_key(|&(deficit, _)| std::cmp::Reverse(deficit));
    deficit_classes
}

/// Rejection-sample deals until a deficit class meets its minimum coverage.
///
/// Returns the number of deals added.
#[allow(clippy::too_many_arguments)]
fn fill_deficit_class(
    sampler: &FlopSampler,
    rng: &mut rand::rngs::StdRng,
    coverage: &mut [usize; HandClass::COUNT],
    deals: &mut Vec<PostflopState>,
    class_idx: u8,
    min_per_class: usize,
    max_rejections: usize,
    stack_depth: u32,
) -> usize {
    let ci = class_idx as usize;
    let mut added = 0usize;
    let mut consecutive_failures = 0usize;

    while coverage[ci] < min_per_class && consecutive_failures < max_rejections {
        let deal = sampler.sample_deal(rng, stack_depth);

        let Some(board) = deal.full_board else {
            consecutive_failures += 1;
            continue;
        };
        let deal_classes = classify_deal_bits(&deal, &board);

        if deal_classes & (1 << class_idx) != 0 {
            update_coverage(coverage, deal_classes);
            deals.push(deal);
            added += 1;
            consecutive_failures = 0;
        } else {
            consecutive_failures += 1;
        }
    }
    added
}

/// Compute the combined hand-class bitmask for both players in a deal.
fn classify_deal_bits(deal: &PostflopState, board: &[Card; 5]) -> u32 {
    let mut bits = 0u32;
    if let Ok(c1) = classify(deal.p1_holding, board) {
        bits |= c1.bits();
    }
    if let Ok(c2) = classify(deal.p2_holding, board) {
        bits |= c2.bits();
    }
    bits
}

/// Increment coverage counters for all classes present in the bitmask.
fn update_coverage(coverage: &mut [usize; HandClass::COUNT], deal_classes: u32) {
    for (i, count) in coverage.iter_mut().enumerate() {
        if deal_classes & (1 << i) != 0 {
            *count += 1;
        }
    }
}

fn print_coverage_summary(coverage: &[usize; HandClass::COUNT]) {
    use crate::hand_class::HandClass;
    let labels = [
        "SF", "4K", "FH", "Fl", "St", "Se", "Tr", "2P", "OP", "Pr", "UP", "OC", "HC", "CD", "FD",
        "BF", "OS", "GS", "BD",
    ];
    let parts: Vec<String> = (0..HandClass::COUNT)
        .filter(|&i| i < HandClass::NUM_MADE)
        .map(|i| format!("{}={}", labels[i], coverage[i]))
        .collect();
    tracing::info!(coverage = %parts.join(", "), "class coverage");
}

/// Number of 2-element subsets of 49 remaining cards: C(49, 2).
const C49_2: usize = 49 * 48 / 2;

/// Decode a combinatorial index to a pair `(a, b)` with `0 <= a < b < n`.
///
/// Uses the combinatorial number system for 2-element subsets.
/// Index space: `0..n*(n-1)/2`.
#[must_use]
fn decode_hole_pair(idx: usize, n: usize) -> (usize, usize) {
    debug_assert!(idx < n * (n - 1) / 2, "idx {idx} out of range for n={n}");
    // b is the largest integer such that C(b,2) = b*(b-1)/2 <= idx.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let mut b = f64::midpoint(1.0, ((8 * idx + 1) as f64).sqrt()) as usize;
    // Correct for floating-point rounding at exact squares
    if b > 0 && b * (b - 1) / 2 > idx {
        b -= 1;
    }
    if (b + 1) * b / 2 <= idx {
        b += 1;
    }
    let a = idx - b * (b - 1) / 2;
    (a, b)
}

/// Count how many deals cover each hand class on the river.
///
/// For each deal, classifies both players' hands against the full 5-card board.
/// A deal counts toward a class if *either* player has that class.
fn count_class_coverage(deals: &[PostflopState]) -> [usize; HandClass::COUNT] {
    let mut counts = [0usize; HandClass::COUNT];
    for deal in deals {
        let Some(board) = deal.full_board else {
            continue;
        };
        let deal_classes = classify_deal_bits(deal, &board);
        update_coverage(&mut counts, deal_classes);
    }
    counts
}

/// Generate all (`P1_hole`, `random_completion`) deals for a single canonical flop.
///
/// Enumerates every 2-card P1 holding from the 49 non-flop cards, and for each
/// randomly selects P2 hole cards, turn, and river from the remaining 47.
fn generate_deals_for_flop(
    flop: &crate::flops::CanonicalFlop,
    deck: &[Card],
    base_seed: u64,
    flop_idx: usize,
    stack_depth: u32,
) -> Vec<PostflopState> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let flop_cards = *flop.cards();
    let remaining: Vec<Card> = deck
        .iter()
        .filter(|c| !flop_cards.contains(c))
        .copied()
        .collect();
    debug_assert_eq!(remaining.len(), 49);

    let flop_seed = base_seed.wrapping_add((flop_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut rng = StdRng::seed_from_u64(flop_seed);
    let mut deals = Vec::with_capacity(C49_2);

    for pair_idx in 0..C49_2 {
        let (a, b) = decode_hole_pair(pair_idx, 49);
        let p1 = [remaining[a], remaining[b]];

        // Build 47-card pool excluding P1's hole cards (stack-allocated)
        let mut pool = [remaining[0]; 47];
        let mut k = 0;
        for (i, &c) in remaining.iter().enumerate() {
            if i != a && i != b {
                pool[k] = c;
                k += 1;
            }
        }
        debug_assert_eq!(k, 47);

        // Fisher-Yates partial shuffle: pick 4 random cards
        for i in 0..4 {
            let j = rng.random_range(i..47);
            pool.swap(i, j);
        }

        let p2 = [pool[0], pool[1]];
        let board = [
            flop_cards[0],
            flop_cards[1],
            flop_cards[2],
            pool[2],
            pool[3],
        ];
        deals.push(PostflopState::new_preflop_with_board(
            p1,
            p2,
            board,
            stack_depth,
        ));
    }

    deals
}

impl Game for HunlPostflop {
    type State = PostflopState;

    fn initial_states(&self) -> Vec<Self::State> {
        let mut deals = match &self.abstraction {
            Some(AbstractionMode::HandClassV2 { .. }) => {
                let base = if self.uniform_deals {
                    self.generate_uniform_deals(self.rng_seed)
                } else {
                    self.generate_flop_deals(self.rng_seed, self.deal_count)
                };
                if self.min_deals_per_class > 0 {
                    self.stratify_pool(
                        base,
                        self.rng_seed,
                        self.min_deals_per_class,
                        self.max_rejections_per_class,
                    )
                } else {
                    base
                }
            }
            _ => self.generate_random_deals(self.rng_seed, self.deal_count),
        };

        // Precompute equity bins and strength for HandClassV2 mode
        if matches!(&self.abstraction, Some(AbstractionMode::HandClassV2 { .. })) {
            tracing::info!(deals = deals.len(), "precomputing equity bins and strength");
            let precomp_start = std::time::Instant::now();
            deals
                .par_iter_mut()
                .for_each(PostflopState::precompute_v2_caches);
            tracing::info!(elapsed = ?precomp_start.elapsed(), "precomputation complete");
        }

        deals
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        state.terminal.is_some()
    }

    fn player(&self, state: &Self::State) -> Player {
        state.active_player()
    }

    fn actions(&self, state: &Self::State) -> Actions {
        let mut actions = Actions::new();
        if state.terminal.is_some() {
            return actions;
        }

        let stack = state.current_stack();
        let to_call = state.to_call;

        // Can fold if facing a bet
        if to_call > 0 {
            actions.push(Action::Fold);
        }

        // Check or call
        if to_call == 0 {
            actions.push(Action::Check);
        } else if stack >= to_call {
            actions.push(Action::Call);
        }

        // Sized bet/raise options (only if under the per-street raise cap).
        // Bets that exceed the effective stack are capped to all-in in
        // `next_state()`.
        if state.street_bets < self.config.max_raises_per_street {
            #[allow(clippy::cast_possible_truncation)]
            for idx in 0..self.config.bet_sizes.len() {
                if actions.is_full() {
                    break;
                }
                let idx_u32 = idx as u32;
                if to_call == 0 {
                    actions.push(Action::Bet(idx_u32));
                } else {
                    actions.push(Action::Raise(idx_u32));
                }
            }
        }

        // All-in is always available regardless of raise cap
        if !actions.is_full() {
            if to_call == 0 {
                actions.push(Action::Bet(ALL_IN));
            } else {
                actions.push(Action::Raise(ALL_IN));
            }
        }

        actions
    }

    fn next_state(&self, state: &Self::State, action: Action) -> Self::State {
        let mut new_state = state.clone();
        new_state.history.push((state.street, action));

        match action {
            Action::Fold => apply_fold(state, &mut new_state),
            Action::Check => self.apply_check(state, &mut new_state),
            Action::Call => self.apply_call(state, &mut new_state),
            Action::Bet(idx) | Action::Raise(idx) => {
                self.apply_bet_or_raise(state, &mut new_state, idx);
            }
        }

        new_state
    }

    fn utility(&self, state: &Self::State, player: Player) -> f64 {
        let Some(terminal) = state.terminal else {
            return 0.0;
        };

        let starting_stack = self.config.stack_depth * 2;
        let p1_invested = starting_stack - state.stacks[0];
        let p2_invested = starting_stack - state.stacks[1];

        let p1_ev = match terminal {
            TerminalType::Fold(folder) => compute_fold_payoff(folder, p1_invested, p2_invested),
            TerminalType::Showdown => compute_showdown_payoff(state, p1_invested, p2_invested),
        };

        if player == Player::Player1 {
            p1_ev
        } else {
            -p1_ev
        }
    }

    fn info_set_key(&self, state: &Self::State) -> u64 {
        use crate::info_key::{
            InfoKey, canonical_hand_index, compute_hand_bits_v2, encode_hand_v2, spr_bucket,
        };

        let holding = state.current_holding();

        // Get hand/bucket bits
        let hand_bits = match &self.abstraction {
            Some(AbstractionMode::Ehs2(abstraction)) if !state.board.is_empty() => {
                match abstraction.get_bucket(&state.board, (holding[0], holding[1])) {
                    Ok(b) => u32::from(b),
                    Err(_) => 0,
                }
            }
            Some(AbstractionMode::HandClassV2 {
                strength_bits,
                equity_bits,
            }) if !state.board.is_empty() => {
                let is_p2 = state.to_act == Some(Player::Player2);
                let cache = if is_p2 {
                    &state.p2_cache
                } else {
                    &state.p1_cache
                };

                if let Some(classification) = cache.class {
                    // Fast path: use precomputed values from deal-time caching
                    let street_idx = street_to_cache_idx(state.street);
                    encode_hand_v2(
                        classification.strongest_made_id(),
                        cache.strength[street_idx],
                        cache.equity[street_idx],
                        classification.draw_flags(),
                        *strength_bits,
                        *equity_bits,
                    )
                } else {
                    // Slow path: compute from scratch
                    compute_hand_bits_v2(holding, &state.board, *strength_bits, *equity_bits)
                }
            }
            _ => u32::from(canonical_hand_index(holding)),
        };

        let street_code = street_to_info_code(state.street);
        let spr = spr_bucket(state.pot, state.stacks[0].min(state.stacks[1]));
        let action_codes = encode_current_street_actions(&state.history, state.street);

        InfoKey::new(hand_bits, street_code, spr, &action_codes).as_u64()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, clippy::items_after_statements)]
    use super::*;
    use crate::hand_class::HandClass;
    use crate::poker::{Suit, Value};
    use test_macros::timed_test;

    fn make_card(value: Value, suit: Suit) -> Card {
        Card::new(value, suit)
    }

    fn sample_holdings() -> ([Card; 2], [Card; 2]) {
        let p1 = [
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::King, Suit::Spade),
        ];
        let p2 = [
            make_card(Value::Queen, Suit::Heart),
            make_card(Value::Jack, Suit::Heart),
        ];
        (p1, p2)
    }

    #[timed_test]
    fn postflop_config_default_values() {
        let config = PostflopConfig::default();
        assert_eq!(config.stack_depth, 100);
        assert_eq!(config.bet_sizes, vec![0.33, 0.5, 0.75, 1.0]);
    }

    #[timed_test]
    fn postflop_config_custom_values() {
        let config = PostflopConfig {
            stack_depth: 50,
            bet_sizes: vec![0.5, 1.0],
            ..PostflopConfig::default()
        };
        assert_eq!(config.stack_depth, 50);
        assert_eq!(config.bet_sizes.len(), 2);
    }

    #[timed_test]
    fn new_preflop_sets_correct_pot() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // Pot should be SB (1) + BB (2) = 3
        assert_eq!(state.pot, 3);
    }

    #[timed_test]
    fn new_preflop_sets_correct_stacks() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // 100 BB = 200 internal units.  SB posted 1, BB posted 2.
        assert_eq!(state.stacks[0], 199);
        assert_eq!(state.stacks[1], 198);
    }

    #[timed_test]
    fn new_preflop_sb_acts_first() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert_eq!(state.to_act, Some(Player::Player1));
    }

    #[timed_test]
    fn new_preflop_to_call_is_half_bb() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // SB needs 1 more to match BB's 2
        assert_eq!(state.to_call, 1);
    }

    #[timed_test]
    fn new_preflop_street_is_preflop() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert_eq!(state.street, Street::Preflop);
    }

    #[timed_test]
    fn new_preflop_board_is_empty() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert!(state.board.is_empty());
    }

    #[timed_test]
    fn new_preflop_is_not_terminal() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert!(state.terminal.is_none());
    }

    #[timed_test]
    fn new_preflop_history_is_empty() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert!(state.history.is_empty());
    }

    #[timed_test]
    fn new_preflop_street_bets_is_one() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // BB's post counts as first bet
        assert_eq!(state.street_bets, 1);
    }

    #[timed_test]
    fn current_stack_returns_sb_stack_when_sb_to_act() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // SB acts first, should return SB's stack (200 - 1)
        assert_eq!(state.current_stack(), 199);
    }

    #[timed_test]
    fn current_stack_returns_bb_stack_when_bb_to_act() {
        let (p1, p2) = sample_holdings();
        let mut state = PostflopState::new_preflop(p1, p2, 100);
        state.to_act = Some(Player::Player2);

        assert_eq!(state.current_stack(), 198);
    }

    #[timed_test]
    fn opponent_stack_returns_bb_stack_when_sb_to_act() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        // SB acts first, opponent is BB (200 - 2)
        assert_eq!(state.opponent_stack(), 198);
    }

    #[timed_test]
    fn opponent_stack_returns_sb_stack_when_bb_to_act() {
        let (p1, p2) = sample_holdings();
        let mut state = PostflopState::new_preflop(p1, p2, 100);
        state.to_act = Some(Player::Player2);

        assert_eq!(state.opponent_stack(), 199);
    }

    #[timed_test]
    fn current_holding_returns_p1_holding_when_sb_to_act() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert_eq!(state.current_holding(), p1);
    }

    #[timed_test]
    fn current_holding_returns_p2_holding_when_bb_to_act() {
        let (p1, p2) = sample_holdings();
        let mut state = PostflopState::new_preflop(p1, p2, 100);
        state.to_act = Some(Player::Player2);

        assert_eq!(state.current_holding(), p2);
    }

    #[timed_test]
    fn terminal_type_fold_equality() {
        let fold_p1 = TerminalType::Fold(Player::Player1);
        let fold_p1_again = TerminalType::Fold(Player::Player1);
        let fold_p2 = TerminalType::Fold(Player::Player2);

        assert_eq!(fold_p1, fold_p1_again);
        assert_ne!(fold_p1, fold_p2);
    }

    #[timed_test]
    fn terminal_type_showdown_equality() {
        let showdown1 = TerminalType::Showdown;
        let showdown2 = TerminalType::Showdown;

        assert_eq!(showdown1, showdown2);
    }

    #[timed_test]
    fn postflop_state_stores_holdings() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);

        assert_eq!(state.p1_holding, p1);
        assert_eq!(state.p2_holding, p2);
    }

    #[timed_test]
    fn different_stack_depths() {
        let (p1, p2) = sample_holdings();

        let state_50bb = PostflopState::new_preflop(p1, p2, 50);
        assert_eq!(state_50bb.stacks[0], 99); // 50*2 - 1
        assert_eq!(state_50bb.stacks[1], 98); // 50*2 - 2

        let state_200bb = PostflopState::new_preflop(p1, p2, 200);
        assert_eq!(state_200bb.stacks[0], 399); // 200*2 - 1
        assert_eq!(state_200bb.stacks[1], 398); // 200*2 - 2
    }

    // ==================== HunlPostflop Game trait tests ====================

    fn create_game() -> HunlPostflop {
        HunlPostflop::new(PostflopConfig::default(), None, 1)
    }

    #[timed_test]
    fn preflop_actions_include_fold_call_raise() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        let actions = game.actions(&state);

        // SB facing BB: can fold, call, or raise
        assert!(
            actions.contains(&Action::Fold),
            "Actions should contain Fold"
        );
        assert!(
            actions.contains(&Action::Call),
            "Actions should contain Call"
        );
        // Should have at least one raise size
        assert!(
            actions.iter().any(|a| matches!(a, Action::Raise(_))),
            "Actions should contain at least one Raise"
        );
    }

    #[timed_test]
    fn fold_ends_game() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        let new_state = game.next_state(&state, Action::Fold);

        assert!(game.is_terminal(&new_state), "Fold should end the game");
        assert_eq!(
            new_state.terminal,
            Some(TerminalType::Fold(Player::Player1)),
            "Terminal should be Fold(Player1)"
        );
    }

    #[timed_test]
    fn check_check_advances_street() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        // SB calls (limps) - BB gets option
        let after_limp = game.next_state(&state, Action::Call);
        assert_eq!(
            after_limp.to_act,
            Some(Player::Player2),
            "After SB limp, BB should get option"
        );
        assert_eq!(
            after_limp.street,
            Street::Preflop,
            "Still preflop after limp"
        );

        // BB checks (option) - goes to flop
        let after_bb_check = game.next_state(&after_limp, Action::Check);
        assert_eq!(after_bb_check.street, Street::Flop, "BB check goes to flop");
        assert_eq!(
            after_bb_check.to_act,
            Some(Player::Player1),
            "SB acts first on flop"
        );

        // SB checks on flop
        let after_sb_check = game.next_state(&after_bb_check, Action::Check);
        assert_eq!(
            after_sb_check.to_act,
            Some(Player::Player2),
            "After SB check on flop, BB to act"
        );

        // BB checks on flop - advance to turn
        let after_bb_check_flop = game.next_state(&after_sb_check, Action::Check);
        assert_eq!(
            after_bb_check_flop.street,
            Street::Turn,
            "Both checks should advance street to Turn"
        );
    }

    #[timed_test]
    fn call_after_raise_advances_street() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        // Find a raise action
        let actions = game.actions(&state);
        let raise = actions
            .iter()
            .find(|a| matches!(a, Action::Raise(_)))
            .expect("Should have a raise action");

        // SB raises
        let after_raise = game.next_state(&state, *raise);
        assert_eq!(
            after_raise.to_act,
            Some(Player::Player2),
            "After SB raise, BB should be to act"
        );

        // BB calls
        let after_call = game.next_state(&after_raise, Action::Call);
        // Call after raise should advance to next street (flop)
        assert_eq!(
            after_call.street,
            Street::Flop,
            "Call after raise should advance to Flop"
        );
    }

    #[timed_test]
    fn initial_states_returns_random_deals() {
        let game = HunlPostflop::new(PostflopConfig::default(), None, 50);
        let states = game.initial_states();

        assert_eq!(states.len(), 50, "Should have 50 random deals");

        for state in &states {
            assert_eq!(state.street, Street::Preflop);
            assert!(state.terminal.is_none());
            assert_eq!(state.to_act, Some(Player::Player1));
            assert!(
                state.full_board.is_some(),
                "Random deal should have full_board"
            );
        }
    }

    #[timed_test]
    fn all_canonical_hands_generates_169() {
        let states = HunlPostflop::all_canonical_hands(100);
        assert_eq!(states.len(), 169, "Should have 169 canonical hands");
    }

    #[timed_test]
    fn fold_utility_correct() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        // SB folds (loses 0.5 BB)
        let folded = game.next_state(&state, Action::Fold);

        let p1_utility = game.utility(&folded, Player::Player1);
        let p2_utility = game.utility(&folded, Player::Player2);

        // SB posted 1 unit (0.5 BB), loses it
        assert!(
            (p1_utility - (-0.5)).abs() < 0.01,
            "P1 should lose 0.5 BB, got {p1_utility}"
        );
        assert!(
            (p2_utility - 0.5).abs() < 0.01,
            "P2 should win 0.5 BB, got {p2_utility}"
        );
    }

    #[timed_test]
    fn utilities_are_zero_sum() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        // Test fold
        let folded = game.next_state(&state, Action::Fold);
        let u1 = game.utility(&folded, Player::Player1);
        let u2 = game.utility(&folded, Player::Player2);
        assert!(
            (u1 + u2).abs() < 0.001,
            "Fold utilities should be zero-sum: {u1} + {u2} = {}",
            u1 + u2
        );
    }

    #[timed_test]
    fn info_set_key_different_for_different_hands() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        let info_set1 = game.info_set_key(&state);

        // After a raise, key should change (different player, different actions)
        let actions = game.actions(&state);
        let raise = actions
            .iter()
            .find(|a| matches!(a, Action::Raise(_)))
            .expect("Should have a raise");
        let after_raise = game.next_state(&state, *raise);

        let info_set2 = game.info_set_key(&after_raise);
        assert_ne!(
            info_set1, info_set2,
            "Info set should change after raise action"
        );
    }

    #[timed_test]
    fn actions_include_all_in() {
        let (p1, p2) = sample_holdings();
        let board = sample_board();
        let state = PostflopState::new_preflop_with_board(p1, p2, board, 100);
        let game = create_game();

        // Limp → BB check → flop
        let s = game.next_state(&state, Action::Call);
        let s = game.next_state(&s, Action::Check);

        let actions = game.actions(&s);
        // Should have an ALL_IN action since effective stack >> pot-fraction bets
        assert!(
            actions.iter().any(|a| matches!(a, Action::Bet(ALL_IN))),
            "Actions should include all-in bet: {actions:?}"
        );
    }

    #[timed_test]
    fn all_in_available_at_raise_cap() {
        let (p1, p2) = sample_holdings();
        let board = sample_board();
        let state = PostflopState::new_preflop_with_board(p1, p2, board, 100);
        let game = create_game(); // max_raises_per_street = 3

        // Limp → BB check → flop, then exhaust raise cap: bet → raise → raise
        let s = game.next_state(&state, Action::Call);
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Bet(0));
        let s = game.next_state(&s, Action::Raise(0));
        let s = game.next_state(&s, Action::Raise(0));

        // At raise cap: sized raises should be gone, but all-in must remain
        let actions = game.actions(&s);
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, Action::Raise(idx) if *idx != ALL_IN)),
            "Sized raises should not be available at raise cap: {actions:?}"
        );
        assert!(
            actions.iter().any(|a| matches!(a, Action::Raise(ALL_IN))),
            "All-in raise should be available at raise cap: {actions:?}"
        );
    }

    #[timed_test]
    fn resolve_bet_amount_all_in_returns_effective_stack() {
        let game = create_game();
        let amount = game.resolve_bet_amount(ALL_IN, 200, 500);
        assert_eq!(amount, 500, "ALL_IN should return effective stack");
    }

    #[timed_test]
    fn resolve_bet_amount_index_returns_pot_fraction() {
        let config = PostflopConfig {
            bet_sizes: vec![0.5, 1.0],
            ..PostflopConfig::default()
        };
        let game = HunlPostflop::new(config, None, 1);

        // Index 0 = 0.5 pot, pot=200, effective=1000
        let amount = game.resolve_bet_amount(0, 200, 1000);
        assert_eq!(amount, 100, "0.5 * 200 = 100");

        // Index 1 = 1.0 pot, pot=200, effective=1000
        let amount = game.resolve_bet_amount(1, 200, 1000);
        assert_eq!(amount, 200, "1.0 * 200 = 200");
    }

    #[timed_test]
    fn resolve_bet_amount_capped_at_effective_stack() {
        let config = PostflopConfig {
            bet_sizes: vec![1.0],
            ..PostflopConfig::default()
        };
        let game = HunlPostflop::new(config, None, 1);

        // 1.0 * 1000 = 1000, but effective stack is 50
        let amount = game.resolve_bet_amount(0, 1000, 50);
        assert_eq!(amount, 50, "Should cap at effective stack");
    }

    #[timed_test]
    fn info_set_key_uses_bet_index_encoding() {
        // A bet at index 0 and index 1 should produce different keys
        let config = PostflopConfig {
            stack_depth: 100,
            bet_sizes: vec![0.5, 1.0],
            ..PostflopConfig::default()
        };
        let game = HunlPostflop::new(config, None, 1);

        let (p1, p2) = sample_holdings();
        let board = sample_board();
        let state = PostflopState::new_preflop_with_board(p1, p2, board, 100);

        // Path 1: limp → check → bet index 0
        let s = game.next_state(&state, Action::Call);
        let s = game.next_state(&s, Action::Check);
        let s1 = game.next_state(&s, Action::Bet(0));
        let key1 = game.info_set_key(&s1);

        // Path 2: limp → check → bet index 1
        let s2 = game.next_state(&s, Action::Bet(1));
        let key2 = game.info_set_key(&s2);

        // Different bet indices → different keys
        assert_ne!(
            key1, key2,
            "Different bet indices should produce different keys"
        );
    }

    #[timed_test]
    fn raise_switches_action() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        let actions = game.actions(&state);
        let raise = actions
            .iter()
            .find(|a| matches!(a, Action::Raise(_)))
            .expect("Should have a raise");

        let after_raise = game.next_state(&state, *raise);

        assert!(!game.is_terminal(&after_raise));
        assert_eq!(game.player(&after_raise), Player::Player2);
    }

    #[timed_test]
    fn bet_after_check_allowed() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        // SB calls to go to flop
        let after_call = game.next_state(&state, Action::Call);

        // P2 (BB) should be able to check or bet
        let actions = game.actions(&after_call);
        assert!(actions.contains(&Action::Check), "Should be able to check");
        assert!(
            actions.iter().any(|a| matches!(a, Action::Bet(_))),
            "Should be able to bet"
        );
    }

    #[timed_test]
    fn river_check_check_is_showdown() {
        let (p1, p2) = sample_holdings();
        let mut state = PostflopState::new_preflop(p1, p2, 100);
        state.street = Street::River;
        state.to_call = 0;
        state.to_act = Some(Player::Player1);

        let game = create_game();

        // P1 checks
        let after_p1_check = game.next_state(&state, Action::Check);
        assert!(!game.is_terminal(&after_p1_check));

        // P2 checks
        let after_p2_check = game.next_state(&after_p1_check, Action::Check);
        assert!(
            game.is_terminal(&after_p2_check),
            "River check-check should be showdown"
        );
        assert_eq!(
            after_p2_check.terminal,
            Some(TerminalType::Showdown),
            "Terminal should be Showdown"
        );
    }

    #[timed_test]
    fn showdown_utility_uses_hand_evaluation() {
        // P1 has AA (Aces)
        let p1 = [
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::Ace, Suit::Heart),
        ];
        // P2 has 72o (Seven-Two offsuit - worst hand)
        let p2 = [
            make_card(Value::Seven, Suit::Club),
            make_card(Value::Two, Suit::Diamond),
        ];

        // Create a 5-card board that doesn't help P2
        // Board: Kh Qd Jc 8s 3h (no straight, flush, or pair for 72o)
        let board = ArrayVec::from([
            make_card(Value::King, Suit::Heart),
            make_card(Value::Queen, Suit::Diamond),
            make_card(Value::Jack, Suit::Club),
            make_card(Value::Eight, Suit::Spade),
            make_card(Value::Three, Suit::Heart),
        ]);

        let mut state = PostflopState::new_preflop(p1, p2, 100);
        state.board = board;
        state.street = Street::River;
        state.terminal = Some(TerminalType::Showdown);
        state.to_act = None;

        let game = create_game();

        // P1 should win with pair of Aces vs P2's high card Seven
        let p1_utility = game.utility(&state, Player::Player1);
        let p2_utility = game.utility(&state, Player::Player2);

        // P1 wins, so P1 utility should be positive
        assert!(
            p1_utility > 0.0,
            "P1 with AA should have positive utility, got {p1_utility}"
        );

        // P2 loses, so P2 utility should be negative
        assert!(
            p2_utility < 0.0,
            "P2 with 72o should have negative utility, got {p2_utility}"
        );

        // Utilities should be zero-sum
        assert!(
            (p1_utility + p2_utility).abs() < 0.001,
            "Utilities should be zero-sum: {p1_utility} + {p2_utility} = {}",
            p1_utility + p2_utility
        );
    }

    #[timed_test]
    fn showdown_utility_tie_splits_pot() {
        // Both players have same pocket pair
        let p1 = [
            make_card(Value::Nine, Suit::Spade),
            make_card(Value::Nine, Suit::Heart),
        ];
        let p2 = [
            make_card(Value::Nine, Suit::Club),
            make_card(Value::Nine, Suit::Diamond),
        ];

        // Board with higher cards - both make same hand
        // Board: AKQJT (broadway) - both have same straight
        let board = ArrayVec::from([
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::King, Suit::Diamond),
            make_card(Value::Queen, Suit::Club),
            make_card(Value::Jack, Suit::Heart),
            make_card(Value::Ten, Suit::Spade),
        ]);

        let mut state = PostflopState::new_preflop(p1, p2, 100);
        state.board = board;
        state.street = Street::River;
        state.terminal = Some(TerminalType::Showdown);
        state.to_act = None;

        let game = create_game();

        let p1_utility = game.utility(&state, Player::Player1);
        let p2_utility = game.utility(&state, Player::Player2);

        // Both should get zero (split pot equals their investment)
        // P1 invested 1 (0.5BB), P2 invested 2 (1BB)
        // With a tie, pot is split - this test verifies zero-sum property
        assert!(
            (p1_utility + p2_utility).abs() < 0.001,
            "Tie utilities should be zero-sum: {p1_utility} + {p2_utility} = {}",
            p1_utility + p2_utility
        );
    }

    // ==================== Board dealing tests ====================

    fn sample_board() -> [Card; 5] {
        [
            make_card(Value::King, Suit::Heart),
            make_card(Value::Queen, Suit::Diamond),
            make_card(Value::Jack, Suit::Club),
            make_card(Value::Eight, Suit::Spade),
            make_card(Value::Three, Suit::Heart),
        ]
    }

    #[timed_test]
    fn new_preflop_with_board_stores_full_board() {
        let (p1, p2) = sample_holdings();
        let board = sample_board();
        let state = PostflopState::new_preflop_with_board(p1, p2, board, 100);

        assert_eq!(state.full_board, Some(board));
        assert!(state.board.is_empty(), "Board should be empty at preflop");
        assert_eq!(state.street, Street::Preflop);
    }

    #[timed_test]
    fn advance_street_reveals_flop_cards() {
        let (p1, p2) = sample_holdings();
        let board = sample_board();
        let state = PostflopState::new_preflop_with_board(p1, p2, board, 100);
        let game = create_game();

        // SB calls, BB checks → advance to flop
        let after_limp = game.next_state(&state, Action::Call);
        let after_bb_check = game.next_state(&after_limp, Action::Check);

        assert_eq!(after_bb_check.street, Street::Flop);
        assert_eq!(after_bb_check.board.len(), 3, "Flop should have 3 cards");
        assert_eq!(after_bb_check.board[0], board[0]);
        assert_eq!(after_bb_check.board[1], board[1]);
        assert_eq!(after_bb_check.board[2], board[2]);
    }

    #[timed_test]
    fn advance_street_reveals_turn_card() {
        let (p1, p2) = sample_holdings();
        let board = sample_board();
        let state = PostflopState::new_preflop_with_board(p1, p2, board, 100);
        let game = create_game();

        // Limp → BB check → flop check-check → turn
        let s = game.next_state(&state, Action::Call);
        let s = game.next_state(&s, Action::Check); // → flop
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check); // → turn

        assert_eq!(s.street, Street::Turn);
        assert_eq!(s.board.len(), 4, "Turn should have 4 cards");
        assert_eq!(s.board[3], board[3]);
    }

    #[timed_test]
    fn advance_street_reveals_river_card() {
        let (p1, p2) = sample_holdings();
        let board = sample_board();
        let state = PostflopState::new_preflop_with_board(p1, p2, board, 100);
        let game = create_game();

        // Limp → BB check → check-check (flop) → check-check (turn) → river
        let s = game.next_state(&state, Action::Call);
        let s = game.next_state(&s, Action::Check); // → flop
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check); // → turn
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check); // → river

        assert_eq!(s.street, Street::River);
        assert_eq!(s.board.len(), 5, "River should have 5 cards");
        assert_eq!(s.board[4], board[4]);
    }

    #[timed_test]
    fn full_game_check_through_reaches_showdown() {
        let (p1, p2) = sample_holdings();
        let board = sample_board();
        let state = PostflopState::new_preflop_with_board(p1, p2, board, 100);
        let game = create_game();

        // Play through all 4 streets with limp + check-through
        let s = game.next_state(&state, Action::Call); // SB limps
        let s = game.next_state(&s, Action::Check); // BB checks → flop
        let s = game.next_state(&s, Action::Check); // SB checks
        let s = game.next_state(&s, Action::Check); // BB checks → turn
        let s = game.next_state(&s, Action::Check); // SB checks
        let s = game.next_state(&s, Action::Check); // BB checks → river
        let s = game.next_state(&s, Action::Check); // SB checks
        let s = game.next_state(&s, Action::Check); // BB checks → showdown

        assert!(
            game.is_terminal(&s),
            "Should reach terminal after all streets"
        );
        assert_eq!(s.terminal, Some(TerminalType::Showdown));
        assert_eq!(
            s.board.len(),
            5,
            "Board should have all 5 cards at showdown"
        );
    }

    #[timed_test]
    fn showdown_with_dealt_board_correct_winner() {
        // P1 has AA
        let p1 = [
            make_card(Value::Ace, Suit::Spade),
            make_card(Value::Ace, Suit::Heart),
        ];
        // P2 has 72o
        let p2 = [
            make_card(Value::Seven, Suit::Club),
            make_card(Value::Two, Suit::Diamond),
        ];
        // Board that doesn't help 72o
        let board = [
            make_card(Value::King, Suit::Heart),
            make_card(Value::Queen, Suit::Diamond),
            make_card(Value::Jack, Suit::Club),
            make_card(Value::Eight, Suit::Spade),
            make_card(Value::Three, Suit::Heart),
        ];

        let state = PostflopState::new_preflop_with_board(p1, p2, board, 100);
        let game = create_game();

        // Play through to showdown
        let s = game.next_state(&state, Action::Call);
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check);
        let s = game.next_state(&s, Action::Check);

        assert!(game.is_terminal(&s));
        let p1_util = game.utility(&s, Player::Player1);
        assert!(p1_util > 0.0, "AA should beat 72o, got {p1_util}");
    }

    #[timed_test]
    fn random_deals_no_card_conflicts() {
        let game = HunlPostflop::new(PostflopConfig::default(), None, 100);
        let deals = game.initial_states();

        for (i, deal) in deals.iter().enumerate() {
            let board = deal.full_board.expect("should have board");
            let mut all_cards = vec![
                deal.p1_holding[0],
                deal.p1_holding[1],
                deal.p2_holding[0],
                deal.p2_holding[1],
                board[0],
                board[1],
                board[2],
                board[3],
                board[4],
            ];
            let orig_len = all_cards.len();
            all_cards.sort_by_key(|c| (c.value as u8, c.suit as u8));
            all_cards.dedup();
            assert_eq!(all_cards.len(), orig_len, "Deal {i} has duplicate cards");
        }
    }

    #[timed_test]
    fn random_deals_deterministic() {
        let game = HunlPostflop::new(PostflopConfig::default(), None, 10);

        let deals1 = game.initial_states();
        let deals2 = game.initial_states();

        for (d1, d2) in deals1.iter().zip(deals2.iter()) {
            assert_eq!(d1.p1_holding, d2.p1_holding);
            assert_eq!(d1.p2_holding, d2.p2_holding);
            assert_eq!(d1.full_board, d2.full_board);
        }
    }

    #[timed_test]
    fn new_preflop_has_none_full_board() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        assert!(
            state.full_board.is_none(),
            "Plain new_preflop should have no full_board"
        );
    }

    #[timed_test]
    fn advance_street_without_board_leaves_board_empty() {
        let (p1, p2) = sample_holdings();
        let state = PostflopState::new_preflop(p1, p2, 100);
        let game = create_game();

        // SB calls → BB checks → flop (no board cards dealt without full_board)
        let s = game.next_state(&state, Action::Call);
        let s = game.next_state(&s, Action::Check);

        assert_eq!(s.street, Street::Flop);
        assert!(
            s.board.is_empty(),
            "Board should stay empty without full_board"
        );
    }

    // ==================== Stratification tests ====================

    #[timed_test]
    fn count_class_coverage_empty() {
        let coverage = super::count_class_coverage(&[]);
        assert_eq!(coverage, [0usize; HandClass::COUNT]);
    }

    #[timed_test]
    fn count_class_coverage_counts_both_players() {
        // P1: Ah Kh (flush on heart board)
        // P2: 7c 2d (low pair on this board)
        // Board: Qh Jh 5h 3h 2c → P1 has Flush, P2 has Pair (pairs the 2c)
        let p1 = [
            make_card(Value::Ace, Suit::Heart),
            make_card(Value::King, Suit::Heart),
        ];
        let p2 = [
            make_card(Value::Seven, Suit::Club),
            make_card(Value::Two, Suit::Diamond),
        ];
        let board = [
            make_card(Value::Queen, Suit::Heart),
            make_card(Value::Jack, Suit::Heart),
            make_card(Value::Five, Suit::Heart),
            make_card(Value::Three, Suit::Heart),
            make_card(Value::Two, Suit::Club),
        ];
        let deal = PostflopState::new_preflop_with_board(p1, p2, board, 100);
        let coverage = super::count_class_coverage(&[deal]);

        // P1 should have Flush
        assert!(
            coverage[HandClass::Flush as usize] > 0,
            "Flush should be counted"
        );
        // P2 pairs the 2 on board → Pair
        assert!(
            coverage[HandClass::Pair as usize] > 0,
            "Pair should be counted"
        );
    }

    #[timed_test(5)]
    fn stratified_zero_min_equals_base() {
        let config = PostflopConfig {
            stack_depth: 100,
            bet_sizes: vec![1.0],
            ..PostflopConfig::default()
        };
        // min=0 means stratification is disabled in initial_states
        let base_game = HunlPostflop::new(
            config.clone(),
            Some(AbstractionMode::HandClassV2 {
                strength_bits: 0,
                equity_bits: 0,
            }),
            50,
        );
        let strat_game = HunlPostflop::new(
            config,
            Some(AbstractionMode::HandClassV2 {
                strength_bits: 0,
                equity_bits: 0,
            }),
            50,
        )
        .with_stratification(0, 500_000);

        let base = base_game.initial_states();
        let strat = strat_game.initial_states();

        // Same length and same deals (stratification disabled when min=0)
        assert_eq!(base.len(), strat.len());
        for (b, s) in base.iter().zip(strat.iter()) {
            assert_eq!(b.p1_holding, s.p1_holding);
            assert_eq!(b.full_board, s.full_board);
        }
    }

    #[timed_test(15)]
    fn stratified_deals_deterministic() {
        let config = PostflopConfig {
            stack_depth: 100,
            bet_sizes: vec![1.0],
            ..PostflopConfig::default()
        };
        let game1 = HunlPostflop::new(
            config.clone(),
            Some(AbstractionMode::HandClassV2 {
                strength_bits: 0,
                equity_bits: 0,
            }),
            100,
        )
        .with_stratification(2, 10_000);
        let game2 = HunlPostflop::new(
            config,
            Some(AbstractionMode::HandClassV2 {
                strength_bits: 0,
                equity_bits: 0,
            }),
            100,
        )
        .with_stratification(2, 10_000);

        let deals1 = game1.initial_states();
        let deals2 = game2.initial_states();

        assert_eq!(deals1.len(), deals2.len());
        for (d1, d2) in deals1.iter().zip(deals2.iter()) {
            assert_eq!(d1.p1_holding, d2.p1_holding);
            assert_eq!(d1.p2_holding, d2.p2_holding);
            assert_eq!(d1.full_board, d2.full_board);
        }
    }

    #[timed_test(10)]
    fn stratified_deals_no_card_conflicts() {
        let config = PostflopConfig {
            stack_depth: 100,
            bet_sizes: vec![1.0],
            ..PostflopConfig::default()
        };
        let game = HunlPostflop::new(
            config,
            Some(AbstractionMode::HandClassV2 {
                strength_bits: 0,
                equity_bits: 0,
            }),
            100,
        )
        .with_stratification(2, 10_000);
        let deals = game.initial_states();

        for (i, deal) in deals.iter().enumerate() {
            let board = deal.full_board.expect("should have board");
            let mut all_cards = vec![
                deal.p1_holding[0],
                deal.p1_holding[1],
                deal.p2_holding[0],
                deal.p2_holding[1],
                board[0],
                board[1],
                board[2],
                board[3],
                board[4],
            ];
            let orig_len = all_cards.len();
            all_cards.sort_by_key(|c| (c.value as u8, c.suit as u8));
            all_cards.dedup();
            assert_eq!(all_cards.len(), orig_len, "Deal {i} has duplicate cards");
        }
    }

    #[timed_test(10)]
    fn stratified_deals_improve_rare_coverage() {
        // Use generate_stratified_deals directly with smaller parameters
        let config = PostflopConfig {
            stack_depth: 100,
            bet_sizes: vec![1.0],
            ..PostflopConfig::default()
        };
        let game = HunlPostflop::new(
            config,
            Some(AbstractionMode::HandClassV2 {
                strength_bits: 0,
                equity_bits: 0,
            }),
            200,
        );

        let base = game.generate_flop_deals(42, 200);
        let base_coverage = super::count_class_coverage(&base);

        let strat = game.generate_stratified_deals(42, 200, 2, 50_000);
        let strat_coverage = super::count_class_coverage(&strat);

        // Stratified pool should be at least as large
        assert!(
            strat.len() >= base.len(),
            "Stratified pool ({}) should be >= base ({})",
            strat.len(),
            base.len()
        );

        // Every made-hand class should have coverage >= base
        for i in 0..HandClass::NUM_MADE {
            assert!(
                strat_coverage[i] >= base_coverage[i],
                "Class {} coverage dropped: strat {} < base {}",
                i,
                strat_coverage[i],
                base_coverage[i]
            );
        }
    }

    #[timed_test(10)]
    fn stratified_deals_meet_minimum() {
        let config = PostflopConfig {
            stack_depth: 100,
            bet_sizes: vec![1.0],
            ..PostflopConfig::default()
        };
        let min_per_class = 2;
        let game = HunlPostflop::new(
            config,
            Some(AbstractionMode::HandClassV2 {
                strength_bits: 0,
                equity_bits: 0,
            }),
            200,
        );
        let deals = game.generate_stratified_deals(42, 200, min_per_class, 50_000);
        let coverage = super::count_class_coverage(&deals);

        // Common classes should easily meet the minimum
        let common = [
            HandClass::Pair as usize,
            HandClass::TwoPair as usize,
            HandClass::HighCard as usize,
        ];
        for &ci in &common {
            assert!(
                coverage[ci] >= min_per_class,
                "Common class {} has {} < {} minimum",
                ci,
                coverage[ci],
                min_per_class
            );
        }
    }

    #[timed_test]
    fn with_stratification_builder() {
        let config = PostflopConfig::default();
        let game = HunlPostflop::new(config, None, 100).with_stratification(50, 100_000);
        assert_eq!(game.min_deals_per_class, 50);
        assert_eq!(game.max_rejections_per_class, 100_000);
    }

    // ==================== decode_hole_pair tests ====================

    #[timed_test]
    fn decode_hole_pair_first_index() {
        let (a, b) = super::decode_hole_pair(0, 49);
        assert_eq!((a, b), (0, 1), "Index 0 should give pair (0, 1)");
    }

    #[timed_test]
    fn decode_hole_pair_last_index() {
        let (a, b) = super::decode_hole_pair(C49_2 - 1, 49);
        assert_eq!((a, b), (47, 48), "Last index should give pair (47, 48)");
    }

    #[timed_test]
    fn decode_hole_pair_round_trip_all_indices() {
        use std::collections::HashSet;
        let n = 49;
        let total = n * (n - 1) / 2;
        assert_eq!(total, C49_2);

        let mut seen = HashSet::new();
        for idx in 0..total {
            let (a, b) = super::decode_hole_pair(idx, n);
            assert!(a < b, "Expected a < b, got a={a}, b={b} at idx={idx}");
            assert!(b < n, "Expected b < {n}, got b={b} at idx={idx}");
            assert!(
                seen.insert((a, b)),
                "Duplicate pair ({a}, {b}) at idx={idx}"
            );
        }

        assert_eq!(
            seen.len(),
            total,
            "Should have exactly {total} unique pairs"
        );
    }

    #[timed_test]
    fn decode_hole_pair_small_n() {
        // C(4, 2) = 6 pairs: (0,1) (0,2) (1,2) (0,3) (1,3) (2,3)
        let expected = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)];
        for (idx, &exp) in expected.iter().enumerate() {
            let result = super::decode_hole_pair(idx, 4);
            assert_eq!(result, exp, "idx={idx}: expected {exp:?}, got {result:?}");
        }
    }

    // ==================== uniform deal generation tests ====================

    #[timed_test]
    fn generate_deals_for_single_flop_count() {
        let flops = crate::flops::all_flops();
        let deck = HunlPostflop::full_deck();

        let deals = super::generate_deals_for_flop(&flops[0], &deck, 42, 0, 25);

        assert_eq!(
            deals.len(),
            C49_2,
            "Should produce C(49,2)={} deals per flop, got {}",
            C49_2,
            deals.len()
        );
    }

    #[timed_test]
    fn generate_deals_for_flop_no_card_conflicts() {
        let flops = crate::flops::all_flops();
        let deck = HunlPostflop::full_deck();

        let deals = super::generate_deals_for_flop(&flops[0], &deck, 42, 0, 25);

        for (i, deal) in deals.iter().enumerate() {
            let board = deal.full_board.expect("should have board");
            let mut all_cards = vec![
                deal.p1_holding[0],
                deal.p1_holding[1],
                deal.p2_holding[0],
                deal.p2_holding[1],
                board[0],
                board[1],
                board[2],
                board[3],
                board[4],
            ];
            let orig_len = all_cards.len();
            all_cards.sort_by_key(|c| (c.value as u8, c.suit as u8));
            all_cards.dedup();
            assert_eq!(all_cards.len(), orig_len, "Deal {i} has duplicate cards");
        }
    }

    #[timed_test]
    fn generate_deals_for_flop_all_p1_pairs_unique() {
        use std::collections::HashSet;

        let flops = crate::flops::all_flops();
        let deck = HunlPostflop::full_deck();

        let deals = super::generate_deals_for_flop(&flops[0], &deck, 42, 0, 25);

        let mut seen = HashSet::new();
        for deal in &deals {
            let mut pair = [deal.p1_holding[0], deal.p1_holding[1]];
            pair.sort_by_key(|c| (c.value as u8, c.suit as u8));
            assert!(
                seen.insert((
                    pair[0].value as u8,
                    pair[0].suit as u8,
                    pair[1].value as u8,
                    pair[1].suit as u8
                )),
                "Duplicate P1 pair"
            );
        }
        assert_eq!(seen.len(), C49_2);
    }

    #[timed_test]
    fn generate_deals_for_flop_p1_uses_only_remaining_cards() {
        let flops = crate::flops::all_flops();
        let deck = HunlPostflop::full_deck();

        let deals = super::generate_deals_for_flop(&flops[0], &deck, 42, 0, 25);
        let flop_cards = *flops[0].cards();

        for deal in &deals {
            for &hole_card in &deal.p1_holding {
                assert!(
                    !flop_cards.contains(&hole_card),
                    "P1 hole card {:?} collides with flop {:?}",
                    hole_card,
                    flop_cards
                );
            }
        }
    }

    #[timed_test]
    fn generate_deals_for_flop_preserves_flop() {
        let flops = crate::flops::all_flops();
        let deck = HunlPostflop::full_deck();

        let deals = super::generate_deals_for_flop(&flops[42], &deck, 77, 42, 25);
        let flop_cards = *flops[42].cards();

        for deal in &deals {
            let board = deal.full_board.expect("should have board");
            assert_eq!(board[0], flop_cards[0]);
            assert_eq!(board[1], flop_cards[1]);
            assert_eq!(board[2], flop_cards[2]);
        }
    }

    #[timed_test]
    fn generate_deals_for_flop_p2_diversity() {
        use std::collections::HashSet;

        let flops = crate::flops::all_flops();
        let deck = HunlPostflop::full_deck();

        let deals = super::generate_deals_for_flop(&flops[0], &deck, 42, 0, 25);

        // For a given P1 combo, P2 should vary across different P1 combos
        // Check that not all deals have the same P2
        let p2_set: HashSet<_> = deals
            .iter()
            .map(|d| {
                (
                    d.p2_holding[0].value as u8,
                    d.p2_holding[0].suit as u8,
                    d.p2_holding[1].value as u8,
                    d.p2_holding[1].suit as u8,
                )
            })
            .collect();

        assert!(
            p2_set.len() > 1,
            "P2 holdings should vary across deals, but all {} deals have same P2",
            deals.len()
        );
    }

    #[timed_test]
    fn generate_deals_for_flop_deterministic() {
        let flops = crate::flops::all_flops();
        let deck = HunlPostflop::full_deck();

        let deals1 = super::generate_deals_for_flop(&flops[0], &deck, 42, 0, 25);
        let deals2 = super::generate_deals_for_flop(&flops[0], &deck, 42, 0, 25);

        for (d1, d2) in deals1.iter().zip(deals2.iter()) {
            assert_eq!(d1.p1_holding, d2.p1_holding);
            assert_eq!(d1.p2_holding, d2.p2_holding);
            assert_eq!(d1.full_board, d2.full_board);
        }
    }

    #[timed_test]
    fn with_uniform_deals_builder() {
        let config = PostflopConfig::default();
        let game = HunlPostflop::new(
            config,
            Some(AbstractionMode::HandClassV2 {
                strength_bits: 0,
                equity_bits: 0,
            }),
            1,
        )
        .with_uniform_deals();
        assert!(game.uniform_deals);
    }
}
