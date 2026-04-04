//! Game tree construction helpers.
//!
//! Extracted from `turn_generate.rs` so the domain pipeline can use them
//! without depending on the old monolithic datagen module.

use poker_solver_core::poker::{Card, Suit, Value};
use rand::Rng;
use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
use range_solver::bet_size::{BetSize, BetSizeOptions};
use range_solver::card::{CardConfig, NOT_DEALT};
use range_solver::game::PostFlopGame;
use range_solver::range::Range as RsRange;

use crate::config::BetSizeConfig;
use crate::datagen::range_gen::NUM_COMBOS;

/// Convert a range-solver `u8` card to an `rs_poker::core::Card`.
pub fn u8_to_rs_card(id: u8) -> Card {
    let rank = id / 4;
    let suit_id = id % 4;
    let value = Value::from(rank);
    let suit = match suit_id {
        0 => Suit::Club,
        1 => Suit::Diamond,
        2 => Suit::Heart,
        3 => Suit::Spade,
        _ => unreachable!(),
    };
    Card::new(value, suit)
}

/// Parse a single depth's bet size strings (e.g. `["50%", "100%", "a"]`) into pot fractions.
///
/// Entries like "a" (all-in) are skipped -- the game tree builder adds all-in
/// automatically.
fn parse_bet_sizes_depth(sizes: &[String]) -> Vec<f64> {
    sizes
        .iter()
        .filter_map(|s| {
            let trimmed = s.trim();
            if trimmed.eq_ignore_ascii_case("a") {
                return None;
            }
            let num_str = trimmed.trim_end_matches('%');
            num_str.parse::<f64>().ok().map(|v| v / 100.0)
        })
        .collect()
}

/// Parse all depths from a BetSizeConfig into `Vec<Vec<f64>>`.
pub fn parse_bet_sizes_all(config: &BetSizeConfig) -> Vec<Vec<f64>> {
    config
        .depths()
        .iter()
        .map(|d| parse_bet_sizes_depth(d))
        .collect()
}

/// Perturb bet sizes by multiplying each by `1.0 + uniform(-fuzz, +fuzz)`.
///
/// Returns the original sizes unchanged if `fuzz <= 0.0`.
/// Each fuzzed value is clamped to a minimum of 0.01 to avoid non-positive sizes.
pub fn fuzz_bet_sizes(
    bet_sizes: &[Vec<f64>],
    fuzz: f64,
    rng: &mut impl Rng,
) -> Vec<Vec<f64>> {
    if fuzz <= 0.0 {
        return bet_sizes.to_vec();
    }
    bet_sizes
        .iter()
        .map(|depth| {
            depth
                .iter()
                .map(|&size| {
                    let perturbation = 1.0 + rng.gen_range(-fuzz..fuzz);
                    (size * perturbation).max(0.01)
                })
                .collect()
        })
        .collect()
}

/// Build a turn game tree for a situation.
///
/// When `exact` is false (model mode): uses `depth_limit: Some(0)` to create boundary
/// nodes at the river transition, with empty river bet sizes.
/// When `exact` is true: uses `depth_limit: None` so the tree extends through the
/// river to showdown, with river bet sizes matching turn sizes.
///
/// Returns `None` for degenerate situations (effective_stack <= 0, or game construction fails).
fn build_turn_game_inner(
    board_u8: &[u8],
    pot: f64,
    effective_stack: f64,
    ranges: &[[f32; NUM_COMBOS]; 2],
    bet_sizes: &[Vec<f64>],
    exact: bool,
) -> Option<PostFlopGame> {
    let oop_range = RsRange::from_raw_data(&ranges[0]).expect("valid OOP range");
    let ip_range = RsRange::from_raw_data(&ranges[1]).expect("valid IP range");

    // bet_sizes[0] = first bet sizes, bet_sizes[1+] = raise sizes.
    let bet = bet_sizes
        .first()
        .map(|v| v.iter().map(|&f| BetSize::PotRelative(f)).collect())
        .unwrap_or_default();
    let raise = if bet_sizes.len() > 1 {
        bet_sizes[1..]
            .iter()
            .flat_map(|v| v.iter().map(|&f| BetSize::PotRelative(f)))
            .collect()
    } else {
        Vec::new()
    };
    let bet_size_opts = BetSizeOptions { bet, raise };

    let is_river = board_u8.len() >= 5;

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [board_u8[0], board_u8[1], board_u8[2]],
        turn: board_u8[3],
        river: if is_river { board_u8[4] } else { NOT_DEALT },
    };

    let (river_bet_sizes, depth_limit) = if exact || is_river {
        ([bet_size_opts.clone(), bet_size_opts.clone()], None)
    } else {
        (
            [BetSizeOptions::default(), BetSizeOptions::default()],
            Some(0),
        )
    };

    let initial_state = if is_river {
        BoardState::River
    } else {
        BoardState::Turn
    };

    let tree_config = TreeConfig {
        initial_state,
        starting_pot: pot as i32,
        effective_stack: effective_stack as i32,
        turn_bet_sizes: [bet_size_opts.clone(), bet_size_opts],
        river_bet_sizes,
        depth_limit,
        add_allin_threshold: 0.0,
        force_allin_threshold: 0.0,
        merging_threshold: 0.0,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).expect("valid action tree");
    let mut game = PostFlopGame::with_config(card_config, action_tree).expect("valid game");
    game.allocate_memory(true); // compressed (16-bit) storage to reduce memory ~4x
    use std::sync::atomic::{AtomicBool, Ordering as AO};
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, AO::Relaxed) {
        let (mem, _) = game.memory_usage();
        eprintln!(
            "[tree] memory per game: {:.1} MB",
            mem as f64 / 1_048_576.0
        );
    }
    Some(game)
}

/// Build a depth-limited turn game tree (model mode, with boundary nodes at river).
pub fn build_turn_game(
    board_u8: &[u8],
    pot: f64,
    effective_stack: f64,
    ranges: &[[f32; NUM_COMBOS]; 2],
    bet_sizes: &[Vec<f64>],
) -> Option<PostFlopGame> {
    build_turn_game_inner(board_u8, pot, effective_stack, ranges, bet_sizes, false)
}

/// Build a full turn+river game tree (exact mode, no boundaries).
pub fn build_turn_game_exact(
    board_u8: &[u8],
    pot: f64,
    effective_stack: f64,
    ranges: &[[f32; NUM_COMBOS]; 2],
    bet_sizes: &[Vec<f64>],
) -> Option<PostFlopGame> {
    build_turn_game_inner(board_u8, pot, effective_stack, ranges, bet_sizes, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_to_rs_card_converts_correctly() {
        // Card 0 = 2 of Clubs (rank=0, suit=0)
        let card = u8_to_rs_card(0);
        assert_eq!(card.value, Value::Two);
        assert_eq!(card.suit, Suit::Club);

        // Card 1 = 2 of Diamonds (rank=0, suit=1)
        let card = u8_to_rs_card(1);
        assert_eq!(card.value, Value::Two);
        assert_eq!(card.suit, Suit::Diamond);

        // Card 51 = Ace of Spades (rank=12, suit=3)
        let card = u8_to_rs_card(51);
        assert_eq!(card.value, Value::Ace);
        assert_eq!(card.suit, Suit::Spade);
    }

    #[test]
    fn parse_bet_sizes_depth_parses_percentages() {
        let sizes = vec!["50%".into(), "100%".into()];
        let parsed = parse_bet_sizes_depth(&sizes);
        assert_eq!(parsed.len(), 2);
        assert!((parsed[0] - 0.5).abs() < 1e-12);
        assert!((parsed[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn parse_bet_sizes_depth_skips_allin() {
        let sizes = vec!["50%".into(), "a".into(), "100%".into()];
        let parsed = parse_bet_sizes_depth(&sizes);
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn parse_bet_sizes_depth_skips_allin_case_insensitive() {
        let sizes = vec!["A".into(), "50%".into()];
        let parsed = parse_bet_sizes_depth(&sizes);
        assert_eq!(parsed.len(), 1);
        assert!((parsed[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn parse_bet_sizes_depth_handles_empty() {
        let sizes: Vec<String> = vec![];
        let parsed = parse_bet_sizes_depth(&sizes);
        assert!(parsed.is_empty());
    }

    #[test]
    fn fuzz_bet_sizes_noop_when_zero() {
        let sizes = vec![vec![0.5, 1.0]];
        let mut rng = rand::thread_rng();
        let result = fuzz_bet_sizes(&sizes, 0.0, &mut rng);
        assert_eq!(result, sizes);
    }

    #[test]
    fn fuzz_bet_sizes_noop_when_negative() {
        let sizes = vec![vec![0.5, 1.0]];
        let mut rng = rand::thread_rng();
        let result = fuzz_bet_sizes(&sizes, -0.1, &mut rng);
        assert_eq!(result, sizes);
    }

    #[test]
    fn fuzz_bet_sizes_perturbs_values() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let sizes = vec![vec![0.5, 1.0]];
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let result = fuzz_bet_sizes(&sizes, 0.2, &mut rng);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2);
        // Values should be perturbed but within 20% of original
        assert!((result[0][0] - 0.5).abs() < 0.5 * 0.2 + 0.01);
        assert!((result[0][1] - 1.0).abs() < 1.0 * 0.2 + 0.01);
    }

    #[test]
    fn fuzz_bet_sizes_clamps_minimum() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let sizes = vec![vec![0.001]]; // Very small value
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let result = fuzz_bet_sizes(&sizes, 0.99, &mut rng);
        assert!(result[0][0] >= 0.01, "should clamp to minimum 0.01");
    }
}
