//! Terminal ASCII visualization of LHE preflop strategies.
//!
//! Renders a 13x13 hand matrix with color-coded action distributions.
//! Upper triangle = suited, lower triangle = offsuit, diagonal = pairs.
//!
//! Colors:
//! - Red = fold
//! - Green = call
//! - Yellow = raise/bet

use poker_solver_core::game::{
    Action, Game, LimitHoldem, LimitHoldemConfig, LimitHoldemState, Player,
};
use poker_solver_core::poker::{Card, Suit, Value};
use poker_solver_deep_cfr::eval::ExplicitPolicy;
use poker_solver_deep_cfr::lhe_encoder::LheEncoder;
use poker_solver_deep_cfr::traverse::StateEncoder;

use rand::SeedableRng;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Strategy frequencies for a single hand.
#[derive(Debug, Clone, Copy)]
pub struct HandStrategy {
    pub fold: f32,
    pub call: f32,
    pub raise: f32,
}

/// A 13x13 matrix of hand strategies.
///
/// Indexed by rank (A=0 down to 2=12) for rows and columns.
/// Upper triangle = suited, lower triangle = offsuit, diagonal = pairs.
pub type HandMatrix = [[Option<HandStrategy>; 13]; 13];

// ---------------------------------------------------------------------------
// Rank ordering: A=0, K=1, ..., 2=12
// ---------------------------------------------------------------------------

/// Rank values in display order: Ace(12) down to Two(0).
const RANK_ORDER: [Value; 13] = [
    Value::Ace,
    Value::King,
    Value::Queen,
    Value::Jack,
    Value::Ten,
    Value::Nine,
    Value::Eight,
    Value::Seven,
    Value::Six,
    Value::Five,
    Value::Four,
    Value::Three,
    Value::Two,
];

/// Display label for a rank.
fn rank_label(v: Value) -> &'static str {
    match v {
        Value::Ace => "A",
        Value::King => "K",
        Value::Queen => "Q",
        Value::Jack => "J",
        Value::Ten => "T",
        Value::Nine => "9",
        Value::Eight => "8",
        Value::Seven => "7",
        Value::Six => "6",
        Value::Five => "5",
        Value::Four => "4",
        Value::Three => "3",
        Value::Two => "2",
    }
}

/// Hand label for a matrix cell (e.g., "AKs", "QQ", "72o").
#[cfg(test)]
fn hand_label(row: usize, col: usize) -> String {
    let r1 = rank_label(RANK_ORDER[row]);
    let r2 = rank_label(RANK_ORDER[col]);
    if row == col {
        format!("{r1}{r2}")
    } else if row < col {
        format!("{r1}{r2}s")
    } else {
        format!("{r1}{r2}o")
    }
}

// ---------------------------------------------------------------------------
// Strategy computation
// ---------------------------------------------------------------------------

/// Configuration for computing a preflop strategy matrix.
struct MatrixConfig<'a> {
    policies: &'a [ExplicitPolicy; 2],
    lhe_config: &'a LimitHoldemConfig,
    board_samples: usize,
    seed: u64,
    target_player: Player,
    preceding_action: Option<Action>,
}

/// Build the preflop RFI (raise first in) strategy matrix for SB.
///
/// At the preflop root, SB faces: Fold / Call / Raise.
/// Averages over `board_samples` random boards for stability.
pub fn preflop_rfi_matrix(
    policies: &[ExplicitPolicy; 2],
    lhe_config: &LimitHoldemConfig,
    _num_actions: usize,
    board_samples: usize,
    seed: u64,
) -> HandMatrix {
    compute_preflop_matrix(&MatrixConfig {
        policies,
        lhe_config,
        board_samples,
        seed,
        target_player: Player::Player1,
        preceding_action: None,
    })
}

/// Build the preflop response matrix for BB facing SB raise.
///
/// After SB raises, BB faces: Fold / Call / Raise(3-bet).
/// Averages over `board_samples` random boards for stability.
pub fn preflop_response_matrix(
    policies: &[ExplicitPolicy; 2],
    lhe_config: &LimitHoldemConfig,
    _num_actions: usize,
    board_samples: usize,
    seed: u64,
) -> HandMatrix {
    compute_preflop_matrix(&MatrixConfig {
        policies,
        lhe_config,
        board_samples,
        seed,
        target_player: Player::Player2,
        preceding_action: Some(Action::Raise(0)),
    })
}

/// Generic preflop matrix computation.
///
/// Creates deals with specific hole cards for the target player,
/// optionally applies a preceding action, then queries the policy.
fn compute_preflop_matrix(cfg: &MatrixConfig<'_>) -> HandMatrix {
    let encoder = LheEncoder::new();
    let game = LimitHoldem::new(cfg.lhe_config.clone(), 1, cfg.seed);
    let mut matrix = [[None; 13]; 13];

    for (row, matrix_row) in matrix.iter_mut().enumerate() {
        for (col, cell) in matrix_row.iter_mut().enumerate() {
            *cell = Some(average_strategy_for_hand(&game, &encoder, cfg, row, col));
        }
    }

    matrix
}

/// Average strategy for a specific hand (row, col) over random boards.
fn average_strategy_for_hand(
    game: &LimitHoldem,
    encoder: &LheEncoder,
    cfg: &MatrixConfig<'_>,
    row: usize,
    col: usize,
) -> HandStrategy {
    let hole = make_hole_cards(row, col);
    let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(row as u64 * 13 + col as u64));
    let deck = build_remaining_deck(&hole);

    let mut fold_sum = 0.0f64;
    let mut call_sum = 0.0f64;
    let mut raise_sum = 0.0f64;
    let mut count = 0;

    for _ in 0..cfg.board_samples {
        let board = sample_board(&deck, &mut rng);
        let state = build_state(hole, board, cfg.target_player, cfg.lhe_config);

        let state = match cfg.preceding_action {
            Some(action) => game.next_state(&state, action),
            None => state,
        };

        let pi = player_index(cfg.target_player);
        let features = encoder.encode(&state, cfg.target_player);
        let probs = match cfg.policies[pi].strategy(&features) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let actions = game.actions(&state);
        let (f, c, r) = classify_action_probs(&actions, &probs);
        fold_sum += f64::from(f);
        call_sum += f64::from(c);
        raise_sum += f64::from(r);
        count += 1;
    }

    if count == 0 {
        return HandStrategy {
            fold: 1.0,
            call: 0.0,
            raise: 0.0,
        };
    }

    let n = count as f64;
    HandStrategy {
        fold: (fold_sum / n) as f32,
        call: (call_sum / n) as f32,
        raise: (raise_sum / n) as f32,
    }
}

/// Create hole cards for a given matrix cell.
fn make_hole_cards(row: usize, col: usize) -> [Card; 2] {
    let rank1 = RANK_ORDER[row];
    let rank2 = RANK_ORDER[col];
    let is_suited = row < col;

    let (suit1, suit2) = if is_suited {
        (Suit::Spade, Suit::Spade)
    } else {
        // Pairs and offsuit both use different suits
        (Suit::Spade, Suit::Heart)
    };

    [Card::new(rank1, suit1), Card::new(rank2, suit2)]
}

/// Build a state with specific hole cards for the target player.
///
/// Picks dummy opponent cards that don't collide with hole or board.
fn build_state(
    hole: [Card; 2],
    board: [Card; 5],
    target_player: Player,
    config: &LimitHoldemConfig,
) -> LimitHoldemState {
    let dummy_opp = pick_dummy_opponent(&hole, &board);

    match target_player {
        Player::Player1 => {
            LimitHoldemState::new_preflop(hole, dummy_opp, board, config.stack_depth)
        }
        Player::Player2 => {
            LimitHoldemState::new_preflop(dummy_opp, hole, board, config.stack_depth)
        }
    }
}

/// Pick two cards that don't collide with hole cards or board.
fn pick_dummy_opponent(hole: &[Card; 2], board: &[Card; 5]) -> [Card; 2] {
    let used: std::collections::HashSet<Card> =
        hole.iter().chain(board.iter()).copied().collect();
    let mut dummy = Vec::with_capacity(2);
    for card in poker_solver_core::poker::full_deck() {
        if !used.contains(&card) {
            dummy.push(card);
            if dummy.len() == 2 {
                break;
            }
        }
    }
    [dummy[0], dummy[1]]
}

/// Classify action probabilities into fold/call/raise buckets.
fn classify_action_probs(actions: &[Action], probs: &[f32]) -> (f32, f32, f32) {
    let mut fold = 0.0f32;
    let mut call = 0.0f32;
    let mut raise = 0.0f32;

    for (i, &action) in actions.iter().enumerate() {
        let p = if i < probs.len() { probs[i] } else { 0.0 };
        match action {
            Action::Fold => fold += p,
            Action::Check | Action::Call => call += p,
            Action::Bet(_) | Action::Raise(_) => raise += p,
        }
    }

    (fold, call, raise)
}

/// Build a deck without the given hole cards.
fn build_remaining_deck(hole: &[Card; 2]) -> Vec<Card> {
    poker_solver_core::poker::full_deck()
        .into_iter()
        .filter(|c| *c != hole[0] && *c != hole[1])
        .collect()
}

/// Sample 5 board cards from the remaining deck.
fn sample_board(deck: &[Card], rng: &mut StdRng) -> [Card; 5] {
    let mut pool = deck.to_vec();
    pool.shuffle(rng);
    [pool[0], pool[1], pool[2], pool[3], pool[4]]
}

/// Map Player enum to array index.
const fn player_index(player: Player) -> usize {
    match player {
        Player::Player1 => 0,
        Player::Player2 => 1,
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Bar width for the color-coded display.
const BAR_WIDTH: usize = 5;

/// ANSI color codes.
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";

/// Print a hand matrix with color-coded bars.
pub fn print_hand_matrix(matrix: &HandMatrix, title: &str) {
    println!("\n{BOLD}{title}{RESET}");
    println!();

    // Header row — "s" suffix marks the suited (upper-triangle) dimension
    print!("     ");
    for &rank in &RANK_ORDER {
        print!("{:>4}s ", rank_label(rank));
    }
    println!();

    for (row, matrix_row) in matrix.iter().enumerate() {
        print!("  {} ", rank_label(RANK_ORDER[row]));
        for cell in matrix_row {
            match cell {
                Some(s) => print!(" {}", render_bar(*s)),
                None => print!("  --- "),
            }
        }
        println!();
    }

    // Legend
    println!();
    println!(
        "  Legend: {RED}F{RESET}=fold {GREEN}C{RESET}=call/check {YELLOW}R{RESET}=raise/bet  |  Upper-right=suited  Lower-left=offsuit  Diagonal=pairs"
    );
}

/// Render a single cell as a stacked color bar of `BAR_WIDTH` characters.
fn render_bar(s: HandStrategy) -> String {
    let fold_chars = (s.fold * BAR_WIDTH as f32).round() as usize;
    let raise_chars = (s.raise * BAR_WIDTH as f32).round() as usize;
    let call_chars = BAR_WIDTH
        .saturating_sub(fold_chars)
        .saturating_sub(raise_chars);

    let mut bar = String::with_capacity(BAR_WIDTH + 20);
    for _ in 0..raise_chars {
        bar.push_str(YELLOW);
        bar.push('#');
    }
    for _ in 0..call_chars {
        bar.push_str(GREEN);
        bar.push('#');
    }
    for _ in 0..fold_chars {
        bar.push_str(RED);
        bar.push('.');
    }
    bar.push_str(RESET);

    bar
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // 1. Hand label correctness
    // -----------------------------------------------------------------------

    #[test]
    fn hand_label_pairs() {
        assert_eq!(hand_label(0, 0), "AA");
        assert_eq!(hand_label(1, 1), "KK");
        assert_eq!(hand_label(12, 12), "22");
    }

    #[test]
    fn hand_label_suited() {
        // Upper triangle: row < col
        assert_eq!(hand_label(0, 1), "AKs");
        assert_eq!(hand_label(0, 12), "A2s");
    }

    #[test]
    fn hand_label_offsuit() {
        // Lower triangle: row > col
        assert_eq!(hand_label(1, 0), "KAo");
        assert_eq!(hand_label(12, 0), "2Ao");
    }

    // -----------------------------------------------------------------------
    // 2. Classify action probabilities
    // -----------------------------------------------------------------------

    #[test]
    fn classify_fold_call_raise() {
        let actions = [Action::Fold, Action::Call, Action::Raise(0)];
        let probs = [0.2f32, 0.3, 0.5];
        let (f, c, r) = classify_action_probs(&actions, &probs);
        assert!((f - 0.2).abs() < 1e-6);
        assert!((c - 0.3).abs() < 1e-6);
        assert!((r - 0.5).abs() < 1e-6);
    }

    #[test]
    fn classify_check_bet() {
        let actions = [Action::Check, Action::Bet(0)];
        let probs = [0.7f32, 0.3];
        let (f, c, r) = classify_action_probs(&actions, &probs);
        assert!((f - 0.0).abs() < 1e-6);
        assert!((c - 0.7).abs() < 1e-6);
        assert!((r - 0.3).abs() < 1e-6);
    }

    #[test]
    fn classify_handles_empty_probs() {
        let actions = [Action::Fold, Action::Call];
        let probs: &[f32] = &[];
        let (f, c, r) = classify_action_probs(&actions, probs);
        assert!((f - 0.0).abs() < 1e-6);
        assert!((c - 0.0).abs() < 1e-6);
        assert!((r - 0.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // 3. Render bar
    // -----------------------------------------------------------------------

    #[test]
    fn render_bar_all_raise() {
        let s = HandStrategy {
            fold: 0.0,
            call: 0.0,
            raise: 1.0,
        };
        let bar = render_bar(s);
        assert!(bar.contains('#'));
        assert!(!bar.contains('.'));
    }

    #[test]
    fn render_bar_all_fold() {
        let s = HandStrategy {
            fold: 1.0,
            call: 0.0,
            raise: 0.0,
        };
        let bar = render_bar(s);
        assert!(bar.contains('.'));
    }

    // -----------------------------------------------------------------------
    // 4. Build remaining deck excludes hole cards
    // -----------------------------------------------------------------------

    #[test]
    fn remaining_deck_has_50_cards() {
        let hole = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let deck = build_remaining_deck(&hole);
        assert_eq!(deck.len(), 50);
        assert!(!deck.contains(&hole[0]));
        assert!(!deck.contains(&hole[1]));
    }

    // -----------------------------------------------------------------------
    // 5. Sample board returns 5 unique cards
    // -----------------------------------------------------------------------

    #[test]
    fn sample_board_returns_five_cards() {
        let hole = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let deck = build_remaining_deck(&hole);
        let mut rng = StdRng::seed_from_u64(42);
        let board = sample_board(&deck, &mut rng);

        assert_eq!(board.len(), 5);
        let unique: std::collections::HashSet<_> = board.iter().collect();
        assert_eq!(unique.len(), 5);
    }

    // -----------------------------------------------------------------------
    // 6. Rank ordering is correct
    // -----------------------------------------------------------------------

    #[test]
    fn rank_order_starts_with_ace() {
        assert_eq!(RANK_ORDER[0], Value::Ace);
        assert_eq!(RANK_ORDER[12], Value::Two);
    }

    // -----------------------------------------------------------------------
    // 7. Make hole cards
    // -----------------------------------------------------------------------

    #[test]
    fn make_hole_cards_suited() {
        let cards = make_hole_cards(0, 1); // AKs
        assert_eq!(cards[0].suit, Suit::Spade);
        assert_eq!(cards[1].suit, Suit::Spade);
    }

    #[test]
    fn make_hole_cards_offsuit() {
        let cards = make_hole_cards(1, 0); // KAo
        assert_ne!(cards[0].suit, cards[1].suit);
    }

    #[test]
    fn make_hole_cards_pair() {
        let cards = make_hole_cards(0, 0); // AA
        assert_ne!(cards[0].suit, cards[1].suit);
        assert_eq!(cards[0].value, cards[1].value);
    }
}
