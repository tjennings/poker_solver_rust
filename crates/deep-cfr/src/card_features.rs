//! Card feature encoding for Single Deep CFR.
//!
//! Provides suit-isomorphic card canonicalization and bet-history encoding,
//! producing tensors suitable for the advantage network.

use candle_core::{Device, Tensor};
use rs_poker::core::Card;

/// Sentinel value for cards not yet dealt (turn/river on earlier streets).
const ABSENT: i8 = -1;

/// Maximum actions per betting round.
const MAX_ACTIONS_PER_ROUND: usize = 6;

/// Number of betting rounds (preflop, flop, turn, river).
const NUM_ROUNDS: usize = 4;

/// Features per action slot: (occurred, pot_fraction).
const FEATURES_PER_SLOT: usize = 2;

/// Total bet feature dimensionality: 4 rounds x 6 slots x 2 features.
pub const BET_FEATURES: usize = NUM_ROUNDS * MAX_ACTIONS_PER_ROUND * FEATURES_PER_SLOT;

/// Pre-encoded features for one info set (one player's perspective at one decision point).
#[derive(Debug, Clone)]
pub struct InfoSetFeatures {
    /// Canonicalized card indices: [hole0, hole1, flop0, flop1, flop2, turn, river].
    /// Each value is 0..51 (rank * 4 + suit) or -1 for absent cards.
    pub cards: [i8; 7],
    /// Bet features: 4 rounds x 6 positions x (occurred, pot_fraction) = 48 floats.
    pub bets: [f32; BET_FEATURES],
}

impl InfoSetFeatures {
    /// Convert a batch of features to tensors for the neural network.
    ///
    /// Returns `(card_indices: [B, 7] i64, bets: [B, 48] f32)`.
    pub fn to_tensors(
        batch: &[InfoSetFeatures],
        device: &Device,
    ) -> Result<(Tensor, Tensor), candle_core::Error> {
        let b = batch.len();
        let card_data: Vec<i64> = batch
            .iter()
            .flat_map(|f| f.cards.iter().map(|&c| i64::from(c)))
            .collect();
        let bet_data: Vec<f32> = batch.iter().flat_map(|f| f.bets.iter().copied()).collect();

        let cards = Tensor::from_vec(card_data, &[b, 7], device)?;
        let bets = Tensor::from_vec(bet_data, &[b, BET_FEATURES], device)?;
        Ok((cards, bets))
    }
}

// ---------------------------------------------------------------------------
// Suit isomorphism
// ---------------------------------------------------------------------------

/// Build a suit permutation by assigning canonical ids in order of first
/// appearance, scanning: flop[0], flop[1], flop[2], hole[0], hole[1], turn, river.
///
/// Returns a 4-element map where `map[original_suit] = canonical_suit`.
/// Suits that never appear get the next available canonical id.
fn build_suit_map(hole: [Card; 2], board: &[Card]) -> [u8; 4] {
    let scan_order = board_then_hole_then_extra(hole, board);
    let mut map = [u8::MAX; 4];
    let mut next_id = 0u8;

    for card in scan_order {
        let s = card.suit as u8;
        if map[s as usize] == u8::MAX {
            map[s as usize] = next_id;
            next_id += 1;
        }
    }
    // Fill any unseen suits with remaining ids
    for slot in &mut map {
        if *slot == u8::MAX {
            *slot = next_id;
            next_id += 1;
        }
    }
    map
}

/// Produce the scan order: flop cards, then hole cards, then turn, then river.
fn board_then_hole_then_extra(hole: [Card; 2], board: &[Card]) -> Vec<Card> {
    let flop = &board[..board.len().min(3)];
    let turn = board.get(3);
    let river = board.get(4);

    let mut out = Vec::with_capacity(7);
    out.extend_from_slice(flop);
    out.extend_from_slice(&hole);
    if let Some(&t) = turn {
        out.push(t);
    }
    if let Some(&r) = river {
        out.push(r);
    }
    out
}

/// Encode a single card as `rank * 4 + canonical_suit` (0..51).
fn encode_card(card: Card, suit_map: &[u8; 4]) -> i8 {
    let rank = card.value as u8; // 0=Two .. 12=Ace
    let canonical_suit = suit_map[card.suit as usize];
    (rank * 4 + canonical_suit) as i8
}

/// Canonicalize hole cards and board using suit isomorphism.
///
/// Returns 7 card slots: `[hole0, hole1, flop0, flop1, flop2, turn, river]`.
/// Absent cards (turn/river not yet dealt) are encoded as -1.
pub fn canonicalize(hole: [Card; 2], board: &[Card]) -> [i8; 7] {
    let suit_map = build_suit_map(hole, board);
    let mut result = [ABSENT; 7];

    result[0] = encode_card(hole[0], &suit_map);
    result[1] = encode_card(hole[1], &suit_map);

    for (i, card) in board.iter().enumerate().take(5) {
        // flop = indices 2,3,4; turn = 5; river = 6
        result[i + 2] = encode_card(*card, &suit_map);
    }
    result
}

// ---------------------------------------------------------------------------
// Bet encoding
// ---------------------------------------------------------------------------

/// A single betting action paired with its pot-fraction at the time.
pub type BetAction = (poker_solver_core::Action, f64);

/// Encode betting history as a 48-element feature vector.
///
/// `action_history` contains `(Action, pot_fraction)` pairs ordered
/// chronologically. Actions are distributed across 4 rounds; a new round
/// starts when a `Check`/`Call` closes the current round's action or when
/// we run out of slots.
///
/// Each slot gets two features: `(occurred: 0|1, pot_fraction)`.
pub fn encode_bets(action_history: &[BetAction]) -> [f32; BET_FEATURES] {
    let mut bets = [0.0f32; BET_FEATURES];
    let mut round = 0usize;
    let mut slot = 0usize;

    for &(action, pot_frac) in action_history {
        if round >= NUM_ROUNDS {
            break;
        }
        if slot >= MAX_ACTIONS_PER_ROUND {
            // Overflow: advance to next round
            round += 1;
            slot = 0;
            if round >= NUM_ROUNDS {
                break;
            }
        }

        let base = (round * MAX_ACTIONS_PER_ROUND + slot) * FEATURES_PER_SLOT;
        bets[base] = 1.0;
        bets[base + 1] = pot_frac as f32;
        slot += 1;

        // A check or call closes the betting round (when it's the second
        // closing action, i.e. both players have acted). We use a simple
        // heuristic: call always advances, check advances only when it's
        // not the opening check of a round (slot > 1 means at least one
        // prior action exists in this round).
        let advances_round = match action {
            poker_solver_core::Action::Call => true,
            poker_solver_core::Action::Check if slot > 1 => true,
            _ => false,
        };
        if advances_round {
            round += 1;
            slot = 0;
        }
    }

    bets
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use poker_solver_core::Action;
    use rs_poker::core::{Card, Suit, Value};

    /// Helper: make a card from value and suit.
    fn card(value: Value, suit: Suit) -> Card {
        Card::new(value, suit)
    }

    #[test]
    fn isomorphic_hands_produce_same_features() {
        // Hand A: hole = Ah Kh, flop = Qh Jh Td
        let hole_a = [
            card(Value::Ace, Suit::Heart),
            card(Value::King, Suit::Heart),
        ];
        let board_a = [
            card(Value::Queen, Suit::Heart),
            card(Value::Jack, Suit::Heart),
            card(Value::Ten, Suit::Diamond),
        ];

        // Hand B: same ranks, but spades instead of hearts, clubs instead of diamonds
        let hole_b = [
            card(Value::Ace, Suit::Spade),
            card(Value::King, Suit::Spade),
        ];
        let board_b = [
            card(Value::Queen, Suit::Spade),
            card(Value::Jack, Suit::Spade),
            card(Value::Ten, Suit::Club),
        ];

        let canon_a = canonicalize(hole_a, &board_a);
        let canon_b = canonicalize(hole_b, &board_b);

        assert_eq!(
            canon_a, canon_b,
            "Isomorphic hands should produce identical canonical encodings.\n  A: {canon_a:?}\n  B: {canon_b:?}"
        );
    }

    #[test]
    fn absent_cards_encoded_as_negative_one() {
        let hole = [
            card(Value::Ace, Suit::Spade),
            card(Value::King, Suit::Heart),
        ];
        // Flop only, no turn or river
        let board = [
            card(Value::Queen, Suit::Diamond),
            card(Value::Jack, Suit::Club),
            card(Value::Ten, Suit::Spade),
        ];

        let canon = canonicalize(hole, &board);

        // Slots 0-4 should be valid (>= 0), turn and river should be -1
        for &c in &canon[..5] {
            assert!(c >= 0, "dealt card should not be -1, got {c}");
        }
        assert_eq!(canon[5], ABSENT, "turn should be -1 when not dealt");
        assert_eq!(canon[6], ABSENT, "river should be -1 when not dealt");
    }

    #[test]
    fn canonicalize_preserves_rank() {
        let hole = [
            card(Value::Ace, Suit::Heart),
            card(Value::King, Suit::Diamond),
        ];
        let board = [
            card(Value::Two, Suit::Spade),
            card(Value::Five, Suit::Club),
            card(Value::Ten, Suit::Heart),
            card(Value::Jack, Suit::Diamond),
            card(Value::Queen, Suit::Spade),
        ];

        let canon = canonicalize(hole, &board);
        let original_ranks = [
            Value::Ace as i8,   // hole0
            Value::King as i8,  // hole1
            Value::Two as i8,   // flop0
            Value::Five as i8,  // flop1
            Value::Ten as i8,   // flop2
            Value::Jack as i8,  // turn
            Value::Queen as i8, // river
        ];

        for (i, (&encoded, &expected_rank)) in canon.iter().zip(original_ranks.iter()).enumerate() {
            // rank = encoded / 4
            let decoded_rank = encoded / 4;
            assert_eq!(
                decoded_rank, expected_rank,
                "slot {i}: rank mismatch. encoded={encoded}, decoded_rank={decoded_rank}, expected={expected_rank}"
            );
        }
    }

    #[test]
    fn bet_encoding_correct_positions() {
        // Preflop: raise 2x pot, call
        let history: Vec<BetAction> = vec![(Action::Raise(0), 2.0), (Action::Call, 0.5)];

        let bets = encode_bets(&history);

        // Round 0, slot 0: occurred=1.0, pot_fraction=2.0
        assert_eq!(bets[0], 1.0, "round 0 slot 0 occurred");
        assert!(
            (bets[1] - 2.0).abs() < f32::EPSILON,
            "round 0 slot 0 pot_frac"
        );

        // Round 0, slot 1: occurred=1.0, pot_fraction=0.5
        assert_eq!(bets[2], 1.0, "round 0 slot 1 occurred");
        assert!(
            (bets[3] - 0.5).abs() < f32::EPSILON,
            "round 0 slot 1 pot_frac"
        );

        // Round 0, slot 2 onward should be zero (call advanced the round)
        for (i, &val) in bets.iter().enumerate().take(12).skip(4) {
            assert_eq!(val, 0.0, "round 0 slot {} should be zero", (i - 4) / 2 + 2);
        }
    }

    #[test]
    fn empty_action_history_all_zeros() {
        let bets = encode_bets(&[]);
        assert!(
            bets.iter().all(|&v| v == 0.0),
            "empty history should produce all-zero bet features"
        );
    }

    #[test]
    fn to_tensors_shape_is_correct() {
        let batch = vec![
            InfoSetFeatures {
                cards: [0, 4, 8, 12, 16, 20, 24],
                bets: [0.0; BET_FEATURES],
            },
            InfoSetFeatures {
                cards: [1, 5, 9, 13, 17, ABSENT, ABSENT],
                bets: [0.0; BET_FEATURES],
            },
            InfoSetFeatures {
                cards: [2, 6, 10, 14, 18, 22, ABSENT],
                bets: [0.0; BET_FEATURES],
            },
        ];

        let (card_tensor, bet_tensor) = InfoSetFeatures::to_tensors(&batch, &Device::Cpu).unwrap();

        assert_eq!(card_tensor.dims(), &[3, 7], "card tensor shape");
        assert_eq!(bet_tensor.dims(), &[3, BET_FEATURES], "bet tensor shape");
    }

    #[test]
    fn suit_map_assigns_in_first_appearance_order() {
        // flop: Qd Jc Ts, hole: Ah Kd
        // Scan: Qd(Diamond=3), Jc(Club=1), Ts(Spade=0), Ah(Heart=2), Kd(Diamond=3)
        // First appearances: Diamond->0, Club->1, Spade->2, Heart->3
        let hole = [
            card(Value::Ace, Suit::Heart),
            card(Value::King, Suit::Diamond),
        ];
        let board = [
            card(Value::Queen, Suit::Diamond),
            card(Value::Jack, Suit::Club),
            card(Value::Ten, Suit::Spade),
        ];

        let suit_map = build_suit_map(hole, &board);

        assert_eq!(suit_map[Suit::Diamond as usize], 0, "Diamond seen first");
        assert_eq!(suit_map[Suit::Club as usize], 1, "Club seen second");
        assert_eq!(suit_map[Suit::Spade as usize], 2, "Spade seen third");
        assert_eq!(suit_map[Suit::Heart as usize], 3, "Heart seen fourth");
    }

    #[test]
    fn encode_card_produces_rank_times_4_plus_suit() {
        let suit_map = [0, 1, 2, 3]; // identity map
        let c = card(Value::Ace, Suit::Spade); // rank=12, suit=0
        assert_eq!(encode_card(c, &suit_map), 48); // 12*4 + 0

        let c2 = card(Value::Two, Suit::Diamond); // rank=0, suit=3
        assert_eq!(encode_card(c2, &suit_map), 3); // 0*4 + 3
    }

    #[test]
    fn multi_round_bet_encoding() {
        // Preflop: bet, call -> round 0 done
        // Flop: check, bet, raise, call -> round 1 done
        let history: Vec<BetAction> = vec![
            (Action::Bet(0), 0.5),   // preflop slot 0
            (Action::Call, 0.5),     // preflop slot 1 -> advances
            (Action::Check, 0.0),    // flop slot 0
            (Action::Bet(0), 0.75),  // flop slot 1
            (Action::Raise(0), 2.0), // flop slot 2
            (Action::Call, 1.0),     // flop slot 3 -> advances
        ];

        let bets = encode_bets(&history);

        // Preflop: round=0, 2 slots filled
        assert_eq!(bets[0], 1.0); // round 0, slot 0, occurred
        assert_eq!(bets[2], 1.0); // round 0, slot 1, occurred
        assert_eq!(bets[4], 0.0); // round 0, slot 2, not used

        // Flop: round=1, 4 slots filled
        let flop_base = MAX_ACTIONS_PER_ROUND * FEATURES_PER_SLOT; // 12
        assert_eq!(bets[flop_base], 1.0); // round 1, slot 0 (check)
        assert_eq!(bets[flop_base + 1], 0.0); // pot_frac = 0
        assert_eq!(bets[flop_base + 2], 1.0); // round 1, slot 1 (bet)
        assert!((bets[flop_base + 3] - 0.75).abs() < f32::EPSILON);
        assert_eq!(bets[flop_base + 4], 1.0); // round 1, slot 2 (raise)
        assert!((bets[flop_base + 5] - 2.0).abs() < f32::EPSILON);
        assert_eq!(bets[flop_base + 6], 1.0); // round 1, slot 3 (call)
        assert!((bets[flop_base + 7] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn to_tensors_preserves_card_values() {
        let features = InfoSetFeatures {
            cards: [48, 44, 0, 4, 8, ABSENT, ABSENT],
            bets: [0.0; BET_FEATURES],
        };

        let (card_tensor, _) = InfoSetFeatures::to_tensors(&[features], &Device::Cpu).unwrap();
        let card_data = card_tensor.to_vec2::<i64>().unwrap();

        assert_eq!(card_data[0], vec![48, 44, 0, 4, 8, -1, -1]);
    }
}
