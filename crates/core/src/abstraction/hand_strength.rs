//! Hand Strength Metrics for Card Abstraction
//!
//! Provides types for measuring hand strength including Expected Hand Strength (EHS),
//! potential metrics, and the EHS2 formula that combines current strength with
//! hand potential.

use crate::poker::{Card, Hand, Rank, Rankable, Suit, Value};
use std::collections::HashSet;

/// Hand strength metrics for poker hand evaluation.
///
/// This struct captures both the current hand strength and the potential
/// for improvement or degradation as more cards are dealt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandStrength {
    /// Expected Hand Strength - equity vs random opponent now
    pub ehs: f32,
    /// Positive potential - P(behind now, ahead after runout)
    pub ppot: f32,
    /// Negative potential - P(ahead now, behind after runout)
    pub npot: f32,
    /// `EHS2 = EHS + (1-EHS)*PPot - EHS*NPot`
    pub ehs2: f32,
}

impl HandStrength {
    /// Create a new `HandStrength` with all metrics.
    ///
    /// Computes EHS2 automatically from the provided values using:
    /// `EHS2 = EHS + (1-EHS)*PPot - EHS*NPot`
    ///
    /// # Arguments
    /// * `ehs` - Expected Hand Strength (current equity)
    /// * `ppot` - Positive potential (probability of improving from behind)
    /// * `npot` - Negative potential (probability of falling behind from ahead)
    ///
    /// # Examples
    /// ```
    /// use poker_solver_core::abstraction::HandStrength;
    ///
    /// let hs = HandStrength::new(0.5, 0.2, 0.1);
    /// assert!((hs.ehs2 - 0.55).abs() < 0.001);
    /// ```
    #[must_use]
    pub fn new(ehs: f32, ppot: f32, npot: f32) -> Self {
        let ehs2 = ehs + (1.0 - ehs) * ppot - ehs * npot;
        Self {
            ehs,
            ppot,
            npot,
            ehs2,
        }
    }

    /// Create a river hand strength (no potential, just EHS).
    ///
    /// On the river, there are no more cards to come, so positive and negative
    /// potential are both zero, and EHS2 equals EHS.
    ///
    /// # Arguments
    /// * `ehs` - Expected Hand Strength (current equity)
    ///
    /// # Examples
    /// ```
    /// use poker_solver_core::abstraction::HandStrength;
    ///
    /// let hs = HandStrength::river(0.75);
    /// assert_eq!(hs.ehs, 0.75);
    /// assert_eq!(hs.ehs2, 0.75);
    /// assert_eq!(hs.ppot, 0.0);
    /// assert_eq!(hs.npot, 0.0);
    /// ```
    #[must_use]
    pub fn river(ehs: f32) -> Self {
        Self {
            ehs,
            ppot: 0.0,
            npot: 0.0,
            ehs2: ehs,
        }
    }
}

/// Counters accumulated during opponent enumeration for EHS and potential.
#[derive(Default)]
struct EhsCounters {
    total_ahead: u64,
    total_tied: u64,
    total_behind: u64,
    ahead_stays_ahead: u64,
    ahead_falls_behind: u64,
    behind_moves_ahead: u64,
    behind_stays_behind: u64,
}

impl EhsCounters {
    /// Record the current-street comparison result.
    fn record_current(&mut self, our_rank: Rank, opp_rank: Rank) {
        match our_rank.cmp(&opp_rank) {
            std::cmp::Ordering::Greater => self.total_ahead += 1,
            std::cmp::Ordering::Less => self.total_behind += 1,
            std::cmp::Ordering::Equal => self.total_tied += 1,
        }
    }

    /// Record a runout comparison result given the current-street comparison.
    fn record_runout(
        &mut self,
        currently_ahead: bool,
        currently_behind: bool,
        our_rank: Rank,
        opp_rank: Rank,
    ) {
        let finally_ahead = our_rank > opp_rank;
        let tied_or_ahead = finally_ahead || our_rank == opp_rank;

        if currently_ahead {
            if tied_or_ahead {
                self.ahead_stays_ahead += 1;
            } else {
                self.ahead_falls_behind += 1;
            }
        } else if currently_behind {
            if tied_or_ahead {
                self.behind_moves_ahead += 1;
            } else {
                self.behind_stays_behind += 1;
            }
        }
    }

    /// Compute final EHS, `PPot`, `NPot` from accumulated counters.
    fn to_hand_strength(&self) -> HandStrength {
        let total = self.total_ahead + self.total_tied + self.total_behind;
        #[allow(clippy::cast_precision_loss)]
        let ehs = if total > 0 {
            (self.total_ahead as f32 + self.total_tied as f32 / 2.0) / total as f32
        } else {
            0.5
        };

        let ppot = compute_potential(
            self.behind_moves_ahead,
            self.behind_stays_behind,
            self.total_behind,
        );
        let npot = compute_potential(
            self.ahead_falls_behind,
            self.ahead_stays_ahead,
            self.total_ahead,
        );

        HandStrength::new(ehs, ppot, npot)
    }
}

/// Compute a potential metric: P(outcome changed) given a category total.
///
/// `changed` = count where outcome flipped, `stayed` = count where it didn't,
/// `category_total` = total current-street matchups in this category.
#[allow(clippy::cast_precision_loss)]
fn compute_potential(changed: u64, stayed: u64, category_total: u64) -> f32 {
    if category_total == 0 {
        return 0.0;
    }
    let outcomes = changed + stayed;
    if outcomes > 0 {
        changed as f32 / outcomes as f32
    } else {
        0.0
    }
}

/// Build the live deck (all cards not in the dead set).
fn build_deck(holding: (Card, Card), board: &[Card]) -> Vec<Card> {
    let mut dead: HashSet<Card> = HashSet::new();
    dead.insert(holding.0);
    dead.insert(holding.1);
    for &card in board {
        dead.insert(card);
    }
    all_cards().filter(|c| !dead.contains(c)).collect()
}

/// Build a hand from board + hole cards.
fn build_hand(board: &[Card], h1: Card, h2: Card) -> Vec<Card> {
    let mut hand: Vec<Card> = board.to_vec();
    hand.push(h1);
    hand.push(h2);
    hand
}

/// Enumerate flop runouts (turn + river) and record potential transitions.
#[allow(clippy::too_many_arguments)]
fn enumerate_flop_runouts(
    our_hand: &mut Vec<Card>,
    opp_hand: &mut Vec<Card>,
    deck: &[Card],
    opp1: Card,
    opp2: Card,
    currently_ahead: bool,
    currently_behind: bool,
    counters: &mut EhsCounters,
) {
    our_hand.reserve(2);
    opp_hand.reserve(2);

    for (ti, &turn) in deck.iter().enumerate() {
        if turn == opp1 || turn == opp2 {
            continue;
        }
        for &river in deck.iter().skip(ti + 1) {
            if river == opp1 || river == opp2 {
                continue;
            }

            our_hand.push(turn);
            our_hand.push(river);
            let our_rank = Hand::new_with_cards(our_hand.clone()).rank();
            our_hand.pop();
            our_hand.pop();

            opp_hand.push(turn);
            opp_hand.push(river);
            let opp_rank = Hand::new_with_cards(opp_hand.clone()).rank();
            opp_hand.pop();
            opp_hand.pop();

            counters.record_runout(currently_ahead, currently_behind, our_rank, opp_rank);
        }
    }
}

/// Enumerate turn runouts (river only) and record potential transitions.
#[allow(clippy::too_many_arguments)]
fn enumerate_turn_runouts(
    our_hand: &mut Vec<Card>,
    opp_hand: &mut Vec<Card>,
    deck: &[Card],
    opp1: Card,
    opp2: Card,
    currently_ahead: bool,
    currently_behind: bool,
    counters: &mut EhsCounters,
) {
    our_hand.reserve(1);
    opp_hand.reserve(1);

    for &river in deck {
        if river == opp1 || river == opp2 {
            continue;
        }

        our_hand.push(river);
        let our_rank = Hand::new_with_cards(our_hand.clone()).rank();
        our_hand.pop();

        opp_hand.push(river);
        let opp_rank = Hand::new_with_cards(opp_hand.clone()).rank();
        opp_hand.pop();

        counters.record_runout(currently_ahead, currently_behind, our_rank, opp_rank);
    }
}

/// Calculator for hand strength metrics.
///
/// Provides methods for computing Expected Hand Strength (EHS) and related
/// metrics at different streets.
#[derive(Debug)]
pub struct HandStrengthCalculator;

impl HandStrengthCalculator {
    /// Create a new `HandStrengthCalculator`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Calculate EHS2 on the flop (enumerate turn+river cards).
    ///
    /// This is the most expensive calculation (~1M evaluations per holding).
    /// Enumerates all possible opponent holdings and turn+river cards to calculate
    /// hand strength with positive and negative potential.
    #[must_use]
    pub fn calculate_flop(&self, board: &[Card], holding: (Card, Card)) -> HandStrength {
        let deck = build_deck(holding, board);
        let mut our_hand = build_hand(board, holding.0, holding.1);
        let mut counters = EhsCounters::default();

        for (i, &opp1) in deck.iter().enumerate() {
            for &opp2 in deck.iter().skip(i + 1) {
                let mut opp_hand = build_hand(board, opp1, opp2);
                let our_rank = Hand::new_with_cards(our_hand.clone()).rank();
                let opp_rank = Hand::new_with_cards(opp_hand.clone()).rank();

                counters.record_current(our_rank, opp_rank);
                let currently_ahead = our_rank > opp_rank;
                let currently_behind = our_rank < opp_rank;

                enumerate_flop_runouts(
                    &mut our_hand,
                    &mut opp_hand,
                    &deck,
                    opp1,
                    opp2,
                    currently_ahead,
                    currently_behind,
                    &mut counters,
                );

                our_hand.truncate(board.len() + 2);
            }
        }

        counters.to_hand_strength()
    }

    /// Calculate EHS2 on the turn (enumerate river cards).
    ///
    /// Enumerates all possible opponent holdings and river cards to calculate
    /// hand strength with positive and negative potential.
    #[must_use]
    pub fn calculate_turn(&self, board: &[Card], holding: (Card, Card)) -> HandStrength {
        let deck = build_deck(holding, board);
        let mut our_hand = build_hand(board, holding.0, holding.1);
        let mut counters = EhsCounters::default();

        for (i, &opp1) in deck.iter().enumerate() {
            for &opp2 in deck.iter().skip(i + 1) {
                let mut opp_hand = build_hand(board, opp1, opp2);
                let our_rank = Hand::new_with_cards(our_hand.clone()).rank();
                let opp_rank = Hand::new_with_cards(opp_hand.clone()).rank();

                counters.record_current(our_rank, opp_rank);
                let currently_ahead = our_rank > opp_rank;
                let currently_behind = our_rank < opp_rank;

                enumerate_turn_runouts(
                    &mut our_hand,
                    &mut opp_hand,
                    &deck,
                    opp1,
                    opp2,
                    currently_ahead,
                    currently_behind,
                    &mut counters,
                );
            }
        }

        counters.to_hand_strength()
    }

    /// Calculate EHS on the river using exhaustive enumeration.
    ///
    /// Enumerates all possible opponent holdings and calculates the probability
    /// of winning against a random opponent hand.
    ///
    /// # Arguments
    /// * `board` - The 5 community cards on the river
    /// * `holding` - The player's two hole cards
    ///
    /// # Returns
    /// A `HandStrength` with river EHS (no potential, since no more cards to come).
    #[must_use]
    pub fn calculate_river(&self, board: &[Card], holding: (Card, Card)) -> HandStrength {
        let (h1, h2) = holding;

        // Build set of dead cards
        let mut dead: HashSet<Card> = HashSet::new();
        dead.insert(h1);
        dead.insert(h2);
        for &card in board {
            dead.insert(card);
        }

        // Our 7-card hand
        let mut our_cards: Vec<Card> = board.to_vec();
        our_cards.push(h1);
        our_cards.push(h2);
        let our_hand = Hand::new_with_cards(our_cards);
        let our_rank = our_hand.rank();

        // Enumerate all opponent holdings
        let mut wins = 0u32;
        let mut ties = 0u32;
        let mut losses = 0u32;

        let all_cards_vec: Vec<Card> = all_cards().collect();

        for (i, &opp1) in all_cards_vec.iter().enumerate() {
            if dead.contains(&opp1) {
                continue;
            }
            for &opp2 in all_cards_vec.iter().skip(i + 1) {
                if dead.contains(&opp2) {
                    continue;
                }

                // Opponent's 7-card hand
                let mut opp_cards: Vec<Card> = board.to_vec();
                opp_cards.push(opp1);
                opp_cards.push(opp2);
                let opp_hand = Hand::new_with_cards(opp_cards);
                let opp_rank = opp_hand.rank();

                match our_rank.cmp(&opp_rank) {
                    std::cmp::Ordering::Greater => wins += 1,
                    std::cmp::Ordering::Less => losses += 1,
                    std::cmp::Ordering::Equal => ties += 1,
                }
            }
        }

        let total = wins + ties + losses;
        #[allow(clippy::cast_precision_loss)]
        let ehs = if total > 0 {
            // Precision loss is acceptable for EHS calculation (max ~990 opponent combinations)
            (wins as f32 + ties as f32 / 2.0) / total as f32
        } else {
            0.5
        };

        HandStrength::river(ehs)
    }
}

impl Default for HandStrengthCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate an iterator over all 52 cards in a standard deck.
fn all_cards() -> impl Iterator<Item = Card> {
    const VALUES: [Value; 13] = [
        Value::Two,
        Value::Three,
        Value::Four,
        Value::Five,
        Value::Six,
        Value::Seven,
        Value::Eight,
        Value::Nine,
        Value::Ten,
        Value::Jack,
        Value::Queen,
        Value::King,
        Value::Ace,
    ];
    const SUITS: [Suit; 4] = [Suit::Spade, Suit::Heart, Suit::Diamond, Suit::Club];

    VALUES
        .into_iter()
        .flat_map(move |v| SUITS.into_iter().map(move |s| Card::new(v, s)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn river_ehs2_equals_ehs() {
        let hs = HandStrength::river(0.75);
        assert_eq!(hs.ehs2, hs.ehs);
        assert_eq!(hs.ppot, 0.0);
        assert_eq!(hs.npot, 0.0);
    }

    #[timed_test]
    fn ehs2_formula_correct() {
        let hs = HandStrength::new(0.5, 0.2, 0.1);
        // EHS2 = 0.5 + (1-0.5)*0.2 - 0.5*0.1 = 0.5 + 0.1 - 0.05 = 0.55
        assert!((hs.ehs2 - 0.55).abs() < 0.001);
    }

    #[timed_test]
    fn river_with_zero_ehs() {
        let hs = HandStrength::river(0.0);
        assert_eq!(hs.ehs, 0.0);
        assert_eq!(hs.ehs2, 0.0);
    }

    #[timed_test]
    fn river_with_full_ehs() {
        let hs = HandStrength::river(1.0);
        assert_eq!(hs.ehs, 1.0);
        assert_eq!(hs.ehs2, 1.0);
    }

    #[timed_test]
    fn ehs2_with_high_positive_potential() {
        // A drawing hand with high positive potential
        let hs = HandStrength::new(0.3, 0.5, 0.0);
        // EHS2 = 0.3 + (1-0.3)*0.5 - 0.3*0.0 = 0.3 + 0.35 = 0.65
        assert!((hs.ehs2 - 0.65).abs() < 0.001);
    }

    #[timed_test]
    fn ehs2_with_high_negative_potential() {
        // A vulnerable hand with high negative potential
        let hs = HandStrength::new(0.8, 0.0, 0.3);
        // EHS2 = 0.8 + (1-0.8)*0.0 - 0.8*0.3 = 0.8 - 0.24 = 0.56
        assert!((hs.ehs2 - 0.56).abs() < 0.001);
    }

    #[timed_test]
    fn hand_strength_is_copy() {
        let hs1 = HandStrength::new(0.5, 0.2, 0.1);
        let hs2 = hs1;
        assert_eq!(hs1, hs2);
    }

    #[timed_test]
    fn hand_strength_is_clone() {
        let hs1 = HandStrength::new(0.5, 0.2, 0.1);
        let hs2 = hs1;
        assert_eq!(hs1, hs2);
    }

    #[timed_test]
    fn hand_strength_debug() {
        let hs = HandStrength::river(0.5);
        let debug_str = format!("{hs:?}");
        assert!(debug_str.contains("HandStrength"));
        assert!(debug_str.contains("0.5"));
    }

    #[timed_test]
    fn river_ehs_nut_flush_near_one() {
        // As Ks on 2s 5s 8s Tc Qd - nut flush
        let board = vec![
            Card::new(Value::Two, Suit::Spade),
            Card::new(Value::Five, Suit::Spade),
            Card::new(Value::Eight, Suit::Spade),
            Card::new(Value::Ten, Suit::Club),
            Card::new(Value::Queen, Suit::Diamond),
        ];
        let holding = (
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        );

        let calc = HandStrengthCalculator::new();
        let hs = calc.calculate_river(&board, holding);

        // Nut flush should have very high equity
        assert!(hs.ehs > 0.90, "Expected EHS > 0.90, got {}", hs.ehs);
    }

    #[timed_test]
    fn river_ehs_weak_hand_low() {
        // 7h 2c on As Ks Qs Js 9d - no pair, no flush
        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Spade),
            Card::new(Value::Jack, Suit::Spade),
            Card::new(Value::Nine, Suit::Diamond),
        ];
        let holding = (
            Card::new(Value::Seven, Suit::Heart),
            Card::new(Value::Two, Suit::Club),
        );

        let calc = HandStrengthCalculator::new();
        let hs = calc.calculate_river(&board, holding);

        // Weak hand on scary board should have low equity
        assert!(hs.ehs < 0.30, "Expected EHS < 0.30, got {}", hs.ehs);
    }

    #[timed_test]
    fn all_cards_generates_52_cards() {
        let cards: Vec<Card> = all_cards().collect();
        assert_eq!(cards.len(), 52);
    }

    #[timed_test]
    fn all_cards_generates_unique_cards() {
        let cards: Vec<Card> = all_cards().collect();
        let unique: HashSet<Card> = cards.into_iter().collect();
        assert_eq!(unique.len(), 52);
    }

    #[timed_test]
    fn calculator_default_trait() {
        let calc = HandStrengthCalculator;
        // Just verify we can create via default
        let board = vec![
            Card::new(Value::Two, Suit::Spade),
            Card::new(Value::Three, Suit::Heart),
            Card::new(Value::Four, Suit::Diamond),
            Card::new(Value::Five, Suit::Club),
            Card::new(Value::Six, Suit::Spade),
        ];
        let holding = (
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Heart),
        );
        let hs = calc.calculate_river(&board, holding);
        // Just check it computed something valid
        assert!(hs.ehs >= 0.0 && hs.ehs <= 1.0);
    }

    #[timed_test]
    fn turn_ehs2_flush_draw_higher_than_ehs() {
        // Flush draw: As 5s on 2s 8s Tc Qd (4 to flush on turn)
        let board = vec![
            Card::new(Value::Two, Suit::Spade),
            Card::new(Value::Eight, Suit::Spade),
            Card::new(Value::Ten, Suit::Club),
            Card::new(Value::Queen, Suit::Diamond),
        ];
        let holding = (
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Five, Suit::Spade),
        );

        let calc = HandStrengthCalculator::new();
        let hs = calc.calculate_turn(&board, holding);

        // Flush draw should have positive potential
        assert!(hs.ppot > 0.1, "Expected PPot > 0.1, got {}", hs.ppot);
        assert!(
            hs.ehs2 > hs.ehs,
            "Expected EHS2 > EHS for flush draw, got EHS2={}, EHS={}",
            hs.ehs2,
            hs.ehs
        );
    }

    #[timed_test]
    fn turn_ehs2_made_hand_npot() {
        // Top pair on draw-heavy board: Ah Kc on As 8s 7s 2d
        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Eight, Suit::Spade),
            Card::new(Value::Seven, Suit::Spade),
            Card::new(Value::Two, Suit::Diamond),
        ];
        let holding = (
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Club),
        );

        let calc = HandStrengthCalculator::new();
        let hs = calc.calculate_turn(&board, holding);

        // Made hand on flush board should have negative potential
        assert!(hs.npot > 0.05, "Expected NPot > 0.05, got {}", hs.npot);
    }

    #[timed_test(60)]
    #[ignore = "slow"]
    fn flop_ehs2_open_ended_straight_draw() {
        // 9h 8h on 7c 6d 2s - open-ended straight draw
        let board = vec![
            Card::new(Value::Seven, Suit::Club),
            Card::new(Value::Six, Suit::Diamond),
            Card::new(Value::Two, Suit::Spade),
        ];
        let holding = (
            Card::new(Value::Nine, Suit::Heart),
            Card::new(Value::Eight, Suit::Heart),
        );

        let calc = HandStrengthCalculator::new();
        let hs = calc.calculate_flop(&board, holding);

        // Straight draw should have significant positive potential
        assert!(
            hs.ppot > 0.15,
            "Expected PPot > 0.15 for OESD, got {}",
            hs.ppot
        );
        assert!(hs.ehs2 > hs.ehs, "Expected EHS2 > EHS for draw");
    }
}
