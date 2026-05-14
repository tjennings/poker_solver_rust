//! Board-aware nut-distance diagnostics for postflop bucket quality.
//!
//! This module starts with river-only features. It is intentionally explicit
//! rather than clever: enumerate legal opponent holdings on a complete board
//! and measure how many same-family hands dominate the hero hand.

use std::collections::BTreeSet;
use std::fmt;

use crate::hand_class::{HandClass, classify};
use crate::poker::{ALL_SUITS, ALL_VALUES, Card};
use crate::showdown_equity::{rank_hand, rank_to_ordinal};

/// Error returned when nut features cannot be computed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NutFeatureError {
    /// Nut features are currently river-only.
    InvalidBoardSize(usize),
    /// Hand classification failed for the hero or an opponent holding.
    Classification(String),
}

impl fmt::Display for NutFeatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBoardSize(size) => {
                write!(f, "nut features require a 5-card river board, got {size}")
            }
            Self::Classification(msg) => write!(f, "hand classification failed: {msg}"),
        }
    }
}

impl std::error::Error for NutFeatureError {}

/// River nut-distance features for one `(hole, board)` situation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NutFeatures {
    /// Strongest made-hand family for this situation.
    pub made_class: HandClass,
    /// Number of distinct stronger same-family hand ordinals available to
    /// legal opponent holdings.
    pub class_gap: u16,
    /// Number of legal same-family opponent combos that beat this hand.
    pub same_family_beaters: u16,
    /// Number of legal opponent combos in the same made-hand family.
    pub same_family_total: u16,
    /// Fraction of all legal opponent combos beaten, ties counted half.
    pub global_rank_percentile: f32,
    /// Fraction of same-family opponent combos that beat this hand.
    pub dominance_margin: f32,
    /// Whether at least one hero card blocks a board-legal nut combo.
    pub blocker_to_nuts: bool,
    /// River has no future redraws; kept in the feature contract for street
    /// compatibility.
    pub redraw_to_nuts: bool,
}

/// Compute river nut-distance features for a hero holding on a complete board.
///
/// # Errors
///
/// Returns [`NutFeatureError::InvalidBoardSize`] unless `board.len() == 5`, or
/// [`NutFeatureError::Classification`] if local hand classification fails.
pub fn river_nut_features(hole: [Card; 2], board: &[Card]) -> Result<NutFeatures, NutFeatureError> {
    if board.len() != 5 {
        return Err(NutFeatureError::InvalidBoardSize(board.len()));
    }

    let hero_class = made_class(hole, board)?;
    let hero_rank = rank_hand(hole, board);
    let hero_ord = rank_to_ordinal(hero_rank);
    let legal_opp = legal_opponent_combos(hole, board);
    let board_legal = board_legal_combos(board);
    let nut_ord = board_legal
        .iter()
        .map(|&opp| rank_to_ordinal(rank_hand(opp, board)))
        .max()
        .unwrap_or(hero_ord);

    let mut same_family_total = 0_u16;
    let mut same_family_beaters = 0_u16;
    let mut stronger_same_family_ordinals = BTreeSet::new();
    let mut wins = 0_u16;
    let mut ties = 0_u16;

    for opp in &legal_opp {
        let opp_rank = rank_hand(*opp, board);
        let opp_ord = rank_to_ordinal(opp_rank);
        match hero_ord.cmp(&opp_ord) {
            std::cmp::Ordering::Greater => wins += 1,
            std::cmp::Ordering::Equal => ties += 1,
            std::cmp::Ordering::Less => {}
        }

        let opp_class = made_class(*opp, board)?;
        if opp_class == hero_class {
            same_family_total += 1;
            if opp_ord > hero_ord {
                same_family_beaters += 1;
                stronger_same_family_ordinals.insert(opp_ord);
            }
        }
    }

    let global_rank_percentile = if legal_opp.is_empty() {
        0.5
    } else {
        (f32::from(wins) + f32::from(ties) * 0.5) / legal_opp.len() as f32
    };
    let dominance_margin = if same_family_total == 0 {
        0.0
    } else {
        f32::from(same_family_beaters) / f32::from(same_family_total)
    };

    Ok(NutFeatures {
        made_class: hero_class,
        class_gap: stronger_same_family_ordinals.len() as u16,
        same_family_beaters,
        same_family_total,
        global_rank_percentile,
        dominance_margin,
        blocker_to_nuts: blocks_nut_combo(hole, board, nut_ord),
        redraw_to_nuts: false,
    })
}

fn made_class(hole: [Card; 2], board: &[Card]) -> Result<HandClass, NutFeatureError> {
    let classification =
        classify(hole, board).map_err(|e| NutFeatureError::Classification(e.to_string()))?;
    let class_id = classification.strongest_made_id();
    HandClass::from_discriminant(class_id).ok_or_else(|| {
        NutFeatureError::Classification(format!("no made hand for class id {class_id}"))
    })
}

fn legal_opponent_combos(hero: [Card; 2], board: &[Card]) -> Vec<[Card; 2]> {
    board_legal_combos(board)
        .into_iter()
        .filter(|combo| !combo.contains(&hero[0]) && !combo.contains(&hero[1]))
        .collect()
}

fn board_legal_combos(board: &[Card]) -> Vec<[Card; 2]> {
    let deck = full_deck();
    let mut combos = Vec::with_capacity(1081);
    for i in 0..deck.len() {
        if board.contains(&deck[i]) {
            continue;
        }
        for j in i + 1..deck.len() {
            if board.contains(&deck[j]) {
                continue;
            }
            combos.push([deck[i], deck[j]]);
        }
    }
    combos
}

fn full_deck() -> Vec<Card> {
    let mut deck = Vec::with_capacity(52);
    for &value in &ALL_VALUES {
        for &suit in &ALL_SUITS {
            deck.push(Card::new(value, suit));
        }
    }
    deck
}

fn blocks_nut_combo(hero: [Card; 2], board: &[Card], nut_ord: u32) -> bool {
    board_legal_combos(board)
        .into_iter()
        .filter(|combo| combo.contains(&hero[0]) || combo.contains(&hero[1]))
        .any(|combo| rank_to_ordinal(rank_hand(combo, board)) == nut_ord)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::Suit::{Club, Diamond, Heart, Spade};
    use crate::poker::Value::{Ace, Jack, King, Nine, Queen, Seven, Six, Ten, Three, Two};

    fn c(value: crate::poker::Value, suit: crate::poker::Suit) -> Card {
        Card::new(value, suit)
    }

    #[test]
    fn nut_flush_has_zero_class_gap() {
        let board = [
            c(Queen, Spade),
            c(Jack, Spade),
            c(Seven, Spade),
            c(Two, Heart),
            c(Three, Diamond),
        ];
        let features = river_nut_features([c(Ace, Spade), c(King, Spade)], &board).unwrap();

        assert_eq!(features.made_class, HandClass::Flush);
        assert_eq!(features.class_gap, 0);
        assert_eq!(features.same_family_beaters, 0);
        assert_eq!(features.dominance_margin, 0.0);
    }

    #[test]
    fn dominated_flush_has_same_family_beaters() {
        let board = [
            c(Queen, Spade),
            c(Jack, Spade),
            c(Seven, Spade),
            c(Two, Heart),
            c(Three, Diamond),
        ];
        let features = river_nut_features([c(King, Spade), c(Ten, Spade)], &board).unwrap();

        assert_eq!(features.made_class, HandClass::Flush);
        assert!(features.class_gap > 0);
        assert!(features.same_family_beaters > 0);
        assert!(features.dominance_margin > 0.0);
    }

    #[test]
    fn top_set_is_less_dominated_than_bottom_set() {
        let board = [
            c(Ace, Heart),
            c(Nine, Diamond),
            c(Two, Club),
            c(Six, Spade),
            c(Seven, Heart),
        ];
        let top_set = river_nut_features([c(Ace, Spade), c(Ace, Diamond)], &board).unwrap();
        let bottom_set = river_nut_features([c(Two, Spade), c(Two, Diamond)], &board).unwrap();

        assert_eq!(top_set.made_class, HandClass::Set);
        assert_eq!(bottom_set.made_class, HandClass::Set);
        assert!(top_set.class_gap < bottom_set.class_gap);
        assert!(top_set.dominance_margin < bottom_set.dominance_margin);
    }

    #[test]
    fn rejects_non_river_board() {
        let board = [c(Queen, Spade), c(Jack, Spade), c(Seven, Spade)];
        let err = river_nut_features([c(Ace, Spade), c(King, Spade)], &board).unwrap_err();

        assert_eq!(err, NutFeatureError::InvalidBoardSize(3));
    }
}
