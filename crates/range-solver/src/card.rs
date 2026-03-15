use std::mem;

use crate::game::StrengthItem;
use crate::hand::Hand;
use crate::range::Range;

/// A type representing a card, defined as an alias of `u8`.
///
/// The correspondence between the card and its ID is defined as follows:
/// - `card_id = 4 * rank + suit` (where `0 <= card_id < 52`)
///   - `rank`: 2 => `0`, 3 => `1`, 4 => `2`, ..., A => `12`
///   - `suit`: club => `0`, diamond => `1`, heart => `2`, spade => `3`
pub type Card = u8;

/// Constant representing that the card is not yet dealt.
pub const NOT_DEALT: Card = Card::MAX;

/// A struct containing the card configuration.
#[derive(Debug, Clone)]
pub struct CardConfig {
    /// Initial range of each player.
    pub range: [Range; 2],

    /// Flop cards: each card must be unique.
    pub flop: [Card; 3],

    /// Turn card: must be in range [`0`, `52`) or [`NOT_DEALT`].
    pub turn: Card,

    /// River card: must be in range [`0`, `52`) or [`NOT_DEALT`].
    pub river: Card,
}

impl Default for CardConfig {
    #[inline]
    fn default() -> Self {
        Self {
            range: Default::default(),
            flop: [NOT_DEALT; 3],
            turn: NOT_DEALT,
            river: NOT_DEALT,
        }
    }
}

/// Returns an index of the given card pair.
///
/// Examples: 2d2c => `0`, 2h2c => `1`, 2s2c => `2`, ..., AsAh => `1325`
#[inline]
pub fn card_pair_to_index(mut card1: Card, mut card2: Card) -> usize {
    if card1 > card2 {
        mem::swap(&mut card1, &mut card2);
    }
    card1 as usize * (101 - card1 as usize) / 2 + card2 as usize - 1
}

/// Returns a card pair from the given index.
///
/// Examples: `0` => 2d2c, `1` => 2h2c , `2` => 2s2c, ..., `1325` => AsAh
#[inline]
pub fn index_to_card_pair(index: usize) -> (Card, Card) {
    let card1 = (103 - (103.0 * 103.0 - 8.0 * index as f64).sqrt().ceil() as u16) / 2;
    let card2 = index as u16 - card1 * (101 - card1) / 2 + 1;
    (card1 as Card, card2 as Card)
}

// ---------------------------------------------------------------------------
// Character / string conversions
// ---------------------------------------------------------------------------

type PrivateCards = [Vec<(Card, Card)>; 2];
type Indices = [Vec<u16>; 2];

impl CardConfig {
    /// Computes valid hand indices for each board runout.
    ///
    /// Returns `(flop_indices, turn_indices, river_indices)`:
    /// - `flop_indices`: per-player indices not blocked by any board card beyond the flop
    /// - `turn_indices`: indexed by turn card (0..52)
    /// - `river_indices`: indexed by `card_pair_to_index(turn, river)` (0..1326)
    pub(crate) fn valid_indices(
        &self,
        private_cards: &PrivateCards,
    ) -> (Indices, Vec<Indices>, Vec<Indices>) {
        let ret_flop = if self.turn == NOT_DEALT {
            [
                (0..private_cards[0].len() as u16).collect(),
                (0..private_cards[1].len() as u16).collect(),
            ]
        } else {
            Indices::default()
        };

        let mut ret_turn = vec![Indices::default(); 52];
        for board in 0..52 {
            if !self.flop.contains(&board)
                && (self.turn == NOT_DEALT || self.turn == board)
                && self.river == NOT_DEALT
            {
                ret_turn[board as usize] =
                    Self::valid_indices_internal(private_cards, board, NOT_DEALT);
            }
        }

        let mut ret_river = vec![Indices::default(); 52 * 51 / 2];
        for board1 in 0..52 {
            for board2 in board1 + 1..52 {
                if !self.flop.contains(&board1)
                    && !self.flop.contains(&board2)
                    && (self.turn == NOT_DEALT || board1 == self.turn || board2 == self.turn)
                    && (self.river == NOT_DEALT || board1 == self.river || board2 == self.river)
                {
                    let index = card_pair_to_index(board1, board2);
                    ret_river[index] =
                        Self::valid_indices_internal(private_cards, board1, board2);
                }
            }
        }

        (ret_flop, ret_turn, ret_river)
    }

    fn valid_indices_internal(
        private_cards: &PrivateCards,
        board1: Card,
        board2: Card,
    ) -> Indices {
        let mut ret = [
            Vec::with_capacity(private_cards[0].len()),
            Vec::with_capacity(private_cards[1].len()),
        ];

        let mut board_mask: u64 = 0;
        if board1 != NOT_DEALT {
            board_mask |= 1 << board1;
        }
        if board2 != NOT_DEALT {
            board_mask |= 1 << board2;
        }

        for player in 0..2 {
            ret[player].extend(private_cards[player].iter().enumerate().filter_map(
                |(index, &(c1, c2))| {
                    let hand_mask: u64 = (1 << c1) | (1 << c2);
                    if hand_mask & board_mask == 0 {
                        Some(index as u16)
                    } else {
                        None
                    }
                },
            ));
            ret[player].shrink_to_fit();
        }

        ret
    }

    /// Computes per-runout hand strength arrays for showdown evaluation.
    ///
    /// Returns a `Vec` indexed by `card_pair_to_index(turn, river)`. Each entry
    /// is a `[Vec<StrengthItem>; 2]` containing sorted (ascending) strength
    /// items for each player, bracketed by sentinel items at both ends.
    pub(crate) fn hand_strength(
        &self,
        private_cards: &PrivateCards,
    ) -> Vec<[Vec<StrengthItem>; 2]> {
        let mut ret = vec![Default::default(); 52 * 51 / 2];

        let mut board = Hand::new();
        for &card in &self.flop {
            board = board.add_card(card as usize);
        }

        for board1 in 0..52 {
            for board2 in board1 + 1..52 {
                if !board.contains(board1 as usize)
                    && !board.contains(board2 as usize)
                    && (self.turn == NOT_DEALT || board1 == self.turn || board2 == self.turn)
                    && (self.river == NOT_DEALT || board1 == self.river || board2 == self.river)
                {
                    let full_board = board.add_card(board1 as usize).add_card(board2 as usize);
                    let mut strength = [
                        Vec::with_capacity(private_cards[0].len() + 2),
                        Vec::with_capacity(private_cards[1].len() + 2),
                    ];

                    for player in 0..2 {
                        // Sentinel: weakest possible
                        strength[player].push(StrengthItem {
                            strength: 0,
                            index: 0,
                        });
                        // Sentinel: strongest possible
                        strength[player].push(StrengthItem {
                            strength: u16::MAX,
                            index: u16::MAX,
                        });

                        strength[player].extend(
                            private_cards[player].iter().enumerate().filter_map(
                                |(index, &(c1, c2))| {
                                    let (c1, c2) = (c1 as usize, c2 as usize);
                                    if full_board.contains(c1) || full_board.contains(c2) {
                                        None
                                    } else {
                                        let hand = full_board.add_card(c1).add_card(c2);
                                        Some(StrengthItem {
                                            strength: hand.evaluate() + 1,
                                            index: index as u16,
                                        })
                                    }
                                },
                            ),
                        );

                        strength[player].shrink_to_fit();
                        strength[player].sort_unstable();
                    }

                    ret[card_pair_to_index(board1, board2)] = strength;
                }
            }
        }

        ret
    }
}

/// Evaluates a 7-card poker hand and returns a strength value.
///
/// Cards use the encoding `card_id = 4 * rank + suit` where rank 0 = 2, ..., 12 = A
/// and suit 0 = clubs, 1 = diamonds, 2 = hearts, 3 = spades.
///
/// Board must have exactly 5 cards. Hole cards are 2 cards.
/// Higher return values indicate stronger hands. Returns 0 if any card conflicts.
///
/// The returned `u16` is an index into the internal hand table and is comparable:
/// equal values mean equal-strength hands.
pub fn evaluate_hand_strength(board: &[Card; 5], hole: (Card, Card)) -> u16 {
    let mut hand = Hand::new();
    for &c in board {
        if hand.contains(c as usize) {
            return 0;
        }
        hand = hand.add_card(c as usize);
    }
    if hand.contains(hole.0 as usize) || hand.contains(hole.1 as usize) {
        return 0;
    }
    if hole.0 == hole.1 {
        return 0;
    }
    let hand = hand.add_card(hole.0 as usize).add_card(hole.1 as usize);
    hand.evaluate()
}

/// Attempts to convert a rank character to a rank index.
///
/// `'A'` => `12`, `'K'` => `11`, ..., `'2'` => `0`.
#[inline]
pub(crate) fn char_to_rank(c: char) -> Result<u8, String> {
    match c {
        'A' | 'a' => Ok(12),
        'K' | 'k' => Ok(11),
        'Q' | 'q' => Ok(10),
        'J' | 'j' => Ok(9),
        'T' | 't' => Ok(8),
        '2'..='9' => Ok(c as u8 - b'2'),
        _ => Err(format!("Expected rank character: {c}")),
    }
}

/// Attempts to convert a suit character to a suit index.
///
/// `'c'` => `0`, `'d'` => `1`, `'h'` => `2`, `'s'` => `3`.
#[inline]
pub(crate) fn char_to_suit(c: char) -> Result<u8, String> {
    match c {
        'c' => Ok(0),
        'd' => Ok(1),
        'h' => Ok(2),
        's' => Ok(3),
        _ => Err(format!("Expected suit character: {c}")),
    }
}

/// Attempts to convert a rank index to a rank character.
///
/// `12` => `'A'`, `11` => `'K'`, ..., `0` => `'2'`.
#[inline]
pub(crate) fn rank_to_char(rank: u8) -> Result<char, String> {
    match rank {
        12 => Ok('A'),
        11 => Ok('K'),
        10 => Ok('Q'),
        9 => Ok('J'),
        8 => Ok('T'),
        0..=7 => Ok((rank + b'2') as char),
        _ => Err(format!("Invalid rank: {rank}")),
    }
}

/// Attempts to convert a suit index to a suit character.
///
/// `0` => `'c'`, `1` => `'d'`, `2` => `'h'`, `3` => `'s'`.
#[inline]
pub(crate) fn suit_to_char(suit: u8) -> Result<char, String> {
    match suit {
        0 => Ok('c'),
        1 => Ok('d'),
        2 => Ok('h'),
        3 => Ok('s'),
        _ => Err(format!("Invalid suit: {suit}")),
    }
}

#[inline]
pub(crate) fn check_card(card: Card) -> Result<(), String> {
    if card < 52 {
        Ok(())
    } else {
        Err(format!("Invalid card: {card}"))
    }
}

/// Attempts to convert a card into a string.
///
/// # Examples
/// ```
/// use range_solver::card::card_to_string;
///
/// assert_eq!(card_to_string(0), Ok("2c".to_string()));
/// assert_eq!(card_to_string(51), Ok("As".to_string()));
/// assert!(card_to_string(52).is_err());
/// ```
#[inline]
pub fn card_to_string(card: Card) -> Result<String, String> {
    check_card(card)?;
    let rank = card >> 2;
    let suit = card & 3;
    Ok(format!("{}{}", rank_to_char(rank)?, suit_to_char(suit)?))
}

/// Attempts to convert hole cards into a string.
///
/// The card order in the input does not matter, but the output string is sorted
/// in descending order of card IDs.
#[inline]
pub fn hole_to_string(hole: (Card, Card)) -> Result<String, String> {
    let max_card = Card::max(hole.0, hole.1);
    let min_card = Card::min(hole.0, hole.1);
    Ok(format!(
        "{}{}",
        card_to_string(max_card)?,
        card_to_string(min_card)?
    ))
}

/// Attempts to convert a list of hole cards into a list of strings.
#[inline]
pub fn holes_to_strings(holes: &[(Card, Card)]) -> Result<Vec<String>, String> {
    holes.iter().map(|&hole| hole_to_string(hole)).collect()
}

/// Attempts to read the next card from a char iterator.
///
/// # Examples
/// ```
/// use range_solver::card::card_from_chars;
///
/// let mut chars = "2c3d4hAs".chars();
/// assert_eq!(card_from_chars(&mut chars), Ok(0));
/// assert_eq!(card_from_chars(&mut chars), Ok(5));
/// assert_eq!(card_from_chars(&mut chars), Ok(10));
/// assert_eq!(card_from_chars(&mut chars), Ok(51));
/// assert!(card_from_chars(&mut chars).is_err());
/// ```
#[inline]
pub fn card_from_chars<T: Iterator<Item = char>>(chars: &mut T) -> Result<Card, String> {
    let rank_char = chars.next().ok_or_else(|| "Unexpected end".to_string())?;
    let suit_char = chars.next().ok_or_else(|| "Unexpected end".to_string())?;

    let rank = char_to_rank(rank_char)?;
    let suit = char_to_suit(suit_char)?;

    Ok((rank << 2) | suit)
}

/// Attempts to convert a string into a card.
///
/// # Examples
/// ```
/// use range_solver::card::card_from_str;
///
/// assert_eq!(card_from_str("2c"), Ok(0));
/// assert_eq!(card_from_str("As"), Ok(51));
/// ```
#[inline]
pub fn card_from_str(s: &str) -> Result<Card, String> {
    let mut chars = s.chars();
    let result = card_from_chars(&mut chars)?;

    if chars.next().is_some() {
        return Err("Expected exactly two characters".to_string());
    }

    Ok(result)
}

/// Attempts to convert an optionally space-separated string into a sorted flop array.
///
/// # Examples
/// ```
/// use range_solver::card::flop_from_str;
///
/// assert_eq!(flop_from_str("2c3d4h"), Ok([0, 5, 10]));
/// assert_eq!(flop_from_str("As Ah Ks"), Ok([47, 50, 51]));
/// assert!(flop_from_str("2c3d4h5s").is_err());
/// ```
#[inline]
pub fn flop_from_str(s: &str) -> Result<[Card; 3], String> {
    let mut result = [0; 3];
    let mut chars = s.chars();

    result[0] = card_from_chars(&mut chars)?;
    result[1] = card_from_chars(&mut chars.by_ref().skip_while(|c| c.is_whitespace()))?;
    result[2] = card_from_chars(&mut chars.by_ref().skip_while(|c| c.is_whitespace()))?;

    if chars.next().is_some() {
        return Err("Expected exactly three cards".to_string());
    }

    result.sort_unstable();

    if result[0] == result[1] || result[1] == result[2] {
        return Err("Cards must be unique".to_string());
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_card_encoding() {
        assert_eq!(card_from_str("2c").unwrap(), 0);
        assert_eq!(card_from_str("As").unwrap(), 51);
        assert_eq!(card_to_string(0).unwrap(), "2c");
        assert_eq!(card_to_string(51).unwrap(), "As");
    }

    #[test]
    fn test_pair_index_roundtrip() {
        for i in 0..1326 {
            let (c1, c2) = index_to_card_pair(i);
            assert_eq!(card_pair_to_index(c1, c2), i);
        }
    }

    #[test]
    fn test_card_pair_index_exhaustive() {
        let mut k = 0;
        for i in 0..52u8 {
            for j in (i + 1)..52u8 {
                assert_eq!(card_pair_to_index(i, j), k);
                assert_eq!(card_pair_to_index(j, i), k);
                assert_eq!(index_to_card_pair(k), (i, j));
                k += 1;
            }
        }
    }

    #[test]
    fn test_card_to_string_boundaries() {
        assert_eq!(card_to_string(0), Ok("2c".to_string()));
        assert_eq!(card_to_string(5), Ok("3d".to_string()));
        assert_eq!(card_to_string(10), Ok("4h".to_string()));
        assert_eq!(card_to_string(51), Ok("As".to_string()));
        assert!(card_to_string(52).is_err());
    }

    #[test]
    fn test_card_from_chars() {
        let mut chars = "2c3d4hAs".chars();
        assert_eq!(card_from_chars(&mut chars), Ok(0));
        assert_eq!(card_from_chars(&mut chars), Ok(5));
        assert_eq!(card_from_chars(&mut chars), Ok(10));
        assert_eq!(card_from_chars(&mut chars), Ok(51));
        assert!(card_from_chars(&mut chars).is_err());
    }

    #[test]
    fn test_hole_to_string() {
        assert_eq!(hole_to_string((0, 5)), Ok("3d2c".to_string()));
        assert_eq!(hole_to_string((10, 51)), Ok("As4h".to_string()));
    }

    #[test]
    fn test_holes_to_strings() {
        assert_eq!(
            holes_to_strings(&[(0, 5), (10, 51)]),
            Ok(vec!["3d2c".to_string(), "As4h".to_string()])
        );
    }

    #[test]
    fn test_flop_from_str() {
        assert_eq!(flop_from_str("2c3d4h"), Ok([0, 5, 10]));
        assert_eq!(flop_from_str("As Ah Ks"), Ok([47, 50, 51]));
        assert!(flop_from_str("2c3d4h5s").is_err());
    }

    #[test]
    fn test_flop_from_str_duplicate() {
        assert!(flop_from_str("2c2c3d").is_err());
    }
}
