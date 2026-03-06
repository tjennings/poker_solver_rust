use std::mem;

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
