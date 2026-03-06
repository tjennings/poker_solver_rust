use crate::hand_table::HAND_TABLE;

/// A 7-card poker hand accumulator for evaluation.
///
/// Cards are added one at a time via [`Hand::add_card`]. Once all 7 cards
/// (5 board + 2 hole) are present, [`Hand::evaluate`] returns a `u16`
/// strength value where higher values correspond to stronger hands.
///
/// The strength is an index into [`HAND_TABLE`], so identical hand rankings
/// produce identical values. This is critical for correct showdown evaluation.
#[derive(Clone, Copy, Default)]
pub(crate) struct Hand {
    cards: [usize; 7],
    num_cards: usize,
}

/// Keeps the `n` most-significant set bits of `x`, clearing all others.
#[inline]
fn keep_n_msb(mut x: i32, n: i32) -> i32 {
    let mut ret = 0;
    for _ in 0..n {
        let bit = 1 << (x.leading_zeros() ^ 31);
        x ^= bit;
        ret |= bit;
    }
    ret
}

/// Detects a 5-card straight in a rank bitset. Returns the top-rank bit
/// of the straight, or 0 if none exists. Handles the wheel (A-2-3-4-5).
#[inline]
fn find_straight(rankset: i32) -> i32 {
    const WHEEL: i32 = 0b1_0000_0000_1111;
    let is_straight = rankset & (rankset << 1) & (rankset << 2) & (rankset << 3) & (rankset << 4);
    if is_straight != 0 {
        keep_n_msb(is_straight, 1)
    } else if (rankset & WHEEL) == WHEEL {
        1 << 3
    } else {
        0
    }
}

impl Hand {
    #[inline]
    pub fn new() -> Hand {
        Hand::default()
    }

    #[inline]
    pub fn add_card(&self, card: usize) -> Hand {
        let mut hand = *self;
        hand.cards[hand.num_cards] = card;
        hand.num_cards += 1;
        hand
    }

    #[inline]
    pub fn contains(&self, card: usize) -> bool {
        self.cards[0..self.num_cards].contains(&card)
    }

    /// Evaluates the hand and returns a `u16` strength index.
    ///
    /// Higher values indicate stronger hands. The value is the position
    /// of the hand's internal encoding in the sorted [`HAND_TABLE`].
    #[inline]
    pub fn evaluate(&self) -> u16 {
        // INVARIANT: evaluate_internal() always produces a value present in
        // HAND_TABLE. The table is sorted and covers all 4824 distinct
        // 7-card hand equivalence classes.
        HAND_TABLE
            .binary_search(&self.evaluate_internal())
            .unwrap() as u16
    }

    fn evaluate_internal(&self) -> i32 {
        let mut rankset = 0i32;
        let mut rankset_suit = [0i32; 4];
        let mut rankset_of_count = [0i32; 5];
        let mut rank_count = [0i32; 13];

        for &card in &self.cards {
            let rank = card / 4;
            let suit = card % 4;
            rankset |= 1 << rank;
            rankset_suit[suit] |= 1 << rank;
            rank_count[rank] += 1;
        }

        for rank in 0..13 {
            rankset_of_count[rank_count[rank] as usize] |= 1 << rank;
        }

        let mut flush_suit: i32 = -1;
        for suit in 0..4 {
            if rankset_suit[suit as usize].count_ones() >= 5 {
                flush_suit = suit;
            }
        }

        let is_straight = find_straight(rankset);

        if flush_suit >= 0 {
            let is_straight_flush = find_straight(rankset_suit[flush_suit as usize]);
            if is_straight_flush != 0 {
                // straight flush
                (8 << 26) | is_straight_flush
            } else {
                // flush
                (5 << 26) | keep_n_msb(rankset_suit[flush_suit as usize], 5)
            }
        } else if rankset_of_count[4] != 0 {
            // four of a kind
            let remaining = keep_n_msb(rankset ^ rankset_of_count[4], 1);
            (7 << 26) | (rankset_of_count[4] << 13) | remaining
        } else if rankset_of_count[3].count_ones() == 2 {
            // full house (two trips -> best trips + second as pair)
            let trips = keep_n_msb(rankset_of_count[3], 1);
            let pair = rankset_of_count[3] ^ trips;
            (6 << 26) | (trips << 13) | pair
        } else if rankset_of_count[3] != 0 && rankset_of_count[2] != 0 {
            // full house (trips + pair)
            let pair = keep_n_msb(rankset_of_count[2], 1);
            (6 << 26) | (rankset_of_count[3] << 13) | pair
        } else if is_straight != 0 {
            // straight
            (4 << 26) | is_straight
        } else if rankset_of_count[3] != 0 {
            // three of a kind
            let remaining = keep_n_msb(rankset_of_count[1], 2);
            (3 << 26) | (rankset_of_count[3] << 13) | remaining
        } else if rankset_of_count[2].count_ones() >= 2 {
            // two pair
            let pairs = keep_n_msb(rankset_of_count[2], 2);
            let remaining = keep_n_msb(rankset ^ pairs, 1);
            (2 << 26) | (pairs << 13) | remaining
        } else if rankset_of_count[2] != 0 {
            // one pair
            let remaining = keep_n_msb(rankset_of_count[1], 3);
            (1 << 26) | (rankset_of_count[2] << 13) | remaining
        } else {
            // high card
            keep_n_msb(rankset, 5)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a Hand from a slice of card indices.
    fn hand_from(cards: &[usize]) -> Hand {
        let mut h = Hand::new();
        for &c in cards {
            h = h.add_card(c);
        }
        h
    }

    /// Helper: convert rank+suit to card index.
    /// rank: 0=2, 1=3, ..., 12=A.  suit: 0=c, 1=d, 2=h, 3=s.
    fn card(rank: usize, suit: usize) -> usize {
        rank * 4 + suit
    }

    #[test]
    fn test_hand_ranking_order() {
        // Royal flush: Ac Kc Qc Jc Tc + two irrelevant cards
        let royal_flush = hand_from(&[
            card(12, 0), // Ac
            card(11, 0), // Kc
            card(10, 0), // Qc
            card(9, 0),  // Jc
            card(8, 0),  // Tc
            card(0, 1),  // 2d
            card(1, 2),  // 3h
        ]);

        // Straight flush: 9c 8c 7c 6c 5c + two irrelevant
        let straight_flush = hand_from(&[
            card(7, 0), // 9c
            card(6, 0), // 8c
            card(5, 0), // 7c
            card(4, 0), // 6c
            card(3, 0), // 5c
            card(0, 1), // 2d
            card(1, 2), // 3h
        ]);

        // Four of a kind: 4x Aces + kickers
        let quads = hand_from(&[
            card(12, 0), // Ac
            card(12, 1), // Ad
            card(12, 2), // Ah
            card(12, 3), // As
            card(11, 0), // Kc
            card(0, 1),  // 2d
            card(1, 2),  // 3h
        ]);

        // Full house: 3x Kings + 2x Queens
        let full_house = hand_from(&[
            card(11, 0), // Kc
            card(11, 1), // Kd
            card(11, 2), // Kh
            card(10, 0), // Qc
            card(10, 1), // Qd
            card(0, 2),  // 2h
            card(1, 3),  // 3s
        ]);

        // Flush: Ac Tc 8c 6c 4c + two off-suit
        let flush = hand_from(&[
            card(12, 0), // Ac
            card(8, 0),  // Tc
            card(6, 0),  // 8c
            card(4, 0),  // 6c
            card(2, 0),  // 4c
            card(0, 1),  // 2d
            card(1, 2),  // 3h
        ]);

        // Straight: A K Q J T (no flush)
        let straight = hand_from(&[
            card(12, 0), // Ac
            card(11, 1), // Kd
            card(10, 2), // Qh
            card(9, 3),  // Js
            card(8, 0),  // Tc
            card(0, 1),  // 2d
            card(1, 2),  // 3h
        ]);

        // Three of a kind: 3x Jacks
        let trips = hand_from(&[
            card(9, 0),  // Jc
            card(9, 1),  // Jd
            card(9, 2),  // Jh
            card(12, 3), // As
            card(11, 0), // Kc
            card(0, 1),  // 2d
            card(1, 2),  // 3h
        ]);

        // Two pair: AA + KK
        let two_pair = hand_from(&[
            card(12, 0), // Ac
            card(12, 1), // Ad
            card(11, 2), // Kh
            card(11, 3), // Ks
            card(10, 0), // Qc
            card(0, 1),  // 2d
            card(1, 2),  // 3h
        ]);

        // One pair: pair of Aces
        let one_pair = hand_from(&[
            card(12, 0), // Ac
            card(12, 1), // Ad
            card(11, 2), // Kh
            card(10, 3), // Qs
            card(9, 0),  // Jc
            card(0, 1),  // 2d
            card(1, 2),  // 3h
        ]);

        // High card: A K Q J 9 (no straight, no flush)
        let high_card = hand_from(&[
            card(12, 0), // Ac
            card(11, 1), // Kd
            card(10, 2), // Qh
            card(9, 3),  // Js
            card(7, 0),  // 9c
            card(0, 1),  // 2d
            card(1, 2),  // 3h
        ]);

        let strengths = [
            royal_flush.evaluate(),
            straight_flush.evaluate(),
            quads.evaluate(),
            full_house.evaluate(),
            flush.evaluate(),
            straight.evaluate(),
            trips.evaluate(),
            two_pair.evaluate(),
            one_pair.evaluate(),
            high_card.evaluate(),
        ];

        // Each category must be strictly stronger than the next.
        for i in 0..strengths.len() - 1 {
            assert!(
                strengths[i] > strengths[i + 1],
                "Expected strength[{i}]={} > strength[{}]={}, categories: {:?}",
                strengths[i],
                i + 1,
                strengths[i + 1],
                strengths,
            );
        }
    }

    #[test]
    fn test_identical_hands_equal_strength() {
        // Same effective hand with different kickers that don't matter:
        // Both are Ace-high flushes in clubs: Ac Kc Qc Jc Tc
        let hand_a = hand_from(&[
            card(12, 0),
            card(11, 0),
            card(10, 0),
            card(9, 0),
            card(8, 0),
            card(0, 1), // 2d
            card(1, 2), // 3h
        ]);

        let hand_b = hand_from(&[
            card(12, 0),
            card(11, 0),
            card(10, 0),
            card(9, 0),
            card(8, 0),
            card(5, 1), // 7d
            card(6, 2), // 8h
        ]);

        assert_eq!(hand_a.evaluate(), hand_b.evaluate());

        // Same pair of aces with same best kickers (K Q J),
        // differing only in the two worst kickers.
        let pair_a = hand_from(&[
            card(12, 0), // Ac
            card(12, 1), // Ad
            card(11, 2), // Kh
            card(10, 3), // Qs
            card(9, 0),  // Jc
            card(0, 1),  // 2d
            card(1, 2),  // 3h
        ]);

        let pair_b = hand_from(&[
            card(12, 0), // Ac
            card(12, 1), // Ad
            card(11, 2), // Kh
            card(10, 3), // Qs
            card(9, 0),  // Jc
            card(2, 1),  // 4d
            card(3, 2),  // 5h
        ]);

        assert_eq!(pair_a.evaluate(), pair_b.evaluate());
    }

    #[test]
    fn test_wheel_straight() {
        // A-2-3-4-5 wheel (lowest straight)
        let wheel = hand_from(&[
            card(12, 0), // Ac
            card(0, 1),  // 2d
            card(1, 2),  // 3h
            card(2, 3),  // 4s
            card(3, 0),  // 5c
            card(5, 1),  // 7d
            card(6, 2),  // 8h
        ]);

        // 2-3-4-5-6 (next lowest straight)
        let six_high = hand_from(&[
            card(0, 0), // 2c
            card(1, 1), // 3d
            card(2, 2), // 4h
            card(3, 3), // 5s
            card(4, 0), // 6c
            card(7, 1), // 9d
            card(8, 2), // Th
        ]);

        assert!(
            six_high.evaluate() > wheel.evaluate(),
            "6-high straight should beat wheel"
        );
    }

    #[test]
    fn test_add_card_and_contains() {
        let h = Hand::new();
        assert!(!h.contains(0));

        let h = h.add_card(0);
        assert!(h.contains(0));
        assert!(!h.contains(1));

        let h = h.add_card(51);
        assert!(h.contains(0));
        assert!(h.contains(51));
        assert!(!h.contains(25));
    }

    /// Exhaustive test: verify every 7-card combination maps to a valid
    /// HAND_TABLE entry and that the hand-category counts match the known
    /// combinatorial values.
    ///
    /// This mirrors the original postflop-solver's test_all_hands.
    #[test]
    #[ignore] // ~30s in release mode; run with `cargo test -- --ignored`
    fn test_all_hands_exhaustive() {
        let mut appeared = vec![false; HAND_TABLE.len()];
        let mut counter = [0u64; 9];

        for i in 0..52usize {
            let h = Hand::new().add_card(i);
            for j in (i + 1)..52 {
                let h = h.add_card(j);
                for k in (j + 1)..52 {
                    let h = h.add_card(k);
                    for m in (k + 1)..52 {
                        let h = h.add_card(m);
                        for n in (m + 1)..52 {
                            let h = h.add_card(n);
                            for p in (n + 1)..52 {
                                let h = h.add_card(p);
                                for q in (p + 1)..52 {
                                    let h = h.add_card(q);
                                    let raw = h.evaluate_internal();
                                    let idx = HAND_TABLE.binary_search(&raw).unwrap();
                                    appeared[idx] = true;
                                    counter[(raw >> 26) as usize] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        assert!(appeared.iter().all(|&x| x));
        assert_eq!(counter[8], 41584);    // straight flush
        assert_eq!(counter[7], 224848);   // four of a kind
        assert_eq!(counter[6], 3473184);  // full house
        assert_eq!(counter[5], 4047644);  // flush
        assert_eq!(counter[4], 6180020);  // straight
        assert_eq!(counter[3], 6461620);  // three of a kind
        assert_eq!(counter[2], 31433400); // two pair
        assert_eq!(counter[1], 58627800); // one pair
        assert_eq!(counter[0], 23294460); // high card
    }
}
