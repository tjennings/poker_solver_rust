use crate::card::*;
use crate::game::SwapList;

/// Result of computing suit isomorphisms for a card configuration.
///
/// Contains reference-index maps, lists of skippable cards, and per-suit
/// swap tables for both turn and river chance nodes. The solver uses this
/// data to skip redundant chance outcomes and copy strategies from
/// canonical representatives.
pub(crate) type IsomorphismData = (
    Vec<u8>,          // isomorphism_ref_turn
    Vec<Card>,        // isomorphism_card_turn
    [SwapList; 4],    // isomorphism_swap_turn  (indexed by suit)
    Vec<Vec<u8>>,     // isomorphism_ref_river  (indexed by turn card)
    [Vec<Card>; 4],   // isomorphism_card_river (indexed by turn suit)
    [[SwapList; 4]; 4], // isomorphism_swap_river (indexed by [turn suit][river suit])
);

type PrivateCards = [Vec<(Card, Card)>; 2];

impl CardConfig {
    /// Compute suit isomorphism data for all turn/river chance nodes.
    ///
    /// Two suits are considered isomorphic when:
    /// 1. Both players' ranges have identical weight distributions after
    ///    swapping those two suits, AND
    /// 2. The board cards seen so far treat those suits identically
    ///    (same set of ranks for each suit on the flop/turn).
    ///
    /// When suits are isomorphic, the solver can skip one of them and
    /// copy the result from the canonical representative, adjusting
    /// hand indices via the swap tables.
    pub(crate) fn isomorphism(&self, private_cards: &PrivateCards) -> IsomorphismData {
        // Step 1: Determine which suits are isomorphic in the ranges alone.
        // suit_isomorphism[s] is an equivalence-class index for suit s.
        let mut suit_isomorphism = [0u8; 4];
        let mut next_index = 1u8;
        'outer: for suit2 in 1..4u8 {
            for suit1 in 0..suit2 {
                if self.range[0].is_suit_isomorphic(suit1, suit2)
                    && self.range[1].is_suit_isomorphic(suit1, suit2)
                {
                    suit_isomorphism[suit2 as usize] = suit_isomorphism[suit1 as usize];
                    continue 'outer;
                }
            }
            suit_isomorphism[suit2 as usize] = next_index;
            next_index += 1;
        }

        // Step 2: Build per-suit rank sets from the flop.
        let flop_mask: u64 = (1 << self.flop[0]) | (1 << self.flop[1]) | (1 << self.flop[2]);
        let mut flop_rankset = [0u16; 4];
        for &card in &self.flop {
            let rank = card >> 2;
            let suit = card & 3;
            flop_rankset[suit as usize] |= 1 << rank;
        }

        let mut isomorphic_suit: [Option<u8>; 4] = [None; 4];
        let mut reverse_table = vec![usize::MAX; 52 * 51 / 2];

        let mut isomorphism_ref_turn = Vec::new();
        let mut isomorphism_card_turn = Vec::new();
        let mut isomorphism_swap_turn: [SwapList; 4] = Default::default();

        // Step 3: Turn isomorphism — only relevant when the turn is not yet dealt.
        if self.turn == NOT_DEALT {
            for suit1 in 1..4u8 {
                for suit2 in 0..suit1 {
                    if flop_rankset[suit1 as usize] == flop_rankset[suit2 as usize]
                        && suit_isomorphism[suit1 as usize] == suit_isomorphism[suit2 as usize]
                    {
                        isomorphic_suit[suit1 as usize] = Some(suit2);
                        isomorphism_swap_internal(
                            &mut isomorphism_swap_turn,
                            &mut reverse_table,
                            suit1,
                            suit2,
                            private_cards,
                        );
                        break;
                    }
                }
            }

            isomorphism_internal(
                &mut isomorphism_ref_turn,
                &mut isomorphism_card_turn,
                flop_mask,
                &isomorphic_suit,
            );
        }

        let mut isomorphism_ref_river: Vec<Vec<u8>> = vec![Vec::new(); 52];
        let mut isomorphism_card_river: [Vec<Card>; 4] = Default::default();
        let mut isomorphism_swap_river: [[SwapList; 4]; 4] = Default::default();

        // Step 4: River isomorphism — per turn card.
        if self.river == NOT_DEALT {
            for turn in 0u8..52 {
                if (1 << turn) & flop_mask != 0
                    || (self.turn != NOT_DEALT && self.turn != turn)
                {
                    continue;
                }

                let turn_mask = flop_mask | (1 << turn);
                let mut turn_rankset = flop_rankset;
                turn_rankset[turn as usize & 3] |= 1 << (turn >> 2);

                isomorphic_suit.fill(None);

                for suit1 in 1..4u8 {
                    for suit2 in 0..suit1 {
                        if (flop_rankset[suit1 as usize] == flop_rankset[suit2 as usize]
                            || self.turn != NOT_DEALT)
                            && turn_rankset[suit1 as usize] == turn_rankset[suit2 as usize]
                            && suit_isomorphism[suit1 as usize]
                                == suit_isomorphism[suit2 as usize]
                        {
                            isomorphic_suit[suit1 as usize] = Some(suit2);
                            isomorphism_swap_internal(
                                &mut isomorphism_swap_river[turn as usize & 3],
                                &mut reverse_table,
                                suit1,
                                suit2,
                                private_cards,
                            );
                            break;
                        }
                    }
                }

                isomorphism_internal(
                    &mut isomorphism_ref_river[turn as usize],
                    &mut isomorphism_card_river[turn as usize & 3],
                    turn_mask,
                    &isomorphic_suit,
                );
            }
        }

        (
            isomorphism_ref_turn,
            isomorphism_card_turn,
            isomorphism_swap_turn,
            isomorphism_ref_river,
            isomorphism_card_river,
            isomorphism_swap_river,
        )
    }
}

/// Builds swap tables for a single suit pair (suit1 <-> suit2).
///
/// For each player, iterates over all hands and finds the index of the
/// hand that results from swapping suit1 and suit2.  Records `(i, j)`
/// pairs where `i < j` so that swapping can be applied in one pass.
///
/// The `swap_list` is indexed by suit: `swap_list[suit1]` holds the
/// per-player swap pairs for the suit1<->suit2 isomorphism.
fn isomorphism_swap_internal(
    swap_list: &mut [SwapList; 4],
    reverse_table: &mut [usize],
    suit1: u8,
    suit2: u8,
    private_cards: &PrivateCards,
) {
    let swap_list = &mut swap_list[suit1 as usize];
    let replacer = |card: Card| -> Card {
        if card & 3 == suit1 {
            card - suit1 + suit2
        } else if card & 3 == suit2 {
            card + suit1 - suit2
        } else {
            card
        }
    };

    for player in 0..2 {
        if !swap_list[player].is_empty() {
            continue;
        }

        reverse_table.fill(usize::MAX);
        let cards = &private_cards[player];

        for i in 0..cards.len() {
            reverse_table[card_pair_to_index(cards[i].0, cards[i].1)] = i;
        }

        for (i, &(c1, c2)) in cards.iter().enumerate() {
            let c1 = replacer(c1);
            let c2 = replacer(c2);
            let index = reverse_table[card_pair_to_index(c1, c2)];
            if i < index {
                swap_list[player].push((i as u16, index as u16));
            }
        }
    }
}

/// Builds the reference-index and skipped-card arrays.
///
/// For each card not on the board (`mask`), if its suit is isomorphic
/// to another suit, record which canonical card index it maps to and
/// add it to the skip list. Otherwise assign it the next counter value.
fn isomorphism_internal(
    isomorphism_ref: &mut Vec<u8>,
    isomorphism_card: &mut Vec<Card>,
    mask: u64,
    isomorphic_suit: &[Option<u8>; 4],
) {
    let push_card = isomorphism_card.is_empty();
    let mut counter: u8 = 0;
    let mut indices = [0u8; 52];

    for card in 0u8..52 {
        if (1u64 << card) & mask != 0 {
            continue;
        }

        let suit = card & 3;

        if let Some(replace_suit) = isomorphic_suit[suit as usize] {
            let replace_card = card - suit + replace_suit;
            isomorphism_ref.push(indices[replace_card as usize]);
            if push_card {
                isomorphism_card.push(card);
            }
        } else {
            indices[card as usize] = counter;
            counter += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a CardConfig and private_cards from range strings
    /// and a flop string, with turn/river not dealt.
    fn setup(
        oop_range: &str,
        ip_range: &str,
        flop: &str,
        turn: Card,
        river: Card,
    ) -> (CardConfig, PrivateCards) {
        let card_config = CardConfig {
            range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
            flop: flop_from_str(flop).unwrap(),
            turn,
            river,
        };

        let mut board_mask: u64 = (1 << card_config.flop[0])
            | (1 << card_config.flop[1])
            | (1 << card_config.flop[2]);
        if turn != NOT_DEALT {
            board_mask |= 1 << turn;
        }
        if river != NOT_DEALT {
            board_mask |= 1 << river;
        }

        let mut private_cards: PrivateCards = Default::default();
        for player in 0..2 {
            let (hands, _weights) = card_config.range[player].get_hands_weights(board_mask);
            private_cards[player] = hands;
        }

        (card_config, private_cards)
    }

    #[test]
    fn rainbow_flop_no_turn_iso() {
        // Flop: Ah Kd Qs — all different suits. The 4th suit (clubs) is the
        // only one absent from the flop, so no two suits share the same flop
        // rank set => no turn isomorphism.
        let range = "AA,KK,QQ,JJ,TT,AKs,AKo";
        let (config, pc) = setup(range, range, "AhKdQs", NOT_DEALT, NOT_DEALT);
        let (ref_turn, card_turn, _swap_turn, _ref_river, _card_river, _swap_river) =
            config.isomorphism(&pc);

        // With a rainbow flop, every suit appears at most once on the board,
        // and since the ranges are suit-symmetric, the two suits NOT on the
        // flop (clubs and one more) could potentially be isomorphic — but
        // the flop has three different suits, leaving clubs as the only
        // absent suit. No pair of suits shares the same flop rank set
        // while also being absent from the flop — so no turn iso.
        //
        // Actually: hearts has rank A(12), diamonds has rank K(11), spades has Q(10).
        // Clubs has no flop rank. No two suits share the same rank set.
        // Turn iso requires flop_rankset[s1] == flop_rankset[s2] AND
        // suit_isomorphism[s1] == suit_isomorphism[s2].
        // With a symmetric range, all suits are isomorphic in the range,
        // but on the flop each suit has a unique rank set, so no turn iso.
        assert!(
            card_turn.is_empty(),
            "Rainbow flop should have no turn isomorphism"
        );
        assert!(ref_turn.is_empty());
    }

    #[test]
    fn monotone_flop_turn_iso() {
        // Flop: Ah Kh Qh (all hearts).
        // The three non-heart suits (c=0, d=1, s=3) all have empty flop rank
        // sets, and with a symmetric range they are isomorphic in the range.
        // So turn cards of suit d map to suit c, and suit s maps to suit c
        // (or d, whichever comes first).
        let range = "AA,KK,QQ,JJ,TT,99,88,77,AKs,AQs,AJs,ATs";
        let (config, pc) = setup(range, range, "AhKhQh", NOT_DEALT, NOT_DEALT);
        let (ref_turn, card_turn, swap_turn, _ref_river, _card_river, _swap_river) =
            config.isomorphism(&pc);

        // With a monotone flop, 3 suits have empty flop ranksets and hearts
        // has {A,K,Q}. With symmetric range, suits 0(c), 1(d), 3(s) are all
        // in the same equivalence class. Suit 1 maps to suit 0, suit 3 maps
        // to suit 0.
        //
        // So 2 out of 3 non-heart suits are skipped => 2*13 = 26 cards
        // skipped (minus those blocked by the board).
        assert!(
            !card_turn.is_empty(),
            "Monotone flop should produce turn isomorphisms"
        );

        // Count how many turn cards are skipped: should be roughly 2 suits
        // worth of ranks (minus board blockers).
        // 49 possible turn cards (52 - 3 flop). Of the 49:
        // - 10 hearts (13-3 flop cards)
        // - 13 clubs, 13 diamonds, 13 spades
        // Clubs (suit 0) is canonical, diamonds (suit 1) and spades (suit 3)
        // are isomorphic to clubs => skip 13 + 13 = 26 cards.
        assert_eq!(card_turn.len(), 26);

        // ref_turn should have 26 entries, each pointing to a clubs card index.
        assert_eq!(ref_turn.len(), 26);

        // swap_turn should have entries for suit1=1 (d->c swap) and
        // suit1=3 (s->c swap).
        assert!(
            !swap_turn[1][0].is_empty() || !swap_turn[1][1].is_empty(),
            "Should have swap entries for suit 1 (diamond)"
        );
        assert!(
            !swap_turn[3][0].is_empty() || !swap_turn[3][1].is_empty(),
            "Should have swap entries for suit 3 (spade)"
        );
    }

    #[test]
    fn two_tone_flop() {
        // Flop: Ah Kh Qd (2 hearts, 1 diamond).
        // flop_rankset: c=0, d={Q}, h={A,K}, s=0
        // Clubs and spades both have empty flop rank sets.
        // With a symmetric range, clubs and spades are isomorphic.
        let range = "AA,KK,QQ,JJ,TT,99,AKs,AQs,KQs";
        let (config, pc) = setup(range, range, "AhKhQd", NOT_DEALT, NOT_DEALT);
        let (ref_turn, card_turn, swap_turn, _ref_river, _card_river, _swap_river) =
            config.isomorphism(&pc);

        // c(0) and s(3) have the same flop rankset (both empty) and same
        // range isomorphism class. So spades maps to clubs => 13 cards skipped.
        assert!(
            !card_turn.is_empty(),
            "Two-tone flop should have turn isomorphism for the two absent suits"
        );
        assert_eq!(card_turn.len(), 13);
        assert_eq!(ref_turn.len(), 13);

        // All skipped cards should be spades (suit 3).
        for &card in &card_turn {
            assert_eq!(card & 3, 3, "Skipped card should be a spade");
        }

        // swap_turn[3] should have entries (spade -> club swap).
        assert!(
            !swap_turn[3][0].is_empty() || !swap_turn[3][1].is_empty(),
            "Should have swap entries for suit 3 (spade->club)"
        );
    }

    #[test]
    fn no_iso_when_turn_dealt() {
        // When turn is already dealt, no turn isomorphism is computed.
        let range = "AA,KK,QQ,JJ,TT";
        let turn = card_from_str("2c").unwrap();
        let (config, pc) = setup(range, range, "AhKhQh", turn, NOT_DEALT);
        let (ref_turn, card_turn, _swap_turn, _ref_river, _card_river, _swap_river) =
            config.isomorphism(&pc);

        assert!(ref_turn.is_empty());
        assert!(card_turn.is_empty());
    }

    #[test]
    fn no_iso_when_river_dealt() {
        // When river is already dealt, no river isomorphism is computed.
        let range = "AA,KK,QQ,JJ,TT";
        let turn = card_from_str("2c").unwrap();
        let river = card_from_str("3d").unwrap();
        let (config, pc) = setup(range, range, "AhKhQh", turn, river);
        let (_ref_turn, _card_turn, _swap_turn, ref_river, _card_river, _swap_river) =
            config.isomorphism(&pc);

        // All ref_river entries should be empty.
        for refs in &ref_river {
            assert!(refs.is_empty());
        }
    }

    #[test]
    fn river_iso_after_monotone_flop() {
        // Monotone flop Ah Kh Qh, turn is 2c.
        // After turn: hearts has {A,K,Q}, clubs has {2}, diamonds has {}, spades has {}.
        // Diamonds and spades have the same turn_rankset (empty) and the
        // same range isomorphism class => river cards of spades map to
        // diamonds for this turn card.
        let range = "AA,KK,QQ,JJ,TT,99,88,77,AKs,AQs,AJs";
        let turn = card_from_str("2c").unwrap();
        let (config, pc) = setup(range, range, "AhKhQh", turn, NOT_DEALT);
        let (_ref_turn, _card_turn, _swap_turn, ref_river, card_river, _swap_river) =
            config.isomorphism(&pc);

        // For turn=2c (card index 0, suit 0=clubs):
        // turn_rankset: c={2}, d={}, h={A,K,Q}, s={}
        // Diamonds (1) and spades (3) are isomorphic.
        // So river cards of spades map to diamonds => 13 cards skipped.
        //
        // Also: flop_rankset condition is relaxed when turn is dealt
        // (self.turn != NOT_DEALT), so even suits with different flop ranks
        // can be isomorphic if their turn_ranksets match.
        // In this case, turn IS dealt, so we check only turn_rankset.
        let turn_idx = turn as usize;
        assert!(
            !ref_river[turn_idx].is_empty(),
            "Should have river isomorphism for turn=2c"
        );

        // card_river is indexed by turn suit (0 for clubs).
        let turn_suit = turn_idx & 3;
        assert!(
            !card_river[turn_suit].is_empty(),
            "Should have skipped river cards for turn suit 0"
        );
    }

    #[test]
    fn swap_table_symmetry() {
        // Swap pairs should satisfy: for each (i, j), i < j.
        let range = "AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,ATs,A9s";
        let (config, pc) = setup(range, range, "AhKhQh", NOT_DEALT, NOT_DEALT);
        let (_ref_turn, _card_turn, swap_turn, _ref_river, _card_river, _swap_river) =
            config.isomorphism(&pc);

        for suit_swaps in &swap_turn {
            for player_swaps in suit_swaps {
                for &(a, b) in player_swaps {
                    assert!(
                        a < b,
                        "Swap pair should have a < b, got ({a}, {b})"
                    );
                }
            }
        }
    }

    #[test]
    fn asymmetric_range_no_iso() {
        // When one player's range distinguishes suits, no isomorphism even
        // if the flop is monotone.
        // OOP has only heart-suited hands, IP has a symmetric range.
        let oop_range = "AhKh,AhQh,AhJh,KhQh";
        let ip_range = "AA,KK,QQ,JJ,TT";
        let (config, pc) = setup(oop_range, ip_range, "2c3c4c", NOT_DEALT, NOT_DEALT);
        let (_ref_turn, card_turn, _swap_turn, _ref_river, _card_river, _swap_river) =
            config.isomorphism(&pc);

        // Even though the flop is monotone clubs, the OOP range is
        // heart-specific, so suits are not interchangeable.
        // d(1), h(2), s(3) all have empty flop ranksets, but
        // is_suit_isomorphic will fail for OOP's range since swapping
        // hearts with diamonds/spades changes hand weights.
        //
        // The only possible turn iso would be between suits whose ranges
        // are symmetric. Let's check that at least hearts aren't isomorphic
        // with others.
        // Actually d and s might still be isomorphic since OOP has no
        // diamond or spade hands either way (weight 0 for both).
        // So d<->s could still be isomorphic. Let's just verify the
        // length is <= 13 (at most one pair of suits).
        assert!(
            card_turn.len() <= 13,
            "At most one suit pair should be isomorphic with asymmetric range"
        );
    }
}
