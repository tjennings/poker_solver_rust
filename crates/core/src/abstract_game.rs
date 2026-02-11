//! Exhaustive abstract deal enumeration for tabular CFR.
//!
//! When using hand-class abstractions, the set of distinct info-set-key
//! trajectories is finite and enumerable. Instead of Monte Carlo sampling
//! (MCCFR), we enumerate **all** abstract deal trajectories, compute their
//! expected showdown equity and multiplicity (weight), and pass them to
//! the sequence-form or GPU CFR solvers for exact convergence.
//!
//! # Architecture
//!
//! ```text
//! for each (P1_preflop, P2_preflop) hand pair:
//!     for each compatible canonical flop (weighted):
//!         for each turn card:
//!             for each river card:
//!                 encode per-street hand bits for both players
//!                 determine winner via 7-card rank comparison
//!                 accumulate into TrajectoryKey → (weight, equity_sum)
//! ```
//!
//! Each unique trajectory key maps to one [`AbstractDeal`] with averaged
//! equity and summed weight, ready for use as a [`DealInfo`].

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::cfr::DealInfo;
use crate::flops::{self, CanonicalFlop};
use crate::hand_class::{classify, intra_class_strength, HandClass};
use crate::info_key::{canonical_hand_index, encode_hand_v2};
use crate::poker::{Card, Hand, Rank, Rankable};
use crate::showdown_equity;

/// A fully-enumerated abstract deal with averaged showdown equity.
#[derive(Debug, Clone)]
pub struct AbstractDeal {
    /// Per-street hand-class encodings for P1: `[preflop, flop, turn, river]`.
    pub hand_bits_p1: [u32; 4],
    /// Per-street hand-class encodings for P2: `[preflop, flop, turn, river]`.
    pub hand_bits_p2: [u32; 4],
    /// Average P1 showdown equity across all concrete deals in this group.
    pub p1_equity: f64,
    /// Total weight (number of concrete deals mapped to this trajectory).
    pub weight: f64,
}

impl From<AbstractDeal> for DealInfo {
    fn from(ad: AbstractDeal) -> Self {
        DealInfo {
            hand_bits_p1: ad.hand_bits_p1,
            hand_bits_p2: ad.hand_bits_p2,
            p1_equity: ad.p1_equity,
            weight: ad.weight,
        }
    }
}

/// Configuration for abstract deal generation.
#[derive(Debug, Clone)]
pub struct AbstractDealConfig {
    /// Stack depth in BB.
    pub stack_depth: u32,
    /// Strength bits for v2 encoding.
    pub strength_bits: u8,
    /// Equity bits for v2 encoding.
    pub equity_bits: u8,
    /// Maximum number of hole card pairs to enumerate (0 = all 1326).
    /// Useful for testing; production should use 0.
    pub max_hole_pairs: usize,
}

/// Key for deduplicating abstract deal trajectories.
#[derive(Hash, Eq, PartialEq, Clone, Copy)]
struct TrajectoryKey {
    p1_bits: [u32; 4],
    p2_bits: [u32; 4],
}

/// Accumulator for a single trajectory: total weight and equity sum.
struct TrajectoryAccum {
    weight: f64,
    equity_sum: f64,
}

/// Summary statistics from abstract deal generation.
#[derive(Debug, Clone)]
pub struct GenerationStats {
    /// Total number of concrete deals enumerated.
    pub concrete_deals: u64,
    /// Number of distinct abstract trajectories.
    pub abstract_deals: usize,
    /// Compression ratio (concrete / abstract).
    pub compression_ratio: f64,
}

/// Estimate the number of abstract deals without generating them.
///
/// Returns `(estimated_abstract_deals, concrete_deals)`.
#[must_use]
pub fn estimate_deal_count(config: &AbstractDealConfig) -> (u64, u64) {
    let canonical_flops = flops::all_flops();
    // C(49,2) P1 pairs × C(47,2) P2 pairs per flop × 46 turns × 45 rivers
    // But this overcounts — abstract dedup reduces it significantly.
    // Rough estimate: concrete = sum of flop_weight * C(49,2) * 46 * 45
    let concrete: u64 = canonical_flops.iter()
        .map(|f| u64::from(f.weight()) * 1176 * 46 * 45)
        .sum();

    // Rough compression based on bit config
    let hand_encodings = estimate_hand_encodings(config.strength_bits, config.equity_bits);
    let estimated_abstract = hand_encodings * hand_encodings * 4; // rough per-street factor
    (estimated_abstract.min(concrete), concrete)
}

fn estimate_hand_encodings(strength_bits: u8, equity_bits: u8) -> u64 {
    let classes = HandClass::COUNT as u64;
    let strength_levels = if strength_bits == 0 { 1 } else { 1u64 << strength_bits };
    let equity_levels = if equity_bits == 0 { 1 } else { 1u64 << equity_bits };
    let draw_combos = 16u64; // rough estimate of common draw flag combinations
    classes * strength_levels * equity_levels * draw_combos
}

/// Generate all abstract deals by exhaustive enumeration.
///
/// Iterates over all hand pairs and canonical flops in parallel,
/// enumerating every turn+river completion to compute per-trajectory
/// weights and equities.
#[must_use]
pub fn generate_abstract_deals(config: &AbstractDealConfig) -> (Vec<AbstractDeal>, GenerationStats) {
    let canonical_flops = flops::all_flops();
    let deck = crate::poker::full_deck();

    println!(
        "Generating abstract deals (strength_bits={}, equity_bits={})...",
        config.strength_bits, config.equity_bits
    );

    // Generate hole card pairs (optionally limited for testing)
    let all_pairs = generate_hole_pairs(&deck);
    let hole_pairs = if config.max_hole_pairs > 0 && config.max_hole_pairs < all_pairs.len() {
        // Sample evenly across the pair space to ensure card diversity
        let step = all_pairs.len() / config.max_hole_pairs;
        all_pairs.into_iter().step_by(step).take(config.max_hole_pairs).collect()
    } else {
        all_pairs
    };

    // For each ordered (P1, P2) hand pair, enumerate all boards
    let total_pairs = hole_pairs.len() * (hole_pairs.len() - 1);
    println!("  {} P1×P2 hand pairs × {} canonical flops", total_pairs, canonical_flops.len());

    // Generate all valid (P1, P2) pairs
    let hand_pairs: Vec<([Card; 2], [Card; 2])> = hole_pairs
        .iter()
        .enumerate()
        .flat_map(|(i, &p1)| {
            hole_pairs.iter().enumerate()
                .filter(move |(j, _)| *j != i)
                .filter(move |(_, p2)| !cards_overlap(p1, **p2))
                .map(move |(_, p2)| (p1, *p2))
        })
        .collect();

    println!("  {} valid non-overlapping hand pairs", hand_pairs.len());

    let results: Vec<(FxHashMap<TrajectoryKey, TrajectoryAccum>, u64)> = hand_pairs
        .par_iter()
        .fold(
            || (FxHashMap::<TrajectoryKey, TrajectoryAccum>::default(), 0u64),
            |(mut map, mut count), &(p1, p2)| {
                count += enumerate_boards_for_pair(
                    p1, p2, &canonical_flops, &deck, config, &mut map,
                );
                (map, count)
            },
        )
        .collect();

    // Merge all thread-local maps
    let mut global_map = FxHashMap::<TrajectoryKey, TrajectoryAccum>::default();
    let mut concrete_deals = 0u64;

    for (local_map, local_count) in results {
        concrete_deals += local_count;
        for (key, accum) in local_map {
            let entry = global_map.entry(key).or_insert(TrajectoryAccum {
                weight: 0.0,
                equity_sum: 0.0,
            });
            entry.weight += accum.weight;
            entry.equity_sum += accum.equity_sum;
        }
    }

    // Convert to AbstractDeal vec
    let deals: Vec<AbstractDeal> = global_map
        .into_iter()
        .map(|(key, accum)| AbstractDeal {
            hand_bits_p1: key.p1_bits,
            hand_bits_p2: key.p2_bits,
            p1_equity: accum.equity_sum / accum.weight,
            weight: accum.weight,
        })
        .collect();

    let abstract_count = deals.len();
    #[allow(clippy::cast_precision_loss)]
    let compression = if abstract_count > 0 {
        concrete_deals as f64 / abstract_count as f64
    } else {
        0.0
    };

    let stats = GenerationStats {
        concrete_deals,
        abstract_deals: abstract_count,
        compression_ratio: compression,
    };

    println!(
        "  {} concrete → {} abstract deals ({:.1}x compression)",
        stats.concrete_deals, stats.abstract_deals, stats.compression_ratio
    );

    (deals, stats)
}

/// Generate all C(52,2) = 1326 hole card pairs.
fn generate_hole_pairs(deck: &[Card]) -> Vec<[Card; 2]> {
    let mut pairs = Vec::with_capacity(1326);
    for i in 0..deck.len() {
        for j in (i + 1)..deck.len() {
            pairs.push([deck[i], deck[j]]);
        }
    }
    pairs
}

/// Check if two hole card pairs share any card.
fn cards_overlap(a: [Card; 2], b: [Card; 2]) -> bool {
    a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1]
}

/// Enumerate all board completions for a given (P1, P2) hand pair.
///
/// For each canonical flop compatible with both hands, iterates over all
/// turn and river cards, computing per-street hand-class encodings and
/// showdown outcomes.
///
/// Returns the number of concrete deals enumerated.
fn enumerate_boards_for_pair(
    p1: [Card; 2],
    p2: [Card; 2],
    canonical_flops: &[CanonicalFlop],
    deck: &[Card],
    config: &AbstractDealConfig,
    map: &mut FxHashMap<TrajectoryKey, TrajectoryAccum>,
) -> u64 {
    let dead_cards = [p1[0], p1[1], p2[0], p2[1]];
    let preflop_p1 = u32::from(canonical_hand_index(p1));
    let preflop_p2 = u32::from(canonical_hand_index(p2));
    let mut count = 0u64;

    for flop in canonical_flops {
        let flop_cards = *flop.cards();

        // Skip flops that conflict with hole cards
        if flop_cards.iter().any(|c| dead_cards.contains(c)) {
            continue;
        }

        let flop_weight = f64::from(flop.weight());
        let flop_p1 = encode_postflop(p1, &flop_cards, config);
        let flop_p2 = encode_postflop(p2, &flop_cards, config);

        // Remaining cards for turn/river
        let remaining: Vec<Card> = deck.iter()
            .filter(|c| !dead_cards.contains(c) && !flop_cards.contains(c))
            .copied()
            .collect();

        for (ti, &turn) in remaining.iter().enumerate() {
            let turn_board = [flop_cards[0], flop_cards[1], flop_cards[2], turn];
            let turn_p1 = encode_postflop(p1, &turn_board, config);
            let turn_p2 = encode_postflop(p2, &turn_board, config);

            for &river in &remaining[(ti + 1)..] {
                let river_board = [flop_cards[0], flop_cards[1], flop_cards[2], turn, river];
                let river_p1 = encode_postflop(p1, &river_board, config);
                let river_p2 = encode_postflop(p2, &river_board, config);

                // Determine showdown winner
                let p1_rank = rank_7cards(p1, &river_board);
                let p2_rank = rank_7cards(p2, &river_board);
                let equity = match p1_rank.cmp(&p2_rank) {
                    std::cmp::Ordering::Greater => 1.0,
                    std::cmp::Ordering::Less => 0.0,
                    std::cmp::Ordering::Equal => 0.5,
                };

                let key = TrajectoryKey {
                    p1_bits: [preflop_p1, flop_p1, turn_p1, river_p1],
                    p2_bits: [preflop_p2, flop_p2, turn_p2, river_p2],
                };

                let entry = map.entry(key).or_insert(TrajectoryAccum {
                    weight: 0.0,
                    equity_sum: 0.0,
                });
                entry.weight += flop_weight;
                entry.equity_sum += equity * flop_weight;
                count += 1;
            }
        }
    }
    count
}

/// Encode a holding on a partial board using `HandClassV2`.
fn encode_postflop(holding: [Card; 2], board: &[Card], config: &AbstractDealConfig) -> u32 {
    let classification = classify(holding, board).unwrap_or_default();
    let class_id = classification.strongest_made_id();
    let draw_flags = classification.draw_flags();
    let strength = if config.strength_bits > 0 && HandClass::is_made_hand_id(class_id) {
        intra_class_strength(holding, board, HandClass::ALL[class_id as usize])
    } else {
        0
    };
    let eq_bin = if config.equity_bits > 0 {
        let eq = showdown_equity::compute_equity(holding, board);
        showdown_equity::equity_bin(eq, 16)
    } else {
        0
    };
    encode_hand_v2(
        class_id, strength, eq_bin, draw_flags,
        config.strength_bits, config.equity_bits,
    )
}

/// Rank a 7-card hand (2 hole + 5 board).
fn rank_7cards(holding: [Card; 2], board: &[Card; 5]) -> Rank {
    let mut hand = Hand::default();
    for &c in board {
        hand.insert(c);
    }
    for &c in &holding {
        hand.insert(c);
    }
    hand.rank()
}

/// Convert abstract deals to `DealInfo` for use with the sequence-form solver.
#[must_use]
pub fn to_deal_infos(deals: &[AbstractDeal]) -> Vec<DealInfo> {
    deals.iter().map(|ad| DealInfo {
        hand_bits_p1: ad.hand_bits_p1,
        hand_bits_p2: ad.hand_bits_p2,
        p1_equity: ad.p1_equity,
        weight: ad.weight,
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    fn small_config() -> AbstractDealConfig {
        AbstractDealConfig {
            stack_depth: 10,
            strength_bits: 0,
            equity_bits: 0,
            max_hole_pairs: 10,
        }
    }

    #[test]
    fn abstract_deal_generation_correctness() {
        // Single test to avoid parallel contention with timed_test timeout
        let config = small_config();

        // Verify single-pair enumeration works
        let deck = crate::poker::full_deck();
        let hole_pairs = super::generate_hole_pairs(&deck);
        let canonical_flops = crate::flops::all_flops();
        let p1 = hole_pairs[0];
        let p2 = hole_pairs.iter()
            .find(|p| !super::cards_overlap(p1, **p))
            .expect("should find a non-overlapping pair");
        let mut map = FxHashMap::default();
        let count = super::enumerate_boards_for_pair(p1, *p2, &canonical_flops, &deck, &config, &mut map);
        assert!(count > 0, "Single pair should produce concrete deals");

        // Full generation
        let (deals, stats) = generate_abstract_deals(&config);

        // Weight sum positive
        assert!(!deals.is_empty(), "Should generate at least one abstract deal");
        assert!(stats.concrete_deals > 0);
        assert!(stats.compression_ratio >= 1.0);
        let total_weight: f64 = deals.iter().map(|d| d.weight).sum();
        assert!(total_weight > 0.0, "Total weight should be positive");

        // Equity in range
        for deal in &deals {
            assert!(
                (0.0..=1.0).contains(&deal.p1_equity),
                "Equity {:.4} out of range for deal {:?}",
                deal.p1_equity, deal.hand_bits_p1
            );
        }

        // Expected value ~0.5 (symmetric game)
        let weighted_eq: f64 = deals.iter().map(|d| d.p1_equity * d.weight).sum();
        let avg_equity = weighted_eq / total_weight;
        assert!(
            (avg_equity - 0.5).abs() < 0.05,
            "Average equity should be ~0.5, got {avg_equity:.4}"
        );
    }

    #[timed_test]
    fn to_deal_infos_preserves_fields() {
        let ad = AbstractDeal {
            hand_bits_p1: [1, 2, 3, 4],
            hand_bits_p2: [5, 6, 7, 8],
            p1_equity: 0.75,
            weight: 3.0,
        };
        let di = DealInfo::from(ad.clone());
        assert_eq!(di.hand_bits_p1, [1, 2, 3, 4]);
        assert_eq!(di.hand_bits_p2, [5, 6, 7, 8]);
        assert!((di.p1_equity - 0.75).abs() < f64::EPSILON);
        assert!((di.weight - 3.0).abs() < f64::EPSILON);
    }

    #[timed_test]
    fn estimate_deal_count_returns_positive() {
        let config = small_config();
        let (est, concrete) = estimate_deal_count(&config);
        assert!(est > 0);
        assert!(concrete > 0);
        assert!(est <= concrete);
    }
}
