//! Exhaustive abstract deal enumeration for tabular CFR.
//!
//! When using hand-class abstractions, the set of distinct info-set-key
//! trajectories is finite and enumerable. Instead of Monte Carlo sampling
//! (MCCFR), we enumerate **all** abstract deal trajectories, compute their
//! expected showdown equity and multiplicity (weight), and pass them to
//! the sequence-form or GPU CFR solvers for exact convergence.
//!
//! # Architecture (board-centric)
//!
//! ```text
//! parallel for each canonical flop (1,755):
//!     precompute equity bins and flop encodings for ~1,176 compatible hands
//!     for each (turn, river) from remaining cards:
//!         encode all ~1,081 compatible hands → group by encoding
//!         cross-join groups: for each (group_a, group_b) pair,
//!             compare showdown ranks, accumulate into trajectory map
//! merge per-flop maps → final result
//! ```
//!
//! Each unique trajectory key maps to one [`AbstractDeal`] with averaged
//! equity and summed weight, ready for use as a [`DealInfo`].

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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

// ---------------------------------------------------------------------------
// Bitmask utilities
// ---------------------------------------------------------------------------

/// Map a card to a unique bit index 0..51.
fn card_bit(card: Card) -> u32 {
    card.value as u32 * 4 + card.suit as u32
}

/// Fold a slice of cards into a u64 bitmask.
fn card_mask(cards: &[Card]) -> u64 {
    cards.iter().fold(0u64, |m, &c| m | (1u64 << card_bit(c)))
}

/// Bitmask for a 2-card holding (avoids slice overhead).
fn card_mask_pair(pair: [Card; 2]) -> u64 {
    (1u64 << card_bit(pair[0])) | (1u64 << card_bit(pair[1]))
}

/// True if any card appears in both masks.
fn masks_overlap(a: u64, b: u64) -> bool {
    a & b != 0
}

// ---------------------------------------------------------------------------
// Fast encoding helpers
// ---------------------------------------------------------------------------

/// Encode a holding on a partial board, using a pre-supplied equity bin.
///
/// Same as `encode_postflop` but skips the expensive `compute_equity` call.
fn encode_postflop_fast(
    holding: [Card; 2],
    board: &[Card],
    config: &AbstractDealConfig,
    eq_bin: u8,
) -> u32 {
    let classification = classify(holding, board).unwrap_or_default();
    let class_id = classification.strongest_made_id();
    let draw_flags = classification.draw_flags();
    let strength = if config.strength_bits > 0 && HandClass::is_made_hand_id(class_id) {
        intra_class_strength(holding, board, HandClass::ALL[class_id as usize])
    } else {
        0
    };
    encode_hand_v2(
        class_id, strength, eq_bin, draw_flags,
        config.strength_bits, config.equity_bits,
    )
}

/// Convert a rank comparison to P1 equity (1.0 = win, 0.5 = tie, 0.0 = loss).
fn rank_equity(p1_rank: Rank, p2_rank: Rank) -> f64 {
    match p1_rank.cmp(&p2_rank) {
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Equal => 0.5,
        std::cmp::Ordering::Less => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Board-centric data structures
// ---------------------------------------------------------------------------

/// A hand encoded for a specific 5-card board.
struct EncodedHand {
    /// Per-street trajectory: `[preflop, flop, turn, river]`.
    enc: [u32; 4],
    /// 7-card showdown rank.
    rank: Rank,
    /// Hole card bitmask for fast overlap detection.
    hole_mask: u64,
}

/// Pre-computed turn-level info for a hand compatible with flop + turn.
struct TurnInfo {
    /// Index into the per-flop compatibility arrays.
    ci: usize,
    /// Index into the global `hole_pairs` / `preflop_encs` / `hole_masks`.
    gi: usize,
    /// Pre-computed turn-street encoding.
    turn_enc: u32,
}

/// Shared context for board-centric processing of a single flop.
struct FlopContext<'a> {
    flop_cards: [Card; 3],
    hole_pairs: &'a [[Card; 2]],
    preflop_encs: &'a [u32],
    hole_masks: &'a [u64],
    config: &'a AbstractDealConfig,
    /// Per-flop compatible indices, equity bins, and flop encodings.
    compat: Vec<usize>,
    eq_bins: Vec<u8>,
    flop_encs: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Cross-join: group hands by encoding, pair groups, accumulate trajectories
// ---------------------------------------------------------------------------

/// Identify contiguous group boundaries in a sorted slice.
///
/// Returns `(start, end)` ranges where all elements share the same encoding.
fn group_boundaries(hands: &[EncodedHand]) -> Vec<(usize, usize)> {
    if hands.is_empty() {
        return Vec::new();
    }
    let mut groups = Vec::new();
    let mut start = 0;
    for i in 1..hands.len() {
        if hands[i].enc != hands[start].enc {
            groups.push((start, i));
            start = i;
        }
    }
    groups.push((start, hands.len()));
    groups
}

/// Accumulate one group pair (a, b) into the trajectory map.
///
/// Handles both orderings: `(enc_a, enc_b)` and `(enc_b, enc_a)`.
/// For same-group pairs, iterates `i < j` to avoid double-counting.
/// Returns the number of concrete ordered pairs processed.
fn accumulate_group_pair(
    hands: &[EncodedHand],
    ga: (usize, usize),
    gb: (usize, usize),
    same_group: bool,
    flop_weight: f64,
    map: &mut FxHashMap<TrajectoryKey, TrajectoryAccum>,
) -> u64 {
    let enc_a = hands[ga.0].enc;
    let enc_b = hands[gb.0].enc;
    let mut w_ab = 0.0f64;
    let mut eq_ab = 0.0f64;
    let mut count = 0u64;

    if same_group {
        for i in ga.0..ga.1 {
            for j in (i + 1)..ga.1 {
                if masks_overlap(hands[i].hole_mask, hands[j].hole_mask) {
                    continue;
                }
                let eq = rank_equity(hands[i].rank, hands[j].rank);
                // Both orderings map to the same key (enc_a == enc_b).
                w_ab += 2.0 * flop_weight;
                eq_ab += (eq + (1.0 - eq)) * flop_weight; // = flop_weight
                count += 2;
            }
        }
        if w_ab > 0.0 {
            let key = TrajectoryKey { p1_bits: enc_a, p2_bits: enc_b };
            let e = map.entry(key).or_insert(TrajectoryAccum { weight: 0.0, equity_sum: 0.0 });
            e.weight += w_ab;
            e.equity_sum += eq_ab;
        }
    } else {
        let mut w_ba = 0.0f64;
        let mut eq_ba = 0.0f64;
        for i in ga.0..ga.1 {
            for j in gb.0..gb.1 {
                if masks_overlap(hands[i].hole_mask, hands[j].hole_mask) {
                    continue;
                }
                let eq = rank_equity(hands[i].rank, hands[j].rank);
                w_ab += flop_weight;
                eq_ab += eq * flop_weight;
                w_ba += flop_weight;
                eq_ba += (1.0 - eq) * flop_weight;
                count += 2;
            }
        }
        if w_ab > 0.0 {
            let k = TrajectoryKey { p1_bits: enc_a, p2_bits: enc_b };
            let e = map.entry(k).or_insert(TrajectoryAccum { weight: 0.0, equity_sum: 0.0 });
            e.weight += w_ab;
            e.equity_sum += eq_ab;
        }
        if w_ba > 0.0 {
            let k = TrajectoryKey { p1_bits: enc_b, p2_bits: enc_a };
            let e = map.entry(k).or_insert(TrajectoryAccum { weight: 0.0, equity_sum: 0.0 });
            e.weight += w_ba;
            e.equity_sum += eq_ba;
        }
    }
    count
}

/// Group hands by encoding, cross-join all group pairs, and accumulate.
///
/// Sorts `hands` in place by encoding, then iterates all `(gi, gj)` pairs
/// with `gi <= gj`. Returns the number of concrete ordered pairs processed.
fn cross_join_and_accumulate(
    hands: &mut [EncodedHand],
    flop_weight: f64,
    map: &mut FxHashMap<TrajectoryKey, TrajectoryAccum>,
) -> u64 {
    if hands.is_empty() {
        return 0;
    }
    hands.sort_unstable_by_key(|h| h.enc);
    let groups = group_boundaries(hands);
    let mut concrete = 0u64;
    for (gi, &ga) in groups.iter().enumerate() {
        for &gb in &groups[gi..] {
            concrete += accumulate_group_pair(
                hands, ga, gb, ga.0 == gb.0, flop_weight, map,
            );
        }
    }
    concrete
}

// ---------------------------------------------------------------------------
// Board-centric flop processing
// ---------------------------------------------------------------------------

impl<'a> FlopContext<'a> {
    /// Build context for a flop, precomputing compatibility and encodings.
    fn new(
        flop_cards: [Card; 3],
        hole_pairs: &'a [[Card; 2]],
        preflop_encs: &'a [u32],
        hole_masks: &'a [u64],
        config: &'a AbstractDealConfig,
    ) -> Self {
        let flop_mask = card_mask(&flop_cards);
        let compat: Vec<usize> = (0..hole_pairs.len())
            .filter(|&i| !masks_overlap(hole_masks[i], flop_mask))
            .collect();
        let eq_bins: Vec<u8> = if config.equity_bits > 0 {
            compat.iter().map(|&i| {
                let eq = showdown_equity::compute_equity(hole_pairs[i], &flop_cards);
                showdown_equity::equity_bin(eq, 16)
            }).collect()
        } else {
            vec![0u8; compat.len()]
        };
        let flop_encs: Vec<u32> = compat.iter().enumerate().map(|(ci, &i)| {
            encode_postflop_fast(hole_pairs[i], &flop_cards, config, eq_bins[ci])
        }).collect();
        Self { flop_cards, hole_pairs, preflop_encs, hole_masks, config, compat, eq_bins, flop_encs }
    }

    /// Build `TurnInfo` for all hands compatible with flop + turn.
    fn build_turn_info(&self, turn: Card) -> Vec<TurnInfo> {
        let turn_mask = 1u64 << card_bit(turn);
        let tb = [self.flop_cards[0], self.flop_cards[1], self.flop_cards[2], turn];
        self.compat.iter().copied().enumerate()
            .filter(|(_, i)| !masks_overlap(self.hole_masks[*i], turn_mask))
            .map(|(ci, i)| TurnInfo {
                ci,
                gi: i,
                turn_enc: encode_postflop_fast(
                    self.hole_pairs[i], &tb, self.config, self.eq_bins[ci],
                ),
            })
            .collect()
    }

    /// Encode hands for a full 5-card board, writing into `out`.
    fn encode_river_hands(
        &self, out: &mut Vec<EncodedHand>, turn_hands: &[TurnInfo], turn: Card, river: Card,
    ) {
        let rm = 1u64 << card_bit(river);
        let rb = [self.flop_cards[0], self.flop_cards[1], self.flop_cards[2], turn, river];
        out.clear();
        for th in turn_hands {
            if masks_overlap(self.hole_masks[th.gi], rm) {
                continue;
            }
            let re = encode_postflop_fast(
                self.hole_pairs[th.gi], &rb, self.config, self.eq_bins[th.ci],
            );
            out.push(EncodedHand {
                enc: [self.preflop_encs[th.gi], self.flop_encs[th.ci], th.turn_enc, re],
                rank: rank_7cards(self.hole_pairs[th.gi], &rb),
                hole_mask: self.hole_masks[th.gi],
            });
        }
    }
}

/// Process one canonical flop: enumerate all turn-river boards, encode hands,
/// cross-join, and accumulate into a local trajectory map.
///
/// Returns `(local_map, concrete_deal_count)`.
fn process_flop_board_centric(
    flop: &CanonicalFlop,
    hole_pairs: &[[Card; 2]],
    preflop_encs: &[u32],
    hole_masks: &[u64],
    deck: &[Card],
    config: &AbstractDealConfig,
) -> (FxHashMap<TrajectoryKey, TrajectoryAccum>, u64) {
    let flop_cards = *flop.cards();
    let flop_mask = card_mask(&flop_cards);
    let flop_weight = f64::from(flop.weight());
    let ctx = FlopContext::new(flop_cards, hole_pairs, preflop_encs, hole_masks, config);

    let remaining: Vec<Card> = deck.iter().copied()
        .filter(|&c| !masks_overlap(1u64 << card_bit(c), flop_mask))
        .collect();

    let mut local_map = FxHashMap::<TrajectoryKey, TrajectoryAccum>::default();
    let mut concrete = 0u64;
    let mut buf = Vec::with_capacity(ctx.compat.len());

    for (ti, &turn) in remaining.iter().enumerate() {
        let turn_hands = ctx.build_turn_info(turn);
        for &river in &remaining[(ti + 1)..] {
            ctx.encode_river_hands(&mut buf, &turn_hands, turn, river);
            concrete += cross_join_and_accumulate(&mut buf, flop_weight, &mut local_map);
        }
    }

    (local_map, concrete)
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

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn format_duration(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m {}s", s / 60, s % 60)
    } else {
        format!("{}h {}m", s / 3600, (s % 3600) / 60)
    }
}


/// Generate all abstract deals by board-centric exhaustive enumeration.
///
/// For each canonical flop (in parallel), iterates all turn+river boards,
/// encodes every compatible hand, groups by encoding, and cross-joins to
/// produce trajectory-keyed equity/weight accumulators.
#[must_use]
pub fn generate_abstract_deals(config: &AbstractDealConfig) -> (Vec<AbstractDeal>, GenerationStats) {
    let canonical_flops = flops::all_flops();
    let deck = crate::poker::full_deck();

    println!(
        "Generating abstract deals (strength_bits={}, equity_bits={}, board-centric)...",
        config.strength_bits, config.equity_bits
    );

    // Generate hole card pairs (optionally limited for testing)
    let hole_pairs = select_hole_pairs(&deck, config.max_hole_pairs);

    // Precompute per-hand invariants
    let preflop_encs: Vec<u32> = hole_pairs.iter()
        .map(|h| u32::from(canonical_hand_index(*h)))
        .collect();
    let hole_masks: Vec<u64> = hole_pairs.iter()
        .map(|h| card_mask_pair(*h))
        .collect();

    let total_flops = canonical_flops.len();
    println!("  {} hole pairs × {} canonical flops (board-centric)", hole_pairs.len(), total_flops);

    let flops_done = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let progress_handle = spawn_flop_progress_thread(
        flops_done.clone(), stop.clone(), total_flops as u64,
    );

    // Parallel over canonical flops
    let flop_results: Vec<(FxHashMap<TrajectoryKey, TrajectoryAccum>, u64)> = canonical_flops
        .par_iter()
        .map(|flop| {
            let result = process_flop_board_centric(
                flop, &hole_pairs, &preflop_encs, &hole_masks, &deck, config,
            );
            flops_done.fetch_add(1, Ordering::Relaxed);
            result
        })
        .collect();

    stop.store(true, Ordering::Relaxed);
    let _ = progress_handle.join();

    // Merge per-flop maps
    let (global_map, concrete_deals) = merge_flop_results(flop_results);
    let (deals, stats) = finalize_deals(global_map, concrete_deals);

    println!(
        "  {} concrete → {} abstract deals ({:.1}x compression)",
        stats.concrete_deals, stats.abstract_deals, stats.compression_ratio
    );

    (deals, stats)
}

/// Select and optionally subsample hole card pairs.
fn select_hole_pairs(deck: &[Card], max_pairs: usize) -> Vec<[Card; 2]> {
    let all = generate_hole_pairs(deck);
    if max_pairs > 0 && max_pairs < all.len() {
        let step = all.len() / max_pairs;
        all.into_iter().step_by(step).take(max_pairs).collect()
    } else {
        all
    }
}

/// Progress thread that tracks flop completion.
fn spawn_flop_progress_thread(
    done: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    total: u64,
) -> std::thread::JoinHandle<()> {
    #[allow(clippy::cast_precision_loss)]
    std::thread::spawn(move || {
        let start = Instant::now();
        while !stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_secs(2));
            let completed = done.load(Ordering::Relaxed);
            let elapsed = start.elapsed().as_secs_f64();
            let pct = completed as f64 / total as f64 * 100.0;
            let rate = completed as f64 / elapsed;
            let eta = if rate > 0.0 {
                format_duration((total - completed) as f64 / rate)
            } else {
                "--".to_string()
            };
            print!(
                "\r  flop {completed}/{total} ({pct:.1}%) [{}, ETA ~{eta}]     ",
                format_duration(elapsed),
            );
            let _ = std::io::stdout().flush();
        }
        println!();
    })
}

/// Merge per-flop trajectory maps into a single global map.
fn merge_flop_results(
    results: Vec<(FxHashMap<TrajectoryKey, TrajectoryAccum>, u64)>,
) -> (FxHashMap<TrajectoryKey, TrajectoryAccum>, u64) {
    let mut global = FxHashMap::<TrajectoryKey, TrajectoryAccum>::default();
    let mut concrete = 0u64;
    for (local_map, local_count) in results {
        concrete += local_count;
        for (key, accum) in local_map {
            let e = global.entry(key).or_insert(TrajectoryAccum {
                weight: 0.0, equity_sum: 0.0,
            });
            e.weight += accum.weight;
            e.equity_sum += accum.equity_sum;
        }
    }
    (global, concrete)
}

/// Convert trajectory map into final `AbstractDeal` vec + stats.
fn finalize_deals(
    map: FxHashMap<TrajectoryKey, TrajectoryAccum>,
    concrete_deals: u64,
) -> (Vec<AbstractDeal>, GenerationStats) {
    let deals: Vec<AbstractDeal> = map.into_iter()
        .map(|(key, accum)| AbstractDeal {
            hand_bits_p1: key.p1_bits,
            hand_bits_p2: key.p2_bits,
            p1_equity: accum.equity_sum / accum.weight,
            weight: accum.weight,
        })
        .collect();
    let n = deals.len();
    #[allow(clippy::cast_precision_loss)]
    let compression = if n > 0 { concrete_deals as f64 / n as f64 } else { 0.0 };
    let stats = GenerationStats {
        concrete_deals,
        abstract_deals: n,
        compression_ratio: compression,
    };
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

    // -----------------------------------------------------------------------
    // Legacy pair-centric implementation (kept for oracle testing)
    // -----------------------------------------------------------------------

    fn cards_overlap(a: [Card; 2], b: [Card; 2]) -> bool {
        a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1]
    }

    fn encode_postflop_legacy(
        holding: [Card; 2], board: &[Card], config: &AbstractDealConfig,
    ) -> u32 {
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

    fn enumerate_flop_for_pair_legacy(
        p1: [Card; 2], p2: [Card; 2],
        flop: &crate::flops::CanonicalFlop,
        deck: &[Card], config: &AbstractDealConfig,
        map: &mut FxHashMap<TrajectoryKey, TrajectoryAccum>,
    ) -> u64 {
        let dead = [p1[0], p1[1], p2[0], p2[1]];
        let fc = *flop.cards();
        if fc.iter().any(|c| dead.contains(c)) { return 0; }

        let pp1 = u32::from(canonical_hand_index(p1));
        let pp2 = u32::from(canonical_hand_index(p2));
        let fw = f64::from(flop.weight());
        let fp1 = encode_postflop_legacy(p1, &fc, config);
        let fp2 = encode_postflop_legacy(p2, &fc, config);

        let remaining: Vec<Card> = deck.iter()
            .filter(|c| !dead.contains(c) && !fc.contains(c))
            .copied().collect();

        let mut count = 0u64;
        for (ti, &turn) in remaining.iter().enumerate() {
            let tb = [fc[0], fc[1], fc[2], turn];
            let tp1 = encode_postflop_legacy(p1, &tb, config);
            let tp2 = encode_postflop_legacy(p2, &tb, config);
            for &river in &remaining[(ti + 1)..] {
                let rb = [fc[0], fc[1], fc[2], turn, river];
                let rp1 = encode_postflop_legacy(p1, &rb, config);
                let rp2 = encode_postflop_legacy(p2, &rb, config);
                let r1 = rank_7cards(p1, &rb);
                let r2 = rank_7cards(p2, &rb);
                let eq = rank_equity(r1, r2);
                let key = TrajectoryKey {
                    p1_bits: [pp1, fp1, tp1, rp1],
                    p2_bits: [pp2, fp2, tp2, rp2],
                };
                let e = map.entry(key).or_insert(TrajectoryAccum {
                    weight: 0.0, equity_sum: 0.0,
                });
                e.weight += fw;
                e.equity_sum += eq * fw;
                count += 1;
            }
        }
        count
    }

    fn generate_abstract_deals_legacy(
        config: &AbstractDealConfig,
    ) -> (Vec<AbstractDeal>, GenerationStats) {
        let canonical_flops = crate::flops::all_flops();
        let deck = crate::poker::full_deck();
        let hole_pairs = select_hole_pairs(&deck, config.max_hole_pairs);

        let hand_pairs: Vec<([Card; 2], [Card; 2])> = hole_pairs.iter().enumerate()
            .flat_map(|(i, &p1)| {
                hole_pairs.iter().enumerate()
                    .filter(move |(j, _)| *j != i)
                    .filter(move |(_, p2)| !cards_overlap(p1, **p2))
                    .map(move |(_, p2)| (p1, *p2))
            })
            .collect();

        let mut global_map = FxHashMap::<TrajectoryKey, TrajectoryAccum>::default();
        let mut concrete = 0u64;
        for flop in &canonical_flops {
            for &(p1, p2) in &hand_pairs {
                concrete += enumerate_flop_for_pair_legacy(
                    p1, p2, flop, &deck, config, &mut global_map,
                );
            }
        }
        finalize_deals(global_map, concrete)
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn abstract_deal_generation_correctness() {
        let config = small_config();
        let (deals, stats) = generate_abstract_deals(&config);

        assert!(!deals.is_empty(), "Should generate at least one abstract deal");
        assert!(stats.concrete_deals > 0);
        assert!(stats.compression_ratio >= 1.0);

        let total_weight: f64 = deals.iter().map(|d| d.weight).sum();
        assert!(total_weight > 0.0, "Total weight should be positive");

        for deal in &deals {
            assert!(
                (0.0..=1.0).contains(&deal.p1_equity),
                "Equity {:.4} out of range for deal {:?}",
                deal.p1_equity, deal.hand_bits_p1
            );
        }

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

    #[test]
    #[ignore] // ~100s in debug — run manually to verify equivalence
    fn oracle_board_centric_matches_legacy() {
        let config = AbstractDealConfig {
            stack_depth: 10,
            strength_bits: 0,
            equity_bits: 0,
            max_hole_pairs: 5,
        };

        let (legacy_deals, legacy_stats) = generate_abstract_deals_legacy(&config);
        let (new_deals, new_stats) = generate_abstract_deals(&config);

        assert_eq!(
            legacy_stats.concrete_deals, new_stats.concrete_deals,
            "concrete deal count mismatch: legacy={}, new={}",
            legacy_stats.concrete_deals, new_stats.concrete_deals,
        );
        assert_eq!(
            legacy_deals.len(), new_deals.len(),
            "abstract deal count mismatch: legacy={}, new={}",
            legacy_deals.len(), new_deals.len(),
        );

        // Build legacy lookup for comparison
        let legacy_map: FxHashMap<([u32; 4], [u32; 4]), (f64, f64)> = legacy_deals.iter()
            .map(|d| ((d.hand_bits_p1, d.hand_bits_p2), (d.weight, d.p1_equity)))
            .collect();

        for deal in &new_deals {
            let key = (deal.hand_bits_p1, deal.hand_bits_p2);
            let &(lw, le) = legacy_map.get(&key)
                .unwrap_or_else(|| panic!("Missing trajectory in legacy: {key:?}"));
            assert!(
                (deal.weight - lw).abs() < 1e-6,
                "Weight mismatch for {key:?}: new={}, legacy={lw}", deal.weight,
            );
            assert!(
                (deal.p1_equity - le).abs() < 1e-6,
                "Equity mismatch for {key:?}: new={}, legacy={le}", deal.p1_equity,
            );
        }
    }
}
