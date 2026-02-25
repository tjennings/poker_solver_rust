//! MCCFR postflop solve backend.
//!
//! Instead of abstract bucket-to-bucket transitions at street changes, this
//! backend uses **concrete hands** (actual `Card` pairs). For each sample it
//! picks two non-conflicting hands from flop buckets and a random turn + river.
//! At showdown terminals it uses `rank_hand()` to evaluate actual hands against
//! the full 5-card board. Strategies and regrets are still indexed by **flop
//! bucket** (not by concrete hand).

use std::sync::atomic::{AtomicUsize, Ordering};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::hand_buckets::{self, StreetBuckets};
use super::postflop_abstraction::{
    BuildPhase, FlopSolveResult, FlopStage, PostflopLayout, PostflopValues,
    add_buffers, regret_matching_into, normalize_strategy_sum,
    weighted_avg_strategy_delta, MAX_POSTFLOP_ACTIONS,
};
use super::postflop_model::PostflopModelConfig;
use super::postflop_tree::{PostflopNode, PostflopTerminalType, PostflopTree};
use crate::abstraction::Street;
use crate::hands::CanonicalHand;
use crate::poker::Card;
use crate::showdown_equity::rank_hand;

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// Maps flop_bucket → list of concrete (Card, Card) hand combos in that bucket.
struct FlopBucketMap {
    /// `bucket_hands[bucket]` = list of hands assigned to that bucket.
    bucket_hands: Vec<Vec<(Card, Card)>>,
    num_buckets: usize,
}

impl FlopBucketMap {
    /// Build from flop bucket assignments and the canonical 169 hands.
    ///
    /// `assignments[hand_idx]` → bucket index for each of the 169 canonical
    /// hands. Each canonical hand is expanded into its concrete combos and
    /// only non-conflicting combos (with the flop) are kept.
    fn build(assignments: &[u16], num_buckets: usize, flop: &[Card; 3]) -> Self {
        let hands: Vec<CanonicalHand> = crate::hands::all_hands().collect();
        let mut bucket_hands = vec![Vec::new(); num_buckets];
        for (i, &bucket) in assignments.iter().enumerate() {
            for &(c1, c2) in &hands[i].combos() {
                if hand_buckets::board_conflicts([c1, c2], flop) {
                    continue;
                }
                bucket_hands[bucket as usize].push((c1, c2));
            }
        }
        Self {
            bucket_hands,
            num_buckets,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Entry point
// ──────────────────────────────────────────────────────────────────────────────

/// Build postflop abstraction using MCCFR with sampled concrete hands.
///
/// Pipeline per flop (parallelised via rayon):
/// 1. Compute flop histogram features and cluster into `num_hand_buckets_flop` buckets.
/// 2. Build `FlopBucketMap` mapping bucket → concrete card combos.
/// 3. Run MCCFR training loop (`mccfr_solve_one_flop`).
/// 4. Extract EV table from converged strategy (`mccfr_extract_values`).
///
/// Returns `(StreetBuckets, PostflopValues)`. Turn/river bucket fields are
/// empty since MCCFR uses only flop buckets with concrete runouts.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_mccfr(
    config: &PostflopModelConfig,
    tree: &PostflopTree,
    layout: &PostflopLayout,
    node_streets: &[Street],
    flops: &[[Card; 3]],
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> (StreetBuckets, PostflopValues) {
    let hands: Vec<CanonicalHand> = crate::hands::all_hands().collect();
    let num_flops = flops.len();
    let nfb = config.num_hand_buckets_flop as usize;
    let num_iterations = config.postflop_solve_iterations as usize;
    let _ = node_streets; // Streets are embedded in the layout already.
    let completed = AtomicUsize::new(0);

    // Phase 1: per-flop solve
    let results: Vec<(Vec<u16>, Vec<f64>)> = (0..num_flops)
        .into_par_iter()
        .map(|flop_idx| {
            let flop = &flops[flop_idx];
            let flop_name = format!("{}{}{}", flop[0], flop[1], flop[2]);

            // Step 1: Assign hands to buckets.
            // When num_buckets >= num_hands (e.g. 169 buckets for 169 hands),
            // each hand gets its own bucket — skip expensive histogram + k-means.
            let assignments = if nfb >= hands.len() {
                #[allow(clippy::cast_possible_truncation)]
                let id: Vec<u16> = (0..hands.len() as u16).collect();
                id
            } else {
                on_progress(BuildPhase::FlopProgress {
                    flop_name: flop_name.clone(),
                    stage: FlopStage::Bucketing { step: 0, total_steps: 3 },
                });
                let flop_features = hand_buckets::compute_flop_histograms(
                    &hands,
                    std::slice::from_ref(flop),
                    &|_| {},
                );
                on_progress(BuildPhase::FlopProgress {
                    flop_name: flop_name.clone(),
                    stage: FlopStage::Bucketing { step: 1, total_steps: 3 },
                });

                let a = hand_buckets::cluster_histograms(
                    &flop_features,
                    config.num_hand_buckets_flop,
                );
                on_progress(BuildPhase::FlopProgress {
                    flop_name: flop_name.clone(),
                    stage: FlopStage::Bucketing { step: 2, total_steps: 3 },
                });
                a
            };

            // Step 2: Build bucket map
            let bucket_map = FlopBucketMap::build(&assignments, nfb, flop);

            // Step 3: Compute sample count
            let total_non_conflicting: usize =
                bucket_map.bucket_hands.iter().map(Vec::len).sum();
            // Remaining deck after flop (3 cards) and one hand (2 cards) = 47 cards
            // for turn, then 46 for river. But we're choosing turn+river from deck
            // minus flop minus both hands (7 used), so 45 cards for turn, 44 for river.
            let live_runouts = 45usize * 44;
            let total_space = total_non_conflicting.saturating_mul(live_runouts);
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_precision_loss,
                clippy::cast_sign_loss
            )]
            let samples_per_iter = ((total_space as f64 * config.mccfr_sample_pct) as usize
                / num_iterations.max(1))
            .max(1);

            // Step 4: Solve
            let result = mccfr_solve_one_flop(
                tree,
                layout,
                &bucket_map,
                flop,
                num_iterations,
                samples_per_iter,
                config.cfr_delta_threshold,
                &flop_name,
                on_progress,
            );

            // Step 5: Extract values
            let values = mccfr_extract_values(
                tree,
                layout,
                &result.strategy_sum,
                &bucket_map,
                flop,
                config.value_extraction_samples as usize,
                nfb,
                config.ev_convergence_threshold,
                &flop_name,
                on_progress,
            );

            on_progress(BuildPhase::FlopProgress {
                flop_name: flop_name.clone(),
                stage: FlopStage::Done,
            });

            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            on_progress(BuildPhase::MccfrFlopsCompleted { completed: done, total: num_flops });

            (assignments, values)
        })
        .collect();

    // Assemble results
    let mut flop_assignments: Vec<Vec<u16>> = Vec::with_capacity(num_flops);
    let mut all_values = vec![0.0f64; num_flops * 2 * nfb * nfb];
    for (flop_idx, (fb, vals)) in results.into_iter().enumerate() {
        flop_assignments.push(fb);
        let offset = flop_idx * 2 * nfb * nfb;
        let copy_len = vals.len().min(2 * nfb * nfb);
        all_values[offset..offset + copy_len].copy_from_slice(&vals[..copy_len]);
    }

    let buckets = StreetBuckets {
        flop: flop_assignments,
        num_flop_buckets: config.num_hand_buckets_flop,
        turn: Vec::new(),
        num_turn_buckets: 0,
        river: Vec::new(),
        num_river_buckets: 0,
    };

    let values = PostflopValues::from_raw(all_values, nfb, num_flops);

    (buckets, values)
}

// ──────────────────────────────────────────────────────────────────────────────
// Core solve loop
// ──────────────────────────────────────────────────────────────────────────────

/// Training loop for a single flop using MCCFR with concrete hands.
#[allow(clippy::too_many_arguments)]
fn mccfr_solve_one_flop(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    bucket_map: &FlopBucketMap,
    flop: &[Card; 3],
    num_iterations: usize,
    samples_per_iter: usize,
    delta_threshold: f64,
    flop_name: &str,
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> FlopSolveResult {
    let buf_size = layout.total_size;
    let mut regret_sum = vec![0.0f64; buf_size];
    let mut strategy_sum = vec![0.0f64; buf_size];
    let mut final_delta = 0.0;
    let mut iterations_used = 0;

    for iter in 0..num_iterations {
        let iteration = iter as u64 + 1;
        let prev_regrets = regret_sum.clone();

        // Sample deals and traverse
        let seed = iteration * 1_000_003 + flop_name.len() as u64;
        let mut rng = SmallRng::seed_from_u64(seed);

        let mut dr = vec![0.0f64; buf_size];
        let mut ds = vec![0.0f64; buf_size];

        for _ in 0..samples_per_iter {
            let deal = sample_deal(bucket_map, flop, &mut rng);
            let Some((hb, ob, hero_hand, opp_hand, turn, river)) = deal else {
                continue;
            };
            let board = [flop[0], flop[1], flop[2], turn, river];

            for hero_pos in 0..2u8 {
                mccfr_traverse(
                    tree,
                    layout,
                    &regret_sum,
                    &mut dr,
                    &mut ds,
                    0,
                    hb,
                    ob,
                    hero_pos,
                    hero_hand,
                    opp_hand,
                    &board,
                    1.0,
                    1.0,
                    iteration,
                );
            }
        }

        add_buffers(&mut regret_sum, &dr);
        add_buffers(&mut strategy_sum, &ds);
        iterations_used = iter + 1;

        if iter >= 1 {
            final_delta =
                weighted_avg_strategy_delta(&prev_regrets, &regret_sum, layout, tree);
        }

        on_progress(BuildPhase::FlopProgress {
            flop_name: flop_name.to_string(),
            stage: FlopStage::Solving {
                iteration: iterations_used,
                max_iterations: num_iterations,
                delta: final_delta,
            },
        });

        if iter >= 1 && final_delta < delta_threshold {
            break;
        }
    }

    FlopSolveResult {
        strategy_sum,
        final_delta,
        iterations_used,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Deal sampling
// ──────────────────────────────────────────────────────────────────────────────

/// Sample a concrete deal: two non-conflicting hands from random buckets plus
/// random turn + river cards.
///
/// Returns `(hero_bucket, opp_bucket, hero_hand, opp_hand, turn_card, river_card)`.
#[allow(clippy::cast_possible_truncation)]
fn sample_deal(
    bucket_map: &FlopBucketMap,
    flop: &[Card; 3],
    rng: &mut SmallRng,
) -> Option<(u16, u16, [Card; 2], [Card; 2], Card, Card)> {
    let hb = rng.random_range(0..bucket_map.num_buckets as u16);
    let ob = rng.random_range(0..bucket_map.num_buckets as u16);

    let hero_hands = &bucket_map.bucket_hands[hb as usize];
    let opp_hands = &bucket_map.bucket_hands[ob as usize];
    if hero_hands.is_empty() || opp_hands.is_empty() {
        return None;
    }

    let hero_idx = rng.random_range(0..hero_hands.len());
    let (h1, h2) = hero_hands[hero_idx];

    // Find a non-conflicting opponent hand (try up to 10 times)
    let mut opp_hand = None;
    for _ in 0..10 {
        let idx = rng.random_range(0..opp_hands.len());
        let (o1, o2) = opp_hands[idx];
        if o1 != h1 && o1 != h2 && o2 != h1 && o2 != h2 {
            opp_hand = Some((o1, o2));
            break;
        }
    }
    let (o1, o2) = opp_hand?;

    // Build remaining deck and pick turn + river
    let used = [flop[0], flop[1], flop[2], h1, h2, o1, o2];
    let deck: Vec<Card> = hand_buckets::all_cards_vec()
        .into_iter()
        .filter(|c| !used.contains(c))
        .collect();
    if deck.len() < 2 {
        return None;
    }

    let t_idx = rng.random_range(0..deck.len());
    let mut r_idx = rng.random_range(0..deck.len() - 1);
    if r_idx >= t_idx {
        r_idx += 1;
    }

    Some((hb, ob, [h1, h2], [o1, o2], deck[t_idx], deck[r_idx]))
}

// ──────────────────────────────────────────────────────────────────────────────
// CFR traversal with concrete hands
// ──────────────────────────────────────────────────────────────────────────────

/// Recursive CFR traversal using concrete hands (not bucket transitions).
///
/// At chance nodes the board is already dealt, so we just pass through to
/// the single structural child. At showdown terminals we use `rank_hand`
/// for card-based evaluation.
#[allow(clippy::too_many_arguments)]
fn mccfr_traverse(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    snapshot: &[f64],
    dr: &mut [f64],
    ds: &mut [f64],
    node_idx: u32,
    hero_bucket: u16,
    opp_bucket: u16,
    hero_pos: u8,
    hero_hand: [Card; 2],
    opp_hand: [Card; 2],
    board: &[Card; 5],
    reach_hero: f64,
    reach_opp: f64,
    iteration: u64,
) -> f64 {
    match &tree.nodes[node_idx as usize] {
        PostflopNode::Terminal {
            terminal_type,
            pot_fraction,
        } => match terminal_type {
            PostflopTerminalType::Fold { folder } => {
                if *folder == hero_pos {
                    -pot_fraction / 2.0
                } else {
                    pot_fraction / 2.0
                }
            }
            PostflopTerminalType::Showdown => {
                let hero_rank = rank_hand(hero_hand, board);
                let opp_rank = rank_hand(opp_hand, board);
                let eq = match hero_rank.cmp(&opp_rank) {
                    std::cmp::Ordering::Greater => 1.0,
                    std::cmp::Ordering::Less => 0.0,
                    std::cmp::Ordering::Equal => 0.5,
                };
                eq * pot_fraction - pot_fraction / 2.0
            }
        },

        PostflopNode::Chance { children, .. } => {
            // Board already dealt — pass through to single structural child.
            debug_assert!(
                !children.is_empty(),
                "chance node must have at least one child"
            );
            mccfr_traverse(
                tree,
                layout,
                snapshot,
                dr,
                ds,
                children[0],
                hero_bucket,
                opp_bucket,
                hero_pos,
                hero_hand,
                opp_hand,
                board,
                reach_hero,
                reach_opp,
                iteration,
            )
        }

        PostflopNode::Decision {
            position, children, ..
        } => {
            let is_hero = *position == hero_pos;
            let bucket = if is_hero { hero_bucket } else { opp_bucket };
            let (start, _) = layout.slot(node_idx, bucket);
            let num_actions = children.len();

            let mut strategy = [0.0f64; MAX_POSTFLOP_ACTIONS];
            regret_matching_into(snapshot, start, &mut strategy[..num_actions]);

            if is_hero {
                let mut action_values = [0.0f64; MAX_POSTFLOP_ACTIONS];
                for (i, &child) in children.iter().enumerate() {
                    action_values[i] = mccfr_traverse(
                        tree,
                        layout,
                        snapshot,
                        dr,
                        ds,
                        child,
                        hero_bucket,
                        opp_bucket,
                        hero_pos,
                        hero_hand,
                        opp_hand,
                        board,
                        reach_hero * strategy[i],
                        reach_opp,
                        iteration,
                    );
                }
                let node_value: f64 = strategy[..num_actions]
                    .iter()
                    .zip(&action_values[..num_actions])
                    .map(|(s, v)| s * v)
                    .sum();

                #[allow(clippy::cast_precision_loss)]
                let weight = iteration as f64; // LCFR weighting
                for (i, val) in action_values[..num_actions].iter().enumerate() {
                    dr[start + i] += weight * reach_opp * (val - node_value);
                }
                for (i, &s) in strategy[..num_actions].iter().enumerate() {
                    ds[start + i] += weight * reach_hero * s;
                }
                node_value
            } else {
                // Opponent: weighted traversal
                children
                    .iter()
                    .enumerate()
                    .map(|(i, &child)| {
                        strategy[i]
                            * mccfr_traverse(
                                tree,
                                layout,
                                snapshot,
                                dr,
                                ds,
                                child,
                                hero_bucket,
                                opp_bucket,
                                hero_pos,
                                hero_hand,
                                opp_hand,
                                board,
                                reach_hero,
                                reach_opp * strategy[i],
                                iteration,
                            )
                    })
                    .sum()
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Value extraction
// ──────────────────────────────────────────────────────────────────────────────

/// Monte Carlo value extraction after convergence.
///
/// Samples random deals, evaluates using the averaged strategy, and
/// accumulates per `(hero_pos, hero_bucket, opp_bucket)`. Returns a flat
/// `Vec<f64>` of size `2 * n * n`.
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
fn mccfr_extract_values(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    strategy_sum: &[f64],
    bucket_map: &FlopBucketMap,
    flop: &[Card; 3],
    num_samples: usize,
    num_flop_buckets: usize,
    convergence_threshold: f64,
    flop_name: &str,
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> Vec<f64> {
    let n = num_flop_buckets;
    let mut values = vec![0.0f64; 2 * n * n];
    let mut counts = vec![0u32; 2 * n * n];
    let mut prev_means = vec![0.0f64; 2 * n * n];

    let mut rng = SmallRng::seed_from_u64(42);
    // Need at least 2 checkpoints to measure convergence.
    let min_samples = 2000usize.min(num_samples);

    let mut current_avg_delta = f64::INFINITY;
    for sample_idx in 0..num_samples {
        if sample_idx % 1000 == 0 {
            // Check convergence: weighted-average change in bucket pair means.
            // Weight by sample count so sparse cells don't dominate.
            if sample_idx >= min_samples {
                let mut weighted_delta_sum = 0.0f64;
                let mut weight_sum = 0u64;
                for (i, &sum) in values.iter().enumerate() {
                    if counts[i] > 0 {
                        let mean = sum / f64::from(counts[i]);
                        let delta = (mean - prev_means[i]).abs();
                        weighted_delta_sum += delta * f64::from(counts[i]);
                        weight_sum += u64::from(counts[i]);
                        prev_means[i] = mean;
                    }
                }
                let avg_delta = if weight_sum > 0 {
                    weighted_delta_sum / weight_sum as f64
                } else {
                    f64::INFINITY
                };
                current_avg_delta = avg_delta;
                if avg_delta < convergence_threshold {
                    on_progress(BuildPhase::FlopProgress {
                        flop_name: flop_name.to_string(),
                        stage: FlopStage::EstimatingEv {
                            sample: sample_idx,
                            total_samples: num_samples,
                            avg_delta: current_avg_delta,
                        },
                    });
                    break;
                }
            } else if sample_idx > 0 {
                // Snapshot current means for next comparison.
                for (i, &sum) in values.iter().enumerate() {
                    if counts[i] > 0 {
                        prev_means[i] = sum / f64::from(counts[i]);
                    }
                }
            }

            on_progress(BuildPhase::FlopProgress {
                flop_name: flop_name.to_string(),
                stage: FlopStage::EstimatingEv {
                    sample: sample_idx,
                    total_samples: num_samples,
                    avg_delta: current_avg_delta,
                },
            });
        }

        let Some((hb, ob, hero_hand, opp_hand, turn, river)) =
            sample_deal(bucket_map, flop, &mut rng)
        else {
            continue;
        };
        let board = [flop[0], flop[1], flop[2], turn, river];

        for hero_pos in 0..2u8 {
            let ev = mccfr_eval_with_avg_strategy(
                tree,
                layout,
                strategy_sum,
                0,
                hb,
                ob,
                hero_pos,
                hero_hand,
                opp_hand,
                &board,
            );
            let idx = hero_pos as usize * n * n + hb as usize * n + ob as usize;
            values[idx] += ev;
            counts[idx] += 1;
        }
    }

    // Normalize by count
    for (i, val) in values.iter_mut().enumerate() {
        if counts[i] > 0 {
            *val /= f64::from(counts[i]);
        }
    }

    values
}

/// Walk tree using averaged (converged) strategy with concrete hands.
///
/// Same structure as `mccfr_traverse` but read-only: uses
/// `normalize_strategy_sum` and does no regret updates.
#[allow(clippy::too_many_arguments)]
fn mccfr_eval_with_avg_strategy(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    strategy_sum: &[f64],
    node_idx: u32,
    hero_bucket: u16,
    opp_bucket: u16,
    hero_pos: u8,
    hero_hand: [Card; 2],
    opp_hand: [Card; 2],
    board: &[Card; 5],
) -> f64 {
    match &tree.nodes[node_idx as usize] {
        PostflopNode::Terminal {
            terminal_type,
            pot_fraction,
        } => match terminal_type {
            PostflopTerminalType::Fold { folder } => {
                if *folder == hero_pos {
                    -pot_fraction / 2.0
                } else {
                    pot_fraction / 2.0
                }
            }
            PostflopTerminalType::Showdown => {
                let hero_rank = rank_hand(hero_hand, board);
                let opp_rank = rank_hand(opp_hand, board);
                let eq = match hero_rank.cmp(&opp_rank) {
                    std::cmp::Ordering::Greater => 1.0,
                    std::cmp::Ordering::Less => 0.0,
                    std::cmp::Ordering::Equal => 0.5,
                };
                eq * pot_fraction - pot_fraction / 2.0
            }
        },

        PostflopNode::Chance { children, .. } => {
            debug_assert!(
                !children.is_empty(),
                "chance node must have at least one child"
            );
            mccfr_eval_with_avg_strategy(
                tree,
                layout,
                strategy_sum,
                children[0],
                hero_bucket,
                opp_bucket,
                hero_pos,
                hero_hand,
                opp_hand,
                board,
            )
        }

        PostflopNode::Decision {
            position, children, ..
        } => {
            let bucket = if *position == hero_pos {
                hero_bucket
            } else {
                opp_bucket
            };
            let (start, _) = layout.slot(node_idx, bucket);
            let num_actions = children.len();

            let strategy = normalize_strategy_sum(strategy_sum, start, num_actions);

            children
                .iter()
                .enumerate()
                .map(|(i, &child)| {
                    strategy[i]
                        * mccfr_eval_with_avg_strategy(
                            tree,
                            layout,
                            strategy_sum,
                            child,
                            hero_bucket,
                            opp_bucket,
                            hero_pos,
                            hero_hand,
                            opp_hand,
                            board,
                        )
                })
                .sum()
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::{Suit, Value};
    use crate::preflop::postflop_abstraction::annotate_streets;
    use test_macros::timed_test;

    fn card(v: Value, s: Suit) -> Card {
        Card::new(v, s)
    }

    /// A rainbow low flop: 2s 7h Qd — no suits overlap.
    fn test_flop() -> [Card; 3] {
        [
            card(Value::Two, Suit::Spade),
            card(Value::Seven, Suit::Heart),
            card(Value::Queen, Suit::Diamond),
        ]
    }

    /// Build a minimal config and tree for testing.
    fn test_config(num_buckets: u16) -> PostflopModelConfig {
        PostflopModelConfig {
            num_hand_buckets_flop: num_buckets,
            num_hand_buckets_turn: num_buckets,
            num_hand_buckets_river: num_buckets,
            bet_sizes: vec![1.0],
            max_raises_per_street: 1,
            postflop_solve_iterations: 50,
            postflop_solve_samples: 0,
            postflop_sprs: vec![3.5],
            rebucket_rounds: 1,
            cfr_delta_threshold: 0.001,
            max_flop_boards: 1,
            fixed_flops: None,
            equity_rollout_fraction: 1.0,
            solve_type: super::super::postflop_model::PostflopSolveType::Mccfr,
            mccfr_sample_pct: 0.01,
            value_extraction_samples: 1000,
            ev_convergence_threshold: 0.001,
        }
    }

    #[timed_test]
    fn flop_bucket_map_construction() {
        let flop = test_flop();
        // Build simple assignments: all hands → bucket 0 or 1
        let assignments: Vec<u16> = (0..169).map(|i| (i % 3) as u16).collect();
        let map = FlopBucketMap::build(&assignments, 3, &flop);

        assert_eq!(map.num_buckets, 3);
        // Every bucket should have some hands (169 hands across 3 buckets)
        for b in 0..3 {
            assert!(
                !map.bucket_hands[b].is_empty(),
                "bucket {b} should not be empty"
            );
        }
        // Verify no card conflicts with flop
        for bucket_hands in &map.bucket_hands {
            for &(c1, c2) in bucket_hands {
                assert!(
                    !hand_buckets::board_conflicts([c1, c2], &flop),
                    "hand ({c1}, {c2}) conflicts with flop"
                );
            }
        }
    }

    #[timed_test]
    fn mccfr_traverse_fold_terminal() {
        let config = test_config(3);
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let nfb = 3;
        let layout = PostflopLayout::build(&tree, &node_streets, nfb, nfb, nfb);

        let buf_size = layout.total_size;
        let snapshot = vec![0.0f64; buf_size];
        let mut dr = vec![0.0f64; buf_size];
        let mut ds = vec![0.0f64; buf_size];

        // Find a fold terminal
        let fold_node = tree.nodes.iter().enumerate().find_map(|(i, n)| {
            if let PostflopNode::Terminal {
                terminal_type: PostflopTerminalType::Fold { folder },
                pot_fraction,
            } = n
            {
                Some((i as u32, *folder, *pot_fraction))
            } else {
                None
            }
        });

        if let Some((node_idx, folder, pot_fraction)) = fold_node {
            let hero_hand = [
                card(Value::Ace, Suit::Spade),
                card(Value::King, Suit::Club),
            ];
            let opp_hand = [
                card(Value::Ten, Suit::Heart),
                card(Value::Nine, Suit::Diamond),
            ];
            let board = [
                card(Value::Two, Suit::Spade),
                card(Value::Seven, Suit::Heart),
                card(Value::Queen, Suit::Diamond),
                card(Value::Three, Suit::Club),
                card(Value::Eight, Suit::Heart),
            ];

            let ev = mccfr_traverse(
                &tree,
                &layout,
                &snapshot,
                &mut dr,
                &mut ds,
                node_idx,
                0,
                1,
                0,
                hero_hand,
                opp_hand,
                &board,
                1.0,
                1.0,
                1,
            );

            if folder == 0 {
                // Hero folds → loses
                assert!(
                    (ev - (-pot_fraction / 2.0)).abs() < 1e-9,
                    "fold: hero folds, ev={ev}, expected {}",
                    -pot_fraction / 2.0
                );
            } else {
                // Opponent folds → hero wins
                assert!(
                    (ev - (pot_fraction / 2.0)).abs() < 1e-9,
                    "fold: opp folds, ev={ev}, expected {}",
                    pot_fraction / 2.0
                );
            }
        }
    }

    #[timed_test]
    fn mccfr_traverse_showdown_terminal() {
        let config = test_config(3);
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let nfb = 3;
        let layout = PostflopLayout::build(&tree, &node_streets, nfb, nfb, nfb);

        let buf_size = layout.total_size;
        let snapshot = vec![0.0f64; buf_size];
        let mut dr = vec![0.0f64; buf_size];
        let mut ds = vec![0.0f64; buf_size];

        // Find a showdown terminal
        let showdown_node = tree.nodes.iter().enumerate().find_map(|(i, n)| {
            if let PostflopNode::Terminal {
                terminal_type: PostflopTerminalType::Showdown,
                pot_fraction,
            } = n
            {
                Some((i as u32, *pot_fraction))
            } else {
                None
            }
        });

        if let Some((node_idx, pot_fraction)) = showdown_node {
            // AA vs KK on a low board — AA should win
            let hero_hand = [
                card(Value::Ace, Suit::Spade),
                card(Value::Ace, Suit::Club),
            ];
            let opp_hand = [
                card(Value::King, Suit::Heart),
                card(Value::King, Suit::Diamond),
            ];
            let board = [
                card(Value::Two, Suit::Spade),
                card(Value::Seven, Suit::Heart),
                card(Value::Queen, Suit::Diamond),
                card(Value::Three, Suit::Club),
                card(Value::Eight, Suit::Diamond),
            ];

            let ev = mccfr_traverse(
                &tree,
                &layout,
                &snapshot,
                &mut dr,
                &mut ds,
                node_idx,
                0,
                1,
                0,
                hero_hand,
                opp_hand,
                &board,
                1.0,
                1.0,
                1,
            );

            // AA beats KK → eq=1.0 → ev = 1.0 * pot_frac - pot_frac/2 = pot_frac/2
            let expected = pot_fraction / 2.0;
            assert!(
                (ev - expected).abs() < 1e-9,
                "showdown: AA vs KK, ev={ev}, expected={expected}"
            );
        }
    }

    #[timed_test]
    fn mccfr_solve_tiny_tree_converges() {
        let config = test_config(3);
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let nfb = 3;
        let layout = PostflopLayout::build(&tree, &node_streets, nfb, nfb, nfb);

        let flop = test_flop();
        let assignments: Vec<u16> = (0..169).map(|i| (i % 3) as u16).collect();
        let bucket_map = FlopBucketMap::build(&assignments, nfb, &flop);

        let result = mccfr_solve_one_flop(
            &tree,
            &layout,
            &bucket_map,
            &flop,
            50,
            20,
            0.0001,
            "test",
            &|_| {},
        );

        assert!(result.iterations_used > 0, "should complete at least 1 iteration");
        // Strategy sum should have non-zero values
        let has_nonzero = result.strategy_sum.iter().any(|&v| v.abs() > 1e-15);
        assert!(has_nonzero, "strategy_sum should have non-zero entries");
    }

    #[timed_test]
    fn mccfr_extract_values_correct_dimensions() {
        let config = test_config(3);
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let nfb = 3;
        let layout = PostflopLayout::build(&tree, &node_streets, nfb, nfb, nfb);

        let flop = test_flop();
        let assignments: Vec<u16> = (0..169).map(|i| (i % 3) as u16).collect();
        let bucket_map = FlopBucketMap::build(&assignments, nfb, &flop);

        // Solve first
        let result = mccfr_solve_one_flop(
            &tree,
            &layout,
            &bucket_map,
            &flop,
            20,
            10,
            0.001,
            "test",
            &|_| {},
        );

        let values =
            mccfr_extract_values(&tree, &layout, &result.strategy_sum, &bucket_map, &flop, 500, nfb, 0.001, "test", &|_| {});

        assert_eq!(values.len(), 2 * nfb * nfb, "values should be 2*n*n");
        // All values should be finite
        assert!(
            values.iter().all(|v| v.is_finite()),
            "all values should be finite"
        );
    }

    #[test]
    #[ignore = "slow: full MCCFR pipeline for single flop"]
    fn build_mccfr_single_flop() {
        let config = test_config(5);
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let nfb = 5;
        let layout = PostflopLayout::build(&tree, &node_streets, nfb, nfb, nfb);

        let flop = [test_flop()];

        let (buckets, values) =
            build_mccfr(&config, &tree, &layout, &node_streets, &flop, &|_| {});

        // Verify bucket structure
        assert_eq!(buckets.flop.len(), 1, "one flop");
        assert_eq!(
            buckets.flop[0].len(),
            169,
            "169 canonical hands bucketed"
        );
        assert!(buckets.turn.is_empty(), "MCCFR has no turn buckets");
        assert!(buckets.river.is_empty(), "MCCFR has no river buckets");

        // Verify values dimensions
        assert_eq!(values.len(), 2 * nfb * nfb, "values = 2*n*n");

        // Check approximately zero-sum: sum of pos0 + pos1 should be near 0
        let n = nfb;
        let mut total_ev = 0.0f64;
        let mut count = 0;
        for hb in 0..n as u16 {
            for ob in 0..n as u16 {
                let ev0 = values.get_by_flop(0, 0, hb, ob);
                let ev1 = values.get_by_flop(0, 1, hb, ob);
                // These are from the same bucket matchup; in a fair game
                // the sum over all matchups should tend toward zero.
                total_ev += ev0 + ev1;
                count += 1;
            }
        }
        // Zero-sum check: average should be close to 0
        let avg = total_ev / count as f64;
        assert!(
            avg.abs() < 0.5,
            "average EV across all matchups should be roughly zero-sum, got {avg}"
        );
    }
}
