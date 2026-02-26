//! MCCFR postflop solve backend.
//!
//! Instead of abstract bucket-to-bucket transitions at street changes, this
//! backend uses **concrete hands** (actual `Card` pairs). For each sample it
//! picks two non-conflicting hands from canonical hand combos and a random turn
//! + river. At showdown terminals it uses `rank_hand()` to evaluate actual hands
//! against the full 5-card board. Strategies and regrets are indexed by canonical
//! hand index (0..168).

use std::sync::atomic::{AtomicUsize, Ordering};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::postflop_abstraction::{
    BuildPhase, FlopSolveResult, FlopStage, PostflopLayout, PostflopValues,
    add_buffers, regret_matching_into, normalize_strategy_sum, weighted_avg_strategy_delta,
    MAX_POSTFLOP_ACTIONS,
};
use super::postflop_hands::{build_combo_map, all_cards_vec, NUM_CANONICAL_HANDS};
use super::postflop_model::PostflopModelConfig;
use super::postflop_tree::{PostflopNode, PostflopTerminalType, PostflopTree};
use crate::abstraction::Street;
use crate::poker::Card;
use crate::showdown_equity::rank_hand;

// ──────────────────────────────────────────────────────────────────────────────
// Entry point
// ──────────────────────────────────────────────────────────────────────────────

/// Build postflop abstraction using MCCFR with sampled concrete hands.
///
/// Pipeline per flop (parallelised via rayon):
/// 1. Build combo map — each canonical hand (0..168) maps to its concrete combos.
/// 2. Run MCCFR training loop (`mccfr_solve_one_flop`).
/// 3. Extract EV table from converged strategy (`mccfr_extract_values`).
///
/// Returns `PostflopValues` indexed by canonical hand indices (169).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_mccfr(
    config: &PostflopModelConfig,
    tree: &PostflopTree,
    layout: &PostflopLayout,
    node_streets: &[Street],
    flops: &[[Card; 3]],
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> PostflopValues {
    let num_flops = flops.len();
    let n = NUM_CANONICAL_HANDS;
    let num_iterations = config.postflop_solve_iterations as usize;
    let _ = node_streets; // Streets are embedded in the layout already.
    let completed = AtomicUsize::new(0);

    let results: Vec<Vec<f64>> = (0..num_flops)
        .into_par_iter()
        .map(|flop_idx| {
            let flop = &flops[flop_idx];
            let flop_name = format!("{}{}{}", flop[0], flop[1], flop[2]);

            // Build combo map — no clustering needed
            let combo_map = build_combo_map(flop);

            // Compute sample count
            let total_non_conflicting: usize = combo_map.iter().map(Vec::len).sum();
            let live_runouts = 45usize * 44;
            let total_space = total_non_conflicting.saturating_mul(live_runouts);
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]
            let samples_per_iter = ((total_space as f64 * config.mccfr_sample_pct) as usize
                / num_iterations.max(1)).max(1);

            // Solve
            let result = mccfr_solve_one_flop(
                tree, layout, &combo_map, flop,
                num_iterations, samples_per_iter,
                config.cfr_convergence_threshold,
                &flop_name, on_progress,
            );

            // Extract values
            let values = mccfr_extract_values(
                tree, layout, &result.strategy_sum, &combo_map, flop,
                config.value_extraction_samples as usize, n,
                config.ev_convergence_threshold, &flop_name, on_progress,
            );

            on_progress(BuildPhase::FlopProgress { flop_name: flop_name.clone(), stage: FlopStage::Done });
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            on_progress(BuildPhase::MccfrFlopsCompleted { completed: done, total: num_flops });

            values
        })
        .collect();

    // Assemble
    let mut all_values = vec![0.0f64; num_flops * 2 * n * n];
    for (flop_idx, vals) in results.into_iter().enumerate() {
        let offset = flop_idx * 2 * n * n;
        let copy_len = vals.len().min(2 * n * n);
        all_values[offset..offset + copy_len].copy_from_slice(&vals[..copy_len]);
    }

    PostflopValues::from_raw(all_values, n, num_flops)
}

// ──────────────────────────────────────────────────────────────────────────────
// Core solve loop
// ──────────────────────────────────────────────────────────────────────────────

/// Training loop for a single flop using MCCFR with concrete hands.
///
/// Convergence is measured via `weighted_avg_strategy_delta`: the regret-weighted
/// average of max per-action strategy probability change between consecutive
/// iterations. This avoids the decay problem of `avg_positive_regret_flat` which
/// divides cumulative regret by buffer_size × iterations.
#[allow(clippy::too_many_arguments)]
fn mccfr_solve_one_flop(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    combo_map: &[Vec<(Card, Card)>],
    flop: &[Card; 3],
    num_iterations: usize,
    samples_per_iter: usize,
    convergence_threshold: f64,
    flop_name: &str,
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> FlopSolveResult {
    let buf_size = layout.total_size;
    let mut regret_sum = vec![0.0f64; buf_size];
    let mut strategy_sum = vec![0.0f64; buf_size];
    let mut current_delta = 0.0;
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
            let deal = sample_deal(combo_map, flop, &mut rng);
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
            current_delta = weighted_avg_strategy_delta(&prev_regrets, &regret_sum, layout, tree);
        }

        on_progress(BuildPhase::FlopProgress {
            flop_name: flop_name.to_string(),
            stage: FlopStage::Solving {
                iteration: iterations_used,
                max_iterations: num_iterations,
                delta: current_delta,
                metric_label: "\u{03b4}".to_string(),
            },
        });

        if iter >= 1 && current_delta < convergence_threshold {
            break;
        }
    }

    FlopSolveResult {
        strategy_sum,
        delta: current_delta,
        iterations_used,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Deal sampling
// ──────────────────────────────────────────────────────────────────────────────

/// Sample a concrete deal: two non-conflicting hands from random canonical hand
/// indices plus random turn + river cards.
///
/// Returns `(hero_hand_idx, opp_hand_idx, hero_hand, opp_hand, turn_card, river_card)`.
#[allow(clippy::cast_possible_truncation)]
fn sample_deal(
    combo_map: &[Vec<(Card, Card)>],
    flop: &[Card; 3],
    rng: &mut SmallRng,
) -> Option<(u16, u16, [Card; 2], [Card; 2], Card, Card)> {
    let n = combo_map.len();
    let hero_hand_idx = rng.random_range(0..n) as u16;
    let opp_hand_idx = rng.random_range(0..n) as u16;

    let hero_combos = &combo_map[hero_hand_idx as usize];
    let opp_combos = &combo_map[opp_hand_idx as usize];
    if hero_combos.is_empty() || opp_combos.is_empty() {
        return None;
    }

    let hero_idx = rng.random_range(0..hero_combos.len());
    let (h1, h2) = hero_combos[hero_idx];

    let mut opp_hand = None;
    for _ in 0..10 {
        let idx = rng.random_range(0..opp_combos.len());
        let (o1, o2) = opp_combos[idx];
        if o1 != h1 && o1 != h2 && o2 != h1 && o2 != h2 {
            opp_hand = Some((o1, o2));
            break;
        }
    }
    let (o1, o2) = opp_hand?;

    let used = [flop[0], flop[1], flop[2], h1, h2, o1, o2];
    let deck: Vec<Card> = all_cards_vec()
        .into_iter()
        .filter(|c| !used.contains(c))
        .collect();
    if deck.len() < 2 { return None; }

    let t_idx = rng.random_range(0..deck.len());
    let mut r_idx = rng.random_range(0..deck.len() - 1);
    if r_idx >= t_idx { r_idx += 1; }

    Some((hero_hand_idx, opp_hand_idx, [h1, h2], [o1, o2], deck[t_idx], deck[r_idx]))
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
/// accumulates per `(hero_pos, hero_hand_idx, opp_hand_idx)`. Returns a flat
/// `Vec<f64>` of size `2 * n * n`.
#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
/// Extract EV values by sampling a fixed number of runouts per hand pair.
///
/// `samples_per_pair` runouts are drawn for each of the `n × n` canonical hand
/// pairs, guaranteeing 100% coverage. Pairs where no non-conflicting concrete
/// combos exist on this flop are marked NaN (impossible on this board).
fn mccfr_extract_values(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    strategy_sum: &[f64],
    combo_map: &[Vec<(Card, Card)>],
    flop: &[Card; 3],
    samples_per_pair: usize,
    num_flop_buckets: usize,
    _convergence_threshold: f64,
    flop_name: &str,
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> Vec<f64> {
    let n = num_flop_buckets;
    let mut values = vec![0.0f64; 2 * n * n];
    let mut counts = vec![0u32; 2 * n * n];

    let mut rng = SmallRng::seed_from_u64(42);
    let total_pairs = n * n;
    let mut pairs_done = 0usize;

    for hb in 0..n as u16 {
        let hero_combos = &combo_map[hb as usize];
        if hero_combos.is_empty() {
            pairs_done += n;
            continue;
        }

        for ob in 0..n as u16 {
            pairs_done += 1;
            if pairs_done % 1000 == 0 {
                on_progress(BuildPhase::FlopProgress {
                    flop_name: flop_name.to_string(),
                    stage: FlopStage::EstimatingEv {
                        sample: pairs_done,
                        total_samples: total_pairs,
                    },
                });
            }

            let opp_combos = &combo_map[ob as usize];
            if opp_combos.is_empty() {
                continue;
            }

            for _ in 0..samples_per_pair {
                let Some((hero_hand, opp_hand, turn, river)) =
                    sample_runout_for_pair(hero_combos, opp_combos, flop, &mut rng)
                else {
                    break; // no non-conflicting combos exist for this pair
                };
                let board = [flop[0], flop[1], flop[2], turn, river];

                for hero_pos in 0..2u8 {
                    let ev = mccfr_eval_with_avg_strategy(
                        tree, layout, strategy_sum, 0,
                        hb, ob, hero_pos, hero_hand, opp_hand, &board,
                    );
                    let idx = hero_pos as usize * n * n + hb as usize * n + ob as usize;
                    values[idx] += ev;
                    counts[idx] += 1;
                }
            }
        }
    }

    on_progress(BuildPhase::FlopProgress {
        flop_name: flop_name.to_string(),
        stage: FlopStage::EstimatingEv {
            sample: total_pairs,
            total_samples: total_pairs,
        },
    });

    // Normalize; unsampled cells (impossible combos on this flop) become NaN.
    for (i, val) in values.iter_mut().enumerate() {
        if counts[i] > 0 {
            *val /= f64::from(counts[i]);
        } else {
            *val = f64::NAN;
        }
    }

    values
}

/// Sample a single runout (concrete hero/opp combo + turn + river) for a
/// specific canonical hand pair on a given flop.
fn sample_runout_for_pair(
    hero_combos: &[(Card, Card)],
    opp_combos: &[(Card, Card)],
    flop: &[Card; 3],
    rng: &mut SmallRng,
) -> Option<([Card; 2], [Card; 2], Card, Card)> {
    let hero_idx = rng.random_range(0..hero_combos.len());
    let (h1, h2) = hero_combos[hero_idx];

    // Find a non-conflicting opponent combo (up to 20 attempts).
    let mut opp_hand = None;
    for _ in 0..20 {
        let idx = rng.random_range(0..opp_combos.len());
        let (o1, o2) = opp_combos[idx];
        if o1 != h1 && o1 != h2 && o2 != h1 && o2 != h2 {
            opp_hand = Some((o1, o2));
            break;
        }
    }
    let (o1, o2) = opp_hand?;

    let used = [flop[0], flop[1], flop[2], h1, h2, o1, o2];
    let deck: Vec<Card> = all_cards_vec()
        .into_iter()
        .filter(|c| !used.contains(c))
        .collect();
    if deck.len() < 2 { return None; }

    let t_idx = rng.random_range(0..deck.len());
    let mut r_idx = rng.random_range(0..deck.len() - 1);
    if r_idx >= t_idx { r_idx += 1; }

    Some(([h1, h2], [o1, o2], deck[t_idx], deck[r_idx]))
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
    use crate::preflop::postflop_model::PostflopSolveType;
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

    /// Build a minimal config for testing.
    fn test_config() -> PostflopModelConfig {
        PostflopModelConfig {
            bet_sizes: vec![1.0],
            max_raises_per_street: 1,
            postflop_solve_iterations: 50,
            postflop_sprs: vec![3.5],
            cfr_convergence_threshold: 0.01,
            max_flop_boards: 1,
            fixed_flops: None,
            solve_type: PostflopSolveType::Mccfr,
            mccfr_sample_pct: 0.01,
            value_extraction_samples: 1000,
            ev_convergence_threshold: 0.001,
        }
    }

    #[timed_test]
    fn combo_map_construction() {
        let flop = test_flop();
        let combo_map = build_combo_map(&flop);

        assert_eq!(combo_map.len(), NUM_CANONICAL_HANDS);
        // Every hand should have some combos (except maybe those conflicting with flop)
        let non_empty = combo_map.iter().filter(|c| !c.is_empty()).count();
        assert!(non_empty > 100, "most hands should have non-empty combos, got {non_empty}");
        // Verify no card conflicts with flop
        for combos in &combo_map {
            for &(c1, c2) in combos {
                assert!(
                    !super::super::postflop_hands::board_conflicts([c1, c2], &flop),
                    "hand ({c1}, {c2}) conflicts with flop"
                );
            }
        }
    }

    #[timed_test]
    fn mccfr_traverse_fold_terminal() {
        let config = test_config();
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);

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
        let config = test_config();
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);

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
        let config = test_config();
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);

        let flop = test_flop();
        let combo_map = build_combo_map(&flop);

        let result = mccfr_solve_one_flop(
            &tree,
            &layout,
            &combo_map,
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
    fn mccfr_extract_values_subset() {
        // Use only the first 10 canonical hands to keep the test fast (10² = 100 pairs).
        let config = test_config();
        let n_subset = 10;
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let layout = PostflopLayout::build(&tree, &node_streets, n_subset, n_subset, n_subset);

        let flop = test_flop();
        let full_combo_map = build_combo_map(&flop);
        let combo_map: Vec<Vec<(Card, Card)>> = full_combo_map[..n_subset].to_vec();

        // Solve with the subset
        let result = mccfr_solve_one_flop(
            &tree,
            &layout,
            &combo_map,
            &flop,
            20,
            10,
            0.001,
            "test",
            &|_| {},
        );

        let values = mccfr_extract_values(
            &tree, &layout, &result.strategy_sum, &combo_map, &flop,
            3, n_subset, 0.001, "test", &|_| {},
        );

        assert_eq!(values.len(), 2 * n_subset * n_subset, "values should be 2*n*n");
        // Values should be finite or NaN (NaN = impossible combo on this flop)
        let finite_count = values.iter().filter(|v| v.is_finite()).count();
        assert!(
            finite_count > 0,
            "should have at least some finite values"
        );
        assert!(
            values.iter().all(|v| v.is_finite() || v.is_nan()),
            "values should be finite or NaN, no infinities"
        );
    }

    #[timed_test]
    fn sample_runout_for_pair_produces_valid_deals() {
        let flop = test_flop();
        let combo_map = build_combo_map(&flop);
        let mut rng = SmallRng::seed_from_u64(42);

        // AA combos (index 0) should have valid deals against KK (index 1)
        let hero_combos = &combo_map[0];
        let opp_combos = &combo_map[1];
        assert!(!hero_combos.is_empty(), "AA should have combos");
        assert!(!opp_combos.is_empty(), "KK should have combos");

        let mut got_deal = false;
        for _ in 0..10 {
            if let Some((hero, opp, turn, river)) =
                sample_runout_for_pair(hero_combos, opp_combos, &flop, &mut rng)
            {
                // Cards should all be distinct
                let all = [flop[0], flop[1], flop[2], hero[0], hero[1], opp[0], opp[1], turn, river];
                for i in 0..all.len() {
                    for j in (i+1)..all.len() {
                        assert_ne!(all[i], all[j], "duplicate card in deal");
                    }
                }
                got_deal = true;
                break;
            }
        }
        assert!(got_deal, "should produce at least one valid deal");
    }

    #[test]
    #[ignore = "slow: full MCCFR pipeline for single flop"]
    fn build_mccfr_single_flop() {
        let config = test_config();
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);

        let flop = [test_flop()];

        let values =
            build_mccfr(&config, &tree, &layout, &node_streets, &flop, &|_| {});

        // Verify values dimensions
        assert_eq!(values.len(), 2 * n * n, "values = 2*n*n");

        // Check approximately zero-sum: sum of pos0 + pos1 should be near 0
        let mut total_ev = 0.0f64;
        let mut count = 0;
        for hb in 0..n as u16 {
            for ob in 0..n as u16 {
                let ev0 = values.get_by_flop(0, 0, hb, ob);
                let ev1 = values.get_by_flop(0, 1, hb, ob);
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
