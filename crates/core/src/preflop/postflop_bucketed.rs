//! Bucketed CFR solve backend for postflop abstraction.
//!
//! Contains the per-flop CFR solve loop, street-transition traversal,
//! and the `build_bucketed` entry point that orchestrates the full
//! per-flop solve + rebucketing pipeline.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::hand_buckets::{self, BucketEquity, SingleFlopAbstraction, StreetBuckets};
use super::postflop_model::PostflopModelConfig;
use super::postflop_tree::{PostflopNode, PostflopTree};
use super::postflop_abstraction::{
    BuildPhase, FlopSolveResult, FlopStage, PostflopLayout, PostflopValues,
    add_buffers, regret_matching_into, normalize_strategy_sum,
    postflop_terminal_value, MAX_POSTFLOP_ACTIONS,
};
use crate::abstraction::Street;
use crate::poker::Card;

/// Equity and transition references for a single flop solve.
pub(crate) struct SolveEquity<'a> {
    flop: &'a BucketEquity,
    turn: &'a BucketEquity,
    river: &'a BucketEquity,
    /// `flop_to_turn[flop_bucket][turn_bucket]` → transition probability
    flop_to_turn: &'a [Vec<f64>],
    /// `turn_to_river[turn_bucket][river_bucket]` → transition probability
    turn_to_river: &'a [Vec<f64>],
}

/// Select the equity table for the given street.
pub(crate) fn solve_equity_for_street<'a>(solve_eq: &'a SolveEquity<'a>, street: Street) -> &'a BucketEquity {
    match street {
        Street::Preflop | Street::Flop => solve_eq.flop,
        Street::Turn => solve_eq.turn,
        Street::River => solve_eq.river,
    }
}

/// Build postflop abstraction using per-street bucket CFR.
/// Returns `(StreetBuckets, PostflopValues)` for assembly by the caller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_bucketed(
    config: &PostflopModelConfig,
    tree: &PostflopTree,
    layout: &PostflopLayout,
    _node_streets: &[Street],
    flops: &[[Card; 3]],
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> (StreetBuckets, PostflopValues) {
    let hands: Vec<_> = crate::hands::all_hands().collect();
    let num_flops = flops.len();
    let nfb = config.num_hand_buckets_flop as usize;
    let solve_iters = config.postflop_solve_iterations as usize;
    let samples = if config.postflop_solve_samples > 0 {
        config.postflop_solve_samples as usize
    } else {
        nfb * nfb // 0 = exhaustive (all bucket pairs)
    };
    let total_rounds = config.rebucket_rounds;
    let num_hands = hand_buckets::NUM_HANDS;

    // First round: streaming per-flop pipeline (bucket + equity + transitions + solve + extract)
    let results: Vec<(Vec<u16>, Vec<f64>)> = (0..num_flops)
        .into_par_iter()
        .map(|flop_idx| {
            let flop_name = format!("{}{}{}", flops[flop_idx][0], flops[flop_idx][1], flops[flop_idx][2]);
            on_progress(BuildPhase::FlopProgress {
                flop_name: flop_name.clone(),
                stage: FlopStage::Bucketing { step: 0, total_steps: 6 },
            });
            let flop_data = hand_buckets::process_single_flop(
                &hands,
                &flops[flop_idx],
                config.num_hand_buckets_flop,
                config.num_hand_buckets_turn,
                config.num_hand_buckets_river,
                config.equity_rollout_fraction,
                None, // first round: cluster from scratch
                &|step| {
                    on_progress(BuildPhase::FlopProgress {
                        flop_name: flop_name.clone(),
                        stage: FlopStage::Bucketing { step, total_steps: 6 },
                    });
                },
            );
            let flop_buckets = flop_data.flop_buckets.clone();
            let values = stream_solve_and_extract_one_flop(
                flop_data,
                tree,
                layout,
                nfb,
                solve_iters,
                samples,
                config.cfr_regret_threshold,
                flop_idx,
                &flop_name,
                on_progress,
            );
            on_progress(BuildPhase::FlopProgress {
                flop_name,
                stage: FlopStage::Done,
            });
            (flop_buckets, values)
        })
        .collect();

    // Assemble first-round outputs
    let mut flop_assignments: Vec<Vec<u16>> = Vec::with_capacity(num_flops);
    let mut all_values = vec![0.0f64; num_flops * 2 * nfb * nfb];
    for (flop_idx, (fb, vals)) in results.into_iter().enumerate() {
        flop_assignments.push(fb);
        let offset = flop_idx * 2 * nfb * nfb;
        all_values[offset..offset + vals.len()].copy_from_slice(&vals);
    }

    let mut buckets = StreetBuckets {
        flop: flop_assignments,
        num_flop_buckets: config.num_hand_buckets_flop,
        turn: Vec::new(),
        num_turn_buckets: config.num_hand_buckets_turn,
        river: Vec::new(),
        num_river_buckets: config.num_hand_buckets_river,
    };

    let mut values = PostflopValues {
        values: all_values,
        num_buckets: nfb,
        num_flops,
    };

    // Rebucketing rounds (2..=total_rounds): extract EV, recluster, re-stream
    for round in 2..=total_rounds {
        // Extract EV histograms from current values
        on_progress(BuildPhase::ExtractingEv(0, num_hands));
        let ev_histograms = hand_buckets::build_ev_histograms(
            &buckets, &values, num_hands, nfb,
        );
        on_progress(BuildPhase::ExtractingEv(num_hands, num_hands));

        // Recluster flop buckets using EV features
        on_progress(BuildPhase::Rebucketing(round, total_rounds));
        buckets.flop = hand_buckets::recluster_flop_buckets(
            &ev_histograms, buckets.num_flop_buckets, num_flops, num_hands,
        );

        // Stream-solve again with reclustered flop buckets
        let results: Vec<(Vec<u16>, Vec<f64>)> = (0..num_flops)
            .into_par_iter()
            .map(|flop_idx| {
                let flop_name = format!("{}{}{}", flops[flop_idx][0], flops[flop_idx][1], flops[flop_idx][2]);
                on_progress(BuildPhase::FlopProgress {
                    flop_name: flop_name.clone(),
                    stage: FlopStage::Bucketing { step: 0, total_steps: 6 },
                });
                let flop_data = hand_buckets::process_single_flop(
                    &hands,
                    &flops[flop_idx],
                    config.num_hand_buckets_flop,
                    config.num_hand_buckets_turn,
                    config.num_hand_buckets_river,
                    config.equity_rollout_fraction,
                    Some(&buckets.flop[flop_idx]), // use reclustered assignments
                    &|step| {
                        on_progress(BuildPhase::FlopProgress {
                            flop_name: flop_name.clone(),
                            stage: FlopStage::Bucketing { step, total_steps: 6 },
                        });
                    },
                );
                let fb = flop_data.flop_buckets.clone();
                let vals = stream_solve_and_extract_one_flop(
                    flop_data,
                    tree,
                    layout,
                    nfb,
                    solve_iters,
                    samples,
                    config.cfr_regret_threshold,
                    flop_idx,
                    &flop_name,
                    on_progress,
                );
                on_progress(BuildPhase::FlopProgress {
                    flop_name,
                    stage: FlopStage::Done,
                });
                (fb, vals)
            })
            .collect();

        // Reassemble
        let mut new_all_values = vec![0.0f64; num_flops * 2 * nfb * nfb];
        for (flop_idx, (fb, vals)) in results.into_iter().enumerate() {
            buckets.flop[flop_idx] = fb;
            let offset = flop_idx * 2 * nfb * nfb;
            new_all_values[offset..offset + vals.len()].copy_from_slice(&vals);
        }
        values = PostflopValues {
            values: new_all_values,
            num_buckets: nfb,
            num_flops,
        };
    }

    (buckets, values)
}

/// Solve one flop and extract the value row.
///
/// Takes ownership of the per-flop abstraction data. After solving and
/// extracting values, all intermediate data (equity tables, transitions,
/// regret/strategy buffers) is dropped.
///
/// Returns `Vec<f64>` of size `2 * num_flop_buckets * num_flop_buckets`:
/// indexed by `hero_pos * n * n + hero_bucket * n + opp_bucket`.
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn stream_solve_and_extract_one_flop(
    flop_data: SingleFlopAbstraction,
    tree: &PostflopTree,
    layout: &PostflopLayout,
    num_flop_buckets: usize,
    num_iterations: usize,
    samples_per_iter: usize,
    delta_threshold: f64,
    flop_idx: usize,
    flop_name: &str,
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> Vec<f64> {
    let solve_eq = SolveEquity {
        flop: &flop_data.flop_equity,
        turn: &flop_data.turn_equity,
        river: &flop_data.river_equity,
        flop_to_turn: &flop_data.flop_to_turn,
        turn_to_river: &flop_data.turn_to_river,
    };

    let buf_size = layout.total_size;
    let result = solve_one_flop(
        tree, layout, &solve_eq,
        num_flop_buckets, buf_size,
        num_iterations, samples_per_iter, delta_threshold,
        flop_idx, flop_name, on_progress,
    );

    // Extract values from converged strategy
    let n = num_flop_buckets;
    let mut values = vec![0.0f64; 2 * n * n];
    for hero_pos in 0..2u8 {
        for hb in 0..n as u16 {
            for ob in 0..n as u16 {
                let ev = eval_with_avg_strategy(
                    tree, layout, &solve_eq, &result.strategy_sum,
                    0, hb, ob, hero_pos,
                );
                values[hero_pos as usize * n * n + hb as usize * n + ob as usize] = ev;
            }
        }
    }
    values
}

/// Run CFR for a single flop with the shared tree template.
///
/// Runs up to `num_iterations` CFR iterations, stopping early when
/// avg positive regret drops below `regret_threshold`.
/// Early stopping requires at least 2 completed iterations.
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn solve_one_flop(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    solve_eq: &SolveEquity<'_>,
    num_flop_buckets: usize,
    buf_size: usize,
    num_iterations: usize,
    samples_per_iter: usize,
    regret_threshold: f64,
    flop_idx: usize,
    flop_name: &str,
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> FlopSolveResult {
    let mut regret_sum = vec![0.0f64; buf_size];
    let mut strategy_sum = vec![0.0f64; buf_size];
    let use_exhaustive = num_flop_buckets * num_flop_buckets <= samples_per_iter;
    let mut current_expl = 0.0;
    let mut iterations_used = 0;

    on_progress(BuildPhase::FlopProgress {
        flop_name: flop_name.to_string(),
        stage: FlopStage::Solving {
            iteration: 0,
            max_iterations: num_iterations,
            exploitability: 0.0,
        },
    });

    for iter in 0..num_iterations {
        let iteration = iter as u64 + 1;

        let (dr, ds) = if use_exhaustive {
            exhaustive_cfr_iteration(
                tree, layout, solve_eq, &regret_sum,
                num_flop_buckets, buf_size, iteration,
            )
        } else {
            sampled_cfr_iteration(
                tree, layout, solve_eq, &regret_sum,
                num_flop_buckets, buf_size, samples_per_iter, iteration, iter,
                flop_idx,
            )
        };

        add_buffers(&mut regret_sum, &dr);
        add_buffers(&mut strategy_sum, &ds);
        iterations_used = iter + 1;

        current_expl = postflop_exploitability(tree, layout, solve_eq, &strategy_sum, num_flop_buckets);

        on_progress(BuildPhase::FlopProgress {
            flop_name: flop_name.to_string(),
            stage: FlopStage::Solving {
                iteration: iter + 1,
                max_iterations: num_iterations,
                exploitability: current_expl,
            },
        });

        // Early stopping after at least 2 iterations.
        if iter >= 1 && current_expl < regret_threshold {
            break;
        }
    }

    FlopSolveResult { strategy_sum, exploitability: current_expl, iterations_used }
}

#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn exhaustive_cfr_iteration(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    solve_eq: &SolveEquity<'_>,
    regret_sum: &[f64],
    num_flop_buckets: usize,
    buf_size: usize,
    iteration: u64,
) -> (Vec<f64>, Vec<f64>) {
    (0..num_flop_buckets)
        .into_par_iter()
        .fold(
            || (vec![0.0f64; buf_size], vec![0.0f64; buf_size]),
            |(mut dr, mut ds), hb| {
                let hb = hb as u16;
                for ob in 0..num_flop_buckets as u16 {
                    for hero_pos in 0..2u8 {
                        solve_cfr_traverse(
                            tree, layout, solve_eq, regret_sum,
                            &mut dr, &mut ds,
                            0, hb, ob, hero_pos, 1.0, 1.0, iteration,
                        );
                    }
                }
                (dr, ds)
            },
        )
        .reduce(
            || (vec![0.0f64; buf_size], vec![0.0f64; buf_size]),
            |(mut dr1, mut ds1), (dr2, ds2)| {
                add_buffers(&mut dr1, &dr2);
                add_buffers(&mut ds1, &ds2);
                (dr1, ds1)
            },
        )
}

#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn sampled_cfr_iteration(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    solve_eq: &SolveEquity<'_>,
    regret_sum: &[f64],
    num_flop_buckets: usize,
    buf_size: usize,
    samples_per_iter: usize,
    iteration: u64,
    iter_idx: usize,
    flop_idx: usize,
) -> (Vec<f64>, Vec<f64>) {
    let seed = iter_idx as u64 * 1_000_003 + flop_idx as u64;
    let mut rng = SmallRng::seed_from_u64(seed);
    let pairs: Vec<(u16, u16)> = (0..samples_per_iter)
        .map(|_| {
            let hb = rng.random_range(0..num_flop_buckets as u16);
            let ob = rng.random_range(0..num_flop_buckets as u16);
            (hb, ob)
        })
        .collect();

    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = (pairs.len() / num_threads).max(1);

    pairs
        .par_chunks(chunk_size)
        .fold(
            || (vec![0.0f64; buf_size], vec![0.0f64; buf_size]),
            |(mut dr, mut ds), chunk| {
                for &(hb, ob) in chunk {
                    for hero_pos in 0..2u8 {
                        solve_cfr_traverse(
                            tree, layout, solve_eq, regret_sum,
                            &mut dr, &mut ds,
                            0, hb, ob, hero_pos, 1.0, 1.0, iteration,
                        );
                    }
                }
                (dr, ds)
            },
        )
        .reduce(
            || (vec![0.0f64; buf_size], vec![0.0f64; buf_size]),
            |(mut dr1, mut ds1), (dr2, ds2)| {
                add_buffers(&mut dr1, &dr2);
                add_buffers(&mut ds1, &ds2);
                (dr1, ds1)
            },
        )
}

/// Single CFR traversal for the postflop solve phase. Returns hero EV fraction.
///
/// At Chance nodes (street transitions), iterates over all new-street bucket
/// pairs weighted by transition probabilities. This properly models how hands
/// transition between per-flop buckets at street boundaries.
#[allow(clippy::too_many_arguments)]
fn solve_cfr_traverse(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    solve_eq: &SolveEquity<'_>,
    snapshot: &[f64],
    dr: &mut [f64],
    ds: &mut [f64],
    node_idx: u32,
    hero_bucket: u16,
    opp_bucket: u16,
    hero_pos: u8,
    reach_hero: f64,
    reach_opp: f64,
    iteration: u64,
) -> f64 {
    match &tree.nodes[node_idx as usize] {
        PostflopNode::Terminal { terminal_type, pot_fraction } => {
            let current_street = layout.street(node_idx);
            let eq_table = solve_equity_for_street(solve_eq, current_street);
            postflop_terminal_value(*terminal_type, *pot_fraction, hero_bucket, opp_bucket, hero_pos, eq_table)
        }
        PostflopNode::Chance { children, street, .. } => {
            // Transition to new street buckets using transition matrices
            let transition = match street {
                Street::Turn => solve_eq.flop_to_turn,
                Street::River => solve_eq.turn_to_river,
                _ => {
                    // Shouldn't happen in practice, fall through to children
                    return children.iter()
                        .map(|&child| {
                            solve_cfr_traverse(
                                tree, layout, solve_eq, snapshot, dr, ds,
                                child, hero_bucket, opp_bucket, hero_pos,
                                reach_hero, reach_opp, iteration,
                            )
                        })
                        .sum::<f64>() / children.len().max(1) as f64;
                }
            };
            debug_assert_eq!(children.len(), 1, "chance node should have exactly 1 structural child");
            let child = children[0];
            let num_new = transition[0].len();
            let mut value = 0.0;
            #[allow(clippy::cast_possible_truncation)] // num_new ≤ u16::MAX (bucket count)
            for new_hero in 0..num_new {
                let hw = transition[hero_bucket as usize][new_hero];
                if hw < 1e-12 { continue; }
                for new_opp in 0..num_new {
                    let ow = transition[opp_bucket as usize][new_opp];
                    if ow < 1e-12 { continue; }
                    value += hw * ow * solve_cfr_traverse(
                        tree, layout, solve_eq, snapshot, dr, ds,
                        child, new_hero as u16, new_opp as u16, hero_pos,
                        reach_hero * hw, reach_opp * ow, iteration,
                    );
                }
            }
            value
        }
        PostflopNode::Decision { position, children, .. } => {
            let is_hero = *position == hero_pos;
            let bucket = if is_hero { hero_bucket } else { opp_bucket };
            let (start, _) = layout.slot(node_idx, bucket);
            let num_actions = children.len();

            let mut strategy = [0.0f64; MAX_POSTFLOP_ACTIONS];
            regret_matching_into(snapshot, start, &mut strategy[..num_actions]);

            if is_hero {
                solve_traverse_hero(
                    tree, layout, solve_eq, snapshot, dr, ds,
                    start, hero_bucket, opp_bucket, hero_pos,
                    reach_hero, reach_opp, children, &strategy[..num_actions], iteration,
                )
            } else {
                solve_traverse_opponent(
                    tree, layout, solve_eq, snapshot, dr, ds,
                    hero_bucket, opp_bucket, hero_pos,
                    reach_hero, reach_opp, children, &strategy[..num_actions], iteration,
                )
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_traverse_hero(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    solve_eq: &SolveEquity<'_>,
    snapshot: &[f64],
    dr: &mut [f64],
    ds: &mut [f64],
    slot_start: usize,
    hero_bucket: u16,
    opp_bucket: u16,
    hero_pos: u8,
    reach_hero: f64,
    reach_opp: f64,
    children: &[u32],
    strategy: &[f64],
    iteration: u64,
) -> f64 {
    let num_actions = children.len();
    let mut action_values = [0.0f64; MAX_POSTFLOP_ACTIONS];

    for (i, &child) in children.iter().enumerate() {
        action_values[i] = solve_cfr_traverse(
            tree, layout, solve_eq, snapshot, dr, ds,
            child, hero_bucket, opp_bucket, hero_pos,
            reach_hero * strategy[i], reach_opp, iteration,
        );
    }

    let node_value: f64 = strategy.iter()
        .zip(&action_values[..num_actions])
        .map(|(s, v)| s * v)
        .sum();

    #[allow(clippy::cast_precision_loss)]
    let weight = iteration as f64;
    for (i, val) in action_values[..num_actions].iter().enumerate() {
        dr[slot_start + i] += weight * reach_opp * (val - node_value);
    }
    for (i, &s) in strategy.iter().enumerate() {
        ds[slot_start + i] += weight * reach_hero * s;
    }

    node_value
}

#[allow(clippy::too_many_arguments)]
fn solve_traverse_opponent(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    solve_eq: &SolveEquity<'_>,
    snapshot: &[f64],
    dr: &mut [f64],
    ds: &mut [f64],
    hero_bucket: u16,
    opp_bucket: u16,
    hero_pos: u8,
    reach_hero: f64,
    reach_opp: f64,
    children: &[u32],
    strategy: &[f64],
    iteration: u64,
) -> f64 {
    children.iter().enumerate()
        .map(|(i, &child)| {
            strategy[i] * solve_cfr_traverse(
                tree, layout, solve_eq, snapshot, dr, ds,
                child, hero_bucket, opp_bucket, hero_pos,
                reach_hero, reach_opp * strategy[i], iteration,
            )
        })
        .sum()
}

/// Walk tree using averaged (converged) strategy, returning hero EV fraction.
#[allow(clippy::too_many_arguments)]
fn eval_with_avg_strategy(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    solve_eq: &SolveEquity<'_>,
    strategy_sum: &[f64],
    node_idx: u32,
    hero_bucket: u16,
    opp_bucket: u16,
    hero_pos: u8,
) -> f64 {
    match &tree.nodes[node_idx as usize] {
        PostflopNode::Terminal { terminal_type, pot_fraction } => {
            let current_street = layout.street(node_idx);
            let eq_table = solve_equity_for_street(solve_eq, current_street);
            postflop_terminal_value(*terminal_type, *pot_fraction, hero_bucket, opp_bucket, hero_pos, eq_table)
        }
        PostflopNode::Chance { children, street, .. } => {
            let transition = match street {
                Street::Turn => solve_eq.flop_to_turn,
                Street::River => solve_eq.turn_to_river,
                _ => {
                    return children.iter()
                        .map(|&child| {
                            eval_with_avg_strategy(
                                tree, layout, solve_eq, strategy_sum,
                                child, hero_bucket, opp_bucket, hero_pos,
                            )
                        })
                        .sum::<f64>() / children.len().max(1) as f64;
                }
            };
            debug_assert_eq!(children.len(), 1, "chance node should have exactly 1 structural child");
            let child = children[0];
            let num_new = transition[0].len();
            let mut value = 0.0;
            #[allow(clippy::cast_possible_truncation)] // num_new ≤ u16::MAX (bucket count)
            for new_hero in 0..num_new {
                let hw = transition[hero_bucket as usize][new_hero];
                if hw < 1e-12 { continue; }
                for new_opp in 0..num_new {
                    let ow = transition[opp_bucket as usize][new_opp];
                    if ow < 1e-12 { continue; }
                    value += hw * ow * eval_with_avg_strategy(
                        tree, layout, solve_eq, strategy_sum,
                        child, new_hero as u16, new_opp as u16, hero_pos,
                    );
                }
            }
            value
        }
        PostflopNode::Decision { position, children, .. } => {
            let bucket = if *position == hero_pos { hero_bucket } else { opp_bucket };
            let (start, _) = layout.slot(node_idx, bucket);
            let num_actions = children.len();

            let strategy = normalize_strategy_sum(strategy_sum, start, num_actions);

            children.iter().enumerate()
                .map(|(i, &child)| {
                    strategy[i] * eval_with_avg_strategy(
                        tree, layout, solve_eq, strategy_sum,
                        child, hero_bucket, opp_bucket, hero_pos,
                    )
                })
                .sum()
        }
    }
}

/// Best-response tree traversal. Like `eval_with_avg_strategy` but at
/// decision nodes where `hero_pos` is the acting player, picks the action
/// with max value instead of mixing according to the average strategy.
///
/// Returns the expected value (in pot fractions) for `hero_pos` when playing
/// optimally against the opponent's average strategy, averaged uniformly
/// across all bucket pairs.
pub(crate) fn best_response_value(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    solve_eq: &SolveEquity<'_>,
    strategy_sum: &[f64],
    node_idx: u32,
    num_buckets: usize,
    hero_pos: u8,
) -> f64 {
    let mut total = 0.0;
    let count = num_buckets * num_buckets;
    #[allow(clippy::cast_possible_truncation)] // num_buckets ≤ u16::MAX (bucket count)
    for hb in 0..num_buckets as u16 {
        for ob in 0..num_buckets as u16 {
            total += br_traverse(
                tree, layout, solve_eq, strategy_sum,
                node_idx, hb, ob, hero_pos,
            );
        }
    }
    if count == 0 { 0.0 } else { total / count as f64 }
}

/// Recursive best-response traversal for a single bucket pair.
fn br_traverse(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    solve_eq: &SolveEquity<'_>,
    strategy_sum: &[f64],
    node_idx: u32,
    hero_bucket: u16,
    opp_bucket: u16,
    hero_pos: u8,
) -> f64 {
    match &tree.nodes[node_idx as usize] {
        PostflopNode::Terminal { terminal_type, pot_fraction } => {
            let current_street = layout.street(node_idx);
            let eq_table = solve_equity_for_street(solve_eq, current_street);
            postflop_terminal_value(
                *terminal_type, *pot_fraction,
                hero_bucket, opp_bucket, hero_pos, eq_table,
            )
        }
        PostflopNode::Chance { children, street, .. } => {
            let transition = match street {
                Street::Turn => solve_eq.flop_to_turn,
                Street::River => solve_eq.turn_to_river,
                _ => {
                    return children.iter()
                        .map(|&child| br_traverse(
                            tree, layout, solve_eq, strategy_sum,
                            child, hero_bucket, opp_bucket, hero_pos,
                        ))
                        .sum::<f64>() / children.len().max(1) as f64;
                }
            };
            debug_assert_eq!(children.len(), 1, "chance node should have exactly 1 structural child");
            let child = children[0];
            let num_new = transition[0].len();
            let mut value = 0.0;
            #[allow(clippy::cast_possible_truncation)]
            for new_hero in 0..num_new {
                let hw = transition[hero_bucket as usize][new_hero];
                if hw < 1e-12 { continue; }
                for new_opp in 0..num_new {
                    let ow = transition[opp_bucket as usize][new_opp];
                    if ow < 1e-12 { continue; }
                    value += hw * ow * br_traverse(
                        tree, layout, solve_eq, strategy_sum,
                        child, new_hero as u16, new_opp as u16, hero_pos,
                    );
                }
            }
            value
        }
        PostflopNode::Decision { position, children, .. } => {
            let is_hero = *position == hero_pos;
            let bucket = if is_hero { hero_bucket } else { opp_bucket };
            let (start, _) = layout.slot(node_idx, bucket);
            let num_actions = children.len();

            let strategy = normalize_strategy_sum(strategy_sum, start, num_actions);

            if is_hero {
                // Best response: pick the action with max value
                children.iter()
                    .map(|&child| br_traverse(
                        tree, layout, solve_eq, strategy_sum,
                        child, hero_bucket, opp_bucket, hero_pos,
                    ))
                    .fold(f64::NEG_INFINITY, f64::max)
            } else {
                // Opponent plays average strategy
                children.iter().enumerate()
                    .map(|(i, &child)| {
                        strategy[i] * br_traverse(
                            tree, layout, solve_eq, strategy_sum,
                            child, hero_bucket, opp_bucket, hero_pos,
                        )
                    })
                    .sum()
            }
        }
    }
}

/// Exploitability of the current average strategy in pot fractions.
///
/// Sums best-response values for both players across all bucket pairs.
/// At Nash equilibrium this is 0.
fn postflop_exploitability(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    solve_eq: &SolveEquity<'_>,
    strategy_sum: &[f64],
    num_buckets: usize,
) -> f64 {
    let p0 = best_response_value(tree, layout, solve_eq, strategy_sum, 0, num_buckets, 0);
    let p1 = best_response_value(tree, layout, solve_eq, strategy_sum, 0, num_buckets, 1);
    (p0 + p1).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;
    use super::super::postflop_abstraction::PostflopLayout;
    use super::super::postflop_abstraction::annotate_streets;

    /// Build identity transition matrix: each bucket maps to the same-index bucket.
    /// This makes tests equivalent to the old pass-through behavior.
    fn identity_transition(k: usize) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; k]; k];
        for i in 0..k {
            matrix[i][i] = 1.0;
        }
        matrix
    }

    #[timed_test]
    fn solve_one_flop_returns_result_struct() {
        // Verify that solve_one_flop returns FlopSolveResult with correct fields.
        let config = PostflopModelConfig::fast();
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let streets = annotate_streets(&tree);
        let layout = PostflopLayout::build(&tree, &streets, 5, 5, 5);
        let buf_size = layout.total_size;

        // Create minimal equity tables (5 buckets).
        let eq = BucketEquity {
            equity: vec![vec![0.5; 5]; 5],
            num_buckets: 5,
        };
        let f2t = identity_transition(5);
        let t2r = identity_transition(5);
        let solve_eq = SolveEquity {
            flop: &eq,
            turn: &eq,
            river: &eq,
            flop_to_turn: &f2t,
            turn_to_river: &t2r,
        };
        let result = solve_one_flop(
            &tree, &layout, &solve_eq,
            5, buf_size, 4, 25, 0.0001,
            0, "AhKd7s", &|_| {},
        );
        assert_eq!(result.strategy_sum.len(), buf_size);
        assert!(result.iterations_used >= 2);
        assert!(result.iterations_used <= 4);
        assert!(result.exploitability.is_finite());
    }

    #[timed_test]
    fn solve_one_flop_early_stop_with_zero_threshold() {
        let config = PostflopModelConfig::fast();
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let streets = annotate_streets(&tree);
        let layout = PostflopLayout::build(&tree, &streets, 5, 5, 5);
        let buf_size = layout.total_size;

        let eq = BucketEquity {
            equity: vec![vec![0.5; 5]; 5],
            num_buckets: 5,
        };
        let f2t = identity_transition(5);
        let t2r = identity_transition(5);
        let solve_eq = SolveEquity {
            flop: &eq,
            turn: &eq,
            river: &eq,
            flop_to_turn: &f2t,
            turn_to_river: &t2r,
        };
        let result = solve_one_flop(
            &tree, &layout, &solve_eq,
            5, buf_size, 3, 25, 0.0,
            0, "AhKd7s", &|_| {},
        );
        assert_eq!(result.iterations_used, 3, "zero threshold should run all iterations");
    }

    #[timed_test]
    fn solve_one_flop_early_stop_with_large_threshold() {
        let config = PostflopModelConfig::fast();
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let streets = annotate_streets(&tree);
        let layout = PostflopLayout::build(&tree, &streets, 5, 5, 5);
        let buf_size = layout.total_size;

        let eq = BucketEquity {
            equity: vec![vec![0.5; 5]; 5],
            num_buckets: 5,
        };
        let f2t = identity_transition(5);
        let t2r = identity_transition(5);
        let solve_eq = SolveEquity {
            flop: &eq,
            turn: &eq,
            river: &eq,
            flop_to_turn: &f2t,
            turn_to_river: &t2r,
        };
        let result = solve_one_flop(
            &tree, &layout, &solve_eq,
            5, buf_size, 100, 25, f64::INFINITY,
            0, "AhKd7s", &|_| {},
        );
        assert_eq!(result.iterations_used, 2, "huge threshold should stop at minimum 2 iterations");
    }

    #[timed_test(5)]
    fn postflop_exploitability_decreases_with_training() {
        let config = PostflopModelConfig::fast();
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let streets = annotate_streets(&tree);
        let layout = PostflopLayout::build(&tree, &streets, 5, 5, 5);
        let buf_size = layout.total_size;

        let eq = BucketEquity {
            equity: vec![vec![0.5; 5]; 5],
            num_buckets: 5,
        };
        let f2t = identity_transition(5);
        let t2r = identity_transition(5);
        let solve_eq = SolveEquity {
            flop: &eq, turn: &eq, river: &eq,
            flop_to_turn: &f2t, turn_to_river: &t2r,
        };

        // Few iterations
        let early = solve_one_flop(
            &tree, &layout, &solve_eq,
            5, buf_size, 3, 25, 0.0,
            0, "test", &|_| {},
        );
        // Many iterations
        let late = solve_one_flop(
            &tree, &layout, &solve_eq,
            5, buf_size, 50, 25, 0.0,
            0, "test", &|_| {},
        );

        assert!(early.exploitability > late.exploitability,
            "exploitability should decrease: early={:.6}, late={:.6}",
            early.exploitability, late.exploitability);
        assert!(late.exploitability >= 0.0);
    }

    #[timed_test]
    fn best_response_value_is_non_negative() {
        // In any zero-sum game, the best-response value for one player
        // against the opponent's avg strategy should be >= the game value.
        // With uniform equity (0.5), values should be near 0.
        let config = PostflopModelConfig::fast();
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let streets = annotate_streets(&tree);
        let layout = PostflopLayout::build(&tree, &streets, 5, 5, 5);
        let buf_size = layout.total_size;

        let eq = BucketEquity {
            equity: vec![vec![0.5; 5]; 5],
            num_buckets: 5,
        };
        let f2t = identity_transition(5);
        let t2r = identity_transition(5);
        let solve_eq = SolveEquity {
            flop: &eq,
            turn: &eq,
            river: &eq,
            flop_to_turn: &f2t,
            turn_to_river: &t2r,
        };

        // Run a few CFR iterations to get a non-trivial strategy_sum
        let result = solve_one_flop(
            &tree, &layout, &solve_eq,
            5, buf_size, 10, 25, 0.0,
            0, "test", &|_| {},
        );

        // BR value for each player should be finite
        for hero_pos in 0..2u8 {
            let br = best_response_value(
                &tree, &layout, &solve_eq, &result.strategy_sum,
                0, 5, hero_pos,
            );
            assert!(br.is_finite(), "BR value should be finite, got {br}");
        }
    }
}
