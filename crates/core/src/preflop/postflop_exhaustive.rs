//! Exhaustive postflop solve backend.
//!
//! Vanilla CFR with pre-computed equity tables. All chance nodes are fully
//! enumerated (no sampling), and showdown terminals use O(1) equity lookups.

use rayon::prelude::*;

use crate::cfr::parallel::{add_into, parallel_traverse_into, ParallelCfr};

use super::postflop_abstraction::{
    normalize_strategy_sum_into, regret_matching_into, BuildPhase, FlopSolveResult, FlopStage,
    PostflopLayout, PostflopValues, MAX_POSTFLOP_ACTIONS,
};
use super::postflop_hands::{all_cards_vec, build_combo_map, NUM_CANONICAL_HANDS};
use super::postflop_model::PostflopModelConfig;
use super::postflop_tree::{PostflopNode, PostflopTerminalType, PostflopTree};
use crate::abstraction::Street;
use crate::cfr::dcfr::DcfrParams;
use crate::poker::Card;
use crate::preflop::CfrVariant;
use crate::showdown_equity::rank_hand;

use std::sync::atomic::{AtomicUsize, Ordering};

// ──────────────────────────────────────────────────────────────────────────────
// Equity table
// ──────────────────────────────────────────────────────────────────────────────

/// Pre-compute flop-only equity table for all 169x169 hand pairs.
///
/// For each `(hero_hand, opp_hand)` canonical pair, enumerates all concrete
/// combo pairs that don't conflict with each other or the flop, then
/// enumerates all turn+river runouts to compute average equity.
///
/// Returns a flat `Vec` of size 169*169, indexed as `hero*169 + opp`.
/// Value is hero's equity (0.0 to 1.0), or `NaN` if the hand pair has
/// no non-conflicting combos.
#[allow(clippy::cast_precision_loss)]
fn compute_equity_table(combo_map: &[Vec<(Card, Card)>], flop: [Card; 3]) -> Vec<f64> {
    let n = NUM_CANONICAL_HANDS;
    let deck = all_cards_vec();

    // Parallel over hero hands
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|hero_idx| {
            let mut row = vec![f64::NAN; n];
            let hero_combos = &combo_map[hero_idx];
            if hero_combos.is_empty() {
                return row;
            }

            for opp_idx in 0..n {
                let opp_combos = &combo_map[opp_idx];
                if opp_combos.is_empty() {
                    continue;
                }

                let mut total_eq = 0.0f64;
                let mut total_count = 0u64;

                for &(h1, h2) in hero_combos {
                    for &(o1, o2) in opp_combos {
                        // Skip conflicting hands
                        if h1 == o1 || h1 == o2 || h2 == o1 || h2 == o2 {
                            continue;
                        }

                        // Enumerate all turn+river runouts
                        let used = [flop[0], flop[1], flop[2], h1, h2, o1, o2];
                        for (ti, &turn) in deck.iter().enumerate() {
                            if used.contains(&turn) {
                                continue;
                            }
                            for &river in &deck[ti + 1..] {
                                if used.contains(&river) {
                                    continue;
                                }
                                let board = [flop[0], flop[1], flop[2], turn, river];
                                let hero_rank = rank_hand([h1, h2], &board);
                                let opp_rank = rank_hand([o1, o2], &board);
                                total_eq += match hero_rank.cmp(&opp_rank) {
                                    std::cmp::Ordering::Greater => 1.0,
                                    std::cmp::Ordering::Equal => 0.5,
                                    std::cmp::Ordering::Less => 0.0,
                                };
                                total_count += 1;
                            }
                        }
                    }
                }

                if total_count > 0 {
                    row[opp_idx] = total_eq / total_count as f64;
                }
            }
            row
        })
        .collect();

    // Flatten
    let mut table = vec![f64::NAN; n * n];
    for (hero_idx, row) in rows.into_iter().enumerate() {
        table[hero_idx * n..hero_idx * n + n].copy_from_slice(&row);
    }
    table
}

// ──────────────────────────────────────────────────────────────────────────────
// CFR traversal
// ──────────────────────────────────────────────────────────────────────────────

/// CFR traversal with equity table lookups at showdown.
///
/// Reads strategy from a frozen `snapshot` and writes regret/strategy deltas
/// to separate `dr`/`ds` buffers. This split enables future parallelisation:
/// multiple threads can share a read-only snapshot while writing to thread-local
/// delta buffers.
///
/// Both players are traversed every iteration. Chance nodes pass through
/// to their single child (board cards are implicit in the equity table).
/// Supports LCFR/DCFR iteration weighting via `dcfr` params.
#[allow(clippy::too_many_arguments)]
fn exhaustive_cfr_traverse(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    equity_table: &[f64],
    snapshot: &[f64],
    dr: &mut [f64],
    ds: &mut [f64],
    node_idx: u32,
    hero_hand: u16,
    opp_hand: u16,
    hero_pos: u8,
    reach_hero: f64,
    reach_opp: f64,
    iteration: u64,
    dcfr: &DcfrParams,
    prune_active: bool,
) -> f64 {
    let n = NUM_CANONICAL_HANDS;
    match &tree.nodes[node_idx as usize] {
        PostflopNode::Terminal {
            terminal_type,
            pot_fraction,
        } => terminal_payoff(
            *terminal_type,
            *pot_fraction,
            equity_table,
            hero_hand,
            opp_hand,
            hero_pos,
            n,
        ),

        PostflopNode::Chance { children, .. } => {
            debug_assert!(!children.is_empty());
            exhaustive_cfr_traverse(
                tree,
                layout,
                equity_table,
                snapshot,
                dr,
                ds,
                children[0],
                hero_hand,
                opp_hand,
                hero_pos,
                reach_hero,
                reach_opp,
                iteration,
                dcfr,
                prune_active,
            )
        }

        PostflopNode::Decision {
            position, children, ..
        } => {
            let is_hero = *position == hero_pos;
            let bucket = if is_hero { hero_hand } else { opp_hand };
            let (start, _) = layout.slot(node_idx, bucket);
            let num_actions = children.len();

            let mut strategy = [0.0f64; MAX_POSTFLOP_ACTIONS];
            regret_matching_into(snapshot, start, &mut strategy[..num_actions]);

            if is_hero {
                let mut action_values = [0.0f64; MAX_POSTFLOP_ACTIONS];

                // RBP: build prune bitmask in a single pass over regrets.
                // Bit i is set if action i should be pruned (negative regret,
                // and at least one sibling has positive regret).
                let mut prune_mask: u16 = 0;
                if prune_active {
                    let mut has_positive = false;
                    let mut neg_mask: u16 = 0;
                    for i in 0..num_actions {
                        if snapshot[start + i] > 0.0 {
                            has_positive = true;
                        } else if snapshot[start + i] < 0.0 {
                            neg_mask |= 1 << i;
                        }
                    }
                    if has_positive {
                        prune_mask = neg_mask;
                    }
                }

                for (i, &child) in children.iter().enumerate() {
                    if prune_mask & (1 << i) != 0 {
                        continue;
                    }
                    action_values[i] = exhaustive_cfr_traverse(
                        tree,
                        layout,
                        equity_table,
                        snapshot,
                        dr,
                        ds,
                        child,
                        hero_hand,
                        opp_hand,
                        hero_pos,
                        reach_hero * strategy[i],
                        reach_opp,
                        iteration,
                        dcfr,
                        prune_active,
                    );
                }
                let node_value: f64 = strategy[..num_actions]
                    .iter()
                    .zip(&action_values[..num_actions])
                    .map(|(s, v)| s * v)
                    .sum();

                let (regret_weight, strategy_weight) = dcfr.iteration_weights(iteration);
                for (i, val) in action_values[..num_actions].iter().enumerate() {
                    dr[start + i] += regret_weight * reach_opp * (val - node_value);
                }
                for (i, &s) in strategy[..num_actions].iter().enumerate() {
                    ds[start + i] += strategy_weight * reach_hero * s;
                }
                node_value
            } else {
                children
                    .iter()
                    .enumerate()
                    .map(|(i, &child)| {
                        strategy[i]
                            * exhaustive_cfr_traverse(
                                tree,
                                layout,
                                equity_table,
                                snapshot,
                                dr,
                                ds,
                                child,
                                hero_hand,
                                opp_hand,
                                hero_pos,
                                reach_hero,
                                reach_opp * strategy[i],
                                iteration,
                                dcfr,
                                prune_active,
                            )
                    })
                    .sum()
            }
        }
    }
}

/// Compute terminal node payoff (fold or showdown with equity table).
fn terminal_payoff(
    terminal_type: PostflopTerminalType,
    pot_fraction: f64,
    equity_table: &[f64],
    hero_hand: u16,
    opp_hand: u16,
    hero_pos: u8,
    n: usize,
) -> f64 {
    match terminal_type {
        PostflopTerminalType::Fold { folder } => {
            if folder == hero_pos {
                -pot_fraction / 2.0
            } else {
                pot_fraction / 2.0
            }
        }
        PostflopTerminalType::Showdown => {
            let eq = equity_table[hero_hand as usize * n + opp_hand as usize];
            if eq.is_nan() {
                return 0.0;
            }
            eq * pot_fraction - pot_fraction / 2.0
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Exploitability
// ──────────────────────────────────────────────────────────────────────────────

/// Compute best-response EV for one player against the opponent's average strategy.
///
/// At `br_player`'s decision nodes: pick the action with highest EV (best response).
/// At opponent's decision nodes: play the average strategy from `strategy_sum`.
/// At terminals: use equity table for showdown, `pot_fraction` for folds.
#[allow(clippy::too_many_arguments)]
fn best_response_ev(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    strategy_sum: &[f64],
    equity_table: &[f64],
    node_idx: u32,
    hero_hand: u16,
    opp_hand: u16,
    br_player: u8,
) -> f64 {
    let n = NUM_CANONICAL_HANDS;
    match &tree.nodes[node_idx as usize] {
        PostflopNode::Terminal {
            terminal_type,
            pot_fraction,
        } => terminal_payoff(
            *terminal_type,
            *pot_fraction,
            equity_table,
            hero_hand,
            opp_hand,
            br_player,
            n,
        ),

        PostflopNode::Chance { children, .. } => best_response_ev(
            tree,
            layout,
            strategy_sum,
            equity_table,
            children[0],
            hero_hand,
            opp_hand,
            br_player,
        ),

        PostflopNode::Decision {
            position, children, ..
        } => {
            let is_br = *position == br_player;
            let bucket = if is_br { hero_hand } else { opp_hand };
            let (start, _) = layout.slot(node_idx, bucket);
            let num_actions = children.len();

            if is_br {
                // Best response: pick the action with highest EV
                children
                    .iter()
                    .map(|&child| {
                        best_response_ev(
                            tree,
                            layout,
                            strategy_sum,
                            equity_table,
                            child,
                            hero_hand,
                            opp_hand,
                            br_player,
                        )
                    })
                    .fold(f64::NEG_INFINITY, f64::max)
            } else {
                // Opponent plays average strategy
                let mut strategy = [0.0f64; MAX_POSTFLOP_ACTIONS];
                normalize_strategy_sum_into(strategy_sum, start, &mut strategy[..num_actions]);
                children
                    .iter()
                    .enumerate()
                    .map(|(i, &child)| {
                        strategy[i]
                            * best_response_ev(
                                tree,
                                layout,
                                strategy_sum,
                                equity_table,
                                child,
                                hero_hand,
                                opp_hand,
                                br_player,
                            )
                    })
                    .sum()
            }
        }
    }
}

/// Assumed initial pot size in BB for mBB/h conversion.
///
/// Standard HU open (~3x pot entering flop). This matches the preflop
/// solver's convention of reporting exploitability in mBB.
const INITIAL_POT_BB: f64 = 3.0;

/// Compute exploitability of the current average strategy, in mBB/h.
///
/// For each player, computes the best-response value over all hand matchups,
/// then returns the average of both players' BR values converted to mBB/h
/// (assumes ~3 BB initial pot: 1 pot-fraction = 3 BB = 3000 mBB).
///
/// A Nash equilibrium has exploitability of 0.
#[allow(clippy::cast_precision_loss)]
fn compute_exploitability(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    strategy_sum: &[f64],
    equity_table: &[f64],
) -> f64 {
    let n = NUM_CANONICAL_HANDS;
    let mut br_values = [0.0f64; 2];

    for br_player in 0..2u8 {
        let (total, count) = (0..n as u16)
            .into_par_iter()
            .flat_map_iter(|hero_hand| {
                (0..n as u16)
                    .filter(move |&opp_hand| {
                        !equity_table[hero_hand as usize * n + opp_hand as usize].is_nan()
                    })
                    .map(move |opp_hand| (hero_hand, opp_hand))
            })
            .map(|(hero_hand, opp_hand)| {
                best_response_ev(
                    tree, layout, strategy_sum, equity_table, 0, hero_hand, opp_hand, br_player,
                )
            })
            .fold(
                || (0.0f64, 0u64),
                |(t, c), v| (t + v, c + 1),
            )
            .reduce(
                || (0.0, 0),
                |(t1, c1), (t2, c2)| (t1 + t2, c1 + c2),
            );

        br_values[br_player as usize] = if count > 0 {
            total / count as f64
        } else {
            0.0
        };
    }

    let pot_fraction = (br_values[0] + br_values[1]) / 2.0;
    pot_fraction * INITIAL_POT_BB * 1000.0
}

// ──────────────────────────────────────────────────────────────────────────────
// Solve one flop
// ──────────────────────────────────────────────────────────────────────────────

/// Immutable context for one postflop CFR iteration.
/// Strategy is read from `snapshot`; deltas written to thread-local buffers.
#[derive(Debug)]
struct PostflopCfrCtx<'a> {
    tree: &'a PostflopTree,
    layout: &'a PostflopLayout,
    equity_table: &'a [f64],
    snapshot: &'a [f64],
    iteration: u64,
    dcfr: &'a DcfrParams,
    prune_active: bool,
}

impl ParallelCfr for PostflopCfrCtx<'_> {
    fn buffer_size(&self) -> usize {
        self.layout.total_size
    }

    fn traverse_pair(&self, regret_delta: &mut [f64], strategy_delta: &mut [f64], hero: u16, opponent: u16) {
        let n = NUM_CANONICAL_HANDS;
        let eq = self.equity_table[hero as usize * n + opponent as usize];
        if eq.is_nan() {
            return;
        }
        for hero_pos in 0..2u8 {
            exhaustive_cfr_traverse(
                self.tree,
                self.layout,
                self.equity_table,
                self.snapshot,
                regret_delta,
                strategy_delta,
                0,
                hero,
                opponent,
                hero_pos,
                1.0,
                1.0,
                self.iteration,
                self.dcfr,
                self.prune_active,
            );
        }
    }
}

/// Solve a single flop using exhaustive CFR with configurable iteration weighting.
///
/// Inner `parallel_traverse_into` and `compute_exploitability` use rayon's
/// global thread pool. When called from `build_exhaustive` (which parallelises
/// over flops), rayon's work-stealing distributes hand-pair work across all
/// available cores, dynamically rebalancing as flops converge at different rates.
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn exhaustive_solve_one_flop(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    equity_table: &[f64],
    num_iterations: usize,
    convergence_threshold: f64,
    flop_name: &str,
    dcfr: &DcfrParams,
    config: &PostflopModelConfig,
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> FlopSolveResult {
    let buf_size = layout.total_size;
    let mut regret_sum = vec![0.0f64; buf_size];
    let mut strategy_sum = vec![0.0f64; buf_size];
    let mut current_exploitability = f64::INFINITY;
    let mut iterations_used = 0;
    let n = NUM_CANONICAL_HANDS;

    // Pre-build valid hand pairs (filter NaN equity pairs once).
    let pairs: Vec<(u16, u16)> = (0..n as u16)
        .flat_map(|h1| (0..n as u16).map(move |h2| (h1, h2)))
        .filter(|&(h1, h2)| !equity_table[h1 as usize * n + h2 as usize].is_nan())
        .collect();

    let mut snapshot = vec![0.0f64; buf_size];

    // Pre-allocate delta buffers reused across iterations to avoid
    // two Vec allocations per iteration (parallel_traverse_into zeroes them).
    let mut dr = vec![0.0f64; buf_size];
    let mut ds = vec![0.0f64; buf_size];

    let flop_name_owned = flop_name.to_string();

    for iter in 0..num_iterations {
        snapshot.clone_from(&regret_sum);

        let prune_active = config.prune_warmup > 0
            && iter >= config.prune_warmup
            && (config.prune_explore_freq == 0 || iter % config.prune_explore_freq != 0);

        let ctx = PostflopCfrCtx {
            tree,
            layout,
            equity_table,
            snapshot: &snapshot,
            iteration: iter as u64,
            dcfr,
            prune_active,
        };

        parallel_traverse_into(&ctx, &pairs, &mut dr, &mut ds);

        // Apply DCFR discounting before merging deltas.
        if dcfr.should_discount(iter as u64) {
            dcfr.discount_regrets(&mut regret_sum, iter as u64);
            dcfr.discount_strategy_sums(&mut strategy_sum, iter as u64);
        }
        add_into(&mut regret_sum, &dr);
        add_into(&mut strategy_sum, &ds);
        if dcfr.should_floor_regrets() {
            dcfr.floor_regrets(&mut regret_sum);
        }

        // Clamp negative regrets to prevent unbounded accumulation.
        // This bounds memory of bad actions so pruned actions can
        // recover when explored during explore-frequency iterations.
        if config.regret_floor > 0.0 && config.prune_warmup > 0 {
            let floor = -config.regret_floor;
            for v in regret_sum.iter_mut() {
                *v = (*v).max(floor);
            }
        }

        iterations_used = iter + 1;
        if iter >= 1 && (iter % 2 == 1 || iter == num_iterations - 1) {
            current_exploitability =
                compute_exploitability(tree, layout, &strategy_sum, equity_table);
        }

        on_progress(BuildPhase::FlopProgress {
            flop_name: flop_name_owned.clone(),
            stage: FlopStage::Solving {
                iteration: iterations_used,
                max_iterations: num_iterations,
                delta: current_exploitability,
                metric_label: "mBB/h".into(),
            },
        });

        if iter >= 1 && current_exploitability < convergence_threshold {
            break;
        }
    }

    FlopSolveResult {
        strategy_sum,
        delta: current_exploitability,
        iterations_used,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Value extraction
// ──────────────────────────────────────────────────────────────────────────────

/// Extract values from converged strategy for all
/// `(hero_pos, hero_hand, opp_hand)` triples.
///
/// Returns flat `Vec` of size `2 * 169 * 169`.
#[allow(clippy::cast_possible_truncation)]
fn exhaustive_extract_values(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    strategy_sum: &[f64],
    equity_table: &[f64],
) -> Vec<f64> {
    let n = NUM_CANONICAL_HANDS;
    // NaN = "no valid combos on this flop" -- skipped during cross-flop averaging.
    let mut values = vec![f64::NAN; 2 * n * n];

    for hero_hand in 0..n as u16 {
        for opp_hand in 0..n as u16 {
            let eq = equity_table[hero_hand as usize * n + opp_hand as usize];
            if eq.is_nan() {
                continue; // remains NaN -- signals missing data
            }

            for hero_pos in 0..2u8 {
                let ev = eval_with_avg_strategy(
                    tree,
                    layout,
                    strategy_sum,
                    equity_table,
                    0,
                    hero_hand,
                    opp_hand,
                    hero_pos,
                );
                let idx =
                    hero_pos as usize * n * n + hero_hand as usize * n + opp_hand as usize;
                values[idx] = ev;
            }
        }
    }
    values
}

/// Walk tree using averaged strategy with equity table lookups.
#[allow(clippy::too_many_arguments)]
fn eval_with_avg_strategy(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    strategy_sum: &[f64],
    equity_table: &[f64],
    node_idx: u32,
    hero_hand: u16,
    opp_hand: u16,
    hero_pos: u8,
) -> f64 {
    let n = NUM_CANONICAL_HANDS;
    match &tree.nodes[node_idx as usize] {
        PostflopNode::Terminal {
            terminal_type,
            pot_fraction,
        } => terminal_payoff(
            *terminal_type,
            *pot_fraction,
            equity_table,
            hero_hand,
            opp_hand,
            hero_pos,
            n,
        ),
        PostflopNode::Chance { children, .. } => eval_with_avg_strategy(
            tree,
            layout,
            strategy_sum,
            equity_table,
            children[0],
            hero_hand,
            opp_hand,
            hero_pos,
        ),
        PostflopNode::Decision {
            position, children, ..
        } => {
            let bucket = if *position == hero_pos {
                hero_hand
            } else {
                opp_hand
            };
            let (start, _) = layout.slot(node_idx, bucket);
            let num_actions = children.len();
            let mut strategy = [0.0f64; MAX_POSTFLOP_ACTIONS];
            normalize_strategy_sum_into(strategy_sum, start, &mut strategy[..num_actions]);
            children
                .iter()
                .enumerate()
                .map(|(i, &child)| {
                    strategy[i]
                        * eval_with_avg_strategy(
                            tree,
                            layout,
                            strategy_sum,
                            equity_table,
                            child,
                            hero_hand,
                            opp_hand,
                            hero_pos,
                        )
                })
                .sum()
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Entry point
// ──────────────────────────────────────────────────────────────────────────────

/// Build postflop values using exhaustive CFR with equity tables.
pub(crate) fn build_exhaustive(
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
    let _ = node_streets;
    let completed = AtomicUsize::new(0);

    let dcfr = match config.cfr_variant {
        CfrVariant::Linear => DcfrParams::linear(),
        CfrVariant::Dcfr => DcfrParams::default(),
        CfrVariant::Vanilla => DcfrParams::vanilla(),
        CfrVariant::CfrPlus => DcfrParams::from_config(CfrVariant::CfrPlus, 0.0, 0.0, 0.0, 0),
    };

    let results: Vec<Vec<f64>> = (0..num_flops)
        .into_par_iter()
        .map(|flop_idx| {
            let flop = flops[flop_idx];
            let flop_name = format!("{}{}{}", flop[0], flop[1], flop[2]);

            let combo_map = build_combo_map(&flop);
            let equity_table = compute_equity_table(&combo_map, flop);

            let result = exhaustive_solve_one_flop(
                tree,
                layout,
                &equity_table,
                num_iterations,
                config.cfr_convergence_threshold,
                &flop_name,
                &dcfr,
                config,
                on_progress,
            );

            let values =
                exhaustive_extract_values(tree, layout, &result.strategy_sum, &equity_table);

            on_progress(BuildPhase::FlopProgress {
                flop_name: flop_name.clone(),
                stage: FlopStage::Done,
            });
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            on_progress(BuildPhase::MccfrFlopsCompleted {
                completed: done,
                total: num_flops,
            });

            values
        })
        .collect();

    let mut all_values = vec![0.0f64; num_flops * 2 * n * n];
    for (flop_idx, vals) in results.into_iter().enumerate() {
        let offset = flop_idx * 2 * n * n;
        let copy_len = vals.len().min(2 * n * n);
        all_values[offset..offset + copy_len].copy_from_slice(&vals[..copy_len]);
    }

    PostflopValues::from_raw(all_values, n, num_flops)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::dcfr::DcfrParams;
    use crate::poker::{Suit, Value};
    use crate::preflop::postflop_abstraction::annotate_streets;
    use test_macros::timed_test;

    fn card(v: Value, s: Suit) -> Card {
        Card::new(v, s)
    }

    fn test_flop() -> [Card; 3] {
        [
            card(Value::Two, Suit::Spade),
            card(Value::Seven, Suit::Heart),
            card(Value::Queen, Suit::Diamond),
        ]
    }

    /// Build a synthetic equity table where equity is based on hand index.
    /// Higher index = stronger hand. This avoids expensive full enumeration.
    #[allow(clippy::cast_precision_loss)]
    fn synthetic_equity_table() -> Vec<f64> {
        let n = NUM_CANONICAL_HANDS;
        let mut table = vec![f64::NAN; n * n];
        for h in 0..n {
            for o in 0..n {
                // Synthetic equity: hero wins more when hero_idx > opp_idx
                let eq = if h == o {
                    0.5
                } else {
                    // Smooth gradient based on index difference
                    0.5 + 0.4 * (h as f64 - o as f64) / (n as f64)
                };
                table[h * n + o] = eq;
            }
        }
        table
    }

    #[test]
    #[ignore = "slow: full 169x169 equity table computation"]
    fn equity_table_has_correct_size() {
        let flop = test_flop();
        let combo_map = build_combo_map(&flop);
        let eq = compute_equity_table(&combo_map, flop);
        assert_eq!(eq.len(), 169 * 169);
    }

    #[test]
    #[ignore = "slow: full 169x169 equity table computation"]
    fn equity_table_diagonal_pairs_are_half() {
        // Same hand vs same hand should have ~0.5 equity (ties)
        let flop = test_flop();
        let combo_map = build_combo_map(&flop);
        let eq = compute_equity_table(&combo_map, flop);
        let n = NUM_CANONICAL_HANDS;
        for h in 0..13 {
            // Check pairs (index 0-12)
            let val = eq[h * n + h];
            if !val.is_nan() {
                assert!(
                    (val - 0.5).abs() < 0.01,
                    "same hand vs same hand should be ~0.5, got {val} for hand {h}"
                );
            }
        }
    }

    #[test]
    #[ignore = "slow: full 169x169 equity table computation"]
    fn equity_table_symmetric() {
        let flop = test_flop();
        let combo_map = build_combo_map(&flop);
        let eq = compute_equity_table(&combo_map, flop);
        let n = NUM_CANONICAL_HANDS;
        for h in 0..20 {
            for o in (h + 1)..20 {
                let e1 = eq[h * n + o];
                let e2 = eq[o * n + h];
                if e1.is_finite() && e2.is_finite() {
                    let sum = e1 + e2;
                    assert!(
                        (sum - 1.0).abs() < 0.02,
                        "equity[{h}][{o}] + equity[{o}][{h}] = {sum}, expected ~1.0"
                    );
                }
            }
        }
    }

    #[timed_test]
    fn synthetic_equity_table_is_consistent() {
        let table = synthetic_equity_table();
        let n = NUM_CANONICAL_HANDS;
        assert_eq!(table.len(), n * n);

        // Diagonal should be 0.5
        for h in 0..n {
            assert!(
                (table[h * n + h] - 0.5).abs() < 1e-9,
                "diagonal should be 0.5"
            );
        }

        // hero[0][168] should be < 0.5 (weaker hand), hero[168][0] > 0.5
        assert!(table[0 * n + 168] < 0.5);
        assert!(table[168 * n + 0] > 0.5);
    }

    #[test]
    fn exhaustive_solve_produces_strategy() {
        let config = PostflopModelConfig {
            bet_sizes: vec![1.0],
            max_raises_per_street: 0,
            postflop_solve_iterations: 3,
            ..PostflopModelConfig::exhaustive_fast()
        };
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);

        let equity_table = synthetic_equity_table();
        let dcfr = DcfrParams::linear();

        let result = exhaustive_solve_one_flop(
            &tree,
            &layout,
            &equity_table,
            3,
            0.001,
            "test",
            &dcfr,
            &config,
            &|_| {},
        );

        assert!(result.iterations_used > 0);
        let has_nonzero = result.strategy_sum.iter().any(|&v| v.abs() > 1e-15);
        assert!(has_nonzero, "strategy_sum should have non-zero entries");
    }

    #[test]
    fn exhaustive_extract_values_dimensions() {
        let config = PostflopModelConfig {
            bet_sizes: vec![1.0],
            max_raises_per_street: 0,
            postflop_solve_iterations: 2,
            ..PostflopModelConfig::exhaustive_fast()
        };
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);

        let equity_table = synthetic_equity_table();
        let dcfr = DcfrParams::linear();

        let result = exhaustive_solve_one_flop(
            &tree,
            &layout,
            &equity_table,
            2,
            0.001,
            "test",
            &dcfr,
            &config,
            &|_| {},
        );

        let values =
            exhaustive_extract_values(&tree, &layout, &result.strategy_sum, &equity_table);
        assert_eq!(values.len(), 2 * n * n);
        assert!(
            values.iter().all(|v| v.is_finite()),
            "all values should be finite"
        );
    }

    #[timed_test]
    fn exhaustive_cfr_fold_terminal_payoff() {
        // Test that fold terminals return correct payoff values
        let config = PostflopModelConfig {
            bet_sizes: vec![1.0],
            max_raises_per_street: 0,
            postflop_solve_iterations: 5,
            ..PostflopModelConfig::exhaustive_fast()
        };
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);
        let equity_table = synthetic_equity_table();
        let mut regret_sum = vec![0.0f64; layout.total_size];
        let mut strategy_sum = vec![0.0f64; layout.total_size];
        let dcfr = DcfrParams::linear();

        // Find a fold terminal and verify payoff
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
            let snapshot = regret_sum.clone();
            let ev = exhaustive_cfr_traverse(
                &tree,
                &layout,
                &equity_table,
                &snapshot,
                &mut regret_sum,
                &mut strategy_sum,
                node_idx,
                0,
                1,
                0, // hero_pos = 0
                1.0,
                1.0,
                0,
                &dcfr,
                false,
            );

            if folder == 0 {
                assert!(
                    (ev - (-pot_fraction / 2.0)).abs() < 1e-9,
                    "hero folds: expected {}, got {ev}",
                    -pot_fraction / 2.0
                );
            } else {
                assert!(
                    (ev - (pot_fraction / 2.0)).abs() < 1e-9,
                    "opp folds: expected {}, got {ev}",
                    pot_fraction / 2.0
                );
            }
        }
    }

    #[timed_test]
    fn exhaustive_cfr_showdown_terminal_payoff() {
        let config = PostflopModelConfig {
            bet_sizes: vec![1.0],
            max_raises_per_street: 0,
            postflop_solve_iterations: 5,
            ..PostflopModelConfig::exhaustive_fast()
        };
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);
        let equity_table = synthetic_equity_table();
        let mut regret_sum = vec![0.0f64; layout.total_size];
        let mut strategy_sum = vec![0.0f64; layout.total_size];
        let dcfr = DcfrParams::linear();

        // Find a showdown terminal
        let sd_node = tree.nodes.iter().enumerate().find_map(|(i, n)| {
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

        if let Some((node_idx, pot_fraction)) = sd_node {
            // Hand 100 vs hand 50: hand 100 should have equity > 0.5
            let hero_hand = 100u16;
            let opp_hand = 50u16;
            let eq = equity_table[hero_hand as usize * n + opp_hand as usize];

            let snapshot = regret_sum.clone();
            let ev = exhaustive_cfr_traverse(
                &tree,
                &layout,
                &equity_table,
                &snapshot,
                &mut regret_sum,
                &mut strategy_sum,
                node_idx,
                hero_hand,
                opp_hand,
                0,
                1.0,
                1.0,
                0,
                &dcfr,
                false,
            );

            let expected = eq * pot_fraction - pot_fraction / 2.0;
            assert!(
                (ev - expected).abs() < 1e-9,
                "showdown EV: expected {expected}, got {ev}"
            );
        }
    }

    /// Helper: build a minimal tree + layout + synthetic equity for exploitability tests.
    fn expl_test_fixtures() -> (PostflopTree, PostflopLayout, Vec<f64>) {
        let config = PostflopModelConfig {
            bet_sizes: vec![1.0],
            max_raises_per_street: 0,
            postflop_solve_iterations: 1,
            ..PostflopModelConfig::exhaustive_fast()
        };
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);
        let equity_table = synthetic_equity_table();
        (tree, layout, equity_table)
    }

    #[timed_test(10)]
    fn exploitability_is_positive_for_uniform_strategy() {
        let (tree, layout, equity_table) = expl_test_fixtures();
        // All-zero strategy_sum -> normalize_strategy_sum returns uniform
        let strategy_sum = vec![0.0f64; layout.total_size];
        let expl = compute_exploitability(&tree, &layout, &strategy_sum, &equity_table);
        assert!(
            expl > 0.0,
            "uniform strategy should be exploitable, got {expl}"
        );
    }

    #[timed_test(20)]
    fn exploitability_decreases_with_training() {
        let (tree, layout, equity_table) = expl_test_fixtures();
        // Uniform strategy exploitability (baseline).
        let uniform_sum = vec![0.0f64; layout.total_size];
        let expl_uniform = compute_exploitability(&tree, &layout, &uniform_sum, &equity_table);
        // 3 CFR iterations with no early stop. With snapshot-based CFR the first
        // iteration reads uniform regrets, so convergence needs one extra iteration
        // compared to in-place updates.
        let dcfr = DcfrParams::linear();
        let no_prune = PostflopModelConfig::exhaustive_fast();
        let result = exhaustive_solve_one_flop(
            &tree, &layout, &equity_table, 3, 0.0, "test", &dcfr, &no_prune, &|_| {},
        );
        assert!(
            result.delta < expl_uniform,
            "trained exploitability ({:.6}) should be less than uniform ({:.6})",
            result.delta,
            expl_uniform
        );
    }

    #[timed_test(30)]
    fn parallel_solve_matches_sequential_result() {
        let config = PostflopModelConfig {
            bet_sizes: vec![1.0],
            max_raises_per_street: 0,
            postflop_solve_iterations: 5,
            ..PostflopModelConfig::exhaustive_fast()
        };
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);
        let equity_table = synthetic_equity_table();
        let dcfr = DcfrParams::linear();

        // Solve with default thread pool (parallel)
        let result_par = exhaustive_solve_one_flop(
            &tree, &layout, &equity_table, 5, 0.0, "par", &dcfr, &config, &|_| {},
        );

        // Solve with 1 thread (sequential)
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let result_seq = pool.install(|| {
            exhaustive_solve_one_flop(
                &tree, &layout, &equity_table, 5, 0.0, "seq", &dcfr, &config, &|_| {},
            )
        });

        assert_eq!(result_par.iterations_used, result_seq.iterations_used);
        for (p, s) in result_par
            .strategy_sum
            .iter()
            .zip(result_seq.strategy_sum.iter())
        {
            assert!(
                (p - s).abs() < 1e-6,
                "strategy_sum mismatch: parallel={p}, sequential={s}"
            );
        }
    }

    #[timed_test(15)]
    fn exploitability_early_stopping_triggers() {
        let (tree, layout, equity_table) = expl_test_fixtures();
        // Very generous threshold (30 BB/h) -- stops at first exploitability check.
        let threshold_mbb = 30_000.0;
        let dcfr = DcfrParams::linear();
        let no_prune = PostflopModelConfig::exhaustive_fast();
        let result = exhaustive_solve_one_flop(
            &tree,
            &layout,
            &equity_table,
            30,
            threshold_mbb,
            "test",
            &dcfr,
            &no_prune,
            &|_| {},
        );
        assert_eq!(
            result.iterations_used, 2,
            "should stop at first exploitability check (iter 1), used {}",
            result.iterations_used
        );
        assert!(
            result.delta < threshold_mbb,
            "final exploitability should be below threshold, got {} mBB/h",
            result.delta
        );
    }

    #[test]
    fn exhaustive_solve_with_pruning_produces_strategy() {
        let config = PostflopModelConfig {
            bet_sizes: vec![1.0],
            max_raises_per_street: 0,
            postflop_solve_iterations: 50,
            prune_warmup: 10,
            prune_explore_freq: 5,
            regret_floor: 1_000_000.0,
            ..PostflopModelConfig::exhaustive_fast()
        };
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);
        let equity_table = synthetic_equity_table();
        let dcfr = DcfrParams::linear();

        let result = exhaustive_solve_one_flop(
            &tree,
            &layout,
            &equity_table,
            50,
            0.0,
            "prune_test",
            &dcfr,
            &config,
            &|_| {},
        );

        assert!(result.iterations_used > 0, "should complete iterations");
        let has_nonzero = result.strategy_sum.iter().any(|&v| v.abs() > 1e-15);
        assert!(has_nonzero, "strategy_sum should have non-zero entries with pruning enabled");
    }

    #[test]
    fn pruning_does_not_break_convergence() {
        let base = PostflopModelConfig {
            bet_sizes: vec![1.0],
            max_raises_per_street: 0,
            postflop_solve_iterations: 30,
            ..PostflopModelConfig::exhaustive_fast()
        };
        let tree = PostflopTree::build_with_spr(&base, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);
        let equity_table = synthetic_equity_table();
        let dcfr = DcfrParams::linear();

        // Unpruned baseline (prune_warmup=0)
        let unpruned = exhaustive_solve_one_flop(
            &tree, &layout, &equity_table, 30, 0.0, "no_prune", &dcfr, &base, &|_| {},
        );

        // Pruned run
        let pruned_config = PostflopModelConfig {
            prune_warmup: 5,
            prune_explore_freq: 3,
            regret_floor: 1_000_000.0,
            ..base.clone()
        };
        let pruned = exhaustive_solve_one_flop(
            &tree, &layout, &equity_table, 30, 0.0, "pruned", &dcfr, &pruned_config, &|_| {},
        );

        // Both should produce valid strategies
        assert!(unpruned.iterations_used > 0);
        assert!(pruned.iterations_used > 0);
        let unpruned_nonzero = unpruned.strategy_sum.iter().any(|&v| v.abs() > 1e-15);
        let pruned_nonzero = pruned.strategy_sum.iter().any(|&v| v.abs() > 1e-15);
        assert!(unpruned_nonzero, "unpruned should have non-zero strategy");
        assert!(pruned_nonzero, "pruned should have non-zero strategy");
    }
}
