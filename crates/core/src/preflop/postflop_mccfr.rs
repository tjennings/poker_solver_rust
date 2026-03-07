//! MCCFR postflop solve backend.
//!
//! Instead of abstract bucket-to-bucket transitions at street changes, this
//! backend uses **concrete hands** (actual `Card` pairs). For each sample it
//! picks two non-conflicting hands from canonical hand combos and a random
//! turn/river. At showdown terminals it uses `rank_hand()` to evaluate actual
//! hands against the full 5-card board. Strategies and regrets are indexed by
//! canonical hand index (0..168).

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

/// Solve one flop and extract EV values using MCCFR.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]
fn solve_and_extract_mccfr_flop(
    config: &PostflopModelConfig,
    tree: &PostflopTree,
    layout: &PostflopLayout,
    flop: [Card; 3],
    num_iterations: usize,
    n: usize,
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> Vec<f64> {
    let flop_name = format!("{}{}{}", flop[0], flop[1], flop[2]);
    let combo_map = build_combo_map(&flop);

    let total_non_conflicting: usize = combo_map.iter().map(Vec::len).sum();
    let total_space = total_non_conflicting.saturating_mul(45 * 44);
    let samples_per_iter = ((total_space as f64 * config.mccfr_sample_pct) as usize
        / num_iterations.max(1)).max(1);

    let result = mccfr_solve_one_flop(
        tree, layout, &combo_map, flop, num_iterations,
        samples_per_iter, config.cfr_convergence_threshold, &flop_name, on_progress,
    );
    let values = mccfr_extract_values(
        tree, layout, &result.strategy_sum, &combo_map, flop,
        config.value_extraction_samples as usize, n, &flop_name, on_progress,
    );
    on_progress(BuildPhase::FlopProgress { flop_name, stage: FlopStage::Done });
    values
}

/// Build postflop abstraction using MCCFR with sampled concrete hands.
///
/// Pipeline per flop (parallelised via rayon):
/// 1. Build combo map -- each canonical hand (0..168) maps to its concrete combos.
/// 2. Run MCCFR training loop (`mccfr_solve_one_flop`).
/// 3. Extract EV table from converged strategy (`mccfr_extract_values`).
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
    let _ = node_streets;
    let completed = AtomicUsize::new(0);

    let results: Vec<Vec<f64>> = (0..num_flops)
        .into_par_iter()
        .map(|flop_idx| {
            let values = solve_and_extract_mccfr_flop(
                config, tree, layout, flops[flop_idx], num_iterations, n, on_progress,
            );
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            on_progress(BuildPhase::MccfrFlopsCompleted { completed: done, total: num_flops });
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
// Core solve loop
// ──────────────────────────────────────────────────────────────────────────────

/// Sample deals and run MCCFR traversal for one iteration, accumulating
/// regret and strategy deltas into `dr`/`ds`.
#[allow(clippy::too_many_arguments)]
fn mccfr_sample_and_traverse(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    regret_sum: &[f64],
    dr: &mut [f64],
    ds: &mut [f64],
    combo_map: &[Vec<(Card, Card)>],
    flop: [Card; 3],
    samples_per_iter: usize,
    iteration: u64,
    flop_name: &str,
) {
    let seed = iteration * 1_000_003 + flop_name.len() as u64;
    let mut rng = SmallRng::seed_from_u64(seed);

    for _ in 0..samples_per_iter {
        let Some((hb, ob, hero_hand, opp_hand, turn, river)) =
            sample_deal(combo_map, flop, &mut rng)
        else {
            continue;
        };
        let board = [flop[0], flop[1], flop[2], turn, river];
        let ctx = MccfrTraverseCtx { tree, layout, snapshot: regret_sum, board: &board, iteration };
        for hero_pos in 0..2u8 {
            let args = MccfrTraverseArgs {
                node_idx: 0, hero_bucket: hb, opp_bucket: ob, hero_pos,
                hero_hand, opp_hand, reach_hero: 1.0, reach_opp: 1.0,
            };
            ctx.traverse(dr, ds, &args);
        }
    }
}

/// Report solving progress for one iteration.
fn report_solve_progress(
    on_progress: &(impl Fn(BuildPhase) + Sync),
    flop_name: &str,
    iterations_used: usize,
    max_iterations: usize,
    delta: f64,
) {
    on_progress(BuildPhase::FlopProgress {
        flop_name: flop_name.to_string(),
        stage: FlopStage::Solving {
            iteration: iterations_used,
            max_iterations,
            delta,
            metric_label: "\u{03b4}".to_string(),
            total_action_slots: 0,
            pruned_action_slots: 0,
            max_positive_regret: 0.0,
            min_negative_regret: 0.0,
        },
    });
}

/// Training loop for a single flop using MCCFR with concrete hands.
///
/// Convergence is measured via `weighted_avg_strategy_delta`: the regret-weighted
/// average of max per-action strategy probability change between consecutive
/// iterations.
#[allow(clippy::too_many_arguments)]
fn mccfr_solve_one_flop(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    combo_map: &[Vec<(Card, Card)>],
    flop: [Card; 3],
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

        let mut dr = vec![0.0f64; buf_size];
        let mut ds = vec![0.0f64; buf_size];
        mccfr_sample_and_traverse(
            tree, layout, &regret_sum, &mut dr, &mut ds,
            combo_map, flop, samples_per_iter, iteration, flop_name,
        );

        add_buffers(&mut regret_sum, &dr);
        add_buffers(&mut strategy_sum, &ds);
        iterations_used = iter + 1;

        if iter >= 1 {
            current_delta = weighted_avg_strategy_delta(&prev_regrets, &regret_sum, layout, tree);
        }
        report_solve_progress(on_progress, flop_name, iterations_used, num_iterations, current_delta);
        if iter >= 1 && current_delta < convergence_threshold {
            break;
        }
    }

    FlopSolveResult { strategy_sum, delta: current_delta, iterations_used }
}

// ──────────────────────────────────────────────────────────────────────────────
// Deal sampling
// ──────────────────────────────────────────────────────────────────────────────

/// `(hero_hand_idx, opp_hand_idx, hero_hand, opp_hand, turn_card, river_card)`.
type SampledDeal = (u16, u16, [Card; 2], [Card; 2], Card, Card);

/// Sample a concrete deal: two non-conflicting hands from random canonical hand
/// indices plus random turn + river cards.
///
/// Returns `(hero_hand_idx, opp_hand_idx, hero_hand, opp_hand, turn_card, river_card)`.
#[allow(clippy::cast_possible_truncation)]
fn sample_deal(
    combo_map: &[Vec<(Card, Card)>],
    flop: [Card; 3],
    rng: &mut SmallRng,
) -> Option<SampledDeal> {
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

/// Immutable context shared across all recursive `mccfr_traverse` calls.
struct MccfrTraverseCtx<'a> {
    tree: &'a PostflopTree,
    layout: &'a PostflopLayout,
    snapshot: &'a [f64],
    board: &'a [Card; 5],
    iteration: u64,
}

/// Per-call varying state for `mccfr_traverse`.
struct MccfrTraverseArgs {
    node_idx: u32,
    hero_bucket: u16,
    opp_bucket: u16,
    hero_pos: u8,
    hero_hand: [Card; 2],
    opp_hand: [Card; 2],
    reach_hero: f64,
    reach_opp: f64,
}

/// Compute terminal payoff for fold or showdown with concrete hand ranking.
#[inline]
fn mccfr_terminal_payoff(
    terminal_type: PostflopTerminalType,
    pot_fraction: f64,
    hero_hand: [Card; 2],
    opp_hand: [Card; 2],
    board: &[Card; 5],
    hero_pos: u8,
) -> f64 {
    match terminal_type {
        PostflopTerminalType::Fold { folder } => {
            if folder == hero_pos { -pot_fraction / 2.0 } else { pot_fraction / 2.0 }
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
    }
}

impl MccfrTraverseCtx<'_> {
    /// Hero decision: traverse all actions, accumulate regret and strategy deltas.
    #[inline]
    fn traverse_hero(
        &self,
        dr: &mut [f64],
        ds: &mut [f64],
        children: &[u32],
        strategy: &[f64],
        start: usize,
        args: &MccfrTraverseArgs,
    ) -> f64 {
        let num_actions = children.len();
        let mut action_values = [0.0f64; MAX_POSTFLOP_ACTIONS];
        for (i, &child) in children.iter().enumerate() {
            let mut child_args = MccfrTraverseArgs { node_idx: child, ..*args };
            child_args.reach_hero = args.reach_hero * strategy[i];
            action_values[i] = self.traverse(dr, ds, &child_args);
        }
        let node_value: f64 = strategy[..num_actions]
            .iter()
            .zip(&action_values[..num_actions])
            .map(|(s, v)| s * v)
            .sum();

        #[allow(clippy::cast_precision_loss)]
        let weight = self.iteration as f64;
        for (i, val) in action_values[..num_actions].iter().enumerate() {
            dr[start + i] += weight * args.reach_opp * (val - node_value);
        }
        for (i, &s) in strategy[..num_actions].iter().enumerate() {
            ds[start + i] += weight * args.reach_hero * s;
        }
        node_value
    }

    /// Opponent decision: strategy-weighted traversal.
    #[inline]
    fn traverse_opponent(
        &self,
        dr: &mut [f64],
        ds: &mut [f64],
        children: &[u32],
        strategy: &[f64],
        args: &MccfrTraverseArgs,
    ) -> f64 {
        children
            .iter()
            .enumerate()
            .map(|(i, &child)| {
                let child_args = MccfrTraverseArgs {
                    node_idx: child,
                    reach_opp: args.reach_opp * strategy[i],
                    ..*args
                };
                strategy[i] * self.traverse(dr, ds, &child_args)
            })
            .sum()
    }

    /// Recursive CFR traversal using concrete hands (not bucket transitions).
    ///
    /// At chance nodes the board is already dealt, so we just pass through to
    /// the single structural child. At showdown terminals we use `rank_hand`
    /// for card-based evaluation.
    fn traverse(&self, dr: &mut [f64], ds: &mut [f64], args: &MccfrTraverseArgs) -> f64 {
        match &self.tree.nodes[args.node_idx as usize] {
            PostflopNode::Terminal { terminal_type, pot_fraction } => {
                mccfr_terminal_payoff(
                    *terminal_type, *pot_fraction,
                    args.hero_hand, args.opp_hand, self.board, args.hero_pos,
                )
            }
            PostflopNode::Chance { children, .. } => {
                debug_assert!(!children.is_empty(), "chance node must have at least one child");
                let child_args = MccfrTraverseArgs { node_idx: children[0], ..*args };
                self.traverse(dr, ds, &child_args)
            }
            PostflopNode::Decision { position, children, .. } => {
                let is_hero = *position == args.hero_pos;
                let bucket = if is_hero { args.hero_bucket } else { args.opp_bucket };
                let (start, _) = self.layout.slot(args.node_idx, bucket);
                let mut strategy = [0.0f64; MAX_POSTFLOP_ACTIONS];
                regret_matching_into(self.snapshot, start, &mut strategy[..children.len()]);
                if is_hero {
                    self.traverse_hero(dr, ds, children, &strategy, start, args)
                } else {
                    self.traverse_opponent(dr, ds, children, &strategy, args)
                }
            }
        }
    }
}


// ──────────────────────────────────────────────────────────────────────────────
// Value extraction
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluate one hand pair by sampling runouts and accumulating EV.
#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
fn evaluate_hand_pair(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    strategy_sum: &[f64],
    hero_combos: &[(Card, Card)],
    opp_combos: &[(Card, Card)],
    flop: [Card; 3],
    hb: u16,
    ob: u16,
    n: usize,
    samples_per_pair: usize,
    values: &mut [f64],
    counts: &mut [u32],
    rng: &mut SmallRng,
) {
    for _ in 0..samples_per_pair {
        let Some((hero_hand, opp_hand, turn, river)) =
            sample_runout_for_pair(hero_combos, opp_combos, flop, rng)
        else {
            break;
        };
        let board = [flop[0], flop[1], flop[2], turn, river];
        for hero_pos in 0..2u8 {
            let ctx = MccfrEvalCtx {
                tree, layout, strategy_sum, board: &board,
                hero_bucket: hb, opp_bucket: ob, hero_pos, hero_hand, opp_hand,
            };
            let ev = ctx.eval(0);
            let idx = hero_pos as usize * n * n + hb as usize * n + ob as usize;
            values[idx] += ev;
            counts[idx] += 1;
        }
    }
}

/// Normalize accumulated EV values; unsampled cells become NaN.
fn normalize_ev_values(values: &mut [f64], counts: &[u32]) {
    for (i, val) in values.iter_mut().enumerate() {
        if counts[i] > 0 {
            *val /= f64::from(counts[i]);
        } else {
            *val = f64::NAN;
        }
    }
}

/// Extract EV values by sampling a fixed number of runouts per hand pair.
///
/// `samples_per_pair` runouts are drawn for each of the `n * n` canonical hand
/// pairs, guaranteeing 100% coverage. Pairs where no non-conflicting concrete
/// combos exist on this flop are marked NaN (impossible on this board).
#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
fn mccfr_extract_values(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    strategy_sum: &[f64],
    combo_map: &[Vec<(Card, Card)>],
    flop: [Card; 3],
    samples_per_pair: usize,
    num_flop_buckets: usize,
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
            if pairs_done.is_multiple_of(1000) {
                on_progress(BuildPhase::FlopProgress {
                    flop_name: flop_name.to_string(),
                    stage: FlopStage::EstimatingEv { sample: pairs_done, total_samples: total_pairs },
                });
            }
            let opp_combos = &combo_map[ob as usize];
            if opp_combos.is_empty() { continue; }
            evaluate_hand_pair(
                tree, layout, strategy_sum, hero_combos, opp_combos,
                flop, hb, ob, n, samples_per_pair, &mut values, &mut counts, &mut rng,
            );
        }
    }

    on_progress(BuildPhase::FlopProgress {
        flop_name: flop_name.to_string(),
        stage: FlopStage::EstimatingEv { sample: total_pairs, total_samples: total_pairs },
    });
    normalize_ev_values(&mut values, &counts);
    values
}

/// Sample a single runout (concrete hero/opp combo + turn + river) for a
/// specific canonical hand pair on a given flop.
fn sample_runout_for_pair(
    hero_combos: &[(Card, Card)],
    opp_combos: &[(Card, Card)],
    flop: [Card; 3],
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

/// Immutable context for evaluating converged strategy (read-only traversal).
struct MccfrEvalCtx<'a> {
    tree: &'a PostflopTree,
    layout: &'a PostflopLayout,
    strategy_sum: &'a [f64],
    board: &'a [Card; 5],
    hero_bucket: u16,
    opp_bucket: u16,
    hero_pos: u8,
    hero_hand: [Card; 2],
    opp_hand: [Card; 2],
}

impl MccfrEvalCtx<'_> {
    /// Walk tree using averaged (converged) strategy with concrete hands.
    ///
    /// Same structure as `MccfrTraverseCtx::traverse` but read-only: uses
    /// `normalize_strategy_sum` and does no regret updates.
    fn eval(&self, node_idx: u32) -> f64 {
        match &self.tree.nodes[node_idx as usize] {
            PostflopNode::Terminal { terminal_type, pot_fraction } => {
                mccfr_terminal_payoff(
                    *terminal_type, *pot_fraction,
                    self.hero_hand, self.opp_hand, self.board, self.hero_pos,
                )
            }
            PostflopNode::Chance { children, .. } => {
                debug_assert!(!children.is_empty(), "chance node must have at least one child");
                self.eval(children[0])
            }
            PostflopNode::Decision { position, children, .. } => {
                let bucket = if *position == self.hero_pos { self.hero_bucket } else { self.opp_bucket };
                let (start, _) = self.layout.slot(node_idx, bucket);
                let strategy = normalize_strategy_sum(self.strategy_sum, start, children.len());
                children.iter().enumerate()
                    .map(|(i, &child)| strategy[i] * self.eval(child))
                    .sum()
            }
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
    use crate::preflop::config::CfrVariant;
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
            cfr_variant: CfrVariant::Linear,
            prune_warmup: 0,
            prune_explore_freq: 20,
            prune_regret_threshold: 0.0,
            regret_floor: 1_000_000.0,
            exploitability_freq: 2,
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
                #[allow(clippy::cast_possible_truncation)]
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

            let ctx = MccfrTraverseCtx {
                tree: &tree, layout: &layout, snapshot: &snapshot,
                board: &board, iteration: 1,
            };
            let args = MccfrTraverseArgs {
                node_idx, hero_bucket: 0, opp_bucket: 1, hero_pos: 0,
                hero_hand, opp_hand, reach_hero: 1.0, reach_opp: 1.0,
            };
            let ev = ctx.traverse(&mut dr, &mut ds, &args);

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
                #[allow(clippy::cast_possible_truncation)]
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

            let ctx = MccfrTraverseCtx {
                tree: &tree, layout: &layout, snapshot: &snapshot,
                board: &board, iteration: 1,
            };
            let args = MccfrTraverseArgs {
                node_idx, hero_bucket: 0, opp_bucket: 1, hero_pos: 0,
                hero_hand, opp_hand, reach_hero: 1.0, reach_opp: 1.0,
            };
            let ev = ctx.traverse(&mut dr, &mut ds, &args);

            // AA beats KK → eq=1.0 → ev = 1.0 * pot_frac - pot_frac/2 = pot_frac/2
            let expected = pot_fraction / 2.0;
            assert!(
                (ev - expected).abs() < 1e-9,
                "showdown: AA vs KK, ev={ev}, expected={expected}"
            );
        }
    }

    #[timed_test(3)]
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
            flop,
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
            flop,
            20,
            10,
            0.001,
            "test",
            &|_| {},
        );

        let values = mccfr_extract_values(
            &tree, &layout, &result.strategy_sum, &combo_map, flop,
            3, n_subset, "test", &|_| {},
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
                sample_runout_for_pair(hero_combos, opp_combos, flop, &mut rng)
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

    #[timed_test(10)]
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
        #[allow(clippy::cast_possible_truncation)]
        let n_u16 = n as u16;
        for hb in 0..n_u16 {
            for ob in 0..n_u16 {
                let ev0 = values.get_by_flop(0, 0, hb, ob);
                let ev1 = values.get_by_flop(0, 1, hb, ob);
                total_ev += ev0 + ev1;
                count += 1;
            }
        }
        // Zero-sum check: average should be close to 0
        let avg = total_ev / f64::from(count);
        assert!(
            avg.abs() < 0.5,
            "average EV across all matchups should be roughly zero-sum, got {avg}"
        );
    }
}
