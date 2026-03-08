//! Exhaustive postflop solve backend.
//!
//! Vanilla CFR with pre-computed equity tables. All chance nodes are fully
//! enumerated (no sampling), and showdown terminals use O(1) equity lookups.

use rayon::prelude::*;
use std::cell::RefCell;

use crate::cfr::parallel::{add_into, parallel_traverse_pooled, ParallelCfr};

use super::postflop_abstraction::{
    normalize_strategy_sum_into, regret_matching_into, BuildPhase, FlopStage, PostflopLayout,
    PostflopValues, MAX_POSTFLOP_ACTIONS,
};
use super::postflop_hands::{all_cards_vec, build_combo_map, NUM_CANONICAL_HANDS};
use super::postflop_model::PostflopModelConfig;
use super::postflop_tree::{PostflopNode, PostflopTerminalType, PostflopTree};
use crate::abstraction::Street;
use crate::cfr::dcfr::DcfrParams;
use crate::poker::Card;
use crate::preflop::CfrVariant;
use crate::showdown_equity::{rank_hand, rank_to_ordinal};

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Optional atomic counters for external progress monitoring (e.g. TUI).
///
/// When provided, the solver hot path increments these with `Relaxed` ordering
/// so an observer thread can sample throughput and pruning rates without
/// blocking the solver.
#[derive(Debug, Default)]
pub struct SolverCounters {
    /// Number of `traverse_pair` calls (one per hero/opponent hand pair per iteration).
    pub traversal_count: AtomicU64,
    /// Number of `traverse_pair` calls where pruning was active.
    pub pruned_traversal_count: AtomicU64,
    /// Total action slots visited at hero decision nodes.
    pub total_action_slots: AtomicU64,
    /// Action slots skipped by regret-based pruning.
    pub pruned_action_slots: AtomicU64,
    /// Total expected `traverse_pair` calls for the current SPR.
    /// Each flop adds `pairs.len() * num_iterations` when it starts solving.
    pub total_expected_traversals: AtomicU64,
}

/// Reusable per-thread buffers for exhaustive CFR solving.
///
/// Allocated once per rayon worker thread and reused across flops to avoid
/// repeated allocation of large vectors (e.g. 277 MB per flop at SPR=6).
pub(crate) struct FlopBuffers {
    pub regret_sum: Vec<f64>,
    pub strategy_sum: Vec<f64>,
    pub delta: (Vec<f64>, Vec<f64>),
}

impl FlopBuffers {
    /// Allocate zeroed buffers of the given size.
    pub(crate) fn new(size: usize) -> Self {
        Self {
            regret_sum: vec![0.0; size],
            strategy_sum: vec![0.0; size],
            delta: (vec![0.0; size], vec![0.0; size]),
        }
    }

    /// Zero all buffers for reuse with the next flop.
    fn reset(&mut self) {
        self.regret_sum.fill(0.0);
        self.strategy_sum.fill(0.0);
        // delta buffers are zeroed per-iteration by parallel_traverse_pooled,
        // but zero them here too for clean state between flops.
        self.delta.0.fill(0.0);
        self.delta.1.fill(0.0);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Equity table
// ──────────────────────────────────────────────────────────────────────────────

/// Map a card to a unique bit position (0..51) for bitmask collision checks.
///
/// Encoding: `value * 4 + suit` using `#[repr(u8)]` discriminants.
/// This is the same scheme used in `showdown_equity::card_bit`.
#[inline]
fn card_bit(card: Card) -> u8 {
    card.value as u8 * 4 + card.suit as u8
}

/// Reference implementation: pre-compute flop-only equity table for all 169x169 hand pairs.
///
/// For each `(hero_hand, opp_hand)` canonical pair, enumerates all concrete
/// combo pairs that don't conflict with each other or the flop, then
/// enumerates all turn+river runouts to compute average equity.
///
/// Returns a flat `Vec` of size 169*169, indexed as `hero*169 + opp`.
/// Value is hero's equity (0.0 to 1.0), or `NaN` if the hand pair has
/// no non-conflicting combos.
///
/// Kept behind `#[cfg(test)]` for correctness verification against
/// [`compute_equity_table`].
#[cfg(test)]
#[must_use]
#[allow(clippy::cast_precision_loss)]
fn compute_equity_table_reference(combo_map: &[Vec<(Card, Card)>], flop: [Card; 3]) -> Vec<f64> {
    let n = NUM_CANONICAL_HANDS;
    let deck = all_cards_vec();

    // Pre-compute bit position for each deck index, avoiding repeated
    // value/suit → u8 conversions in the inner loop.
    let mut deck_bits = [0u8; 52];
    for (i, &c) in deck.iter().enumerate() {
        deck_bits[i] = card_bit(c);
    }

    // Flop mask — set once, reused by every thread.
    let flop_mask: u64 = (1u64 << card_bit(flop[0]))
        | (1u64 << card_bit(flop[1]))
        | (1u64 << card_bit(flop[2]));

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

                        // 64-bit bitmask of all 7 used cards (flop + both holes).
                        let used: u64 = flop_mask
                            | (1u64 << card_bit(h1))
                            | (1u64 << card_bit(h2))
                            | (1u64 << card_bit(o1))
                            | (1u64 << card_bit(o2));

                        // Enumerate all turn+river runouts
                        for (ti, &turn) in deck.iter().enumerate() {
                            if used & (1u64 << deck_bits[ti]) != 0 {
                                continue;
                            }
                            for (ri, &river) in deck[ti + 1..].iter().enumerate() {
                                let river_idx = ti + 1 + ri;
                                if used & (1u64 << deck_bits[river_idx]) != 0 {
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

/// Pre-compute flop-only equity table for all 169x169 hand pairs.
///
/// Two-phase implementation: evaluates each concrete combo exactly once per
/// (turn, river) runout via `rank_to_ordinal(rank_hand(...))`, then derives
/// equity via cheap `u32` comparison of precomputed rank ordinals.
///
/// **Phase 1** (per board): compute `rank_to_ordinal(rank_hand(...))` for every
/// concrete combo whose cards do not conflict with the 5-card board.
///
/// **Phase 2** (per board): for every canonical (hero, opp) pair, compare
/// precomputed rank ordinals using `u32` comparison. Accumulate `(equity_sum,
/// count)` per canonical pair across all boards.
///
/// Returns a flat `Vec` of size 169*169, indexed as `hero*169 + opp`.
/// Value is hero's equity (0.0 to 1.0), or `NaN` if the hand pair has
/// no non-conflicting combos.
/// Flat combo index for efficient equity computation.
///
/// Maps canonical hands to a flat array of concrete combos, enabling
/// O(1) bitmask-based conflict checks during board enumeration.
struct ComboIndex {
    cards: Vec<(Card, Card)>,
    masks: Vec<u64>,
    ranges: Vec<std::ops::Range<usize>>,
}

/// Build a flat combo index from the canonical combo map.
///
/// `cards[i]` = the two hole cards for flat combo i,
/// `masks[i]` = bitmask of those two cards (for conflict checks),
/// `ranges[h]` = `Range<usize>` into `cards` for canonical hand h.
fn build_combo_index(combo_map: &[Vec<(Card, Card)>], n: usize) -> ComboIndex {
    let mut cards: Vec<(Card, Card)> = Vec::new();
    let mut masks: Vec<u64> = Vec::new();
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(n);

    debug_assert_eq!(combo_map.len(), n, "combo_map must have exactly n entries");
    for hand_combos in combo_map {
        let start = cards.len();
        for &(c1, c2) in hand_combos {
            cards.push((c1, c2));
            masks.push((1u64 << card_bit(c1)) | (1u64 << card_bit(c2)));
        }
        ranges.push(start..cards.len());
    }
    ComboIndex { cards, masks, ranges }
}

/// Enumerate all (`turn_deck_idx`, `river_deck_idx`) pairs that don't overlap
/// with the flop. Returns deck-index pairs for ordered (turn < river).
fn enumerate_non_flop_boards(deck_bits: &[u8; 52], flop_mask: u64) -> Vec<(usize, usize)> {
    let mut boards = Vec::new();
    for (ti, &ti_bit) in deck_bits.iter().enumerate() {
        if flop_mask & (1u64 << ti_bit) != 0 {
            continue;
        }
        for (ri, &ri_bit) in deck_bits.iter().enumerate().skip(ti + 1) {
            if flop_mask & (1u64 << ri_bit) != 0 {
                continue;
            }
            boards.push((ti, ri));
        }
    }
    boards
}

/// Sentinel value for combos that conflict with the board.
const COMBO_CONFLICT: u32 = u32::MAX;

#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_equity_table(combo_map: &[Vec<(Card, Card)>], flop: [Card; 3]) -> Vec<f64> {
    let n = NUM_CANONICAL_HANDS;
    let deck = all_cards_vec();
    let idx = build_combo_index(combo_map, n);

    let mut deck_bits = [0u8; 52];
    for (i, &c) in deck.iter().enumerate() {
        deck_bits[i] = card_bit(c);
    }
    let flop_mask: u64 = (1u64 << card_bit(flop[0]))
        | (1u64 << card_bit(flop[1]))
        | (1u64 << card_bit(flop[2]));

    let boards = enumerate_non_flop_boards(&deck_bits, flop_mask);
    let total_combos = idx.cards.len();

    let (eq_sum, eq_count) = boards
        .par_iter()
        .fold(
            || (vec![0.0f64; n * n], vec![0u64; n * n]),
            |(mut local_eq, mut local_count), &(ti, ri)| {
                let board = [flop[0], flop[1], flop[2], deck[ti], deck[ri]];
                let board_mask: u64 =
                    flop_mask | (1u64 << deck_bits[ti]) | (1u64 << deck_bits[ri]);

                // Phase 1: evaluate every combo against this board.
                let mut ranks = vec![COMBO_CONFLICT; total_combos];
                for (ci, &(c1, c2)) in idx.cards.iter().enumerate() {
                    if idx.masks[ci] & board_mask != 0 {
                        continue;
                    }
                    ranks[ci] = rank_to_ordinal(rank_hand([c1, c2], &board));
                }

                // Phase 2: for each canonical (hero, opp) pair, compare ranks.
                for (h_idx, h_range) in idx.ranges.iter().enumerate() {
                    for (o_idx, o_range) in idx.ranges.iter().enumerate() {
                        let pair_idx = h_idx * n + o_idx;
                        for hi in h_range.clone() {
                            let h_rank = ranks[hi];
                            if h_rank == COMBO_CONFLICT {
                                continue;
                            }
                            let h_mask = idx.masks[hi];
                            for oi in o_range.clone() {
                                let o_rank = ranks[oi];
                                if o_rank == COMBO_CONFLICT || h_mask & idx.masks[oi] != 0 {
                                    continue;
                                }
                                local_eq[pair_idx] += match h_rank.cmp(&o_rank) {
                                    std::cmp::Ordering::Greater => 1.0,
                                    std::cmp::Ordering::Equal => 0.5,
                                    std::cmp::Ordering::Less => 0.0,
                                };
                                local_count[pair_idx] += 1;
                            }
                        }
                    }
                }
                (local_eq, local_count)
            },
        )
        .reduce(
            || (vec![0.0f64; n * n], vec![0u64; n * n]),
            |(mut a_eq, mut a_count), (b_eq, b_count)| {
                for i in 0..a_eq.len() {
                    a_eq[i] += b_eq[i];
                    a_count[i] += b_count[i];
                }
                (a_eq, a_count)
            },
        );

    let mut table = vec![f64::NAN; n * n];
    for i in 0..n * n {
        if eq_count[i] > 0 {
            table[i] = eq_sum[i] / eq_count[i] as f64;
        }
    }
    table
}

/// Pre-compute card-removal weights for all 169x169 canonical hand pairs.
///
/// For each `(hero_hand, opp_hand)` canonical pair, counts the number of
/// concrete (`combo_h`, `combo_o`) pairs where all four hole cards are distinct.
/// This weight determines how much each matchup contributes to regret updates,
/// correcting for card-removal effects (e.g. AA vs KK has 36 combos while
/// AA vs AK has only 12).
///
/// Returns a flat `Vec` of size 169*169, indexed as `hero*169 + opp`.
#[must_use]
pub fn compute_weight_table(combo_map: &[Vec<(Card, Card)>]) -> Vec<f64> {
    let n = NUM_CANONICAL_HANDS;
    let mut weights = vec![0.0f64; n * n];

    for h_idx in 0..n {
        let hero_combos = &combo_map[h_idx];
        if hero_combos.is_empty() {
            continue;
        }

        // Pre-compute hero combo masks to avoid recomputing in inner loop.
        let hero_masks: Vec<u64> = hero_combos
            .iter()
            .map(|&(c1, c2)| (1u64 << card_bit(c1)) | (1u64 << card_bit(c2)))
            .collect();

        for o_idx in 0..n {
            let opp_combos = &combo_map[o_idx];
            if opp_combos.is_empty() {
                continue;
            }

            let mut count = 0u64;
            for &h_mask in &hero_masks {
                for &(o1, o2) in opp_combos {
                    let o_mask = (1u64 << card_bit(o1)) | (1u64 << card_bit(o2));
                    if h_mask & o_mask == 0 {
                        count += 1;
                    }
                }
            }
            #[allow(clippy::cast_precision_loss)]
            {
                weights[h_idx * n + o_idx] = count as f64;
            }
        }
    }
    weights
}

// ──────────────────────────────────────────────────────────────────────────────
// CFR traversal
// ──────────────────────────────────────────────────────────────────────────────

/// Per-call varying arguments for CFR traversal.
///
/// Bundles the state that changes across recursive calls, keeping method
/// signatures focused and avoiding `clippy::too_many_arguments`.
#[derive(Clone, Copy)]
struct TraverseArgs {
    node_idx: u32,
    hero_hand: u16,
    opp_hand: u16,
    hero_pos: u8,
    reach_hero: f64,
    reach_opp: f64,
}

/// Immutable arguments for recursive CFR traversal.
///
/// Bundles the read-only state that never changes across recursive calls,
/// keeping the per-call `traverse` signature focused on what varies:
/// node index, hands, reach probabilities, and the mutable delta buffers.
struct TraverseCtx<'a> {
    tree: &'a PostflopTree,
    layout: &'a PostflopLayout,
    equity_table: &'a [f64],
    snapshot: &'a [f64],
    iteration: u64,
    dcfr: &'a DcfrParams,
    prune_active: bool,
    prune_regret_threshold: f64,
    counters: Option<&'a SolverCounters>,
}

impl TraverseCtx<'_> {
    /// CFR traversal with equity table lookups at showdown.
    ///
    /// Dispatches on node type: terminals return payoff, chance nodes pass
    /// through, decision nodes delegate to hero/opponent helpers.
    #[inline]
    fn traverse(
        &self,
        dr: &mut [f64],
        ds: &mut [f64],
        args: TraverseArgs,
    ) -> f64 {
        match &self.tree.nodes[args.node_idx as usize] {
            PostflopNode::Terminal {
                terminal_type,
                pot_fraction,
            } => terminal_payoff(
                *terminal_type,
                *pot_fraction,
                self.equity_table,
                args.hero_hand,
                args.opp_hand,
                args.hero_pos,
                NUM_CANONICAL_HANDS,
            ),

            PostflopNode::Chance { children, .. } => {
                debug_assert!(!children.is_empty());
                self.traverse(dr, ds, TraverseArgs { node_idx: children[0], ..args })
            }

            PostflopNode::Decision {
                position, children, ..
            } => {
                let is_hero = *position == args.hero_pos;
                let bucket = if is_hero { args.hero_hand } else { args.opp_hand };
                let (start, _) = self.layout.slot(args.node_idx, bucket);
                let num_actions = children.len();

                let mut strategy = [0.0f64; MAX_POSTFLOP_ACTIONS];
                regret_matching_into(self.snapshot, start, &mut strategy[..num_actions]);

                if is_hero {
                    self.traverse_hero(dr, ds, children, &strategy, start, args)
                } else {
                    self.traverse_opponent(dr, ds, children, &strategy, args)
                }
            }
        }
    }

    /// Hero decision: compute per-action values, apply RBP pruning, update
    /// regret and strategy deltas.
    #[inline]
    fn traverse_hero(
        &self,
        dr: &mut [f64],
        ds: &mut [f64],
        children: &[u32],
        strategy: &[f64; MAX_POSTFLOP_ACTIONS],
        start: usize,
        args: TraverseArgs,
    ) -> f64 {
        let num_actions = children.len();
        let prune_mask = self.build_prune_mask(start, num_actions);

        if let Some(c) = self.counters {
            c.total_action_slots
                .fetch_add(num_actions as u64, Ordering::Relaxed);
            c.pruned_action_slots
                .fetch_add(u64::from(prune_mask.count_ones()), Ordering::Relaxed);
        }

        let mut action_values = [0.0f64; MAX_POSTFLOP_ACTIONS];
        for (i, &child) in children.iter().enumerate() {
            if prune_mask & (1 << i) != 0 {
                continue;
            }
            action_values[i] = self.traverse(
                dr,
                ds,
                TraverseArgs {
                    node_idx: child,
                    reach_hero: args.reach_hero * strategy[i],
                    ..args
                },
            );
        }

        let node_value: f64 = strategy[..num_actions]
            .iter()
            .zip(&action_values[..num_actions])
            .map(|(s, v)| s * v)
            .sum();

        let (rw, sw) = self.dcfr.iteration_weights(self.iteration);
        for (i, val) in action_values[..num_actions].iter().enumerate() {
            if prune_mask & (1 << i) == 0 {
                dr[start + i] += rw * args.reach_opp * (val - node_value);
            }
        }
        for (i, &s) in strategy[..num_actions].iter().enumerate() {
            if prune_mask & (1 << i) == 0 {
                ds[start + i] += sw * args.reach_hero * s;
            }
        }
        node_value
    }

    /// Opponent decision: weight-sum over all actions using opponent's strategy.
    #[inline]
    fn traverse_opponent(
        &self,
        dr: &mut [f64],
        ds: &mut [f64],
        children: &[u32],
        strategy: &[f64; MAX_POSTFLOP_ACTIONS],
        args: TraverseArgs,
    ) -> f64 {
        children
            .iter()
            .enumerate()
            .map(|(i, &child)| {
                strategy[i]
                    * self.traverse(
                        dr,
                        ds,
                        TraverseArgs {
                            node_idx: child,
                            reach_opp: args.reach_opp * strategy[i],
                            ..args
                        },
                    )
            })
            .sum()
    }

    /// Build regret-based pruning bitmask. Bit `i` is set when action `i`
    /// has negative regret below threshold and at least one sibling is positive.
    #[inline]
    fn build_prune_mask(&self, start: usize, num_actions: usize) -> u16 {
        if !self.prune_active {
            return 0;
        }
        let mut has_positive = false;
        let mut neg_mask: u16 = 0;
        for i in 0..num_actions {
            if self.snapshot[start + i] > 0.0 {
                has_positive = true;
            } else if self.snapshot[start + i] < self.prune_regret_threshold {
                neg_mask |= 1 << i;
            }
        }
        if has_positive { neg_mask } else { 0 }
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
// Strategy evaluation context
// ──────────────────────────────────────────────────────────────────────────────

/// Read-only context for tree evaluation using the averaged strategy.
///
/// Shared by best-response computation (`best_response_ev`) and
/// value extraction (`eval_with_avg_strategy`), eliminating parameter
/// threading through recursive calls.
struct EvalCtx<'a> {
    tree: &'a PostflopTree,
    layout: &'a PostflopLayout,
    strategy_sum: &'a [f64],
    equity_table: &'a [f64],
}

impl EvalCtx<'_> {
    /// Compute best-response EV for `br_player` against the opponent's
    /// average strategy, starting at `node_idx`.
    fn best_response_ev(
        &self, node_idx: u32, hero_hand: u16, opp_hand: u16, br_player: u8,
    ) -> f64 {
        let n = NUM_CANONICAL_HANDS;
        match &self.tree.nodes[node_idx as usize] {
            PostflopNode::Terminal { terminal_type, pot_fraction } => terminal_payoff(
                *terminal_type, *pot_fraction, self.equity_table,
                hero_hand, opp_hand, br_player, n,
            ),
            PostflopNode::Chance { children, .. } => {
                self.best_response_ev(children[0], hero_hand, opp_hand, br_player)
            }
            PostflopNode::Decision { position, children, .. } => {
                let is_br = *position == br_player;
                let bucket = if is_br { hero_hand } else { opp_hand };
                let (start, _) = self.layout.slot(node_idx, bucket);

                if is_br {
                    self.br_hero_decision(children, hero_hand, opp_hand, br_player)
                } else {
                    self.br_opp_decision(children, start, hero_hand, opp_hand, br_player)
                }
            }
        }
    }

    /// Hero best-response: pick the action with maximum EV.
    fn br_hero_decision(
        &self, children: &[u32], hero_hand: u16, opp_hand: u16, br_player: u8,
    ) -> f64 {
        children.iter()
            .map(|&child| self.best_response_ev(child, hero_hand, opp_hand, br_player))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Opponent decision: weight actions by the opponent's average strategy.
    fn br_opp_decision(
        &self, children: &[u32], start: usize, hero_hand: u16, opp_hand: u16, br_player: u8,
    ) -> f64 {
        let mut strategy = [0.0f64; MAX_POSTFLOP_ACTIONS];
        normalize_strategy_sum_into(self.strategy_sum, start, &mut strategy[..children.len()]);
        children.iter().enumerate()
            .map(|(i, &child)| strategy[i] * self.best_response_ev(child, hero_hand, opp_hand, br_player))
            .sum()
    }

    /// Walk tree using the averaged strategy for both players.
    fn eval_with_avg_strategy(
        &self, node_idx: u32, hero_hand: u16, opp_hand: u16, hero_pos: u8,
    ) -> f64 {
        let n = NUM_CANONICAL_HANDS;
        match &self.tree.nodes[node_idx as usize] {
            PostflopNode::Terminal { terminal_type, pot_fraction } => terminal_payoff(
                *terminal_type, *pot_fraction, self.equity_table,
                hero_hand, opp_hand, hero_pos, n,
            ),
            PostflopNode::Chance { children, .. } => {
                self.eval_with_avg_strategy(children[0], hero_hand, opp_hand, hero_pos)
            }
            PostflopNode::Decision { position, children, .. } => {
                let bucket = if *position == hero_pos { hero_hand } else { opp_hand };
                let (start, _) = self.layout.slot(node_idx, bucket);
                let mut strategy = [0.0f64; MAX_POSTFLOP_ACTIONS];
                normalize_strategy_sum_into(self.strategy_sum, start, &mut strategy[..children.len()]);
                children.iter().enumerate()
                    .map(|(i, &child)| strategy[i] * self.eval_with_avg_strategy(child, hero_hand, opp_hand, hero_pos))
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
    weight_table: &[f64],
) -> f64 {
    let ctx = EvalCtx { tree, layout, strategy_sum, equity_table };
    let n = NUM_CANONICAL_HANDS;
    let mut br_values = [0.0f64; 2];

    for br_player in 0..2u8 {
        #[allow(clippy::cast_possible_truncation)]
        let n_u16 = n as u16;
        let (weighted_total, weight_sum) = (0..n_u16)
            .into_par_iter()
            .flat_map_iter(|hero_hand| {
                (0..n_u16)
                    .filter(move |&opp_hand| {
                        !equity_table[hero_hand as usize * n + opp_hand as usize].is_nan()
                    })
                    .map(move |opp_hand| (hero_hand, opp_hand))
            })
            .map(|(hero_hand, opp_hand)| {
                let w = weight_table[hero_hand as usize * n + opp_hand as usize];
                let br = ctx.best_response_ev(0, hero_hand, opp_hand, br_player);
                (br * w, w)
            })
            .fold(
                || (0.0f64, 0.0f64),
                |(t, c), (v, w)| (t + v, c + w),
            )
            .reduce(
                || (0.0, 0.0),
                |(t1, c1), (t2, c2)| (t1 + t2, c1 + c2),
            );

        br_values[br_player as usize] = if weight_sum > 0.0 {
            weighted_total / weight_sum
        } else {
            0.0
        };
    }

    let pot_fraction = f64::midpoint(br_values[0], br_values[1]);
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
    weight_table: &'a [f64],
    snapshot: &'a [f64],
    iteration: u64,
    dcfr: &'a DcfrParams,
    prune_active: bool,
    /// Cumulative regret below this threshold triggers pruning candidacy.
    prune_regret_threshold: f64,
    counters: Option<&'a SolverCounters>,
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
        let w = self.weight_table[hero as usize * n + opponent as usize];
        if let Some(c) = self.counters {
            c.traversal_count.fetch_add(1, Ordering::Relaxed);
            if self.prune_active {
                c.pruned_traversal_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        let ctx = TraverseCtx {
            tree: self.tree,
            layout: self.layout,
            equity_table: self.equity_table,
            snapshot: self.snapshot,
            iteration: self.iteration,
            dcfr: self.dcfr,
            prune_active: self.prune_active,
            prune_regret_threshold: self.prune_regret_threshold,
            counters: self.counters,
        };
        for hero_pos in 0..2u8 {
            ctx.traverse(regret_delta, strategy_delta, TraverseArgs {
                node_idx: 0,
                hero_hand: hero,
                opp_hand: opponent,
                hero_pos,
                reach_hero: 1.0,
                reach_opp: w,
            });
        }
    }
}

/// Compute max positive and min negative regret values in a regret buffer.
/// Returns `(max_positive, min_negative)`. If no values exist for
/// a sign, returns 0.0.
fn extremal_regrets(regret_sum: &[f64]) -> (f64, f64) {
    let mut max_pos = 0.0f64;
    let mut min_neg = 0.0f64;
    for &v in regret_sum {
        if v > max_pos {
            max_pos = v;
        } else if v < min_neg {
            min_neg = v;
        }
    }
    (max_pos, min_neg)
}

/// Immutable context for solving a single flop via exhaustive CFR.
///
/// Bundles all read-only arguments so the solve loop and its helpers
/// have a single receiver instead of 10+ parameters.
struct SolveLoopCtx<'a, F: Fn(BuildPhase) + Sync> {
    tree: &'a PostflopTree,
    layout: &'a PostflopLayout,
    equity_table: &'a [f64],
    weight_table: &'a [f64],
    dcfr: &'a DcfrParams,
    config: &'a PostflopModelConfig,
    on_progress: &'a F,
    counters: Option<&'a SolverCounters>,
    flop_name: &'a str,
    pairs: &'a [(u16, u16)],
}

impl<F: Fn(BuildPhase) + Sync> SolveLoopCtx<'_, F> {
    /// Run one CFR traversal iteration. Returns the per-flop pruning delta
    /// `(total_action_slots_delta, pruned_action_slots_delta)`.
    #[inline]
    fn run_traversal(&self, bufs: &mut FlopBuffers, iteration: u64, prune_active: bool) -> (u64, u64) {
        let prev_total = self.counters.map_or(0, |c| c.total_action_slots.load(Ordering::Relaxed));
        let prev_pruned = self.counters.map_or(0, |c| c.pruned_action_slots.load(Ordering::Relaxed));

        {
            let ctx = PostflopCfrCtx {
                tree: self.tree,
                layout: self.layout,
                equity_table: self.equity_table,
                weight_table: self.weight_table,
                snapshot: &bufs.regret_sum,
                iteration,
                dcfr: self.dcfr,
                prune_active,
                prune_regret_threshold: self.config.prune_regret_threshold,
                counters: self.counters,
            };
            parallel_traverse_pooled(&ctx, self.pairs, std::slice::from_mut(&mut bufs.delta));
        }

        let cur_total = self.counters.map_or(0, |c| c.total_action_slots.load(Ordering::Relaxed));
        let cur_pruned = self.counters.map_or(0, |c| c.pruned_action_slots.load(Ordering::Relaxed));
        (cur_total.saturating_sub(prev_total), cur_pruned.saturating_sub(prev_pruned))
    }

    /// Apply DCFR discounting, merge per-iteration deltas into accumulators,
    /// and clamp negative regrets when configured.
    #[inline]
    fn apply_dcfr_and_merge_deltas(&self, bufs: &mut FlopBuffers, iteration: u64) {
        if self.dcfr.should_discount(iteration) {
            self.dcfr.discount_regrets(&mut bufs.regret_sum, iteration);
            self.dcfr.discount_strategy_sums(&mut bufs.strategy_sum, iteration);
        }
        add_into(&mut bufs.regret_sum, &bufs.delta.0);
        add_into(&mut bufs.strategy_sum, &bufs.delta.1);
        if self.dcfr.should_floor_regrets() {
            self.dcfr.floor_regrets(&mut bufs.regret_sum);
        }
        if self.config.regret_floor > 0.0 && self.config.prune_warmup > 0 {
            let floor = -self.config.regret_floor;
            for v in &mut bufs.regret_sum {
                *v = (*v).max(floor);
            }
        }
    }

    /// Conditionally compute exploitability if the current iteration matches
    /// the configured frequency. Returns `Some(mBB/h)` when computed.
    fn check_exploitability(&self, bufs: &FlopBuffers, iter: usize, num_iterations: usize) -> Option<f64> {
        let expl_freq = self.config.exploitability_freq.max(1);
        if iter >= 1 && (iter % expl_freq == expl_freq - 1 || iter == num_iterations - 1) {
            Some(compute_exploitability(self.tree, self.layout, &bufs.strategy_sum, self.equity_table, self.weight_table))
        } else {
            None
        }
    }

    /// Emit a `FlopProgress` callback with the current solve metrics.
    fn report_iteration_progress(&self, metrics: &IterationMetrics, num_iterations: usize) {
        (self.on_progress)(BuildPhase::FlopProgress {
            flop_name: self.flop_name.to_owned(),
            stage: FlopStage::Solving {
                iteration: metrics.iterations_used,
                max_iterations: num_iterations,
                delta: metrics.exploitability,
                metric_label: "mBB/h".into(),
                total_action_slots: metrics.flop_total_slots,
                pruned_action_slots: metrics.flop_pruned_slots,
                max_positive_regret: metrics.max_pos,
                min_negative_regret: metrics.min_neg,
            },
        });
    }
}

/// Mutable per-flop iteration state accumulated across the solve loop.
struct IterationMetrics {
    exploitability: f64,
    max_pos: f64,
    min_neg: f64,
    flop_total_slots: u64,
    flop_pruned_slots: u64,
    iterations_used: usize,
}

/// Solve a single flop using exhaustive CFR with configurable iteration weighting.
///
/// Inner `parallel_traverse_pooled` and `compute_exploitability` use rayon's
/// global thread pool. When called from `build_exhaustive` (which parallelises
/// over flops), rayon's work-stealing distributes hand-pair work across all
/// available cores, dynamically rebalancing as flops converge at different rates.
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn exhaustive_solve_one_flop(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    equity_table: &[f64],
    weight_table: &[f64],
    num_iterations: usize,
    convergence_threshold: f64,
    flop_name: &str,
    dcfr: &DcfrParams,
    config: &PostflopModelConfig,
    on_progress: &(impl Fn(BuildPhase) + Sync),
    counters: Option<&SolverCounters>,
    bufs: &mut FlopBuffers,
) -> (f64, usize) {
    let n = NUM_CANONICAL_HANDS;
    let pairs: Vec<(u16, u16)> = (0..n as u16)
        .flat_map(|h1| (0..n as u16).map(move |h2| (h1, h2)))
        .filter(|&(h1, h2)| !equity_table[h1 as usize * n + h2 as usize].is_nan())
        .collect();

    if let Some(c) = counters {
        c.total_expected_traversals
            .fetch_add((pairs.len() * num_iterations) as u64, Ordering::Relaxed);
    }

    let ctx = SolveLoopCtx {
        tree, layout, equity_table, weight_table, dcfr, config, on_progress, counters,
        flop_name,
        pairs: &pairs,
    };
    let mut m = IterationMetrics {
        exploitability: f64::INFINITY,
        max_pos: 0.0, min_neg: 0.0,
        flop_total_slots: 0, flop_pruned_slots: 0,
        iterations_used: 0,
    };
    ctx.report_iteration_progress(&m, num_iterations);

    for iter in 0..num_iterations {
        let iteration = iter as u64 + 1;
        let prune_active = config.prune_warmup > 0
            && iter >= config.prune_warmup
            && (config.prune_explore_freq == 0 || iter % config.prune_explore_freq != 0);

        let (dt, dp) = ctx.run_traversal(bufs, iteration, prune_active);
        m.flop_total_slots += dt;
        m.flop_pruned_slots += dp;

        ctx.apply_dcfr_and_merge_deltas(bufs, iteration);
        m.iterations_used = iter + 1;

        if iter % 10 == 0 || iter == num_iterations - 1 {
            (m.max_pos, m.min_neg) = extremal_regrets(&bufs.regret_sum);
        }
        if let Some(e) = ctx.check_exploitability(bufs, iter, num_iterations) {
            m.exploitability = e;
        }
        ctx.report_iteration_progress(&m, num_iterations);
        if iter >= 1 && m.exploitability < convergence_threshold {
            break;
        }
    }

    (m.exploitability, m.iterations_used)
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
    let ctx = EvalCtx { tree, layout, strategy_sum, equity_table };
    let n = NUM_CANONICAL_HANDS;
    let mut values = vec![f64::NAN; 2 * n * n];

    for hero_hand in 0..n as u16 {
        for opp_hand in 0..n as u16 {
            if equity_table[hero_hand as usize * n + opp_hand as usize].is_nan() {
                continue;
            }
            for hero_pos in 0..2u8 {
                let ev = ctx.eval_with_avg_strategy(0, hero_hand, opp_hand, hero_pos);
                let idx = hero_pos as usize * n * n + hero_hand as usize * n + opp_hand as usize;
                values[idx] = ev;
            }
        }
    }
    values
}

// ──────────────────────────────────────────────────────────────────────────────
// Entry point
// ──────────────────────────────────────────────────────────────────────────────

/// Shared immutable context for building postflop values across all flops.
///
/// Bundles the read-only state that every per-flop solve needs, avoiding
/// 13-parameter function signatures.
struct BuildCtx<'a, F: Fn(BuildPhase) + Sync> {
    config: &'a PostflopModelConfig,
    tree: &'a PostflopTree,
    layout: &'a PostflopLayout,
    flops: &'a [[Card; 3]],
    pre_equity_tables: Option<&'a [Vec<f64>]>,
    dcfr: DcfrParams,
    on_progress: &'a F,
    counters: Option<&'a SolverCounters>,
    completed: AtomicUsize,
    num_iterations: usize,
}

impl<F: Fn(BuildPhase) + Sync> BuildCtx<'_, F> {
    /// Solve one flop and extract its strategy values.
    fn solve_and_extract_flop(&self, flop_idx: usize) -> Vec<f64> {
        let flop = self.flops[flop_idx];
        let flop_name = format!("{}{}{}", flop[0], flop[1], flop[2]);

        let combo_map = build_combo_map(&flop);
        let weight_table = compute_weight_table(&combo_map);
        let equity_table = if let Some(tables) = self.pre_equity_tables {
            tables[flop_idx].clone()
        } else {
            compute_equity_table(&combo_map, flop)
        };

        thread_local! {
            static FLOP_BUFS: RefCell<Option<FlopBuffers>> = const { RefCell::new(None) };
        }

        let buf_size = self.layout.total_size;

        // Take ownership of the cached buffer (if any) from the thread-local,
        // then immediately drop the RefCell borrow. This avoids holding a
        // borrow_mut across nested rayon parallelism (compute_exploitability
        // and parallel_traverse_pooled both use par_iter), which would panic
        // if rayon work-stealing re-enters this function on the same thread.
        let mut bufs = FLOP_BUFS.with(|cell| {
            cell.borrow_mut()
                .take()
                .filter(|b| b.regret_sum.len() == buf_size)
                .map_or_else(|| FlopBuffers::new(buf_size), |mut b| { b.reset(); b })
        });

        exhaustive_solve_one_flop(
            self.tree, self.layout, &equity_table, &weight_table, self.num_iterations,
            self.config.cfr_convergence_threshold, &flop_name, &self.dcfr,
            self.config, self.on_progress, self.counters, &mut bufs,
        );

        let values = exhaustive_extract_values(
            self.tree, self.layout, &bufs.strategy_sum, &equity_table,
        );

        // Return buffer to thread-local cache for reuse by the next flop.
        FLOP_BUFS.with(|cell| { cell.borrow_mut().replace(bufs); });

        (self.on_progress)(BuildPhase::FlopProgress { flop_name, stage: FlopStage::Done });
        let done = self.completed.fetch_add(1, Ordering::Relaxed) + 1;
        (self.on_progress)(BuildPhase::MccfrFlopsCompleted {
            completed: done, total: self.flops.len(),
        });

        values
    }
}

/// Build postflop values using exhaustive CFR with equity tables.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_exhaustive(
    config: &PostflopModelConfig,
    tree: &PostflopTree,
    layout: &PostflopLayout,
    node_streets: &[Street],
    flops: &[[Card; 3]],
    pre_equity_tables: Option<&[Vec<f64>]>,
    on_progress: &(impl Fn(BuildPhase) + Sync),
    counters: Option<&SolverCounters>,
) -> PostflopValues {
    let _ = node_streets;
    let n = NUM_CANONICAL_HANDS;

    let ctx = BuildCtx {
        config, tree, layout, flops, pre_equity_tables,
        dcfr: match config.cfr_variant {
            CfrVariant::Linear => DcfrParams::linear(),
            CfrVariant::Dcfr => DcfrParams::default(),
            CfrVariant::Vanilla => DcfrParams::vanilla(),
            CfrVariant::CfrPlus => DcfrParams::from_config(CfrVariant::CfrPlus, 0.0, 0.0, 0.0, 0),
        },
        on_progress, counters,
        completed: AtomicUsize::new(0),
        num_iterations: config.postflop_solve_iterations as usize,
    };

    let results: Vec<Vec<f64>> = (0..flops.len())
        .into_par_iter()
        .map(|flop_idx| {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ctx.solve_and_extract_flop(flop_idx)
            })) {
                Ok(values) => values,
                Err(panic) => {
                    let flop = flops[flop_idx];
                    let flop_name = format!("{}{}{}", flop[0], flop[1], flop[2]);
                    let msg = if let Some(s) = panic.downcast_ref::<&str>() {
                        (*s).to_string()
                    } else if let Some(s) = panic.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    eprintln!("PANIC solving flop {flop_name} (idx={flop_idx}): {msg}");
                    // Return NaN values so the solve can continue with other flops.
                    vec![f64::NAN; 2 * n * n]
                }
            }
        })
        .collect();

    let num_flops = flops.len();
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

    /// Build a uniform weight table (all 1.0) for tests using synthetic equity.
    /// Preserves existing test behavior since all matchups are equally weighted.
    fn synthetic_weight_table() -> Vec<f64> {
        vec![1.0; NUM_CANONICAL_HANDS * NUM_CANONICAL_HANDS]
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

    #[timed_test]
    fn bitmask_matches_contains_for_card_exclusion() {
        // Verify the bitmask approach agrees with slice::contains for a small
        // set of cards. This guards against bit-encoding mismatches.
        let deck = all_cards_vec();
        let flop = test_flop();

        let mut deck_bits = [0u8; 52];
        for (i, &c) in deck.iter().enumerate() {
            deck_bits[i] = card_bit(c);
        }

        // Pick two hole cards and two opponent cards that don't overlap with
        // the flop. Use deck positions to avoid accidental collisions.
        let h1 = Card::new(Value::Ace, Suit::Spade);
        let h2 = Card::new(Value::King, Suit::Club);
        let o1 = Card::new(Value::Ten, Suit::Heart);
        let o2 = Card::new(Value::Nine, Suit::Diamond);

        let used_arr = [flop[0], flop[1], flop[2], h1, h2, o1, o2];
        let used_mask: u64 = (1u64 << card_bit(flop[0]))
            | (1u64 << card_bit(flop[1]))
            | (1u64 << card_bit(flop[2]))
            | (1u64 << card_bit(h1))
            | (1u64 << card_bit(h2))
            | (1u64 << card_bit(o1))
            | (1u64 << card_bit(o2));

        // Verify every card in the deck: bitmask matches array contains.
        for (i, &c) in deck.iter().enumerate() {
            let arr_hit = used_arr.contains(&c);
            let mask_hit = used_mask & (1u64 << deck_bits[i]) != 0;
            assert_eq!(
                arr_hit, mask_hit,
                "mismatch for card {c} at deck pos {i}: contains={arr_hit}, bitmask={mask_hit}"
            );
        }

        // Also verify that exactly 7 bits are set.
        assert_eq!(used_mask.count_ones(), 7, "should have exactly 7 used cards");
    }

    #[timed_test]
    fn card_bit_produces_unique_indices() {
        let deck = all_cards_vec();
        let mut seen = 0u64;
        for &c in &deck {
            let bit = card_bit(c);
            assert!(bit < 52, "card_bit out of range: {bit}");
            let mask = 1u64 << bit;
            assert_eq!(seen & mask, 0, "duplicate bit {bit} for card {c}");
            seen |= mask;
        }
        assert_eq!(seen.count_ones(), 52);
    }

    #[timed_test(10)]
    #[ignore = "slow: full 169x169 equity table computation"]
    fn equity_table_has_correct_size() {
        let flop = test_flop();
        let combo_map = build_combo_map(&flop);
        let eq = compute_equity_table(&combo_map, flop);
        assert_eq!(eq.len(), 169 * 169);
    }

    #[timed_test(10)]
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

    #[timed_test(10)]
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

    #[allow(clippy::erasing_op, clippy::identity_op)]
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

    #[timed_test(10)]
    #[ignore = "slow: 169-hand exhaustive solve in debug mode"]
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
        let weight_table = synthetic_weight_table();
        let dcfr = DcfrParams::linear();
        let mut bufs = FlopBuffers::new(layout.total_size);

        let (_delta, iterations_used) = exhaustive_solve_one_flop(
            &tree,
            &layout,
            &equity_table,
            &weight_table,
            3,
            0.001,
            "test",
            &dcfr,
            &config,
            &|_| {},
            None,
            &mut bufs,
        );

        assert!(iterations_used > 0);
        let has_nonzero = bufs.strategy_sum.iter().any(|&v| v.abs() > 1e-15);
        assert!(has_nonzero, "strategy_sum should have non-zero entries");
    }

    #[timed_test(10)]
    #[ignore = "slow: 169-hand exhaustive solve in debug mode"]
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
        let weight_table = synthetic_weight_table();
        let dcfr = DcfrParams::linear();
        let mut bufs = FlopBuffers::new(layout.total_size);

        let (_delta, _iterations_used) = exhaustive_solve_one_flop(
            &tree,
            &layout,
            &equity_table,
            &weight_table,
            2,
            0.001,
            "test",
            &dcfr,
            &config,
            &|_| {},
            None,
            &mut bufs,
        );

        let values =
            exhaustive_extract_values(&tree, &layout, &bufs.strategy_sum, &equity_table);
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
                #[allow(clippy::cast_possible_truncation)]
                Some((i as u32, *folder, *pot_fraction))
            } else {
                None
            }
        });

        if let Some((node_idx, folder, pot_fraction)) = fold_node {
            let snapshot = regret_sum.clone();
            let ctx = TraverseCtx {
                tree: &tree,
                layout: &layout,
                equity_table: &equity_table,
                snapshot: &snapshot,
                iteration: 0,
                dcfr: &dcfr,
                prune_active: false,
                prune_regret_threshold: 0.0,
                counters: None,
            };
            let ev = ctx.traverse(
                &mut regret_sum,
                &mut strategy_sum,
                TraverseArgs {
                    node_idx,
                    hero_hand: 0,
                    opp_hand: 1,
                    hero_pos: 0,
                    reach_hero: 1.0,
                    reach_opp: 1.0,
                },
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
                #[allow(clippy::cast_possible_truncation)]
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
            let ctx = TraverseCtx {
                tree: &tree,
                layout: &layout,
                equity_table: &equity_table,
                snapshot: &snapshot,
                iteration: 0,
                dcfr: &dcfr,
                prune_active: false,
                prune_regret_threshold: 0.0,
                counters: None,
            };
            let ev = ctx.traverse(
                &mut regret_sum,
                &mut strategy_sum,
                TraverseArgs {
                    node_idx,
                    hero_hand,
                    opp_hand,
                    hero_pos: 0,
                    reach_hero: 1.0,
                    reach_opp: 1.0,
                },
            );

            let expected = eq * pot_fraction - pot_fraction / 2.0;
            assert!(
                (ev - expected).abs() < 1e-9,
                "showdown EV: expected {expected}, got {ev}"
            );
        }
    }

    /// Helper: build a minimal tree + layout + synthetic equity for exploitability tests.
    fn expl_test_fixtures() -> (PostflopTree, PostflopLayout, Vec<f64>, Vec<f64>) {
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
        let weight_table = synthetic_weight_table();
        (tree, layout, equity_table, weight_table)
    }

    #[timed_test(10)]
    #[ignore = "slow: O(169²) best-response in debug mode"]
    fn exploitability_is_positive_for_uniform_strategy() {
        let (tree, layout, equity_table, weight_table) = expl_test_fixtures();
        // All-zero strategy_sum -> normalize_strategy_sum returns uniform
        let strategy_sum = vec![0.0f64; layout.total_size];
        let expl = compute_exploitability(&tree, &layout, &strategy_sum, &equity_table, &weight_table);
        assert!(
            expl > 0.0,
            "uniform strategy should be exploitable, got {expl}"
        );
    }

    #[timed_test(10)]
    #[ignore = "slow: O(169²) best-response in debug mode"]
    fn exploitability_decreases_with_training() {
        let (tree, layout, equity_table, weight_table) = expl_test_fixtures();
        // Uniform strategy exploitability (baseline).
        let uniform_sum = vec![0.0f64; layout.total_size];
        let expl_uniform = compute_exploitability(&tree, &layout, &uniform_sum, &equity_table, &weight_table);
        // 3 CFR iterations with no early stop. With snapshot-based CFR the first
        // iteration reads uniform regrets, so convergence needs one extra iteration
        // compared to in-place updates.
        let dcfr = DcfrParams::linear();
        let no_prune = PostflopModelConfig::exhaustive_fast();
        let mut bufs = FlopBuffers::new(layout.total_size);
        let (delta, _iterations_used) = exhaustive_solve_one_flop(
            &tree, &layout, &equity_table, &weight_table, 3, 0.0, "test", &dcfr, &no_prune, &|_| {}, None,
            &mut bufs,
        );
        assert!(
            delta < expl_uniform,
            "trained exploitability ({:.6}) should be less than uniform ({:.6})",
            delta,
            expl_uniform
        );
    }

    #[timed_test(10)]
    #[ignore = "slow: sequential + parallel solve comparison in debug mode"]
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
        let weight_table = synthetic_weight_table();
        let dcfr = DcfrParams::linear();

        // Solve with default thread pool (parallel)
        let mut bufs_par = FlopBuffers::new(layout.total_size);
        let (_delta_par, iterations_par) = exhaustive_solve_one_flop(
            &tree, &layout, &equity_table, &weight_table, 5, 0.0, "par", &dcfr, &config, &|_| {}, None,
            &mut bufs_par,
        );

        // Solve with 1 thread (sequential)
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let mut bufs_seq = FlopBuffers::new(layout.total_size);
        let (_delta_seq, iterations_seq) = pool.install(|| {
            exhaustive_solve_one_flop(
                &tree, &layout, &equity_table, &weight_table, 5, 0.0, "seq", &dcfr, &config, &|_| {}, None,
                &mut bufs_seq,
            )
        });

        assert_eq!(iterations_par, iterations_seq);
        for (p, s) in bufs_par
            .strategy_sum
            .iter()
            .zip(bufs_seq.strategy_sum.iter())
        {
            assert!(
                (p - s).abs() < 1e-6,
                "strategy_sum mismatch: parallel={p}, sequential={s}"
            );
        }
    }

    #[timed_test(10)]
    #[ignore = "slow: O(169²) best-response in debug mode"]
    fn exploitability_early_stopping_triggers() {
        let (tree, layout, equity_table, weight_table) = expl_test_fixtures();
        // Very generous threshold (30 BB/h) -- stops at first exploitability check.
        let threshold_mbb = 30_000.0;
        let dcfr = DcfrParams::linear();
        let no_prune = PostflopModelConfig::exhaustive_fast();
        let mut bufs = FlopBuffers::new(layout.total_size);
        let (delta, iterations_used) = exhaustive_solve_one_flop(
            &tree,
            &layout,
            &equity_table,
            &weight_table,
            30,
            threshold_mbb,
            "test",
            &dcfr,
            &no_prune,
            &|_| {},
            None,
            &mut bufs,
        );
        assert_eq!(
            iterations_used, 2,
            "should stop at first exploitability check (iter 1), used {}",
            iterations_used
        );
        assert!(
            delta < threshold_mbb,
            "final exploitability should be below threshold, got {} mBB/h",
            delta
        );
    }

    #[timed_test(10)]
    #[ignore = "slow: 169-hand exhaustive solve in debug mode"]
    fn exhaustive_solve_with_pruning_produces_strategy() {
        let config = PostflopModelConfig {
            bet_sizes: vec![1.0],
            max_raises_per_street: 0,
            postflop_solve_iterations: 5,
            prune_warmup: 2,
            prune_explore_freq: 5,
            regret_floor: 1_000_000.0,
            ..PostflopModelConfig::exhaustive_fast()
        };
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);
        let equity_table = synthetic_equity_table();
        let weight_table = synthetic_weight_table();
        let dcfr = DcfrParams::linear();
        let mut bufs = FlopBuffers::new(layout.total_size);

        let (_delta, iterations_used) = exhaustive_solve_one_flop(
            &tree,
            &layout,
            &equity_table,
            &weight_table,
            5,
            0.0,
            "prune_test",
            &dcfr,
            &config,
            &|_| {},
            None,
            &mut bufs,
        );

        assert!(iterations_used > 0, "should complete iterations");
        let has_nonzero = bufs.strategy_sum.iter().any(|&v| v.abs() > 1e-15);
        assert!(has_nonzero, "strategy_sum should have non-zero entries with pruning enabled");
    }

    #[timed_test(30)]
    #[ignore = "slow: 169-hand exhaustive solve in debug mode"]
    fn pruning_does_not_break_convergence() {
        let base = PostflopModelConfig {
            bet_sizes: vec![1.0],
            max_raises_per_street: 0,
            postflop_solve_iterations: 5,
            ..PostflopModelConfig::exhaustive_fast()
        };
        let tree = PostflopTree::build_with_spr(&base, 3.5).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);
        let equity_table = synthetic_equity_table();
        let weight_table = synthetic_weight_table();
        let dcfr = DcfrParams::linear();

        // Unpruned baseline (prune_warmup=0)
        let mut bufs_unpruned = FlopBuffers::new(layout.total_size);
        let (_delta_unpruned, iterations_unpruned) = exhaustive_solve_one_flop(
            &tree, &layout, &equity_table, &weight_table, 5, 0.0, "no_prune", &dcfr, &base, &|_| {}, None,
            &mut bufs_unpruned,
        );

        // Pruned run
        let pruned_config = PostflopModelConfig {
            prune_warmup: 2,
            prune_explore_freq: 3,
            regret_floor: 1_000_000.0,
            ..base.clone()
        };
        let mut bufs_pruned = FlopBuffers::new(layout.total_size);
        let (_delta_pruned, iterations_pruned) = exhaustive_solve_one_flop(
            &tree, &layout, &equity_table, &weight_table, 5, 0.0, "pruned", &dcfr, &pruned_config, &|_| {}, None,
            &mut bufs_pruned,
        );

        // Both should produce valid strategies
        assert!(iterations_unpruned > 0);
        assert!(iterations_pruned > 0);
        let unpruned_nonzero = bufs_unpruned.strategy_sum.iter().any(|&v| v.abs() > 1e-15);
        let pruned_nonzero = bufs_pruned.strategy_sum.iter().any(|&v| v.abs() > 1e-15);
        assert!(unpruned_nonzero, "unpruned should have non-zero strategy");
        assert!(pruned_nonzero, "pruned should have non-zero strategy");
    }

    #[timed_test(30)]
    #[ignore = "slow: 169-hand exhaustive solve in debug mode"]
    fn solver_counters_are_incremented() {
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
        let weight_table = synthetic_weight_table();
        let dcfr = DcfrParams::linear();
        let counters = SolverCounters::default();
        let mut bufs = FlopBuffers::new(layout.total_size);
        let _result = exhaustive_solve_one_flop(
            &tree, &layout, &equity_table, &weight_table, 5, 0.0, "test", &dcfr, &config,
            &|_| {}, Some(&counters), &mut bufs,
        );
        let traversals = counters.traversal_count.load(Ordering::Relaxed);
        assert!(traversals > 0, "traversal counter should be incremented, got {traversals}");
        let total_actions = counters.total_action_slots.load(Ordering::Relaxed);
        assert!(total_actions > 0, "action counter should be incremented, got {total_actions}");
    }

    /// Test with full.yaml-equivalent config (2 bet sizes, 2 max raises, SPR=2).
    /// This exercises the deeper tree that the production solve uses.
    #[timed_test(120)]
    #[ignore = "slow: 169-hand exhaustive solve in debug mode"]
    fn full_config_spr2_does_not_panic() {
        let config = PostflopModelConfig {
            bet_sizes: vec![0.3, 0.75],
            max_raises_per_street: 2,
            postflop_solve_iterations: 3,
            cfr_convergence_threshold: 100.0,
            prune_warmup: 2,
            prune_explore_freq: 10,
            prune_regret_threshold: 0.0,
            ..PostflopModelConfig::exhaustive_fast()
        };
        let tree = PostflopTree::build_with_spr(&config, 2.0).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);

        eprintln!("tree nodes: {}, layout total_size: {}", tree.node_count(), layout.total_size);

        let equity_table = synthetic_equity_table();
        let weight_table = synthetic_weight_table();
        let dcfr = DcfrParams::default(); // DCFR variant
        let mut bufs = FlopBuffers::new(layout.total_size);

        let (_delta, iterations_used) = exhaustive_solve_one_flop(
            &tree, &layout, &equity_table, &weight_table, 3, 100.0, "test_full", &dcfr,
            &config, &|_| {}, None, &mut bufs,
        );

        assert!(iterations_used > 0, "should complete at least 1 iteration");
        let has_nonzero = bufs.strategy_sum.iter().any(|&v| v.abs() > 1e-15);
        assert!(has_nonzero, "strategy_sum should have non-zero entries");
    }

    /// Test build_exhaustive with full.yaml config on multiple flops to exercise
    /// the parallel solve path and catch_unwind protection.
    #[timed_test(120)]
    #[ignore = "slow: 169-hand exhaustive solve in debug mode"]
    fn full_config_parallel_multi_flop() {
        use crate::preflop::postflop_hands::sample_canonical_flops;

        let config = PostflopModelConfig {
            bet_sizes: vec![0.3, 0.75],
            max_raises_per_street: 2,
            postflop_solve_iterations: 3,
            cfr_convergence_threshold: 100.0,
            prune_warmup: 2,
            prune_explore_freq: 10,
            prune_regret_threshold: 0.0,
            max_flop_boards: 10,
            ..PostflopModelConfig::exhaustive_fast()
        };
        let tree = PostflopTree::build_with_spr(&config, 2.0).unwrap();
        let node_streets = annotate_streets(&tree);
        let n = NUM_CANONICAL_HANDS;
        let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);
        let flops = sample_canonical_flops(config.max_flop_boards);

        let values = build_exhaustive(
            &config, &tree, &layout, &node_streets, &flops, None, &|_| {}, None,
        );

        assert_eq!(values.num_flops, flops.len());
    }

    #[timed_test(300)]
    #[ignore = "slow: compares old vs new equity table"]
    fn restructured_equity_table_matches_original() {
        let flop = test_flop();
        let combo_map = build_combo_map(&flop);
        let original = compute_equity_table_reference(&combo_map, flop);
        let restructured = compute_equity_table(&combo_map, flop);
        assert_eq!(original.len(), restructured.len());
        let n = NUM_CANONICAL_HANDS;
        for h in 0..n {
            for o in 0..n {
                let idx = h * n + o;
                let a = original[idx];
                let b = restructured[idx];
                if a.is_nan() {
                    assert!(b.is_nan(), "at [{h}][{o}]: original=NaN, new={b}");
                } else {
                    assert!((a - b).abs() < 1e-10, "at [{h}][{o}]: orig={a}, new={b}, diff={}", (a - b).abs());
                }
            }
        }
    }

    #[timed_test]
    fn weight_table_basic_properties() {
        let flop = test_flop();
        let combo_map = build_combo_map(&flop);
        let weights = compute_weight_table(&combo_map);
        let n = NUM_CANONICAL_HANDS;

        assert_eq!(weights.len(), n * n);

        // All weights should be non-negative.
        assert!(
            weights.iter().all(|&w| w >= 0.0),
            "weights must be non-negative"
        );

        // Symmetry: weight[h][o] == weight[o][h].
        for h in 0..n {
            for o in h + 1..n {
                assert_eq!(
                    weights[h * n + o],
                    weights[o * n + h],
                    "weight table should be symmetric for ({h}, {o})"
                );
            }
        }

        // Hands with empty combo maps should have zero weight.
        for h in 0..n {
            if combo_map[h].is_empty() {
                for o in 0..n {
                    assert_eq!(weights[h * n + o], 0.0);
                    assert_eq!(weights[o * n + h], 0.0);
                }
            }
        }

        // At least some weights should be > 0.
        assert!(
            weights.iter().any(|&w| w > 0.0),
            "should have positive weights"
        );
    }

    #[timed_test(10)]
    #[ignore = "slow: full 169x169 equity + weight table computation"]
    fn weight_table_agrees_with_equity_table_validity() {
        // Any pair with NaN equity should have 0 weight, and vice versa.
        let flop = test_flop();
        let combo_map = build_combo_map(&flop);
        let weights = compute_weight_table(&combo_map);
        let equity = compute_equity_table(&combo_map, flop);
        let n = NUM_CANONICAL_HANDS;

        for h in 0..n {
            for o in 0..n {
                let idx = h * n + o;
                if equity[idx].is_nan() {
                    assert_eq!(
                        weights[idx], 0.0,
                        "NaN equity at ({h},{o}) should have 0 weight, got {}",
                        weights[idx]
                    );
                } else {
                    assert!(
                        weights[idx] > 0.0,
                        "valid equity at ({h},{o}) should have positive weight, got {}",
                        weights[idx]
                    );
                }
            }
        }
    }
}
