//! Backward-induction computation of counterfactual boundary values (CBVs).
//!
//! After a blueprint training snapshot is saved, this module walks the
//! abstract game tree bottom-up and computes, for each player and each
//! street-boundary (Chance) node, the expected value per bucket of
//! continuing play under the blueprint strategy.
//!
//! These CBVs are stored in a [`CbvTable`] and persisted alongside the
//! snapshot so that the real-time subgame solver can use them as leaf
//! values without re-traversing the full tree.

// Arena indices are u32, bucket counts are u16. Truncation is safe.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use super::cbv::CbvTable;
use crate::blueprint_v2::bundle::BlueprintV2Strategy;
use crate::blueprint_v2::game_tree::{GameNode, GameTree, TerminalKind};

/// Sparse transition distribution from one street's buckets to the next.
///
/// `dist[prev_bucket]` is a list of `(next_bucket, probability)` pairs.
/// Each row sums to 1.0.
#[derive(Debug, Clone)]
pub struct BucketTransition {
    /// `dist[prev_bucket]` = vec of `(next_bucket, weight)` pairs.
    pub dist: Vec<Vec<(u16, f32)>>,
}

/// Shared context for the recursive backward-induction traversal.
///
/// Groups the immutable references and per-player state needed at
/// every node, keeping the recursive function signature small.
struct TraversalCtx<'a> {
    tree: &'a GameTree,
    strategy: &'a BlueprintV2Strategy,
    decision_idx_map: &'a [u32],
    bucket_counts: [u16; 4],
    /// Arena-index to position in the [`CbvTable`]'s node list
    /// (`u32::MAX` for non-chance nodes).
    chance_pos: &'a [u32],
    player: u8,
}

/// Compute CBV tables for both players via backward induction through
/// the abstract game tree, without transition matrices.
///
/// Without transitions, Chance nodes use a uniform distribution across
/// all next-street buckets. This is a reasonable approximation when
/// bucket files are not available.
///
/// Returns `[player_0_cbvs, player_1_cbvs]`. Each [`CbvTable`] contains
/// one entry per `(chance_node, bucket)` pair, where chance nodes are
/// the street-boundary nodes in the tree.
///
/// # Arguments
///
/// * `strategy` — The extracted blueprint strategy (action probabilities).
/// * `tree` — The abstract game tree.
/// * `bucket_counts` — Number of buckets per street `[preflop, flop, turn, river]`.
#[must_use]
pub fn compute_cbvs(
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    bucket_counts: [u16; 4],
) -> [CbvTable; 2] {
    let no_transitions: [Option<BucketTransition>; 3] = [None, None, None];
    compute_cbvs_with_transitions(strategy, tree, bucket_counts, &no_transitions)
}

/// Compute CBV tables for both players via backward induction through
/// the abstract game tree, using precomputed bucket transition matrices.
///
/// The `transitions` array has 3 slots corresponding to street boundaries:
/// - `transitions[0]`: preflop → flop (street index 0→1)
/// - `transitions[1]`: flop → turn (street index 1→2)
/// - `transitions[2]`: turn → river (street index 2→3)
///
/// When a slot is `None`, the Chance node uses a uniform distribution
/// across all next-street buckets.
///
/// # Two-pass approach
///
/// Phase 1: Iterate over all next-street bucket indices. At each Chance
/// node, recurse into the child subtree with the next-street bucket and
/// store the resulting value in a temporary buffer.
///
/// Phase 2: For each Chance node and each prev-street bucket, compute
/// the transition-weighted average from the buffered child values.
#[must_use]
pub fn compute_cbvs_with_transitions(
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    bucket_counts: [u16; 4],
    transitions: &[Option<BucketTransition>; 3],
) -> [CbvTable; 2] {
    let decision_idx_map = tree.decision_index_map();

    // Identify all chance (street-boundary) nodes.
    let chance_indices: Vec<u32> = tree
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| matches!(n, GameNode::Chance { .. }))
        .map(|(i, _)| i as u32)
        .collect();

    let mut tables = [
        build_empty_table(&chance_indices, &tree.nodes, bucket_counts),
        build_empty_table(&chance_indices, &tree.nodes, bucket_counts),
    ];

    // Map from arena index -> position in chance_indices (for fast lookup).
    let mut chance_pos: Vec<u32> = vec![u32::MAX; tree.nodes.len()];
    for (pos, &arena_idx) in chance_indices.iter().enumerate() {
        chance_pos[arena_idx as usize] = pos as u32;
    }

    let max_buckets = *bucket_counts.iter().max().unwrap_or(&1) as usize;
    let num_chance = chance_indices.len();

    for player in 0..2u8 {
        let ctx = TraversalCtx {
            tree,
            strategy,
            decision_idx_map: &decision_idx_map,
            bucket_counts,
            chance_pos: &chance_pos,
            player,
        };
        let table = &mut tables[player as usize];

        // Phase 1: Compute child values for every next-street bucket at
        // each Chance node. The per-bucket loop recurses through the tree;
        // at Chance nodes, the bucket is treated as a next-street index
        // and the child value is stored in a temporary buffer.
        //
        // `chance_child_values[chance_pos][next_bucket]` holds the child
        // value computed during phase 1.
        let mut chance_child_values: Vec<Vec<f32>> = Vec::with_capacity(num_chance);
        for &arena_idx in &chance_indices {
            if let GameNode::Chance { next_street, .. } = &tree.nodes[arena_idx as usize] {
                let next_b = bucket_counts[*next_street as usize] as usize;
                chance_child_values.push(vec![0.0; next_b]);
            } else {
                chance_child_values.push(Vec::new());
            }
        }

        for bucket in 0..max_buckets {
            compute_node_value_phase1(
                &ctx,
                bucket as u16,
                tree.root,
                &mut chance_child_values,
            );
        }

        // Phase 2: For each Chance node and each prev-street bucket,
        // compute the transition-weighted average of child values.
        for (pos, &arena_idx) in chance_indices.iter().enumerate() {
            if let GameNode::Chance { next_street, .. } = &tree.nodes[arena_idx as usize] {
                let prev_street_idx = (*next_street as usize).saturating_sub(1);
                let prev_b_count = bucket_counts[prev_street_idx] as usize;
                let next_b_count = bucket_counts[*next_street as usize] as usize;

                // Which transition slot? preflop->flop = 0, flop->turn = 1, turn->river = 2
                let transition_slot = prev_street_idx;
                let transition = &transitions[transition_slot];

                for prev_b in 0..prev_b_count {
                    let weighted_value = match transition {
                        Some(t) if prev_b < t.dist.len() => {
                            // Use transition distribution
                            let mut sum = 0.0_f32;
                            for &(next_b, weight) in &t.dist[prev_b] {
                                let nb = next_b as usize;
                                if nb < next_b_count {
                                    sum += weight * chance_child_values[pos][nb];
                                }
                            }
                            sum
                        }
                        _ => {
                            // No transition data: identity mapping --
                            // prev_bucket b maps to next_bucket b (clamped
                            // to the next-street range).
                            let clamped = prev_b.min(next_b_count.saturating_sub(1));
                            chance_child_values[pos][clamped]
                        }
                    };

                    if prev_b < table.buckets_per_node[pos] as usize {
                        table.values[table.node_offsets[pos] + prev_b] = weighted_value;
                    }
                }
            }
        }
    }

    tables
}

/// Phase 1 of backward induction: compute child values for each
/// next-street bucket at every Chance node.
///
/// At Chance nodes, the `bucket` parameter is treated as a next-street
/// bucket index. The child value is stored into `chance_child_values`
/// (not the CBV table) for later aggregation in phase 2.
fn compute_node_value_phase1(
    ctx: &TraversalCtx<'_>,
    bucket: u16,
    node_idx: u32,
    chance_child_values: &mut [Vec<f32>],
) -> f32 {
    match &ctx.tree.nodes[node_idx as usize] {
        GameNode::Terminal { kind, invested, .. } => {
            terminal_bucket_value(*kind, invested, &ctx.tree.blinds, ctx.player, bucket, ctx.bucket_counts)
        }

        GameNode::Chance { next_street, child } => {
            // Clamp the bucket to the next street's range and recurse.
            let next_buckets = ctx.bucket_counts[*next_street as usize];
            let child_bucket = bucket.min(next_buckets.saturating_sub(1));

            let child_value = compute_node_value_phase1(
                ctx, child_bucket, *child, chance_child_values,
            );

            // Store the child value indexed by the next-street bucket.
            let pos = ctx.chance_pos[node_idx as usize];
            if pos != u32::MAX {
                let pos = pos as usize;
                let nb = child_bucket as usize;
                if nb < chance_child_values[pos].len() {
                    chance_child_values[pos][nb] = child_value;
                }
            }

            child_value
        }

        GameNode::Decision {
            street, children, ..
        } => {
            let num_actions = children.len();
            let decision_idx = ctx.decision_idx_map[node_idx as usize];
            if decision_idx == u32::MAX {
                // Should not happen in a well-formed tree.
                return 0.0;
            }

            let street_buckets = ctx.bucket_counts[*street as usize];
            let acting_bucket = bucket.min(street_buckets.saturating_sub(1));

            let probs = ctx
                .strategy
                .get_action_probs(decision_idx as usize, acting_bucket);
            let uniform = 1.0 / num_actions as f32;

            let mut value = 0.0_f32;
            for (a, &child_idx) in children.iter().enumerate() {
                let p = if a < probs.len() { probs[a] } else { uniform };
                if p < 1e-9 {
                    continue;
                }
                let child_value = compute_node_value_phase1(
                    ctx, bucket, child_idx, chance_child_values,
                );
                value += p * child_value;
            }

            value
        }
    }
}

/// Approximate terminal value for an abstract bucket.
///
/// For fold terminals the payoff is deterministic. For showdown
/// terminals we approximate equity from the river bucket index:
/// `equity ~ (bucket + 0.5) / river_bucket_count`.
fn terminal_bucket_value(
    kind: TerminalKind,
    invested: &[f64; 2],
    blinds: &[f64; 2],
    player: u8,
    bucket: u16,
    bucket_counts: [u16; 4],
) -> f32 {
    let p = player as usize;
    let o = 1 - p;
    let initial_pot = blinds[0] + blinds[1];
    // invested is now voluntary-only (blinds already excluded).
    let vol_p = invested[p];
    let vol_o = invested[o];

    match kind {
        TerminalKind::Fold { winner } => {
            if winner == player {
                (initial_pot + vol_o) as f32
            } else {
                -(vol_p as f32)
            }
        }
        TerminalKind::Showdown => {
            let river_buckets = bucket_counts[3].max(1);
            let clamped_bucket = bucket.min(river_buckets - 1);
            let equity = (f64::from(clamped_bucket) + 0.5) / f64::from(river_buckets);

            // EV = equity * (initial_pot + vol_opponent) - (1 - equity) * vol_self
            let ev = equity * (initial_pot + vol_o) - (1.0 - equity) * vol_p;
            ev as f32
        }
        TerminalKind::DepthBoundary => {
            // Depth-boundary nodes are not reachable during CBV computation;
            // they only appear in subgame trees resolved by a separate solver.
            unreachable!("DepthBoundary should not be reached during CBV computation")
        }
    }
}

/// Build a [`BucketTransition`] from a pair of adjacent-street bucket files.
///
/// Counts how often each `(prev_bucket, next_bucket)` pair occurs across
/// all (board, combo) pairs, then normalizes each row to produce
/// probabilities. Zero-weight entries are omitted.
///
/// The counting logic follows the same approach as
/// [`cross_street_transition_matrix`](super::cluster_diagnostics::cross_street_transition_matrix):
/// when the from-file has 1 board and the to-file has many, every
/// to-board is paired with the single from-board; otherwise boards are
/// paired by index.
#[must_use]
pub fn build_transition_from_pair(
    from_bf: &super::bucket_file::BucketFile,
    to_bf: &super::bucket_file::BucketFile,
) -> BucketTransition {
    let from_k = from_bf.header.bucket_count as usize;
    let to_k = to_bf.header.bucket_count as usize;
    let from_boards = from_bf.header.board_count as usize;
    let to_boards = to_bf.header.board_count as usize;
    let combos = from_bf.header.combos_per_board as usize;

    // Dense count matrix: counts[from_bucket][to_bucket]
    let mut counts = vec![vec![0_u64; to_k]; from_k];

    #[allow(clippy::cast_possible_truncation)]
    if from_boards == 1 && to_boards > 1 {
        // Preflop->flop style: single from-board, many to-boards
        for combo_idx in 0..combos {
            let from_bucket = from_bf.get_bucket(0, combo_idx as u16) as usize;
            for board_idx in 0..to_boards {
                let to_bucket = to_bf.get_bucket(board_idx as u32, combo_idx as u16) as usize;
                if from_bucket < from_k && to_bucket < to_k {
                    counts[from_bucket][to_bucket] += 1;
                }
            }
        }
    } else {
        // Same-arity: pair boards by index (flop->turn, turn->river)
        let boards = from_boards.min(to_boards);
        #[allow(clippy::cast_possible_truncation)]
        for board_idx in 0..boards {
            for combo_idx in 0..combos {
                let from_bucket = from_bf.get_bucket(board_idx as u32, combo_idx as u16) as usize;
                let to_bucket = to_bf.get_bucket(board_idx as u32, combo_idx as u16) as usize;
                if from_bucket < from_k && to_bucket < to_k {
                    counts[from_bucket][to_bucket] += 1;
                }
            }
        }
    }

    // Normalize rows into sparse probability distributions
    let dist = counts
        .iter()
        .map(|row| {
            let total: u64 = row.iter().sum();
            if total == 0 {
                return Vec::new();
            }
            let inv = 1.0 / total as f32;
            row.iter()
                .enumerate()
                .filter(|&(_, &c)| c > 0)
                .map(|(bucket, &c)| (bucket as u16, c as f32 * inv))
                .collect()
        })
        .collect();

    BucketTransition { dist }
}

/// Build transition matrices for all street boundaries from bucket files.
///
/// Returns `[preflop->flop, flop->turn, turn->river]`.
/// Slots where either the from or to bucket file is missing are `None`.
#[must_use]
pub fn build_transitions_from_buckets(
    bucket_files: &[Option<super::bucket_file::BucketFile>; 4],
) -> [Option<BucketTransition>; 3] {
    let mut result: [Option<BucketTransition>; 3] = [None, None, None];
    // Street pairs: (from_idx, to_idx, result_slot)
    let pairs = [(0, 1, 0), (1, 2, 1), (2, 3, 2)];
    for (from_idx, to_idx, slot) in pairs {
        if let (Some(from_bf), Some(to_bf)) = (&bucket_files[from_idx], &bucket_files[to_idx]) {
            eprintln!(
                "  CBV: computing bucket transition {:?} -> {:?} ...",
                from_bf.header.street, to_bf.header.street,
            );
            result[slot] = Some(build_transition_from_pair(from_bf, to_bf));
        }
    }
    result
}

/// Build an empty [`CbvTable`] with the right dimensions for the given
/// chance nodes.
fn build_empty_table(
    chance_indices: &[u32],
    nodes: &[GameNode],
    bucket_counts: [u16; 4],
) -> CbvTable {
    let mut node_offsets = Vec::with_capacity(chance_indices.len());
    let mut buckets_per_node = Vec::with_capacity(chance_indices.len());
    let mut total = 0_usize;

    for &arena_idx in chance_indices {
        if let GameNode::Chance { next_street, .. } = &nodes[arena_idx as usize] {
            // The CBV at a chance node is indexed by the bucket of the
            // street that just ended (next_street - 1).
            let prev_street_idx = (*next_street as usize).saturating_sub(1);
            let buckets = bucket_counts[prev_street_idx];
            node_offsets.push(total);
            buckets_per_node.push(buckets);
            total += buckets as usize;
        }
    }

    CbvTable {
        values: vec![0.0; total],
        node_offsets,
        buckets_per_node,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_v2::bundle::BlueprintV2Strategy;
    use crate::blueprint_v2::game_tree::{GameNode, GameTree, TerminalKind, TreeAction};
    use crate::blueprint_v2::storage::BlueprintStorage;

    /// Build a minimal hand-crafted tree for testing:
    ///
    /// ```text
    ///   Root: Decision(player=0, street=Flop, [Check, Bet])
    ///     Check -> Decision(player=1, street=Flop, [Check, Bet])
    ///       Check -> Chance(next=Turn, child=...)
    ///         -> Terminal(Showdown, pot=2.0, invested=[1.0, 1.0])
    ///       Bet -> Decision(player=0, street=Flop, [Fold, Call])
    ///         Fold -> Terminal(Fold{winner=1}, pot=2.0, invested=[1.0, 1.0])
    ///         Call -> Chance(next=Turn, child=...)
    ///           -> Terminal(Showdown, pot=4.0, invested=[2.0, 2.0])
    ///     Bet -> Terminal(Fold{winner=0}, pot=2.0, invested=[1.0, 1.0])
    /// ```
    ///
    /// This gives us 2 Chance nodes to compute CBVs for.
    fn build_test_tree() -> GameTree {
        use crate::blueprint_v2::Street;

        let nodes = vec![
            // Node 0: Root decision (player 0, Flop)
            GameNode::Decision {
                player: 0,
                street: Street::Flop,
                actions: vec![TreeAction::Check, TreeAction::Bet(2.0)],
                children: vec![1, 7],
            },
            // Node 1: Player 1 decision after check (Flop)
            GameNode::Decision {
                player: 1,
                street: Street::Flop,
                actions: vec![TreeAction::Check, TreeAction::Bet(2.0)],
                children: vec![2, 4],
            },
            // Node 2: Chance (check-check -> Turn)
            GameNode::Chance {
                next_street: Street::Turn,
                child: 3,
            },
            // Node 3: Terminal showdown (pot=2, each invested 1)
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 2.0,
                invested: [1.0, 1.0],
            },
            // Node 4: Player 0 facing bet (Flop, Fold/Call)
            GameNode::Decision {
                player: 0,
                street: Street::Flop,
                actions: vec![TreeAction::Fold, TreeAction::Call],
                children: vec![5, 6],
            },
            // Node 5: Terminal fold (player 1 wins)
            GameNode::Terminal {
                kind: TerminalKind::Fold { winner: 1 },
                pot: 2.0,
                invested: [1.0, 1.0],
            },
            // Node 6: Chance (bet-call -> Turn)
            GameNode::Chance {
                next_street: Street::Turn,
                child: 8,
            },
            // Node 7: Terminal fold (player 0 wins after bet)
            GameNode::Terminal {
                kind: TerminalKind::Fold { winner: 0 },
                pot: 2.0,
                invested: [1.0, 1.0],
            },
            // Node 8: Terminal showdown (pot=4, each invested 2)
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 4.0,
                invested: [2.0, 2.0],
            },
        ];

        GameTree { nodes, root: 0, blinds: [0.0, 0.0] }
    }

    /// With uniform strategy (fresh storage), verify that the CBV
    /// table has the right shape and plausible values.
    #[test]
    fn cbv_shape_and_sign() {
        let tree = build_test_tree();
        let bucket_counts: [u16; 4] = [4, 4, 4, 4];
        let storage = BlueprintStorage::new(&tree, bucket_counts);
        let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

        let [p0_cbvs, p1_cbvs] = compute_cbvs(&strategy, &tree, bucket_counts);

        // 2 chance nodes in the tree (nodes 2 and 6).
        assert_eq!(p0_cbvs.num_boundary_nodes(), 2);
        assert_eq!(p1_cbvs.num_boundary_nodes(), 2);

        // Each chance node transitions flop->turn, so CBV buckets use
        // the flop bucket count (index 1) = 4.
        assert_eq!(p0_cbvs.buckets_per_node[0], 4);
        assert_eq!(p0_cbvs.buckets_per_node[1], 4);

        // Low buckets (low equity) should have lower values for
        // player 0, higher buckets should have higher values.
        let v0_low = p0_cbvs.lookup(0, 0);
        let v0_high = p0_cbvs.lookup(0, 3);
        assert!(
            v0_high > v0_low,
            "higher equity bucket should have higher EV: low={v0_low}, high={v0_high}"
        );
    }

    /// Manually compute CBVs for a 2-bucket case and verify exact
    /// values.
    #[test]
    fn cbv_manual_verification() {
        let tree = build_test_tree();
        let bucket_counts: [u16; 4] = [2, 2, 2, 2];
        let storage = BlueprintStorage::new(&tree, bucket_counts);
        let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

        let [p0_cbvs, _p1_cbvs] = compute_cbvs(&strategy, &tree, bucket_counts);

        // With 2 buckets and uniform strategy at all 3 decision nodes:
        //
        // Bucket 0: equity = 0.25, Bucket 1: equity = 0.75
        //
        // Node 3 (showdown, invested=[1,1]):
        //   p0 EV = eq*1 - (1-eq)*1 = 2*eq - 1
        //   bucket 0: -0.5, bucket 1: 0.5
        //
        // Node 8 (showdown, invested=[2,2]):
        //   p0 EV = eq*2 - (1-eq)*2 = 4*eq - 2
        //   bucket 0: -1.0, bucket 1: 1.0
        //
        // Chance node 2 (check-check -> node 3):
        //   CBV = showdown value at node 3
        //
        // Chance node 6 (bet-call -> node 8):
        //   CBV = showdown value at node 8

        let eps = 1e-4;

        assert!(
            (p0_cbvs.lookup(0, 0) - (-0.5)).abs() < eps,
            "chance0 bucket0: expected -0.5, got {}",
            p0_cbvs.lookup(0, 0)
        );
        assert!(
            (p0_cbvs.lookup(0, 1) - 0.5).abs() < eps,
            "chance0 bucket1: expected 0.5, got {}",
            p0_cbvs.lookup(0, 1)
        );

        assert!(
            (p0_cbvs.lookup(1, 0) - (-1.0)).abs() < eps,
            "chance1 bucket0: expected -1.0, got {}",
            p0_cbvs.lookup(1, 0)
        );
        assert!(
            (p0_cbvs.lookup(1, 1) - 1.0).abs() < eps,
            "chance1 bucket1: expected 1.0, got {}",
            p0_cbvs.lookup(1, 1)
        );
    }

    /// Verify that fold terminals produce correct values independent
    /// of bucket.
    #[test]
    fn fold_terminal_value_independent_of_bucket() {
        let bucket_counts: [u16; 4] = [4, 4, 4, 4];

        // Player 0 wins fold: gains opponent's investment.
        let v = terminal_bucket_value(
            TerminalKind::Fold { winner: 0 },
            &[1.0, 1.0],
            &[0.0, 0.0],
            0,
            0,
            bucket_counts,
        );
        assert!((v - 1.0).abs() < 1e-6);

        // Same value regardless of bucket.
        let v2 = terminal_bucket_value(
            TerminalKind::Fold { winner: 0 },
            &[1.0, 1.0],
            &[0.0, 0.0],
            0,
            3,
            bucket_counts,
        );
        assert!((v - v2).abs() < 1e-6);

        // Player 0 loses fold: loses own investment.
        let v3 = terminal_bucket_value(
            TerminalKind::Fold { winner: 1 },
            &[1.0, 1.0],
            &[0.0, 0.0],
            0,
            2,
            bucket_counts,
        );
        assert!((v3 - (-1.0)).abs() < 1e-6);
    }

    /// Verify bucket-equity mapping at showdown terminals.
    #[test]
    fn showdown_equity_midpoint() {
        // With 4 buckets: equities = 0.125, 0.375, 0.625, 0.875
        // EV = 2*equity - 1 when both invested 1.0.
        let bucket_counts: [u16; 4] = [4, 4, 4, 4];

        let v0 = terminal_bucket_value(
            TerminalKind::Showdown,
            &[1.0, 1.0],
            &[0.0, 0.0],
            0,
            0,
            bucket_counts,
        );
        let expected_0 = (2.0 * 0.125 - 1.0) as f32;
        assert!(
            (v0 - expected_0).abs() < 1e-5,
            "bucket 0: expected {expected_0}, got {v0}"
        );

        let v3 = terminal_bucket_value(
            TerminalKind::Showdown,
            &[1.0, 1.0],
            &[0.0, 0.0],
            0,
            3,
            bucket_counts,
        );
        let expected_3 = (2.0 * 0.875 - 1.0) as f32;
        assert!(
            (v3 - expected_3).abs() < 1e-5,
            "bucket 3: expected {expected_3}, got {v3}"
        );
    }

    /// With non-zero blinds and voluntary invested, verify terminal values.
    ///
    /// Scenario: blinds = [0.5, 1.0], invested (voluntary) = [2.0, 2.0].
    /// initial_pot = 1.5, vol_p0 = 2.0, vol_p1 = 2.0.
    /// Fold winner=0: p0 gains initial_pot + vol_opp = 1.5 + 2.0 = 3.5
    /// Fold winner=1: p0 loses vol_self = -2.0
    #[test]
    fn terminal_value_with_nonzero_blinds() {
        let bucket_counts: [u16; 4] = [4, 4, 4, 4];

        // Player 0 wins fold: gains dead money + opponent voluntary
        let v = terminal_bucket_value(
            TerminalKind::Fold { winner: 0 },
            &[2.0, 2.0],
            &[0.5, 1.0],
            0,
            0,
            bucket_counts,
        );
        assert!(
            (v - 3.5).abs() < 1e-5,
            "Fold win: expected 3.5 (initial_pot 1.5 + vol_opp 2.0), got {v}"
        );

        // Player 0 loses fold: loses own voluntary
        let v2 = terminal_bucket_value(
            TerminalKind::Fold { winner: 1 },
            &[2.0, 2.0],
            &[0.5, 1.0],
            0,
            0,
            bucket_counts,
        );
        assert!(
            (v2 - (-2.0)).abs() < 1e-5,
            "Fold loss: expected -2.0, got {v2}"
        );
    }

    /// Verify that with a non-trivial transition matrix, the Chance node
    /// CBVs are weighted averages of child values (not identity-mapped).
    ///
    /// Setup: 2 flop buckets, 2 turn buckets. Transition matrix:
    ///   flop bucket 0 -> turn bucket 0 (100%)
    ///   flop bucket 1 -> turn bucket 0 (50%) + turn bucket 1 (50%)
    ///
    /// The child (turn) subtree is a showdown terminal with invested=[1,1]:
    ///   turn bucket 0: equity = 0.25, EV = -0.5
    ///   turn bucket 1: equity = 0.75, EV =  0.5
    ///
    /// Expected CBVs for player 0:
    ///   flop bucket 0: 100% * (-0.5) = -0.5
    ///   flop bucket 1: 50% * (-0.5) + 50% * (0.5) = 0.0
    #[test]
    fn cbv_with_transition_matrix() {
        use crate::blueprint_v2::Street;

        // Minimal tree: Decision -> Chance -> Terminal(Showdown)
        let nodes = vec![
            // Node 0: Root decision (player 0, Flop, Check only)
            GameNode::Decision {
                player: 0,
                street: Street::Flop,
                actions: vec![TreeAction::Check],
                children: vec![1],
            },
            // Node 1: Chance (flop -> turn)
            GameNode::Chance {
                next_street: Street::Turn,
                child: 2,
            },
            // Node 2: Terminal showdown (invested=[1,1])
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 2.0,
                invested: [1.0, 1.0],
            },
        ];
        let tree = GameTree { nodes, root: 0, blinds: [0.0, 0.0] };
        let bucket_counts: [u16; 4] = [2, 2, 2, 2];
        let storage = BlueprintStorage::new(&tree, bucket_counts);
        let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

        // Build transition matrix: flop(1) -> turn(2) = slot 1
        // flop bucket 0 -> turn bucket 0 with probability 1.0
        // flop bucket 1 -> turn bucket 0 with probability 0.5, turn bucket 1 with 0.5
        let mut transitions: [Option<BucketTransition>; 3] = [None, None, None];
        transitions[1] = Some(BucketTransition {
            dist: vec![
                vec![(0, 1.0)],                   // flop bucket 0 -> turn 0
                vec![(0, 0.5), (1, 0.5)],         // flop bucket 1 -> turn 0 (50%) + turn 1 (50%)
            ],
        });

        let [p0_cbvs, _] = compute_cbvs_with_transitions(
            &strategy, &tree, bucket_counts, &transitions,
        );

        let eps = 1e-4;

        // flop bucket 0: maps 100% to turn bucket 0
        // turn bucket 0: equity = 0.25, EV = 2*0.25 - 1 = -0.5
        assert!(
            (p0_cbvs.lookup(0, 0) - (-0.5)).abs() < eps,
            "flop bucket 0: expected -0.5, got {}",
            p0_cbvs.lookup(0, 0)
        );

        // flop bucket 1: maps 50% to turn bucket 0 (-0.5) + 50% to turn bucket 1 (0.5)
        // expected: 0.5 * (-0.5) + 0.5 * (0.5) = 0.0
        assert!(
            (p0_cbvs.lookup(0, 1) - 0.0).abs() < eps,
            "flop bucket 1: expected 0.0, got {}",
            p0_cbvs.lookup(0, 1)
        );
    }

    /// When all prev-street buckets map 100% to the same next-street
    /// bucket, every prev-street bucket gets the same CBV.
    #[test]
    fn cbv_transition_concentrated_on_one_next_bucket() {
        use crate::blueprint_v2::Street;

        let nodes = vec![
            GameNode::Decision {
                player: 0,
                street: Street::Flop,
                actions: vec![TreeAction::Check],
                children: vec![1],
            },
            GameNode::Chance {
                next_street: Street::Turn,
                child: 2,
            },
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 2.0,
                invested: [1.0, 1.0],
            },
        ];
        let tree = GameTree { nodes, root: 0, blinds: [0.0, 0.0] };
        // 3 flop buckets, 4 turn buckets
        let bucket_counts: [u16; 4] = [3, 3, 4, 4];
        let storage = BlueprintStorage::new(&tree, bucket_counts);
        let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

        // All 3 flop buckets map 100% to turn bucket 2
        let mut transitions: [Option<BucketTransition>; 3] = [None, None, None];
        transitions[1] = Some(BucketTransition {
            dist: vec![
                vec![(2, 1.0)],  // flop 0 -> turn 2
                vec![(2, 1.0)],  // flop 1 -> turn 2
                vec![(2, 1.0)],  // flop 2 -> turn 2
            ],
        });

        let [p0_cbvs, _] = compute_cbvs_with_transitions(
            &strategy, &tree, bucket_counts, &transitions,
        );

        // All flop buckets should get the same value: turn bucket 2's value
        // turn bucket 2: equity = (2 + 0.5) / 4 = 0.625
        // EV = 2 * 0.625 - 1 = 0.25
        let eps = 1e-4;
        let expected = (2.0 * 0.625 - 1.0) as f32;
        for b in 0..3 {
            assert!(
                (p0_cbvs.lookup(0, b) - expected).abs() < eps,
                "flop bucket {b}: expected {expected}, got {}",
                p0_cbvs.lookup(0, b)
            );
        }
    }

    /// Test build_transitions_from_buckets with synthetic bucket files.
    #[test]
    fn build_transitions_from_bucket_files() {
        use crate::blueprint_v2::bucket_file::{BucketFile, BucketFileHeader, PackedBoard};
        use crate::blueprint_v2::Street;

        // Create a flop bucket file with 1 board, 4 combos, 2 buckets
        // Combo 0,1 -> bucket 0; Combo 2,3 -> bucket 1
        let flop_bf = BucketFile {
            header: BucketFileHeader {
                street: Street::Flop,
                bucket_count: 2,
                board_count: 1,
                combos_per_board: 4,
                version: 2,
            },
            boards: vec![PackedBoard(0x1000)],
            buckets: vec![0, 0, 1, 1],
        };

        // Create a turn bucket file with 2 boards (the flop board + 2 possible turn cards),
        // 4 combos each, 2 buckets.
        // Board 0: same flop + turn card A -> combos map to [0, 1, 0, 1]
        // Board 1: same flop + turn card B -> combos map to [1, 1, 0, 0]
        let turn_bf = BucketFile {
            header: BucketFileHeader {
                street: Street::Turn,
                bucket_count: 2,
                board_count: 2,
                combos_per_board: 4,
                version: 2,
            },
            boards: vec![PackedBoard(0x1000_0100), PackedBoard(0x1000_0200)],
            buckets: vec![
                0, 1, 0, 1,  // board 0 combos
                1, 1, 0, 0,  // board 1 combos
            ],
        };

        let transitions = build_transition_from_pair(&flop_bf, &turn_bf);

        // Flop bucket 0 contains combos 0,1 (across 1 flop board).
        // For each turn board (2 boards):
        //   Board 0: combo 0 -> turn bucket 0, combo 1 -> turn bucket 1
        //   Board 1: combo 0 -> turn bucket 1, combo 1 -> turn bucket 1
        // Total: turn bucket 0 = 1, turn bucket 1 = 3
        // Probabilities: 0.25, 0.75

        // Flop bucket 1 contains combos 2,3 (across 1 flop board).
        // For each turn board:
        //   Board 0: combo 2 -> turn bucket 0, combo 3 -> turn bucket 1
        //   Board 1: combo 2 -> turn bucket 0, combo 3 -> turn bucket 0
        // Total: turn bucket 0 = 3, turn bucket 1 = 1
        // Probabilities: 0.75, 0.25

        assert_eq!(transitions.dist.len(), 2);

        let eps = 1e-4;
        // Check flop bucket 0 transitions
        let row0 = &transitions.dist[0];
        let p0_to_0 = row0.iter().find(|&&(b, _)| b == 0).map(|&(_, w)| w).unwrap_or(0.0);
        let p0_to_1 = row0.iter().find(|&&(b, _)| b == 1).map(|&(_, w)| w).unwrap_or(0.0);
        assert!((p0_to_0 - 0.25).abs() < eps, "flop0->turn0: expected 0.25, got {p0_to_0}");
        assert!((p0_to_1 - 0.75).abs() < eps, "flop0->turn1: expected 0.75, got {p0_to_1}");

        // Check flop bucket 1 transitions
        let row1 = &transitions.dist[1];
        let p1_to_0 = row1.iter().find(|&&(b, _)| b == 0).map(|&(_, w)| w).unwrap_or(0.0);
        let p1_to_1 = row1.iter().find(|&&(b, _)| b == 1).map(|&(_, w)| w).unwrap_or(0.0);
        assert!((p1_to_0 - 0.75).abs() < eps, "flop1->turn0: expected 0.75, got {p1_to_0}");
        assert!((p1_to_1 - 0.25).abs() < eps, "flop1->turn1: expected 0.25, got {p1_to_1}");
    }

    /// A real-ish tree from [`GameTree::build`] — verify it doesn't
    /// panic and produces the right number of boundary nodes.
    #[test]
    fn cbv_on_real_tree() {
        let tree = GameTree::build(
            10.0,
            0.5,
            1.0,
            &[vec!["2.5bb".into()]],
            &[vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
        );
        let bucket_counts: [u16; 4] = [10, 10, 10, 10];
        let storage = BlueprintStorage::new(&tree, bucket_counts);
        let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

        let [p0_cbvs, p1_cbvs] = compute_cbvs(&strategy, &tree, bucket_counts);

        let chance_count = tree
            .nodes
            .iter()
            .filter(|n| matches!(n, GameNode::Chance { .. }))
            .count();

        assert_eq!(p0_cbvs.num_boundary_nodes(), chance_count);
        assert_eq!(p1_cbvs.num_boundary_nodes(), chance_count);

        for v in &p0_cbvs.values {
            assert!(v.is_finite(), "CBV value should be finite, got {v}");
        }
        for v in &p1_cbvs.values {
            assert!(v.is_finite(), "CBV value should be finite, got {v}");
        }
    }
}
