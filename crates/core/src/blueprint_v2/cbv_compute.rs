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

use std::collections::HashMap;

use super::cbv::CbvTable;
use crate::blueprint_v2::bundle::BlueprintV2Strategy;
use crate::blueprint_v2::game_tree::{GameNode, GameTree, TerminalKind};
use crate::blueprint_v2::Street;

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
/// the abstract game tree.
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

    // Map from arena index → position in chance_indices (for fast lookup).
    let mut chance_pos: Vec<u32> = vec![u32::MAX; tree.nodes.len()];
    for (pos, &arena_idx) in chance_indices.iter().enumerate() {
        chance_pos[arena_idx as usize] = pos as u32;
    }

    let max_buckets = *bucket_counts.iter().max().unwrap_or(&1) as usize;

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

        for bucket in 0..max_buckets {
            compute_node_value(&ctx, bucket as u16, tree.root, table);
        }
    }

    tables
}

/// Recursive backward induction. Returns the expected value for
/// `ctx.player` holding abstract bucket `bucket` at `node_idx`.
///
/// At chance (street-boundary) nodes, the computed child value is
/// stored into the [`CbvTable`] for later retrieval by the subgame
/// solver.
fn compute_node_value(
    ctx: &TraversalCtx<'_>,
    bucket: u16,
    node_idx: u32,
    table: &mut CbvTable,
) -> f32 {
    match &ctx.tree.nodes[node_idx as usize] {
        GameNode::Terminal { kind, pot, .. } => {
            terminal_bucket_value(*kind, *pot, ctx.player, bucket, ctx.bucket_counts)
        }

        GameNode::Chance { next_street, child } => {
            // Clamp the bucket index to the next street's range.
            // This is an approximation — bucket mappings change across
            // streets, but a single traversal per bucket is the best
            // we can do without concrete cards.
            let next_buckets = ctx.bucket_counts[*next_street as usize];
            let child_bucket = bucket.min(next_buckets.saturating_sub(1));

            let child_value = compute_node_value(ctx, child_bucket, *child, table);

            // Store this value if the bucket is in range for this node.
            let pos = ctx.chance_pos[node_idx as usize];
            if pos != u32::MAX {
                let pos = pos as usize;
                let node_buckets = table.buckets_per_node[pos] as usize;
                if (bucket as usize) < node_buckets {
                    table.values[table.node_offsets[pos] + bucket as usize] = child_value;
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
                let child_value = compute_node_value(ctx, bucket, child_idx, table);
                value += p * child_value;
            }

            value
        }
    }
}

/// Approximate terminal value for an abstract bucket.
///
/// Pot-only payoff model: fold winner = pot, loser = 0, showdown uses
/// equity approximated from river bucket index:
/// `equity ~ (bucket + 0.5) / river_bucket_count`.
fn terminal_bucket_value(
    kind: TerminalKind,
    pot: f64,
    player: u8,
    bucket: u16,
    bucket_counts: [u16; 4],
) -> f32 {
    match kind {
        TerminalKind::Fold { winner } => {
            if winner == player {
                pot as f32
            } else {
                0.0
            }
        }
        TerminalKind::Showdown => {
            let river_buckets = bucket_counts[3].max(1);
            let clamped_bucket = bucket.min(river_buckets - 1);
            let equity = (f64::from(clamped_bucket) + 0.5) / f64::from(river_buckets);

            // Pot-only: payoff = equity * pot
            let ev = equity * pot;
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

    let mut counts = vec![vec![0_u64; to_k]; from_k];

    #[allow(clippy::cast_possible_truncation)]
    if from_boards == 1 && to_boards > 1 {
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

    let matrix = counts
        .iter()
        .map(|row| {
            let total: u64 = row.iter().sum();
            if total == 0 {
                return vec![0.0_f32; to_k];
            }
            let inv = 1.0 / total as f32;
            row.iter().map(|&c| c as f32 * inv).collect()
        })
        .collect();

    BucketTransition { matrix }
}

/// Build transition matrices for all street boundaries from bucket files.
#[must_use]
pub fn build_transitions_from_buckets(
    bucket_files: &[Option<super::bucket_file::BucketFile>; 4],
) -> [Option<BucketTransition>; 3] {
    let mut result: [Option<BucketTransition>; 3] = [None, None, None];
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

/// Row-normalized transition probability matrix between adjacent streets.
///
/// `matrix[prev_bucket][next_bucket]` gives the probability that a hand in
/// `prev_bucket` on the previous street transitions to `next_bucket` on the
/// next street. Each row sums to 1.0.
#[derive(Debug, Clone)]
pub struct BucketTransition {
    /// `matrix[from][to]` — probability of transitioning from bucket `from`
    /// to bucket `to`. Dimensions: `prev_buckets x next_buckets`.
    pub matrix: Vec<Vec<f32>>,
}

/// Compute CBV tables using bottom-up street-by-street backward induction
/// with bucket transition matrices.
///
/// Unlike [`compute_cbvs`], which uses bucket clamping at Chance nodes,
/// this function processes each street independently from deepest to
/// shallowest. At each street boundary, the transition matrix converts
/// per-next-street-bucket child values into per-prev-street-bucket
/// continuation values.
///
/// # Arguments
///
/// * `strategy` — The extracted blueprint strategy (action probabilities).
/// * `tree` — The abstract game tree.
/// * `bucket_counts` — Number of buckets per street `[preflop, flop, turn, river]`.
/// * `transitions` — Transition matrices for each street boundary:
///   `[preflop->flop, flop->turn, turn->river]`. `None` entries use identity
///   mapping (same as clamping).
///
/// Returns `[player_0_cbvs, player_1_cbvs]`.
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

    // Map from arena index -> position in chance_indices.
    let mut chance_pos: Vec<u32> = vec![u32::MAX; tree.nodes.len()];
    for (pos, &arena_idx) in chance_indices.iter().enumerate() {
        chance_pos[arena_idx as usize] = pos as u32;
    }

    // Group chance nodes by the transition they represent.
    // transition_slot: 0 = preflop->flop, 1 = flop->turn, 2 = turn->river
    let mut chance_nodes_by_slot: [Vec<u32>; 3] = [vec![], vec![], vec![]];
    for &arena_idx in &chance_indices {
        if let GameNode::Chance { next_street, .. } = &tree.nodes[arena_idx as usize] {
            let slot = match next_street {
                Street::Flop => 0,
                Street::Turn => 1,
                Street::River => 2,
                Street::Preflop => continue, // shouldn't happen
            };
            chance_nodes_by_slot[slot].push(arena_idx);
        }
    }

    for player in 0..2u8 {
        let ctx = TraversalCtx {
            tree,
            strategy,
            decision_idx_map: &decision_idx_map,
            bucket_counts,
            chance_pos: &chance_pos,
            player,
        };

        // Process streets bottom-up: River(slot 2) -> Turn(slot 1) -> Flop(slot 0)
        // `continuation_values` maps arena_idx -> per-prev-bucket values
        // (used as terminal values when the parent street traverses)
        let mut chance_continuation: HashMap<u32, Vec<f32>> = HashMap::new();

        for slot in (0..3).rev() {
            let chance_nodes = &chance_nodes_by_slot[slot];
            if chance_nodes.is_empty() {
                continue;
            }

            let next_street_idx = slot + 1; // flop=1, turn=2, river=3
            let next_buckets = bucket_counts[next_street_idx] as usize;
            let prev_street_idx = slot; // preflop=0, flop=1, turn=2
            let prev_buckets = bucket_counts[prev_street_idx] as usize;

            // Phase 1: For each chance node at this slot, compute child
            // values for ALL next-street buckets.
            for &chance_arena_idx in chance_nodes {
                let child_idx = match &tree.nodes[chance_arena_idx as usize] {
                    GameNode::Chance { child, .. } => *child,
                    _ => unreachable!(),
                };

                let child_values: Vec<f32> = (0..next_buckets)
                    .map(|next_bucket| {
                        compute_street_values(
                            &ctx,
                            next_bucket as u16,
                            child_idx,
                            &chance_continuation,
                        )
                    })
                    .collect();

                // Phase 2: Apply transition matrix to convert next-street
                // bucket values to prev-street bucket values.
                let prev_values = apply_transition(
                    &child_values,
                    transitions[slot].as_ref(),
                    prev_buckets,
                );

                // Store in CBV table
                let pos = chance_pos[chance_arena_idx as usize];
                if pos != u32::MAX {
                    let pos = pos as usize;
                    let table = &mut tables[player as usize];
                    let node_buckets = table.buckets_per_node[pos] as usize;
                    let offset = table.node_offsets[pos];
                    for (b, &val) in prev_values.iter().enumerate().take(prev_buckets.min(node_buckets)) {
                        table.values[offset + b] = val;
                    }
                }

                // Store continuation values for parent street traversal
                chance_continuation.insert(chance_arena_idx, prev_values);
            }
        }
    }

    tables
}

/// Traverse within a single street, stopping at Chance nodes (treating
/// them as terminals with precomputed continuation values).
///
/// Returns the expected value for `ctx.player` holding abstract bucket
/// `bucket` at `node_idx`.
fn compute_street_values(
    ctx: &TraversalCtx<'_>,
    bucket: u16,
    node_idx: u32,
    chance_continuation: &HashMap<u32, Vec<f32>>,
) -> f32 {
    match &ctx.tree.nodes[node_idx as usize] {
        GameNode::Terminal { kind, pot, .. } => {
            terminal_bucket_value(*kind, *pot, ctx.player, bucket, ctx.bucket_counts)
        }

        GameNode::Chance { .. } => {
            // Use precomputed continuation value from deeper street pass.
            // Look up by bucket index in the continuation values.
            if let Some(values) = chance_continuation.get(&node_idx) {
                let b = (bucket as usize).min(values.len().saturating_sub(1));
                values[b]
            } else {
                // No continuation computed yet (shouldn't happen with
                // bottom-up processing, but fall back to 0).
                0.0
            }
        }

        GameNode::Decision {
            street, children, ..
        } => {
            let num_actions = children.len();
            let decision_idx = ctx.decision_idx_map[node_idx as usize];
            if decision_idx == u32::MAX {
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
                let child_value = compute_street_values(ctx, bucket, child_idx, chance_continuation);
                value += p * child_value;
            }

            value
        }
    }
}

/// Apply a transition matrix to convert next-street bucket values to
/// prev-street bucket values.
///
/// For each prev-street bucket `b`, the result is:
/// `sum_j(transition[b][j] * child_values[j])`
///
/// If no transition is provided, uses identity/clamping: prev bucket `b`
/// maps to next bucket `min(b, next_buckets-1)`.
fn apply_transition(
    child_values: &[f32],
    transition: Option<&BucketTransition>,
    prev_buckets: usize,
) -> Vec<f32> {
    let next_buckets = child_values.len();
    let mut result = vec![0.0_f32; prev_buckets];

    match transition {
        Some(trans) => {
            for (b, row) in trans.matrix.iter().enumerate().take(prev_buckets) {
                let mut val = 0.0_f32;
                for (j, &prob) in row.iter().enumerate().take(next_buckets) {
                    val += prob * child_values[j];
                }
                result[b] = val;
            }
        }
        None => {
            // Identity/clamping: each prev bucket maps to same-index next bucket
            for (b, v) in result.iter_mut().enumerate() {
                let clamped = b.min(next_buckets.saturating_sub(1));
                *v = child_values[clamped];
            }
        }
    }

    result
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
    ///         -> Terminal(Showdown, pot=2.0)
    ///       Bet -> Decision(player=0, street=Flop, [Fold, Call])
    ///         Fold -> Terminal(Fold{winner=1}, pot=2.0)
    ///         Call -> Chance(next=Turn, child=...)
    ///           -> Terminal(Showdown, pot=4.0)
    ///     Bet -> Terminal(Fold{winner=0}, pot=2.0)
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
                blueprint_decision_idx: None,
            },
            // Node 1: Player 1 decision after check (Flop)
            GameNode::Decision {
                player: 1,
                street: Street::Flop,
                actions: vec![TreeAction::Check, TreeAction::Bet(2.0)],
                children: vec![2, 4],
                blueprint_decision_idx: None,
            },
            // Node 2: Chance (check-check -> Turn)
            GameNode::Chance {
                next_street: Street::Turn,
                child: 3,
            },
            // Node 3: Terminal showdown (pot=2)
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 2.0,
                stacks: [49.0, 49.0],
            },
            // Node 4: Player 0 facing bet (Flop, Fold/Call)
            GameNode::Decision {
                player: 0,
                street: Street::Flop,
                actions: vec![TreeAction::Fold, TreeAction::Call],
                children: vec![5, 6],
                blueprint_decision_idx: None,
            },
            // Node 5: Terminal fold (player 1 wins)
            GameNode::Terminal {
                kind: TerminalKind::Fold { winner: 1 },
                pot: 2.0,
                stacks: [49.0, 49.0],
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
                stacks: [49.0, 49.0],
            },
            // Node 8: Terminal showdown (pot=4)
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 4.0,
                stacks: [48.0, 48.0],
            },
        ];

        GameTree { nodes, root: 0, dealer: 0, starting_stack: 50.0 }
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
        // Pot-only model:
        // Node 3 (showdown, pot=2.0): EV = equity * 2.0
        //   bucket 0: 0.5, bucket 1: 1.5
        //
        // Node 8 (showdown, pot=4.0): EV = equity * 4.0
        //   bucket 0: 1.0, bucket 1: 3.0
        //
        // Chance node 2 (check-check -> node 3):
        //   CBV = showdown value at node 3
        //
        // Chance node 6 (bet-call -> node 8):
        //   CBV = showdown value at node 8

        let eps = 1e-4;

        assert!(
            (p0_cbvs.lookup(0, 0) - 0.5).abs() < eps,
            "chance0 bucket0: expected 0.5, got {}",
            p0_cbvs.lookup(0, 0)
        );
        assert!(
            (p0_cbvs.lookup(0, 1) - 1.5).abs() < eps,
            "chance0 bucket1: expected 1.5, got {}",
            p0_cbvs.lookup(0, 1)
        );

        assert!(
            (p0_cbvs.lookup(1, 0) - 1.0).abs() < eps,
            "chance1 bucket0: expected 1.0, got {}",
            p0_cbvs.lookup(1, 0)
        );
        assert!(
            (p0_cbvs.lookup(1, 1) - 3.0).abs() < eps,
            "chance1 bucket1: expected 3.0, got {}",
            p0_cbvs.lookup(1, 1)
        );
    }

    /// Verify that fold terminals produce correct values independent
    /// of bucket.
    #[test]
    fn fold_terminal_value_independent_of_bucket() {
        let bucket_counts: [u16; 4] = [4, 4, 4, 4];

        // Player 0 wins fold: pot-only model gives pot.
        let v = terminal_bucket_value(
            TerminalKind::Fold { winner: 0 },
            2.0,
            0,
            0,
            bucket_counts,
        );
        assert!((v - 2.0).abs() < 1e-6);

        // Same value regardless of bucket.
        let v2 = terminal_bucket_value(
            TerminalKind::Fold { winner: 0 },
            2.0,
            0,
            3,
            bucket_counts,
        );
        assert!((v - v2).abs() < 1e-6);

        // Player 0 loses fold: pot-only model gives 0.
        let v3 = terminal_bucket_value(
            TerminalKind::Fold { winner: 1 },
            2.0,
            0,
            2,
            bucket_counts,
        );
        assert!(v3.abs() < 1e-6);
    }

    /// Verify bucket-equity mapping at showdown terminals.
    #[test]
    fn showdown_equity_midpoint() {
        // With 4 buckets: equities = 0.125, 0.375, 0.625, 0.875
        // Pot-only: EV = equity * pot. With pot=2.0:
        let bucket_counts: [u16; 4] = [4, 4, 4, 4];

        let v0 = terminal_bucket_value(
            TerminalKind::Showdown,
            2.0,
            0,
            0,
            bucket_counts,
        );
        let expected_0 = (0.125 * 2.0) as f32;
        assert!(
            (v0 - expected_0).abs() < 1e-5,
            "bucket 0: expected {expected_0}, got {v0}"
        );

        let v3 = terminal_bucket_value(
            TerminalKind::Showdown,
            2.0,
            0,
            3,
            bucket_counts,
        );
        let expected_3 = (0.875 * 2.0) as f32;
        assert!(
            (v3 - expected_3).abs() < 1e-5,
            "bucket 3: expected {expected_3}, got {v3}"
        );
    }

    /// Build a 2-street tree (flop decisions -> turn showdown) for
    /// testing transition-based CBV computation.
    ///
    /// ```text
    ///   Root: Decision(player=0, street=Flop, [Check, Bet])
    ///     Check -> Decision(player=1, street=Flop, [Check])
    ///       Check -> Chance(next=Turn, child=...)
    ///         -> Decision(player=0, street=Turn, [Check])
    ///           Check -> Decision(player=1, street=Turn, [Check])
    ///             Check -> Terminal(Showdown, pot=2.0)
    ///     Bet -> Terminal(Fold{winner=0}, pot=2.0)
    /// ```
    fn build_two_street_tree() -> GameTree {
        use crate::blueprint_v2::Street;

        let nodes = vec![
            // 0: Root (P0, Flop, Check/Bet)
            GameNode::Decision {
                player: 0,
                street: Street::Flop,
                actions: vec![TreeAction::Check, TreeAction::Bet(2.0)],
                children: vec![1, 8],
                blueprint_decision_idx: None,
            },
            // 1: P1 Flop (Check only - keeps it simple)
            GameNode::Decision {
                player: 1,
                street: Street::Flop,
                actions: vec![TreeAction::Check],
                children: vec![2],
            blueprint_decision_idx: None,
            },
            // 2: Chance (flop->turn)
            GameNode::Chance {
                next_street: Street::Turn,
                child: 3,
            },
            // 3: P0 Turn (Check only)
            GameNode::Decision {
                player: 0,
                street: Street::Turn,
                actions: vec![TreeAction::Check],
                children: vec![4],
            blueprint_decision_idx: None,
            },
            // 4: P1 Turn (Check only)
            GameNode::Decision {
                player: 1,
                street: Street::Turn,
                actions: vec![TreeAction::Check],
                children: vec![5],
            blueprint_decision_idx: None,
            },
            // 5: Chance (turn->river)
            GameNode::Chance {
                next_street: Street::River,
                child: 6,
            },
            // 6: Terminal showdown
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 2.0,
                stacks: [49.0, 49.0],
            },
            // 7: (unused slot for alignment)
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 2.0,
                stacks: [49.0, 49.0],
            },
            // 8: Fold terminal (P0 wins)
            GameNode::Terminal {
                kind: TerminalKind::Fold { winner: 0 },
                pot: 2.0,
                stacks: [49.0, 49.0],
            },
        ];

        GameTree { nodes, root: 0, dealer: 0, starting_stack: 50.0 }
    }

    /// Identity transition (each bucket maps 100% to same bucket on
    /// next street) should produce the same results as no transition.
    #[test]
    fn cbv_with_identity_transition_matches_no_transition() {
        let tree = build_test_tree();
        let bucket_counts: [u16; 4] = [2, 2, 2, 2];
        let storage = BlueprintStorage::new(&tree, bucket_counts);
        let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

        let no_transition = compute_cbvs(&strategy, &tree, bucket_counts);

        // Identity transition: bucket i -> bucket i with probability 1.0
        let identity_2x2 = BucketTransition {
            matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        // Transition slots: [preflop->flop, flop->turn, turn->river]
        let transitions: [Option<BucketTransition>; 3] = [
            None,
            Some(identity_2x2.clone()),
            None,
        ];

        let with_transition =
            compute_cbvs_with_transitions(&strategy, &tree, bucket_counts, &transitions);

        let eps = 1e-4;
        for player in 0..2 {
            for node in 0..no_transition[player].num_boundary_nodes() {
                for bucket in 0..no_transition[player].buckets_per_node[node] as usize {
                    let expected = no_transition[player].lookup(node, bucket);
                    let actual = with_transition[player].lookup(node, bucket);
                    assert!(
                        (expected - actual).abs() < eps,
                        "player {player} node {node} bucket {bucket}: \
                         expected {expected}, got {actual}"
                    );
                }
            }
        }
    }

    /// Test that transition matrix properly mixes bucket values.
    ///
    /// With a known transition and known terminal values, verify exact
    /// CBV outputs.
    #[test]
    fn cbv_transition_mixes_bucket_values() {
        use crate::blueprint_v2::Street;

        // Simple tree: single flop decision -> chance -> single turn
        // decision -> showdown. This tests one flop->turn transition.
        //
        //   0: Decision(P0, Flop, [Check])
        //     1: Decision(P1, Flop, [Check])
        //       2: Chance(next=Turn)
        //         3: Terminal(Showdown, pot=2.0)
        let nodes = vec![
            GameNode::Decision {
                player: 0,
                street: Street::Flop,
                actions: vec![TreeAction::Check],
                children: vec![1],
            blueprint_decision_idx: None,
            },
            GameNode::Decision {
                player: 1,
                street: Street::Flop,
                actions: vec![TreeAction::Check],
                children: vec![2],
            blueprint_decision_idx: None,
            },
            GameNode::Chance {
                next_street: Street::Turn,
                child: 3,
            },
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 2.0,
                stacks: [49.0, 49.0],
            },
        ];
        let tree = GameTree { nodes, root: 0, dealer: 0, starting_stack: 50.0 };
        let bucket_counts: [u16; 4] = [2, 2, 2, 2];
        let storage = BlueprintStorage::new(&tree, bucket_counts);
        let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

        // Turn has 2 buckets: equity = (b+0.5)/2
        //   bucket 0: equity = 0.25, EV = 2*0.25-1 = -0.5
        //   bucket 1: equity = 0.75, EV = 2*0.75-1 =  0.5
        //
        // Transition matrix (flop bucket -> turn bucket probabilities):
        //   flop bucket 0 -> [0.8 turn_0, 0.2 turn_1]
        //   flop bucket 1 -> [0.3 turn_0, 0.7 turn_1]
        //
        // Pot-only: turn bucket 0: eq=0.25, EV=0.5; bucket 1: eq=0.75, EV=1.5
        // Expected CBV at chance node 2:
        //   flop bucket 0: 0.8 * 0.5 + 0.2 * 1.5 = 0.4 + 0.3 = 0.7
        //   flop bucket 1: 0.3 * 0.5 + 0.7 * 1.5 = 0.15 + 1.05 = 1.2
        let transition = BucketTransition {
            matrix: vec![vec![0.8, 0.2], vec![0.3, 0.7]],
        };
        let transitions: [Option<BucketTransition>; 3] = [
            None,
            Some(transition),
            None,
        ];

        let [p0_cbvs, _] =
            compute_cbvs_with_transitions(&strategy, &tree, bucket_counts, &transitions);

        let eps = 1e-4;
        assert!(
            (p0_cbvs.lookup(0, 0) - 0.7).abs() < eps,
            "flop bucket 0: expected 0.7, got {}",
            p0_cbvs.lookup(0, 0)
        );
        assert!(
            (p0_cbvs.lookup(0, 1) - 1.2).abs() < eps,
            "flop bucket 1: expected 1.2, got {}",
            p0_cbvs.lookup(0, 1)
        );
    }

    /// Test bottom-up processing with 2 chance nodes at different
    /// street levels (flop->turn and turn->river).
    #[test]
    fn cbv_multi_street_transitions() {
        let tree = build_two_street_tree();
        let bucket_counts: [u16; 4] = [2, 2, 2, 2];
        let storage = BlueprintStorage::new(&tree, bucket_counts);
        let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

        // Pot-only: River has 2 buckets: equity = (b+0.5)/2
        //   bucket 0: eq=0.25, EV = 0.5
        //   bucket 1: eq=0.75, EV = 1.5
        //
        // Turn->river transition:
        //   turn bucket 0 -> [0.9 river_0, 0.1 river_1]
        //   turn bucket 1 -> [0.2 river_0, 0.8 river_1]
        //
        // After turn->river transition, continuation values at turn:
        //   turn bucket 0: 0.9*0.5 + 0.1*1.5 = 0.45 + 0.15 = 0.6
        //   turn bucket 1: 0.2*0.5 + 0.8*1.5 = 0.1 + 1.2 = 1.3
        //
        // Flop->turn transition:
        //   flop bucket 0 -> [0.7 turn_0, 0.3 turn_1]
        //   flop bucket 1 -> [0.1 turn_0, 0.9 turn_1]
        //
        // After flop->turn transition, CBV at flop->turn chance node:
        //   flop bucket 0: 0.7*0.6 + 0.3*1.3 = 0.42 + 0.39 = 0.81
        //   flop bucket 1: 0.1*0.6 + 0.9*1.3 = 0.06 + 1.17 = 1.23

        let turn_river = BucketTransition {
            matrix: vec![vec![0.9, 0.1], vec![0.2, 0.8]],
        };
        let flop_turn = BucketTransition {
            matrix: vec![vec![0.7, 0.3], vec![0.1, 0.9]],
        };
        let transitions: [Option<BucketTransition>; 3] = [
            None,
            Some(flop_turn),
            Some(turn_river),
        ];

        let [p0_cbvs, _] =
            compute_cbvs_with_transitions(&strategy, &tree, bucket_counts, &transitions);

        // There are 2 chance nodes: node 2 (flop->turn) and node 5 (turn->river).
        assert_eq!(p0_cbvs.num_boundary_nodes(), 2);

        let eps = 1e-4;

        // Chance node for flop->turn (node 2, position 0 in table):
        // CBV indexed by flop buckets
        assert!(
            (p0_cbvs.lookup(0, 0) - 0.81).abs() < eps,
            "flop->turn, flop bucket 0: expected 0.81, got {}",
            p0_cbvs.lookup(0, 0)
        );
        assert!(
            (p0_cbvs.lookup(0, 1) - 1.23).abs() < eps,
            "flop->turn, flop bucket 1: expected 1.23, got {}",
            p0_cbvs.lookup(0, 1)
        );

        // Chance node for turn->river (node 5, position 1 in table):
        // CBV indexed by turn buckets
        assert!(
            (p0_cbvs.lookup(1, 0) - 0.6).abs() < eps,
            "turn->river, turn bucket 0: expected 0.6, got {}",
            p0_cbvs.lookup(1, 0)
        );
        assert!(
            (p0_cbvs.lookup(1, 1) - 1.3).abs() < eps,
            "turn->river, turn bucket 1: expected 1.3, got {}",
            p0_cbvs.lookup(1, 1)
        );
    }

    /// Verify that weak buckets get negative CBVs and strong buckets
    /// get positive CBVs at showdown terminals with transitions.
    #[test]
    fn cbv_transition_weak_negative_strong_positive() {
        let tree = build_test_tree();
        let bucket_counts: [u16; 4] = [2, 2, 2, 2];
        let storage = BlueprintStorage::new(&tree, bucket_counts);
        let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

        // Slight mixing: weak hands mostly stay weak, strong mostly stay strong
        let transition = BucketTransition {
            matrix: vec![vec![0.8, 0.2], vec![0.2, 0.8]],
        };
        let transitions: [Option<BucketTransition>; 3] = [
            None,
            Some(transition),
            None,
        ];

        let [p0_cbvs, _] =
            compute_cbvs_with_transitions(&strategy, &tree, bucket_counts, &transitions);

        // For both chance nodes, bucket 1 (strong) should exceed
        // bucket 0 (weak). With pot-only model both are positive.
        for node in 0..p0_cbvs.num_boundary_nodes() {
            let v0 = p0_cbvs.lookup(node, 0);
            let v1 = p0_cbvs.lookup(node, 1);
            assert!(
                v0 > 0.0,
                "node {node} bucket 0 (weak) should be positive with pot-only, got {v0}"
            );
            assert!(
                v1 > v0,
                "node {node} strong bucket should exceed weak: v0={v0}, v1={v1}"
            );
        }
    }

    /// None transitions should produce same result as `compute_cbvs`.
    #[test]
    fn cbv_no_transitions_matches_original() {
        let tree = build_test_tree();
        let bucket_counts: [u16; 4] = [2, 2, 2, 2];
        let storage = BlueprintStorage::new(&tree, bucket_counts);
        let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

        let original = compute_cbvs(&strategy, &tree, bucket_counts);
        let transitions: [Option<BucketTransition>; 3] = [None, None, None];
        let with_none =
            compute_cbvs_with_transitions(&strategy, &tree, bucket_counts, &transitions);

        let eps = 1e-4;
        for player in 0..2 {
            for node in 0..original[player].num_boundary_nodes() {
                for bucket in 0..original[player].buckets_per_node[node] as usize {
                    let expected = original[player].lookup(node, bucket);
                    let actual = with_none[player].lookup(node, bucket);
                    assert!(
                        (expected - actual).abs() < eps,
                        "player {player} node {node} bucket {bucket}: \
                         expected {expected}, got {actual}"
                    );
                }
            }
        }
    }

    /// Real tree with transitions should not panic and should produce
    /// finite values.
    #[test]
    fn cbv_real_tree_with_transitions_finite() {
        let tree = GameTree::build(
            10.0,
            0.5,
            1.0,
            &[vec!["2.5bb".into()]],
            &[vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
        );
        let bucket_counts: [u16; 4] = [4, 4, 4, 4];
        let storage = BlueprintStorage::new(&tree, bucket_counts);
        let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

        // 4x4 slight mixing transition
        let make_transition = |k: usize| -> BucketTransition {
            let mut matrix = vec![vec![0.0_f32; k]; k];
            for i in 0..k {
                for j in 0..k {
                    matrix[i][j] = if i == j { 0.7 } else { 0.1 / (k - 1) as f32 };
                }
                // Normalize
                let sum: f32 = matrix[i].iter().sum();
                for j in 0..k {
                    matrix[i][j] /= sum;
                }
            }
            BucketTransition { matrix }
        };

        let transitions: [Option<BucketTransition>; 3] = [
            Some(make_transition(4)),
            Some(make_transition(4)),
            Some(make_transition(4)),
        ];

        let [p0_cbvs, p1_cbvs] =
            compute_cbvs_with_transitions(&strategy, &tree, bucket_counts, &transitions);

        for v in &p0_cbvs.values {
            assert!(v.is_finite(), "P0 CBV should be finite, got {v}");
        }
        for v in &p1_cbvs.values {
            assert!(v.is_finite(), "P1 CBV should be finite, got {v}");
        }

        // Should have same number of boundary nodes as without transitions
        let chance_count = tree
            .nodes
            .iter()
            .filter(|n| matches!(n, GameNode::Chance { .. }))
            .count();
        assert_eq!(p0_cbvs.num_boundary_nodes(), chance_count);
        assert_eq!(p1_cbvs.num_boundary_nodes(), chance_count);
    }

    /// Test with asymmetric bucket counts: 3 flop buckets, 2 turn buckets.
    /// Transition matrix is 3x2.
    #[test]
    fn cbv_transition_asymmetric_bucket_counts() {
        use crate::blueprint_v2::Street;

        // Simple tree: flop decision -> chance -> terminal showdown
        let nodes = vec![
            // 0: P0 Flop
            GameNode::Decision {
                player: 0,
                street: Street::Flop,
                actions: vec![TreeAction::Check],
                children: vec![1],
            blueprint_decision_idx: None,
            },
            // 1: P1 Flop
            GameNode::Decision {
                player: 1,
                street: Street::Flop,
                actions: vec![TreeAction::Check],
                children: vec![2],
            blueprint_decision_idx: None,
            },
            // 2: Chance (flop->turn)
            GameNode::Chance {
                next_street: Street::Turn,
                child: 3,
            },
            // 3: Terminal showdown
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 2.0,
                stacks: [49.0, 49.0],
            },
        ];
        let tree = GameTree { nodes, root: 0, dealer: 0, starting_stack: 50.0 };
        // 3 flop buckets, 2 turn buckets
        let bucket_counts: [u16; 4] = [3, 3, 2, 2];
        let storage = BlueprintStorage::new(&tree, bucket_counts);
        let strategy = BlueprintV2Strategy::from_storage(&storage, &tree);

        // Pot-only: Turn 2 buckets: equity = (b+0.5)/2
        //   bucket 0: eq=0.25, EV = 0.5
        //   bucket 1: eq=0.75, EV = 1.5
        //
        // 3x2 transition (flop 3 buckets -> turn 2 buckets):
        //   flop 0 -> [0.9, 0.1]  => 0.9*0.5 + 0.1*1.5 = 0.6
        //   flop 1 -> [0.5, 0.5]  => 0.5*0.5 + 0.5*1.5 = 1.0
        //   flop 2 -> [0.1, 0.9]  => 0.1*0.5 + 0.9*1.5 = 1.4
        let transition = BucketTransition {
            matrix: vec![
                vec![0.9, 0.1],
                vec![0.5, 0.5],
                vec![0.1, 0.9],
            ],
        };
        let transitions: [Option<BucketTransition>; 3] = [
            None,
            Some(transition),
            None,
        ];

        let [p0_cbvs, _] =
            compute_cbvs_with_transitions(&strategy, &tree, bucket_counts, &transitions);

        // CBV table indexed by flop buckets (3 buckets)
        assert_eq!(p0_cbvs.num_boundary_nodes(), 1);
        assert_eq!(p0_cbvs.buckets_per_node[0], 3);

        let eps = 1e-4;
        assert!(
            (p0_cbvs.lookup(0, 0) - 0.6).abs() < eps,
            "flop bucket 0: expected 0.6, got {}",
            p0_cbvs.lookup(0, 0)
        );
        assert!(
            (p0_cbvs.lookup(0, 1) - 1.0).abs() < eps,
            "flop bucket 1: expected 1.0, got {}",
            p0_cbvs.lookup(0, 1)
        );
        assert!(
            (p0_cbvs.lookup(0, 2) - 1.4).abs() < eps,
            "flop bucket 2: expected 1.4, got {}",
            p0_cbvs.lookup(0, 2)
        );
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
