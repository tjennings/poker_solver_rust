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
        GameNode::Terminal { kind, invested, .. } => {
            terminal_bucket_value(*kind, invested, &ctx.tree.blinds, ctx.player, bucket, ctx.bucket_counts)
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
