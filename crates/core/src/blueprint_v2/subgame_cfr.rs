//! Subgame hand enumeration, equity computation, and boundary mapping.
//!
//! Provides the foundational types and utility functions used by subgame
//! solvers: combo enumeration, equity matrices, reach precomputation,
//! and lockstep tree walking for boundary mapping.

use rustc_hash::FxHashMap;

use crate::poker::{Card, Hand, Rank, Rankable};

use crate::blueprint_v2::game_tree::{GameNode, GameTree, TerminalKind, TreeAction};

// ---------------------------------------------------------------------------
// SubgameHands -- valid hole card combos for a specific board
// ---------------------------------------------------------------------------

/// All valid hole card combos for a specific board.
#[derive(Debug, Clone)]
pub struct SubgameHands {
    pub combos: Vec<[Card; 2]>,
}

impl SubgameHands {
    /// Enumerate all valid 2-card combos from the 52-card deck excluding board cards.
    #[must_use]
    pub fn enumerate(board: &[Card]) -> Self {
        let deck = remaining_deck(board);
        let mut combos = Vec::with_capacity(deck.len() * (deck.len() - 1) / 2);
        for i in 0..deck.len() {
            for j in (i + 1)..deck.len() {
                combos.push([deck[i], deck[j]]);
            }
        }
        Self { combos }
    }
}

fn remaining_deck(board: &[Card]) -> Vec<Card> {
    crate::poker::full_deck()
        .into_iter()
        .filter(|c| !board.contains(c))
        .collect()
}

// ---------------------------------------------------------------------------
// SubgameStrategy -- the output of a subgame solve
// ---------------------------------------------------------------------------

/// Result of a subgame solve: strategies for all combos at all decision nodes.
#[derive(Debug, Clone)]
pub struct SubgameStrategy {
    /// `(node_idx, combo_idx)` -> action probabilities.
    pub strategies: FxHashMap<u64, Vec<f64>>,
    /// Total number of combos in this subgame.
    pub num_combos: usize,
}

impl SubgameStrategy {
    /// Compute the flat key for `(node_idx, combo_idx)`.
    #[must_use]
    pub fn key(node_idx: u32, combo_idx: u32) -> u64 {
        (u64::from(node_idx) << 32) | u64::from(combo_idx)
    }

    /// Get action probabilities for a combo at a node.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn get_probs(&self, node_idx: u32, combo_idx: usize) -> Vec<f64> {
        self.strategies
            .get(&Self::key(node_idx, combo_idx as u32))
            .cloned()
            .unwrap_or_default()
    }

    /// Get root node probabilities for a combo.
    #[must_use]
    pub fn root_probs(&self, combo_idx: usize) -> Vec<f64> {
        self.get_probs(0, combo_idx)
    }

    /// Number of combos in this subgame.
    #[must_use]
    pub fn num_combos(&self) -> usize {
        self.num_combos
    }
}

// ---------------------------------------------------------------------------
// Precomputation helpers
// ---------------------------------------------------------------------------

/// Precompute total opponent reach for each combo (excluding blocked combos).
pub fn precompute_opp_reach(combos: &[[Card; 2]], opponent_reach: &[f64]) -> Vec<f64> {
    combos
        .iter()
        .map(|&hero| {
            opponent_reach
                .iter()
                .zip(combos)
                .filter(|(_, opp)| !cards_overlap(hero, **opp))
                .map(|(&r, _)| r)
                .sum()
        })
        .collect()
}

/// Build an N x N equity matrix for all combo pairs on the given board.
///
/// For 5-card boards (river), compares hand ranks directly.
/// For shorter boards (flop/turn), enumerates all possible runouts to
/// compute all-in equity that accounts for future streets.
pub fn compute_equity_matrix(combos: &[[Card; 2]], board: &[Card]) -> Vec<Vec<f64>> {
    let n = combos.len();

    if board.len() >= 5 {
        // River: direct rank comparison (no runouts needed).
        return compute_equity_matrix_direct(combos, board);
    }

    // Flop (3 cards) or Turn (4 cards): enumerate runouts.
    let cards_needed = 5 - board.len();
    let remaining = remaining_cards_for_runout(board);

    // Build all possible runouts.
    let runouts: Vec<Vec<Card>> = if cards_needed == 2 {
        // Flop: enumerate all C(remaining, 2) pairs.
        let mut runs = Vec::new();
        for i in 0..remaining.len() {
            for j in (i + 1)..remaining.len() {
                runs.push(vec![remaining[i], remaining[j]]);
            }
        }
        runs
    } else {
        // Turn: enumerate all remaining single cards.
        remaining.iter().map(|&c| vec![c]).collect()
    };

    let start = std::time::Instant::now();
    eprintln!(
        "[equity matrix] {} combos, {} board cards, {} runouts — enumerating all-in equity",
        n, board.len(), runouts.len()
    );

    // Accumulate wins/losses/ties per combo pair across all runouts.
    // Strategy: for each runout, rank ALL combos once (cheap), then do
    // pairwise comparisons using pre-ranked values (integer compare).
    // This avoids redundant hand ranking — O(runouts * n) rankings
    // instead of O(runouts * n^2).

    // wins[i][j] and total[i][j] for i < j only (upper triangle).
    let mut win_counts = vec![vec![0u32; n]; n];
    let mut tie_counts = vec![vec![0u32; n]; n];
    let mut total_counts = vec![vec![0u32; n]; n];

    // Precompute which cards each combo uses (for conflict checking).
    let combo_bits: Vec<u64> = combos
        .iter()
        .map(|c| (1u64 << card_bit_idx(c[0])) | (1u64 << card_bit_idx(c[1])))
        .collect();

    for runout in &runouts {
        let mut runout_bits = 0u64;
        for &c in runout.iter() {
            runout_bits |= 1u64 << card_bit_idx(c);
        }

        // Build full board for this runout.
        let mut full_board = board.to_vec();
        full_board.extend_from_slice(runout);

        // Rank all combos on this full board (skip conflicting ones).
        let ranks: Vec<Option<Rank>> = combos
            .iter()
            .zip(combo_bits.iter())
            .map(|(combo, &bits)| {
                if bits & runout_bits != 0 {
                    None // combo conflicts with runout cards
                } else {
                    Some(rank_combo(*combo, &full_board))
                }
            })
            .collect();

        // Pairwise comparisons using pre-ranked values.
        for i in 0..n {
            let Some(ref rank_i) = ranks[i] else { continue };
            for j in (i + 1)..n {
                if combo_bits[i] & combo_bits[j] != 0 {
                    continue; // combos share cards
                }
                let Some(ref rank_j) = ranks[j] else { continue };
                total_counts[i][j] += 1;
                match rank_i.cmp(rank_j) {
                    std::cmp::Ordering::Greater => win_counts[i][j] += 1,
                    std::cmp::Ordering::Equal => tie_counts[i][j] += 1,
                    std::cmp::Ordering::Less => {}
                }
            }
        }
    }

    // Convert counts to equity matrix.
    let mut matrix = vec![vec![0.5; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let total = total_counts[i][j];
            if total > 0 {
                let eq = (f64::from(win_counts[i][j]) + f64::from(tie_counts[i][j]) * 0.5)
                    / f64::from(total);
                matrix[i][j] = eq;
                matrix[j][i] = 1.0 - eq;
            }
        }
    }

    let elapsed = start.elapsed();
    eprintln!("[equity matrix] completed in {:.2}s", elapsed.as_secs_f64());

    matrix
}

/// Direct rank comparison for 5-card boards (no runout enumeration needed).
fn compute_equity_matrix_direct(combos: &[[Card; 2]], board: &[Card]) -> Vec<Vec<f64>> {
    let n = combos.len();
    let ranks: Vec<Rank> = combos.iter().map(|c| rank_combo(*c, board)).collect();
    let mut matrix = vec![vec![0.5; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            if cards_overlap(combos[i], combos[j]) {
                continue;
            }
            match ranks[i].cmp(&ranks[j]) {
                std::cmp::Ordering::Greater => {
                    matrix[i][j] = 1.0;
                    matrix[j][i] = 0.0;
                }
                std::cmp::Ordering::Less => {
                    matrix[i][j] = 0.0;
                    matrix[j][i] = 1.0;
                }
                std::cmp::Ordering::Equal => {}
            }
        }
    }
    matrix
}

/// Map a card to a bit index (0..51) for 64-bit bitset conflict checking.
fn card_bit_idx(card: Card) -> u32 {
    card.value as u32 * 4 + card.suit as u32
}

/// Collect remaining deck cards excluding the board (for runout enumeration).
fn remaining_cards_for_runout(board: &[Card]) -> Vec<Card> {
    crate::poker::full_deck()
        .into_iter()
        .filter(|c| !board.contains(c))
        .collect()
}

/// Rank a combo (2 hole cards) on a board using `rs_poker`.
fn rank_combo(combo: [Card; 2], board: &[Card]) -> Rank {
    let mut hand = Hand::default();
    for &c in board {
        hand.insert(c);
    }
    hand.insert(combo[0]);
    hand.insert(combo[1]);
    hand.rank()
}

/// Check if two 2-card combos share any card.
pub fn cards_overlap(a: [Card; 2], b: [Card; 2]) -> bool {
    a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1]
}

// ---------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------

/// Compute per-combo equity against an opponent range.
///
/// Returns a vector of equities (0.0 to 1.0) for each combo in `hands`,
/// where equity is the reach-weighted probability of beating the opponent.
/// Combos facing zero opponent reach get equity 0.5 (neutral).
#[must_use]
pub fn compute_combo_equities(
    hands: &SubgameHands,
    board: &[Card],
    opponent_reach: &[f64],
) -> Vec<f64> {
    let equity_matrix = compute_equity_matrix(&hands.combos, board);
    let n = hands.combos.len();
    let mut equities = vec![0.5; n];

    for i in 0..n {
        let mut eq_sum = 0.0;
        let mut reach_sum = 0.0;
        for (j, &opp_r) in opponent_reach.iter().enumerate() {
            if opp_r <= 0.0 || cards_overlap(hands.combos[i], hands.combos[j]) {
                continue;
            }
            eq_sum += opp_r * equity_matrix[i][j];
            reach_sum += opp_r;
        }
        if reach_sum > 0.0 {
            equities[i] = eq_sum / reach_sum;
        }
    }
    equities
}

// ---------------------------------------------------------------------------
// build_boundary_mapping -- parallel tree walk
// ---------------------------------------------------------------------------

/// Map each `DepthBoundary` in `subgame_tree` to the corresponding `Chance`
/// node ordinal in `abstract_tree` by walking both trees in lockstep.
///
/// Both trees must share the same action set at every decision node. Returns
/// a `Vec<usize>` of length equal to the number of `DepthBoundary` nodes,
/// where each value is the ordinal position of the matching `Chance` node
/// among all `Chance` nodes in the abstract tree (matching [`CbvTable`]
/// indexing).
///
/// # Panics
///
/// Panics if any subgame action has no match in the abstract tree, or if a
/// `DepthBoundary` doesn't correspond to a `Chance` node.
#[must_use]
pub fn build_boundary_mapping(
    subgame_tree: &GameTree,
    abstract_tree: &GameTree,
) -> Vec<usize> {
    // Precompute chance node ordinals in the abstract tree.
    let mut chance_ordinals = vec![usize::MAX; abstract_tree.nodes.len()];
    let mut ord = 0;
    for (idx, node) in abstract_tree.nodes.iter().enumerate() {
        if matches!(node, GameNode::Chance { .. }) {
            chance_ordinals[idx] = ord;
            ord += 1;
        }
    }

    let mut mapping = Vec::new();
    walk_trees_lockstep(
        subgame_tree,
        abstract_tree,
        &chance_ordinals,
        subgame_tree.root as usize,
        abstract_tree.root as usize,
        &mut mapping,
    );
    mapping
}

fn walk_trees_lockstep(
    sub: &GameTree,
    abs: &GameTree,
    chance_ordinals: &[usize],
    sub_idx: usize,
    abs_idx: usize,
    mapping: &mut Vec<usize>,
) {
    match (&sub.nodes[sub_idx], &abs.nodes[abs_idx]) {
        // Subgame hits DepthBoundary -> abstract should be at Chance.
        (
            GameNode::Terminal {
                kind: TerminalKind::DepthBoundary,
                ..
            },
            GameNode::Chance { .. },
        ) => {
            let ord = chance_ordinals[abs_idx];
            assert_ne!(
                ord,
                usize::MAX,
                "abstract Chance node {abs_idx} has no ordinal"
            );
            mapping.push(ord);
        }

        // Both are terminals (Fold or Showdown) -- nothing to map.
        (GameNode::Terminal { .. }, GameNode::Terminal { .. }) => {}

        // Subgame terminal vs abstract Chance/Decision -- the subgame
        // resolved this path (e.g. all-in showdown) while the abstract
        // tree continues through more streets. No boundaries needed.
        (GameNode::Terminal { .. }, GameNode::Chance { .. } | GameNode::Decision { .. }) => {}

        // Both are Chance nodes -- recurse into children.
        (
            GameNode::Chance {
                child: sub_child, ..
            },
            GameNode::Chance {
                child: abs_child, ..
            },
        ) => {
            walk_trees_lockstep(
                sub,
                abs,
                chance_ordinals,
                *sub_child as usize,
                *abs_child as usize,
                mapping,
            );
        }

        // Both are Decision nodes -- match actions and recurse.
        (
            GameNode::Decision {
                actions: sub_actions,
                children: sub_children,
                ..
            },
            GameNode::Decision {
                actions: abs_actions,
                children: abs_children,
                ..
            },
        ) => {
            // Match actions by TYPE (Fold<->Fold, Call<->Call, AllIn<->AllIn,
            // Bet/Raise by position among sized actions).
            // The subgame tree may have fewer raise depths than the abstract
            // tree, so we match what exists and skip abstract-only actions.
            for (sub_a_idx, sub_action) in sub_actions.iter().enumerate() {
                let abs_a_idx = match sub_action {
                    TreeAction::Fold => abs_actions.iter().position(|a| matches!(a, TreeAction::Fold)),
                    TreeAction::Check => abs_actions.iter().position(|a| matches!(a, TreeAction::Check)),
                    // Call and AllIn are interchangeable — calling can be all-in
                    // in one tree but not the other due to different unit scaling.
                    TreeAction::Call => abs_actions.iter().position(|a| matches!(a, TreeAction::Call | TreeAction::AllIn)),
                    TreeAction::AllIn => abs_actions.iter().position(|a| matches!(a, TreeAction::AllIn | TreeAction::Call)),
                    TreeAction::Bet(_) | TreeAction::Raise(_) => {
                        let sub_sized_idx = sub_actions[..sub_a_idx]
                            .iter()
                            .filter(|a| matches!(a, TreeAction::Bet(_) | TreeAction::Raise(_)))
                            .count();
                        abs_actions.iter().enumerate()
                            .filter(|(_, a)| matches!(a, TreeAction::Bet(_) | TreeAction::Raise(_)))
                            .nth(sub_sized_idx)
                            .map(|(i, _)| i)
                    }
                };
                let abs_a_idx = abs_a_idx.unwrap_or_else(|| panic!(
                    "no matching action for {sub_action:?} in abstract tree at node {abs_idx}: \
                     sub={sub_actions:?} abs={abs_actions:?}"
                ));
                walk_trees_lockstep(
                    sub,
                    abs,
                    chance_ordinals,
                    sub_children[sub_a_idx] as usize,
                    abs_children[abs_a_idx] as usize,
                    mapping,
                );
            }
        }

        // Subgame has an all-in response Decision [Fold, Call/AllIn] where
        // the abstract tree has a Chance node (the call was already resolved
        // in the full-depth tree). Walk the Call/AllIn child against the
        // Chance node; the Fold child leads to a terminal (no boundaries).
        (
            GameNode::Decision { actions: sub_actions, children: sub_children, .. },
            GameNode::Chance { .. },
        ) => {
            for (sub_a_idx, sub_action) in sub_actions.iter().enumerate() {
                match sub_action {
                    TreeAction::Call | TreeAction::AllIn => {
                        walk_trees_lockstep(
                            sub, abs, chance_ordinals,
                            sub_children[sub_a_idx] as usize,
                            abs_idx, // stay at same abstract Chance node
                            mapping,
                        );
                    }
                    TreeAction::Fold => {
                        // Fold leads to a terminal — no boundaries to map.
                    }
                    _ => {}
                }
            }
        }

        (sub_node, abs_node) => {
            panic!(
                "tree structure mismatch at sub={sub_idx} abs={abs_idx}: \
                 sub={sub_node:?}, abs={abs_node:?}"
            );
        }
    }
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::{Card, Suit, Value};
    use crate::blueprint_v2::Street;
    use test_macros::timed_test;

    fn river_board() -> Vec<Card> {
        vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Four, Suit::Club),
            Card::new(Value::Ten, Suit::Club),
        ]
    }

    /// Build a small hand set (first `n` combos) for fast testing.
    fn small_hands(board: &[Card], n: usize) -> SubgameHands {
        let full = SubgameHands::enumerate(board);
        SubgameHands {
            combos: full.combos.into_iter().take(n).collect(),
        }
    }

    #[timed_test]
    fn equity_matrix_is_symmetric() {
        let board = river_board();
        let hands = small_hands(&board, 50);
        let matrix = compute_equity_matrix(&hands.combos, &board);
        let n = hands.combos.len();
        for (i, row_i) in matrix.iter().enumerate() {
            for j in (i + 1)..n {
                let sum = row_i[j] + matrix[j][i];
                assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "equity[{i}][{j}] + equity[{j}][{i}] = {sum}, expected 1.0"
                );
            }
        }
    }

    #[timed_test]
    fn cards_overlap_detects_shared_cards() {
        let a = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
        ];
        let b = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Queen, Suit::Diamond),
        ];
        let c = [
            Card::new(Value::Two, Suit::Club),
            Card::new(Value::Three, Suit::Diamond),
        ];
        assert!(cards_overlap(a, b));
        assert!(!cards_overlap(a, c));
    }

    #[timed_test]
    fn build_boundary_mapping_matches_trees() {
        // Build abstract full-depth flop tree.
        let abstract_tree = GameTree::build_subgame(
            Street::Flop,
            100.0,
            [50.0, 50.0],
            250.0,
            &[vec![1.0]],
            None, // full depth
            0,
        );

        // Build subgame depth-limited flop tree with SAME bet sizes.
        let subgame_tree = GameTree::build_subgame(
            Street::Flop,
            100.0,
            [50.0, 50.0],
            250.0,
            &[vec![1.0]],
            Some(1), // depth limit = 1 street
            0,
        );

        // Count expected boundaries.
        let boundary_count = subgame_tree
            .nodes
            .iter()
            .filter(|n| {
                matches!(
                    n,
                    GameNode::Terminal {
                        kind: TerminalKind::DepthBoundary,
                        ..
                    }
                )
            })
            .count();
        assert!(boundary_count > 0, "subgame should have boundaries");

        // Count chance nodes in abstract tree.
        let chance_count = abstract_tree
            .nodes
            .iter()
            .filter(|n| matches!(n, GameNode::Chance { .. }))
            .count();

        let mapping = build_boundary_mapping(&subgame_tree, &abstract_tree);

        assert_eq!(mapping.len(), boundary_count);
        // Every mapped ordinal must be valid.
        for &ord in &mapping {
            assert!(
                ord < chance_count,
                "ordinal {ord} out of range (chance_count={chance_count})"
            );
        }
    }

    #[timed_test]
    #[should_panic(expected = "tree structure mismatch")]
    fn build_boundary_mapping_panics_on_mismatch() {
        let tree_a = GameTree::build_subgame(
            Street::Flop,
            100.0,
            [50.0, 50.0],
            250.0,
            &[vec![1.0]],
            Some(1),
            0,
        );
        // Abstract tree is a bare terminal — structural mismatch.
        let tree_b = GameTree {
            nodes: vec![GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 100.0,
                stacks: [200.0, 200.0],
            }],
            root: 0,
            dealer: 0,
            starting_stack: 250.0,
        };
        let _ = build_boundary_mapping(&tree_a, &tree_b);
    }
}
