//! MCCFR solver adapter for the convergence harness.
//!
//! Wraps `BlueprintTrainer` from poker-solver-core, implementing the
//! `ConvergenceSolver` trait with fixed-flop deals. Uses canonical preflop
//! hand bucketing (169 buckets) for preflop/flop streets and real
//! potential-aware clustering via `cluster_single_flop()` for turn/river.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use poker_solver_core::blueprint_v2::cluster_pipeline::{cluster_single_flop, combo_index};
use poker_solver_core::blueprint_v2::config::{
    ActionAbstractionConfig, BlueprintV2Config, ClusteringAlgorithm, ClusteringConfig, GameConfig,
    SnapshotConfig, StreetClusterConfig, TrainingConfig,
};
use poker_solver_core::blueprint_v2::game_tree::{GameNode, GameTree};
use poker_solver_core::blueprint_v2::mccfr::{Deal, DealWithBuckets, traverse_external};
use poker_solver_core::blueprint_v2::per_flop_bucket_file::PerFlopBucketFile;
use poker_solver_core::blueprint_v2::storage::BlueprintStorage;
use poker_solver_core::blueprint_v2::trainer::BlueprintTrainer;
use poker_solver_core::blueprint_v2::Street;
use poker_solver_core::hands::CanonicalHand;
use poker_solver_core::poker::{Card, Suit, Value, ALL_SUITS, ALL_VALUES};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use range_solver::card::flop_from_str;
use std::time::Instant;

use crate::baseline::Baseline;
use crate::game::FlopPokerConfig;
use crate::solver_trait::{ComboEvMap, ConvergenceSolver, SolverMetrics, StrategyMap};

/// Number of MCCFR iterations per `solve_step()` call.
const BATCH_SIZE: u64 = 1000;

/// Number of canonical preflop hand types.
const NUM_BUCKETS: u16 = 169;

// ---------------------------------------------------------------------------
// Card type conversion
// ---------------------------------------------------------------------------

/// Convert a range-solver card (u8: `4*rank + suit`) to a core `Card`.
///
/// Range-solver encoding: rank 2=0..A=12, suit c=0,d=1,h=2,s=3.
/// Core encoding: Value Two=0..Ace=12, Suit Spade=0,Club=1,Heart=2,Diamond=3.
fn rs_card_to_core_card(card_id: u8) -> Card {
    let rank = card_id >> 2;
    let rs_suit = card_id & 3;
    let value = Value::from_u8(rank);
    let suit = match rs_suit {
        0 => Suit::Club,
        1 => Suit::Diamond,
        2 => Suit::Heart,
        3 => Suit::Spade,
        _ => unreachable!(),
    };
    Card::new(value, suit)
}

// ---------------------------------------------------------------------------
// Fixed-flop deal sampling
// ---------------------------------------------------------------------------

/// The fixed flop cards: Qh Jd Th.
fn flop_cards() -> [Card; 3] {
    [
        Card::new(Value::Queen, Suit::Heart),
        Card::new(Value::Jack, Suit::Diamond),
        Card::new(Value::Ten, Suit::Heart),
    ]
}

/// Build the full 52-card deck in value-major order (matching core's canonical deck).
fn build_deck() -> [Card; 52] {
    let mut deck = [Card::new(ALL_VALUES[0], ALL_SUITS[0]); 52];
    let mut idx = 0;
    for &v in &ALL_VALUES {
        for &s in &ALL_SUITS {
            deck[idx] = Card::new(v, s);
            idx += 1;
        }
    }
    deck
}

/// Sample a deal with a fixed flop (QhJdTh).
/// Hole cards, turn, and river are random from the remaining 49 cards.
#[cfg(test)]
fn sample_fixed_flop_deal(rng: &mut impl Rng) -> Deal {
    sample_deal_for_flop(rng, &flop_cards())
}

/// Sample a deal for a specific flop.
/// Hole cards, turn, and river are random from the remaining 49 cards.
fn sample_deal_for_flop(rng: &mut impl Rng, flop: &[Card; 3]) -> Deal {
    let full_deck = build_deck();

    let mut deck: Vec<Card> = full_deck
        .iter()
        .copied()
        .filter(|c| !flop.contains(c))
        .collect();

    // Partial Fisher-Yates: shuffle first 6 positions (2+2 hole + turn + river)
    for i in 0..6 {
        let j = rng.random_range(i..deck.len());
        deck.swap(i, j);
    }

    Deal {
        hole_cards: [[deck[0], deck[1]], [deck[2], deck[3]]],
        board: [flop[0], flop[1], flop[2], deck[4], deck[5]],
    }
}

/// Convert a flop string (e.g. "QhJdTh") to core `Card` types.
///
/// Uses the range-solver's `flop_from_str` for parsing, then converts
/// each range-solver card ID to a core `Card`.
pub fn flop_str_to_core_cards(flop_str: &str) -> [Card; 3] {
    let rs_cards = flop_from_str(flop_str)
        .unwrap_or_else(|e| panic!("Invalid flop string '{}': {}", flop_str, e));
    [
        rs_card_to_core_card(rs_cards[0]),
        rs_card_to_core_card(rs_cards[1]),
        rs_card_to_core_card(rs_cards[2]),
    ]
}

/// Assign canonical preflop hand index as bucket for ALL streets.
/// This ignores board interaction -- pipeline validation only.
/// Retained for tests; production code uses `MccfrSolver::clustered_buckets`.
#[cfg(test)]
fn canonical_buckets(deal: &Deal) -> [[u16; 4]; 2] {
    let mut result = [[0u16; 4]; 2];
    for (player, row) in result.iter_mut().enumerate() {
        let hole = deal.hole_cards[player];
        let hand = CanonicalHand::from_cards(hole[0], hole[1]);
        let idx = hand.index() as u16;
        *row = [idx; 4];
    }
    result
}

// ---------------------------------------------------------------------------
// Config translation
// ---------------------------------------------------------------------------

/// Parse a percentage-based bet size string (e.g. "33%,67%") into pot fractions
/// for the blueprint tree builder.
///
/// Returns a `Vec<Vec<f64>>` where each inner vec is a raise depth.
/// - `"67%"` => `[[0.67]]` — one raise depth with 67% pot
/// - `"33%,67%"` => `[[0.33, 0.67]]` — one depth with two sizes
/// - `"a"` => `[]` — all-in only (no sized bets; all-in added automatically)
fn parse_bet_sizes(s: &str) -> Vec<Vec<f64>> {
    // "a" means all-in only — no pot-fraction sizes needed since the
    // blueprint tree builder always adds all-in.
    if s.trim().eq_ignore_ascii_case("a") {
        return vec![];
    }

    let sizes: Vec<f64> = s
        .split(',')
        .filter_map(|part| {
            let trimmed = part.trim().trim_end_matches('%');
            trimmed.parse::<f64>().ok().map(|v| v / 100.0)
        })
        .collect();
    if sizes.is_empty() {
        vec![vec![0.67]]
    } else {
        vec![sizes]
    }
}

/// Build a `BlueprintV2Config` from a `FlopPokerConfig`.
fn build_mccfr_config(
    config: &FlopPokerConfig,
    turn_buckets: u16,
    river_buckets: u16,
) -> BlueprintV2Config {
    let street_cluster = |buckets| StreetClusterConfig {
        buckets,
        delta_bins: None,
        expected_delta: false,
        sample_boards: None,
    };

    let postflop_sizes = parse_bet_sizes(&config.bet_sizes);

    BlueprintV2Config {
        game: GameConfig {
            name: "Flop Poker convergence test".into(),
            players: 2,
            stack_depth: config.effective_stack as f64,
            small_blind: 0.5,
            big_blind: 1.0,
            rake_rate: 0.0,
            rake_cap: 0.0,
        },
        clustering: ClusteringConfig {
            algorithm: ClusteringAlgorithm::PotentialAwareEmd,
            preflop: street_cluster(NUM_BUCKETS),
            flop: street_cluster(NUM_BUCKETS),
            turn: street_cluster(turn_buckets),
            river: street_cluster(river_buckets),
            seed: 42,
            kmeans_iterations: 0,
            cfvnet_river_data: None,
            per_flop: None,
        },
        action_abstraction: ActionAbstractionConfig {
            preflop: vec![vec!["2bb".into()]],
            flop: postflop_sizes.clone(),
            turn: postflop_sizes.clone(),
            river: postflop_sizes,
        },
        training: TrainingConfig {
            cluster_path: None,
            iterations: Some(1_000_000),
            time_limit_minutes: None,
            lcfr_warmup_iterations: 10_000,
            lcfr_discount_interval: 10_000,
            prune_after_iterations: 50_000,
            prune_threshold: -300,
            prune_explore_pct: 0.05,
            print_every_minutes: 60,
            batch_size: BATCH_SIZE,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.5,
            dcfr_gamma: 2.0,
            dcfr_epoch_cap: None,
            target_strategy_delta: None,
            purify_threshold: 0.0,
            equity_cache_path: None,
            optimizer: "dcfr".to_string(),
            sapcfr_eta: 0.5,
            use_baselines: false,
            baseline_alpha: 0.01,
        },
        snapshots: SnapshotConfig {
            warmup_minutes: 9999,
            snapshot_every_minutes: 9999,
            output_dir: "/tmp/convergence-harness-mccfr".into(),
            resume: false,
            max_snapshots: None,
        },
    }
}

// ---------------------------------------------------------------------------
// Strategy lifting
// ---------------------------------------------------------------------------

/// Lift a bucket-level strategy to combo-level.
///
/// For each combo (represented as a range-solver card pair), looks up the
/// canonical hand bucket and copies the bucket's strategy probabilities.
/// Returns a flat Vec in layout: `action_idx * num_combos + combo_idx`.
fn lift_bucket_strategy_for_node(
    storage: &BlueprintStorage,
    node_idx: u32,
    street: Street,
    num_actions: usize,
    combos: &[(u8, u8)],
    per_flop: Option<&PerFlopBucketFile>,
    turn_card: Option<Card>,
    river_card: Option<Card>,
) -> Vec<f32> {
    let num_combos = combos.len();
    let mut result = vec![0.0f32; num_actions * num_combos];

    // For turn/river, precompute card indices once
    let turn_idx = if street == Street::Turn || street == Street::River {
        per_flop.and_then(|pf| {
            turn_card.and_then(|tc| pf.turn_cards.iter().position(|&c| c == tc))
        })
    } else {
        None
    };
    let river_idx = if street == Street::River {
        turn_idx.and_then(|ti| {
            per_flop.and_then(|pf| {
                river_card.and_then(|rc| {
                    pf.river_cards_per_turn[ti].iter().position(|&c| c == rc)
                })
            })
        })
    } else {
        None
    };

    for (combo_idx, &(c1, c2)) in combos.iter().enumerate() {
        let core_c1 = rs_card_to_core_card(c1);
        let core_c2 = rs_card_to_core_card(c2);

        let bucket = match street {
            Street::Preflop | Street::Flop => {
                let hand = CanonicalHand::from_cards(core_c1, core_c2);
                hand.index() as u16
            }
            Street::Turn => {
                if let (Some(pf), Some(ti)) = (per_flop, turn_idx) {
                    let ci = combo_index(core_c1, core_c2) as usize;
                    pf.get_turn_bucket(ti, ci)
                } else {
                    let hand = CanonicalHand::from_cards(core_c1, core_c2);
                    hand.index() as u16
                }
            }
            Street::River => {
                if let (Some(pf), Some(ti), Some(ri)) = (per_flop, turn_idx, river_idx) {
                    let ci = combo_index(core_c1, core_c2) as usize;
                    pf.get_river_bucket(ti, ri, ci)
                } else {
                    let hand = CanonicalHand::from_cards(core_c1, core_c2);
                    hand.index() as u16
                }
            }
        };

        let strat = storage.average_strategy(node_idx, bucket);
        for (action_idx, &prob) in strat.iter().enumerate() {
            if action_idx < num_actions {
                result[action_idx * num_combos + combo_idx] = prob as f32;
            }
        }
    }

    result
}

/// Walk the blueprint game tree and extract lifted strategies at every
/// decision node. Uses arena index as node key.
fn extract_lifted_strategies(
    tree: &GameTree,
    storage: &BlueprintStorage,
    combos: &[(u8, u8)],
    per_flop: Option<&PerFlopBucketFile>,
) -> StrategyMap {
    let mut result = StrategyMap::new();

    for (idx, node) in tree.nodes.iter().enumerate() {
        if let GameNode::Decision { actions, street, .. } = node {
            let num_actions = actions.len();
            // Note: for tree-level extraction we don't have specific board cards.
            // Turn/river nodes need board context which we don't have here.
            // This is only used for the ConvergenceSolver trait's average_strategy().
            let strat = lift_bucket_strategy_for_node(
                storage,
                idx as u32,
                *street,
                num_actions,
                combos,
                per_flop,
                None, // no turn card context in bulk extraction
                None, // no river card context
            );
            result.insert(idx as u64, strat);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Exploitability computation
// ---------------------------------------------------------------------------

/// Find the first Flop decision node in the blueprint tree.
///
/// The blueprint tree starts with preflop decisions and chance nodes. This
/// walks from the root, following through preflop decisions and the
/// preflop-to-flop chance node, until it reaches the first Flop Decision.
pub fn find_flop_root(tree: &GameTree) -> u32 {
    let mut idx = tree.root;
    loop {
        match &tree.nodes[idx as usize] {
            GameNode::Decision { street, children, .. } => {
                if *street != Street::Preflop {
                    return idx;
                }
                // Follow the first child through preflop (trivial preflop
                // has one meaningful path). We follow the LAST child which
                // is typically the action that continues the hand (call/check)
                // rather than fold. But with trivial preflop (SB limps, BB
                // checks), every path eventually reaches the flop. The
                // structure is: SB acts -> [fold, call, ...]. We want the
                // call path, which is the last non-fold child. In our
                // trivial preflop tree, the SB has [Fold, Call] or similar.
                // The call leads to BB acting, BB checks, then flop chance.
                //
                // Walk ALL children via DFS to find the flop.
                for &child in children {
                    if let Some(flop_idx) = find_flop_root_dfs(tree, child) {
                        return flop_idx;
                    }
                }
                panic!("No flop decision found from preflop root");
            }
            GameNode::Chance { child, .. } => {
                idx = *child;
            }
            GameNode::Terminal { .. } => {
                panic!("Reached terminal before finding flop decision");
            }
        }
    }
}

/// DFS helper to find the first Flop decision node reachable from `idx`.
fn find_flop_root_dfs(tree: &GameTree, idx: u32) -> Option<u32> {
    match &tree.nodes[idx as usize] {
        GameNode::Decision { street, children, .. } => {
            if *street != Street::Preflop {
                return Some(idx);
            }
            for &child in children {
                if let Some(found) = find_flop_root_dfs(tree, child) {
                    return Some(found);
                }
            }
            None
        }
        GameNode::Chance { child, .. } => find_flop_root_dfs(tree, *child),
        GameNode::Terminal { .. } => None,
    }
}

/// Compute exploitability of the MCCFR strategy in the full (unabstracted) game.
///
/// Builds a range-solver game with the same parameters, walks the tree in
/// parallel with the blueprint tree, locks the lifted MCCFR strategy at
/// every decision node, and then runs best-response exploitability.
///
/// `flop_idx` selects which flop's per-flop bucket file to use for
/// turn/river bucket lookups.
pub fn compute_mccfr_exploitability(
    solver: &MccfrSolver,
    config: &FlopPokerConfig,
    flop_idx: usize,
) -> Result<f64, String> {
    // 1. Build range-solver game with matching tree structure.
    let mut game = crate::game::build_flop_poker_game_with_config(config)?;
    game.allocate_memory(false);

    // 2. Find the flop root in the blueprint tree
    let bp_flop_root = find_flop_root(solver.tree());

    // 3. Walk both trees in parallel, locking MCCFR strategy at each decision node
    let mut history: Vec<usize> = Vec::new();
    let mut board_cards = (None, None);
    lock_strategy_recursive(
        &mut game,
        solver.tree(),
        solver.storage(),
        Some(solver.per_flop_buckets_for(flop_idx)),
        bp_flop_root,
        &mut history,
        &mut board_cards,
    );

    // 4. Compute exploitability
    Ok(range_solver::compute_exploitability(&game) as f64)
}

/// Recursively walk the range-solver game tree and blueprint tree in parallel,
/// locking the MCCFR lifted strategy at every decision node.
///
/// `board_cards` tracks the current turn/river cards as we traverse chance nodes,
/// needed for street-aware bucket lookups.
pub fn lock_strategy_recursive(
    game: &mut range_solver::PostFlopGame,
    tree: &GameTree,
    storage: &BlueprintStorage,
    per_flop: Option<&PerFlopBucketFile>,
    bp_node_idx: u32,
    history: &mut Vec<usize>,
    board_cards: &mut (Option<Card>, Option<Card>), // (turn, river)
) {
    if game.is_terminal_node() {
        return;
    }

    if game.is_chance_node() {
        let (bp_child, next_street) = match &tree.nodes[bp_node_idx as usize] {
            GameNode::Chance { child, next_street } => (*child, *next_street),
            other => panic!(
                "Expected blueprint Chance node at idx {}, got {:?}",
                bp_node_idx,
                std::mem::discriminant(other)
            ),
        };

        let num_chance_actions = game.available_actions().len();
        for i in 0..num_chance_actions {
            let actions = game.available_actions();
            if let range_solver::action_tree::Action::Chance(card) = actions[i] {
                // Track the dealt card for bucket lookups
                let core_card = rs_card_to_core_card(card);
                let prev = *board_cards;
                match next_street {
                    Street::Turn => board_cards.0 = Some(core_card),
                    Street::River => board_cards.1 = Some(core_card),
                    _ => {}
                }

                history.push(card as usize);
                game.play(card as usize);
                lock_strategy_recursive(game, tree, storage, per_flop, bp_child, history, board_cards);
                history.pop();
                *board_cards = prev;
                crate::evaluator::navigate_back(game, history);
            }
        }
        return;
    }

    // Decision node -- extract blueprint info
    let bp_node = &tree.nodes[bp_node_idx as usize];
    let (bp_street, bp_actions, bp_children) = match bp_node {
        GameNode::Decision {
            street, actions, children, ..
        } => (*street, actions, children),
        other => panic!(
            "Expected blueprint Decision node at idx {}, got {:?}",
            bp_node_idx,
            std::mem::discriminant(other)
        ),
    };

    let rs_actions = game.available_actions();
    let num_actions = rs_actions.len();
    assert_eq!(
        num_actions,
        bp_actions.len(),
        "Action count mismatch at history {:?}: range-solver has {}, blueprint has {}",
        history,
        num_actions,
        bp_actions.len()
    );

    // Build combo-level strategy with street-aware bucket lookup and lock it
    let player = game.current_player();
    let combos = game.private_cards(player);
    let strategy = lift_bucket_strategy_for_node(
        storage,
        bp_node_idx,
        bp_street,
        num_actions,
        combos,
        per_flop,
        board_cards.0,
        board_cards.1,
    );
    game.lock_current_strategy(&strategy);

    // Recurse into children
    for (action_idx, &bp_child_idx) in bp_children.iter().enumerate() {
        history.push(action_idx);
        game.play(action_idx);
        lock_strategy_recursive(game, tree, storage, per_flop, bp_child_idx, history, board_cards);
        history.pop();
        crate::evaluator::navigate_back(game, history);
    }
}

// ---------------------------------------------------------------------------
// Head-to-head EV computation
// ---------------------------------------------------------------------------

/// Compute head-to-head EV: exact baseline vs MCCFR strategy for a single flop.
///
/// Builds two range-solver games. In each, one player uses the exact baseline
/// strategy and the other uses the MCCFR (lifted bucket) strategy. The EV
/// difference from the Nash value measures how much the MCCFR strategy loses.
///
/// `flop_idx` selects which flop's per-flop bucket file to use.
///
/// Returns `(oop_delta, ip_delta, average)` in mbb/hand from MCCFR's
/// perspective. Negative means MCCFR loses to the baseline strategy.
pub fn compute_head_to_head_ev(
    solver: &MccfrSolver,
    baseline: &Baseline,
    config: &FlopPokerConfig,
    flop_idx: usize,
) -> Result<(f64, f64, f64), String> {
    // Compute Nash EV: both players use baseline strategy
    let nash_ev = {
        let mut game = crate::game::build_flop_poker_game_with_config(config)?;
        game.allocate_memory(false);
        lock_baseline_strategy(&mut game, &baseline.strategy);
        range_solver::compute_current_ev(&game)
    };

    let bp_flop_root = find_flop_root(solver.tree());
    let pf_buckets = solver.per_flop_buckets_for(flop_idx);

    // MCCFR as OOP (player 0), baseline as IP (player 1)
    let ev_mccfr_oop = {
        let mut game = crate::game::build_flop_poker_game_with_config(config)?;
        game.allocate_memory(false);
        let mut history = Vec::new();
        let mut board_cards = (None, None);
        lock_head_to_head_recursive(
            &mut game,
            solver.tree(),
            solver.storage(),
            Some(pf_buckets),
            &baseline.strategy,
            bp_flop_root,
            &mut history,
            0, // mccfr plays OOP
            &mut board_cards,
        );
        range_solver::compute_current_ev(&game)
    };

    // MCCFR as IP (player 1), baseline as OOP (player 0)
    let ev_mccfr_ip = {
        let mut game = crate::game::build_flop_poker_game_with_config(config)?;
        game.allocate_memory(false);
        let mut history = Vec::new();
        let mut board_cards = (None, None);
        lock_head_to_head_recursive(
            &mut game,
            solver.tree(),
            solver.storage(),
            Some(pf_buckets),
            &baseline.strategy,
            bp_flop_root,
            &mut history,
            1, // mccfr plays IP
            &mut board_cards,
        );
        range_solver::compute_current_ev(&game)
    };

    // EV delta from MCCFR's perspective: negative means MCCFR loses vs baseline.
    // mccfr_ev - nash_ev: negative when MCCFR is worse (losing to the exact strategy).
    let oop_delta_raw = (ev_mccfr_oop[0] - nash_ev[0]) as f64;
    let ip_delta_raw = (ev_mccfr_ip[1] - nash_ev[1]) as f64;

    // Convert to mbb/hand: multiply by 1000 (milli big blinds)
    let oop_mbb = oop_delta_raw * 1000.0;
    let ip_mbb = ip_delta_raw * 1000.0;
    let avg_mbb = (oop_mbb + ip_mbb) / 2.0;

    Ok((oop_mbb, ip_mbb, avg_mbb))
}

/// Lock the baseline (exact) strategy at every decision node in the
/// range-solver game tree.
fn lock_baseline_strategy(
    game: &mut range_solver::PostFlopGame,
    baseline_strategy: &std::collections::BTreeMap<u64, Vec<f32>>,
) {
    let mut history = Vec::new();
    lock_baseline_recursive(game, baseline_strategy, &mut history);
    game.back_to_root();
}

/// Recursively walk the range-solver tree, locking the baseline strategy
/// at every decision node.
fn lock_baseline_recursive(
    game: &mut range_solver::PostFlopGame,
    baseline_strategy: &std::collections::BTreeMap<u64, Vec<f32>>,
    history: &mut Vec<usize>,
) {
    if game.is_terminal_node() {
        return;
    }

    if game.is_chance_node() {
        let num_chance_actions = game.available_actions().len();
        for i in 0..num_chance_actions {
            let actions = game.available_actions();
            if let range_solver::action_tree::Action::Chance(card) = actions[i] {
                history.push(card as usize);
                game.play(card as usize);
                lock_baseline_recursive(game, baseline_strategy, history);
                history.pop();
                crate::evaluator::navigate_back(game, history);
            }
        }
        return;
    }

    // Decision node: lock the baseline strategy
    let nid = crate::evaluator::node_id(history);
    if let Some(strat) = baseline_strategy.get(&nid) {
        game.lock_current_strategy(strat);
    }

    let num_actions = game.available_actions().len();
    for action_idx in 0..num_actions {
        history.push(action_idx);
        game.play(action_idx);
        lock_baseline_recursive(game, baseline_strategy, history);
        history.pop();
        crate::evaluator::navigate_back(game, history);
    }
}

/// Recursively walk both trees, locking MCCFR strategy for `mccfr_player`
/// and baseline strategy for the other player at each decision node.
fn lock_head_to_head_recursive(
    game: &mut range_solver::PostFlopGame,
    tree: &GameTree,
    storage: &BlueprintStorage,
    per_flop: Option<&PerFlopBucketFile>,
    baseline_strategy: &std::collections::BTreeMap<u64, Vec<f32>>,
    bp_node_idx: u32,
    history: &mut Vec<usize>,
    mccfr_player: usize,
    board_cards: &mut (Option<Card>, Option<Card>),
) {
    if game.is_terminal_node() {
        return;
    }

    if game.is_chance_node() {
        let (bp_child, next_street) = match &tree.nodes[bp_node_idx as usize] {
            GameNode::Chance { child, next_street } => (*child, *next_street),
            other => panic!(
                "Expected blueprint Chance node at idx {}, got {:?}",
                bp_node_idx,
                std::mem::discriminant(other)
            ),
        };

        let num_chance_actions = game.available_actions().len();
        for i in 0..num_chance_actions {
            let actions = game.available_actions();
            if let range_solver::action_tree::Action::Chance(card) = actions[i] {
                let core_card = rs_card_to_core_card(card);
                let prev = *board_cards;
                match next_street {
                    Street::Turn => board_cards.0 = Some(core_card),
                    Street::River => board_cards.1 = Some(core_card),
                    _ => {}
                }

                history.push(card as usize);
                game.play(card as usize);
                lock_head_to_head_recursive(
                    game, tree, storage, per_flop, baseline_strategy,
                    bp_child, history, mccfr_player, board_cards,
                );
                history.pop();
                *board_cards = prev;
                crate::evaluator::navigate_back(game, history);
            }
        }
        return;
    }

    // Decision node
    let bp_node = &tree.nodes[bp_node_idx as usize];
    let (bp_street, bp_actions, bp_children) = match bp_node {
        GameNode::Decision {
            street, actions, children, ..
        } => (*street, actions, children),
        other => panic!(
            "Expected blueprint Decision node at idx {}, got {:?}",
            bp_node_idx,
            std::mem::discriminant(other)
        ),
    };

    let rs_actions = game.available_actions();
    let num_actions = rs_actions.len();
    assert_eq!(
        num_actions,
        bp_actions.len(),
        "Action count mismatch at history {:?}: range-solver has {}, blueprint has {}",
        history,
        num_actions,
        bp_actions.len()
    );

    let player = game.current_player();
    if player == mccfr_player {
        // Lock MCCFR lifted strategy with street-aware bucket lookup
        let combos = game.private_cards(player);
        let strategy = lift_bucket_strategy_for_node(
            storage, bp_node_idx, bp_street, num_actions, combos,
            per_flop, board_cards.0, board_cards.1,
        );
        game.lock_current_strategy(&strategy);
    } else {
        // Lock baseline strategy
        let nid = crate::evaluator::node_id(history);
        if let Some(strat) = baseline_strategy.get(&nid) {
            game.lock_current_strategy(strat);
        }
    }

    // Recurse into children
    for (action_idx, &bp_child_idx) in bp_children.iter().enumerate() {
        history.push(action_idx);
        game.play(action_idx);
        lock_head_to_head_recursive(
            game, tree, storage, per_flop, baseline_strategy,
            bp_child_idx, history, mccfr_player, board_cards,
        );
        history.pop();
        crate::evaluator::navigate_back(game, history);
    }
}

// ---------------------------------------------------------------------------
// MccfrSolver
// ---------------------------------------------------------------------------

/// MCCFR solver adapter: runs blueprint MCCFR with potential-aware clustering.
pub struct MccfrSolver {
    tree: GameTree,
    storage: BlueprintStorage,
    /// Combo list from range-solver (card pairs using range-solver u8 encoding).
    combos: Vec<(u8, u8)>,
    /// Flop boards (core Card types).
    flops: Vec<[Card; 3]>,
    /// Clustered turn/river bucket assignments, one per flop.
    all_per_flop_buckets: Vec<PerFlopBucketFile>,
    /// Pre-computed solver name string.
    name_str: String,
    iteration: u64,
    rng: SmallRng,
}

impl MccfrSolver {
    /// Create a new MCCFR solver from a `FlopPokerConfig`.
    ///
    /// `turn_buckets` and `river_buckets` control the granularity of the
    /// potential-aware clustering used for turn and river streets.
    pub fn new(config: FlopPokerConfig, turn_buckets: u16, river_buckets: u16) -> Self {
        Self::new_multi_flop(
            config,
            &[flop_cards()],
            turn_buckets,
            river_buckets,
        )
    }

    /// Create a new MCCFR solver with multiple flops.
    ///
    /// All flops share the same game tree structure and storage (since bet
    /// sizes are identical). Each flop gets its own per-flop bucket file
    /// for turn/river clustering.
    pub fn new_multi_flop(
        config: FlopPokerConfig,
        flops: &[[Card; 3]],
        turn_buckets: u16,
        river_buckets: u16,
    ) -> Self {
        assert!(!flops.is_empty(), "Must provide at least one flop");

        // Build a range-solver game to get the combo list (using the first flop).
        let rs_game = crate::game::build_flop_poker_game_with_config(&config)
            .expect("Failed to build range-solver game for combo list");
        let combos: Vec<(u8, u8)> = rs_game.private_cards(0).to_vec();

        // Cluster each flop independently.
        let mut all_per_flop_buckets = Vec::with_capacity(flops.len());
        for flop in flops {
            eprintln!(
                "Clustering flop {:?} with {} turn / {} river buckets...",
                flop, turn_buckets, river_buckets
            );
            let cluster_start = Instant::now();
            let per_flop = cluster_single_flop(
                *flop,
                turn_buckets,
                river_buckets,
                50,
                42,
                |phase, done, total| {
                    if done == total {
                        eprintln!("  clustering phase {}: done", phase);
                    }
                },
            );
            eprintln!(
                "Clustering complete in {:.1}s ({} turns, {} rivers/turn avg)",
                cluster_start.elapsed().as_secs_f64(),
                per_flop.turn_cards.len(),
                if per_flop.turn_cards.is_empty() {
                    0
                } else {
                    per_flop
                        .river_cards_per_turn
                        .iter()
                        .map(Vec::len)
                        .sum::<usize>()
                        / per_flop.turn_cards.len()
                },
            );
            all_per_flop_buckets.push(per_flop);
        }

        let mccfr_config = build_mccfr_config(&config, turn_buckets, river_buckets);
        let mut trainer = BlueprintTrainer::new(mccfr_config);
        trainer.skip_bucket_validation = true;

        // Extract the tree and storage from the trainer so we can call
        // traverse_external directly without the full training loop.
        let tree = std::mem::replace(
            &mut trainer.tree,
            GameTree {
                nodes: Vec::new(),
                root: 0,
                dealer: 0,
                starting_stack: 0.0,
            },
        );
        let storage = std::mem::replace(
            &mut trainer.storage,
            BlueprintStorage::new(
                &GameTree {
                    nodes: Vec::new(),
                    root: 0,
                    dealer: 0,
                    starting_stack: 0.0,
                },
                [1, 1, 1, 1],
            ),
        );

        let name_str = format!(
            "MCCFR ({}t/{}r buckets, {} flops)",
            turn_buckets,
            river_buckets,
            flops.len()
        );

        Self {
            tree,
            storage,
            combos,
            flops: flops.to_vec(),
            all_per_flop_buckets,
            name_str,
            iteration: 0,
            rng: SmallRng::seed_from_u64(42),
        }
    }

    /// Access the blueprint game tree.
    pub fn tree(&self) -> &GameTree {
        &self.tree
    }

    /// Access the blueprint storage.
    pub fn storage(&self) -> &BlueprintStorage {
        &self.storage
    }

    /// Access the per-flop bucket file for a specific flop index.
    pub fn per_flop_buckets_for(&self, flop_idx: usize) -> &PerFlopBucketFile {
        &self.all_per_flop_buckets[flop_idx]
    }

    /// Access all per-flop bucket files.
    pub fn all_per_flop_buckets(&self) -> &[PerFlopBucketFile] {
        &self.all_per_flop_buckets
    }

    /// Access the flop boards.
    pub fn flops(&self) -> &[[Card; 3]] {
        &self.flops
    }

    /// Assign buckets using real potential-aware clustering for a specific flop.
    ///
    /// Preflop and flop use canonical hand index (169 buckets).
    /// Turn and river use the per-flop clustered bucket assignments.
    fn clustered_buckets(&self, deal: &Deal, flop_idx: usize) -> [[u16; 4]; 2] {
        let pf = &self.all_per_flop_buckets[flop_idx];
        let mut result = [[0u16; 4]; 2];
        let turn_card = deal.board[3];
        let river_card = deal.board[4];
        let turn_idx = pf
            .turn_cards
            .iter()
            .position(|&c| c == turn_card)
            .unwrap_or_else(|| panic!("Turn card {:?} not found in per-flop bucket file", turn_card));
        let river_idx = pf.river_cards_per_turn[turn_idx]
            .iter()
            .position(|&c| c == river_card)
            .unwrap_or_else(|| {
                panic!(
                    "River card {:?} not found for turn_idx {} in per-flop bucket file",
                    river_card, turn_idx
                )
            });

        for (player, row) in result.iter_mut().enumerate() {
            let hole = deal.hole_cards[player];

            // Preflop: canonical hand index (169)
            let hand = CanonicalHand::from_cards(hole[0], hole[1]);
            row[0] = hand.index() as u16;

            // Flop: same as preflop (no real flop clustering)
            row[1] = row[0];

            // Turn: look up in per-flop file
            let ci = combo_index(hole[0], hole[1]) as usize;
            row[2] = pf.get_turn_bucket(turn_idx, ci);

            // River: look up in per-flop file
            row[3] = pf.get_river_bucket(turn_idx, river_idx, ci);
        }
        result
    }
}

impl ConvergenceSolver for MccfrSolver {
    fn name(&self) -> &str {
        &self.name_str
    }

    fn solve_step(&mut self) {
        let num_flops = self.flops.len();
        for _ in 0..BATCH_SIZE {
            let seed: u64 = self.rng.random();
            let mut deal_rng = SmallRng::seed_from_u64(seed);
            // Pick a random flop (skip RNG for single-flop case)
            let flop_idx = if num_flops == 1 {
                0
            } else {
                deal_rng.random_range(0..num_flops)
            };
            let deal = sample_deal_for_flop(&mut deal_rng, &self.flops[flop_idx]);
            let buckets = self.clustered_buckets(&deal, flop_idx);
            let dwb = DealWithBuckets { deal, buckets };

            // Traverse for both players (external sampling MCCFR)
            traverse_external(
                &self.tree,
                &self.storage,
                &dwb,
                0,
                self.tree.root,
                false,
                0,
                &mut deal_rng,
                0.0,
                0.0,
                None,
                None,
                0.0,
            );
            traverse_external(
                &self.tree,
                &self.storage,
                &dwb,
                1,
                self.tree.root,
                false,
                0,
                &mut deal_rng,
                0.0,
                0.0,
                None,
                None,
                0.0,
            );
        }
        self.iteration += BATCH_SIZE;
    }

    fn iterations(&self) -> u64 {
        self.iteration
    }

    fn average_strategy(&self) -> StrategyMap {
        // Use the first flop's bucket file for bulk extraction. Turn/river
        // nodes won't have board context anyway (None turn/river cards in
        // extract_lifted_strategies), so the specific bucket file used
        // doesn't affect preflop/flop extraction.
        extract_lifted_strategies(
            &self.tree,
            &self.storage,
            &self.combos,
            self.all_per_flop_buckets.first(),
        )
    }

    fn combo_evs(&self) -> ComboEvMap {
        ComboEvMap::new()
    }

    fn self_reported_metrics(&self) -> SolverMetrics {
        SolverMetrics::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;

    #[test]
    fn test_fixed_flop_deal_has_correct_board() {
        let mut rng = SmallRng::seed_from_u64(42);
        let deal = sample_fixed_flop_deal(&mut rng);
        let expected = flop_cards();
        assert_eq!(deal.board[0], expected[0], "board[0] should be Qh");
        assert_eq!(deal.board[1], expected[1], "board[1] should be Jd");
        assert_eq!(deal.board[2], expected[2], "board[2] should be Th");
    }

    #[test]
    fn test_fixed_flop_deal_no_duplicate_cards() {
        let mut rng = SmallRng::seed_from_u64(42);
        let deal = sample_fixed_flop_deal(&mut rng);
        let mut all_cards: Vec<Card> = vec![
            deal.hole_cards[0][0],
            deal.hole_cards[0][1],
            deal.hole_cards[1][0],
            deal.hole_cards[1][1],
            deal.board[0],
            deal.board[1],
            deal.board[2],
            deal.board[3],
            deal.board[4],
        ];
        all_cards.sort();
        all_cards.dedup();
        assert_eq!(all_cards.len(), 9, "All 9 dealt cards must be unique");
    }

    #[test]
    fn test_canonical_buckets_same_for_all_streets() {
        let mut rng = SmallRng::seed_from_u64(42);
        let deal = sample_fixed_flop_deal(&mut rng);
        let buckets = canonical_buckets(&deal);
        for player_buckets in &buckets {
            assert_eq!(player_buckets[0], player_buckets[1]);
            assert_eq!(player_buckets[1], player_buckets[2]);
            assert_eq!(player_buckets[2], player_buckets[3]);
            assert!(
                player_buckets[0] < 169,
                "Bucket must be valid canonical index"
            );
        }
    }

    #[test]
    fn test_canonical_buckets_different_hands_get_different_buckets() {
        let mut rng = SmallRng::seed_from_u64(99);
        let deal = sample_fixed_flop_deal(&mut rng);
        let buckets = canonical_buckets(&deal);
        assert!(buckets[0][0] < 169);
        assert!(buckets[1][0] < 169);
    }

    #[test]
    fn test_range_solver_card_to_core_card_round_trip() {
        // As = 4*12 + 3 = 51
        let core_card = rs_card_to_core_card(51);
        assert_eq!(core_card.value, Value::Ace);
        assert_eq!(core_card.suit, Suit::Spade);

        // 2c = 4*0 + 0 = 0
        let core_card = rs_card_to_core_card(0);
        assert_eq!(core_card.value, Value::Two);
        assert_eq!(core_card.suit, Suit::Club);

        // Qh = 4*10 + 2 = 42
        let core_card = rs_card_to_core_card(42);
        assert_eq!(core_card.value, Value::Queen);
        assert_eq!(core_card.suit, Suit::Heart);

        // Jd = 4*9 + 1 = 37
        let core_card = rs_card_to_core_card(37);
        assert_eq!(core_card.value, Value::Jack);
        assert_eq!(core_card.suit, Suit::Diamond);
    }

    #[test]
    fn test_range_solver_card_to_core_card_all_52() {
        // Every card should convert without panic and have valid value/suit
        for card_id in 0..52u8 {
            let core_card = rs_card_to_core_card(card_id);
            let rank = card_id >> 2;
            assert_eq!(core_card.value as u8, rank);
        }
    }

    // -----------------------------------------------------------------------
    // parse_bet_sizes tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_bet_sizes_percentage() {
        assert_eq!(parse_bet_sizes("67%"), vec![vec![0.67]]);
    }

    #[test]
    fn test_parse_bet_sizes_multiple() {
        assert_eq!(parse_bet_sizes("33%,67%"), vec![vec![0.33, 0.67]]);
    }

    #[test]
    fn test_parse_bet_sizes_allin_returns_empty() {
        let result = parse_bet_sizes("a");
        assert!(result.is_empty(), "All-in should return empty sizes");
    }

    #[test]
    fn test_parse_bet_sizes_allin_case_insensitive() {
        assert!(parse_bet_sizes("A").is_empty());
    }

    #[test]
    fn test_parse_bet_sizes_fallback_to_default() {
        // Non-numeric, non-"a" input falls back to 67%
        assert_eq!(parse_bet_sizes("invalid"), vec![vec![0.67]]);
    }

    #[test]
    fn test_sample_deal_for_flop_uses_given_flop() {
        let mut rng = SmallRng::seed_from_u64(42);
        let flop = [
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Two, Suit::Club),
        ];
        let deal = sample_deal_for_flop(&mut rng, &flop);
        assert_eq!(deal.board[0], flop[0]);
        assert_eq!(deal.board[1], flop[1]);
        assert_eq!(deal.board[2], flop[2]);
    }

    #[test]
    fn test_sample_deal_for_flop_no_duplicates() {
        let mut rng = SmallRng::seed_from_u64(7);
        let flop = [
            Card::new(Value::Eight, Suit::Club),
            Card::new(Value::Eight, Suit::Diamond),
            Card::new(Value::Three, Suit::Heart),
        ];
        let deal = sample_deal_for_flop(&mut rng, &flop);
        let mut all_cards: Vec<Card> = vec![
            deal.hole_cards[0][0],
            deal.hole_cards[0][1],
            deal.hole_cards[1][0],
            deal.hole_cards[1][1],
            deal.board[0],
            deal.board[1],
            deal.board[2],
            deal.board[3],
            deal.board[4],
        ];
        all_cards.sort();
        all_cards.dedup();
        assert_eq!(all_cards.len(), 9, "All 9 dealt cards must be unique");
    }

    #[test]
    fn test_flop_from_str_helper_parses_valid_flop() {
        let cards = flop_str_to_core_cards("QhJdTh");
        let mut sorted = cards;
        sorted.sort();
        let mut expected = [
            Card::new(Value::Queen, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
            Card::new(Value::Ten, Suit::Heart),
        ];
        expected.sort();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn test_flop_from_str_helper_different_flop() {
        let cards = flop_str_to_core_cards("Ks7d2c");
        let mut sorted = cards;
        sorted.sort();
        let mut expected = [
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Two, Suit::Club),
        ];
        expected.sort();
        assert_eq!(sorted, expected);
    }
}
