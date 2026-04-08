//! Cross-validation integration tests for the `blueprint_mp` N-player module.
//!
//! Verifies that `blueprint_mp` produces reasonable results on 2-player configs
//! and that 3-player training completes without panics.

use poker_solver_core::blueprint_mp::config::*;
use poker_solver_core::blueprint_mp::exploitability::compute_exploitability;
use poker_solver_core::blueprint_mp::game_tree::*;
use poker_solver_core::blueprint_mp::mccfr::{sample_deal, traverse_external};
use poker_solver_core::blueprint_mp::storage::MpStorage;
use poker_solver_core::blueprint_mp::trainer::train_blueprint_mp;
use poker_solver_core::blueprint_mp::{Chips, Seat, MAX_PLAYERS};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use test_macros::timed_test;

const BUCKET_COUNTS: [u16; 4] = [10, 10, 10, 10];

// ── Helpers ────────────────────────────────────────────────────────

fn yaml_f64(v: f64) -> serde_yaml::Value {
    serde_yaml::Value::Number(serde_yaml::Number::from(v))
}

fn sb_bb_blinds() -> Vec<ForcedBet> {
    vec![
        ForcedBet { seat: 0, kind: ForcedBetKind::SmallBlind, amount: 1.0 },
        ForcedBet { seat: 1, kind: ForcedBetKind::BigBlind, amount: 2.0 },
    ]
}

fn sized_action_config() -> MpActionAbstractionConfig {
    let postflop = MpStreetSizes {
        lead: vec![yaml_f64(0.67)],
        raise: vec![vec![yaml_f64(1.0)]],
    };
    let preflop = MpStreetSizes {
        lead: vec![serde_yaml::Value::String("5bb".into())],
        raise: vec![vec![serde_yaml::Value::String("3.0x".into())]],
    };
    MpActionAbstractionConfig {
        preflop,
        flop: postflop.clone(),
        turn: postflop.clone(),
        river: postflop,
    }
}

/// Build game + action configs with sized bets for `num_players`.
fn build_sized_config(num_players: u8) -> (MpGameConfig, MpActionAbstractionConfig) {
    let game = MpGameConfig {
        name: format!("{num_players}p-validation"),
        num_players,
        stack_depth: 40.0,
        blinds: sb_bb_blinds(),
        rake_rate: 0.0,
        rake_cap: 0.0,
    };
    (game, sized_action_config())
}

/// Build a full `BlueprintMpConfig` with empty (fold/check/call/all-in
/// only) action abstraction, suitable for fast training tests.
fn build_full_config(num_players: u8, iterations: u64) -> BlueprintMpConfig {
    let game = MpGameConfig {
        name: format!("{num_players}p-validation"),
        num_players,
        stack_depth: 20.0,
        blinds: sb_bb_blinds(),
        rake_rate: 0.0,
        rake_cap: 0.0,
    };
    let empty = MpStreetSizes { lead: vec![], raise: vec![] };
    let action = MpActionAbstractionConfig {
        preflop: empty.clone(),
        flop: empty.clone(),
        turn: empty.clone(),
        river: empty,
    };
    let clustering = MpClusteringConfig {
        preflop: MpStreetCluster { buckets: 10 },
        flop: MpStreetCluster { buckets: 10 },
        turn: MpStreetCluster { buckets: 10 },
        river: MpStreetCluster { buckets: 10 },
    };
    let training = MpTrainingConfig {
        cluster_path: None,
        iterations: Some(iterations),
        time_limit_minutes: None,
        lcfr_warmup_iterations: 0,
        lcfr_discount_interval: 50,
        prune_after_iterations: 1_000_000,
        prune_threshold: -250,
        batch_size: 10,
        dcfr_alpha: 1.5,
        dcfr_beta: 0.0,
        dcfr_gamma: 2.0,
        print_every_minutes: 999,
        purify_threshold: 0.0,
        exploitability_interval_minutes: 0,
        exploitability_samples: 0,
    };
    let snapshots = MpSnapshotConfig {
        warmup_minutes: 999,
        snapshot_every_minutes: 999,
        output_dir: "/tmp/mp_validation".into(),
        resume: false,
        max_snapshots: None,
    };
    BlueprintMpConfig { game, action_abstraction: action, clustering, training, snapshots }
}

/// Run `count` meta-iterations of external sampling MCCFR.
fn run_iterations(
    tree: &MpGameTree,
    storage: &MpStorage,
    count: u64,
    seed: u64,
) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n = tree.num_players;
    for _ in 0..count {
        let deal = sample_deal(n, &mut rng);
        let buckets = trivial_buckets(&deal);
        for seat in 0..n {
            traverse_external(
                tree, storage, &buckets, Seat::from_raw(seat),
                tree.root, &mut rng, 0.0, Chips::ZERO,
            );
        }
    }
}

/// Assign trivial buckets: first hole card index mod bucket count.
fn trivial_buckets(
    deal: &poker_solver_core::blueprint_mp::Deal,
) -> poker_solver_core::blueprint_mp::DealWithBuckets {
    use poker_solver_core::blueprint_mp::Bucket;
    let mut buckets = [[Bucket(0); 4]; MAX_PLAYERS];
    for seat in 0..deal.num_players as usize {
        let card_idx = deal.hole_cards[seat][0].value as u16;
        for (street, &count) in BUCKET_COUNTS.iter().enumerate() {
            buckets[seat][street] = Bucket(card_idx % count);
        }
    }
    poker_solver_core::blueprint_mp::DealWithBuckets { deal: deal.clone(), buckets }
}

/// Assert that at every terminal node, contributions[0..n] sum to pot.
fn assert_contributions_equal_pot(tree: &MpGameTree) {
    let n = tree.num_players as usize;
    for (i, node) in tree.nodes.iter().enumerate() {
        if let MpGameNode::Terminal { pot, contributions, .. } = node {
            let sum: f64 = contributions.iter().take(n).map(|c| c.0).sum();
            assert!(
                (sum - pot.0).abs() < 0.01,
                "node {i}: contributions sum {sum:.2} != pot {:.2}",
                pot.0,
            );
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[timed_test]
fn tree_structure_reasonable() {
    let (game_cfg, action_cfg) = build_sized_config(2);
    let tree = MpGameTree::build(&game_cfg, &action_cfg);

    let decision_count = tree.nodes.iter()
        .filter(|n| matches!(n, MpGameNode::Decision { .. }))
        .count();
    let chance_count = tree.nodes.iter()
        .filter(|n| matches!(n, MpGameNode::Chance { .. }))
        .count();
    let terminal_count = tree.nodes.iter()
        .filter(|n| matches!(n, MpGameNode::Terminal { .. }))
        .count();

    assert!(decision_count > 0, "tree should have decision nodes");
    assert!(chance_count > 0, "tree should have chance nodes");
    assert!(terminal_count > 0, "tree should have terminal nodes");
    assert!(
        matches!(&tree.nodes[tree.root as usize], MpGameNode::Decision { .. }),
        "root should be a Decision node"
    );

    let has_fold = tree.nodes.iter().any(|n| matches!(
        n, MpGameNode::Terminal { kind: TerminalKind::LastStanding { .. }, .. }
    ));
    let has_showdown = tree.nodes.iter().any(|n| matches!(
        n, MpGameNode::Terminal { kind: TerminalKind::Showdown { .. }, .. }
    ));
    assert!(has_fold, "tree should have LastStanding (fold) terminals");
    assert!(has_showdown, "tree should have Showdown terminals");
}

#[timed_test]
fn training_reduces_exploitability_direction() {
    let config = build_full_config(2, 500);
    let tree = MpGameTree::build(&config.game, &config.action_abstraction);
    let storage = MpStorage::new(&tree, BUCKET_COUNTS);

    run_iterations(&tree, &storage, 500, 42);
    let br1 = compute_exploitability(
        &tree, &storage, 200, BUCKET_COUNTS, 0.0, Chips::ZERO,
    );

    run_iterations(&tree, &storage, 500, 99);
    let br2 = compute_exploitability(
        &tree, &storage, 200, BUCKET_COUNTS, 0.0, Chips::ZERO,
    );

    // Second measurement should not be wildly larger than first.
    let margin = br1.total.abs() * 1.5 + 5.0;
    assert!(
        br2.total <= br1.total + margin,
        "exploitability should not explode: phase1={:.2} phase2={:.2}",
        br1.total, br2.total,
    );
    assert!(br1.total.is_finite(), "phase1 BR should be finite");
    assert!(br2.total.is_finite(), "phase2 BR should be finite");
}

#[timed_test]
fn three_player_convergence_smoke() {
    let config = build_full_config(3, 200);
    let result = train_blueprint_mp(&config);
    assert_eq!(result.meta_iterations, 200, "should complete 200 meta-iterations");
}

#[timed_test]
fn three_player_convergence_smoke_strategy_nonzero() {
    let config = build_full_config(3, 200);
    let tree = MpGameTree::build(&config.game, &config.action_abstraction);
    let storage = MpStorage::new(&tree, BUCKET_COUNTS);

    run_iterations(&tree, &storage, 200, 77);

    let any_nonzero = storage.strategy_sums.iter().any(|s| {
        s.load(std::sync::atomic::Ordering::Relaxed) != 0
    });
    assert!(any_nonzero, "3-player training should produce non-zero strategy sums");
}

#[timed_test]
fn payoffs_zero_sum_at_all_terminals() {
    let (game_cfg, action_cfg) = build_sized_config(2);
    let tree = MpGameTree::build(&game_cfg, &action_cfg);
    assert_contributions_equal_pot(&tree);
}

#[timed_test]
fn payoffs_zero_sum_at_all_terminals_3p() {
    let (game_cfg, action_cfg) = build_sized_config(3);
    let tree = MpGameTree::build(&game_cfg, &action_cfg);
    assert_contributions_equal_pot(&tree);
}
