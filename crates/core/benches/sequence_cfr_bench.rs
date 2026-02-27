//! Benchmarks for SequenceCfrSolver to track allocation optimization impact.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use poker_solver_core::cfr::game_tree::materialize;
use poker_solver_core::cfr::{DealInfo, SequenceCfrConfig, SequenceCfrSolver};
use poker_solver_core::game::{Game, KuhnPoker};
use poker_solver_core::info_key::InfoKey;

/// Build per-deal trees and DealInfo for all 6 Kuhn deals.
fn kuhn_trees_and_deals() -> (Vec<poker_solver_core::cfr::GameTree>, Vec<DealInfo>) {
    let game = KuhnPoker::new();
    let states = game.initial_states();
    let mut trees = Vec::new();
    let mut deals = Vec::new();

    for state in &states {
        let tree = materialize(&game, state);
        let key_p1 = game.info_set_key(state);
        let hand_bits_p1 = InfoKey::from_raw(key_p1).hand_bits();

        let next = game.next_state(state, game.actions(state)[0]);
        let key_p2 = game.info_set_key(&next);
        let hand_bits_p2 = InfoKey::from_raw(key_p2).hand_bits();

        let p1_equity = if hand_bits_p1 > hand_bits_p2 {
            1.0
        } else {
            0.0
        };

        trees.push(tree);
        deals.push(DealInfo {
            hand_bits_p1: [hand_bits_p1; 4],
            hand_bits_p2: [hand_bits_p2; 4],
            p1_equity,
            weight: 1.0,
        });
    }

    (trees, deals)
}

fn bench_kuhn_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequence_cfr_kuhn");

    for &iters in &[100u64, 1000, 5000] {
        group.bench_with_input(
            BenchmarkId::new("train", iters),
            &iters,
            |b, &iters| {
                b.iter(|| {
                    let (trees, deals) = kuhn_trees_and_deals();
                    let config = SequenceCfrConfig::default();
                    let mut solver =
                        SequenceCfrSolver::from_per_deal_trees(trees, deals, config);
                    solver.train(iters);
                    solver.iterations() // prevent dead-code elimination
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_kuhn_training);
criterion_main!(benches);
