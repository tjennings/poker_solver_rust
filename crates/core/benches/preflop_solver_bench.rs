//! Benchmarks for PreflopSolver to track allocation optimization impact.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use poker_solver_core::preflop::{PreflopConfig, PreflopSolver};

/// A minimal config with a tiny tree (matches unit test tiny_config).
fn tiny_config() -> PreflopConfig {
    let mut config = PreflopConfig::heads_up(3);
    config.raise_sizes = vec![vec![3.0]];
    config.raise_cap = 1;
    config
}

fn bench_preflop_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("preflop_solver");

    for &iters in &[50u64, 200, 500] {
        group.bench_with_input(
            BenchmarkId::new("train_tiny", iters),
            &iters,
            |b, &iters| {
                b.iter(|| {
                    let config = tiny_config();
                    let mut solver = PreflopSolver::new(&config);
                    solver.train(iters);
                    solver.iteration() // prevent dead-code elimination
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_preflop_training);
criterion_main!(benches);
