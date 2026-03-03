//! Benchmark for `compute_equity_table` to track optimization impact.

use criterion::{criterion_group, criterion_main, Criterion};

use poker_solver_core::poker::{Card, Suit, Value};
use poker_solver_core::preflop::postflop_exhaustive::compute_equity_table;
use poker_solver_core::preflop::postflop_hands::build_combo_map;
use poker_solver_core::preflop::rank_array_cache::{compute_rank_arrays, derive_equity_table};

fn flop_akq() -> [Card; 3] {
    [
        Card::new(Value::Ace, Suit::Heart),
        Card::new(Value::King, Suit::Spade),
        Card::new(Value::Queen, Suit::Diamond),
    ]
}

fn bench_equity_table(c: &mut Criterion) {
    let mut group = c.benchmark_group("equity_table");
    group.sample_size(10);

    let flop = flop_akq();
    let combo_map = build_combo_map(&flop);

    group.bench_function("AKQr", |b| {
        b.iter(|| compute_equity_table(&combo_map, flop));
    });

    // Bench rank array computation (Phase 1: hand evaluation only)
    group.bench_function("AKQr_rank_arrays", |b| {
        b.iter(|| compute_rank_arrays(&combo_map, flop));
    });

    // Bench equity derivation from cached ranks (Phase 2: integer comparison only)
    let rank_data = compute_rank_arrays(&combo_map, flop);
    group.bench_function("AKQr_derive_equity", |b| {
        b.iter(|| derive_equity_table(&rank_data, &combo_map));
    });

    group.finish();
}

criterion_group!(benches, bench_equity_table);
criterion_main!(benches);
