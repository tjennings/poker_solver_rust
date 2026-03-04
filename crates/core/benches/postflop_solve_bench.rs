use std::path::Path;

use criterion::{criterion_group, criterion_main, Criterion};

use poker_solver_core::cfr::dcfr::DcfrParams;
use poker_solver_core::poker::Card;
use poker_solver_core::preflop::equity_table_cache::EquityTableCache;
use poker_solver_core::preflop::postflop_abstraction::{annotate_streets, PostflopLayout};
use poker_solver_core::preflop::postflop_exhaustive::{
    compute_equity_table, exhaustive_solve_one_flop, FlopBuffers,
};
use poker_solver_core::preflop::postflop_hands::{build_combo_map, canonical_flops, NUM_CANONICAL_HANDS};
use poker_solver_core::preflop::rank_array_cache::{derive_equity_table, RankArrayCache};
use poker_solver_core::preflop::{PostflopModelConfig, PostflopTree};

/// Pick the first canonical flop. Using a canonical flop directly guarantees
/// cache lookups match, since the caches store canonical representations.
fn bench_flop() -> [Card; 3] {
    canonical_flops()[0]
}

/// Load equity table from cache if available, otherwise compute from scratch.
///
/// Priority:
/// 1. `cache/equity_tables.bin` — extract directly (canonical key guaranteed to match)
/// 2. `cache/rank_arrays.bin`  — derive via integer comparison (no hand eval)
/// 3. Full computation          — evaluate every (turn, river) x combo pair
fn load_or_compute_equity(flop: [Card; 3], combo_map: &[Vec<(Card, Card)>]) -> Vec<f64> {
    // CARGO_MANIFEST_DIR = crates/core; go up two levels to workspace root
    let ws_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("CARGO_MANIFEST_DIR must have a grandparent (workspace root)");

    let eq_path = ws_root.join("cache/equity_tables.bin");
    if let Some(cache) = EquityTableCache::load(&eq_path) {
        if let Some(tables) = cache.extract_tables(&[flop]) {
            eprintln!("[bench] equity table loaded from cache");
            return tables.into_iter().next().expect("single table");
        }
    }

    let rank_path = ws_root.join("cache/rank_arrays.bin");
    if let Some(cache) = RankArrayCache::load(&rank_path) {
        if let Some(data) = cache.get_flop_data(&flop) {
            eprintln!("[bench] equity table derived from rank cache");
            return derive_equity_table(data, combo_map);
        }
    }

    eprintln!("[bench] no cache found — computing equity table from scratch");
    compute_equity_table(combo_map, flop)
}

fn bench_postflop_solve(c: &mut Criterion) {
    // Setup: build tree, layout, equity table (done once, outside the benchmark loop)
    let config = PostflopModelConfig {
        bet_sizes: vec![0.3, 0.5],
        max_raises_per_street: 2,
        postflop_solve_iterations: 1,
        ..PostflopModelConfig::exhaustive_fast()
    };
    let tree = PostflopTree::build_with_spr(&config, 3.0).unwrap();
    let node_streets = annotate_streets(&tree);
    let n = NUM_CANONICAL_HANDS;
    let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);

    let flop = bench_flop();
    let combo_map = build_combo_map(&flop);
    let equity_table = load_or_compute_equity(flop, &combo_map);

    let dcfr = DcfrParams::linear();

    let mut group = c.benchmark_group("postflop_solve");
    group.sample_size(10);

    group.bench_function("solve_1_iter", |b| {
        b.iter_batched(
            || FlopBuffers::new(layout.total_size),
            |mut bufs| {
                exhaustive_solve_one_flop(
                    &tree,
                    &layout,
                    &equity_table,
                    1,     // num_iterations
                    0.0,   // convergence_threshold (no early stop)
                    "bench_flop",
                    &dcfr,
                    &config,
                    &|_| {},
                    None,
                    &mut bufs,
                )
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_postflop_solve);
criterion_main!(benches);
