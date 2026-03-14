use criterion::{criterion_group, criterion_main, Criterion};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use cfvnet::config::CfvnetConfig;
use cfvnet::datagen::sampler::sample_situation;
use cfvnet::datagen::solver::{SolveConfig, solve_situation};

fn load_config() -> CfvnetConfig {
    // Use the sample river config if available, otherwise a minimal default.
    let yaml = std::fs::read_to_string("../../sample_configurations/river_cfvnet.yaml")
        .unwrap_or_else(|_| {
            r#"
game:
  initial_stack: 200
  bet_sizes: ["50%", "100%", "a"]
datagen:
  num_samples: 1
  solver_iterations: 200
  target_exploitability: 0.01
"#
            .to_string()
        });
    serde_yaml::from_str(&yaml).unwrap()
}

fn make_solve_config(cfg: &CfvnetConfig) -> SolveConfig {
    let bet_str = cfg.game.bet_sizes.join(",");
    let bet_sizes = range_solver::bet_size::BetSizeOptions::try_from((bet_str.as_str(), ""))
        .expect("valid bet sizes");
    SolveConfig {
        bet_sizes,
        solver_iterations: cfg.datagen.solver_iterations,
        target_exploitability: cfg.datagen.target_exploitability,
        add_allin_threshold: cfg.game.add_allin_threshold,
        force_allin_threshold: cfg.game.force_allin_threshold,
    }
}

fn bench_solve_situation(c: &mut Criterion) {
    let cfg = load_config();
    let solve_config = make_solve_config(&cfg);
    let board_size = cfg.game.board_size;

    // Pre-sample a situation so we benchmark solving, not sampling.
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let situation = sample_situation(&cfg.datagen, cfg.game.initial_stack, board_size, &mut rng);

    c.bench_function("solve_situation", |b| {
        b.iter(|| {
            solve_situation(criterion::black_box(&situation), criterion::black_box(&solve_config))
        })
    });
}

fn bench_sample_and_solve(c: &mut Criterion) {
    let cfg = load_config();
    let solve_config = make_solve_config(&cfg);
    let board_size = cfg.game.board_size;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    c.bench_function("sample_and_solve", |b| {
        b.iter(|| {
            let situation = sample_situation(
                criterion::black_box(&cfg.datagen),
                cfg.game.initial_stack,
                board_size,
                &mut rng,
            );
            if situation.effective_stack > 0 {
                let _ = solve_situation(&situation, &solve_config);
            }
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_solve_situation, bench_sample_and_solve
}
criterion_main!(benches);
