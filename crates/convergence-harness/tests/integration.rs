use convergence_harness::baseline::Baseline;
use convergence_harness::game::FlopPokerConfig;
use convergence_harness::harness;
use convergence_harness::solvers::mccfr::{compute_mccfr_exploitability, MccfrSolver};
use convergence_harness::solver_trait::ConvergenceSolver;

/// Full end-to-end pipeline test: generate baseline with minimal config,
/// save to disk, load back, and verify all artifacts.
///
/// Uses a tiny game (10bb effective, all-in only) with 5 iterations
/// so the test completes quickly. Verifies:
/// - All expected files are written
/// - Round-trip save/load preserves data
/// - Strategy is non-empty with valid probabilities in [0, 1]
/// - Convergence curve has samples (at least initial + final)
/// - Combo EVs are non-empty with both OOP and IP vectors
#[test]
fn end_to_end_baseline_pipeline() {
    let config = FlopPokerConfig {
        effective_stack: 10,
        bet_sizes: "a".into(),
        raise_sizes: "a".into(),
        add_allin_threshold: 0.0,
        force_allin_threshold: 0.0,
        ..Default::default()
    };

    let baseline = harness::generate_baseline_with_config(&config, 5, 100.0).unwrap();

    let dir = tempfile::TempDir::new().unwrap();
    baseline.save(dir.path()).unwrap();

    // Verify files exist
    assert!(dir.path().join("summary.json").exists());
    assert!(dir.path().join("convergence.csv").exists());
    assert!(dir.path().join("strategy.bin").exists());
    assert!(dir.path().join("combo_ev.bin").exists());

    // Load back and verify round-trip
    let loaded = Baseline::load(dir.path()).unwrap();
    assert_eq!(loaded.summary.solver_name, "Exhaustive DCFR");
    assert!(loaded.summary.total_iterations > 0);

    // Strategy: non-empty, valid probabilities
    assert!(!loaded.strategy.is_empty(), "Strategy should not be empty");
    for strat in loaded.strategy.values() {
        for &v in strat {
            assert!(v >= 0.0, "Strategy probability should be >= 0, got {}", v);
            assert!(
                v <= 1.0001,
                "Strategy probability should be <= 1, got {}",
                v
            );
        }
    }

    // Convergence curve: at least initial + final sample
    assert!(
        loaded.convergence_curve.len() >= 2,
        "Expected at least 2 convergence samples (initial + final), got {}",
        loaded.convergence_curve.len()
    );
    assert_eq!(loaded.convergence_curve[0].iteration, 0);

    // Combo EVs: non-empty, both players present
    assert!(
        !loaded.combo_evs.is_empty(),
        "Combo EVs should not be empty"
    );
    for [oop_evs, ip_evs] in loaded.combo_evs.values() {
        assert!(!oop_evs.is_empty(), "OOP EVs should not be empty");
        assert!(!ip_evs.is_empty(), "IP EVs should not be empty");
    }
}

/// Full MCCFR pipeline end-to-end test:
/// 1. Generate a baseline with exhaustive DCFR
/// 2. Run MCCFR solver for a few steps
/// 3. Compute exploitability (verifies strategy injection works)
/// 4. Extract strategy (verifies lifting works)
///
/// Uses all-in-only config for tree correspondence between
/// blueprint and range-solver trees.
#[test]
fn test_mccfr_pipeline_end_to_end() {
    // Use all-in-only config for tree correspondence
    let config = FlopPokerConfig {
        effective_stack: 10,
        bet_sizes: "a".into(),
        raise_sizes: "a".into(),
        ..Default::default()
    };

    // 1. Generate baseline
    let baseline = harness::generate_baseline_with_config(&config, 20, 10.0).unwrap();
    assert!(!baseline.strategy.is_empty());

    // 2. Run MCCFR with real clustering (small bucket counts for speed)
    let mut solver = MccfrSolver::new(config.clone(), 10, 10);
    for _ in 0..5 {
        solver.solve_step();
    }

    // 3. Compute exploitability
    let expl = compute_mccfr_exploitability(&solver, &config).unwrap();
    assert!(expl > 0.0, "Exploitability should be positive, got {}", expl);
    assert!(expl.is_finite(), "Exploitability should be finite, got {}", expl);

    // 4. Extract strategy
    let strategy = solver.average_strategy();
    assert!(!strategy.is_empty(), "Strategy should not be empty after MCCFR training");
}

/// Test run_mccfr_solver produces a Baseline with convergence data and can
/// be saved/loaded for comparison.
#[test]
fn test_run_mccfr_solver_produces_baseline() {
    let checkpoints = vec![1000, 3000, 5000];
    let result = convergence_harness::harness::run_mccfr_solver(5000, &checkpoints, None, 10, 10).unwrap();

    // Should have convergence samples at each checkpoint
    assert!(
        !result.convergence_curve.is_empty(),
        "Convergence curve should not be empty"
    );

    // Final exploitability is 0.0 when no baseline is provided (no h2h computation)
    assert!(
        result.summary.final_exploitability.is_finite(),
        "Final exploitability should be finite"
    );

    // Solver name should identify MCCFR with bucket counts
    assert!(
        result.summary.solver_name.contains("MCCFR"),
        "Solver name should contain MCCFR, got: {}",
        result.summary.solver_name
    );
    assert!(
        result.summary.solver_name.contains("10t/10r"),
        "Solver name should contain bucket counts, got: {}",
        result.summary.solver_name
    );

    // Strategy should be non-empty
    assert!(
        !result.strategy.is_empty(),
        "Strategy should not be empty"
    );

    // Should round-trip through save/load
    let dir = tempfile::TempDir::new().unwrap();
    result.save(dir.path()).unwrap();
    let loaded = Baseline::load(dir.path()).unwrap();
    assert_eq!(loaded.summary.solver_name, result.summary.solver_name);
    assert_eq!(loaded.summary.total_iterations, result.summary.total_iterations);
}
