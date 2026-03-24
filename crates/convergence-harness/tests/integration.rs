use convergence_harness::baseline::Baseline;
use convergence_harness::game::FlopPokerConfig;
use convergence_harness::harness;

/// Full pipeline: generate baseline with minimal config, save, load, verify.
///
/// Uses a tiny game (10bb effective, all-in only) and few iterations
/// so the test completes quickly.
#[test]
fn end_to_end_baseline_generation_and_reload() {
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

    // Load back
    let loaded = Baseline::load(dir.path()).unwrap();
    assert_eq!(loaded.summary.solver_name, "Exhaustive DCFR");
    assert!(loaded.summary.total_iterations > 0);
}

#[test]
fn end_to_end_strategy_has_valid_probabilities() {
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
    let loaded = Baseline::load(dir.path()).unwrap();

    assert!(!loaded.strategy.is_empty(), "Strategy should not be empty");
    for (_node_id, strat) in &loaded.strategy {
        for &v in strat {
            assert!(v >= 0.0, "Strategy probability should be >= 0, got {}", v);
            assert!(
                v <= 1.0001,
                "Strategy probability should be <= 1, got {}",
                v
            );
        }
    }
}

#[test]
fn end_to_end_convergence_curve_has_samples() {
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
    let loaded = Baseline::load(dir.path()).unwrap();

    assert!(
        loaded.convergence_curve.len() >= 2,
        "Expected at least 2 convergence samples (initial + final), got {}",
        loaded.convergence_curve.len()
    );

    // First sample should be iteration 0
    assert_eq!(loaded.convergence_curve[0].iteration, 0);
}

#[test]
fn end_to_end_combo_evs_are_nonempty() {
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
    let loaded = Baseline::load(dir.path()).unwrap();

    assert!(
        !loaded.combo_evs.is_empty(),
        "Combo EVs should not be empty"
    );

    for (_node_id, [oop_evs, ip_evs]) in &loaded.combo_evs {
        assert!(!oop_evs.is_empty(), "OOP EVs should not be empty");
        assert!(!ip_evs.is_empty(), "IP EVs should not be empty");
    }
}
