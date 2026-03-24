use convergence_harness::baseline::Baseline;
use convergence_harness::game::FlopPokerConfig;
use convergence_harness::harness;

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
    for (_node_id, [oop_evs, ip_evs]) in &loaded.combo_evs {
        assert!(!oop_evs.is_empty(), "OOP EVs should not be empty");
        assert!(!ip_evs.is_empty(), "IP EVs should not be empty");
    }
}
