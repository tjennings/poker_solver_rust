//! Integration test for the blueprint explorer "Load Strategy" flow.
//!
//! Requires a trained blueprint directory on disk. Run with:
//! ```
//! BLUEPRINT_TEST_DIR=/path/to/blueprint cargo test -p poker-solver-tauri -- --ignored
//! ```

use poker_solver_tauri::{ExplorationState, PreflopRanges};

/// Resolve the blueprint directory from the environment or fall back to a
/// well-known relative path. Returns `None` if the directory does not exist.
fn blueprint_dir() -> Option<String> {
    let dir = std::env::var("BLUEPRINT_TEST_DIR")
        .unwrap_or_else(|_| "../../blueprints".to_string());

    if std::path::Path::new(&dir).exists() {
        Some(dir)
    } else {
        None
    }
}

/// Find the first sub-directory inside `dir` that contains a `config.yaml`
/// (the marker for a valid blueprint). Returns the path as a `String`.
fn find_first_blueprint(dir: &str) -> Option<String> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && path.join("config.yaml").exists() {
            return path.to_str().map(String::from);
        }
    }
    // The dir itself might be a blueprint.
    if std::path::Path::new(dir).join("config.yaml").exists() {
        return Some(dir.to_string());
    }
    None
}

#[tokio::test]
#[ignore] // Requires trained blueprint — run with: cargo test -p poker-solver-tauri -- --ignored
async fn test_load_strategy_and_get_preflop_ranges() {
    let Some(base_dir) = blueprint_dir() else {
        eprintln!("Skipping: no blueprint directory found (set BLUEPRINT_TEST_DIR)");
        return;
    };

    let Some(blueprint_path) = find_first_blueprint(&base_dir) else {
        eprintln!(
            "Skipping: no blueprint with config.yaml found in {}",
            base_dir
        );
        return;
    };

    // --- Step 1: Load the blueprint ---
    let state = ExplorationState::default();

    let bundle_info = poker_solver_tauri::load_blueprint_v2_core(&state, blueprint_path.clone())
        .await
        .unwrap_or_else(|e| panic!("load_blueprint_v2_core failed for {blueprint_path}: {e}"));

    assert!(
        bundle_info.stack_depth > 0,
        "stack_depth should be positive, got {}",
        bundle_info.stack_depth
    );
    assert!(
        bundle_info.info_sets > 0,
        "info_sets should be positive, got {}",
        bundle_info.info_sets
    );

    // --- Step 2: Get preflop ranges for a simple action sequence ---
    // Try "c" (call/check) first — SB limps. This should always be valid in
    // standard HU preflop trees where SB acts first with fold/call/raise.
    let history_to_try: &[&[&str]] = &[
        &["c", "c"],       // SB limps, BB checks → both reach flop
        &["c"],            // SB limps (single action)
        &["r:0", "c"],     // SB raises (smallest), BB calls
    ];

    let mut found_valid = false;

    for actions in history_to_try {
        let history: Vec<String> = actions.iter().map(|s| s.to_string()).collect();

        match poker_solver_tauri::get_preflop_ranges_core(&state, history.clone()) {
            Ok(ranges) => {
                validate_preflop_ranges(&ranges, actions);
                found_valid = true;
                break;
            }
            Err(e) => {
                eprintln!(
                    "History {:?} not valid for this tree: {}",
                    actions, e
                );
            }
        }
    }

    assert!(
        found_valid,
        "None of the attempted action histories produced valid preflop ranges"
    );
}

/// Validate structural invariants of a `PreflopRanges` result.
fn validate_preflop_ranges(ranges: &PreflopRanges, history: &[&str]) {
    assert_eq!(
        ranges.oop_weights.len(),
        1326,
        "OOP weights should have 1326 combos, got {}",
        ranges.oop_weights.len()
    );
    assert_eq!(
        ranges.ip_weights.len(),
        1326,
        "IP weights should have 1326 combos, got {}",
        ranges.ip_weights.len()
    );

    let oop_sum: f32 = ranges.oop_weights.iter().sum();
    let ip_sum: f32 = ranges.ip_weights.iter().sum();

    assert!(
        oop_sum > 0.0,
        "OOP weight sum should be positive for history {history:?}, got {oop_sum}"
    );
    assert!(
        ip_sum > 0.0,
        "IP weight sum should be positive for history {history:?}, got {ip_sum}"
    );

    assert!(
        ranges.pot > 0.0,
        "pot should be positive, got {}",
        ranges.pot
    );
    assert!(
        ranges.effective_stack > 0.0,
        "effective_stack should be positive, got {}",
        ranges.effective_stack
    );

    // Weights should be in [0, 1] range (they are cumulative products of probabilities).
    for (i, &w) in ranges.oop_weights.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&w),
            "OOP weight[{i}] = {w} is out of [0, 1]"
        );
    }
    for (i, &w) in ranges.ip_weights.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&w),
            "IP weight[{i}] = {w} is out of [0, 1]"
        );
    }

    eprintln!(
        "Validated ranges for history {:?}: pot={}, eff_stack={}, oop_sum={:.1}, ip_sum={:.1}",
        history, ranges.pot, ranges.effective_stack, oop_sum, ip_sum
    );
}
