//! `inspect-spot` subcommand: load a blueprint and dump strategy/EV data for a spot.

use std::path::{Path, PathBuf};

use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::config::BlueprintV2Config;
use poker_solver_core::blueprint_v2::game_tree::GameTree as V2GameTree;
use poker_solver_tauri::{GameMatrixCell, GameSession, GameState};
#[cfg(test)]
use poker_solver_tauri::GameAction;

/// Load a blueprint bundle and return the components needed for a GameSession.
///
/// Replicates the core loading logic from `load_blueprint_v2_core` without
/// Tauri async or State wrappers.
fn load_blueprint(
    config_path: &Path,
) -> Result<(BlueprintV2Config, BlueprintV2Strategy, V2GameTree, Vec<u32>), String> {
    let yaml = std::fs::read_to_string(config_path)
        .map_err(|e| format!("Failed to read config: {e}"))?;
    let config: BlueprintV2Config =
        serde_yaml::from_str(&yaml).map_err(|e| format!("Failed to parse config: {e}"))?;

    // Determine strategy path: look relative to the config file's directory
    let output_dir = PathBuf::from(&config.snapshots.output_dir);

    // Search for strategy.bin in standard locations
    let strat_path = if output_dir.join("final/strategy.bin").exists() {
        output_dir.join("final/strategy.bin")
    } else if output_dir.join("strategy.bin").exists() {
        output_dir.join("strategy.bin")
    } else {
        // Look for the latest snapshot directory
        let mut snapshots: Vec<_> = std::fs::read_dir(&output_dir)
            .map_err(|e| format!("Cannot read output directory '{}': {e}", output_dir.display()))?
            .filter_map(Result::ok)
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|n| n.starts_with("snapshot_"))
            })
            .collect();
        snapshots.sort_by_key(|e| e.file_name());
        match snapshots.last() {
            Some(entry) => entry.path().join("strategy.bin"),
            None => {
                return Err(format!(
                    "No strategy.bin found in output directory '{}'",
                    output_dir.display()
                ))
            }
        }
    };

    eprintln!("Loading strategy from {}...", strat_path.display());
    let strategy = BlueprintV2Strategy::load(&strat_path)
        .map_err(|e| format!("Failed to load strategy.bin: {e}"))?;

    let aa = &config.action_abstraction;
    let tree = V2GameTree::build(
        config.game.stack_depth,
        config.game.small_blind,
        config.game.big_blind,
        &aa.preflop,
        &aa.flop,
        &aa.turn,
        &aa.river,
    );
    let decision_map = tree.decision_index_map();

    Ok((config, strategy, tree, decision_map))
}

/// Format a strategy report from a `GameState` and print it to stdout.
fn format_report(spot: &str, state: &GameState) -> String {
    let mut out = String::new();

    // Header
    out.push_str(&format!("Spot: {spot}\n"));
    out.push_str(&format!("Street: {}\n", state.street));
    out.push_str(&format!(
        "Board: {}\n",
        if state.board.is_empty() {
            "-".to_string()
        } else {
            state.board.join(" ")
        }
    ));
    out.push_str(&format!(
        "Position: {}{}\n",
        state.position,
        if state.position.is_empty() {
            String::new()
        } else {
            " (to act)".to_string()
        }
    ));
    out.push_str(&format!(
        "Pot: {}BB | Stacks: BB {}BB / SB {}BB\n",
        state.pot / 2,
        state.stacks[0] / 2,
        state.stacks[1] / 2,
    ));

    if state.is_terminal {
        out.push_str("\n(Terminal node)\n");
        return out;
    }

    // Strategy matrix
    if let Some(ref matrix) = state.matrix {
        let actions = &matrix.actions;
        let mut hands: Vec<&GameMatrixCell> = matrix
            .cells
            .iter()
            .flatten()
            .filter(|c| c.combo_count > 0)
            .collect();

        // Sort by EV descending (best hands first)
        hands.sort_by(|a, b| {
            b.ev
                .unwrap_or(f32::NEG_INFINITY)
                .partial_cmp(&a.ev.unwrap_or(f32::NEG_INFINITY))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        out.push_str(&format!("\n=== Strategy ({} hands) ===\n", hands.len()));

        // Header row
        out.push_str(&format!("{:<8}", "Hand"));
        for action in actions {
            out.push_str(&format!("{:>8}", action.label));
        }
        out.push_str(&format!("{:>8}\n", "EV"));

        // Hand rows
        for cell in &hands {
            out.push_str(&format!("{:<8}", cell.hand));
            for prob in &cell.probabilities {
                out.push_str(&format!("{:>7.1}%", prob * 100.0));
            }
            match cell.ev {
                Some(ev) => out.push_str(&format!("{:>+8.1}\n", ev)),
                None => out.push_str(&format!("{:>8}\n", "-")),
            }
        }

        // Action summary
        out.push_str("\n=== Action Summary ===\n");
        let total_weight: f32 = hands.iter().map(|c| c.weight).sum();
        if total_weight > 0.0 {
            for (ai, action) in actions.iter().enumerate() {
                let action_weight: f32 = hands
                    .iter()
                    .map(|c| {
                        c.probabilities
                            .get(ai)
                            .copied()
                            .unwrap_or(0.0)
                            * c.weight
                    })
                    .sum::<f32>();
                let pct = action_weight / total_weight * 100.0;
                out.push_str(&format!("{:<8} {:>5.1}% of range\n", action.label, pct));
            }
        }
    } else {
        out.push_str("\n(No strategy matrix available at this node)\n");
    }

    out
}

/// Run the inspect-spot command.
pub fn run(config_path: &Path, spot: &str) -> Result<(), String> {
    let (config, strategy, tree, decision_map) = load_blueprint(config_path)?;

    eprintln!(
        "Blueprint: {} ({}BB stacks)",
        config.game.name, config.game.stack_depth
    );

    let mut session = GameSession::new(config, strategy, tree, decision_map, None);

    session.load_spot(spot)?;
    let state = session.get_state();

    let report = format_report(spot, &state);
    print!("{report}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mock_state(
        actions: Vec<GameAction>,
        cells: Vec<GameMatrixCell>,
    ) -> GameState {
        use poker_solver_tauri::{GameMatrix, GameState};
        GameState {
            street: "Flop".to_string(),
            position: "BB".to_string(),
            board: vec!["Td".to_string(), "9d".to_string(), "6h".to_string()],
            pot: 8,     // 4BB in half-BB chips
            stacks: [96, 96],
            matrix: Some(GameMatrix {
                cells: vec![vec![cells[0].clone()], vec![cells[1].clone()]],
                actions: actions.clone(),
            }),
            actions,
            action_history: vec![],
            is_terminal: false,
            is_chance: false,
            solve: None,
        }
    }

    #[test]
    fn format_report_shows_spot_header() {
        let state = make_mock_state(
            vec![
                GameAction { id: "0".into(), label: "Fold".into(), action_type: "fold".into() },
                GameAction { id: "1".into(), label: "Call".into(), action_type: "call".into() },
            ],
            vec![
                GameMatrixCell {
                    hand: "AA".into(),
                    suited: false,
                    pair: true,
                    probabilities: vec![0.0, 1.0],
                    combo_count: 6,
                    weight: 1.0,
                    ev: Some(5.2),
                    combos: vec![],
                },
                GameMatrixCell {
                    hand: "AKs".into(),
                    suited: true,
                    pair: false,
                    probabilities: vec![0.3, 0.7],
                    combo_count: 4,
                    weight: 0.8,
                    ev: Some(2.1),
                    combos: vec![],
                },
            ],
        );

        let report = format_report("sb:2bb,bb:call|Td9d6h", &state);
        assert!(report.contains("Spot: sb:2bb,bb:call|Td9d6h"));
        assert!(report.contains("Street: Flop"));
        assert!(report.contains("Board: Td 9d 6h"));
        assert!(report.contains("BB (to act)"));
        assert!(report.contains("Pot: 4BB"));
    }

    #[test]
    fn format_report_hands_sorted_by_ev_descending() {
        let state = make_mock_state(
            vec![
                GameAction { id: "0".into(), label: "Fold".into(), action_type: "fold".into() },
                GameAction { id: "1".into(), label: "Call".into(), action_type: "call".into() },
            ],
            vec![
                GameMatrixCell {
                    hand: "AKs".into(),
                    suited: true,
                    pair: false,
                    probabilities: vec![0.3, 0.7],
                    combo_count: 4,
                    weight: 0.8,
                    ev: Some(2.1),
                    combos: vec![],
                },
                GameMatrixCell {
                    hand: "AA".into(),
                    suited: false,
                    pair: true,
                    probabilities: vec![0.0, 1.0],
                    combo_count: 6,
                    weight: 1.0,
                    ev: Some(5.2),
                    combos: vec![],
                },
            ],
        );

        let report = format_report("test", &state);
        let aa_pos = report.find("AA").unwrap();
        let aks_pos = report.find("AKs").unwrap();
        assert!(
            aa_pos < aks_pos,
            "AA (EV 5.2) should appear before AKs (EV 2.1)"
        );
    }

    #[test]
    fn format_report_shows_action_summary() {
        let state = make_mock_state(
            vec![
                GameAction { id: "0".into(), label: "Fold".into(), action_type: "fold".into() },
                GameAction { id: "1".into(), label: "Call".into(), action_type: "call".into() },
            ],
            vec![
                GameMatrixCell {
                    hand: "AA".into(),
                    suited: false,
                    pair: true,
                    probabilities: vec![0.0, 1.0],
                    combo_count: 6,
                    weight: 1.0,
                    ev: Some(5.2),
                    combos: vec![],
                },
                GameMatrixCell {
                    hand: "72o".into(),
                    suited: false,
                    pair: false,
                    probabilities: vec![1.0, 0.0],
                    combo_count: 12,
                    weight: 1.0,
                    ev: Some(-3.0),
                    combos: vec![],
                },
            ],
        );

        let report = format_report("test", &state);
        assert!(report.contains("Action Summary"));
        assert!(report.contains("Fold"));
        assert!(report.contains("Call"));
        assert!(report.contains("% of range"));
    }

    #[test]
    fn format_report_terminal_node() {
        let state = GameState {
            street: "Preflop".to_string(),
            position: String::new(),
            board: vec![],
            pot: 3,
            stacks: [199, 201],
            matrix: None,
            actions: vec![],
            action_history: vec![],
            is_terminal: true,
            is_chance: false,
            solve: None,
        };

        let report = format_report("sb:fold", &state);
        assert!(report.contains("Terminal node"));
    }

    #[test]
    fn format_report_empty_board_shows_dash() {
        let state = GameState {
            street: "Preflop".to_string(),
            position: "SB".to_string(),
            board: vec![],
            pot: 3,
            stacks: [199, 199],
            matrix: None,
            actions: vec![],
            action_history: vec![],
            is_terminal: false,
            is_chance: false,
            solve: None,
        };

        let report = format_report("", &state);
        assert!(report.contains("Board: -"));
    }

    #[test]
    fn format_report_action_percentages_weighted() {
        // Two hands with equal weight 1.0
        // AA: 100% Call, 72o: 100% Fold
        // Expected: Fold 50%, Call 50%
        let state = make_mock_state(
            vec![
                GameAction { id: "0".into(), label: "Fold".into(), action_type: "fold".into() },
                GameAction { id: "1".into(), label: "Call".into(), action_type: "call".into() },
            ],
            vec![
                GameMatrixCell {
                    hand: "AA".into(),
                    suited: false,
                    pair: true,
                    probabilities: vec![0.0, 1.0],
                    combo_count: 6,
                    weight: 1.0,
                    ev: Some(5.2),
                    combos: vec![],
                },
                GameMatrixCell {
                    hand: "72o".into(),
                    suited: false,
                    pair: false,
                    probabilities: vec![1.0, 0.0],
                    combo_count: 12,
                    weight: 1.0,
                    ev: Some(-3.0),
                    combos: vec![],
                },
            ],
        );

        let report = format_report("test", &state);
        // Both hands have weight 1.0, so Fold = 50%, Call = 50%
        assert!(report.contains("50.0% of range"), "Report:\n{report}");
    }

    #[test]
    fn format_report_skips_zero_combo_hands() {
        let state = make_mock_state(
            vec![
                GameAction { id: "0".into(), label: "Fold".into(), action_type: "fold".into() },
            ],
            vec![
                GameMatrixCell {
                    hand: "AA".into(),
                    suited: false,
                    pair: true,
                    probabilities: vec![1.0],
                    combo_count: 6,
                    weight: 1.0,
                    ev: Some(5.2),
                    combos: vec![],
                },
                GameMatrixCell {
                    hand: "KK".into(),
                    suited: false,
                    pair: true,
                    probabilities: vec![1.0],
                    combo_count: 0,  // blocked
                    weight: 0.0,
                    ev: None,
                    combos: vec![],
                },
            ],
        );

        let report = format_report("test", &state);
        assert!(report.contains("1 hands"), "Report:\n{report}");
        assert!(!report.contains("KK"), "Blocked hand KK should not appear");
    }
}
