//! Reusable resolution functions for TUI scenario and audit configuration.
//!
//! Extracts the scenario/audit resolution logic so it can be called both at
//! startup and on live config reload.

use poker_solver_core::blueprint_v2::game_tree::{GameNode, GameTree};
use poker_solver_core::blueprint_v2::storage::BlueprintStorage;
use poker_solver_core::poker::Card;

use crate::blueprint_tui::ResolvedScenario;
use crate::blueprint_tui_audit;
use crate::blueprint_tui_audit_widget;
use crate::blueprint_tui_config::{RegretAuditConfig, ScenarioConfig};
use crate::blueprint_tui_scenarios;
use crate::blueprint_tui_widgets::HandGridState;

/// Result of resolving scenario configs against a game tree.
pub struct ResolvedScenarios {
    pub scenarios: Vec<ResolvedScenario>,
    pub boards: Vec<Vec<Card>>,
}

/// Result of resolving audit configs against a game tree.
pub struct ResolvedAudits {
    pub metas: Vec<blueprint_tui_audit_widget::AuditMeta>,
    pub audits: Vec<blueprint_tui_audit::ResolvedRegretAudit>,
    pub panel: Option<blueprint_tui_audit_widget::AuditPanelState>,
}

/// Resolve scenario configurations into concrete game-tree nodes and boards.
///
/// Invalid spots fall back to the tree root with an error message attached.
pub fn resolve_scenarios(
    tree: &GameTree,
    storage: &BlueprintStorage,
    configs: &[ScenarioConfig],
) -> ResolvedScenarios {
    let (scenarios, boards): (Vec<ResolvedScenario>, Vec<Vec<Card>>) = configs
        .iter()
        .map(|sc| {
            let (node_idx, board, error_message) =
                match blueprint_tui_scenarios::resolve_spot(tree, &sc.spot) {
                    Some((idx, board)) => (idx, board, None),
                    None => {
                        let msg = format!("Spot failed to resolve: {}", sc.spot);
                        eprintln!("WARNING: scenario '{}': {msg}", sc.name);
                        (tree.root, vec![], Some(msg))
                    }
                };
            let grid = blueprint_tui_scenarios::extract_strategy_grid(
                tree, storage, node_idx, &board, None,
            );
            let street_label = match &tree.nodes[node_idx as usize] {
                GameNode::Decision { street, .. } => format!("{street:?}"),
                _ => "Preflop".to_string(),
            };
            let scenario = ResolvedScenario {
                name: sc.name.clone(),
                node_idx,
                grid: HandGridState {
                    cells: grid,
                    prev_cells: None,
                    scenario_name: sc.name.clone(),
                    action_path: vec![sc.spot.clone()],
                    board_display: if board.is_empty() {
                        None
                    } else {
                        Some(
                            board
                                .iter()
                                .map(|c| format!("{c}"))
                                .collect::<Vec<_>>()
                                .join(" "),
                        )
                    },
                    cluster_id: None,
                    street_label,
                    iteration_at_snapshot: 0,
                    error_message,
                },
            };
            (scenario, board)
        })
        .unzip();

    ResolvedScenarios { scenarios, boards }
}

/// Resolve regret audit configurations into concrete storage coordinates.
///
/// Returns metas, live audit state, and an optional panel for the TUI.
pub fn resolve_audits(
    tree: &GameTree,
    storage: &BlueprintStorage,
    configs: &[RegretAuditConfig],
    sparkline_window: usize,
) -> ResolvedAudits {
    let (metas, audits): (Vec<_>, Vec<_>) = configs
        .iter()
        .map(|ac| {
            let audit = blueprint_tui_audit::resolve_regret_audit(
                tree,
                storage,
                &ac.name,
                &ac.spot,
                &ac.hand,
                ac.player,
                sparkline_window,
            );
            let meta = blueprint_tui_audit_widget::AuditMeta {
                name: ac.name.clone(),
                spot: ac.spot.clone(),
                hand: ac.hand.clone(),
                player: ac.player,
                bucket_trail: audit.bucket_trail.clone(),
                action_labels: audit.action_labels.clone(),
                error: audit.error.clone(),
            };
            (meta, audit)
        })
        .unzip();

    let panel = if !metas.is_empty() {
        let initial_snapshots: Vec<_> = audits.iter().map(|a| a.snapshot()).collect();
        Some(blueprint_tui_audit_widget::AuditPanelState {
            metas: metas.clone(),
            snapshots: initial_snapshots,
            active_tab: 0,
            iteration: 0,
        })
    } else {
        None
    };

    ResolvedAudits {
        metas,
        audits,
        panel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_tui_config::{PlayerLabel, RegretAuditConfig, ScenarioConfig};
    use test_macros::timed_test;

    fn toy_tree() -> GameTree {
        GameTree::build(
            20.0,
            1.0,
            2.0,
            &[vec!["5bb".into()]],
            &[vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
        )
    }

    #[timed_test(10)]
    fn resolve_scenarios_empty_config() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let result = resolve_scenarios(&tree, &storage, &[]);
        assert!(result.scenarios.is_empty());
        assert!(result.boards.is_empty());
    }

    #[timed_test(10)]
    fn resolve_scenarios_root_spot() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let configs = vec![ScenarioConfig {
            name: "SB open".to_string(),
            spot: String::new(),
        }];
        let result = resolve_scenarios(&tree, &storage, &configs);
        assert_eq!(result.scenarios.len(), 1);
        assert_eq!(result.boards.len(), 1);
        assert_eq!(result.scenarios[0].name, "SB open");
        assert_eq!(result.scenarios[0].node_idx, tree.root);
    }

    #[timed_test(10)]
    fn resolve_scenarios_invalid_spot_still_produces_entry() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let configs = vec![ScenarioConfig {
            name: "bad spot".to_string(),
            spot: "sb:999bb".to_string(),
        }];
        let result = resolve_scenarios(&tree, &storage, &configs);
        assert_eq!(result.scenarios.len(), 1);
        assert_eq!(result.scenarios[0].node_idx, tree.root);
        assert!(result.scenarios[0].grid.error_message.is_some());
    }

    #[timed_test(10)]
    fn resolve_scenarios_boards_match_count() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let configs = vec![
            ScenarioConfig {
                name: "A".to_string(),
                spot: String::new(),
            },
            ScenarioConfig {
                name: "B".to_string(),
                spot: String::new(),
            },
        ];
        let result = resolve_scenarios(&tree, &storage, &configs);
        assert_eq!(result.scenarios.len(), 2);
        assert_eq!(result.boards.len(), 2);
    }

    #[timed_test(10)]
    fn resolve_audits_empty_config() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let result = resolve_audits(&tree, &storage, &[], 60);
        assert!(result.metas.is_empty());
        assert!(result.audits.is_empty());
        assert!(result.panel.is_none());
    }

    #[timed_test(10)]
    fn resolve_audits_creates_panel() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let configs = vec![RegretAuditConfig {
            name: "AKo SB open".to_string(),
            spot: String::new(),
            hand: "AKo".to_string(),
            player: PlayerLabel::Sb,
        }];
        let result = resolve_audits(&tree, &storage, &configs, 60);
        assert_eq!(result.metas.len(), 1);
        assert_eq!(result.audits.len(), 1);
        assert!(result.panel.is_some());
        let panel = result.panel.unwrap();
        assert_eq!(panel.metas.len(), 1);
        assert_eq!(panel.metas[0].name, "AKo SB open");
        assert_eq!(panel.active_tab, 0);
    }

    #[timed_test(10)]
    fn resolve_audits_panel_has_initial_snapshots() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let configs = vec![RegretAuditConfig {
            name: "AKo SB open".to_string(),
            spot: String::new(),
            hand: "AKo".to_string(),
            player: PlayerLabel::Sb,
        }];
        let result = resolve_audits(&tree, &storage, &configs, 60);
        let panel = result.panel.unwrap();
        assert_eq!(panel.snapshots.len(), 1);
        // Initial regrets should all be zero
        for &r in &panel.snapshots[0].regrets {
            assert_eq!(r, 0.0);
        }
    }
}
