#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(poker_solver_tauri::ExplorationState::default())
        .manage(poker_solver_tauri::SimulationState::default())
        .manage(Arc::new(poker_solver_tauri::PostflopState::default()))
        .invoke_handler(tauri::generate_handler![
            // Exploration commands
            poker_solver_tauri::load_bundle,
            poker_solver_tauri::load_preflop_solve,
            poker_solver_tauri::solve_preflop_live,
            poker_solver_tauri::load_blueprint_v2,
            poker_solver_tauri::load_subgame_source,
            poker_solver_tauri::get_strategy_matrix,
            poker_solver_tauri::get_available_actions,
            poker_solver_tauri::is_bundle_loaded,
            poker_solver_tauri::get_bundle_info,
            poker_solver_tauri::canonicalize_board,
            poker_solver_tauri::start_bucket_computation,
            poker_solver_tauri::get_computation_status,
            poker_solver_tauri::is_board_cached,
            poker_solver_tauri::list_agents,
            poker_solver_tauri::list_blueprints,
            poker_solver_tauri::get_combo_classes,
            poker_solver_tauri::get_hand_equity,
            poker_solver_tauri::get_preflop_ranges,
            // Simulation commands
            poker_solver_tauri::list_strategy_sources,
            poker_solver_tauri::start_simulation,
            poker_solver_tauri::stop_simulation,
            poker_solver_tauri::get_simulation_result,
            // Postflop solver commands
            poker_solver_tauri::postflop_set_config,
            poker_solver_tauri::postflop_set_filtered_weights,
            poker_solver_tauri::postflop_solve_street,
            poker_solver_tauri::postflop_cancel_solve,
            poker_solver_tauri::postflop_get_progress,
            poker_solver_tauri::postflop_play_action,
            poker_solver_tauri::postflop_navigate_to,
            poker_solver_tauri::postflop_close_street,
            poker_solver_tauri::postflop_set_cache_dir,
            poker_solver_tauri::postflop_check_cache,
            poker_solver_tauri::postflop_load_cached,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
