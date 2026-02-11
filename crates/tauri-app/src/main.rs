#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(poker_solver_tauri::ExplorationState::default())
        .manage(poker_solver_tauri::SimulationState::default())
        .invoke_handler(tauri::generate_handler![
            // Exploration commands
            poker_solver_tauri::load_bundle,
            poker_solver_tauri::get_strategy_matrix,
            poker_solver_tauri::get_available_actions,
            poker_solver_tauri::is_bundle_loaded,
            poker_solver_tauri::get_bundle_info,
            poker_solver_tauri::canonicalize_board,
            poker_solver_tauri::start_bucket_computation,
            poker_solver_tauri::get_computation_status,
            poker_solver_tauri::is_board_cached,
            poker_solver_tauri::list_agents,
            poker_solver_tauri::get_combo_classes,
            // Simulation commands
            poker_solver_tauri::list_strategy_sources,
            poker_solver_tauri::start_simulation,
            poker_solver_tauri::stop_simulation,
            poker_solver_tauri::get_simulation_result,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
