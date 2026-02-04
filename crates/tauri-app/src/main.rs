#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            poker_solver_tauri::run_kuhn_training,
            poker_solver_tauri::train_with_checkpoints
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
