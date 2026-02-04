#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            poker_solver_tauri::run_kuhn_training
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
