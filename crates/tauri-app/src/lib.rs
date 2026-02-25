mod exploration;
mod simulation;

pub use exploration::{
    // Tauri command wrappers
    canonicalize_board, get_available_actions, get_bundle_info, get_combo_classes,
    get_computation_status, get_hand_equity, get_strategy_matrix, is_board_cached,
    is_bundle_loaded, list_agents, load_bundle, load_preflop_solve, load_subgame_source,
    solve_preflop_live, start_bucket_computation,
    // Core functions (no Tauri dependency, usable from Axum or other runtimes)
    canonicalize_board_core, get_available_actions_core, get_bundle_info_core,
    get_combo_classes_core, get_computation_status_core, get_hand_equity_core,
    get_strategy_matrix_core, is_board_cached_core, is_bundle_loaded_core, load_bundle_core,
    load_preflop_solve_core, load_subgame_source_core, solve_preflop_live_core,
    start_bucket_computation_core,
    // Types
    ExplorationState, SubgameProgressEvent,
    // Additional types needed by consumers of core functions
    ActionInfo, ActionProb, BucketProgressEvent, BundleInfo, CanonicalizeResult,
    ComboGroup, ComboGroupInfo, ComputationStatus, ExplorationPosition, HandEquity,
    MatrixCell, StrategyMatrix,
};

pub use simulation::{
    get_simulation_result, list_strategy_sources, start_simulation, stop_simulation,
    SimulationState,
};
