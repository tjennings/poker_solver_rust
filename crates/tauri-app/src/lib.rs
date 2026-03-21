mod exploration;
pub mod postflop;
mod simulation;

pub use exploration::{
    // Tauri command wrappers
    canonicalize_board, get_available_actions, get_bundle_info, get_combo_classes,
    get_computation_status, get_hand_equity, get_preflop_ranges, get_strategy_matrix,
    is_board_cached, is_bundle_loaded, list_agents, list_blueprints, load_blueprint_v2, load_bundle,
    start_bucket_computation,
    // Core functions (no Tauri dependency, usable from Axum or other runtimes)
    canonicalize_board_core, get_available_actions_core, get_bundle_info_core,
    get_combo_classes_core, get_computation_status_core, get_hand_equity_core,
    get_preflop_ranges_core, get_strategy_matrix_core, is_board_cached_core,
    is_bundle_loaded_core, list_blueprints_core, load_blueprint_v2_core, load_bundle_core,
    start_bucket_computation_core,
    // Helpers
    blueprint_sizes_to_range_solver,
    // Types
    BlueprintListEntry, ExplorationState, SubgameProgressEvent,
    // Additional types needed by consumers of core functions
    ActionInfo, ActionProb, BucketProgressEvent, BundleInfo, CanonicalizeResult,
    ComboGroup, ComboGroupInfo, ComputationStatus, ExplorationPosition, HandEquity,
    MatrixCell, PreflopRanges, StrategyMatrix,
};

pub use postflop::{
    postflop_check_cache, postflop_check_cache_core, postflop_close_street,
    postflop_close_street_core, postflop_get_progress, postflop_get_progress_core,
    postflop_load_cached, postflop_load_cached_core, postflop_navigate_to,
    postflop_navigate_to_core, postflop_play_action, postflop_play_action_core,
    postflop_set_cache_dir, postflop_set_cache_dir_core, postflop_set_config,
    postflop_cancel_solve, postflop_cancel_solve_core, postflop_set_config_core,
    postflop_set_filtered_weights, postflop_set_filtered_weights_core,
    postflop_solve_street, postflop_solve_street_core,
    CbvContext, PostflopState, set_cbv_context,
};

pub use simulation::{
    // Tauri command wrappers
    get_simulation_result, list_strategy_sources, start_simulation, stop_simulation,
    // Core functions (no Tauri dependency, usable from Axum or other runtimes)
    get_simulation_result_core, list_strategy_sources_core, start_simulation_core,
    stop_simulation_core,
    // Trait and types
    SimEventSink, SimProgressEvent, SimResultResponse, SimulationState, StrategySourceInfo,
};
