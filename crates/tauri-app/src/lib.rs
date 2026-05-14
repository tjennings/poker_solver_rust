pub mod boundary_trace;
pub mod exact_subtree;
mod exploration;
pub mod gadget;
pub mod game_session;
pub mod postflop;
mod simulation;

pub use exploration::{
    // Core functions (no Tauri dependency, usable from Axum or other runtimes)
    blueprint_propagate_ranges,
    // Tauri command wrappers
    blueprint_propagate_ranges_cmd,
    // Helpers
    blueprint_sizes_to_range_solver,
    canonicalize_board,
    canonicalize_board_core,
    get_available_actions,
    get_available_actions_core,
    get_bundle_info,
    get_bundle_info_core,
    get_combo_classes,
    get_combo_classes_core,
    get_computation_status,
    get_computation_status_core,
    get_hand_equity,
    get_hand_equity_core,
    get_preflop_ranges,
    get_preflop_ranges_core,
    get_strategy_matrix,
    get_strategy_matrix_core,
    is_board_cached,
    is_board_cached_core,
    is_bundle_loaded,
    is_bundle_loaded_core,
    list_agents,
    list_blueprints,
    list_blueprints_core,
    list_snapshots,
    list_snapshots_core,
    load_blueprint_v2,
    load_blueprint_v2_core,
    load_bundle,
    load_bundle_core,
    load_hand_ev_bin,
    populate_cbv_context,
    range_solver_to_rs_card,
    rs_card_to_range_solver,
    start_bucket_computation,
    start_bucket_computation_core,
    // Additional types needed by consumers of core functions
    ActionInfo,
    ActionProb,
    // Types
    BlueprintListEntry,
    BucketProgressEvent,
    BundleInfo,
    CanonicalizeResult,
    ComboGroup,
    ComboGroupInfo,
    ComputationStatus,
    ExplorationPosition,
    ExplorationState,
    HandEquity,
    MatrixCell,
    PreflopRanges,
    SnapshotEntry,
    StrategyMatrix,
    SubgameProgressEvent,
};

pub use postflop::{
    compute_boundary_reach, parse_rs_poker_card, postflop_cancel_solve, postflop_cancel_solve_core,
    postflop_check_cache, postflop_check_cache_core, postflop_close_street,
    postflop_close_street_core, postflop_get_progress, postflop_get_progress_core,
    postflop_load_cached, postflop_load_cached_core, postflop_navigate_to,
    postflop_navigate_to_core, postflop_play_action, postflop_play_action_core,
    postflop_set_cache_dir, postflop_set_cache_dir_core, postflop_set_config,
    postflop_set_config_core, postflop_set_filtered_weights, postflop_set_filtered_weights_core,
    postflop_solve_street, postflop_solve_street_core, seed_solver_with_blueprint, set_cbv_context,
    CbvContext, PostflopState, RolloutLeafEvaluator,
};

pub use simulation::{
    // Tauri command wrappers
    get_simulation_result,
    // Core functions (no Tauri dependency, usable from Axum or other runtimes)
    get_simulation_result_core,
    list_strategy_sources,
    list_strategy_sources_core,
    start_simulation,
    start_simulation_core,
    stop_simulation,
    stop_simulation_core,
    // Trait and types
    SimEventSink,
    SimProgressEvent,
    SimResultResponse,
    SimulationState,
    StrategySourceInfo,
};

pub use game_session::{
    build_solve_game,
    build_solve_game_parts,
    build_solve_game_parts_with_root,
    build_solve_game_with_root,
    effective_stack_for_solve_root,
    game_back,
    game_back_core,
    game_cancel_solve,
    game_cancel_solve_core,
    game_deal_card,
    game_deal_card_core,
    game_encode_spot,
    game_encode_spot_core,
    game_get_state,
    game_get_state_core,
    game_load_spot,
    game_load_spot_core,
    // Tauri command wrappers
    game_new,
    // Core functions (no Tauri dependency, usable from Axum or other runtimes)
    game_new_core,
    game_play_action,
    game_play_action_core,
    game_solve,
    game_solve_core,
    resolve_street_boundary,
    validate_cfvnet_boundary_cut,
    BoundaryKind,
    GameAction,
    GameMatrix,
    GameMatrixCell,
    // Types
    GameSession,
    GameSessionState,
    GameState,
    SolveBoundaryEvaluator,
    SolveGameRoot,
    StreetBoundaryConfig,
    StreetBoundaryMode,
};
