use std::sync::Arc;

use axum::{
    extract::State as AxumState,
    http::{HeaderValue, Method},
    routing::post,
    Extension, Json, Router,
};
use serde::Deserialize;
use tower_http::cors::CorsLayer;

use poker_solver_tauri::ExplorationState;
use poker_solver_tauri::PostflopState;

type AppState = Arc<ExplorationState>;

// ---------------------------------------------------------------------------
// Request param structs
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct PathParams {
    path: String,
}

#[derive(Deserialize)]
struct SolvePreflopParams {
    stack_depth: u32,
    iterations: u64,
}

#[derive(Deserialize)]
struct BlueprintPathParams {
    blueprint_path: String,
}

#[derive(Deserialize)]
struct StrategyMatrixParams {
    position: poker_solver_tauri::ExplorationPosition,
    threshold: Option<f32>,
    street_histories: Option<Vec<Vec<String>>>,
}

#[derive(Deserialize)]
struct PositionParams {
    position: poker_solver_tauri::ExplorationPosition,
}

#[derive(Deserialize)]
struct BoardParams {
    board: Vec<String>,
}

#[derive(Deserialize)]
struct CardsParams {
    cards: Vec<String>,
}

#[derive(Deserialize)]
struct ComboClassesParams {
    position: poker_solver_tauri::ExplorationPosition,
    hand: String,
}

#[derive(Deserialize)]
struct HandEquityParams {
    hand: String,
    villain_hand: Option<String>,
}

#[derive(Deserialize)]
struct ListBlueprintsParams {
    dir: String,
}

#[derive(Deserialize)]
struct PreflopRangesParams {
    history: Vec<String>,
}

#[derive(Deserialize)]
struct PostflopConfigParams {
    config: poker_solver_tauri::postflop::PostflopConfig,
}

#[derive(Deserialize)]
struct PostflopSolveParams {
    board: Vec<String>,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
}

#[derive(Deserialize)]
struct PostflopActionParams {
    action: usize,
}

#[derive(Deserialize)]
struct PostflopNavigateToParams {
    history: Vec<usize>,
}

#[derive(Deserialize)]
struct PostflopCloseStreetParams {
    action_history: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a serializable value to a `Json<serde_json::Value>`.
/// Safe because all response types implement `Serialize`.
fn to_json_value<T: serde::Serialize>(v: T) -> Json<serde_json::Value> {
    // Serialization of our known types is infallible.
    Json(serde_json::to_value(v).expect("response type implements Serialize"))
}

/// Map a `Result<T, String>` into an Axum response.
fn result_to_response<T: serde::Serialize>(
    r: Result<T, String>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    r.map(to_json_value)
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e))
}

// ---------------------------------------------------------------------------
// Handlers — async core functions
// ---------------------------------------------------------------------------

async fn handle_load_bundle(
    AxumState(state): AxumState<AppState>,
    Json(params): Json<PathParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::load_bundle_core(&state, params.path).await)
}

async fn handle_load_preflop_solve(
    AxumState(state): AxumState<AppState>,
    Json(params): Json<PathParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::load_preflop_solve_core(&state, params.path).await)
}

async fn handle_solve_preflop_live(
    AxumState(state): AxumState<AppState>,
    Json(params): Json<SolvePreflopParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(
        poker_solver_tauri::solve_preflop_live_core(&state, params.stack_depth, params.iterations)
            .await,
    )
}

async fn handle_load_subgame_source(
    AxumState(state): AxumState<AppState>,
    Json(params): Json<BlueprintPathParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(
        poker_solver_tauri::load_subgame_source_core(&state, params.blueprint_path).await,
    )
}

async fn handle_load_blueprint_v2(
    AxumState(state): AxumState<AppState>,
    Json(params): Json<PathParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(
        poker_solver_tauri::load_blueprint_v2_core(&state, params.path).await,
    )
}

// ---------------------------------------------------------------------------
// Handlers — sync core functions (with params)
// ---------------------------------------------------------------------------

async fn handle_get_strategy_matrix(
    AxumState(state): AxumState<AppState>,
    Json(params): Json<StrategyMatrixParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::get_strategy_matrix_core(
        &state,
        params.position,
        params.threshold,
        params.street_histories,
    ))
}

async fn handle_get_available_actions(
    AxumState(state): AxumState<AppState>,
    Json(params): Json<PositionParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::get_available_actions_core(
        &state,
        params.position,
    ))
}

async fn handle_get_bundle_info(
    AxumState(state): AxumState<AppState>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::get_bundle_info_core(&state))
}

async fn handle_canonicalize_board(
    AxumState(state): AxumState<AppState>,
    Json(params): Json<CardsParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::canonicalize_board_core(&state, params.cards))
}

async fn handle_start_bucket_computation(
    AxumState(state): AxumState<AppState>,
    Json(params): Json<BoardParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::start_bucket_computation_core(
        &state,
        params.board,
        None,
    ))
}

async fn handle_get_combo_classes(
    AxumState(state): AxumState<AppState>,
    Json(params): Json<ComboClassesParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::get_combo_classes_core(
        &state,
        params.position,
        params.hand,
    ))
}

async fn handle_get_hand_equity(
    AxumState(state): AxumState<AppState>,
    Json(params): Json<HandEquityParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::get_hand_equity_core(
        &state,
        &params.hand,
        params.villain_hand.as_deref(),
    ))
}

async fn handle_get_preflop_ranges(
    AxumState(state): AxumState<AppState>,
    Json(params): Json<PreflopRangesParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::get_preflop_ranges_core(
        &state,
        params.history,
    ))
}

// ---------------------------------------------------------------------------
// Handlers — sync core functions (no params)
// ---------------------------------------------------------------------------

async fn handle_is_bundle_loaded(
    AxumState(state): AxumState<AppState>,
) -> Json<serde_json::Value> {
    to_json_value(poker_solver_tauri::is_bundle_loaded_core(&state))
}

async fn handle_get_computation_status(
    AxumState(state): AxumState<AppState>,
) -> Json<serde_json::Value> {
    to_json_value(poker_solver_tauri::get_computation_status_core(&state))
}

async fn handle_is_board_cached(
    AxumState(state): AxumState<AppState>,
    Json(params): Json<BoardParams>,
) -> Json<serde_json::Value> {
    to_json_value(poker_solver_tauri::is_board_cached_core(&state, params.board))
}

// ---------------------------------------------------------------------------
// Handler — no state
// ---------------------------------------------------------------------------

async fn handle_list_agents(
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::list_agents())
}

async fn handle_list_blueprints(
    Json(params): Json<ListBlueprintsParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::list_blueprints_core(params.dir))
}

// ---------------------------------------------------------------------------
// Handlers — postflop
// ---------------------------------------------------------------------------

async fn handle_postflop_set_config(
    Extension(state): Extension<Arc<PostflopState>>,
    Json(params): Json<PostflopConfigParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::postflop_set_config_core(
        &state,
        params.config,
    ))
}

async fn handle_postflop_solve_street(
    Extension(state): Extension<Arc<PostflopState>>,
    Json(params): Json<PostflopSolveParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::postflop_solve_street_core(
        &state,
        params.board,
        params.max_iterations,
        params.target_exploitability,
    ))
}

async fn handle_postflop_get_progress(
    Extension(state): Extension<Arc<PostflopState>>,
) -> Json<serde_json::Value> {
    to_json_value(poker_solver_tauri::postflop_get_progress_core(&state))
}

async fn handle_postflop_play_action(
    Extension(state): Extension<Arc<PostflopState>>,
    Json(params): Json<PostflopActionParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::postflop_play_action_core(
        &state,
        params.action,
    ))
}

async fn handle_postflop_navigate_to(
    Extension(state): Extension<Arc<PostflopState>>,
    Json(params): Json<PostflopNavigateToParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::postflop_navigate_to_core(
        &state,
        params.history,
    ))
}

async fn handle_postflop_close_street(
    Extension(state): Extension<Arc<PostflopState>>,
    Json(params): Json<PostflopCloseStreetParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::postflop_close_street_core(
        &state,
        params.action_history,
    ))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let state: AppState = Arc::new(ExplorationState::default());
    let postflop_state: Arc<PostflopState> = Arc::new(PostflopState::default());

    let cors = CorsLayer::new()
        .allow_origin("*".parse::<HeaderValue>().expect("valid header value"))
        .allow_methods([Method::POST])
        .allow_headers(tower_http::cors::Any);

    let app = Router::new()
        .route("/api/load_bundle", post(handle_load_bundle))
        .route("/api/load_preflop_solve", post(handle_load_preflop_solve))
        .route("/api/solve_preflop_live", post(handle_solve_preflop_live))
        .route("/api/load_subgame_source", post(handle_load_subgame_source))
        .route("/api/load_blueprint_v2", post(handle_load_blueprint_v2))
        .route("/api/get_strategy_matrix", post(handle_get_strategy_matrix))
        .route(
            "/api/get_available_actions",
            post(handle_get_available_actions),
        )
        .route("/api/is_bundle_loaded", post(handle_is_bundle_loaded))
        .route("/api/get_bundle_info", post(handle_get_bundle_info))
        .route("/api/canonicalize_board", post(handle_canonicalize_board))
        .route(
            "/api/start_bucket_computation",
            post(handle_start_bucket_computation),
        )
        .route(
            "/api/get_computation_status",
            post(handle_get_computation_status),
        )
        .route("/api/is_board_cached", post(handle_is_board_cached))
        .route("/api/list_agents", post(handle_list_agents))
        .route("/api/list_blueprints", post(handle_list_blueprints))
        .route("/api/get_combo_classes", post(handle_get_combo_classes))
        .route("/api/get_hand_equity", post(handle_get_hand_equity))
        .route(
            "/api/get_preflop_ranges",
            post(handle_get_preflop_ranges),
        )
        // Postflop explorer endpoints
        .route(
            "/api/postflop_set_config",
            post(handle_postflop_set_config),
        )
        .route(
            "/api/postflop_solve_street",
            post(handle_postflop_solve_street),
        )
        .route(
            "/api/postflop_get_progress",
            post(handle_postflop_get_progress),
        )
        .route(
            "/api/postflop_play_action",
            post(handle_postflop_play_action),
        )
        .route(
            "/api/postflop_navigate_to",
            post(handle_postflop_navigate_to),
        )
        .route(
            "/api/postflop_close_street",
            post(handle_postflop_close_street),
        )
        .layer(Extension(postflop_state))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3001")
        .await
        .expect("failed to bind to port 3001");

    println!("devserver listening on http://0.0.0.0:3001");

    axum::serve(listener, app)
        .await
        .expect("server error");
}
