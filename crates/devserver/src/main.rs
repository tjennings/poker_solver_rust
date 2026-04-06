use std::sync::Arc;

use axum::{
    extract::{State as AxumState, WebSocketUpgrade, ws::Message},
    http::{HeaderValue, Method},
    response::IntoResponse,
    routing::{get, post},
    Extension, Json, Router,
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tower_http::cors::CorsLayer;

use poker_solver_tauri::{
    ExplorationState, GameSessionState, PostflopState, SimEventSink, SimProgressEvent,
    SimResultResponse, SimulationState,
};

type AppState = Arc<ExplorationState>;

// ---------------------------------------------------------------------------
// Request param structs
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct PathParams {
    path: String,
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
struct LoadBlueprintParams {
    path: String,
    snapshot: Option<String>,
}

#[derive(Deserialize)]
struct ListBlueprintsParams {
    dir: String,
}

#[derive(Deserialize)]
struct ListSnapshotsParams {
    path: String,
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
    rollout_bias_factor: Option<f64>,
    rollout_num_samples: Option<u32>,
    rollout_opponent_samples: Option<u32>,
    leaf_eval_interval: Option<u32>,
    range_clamp_threshold: Option<f64>,
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

#[derive(Deserialize)]
struct PostflopSetCacheDirParams {
    dir: Option<String>,
}

#[derive(Deserialize)]
struct PostflopSetFilteredWeightsParams {
    oop_weights: Vec<f32>,
    ip_weights: Vec<f32>,
}

#[derive(Deserialize)]
struct PostflopCacheParams {
    board: Vec<String>,
    prior_actions: Vec<Vec<usize>>,
}


#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GamePlayActionParams {
    action_id: String,
    source: Option<String>,
}

#[derive(Deserialize)]
struct GameGetStateParams {
    source: Option<String>,
}

#[derive(Deserialize)]
struct GameCancelSolveParams {
    mode: Option<String>,
}

#[derive(Deserialize)]
struct GameBackParams {
    source: Option<String>,
}

#[derive(Deserialize)]
struct GameDealCardParams {
    card: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GameSolveParams {
    mode: Option<String>,
    max_iterations: Option<u32>,
    target_exploitability: Option<f32>,
    leaf_eval_interval: Option<u32>,
    rollout_bias_factor: Option<f64>,
    rollout_num_samples: Option<u32>,
    rollout_opponent_samples: Option<u32>,
    range_clamp_threshold: Option<f64>,
}

#[derive(Deserialize)]
struct GameLoadSpotParams {
    spot: String,
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
    Extension(postflop): Extension<Arc<PostflopState>>,
    Json(params): Json<PathParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let info = poker_solver_tauri::load_bundle_core(&state, params.path).await;
    if info.is_ok() {
        poker_solver_tauri::populate_cbv_context(&state, &postflop);
    }
    result_to_response(info)
}

async fn handle_load_blueprint_v2(
    AxumState(state): AxumState<AppState>,
    Extension(postflop): Extension<Arc<PostflopState>>,
    Json(params): Json<LoadBlueprintParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let info = poker_solver_tauri::load_blueprint_v2_core(&state, params.path, params.snapshot).await;
    if info.is_ok() {
        poker_solver_tauri::populate_cbv_context(&state, &postflop);
    }
    result_to_response(info)
}

// ---------------------------------------------------------------------------
// Handlers — sync core functions (with params)
// ---------------------------------------------------------------------------

async fn handle_get_strategy_matrix(
    AxumState(state): AxumState<AppState>,
    Extension(postflop): Extension<Arc<PostflopState>>,
    Json(params): Json<StrategyMatrixParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::get_strategy_matrix_core(
        &state,
        params.position,
        params.threshold,
        params.street_histories,
        Some(&postflop),
    ))
}

async fn handle_blueprint_propagate_ranges(
    AxumState(state): AxumState<AppState>,
    Extension(postflop): Extension<Arc<PostflopState>>,
    Json(params): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let board: Vec<String> = serde_json::from_value(params["board"].clone())
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, format!("Invalid board: {e}")))?;
    let action_history: Vec<String> = serde_json::from_value(params["action_history"].clone())
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, format!("Invalid action_history: {e}")))?;
    result_to_response(
        poker_solver_tauri::blueprint_propagate_ranges(&state, &postflop, &board, &action_history)
    )
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

async fn handle_list_snapshots(
    Json(params): Json<ListSnapshotsParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::list_snapshots_core(params.path))
}

// ---------------------------------------------------------------------------
// Handlers — simulation
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ListStrategySourcesParams {
    dir: Option<String>,
}

async fn handle_health() -> Json<serde_json::Value> {
    to_json_value(serde_json::json!({"ok": true}))
}

async fn handle_list_strategy_sources(
    Json(params): Json<ListStrategySourcesParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::list_strategy_sources_core(params.dir))
}

async fn handle_stop_simulation(
    Extension(state): Extension<Arc<SimulationState>>,
) -> Json<serde_json::Value> {
    poker_solver_tauri::stop_simulation_core(&state);
    to_json_value(())
}

async fn handle_get_simulation_result(
    Extension(state): Extension<Arc<SimulationState>>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::get_simulation_result_core(&state))
}

// ---------------------------------------------------------------------------
// WebSocket event streaming
// ---------------------------------------------------------------------------

/// JSON-serializable event sent over the WebSocket.
#[derive(Clone, Serialize)]
struct WsEvent {
    event: String,
    payload: serde_json::Value,
}

/// `SimEventSink` backed by a `tokio::sync::broadcast` channel.
///
/// Allows `start_simulation_core` to emit events that are forwarded
/// to all connected WebSocket clients.
struct BroadcastEventSink {
    tx: broadcast::Sender<WsEvent>,
}

impl SimEventSink for BroadcastEventSink {
    fn emit_progress(&self, event: SimProgressEvent) {
        let _ = self.tx.send(WsEvent {
            event: "simulation-progress".to_string(),
            payload: serde_json::to_value(event).expect("SimProgressEvent serializes"),
        });
    }

    fn emit_complete(&self, event: SimResultResponse) {
        let _ = self.tx.send(WsEvent {
            event: "simulation-complete".to_string(),
            payload: serde_json::to_value(event).expect("SimResultResponse serializes"),
        });
    }

    fn emit_error(&self, msg: String) {
        let _ = self.tx.send(WsEvent {
            event: "simulation-error".to_string(),
            payload: serde_json::to_value(msg).expect("String serializes"),
        });
    }
}

#[derive(Deserialize)]
struct StartSimulationParams {
    p1_path: String,
    p2_path: String,
    num_hands: u64,
    stack_depth: u32,
}

async fn handle_start_simulation(
    Extension(state): Extension<Arc<SimulationState>>,
    Extension(tx): Extension<broadcast::Sender<WsEvent>>,
    Json(params): Json<StartSimulationParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let sink = BroadcastEventSink { tx };
    result_to_response(poker_solver_tauri::start_simulation_core(
        sink,
        &state,
        params.p1_path,
        params.p2_path,
        params.num_hands,
        params.stack_depth,
    ))
}

async fn handle_ws_events(
    Extension(tx): Extension<broadcast::Sender<WsEvent>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| async move {
        let mut rx = tx.subscribe();
        let (mut sender, _receiver) = socket.split();

        while let Ok(event) = rx.recv().await {
            let json = serde_json::to_string(&event).expect("WsEvent serializes");
            if sender.send(Message::Text(json.into())).await.is_err() {
                break;
            }
        }
    })
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
        params.rollout_bias_factor,
        params.rollout_num_samples,
        params.rollout_opponent_samples,
        params.leaf_eval_interval,
        params.range_clamp_threshold,
    ))
}

async fn handle_postflop_cancel_solve(
    Extension(state): Extension<Arc<PostflopState>>,
) -> Json<serde_json::Value> {
    poker_solver_tauri::postflop_cancel_solve_core(&state);
    to_json_value("ok")
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

async fn handle_postflop_set_filtered_weights(
    Extension(state): Extension<Arc<PostflopState>>,
    Json(params): Json<PostflopSetFilteredWeightsParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::postflop_set_filtered_weights_core(
        &state,
        params.oop_weights,
        params.ip_weights,
    ))
}

async fn handle_postflop_set_cache_dir(
    Extension(state): Extension<Arc<PostflopState>>,
    Json(params): Json<PostflopSetCacheDirParams>,
) -> Json<serde_json::Value> {
    poker_solver_tauri::postflop_set_cache_dir_core(&state, params.dir);
    to_json_value(())
}

async fn handle_postflop_check_cache(
    Extension(state): Extension<Arc<PostflopState>>,
    Json(params): Json<PostflopCacheParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::postflop_check_cache_core(
        &state,
        params.board,
        params.prior_actions,
    ))
}

async fn handle_postflop_load_cached(
    Extension(state): Extension<Arc<PostflopState>>,
    Json(params): Json<PostflopCacheParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::postflop_load_cached_core(
        &state,
        params.board,
        params.prior_actions,
    ))
}

// ---------------------------------------------------------------------------
// Handlers — game session
// ---------------------------------------------------------------------------

async fn handle_game_new(
    AxumState(state): AxumState<AppState>,
    Extension(postflop): Extension<Arc<PostflopState>>,
    Extension(session_state): Extension<Arc<GameSessionState>>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::game_new_core(&state, &postflop, &session_state))
}

async fn handle_game_get_state(
    Extension(session_state): Extension<Arc<GameSessionState>>,
    Json(params): Json<GameGetStateParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::game_get_state_core(&session_state, params.source))
}

async fn handle_game_play_action(
    Extension(session_state): Extension<Arc<GameSessionState>>,
    Json(params): Json<GamePlayActionParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::game_play_action_core(
        &session_state,
        &params.action_id,
        params.source,
    ))
}

async fn handle_game_deal_card(
    Extension(session_state): Extension<Arc<GameSessionState>>,
    Json(params): Json<GameDealCardParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::game_deal_card_core(
        &session_state,
        &params.card,
    ))
}

async fn handle_game_back(
    Extension(session_state): Extension<Arc<GameSessionState>>,
    Json(params): Json<GameBackParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::game_back_core(&session_state, params.source))
}

async fn handle_game_solve(
    Extension(session_state): Extension<Arc<GameSessionState>>,
    Json(params): Json<GameSolveParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::game_solve_core(
        &session_state,
        params.mode,
        params.max_iterations,
        params.target_exploitability,
        params.leaf_eval_interval,
        params.rollout_bias_factor,
        params.rollout_num_samples,
        params.rollout_opponent_samples,
        params.range_clamp_threshold,
    ))
}

async fn handle_game_cancel_solve(
    Extension(session_state): Extension<Arc<GameSessionState>>,
    Json(params): Json<GameCancelSolveParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::game_cancel_solve_core(&session_state, params.mode))
}

async fn handle_game_encode_spot(
    Extension(session_state): Extension<Arc<GameSessionState>>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::game_encode_spot_core(&session_state))
}

async fn handle_game_load_spot(
    Extension(session_state): Extension<Arc<GameSessionState>>,
    Json(params): Json<GameLoadSpotParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::game_load_spot_core(
        &session_state,
        &params.spot,
    ))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let state: AppState = Arc::new(ExplorationState::default());
    let postflop_state: Arc<PostflopState> = Arc::new(PostflopState::default());
    let simulation_state: Arc<SimulationState> = Arc::new(SimulationState::default());
    let game_session_state: Arc<GameSessionState> = Arc::new(GameSessionState::default());
    let (ws_tx, _) = broadcast::channel::<WsEvent>(64);

    let cors = CorsLayer::new()
        .allow_origin("*".parse::<HeaderValue>().expect("valid header value"))
        .allow_methods([Method::GET, Method::POST])
        .allow_headers(tower_http::cors::Any);

    let app = Router::new()
        .route("/health", get(handle_health))
        .route("/api/load_bundle", post(handle_load_bundle))
        .route("/api/load_blueprint_v2", post(handle_load_blueprint_v2))
        .route("/api/get_strategy_matrix", post(handle_get_strategy_matrix))
        .route("/api/blueprint_propagate_ranges_cmd", post(handle_blueprint_propagate_ranges))
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
        .route("/api/list_snapshots", post(handle_list_snapshots))
        .route("/api/get_combo_classes", post(handle_get_combo_classes))
        .route("/api/get_hand_equity", post(handle_get_hand_equity))
        .route(
            "/api/get_preflop_ranges",
            post(handle_get_preflop_ranges),
        )
        // Simulation endpoints
        .route(
            "/api/list_strategy_sources",
            post(handle_list_strategy_sources),
        )
        .route("/api/start_simulation", post(handle_start_simulation))
        .route("/api/stop_simulation", post(handle_stop_simulation))
        .route(
            "/api/get_simulation_result",
            post(handle_get_simulation_result),
        )
        .route("/ws/events", get(handle_ws_events))
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
            "/api/postflop_cancel_solve",
            post(handle_postflop_cancel_solve),
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
        .route(
            "/api/postflop_set_filtered_weights",
            post(handle_postflop_set_filtered_weights),
        )
        .route(
            "/api/postflop_set_cache_dir",
            post(handle_postflop_set_cache_dir),
        )
        .route(
            "/api/postflop_check_cache",
            post(handle_postflop_check_cache),
        )
        .route(
            "/api/postflop_load_cached",
            post(handle_postflop_load_cached),
        )
        // Game session endpoints
        .route("/api/game_new", post(handle_game_new))
        .route("/api/game_get_state", post(handle_game_get_state))
        .route("/api/game_play_action", post(handle_game_play_action))
        .route("/api/game_deal_card", post(handle_game_deal_card))
        .route("/api/game_back", post(handle_game_back))
        .route("/api/game_solve", post(handle_game_solve))
        .route("/api/game_cancel_solve", post(handle_game_cancel_solve))
        .route("/api/game_encode_spot", post(handle_game_encode_spot))
        .route("/api/game_load_spot", post(handle_game_load_spot))
        .layer(Extension(ws_tx))
        .layer(Extension(simulation_state))
        .layer(Extension(game_session_state))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_get_state_params_deserializes_with_source() {
        let json = r#"{"source": "subgame"}"#;
        let params: GameGetStateParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.source.as_deref(), Some("subgame"));
    }

    #[test]
    fn game_get_state_params_deserializes_empty_body() {
        let json = r#"{}"#;
        let params: GameGetStateParams = serde_json::from_str(json).unwrap();
        assert!(params.source.is_none());
    }

    #[test]
    fn game_cancel_solve_params_deserializes_with_mode() {
        let json = r#"{"mode": "exact"}"#;
        let params: GameCancelSolveParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.mode.as_deref(), Some("exact"));
    }

    #[test]
    fn game_cancel_solve_params_deserializes_empty_body() {
        let json = r#"{}"#;
        let params: GameCancelSolveParams = serde_json::from_str(json).unwrap();
        assert!(params.mode.is_none());
    }

    #[test]
    fn game_play_action_params_deserializes_with_source() {
        let json = r#"{"actionId": "raise_100", "source": "exact"}"#;
        let params: GamePlayActionParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.action_id, "raise_100");
        assert_eq!(params.source.as_deref(), Some("exact"));
    }

    #[test]
    fn game_play_action_params_deserializes_without_source() {
        let json = r#"{"actionId": "fold"}"#;
        let params: GamePlayActionParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.action_id, "fold");
        assert!(params.source.is_none());
    }

    #[test]
    fn game_back_params_deserializes_with_source() {
        let json = r#"{"source": "blueprint"}"#;
        let params: GameBackParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.source.as_deref(), Some("blueprint"));
    }

    #[test]
    fn game_back_params_deserializes_empty_body() {
        let json = r#"{}"#;
        let params: GameBackParams = serde_json::from_str(json).unwrap();
        assert!(params.source.is_none());
    }

    #[test]
    fn game_solve_params_includes_mode() {
        let json = r#"{"mode": "exact"}"#;
        let params: GameSolveParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.mode.as_deref(), Some("exact"));
        assert!(params.max_iterations.is_none());
    }

    #[test]
    fn game_play_action_params_camel_case() {
        let json = r#"{"actionId": "call", "source": "subgame"}"#;
        let params: GamePlayActionParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.action_id, "call");
        assert_eq!(params.source.as_deref(), Some("subgame"));
    }
}
