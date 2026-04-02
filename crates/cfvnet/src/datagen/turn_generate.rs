//! Turn training data generation pipeline.
//!
//! Uses a 3-stage producer-consumer architecture:
//!
//! ```text
//! [Deal Generator] -> channel(5K) -> [GPU Inference] -> channel(5K) -> [DCFR Solvers] -> output
//!    1 thread                          1 thread                         N rayon threads
//! ```
//!
//! Stage 1 samples random turn situations and builds `PostFlopGame` trees.
//! Stage 2 loads a single river model and evaluates depth-boundary CFVs.
//! Stage 3 solves the games with DCFR in parallel and writes [`TrainingRecord`]s.

use crate::config::BetSizeConfig;
use std::io::BufWriter;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use burn::backend::wgpu::Wgpu;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
use indicatif::{ProgressBar, ProgressStyle};
use poker_solver_core::blueprint_v2::LeafEvaluator;
use poker_solver_core::poker::{Card, Suit, Value};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
use range_solver::bet_size::{BetSize, BetSizeOptions};
use range_solver::card::{card_pair_to_index, CardConfig, NOT_DEALT};
use range_solver::game::PostFlopGame;
use range_solver::range::Range as RsRange;
use range_solver::solve;
use range_solver::solve_step;
use rayon::prelude::*;

use super::range_gen::NUM_COMBOS;
use super::sampler::{sample_situation, Situation};
use super::storage::{write_record, TrainingRecord};
use crate::config::CfvnetConfig;
use crate::eval::river_net_evaluator::RiverNetEvaluator;
use crate::model::network::{CfvNet, INPUT_SIZE};

type B = Wgpu;

/// Bounded channel capacity for the pipeline stages.
/// River trees are tiny (~1MB) so we can buffer many more; turn trees are large (~200MB).
const PIPELINE_CHANNEL_CAPACITY_TURN: usize = 16;
const PIPELINE_CHANNEL_CAPACITY_RIVER: usize = 512;

/// Batch size for Stage 3 rayon solve dispatch.
const SOLVE_BATCH_SIZE: usize = 128;


/// Convert a range-solver `u8` card to an `rs_poker::core::Card`.
fn u8_to_rs_card(id: u8) -> Card {
    let rank = id / 4;
    let suit_id = id % 4;
    let value = Value::from(rank);
    let suit = match suit_id {
        0 => Suit::Club,
        1 => Suit::Diamond,
        2 => Suit::Heart,
        3 => Suit::Spade,
        _ => unreachable!(),
    };
    Card::new(value, suit)
}

/// Convert an `rs_poker::core::Card` to a range-solver `u8` card.
#[cfg(test)]
fn rs_card_to_u8(card: Card) -> u8 {
    let rank = card.value as u8;
    let suit = match card.suit {
        Suit::Club => 0,
        Suit::Diamond => 1,
        Suit::Heart => 2,
        Suit::Spade => 3,
    };
    4 * rank + suit
}

/// Parse a single depth's bet size strings (e.g. `["50%", "100%", "a"]`) into pot fractions.
///
/// Entries like "a" (all-in) are skipped — the game tree builder adds all-in
/// automatically.
fn parse_bet_sizes_depth(sizes: &[String]) -> Vec<f64> {
    sizes
        .iter()
        .filter_map(|s| {
            let trimmed = s.trim();
            if trimmed.eq_ignore_ascii_case("a") {
                return None;
            }
            let num_str = trimmed.trim_end_matches('%');
            num_str.parse::<f64>().ok().map(|v| v / 100.0)
        })
        .collect()
}

/// Parse all depths from a BetSizeConfig into `Vec<Vec<f64>>`.
fn parse_bet_sizes_all(config: &crate::config::BetSizeConfig) -> Vec<Vec<f64>> {
    config.depths().iter().map(|d| parse_bet_sizes_depth(d)).collect()
}

/// Solve a single turn situation and return per-player root CFVs mapped to 1326 indices.
///
/// Uses the range-solver with `depth_limit: Some(1)` so that river transitions
/// become depth boundary terminals. The leaf evaluator (river net) is called once
/// per boundary to fill in the boundary CFVs before solving.
///
/// Returns `(oop_cfvs_1326, ip_cfvs_1326, valid_mask, game_value_oop, game_value_ip)`.
#[cfg(test)]
#[allow(clippy::type_complexity)]
fn solve_turn_situation(
    board_u8: &[u8],
    pot: f64,
    effective_stack: f64,
    ranges: &[[f32; NUM_COMBOS]; 2],
    bet_sizes: &[Vec<f64>],
    solver_iterations: u32,
    target_exploitability: f32,
    evaluator: &dyn LeafEvaluator,
) -> (
    [f32; NUM_COMBOS],
    [f32; NUM_COMBOS],
    [u8; NUM_COMBOS],
    f32,
    f32,
    f32,
) {
    let mut game = build_turn_game(board_u8, pot, effective_stack, ranges, bet_sizes)
        .expect("valid game");
    evaluate_game_boundaries(&mut game, board_u8, pot, effective_stack, ranges, evaluator);
    solve_and_extract(&mut game, pot, ranges, solver_iterations, target_exploitability)
}

/// Compute `sum(range[i] * cfvs[i])` for all combos.
fn weighted_sum(range: &[f32; NUM_COMBOS], cfvs: &[f32; NUM_COMBOS]) -> f32 {
    range.iter().zip(cfvs.iter()).map(|(&r, &c)| r * c).sum()
}

/// Evaluate all boundary nodes of a turn game using a leaf evaluator.
///
/// Fills in boundary CFVs for every ordinal so the game is ready to solve.
fn evaluate_game_boundaries(
    game: &mut PostFlopGame,
    board_u8: &[u8],
    pot: f64,
    effective_stack: f64,
    ranges: &[[f32; NUM_COMBOS]; 2],
    evaluator: &dyn LeafEvaluator,
) {
    let num_boundaries = game.num_boundary_nodes();
    if num_boundaries == 0 {
        return;
    }

    // Convert board from u8 to poker_solver_core::Card for the evaluator.
    let board_cards: Vec<Card> = board_u8.iter().map(|&c| u8_to_rs_card(c)).collect();

    // Both players share the same hands on a 4-card board.
    let hands = game.private_cards(0);

    // Convert to [[Card; 2]] for the evaluator.
    let combos: Vec<[Card; 2]> = hands
        .iter()
        .map(|&(c0, c1)| [u8_to_rs_card(c0), u8_to_rs_card(c1)])
        .collect();

    // Build per-combo range arrays from the 1326-indexed input ranges.
    let oop_reach: Vec<f64> = hands
        .iter()
        .map(|&(c0, c1)| f64::from(ranges[0][card_pair_to_index(c0, c1)]))
        .collect();
    let ip_reach: Vec<f64> = hands
        .iter()
        .map(|&(c0, c1)| f64::from(ranges[1][card_pair_to_index(c0, c1)]))
        .collect();

    // Collect all (pot, eff_stack, player) requests.
    let mut requests: Vec<(f64, f64, u8)> = Vec::with_capacity(num_boundaries * 2);
    for ordinal in 0..num_boundaries {
        let bpot = game.boundary_pot(ordinal) as f64;
        let eff_stack_at_boundary = effective_stack - (bpot - pot) / 2.0;
        for player in 0..2u8 {
            requests.push((bpot, eff_stack_at_boundary, player));
        }
    }

    // One batched call -- GPU implementations do one forward pass.
    let all_cfvs = evaluator.evaluate_boundaries(
        &combos,
        &board_cards,
        &oop_reach,
        &ip_reach,
        &requests,
    );

    // Scatter results back to game.
    for ordinal in 0..num_boundaries {
        for player in 0..2usize {
            let req_idx = ordinal * 2 + player;
            let cfvs_f32: Vec<f32> = all_cfvs[req_idx].iter().map(|&v| v as f32).collect();
            game.set_boundary_cfvs(ordinal, player, cfvs_f32);
        }
    }
}

/// Build a turn game tree for a situation.
///
/// When `exact` is false (model mode): uses `depth_limit: Some(0)` to create boundary
/// nodes at the river transition, with empty river bet sizes.
/// When `exact` is true: uses `depth_limit: None` so the tree extends through the
/// river to showdown, with river bet sizes matching turn sizes.
///
/// Returns `None` for degenerate situations (effective_stack <= 0, or game construction fails).
fn build_turn_game_inner(
    board_u8: &[u8],
    pot: f64,
    effective_stack: f64,
    ranges: &[[f32; NUM_COMBOS]; 2],
    bet_sizes: &[Vec<f64>],
    exact: bool,
) -> Option<PostFlopGame> {
    let oop_range = RsRange::from_raw_data(&ranges[0]).expect("valid OOP range");
    let ip_range = RsRange::from_raw_data(&ranges[1]).expect("valid IP range");

    // bet_sizes[0] = first bet sizes, bet_sizes[1+] = raise sizes.
    let bet = bet_sizes.first()
        .map(|v| v.iter().map(|&f| BetSize::PotRelative(f)).collect())
        .unwrap_or_default();
    let raise = if bet_sizes.len() > 1 {
        bet_sizes[1..].iter()
            .flat_map(|v| v.iter().map(|&f| BetSize::PotRelative(f)))
            .collect()
    } else {
        Vec::new()
    };
    let bet_size_opts = BetSizeOptions { bet, raise };

    let is_river = board_u8.len() >= 5;

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [board_u8[0], board_u8[1], board_u8[2]],
        turn: board_u8[3],
        river: if is_river { board_u8[4] } else { NOT_DEALT },
    };

    let (river_bet_sizes, depth_limit) = if exact || is_river {
        ([bet_size_opts.clone(), bet_size_opts.clone()], None)
    } else {
        ([BetSizeOptions::default(), BetSizeOptions::default()], Some(0))
    };

    let initial_state = if is_river { BoardState::River } else { BoardState::Turn };

    let tree_config = TreeConfig {
        initial_state,
        starting_pot: pot as i32,
        effective_stack: effective_stack as i32,
        turn_bet_sizes: [bet_size_opts.clone(), bet_size_opts],
        river_bet_sizes,
        depth_limit,
        add_allin_threshold: 0.0,
        force_allin_threshold: 0.0,
        merging_threshold: 0.0,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).expect("valid action tree");
    let mut game = PostFlopGame::with_config(card_config, action_tree).expect("valid game");
    game.allocate_memory(true); // compressed (16-bit) storage to reduce memory ~4x
    use std::sync::atomic::{AtomicBool, Ordering as AO};
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, AO::Relaxed) {
        let (mem, _) = game.memory_usage();
        eprintln!("[tree] memory per game: {:.1} MB", mem as f64 / 1_048_576.0);
    }
    Some(game)
}

/// Build a depth-limited turn game tree (model mode, with boundary nodes at river).
fn build_turn_game(
    board_u8: &[u8],
    pot: f64,
    effective_stack: f64,
    ranges: &[[f32; NUM_COMBOS]; 2],
    bet_sizes: &[Vec<f64>],
) -> Option<PostFlopGame> {
    build_turn_game_inner(board_u8, pot, effective_stack, ranges, bet_sizes, false)
}

/// Build a full turn+river game tree (exact mode, no boundaries).
fn build_turn_game_exact(
    board_u8: &[u8],
    pot: f64,
    effective_stack: f64,
    ranges: &[[f32; NUM_COMBOS]; 2],
    bet_sizes: &[Vec<f64>],
) -> Option<PostFlopGame> {
    build_turn_game_inner(board_u8, pot, effective_stack, ranges, bet_sizes, true)
}

/// Perturb bet sizes by multiplying each by `1.0 + uniform(-fuzz, +fuzz)`.
///
/// Returns the original sizes unchanged if `fuzz <= 0.0`.
/// Each fuzzed value is clamped to a minimum of 0.01 to avoid non-positive sizes.
fn fuzz_bet_sizes(bet_sizes: &[Vec<f64>], fuzz: f64, rng: &mut impl Rng) -> Vec<Vec<f64>> {
    if fuzz <= 0.0 {
        return bet_sizes.to_vec();
    }
    bet_sizes.iter().map(|depth| {
        depth.iter().map(|&size| {
            let perturbation = 1.0 + rng.gen_range(-fuzz..fuzz);
            (size * perturbation).max(0.01)
        }).collect()
    }).collect()
}

/// Solve a game with boundaries already set. Returns 1326-indexed CFVs + exploitability.
#[allow(clippy::type_complexity)]
fn solve_and_extract(
    game: &mut PostFlopGame,
    pot: f64,
    ranges: &[[f32; NUM_COMBOS]; 2],
    solver_iterations: u32,
    target_exploitability: f32,
) -> ([f32; NUM_COMBOS], [f32; NUM_COMBOS], [u8; NUM_COMBOS], f32, f32, f32) {
    // Use solve() with early exit at target exploitability.
    // This checks exploitability every 10 iterations and exits early when converged.
    let abs_target = target_exploitability * pot as f32;
    let exploit = solve(game, solver_iterations, abs_target, false);

    game.back_to_root();
    game.cache_normalized_weights();
    let raw_oop = game.expected_values(0);
    let raw_ip = game.expected_values(1);

    let oop_hands = game.private_cards(0);
    let ip_hands = game.private_cards(1);

    let half_pot = pot / 2.0;
    let norm = if half_pot > 0.0 { half_pot } else { 1.0 };

    let mut oop_cfvs = [0.0_f32; NUM_COMBOS];
    let mut ip_cfvs = [0.0_f32; NUM_COMBOS];
    let mut valid_mask = [0_u8; NUM_COMBOS];

    for (i, &(c0, c1)) in oop_hands.iter().enumerate() {
        let idx = card_pair_to_index(c0, c1);
        oop_cfvs[idx] = ((f64::from(raw_oop[i]) - half_pot) / norm) as f32;
        valid_mask[idx] = 1;
    }
    for (i, &(c0, c1)) in ip_hands.iter().enumerate() {
        let idx = card_pair_to_index(c0, c1);
        ip_cfvs[idx] = ((f64::from(raw_ip[i]) - half_pot) / norm) as f32;
        valid_mask[idx] = 1;
    }

    let oop_gv = weighted_sum(&ranges[0], &oop_cfvs);
    let ip_gv = weighted_sum(&ranges[1], &ip_cfvs);

    (oop_cfvs, ip_cfvs, valid_mask, oop_gv, ip_gv, exploit)
}

/// Extract river-level training records from a solved turn+river game.
///
/// After an exact-mode solve (turn + river to showdown), this walks the game
/// tree to find each river chance outcome and extracts per-player CFVs at the
/// first river decision node. Each river card produces two records (OOP + IP).
///
/// The game must already be finalized (`range_solver::finalize` called).
fn extract_river_records(
    game: &mut PostFlopGame,
    sit: &Situation,
    starting_pot: i32,
) -> Vec<TrainingRecord> {
    let mut records = Vec::new();
    let mut path: Vec<usize> = Vec::new();

    game.back_to_root();
    collect_river_nodes(game, &mut path, sit, starting_pot, &mut records);
    game.back_to_root();

    records
}

/// Recursively walk the game tree to find river chance outcomes.
///
/// At each chance node (river deal), iterates over possible river cards,
/// navigates to the resulting decision node, and extracts CFVs for both players.
/// Uses `apply_history` for backtracking since `PostFlopGame` has no `undo()`.
fn collect_river_nodes(
    game: &mut PostFlopGame,
    path: &mut Vec<usize>,
    sit: &Situation,
    starting_pot: i32,
    records: &mut Vec<TrainingRecord>,
) {
    if game.is_terminal_node() {
        return;
    }

    if game.is_chance_node() {
        // This is the river chance node. Iterate over possible river cards.
        let possible = game.possible_cards();
        for card in 0u8..52 {
            if possible & (1u64 << card) == 0 {
                continue;
            }
            // Deal this river card.
            path.push(card as usize);
            game.apply_history(path);

            // Now we're at the first river decision node (or possibly terminal
            // if both players went all-in on the turn).
            if !game.is_terminal_node() {
                extract_river_data_at_node(game, sit, starting_pot, records);
            }

            path.pop();
            game.apply_history(path);
        }
        return;
    }

    // Decision node on the turn. Only follow the first action (check/first bet)
    // to find ONE chance node. Extracting from every turn action path would
    // produce thousands of river records per sample (one per action sequence × 46 cards).
    // One representative path per sample is sufficient for training diversity.
    let num_actions = game.available_actions().len();
    if num_actions > 0 {
        path.push(0); // Follow first action only.
        game.apply_history(path);
        collect_river_nodes(game, path, sit, starting_pot, records);
        path.pop();
        game.apply_history(path);
    }
}

/// Extract training records for both players at the current river decision node.
///
/// The game must be navigated to a river decision node (5-card board).
/// Produces two `TrainingRecord`s (one per player) with pot-normalized CFVs.
fn extract_river_data_at_node(
    game: &mut PostFlopGame,
    sit: &Situation,
    starting_pot: i32,
    records: &mut Vec<TrainingRecord>,
) {
    let board = game.current_board();
    if board.len() != 5 {
        return; // Not a river node.
    }

    // Compute pot at this node from the total bet amounts.
    let bet_amounts = game.total_bet_amount();
    let node_pot = starting_pot + bet_amounts[0] + bet_amounts[1];
    let pot_f64 = f64::from(node_pot);
    let half_pot = pot_f64 / 2.0;
    let norm = if half_pot > 0.0 { half_pot } else { 1.0 };

    game.cache_normalized_weights();

    // Build reach-weighted ranges once for both players.
    let mut oop_range = [0.0_f32; NUM_COMBOS];
    let mut ip_range = [0.0_f32; NUM_COMBOS];

    for player_idx in 0..2usize {
        let hands = game.private_cards(player_idx);
        let weights = game.weights(player_idx);
        let range = if player_idx == 0 { &mut oop_range } else { &mut ip_range };

        let weight_sum: f64 = hands.iter().zip(weights.iter())
            .filter(|&(&(c0, c1), _)| !board.contains(&c0) && !board.contains(&c1))
            .map(|(_, &w)| f64::from(w))
            .sum();

        for (i, &(c0, c1)) in hands.iter().enumerate() {
            if !board.contains(&c0) && !board.contains(&c1) {
                let idx = card_pair_to_index(c0, c1);
                range[idx] = if weight_sum > 0.0 {
                    (f64::from(weights[i]) / weight_sum) as f32
                } else {
                    0.0
                };
            }
        }
    }

    let effective_stack = (sit.effective_stack - (node_pot - starting_pot) / 2) as f32;

    for player in 0..2usize {
        let raw_evs = game.expected_values(player);
        let hands = game.private_cards(player);

        let mut cfvs = [0.0_f32; NUM_COMBOS];
        let mut valid_mask = [0_u8; NUM_COMBOS];

        for (i, &(c0, c1)) in hands.iter().enumerate() {
            let idx = card_pair_to_index(c0, c1);
            cfvs[idx] = ((f64::from(raw_evs[i]) - half_pot) / norm) as f32;
            valid_mask[idx] = 1;
        }

        let game_value = weighted_sum(
            if player == 0 { &oop_range } else { &ip_range },
            &cfvs,
        );

        records.push(TrainingRecord {
            board: board.clone(),
            pot: node_pot as f32,
            effective_stack,
            player: player as u8,
            game_value,
            oop_range,
            ip_range,
            cfvs,
            valid_mask,
        });
    }
}

/// Message types for the Stage 4 storage channel.
enum StorageMsg {
    /// A turn-level training record (4-card board).
    TurnRecord(TrainingRecord),
    /// A river-level training record (5-card board).
    RiverRecord(TrainingRecord),
    /// Flush remaining buffered records and shut down.
    Flush,
}

/// Flush a buffer of training records to a new numbered file.
///
/// Generates a filename like `{stem}_{count:05}.bin` in the same directory as `base_path`.
/// Increments `file_count` after writing.
fn flush_buffer(
    base_path: &Path,
    buffer: &mut Vec<TrainingRecord>,
    file_count: &mut u32,
) -> Result<(), String> {
    let count = buffer.len();
    let stem = base_path.file_stem().unwrap_or_default().to_string_lossy();
    let parent = base_path.parent().unwrap_or(Path::new("."));
    let path = parent.join(format!("{}_{:05}.bin", stem, file_count));
    *file_count += 1;

    let file =
        std::fs::File::create(&path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for rec in buffer.drain(..) {
        write_record(&mut writer, &rec).map_err(|e| format!("write: {e}"))?;
    }
    eprintln!("[storage] wrote {count} records to {}", path.display());
    Ok(())
}

/// GPU-accelerated turn datagen with pipelined GPU/CPU phases.
///
/// Phase 1 (CPU parallel): Build game trees for each situation.
/// Phase 2 (GPU sequential): Evaluate all boundary nodes across all games.
/// Phase 3 (CPU parallel): Solve all games (no GPU access needed).
#[cfg(feature = "cuda")]
fn generate_turn_training_data_cuda(
    config: &CfvnetConfig,
    output_path: &Path,
) -> Result<(), String> {
    use burn::backend::cuda_jit::CudaDevice;
    use burn::backend::CudaJit;
    use burn::tensor::{Tensor, TensorData};
    use crate::model::network::OUTPUT_SIZE;

    type CudaB = CudaJit<f32>;

    let river_model_path = config
        .game
        .river_model_path
        .as_deref()
        .ok_or("river_model_path is required for turn datagen")?;

    let num_samples = config.datagen.num_samples;
    let seed = crate::config::resolve_seed(config.datagen.seed);
    let threads = config.datagen.threads;
    let solver_iterations = config.datagen.solver_iterations;
    let target_exploitability = config.datagen.target_exploitability.unwrap_or(-1.0);
    let bet_sizes_f64 = parse_bet_sizes_all(&config.game.bet_sizes);
    if bet_sizes_f64.is_empty() {
        return Err("no valid percentage bet sizes found in config".into());
    }
    let bet_sizes_vec = bet_sizes_f64;

    // Load river model on CUDA.
    let device = CudaDevice::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let model = CfvNet::<CudaB>::new(
        &device,
        config.training.hidden_layers,
        config.training.hidden_size,
        INPUT_SIZE,
    )
    .load_file(river_model_path, &recorder, &device)
    .map_err(|e| format!("failed to load river model: {e}"))?;

    println!("River model loaded on CUDA");

    let wall_start = std::time::Instant::now();

    // Dedicated GPU thread: receives input batches, returns output vectors.
    let (gpu_tx, gpu_rx) = std::sync::mpsc::sync_channel::<(Vec<f32>, usize)>(1);
    let (result_tx, result_rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(1);
    let gpu_thread = std::thread::spawn(move || {
        while let Ok((inputs, total_rows)) = gpu_rx.recv() {
            let data = TensorData::new(inputs, [total_rows, INPUT_SIZE]);
            let input_tensor = Tensor::<CudaB, 2>::from_data(data, &device);
            let output = model.forward(input_tensor);
            let out_data = output.into_data();
            let out_vec: Vec<f32> = out_data.to_vec().expect("output tensor conversion");
            if result_tx.send(out_vec).is_err() {
                break;
            }
        }
    });

    let pb = ProgressBar::new(num_samples);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} [{elapsed_precise}] ETA {eta} ({per_sec})")
            .expect("valid progress bar template"),
    );
    pb.enable_steady_tick(std::time::Duration::from_secs(2));

    let file =
        std::fs::File::create(output_path).map_err(|e| format!("create output: {e}"))?;
    let mut writer = BufWriter::new(file);

    let pool = if threads > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| format!("thread pool: {e}"))?,
        )
    } else {
        None
    };

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut remaining = num_samples;
    let cuda_chunk_size = 128u64;

    // Reusable types for boundary evaluation.
    struct BoundaryRequest {
        game_idx: usize,
        ordinal: usize,
        player: usize,
        combo_indices: Vec<usize>,
        river_valid_masks: Vec<Vec<bool>>,
        num_combos: usize,
    }

    /// Prefix length: ranges (2×1326) + board_onehot (52) + rank_presence (13).
    const PREFIX_LEN: usize = OUTPUT_SIZE * 2 + 52 + 13;

    fn build_game_inputs(
        gi: usize,
        game: &PostFlopGame,
        sit: &super::sampler::Situation,
        all_inputs: &mut Vec<f32>,
        requests: &mut Vec<BoundaryRequest>,
        rows_per: &mut Vec<usize>,
    ) {
        let num_boundaries = game.num_boundary_nodes();
        if num_boundaries == 0 {
            return;
        }
        let hands = game.private_cards(0);
        let combos_u8: Vec<[u8; 2]> = hands.iter().map(|&(c0, c1)| [c0, c1]).collect();
        let combo_indices: Vec<usize> = combos_u8.iter().map(|c| card_pair_to_index(c[0], c[1])).collect();
        let board_u8 = sit.board_cards();
        let valid_rivers: Vec<u8> = (0u8..52).filter(|r| !board_u8.contains(r)).collect();
        let num_rivers = valid_rivers.len();
        let river_valid_masks: Vec<Vec<bool>> = valid_rivers.iter()
            .map(|&r| combos_u8.iter().map(|c| c[0] != r && c[1] != r).collect())
            .collect();
        let oop_rc: Vec<f32> = hands.iter().map(|&(c0, c1)| sit.ranges[0][card_pair_to_index(c0, c1)]).collect();
        let ip_rc: Vec<f32> = hands.iter().map(|&(c0, c1)| sit.ranges[1][card_pair_to_index(c0, c1)]).collect();
        let pot = f64::from(sit.pot);
        let eff = f64::from(sit.effective_stack);

        let mut prefixes = vec![[0.0_f32; PREFIX_LEN]; num_rivers];
        for (ri, &river_u8) in valid_rivers.iter().enumerate() {
            let p = &mut prefixes[ri];
            for (i, &idx) in combo_indices.iter().enumerate() {
                if river_valid_masks[ri][i] {
                    p[idx] = oop_rc[i];
                    p[OUTPUT_SIZE + idx] = ip_rc[i];
                }
            }
            let bs = OUTPUT_SIZE * 2;
            for &card in board_u8 { p[bs + card as usize] = 1.0; }
            p[bs + river_u8 as usize] = 1.0;
            let rs = bs + 52;
            for &card in board_u8 { p[rs + (card / 4) as usize] = 1.0; }
            p[rs + (river_u8 / 4) as usize] = 1.0;
        }

        for ordinal in 0..num_boundaries {
            let bpot = game.boundary_pot(ordinal) as f64;
            let es = eff - (bpot - pot) / 2.0;
            let pn = bpot as f32 / 400.0;
            let sn = es as f32 / 400.0;
            for player in 0..2usize {
                let pi = if player == 0 { 0.0_f32 } else { 1.0 };
                for prefix in &prefixes {
                    all_inputs.extend_from_slice(prefix);
                    all_inputs.push(pn);
                    all_inputs.push(sn);
                    all_inputs.push(pi);
                }
                requests.push(BoundaryRequest {
                    game_idx: gi, ordinal, player,
                    combo_indices: combo_indices.clone(),
                    river_valid_masks: river_valid_masks.clone(),
                    num_combos: hands.len(),
                });
                rows_per.push(num_rivers);
            }
        }
    }

    fn scatter_gpu_results(
        out_vec: &[f32],
        requests: &[BoundaryRequest],
        rows_per: &[usize],
        games: &mut [Option<PostFlopGame>],
    ) {
        let mut row_offset = 0;
        for (req, &nr) in requests.iter().zip(rows_per.iter()) {
            let mut cfv_sum = vec![0.0_f64; req.num_combos];
            let mut cfv_count = vec![0_u32; req.num_combos];
            for (ri, mask) in req.river_valid_masks.iter().enumerate() {
                let rs = (row_offset + ri) * OUTPUT_SIZE;
                for (i, &idx) in req.combo_indices.iter().enumerate() {
                    if mask[i] {
                        cfv_sum[i] += f64::from(out_vec[rs + idx]);
                        cfv_count[i] += 1;
                    }
                }
            }
            row_offset += nr;
            let cfvs: Vec<f32> = cfv_sum.iter().zip(cfv_count.iter())
                .map(|(&s, &c)| if c > 0 { (s / f64::from(c)) as f32 } else { 0.0 })
                .collect();
            if let Some(game) = &mut games[req.game_idx] {
                game.set_boundary_cfvs(req.ordinal, req.player, cfvs);
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn write_chunk_results(
        situations: &[super::sampler::Situation],
        results: Vec<Option<([f32; NUM_COMBOS], [f32; NUM_COMBOS], [u8; NUM_COMBOS], f32, f32)>>,
        writer: &mut BufWriter<std::fs::File>,
        pb: &ProgressBar,
    ) -> Result<(), String> {
        for (sit, result) in situations.iter().zip(results) {
            if let Some((oop_cfvs, ip_cfvs, valid_mask, oop_gv, ip_gv, _exploit)) = result {
                let board_vec = sit.board_cards().to_vec();
                let oop_rec = TrainingRecord {
                    board: board_vec.clone(), pot: sit.pot as f32,
                    effective_stack: sit.effective_stack as f32, player: 0,
                    game_value: oop_gv, oop_range: sit.ranges[0],
                    ip_range: sit.ranges[1], cfvs: oop_cfvs, valid_mask,
                };
                write_record(writer, &oop_rec).map_err(|e| format!("write OOP: {e}"))?;
                let ip_rec = TrainingRecord {
                    board: board_vec, pot: sit.pot as f32,
                    effective_stack: sit.effective_stack as f32, player: 1,
                    game_value: ip_gv, oop_range: sit.ranges[0],
                    ip_range: sit.ranges[1], cfvs: ip_cfvs, valid_mask,
                };
                write_record(writer, &ip_rec).map_err(|e| format!("write IP: {e}"))?;
            } else {
                pb.inc(1);
            }
        }
        Ok(())
    }

    // Pipeline state: previous chunk waiting for solve.
    type ChunkState = (
        Vec<Option<PostFlopGame>>,
        Vec<super::sampler::Situation>,
    );
    let mut prev_chunk: Option<ChunkState> = None;

    while remaining > 0 {
        let chunk_len = remaining.min(cuda_chunk_size);
        remaining -= chunk_len;

        // Sample + build trees for current chunk.
        let situations: Vec<_> = (0..chunk_len)
            .map(|_| sample_situation(&config.datagen, config.game.initial_stack, config.game.board_size, &mut rng))
            .collect();

        let build_one = |sit: &super::sampler::Situation| -> Option<PostFlopGame> {
            if sit.effective_stack <= 0 { return None; }
            build_turn_game(sit.board_cards(), f64::from(sit.pot),
                f64::from(sit.effective_stack), &sit.ranges, &bet_sizes_vec)
        };
        let mut games: Vec<Option<PostFlopGame>> = match &pool {
            Some(pool) => pool.install(|| situations.par_iter().map(build_one).collect()),
            None => situations.iter().map(build_one).collect(),
        };

        // Build GPU inputs for current chunk.
        let mut all_inputs: Vec<f32> = Vec::new();
        let mut all_requests: Vec<BoundaryRequest> = Vec::new();
        let mut all_rows_per: Vec<usize> = Vec::new();
        for (gi, (game_opt, sit)) in games.iter().zip(situations.iter()).enumerate() {
            if let Some(game) = game_opt.as_ref() {
                build_game_inputs(gi, game, sit, &mut all_inputs, &mut all_requests, &mut all_rows_per);
            }
        }

        // Submit current chunk to GPU (non-blocking — GPU thread processes it).
        let has_gpu_work = if all_inputs.is_empty() {
            false
        } else {
            let total_rows = all_inputs.len() / INPUT_SIZE;
            gpu_tx.send((all_inputs, total_rows)).expect("GPU thread alive");
            true
        };

        // While GPU runs: solve previous chunk on CPU.
        if let Some((mut prev_games, prev_sits)) = prev_chunk.take() {
            let results: Vec<_> = match &pool {
                Some(pool) => pool.install(|| prev_games.par_iter_mut().zip(prev_sits.par_iter())
                    .map(|(g, s)| { let game = g.as_mut()?; pb.inc(1);
                        Some(solve_and_extract(game, f64::from(s.pot), &s.ranges, solver_iterations, target_exploitability))
                    }).collect()),
                None => prev_games.iter_mut().zip(prev_sits.iter())
                    .map(|(g, s)| { let game = g.as_mut()?; pb.inc(1);
                        Some(solve_and_extract(game, f64::from(s.pot), &s.ranges, solver_iterations, target_exploitability))
                    }).collect(),
            };
            write_chunk_results(&prev_sits, results, &mut writer, &pb)?;
        }

        // Collect GPU output (blocks until forward pass completes).
        let gpu_output = if has_gpu_work {
            Some(result_rx.recv().expect("GPU result"))
        } else {
            None
        };

        // Scatter GPU results to current chunk's games.
        if let Some(out_vec) = gpu_output {
            scatter_gpu_results(&out_vec, &all_requests, &all_rows_per, &mut games);
        }

        // Queue current chunk for solving in next iteration.
        prev_chunk = Some((games, situations));
    }

    // Solve the final chunk.
    if let Some((mut prev_games, prev_sits)) = prev_chunk.take() {
        let results: Vec<_> = match &pool {
            Some(pool) => pool.install(|| prev_games.par_iter_mut().zip(prev_sits.par_iter())
                .map(|(g, s)| { let game = g.as_mut()?; pb.inc(1);
                    Some(solve_and_extract(game, f64::from(s.pot), &s.ranges, solver_iterations, target_exploitability))
                }).collect()),
            None => prev_games.iter_mut().zip(prev_sits.iter())
                .map(|(g, s)| { let game = g.as_mut()?; pb.inc(1);
                    Some(solve_and_extract(game, f64::from(s.pot), &s.ranges, solver_iterations, target_exploitability))
                }).collect(),
        };
        write_chunk_results(&prev_sits, results, &mut writer, &pb)?;
    }

    // Shut down GPU thread.
    drop(gpu_tx);
    let _ = gpu_thread.join();

    pb.finish_with_message("done");

    let wall_secs = wall_start.elapsed().as_secs_f64();
    println!("Done in {wall_secs:.1}s ({:.1} samples/sec)", num_samples as f64 / wall_secs);

    Ok(())
}

/// Generate turn training data in exact mode (no neural net).
///
/// Uses a 3-stage pipeline:
///
/// ```text
/// [Stage 1: Deal Gen] -> channel -> [Stage 3: Solve] -> channel -> [Stage 4: Storage Writer]
/// ```
///
/// Stage 1 builds full turn+river trees (depth_limit: None).
/// Stage 3 solves them to showdown and sends records via channel.
/// Stage 4 buffers records and flushes to numbered files.
fn generate_turn_training_data_exact(
    config: &CfvnetConfig,
    output_path: &Path,
) -> Result<(), String> {
    let num_samples = config.datagen.num_samples;
    let seed = crate::config::resolve_seed(config.datagen.seed);
    let threads = config.datagen.threads;
    let solver_iterations = config.datagen.solver_iterations;
    let target_exploitability = config.datagen.target_exploitability.unwrap_or(-1.0);
    let bet_sizes_f64 = parse_bet_sizes_all(&config.game.bet_sizes);
    if bet_sizes_f64.is_empty() {
        return Err("no valid percentage bet sizes found in config".into());
    }
    let bet_sizes_vec = bet_sizes_f64;
    let initial_stack = config.game.initial_stack;
    let board_size = config.game.board_size;
    let channel_capacity = if board_size >= 5 { PIPELINE_CHANNEL_CAPACITY_RIVER } else { PIPELINE_CHANNEL_CAPACITY_TURN };
    let bet_size_fuzz = config.datagen.bet_size_fuzz;
    let river_output_path = config.datagen.river_output.as_ref().map(Path::new);
    let per_file = config.datagen.per_file.unwrap_or(u64::MAX);
    let extract_river = river_output_path.is_some();

    // Load precomputed blueprint ranges if configured.
    let precomputed_ranges = if let Some(ref bp) = config.datagen.blueprint_path {
        let bp_path = Path::new(bp);
        // Check if it's a .bin file (precomputed) or a bundle dir (compute on the fly)
        if bp_path.extension().map_or(false, |e| e == "bin") {
            let ranges = super::precompute_ranges::PrecomputedRanges::load(bp_path)
                .map_err(|e| format!("load precomputed ranges: {e}"))?;
            eprintln!("[turn datagen] loaded {} precomputed range paths from {}", ranges.paths.len(), bp);
            Some(ranges)
        } else {
            // It's a blueprint bundle dir — compute paths on the fly
            let bp_gen = super::blueprint_ranges::BlueprintRangeGenerator::load(bp_path)
                .map_err(|e| format!("load blueprint: {e}"))?;
            let ranges = super::precompute_ranges::compute_preflop_paths(
                bp_gen.strategy(), bp_gen.tree(), bp_gen.decision_map(),
            );
            eprintln!("[turn datagen] computed {} preflop paths from blueprint", ranges.paths.len());
            for path in &ranges.paths {
                eprintln!("  {:<40} freq={:.4}  OOP ~{} combos, IP ~{} combos",
                    path.label, path.frequency, path.oop_nonzero, path.ip_nonzero);
            }
            Some(ranges)
        }
    } else {
        None
    };

    let street_label = if board_size >= 5 { "river" } else { "turn" };
    eprintln!("[{street_label} datagen] exact mode: solving to showdown (no neural net)");
    if precomputed_ranges.is_some() {
        eprintln!("[{street_label} datagen] using blueprint ranges instead of RSP");
    }
    if let Some(rp) = river_output_path {
        eprintln!("[{street_label} datagen] river records will be extracted to: {}", rp.display());
    }

    let pb = Arc::new(ProgressBar::new(num_samples));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} [{elapsed_precise}] ETA {eta} ({per_sec}) {msg}")
            .expect("valid progress bar template"),
    );

    let stage1_count_for_pb = Arc::new(AtomicU64::new(0));
    let stage3_count_for_pb = Arc::new(AtomicU64::new(0));
    let turn_buf_depth = Arc::new(AtomicU64::new(0));
    let river_buf_depth = Arc::new(AtomicU64::new(0));
    let s1_pb = Arc::clone(&stage1_count_for_pb);
    let s3_pb = Arc::clone(&stage3_count_for_pb);
    let tb_pb = Arc::clone(&turn_buf_depth);
    let rb_pb = Arc::clone(&river_buf_depth);
    let exploit_sum_pb = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let exploit_count_pb = Arc::new(AtomicU64::new(0));
    let es_pb = Arc::clone(&exploit_sum_pb);
    let ec_pb = Arc::clone(&exploit_count_pb);
    let pb_ticker = Arc::clone(&pb);
    let ticker = std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
            let s1 = s1_pb.load(Ordering::Relaxed);
            let s3 = s3_pb.load(Ordering::Relaxed);
            let buf = s1.saturating_sub(s3);
            let tb = tb_pb.load(Ordering::Relaxed);
            let rb = rb_pb.load(Ordering::Relaxed);
            let ec = ec_pb.load(Ordering::Relaxed);
            let avg_exploit = if ec > 0 {
                (es_pb.load(Ordering::Relaxed) as f64 / 100.0) / ec as f64
            } else {
                0.0
            };
            let disk_status = if rb > 0 {
                format!("[turn:{tb} river:{rb}]")
            } else {
                format!("[{tb}]")
            };
            pb_ticker.set_message(format!(
                "deal\u{2192}[{buf}]\u{2192}solve\u{2192}{disk_status}\u{2192}disk  expl:{avg_exploit:.1} mbb/h"
            ));
            if pb_ticker.is_finished() {
                break;
            }
        }
    });

    let stage1_count = Arc::clone(&stage1_count_for_pb);
    let stage3_count = Arc::clone(&stage3_count_for_pb);

    let wall_start = std::time::Instant::now();

    // Channel: Stage 1 -> Stage 3.
    let (tx, rx) = std::sync::mpsc::sync_channel::<(Situation, PostFlopGame)>(
        channel_capacity,
    );

    // Channel: Stage 3 -> Stage 4. Keep small to limit memory.
    const STORAGE_CHANNEL_CAPACITY: usize = 50;
    let (storage_tx, storage_rx) =
        std::sync::mpsc::sync_channel::<StorageMsg>(STORAGE_CHANNEL_CAPACITY);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|e| format!("thread pool: {e}"))?;

    // --- Stage 1: Deal Generator (exact mode) ---
    let datagen_config = config.datagen.clone();
    let bet_sizes_for_stage1 = bet_sizes_vec;
    let stage1_count_ref = Arc::clone(&stage1_count);
    let stage1 = std::thread::spawn(move || {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for _ in 0..num_samples {
            let sit = if let Some(ref precomp) = precomputed_ranges {
                super::sampler::sample_situation_with_blueprint(
                    &datagen_config, initial_stack, board_size, precomp, &mut rng,
                )
            } else {
                sample_situation(&datagen_config, initial_stack, board_size, &mut rng)
            };
            if sit.effective_stack <= 0 {
                stage1_count_ref.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let sizes = fuzz_bet_sizes(&bet_sizes_for_stage1, bet_size_fuzz, &mut rng);
            let game = build_turn_game_exact(
                sit.board_cards(),
                f64::from(sit.pot),
                f64::from(sit.effective_stack),
                &sit.ranges,
                &sizes,
            );
            if let Some(game) = game {
                stage1_count_ref.fetch_add(1, Ordering::Relaxed);
                if tx.send((sit, game)).is_err() {
                    break;
                }
            } else {
                stage1_count_ref.fetch_add(1, Ordering::Relaxed);
            }
        }
    });

    // --- Stage 4: Storage Writer (streaming, no buffering) ---
    let turn_output_path = output_path.to_path_buf();
    let river_out_for_stage4 = river_output_path.map(|p| p.to_path_buf());
    let turn_buf_depth_ref = Arc::clone(&turn_buf_depth);
    let river_buf_depth_ref = Arc::clone(&river_buf_depth);
    let stage4 = std::thread::spawn(move || -> Result<(), String> {
        // Open first files immediately. Rotate at per_file threshold.
        let mut turn_file_count = 0u32;
        let mut river_file_count = 0u32;
        let mut turn_records_in_file = 0u64;
        let mut river_records_in_file = 0u64;

        let open_file = |base: &Path, count: &mut u32| -> Result<BufWriter<std::fs::File>, String> {
            let stem = base.file_stem().unwrap_or_default().to_string_lossy();
            let parent = base.parent().unwrap_or(Path::new("."));
            let path = parent.join(format!("{}_{:05}.bin", stem, count));
            *count += 1;
            let f = std::fs::File::create(&path).map_err(|e| format!("create {}: {e}", path.display()))?;
            eprintln!("[storage] opened {}", path.display());
            Ok(BufWriter::new(f))
        };

        let mut turn_writer = open_file(&turn_output_path, &mut turn_file_count)?;
        let mut river_writer: Option<BufWriter<std::fs::File>> = match &river_out_for_stage4 {
            Some(rp) => Some(open_file(rp, &mut river_file_count)?),
            None => None,
        };

        while let Ok(msg) = storage_rx.recv() {
            match msg {
                StorageMsg::TurnRecord(rec) => {
                    if turn_records_in_file >= per_file {
                        drop(turn_writer);
                        turn_writer = open_file(&turn_output_path, &mut turn_file_count)?;
                        turn_records_in_file = 0;
                        turn_buf_depth_ref.store(0, Ordering::Relaxed);
                    }
                    write_record(&mut turn_writer, &rec).map_err(|e| format!("write turn: {e}"))?;
                    turn_records_in_file += 1;
                    turn_buf_depth_ref.store(turn_records_in_file, Ordering::Relaxed);
                }
                StorageMsg::RiverRecord(rec) => {
                    if river_writer.is_some() && river_records_in_file >= per_file {
                        if let Some(ref rp) = river_out_for_stage4 {
                            drop(river_writer.take());
                            river_writer = Some(open_file(rp, &mut river_file_count)?);
                            river_records_in_file = 0;
                            river_buf_depth_ref.store(0, Ordering::Relaxed);
                        }
                    }
                    if let Some(ref mut rw) = river_writer {
                        write_record(rw, &rec).map_err(|e| format!("write river: {e}"))?;
                        river_records_in_file += 1;
                        river_buf_depth_ref.store(river_records_in_file, Ordering::Relaxed);
                    }
                }
                StorageMsg::Flush => break,
            }
        }

        Ok(())
    });

    // --- Stage 3: DCFR Solvers (no Stage 2 in exact mode) ---
    let pb_ref = Arc::clone(&pb);
    let stage3_count_ref = Arc::clone(&stage3_count);
    let exploit_sum_ref = Arc::clone(&exploit_sum_pb);
    let exploit_count_ref = Arc::clone(&exploit_count_pb);
    let stage3 = std::thread::spawn(move || -> Result<(), String> {
        let mut batch: Vec<(Situation, PostFlopGame)> = Vec::with_capacity(SOLVE_BATCH_SIZE);

        loop {
            match rx.recv() {
                Ok(item) => batch.push(item),
                Err(_) => break,
            }
            while batch.len() < SOLVE_BATCH_SIZE {
                match rx.try_recv() {
                    Ok(item) => batch.push(item),
                    Err(_) => break,
                }
            }

            type TurnResult = ([f32; NUM_COMBOS], [f32; NUM_COMBOS], [u8; NUM_COMBOS], f32, f32, f32);
            let results: Vec<(Situation, TurnResult, Vec<TrainingRecord>)> = pool.install(|| {
                batch
                    .par_drain(..)
                    .map(|(sit, mut game)| {
                        let pot = f64::from(sit.pot);
                        let result = solve_and_extract(
                            &mut game,
                            pot,
                            &sit.ranges,
                            solver_iterations,
                            target_exploitability,
                        );
                        let count = stage3_count_ref.fetch_add(1, Ordering::Relaxed);
                        let (_, _, _, _, _, exploit_chips) = &result;
                        if count % 10 == 0 && *exploit_chips >= 0.0 {
                            let bb = initial_stack as f32 / 100.0;
                            let exploit_mbb = if bb > 0.0 { exploit_chips / bb * 1000.0 } else { 0.0 };
                            exploit_sum_ref.fetch_add((exploit_mbb * 100.0) as u64, Ordering::Relaxed);
                            exploit_count_ref.fetch_add(1, Ordering::Relaxed);
                        }
                        // Extract river records if configured (game is already finalized).
                        let river_recs = if extract_river {
                            extract_river_records(&mut game, &sit, sit.pot)
                        } else {
                            Vec::new()
                        };
                        pb_ref.inc(1);
                        (sit, result, river_recs)
                    })
                    .collect()
            });

            for (sit, (oop_cfvs, ip_cfvs, valid_mask, oop_gv, ip_gv, _exploit), river_recs) in results {
                let board_vec = sit.board_cards().to_vec();

                let oop_rec = TrainingRecord {
                    board: board_vec.clone(),
                    pot: sit.pot as f32,
                    effective_stack: sit.effective_stack as f32,
                    player: 0,
                    game_value: oop_gv,
                    oop_range: sit.ranges[0],
                    ip_range: sit.ranges[1],
                    cfvs: oop_cfvs,
                    valid_mask,
                };
                storage_tx.send(StorageMsg::TurnRecord(oop_rec))
                    .map_err(|e| format!("send turn OOP: {e}"))?;

                let ip_rec = TrainingRecord {
                    board: board_vec,
                    pot: sit.pot as f32,
                    effective_stack: sit.effective_stack as f32,
                    player: 1,
                    game_value: ip_gv,
                    oop_range: sit.ranges[0],
                    ip_range: sit.ranges[1],
                    cfvs: ip_cfvs,
                    valid_mask,
                };
                storage_tx.send(StorageMsg::TurnRecord(ip_rec))
                    .map_err(|e| format!("send turn IP: {e}"))?;

                for river_rec in river_recs {
                    storage_tx.send(StorageMsg::RiverRecord(river_rec))
                        .map_err(|e| format!("send river: {e}"))?;
                }
            }
        }

        // Signal Stage 4 to flush and shut down.
        storage_tx.send(StorageMsg::Flush)
            .map_err(|e| format!("send flush: {e}"))?;
        Ok(())
    });

    stage1.join().map_err(|e| format!("stage 1 panicked: {e:?}"))?;
    stage3.join().map_err(|e| format!("stage 3 panicked: {e:?}"))??;
    stage4.join().map_err(|e| format!("stage 4 panicked: {e:?}"))??;

    pb.finish_with_message("done");
    let _ = ticker.join();

    let wall_secs = wall_start.elapsed().as_secs_f64();
    let s1 = stage1_count.load(Ordering::Relaxed);
    let s3 = stage3_count.load(Ordering::Relaxed);
    eprintln!("Stage 1 (deal gen):   {s1} items in {wall_secs:.1}s ({:.1}/s)", s1 as f64 / wall_secs);
    eprintln!("Stage 3 (DCFR solve): {s3} items in {wall_secs:.1}s ({:.1}/s)", s3 as f64 / wall_secs);
    eprintln!("Total wall time: {wall_secs:.1}s ({:.1} samples/sec)", num_samples as f64 / wall_secs);

    Ok(())
}

/// Generate turn training data using a 3-stage producer-consumer pipeline.
///
/// Stage 1 (deal generator): samples situations and builds game trees.
/// Stage 2 (GPU inference): loads one river model, evaluates boundary CFVs.
/// Stage 3 (DCFR solvers): solves games in parallel via rayon, writes output.
///
/// Stages are connected by bounded channels (capacity 5000) providing backpressure.
///
/// # Errors
///
/// Returns an error if the config is invalid, the river model cannot be
/// loaded, or file IO fails.
pub fn generate_turn_training_data(
    config: &CfvnetConfig,
    output_path: &Path,
    backend: &str,
) -> Result<(), String> {
    let exact_mode = config.datagen.mode == "exact";

    if exact_mode {
        return generate_turn_training_data_exact(config, output_path);
    }

    match backend {
        #[cfg(feature = "cuda")]
        "cuda" => return generate_turn_training_data_cuda(config, output_path),
        #[cfg(not(feature = "cuda"))]
        "cuda" => return Err("CUDA backend not enabled. Rebuild with: cargo build -p cfvnet --features cuda --release".into()),
        _ => {} // fall through to NdArray path
    }
    let river_model_path = config
        .game
        .river_model_path
        .as_deref()
        .ok_or("river_model_path is required for turn datagen")?;

    let num_samples = config.datagen.num_samples;
    let seed = crate::config::resolve_seed(config.datagen.seed);
    let threads = config.datagen.threads;
    let solver_iterations = config.datagen.solver_iterations;
    let target_exploitability = config.datagen.target_exploitability.unwrap_or(-1.0);
    let bet_sizes_f64 = parse_bet_sizes_all(&config.game.bet_sizes);
    if bet_sizes_f64.is_empty() {
        return Err("no valid percentage bet sizes found in config".into());
    }
    let bet_sizes_vec = bet_sizes_f64;

    // Load river model's config to get its architecture (may differ from turn config).
    let river_model_dir = std::path::Path::new(river_model_path)
        .parent()
        .ok_or("river_model_path has no parent directory")?;
    let river_config_path = river_model_dir.join("config.yaml");
    let (river_hidden_layers, river_hidden_size) = if river_config_path.exists() {
        let river_yaml = std::fs::read_to_string(&river_config_path)
            .map_err(|e| format!("read river config: {e}"))?;
        let river_cfg: CfvnetConfig = serde_yaml::from_str(&river_yaml)
            .map_err(|e| format!("parse river config: {e}"))?;
        eprintln!("[turn datagen] river model architecture: {}×{} (from {})",
            river_cfg.training.hidden_layers, river_cfg.training.hidden_size,
            river_config_path.display());
        (river_cfg.training.hidden_layers, river_cfg.training.hidden_size)
    } else {
        eprintln!("[turn datagen] warning: no river config.yaml found, using turn config architecture");
        (config.training.hidden_layers, config.training.hidden_size)
    };

    // Verify initial load works (fail fast before spawning threads).
    let device = <B as burn::tensor::backend::Backend>::Device::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let model = CfvNet::<B>::new(
        &device,
        river_hidden_layers,
        river_hidden_size,
        INPUT_SIZE,
    )
    .load_file(river_model_path, &recorder, &device)
    .map_err(|e| format!("failed to load river model: {e}"))?;
    drop(model);

    let pb = Arc::new(ProgressBar::new(num_samples));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} [{elapsed_precise}] ETA {eta} ({per_sec}) {msg}")
            .expect("valid progress bar template"),
    );

    // Show pipeline buffer depths in the progress bar message.
    let stage1_count_for_pb = Arc::new(AtomicU64::new(0));
    let stage2_count_for_pb = Arc::new(AtomicU64::new(0));
    let stage3_count_for_pb = Arc::new(AtomicU64::new(0));
    let s1_pb = Arc::clone(&stage1_count_for_pb);
    let s2_pb = Arc::clone(&stage2_count_for_pb);
    let s3_pb = Arc::clone(&stage3_count_for_pb);
    let exploit_sum_pb = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let exploit_count_pb = Arc::new(AtomicU64::new(0));
    let es_pb = Arc::clone(&exploit_sum_pb);
    let ec_pb = Arc::clone(&exploit_count_pb);
    let pb_ticker = Arc::clone(&pb);
    let ticker = std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
            let s1 = s1_pb.load(Ordering::Relaxed);
            let s2 = s2_pb.load(Ordering::Relaxed);
            let s3 = s3_pb.load(Ordering::Relaxed);
            let buf1 = s1.saturating_sub(s2);
            let buf2 = s2.saturating_sub(s3);
            let ec = ec_pb.load(Ordering::Relaxed);
            let avg_exploit = if ec > 0 {
                (es_pb.load(Ordering::Relaxed) as f64 / 100.0) / ec as f64
            } else {
                0.0
            };
            pb_ticker.set_message(format!("deal→[{buf1}]→gpu→[{buf2}]→solve  expl:{avg_exploit:.1} mbb/h"));
            if pb_ticker.is_finished() {
                break;
            }
        }
    });

    // Open output file with mutex for thread-safe writing from Stage 3.
    let file =
        std::fs::File::create(output_path).map_err(|e| format!("create output: {e}"))?;
    let writer = Arc::new(Mutex::new(BufWriter::new(file)));

    // Per-stage throughput counters (shared with progress bar ticker).
    let stage1_count = Arc::clone(&stage1_count_for_pb);
    let stage2_count = Arc::clone(&stage2_count_for_pb);
    let stage3_count = Arc::clone(&stage3_count_for_pb);

    let wall_start = std::time::Instant::now();

    // --- 3-stage pipeline connected by bounded channels ---
    //
    // [Deal Generator] -> ch1(5K) -> [GPU Inference] -> ch2(5K) -> [DCFR Solvers] -> output
    //    1 thread                      1 thread                     N rayon threads

    let (tx1, rx1) = std::sync::mpsc::sync_channel::<(Situation, PostFlopGame)>(
        PIPELINE_CHANNEL_CAPACITY_TURN,
    );
    let (tx2, rx2) = std::sync::mpsc::sync_channel::<(Situation, PostFlopGame)>(
        PIPELINE_CHANNEL_CAPACITY_TURN,
    );

    // Build rayon thread pool for Stage 3.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|e| format!("thread pool: {e}"))?;

    // --- Stage 1: Deal Generator (1 thread) ---
    // Samples situations, builds PostFlopGame, sends to Stage 2.
    let datagen_config = config.datagen.clone();
    let initial_stack = config.game.initial_stack;
    let board_size = config.game.board_size;
    let bet_sizes_for_stage1 = bet_sizes_vec.clone();
    let bet_size_fuzz = config.datagen.bet_size_fuzz;
    let stage1_count_ref = Arc::clone(&stage1_count);
    let stage1 = std::thread::spawn(move || {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for _ in 0..num_samples {
            let sit = sample_situation(&datagen_config, initial_stack, board_size, &mut rng);
            if sit.effective_stack <= 0 {
                stage1_count_ref.fetch_add(1, Ordering::Relaxed);
                continue; // Skip degenerate situations.
            }
            let sizes = fuzz_bet_sizes(&bet_sizes_for_stage1, bet_size_fuzz, &mut rng);
            let game = build_turn_game(
                sit.board_cards(),
                f64::from(sit.pot),
                f64::from(sit.effective_stack),
                &sit.ranges,
                &sizes,
            );
            if let Some(game) = game {
                stage1_count_ref.fetch_add(1, Ordering::Relaxed);
                if tx1.send((sit, game)).is_err() {
                    break; // Receiver dropped.
                }
            } else {
                stage1_count_ref.fetch_add(1, Ordering::Relaxed);
            }
        }
        // tx1 drops here, closing the channel.
    });

    // --- Stage 2: GPU Inference (N_GPU_THREADS threads) ---
    // Each thread loads its own model, pulls from shared rx1, sends to tx2.
    const N_GPU_THREADS: usize = 2;
    const GPU_BATCH_SIZE: usize = 32;
    let rx1 = Arc::new(Mutex::new(rx1));
    let mut stage2_handles = Vec::with_capacity(N_GPU_THREADS);
    for gpu_id in 0..N_GPU_THREADS {
        let rx1_ref = Arc::clone(&rx1);
        let tx2_ref = tx2.clone();
        let stage2_count_ref = Arc::clone(&stage2_count);
        let river_model_path_owned = river_model_path.to_string();
        stage2_handles.push(std::thread::spawn(move || {
            let device = <B as burn::tensor::backend::Backend>::Device::default();
            let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
            let model = CfvNet::<B>::new(
                &device,
                river_hidden_layers,
                river_hidden_size,
                INPUT_SIZE,
            )
            .load_file(&river_model_path_owned, &recorder, &device)
            .unwrap_or_else(|e| panic!("GPU thread {gpu_id}: load river model: {e}"));
            let evaluator = RiverNetEvaluator::new(model, device);

            let mut batch: Vec<(Situation, PostFlopGame)> = Vec::with_capacity(GPU_BATCH_SIZE);

            loop {
                // Lock rx1, block on first recv, drain non-blocking up to batch size.
                {
                    let rx = rx1_ref.lock().expect("rx1 lock");
                    match rx.recv() {
                        Ok(item) => batch.push(item),
                        Err(_) => break,
                    }
                    while batch.len() < GPU_BATCH_SIZE {
                        match rx.try_recv() {
                            Ok(item) => batch.push(item),
                            Err(_) => break,
                        }
                    }
                } // Drop lock before GPU work.

                for (sit, game) in batch.iter_mut() {
                    evaluate_game_boundaries(
                        game,
                        sit.board_cards(),
                        f64::from(sit.pot),
                        f64::from(sit.effective_stack),
                        &sit.ranges,
                        &evaluator,
                    );
                    stage2_count_ref.fetch_add(1, Ordering::Relaxed);
                }

                for item in batch.drain(..) {
                    if tx2_ref.send(item).is_err() {
                        return;
                    }
                }
            }

            // Process remaining.
            for (sit, game) in batch.iter_mut() {
                evaluate_game_boundaries(
                    game,
                    sit.board_cards(),
                    f64::from(sit.pot),
                    f64::from(sit.effective_stack),
                    &sit.ranges,
                    &evaluator,
                );
                stage2_count_ref.fetch_add(1, Ordering::Relaxed);
            }
            for item in batch.drain(..) {
                let _ = tx2_ref.send(item);
            }
        }));
    }
    drop(tx2); // Drop the original tx2 so channel closes when all GPU threads finish.

    // --- Stage 3: DCFR Solvers (rayon thread pool, N threads) ---
    // Receives games with boundaries set, solves in parallel batches, writes output.
    let pb_ref = Arc::clone(&pb);
    let writer_ref = Arc::clone(&writer);
    let stage3_count_ref = Arc::clone(&stage3_count);
    let exploit_sum_ref = Arc::clone(&exploit_sum_pb);
    let exploit_count_ref = Arc::clone(&exploit_count_pb);
    let stage3 = std::thread::spawn(move || -> Result<(), String> {
        let mut batch: Vec<(Situation, PostFlopGame)> = Vec::with_capacity(SOLVE_BATCH_SIZE);

        loop {
            // Block on first receive; drain up to SOLVE_BATCH_SIZE non-blocking.
            match rx2.recv() {
                Ok(item) => batch.push(item),
                Err(_) => break, // Channel closed.
            }
            while batch.len() < SOLVE_BATCH_SIZE {
                match rx2.try_recv() {
                    Ok(item) => batch.push(item),
                    Err(_) => break, // Empty or closed; process what we have.
                }
            }

            // Solve batch in parallel. Compute exploitability every 100th sample.
            let results: Vec<_> = pool.install(|| {
                batch
                    .par_drain(..)
                    .map(|(sit, mut game)| {
                        let pot = f64::from(sit.pot);
                        let result = solve_and_extract(
                            &mut game,
                            pot,
                            &sit.ranges,
                            solver_iterations,
                            target_exploitability,
                        );
                        let count = stage3_count_ref.fetch_add(1, Ordering::Relaxed);
                        // Use the exploitability returned by solve() (6th tuple element).
                        let (_, _, _, _, _, exploit_chips) = &result;
                        if count % 10 == 0 && *exploit_chips >= 0.0 {
                            let bb = initial_stack as f32 / 100.0;
                            let exploit_mbb = if bb > 0.0 { exploit_chips / bb * 1000.0 } else { 0.0 };
                            exploit_sum_ref.fetch_add((exploit_mbb * 100.0) as u64, Ordering::Relaxed);
                            exploit_count_ref.fetch_add(1, Ordering::Relaxed);
                        }
                        pb_ref.inc(1);
                        (sit, result)
                    })
                    .collect()
            });

            // Write results sequentially under lock.
            let mut w = writer_ref.lock().expect("writer lock");
            for (sit, (oop_cfvs, ip_cfvs, valid_mask, oop_gv, ip_gv, _exploit)) in results {
                let board_vec = sit.board_cards().to_vec();

                let oop_rec = TrainingRecord {
                    board: board_vec.clone(),
                    pot: sit.pot as f32,
                    effective_stack: sit.effective_stack as f32,
                    player: 0,
                    game_value: oop_gv,
                    oop_range: sit.ranges[0],
                    ip_range: sit.ranges[1],
                    cfvs: oop_cfvs,
                    valid_mask,
                };
                write_record(&mut *w, &oop_rec).map_err(|e| format!("write OOP: {e}"))?;

                let ip_rec = TrainingRecord {
                    board: board_vec,
                    pot: sit.pot as f32,
                    effective_stack: sit.effective_stack as f32,
                    player: 1,
                    game_value: ip_gv,
                    oop_range: sit.ranges[0],
                    ip_range: sit.ranges[1],
                    cfvs: ip_cfvs,
                    valid_mask,
                };
                write_record(&mut *w, &ip_rec).map_err(|e| format!("write IP: {e}"))?;
            }
        }

        Ok(())
    });

    // Wait for all stages to complete.
    stage1.join().map_err(|e| format!("stage 1 panicked: {e:?}"))?;
    for (i, handle) in stage2_handles.into_iter().enumerate() {
        handle.join().map_err(|e| format!("stage 2 thread {i} panicked: {e:?}"))?;
    }
    stage3.join().map_err(|e| format!("stage 3 panicked: {e:?}"))??;

    pb.finish_with_message("done");
    let _ = ticker.join();

    // Report per-stage throughput.
    let wall_secs = wall_start.elapsed().as_secs_f64();
    let s1 = stage1_count.load(Ordering::Relaxed);
    let s2 = stage2_count.load(Ordering::Relaxed);
    let s3 = stage3_count.load(Ordering::Relaxed);
    eprintln!("Stage 1 (deal gen):   {s1} items in {wall_secs:.1}s ({:.1}/s)", s1 as f64 / wall_secs);
    eprintln!("Stage 2 (GPU eval):   {s2} items in {wall_secs:.1}s ({:.1}/s)", s2 as f64 / wall_secs);
    eprintln!("Stage 3 (DCFR solve): {s3} items in {wall_secs:.1}s ({:.1}/s)", s3 as f64 / wall_secs);
    eprintln!("Total wall time: {wall_secs:.1}s ({:.1} samples/sec)", num_samples as f64 / wall_secs);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        CfvnetConfig, DatagenConfig, EvaluationConfig, GameConfig, TrainingConfig,
    };
    use crate::datagen::storage;
    use tempfile::NamedTempFile;

    fn turn_test_config(num_samples: u64) -> CfvnetConfig {
        CfvnetConfig {
            game: GameConfig {
                initial_stack: 200,
                bet_sizes: BetSizeConfig(vec![vec!["50%".into(), "a".into()]]),
                board_size: 4,
                river_model_path: None, // tests don't load a real model
                ..Default::default()
            },
            datagen: DatagenConfig {
                num_samples,
                street: "turn".into(),
                solver_iterations: 50,
                target_exploitability: Some(0.05),
                threads: 1,
                seed: Some(42),
                ..Default::default()
            },
            training: TrainingConfig {
                hidden_layers: 1,
                hidden_size: 8,
                ..Default::default()
            },
            evaluation: EvaluationConfig::default(),
        }
    }

    #[test]
    fn parse_bet_sizes_basic() {
        let sizes = vec!["50%".into(), "100%".into(), "a".into()];
        let parsed = parse_bet_sizes_depth(&sizes);
        assert_eq!(parsed.len(), 2);
        assert!((parsed[0] - 0.5).abs() < 1e-10);
        assert!((parsed[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn parse_bet_sizes_only_allin() {
        let sizes = vec!["a".into()];
        let parsed = parse_bet_sizes_depth(&sizes);
        assert!(parsed.is_empty());
    }

    #[test]
    fn u8_card_roundtrip() {
        for id in 0u8..52 {
            let card = u8_to_rs_card(id);
            let back = rs_card_to_u8(card);
            assert_eq!(id, back, "roundtrip failed for card {id}");
        }
    }

    #[test]
    fn solve_single_turn_situation() {
        // Use a tiny untrained model as the river evaluator.
        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let model = CfvNet::<B>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = RiverNetEvaluator::new(model, device);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let datagen_config = DatagenConfig {
            num_samples: 1,
            street: "turn".into(),
            solver_iterations: 20,
            target_exploitability: Some(0.05),
            threads: 1,
            seed: Some(42),
            ..Default::default()
        };
        let sit = sample_situation(&datagen_config, 200, 4, &mut rng);
        assert_eq!(sit.board_size, 4);

        if sit.effective_stack <= 0 {
            return; // Skip degenerate situation.
        }

        let bet_sizes_f64 = parse_bet_sizes_depth(&["50%".into(), "a".into()]);
        let (oop_cfvs, ip_cfvs, valid_mask, oop_gv, ip_gv, _exploit) = solve_turn_situation(
            sit.board_cards(),
            f64::from(sit.pot),
            f64::from(sit.effective_stack),
            &sit.ranges,
            &[bet_sizes_f64],
            20,
            0.0,
            &evaluator,
        );

        // Verify shapes.
        assert_eq!(oop_cfvs.len(), NUM_COMBOS);
        assert_eq!(ip_cfvs.len(), NUM_COMBOS);
        assert_eq!(valid_mask.len(), NUM_COMBOS);

        // Some combos must be valid.
        let num_valid: usize = valid_mask.iter().map(|&v| v as usize).sum();
        assert!(num_valid > 0, "expected some valid combos");

        // All values should be finite.
        for (i, &cfv) in oop_cfvs.iter().enumerate() {
            assert!(cfv.is_finite(), "OOP combo {i}: non-finite CFV {cfv}");
        }
        for (i, &cfv) in ip_cfvs.iter().enumerate() {
            assert!(cfv.is_finite(), "IP combo {i}: non-finite CFV {cfv}");
        }
        assert!(oop_gv.is_finite(), "OOP game value not finite");
        assert!(ip_gv.is_finite(), "IP game value not finite");
    }

    /// Test that `evaluate_game_boundaries` sets boundary CFVs on a game,
    /// producing finite values for all boundary nodes.
    #[test]
    fn evaluate_game_boundaries_sets_finite_cfvs() {
        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let model = CfvNet::<B>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = RiverNetEvaluator::new(model, device);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let datagen_config = DatagenConfig {
            num_samples: 1,
            street: "turn".into(),
            solver_iterations: 20,
            target_exploitability: Some(0.05),
            threads: 1,
            seed: Some(42),
            ..Default::default()
        };
        let sit = sample_situation(&datagen_config, 200, 4, &mut rng);
        if sit.effective_stack <= 0 {
            return;
        }

        let bet_sizes_f64 = parse_bet_sizes_depth(&["50%".into(), "a".into()]);
        let mut game = build_turn_game(
            sit.board_cards(),
            f64::from(sit.pot),
            f64::from(sit.effective_stack),
            &sit.ranges,
            &[bet_sizes_f64],
        )
        .expect("game should be built");

        let num_boundaries = game.num_boundary_nodes();
        assert!(num_boundaries > 0, "turn game should have boundary nodes");

        // Call the extracted function.
        evaluate_game_boundaries(
            &mut game,
            sit.board_cards(),
            f64::from(sit.pot),
            f64::from(sit.effective_stack),
            &sit.ranges,
            &evaluator,
        );

        // After evaluation, the game should be solvable. We verify by solving
        // and checking that results are finite.
        let (oop_cfvs, ip_cfvs, _valid, oop_gv, ip_gv, _exploit) = solve_and_extract(
            &mut game,
            f64::from(sit.pot),
            &sit.ranges,
            20,
            0.0,
        );

        for (i, &cfv) in oop_cfvs.iter().enumerate() {
            assert!(cfv.is_finite(), "OOP combo {i}: non-finite CFV {cfv}");
        }
        for (i, &cfv) in ip_cfvs.iter().enumerate() {
            assert!(cfv.is_finite(), "IP combo {i}: non-finite CFV {cfv}");
        }
        assert!(oop_gv.is_finite());
        assert!(ip_gv.is_finite());
    }

    /// Test that the 3-step pipeline (build + evaluate_boundaries + solve_and_extract)
    /// produces the same result as the monolithic `solve_turn_situation`.
    #[test]
    fn pipeline_matches_monolithic_solve() {
        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let model = CfvNet::<B>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = RiverNetEvaluator::new(model, device);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let datagen_config = DatagenConfig {
            num_samples: 1,
            street: "turn".into(),
            solver_iterations: 20,
            target_exploitability: Some(0.05),
            threads: 1,
            seed: Some(42),
            ..Default::default()
        };
        let sit = sample_situation(&datagen_config, 200, 4, &mut rng);
        if sit.effective_stack <= 0 {
            return;
        }

        let bet_sizes_f64 = parse_bet_sizes_depth(&["50%".into(), "a".into()]);
        let pot = f64::from(sit.pot);
        let eff = f64::from(sit.effective_stack);

        // Monolithic path.
        let (mono_oop, mono_ip, mono_mask, mono_oop_gv, mono_ip_gv, _exploit) = solve_turn_situation(
            sit.board_cards(),
            pot,
            eff,
            &sit.ranges,
            &[bet_sizes_f64.clone()],
            20,
            0.0,
            &evaluator,
        );

        // Pipeline path: build -> evaluate boundaries -> solve_and_extract.
        let mut game = build_turn_game(
            sit.board_cards(),
            pot,
            eff,
            &sit.ranges,
            &[bet_sizes_f64],
        )
        .expect("game should be built");

        evaluate_game_boundaries(
            &mut game,
            sit.board_cards(),
            pot,
            eff,
            &sit.ranges,
            &evaluator,
        );

        let (pipe_oop, pipe_ip, pipe_mask, pipe_oop_gv, pipe_ip_gv, _exploit) = solve_and_extract(
            &mut game,
            pot,
            &sit.ranges,
            20,
            0.0,
        );

        // The masks should be identical (same game tree).
        assert_eq!(mono_mask, pipe_mask, "valid masks should match");

        // CFVs should match exactly (same solver, same boundaries, same iterations).
        for i in 0..NUM_COMBOS {
            assert!(
                (mono_oop[i] - pipe_oop[i]).abs() < 1e-5,
                "OOP combo {i} mismatch: mono={} pipe={}",
                mono_oop[i],
                pipe_oop[i],
            );
            assert!(
                (mono_ip[i] - pipe_ip[i]).abs() < 1e-5,
                "IP combo {i} mismatch: mono={} pipe={}",
                mono_ip[i],
                pipe_ip[i],
            );
        }

        assert!(
            (mono_oop_gv - pipe_oop_gv).abs() < 1e-5,
            "OOP game value mismatch: mono={mono_oop_gv} pipe={pipe_oop_gv}"
        );
        assert!(
            (mono_ip_gv - pipe_ip_gv).abs() < 1e-5,
            "IP game value mismatch: mono={mono_ip_gv} pipe={pipe_ip_gv}"
        );
    }

    #[test]
    fn generate_turn_requires_river_model_path() {
        let config = turn_test_config(1);
        let output = NamedTempFile::new().unwrap();
        let result = generate_turn_training_data(&config, output.path(), "ndarray");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("river_model_path"));
    }

    #[test]
    fn solve_writes_4_card_board_records() {
        // Directly test the record writing path without needing a model file.
        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let model = CfvNet::<B>::new(&device, 1, 8, INPUT_SIZE);
        let evaluator = RiverNetEvaluator::new(model, device);

        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let datagen_config = DatagenConfig {
            num_samples: 1,
            street: "turn".into(),
            solver_iterations: 10,
            target_exploitability: Some(0.05),
            threads: 1,
            seed: Some(123),
            ..Default::default()
        };
        let sit = sample_situation(&datagen_config, 200, 4, &mut rng);
        if sit.effective_stack <= 0 {
            return;
        }

        let bet_sizes_f64 = parse_bet_sizes_depth(&["50%".into()]);
        let (oop_cfvs, ip_cfvs, valid_mask, oop_gv, ip_gv, _exploit) = solve_turn_situation(
            sit.board_cards(),
            f64::from(sit.pot),
            f64::from(sit.effective_stack),
            &sit.ranges,
            &[bet_sizes_f64],
            10,
            0.0,
            &evaluator,
        );

        // Write records and verify they round-trip correctly.
        let output = NamedTempFile::new().unwrap();
        {
            let file = std::fs::File::create(output.path()).unwrap();
            let mut writer = BufWriter::new(file);

            let board_vec = sit.board_cards().to_vec();

            let oop_rec = TrainingRecord {
                board: board_vec.clone(),
                pot: sit.pot as f32,
                effective_stack: sit.effective_stack as f32,
                player: 0,
                game_value: oop_gv,
                oop_range: sit.ranges[0],
                ip_range: sit.ranges[1],
                cfvs: oop_cfvs,
                valid_mask,
            };
            write_record(&mut writer, &oop_rec).unwrap();

            let ip_rec = TrainingRecord {
                board: board_vec,
                pot: sit.pot as f32,
                effective_stack: sit.effective_stack as f32,
                player: 1,
                game_value: ip_gv,
                oop_range: sit.ranges[0],
                ip_range: sit.ranges[1],
                cfvs: ip_cfvs,
                valid_mask,
            };
            write_record(&mut writer, &ip_rec).unwrap();
        }

        // Read back and verify.
        let mut file = std::fs::File::open(output.path()).unwrap();
        let rec0 = storage::read_record(&mut file).unwrap();
        let rec1 = storage::read_record(&mut file).unwrap();

        assert_eq!(rec0.board.len(), 4, "OOP record should have 4-card board");
        assert_eq!(rec1.board.len(), 4, "IP record should have 4-card board");
        assert_eq!(rec0.player, 0);
        assert_eq!(rec1.player, 1);
        assert!(rec0.pot > 0.0);
        assert!(rec0.effective_stack > 0.0);
    }

    #[test]
    fn fuzz_bet_sizes_zero_fuzz_returns_identical() {
        let sizes = vec![vec![0.5, 1.0], vec![0.75]];
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let result = fuzz_bet_sizes(&sizes, 0.0, &mut rng);
        assert_eq!(result, sizes);
    }

    #[test]
    fn fuzz_bet_sizes_negative_fuzz_returns_identical() {
        let sizes = vec![vec![0.5, 1.0]];
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let result = fuzz_bet_sizes(&sizes, -0.1, &mut rng);
        assert_eq!(result, sizes);
    }

    #[test]
    fn fuzz_bet_sizes_produces_different_values() {
        let sizes = vec![vec![0.5, 1.0]];
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let result1 = fuzz_bet_sizes(&sizes, 0.1, &mut rng);
        let result2 = fuzz_bet_sizes(&sizes, 0.1, &mut rng);
        // With different rng state, results should differ.
        assert_ne!(result1, result2, "two fuzz calls should produce different sizes");
    }

    #[test]
    fn fuzz_bet_sizes_values_within_expected_range() {
        let sizes = vec![vec![0.5, 1.0]];
        let fuzz = 0.1;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..100 {
            let result = fuzz_bet_sizes(&sizes, fuzz, &mut rng);
            for (depth_idx, depth) in result.iter().enumerate() {
                for (size_idx, &val) in depth.iter().enumerate() {
                    let original = sizes[depth_idx][size_idx];
                    // Value should be within [original * 0.9, original * 1.1]
                    assert!(val >= original * (1.0 - fuzz) - 1e-9,
                        "fuzzed {val} below min for original {original}");
                    assert!(val <= original * (1.0 + fuzz) + 1e-9,
                        "fuzzed {val} above max for original {original}");
                }
            }
        }
    }

    #[test]
    fn fuzz_bet_sizes_clamps_to_minimum() {
        // Very small bet size with large fuzz should not go below 0.01.
        let sizes = vec![vec![0.01]];
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..100 {
            let result = fuzz_bet_sizes(&sizes, 0.99, &mut rng);
            assert!(result[0][0] >= 0.01,
                "fuzzed size {} below 0.01 minimum", result[0][0]);
        }
    }

    #[test]
    fn fuzz_bet_sizes_preserves_structure() {
        let sizes = vec![vec![0.25, 0.5, 1.0], vec![0.75, 1.5]];
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let result = fuzz_bet_sizes(&sizes, 0.1, &mut rng);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 3);
        assert_eq!(result[1].len(), 2);
    }

    #[test]
    fn build_turn_game_exact_has_no_boundaries() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let datagen_config = DatagenConfig {
            num_samples: 1,
            street: "turn".into(),
            ..Default::default()
        };
        let sit = sample_situation(&datagen_config, 200, 4, &mut rng);
        if sit.effective_stack <= 0 {
            return;
        }
        let bet_sizes_f64 = parse_bet_sizes_depth(&["50%".into(), "a".into()]);
        let game = build_turn_game_exact(
            sit.board_cards(),
            f64::from(sit.pot),
            f64::from(sit.effective_stack),
            &sit.ranges,
            &[bet_sizes_f64],
        )
        .expect("game should be built");
        assert_eq!(game.num_boundary_nodes(), 0,
            "exact mode game should have no boundary nodes");
    }

    #[test]
    fn exact_mode_solve_produces_finite_cfvs() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let datagen_config = DatagenConfig {
            num_samples: 1,
            street: "turn".into(),
            ..Default::default()
        };
        let sit = sample_situation(&datagen_config, 200, 4, &mut rng);
        if sit.effective_stack <= 0 {
            return;
        }
        let bet_sizes_f64 = parse_bet_sizes_depth(&["50%".into(), "a".into()]);
        let mut game = build_turn_game_exact(
            sit.board_cards(),
            f64::from(sit.pot),
            f64::from(sit.effective_stack),
            &sit.ranges,
            &[bet_sizes_f64],
        )
        .expect("game should be built");

        // No boundary evaluation needed — solve directly.
        let (oop_cfvs, ip_cfvs, valid_mask, oop_gv, ip_gv, _exploit) = solve_and_extract(
            &mut game,
            f64::from(sit.pot),
            &sit.ranges,
            50,
            0.0,
        );

        let num_valid: usize = valid_mask.iter().map(|&v| v as usize).sum();
        assert!(num_valid > 0, "expected some valid combos in exact mode");

        for (i, &cfv) in oop_cfvs.iter().enumerate() {
            assert!(cfv.is_finite(), "exact OOP combo {i}: non-finite CFV {cfv}");
        }
        for (i, &cfv) in ip_cfvs.iter().enumerate() {
            assert!(cfv.is_finite(), "exact IP combo {i}: non-finite CFV {cfv}");
        }
        assert!(oop_gv.is_finite(), "exact OOP game value not finite");
        assert!(ip_gv.is_finite(), "exact IP game value not finite");
    }

    #[test]
    fn generate_turn_exact_mode_skips_river_model() {
        // In exact mode, river_model_path is not needed.
        let mut config = turn_test_config(1);
        config.datagen.mode = "exact".into();
        config.game.river_model_path = None;
        let output = NamedTempFile::new().unwrap();
        let result = generate_turn_training_data(&config, output.path(), "ndarray");
        // Should NOT error about river_model_path.
        assert!(result.is_ok(), "exact mode should not require river_model_path: {:?}", result.err());
    }

    #[test]
    fn extract_river_records_produces_5_card_boards() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let datagen_config = DatagenConfig {
            num_samples: 1,
            street: "turn".into(),
            ..Default::default()
        };
        let sit = sample_situation(&datagen_config, 200, 4, &mut rng);
        if sit.effective_stack <= 0 {
            return;
        }
        let bet_sizes_f64 = parse_bet_sizes_depth(&["50%".into(), "a".into()]);
        let mut game = build_turn_game_exact(
            sit.board_cards(),
            f64::from(sit.pot),
            f64::from(sit.effective_stack),
            &sit.ranges,
            &[bet_sizes_f64],
        )
        .expect("game should be built");

        // Solve the game fully.
        for t in 0..50 {
            solve_step(&mut game, t);
        }
        range_solver::finalize(&mut game);

        let records = extract_river_records(&mut game, &sit, sit.pot);
        assert!(!records.is_empty(), "should produce river records from exact solve");

        for (i, rec) in records.iter().enumerate() {
            assert_eq!(rec.board.len(), 5, "river record {i} should have 5-card board");
            // Board should start with the original 4 turn cards.
            assert_eq!(&rec.board[..4], sit.board_cards(), "river record {i} board prefix mismatch");
            // River card should be valid (0..52) and not duplicate any turn card.
            let river_card = rec.board[4];
            assert!(river_card < 52, "river record {i} river card out of range");
            assert!(!sit.board_cards().contains(&river_card),
                "river record {i} river card {river_card} conflicts with turn board");
        }
    }

    #[test]
    fn extract_river_records_have_finite_cfvs() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let datagen_config = DatagenConfig {
            num_samples: 1,
            street: "turn".into(),
            ..Default::default()
        };
        let sit = sample_situation(&datagen_config, 200, 4, &mut rng);
        if sit.effective_stack <= 0 {
            return;
        }
        let bet_sizes_f64 = parse_bet_sizes_depth(&["50%".into(), "a".into()]);
        let mut game = build_turn_game_exact(
            sit.board_cards(),
            f64::from(sit.pot),
            f64::from(sit.effective_stack),
            &sit.ranges,
            &[bet_sizes_f64],
        )
        .expect("game should be built");

        for t in 0..50 {
            solve_step(&mut game, t);
        }
        range_solver::finalize(&mut game);

        let records = extract_river_records(&mut game, &sit, sit.pot);
        assert!(!records.is_empty(), "should produce river records");

        for (i, rec) in records.iter().enumerate() {
            assert!(rec.pot > 0.0, "river record {i} pot should be positive");
            assert!(rec.effective_stack >= 0.0, "river record {i} stack should be non-negative");
            assert!(rec.player <= 1, "river record {i} player should be 0 or 1");

            let num_valid: usize = rec.valid_mask.iter().map(|&v| v as usize).sum();
            assert!(num_valid > 0, "river record {i} should have valid combos");

            for (j, &cfv) in rec.cfvs.iter().enumerate() {
                assert!(cfv.is_finite(), "river record {i} combo {j}: non-finite CFV {cfv}");
            }
            assert!(rec.game_value.is_finite(), "river record {i} game value not finite");
        }
    }

    #[test]
    fn extract_river_records_have_both_players() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let datagen_config = DatagenConfig {
            num_samples: 1,
            street: "turn".into(),
            ..Default::default()
        };
        let sit = sample_situation(&datagen_config, 200, 4, &mut rng);
        if sit.effective_stack <= 0 {
            return;
        }
        let bet_sizes_f64 = parse_bet_sizes_depth(&["50%".into(), "a".into()]);
        let mut game = build_turn_game_exact(
            sit.board_cards(),
            f64::from(sit.pot),
            f64::from(sit.effective_stack),
            &sit.ranges,
            &[bet_sizes_f64],
        )
        .expect("game should be built");

        for t in 0..50 {
            solve_step(&mut game, t);
        }
        range_solver::finalize(&mut game);

        let records = extract_river_records(&mut game, &sit, sit.pot);
        // Records come in pairs (OOP, IP) for each river card.
        assert!(records.len() % 2 == 0, "river records should come in pairs");

        let has_oop = records.iter().any(|r| r.player == 0);
        let has_ip = records.iter().any(|r| r.player == 1);
        assert!(has_oop, "should have OOP records");
        assert!(has_ip, "should have IP records");
    }

    #[test]
    fn exact_mode_pipeline_writes_river_records_when_configured() {
        use tempfile::TempDir;

        let mut config = turn_test_config(2);
        config.datagen.mode = "exact".into();
        config.game.river_model_path = None;

        let tmp_dir = TempDir::new().unwrap();
        let turn_output = tmp_dir.path().join("turn_data.bin");
        let river_output = tmp_dir.path().join("river_data.bin");
        config.datagen.river_output = Some(river_output.to_string_lossy().into_owned());

        let result = generate_turn_training_data(&config, &turn_output, "ndarray");
        assert!(result.is_ok(), "pipeline failed: {:?}", result.err());

        // Storage stage writes numbered files: turn_data_00000.bin, river_data_00000.bin
        let turn_file = tmp_dir.path().join("turn_data_00000.bin");
        let turn_meta = std::fs::metadata(&turn_file).unwrap();
        assert!(turn_meta.len() > 0, "turn output should be non-empty");

        let river_file = tmp_dir.path().join("river_data_00000.bin");
        assert!(river_file.exists(), "river output file should exist");
        let river_meta = std::fs::metadata(&river_file).unwrap();
        assert!(river_meta.len() > 0, "river output should be non-empty");

        // Read and verify river records.
        let mut rf = std::fs::File::open(&river_file).unwrap();
        let rec = storage::read_record(&mut rf).unwrap();
        assert_eq!(rec.board.len(), 5, "river record should have 5-card board");
    }

    fn make_test_record(board_size: usize, player: u8) -> TrainingRecord {
        TrainingRecord {
            board: (0..board_size as u8).map(|i| i * 4).collect(),
            pot: 100.0,
            effective_stack: 50.0,
            player,
            game_value: 0.05,
            oop_range: [0.0f32; NUM_COMBOS],
            ip_range: [0.0f32; NUM_COMBOS],
            cfvs: [0.0f32; NUM_COMBOS],
            valid_mask: [0u8; NUM_COMBOS],
        }
    }

    #[test]
    fn flush_buffer_writes_correct_record_count() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path().join("turn_exact.bin");

        let mut buffer = vec![
            make_test_record(4, 0),
            make_test_record(4, 1),
            make_test_record(4, 0),
        ];
        let mut file_count = 0u32;

        flush_buffer(&base_path, &mut buffer, &mut file_count).unwrap();

        assert!(buffer.is_empty(), "buffer should be drained after flush");
        assert_eq!(file_count, 1, "file_count should be incremented");

        // Read back records from the file.
        let file_path = tmp.path().join("turn_exact_00000.bin");
        assert!(file_path.exists(), "output file should exist");
        let mut f = std::fs::File::open(&file_path).unwrap();
        let r1 = storage::read_record(&mut f).unwrap();
        let r2 = storage::read_record(&mut f).unwrap();
        let r3 = storage::read_record(&mut f).unwrap();
        assert_eq!(r1.player, 0);
        assert_eq!(r2.player, 1);
        assert_eq!(r3.player, 0);
        // No more records.
        assert!(storage::read_record(&mut f).is_err());
    }

    #[test]
    fn flush_buffer_increments_file_count() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path().join("data.bin");

        let mut file_count = 0u32;

        let mut buf1 = vec![make_test_record(4, 0)];
        flush_buffer(&base_path, &mut buf1, &mut file_count).unwrap();
        assert_eq!(file_count, 1);

        let mut buf2 = vec![make_test_record(4, 1)];
        flush_buffer(&base_path, &mut buf2, &mut file_count).unwrap();
        assert_eq!(file_count, 2);

        assert!(tmp.path().join("data_00000.bin").exists());
        assert!(tmp.path().join("data_00001.bin").exists());
    }

    #[test]
    fn flush_buffer_empty_buffer_writes_empty_file() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let base_path = tmp.path().join("empty.bin");

        let mut buffer: Vec<TrainingRecord> = Vec::new();
        let mut file_count = 0u32;

        flush_buffer(&base_path, &mut buffer, &mut file_count).unwrap();

        let file_path = tmp.path().join("empty_00000.bin");
        assert!(file_path.exists());
        let meta = std::fs::metadata(&file_path).unwrap();
        assert_eq!(meta.len(), 0, "empty buffer should produce zero-byte file");
        assert_eq!(file_count, 1);
    }

    #[test]
    fn storage_stage_writes_turn_and_river_to_separate_files() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let turn_path = tmp.path().join("turn.bin");
        let river_path = tmp.path().join("river.bin");

        let (tx, rx) = std::sync::mpsc::sync_channel::<StorageMsg>(100);

        let turn_out = turn_path.clone();
        let river_out = river_path.clone();
        let per_file = u64::MAX; // flush only on Flush message
        let turn_stored = Arc::new(AtomicU64::new(0));
        let river_stored = Arc::new(AtomicU64::new(0));
        let ts = Arc::clone(&turn_stored);
        let rs = Arc::clone(&river_stored);

        let stage4 = std::thread::spawn(move || -> Result<(), String> {
            let mut turn_buffer: Vec<TrainingRecord> = Vec::new();
            let mut river_buffer: Vec<TrainingRecord> = Vec::new();
            let mut turn_file_count = 0u32;
            let mut river_file_count = 0u32;

            while let Ok(msg) = rx.recv() {
                match msg {
                    StorageMsg::TurnRecord(rec) => {
                        turn_buffer.push(rec);
                        ts.store(turn_buffer.len() as u64, Ordering::Relaxed);
                        if turn_buffer.len() as u64 >= per_file {
                            flush_buffer(&turn_out, &mut turn_buffer, &mut turn_file_count)?;
                        }
                    }
                    StorageMsg::RiverRecord(rec) => {
                        river_buffer.push(rec);
                        rs.store(river_buffer.len() as u64, Ordering::Relaxed);
                        if river_buffer.len() as u64 >= per_file {
                            flush_buffer(&river_out, &mut river_buffer, &mut river_file_count)?;
                        }
                    }
                    StorageMsg::Flush => break,
                }
            }

            if !turn_buffer.is_empty() {
                flush_buffer(&turn_out, &mut turn_buffer, &mut turn_file_count)?;
            }
            if !river_buffer.is_empty() {
                flush_buffer(&river_out, &mut river_buffer, &mut river_file_count)?;
            }
            Ok(())
        });

        // Send 2 turn records and 3 river records.
        tx.send(StorageMsg::TurnRecord(make_test_record(4, 0))).unwrap();
        tx.send(StorageMsg::TurnRecord(make_test_record(4, 1))).unwrap();
        tx.send(StorageMsg::RiverRecord(make_test_record(5, 0))).unwrap();
        tx.send(StorageMsg::RiverRecord(make_test_record(5, 1))).unwrap();
        tx.send(StorageMsg::RiverRecord(make_test_record(5, 0))).unwrap();
        tx.send(StorageMsg::Flush).unwrap();

        stage4.join().unwrap().unwrap();

        // Read turn records.
        let turn_file = tmp.path().join("turn_00000.bin");
        assert!(turn_file.exists(), "turn file should exist");
        let mut f = std::fs::File::open(&turn_file).unwrap();
        let t1 = storage::read_record(&mut f).unwrap();
        let t2 = storage::read_record(&mut f).unwrap();
        assert_eq!(t1.board.len(), 4);
        assert_eq!(t2.board.len(), 4);
        assert!(storage::read_record(&mut f).is_err(), "no more turn records");

        // Read river records.
        let river_file = tmp.path().join("river_00000.bin");
        assert!(river_file.exists(), "river file should exist");
        let mut f = std::fs::File::open(&river_file).unwrap();
        let r1 = storage::read_record(&mut f).unwrap();
        let r2 = storage::read_record(&mut f).unwrap();
        let r3 = storage::read_record(&mut f).unwrap();
        assert_eq!(r1.board.len(), 5);
        assert_eq!(r2.board.len(), 5);
        assert_eq!(r3.board.len(), 5);
        assert!(storage::read_record(&mut f).is_err(), "no more river records");
    }

    #[test]
    fn storage_stage_splits_files_by_per_file_limit() {
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let turn_path = tmp.path().join("turn.bin");
        let river_path = tmp.path().join("river.bin");

        let (tx, rx) = std::sync::mpsc::sync_channel::<StorageMsg>(100);

        let turn_out = turn_path.clone();
        let river_out = river_path.clone();
        let per_file = 2u64; // flush every 2 records
        let turn_stored = Arc::new(AtomicU64::new(0));
        let river_stored = Arc::new(AtomicU64::new(0));
        let ts = Arc::clone(&turn_stored);
        let rs = Arc::clone(&river_stored);

        let stage4 = std::thread::spawn(move || -> Result<(), String> {
            let mut turn_buffer: Vec<TrainingRecord> = Vec::new();
            let mut river_buffer: Vec<TrainingRecord> = Vec::new();
            let mut turn_file_count = 0u32;
            let mut river_file_count = 0u32;

            while let Ok(msg) = rx.recv() {
                match msg {
                    StorageMsg::TurnRecord(rec) => {
                        turn_buffer.push(rec);
                        ts.store(turn_buffer.len() as u64, Ordering::Relaxed);
                        if turn_buffer.len() as u64 >= per_file {
                            flush_buffer(&turn_out, &mut turn_buffer, &mut turn_file_count)?;
                            ts.store(0, Ordering::Relaxed);
                        }
                    }
                    StorageMsg::RiverRecord(rec) => {
                        river_buffer.push(rec);
                        rs.store(river_buffer.len() as u64, Ordering::Relaxed);
                        if river_buffer.len() as u64 >= per_file {
                            flush_buffer(&river_out, &mut river_buffer, &mut river_file_count)?;
                            rs.store(0, Ordering::Relaxed);
                        }
                    }
                    StorageMsg::Flush => break,
                }
            }

            if !turn_buffer.is_empty() {
                flush_buffer(&turn_out, &mut turn_buffer, &mut turn_file_count)?;
            }
            if !river_buffer.is_empty() {
                flush_buffer(&river_out, &mut river_buffer, &mut river_file_count)?;
            }
            Ok(())
        });

        // Send 5 turn records: should produce files of 2, 2, 1.
        for i in 0..5u8 {
            tx.send(StorageMsg::TurnRecord(make_test_record(4, i % 2))).unwrap();
        }
        tx.send(StorageMsg::Flush).unwrap();

        stage4.join().unwrap().unwrap();

        // Should have 3 turn files.
        assert!(tmp.path().join("turn_00000.bin").exists());
        assert!(tmp.path().join("turn_00001.bin").exists());
        assert!(tmp.path().join("turn_00002.bin").exists());
        assert!(!tmp.path().join("turn_00003.bin").exists());

        // First file: 2 records.
        let mut f = std::fs::File::open(tmp.path().join("turn_00000.bin")).unwrap();
        storage::read_record(&mut f).unwrap();
        storage::read_record(&mut f).unwrap();
        assert!(storage::read_record(&mut f).is_err());

        // Second file: 2 records.
        let mut f = std::fs::File::open(tmp.path().join("turn_00001.bin")).unwrap();
        storage::read_record(&mut f).unwrap();
        storage::read_record(&mut f).unwrap();
        assert!(storage::read_record(&mut f).is_err());

        // Third file: 1 record (remainder).
        let mut f = std::fs::File::open(tmp.path().join("turn_00002.bin")).unwrap();
        storage::read_record(&mut f).unwrap();
        assert!(storage::read_record(&mut f).is_err());
    }

    #[test]
    fn exact_mode_pipeline_splits_output_with_per_file() {
        use tempfile::TempDir;

        let mut config = turn_test_config(3);
        config.datagen.mode = "exact".into();
        config.game.river_model_path = None;
        config.datagen.per_file = Some(2); // flush every 2 turn situations (= 4 records: 2 OOP + 2 IP)

        let tmp_dir = TempDir::new().unwrap();
        let turn_output = tmp_dir.path().join("turn_exact.bin");

        let result = generate_turn_training_data(&config, &turn_output, "ndarray");
        assert!(result.is_ok(), "pipeline failed: {:?}", result.err());

        // With 3 samples at per_file=2, we expect multiple turn files.
        // Each sample produces 2 turn records (OOP + IP).
        // per_file=2 means flush after 2 records, so 6 records / 2 = 3 files.
        let turn_files: Vec<_> = std::fs::read_dir(tmp_dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with("turn_exact"))
            .collect();
        assert!(turn_files.len() > 1,
            "expected multiple turn files with per_file=2, got {}", turn_files.len());

        // All files should contain valid 4-card board records.
        for entry in &turn_files {
            let mut f = std::fs::File::open(entry.path()).unwrap();
            let rec = storage::read_record(&mut f).unwrap();
            assert_eq!(rec.board.len(), 4,
                "file {} should contain 4-card board records", entry.file_name().to_string_lossy());
        }
    }

    #[test]
    fn storage_stage_drops_river_records_when_no_river_path() {
        // When river output is not configured, river records should be dropped
        // (not sent to storage). This is tested at the pipeline level below.
        // Here we test the Stage 4 pattern with only turn records.
        use tempfile::TempDir;
        let tmp = TempDir::new().unwrap();
        let turn_path = tmp.path().join("turn.bin");

        let (tx, rx) = std::sync::mpsc::sync_channel::<StorageMsg>(100);

        let turn_out = turn_path.clone();
        let per_file = u64::MAX;

        let stage4 = std::thread::spawn(move || -> Result<(), String> {
            let mut turn_buffer: Vec<TrainingRecord> = Vec::new();
            let mut turn_file_count = 0u32;

            while let Ok(msg) = rx.recv() {
                match msg {
                    StorageMsg::TurnRecord(rec) => {
                        turn_buffer.push(rec);
                        if turn_buffer.len() as u64 >= per_file {
                            flush_buffer(&turn_out, &mut turn_buffer, &mut turn_file_count)?;
                        }
                    }
                    StorageMsg::RiverRecord(_) => {
                        // Drop river records if no river path.
                    }
                    StorageMsg::Flush => break,
                }
            }

            if !turn_buffer.is_empty() {
                flush_buffer(&turn_out, &mut turn_buffer, &mut turn_file_count)?;
            }
            Ok(())
        });

        tx.send(StorageMsg::TurnRecord(make_test_record(4, 0))).unwrap();
        // River record should be silently ignored.
        tx.send(StorageMsg::RiverRecord(make_test_record(5, 0))).unwrap();
        tx.send(StorageMsg::TurnRecord(make_test_record(4, 1))).unwrap();
        tx.send(StorageMsg::Flush).unwrap();

        stage4.join().unwrap().unwrap();

        // Only turn file should exist with 2 records.
        let turn_file = tmp.path().join("turn_00000.bin");
        assert!(turn_file.exists());
        let mut f = std::fs::File::open(&turn_file).unwrap();
        storage::read_record(&mut f).unwrap();
        storage::read_record(&mut f).unwrap();
        assert!(storage::read_record(&mut f).is_err());

        // No river file should exist.
        assert!(!tmp.path().join("river_00000.bin").exists());
    }
}
