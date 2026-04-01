//! Turn training data generation pipeline.
//!
//! Samples random turn situations, solves them using `PostFlopGame` with
//! a `RiverNetEvaluator` as the leaf evaluator, extracts root CFVs, and
//! writes [`TrainingRecord`]s with 4-card boards.

use std::io::BufWriter;
use std::path::Path;

use burn::backend::wgpu::Wgpu;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
use indicatif::{ProgressBar, ProgressStyle};
use poker_solver_core::blueprint_v2::LeafEvaluator;
use poker_solver_core::poker::{Card, Suit, Value};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
use range_solver::bet_size::{BetSize, BetSizeOptions};
use range_solver::card::{card_pair_to_index, CardConfig, NOT_DEALT};
use range_solver::game::PostFlopGame;
use range_solver::range::Range as RsRange;
use range_solver::solve;
use rayon::prelude::*;

use super::range_gen::NUM_COMBOS;
use super::sampler::sample_situation;
use super::storage::{write_record, TrainingRecord};
use crate::config::CfvnetConfig;
use crate::eval::river_net_evaluator::RiverNetEvaluator;
use crate::model::network::{CfvNet, INPUT_SIZE};

type B = Wgpu;

/// Number of situations to generate, solve, and write per chunk.
/// Keeps peak memory bounded regardless of total `num_samples`.
const CHUNK_SIZE: u64 = 128;


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

/// Parse config bet size strings (e.g. `["50%", "100%", "a"]`) into pot fractions.
///
/// Entries like "a" (all-in) are skipped — the game tree builder adds all-in
/// automatically.
fn parse_bet_sizes(sizes: &[String]) -> Vec<f64> {
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

/// Solve a single turn situation and return per-player root CFVs mapped to 1326 indices.
///
/// Uses the range-solver with `depth_limit: Some(1)` so that river transitions
/// become depth boundary terminals. The leaf evaluator (river net) is called once
/// per boundary to fill in the boundary CFVs before solving.
///
/// Returns `(oop_cfvs_1326, ip_cfvs_1326, valid_mask, game_value_oop, game_value_ip)`.
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
) {
    // Build range-solver Range objects from 1326-indexed f32 arrays.
    let oop_range = RsRange::from_raw_data(&ranges[0]).expect("valid OOP range");
    let ip_range = RsRange::from_raw_data(&ranges[1]).expect("valid IP range");

    // Build bet size options from pot fractions.
    let sizes: Vec<BetSize> = bet_sizes
        .iter()
        .flat_map(|v| v.iter().map(|&f| BetSize::PotRelative(f)))
        .collect();
    let bet_size_opts = BetSizeOptions {
        bet: sizes.clone(),
        raise: Vec::new(),
    };

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [board_u8[0], board_u8[1], board_u8[2]],
        turn: board_u8[3],
        river: NOT_DEALT,
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: pot as i32,
        effective_stack: effective_stack as i32,
        turn_bet_sizes: [bet_size_opts.clone(), bet_size_opts],
        river_bet_sizes: [BetSizeOptions::default(), BetSizeOptions::default()],
        depth_limit: Some(0),
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).expect("valid action tree");
    let mut game = PostFlopGame::with_config(card_config, action_tree).expect("valid game");
    game.allocate_memory(false);

    // Convert board from u8 to poker_solver_core::Card for the evaluator.
    let board_cards: Vec<Card> = board_u8.iter().map(|&c| u8_to_rs_card(c)).collect();

    // Evaluate all boundary nodes using batched evaluation (one forward pass).
    let num_boundaries = game.num_boundary_nodes();
    if num_boundaries > 0 {
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

        // One batched call — GPU implementations do one forward pass.
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
                let cfvs_f32: Vec<f32> =
                    all_cfvs[req_idx].iter().map(|&v| v as f32).collect();
                game.set_boundary_cfvs(ordinal, player, cfvs_f32);
            }
        }
    }
    // Solve the depth-limited turn game.
    // target_exploitability is pot-relative; scale to absolute chips.
    let abs_target = target_exploitability * pot as f32;
    solve(&mut game, solver_iterations, abs_target, false);

    // Extract root EVs.
    game.back_to_root();
    game.cache_normalized_weights();
    let raw_oop = game.expected_values(0);
    let raw_ip = game.expected_values(1);

    let oop_hands = game.private_cards(0);
    let ip_hands = game.private_cards(1);

    // Map solver EVs (absolute chips) to 1326-indexed pot-relative arrays.
    // expected_values() returns total payoff (includes player's contribution).
    // Net gain = ev - pot/2. Normalize to half-pot-relative: (ev - pot/2) / (pot/2).
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

    // Compute weighted game values.
    let oop_gv = weighted_sum(&ranges[0], &oop_cfvs);
    let ip_gv = weighted_sum(&ranges[1], &ip_cfvs);

    (oop_cfvs, ip_cfvs, valid_mask, oop_gv, ip_gv)
}

/// Compute `sum(range[i] * cfvs[i])` for all combos.
fn weighted_sum(range: &[f32; NUM_COMBOS], cfvs: &[f32; NUM_COMBOS]) -> f32 {
    range.iter().zip(cfvs.iter()).map(|(&r, &c)| r * c).sum()
}

/// Build a depth-limited turn game tree for a situation.
///
/// Returns `None` for degenerate situations (effective_stack <= 0).
#[allow(dead_code)]
fn build_turn_game(
    board_u8: &[u8],
    pot: f64,
    effective_stack: f64,
    ranges: &[[f32; NUM_COMBOS]; 2],
    bet_sizes: &[Vec<f64>],
) -> Option<PostFlopGame> {
    let oop_range = RsRange::from_raw_data(&ranges[0]).expect("valid OOP range");
    let ip_range = RsRange::from_raw_data(&ranges[1]).expect("valid IP range");

    let sizes: Vec<BetSize> = bet_sizes
        .iter()
        .flat_map(|v| v.iter().map(|&f| BetSize::PotRelative(f)))
        .collect();
    let bet_size_opts = BetSizeOptions {
        bet: sizes.clone(),
        raise: Vec::new(),
    };

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: [board_u8[0], board_u8[1], board_u8[2]],
        turn: board_u8[3],
        river: NOT_DEALT,
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: pot as i32,
        effective_stack: effective_stack as i32,
        turn_bet_sizes: [bet_size_opts.clone(), bet_size_opts],
        river_bet_sizes: [BetSizeOptions::default(), BetSizeOptions::default()],
        depth_limit: Some(0),
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
        ..Default::default()
    };

    let action_tree = ActionTree::new(tree_config).expect("valid action tree");
    let mut game = PostFlopGame::with_config(card_config, action_tree).expect("valid game");
    game.allocate_memory(false);
    Some(game)
}

/// Solve a game with boundaries already set. Returns 1326-indexed CFVs.
#[allow(dead_code, clippy::type_complexity)]
fn solve_and_extract(
    game: &mut PostFlopGame,
    pot: f64,
    ranges: &[[f32; NUM_COMBOS]; 2],
    solver_iterations: u32,
    target_exploitability: f32,
) -> ([f32; NUM_COMBOS], [f32; NUM_COMBOS], [u8; NUM_COMBOS], f32, f32) {
    let abs_target = target_exploitability * pot as f32;
    solve(game, solver_iterations, abs_target, false);

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

    (oop_cfvs, ip_cfvs, valid_mask, oop_gv, ip_gv)
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
    let target_exploitability = config.datagen.target_exploitability as f32;
    let bet_sizes_f64 = parse_bet_sizes(&config.game.bet_sizes);
    if bet_sizes_f64.is_empty() {
        return Err("no valid percentage bet sizes found in config".into());
    }
    let bet_sizes_vec = vec![bet_sizes_f64];

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
            if let Some((oop_cfvs, ip_cfvs, valid_mask, oop_gv, ip_gv)) = result {
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
            .map(|_| sample_situation(&config.datagen, config.game.initial_stack, 4, &mut rng))
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

/// Generate turn training data by sampling situations, solving with
/// `PostFlopGame` + `RiverNetEvaluator`, and writing paired records.
///
/// The river model is loaded once and cloned for each situation's evaluator.
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
    let target_exploitability = config.datagen.target_exploitability as f32;
    let bet_sizes_f64 = parse_bet_sizes(&config.game.bet_sizes);
    if bet_sizes_f64.is_empty() {
        return Err("no valid percentage bet sizes found in config".into());
    }
    let bet_sizes_vec = vec![bet_sizes_f64];

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

    let pb = ProgressBar::new(num_samples);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} [{elapsed_precise}] ETA {eta} ({per_sec}) {msg}")
            .expect("valid progress bar template"),
    );
    pb.enable_steady_tick(std::time::Duration::from_secs(1));

    // Open output file once, write incrementally across chunks.
    let file =
        std::fs::File::create(output_path).map_err(|e| format!("create output: {e}"))?;
    let mut writer = BufWriter::new(file);

    // Build thread pool once if multi-threaded.
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

    while remaining > 0 {
        let chunk_len = remaining.min(CHUNK_SIZE);
        remaining -= chunk_len;

        // Sample situations sequentially for determinism.
        let situations: Vec<_> = (0..chunk_len)
            .map(|_| sample_situation(&config.datagen, config.game.initial_stack, 4, &mut rng))
            .collect();

        // Solve chunk in parallel. Each rayon thread loads its own model
        // on first use via map_init — no cloning, no sharing.
        let river_model_path = river_model_path.to_string();
        let hidden_layers = river_hidden_layers;
        let hidden_size = river_hidden_size;

        let results: Vec<_> = match &pool {
            Some(pool) => pool.install(|| {
                situations.par_iter().map_init(
                    || {
                        let device = <B as burn::tensor::backend::Backend>::Device::default();
                        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
                        let m = CfvNet::<B>::new(&device, hidden_layers, hidden_size, INPUT_SIZE)
                            .load_file(&river_model_path, &recorder, &device)
                            .expect("load river model in worker thread");
                        RiverNetEvaluator::new(m, device)
                    },
                    |evaluator, sit| {
                        if sit.effective_stack <= 0 {
                            pb.inc(1);
                            return None;
                        }
                        let result = solve_turn_situation(
                            sit.board_cards(),
                            f64::from(sit.pot),
                            f64::from(sit.effective_stack),
                            &sit.ranges,
                            &bet_sizes_vec,
                            solver_iterations,
                            target_exploitability,
                            evaluator,
                        );
                        pb.inc(1);
                        Some(result)
                    },
                ).collect()
            }),
            None => {
                // Sequential fallback — single evaluator, no cloning.
                let device = <B as burn::tensor::backend::Backend>::Device::default();
                let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
                let m = CfvNet::<B>::new(&device, hidden_layers, hidden_size, INPUT_SIZE)
                    .load_file(&river_model_path, &recorder, &device)
                    .expect("load river model");
                let evaluator = RiverNetEvaluator::new(m, device);
                situations.iter().map(|sit| {
                    if sit.effective_stack <= 0 {
                        pb.inc(1);
                        return None;
                    }
                    let result = solve_turn_situation(
                        sit.board_cards(),
                        f64::from(sit.pot),
                        f64::from(sit.effective_stack),
                        &sit.ranges,
                        &bet_sizes_vec,
                        solver_iterations,
                        target_exploitability,
                        &evaluator,
                    );
                    pb.inc(1);
                    Some(result)
                }).collect()
            },
        };

        // Write results sequentially.
        for (sit, result) in situations.iter().zip(results) {
            if let Some((oop_cfvs, ip_cfvs, valid_mask, oop_gv, ip_gv)) = result {
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
                write_record(&mut writer, &oop_rec).map_err(|e| format!("write OOP: {e}"))?;

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
                write_record(&mut writer, &ip_rec).map_err(|e| format!("write IP: {e}"))?;
            }
        }
    }

    pb.finish_with_message("done");
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
                bet_sizes: vec!["50%".into(), "a".into()],
                board_size: 4,
                river_model_path: None, // tests don't load a real model
                ..Default::default()
            },
            datagen: DatagenConfig {
                num_samples,
                street: "turn".into(),
                solver_iterations: 50,
                target_exploitability: 0.05,
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
        let parsed = parse_bet_sizes(&sizes);
        assert_eq!(parsed.len(), 2);
        assert!((parsed[0] - 0.5).abs() < 1e-10);
        assert!((parsed[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn parse_bet_sizes_only_allin() {
        let sizes = vec!["a".into()];
        let parsed = parse_bet_sizes(&sizes);
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
            target_exploitability: 0.05,
            threads: 1,
            seed: Some(42),
            ..Default::default()
        };
        let sit = sample_situation(&datagen_config, 200, 4, &mut rng);
        assert_eq!(sit.board_size, 4);

        if sit.effective_stack <= 0 {
            return; // Skip degenerate situation.
        }

        let bet_sizes_f64 = parse_bet_sizes(&["50%".into(), "a".into()]);
        let (oop_cfvs, ip_cfvs, valid_mask, oop_gv, ip_gv) = solve_turn_situation(
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
            target_exploitability: 0.05,
            threads: 1,
            seed: Some(123),
            ..Default::default()
        };
        let sit = sample_situation(&datagen_config, 200, 4, &mut rng);
        if sit.effective_stack <= 0 {
            return;
        }

        let bet_sizes_f64 = parse_bet_sizes(&["50%".into()]);
        let (oop_cfvs, ip_cfvs, valid_mask, oop_gv, ip_gv) = solve_turn_situation(
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
}
