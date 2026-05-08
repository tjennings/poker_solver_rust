use std::path::Path;
#[cfg(feature = "onnx")]
use std::sync::Arc;

#[cfg(feature = "onnx")]
use indicatif::{ProgressBar, ProgressStyle};
#[cfg(feature = "onnx")]
use rand::SeedableRng;
#[cfg(feature = "onnx")]
use rand_chacha::ChaCha8Rng;

use crate::config::CfvnetConfig;
#[cfg(feature = "onnx")]
use crate::datagen::domain::game::Game;
#[cfg(feature = "onnx")]
use crate::datagen::domain::solver::{SolvedGame, SolverConfig};
#[cfg(feature = "onnx")]
use crate::datagen::manifest::{SourceMetadata, TargetSource};
#[cfg(feature = "onnx")]
use crate::datagen::turn_boundary_dataset::FlopBoundaryDatasetWriter;

/// Generate direct flop-boundary records by solving flop games to 4-card turn
/// boundary nodes, evaluating those nodes with the direct turn-boundary net.
pub fn generate_flop_boundary_data(
    config: &CfvnetConfig,
    output_path: &Path,
) -> Result<(), String> {
    generate_flop_boundary_data_impl(config, output_path)
}

#[cfg(not(feature = "onnx"))]
fn generate_flop_boundary_data_impl(
    _config: &CfvnetConfig,
    _output_path: &Path,
) -> Result<(), String> {
    Err("flop_boundary turn_net datagen requires --features onnx or gpu-turn-datagen".into())
}

#[cfg(feature = "onnx")]
fn generate_flop_boundary_data_impl(
    config: &CfvnetConfig,
    output_path: &Path,
) -> Result<(), String> {
    let target_source = parse_target_source(&config.datagen.turn_boundary_target_source)?;
    match target_source {
        TargetSource::TurnNet => generate_with_turn_net(config, output_path),
        TargetSource::ExactTurn => {
            Err("flop_boundary target source 'exact_turn' is not implemented yet".into())
        }
        TargetSource::Mixed => Err("flop_boundary target source 'mixed' is manifest-only".into()),
        other => Err(format!(
            "flop_boundary target source '{}' is invalid; expected turn_net or exact_turn",
            other.as_str()
        )),
    }
}

#[cfg(feature = "onnx")]
fn parse_target_source(value: &str) -> Result<TargetSource, String> {
    match value {
        "turn_net" => Ok(TargetSource::TurnNet),
        "exact_turn" => Ok(TargetSource::ExactTurn),
        "mixed" => Ok(TargetSource::Mixed),
        "river_net" => Ok(TargetSource::RiverNet),
        "exact_river" => Ok(TargetSource::ExactRiver),
        other => Err(format!(
            "unknown flop_boundary target source '{other}', expected turn_net or exact_turn"
        )),
    }
}

#[cfg(feature = "onnx")]
fn generate_with_turn_net(config: &CfvnetConfig, output_path: &Path) -> Result<(), String> {
    use crate::datagen::domain::game::GameBuilder;
    use crate::datagen::domain::game_tree::parse_bet_sizes_all;
    use crate::datagen::domain::situation::{RangeSource, SituationGenerator};
    use crate::eval::boundary_evaluator::load_shared_onnx_session;

    if config.game.board_size != 3 {
        return Err(format!(
            "flop_boundary datagen requires game.board_size=3, got {}",
            config.game.board_size
        ));
    }

    let model_path = config
        .game
        .river_model_path
        .as_deref()
        .ok_or("river_model_path must point to the direct turn-boundary ONNX model")?;
    let session = load_shared_onnx_session(Path::new(model_path))?;
    let source = SourceMetadata {
        source_model_path: Some(model_path.to_string()),
        ..SourceMetadata::default()
    };

    let num_samples = config.datagen.num_samples;
    let seed = crate::config::resolve_seed(config.datagen.seed);
    let range_source_label = if config.datagen.blueprint_path.is_some() {
        "blueprint"
    } else {
        "rsp"
    };
    let range_source = RangeSource::from_config(&config.datagen)?;
    let bet_sizes = parse_bet_sizes_all(&config.game.bet_sizes);
    if bet_sizes.is_empty() {
        return Err("no valid bet sizes".into());
    }

    let mut sit_gen = SituationGenerator::new(
        &config.datagen,
        config.game.initial_stack,
        3,
        seed,
        num_samples,
    )
    .with_range_source(range_source);
    let builder = GameBuilder::depth_limited(bet_sizes).with_fuzz(config.datagen.bet_size_fuzz);
    let solver_config = SolverConfig {
        max_iterations: config.datagen.solver_iterations,
        target_exploitability: config.datagen.target_exploitability,
        leaf_eval_interval: config.datagen.leaf_eval_interval,
    };
    let mut writer = FlopBoundaryDatasetWriter::create(
        output_path,
        config.datagen.per_file,
        TargetSource::TurnNet,
        source,
    )?;

    let pb = ProgressBar::new(num_samples);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} [{elapsed_precise}] ETA {eta} ({per_sec}) {msg}")
            .expect("valid template"),
    );

    let _force_sequential = ForceSequentialGuard::new(true);
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x66c9_58d3_9a2f_0b51);
    let mut solved_count = 0_u64;
    let mut exploit_sum = 0.0_f64;
    let mut exploit_max = 0.0_f32;

    while let Some(sit) = sit_gen.next() {
        let Some(mut game) = builder.build(&sit, &mut rng) else {
            pb.inc(1);
            continue;
        };
        install_turn_boundary_evaluators(&mut game, Arc::clone(&session))?;

        let solved = solve_game(game, &solver_config);
        exploit_sum += f64::from(solved.exploitability.max(0.0));
        exploit_max = exploit_max.max(solved.exploitability);
        solved_count += 1;

        let records = solved.extract_records();
        writer.write_with_coverage(
            &records,
            range_source_label,
            "flop_subgame",
            "turn_boundary",
        )?;
        pb.inc(1);
        let avg_expl = exploit_sum / solved_count.max(1) as f64;
        pb.set_message(format!(
            "expl:{avg_expl:.3} max:{exploit_max:.3} written:{}",
            writer.count()
        ));
    }
    let manifest = writer.finish().map_err(|e| e.to_string())?;
    pb.finish_with_message("done");
    let avg_expl = exploit_sum / solved_count.max(1) as f64;
    eprintln!(
        "Wrote {} flop-boundary records to {} (avg exploitability {:.3}, max {:.3})",
        manifest.coverage.total_records,
        output_path.display(),
        avg_expl,
        exploit_max
    );
    Ok(())
}

#[cfg(feature = "onnx")]
struct ForceSequentialGuard(bool);

#[cfg(feature = "onnx")]
impl ForceSequentialGuard {
    fn new(force: bool) -> Self {
        Self(range_solver::set_force_sequential(force))
    }
}

#[cfg(feature = "onnx")]
impl Drop for ForceSequentialGuard {
    fn drop(&mut self) {
        range_solver::set_force_sequential(self.0);
    }
}

#[cfg(feature = "onnx")]
fn install_turn_boundary_evaluators(
    game: &mut Game,
    session: Arc<ort::session::Session>,
) -> Result<(), String> {
    use crate::eval::boundary_evaluator::{
        neural_boundary_evaluator_from_shared_with_mode, BoundaryInferenceMode,
    };

    let boundary_boards = game.inner().boundary_boards();
    if boundary_boards.is_empty() {
        return Err("flop game produced no turn boundary boards".into());
    }
    if let Some((ordinal, board)) = boundary_boards
        .iter()
        .enumerate()
        .find(|(_, board)| board.len() != 4)
    {
        return Err(format!(
            "turn-boundary model requires 4-card boundary board, ordinal {ordinal} has {} cards",
            board.len()
        ));
    }

    let private_cards = [
        game.inner().private_cards(0).to_vec(),
        game.inner().private_cards(1).to_vec(),
    ];
    let evaluators = boundary_boards
        .into_iter()
        .map(|board| {
            Arc::new(neural_boundary_evaluator_from_shared_with_mode(
                Arc::clone(&session),
                board,
                private_cards.clone(),
                BoundaryInferenceMode::Direct,
            )) as Arc<dyn range_solver::game::BoundaryEvaluator>
        })
        .collect();

    game.inner_mut().per_boundary_evaluators = evaluators;
    game.inner_mut().boundary_evaluator = None;
    Ok(())
}

#[cfg(feature = "onnx")]
fn solve_game(mut game: Game, config: &SolverConfig) -> SolvedGame {
    for iteration in 0..config.max_iterations {
        if config.leaf_eval_interval > 0
            && iteration > 0
            && iteration % config.leaf_eval_interval == 0
        {
            game.inner().flush_boundary_caches();
        }
        game.solve_step(iteration);
        if let Some(target) = config.target_exploitability {
            if iteration > 0 && iteration % 10 == 0 {
                game.inner().flush_boundary_caches();
                let exploit = game.compute_exploitability();
                let abs_target = target * game.situation().pot as f32;
                if exploit <= abs_target {
                    game.finalize();
                    game.back_to_root();
                    game.cache_normalized_weights();
                    return SolvedGame {
                        game,
                        exploitability: exploit,
                    };
                }
            }
        }
    }

    game.inner().flush_boundary_caches();
    game.finalize();
    game.back_to_root();
    game.cache_normalized_weights();
    let exploitability = game.compute_exploitability();
    SolvedGame {
        game,
        exploitability,
    }
}
