use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};

use crate::config::CfvnetConfig;
use crate::datagen::manifest::{SourceMetadata, TargetSource};
use crate::datagen::turn_boundary_dataset::TurnBoundaryDatasetWriter;
#[cfg(feature = "gpu-turn-datagen")]
use crate::datagen::turn_boundary_oracle::BoundaryNetRiverRunoutOracle;
use crate::datagen::turn_boundary_oracle::{
    build_turn_boundary_record, ExactRiverSolverOracle, RiverRunoutOracle, TurnBoundaryInput,
};

use super::domain::{RangeSource, SituationGenerator};

/// Generate a manifest-backed turn-boundary oracle dataset.
pub fn generate_turn_boundary_data(
    config: &CfvnetConfig,
    output_path: &Path,
) -> Result<(), String> {
    let target_source = parse_target_source(&config.datagen.turn_boundary_target_source)?;
    match target_source {
        TargetSource::RiverNet => {
            #[cfg(feature = "gpu-turn-datagen")]
            {
                let model_path =
                    config.game.river_model_path.as_deref().ok_or(
                        "river_model_path is required for turn_boundary river_net datagen",
                    )?;
                let evaluator = crate::datagen::gpu_boundary_eval::GpuBoundaryEvaluator::load(
                    Path::new(model_path),
                )?;
                let source = SourceMetadata {
                    source_model_path: Some(model_path.to_string()),
                    ..SourceMetadata::default()
                };
                generate_turn_boundary_data_with_oracle(
                    config,
                    output_path,
                    TargetSource::RiverNet,
                    source,
                    &BoundaryNetRiverRunoutOracle::new(evaluator),
                )?;
                Ok(())
            }
            #[cfg(not(feature = "gpu-turn-datagen"))]
            {
                Err("turn_boundary river_net datagen requires --features gpu-turn-datagen".into())
            }
        }
        TargetSource::ExactRiver => {
            let oracle = ExactRiverSolverOracle::new(build_exact_solve_config(config)?);
            generate_turn_boundary_data_with_oracle(
                config,
                output_path,
                TargetSource::ExactRiver,
                SourceMetadata::default(),
                &oracle,
            )?;
            Ok(())
        }
        TargetSource::Mixed => Err("turn_boundary target source 'mixed' is manifest-only".into()),
    }
}

pub fn generate_turn_boundary_data_with_oracle<O: RiverRunoutOracle>(
    config: &CfvnetConfig,
    output_path: &Path,
    target_source: TargetSource,
    source: SourceMetadata,
    oracle: &O,
) -> Result<u64, String> {
    let seed = crate::config::resolve_seed(config.datagen.seed);
    let range_source = RangeSource::from_config(&config.datagen)?;
    let mut sit_gen = SituationGenerator::new(
        &config.datagen,
        config.game.initial_stack,
        4,
        seed,
        config.datagen.num_samples,
    )
    .with_range_source(range_source);

    let mut writer = TurnBoundaryDatasetWriter::create(
        output_path,
        config.datagen.per_file,
        target_source,
        source,
    )?;

    let pb = ProgressBar::new(config.datagen.num_samples);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} [{elapsed_precise}] ETA {eta} ({per_sec}) {msg}")
            .expect("valid template"),
    );

    for sit in &mut sit_gen {
        let input = TurnBoundaryInput {
            board: [sit.board[0], sit.board[1], sit.board[2], sit.board[3]],
            pot: sit.pot as f32,
            effective_stack: sit.effective_stack as f32,
            player: 0,
            oop_range: sit.ranges[0],
            ip_range: sit.ranges[1],
        };
        let oop = build_turn_boundary_record(&input, oracle)?;
        let ip = build_turn_boundary_record(&TurnBoundaryInput { player: 1, ..input }, oracle)?;
        writer.write(&[oop, ip])?;
        pb.inc(1);
        pb.set_message(format!("written:{}", writer.count()));
    }

    pb.finish_with_message("done");
    let manifest = writer.finish().map_err(|e| e.to_string())?;
    eprintln!(
        "Wrote {} turn-boundary records to {} with manifest.yaml",
        manifest.coverage.total_records,
        output_path.display()
    );
    Ok(manifest.coverage.total_records)
}

fn parse_target_source(value: &str) -> Result<TargetSource, String> {
    match value {
        "river_net" => Ok(TargetSource::RiverNet),
        "exact_river" => Ok(TargetSource::ExactRiver),
        "mixed" => Ok(TargetSource::Mixed),
        other => Err(format!(
            "unknown turn_boundary_target_source '{other}', expected river_net or exact_river"
        )),
    }
}

fn build_exact_solve_config(
    config: &CfvnetConfig,
) -> Result<crate::datagen::solver::SolveConfig, String> {
    let bet_str = config.game.bet_sizes.join_flat(",");
    let bet_sizes = range_solver::bet_size::BetSizeOptions::try_from((bet_str.as_str(), ""))
        .map_err(|e| format!("invalid bet sizes: {e}"))?;
    Ok(crate::datagen::solver::SolveConfig {
        bet_sizes,
        solver_iterations: config.datagen.solver_iterations,
        target_exploitability: config.datagen.target_exploitability,
        add_allin_threshold: config.game.add_allin_threshold,
        force_allin_threshold: config.game.force_allin_threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        BetSizeConfig, DatagenConfig, EvaluationConfig, GameConfig, TrainingConfig,
    };
    use crate::datagen::manifest::DatasetManifest;
    use crate::datagen::storage::NUM_COMBOS;
    use crate::datagen::turn_boundary_oracle::{RiverRunoutInput, RiverRunoutOracle};
    use tempfile::TempDir;

    struct ConstantOracle(f32);

    impl RiverRunoutOracle for ConstantOracle {
        fn evaluate(&self, _input: RiverRunoutInput<'_>) -> Result<[f32; NUM_COMBOS], String> {
            Ok([self.0; NUM_COMBOS])
        }
    }

    fn test_config(num_samples: u64, per_file: Option<u64>) -> CfvnetConfig {
        CfvnetConfig {
            game: GameConfig {
                initial_stack: 200,
                bet_sizes: BetSizeConfig(vec![vec!["50%".into(), "a".into()]]),
                board_size: 4,
                ..Default::default()
            },
            datagen: DatagenConfig {
                num_samples,
                street: "turn_boundary".into(),
                turn_boundary_target_source: "exact_river".into(),
                pot_intervals: vec![[20, 21]],
                spr_intervals: Some(vec![[1.0, 1.1]]),
                threads: 1,
                seed: Some(42),
                per_file,
                ..Default::default()
            },
            training: TrainingConfig::default(),
            evaluation: EvaluationConfig::default(),
        }
    }

    #[test]
    fn writes_turn_boundary_records_and_manifest() {
        let dir = TempDir::new().unwrap();
        let output = dir.path().join("turn_boundary.bin");

        let written = generate_turn_boundary_data_with_oracle(
            &test_config(3, Some(2)),
            &output,
            TargetSource::ExactRiver,
            SourceMetadata::default(),
            &ConstantOracle(0.25),
        )
        .unwrap();

        assert_eq!(written, 6);
        let manifest = DatasetManifest::read_yaml(dir.path().join("manifest.yaml")).unwrap();
        manifest.validate_turn_boundary().unwrap();
        assert_eq!(manifest.target_source, TargetSource::ExactRiver);
        assert_eq!(manifest.coverage.total_records, 6);
        assert_eq!(manifest.shards.len(), 3);
    }

    #[test]
    fn rejects_unknown_target_source() {
        let err = parse_target_source("banana").unwrap_err();
        assert!(err.contains("unknown turn_boundary_target_source"));
    }
}
