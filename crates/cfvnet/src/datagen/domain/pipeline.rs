use std::path::Path;
use std::sync::Arc;

use indicatif::{ProgressBar, ProgressStyle};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::config::CfvnetConfig;

use super::evaluator::SolveStrategy;
use super::game::GameBuilder;
use super::neural_net_evaluator::NeuralNetEvaluator;
use super::situation::SituationGenerator;
use super::solver::{Solver, SolverConfig};
use super::writer::RecordWriter;

/// Coordinates the domain datagen pipeline: generate -> build -> solve -> write.
pub struct DomainPipeline;

impl DomainPipeline {
    pub fn run(config: &CfvnetConfig, output_path: &Path) -> Result<(), String> {
        let num_samples = config.datagen.num_samples;
        let seed = crate::config::resolve_seed(config.datagen.seed);
        let initial_stack = config.game.initial_stack;
        let board_size = config.game.board_size;

        // Parse bet sizes from config.
        let bet_sizes =
            crate::datagen::turn_generate::parse_bet_sizes_all(&config.game.bet_sizes);
        if bet_sizes.is_empty() {
            return Err("no valid bet sizes".into());
        }

        // Construct solve strategy: depth-limited with neural net for turn, exact for river.
        let strategy = if board_size < 5 {
            SolveStrategy::DepthLimited {
                evaluator: Arc::new(NeuralNetEvaluator::load(
                    config
                        .game
                        .river_model_path
                        .as_deref()
                        .ok_or("river_model_path required for turn datagen")?,
                    config,
                )?),
            }
        } else {
            SolveStrategy::Exact
        };

        // Construct domain objects.
        let range_source = super::RangeSource::from_config(&config.datagen)?;
        let sit_gen = SituationGenerator::new(&config.datagen, initial_stack, board_size, seed, num_samples)
            .with_range_source(range_source);
        let builder = GameBuilder::new(bet_sizes, &strategy)
            .with_fuzz(config.datagen.bet_size_fuzz);
        let solver_config = SolverConfig {
            max_iterations: config.datagen.solver_iterations,
            target_exploitability: config.datagen.target_exploitability,
            leaf_eval_interval: config.datagen.leaf_eval_interval,
        };

        let mut writer = RecordWriter::create(output_path, config.datagen.per_file)?;

        // Progress bar.
        let pb = ProgressBar::new(num_samples);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{wide_bar} {pos}/{len} [{elapsed_precise}] ETA {eta} ({per_sec}) {msg}")
                .expect("valid template"),
        );

        // Generate loop.
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut exploit_sum = 0.0f64;
        let mut exploit_count = 0u64;

        for sit in sit_gen {
            let game = match builder.build(&sit, &mut rng) {
                Some(g) => g,
                None => {
                    pb.inc(1);
                    continue;
                }
            };

            let mut solver = Solver::new(game, &solver_config, strategy.clone());
            let solved = loop {
                match solver.step() {
                    None => continue,
                    Some(sg) => break sg,
                }
            };

            // Track exploitability.
            if solved.exploitability.is_finite() {
                let bb = initial_stack as f32 / 100.0;
                let mbb = if bb > 0.0 {
                    solved.exploitability / bb * 1000.0
                } else {
                    0.0
                };
                exploit_sum += mbb as f64;
                exploit_count += 1;
            }

            let records = solved.extract_records();
            writer.write(&records)?;
            pb.inc(1);

            // Update progress bar.
            let avg_exploit = if exploit_count > 0 {
                exploit_sum / exploit_count as f64
            } else {
                0.0
            };
            pb.set_message(format!(
                "expl:{avg_exploit:.1} mbb/h  written:{}",
                writer.count()
            ));
        }

        writer.flush()?;
        pb.finish_with_message("done");

        eprintln!(
            "Wrote {} records to {}",
            writer.count(),
            output_path.display()
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CfvnetConfig, DatagenConfig, GameConfig};
    use crate::datagen::storage::read_record;
    use std::io::BufReader;
    use tempfile::NamedTempFile;

    fn test_config(num_samples: u64, board_size: usize) -> CfvnetConfig {
        CfvnetConfig {
            game: GameConfig {
                initial_stack: 200,
                board_size,
                ..Default::default()
            },
            datagen: DatagenConfig {
                num_samples,
                mode: "domain".into(),
                solver_iterations: 20,
                seed: Some(42),
                ..Default::default()
            },
            training: Default::default(),
            evaluation: Default::default(),
        }
    }

    #[test]
    fn pipeline_produces_records_for_river() {
        range_solver::set_force_sequential(true);
        let tmp = NamedTempFile::new().unwrap();
        let config = test_config(3, 5);
        DomainPipeline::run(&config, tmp.path()).unwrap();

        let mut reader = BufReader::new(std::fs::File::open(tmp.path()).unwrap());
        let r0 = read_record(&mut reader).unwrap();
        assert_eq!(r0.board.len(), 5);
        assert!(r0.pot > 0.0);
    }

    #[test]
    fn pipeline_produces_records_for_turn() {
        // Turn mode requires a river model which we can't load in unit tests.
        // Verify it returns the expected error.
        range_solver::set_force_sequential(true);
        let tmp = NamedTempFile::new().unwrap();
        let config = test_config(3, 4);
        let result = DomainPipeline::run(&config, tmp.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("river_model_path"));
    }

    #[test]
    fn pipeline_writes_correct_record_count() {
        range_solver::set_force_sequential(true);
        let tmp = NamedTempFile::new().unwrap();
        let config = test_config(5, 5);
        DomainPipeline::run(&config, tmp.path()).unwrap();

        let mut reader = BufReader::new(std::fs::File::open(tmp.path()).unwrap());
        let mut count = 0;
        while read_record(&mut reader).is_ok() {
            count += 1;
        }
        // Each game produces 2 records (OOP + IP), and we asked for 5 samples.
        // Some may be skipped (degenerate), but we should get at least 2 records.
        assert!(count >= 2, "expected at least 2 records, got {count}");
        assert_eq!(count % 2, 0, "records should come in pairs (OOP+IP)");
    }

    #[test]
    fn pipeline_records_have_valid_fields() {
        range_solver::set_force_sequential(true);
        let tmp = NamedTempFile::new().unwrap();
        let config = test_config(3, 5);
        DomainPipeline::run(&config, tmp.path()).unwrap();

        let mut reader = BufReader::new(std::fs::File::open(tmp.path()).unwrap());
        while let Ok(rec) = read_record(&mut reader) {
            assert_eq!(rec.board.len(), 5);
            assert!(rec.pot > 0.0);
            assert!(rec.effective_stack > 0.0);
            assert!(rec.game_value.is_finite());
            assert!(rec.player == 0 || rec.player == 1);
            for &cfv in &rec.cfvs {
                assert!(cfv.is_finite());
            }
        }
    }
}
