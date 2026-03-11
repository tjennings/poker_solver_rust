use std::io::BufWriter;
use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use super::range_gen::NUM_COMBOS;
use super::sampler::sample_situation;
use super::solver::{solve_situation, SolveConfig, SolveResult};
use super::storage::{write_record, TrainingRecord};
use crate::config::CfvnetConfig;

/// Generate training data by sampling river situations, solving each with DCFR,
/// and writing two records (OOP + IP) per solved situation to `output_path`.
///
/// Situations with `effective_stack == 0` are skipped since the solver requires
/// a positive stack. The progress bar tracks solve completions (including skips).
pub fn generate_training_data(config: &CfvnetConfig, output_path: &Path) -> Result<(), String> {
    let num_samples = config.datagen.num_samples;
    let seed = config.datagen.seed;
    let threads = config.datagen.threads;

    let solve_config = SolveConfig {
        bet_sizes: config.game.bet_sizes.clone(),
        solver_iterations: config.datagen.solver_iterations,
        target_exploitability: config.datagen.target_exploitability,
        add_allin_threshold: config.game.add_allin_threshold,
        force_allin_threshold: config.game.force_allin_threshold,
    };

    // Generate all situations sequentially for determinism with a fixed seed.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let situations: Vec<_> = (0..num_samples)
        .map(|_| sample_situation(&config.datagen, config.game.initial_stack, &mut rng))
        .collect();

    // Solve in parallel (or sequentially for determinism in tests).
    let pb = ProgressBar::new(num_samples);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} [{elapsed_precise}] {msg}")
            .expect("valid progress bar template"),
    );

    let results: Vec<Option<SolveResult>> = if threads <= 1 {
        situations
            .iter()
            .map(|sit| {
                let r = if sit.effective_stack <= 0 {
                    None
                } else {
                    Some(solve_situation(sit, &solve_config))
                };
                pb.inc(1);
                r
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|r| match r {
                None => Ok(None),
                Some(Ok(res)) => Ok(Some(res)),
                Some(Err(e)) => Err(e),
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| format!("thread pool: {e}"))?;
        let raw: Vec<Option<Result<SolveResult, String>>> = pool.install(|| {
            situations
                .par_iter()
                .map(|sit| {
                    let r = if sit.effective_stack <= 0 {
                        None
                    } else {
                        Some(solve_situation(sit, &solve_config))
                    };
                    pb.inc(1);
                    r
                })
                .collect()
        });
        raw.into_iter()
            .map(|r| match r {
                None => Ok(None),
                Some(Ok(res)) => Ok(Some(res)),
                Some(Err(e)) => Err(e),
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    pb.finish_with_message("done");

    // Write results: 2 records per solved situation (OOP + IP).
    let file =
        std::fs::File::create(output_path).map_err(|e| format!("create output: {e}"))?;
    let mut writer = BufWriter::new(file);

    for (sit, result) in situations.iter().zip(results.into_iter()) {
        let result = match result {
            Some(r) => r,
            None => continue, // skipped (effective_stack == 0)
        };

        let valid_mask = bool_mask_to_u8(&result.valid_mask);

        let oop_rec = TrainingRecord {
            board: sit.board,
            pot: sit.pot as f32,
            effective_stack: sit.effective_stack as f32,
            player: 0,
            game_value: result.oop_game_value,
            oop_range: sit.ranges[0],
            ip_range: sit.ranges[1],
            cfvs: result.oop_evs,
            valid_mask,
        };
        write_record(&mut writer, &oop_rec).map_err(|e| format!("write OOP: {e}"))?;

        let ip_rec = TrainingRecord {
            board: sit.board,
            pot: sit.pot as f32,
            effective_stack: sit.effective_stack as f32,
            player: 1,
            game_value: result.ip_game_value,
            oop_range: sit.ranges[0],
            ip_range: sit.ranges[1],
            cfvs: result.ip_evs,
            valid_mask,
        };
        write_record(&mut writer, &ip_rec).map_err(|e| format!("write IP: {e}"))?;
    }

    Ok(())
}

/// Convert a `[bool; N]` mask to `[u8; N]` (1 for true, 0 for false).
fn bool_mask_to_u8(mask: &[bool; NUM_COMBOS]) -> [u8; NUM_COMBOS] {
    let mut out = [0u8; NUM_COMBOS];
    for (i, &v) in mask.iter().enumerate() {
        out[i] = u8::from(v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CfvnetConfig, DatagenConfig, EvaluationConfig, GameConfig, TrainingConfig};
    use crate::datagen::storage;
    use tempfile::NamedTempFile;

    fn test_config(num_samples: u64, seed: u64) -> CfvnetConfig {
        CfvnetConfig {
            game: GameConfig {
                initial_stack: 200,
                bet_sizes: vec!["50%".into(), "a".into()],
                ..Default::default()
            },
            datagen: DatagenConfig {
                num_samples,
                solver_iterations: 100,
                target_exploitability: 0.05,
                threads: 1,
                seed,
                ..Default::default()
            },
            training: TrainingConfig::default(),
            evaluation: EvaluationConfig::default(),
        }
    }

    #[test]
    fn generate_small_batch() {
        let config = test_config(4, 42);
        let output = NamedTempFile::new().unwrap();
        let path = output.path().to_path_buf();

        generate_training_data(&config, &path).unwrap();

        let mut file = std::fs::File::open(&path).unwrap();
        let count = storage::count_records(&mut file).unwrap();

        // Some situations may have effective_stack == 0 and be skipped,
        // so we get at most 8 records (4 situations * 2 players).
        assert!(count > 0, "expected at least one record");
        assert!(
            count <= 8,
            "expected at most 8 records (4 sits * 2), got {count}"
        );
        assert_eq!(count % 2, 0, "records must come in OOP/IP pairs");

        // All records readable with valid data.
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        for i in 0..count {
            let rec = storage::read_record(&mut file).unwrap();
            assert!(rec.pot > 0.0, "record {i} has non-positive pot");
            assert!(rec.effective_stack > 0.0, "record {i} has non-positive stack");
            assert!(rec.player <= 1, "record {i} invalid player {}", rec.player);
        }
    }

    #[test]
    fn generate_is_deterministic() {
        let config = test_config(2, 99);

        let out1 = NamedTempFile::new().unwrap();
        let out2 = NamedTempFile::new().unwrap();

        generate_training_data(&config, out1.path()).unwrap();
        generate_training_data(&config, out2.path()).unwrap();

        let data1 = std::fs::read(out1.path()).unwrap();
        let data2 = std::fs::read(out2.path()).unwrap();
        assert_eq!(data1, data2, "same seed should produce identical output");
    }
}
