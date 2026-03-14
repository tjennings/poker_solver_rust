use std::io::BufWriter;
use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use range_solver::bet_size::BetSizeOptions;

use super::range_gen::NUM_COMBOS;
use super::sampler::sample_situation;
use super::solver::{solve_situation, SolveConfig, SolveResult};
use super::storage::{write_record, TrainingRecord};
use crate::config::CfvnetConfig;

/// Number of situations to generate, solve, and write per chunk.
/// Keeps peak memory bounded regardless of total `num_samples`.
const CHUNK_SIZE: u64 = 10_000;

/// Generate training data by sampling river situations, solving each with DCFR,
/// and writing two records (OOP + IP) per solved situation to `output_path`.
///
/// Processes situations in chunks of [`CHUNK_SIZE`] to bound peak memory.
/// Within each chunk, situations are generated sequentially (preserving RNG
/// determinism), solved (in parallel or sequentially), and written before the
/// next chunk begins.
///
/// Situations with `effective_stack == 0` are skipped since the solver requires
/// a positive stack. The progress bar tracks solve completions (including skips).
pub fn generate_training_data(config: &CfvnetConfig, output_path: &Path) -> Result<(), String> {
    let num_samples = config.datagen.num_samples;
    let seed = crate::config::resolve_seed(config.datagen.seed);
    let threads = config.datagen.threads;

    let bet_str = config.game.bet_sizes.join(",");
    let bet_sizes = BetSizeOptions::try_from((bet_str.as_str(), ""))
        .map_err(|e| format!("invalid bet sizes: {e}"))?;
    let solve_config = SolveConfig {
        bet_sizes,
        solver_iterations: config.datagen.solver_iterations,
        target_exploitability: config.datagen.target_exploitability,
        add_allin_threshold: config.game.add_allin_threshold,
        force_allin_threshold: config.game.force_allin_threshold,
    };

    let board_size = config.game.board_size;

    let pb = ProgressBar::new(num_samples);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} [{elapsed_precise}] ETA {eta} ({per_sec}) {msg}")
            .expect("valid progress bar template"),
    );

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

        // Generate situations sequentially for determinism.
        let situations: Vec<_> = (0..chunk_len)
            .map(|_| {
                sample_situation(&config.datagen, config.game.initial_stack, board_size, &mut rng)
            })
            .collect();

        // Solve chunk (parallel or sequential).
        let results = solve_chunk(&situations, &solve_config, &pb, pool.as_ref())?;

        // Write results immediately, then drop the chunk.
        write_chunk(&situations, results, &mut writer)?;
    }

    pb.finish_with_message("done");
    Ok(())
}

/// Solve a chunk of situations, returning results in the same order.
fn solve_chunk(
    situations: &[super::sampler::Situation],
    solve_config: &SolveConfig,
    pb: &ProgressBar,
    pool: Option<&rayon::ThreadPool>,
) -> Result<Vec<Option<SolveResult>>, String> {
    let solve_one = |sit: &super::sampler::Situation| -> Option<Result<SolveResult, String>> {
        let r = if sit.effective_stack <= 0 {
            None
        } else {
            Some(solve_situation(sit, solve_config))
        };
        pb.inc(1);
        r
    };

    let raw: Vec<Option<Result<SolveResult, String>>> = match pool {
        Some(pool) => pool.install(|| situations.par_iter().map(solve_one).collect()),
        None => situations.iter().map(solve_one).collect(),
    };

    raw.into_iter()
        .map(|r| match r {
            None => Ok(None),
            Some(Ok(res)) => Ok(Some(res)),
            Some(Err(e)) => Err(e),
        })
        .collect()
}

/// Write solved results for one chunk to the output writer.
fn write_chunk(
    situations: &[super::sampler::Situation],
    results: Vec<Option<SolveResult>>,
    writer: &mut BufWriter<std::fs::File>,
) -> Result<(), String> {
    for (sit, result) in situations.iter().zip(results) {
        let result = match result {
            Some(r) => r,
            None => continue,
        };

        let valid_mask = bool_mask_to_u8(&result.valid_mask);

        let board_vec = sit.board_cards().to_vec();

        let oop_rec = TrainingRecord {
            board: board_vec.clone(),
            pot: sit.pot as f32,
            effective_stack: sit.effective_stack as f32,
            player: 0,
            game_value: result.oop_game_value,
            oop_range: sit.ranges[0],
            ip_range: sit.ranges[1],
            cfvs: result.oop_evs,
            valid_mask,
        };
        write_record(writer, &oop_rec).map_err(|e| format!("write OOP: {e}"))?;

        let ip_rec = TrainingRecord {
            board: board_vec,
            pot: sit.pot as f32,
            effective_stack: sit.effective_stack as f32,
            player: 1,
            game_value: result.ip_game_value,
            oop_range: sit.ranges[0],
            ip_range: sit.ranges[1],
            cfvs: result.ip_evs,
            valid_mask,
        };
        write_record(writer, &ip_rec).map_err(|e| format!("write IP: {e}"))?;
    }
    Ok(())
}

/// Convert a `[bool; N]` mask to `[u8; N]` (1 for true, 0 for false).
///
/// Rust guarantees `bool` is represented as `0u8` or `1u8`, so this is a
/// direct reinterpretation of the underlying bytes.
fn bool_mask_to_u8(mask: &[bool; NUM_COMBOS]) -> [u8; NUM_COMBOS] {
    let mut out = [0u8; NUM_COMBOS];
    let src: &[u8] = bytemuck::cast_slice(mask.as_slice());
    out.copy_from_slice(src);
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
                seed: Some(seed),
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
        let count = storage::count_records(&mut file, 5).unwrap();

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
