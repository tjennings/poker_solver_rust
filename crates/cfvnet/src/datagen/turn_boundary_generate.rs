use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::config::{CfvnetConfig, DatagenConfig, TurnBoundarySamplingStratum};
use crate::datagen::manifest::{SourceMetadata, TargetSource};
use crate::datagen::sampler::{Situation, sample_situation, sample_situation_with_blueprint};
use crate::datagen::storage::TrainingRecord;
use crate::datagen::turn_boundary_dataset::TurnBoundaryDatasetWriter;
#[cfg(feature = "gpu-turn-datagen")]
use crate::datagen::turn_boundary_oracle::BoundaryNetRiverRunoutOracle;
use crate::datagen::turn_boundary_oracle::{
    ExactRiverSolverOracle, RiverRunoutOracle, TurnBoundaryInput,
    build_exact_turn_boundary_records, build_exact_turn_boundary_records_parallel,
    build_turn_boundary_record,
};

use super::domain::RangeSource;

const PARALLEL_CHUNK_SIZE: u64 = 256;

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
            generate_turn_boundary_data_with_exact_oracle(config, output_path, &oracle)?;
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
    let range_source_label = if config.datagen.blueprint_path.is_some() {
        "blueprint"
    } else {
        "rsp"
    };
    let range_source = RangeSource::from_config(&config.datagen)?;
    let sampling_policy = SamplingPolicy::new(&config.datagen)?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut validation_builder = ValidationSplitBuilder::new(
        config.training.validation_split,
        seed ^ 0x9e37_79b9_7f4a_7c15,
    );

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

    for _ in 0..config.datagen.num_samples {
        let sampled =
            sample_turn_boundary_situation(config, &range_source, &sampling_policy, &mut rng)?;
        let sit = sampled.situation;
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
        validation_builder.record(&oop, sampled.raise_depth, sampled.boundary_ordinal);
        validation_builder.record(&ip, sampled.raise_depth, sampled.boundary_ordinal);
        writer.write_with_coverage(
            &[oop, ip],
            range_source_label,
            sampled.raise_depth,
            sampled.boundary_ordinal,
        )?;
        pb.inc(1);
        pb.set_message(format!("written:{}", writer.count()));
    }

    pb.finish_with_message("done");
    let manifest = writer.finish().map_err(|e| e.to_string())?;
    if let Some(split) = validation_builder.finish() {
        let split_path = output_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .join("validation_split.yaml");
        split.write_yaml(&split_path)?;
    }
    eprintln!(
        "Wrote {} turn-boundary records to {} with manifest.yaml",
        manifest.coverage.total_records,
        output_path.display()
    );
    Ok(manifest.coverage.total_records)
}

fn generate_turn_boundary_data_with_exact_oracle(
    config: &CfvnetConfig,
    output_path: &Path,
    oracle: &ExactRiverSolverOracle,
) -> Result<u64, String> {
    let seed = crate::config::resolve_seed(config.datagen.seed);
    let range_source_label = if config.datagen.blueprint_path.is_some() {
        "blueprint"
    } else {
        "rsp"
    };
    let range_source = RangeSource::from_config(&config.datagen)?;
    let sampling_policy = SamplingPolicy::new(&config.datagen)?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut validation_builder = ValidationSplitBuilder::new(
        config.training.validation_split,
        seed ^ 0x9e37_79b9_7f4a_7c15,
    );

    let mut writer = TurnBoundaryDatasetWriter::create(
        output_path,
        config.datagen.per_file,
        TargetSource::ExactRiver,
        SourceMetadata::default(),
    )?;
    let pool = if config.datagen.threads > 1 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(config.datagen.threads)
                .build()
                .map_err(|e| format!("thread pool: {e}"))?,
        )
    } else {
        None
    };

    let pb = ProgressBar::new(config.datagen.num_samples);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} [{elapsed_precise}] ETA {eta} ({per_sec}) {msg}")
            .expect("valid template"),
    );

    for _ in 0..config.datagen.num_samples {
        let sampled =
            sample_turn_boundary_situation(config, &range_source, &sampling_policy, &mut rng)?;
        let built = build_exact_turn_boundary_records_for_sample(&sampled, oracle, pool.as_ref())?;
        validation_builder.record(&built.records[0], built.raise_depth, built.boundary_ordinal);
        validation_builder.record(&built.records[1], built.raise_depth, built.boundary_ordinal);
        writer.write_with_coverage(
            &built.records,
            range_source_label,
            built.raise_depth,
            built.boundary_ordinal,
        )?;
        pb.inc(1);
        pb.set_message(format!("written:{}", writer.count()));
    }

    pb.finish_with_message("done");
    let manifest = writer.finish().map_err(|e| e.to_string())?;
    if let Some(split) = validation_builder.finish() {
        let split_path = output_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .join("validation_split.yaml");
        split.write_yaml(&split_path)?;
    }
    eprintln!(
        "Wrote {} turn-boundary records to {} with manifest.yaml",
        manifest.coverage.total_records,
        output_path.display()
    );
    Ok(manifest.coverage.total_records)
}

pub fn generate_turn_boundary_data_with_oracle_parallel<O: RiverRunoutOracle + Sync>(
    config: &CfvnetConfig,
    output_path: &Path,
    target_source: TargetSource,
    source: SourceMetadata,
    oracle: &O,
) -> Result<u64, String> {
    let seed = crate::config::resolve_seed(config.datagen.seed);
    let range_source_label = if config.datagen.blueprint_path.is_some() {
        "blueprint"
    } else {
        "rsp"
    };
    let range_source = RangeSource::from_config(&config.datagen)?;
    let sampling_policy = SamplingPolicy::new(&config.datagen)?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut validation_builder = ValidationSplitBuilder::new(
        config.training.validation_split,
        seed ^ 0x9e37_79b9_7f4a_7c15,
    );

    let mut writer = TurnBoundaryDatasetWriter::create(
        output_path,
        config.datagen.per_file,
        target_source,
        source,
    )?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.datagen.threads.max(1))
        .build()
        .map_err(|e| format!("thread pool: {e}"))?;

    let pb = ProgressBar::new(config.datagen.num_samples);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len} [{elapsed_precise}] ETA {eta} ({per_sec}) {msg}")
            .expect("valid template"),
    );

    let mut remaining = config.datagen.num_samples;
    while remaining > 0 {
        let chunk_len = remaining.min(PARALLEL_CHUNK_SIZE);
        remaining -= chunk_len;

        let sampled: Vec<_> = (0..chunk_len)
            .map(|_| {
                sample_turn_boundary_situation(config, &range_source, &sampling_policy, &mut rng)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let built = pool.install(|| {
            sampled
                .par_iter()
                .map(|sampled| {
                    range_solver::set_force_sequential(true);
                    build_turn_boundary_records_for_sample(sampled, oracle)
                })
                .collect::<Vec<_>>()
        });

        for built in built {
            let built = built?;
            validation_builder.record(&built.records[0], built.raise_depth, built.boundary_ordinal);
            validation_builder.record(&built.records[1], built.raise_depth, built.boundary_ordinal);
            writer.write_with_coverage(
                &built.records,
                range_source_label,
                built.raise_depth,
                built.boundary_ordinal,
            )?;
            pb.inc(1);
            pb.set_message(format!("written:{}", writer.count()));
        }
    }

    pb.finish_with_message("done");
    let manifest = writer.finish().map_err(|e| e.to_string())?;
    if let Some(split) = validation_builder.finish() {
        let split_path = output_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .join("validation_split.yaml");
        split.write_yaml(&split_path)?;
    }
    eprintln!(
        "Wrote {} turn-boundary records to {} with manifest.yaml",
        manifest.coverage.total_records,
        output_path.display()
    );
    Ok(manifest.coverage.total_records)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ValidationSplitManifest {
    schema_version: u32,
    seed: u64,
    total_records: u64,
    train_records: u64,
    validation_records: u64,
    validation_fraction: f64,
    strata: BTreeMap<String, ValidationSplitStratum>,
    validation_indices: Vec<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct ValidationSplitStratum {
    total_records: u64,
    validation_records: u64,
}

impl ValidationSplitManifest {
    fn write_yaml(&self, path: &Path) -> Result<(), String> {
        let file = std::fs::File::create(path)
            .map_err(|e| format!("create validation split {}: {e}", path.display()))?;
        serde_yaml::to_writer(file, self)
            .map_err(|e| format!("write validation split {}: {e}", path.display()))
    }
}

struct ValidationSplitBuilder {
    fraction: f64,
    seed: u64,
    next_index: u64,
    by_stratum: BTreeMap<String, Vec<u64>>,
}

impl ValidationSplitBuilder {
    fn new(fraction: f64, seed: u64) -> Self {
        Self {
            fraction,
            seed,
            next_index: 0,
            by_stratum: BTreeMap::new(),
        }
    }

    fn record(&mut self, rec: &TrainingRecord, raise_depth: &str, boundary_ordinal: &str) {
        if self.fraction <= 0.0 {
            self.next_index += 1;
            return;
        }
        let key = validation_stratum_key(rec, raise_depth, boundary_ordinal);
        self.by_stratum
            .entry(key)
            .or_default()
            .push(self.next_index);
        self.next_index += 1;
    }

    fn finish(self) -> Option<ValidationSplitManifest> {
        if self.fraction <= 0.0 || self.next_index == 0 {
            return None;
        }

        let mut selected = BTreeSet::new();
        let mut strata = BTreeMap::new();
        for (key, mut indices) in self.by_stratum {
            deterministic_shuffle(&mut indices, self.seed ^ stable_hash(&key));
            let target = ((indices.len() as f64) * self.fraction).round() as usize;
            let validation_count = target.min(indices.len());
            for idx in indices.iter().take(validation_count) {
                selected.insert(*idx);
            }
            strata.insert(
                key,
                ValidationSplitStratum {
                    total_records: indices.len() as u64,
                    validation_records: validation_count as u64,
                },
            );
        }

        if selected.is_empty() {
            return None;
        }

        let validation_indices: Vec<u64> = selected.into_iter().collect();
        let validation_records = validation_indices.len() as u64;
        Some(ValidationSplitManifest {
            schema_version: 1,
            seed: self.seed,
            total_records: self.next_index,
            train_records: self.next_index - validation_records,
            validation_records,
            validation_fraction: self.fraction,
            strata,
            validation_indices,
        })
    }
}

fn validation_stratum_key(
    rec: &TrainingRecord,
    raise_depth: &str,
    boundary_ordinal: &str,
) -> String {
    format!(
        "raise={raise_depth}|boundary={boundary_ordinal}|{}|{}|{}|board={}",
        validation_pot_bucket(rec.pot),
        validation_stack_bucket(rec.effective_stack),
        validation_spr_bucket(rec.effective_stack, rec.pot),
        validation_board_texture(&rec.board),
    )
}

fn validation_pot_bucket(pot: f32) -> &'static str {
    match pot {
        p if p < 10.0 => "pot_lt_10",
        p if p < 25.0 => "pot_10_25",
        p if p < 50.0 => "pot_25_50",
        p if p < 100.0 => "pot_50_100",
        p if p < 200.0 => "pot_100_200",
        _ => "pot_200_plus",
    }
}

fn validation_stack_bucket(stack: f32) -> &'static str {
    match stack {
        s if s <= 0.0 => "stack_0",
        s if s < 25.0 => "stack_1_25",
        s if s < 50.0 => "stack_25_50",
        s if s < 100.0 => "stack_50_100",
        s if s < 200.0 => "stack_100_200",
        _ => "stack_200_plus",
    }
}

fn validation_spr_bucket(stack: f32, pot: f32) -> &'static str {
    let spr = if pot > 0.0 {
        stack / pot
    } else {
        f32::INFINITY
    };
    match spr {
        s if s < 0.5 => "spr_lt_0_5",
        s if s < 1.5 => "spr_0_5_1_5",
        s if s < 4.0 => "spr_1_5_4",
        s if s < 8.0 => "spr_4_8",
        s if s < 20.0 => "spr_8_20",
        _ => "spr_20_plus",
    }
}

fn validation_board_texture(board: &[u8]) -> String {
    let mut ranks = [false; 13];
    let mut suit_counts = [0_u8; 4];
    for &card in board {
        ranks[(card / 4) as usize] = true;
        suit_counts[(card % 4) as usize] += 1;
    }
    let suit = match suit_counts.iter().copied().max().unwrap_or(0) {
        4 => "monotone",
        3 => "three_flush",
        2 => "two_tone",
        _ => "rainbow",
    };
    let connected = if ranks.windows(4).any(|window| window.iter().all(|&v| v)) {
        "connected"
    } else {
        "disconnected"
    };
    format!("{suit}_{connected}")
}

fn deterministic_shuffle(values: &mut [u64], seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for i in (1..values.len()).rev() {
        let j = rng.gen_range(0..=i);
        values.swap(i, j);
    }
}

fn stable_hash(value: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in value.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

struct SampledSituation<'a> {
    situation: Situation,
    raise_depth: &'a str,
    boundary_ordinal: &'a str,
}

struct BuiltTurnBoundaryRecords<'a> {
    records: [TrainingRecord; 2],
    raise_depth: &'a str,
    boundary_ordinal: &'a str,
}

fn build_turn_boundary_records_for_sample<'a, O: RiverRunoutOracle>(
    sampled: &SampledSituation<'a>,
    oracle: &O,
) -> Result<BuiltTurnBoundaryRecords<'a>, String> {
    let sit = &sampled.situation;
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
    Ok(BuiltTurnBoundaryRecords {
        records: [oop, ip],
        raise_depth: sampled.raise_depth,
        boundary_ordinal: sampled.boundary_ordinal,
    })
}

fn build_exact_turn_boundary_records_for_sample<'a>(
    sampled: &SampledSituation<'a>,
    oracle: &ExactRiverSolverOracle,
    pool: Option<&rayon::ThreadPool>,
) -> Result<BuiltTurnBoundaryRecords<'a>, String> {
    let sit = &sampled.situation;
    let input = TurnBoundaryInput {
        board: [sit.board[0], sit.board[1], sit.board[2], sit.board[3]],
        pot: sit.pot as f32,
        effective_stack: sit.effective_stack as f32,
        player: 0,
        oop_range: sit.ranges[0],
        ip_range: sit.ranges[1],
    };
    let records = match pool {
        Some(pool) => build_exact_turn_boundary_records_parallel(&input, oracle, pool)?,
        None => build_exact_turn_boundary_records(&input, oracle)?,
    };
    Ok(BuiltTurnBoundaryRecords {
        records,
        raise_depth: sampled.raise_depth,
        boundary_ordinal: sampled.boundary_ordinal,
    })
}

struct SamplingPolicy<'a> {
    strata: &'a [TurnBoundarySamplingStratum],
    total_weight: f64,
}

impl<'a> SamplingPolicy<'a> {
    fn new(config: &'a DatagenConfig) -> Result<Self, String> {
        let total_weight = config
            .turn_boundary_sampling
            .strata
            .iter()
            .map(|stratum| stratum.weight.max(0.0))
            .sum();
        if !config.turn_boundary_sampling.strata.is_empty() && total_weight <= 0.0 {
            return Err("turn_boundary_sampling strata must have positive total weight".into());
        }
        Ok(Self {
            strata: &config.turn_boundary_sampling.strata,
            total_weight,
        })
    }

    fn choose<R: Rng>(&self, rng: &mut R) -> Option<&'a TurnBoundarySamplingStratum> {
        if self.strata.is_empty() {
            return None;
        }
        let mut remaining = rng.gen_range(0.0..self.total_weight);
        for stratum in self.strata {
            let weight = stratum.weight.max(0.0);
            if remaining < weight {
                return Some(stratum);
            }
            remaining -= weight;
        }
        self.strata.last()
    }
}

fn sample_turn_boundary_situation<'a, R: Rng>(
    config: &CfvnetConfig,
    range_source: &RangeSource,
    sampling_policy: &'a SamplingPolicy<'a>,
    rng: &mut R,
) -> Result<SampledSituation<'a>, String> {
    let stratum = sampling_policy.choose(rng);
    let mut sample_config = config.datagen.clone();
    if let Some(stratum) = stratum {
        if let Some(pot_intervals) = &stratum.pot_intervals {
            sample_config.pot_intervals = pot_intervals.clone();
        }
        if let Some(spr_intervals) = &stratum.spr_intervals {
            sample_config.spr_intervals = Some(spr_intervals.clone());
        }
    }

    let situation = match range_source {
        RangeSource::Rsp => sample_situation(&sample_config, config.game.initial_stack, 4, rng),
        RangeSource::Blueprint(precomputed) => sample_situation_with_blueprint(
            &sample_config,
            config.game.initial_stack,
            4,
            precomputed,
            rng,
        ),
    };
    let raise_depth = stratum
        .and_then(|stratum| stratum.raise_depth.as_deref())
        .unwrap_or("sampled_turn_state");
    let boundary_ordinal = stratum
        .and_then(|stratum| stratum.boundary_ordinal.as_deref())
        .or_else(|| stratum.map(|stratum| stratum.name.as_str()))
        .unwrap_or("turn_entry");

    Ok(SampledSituation {
        situation,
        raise_depth,
        boundary_ordinal,
    })
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
        TurnBoundarySamplingConfig, TurnBoundarySamplingStratum,
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
        assert_eq!(
            manifest.coverage.by_target_source.get("exact_river"),
            Some(&6)
        );
        assert_eq!(manifest.coverage.by_range_source.get("rsp"), Some(&6));
        assert_eq!(
            manifest.coverage.by_raise_depth.get("sampled_turn_state"),
            Some(&6)
        );
        assert_eq!(
            manifest.coverage.by_boundary_ordinal.get("turn_entry"),
            Some(&6)
        );
        assert!(!manifest.coverage.by_spr_bucket.is_empty());
        assert!(!manifest.coverage.by_board_texture.is_empty());
    }

    #[test]
    fn parallel_writer_preserves_counts_and_manifest() {
        let dir = TempDir::new().unwrap();
        let output = dir.path().join("turn_boundary.bin");
        let mut config = test_config(5, Some(4));
        config.datagen.threads = 2;

        let written = generate_turn_boundary_data_with_oracle_parallel(
            &config,
            &output,
            TargetSource::ExactRiver,
            SourceMetadata::default(),
            &ConstantOracle(0.25),
        )
        .unwrap();

        assert_eq!(written, 10);
        let manifest = DatasetManifest::read_yaml(dir.path().join("manifest.yaml")).unwrap();
        manifest.validate_turn_boundary().unwrap();
        assert_eq!(manifest.coverage.total_records, 10);
        assert_eq!(
            manifest.coverage.by_target_source.get("exact_river"),
            Some(&10)
        );
        assert_eq!(
            manifest.coverage.by_raise_depth.get("sampled_turn_state"),
            Some(&10)
        );
    }

    #[test]
    fn weighted_sampling_strata_override_pot_spr_and_coverage_labels() {
        let dir = TempDir::new().unwrap();
        let output = dir.path().join("turn_boundary.bin");
        let mut config = test_config(2, None);
        config.datagen.turn_boundary_sampling = TurnBoundarySamplingConfig {
            strata: vec![TurnBoundarySamplingStratum {
                name: "tiny_pot_high_spr".to_string(),
                weight: 1.0,
                pot_intervals: Some(vec![[4, 5]]),
                spr_intervals: Some(vec![[8.0, 20.0]]),
                raise_depth: Some("4bet_plus".to_string()),
                boundary_ordinal: None,
            }],
        };

        let written = generate_turn_boundary_data_with_oracle(
            &config,
            &output,
            TargetSource::ExactRiver,
            SourceMetadata::default(),
            &ConstantOracle(0.25),
        )
        .unwrap();

        assert_eq!(written, 4);
        let manifest = DatasetManifest::read_yaml(dir.path().join("manifest.yaml")).unwrap();
        assert_eq!(manifest.coverage.by_pot_bucket.get("pot_lt_10"), Some(&4));
        assert_eq!(manifest.coverage.by_spr_bucket.get("spr_8_20"), Some(&4));
        assert_eq!(manifest.coverage.by_raise_depth.get("4bet_plus"), Some(&4));
        assert_eq!(
            manifest
                .coverage
                .by_boundary_ordinal
                .get("tiny_pot_high_spr"),
            Some(&4)
        );
    }

    #[test]
    fn rejects_unknown_target_source() {
        let err = parse_target_source("banana").unwrap_err();
        assert!(err.contains("unknown turn_boundary_target_source"));
    }

    #[test]
    fn rejects_sampling_policy_with_no_positive_weight() {
        let mut config = test_config(1, None);
        config.datagen.turn_boundary_sampling = TurnBoundarySamplingConfig {
            strata: vec![TurnBoundarySamplingStratum {
                name: "disabled".to_string(),
                weight: 0.0,
                pot_intervals: None,
                spr_intervals: None,
                raise_depth: None,
                boundary_ordinal: None,
            }],
        };

        let err = generate_turn_boundary_data_with_oracle(
            &config,
            Path::new("/tmp/unused.bin"),
            TargetSource::ExactRiver,
            SourceMetadata::default(),
            &ConstantOracle(0.25),
        )
        .unwrap_err();

        assert!(err.contains("positive total weight"));
    }

    #[test]
    fn writes_frozen_validation_split_when_configured() {
        let dir = TempDir::new().unwrap();
        let output = dir.path().join("turn_boundary.bin");
        let mut config = test_config(10, None);
        config.training.validation_split = 0.2;

        generate_turn_boundary_data_with_oracle(
            &config,
            &output,
            TargetSource::ExactRiver,
            SourceMetadata::default(),
            &ConstantOracle(0.25),
        )
        .unwrap();

        let split_path = dir.path().join("validation_split.yaml");
        assert!(split_path.exists());
        let split: ValidationSplitManifest =
            serde_yaml::from_reader(std::fs::File::open(split_path).unwrap()).unwrap();
        assert_eq!(split.schema_version, 1);
        assert_eq!(split.total_records, 20);
        assert_eq!(split.validation_records, 4);
        assert_eq!(split.train_records, 16);
        assert_eq!(split.validation_indices.len(), 4);
        assert!(split.validation_indices.windows(2).all(|w| w[0] < w[1]));
        assert!(!split.strata.is_empty());
    }

    #[test]
    fn omits_validation_split_when_disabled() {
        let dir = TempDir::new().unwrap();
        let output = dir.path().join("turn_boundary.bin");
        let mut config = test_config(3, None);
        config.training.validation_split = 0.0;

        generate_turn_boundary_data_with_oracle(
            &config,
            &output,
            TargetSource::ExactRiver,
            SourceMetadata::default(),
            &ConstantOracle(0.25),
        )
        .unwrap();

        assert!(!dir.path().join("validation_split.yaml").exists());
    }
}
