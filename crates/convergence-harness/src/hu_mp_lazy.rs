//! Evidence-producing HU blueprint_v2 vs MP lazy-sparse 2-player harness.
//!
//! This module is intentionally a GO/NO-GO gate, not a proof that heads-up
//! training can be retired. The first reusable slice verifies structural
//! preconditions, compares normalized root action schemas, and gates behavior on
//! complete canonical preflop root combo-weighted action-frequency parity. Local
//! row strategy L1 and average-strategy accounting remain diagnostics when the
//! action-frequency gate passes; GO does not prove per-hand, post-root, EV, or
//! exploitability parity.

use std::any::Any;
use std::collections::HashSet;
use std::error::Error;
use std::panic::{catch_unwind, set_hook, take_hook, AssertUnwindSafe};
use std::path::Path;
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use poker_solver_core::blueprint_mp::config::{
    BlueprintMpConfig, ForcedBet, ForcedBetKind, MpActionAbstractionConfig,
    MpChanceContinuationMode, MpClusteringConfig, MpGameConfig, MpNegativeActionPurgeMode,
    MpSnapshotConfig, MpStreetCluster, MpStreetSizes, MpTrainingBackend, MpTrainingConfig,
};
use poker_solver_core::blueprint_mp::game_tree::TreeAction as MpTreeAction;
use poker_solver_core::blueprint_mp::lazy_mccfr::LazyResolvedSpot;
use poker_solver_core::blueprint_mp::sparse_storage::MpInfosetKey;
use poker_solver_core::blueprint_mp::trainer::{run_lazy_training, setup_lazy_training};
use poker_solver_core::blueprint_v2::config::{
    ActionAbstractionConfig, BaselineValidationTrainingConfig, BlueprintV2Config,
    ClusteringAlgorithm, ClusteringConfig, GameConfig, SnapshotConfig, SnapshotFormat,
    StreetClusterConfig, TrainingConfig,
};
use poker_solver_core::blueprint_v2::game_tree::{
    GameNode as HuGameNode, TreeAction as HuTreeAction,
};
use poker_solver_core::blueprint_v2::trainer::BlueprintTrainer;
use poker_solver_core::hands::CanonicalHand;
use poker_solver_core::training_runtime::SnapshotFormat as RuntimeSnapshotFormat;
use serde::{Deserialize, Serialize};

const CHIP_EPSILON: f64 = 0.01;
const PROBABILITY_SUM_EPSILON: f64 = 1.0e-6;
const UNMATCHED_ACTION_MASS_EPSILON: f64 = 1.0e-9;

/// Final gate result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING-KEBAB-CASE")]
pub enum HuMpLazyVerdict {
    Go,
    NoGo,
}

/// Tiny in-memory 2-player fixture and report thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuMpLazyHarnessConfig {
    /// Smoke-training meta-iterations. Keep small; this harness is diagnostic.
    pub iterations: u64,
    /// Maximum root-row L1 distance tolerated before adding a NO-GO reason.
    pub max_l1_tolerance: f64,
    /// Mean root-row L1 distance tolerated before adding a NO-GO reason.
    pub mean_l1_tolerance: f64,
    /// Maximum combo-weighted root action-frequency delta tolerated. The
    /// production gate should use <= 0.01; this tiny smoke default stays looser
    /// because one-iteration sparse coverage is diagnostic, not calibrated.
    pub max_action_frequency_delta_tolerance: f64,
    /// Total combo-weighted root action-frequency L1 tolerated. Production
    /// should be closer to 0.02; this default avoids making tiny smoke fragile.
    pub total_action_frequency_l1_tolerance: f64,
    /// Number of largest bucket/action frequency deltas retained in artifacts.
    pub worst_action_frequency_spots: usize,
    /// Effective stack in chips. The default keeps the fixture tiny while
    /// leaving room for a real sized preflop open below the stack.
    pub stack_chips: f64,
    pub small_blind_chips: f64,
    pub big_blind_chips: f64,
    pub rake_rate: f64,
    pub rake_cap_chips: f64,
    /// Normalized preflop open-to amounts in chips for the shared abstraction.
    pub preflop_open_to_chips: Vec<f64>,
    /// Postflop lead/raise pot fractions.
    pub postflop_bet_fractions: Vec<f64>,
    pub preflop_buckets: u16,
    pub flop_buckets: u16,
    pub turn_buckets: u16,
    pub river_buckets: u16,
    pub allow_preflop_limp: bool,
    /// Seed used by HU clustering/training RNG. MP lazy has its own trajectory;
    /// equality of trajectories is explicitly not a validity criterion.
    pub output_seed: u64,
}

impl Default for HuMpLazyHarnessConfig {
    fn default() -> Self {
        Self {
            iterations: 2,
            max_l1_tolerance: 0.05,
            mean_l1_tolerance: 0.01,
            max_action_frequency_delta_tolerance: 0.05,
            total_action_frequency_l1_tolerance: 0.10,
            worst_action_frequency_spots: 20,
            stack_chips: 10.0,
            small_blind_chips: 1.0,
            big_blind_chips: 2.0,
            rake_rate: 0.0,
            rake_cap_chips: 0.0,
            preflop_open_to_chips: vec![6.0],
            postflop_bet_fractions: vec![1.0],
            preflop_buckets: 169,
            flop_buckets: 8,
            turn_buckets: 8,
            river_buckets: 8,
            allow_preflop_limp: true,
            output_seed: 42,
        }
    }
}

/// One structural precondition and its result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuMpLazyStructuralCheck {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

/// Normalized public action descriptor used for HU/MP schema comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormalizedActionDescriptor {
    pub kind: String,
    pub amount_chips: Option<f64>,
}

/// Schema mismatch suitable for JSON and CSV output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuMpLazySchemaMismatch {
    pub path: String,
    pub action_index: usize,
    pub hu_action: String,
    pub mp_action: String,
    pub reason: String,
}

/// Per-row strategy distance suitable for JSON and CSV output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuMpLazyStrategyDistance {
    pub path: String,
    pub bucket: u16,
    pub hu_strategy: String,
    pub mp_strategy: String,
    pub l1_distance: f64,
    pub max_abs_distance: f64,
    pub mp_row_visited: bool,
}

/// Sparse/dense row coverage observed by the smoke comparison.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HuMpLazyRowCoverage {
    pub compared_rows: usize,
    pub hu_dense_rows: usize,
    pub mp_sparse_rows_total: usize,
    pub mp_root_rows_visited: usize,
    pub missing_mp_root_rows: usize,
}

/// Shared distribution metadata for counterfactual action-frequency aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuMpLazyActionFrequencyDistribution {
    pub name: String,
    pub bucket_count: usize,
    pub total_weight: usize,
    pub normalized_weight_total: f64,
}

/// Coverage diagnostics for the action-frequency evaluator.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HuMpLazyActionFrequencyCoverage {
    pub matched_rows: usize,
    pub missing_mp_rows: usize,
    pub skipped_rows: usize,
    pub invalid_rows: usize,
    pub zero_sum_rows: usize,
    pub off_normalized_rows: usize,
    pub uniform_fallback_rows: usize,
    pub included_normalized_weight_total: f64,
    pub skipped_normalized_weight_total: f64,
    pub hu_unmatched_rows: usize,
    pub mp_unmatched_rows: usize,
    pub hu_unmatched_probability_mass: f64,
    pub mp_unmatched_probability_mass: f64,
    pub hu_unmatched_weighted_mass: f64,
    pub mp_unmatched_weighted_mass: f64,
}

/// Combo-weighted aggregate action-frequency comparison at root.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuMpLazyActionFrequencyAggregateRow {
    pub path: String,
    pub action_index: usize,
    pub action_label: String,
    pub action_kind: String,
    pub amount_chips: Option<f64>,
    pub hu_frequency: f64,
    pub mp_frequency: f64,
    pub delta: f64,
    pub abs_delta: f64,
}

/// Worst bucket/action local frequency differences for attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuMpLazyActionFrequencySpot {
    pub path: String,
    pub bucket: u16,
    pub hand_label: String,
    pub combo_count: usize,
    pub normalized_weight: f64,
    pub action_index: usize,
    pub action_label: String,
    pub hu_probability: f64,
    pub mp_probability: f64,
    pub abs_delta: f64,
    pub row_l1: f64,
    pub weighted_l1_contribution: f64,
    pub mp_row_visited: bool,
}

/// Aggregate action-frequency totals and threshold result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuMpLazyActionFrequencyTotals {
    pub total_l1_delta: f64,
    pub max_abs_delta: f64,
    pub weighted_row_l1: f64,
    pub max_row_l1: f64,
    pub passed_thresholds: bool,
}

/// Root-only counterfactual action-frequency parity report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuMpLazyActionFrequencyReport {
    pub distribution: HuMpLazyActionFrequencyDistribution,
    pub coverage: HuMpLazyActionFrequencyCoverage,
    pub aggregate_actions: Vec<HuMpLazyActionFrequencyAggregateRow>,
    pub worst_spots: Vec<HuMpLazyActionFrequencySpot>,
    pub totals: HuMpLazyActionFrequencyTotals,
}

/// Runtime and coarse memory/storage telemetry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HuMpLazyRuntimeStats {
    pub hu_runtime_ms: u64,
    pub mp_lazy_runtime_ms: u64,
    pub total_runtime_ms: u64,
    pub hu_iterations_completed: u64,
    pub mp_meta_iterations_completed: u64,
    pub hu_realized_rows: usize,
    pub hu_realized_slots: usize,
    pub hu_resident_bytes: usize,
    pub mp_sparse_entries: usize,
    pub mp_sparse_slots: usize,
    pub mp_sparse_approx_bytes: usize,
}

/// Complete reusable report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuMpLazyReport {
    pub verdict: HuMpLazyVerdict,
    pub reasons: Vec<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
    pub structural_checks: Vec<HuMpLazyStructuralCheck>,
    pub root_action_schema_mismatches: Vec<HuMpLazySchemaMismatch>,
    pub row_coverage: HuMpLazyRowCoverage,
    pub strategy_distances: Vec<HuMpLazyStrategyDistance>,
    pub max_l1_distance: Option<f64>,
    pub mean_l1_distance: Option<f64>,
    pub max_abs_distance: Option<f64>,
    pub action_frequency: Option<HuMpLazyActionFrequencyReport>,
    pub runtime_stats: HuMpLazyRuntimeStats,
    pub average_strategy_accounting_reconciled: bool,
}

impl HuMpLazyReport {
    /// Save report artifacts:
    /// - `summary.json`
    /// - `report.txt`
    /// - `root_action_schema_mismatches.csv`
    /// - `strategy_distance.csv`
    /// - `root_action_frequencies.csv`
    /// - `root_action_frequency_spots.csv`
    pub fn save(&self, dir: &Path) -> Result<(), Box<dyn Error>> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(
            dir.join("summary.json"),
            serde_json::to_string_pretty(self)?,
        )?;
        std::fs::write(dir.join("report.txt"), self.human_summary())?;

        let mut schema_writer =
            csv::Writer::from_path(dir.join("root_action_schema_mismatches.csv"))?;
        for mismatch in &self.root_action_schema_mismatches {
            schema_writer.serialize(mismatch)?;
        }
        schema_writer.flush()?;

        let mut distance_writer = csv::Writer::from_path(dir.join("strategy_distance.csv"))?;
        for distance in &self.strategy_distances {
            distance_writer.serialize(distance)?;
        }
        distance_writer.flush()?;

        let mut action_frequency_writer =
            csv::Writer::from_path(dir.join("root_action_frequencies.csv"))?;
        let mut action_frequency_spots_writer =
            csv::Writer::from_path(dir.join("root_action_frequency_spots.csv"))?;
        if let Some(action_frequency) = &self.action_frequency {
            for row in &action_frequency.aggregate_actions {
                action_frequency_writer.serialize(row)?;
            }
            for spot in &action_frequency.worst_spots {
                action_frequency_spots_writer.serialize(spot)?;
            }
        }
        action_frequency_writer.flush()?;
        action_frequency_spots_writer.flush()?;

        Ok(())
    }

    /// Human-readable summary for quick inspection.
    #[must_use]
    pub fn human_summary(&self) -> String {
        let mut out = String::new();
        out.push_str("=== HU vs MP Lazy 2p Equivalence Harness ===\n\n");
        out.push_str(&format!("Verdict: {:?}\n", self.verdict));
        out.push_str(&format!(
            "Average-strategy accounting reconciled: {}\n\n",
            self.average_strategy_accounting_reconciled
        ));
        out.push_str(
            "GO criterion: complete canonical preflop root combo-weighted action frequencies matched within tolerance; this is not per-hand, post-root, EV, or exploitability parity.\n\n",
        );

        out.push_str("--- Reasons (hard NO-GO causes) ---\n");
        if self.reasons.is_empty() {
            out.push_str("none\n");
        } else {
            for reason in &self.reasons {
                out.push_str(&format!("- {reason}\n"));
            }
        }

        out.push_str("\n--- Warnings ---\n");
        if self.warnings.is_empty() {
            out.push_str("none\n");
        } else {
            for warning in &self.warnings {
                out.push_str(&format!("- {warning}\n"));
            }
        }

        out.push_str("\n--- Structural Checks ---\n");
        for check in &self.structural_checks {
            let status = if check.passed { "PASS" } else { "FAIL" };
            out.push_str(&format!("[{status}] {}: {}\n", check.name, check.detail));
        }

        out.push_str("\n--- Root Action Schema ---\n");
        out.push_str(&format!(
            "root mismatches: {}\n",
            self.root_action_schema_mismatches.len()
        ));

        out.push_str("\n--- Strategy Distance ---\n");
        out.push_str(&format!(
            "compared rows: {}\nmean L1: {}\nmax L1: {}\nmax abs: {}\n",
            self.row_coverage.compared_rows,
            format_optional_f64(self.mean_l1_distance),
            format_optional_f64(self.max_l1_distance),
            format_optional_f64(self.max_abs_distance)
        ));

        out.push_str("\n--- Root Action Frequencies ---\n");
        if let Some(action_frequency) = &self.action_frequency {
            out.push_str(&format!(
                "distribution: {} buckets={}, total_weight={}, normalized_weight_total={:.6}\n",
                action_frequency.distribution.name,
                action_frequency.distribution.bucket_count,
                action_frequency.distribution.total_weight,
                action_frequency.distribution.normalized_weight_total
            ));
            out.push_str(&format!(
                "coverage: matched={}, missing_mp={}, skipped={}, invalid={}, zero_sum={}, off_normalized={}, uniform_fallback={}\n",
                action_frequency.coverage.matched_rows,
                action_frequency.coverage.missing_mp_rows,
                action_frequency.coverage.skipped_rows,
                action_frequency.coverage.invalid_rows,
                action_frequency.coverage.zero_sum_rows,
                action_frequency.coverage.off_normalized_rows,
                action_frequency.coverage.uniform_fallback_rows
            ));
            out.push_str(&format!(
                "weight coverage: input={:.6}, included={:.6}, skipped={:.6}\n",
                action_frequency.distribution.normalized_weight_total,
                action_frequency.coverage.included_normalized_weight_total,
                action_frequency.coverage.skipped_normalized_weight_total
            ));
            out.push_str(&format!(
                "unmatched action mass: HU rows={}, raw={:.6}, weighted={:.6}; MP rows={}, raw={:.6}, weighted={:.6}\n",
                action_frequency.coverage.hu_unmatched_rows,
                action_frequency.coverage.hu_unmatched_probability_mass,
                action_frequency.coverage.hu_unmatched_weighted_mass,
                action_frequency.coverage.mp_unmatched_rows,
                action_frequency.coverage.mp_unmatched_probability_mass,
                action_frequency.coverage.mp_unmatched_weighted_mass
            ));
            out.push_str(&format!(
                "total L1: {:.6}, max abs: {:.6}, weighted row L1: {:.6}, max row L1: {:.6}, thresholds passed: {}\n",
                action_frequency.totals.total_l1_delta,
                action_frequency.totals.max_abs_delta,
                action_frequency.totals.weighted_row_l1,
                action_frequency.totals.max_row_l1,
                action_frequency.totals.passed_thresholds
            ));
            for row in &action_frequency.aggregate_actions {
                out.push_str(&format!(
                    "{}: HU {}, MP {}, delta {}\n",
                    row.action_label,
                    format_percent(row.hu_frequency),
                    format_percent(row.mp_frequency),
                    format_signed_percent(row.delta)
                ));
            }
        } else {
            out.push_str("not produced\n");
        }

        out.push_str("\n--- Runtime / Storage ---\n");
        out.push_str(&format!(
            "HU: {} ms, {} iterations, {} rows, {} bytes\n",
            self.runtime_stats.hu_runtime_ms,
            self.runtime_stats.hu_iterations_completed,
            self.runtime_stats.hu_realized_rows,
            self.runtime_stats.hu_resident_bytes
        ));
        out.push_str(&format!(
            "MP lazy: {} ms, {} meta-iterations, {} sparse rows, {} bytes\n",
            self.runtime_stats.mp_lazy_runtime_ms,
            self.runtime_stats.mp_meta_iterations_completed,
            self.runtime_stats.mp_sparse_entries,
            self.runtime_stats.mp_sparse_approx_bytes
        ));

        out
    }
}

/// Build the matching tiny HU and MP lazy configs in memory.
#[must_use]
pub fn build_equivalent_configs(
    harness: &HuMpLazyHarnessConfig,
    output_dir: &Path,
) -> (BlueprintV2Config, BlueprintMpConfig) {
    (
        build_hu_config(harness, &output_dir.join("hu")),
        build_mp_lazy_config(harness, &output_dir.join("mp_lazy")),
    )
}

/// Produce a structural/schema report without running training.
#[must_use]
pub fn preflight_report_for_configs(
    harness: &HuMpLazyHarnessConfig,
    hu: &BlueprintV2Config,
    mp: &BlueprintMpConfig,
) -> HuMpLazyReport {
    let structural_checks = structural_checks(hu, mp);
    let root_action_schema_mismatches = root_action_schema_mismatches(hu, mp);
    let mut reasons = failed_check_reasons(&structural_checks);
    if !root_action_schema_mismatches.is_empty() {
        reasons.push(format!(
            "root action schema mismatch: {} root mismatch(es)",
            root_action_schema_mismatches.len()
        ));
    }

    finalize_report(
        harness,
        reasons,
        Vec::new(),
        structural_checks,
        root_action_schema_mismatches,
        HuMpLazyRowCoverage::default(),
        Vec::new(),
        None,
        HuMpLazyRuntimeStats::default(),
    )
}

/// Run the tiny diagnostic harness end to end.
pub fn run_hu_mp_lazy_harness(
    harness: &HuMpLazyHarnessConfig,
) -> Result<HuMpLazyReport, Box<dyn Error>> {
    let run_dir = std::env::temp_dir().join(format!(
        "hu_mp_lazy_harness_{}_{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    let (hu_config, mp_config) = build_equivalent_configs(harness, &run_dir);
    let preflight = preflight_report_for_configs(harness, &hu_config, &mp_config);
    let preflight_blocked = preflight
        .structural_checks
        .iter()
        .any(|check| !check.passed)
        || !preflight.root_action_schema_mismatches.is_empty();
    if preflight_blocked {
        return Ok(preflight);
    }

    let total_start = Instant::now();

    let mut hu_trainer = BlueprintTrainer::new(hu_config.clone());
    hu_trainer.skip_bucket_validation = true;
    hu_trainer.buckets.equity_fallback = true;
    let hu_start = Instant::now();
    while hu_trainer.run_batch_with_budget(None)? > 0 {}
    let hu_runtime_ms = elapsed_ms(hu_start);

    let mp_ctx = setup_lazy_training(&mp_config);
    let mp_start = Instant::now();
    let mp_result = run_lazy_training(&mp_ctx, &mp_config.training, &mp_config.game);
    let mp_runtime_ms = elapsed_ms(mp_start);

    let (row_coverage, strategy_distances, max_l1, mean_l1, max_abs) =
        compare_root_average_strategies(harness, &hu_trainer, &mp_ctx);
    let action_frequency = compare_root_action_frequencies(harness, &hu_trainer, &mp_ctx)
        .map_or_else(
            |reason| {
                let mut reasons = Vec::new();
                reasons.push(reason);
                (None, reasons)
            },
            |report| (Some(report), Vec::new()),
        );

    let hu_stats = hu_trainer.storage_stats();
    let mp_stats = mp_ctx.storage.stats();
    let runtime_stats = HuMpLazyRuntimeStats {
        hu_runtime_ms,
        mp_lazy_runtime_ms: mp_runtime_ms,
        total_runtime_ms: elapsed_ms(total_start),
        hu_iterations_completed: hu_trainer.iterations,
        mp_meta_iterations_completed: mp_result.meta_iterations,
        hu_realized_rows: hu_stats.realized_rows,
        hu_realized_slots: hu_stats.realized_slots,
        hu_resident_bytes: hu_stats.sparse_resident_bytes,
        mp_sparse_entries: mp_stats.entries,
        mp_sparse_slots: mp_stats.strategy_slots,
        mp_sparse_approx_bytes: mp_stats.approx_bytes,
    };

    let (action_frequency, reasons) = action_frequency;
    let mut warnings = Vec::new();
    if let Some(max_l1) = max_l1 {
        if max_l1 > harness.max_l1_tolerance {
            warnings.push(format!(
                "max root strategy L1 {max_l1:.6} exceeds tolerance {:.6}",
                harness.max_l1_tolerance
            ));
        }
    }
    if let Some(mean_l1) = mean_l1 {
        if mean_l1 > harness.mean_l1_tolerance {
            warnings.push(format!(
                "mean root strategy L1 {mean_l1:.6} exceeds tolerance {:.6}",
                harness.mean_l1_tolerance
            ));
        }
    }

    Ok(finalize_report(
        harness,
        reasons,
        warnings,
        preflight.structural_checks,
        preflight.root_action_schema_mismatches,
        row_coverage,
        strategy_distances,
        action_frequency,
        runtime_stats,
    )
    .with_distances(max_l1, mean_l1, max_abs))
}

fn build_hu_config(harness: &HuMpLazyHarnessConfig, output_dir: &Path) -> BlueprintV2Config {
    BlueprintV2Config {
        game: GameConfig {
            name: "hu-mp-lazy-harness-hu".to_string(),
            players: 2,
            stack_depth: harness.stack_chips,
            small_blind: harness.small_blind_chips,
            big_blind: harness.big_blind_chips,
            rake_rate: harness.rake_rate,
            rake_cap: harness.rake_cap_chips,
            allow_preflop_limp: harness.allow_preflop_limp,
        },
        clustering: ClusteringConfig {
            algorithm: ClusteringAlgorithm::PotentialAwareEmd,
            preflop: street_cluster(harness.preflop_buckets),
            flop: street_cluster(harness.flop_buckets),
            turn: street_cluster(harness.turn_buckets),
            river: street_cluster(harness.river_buckets),
            seed: harness.output_seed,
            kmeans_iterations: 1,
            cfvnet_river_data: None,
            per_flop: None,
        },
        action_abstraction: ActionAbstractionConfig {
            preflop: vec![harness
                .preflop_open_to_chips
                .iter()
                .map(|chips| chips_to_bb_label(*chips, harness.big_blind_chips))
                .collect()],
            flop: vec![harness.postflop_bet_fractions.clone()],
            turn: vec![harness.postflop_bet_fractions.clone()],
            river: vec![harness.postflop_bet_fractions.clone()],
        },
        training: TrainingConfig {
            cluster_path: None,
            iterations: Some(harness.iterations),
            time_limit_minutes: None,
            lcfr_warmup_iterations: u64::MAX / 4,
            lcfr_discount_interval: u64::MAX / 4,
            prune_after_iterations: u64::MAX / 4,
            prune_threshold: 0,
            prune_explore_pct: 0.0,
            print_every_minutes: u64::MAX / 4,
            batch_size: harness.iterations.max(1),
            dcfr_alpha: 1.5,
            dcfr_beta: 0.0,
            dcfr_gamma: 2.0,
            dcfr_epoch_cap: None,
            target_strategy_delta: None,
            purify_threshold: 0.0,
            equity_cache_path: None,
            optimizer: "dcfr".to_string(),
            storage_backend: "dense".to_string(),
            sapcfr_eta: 0.5,
            brcfr_eta: 0.6,
            brcfr_warmup_iterations: 0,
            brcfr_interval: u64::MAX / 4,
            use_baselines: false,
            baseline_alpha: 0.01,
            baseline_validation: BaselineValidationTrainingConfig::default(),
            prune_streets: None,
            regret_floor: None,
            exploitability_interval_minutes: 0,
            exploitability_samples: 1,
        },
        snapshots: SnapshotConfig {
            warmup_minutes: u64::MAX / 4,
            snapshot_every_minutes: u64::MAX / 4,
            output_dir: output_dir.to_string_lossy().to_string(),
            resume: false,
            max_snapshots: None,
            format: SnapshotFormat::Legacy,
        },
    }
}

fn build_mp_lazy_config(harness: &HuMpLazyHarnessConfig, output_dir: &Path) -> BlueprintMpConfig {
    BlueprintMpConfig {
        game: MpGameConfig {
            name: "hu-mp-lazy-harness-mp".to_string(),
            num_players: 2,
            stack_depth: harness.stack_chips,
            allow_preflop_limp: harness.allow_preflop_limp,
            blinds: vec![
                ForcedBet {
                    seat: 0,
                    kind: ForcedBetKind::SmallBlind,
                    amount: harness.small_blind_chips,
                },
                ForcedBet {
                    seat: 1,
                    kind: ForcedBetKind::BigBlind,
                    amount: harness.big_blind_chips,
                },
            ],
            rake_rate: harness.rake_rate,
            rake_cap: harness.rake_cap_chips,
        },
        action_abstraction: MpActionAbstractionConfig {
            max_flop_players: None,
            preflop: MpStreetSizes {
                lead: harness
                    .preflop_open_to_chips
                    .iter()
                    .map(|chips| yaml_string(chips_to_bb_label(*chips, harness.big_blind_chips)))
                    .collect(),
                raise: vec![harness
                    .preflop_open_to_chips
                    .iter()
                    .map(|chips| yaml_string(chips_to_bb_label(*chips, harness.big_blind_chips)))
                    .collect()],
            },
            flop: mp_postflop_sizes(&harness.postflop_bet_fractions),
            turn: mp_postflop_sizes(&harness.postflop_bet_fractions),
            river: mp_postflop_sizes(&harness.postflop_bet_fractions),
        },
        clustering: MpClusteringConfig {
            preflop: MpStreetCluster {
                buckets: harness.preflop_buckets,
            },
            flop: MpStreetCluster {
                buckets: harness.flop_buckets,
            },
            turn: MpStreetCluster {
                buckets: harness.turn_buckets,
            },
            river: MpStreetCluster {
                buckets: harness.river_buckets,
            },
        },
        training: MpTrainingConfig {
            backend: MpTrainingBackend::LazySparse,
            chance_continuation_mode: MpChanceContinuationMode::SampledFullDeal,
            cluster_path: None,
            iterations: Some(harness.iterations),
            time_limit_minutes: None,
            lcfr_warmup_iterations: u64::MAX / 4,
            lcfr_discount_interval: u64::MAX / 4,
            dcfr_discount_interval_seconds: None,
            dcfr_discount_max_passes: None,
            prune_after_iterations: u64::MAX / 4,
            traversal_pruning_enabled: false,
            prune_threshold: 0,
            prune_explore_pct: 0.0,
            negative_action_subtree_purge_enabled: false,
            negative_action_prune_below: -1,
            negative_action_reactivate_at: 0,
            negative_action_purge_mode: MpNegativeActionPurgeMode::ScanHistoryPrefix,
            batch_size: harness.iterations.max(1),
            dcfr_alpha: 1.5,
            dcfr_beta: 0.0,
            dcfr_gamma: 2.0,
            print_every_minutes: u64::MAX / 4,
            purify_threshold: 0.0,
            exploitability_interval_minutes: 0,
            exploitability_samples: 1,
        },
        snapshots: MpSnapshotConfig {
            warmup_minutes: u64::MAX / 4,
            snapshot_every_minutes: u64::MAX / 4,
            output_dir: output_dir.to_string_lossy().to_string(),
            resume: false,
            max_snapshots: None,
            format: RuntimeSnapshotFormat::Legacy,
        },
    }
}

fn structural_checks(
    hu: &BlueprintV2Config,
    mp: &BlueprintMpConfig,
) -> Vec<HuMpLazyStructuralCheck> {
    let mut checks = Vec::new();
    push_check(
        &mut checks,
        "players",
        hu.game.players == 2 && mp.game.num_players == 2,
        format!(
            "HU players={}, MP players={}",
            hu.game.players, mp.game.num_players
        ),
    );
    push_check(
        &mut checks,
        "blind_button_mapping",
        mp_blind_amount(mp, ForcedBetKind::SmallBlind, 0)
            .is_some_and(|amount| approx_eq(amount, hu.game.small_blind))
            && mp_blind_amount(mp, ForcedBetKind::BigBlind, 1)
                .is_some_and(|amount| approx_eq(amount, hu.game.big_blind)),
        format!(
            "HU dealer/SB=0 BB=1, MP SB seat0={:?}, BB seat1={:?}",
            mp_blind_amount(mp, ForcedBetKind::SmallBlind, 0),
            mp_blind_amount(mp, ForcedBetKind::BigBlind, 1)
        ),
    );
    push_check(
        &mut checks,
        "stack_blinds_rake_limp",
        approx_eq(hu.game.stack_depth, mp.game.stack_depth)
            && approx_eq(
                hu.game.small_blind,
                mp_blind_any(mp, ForcedBetKind::SmallBlind),
            )
            && approx_eq(hu.game.big_blind, mp_blind_any(mp, ForcedBetKind::BigBlind))
            && approx_eq(hu.game.rake_rate, mp.game.rake_rate)
            && approx_eq(hu.game.rake_cap, mp.game.rake_cap)
            && hu.game.allow_preflop_limp == mp.game.allow_preflop_limp,
        format!(
            "HU stack/sb/bb/rake/cap/limp=({:.2},{:.2},{:.2},{:.4},{:.2},{}), MP=({:.2},{:?},{:?},{:.4},{:.2},{})",
            hu.game.stack_depth,
            hu.game.small_blind,
            hu.game.big_blind,
            hu.game.rake_rate,
            hu.game.rake_cap,
            hu.game.allow_preflop_limp,
            mp.game.stack_depth,
            mp_blind_any(mp, ForcedBetKind::SmallBlind),
            mp_blind_any(mp, ForcedBetKind::BigBlind),
            mp.game.rake_rate,
            mp.game.rake_cap,
            mp.game.allow_preflop_limp
        ),
    );
    push_check(
        &mut checks,
        "mp_backend_lazy_sparse",
        mp.training.backend == MpTrainingBackend::LazySparse,
        format!("MP backend={:?}", mp.training.backend),
    );
    push_check(
        &mut checks,
        "mp_chance_sampled_full_deal",
        mp.training.chance_continuation_mode == MpChanceContinuationMode::SampledFullDeal,
        format!("MP chance mode={:?}", mp.training.chance_continuation_mode),
    );
    push_check(
        &mut checks,
        "bucket_counts",
        [
            hu.clustering.preflop.buckets,
            hu.clustering.flop.buckets,
            hu.clustering.turn.buckets,
            hu.clustering.river.buckets,
        ] == mp.clustering.bucket_counts(),
        format!(
            "HU={:?}, MP={:?}",
            [
                hu.clustering.preflop.buckets,
                hu.clustering.flop.buckets,
                hu.clustering.turn.buckets,
                hu.clustering.river.buckets
            ],
            mp.clustering.bucket_counts()
        ),
    );
    push_check(
        &mut checks,
        "pruning_baselines_purge_disabled",
        hu.training.prune_after_iterations > hu.training.iterations.unwrap_or(0)
            && !hu.training.use_baselines
            && !mp.training.traversal_pruning_enabled
            && mp.training.prune_after_iterations > mp.training.iterations.unwrap_or(0)
            && !mp.training.negative_action_subtree_purge_enabled,
        format!(
            "HU prune_after={} baselines={}, MP traversal_pruning={} prune_after={} negative_purge={}",
            hu.training.prune_after_iterations,
            hu.training.use_baselines,
            mp.training.traversal_pruning_enabled,
            mp.training.prune_after_iterations,
            mp.training.negative_action_subtree_purge_enabled
        ),
    );
    checks
}

fn root_action_schema_mismatches(
    hu: &BlueprintV2Config,
    mp: &BlueprintMpConfig,
) -> Vec<HuMpLazySchemaMismatch> {
    let validation_mismatches = validate_action_schema_inputs(hu, mp);
    if !validation_mismatches.is_empty() {
        return validation_mismatches;
    }

    catch_schema_construction_unwind(|| root_action_schema_mismatches_unchecked(hu, mp))
        .map_or_else(
            |panic| {
                vec![HuMpLazySchemaMismatch {
                    path: "root".to_string(),
                    action_index: 0,
                    hu_action: "<schema-construction-failed>".to_string(),
                    mp_action: "<schema-construction-failed>".to_string(),
                    reason: format!(
                        "schema construction panicked while building root action schemas: {}",
                        panic_payload_message(panic.as_ref())
                    ),
                }]
            },
            |mismatches| mismatches,
        )
}

fn catch_schema_construction_unwind<F>(f: F) -> std::thread::Result<Vec<HuMpLazySchemaMismatch>>
where
    F: FnOnce() -> Vec<HuMpLazySchemaMismatch>,
{
    static PANIC_HOOK_LOCK: Mutex<()> = Mutex::new(());

    let _guard = PANIC_HOOK_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let previous_hook = take_hook();
    set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(f));
    set_hook(previous_hook);
    result
}

fn validate_action_schema_inputs(
    hu: &BlueprintV2Config,
    mp: &BlueprintMpConfig,
) -> Vec<HuMpLazySchemaMismatch> {
    let mut mismatches = Vec::new();

    for (depth_idx, depth) in hu.action_abstraction.preflop.iter().enumerate() {
        for (size_idx, size) in depth.iter().enumerate() {
            if let Err(reason) = validate_preflop_size_label(size) {
                mismatches.push(config_schema_mismatch(
                    format!("hu.preflop[{depth_idx}][{size_idx}]"),
                    size_idx,
                    size.clone(),
                    reason,
                ));
            }
        }
    }
    for (street, depths) in [
        ("hu.flop", &hu.action_abstraction.flop),
        ("hu.turn", &hu.action_abstraction.turn),
        ("hu.river", &hu.action_abstraction.river),
    ] {
        for (depth_idx, depth) in depths.iter().enumerate() {
            for (size_idx, size) in depth.iter().enumerate() {
                if !size.is_finite() {
                    mismatches.push(config_schema_mismatch(
                        format!("{street}[{depth_idx}][{size_idx}]"),
                        size_idx,
                        format!("{size:?}"),
                        "postflop size must be finite".to_string(),
                    ));
                }
            }
        }
    }

    validate_mp_preflop_values(
        "mp.preflop.lead",
        &mp.action_abstraction.preflop.lead,
        &mut mismatches,
    );
    for (depth_idx, depth) in mp.action_abstraction.preflop.raise.iter().enumerate() {
        validate_mp_preflop_values(
            &format!("mp.preflop.raise[{depth_idx}]"),
            depth,
            &mut mismatches,
        );
    }
    validate_mp_postflop_street("mp.flop", &mp.action_abstraction.flop, &mut mismatches);
    validate_mp_postflop_street("mp.turn", &mp.action_abstraction.turn, &mut mismatches);
    validate_mp_postflop_street("mp.river", &mp.action_abstraction.river, &mut mismatches);

    mismatches
}

fn validate_mp_preflop_values(
    path: &str,
    values: &[serde_yaml::Value],
    mismatches: &mut Vec<HuMpLazySchemaMismatch>,
) {
    for (idx, value) in values.iter().enumerate() {
        match value {
            serde_yaml::Value::String(label) => {
                if let Err(reason) = validate_preflop_size_label(label) {
                    mismatches.push(config_schema_mismatch(
                        format!("{path}[{idx}]"),
                        idx,
                        label.clone(),
                        reason,
                    ));
                }
            }
            serde_yaml::Value::Number(number) => mismatches.push(config_schema_mismatch(
                format!("{path}[{idx}]"),
                idx,
                format!("{number}"),
                "preflop size must be a string ending in 'bb' or 'x'".to_string(),
            )),
            other => mismatches.push(config_schema_mismatch(
                format!("{path}[{idx}]"),
                idx,
                format!("{other:?}"),
                "unexpected YAML value type for preflop size".to_string(),
            )),
        }
    }
}

fn validate_mp_postflop_street(
    street: &str,
    sizes: &MpStreetSizes,
    mismatches: &mut Vec<HuMpLazySchemaMismatch>,
) {
    validate_mp_postflop_values(&format!("{street}.lead"), &sizes.lead, mismatches);
    for (depth_idx, depth) in sizes.raise.iter().enumerate() {
        validate_mp_postflop_values(&format!("{street}.raise[{depth_idx}]"), depth, mismatches);
    }
}

fn validate_mp_postflop_values(
    path: &str,
    values: &[serde_yaml::Value],
    mismatches: &mut Vec<HuMpLazySchemaMismatch>,
) {
    for (idx, value) in values.iter().enumerate() {
        match value.as_f64() {
            Some(size) if size.is_finite() => {}
            Some(_) => mismatches.push(config_schema_mismatch(
                format!("{path}[{idx}]"),
                idx,
                format!("{value:?}"),
                "postflop size must be finite".to_string(),
            )),
            None => mismatches.push(config_schema_mismatch(
                format!("{path}[{idx}]"),
                idx,
                format!("{value:?}"),
                "postflop size must be numeric".to_string(),
            )),
        }
    }
}

fn validate_preflop_size_label(label: &str) -> Result<(), String> {
    let trimmed = label.trim();
    let number = if let Some(stripped) = trimmed.strip_suffix("bb") {
        stripped
    } else if let Some(stripped) = trimmed.strip_suffix('x') {
        stripped
    } else {
        return Err("preflop size must end with 'bb' or 'x'".to_string());
    };
    match number.parse::<f64>() {
        Ok(value) if value.is_finite() => Ok(()),
        Ok(_) => Err("preflop size must be finite".to_string()),
        Err(_) => Err("preflop size must contain a valid number".to_string()),
    }
}

fn config_schema_mismatch(
    path: String,
    action_index: usize,
    action: String,
    reason: String,
) -> HuMpLazySchemaMismatch {
    HuMpLazySchemaMismatch {
        path,
        action_index,
        hu_action: action,
        mp_action: "<invalid-config>".to_string(),
        reason,
    }
}

fn root_action_schema_mismatches_unchecked(
    hu: &BlueprintV2Config,
    mp: &BlueprintMpConfig,
) -> Vec<HuMpLazySchemaMismatch> {
    let hu_tree = poker_solver_core::blueprint_v2::game_tree::GameTree::build_with_options(
        hu.game.stack_depth,
        hu.game.small_blind,
        hu.game.big_blind,
        &hu.action_abstraction.preflop,
        &hu.action_abstraction.flop,
        &hu.action_abstraction.turn,
        &hu.action_abstraction.river,
        hu.game.allow_preflop_limp,
    );
    let hu_actions = match &hu_tree.nodes[hu_tree.root as usize] {
        HuGameNode::Decision { actions, .. } => comparable_hu_root_actions(actions, hu)
            .iter()
            .map(|(_, action)| action.clone())
            .collect::<Vec<NormalizedActionDescriptor>>(),
        other => {
            return vec![HuMpLazySchemaMismatch {
                path: "root".to_string(),
                action_index: 0,
                hu_action: format!("{other:?}"),
                mp_action: String::new(),
                reason: "HU root is not a decision node".to_string(),
            }];
        }
    };

    let mp_game = poker_solver_core::blueprint_mp::lazy_mccfr::LazyMpGame::new(
        &mp.game,
        &mp.action_abstraction,
    );
    let mp_root = LazyResolvedSpot::root(&mp_game);
    let mp_actions = mp_root
        .actions(&mp_game)
        .iter()
        .enumerate()
        .map(|(idx, action)| (idx, normalize_mp_root_action(action)))
        .map(|(_, action)| action)
        .collect::<Vec<NormalizedActionDescriptor>>();

    compare_action_descriptors("root", &hu_actions, &mp_actions)
}

fn comparable_hu_root_actions(
    actions: &[HuTreeAction],
    hu: &BlueprintV2Config,
) -> Vec<(usize, NormalizedActionDescriptor)> {
    actions
        .iter()
        .enumerate()
        .filter_map(|(idx, action)| {
            normalize_hu_root_action(action, hu).map(|action| (idx, action))
        })
        .collect()
}

fn panic_payload_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn compare_root_average_strategies(
    harness: &HuMpLazyHarnessConfig,
    hu_trainer: &BlueprintTrainer,
    mp_ctx: &poker_solver_core::blueprint_mp::trainer::LazyTrainContext,
) -> (
    HuMpLazyRowCoverage,
    Vec<HuMpLazyStrategyDistance>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
) {
    let hu_actions = match &hu_trainer.tree.nodes[hu_trainer.tree.root as usize] {
        HuGameNode::Decision { actions, .. } => {
            comparable_hu_root_actions(actions, &hu_trainer.config)
        }
        _ => Vec::new(),
    };
    if hu_actions.is_empty() {
        return Default::default();
    }

    let mp_root = LazyResolvedSpot::root(&mp_ctx.game);
    let mp_root_actions = mp_root.actions(&mp_ctx.game);
    let mp_actions = mp_root_actions
        .iter()
        .enumerate()
        .map(|(idx, action)| (idx, normalize_mp_root_action(action)))
        .collect::<Vec<_>>();
    if hu_actions.len() != mp_actions.len() {
        return Default::default();
    }

    let snapshot = mp_ctx.storage.snapshot_entries();
    let visited: HashSet<MpInfosetKey> = snapshot.iter().map(|entry| entry.key).collect();
    let mut distances = Vec::new();
    let mut mp_root_rows_visited = 0;

    for bucket in 0..harness.preflop_buckets {
        let hu_full_strategy = hu_trainer
            .storage
            .average_strategy(hu_trainer.tree.root, bucket);
        let hu_strategy = hu_actions
            .iter()
            .map(|(idx, _)| hu_full_strategy[*idx])
            .collect::<Vec<_>>();
        let mut mp_full_strategy = vec![0.0; mp_root_actions.len()];
        let key = mp_root.key_for_bucket(bucket);
        let mp_row_visited = visited.contains(&key);
        if mp_row_visited {
            mp_root_rows_visited += 1;
        }
        mp_ctx
            .storage
            .average_strategy(key, mp_root_actions.len(), &mut mp_full_strategy);
        let mp_strategy = mp_actions
            .iter()
            .map(|(idx, _)| mp_full_strategy[*idx])
            .collect::<Vec<_>>();

        let (l1_distance, max_abs_distance) = strategy_distance(&hu_strategy, &mp_strategy);
        distances.push(HuMpLazyStrategyDistance {
            path: "root".to_string(),
            bucket,
            hu_strategy: format_strategy(&hu_strategy),
            mp_strategy: format_strategy(&mp_strategy),
            l1_distance,
            max_abs_distance,
            mp_row_visited,
        });
    }

    let compared_rows = distances.len();
    let max_l1 = distances.iter().map(|d| d.l1_distance).reduce(f64::max);
    let mean_l1 = if distances.is_empty() {
        None
    } else {
        Some(distances.iter().map(|d| d.l1_distance).sum::<f64>() / distances.len() as f64)
    };
    let max_abs = distances
        .iter()
        .map(|d| d.max_abs_distance)
        .reduce(f64::max);

    (
        HuMpLazyRowCoverage {
            compared_rows,
            hu_dense_rows: usize::from(harness.preflop_buckets),
            mp_sparse_rows_total: snapshot.len(),
            mp_root_rows_visited,
            missing_mp_root_rows: usize::from(harness.preflop_buckets) - mp_root_rows_visited,
        },
        distances,
        max_l1,
        mean_l1,
        max_abs,
    )
}

fn compare_root_action_frequencies(
    harness: &HuMpLazyHarnessConfig,
    hu_trainer: &BlueprintTrainer,
    mp_ctx: &poker_solver_core::blueprint_mp::trainer::LazyTrainContext,
) -> Result<HuMpLazyActionFrequencyReport, String> {
    let weights = canonical_preflop_bucket_weights(harness.preflop_buckets)?;
    let hu_actions = match &hu_trainer.tree.nodes[hu_trainer.tree.root as usize] {
        HuGameNode::Decision { actions, .. } => {
            comparable_hu_root_actions(actions, &hu_trainer.config)
        }
        _ => Vec::new(),
    };
    if hu_actions.is_empty() {
        return Err("root action-frequency evidence was not produced: HU root action extraction returned no comparable actions".to_string());
    }

    let mp_root = LazyResolvedSpot::root(&mp_ctx.game);
    let mp_root_actions = mp_root.actions(&mp_ctx.game);
    let mp_actions = mp_root_actions
        .iter()
        .enumerate()
        .map(|(idx, action)| (idx, normalize_mp_root_action(action)))
        .collect::<Vec<_>>();
    let hu_descriptors = hu_actions
        .iter()
        .map(|(_, action)| action.clone())
        .collect::<Vec<_>>();
    let hu_action_indices = hu_actions.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
    let mp_descriptors = mp_actions
        .iter()
        .map(|(_, action)| action.clone())
        .collect::<Vec<_>>();
    let mp_action_indices = mp_actions.iter().map(|(idx, _)| *idx).collect::<Vec<_>>();
    let descriptor_mismatches =
        compare_action_descriptors("root", &hu_descriptors, &mp_descriptors);
    if !descriptor_mismatches.is_empty() {
        return Err(format!(
            "root action-frequency evidence was not produced: {} root action descriptor mismatch(es)",
            descriptor_mismatches.len()
        ));
    }

    let snapshot = mp_ctx.storage.snapshot_entries();
    let visited: HashSet<MpInfosetKey> = snapshot.iter().map(|entry| entry.key).collect();
    let mut rows = Vec::with_capacity(weights.len());
    for weight in weights {
        let hu_full_strategy = hu_trainer
            .storage
            .average_strategy(hu_trainer.tree.root, weight.bucket);

        let key = mp_root.key_for_bucket(weight.bucket);
        let mp_row_visited = visited.contains(&key);
        let mut mp_full_strategy = vec![0.0; mp_root_actions.len()];
        mp_ctx
            .storage
            .average_strategy(key, mp_root_actions.len(), &mut mp_full_strategy);

        rows.push(ActionFrequencyInputRow {
            bucket: weight.bucket,
            hand_label: weight.hand_label,
            combo_count: weight.combo_count,
            normalized_weight: weight.normalized_weight,
            hu_probabilities: hu_full_strategy,
            mp_probabilities: mp_full_strategy,
            mp_row_visited,
        });
    }

    Ok(aggregate_action_frequency_rows(
        "root",
        &hu_descriptors,
        &hu_action_indices,
        &mp_action_indices,
        rows,
        harness.worst_action_frequency_spots,
        harness.max_action_frequency_delta_tolerance,
        harness.total_action_frequency_l1_tolerance,
    ))
}

#[derive(Debug, Clone)]
struct CanonicalPreflopBucketWeight {
    bucket: u16,
    hand_label: String,
    combo_count: usize,
    normalized_weight: f64,
}

#[derive(Debug, Clone)]
struct ActionFrequencyInputRow {
    bucket: u16,
    hand_label: String,
    combo_count: usize,
    normalized_weight: f64,
    hu_probabilities: Vec<f64>,
    mp_probabilities: Vec<f64>,
    mp_row_visited: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbabilityRowError {
    Invalid,
    ZeroSum,
    OffNormalized,
}

fn canonical_preflop_bucket_weights(
    bucket_count: u16,
) -> Result<Vec<CanonicalPreflopBucketWeight>, String> {
    if bucket_count != 169 {
        return Err(format!(
            "root action-frequency evidence was not produced: canonical_combo_count_preflop requires 169 buckets, got {bucket_count}"
        ));
    }

    let mut weights = Vec::with_capacity(169);
    let mut total_weight = 0_usize;
    for bucket in 0..bucket_count {
        let hand = CanonicalHand::from_index(usize::from(bucket)).ok_or_else(|| {
            format!("canonical_combo_count_preflop missing canonical hand for bucket {bucket}")
        })?;
        let combo_count = usize::from(hand.num_combos());
        total_weight += combo_count;
        weights.push(CanonicalPreflopBucketWeight {
            bucket,
            hand_label: hand.to_string(),
            combo_count,
            normalized_weight: 0.0,
        });
    }

    if total_weight != 1326 {
        return Err(format!(
            "canonical_combo_count_preflop total weight {total_weight} != 1326"
        ));
    }
    for weight in &mut weights {
        weight.normalized_weight = weight.combo_count as f64 / total_weight as f64;
    }
    Ok(weights)
}

fn aggregate_action_frequency_rows(
    path: &str,
    actions: &[NormalizedActionDescriptor],
    hu_action_indices: &[usize],
    mp_action_indices: &[usize],
    rows: Vec<ActionFrequencyInputRow>,
    worst_spot_limit: usize,
    max_action_frequency_delta_tolerance: f64,
    total_action_frequency_l1_tolerance: f64,
) -> HuMpLazyActionFrequencyReport {
    let mut coverage = HuMpLazyActionFrequencyCoverage::default();
    let mut hu_action_frequencies = vec![0.0; actions.len()];
    let mut mp_action_frequencies = vec![0.0; actions.len()];
    let mut worst_spots = Vec::new();
    let mut weighted_row_l1 = 0.0;
    let mut max_row_l1 = 0.0_f64;
    let total_weight = rows.iter().map(|row| row.combo_count).sum::<usize>();
    let normalized_weight_total = rows.iter().map(|row| row.normalized_weight).sum::<f64>();
    let bucket_count = rows.len();

    for row in rows {
        let mut skip_row = false;
        if !row.mp_row_visited {
            coverage.missing_mp_rows += 1;
            skip_row = true;
        }

        if let Err(error) = validate_probability_row(&row.hu_probabilities, hu_action_indices) {
            mark_probability_row_error(&mut coverage, error);
            skip_row = true;
        }
        if let Err(error) = validate_probability_row(&row.mp_probabilities, mp_action_indices) {
            mark_probability_row_error(&mut coverage, error);
            skip_row = true;
        }

        if skip_row {
            coverage.skipped_rows += 1;
            coverage.skipped_normalized_weight_total += row.normalized_weight;
            continue;
        }

        coverage.matched_rows += 1;
        coverage.included_normalized_weight_total += row.normalized_weight;

        let hu_unmatched_mass =
            unmatched_probability_mass(&row.hu_probabilities, hu_action_indices);
        let mp_unmatched_mass =
            unmatched_probability_mass(&row.mp_probabilities, mp_action_indices);
        if hu_unmatched_mass > UNMATCHED_ACTION_MASS_EPSILON {
            coverage.hu_unmatched_rows += 1;
            coverage.hu_unmatched_probability_mass += hu_unmatched_mass;
            coverage.hu_unmatched_weighted_mass += row.normalized_weight * hu_unmatched_mass;
        }
        if mp_unmatched_mass > UNMATCHED_ACTION_MASS_EPSILON {
            coverage.mp_unmatched_rows += 1;
            coverage.mp_unmatched_probability_mass += mp_unmatched_mass;
            coverage.mp_unmatched_weighted_mass += row.normalized_weight * mp_unmatched_mass;
        }

        let hu = hu_action_indices
            .iter()
            .map(|idx| row.hu_probabilities[*idx])
            .collect::<Vec<_>>();
        let mp = mp_action_indices
            .iter()
            .map(|idx| row.mp_probabilities[*idx])
            .collect::<Vec<_>>();

        let (row_l1, _) = strategy_distance(&hu, &mp);
        max_row_l1 = max_row_l1.max(row_l1);
        let weighted_l1_contribution = row.normalized_weight * row_l1;
        weighted_row_l1 += weighted_l1_contribution;

        for (action_index, ((hu_probability, mp_probability), action)) in
            hu.iter().zip(&mp).zip(actions).enumerate()
        {
            hu_action_frequencies[action_index] += row.normalized_weight * *hu_probability;
            mp_action_frequencies[action_index] += row.normalized_weight * *mp_probability;
            worst_spots.push(HuMpLazyActionFrequencySpot {
                path: path.to_string(),
                bucket: row.bucket,
                hand_label: row.hand_label.clone(),
                combo_count: row.combo_count,
                normalized_weight: row.normalized_weight,
                action_index,
                action_label: descriptor_label(action),
                hu_probability: *hu_probability,
                mp_probability: *mp_probability,
                abs_delta: (*hu_probability - *mp_probability).abs(),
                row_l1,
                weighted_l1_contribution,
                mp_row_visited: row.mp_row_visited,
            });
        }
    }

    let aggregate_actions = actions
        .iter()
        .enumerate()
        .map(|(action_index, action)| {
            let hu_frequency = hu_action_frequencies[action_index];
            let mp_frequency = mp_action_frequencies[action_index];
            let delta = hu_frequency - mp_frequency;
            HuMpLazyActionFrequencyAggregateRow {
                path: path.to_string(),
                action_index,
                action_label: descriptor_label(action),
                action_kind: action.kind.clone(),
                amount_chips: action.amount_chips,
                hu_frequency,
                mp_frequency,
                delta,
                abs_delta: delta.abs(),
            }
        })
        .collect::<Vec<_>>();
    let total_l1_delta = aggregate_actions
        .iter()
        .map(|row| row.abs_delta)
        .sum::<f64>();
    let max_abs_delta = aggregate_actions
        .iter()
        .map(|row| row.abs_delta)
        .fold(0.0_f64, f64::max);

    worst_spots.sort_by(|a, b| {
        b.abs_delta
            .total_cmp(&a.abs_delta)
            .then_with(|| {
                b.weighted_l1_contribution
                    .total_cmp(&a.weighted_l1_contribution)
            })
            .then_with(|| a.bucket.cmp(&b.bucket))
            .then_with(|| a.action_index.cmp(&b.action_index))
    });
    worst_spots.truncate(worst_spot_limit);

    let coverage_blocks_thresholds = coverage.missing_mp_rows > 0
        || coverage.skipped_rows > 0
        || coverage.invalid_rows > 0
        || coverage.zero_sum_rows > 0
        || coverage.off_normalized_rows > 0
        || coverage.hu_unmatched_weighted_mass > UNMATCHED_ACTION_MASS_EPSILON
        || coverage.mp_unmatched_weighted_mass > UNMATCHED_ACTION_MASS_EPSILON
        || coverage.included_normalized_weight_total + PROBABILITY_SUM_EPSILON
            < normalized_weight_total;

    HuMpLazyActionFrequencyReport {
        distribution: HuMpLazyActionFrequencyDistribution {
            name: "canonical_combo_count_preflop".to_string(),
            bucket_count,
            total_weight,
            normalized_weight_total,
        },
        coverage,
        aggregate_actions,
        worst_spots,
        totals: HuMpLazyActionFrequencyTotals {
            total_l1_delta,
            max_abs_delta,
            weighted_row_l1,
            max_row_l1,
            passed_thresholds: !coverage_blocks_thresholds
                && max_abs_delta <= max_action_frequency_delta_tolerance
                && total_l1_delta <= total_action_frequency_l1_tolerance,
        },
    }
}

fn validate_probability_row(
    probabilities: &[f64],
    matched_action_indices: &[usize],
) -> Result<(), ProbabilityRowError> {
    if probabilities.is_empty()
        || matched_action_indices
            .iter()
            .any(|idx| *idx >= probabilities.len())
        || probabilities
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(ProbabilityRowError::Invalid);
    }
    let total = probabilities.iter().sum::<f64>();
    if total <= 0.0 {
        return Err(ProbabilityRowError::ZeroSum);
    }
    if (total - 1.0).abs() > PROBABILITY_SUM_EPSILON {
        return Err(ProbabilityRowError::OffNormalized);
    }
    Ok(())
}

fn mark_probability_row_error(
    coverage: &mut HuMpLazyActionFrequencyCoverage,
    error: ProbabilityRowError,
) {
    match error {
        ProbabilityRowError::Invalid => coverage.invalid_rows += 1,
        ProbabilityRowError::ZeroSum => coverage.zero_sum_rows += 1,
        ProbabilityRowError::OffNormalized => {
            coverage.invalid_rows += 1;
            coverage.off_normalized_rows += 1;
        }
    }
}

fn unmatched_probability_mass(probabilities: &[f64], matched_action_indices: &[usize]) -> f64 {
    probabilities
        .iter()
        .enumerate()
        .filter(|(idx, _)| !matched_action_indices.contains(idx))
        .map(|(_, probability)| *probability)
        .sum()
}

fn has_unmatched_action_mass(coverage: &HuMpLazyActionFrequencyCoverage) -> bool {
    coverage.hu_unmatched_weighted_mass > UNMATCHED_ACTION_MASS_EPSILON
        || coverage.mp_unmatched_weighted_mass > UNMATCHED_ACTION_MASS_EPSILON
}

fn has_incomplete_action_frequency_coverage(coverage: &HuMpLazyActionFrequencyCoverage) -> bool {
    coverage.missing_mp_rows > 0
        || coverage.skipped_rows > 0
        || coverage.invalid_rows > 0
        || coverage.zero_sum_rows > 0
        || coverage.off_normalized_rows > 0
        || has_unmatched_action_mass(coverage)
}

fn has_action_frequency_missing_reason(reasons: &[String]) -> bool {
    reasons
        .iter()
        .any(|reason| reason.contains("root action-frequency evidence was not produced"))
}

fn finalize_report(
    harness: &HuMpLazyHarnessConfig,
    mut reasons: Vec<String>,
    mut warnings: Vec<String>,
    structural_checks: Vec<HuMpLazyStructuralCheck>,
    root_action_schema_mismatches: Vec<HuMpLazySchemaMismatch>,
    row_coverage: HuMpLazyRowCoverage,
    strategy_distances: Vec<HuMpLazyStrategyDistance>,
    action_frequency: Option<HuMpLazyActionFrequencyReport>,
    runtime_stats: HuMpLazyRuntimeStats,
) -> HuMpLazyReport {
    if strategy_distances.is_empty()
        && structural_checks.iter().all(|check| check.passed)
        && root_action_schema_mismatches.is_empty()
        && harness.iterations > 0
    {
        warnings
            .push("strategy distance rows are empty; no comparable evidence was produced".into());
    }
    if action_frequency.is_none()
        && structural_checks.iter().all(|check| check.passed)
        && root_action_schema_mismatches.is_empty()
        && !has_action_frequency_missing_reason(&reasons)
    {
        reasons.push("root action-frequency evidence was not produced".into());
    }
    if let Some(action_frequency) = &action_frequency {
        if action_frequency.coverage.missing_mp_rows > 0 {
            reasons.push(format!(
                "root action-frequency coverage missing {} MP sparse row(s); those rows were excluded from the primary metric",
                action_frequency.coverage.missing_mp_rows
            ));
        }
        if action_frequency.coverage.skipped_rows > 0 {
            reasons.push(format!(
                "root action-frequency skipped {} row(s); included weight {:.6} of input weight {:.6}",
                action_frequency.coverage.skipped_rows,
                action_frequency.coverage.included_normalized_weight_total,
                action_frequency.distribution.normalized_weight_total
            ));
        }
        if action_frequency.coverage.invalid_rows > 0 {
            reasons.push(format!(
                "root action-frequency saw {} invalid source strategy row(s)",
                action_frequency.coverage.invalid_rows
            ));
        }
        if action_frequency.coverage.zero_sum_rows > 0 {
            reasons.push(format!(
                "root action-frequency saw {} zero-sum source strategy row(s)",
                action_frequency.coverage.zero_sum_rows
            ));
        }
        if action_frequency.coverage.off_normalized_rows > 0 {
            reasons.push(format!(
                "root action-frequency saw {} materially off-normalized source strategy row(s)",
                action_frequency.coverage.off_normalized_rows
            ));
        }
        if action_frequency.coverage.hu_unmatched_weighted_mass > UNMATCHED_ACTION_MASS_EPSILON {
            reasons.push(format!(
                "root action-frequency HU comparable actions omitted weighted probability mass {:.6} across {} row(s)",
                action_frequency.coverage.hu_unmatched_weighted_mass,
                action_frequency.coverage.hu_unmatched_rows
            ));
        }
        if action_frequency.coverage.mp_unmatched_weighted_mass > UNMATCHED_ACTION_MASS_EPSILON {
            reasons.push(format!(
                "root action-frequency MP comparable actions omitted weighted probability mass {:.6} across {} row(s)",
                action_frequency.coverage.mp_unmatched_weighted_mass,
                action_frequency.coverage.mp_unmatched_rows
            ));
        }
        if has_incomplete_action_frequency_coverage(&action_frequency.coverage)
            && action_frequency.totals.passed_thresholds
        {
            reasons.push(
                "root action-frequency thresholds passed despite incomplete coverage".to_string(),
            );
        }
        if !action_frequency.totals.passed_thresholds {
            reasons.push(
                "root action-frequency thresholds did not pass after coverage checks".to_string(),
            );
        }
        if action_frequency.coverage.skipped_rows > 0
            && action_frequency.coverage.skipped_normalized_weight_total > 0.0
        {
            reasons.push(format!(
                "root action-frequency skipped normalized input weight {:.6}",
                action_frequency.coverage.skipped_normalized_weight_total
            ));
        }
        if action_frequency.coverage.uniform_fallback_rows > 0 {
            reasons.push(format!(
                "root action-frequency used {} uniform fallback row(s)",
                action_frequency.coverage.uniform_fallback_rows
            ));
        }
        if action_frequency.coverage.skipped_rows > 0 && action_frequency.coverage.matched_rows == 0
        {
            reasons.push(
                "root action-frequency has no valid included rows after coverage filtering"
                    .to_string(),
            );
        }
        if action_frequency.coverage.skipped_rows > 0
            && action_frequency.coverage.included_normalized_weight_total + PROBABILITY_SUM_EPSILON
                < action_frequency.distribution.normalized_weight_total
        {
            reasons.push(format!(
                "root action-frequency included only {:.6} of {:.6} normalized input weight",
                action_frequency.coverage.included_normalized_weight_total,
                action_frequency.distribution.normalized_weight_total
            ));
        }
        if action_frequency.coverage.skipped_rows > 0
            && action_frequency.coverage.invalid_rows == 0
            && action_frequency.coverage.zero_sum_rows == 0
            && action_frequency.coverage.missing_mp_rows == 0
        {
            reasons.push(format!(
                "root action-frequency skipped {} row(s) for coverage reasons",
                action_frequency.coverage.skipped_rows
            ));
        }
        if action_frequency.totals.max_abs_delta > harness.max_action_frequency_delta_tolerance {
            reasons.push(format!(
                "max root action-frequency delta {:.6} exceeds tolerance {:.6}",
                action_frequency.totals.max_abs_delta, harness.max_action_frequency_delta_tolerance
            ));
        }
        if action_frequency.totals.total_l1_delta > harness.total_action_frequency_l1_tolerance {
            reasons.push(format!(
                "total root action-frequency L1 {:.6} exceeds tolerance {:.6}",
                action_frequency.totals.total_l1_delta, harness.total_action_frequency_l1_tolerance
            ));
        }
    }
    warnings.push(
        "average-strategy accounting is not reconciled: HU sums traverser nodes only, MP lazy also accumulates sampled opponent nodes"
            .to_string(),
    );
    let verdict = if reasons.is_empty() {
        HuMpLazyVerdict::Go
    } else {
        HuMpLazyVerdict::NoGo
    };
    HuMpLazyReport {
        verdict,
        reasons,
        warnings,
        structural_checks,
        root_action_schema_mismatches,
        row_coverage,
        strategy_distances,
        max_l1_distance: None,
        mean_l1_distance: None,
        max_abs_distance: None,
        action_frequency,
        runtime_stats,
        average_strategy_accounting_reconciled: false,
    }
}

trait WithDistances {
    fn with_distances(
        self,
        max_l1: Option<f64>,
        mean_l1: Option<f64>,
        max_abs: Option<f64>,
    ) -> Self;
}

impl WithDistances for HuMpLazyReport {
    fn with_distances(
        mut self,
        max_l1: Option<f64>,
        mean_l1: Option<f64>,
        max_abs: Option<f64>,
    ) -> Self {
        self.max_l1_distance = max_l1;
        self.mean_l1_distance = mean_l1;
        self.max_abs_distance = max_abs;
        self
    }
}

fn failed_check_reasons(checks: &[HuMpLazyStructuralCheck]) -> Vec<String> {
    checks
        .iter()
        .filter(|check| !check.passed)
        .map(|check| format!("structural check failed: {} ({})", check.name, check.detail))
        .collect()
}

fn compare_action_descriptors(
    path: &str,
    hu_actions: &[NormalizedActionDescriptor],
    mp_actions: &[NormalizedActionDescriptor],
) -> Vec<HuMpLazySchemaMismatch> {
    let max_len = hu_actions.len().max(mp_actions.len());
    let mut mismatches = Vec::new();
    for idx in 0..max_len {
        match (hu_actions.get(idx), mp_actions.get(idx)) {
            (Some(hu), Some(mp)) if descriptors_match(hu, mp) => {}
            (Some(hu), Some(mp)) => mismatches.push(HuMpLazySchemaMismatch {
                path: path.to_string(),
                action_index: idx,
                hu_action: descriptor_label(hu),
                mp_action: descriptor_label(mp),
                reason: "descriptor differs".to_string(),
            }),
            (Some(hu), None) => mismatches.push(HuMpLazySchemaMismatch {
                path: path.to_string(),
                action_index: idx,
                hu_action: descriptor_label(hu),
                mp_action: "<missing>".to_string(),
                reason: "MP action missing".to_string(),
            }),
            (None, Some(mp)) => mismatches.push(HuMpLazySchemaMismatch {
                path: path.to_string(),
                action_index: idx,
                hu_action: "<missing>".to_string(),
                mp_action: descriptor_label(mp),
                reason: "HU action missing".to_string(),
            }),
            (None, None) => {}
        }
    }
    mismatches
}

fn descriptors_match(a: &NormalizedActionDescriptor, b: &NormalizedActionDescriptor) -> bool {
    a.kind == b.kind
        && match (a.amount_chips, b.amount_chips) {
            (Some(x), Some(y)) => (x - y).abs() <= CHIP_EPSILON,
            (None, None) => true,
            _ => false,
        }
}

fn normalize_hu_action(action: &HuTreeAction) -> NormalizedActionDescriptor {
    match *action {
        HuTreeAction::Fold => normalized("Fold", None),
        HuTreeAction::Check => normalized("Check", None),
        HuTreeAction::Call => normalized("Call", None),
        HuTreeAction::Bet(amount) => normalized("Lead", Some(amount)),
        HuTreeAction::Raise(amount) => normalized("Raise", Some(amount)),
        HuTreeAction::AllIn => normalized("AllIn", None),
    }
}

fn normalize_hu_root_action(
    action: &HuTreeAction,
    hu: &BlueprintV2Config,
) -> Option<NormalizedActionDescriptor> {
    if matches!(action, HuTreeAction::AllIn) {
        let sb_remaining = hu.game.stack_depth - hu.game.small_blind;
        let root_call = hu.game.big_blind - hu.game.small_blind;
        if root_call >= sb_remaining - CHIP_EPSILON {
            return Some(normalized("Call", None));
        }
        return None;
    }
    Some(normalize_hu_action(action))
}

fn normalize_mp_root_action(action: &MpTreeAction) -> NormalizedActionDescriptor {
    match *action {
        MpTreeAction::Lead(amount) => normalized("Raise", Some(amount)),
        _ => normalize_mp_action(action),
    }
}

fn normalize_mp_action(action: &MpTreeAction) -> NormalizedActionDescriptor {
    match *action {
        MpTreeAction::Fold => normalized("Fold", None),
        MpTreeAction::Check => normalized("Check", None),
        MpTreeAction::Call => normalized("Call", None),
        MpTreeAction::Lead(amount) => normalized("Lead", Some(amount)),
        MpTreeAction::Raise(amount) => normalized("Raise", Some(amount)),
        MpTreeAction::AllIn => normalized("AllIn", None),
    }
}

fn normalized(kind: &str, amount_chips: Option<f64>) -> NormalizedActionDescriptor {
    NormalizedActionDescriptor {
        kind: kind.to_string(),
        amount_chips,
    }
}

fn descriptor_label(descriptor: &NormalizedActionDescriptor) -> String {
    match descriptor.amount_chips {
        Some(amount) => format!("{}({amount:.4})", descriptor.kind),
        None => descriptor.kind.clone(),
    }
}

fn push_check(checks: &mut Vec<HuMpLazyStructuralCheck>, name: &str, passed: bool, detail: String) {
    checks.push(HuMpLazyStructuralCheck {
        name: name.to_string(),
        passed,
        detail,
    });
}

fn mp_blind_amount(mp: &BlueprintMpConfig, kind: ForcedBetKind, seat: u8) -> Option<f64> {
    mp.game
        .blinds
        .iter()
        .find(|blind| blind.kind == kind && blind.seat == seat)
        .map(|blind| blind.amount)
}

fn mp_blind_any(mp: &BlueprintMpConfig, kind: ForcedBetKind) -> f64 {
    mp.game
        .blinds
        .iter()
        .find(|blind| blind.kind == kind)
        .map_or(f64::NAN, |blind| blind.amount)
}

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= CHIP_EPSILON
}

fn strategy_distance(a: &[f64], b: &[f64]) -> (f64, f64) {
    let mut l1 = 0.0;
    let mut max_abs = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = (x - y).abs();
        l1 += diff;
        max_abs = max_abs.max(diff);
    }
    (l1, max_abs)
}

fn format_strategy(strategy: &[f64]) -> String {
    strategy
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join("|")
}

fn format_optional_f64(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".to_string(), |v| format!("{v:.6}"))
}

fn format_percent(value: f64) -> String {
    format!("{:.2}%", value * 100.0)
}

fn format_signed_percent(value: f64) -> String {
    format!("{:+.2}%", value * 100.0)
}

fn elapsed_ms(start: Instant) -> u64 {
    start.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

fn chips_to_bb_label(chips: f64, big_blind_chips: f64) -> String {
    format!("{}bb", chips / big_blind_chips)
}

fn street_cluster(buckets: u16) -> StreetClusterConfig {
    StreetClusterConfig {
        buckets,
        delta_bins: None,
        expected_delta: false,
        sample_boards: None,
        metric: Default::default(),
    }
}

fn mp_postflop_sizes(fractions: &[f64]) -> MpStreetSizes {
    MpStreetSizes {
        lead: fractions.iter().copied().map(yaml_f64).collect(),
        raise: vec![fractions.iter().copied().map(yaml_f64).collect()],
    }
}

fn yaml_string(value: String) -> serde_yaml::Value {
    serde_yaml::Value::String(value)
}

fn yaml_f64(value: f64) -> serde_yaml::Value {
    serde_yaml::to_value(value).expect("finite f64 serializes to yaml")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn default_configs() -> (
        HuMpLazyHarnessConfig,
        BlueprintV2Config,
        BlueprintMpConfig,
        PathBuf,
    ) {
        let harness = HuMpLazyHarnessConfig::default();
        let output_dir = PathBuf::from("/tmp/hu_mp_lazy_harness_tests");
        let (hu, mp) = build_equivalent_configs(&harness, &output_dir);
        (harness, hu, mp, output_dir)
    }

    #[test]
    fn hu_mp_lazy_report_serializes_go_no_go_and_save_writes_expected_files() {
        let report = HuMpLazyReport {
            verdict: HuMpLazyVerdict::Go,
            reasons: Vec::new(),
            warnings: vec!["diagnostic only".to_string()],
            structural_checks: vec![HuMpLazyStructuralCheck {
                name: "sample".to_string(),
                passed: true,
                detail: "ok".to_string(),
            }],
            root_action_schema_mismatches: Vec::new(),
            row_coverage: HuMpLazyRowCoverage {
                compared_rows: 1,
                hu_dense_rows: 1,
                mp_sparse_rows_total: 1,
                mp_root_rows_visited: 1,
                missing_mp_root_rows: 0,
            },
            strategy_distances: vec![HuMpLazyStrategyDistance {
                path: "root".to_string(),
                bucket: 0,
                hu_strategy: "0.5|0.5".to_string(),
                mp_strategy: "0.5|0.5".to_string(),
                l1_distance: 0.0,
                max_abs_distance: 0.0,
                mp_row_visited: true,
            }],
            max_l1_distance: Some(0.0),
            mean_l1_distance: Some(0.0),
            max_abs_distance: Some(0.0),
            action_frequency: Some(sample_action_frequency_report()),
            runtime_stats: HuMpLazyRuntimeStats::default(),
            average_strategy_accounting_reconciled: true,
        };
        let json = serde_json::to_string(&report).expect("serialize GO report");
        assert!(json.contains("GO"));
        assert!(json.contains("warnings"));
        let mut legacy_value = serde_json::to_value(&report).expect("convert report to JSON value");
        legacy_value
            .as_object_mut()
            .expect("report JSON object")
            .remove("warnings");
        let legacy_report: HuMpLazyReport =
            serde_json::from_value(legacy_value).expect("deserialize legacy report");
        assert!(legacy_report.warnings.is_empty());

        let mut no_go = report.clone();
        no_go.verdict = HuMpLazyVerdict::NoGo;
        no_go.reasons.push("blocked".to_string());
        let json = serde_json::to_string(&no_go).expect("serialize NO-GO report");
        assert!(json.contains("NO-GO"));

        let dir = tempfile::tempdir().expect("tempdir");
        no_go.save(dir.path()).expect("save report");
        assert!(dir.path().join("summary.json").exists());
        assert!(dir.path().join("report.txt").exists());
        assert!(dir
            .path()
            .join("root_action_schema_mismatches.csv")
            .exists());
        assert!(dir.path().join("strategy_distance.csv").exists());
        assert!(dir.path().join("root_action_frequencies.csv").exists());
        assert!(dir.path().join("root_action_frequency_spots.csv").exists());

        let summary_json =
            std::fs::read_to_string(dir.path().join("summary.json")).expect("read summary");
        assert!(summary_json.contains("action_frequency"));
        assert!(summary_json.contains("warnings"));
        let text = std::fs::read_to_string(dir.path().join("report.txt")).expect("read report");
        assert!(text.contains("Reasons (hard NO-GO causes)"));
        assert!(text.contains("Warnings"));
        assert!(text.contains("complete canonical preflop root combo-weighted"));
        assert!(text.contains("Root Action Frequencies"));
        assert!(text.contains("Fold"));
    }

    fn sample_action_frequency_report() -> HuMpLazyActionFrequencyReport {
        aggregate_action_frequency_rows(
            "root",
            &[
                normalized("Fold", None),
                normalized("Call", None),
                normalized("Raise", Some(6.0)),
            ],
            &[0, 1, 2],
            &[0, 1, 2],
            vec![ActionFrequencyInputRow {
                bucket: 0,
                hand_label: "AA".to_string(),
                combo_count: 6,
                normalized_weight: 1.0,
                hu_probabilities: vec![0.2, 0.3, 0.5],
                mp_probabilities: vec![0.1, 0.4, 0.5],
                mp_row_visited: true,
            }],
            20,
            0.05,
            0.10,
        )
    }

    fn complete_action_frequency_report_with<F>(
        mut row: F,
        max_delta_tolerance: f64,
        total_l1_tolerance: f64,
    ) -> HuMpLazyActionFrequencyReport
    where
        F: FnMut(usize, &CanonicalPreflopBucketWeight) -> (Vec<f64>, Vec<f64>, bool),
    {
        let rows = canonical_preflop_bucket_weights(169)
            .expect("canonical weights")
            .iter()
            .enumerate()
            .map(|(idx, weight)| {
                let (hu_probabilities, mp_probabilities, mp_row_visited) = row(idx, weight);
                ActionFrequencyInputRow {
                    bucket: weight.bucket,
                    hand_label: weight.hand_label.clone(),
                    combo_count: weight.combo_count,
                    normalized_weight: weight.normalized_weight,
                    hu_probabilities,
                    mp_probabilities,
                    mp_row_visited,
                }
            })
            .collect::<Vec<_>>();

        aggregate_action_frequency_rows(
            "root",
            &[normalized("Fold", None), normalized("Raise", Some(6.0))],
            &[0, 1],
            &[0, 1],
            rows,
            20,
            max_delta_tolerance,
            total_l1_tolerance,
        )
    }

    fn clean_complete_action_frequency_report() -> HuMpLazyActionFrequencyReport {
        complete_action_frequency_report_with(
            |_idx, _weight| (vec![0.25, 0.75], vec![0.25, 0.75], true),
            0.01,
            0.02,
        )
    }

    fn passing_structural_checks() -> Vec<HuMpLazyStructuralCheck> {
        vec![HuMpLazyStructuralCheck {
            name: "sample".to_string(),
            passed: true,
            detail: "ok".to_string(),
        }]
    }

    fn sample_strategy_distance() -> HuMpLazyStrategyDistance {
        HuMpLazyStrategyDistance {
            path: "root".to_string(),
            bucket: 0,
            hu_strategy: "0.25|0.75".to_string(),
            mp_strategy: "0.25|0.75".to_string(),
            l1_distance: 0.0,
            max_abs_distance: 0.0,
            mp_row_visited: true,
        }
    }

    #[test]
    fn canonical_preflop_bucket_weights_sum_to_combo_count_distribution() {
        let weights = canonical_preflop_bucket_weights(169).expect("canonical weights");

        assert_eq!(weights.len(), 169);
        assert_eq!(
            weights
                .iter()
                .map(|weight| weight.combo_count)
                .sum::<usize>(),
            1326
        );
        assert!(
            (weights
                .iter()
                .map(|weight| weight.normalized_weight)
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-12
        );
        assert_eq!(weights[0].hand_label, "AA");
        assert_eq!(weights[0].combo_count, 6);
        assert_eq!(weights[13].hand_label, "AKs");
        assert_eq!(weights[13].combo_count, 4);
        assert_eq!(weights[91].hand_label, "AKo");
        assert_eq!(weights[91].combo_count, 12);

        let err = canonical_preflop_bucket_weights(168).expect_err("wrong bucket count errors");
        assert!(err.contains("requires 169 buckets"));
    }

    #[test]
    fn action_frequency_aggregation_uses_combo_weighted_raw_source_rows() {
        let report = aggregate_action_frequency_rows(
            "root",
            &[normalized("Fold", None), normalized("Raise", Some(6.0))],
            &[0, 1],
            &[0, 1],
            vec![
                ActionFrequencyInputRow {
                    bucket: 0,
                    hand_label: "AA".to_string(),
                    combo_count: 6,
                    normalized_weight: 0.25,
                    hu_probabilities: vec![0.25, 0.75],
                    mp_probabilities: vec![0.5, 0.5],
                    mp_row_visited: true,
                },
                ActionFrequencyInputRow {
                    bucket: 1,
                    hand_label: "AKo".to_string(),
                    combo_count: 18,
                    normalized_weight: 0.75,
                    hu_probabilities: vec![0.0, 1.0],
                    mp_probabilities: vec![0.2, 0.8],
                    mp_row_visited: true,
                },
            ],
            4,
            0.25,
            0.50,
        );

        assert_eq!(report.coverage.matched_rows, 2);
        assert_eq!(report.coverage.missing_mp_rows, 0);
        assert_eq!(report.coverage.uniform_fallback_rows, 0);
        assert_eq!(report.coverage.skipped_rows, 0);
        assert_eq!(report.distribution.total_weight, 24);
        assert!((report.distribution.normalized_weight_total - 1.0).abs() < 1e-12);
        assert!((report.coverage.included_normalized_weight_total - 1.0).abs() < 1e-12);

        let fold = &report.aggregate_actions[0];
        let raise = &report.aggregate_actions[1];
        assert!((fold.hu_frequency - 0.0625).abs() < 1e-12);
        assert!((raise.hu_frequency - 0.9375).abs() < 1e-12);
        assert!((fold.mp_frequency - 0.275).abs() < 1e-12);
        assert!((raise.mp_frequency - 0.725).abs() < 1e-12);
        assert!((report.totals.total_l1_delta - 0.425).abs() < 1e-12);
        assert!((report.totals.max_abs_delta - 0.2125).abs() < 1e-12);
        assert!((report.totals.weighted_row_l1 - 0.425).abs() < 1e-12);
        assert!(report.totals.passed_thresholds);
        assert!(!report.worst_spots.is_empty());
    }

    #[test]
    fn action_frequency_aggregation_surfaces_unmatched_mass_without_renormalizing() {
        let report = aggregate_action_frequency_rows(
            "root",
            &[normalized("Fold", None), normalized("Raise", Some(6.0))],
            &[0, 1],
            &[0, 1],
            vec![ActionFrequencyInputRow {
                bucket: 0,
                hand_label: "AA".to_string(),
                combo_count: 6,
                normalized_weight: 1.0,
                hu_probabilities: vec![0.2, 0.6, 0.2],
                mp_probabilities: vec![0.2, 0.8],
                mp_row_visited: true,
            }],
            4,
            1.0,
            1.0,
        );

        assert_eq!(report.coverage.matched_rows, 1);
        assert_eq!(report.coverage.hu_unmatched_rows, 1);
        assert!((report.coverage.hu_unmatched_probability_mass - 0.2).abs() < 1e-12);
        assert!((report.coverage.hu_unmatched_weighted_mass - 0.2).abs() < 1e-12);
        assert_eq!(report.coverage.mp_unmatched_rows, 0);
        assert!(!report.totals.passed_thresholds);

        let fold = &report.aggregate_actions[0];
        let raise = &report.aggregate_actions[1];
        assert!((fold.hu_frequency - 0.2).abs() < 1e-12);
        assert!((raise.hu_frequency - 0.6).abs() < 1e-12);
        assert!((fold.mp_frequency - 0.2).abs() < 1e-12);
        assert!((raise.mp_frequency - 0.8).abs() < 1e-12);

        let harness = HuMpLazyHarnessConfig {
            max_action_frequency_delta_tolerance: 1.0,
            total_action_frequency_l1_tolerance: 1.0,
            ..Default::default()
        };
        let final_report = finalize_report(
            &harness,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            HuMpLazyRowCoverage::default(),
            Vec::new(),
            Some(report),
            HuMpLazyRuntimeStats::default(),
        );
        assert!(final_report.reasons.iter().any(|reason| {
            reason.contains("HU comparable actions omitted weighted probability mass")
        }));
    }

    #[test]
    fn matching_preflight_is_no_go_without_action_frequency_evidence() {
        let (harness, hu, mp, _) = default_configs();

        let report = preflight_report_for_configs(&harness, &hu, &mp);

        assert_eq!(report.verdict, HuMpLazyVerdict::NoGo);
        assert!(report.action_frequency.is_none());
        assert!(report
            .reasons
            .iter()
            .any(|reason| { reason == "root action-frequency evidence was not produced" }));
        assert_eq!(
            report
                .reasons
                .iter()
                .filter(|reason| reason.contains("root action-frequency evidence was not produced"))
                .count(),
            1
        );
    }

    #[test]
    fn specific_missing_action_frequency_reason_is_not_duplicated() {
        let harness = HuMpLazyHarnessConfig::default();
        let report = finalize_report(
            &harness,
            vec![
                "root action-frequency evidence was not produced: 1 root action descriptor mismatch(es)"
                    .to_string(),
            ],
            Vec::new(),
            passing_structural_checks(),
            Vec::new(),
            HuMpLazyRowCoverage::default(),
            vec![sample_strategy_distance()],
            None,
            HuMpLazyRuntimeStats::default(),
        );

        assert_eq!(report.verdict, HuMpLazyVerdict::NoGo);
        assert_eq!(
            report
                .reasons
                .iter()
                .filter(|reason| reason.contains("root action-frequency evidence was not produced"))
                .count(),
            1
        );
    }

    #[test]
    fn clean_complete_action_frequency_can_go_with_accounting_warning() {
        let harness = HuMpLazyHarnessConfig::default();
        let report = finalize_report(
            &harness,
            Vec::new(),
            Vec::new(),
            passing_structural_checks(),
            Vec::new(),
            HuMpLazyRowCoverage {
                compared_rows: 169,
                hu_dense_rows: 169,
                mp_sparse_rows_total: 169,
                mp_root_rows_visited: 169,
                missing_mp_root_rows: 0,
            },
            vec![sample_strategy_distance()],
            Some(clean_complete_action_frequency_report()),
            HuMpLazyRuntimeStats::default(),
        );

        assert_eq!(report.verdict, HuMpLazyVerdict::Go);
        assert!(report.reasons.is_empty());
        assert!(!report.average_strategy_accounting_reconciled);
        assert!(report
            .warnings
            .iter()
            .any(|warning| { warning.contains("average-strategy accounting is not reconciled") }));
    }

    #[test]
    fn local_row_l1_warning_does_not_block_clean_action_frequency_go() {
        let harness = HuMpLazyHarnessConfig::default();
        let report = finalize_report(
            &harness,
            Vec::new(),
            vec![
                "max root strategy L1 0.500000 exceeds tolerance 0.050000".to_string(),
                "mean root strategy L1 0.100000 exceeds tolerance 0.010000".to_string(),
            ],
            passing_structural_checks(),
            Vec::new(),
            HuMpLazyRowCoverage {
                compared_rows: 169,
                hu_dense_rows: 169,
                mp_sparse_rows_total: 169,
                mp_root_rows_visited: 169,
                missing_mp_root_rows: 0,
            },
            vec![sample_strategy_distance()],
            Some(clean_complete_action_frequency_report()),
            HuMpLazyRuntimeStats::default(),
        );

        assert_eq!(report.verdict, HuMpLazyVerdict::Go);
        assert!(report.reasons.is_empty());
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("max root strategy L1")));
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("mean root strategy L1")));
    }

    #[test]
    fn action_frequency_tolerance_failure_remains_hard_no_go() {
        let harness = HuMpLazyHarnessConfig {
            max_action_frequency_delta_tolerance: 0.01,
            total_action_frequency_l1_tolerance: 0.02,
            ..Default::default()
        };
        let action_frequency = complete_action_frequency_report_with(
            |_idx, _weight| (vec![0.25, 0.75], vec![0.75, 0.25], true),
            harness.max_action_frequency_delta_tolerance,
            harness.total_action_frequency_l1_tolerance,
        );

        let report = finalize_report(
            &harness,
            Vec::new(),
            Vec::new(),
            passing_structural_checks(),
            Vec::new(),
            HuMpLazyRowCoverage {
                compared_rows: 169,
                hu_dense_rows: 169,
                mp_sparse_rows_total: 169,
                mp_root_rows_visited: 169,
                missing_mp_root_rows: 0,
            },
            vec![sample_strategy_distance()],
            Some(action_frequency),
            HuMpLazyRuntimeStats::default(),
        );

        assert_eq!(report.verdict, HuMpLazyVerdict::NoGo);
        assert!(report
            .reasons
            .iter()
            .any(|reason| { reason.contains("root action-frequency thresholds did not pass") }));
        assert!(report
            .reasons
            .iter()
            .any(|reason| { reason.contains("max root action-frequency delta") }));
        assert!(report
            .reasons
            .iter()
            .any(|reason| { reason.contains("total root action-frequency L1") }));
    }

    #[test]
    fn incomplete_action_frequency_coverage_remains_hard_no_go() {
        let harness = HuMpLazyHarnessConfig::default();
        let action_frequency = complete_action_frequency_report_with(
            |idx, _weight| (vec![0.25, 0.75], vec![0.25, 0.75], idx != 0),
            harness.max_action_frequency_delta_tolerance,
            harness.total_action_frequency_l1_tolerance,
        );

        let report = finalize_report(
            &harness,
            Vec::new(),
            Vec::new(),
            passing_structural_checks(),
            Vec::new(),
            HuMpLazyRowCoverage {
                compared_rows: 168,
                hu_dense_rows: 169,
                mp_sparse_rows_total: 168,
                mp_root_rows_visited: 168,
                missing_mp_root_rows: 1,
            },
            vec![sample_strategy_distance()],
            Some(action_frequency),
            HuMpLazyRuntimeStats::default(),
        );

        assert_eq!(report.verdict, HuMpLazyVerdict::NoGo);
        assert!(report
            .reasons
            .iter()
            .any(|reason| { reason.contains("root action-frequency coverage missing") }));
        assert!(report
            .reasons
            .iter()
            .any(|reason| { reason.contains("root action-frequency skipped") }));
        assert!(!report
            .reasons
            .iter()
            .any(|reason| { reason.contains("root action-frequency has no valid included rows") }));
    }

    #[test]
    fn action_frequency_aggregation_surfaces_invalid_zero_and_missing_rows() {
        let report = aggregate_action_frequency_rows(
            "root",
            &[normalized("Fold", None), normalized("Call", None)],
            &[0, 1],
            &[0, 1],
            vec![
                ActionFrequencyInputRow {
                    bucket: 0,
                    hand_label: "AA".to_string(),
                    combo_count: 6,
                    normalized_weight: 0.5,
                    hu_probabilities: vec![0.0, 0.0],
                    mp_probabilities: vec![0.5, 0.5],
                    mp_row_visited: true,
                },
                ActionFrequencyInputRow {
                    bucket: 1,
                    hand_label: "KK".to_string(),
                    combo_count: 6,
                    normalized_weight: 0.5,
                    hu_probabilities: vec![0.5, 0.5],
                    mp_probabilities: vec![f64::NAN, 0.5],
                    mp_row_visited: false,
                },
            ],
            10,
            0.01,
            0.02,
        );

        assert_eq!(report.coverage.matched_rows, 0);
        assert_eq!(report.coverage.skipped_rows, 2);
        assert_eq!(report.coverage.zero_sum_rows, 1);
        assert_eq!(report.coverage.invalid_rows, 1);
        assert_eq!(report.coverage.missing_mp_rows, 1);
        assert_eq!(report.coverage.uniform_fallback_rows, 0);
        assert!((report.coverage.skipped_normalized_weight_total - 1.0).abs() < 1e-12);
        assert!(!report.totals.passed_thresholds);
        assert!(report.worst_spots.is_empty());
    }

    #[test]
    fn hu_mp_lazy_blind_mismatch_is_no_go_with_reason() {
        let (harness, hu, mut mp, _) = default_configs();
        mp.game.blinds[1].amount = 4.0;

        let report = preflight_report_for_configs(&harness, &hu, &mp);

        assert_eq!(report.verdict, HuMpLazyVerdict::NoGo);
        assert!(report.reasons.iter().any(|reason| reason.contains("blind")));
    }

    #[test]
    fn hu_mp_lazy_action_schema_mismatch_is_no_go_with_reason() {
        let (harness, hu, mut mp, _) = default_configs();
        mp.game.stack_depth = 10.0;
        mp.action_abstraction.preflop.lead = vec![yaml_string("2bb".to_string())];

        let report = preflight_report_for_configs(&harness, &hu, &mp);

        assert_eq!(report.verdict, HuMpLazyVerdict::NoGo);
        assert!(!report.root_action_schema_mismatches.is_empty());
        assert!(report
            .reasons
            .iter()
            .any(|reason| reason.contains("action schema")));
    }

    #[test]
    fn hu_mp_lazy_malformed_action_schema_is_no_go_instead_of_panic() {
        let (harness, mut hu, mp, _) = default_configs();
        hu.action_abstraction.preflop = vec![vec!["not-a-size".to_string()]];

        let report = preflight_report_for_configs(&harness, &hu, &mp);

        assert_eq!(report.verdict, HuMpLazyVerdict::NoGo);
        assert_eq!(report.root_action_schema_mismatches.len(), 1);
        assert!(report.root_action_schema_mismatches[0]
            .reason
            .contains("preflop size"));
        assert!(report
            .reasons
            .iter()
            .any(|reason| reason.contains("action schema")));
    }

    #[test]
    fn hu_mp_lazy_default_open_chips_normalize_to_same_bb_label() {
        let (harness, hu, mp, _) = default_configs();

        assert_eq!(harness.big_blind_chips, 2.0);
        assert_eq!(harness.preflop_open_to_chips, vec![6.0]);
        assert_eq!(hu.action_abstraction.preflop, vec![vec!["3bb".to_string()]]);
        assert_eq!(
            mp.action_abstraction.preflop.lead,
            vec![yaml_string("3bb".to_string())]
        );
        assert_eq!(
            mp.action_abstraction.preflop.raise,
            vec![vec![yaml_string("3bb".to_string())]]
        );
    }

    #[test]
    fn hu_mp_lazy_root_schema_matches_for_default_2p_fixture() {
        let (_harness, hu, mp, _) = default_configs();

        let mismatches = root_action_schema_mismatches(&hu, &mp);

        assert!(
            mismatches.is_empty(),
            "default fixture should match root schema, got {mismatches:?}"
        );

        let hu_tree = poker_solver_core::blueprint_v2::game_tree::GameTree::build_with_options(
            hu.game.stack_depth,
            hu.game.small_blind,
            hu.game.big_blind,
            &hu.action_abstraction.preflop,
            &hu.action_abstraction.flop,
            &hu.action_abstraction.turn,
            &hu.action_abstraction.river,
            hu.game.allow_preflop_limp,
        );
        let hu_actions = match &hu_tree.nodes[hu_tree.root as usize] {
            HuGameNode::Decision { actions, .. } => comparable_hu_root_actions(actions, &hu)
                .into_iter()
                .map(|(_, action)| action)
                .collect::<Vec<_>>(),
            other => panic!("HU root should be a decision node, got {other:?}"),
        };
        let mp_game = poker_solver_core::blueprint_mp::lazy_mccfr::LazyMpGame::new(
            &mp.game,
            &mp.action_abstraction,
        );
        let mp_actions = LazyResolvedSpot::root(&mp_game)
            .actions(&mp_game)
            .iter()
            .map(normalize_mp_root_action)
            .collect::<Vec<_>>();

        assert!(hu_actions
            .iter()
            .any(|action| action.kind == "Raise" && action.amount_chips == Some(6.0)));
        assert!(mp_actions
            .iter()
            .any(|action| action.kind == "Raise" && action.amount_chips == Some(6.0)));
    }

    #[test]
    fn hu_mp_lazy_tiny_end_to_end_smoke_produces_report_verdict_reasons_and_stats() {
        let harness = HuMpLazyHarnessConfig {
            iterations: 1,
            ..Default::default()
        };

        let report = run_hu_mp_lazy_harness(&harness).expect("harness runs");

        assert_eq!(report.verdict, HuMpLazyVerdict::NoGo);
        assert!(!report.reasons.is_empty());
        assert!(report.runtime_stats.hu_iterations_completed >= 1);
        assert!(report.runtime_stats.mp_meta_iterations_completed >= 1);
        assert!(report.row_coverage.compared_rows > 0);
        assert!(report.max_l1_distance.is_some());
        let action_frequency = report
            .action_frequency
            .as_ref()
            .expect("action frequency report");
        assert!(!action_frequency.aggregate_actions.is_empty());
    }
}
