//! Preflop baseline strategy validation for `blueprint_v2`.
//!
//! This module compares learned average strategies against a small external
//! preflop baseline without projecting dense strategy exports or invoking a
//! postflop/range solver.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use thiserror::Error;

use super::Street;
use super::game_tree::{GameNode, GameTree, TreeAction};
use super::storage::BlueprintCfrStorage;
use crate::hands::CanonicalHand;

const ACTION_EPSILON: f64 = 0.01;
const MASS_EPSILON: f64 = 1.0e-12;
const EXPECTED_PREFLOP_BUCKETS: u16 = 169;
const EXPECTED_STARTING_STACK: f64 = 40.0;
const EXPECTED_BIG_BLIND: f64 = 2.0;
const EXPECTED_BASELINE_STACK_DEPTH_BB: f64 = 20.0;
const EXPECTED_OPENING_SIZE: &str = "25X";
const PREFLIGHT_SPOT_KEY: &str = "<preflight>";

const EXPECTED_SPOT_SCHEMAS: &[(&str, &[&str], &[&str])] = &[
    ("root", &[], &["F", "R2.5", "RAI"]),
    ("SB:r2.5", &["R2.5"], &["F", "C", "R5", "RAI"]),
    ("SB:rai", &["RAI"], &["F", "C"]),
    ("SB:r2.5, BB:r5", &["R2.5", "R5"], &["F", "C", "RAI"]),
    ("SB:r2.5, BB:rai", &["R2.5", "RAI"], &["F", "C"]),
    (
        "SB:r2.5, BB:r5, SB:rai",
        &["R2.5", "R5", "RAI"],
        &["F", "C"],
    ),
];

/// Parsed baseline JSON document.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct BaselineDocument {
    #[serde(default)]
    pub schema_version: Option<u32>,
    #[serde(default)]
    pub site: Option<String>,
    #[serde(default)]
    pub game: BaselineGameMetadata,
    #[serde(default)]
    pub source: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub actions: BTreeMap<String, BaselineActionMetadata>,
    #[serde(default)]
    pub spots: BTreeMap<String, BaselineSpot>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Baseline game metadata.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct BaselineGameMetadata {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default, rename = "rawGametype")]
    pub raw_gametype: Option<String>,
    #[serde(default, rename = "format")]
    pub game_format: Option<String>,
    #[serde(default)]
    pub players: Option<u8>,
    #[serde(default, rename = "tableType")]
    pub table_type: Option<String>,
    #[serde(default, rename = "stackDepthBb")]
    pub stack_depth_bb: Option<f64>,
    #[serde(default, rename = "evModel")]
    pub ev_model: Option<String>,
    #[serde(default, rename = "openingSize")]
    pub opening_size: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Baseline action metadata from the document-level `actions` map.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct BaselineActionMetadata {
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default, rename = "amountBb")]
    pub amount_bb: Option<f64>,
    #[serde(default, rename = "type")]
    pub action_type: Option<String>,
    #[serde(default, rename = "isTerminal")]
    pub is_terminal: Option<bool>,
    #[serde(default, rename = "nextPosition")]
    pub next_position: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// One preflop baseline spot.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct BaselineSpot {
    #[serde(default)]
    pub spot_key: String,
    #[serde(default)]
    pub path: Vec<String>,
    #[serde(default)]
    pub street: String,
    #[serde(default)]
    pub position_to_act: String,
    #[serde(default)]
    pub pot_bb: Option<f64>,
    #[serde(default)]
    pub actions: Vec<String>,
    #[serde(default)]
    pub action_summary: BTreeMap<String, f64>,
    #[serde(default)]
    pub strategy: BTreeMap<String, BaselineComboStrategy>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Per-canonical-hand baseline strategy row.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct BaselineComboStrategy {
    #[serde(default)]
    pub ev: Option<f64>,
    #[serde(default)]
    pub action_frequencies: BTreeMap<String, f64>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

/// Validation options.
#[derive(Debug, Clone, Copy)]
pub struct BaselineValidationConfig {
    /// Big blind in repo chip units. The target Phase 2 config uses `2.0`.
    pub big_blind: f64,
    /// Number of rows/spots to retain in worst-first summaries.
    pub top_n: usize,
}

impl Default for BaselineValidationConfig {
    fn default() -> Self {
        Self {
            big_blind: 2.0,
            top_n: 5,
        }
    }
}

/// Thin read-only provider boundary for learned average strategies.
pub trait BaselineStrategyProvider {
    fn preflop_bucket_count(&self) -> u16;
    fn average_strategy(&self, node_idx: u32, bucket: u16) -> Vec<f64>;
}

impl<T> BaselineStrategyProvider for T
where
    T: BlueprintCfrStorage + ?Sized,
{
    fn preflop_bucket_count(&self) -> u16 {
        self.bucket_counts()[Street::Preflop as usize]
    }

    fn average_strategy(&self, node_idx: u32, bucket: u16) -> Vec<f64> {
        BlueprintCfrStorage::average_strategy(self, node_idx, bucket)
    }
}

/// Full validation report suitable for later TUI rendering.
#[derive(Debug, Clone, Default)]
pub struct BaselineValidationReport {
    pub aggregate: BaselineAggregateMetrics,
    pub spots: Vec<SpotValidationMetrics>,
    pub worst_spots: Vec<SpotValidationMetrics>,
    pub worst_combo_rows: Vec<ComboMismatchRow>,
    pub precondition_failures: Vec<PreconditionFailure>,
    pub unsupported_spots: Vec<UnsupportedSpot>,
    pub unsupported_actions: Vec<UnsupportedAction>,
    pub invalid_hand_rows: Vec<InvalidHandRow>,
}

/// Aggregate convergence metrics across all scored rows.
#[derive(Debug, Clone, Default)]
pub struct BaselineAggregateMetrics {
    pub spots_total: usize,
    pub spots_scored: usize,
    pub precondition_failures: usize,
    pub unsupported_spots: usize,
    pub unsupported_actions: usize,
    pub combo_rows_scored: usize,
    pub combo_rows_skipped_zero_mass: usize,
    pub combo_rows_invalid_hand: usize,
    pub total_combo_weight: f64,
    pub weighted_total_variation_sum: f64,
    pub mean_total_variation: f64,
    pub weighted_unmapped_candidate_mass_sum: f64,
    pub mean_unmapped_candidate_mass: f64,
    pub worst_spot_total_variation: f64,
}

/// Per-spot strategy validation metrics.
#[derive(Debug, Clone, Default)]
pub struct SpotValidationMetrics {
    pub spot_key: String,
    pub node_idx: u32,
    pub position_to_act: String,
    pub baseline_actions: Vec<String>,
    pub candidate_actions: Vec<String>,
    pub schema_supported: bool,
    pub combo_rows_scored: usize,
    pub combo_rows_skipped_zero_mass: usize,
    pub combo_rows_invalid_hand: usize,
    pub total_combo_weight: f64,
    pub weighted_total_variation_sum: f64,
    pub mean_total_variation: f64,
    pub weighted_unmapped_candidate_mass_sum: f64,
    pub mean_unmapped_candidate_mass: f64,
    pub unsupported_baseline_actions: Vec<String>,
    pub unmapped_candidate_actions: Vec<String>,
}

/// One worst canonical-hand mismatch row.
#[derive(Debug, Clone, Default)]
pub struct ComboMismatchRow {
    pub spot_key: String,
    pub hand: String,
    pub combo_weight: f64,
    pub total_variation: f64,
    pub unmapped_candidate_mass: f64,
    pub action_frequencies: Vec<ActionFrequencyComparison>,
}

/// Baseline-vs-learned frequency for one baseline action label.
#[derive(Debug, Clone, Default)]
pub struct ActionFrequencyComparison {
    pub label: String,
    pub baseline: f64,
    pub learned: f64,
}

/// Spot that could not be resolved or scored.
#[derive(Debug, Clone, Default)]
pub struct UnsupportedSpot {
    pub spot_key: String,
    pub reason: String,
}

/// Action schema mismatch observed at a resolved spot.
#[derive(Debug, Clone, Default)]
pub struct UnsupportedAction {
    pub spot_key: String,
    pub label: String,
    pub reason: String,
}

/// Global validation precondition failure. Any entry here means scoring was
/// refused for the whole baseline/tree/provider tuple.
#[derive(Debug, Clone, Default)]
pub struct PreconditionFailure {
    pub field: String,
    pub expected: String,
    pub actual: String,
    pub reason: String,
}

/// Baseline hand row with an unparsable canonical hand label.
#[derive(Debug, Clone, Default)]
pub struct InvalidHandRow {
    pub spot_key: String,
    pub hand_label: String,
    pub reason: String,
}

/// Mapping from baseline action labels to candidate tree action indices.
#[derive(Debug, Clone)]
pub struct BaselineActionMapping {
    pub node_idx: u32,
    pub entries: Vec<MappedBaselineAction>,
    pub unsupported_baseline_actions: Vec<String>,
    pub unmapped_candidate_actions: Vec<String>,
}

impl BaselineActionMapping {
    #[must_use]
    pub fn is_supported(&self) -> bool {
        self.unsupported_baseline_actions.is_empty() && self.unmapped_candidate_actions.is_empty()
    }

    fn mapped_candidate_indices(&self) -> BTreeSet<usize> {
        self.entries.iter().map(|entry| entry.action_idx).collect()
    }

    fn action_idx_for_label(&self, label: &str) -> Option<usize> {
        self.entries
            .iter()
            .find(|entry| entry.label == label)
            .map(|entry| entry.action_idx)
    }
}

/// One mapped action label.
#[derive(Debug, Clone)]
pub struct MappedBaselineAction {
    pub label: String,
    pub action_idx: usize,
    pub action: TreeAction,
}

#[derive(Debug, Error)]
pub enum BaselineValidationError {
    #[error("baseline JSON parse failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("spot {spot_key} resolved to non-decision node {node_idx}")]
    NonDecisionSpot { spot_key: String, node_idx: u32 },
    #[error("spot {spot_key} path action {label} is unsupported at node {node_idx}")]
    UnsupportedPathAction {
        spot_key: String,
        node_idx: u32,
        label: String,
    },
}

/// Parse a baseline JSON document.
///
/// # Errors
///
/// Returns a serde JSON error if the document is not valid for the baseline
/// shape.
pub fn parse_baseline_json(input: &str) -> Result<BaselineDocument, BaselineValidationError> {
    Ok(serde_json::from_str(input)?)
}

/// Validate a baseline against a learned strategy provider.
#[must_use]
pub fn validate_baseline<P>(
    baseline: &BaselineDocument,
    tree: &GameTree,
    provider: &P,
    config: BaselineValidationConfig,
) -> BaselineValidationReport
where
    P: BaselineStrategyProvider + ?Sized,
{
    let mut report = BaselineValidationReport::default();
    report.aggregate.spots_total = baseline.spots.len();

    report.precondition_failures = validate_preconditions(baseline, tree, provider, config);
    report.aggregate.precondition_failures = report.precondition_failures.len();
    if !report.precondition_failures.is_empty() {
        report.unsupported_spots.push(UnsupportedSpot {
            spot_key: PREFLIGHT_SPOT_KEY.to_string(),
            reason: format!(
                "baseline validation preconditions failed: {}",
                report.precondition_failures.len()
            ),
        });
        report.aggregate.unsupported_spots = report.unsupported_spots.len();
        return report;
    }

    for (key, spot) in &baseline.spots {
        let spot_key = normalized_spot_key(key, spot);
        let node_idx = match resolve_spot_path(tree, spot, config.big_blind) {
            Ok(node_idx) => node_idx,
            Err(err) => {
                report.unsupported_spots.push(UnsupportedSpot {
                    spot_key,
                    reason: err.to_string(),
                });
                continue;
            }
        };

        let mapping = match build_action_mapping(tree, node_idx, &spot.actions, config.big_blind) {
            Ok(mapping) => mapping,
            Err(err) => {
                report.unsupported_spots.push(UnsupportedSpot {
                    spot_key,
                    reason: err.to_string(),
                });
                continue;
            }
        };

        for label in &mapping.unsupported_baseline_actions {
            report.unsupported_actions.push(UnsupportedAction {
                spot_key: spot_key.clone(),
                label: label.clone(),
                reason: "baseline action has no candidate tree action".to_string(),
            });
        }
        for label in &mapping.unmapped_candidate_actions {
            report.unsupported_actions.push(UnsupportedAction {
                spot_key: spot_key.clone(),
                label: label.clone(),
                reason: "candidate tree action is absent from baseline action schema".to_string(),
            });
        }

        let (metrics, mut rows, mut invalid_hand_rows) =
            score_spot(tree, provider, &spot_key, spot, &mapping);
        report.aggregate.spots_scored += 1;
        report.aggregate.combo_rows_scored += metrics.combo_rows_scored;
        report.aggregate.combo_rows_skipped_zero_mass += metrics.combo_rows_skipped_zero_mass;
        report.aggregate.combo_rows_invalid_hand += metrics.combo_rows_invalid_hand;
        report.aggregate.total_combo_weight += metrics.total_combo_weight;
        report.aggregate.weighted_total_variation_sum += metrics.weighted_total_variation_sum;
        report.aggregate.weighted_unmapped_candidate_mass_sum +=
            metrics.weighted_unmapped_candidate_mass_sum;
        report.aggregate.worst_spot_total_variation = report
            .aggregate
            .worst_spot_total_variation
            .max(metrics.mean_total_variation);
        report.spots.push(metrics);
        report.worst_combo_rows.append(&mut rows);
        report.invalid_hand_rows.append(&mut invalid_hand_rows);
    }

    report.aggregate.unsupported_spots = report.unsupported_spots.len();
    report.aggregate.unsupported_actions = report.unsupported_actions.len();
    if report.aggregate.total_combo_weight > 0.0 {
        report.aggregate.mean_total_variation =
            report.aggregate.weighted_total_variation_sum / report.aggregate.total_combo_weight;
        report.aggregate.mean_unmapped_candidate_mass =
            report.aggregate.weighted_unmapped_candidate_mass_sum
                / report.aggregate.total_combo_weight;
    }

    report.worst_spots = report.spots.clone();
    report.worst_spots.sort_by(|a, b| {
        b.mean_total_variation
            .total_cmp(&a.mean_total_variation)
            .then_with(|| a.spot_key.cmp(&b.spot_key))
    });
    report.worst_spots.truncate(config.top_n);

    report.worst_combo_rows.sort_by(|a, b| {
        b.total_variation
            .total_cmp(&a.total_variation)
            .then_with(|| a.spot_key.cmp(&b.spot_key))
            .then_with(|| a.hand.cmp(&b.hand))
    });
    report.worst_combo_rows.truncate(config.top_n);

    report
}

fn validate_preconditions<P>(
    baseline: &BaselineDocument,
    tree: &GameTree,
    provider: &P,
    config: BaselineValidationConfig,
) -> Vec<PreconditionFailure>
where
    P: BaselineStrategyProvider + ?Sized,
{
    let mut failures = Vec::new();

    push_u16_precondition(
        &mut failures,
        "provider.preflop_bucket_count",
        EXPECTED_PREFLOP_BUCKETS,
        provider.preflop_bucket_count(),
        "canonical hand index is used directly as the preflop bucket",
    );
    push_f64_precondition(
        &mut failures,
        "tree.starting_stack",
        EXPECTED_STARTING_STACK,
        tree.starting_stack,
        "baseline all-in label RAI is defined for the 40-chip stack tree",
    );
    push_f64_precondition(
        &mut failures,
        "validation.big_blind",
        EXPECTED_BIG_BLIND,
        config.big_blind,
        "baseline raise labels are mapped from big-blind units to chip amounts",
    );

    match baseline.game.stack_depth_bb {
        Some(stack_depth_bb) if approx_eq(stack_depth_bb, EXPECTED_BASELINE_STACK_DEPTH_BB) => {}
        Some(stack_depth_bb) => failures.push(PreconditionFailure {
            field: "baseline.game.stackDepthBb".to_string(),
            expected: format!("{EXPECTED_BASELINE_STACK_DEPTH_BB:.1}"),
            actual: format!("{stack_depth_bb:.1}"),
            reason: "this validator is pinned to the supplied 20bb heads-up baseline".to_string(),
        }),
        None => failures.push(PreconditionFailure {
            field: "baseline.game.stackDepthBb".to_string(),
            expected: format!("{EXPECTED_BASELINE_STACK_DEPTH_BB:.1}"),
            actual: "missing".to_string(),
            reason: "this validator is pinned to the supplied 20bb heads-up baseline".to_string(),
        }),
    }

    let opening_size = baseline
        .game
        .opening_size
        .as_deref()
        .map(normalize_opening_size);
    if opening_size.as_deref() != Some(EXPECTED_OPENING_SIZE) {
        failures.push(PreconditionFailure {
            field: "baseline.game.openingSize".to_string(),
            expected: EXPECTED_OPENING_SIZE.to_string(),
            actual: opening_size.unwrap_or_else(|| "missing".to_string()),
            reason: "baseline action labels assume the 2.5x opening-size solution".to_string(),
        });
    }

    validate_document_action_metadata(baseline, &mut failures);
    validate_expected_spot_schemas(baseline, tree, config.big_blind, &mut failures);

    failures
}

fn validate_document_action_metadata(
    baseline: &BaselineDocument,
    failures: &mut Vec<PreconditionFailure>,
) {
    let expected = [
        ("F", Some(0.0), Some("FOLD")),
        ("R2.5", Some(2.5), Some("RAISE")),
        ("R5", Some(5.0), Some("RAISE")),
        ("RAI", Some(20.0), Some("RAISE")),
        ("C", Some(20.0), Some("CALL")),
    ];

    for (label, expected_amount, expected_type) in expected {
        let Some(metadata) = baseline.actions.get(label) else {
            failures.push(PreconditionFailure {
                field: format!("baseline.actions.{label}"),
                expected: "present".to_string(),
                actual: "missing".to_string(),
                reason: "expected document-level action metadata is absent".to_string(),
            });
            continue;
        };

        if let Some(expected_amount) = expected_amount {
            match metadata.amount_bb {
                Some(amount) if approx_eq(amount, expected_amount) => {}
                Some(amount) => failures.push(PreconditionFailure {
                    field: format!("baseline.actions.{label}.amountBb"),
                    expected: format!("{expected_amount:.1}"),
                    actual: format!("{amount:.1}"),
                    reason: "baseline action amount does not match the pinned HU 20bb schema"
                        .to_string(),
                }),
                None => failures.push(PreconditionFailure {
                    field: format!("baseline.actions.{label}.amountBb"),
                    expected: format!("{expected_amount:.1}"),
                    actual: "missing".to_string(),
                    reason: "baseline action amount does not match the pinned HU 20bb schema"
                        .to_string(),
                }),
            }
        }

        if let Some(expected_type) = expected_type {
            let actual_type = metadata.action_type.as_deref().map(str::to_ascii_uppercase);
            if actual_type.as_deref() != Some(expected_type) {
                failures.push(PreconditionFailure {
                    field: format!("baseline.actions.{label}.type"),
                    expected: expected_type.to_string(),
                    actual: actual_type.unwrap_or_else(|| "missing".to_string()),
                    reason: "baseline action type does not match the pinned HU 20bb schema"
                        .to_string(),
                });
            }
        }
    }
}

fn validate_expected_spot_schemas(
    baseline: &BaselineDocument,
    tree: &GameTree,
    big_blind: f64,
    failures: &mut Vec<PreconditionFailure>,
) {
    for (spot_key, expected_path, expected_actions) in EXPECTED_SPOT_SCHEMAS {
        let Some(spot) = baseline.spots.get(*spot_key) else {
            failures.push(PreconditionFailure {
                field: format!("baseline.spots.{spot_key}"),
                expected: "present".to_string(),
                actual: "missing".to_string(),
                reason:
                    "the supplied Phase 2 baseline should contain exactly the known preflop spots"
                        .to_string(),
            });
            continue;
        };

        let actual_path: Vec<String> = spot
            .path
            .iter()
            .map(|label| normalize_action_label(label))
            .collect();
        let expected_path: Vec<String> = expected_path
            .iter()
            .map(|label| normalize_action_label(label))
            .collect();
        if actual_path != expected_path {
            failures.push(PreconditionFailure {
                field: format!("baseline.spots.{spot_key}.path"),
                expected: format!("{expected_path:?}"),
                actual: format!("{actual_path:?}"),
                reason: "baseline spot path does not match the pinned HU 20bb schema".to_string(),
            });
            continue;
        }

        let actual_actions: Vec<String> = spot
            .actions
            .iter()
            .map(|label| normalize_action_label(label))
            .collect();
        let expected_actions: Vec<String> = expected_actions
            .iter()
            .map(|label| normalize_action_label(label))
            .collect();
        if actual_actions != expected_actions {
            failures.push(PreconditionFailure {
                field: format!("baseline.spots.{spot_key}.actions"),
                expected: format!("{expected_actions:?}"),
                actual: format!("{actual_actions:?}"),
                reason: "baseline spot action schema does not match the pinned HU 20bb schema"
                    .to_string(),
            });
            continue;
        }

        let node_idx = match resolve_spot_path(tree, spot, big_blind) {
            Ok(node_idx) => node_idx,
            Err(err) => {
                failures.push(PreconditionFailure {
                    field: format!("tree.spots.{spot_key}.path"),
                    expected: "resolvable preflop decision node".to_string(),
                    actual: err.to_string(),
                    reason: "target tree does not support the pinned HU 20bb baseline path"
                        .to_string(),
                });
                continue;
            }
        };

        let mapping = match build_action_mapping(tree, node_idx, &spot.actions, big_blind) {
            Ok(mapping) => mapping,
            Err(err) => {
                failures.push(PreconditionFailure {
                    field: format!("tree.spots.{spot_key}.actions"),
                    expected: "supported action schema".to_string(),
                    actual: err.to_string(),
                    reason: "target tree does not support the pinned HU 20bb baseline actions"
                        .to_string(),
                });
                continue;
            }
        };

        if !mapping.is_supported() {
            failures.push(PreconditionFailure {
                field: format!("tree.spots.{spot_key}.actions"),
                expected: format!("{actual_actions:?}"),
                actual: format!(
                    "unsupported_baseline={:?}, unmapped_candidate={:?}",
                    mapping.unsupported_baseline_actions, mapping.unmapped_candidate_actions
                ),
                reason: "target tree action schema differs from the pinned HU 20bb baseline"
                    .to_string(),
            });
        }
    }
}

fn push_u16_precondition(
    failures: &mut Vec<PreconditionFailure>,
    field: &str,
    expected: u16,
    actual: u16,
    reason: &str,
) {
    if actual != expected {
        failures.push(PreconditionFailure {
            field: field.to_string(),
            expected: expected.to_string(),
            actual: actual.to_string(),
            reason: reason.to_string(),
        });
    }
}

fn push_f64_precondition(
    failures: &mut Vec<PreconditionFailure>,
    field: &str,
    expected: f64,
    actual: f64,
    reason: &str,
) {
    if !approx_eq(actual, expected) {
        failures.push(PreconditionFailure {
            field: field.to_string(),
            expected: format!("{expected:.1}"),
            actual: format!("{actual:.1}"),
            reason: reason.to_string(),
        });
    }
}

/// Resolve a baseline spot path to a `GameTree` decision node.
///
/// # Errors
///
/// Returns an error if any path action cannot be mapped, or if the resolved
/// node is not a decision node.
pub fn resolve_spot_path(
    tree: &GameTree,
    spot: &BaselineSpot,
    big_blind: f64,
) -> Result<u32, BaselineValidationError> {
    let spot_key = if spot.spot_key.is_empty() {
        "root".to_string()
    } else {
        spot.spot_key.clone()
    };
    let mut node_idx = tree.root;

    for label in &spot.path {
        let action_idx =
            map_single_baseline_action(tree, node_idx, label, big_blind).ok_or_else(|| {
                BaselineValidationError::UnsupportedPathAction {
                    spot_key: spot_key.clone(),
                    node_idx,
                    label: label.clone(),
                }
            })?;

        let GameNode::Decision { children, .. } = &tree.nodes[node_idx as usize] else {
            return Err(BaselineValidationError::NonDecisionSpot { spot_key, node_idx });
        };
        node_idx = children[action_idx];
    }

    match &tree.nodes[node_idx as usize] {
        GameNode::Decision {
            street: Street::Preflop,
            ..
        } => Ok(node_idx),
        GameNode::Decision { .. } => Ok(node_idx),
        _ => Err(BaselineValidationError::NonDecisionSpot { spot_key, node_idx }),
    }
}

/// Build the context-aware baseline-to-tree action mapping for a node.
///
/// # Errors
///
/// Returns an error if `node_idx` does not identify a decision node.
pub fn build_action_mapping(
    tree: &GameTree,
    node_idx: u32,
    baseline_actions: &[String],
    big_blind: f64,
) -> Result<BaselineActionMapping, BaselineValidationError> {
    let GameNode::Decision { actions, .. } = &tree.nodes[node_idx as usize] else {
        return Err(BaselineValidationError::NonDecisionSpot {
            spot_key: format!("node:{node_idx}"),
            node_idx,
        });
    };

    let mut entries = Vec::new();
    let mut unsupported_baseline_actions = Vec::new();

    for label in baseline_actions {
        if let Some(action_idx) = action_idx_for_label(actions, label, big_blind) {
            entries.push(MappedBaselineAction {
                label: normalize_action_label(label),
                action_idx,
                action: actions[action_idx],
            });
        } else {
            unsupported_baseline_actions.push(normalize_action_label(label));
        }
    }

    let mapped_indices: BTreeSet<usize> = entries.iter().map(|entry| entry.action_idx).collect();
    let unmapped_candidate_actions = actions
        .iter()
        .enumerate()
        .filter(|(idx, _)| !mapped_indices.contains(idx))
        .map(|(_, action)| format_tree_action(*action))
        .collect();

    Ok(BaselineActionMapping {
        node_idx,
        entries,
        unsupported_baseline_actions,
        unmapped_candidate_actions,
    })
}

fn score_spot<P>(
    tree: &GameTree,
    provider: &P,
    spot_key: &str,
    spot: &BaselineSpot,
    mapping: &BaselineActionMapping,
) -> (
    SpotValidationMetrics,
    Vec<ComboMismatchRow>,
    Vec<InvalidHandRow>,
)
where
    P: BaselineStrategyProvider + ?Sized,
{
    let candidate_actions = match &tree.nodes[mapping.node_idx as usize] {
        GameNode::Decision { actions, .. } => actions
            .iter()
            .map(|action| format_tree_action(*action))
            .collect(),
        _ => Vec::new(),
    };

    let mut metrics = SpotValidationMetrics {
        spot_key: spot_key.to_string(),
        node_idx: mapping.node_idx,
        position_to_act: spot.position_to_act.clone(),
        baseline_actions: spot
            .actions
            .iter()
            .map(|label| normalize_action_label(label))
            .collect(),
        candidate_actions,
        schema_supported: mapping.is_supported(),
        unsupported_baseline_actions: mapping.unsupported_baseline_actions.clone(),
        unmapped_candidate_actions: mapping.unmapped_candidate_actions.clone(),
        ..SpotValidationMetrics::default()
    };

    let mut rows = Vec::new();
    let mut invalid_hand_rows = Vec::new();
    let mapped_candidate_indices = mapping.mapped_candidate_indices();
    let unmapped_candidate_indices: Vec<usize> = candidate_action_indices(tree, mapping.node_idx)
        .into_iter()
        .filter(|idx| !mapped_candidate_indices.contains(idx))
        .collect();

    for (hand_label, baseline_row) in &spot.strategy {
        let hand = match CanonicalHand::parse(hand_label) {
            Ok(hand) => hand,
            Err(err) => {
                metrics.combo_rows_invalid_hand += 1;
                invalid_hand_rows.push(InvalidHandRow {
                    spot_key: spot_key.to_string(),
                    hand_label: hand_label.clone(),
                    reason: err.to_string(),
                });
                continue;
            }
        };

        let baseline_mass = baseline_mass_for_actions(baseline_row, &spot.actions);
        if baseline_mass <= MASS_EPSILON {
            metrics.combo_rows_skipped_zero_mass += 1;
            continue;
        }

        let bucket = hand.index() as u16;
        let combo_weight = f64::from(hand.num_combos());
        let learned = provider.average_strategy(mapping.node_idx, bucket);
        let unmapped_candidate_mass =
            sum_indices(&learned, unmapped_candidate_indices.iter().copied());

        let mut absolute_distance_sum = 0.0;
        let mut action_frequencies = Vec::with_capacity(spot.actions.len());

        for label in &spot.actions {
            let normalized = normalize_action_label(label);
            let baseline_frequency = baseline_row
                .action_frequencies
                .get(label)
                .or_else(|| baseline_row.action_frequencies.get(&normalized))
                .copied()
                .unwrap_or(0.0)
                / baseline_mass;
            let learned_frequency = mapping
                .action_idx_for_label(&normalized)
                .and_then(|idx| learned.get(idx).copied())
                .unwrap_or(0.0);
            absolute_distance_sum += (baseline_frequency - learned_frequency).abs();
            action_frequencies.push(ActionFrequencyComparison {
                label: normalized,
                baseline: baseline_frequency,
                learned: learned_frequency,
            });
        }

        let total_variation = 0.5 * (absolute_distance_sum + unmapped_candidate_mass);
        metrics.combo_rows_scored += 1;
        metrics.total_combo_weight += combo_weight;
        metrics.weighted_total_variation_sum += total_variation * combo_weight;
        metrics.weighted_unmapped_candidate_mass_sum += unmapped_candidate_mass * combo_weight;

        rows.push(ComboMismatchRow {
            spot_key: spot_key.to_string(),
            hand: hand.to_string(),
            combo_weight,
            total_variation,
            unmapped_candidate_mass,
            action_frequencies,
        });
    }

    if metrics.total_combo_weight > 0.0 {
        metrics.mean_total_variation =
            metrics.weighted_total_variation_sum / metrics.total_combo_weight;
        metrics.mean_unmapped_candidate_mass =
            metrics.weighted_unmapped_candidate_mass_sum / metrics.total_combo_weight;
    }

    (metrics, rows, invalid_hand_rows)
}

fn baseline_mass_for_actions(row: &BaselineComboStrategy, baseline_actions: &[String]) -> f64 {
    baseline_actions
        .iter()
        .map(|label| {
            let normalized = normalize_action_label(label);
            row.action_frequencies
                .get(label)
                .or_else(|| row.action_frequencies.get(&normalized))
                .copied()
                .unwrap_or(0.0)
        })
        .sum()
}

fn candidate_action_indices(tree: &GameTree, node_idx: u32) -> Vec<usize> {
    match &tree.nodes[node_idx as usize] {
        GameNode::Decision { actions, .. } => (0..actions.len()).collect(),
        _ => Vec::new(),
    }
}

fn sum_indices(values: &[f64], indices: impl Iterator<Item = usize>) -> f64 {
    indices.filter_map(|idx| values.get(idx)).sum()
}

fn map_single_baseline_action(
    tree: &GameTree,
    node_idx: u32,
    label: &str,
    big_blind: f64,
) -> Option<usize> {
    let GameNode::Decision { actions, .. } = &tree.nodes[node_idx as usize] else {
        return None;
    };
    action_idx_for_label(actions, label, big_blind)
}

fn action_idx_for_label(actions: &[TreeAction], label: &str, big_blind: f64) -> Option<usize> {
    let label = normalize_action_label(label);
    match label.as_str() {
        "F" => actions
            .iter()
            .position(|action| matches!(action, TreeAction::Fold)),
        "C" => actions
            .iter()
            .position(|action| matches!(action, TreeAction::Call))
            .or_else(|| {
                if is_all_in_call_response(actions) {
                    actions
                        .iter()
                        .position(|action| matches!(action, TreeAction::AllIn))
                } else {
                    None
                }
            }),
        "R2.5" => action_idx_for_raise_to(actions, 2.5 * big_blind),
        "R5" => action_idx_for_raise_to(actions, 5.0 * big_blind),
        "RAI" => actions
            .iter()
            .position(|action| matches!(action, TreeAction::AllIn)),
        _ => None,
    }
}

fn action_idx_for_raise_to(actions: &[TreeAction], amount: f64) -> Option<usize> {
    actions.iter().position(|action| match action {
        TreeAction::Raise(value) => (*value - amount).abs() <= ACTION_EPSILON,
        _ => false,
    })
}

fn is_all_in_call_response(actions: &[TreeAction]) -> bool {
    actions.len() == 2
        && matches!(actions[0], TreeAction::Fold)
        && matches!(actions[1], TreeAction::AllIn)
}

fn normalize_action_label(label: &str) -> String {
    label.trim().to_ascii_uppercase()
}

fn normalize_opening_size(value: &str) -> String {
    value.trim().to_ascii_uppercase()
}

fn approx_eq(left: f64, right: f64) -> bool {
    (left - right).abs() <= ACTION_EPSILON
}

fn normalized_spot_key(map_key: &str, spot: &BaselineSpot) -> String {
    if spot.spot_key.is_empty() {
        map_key.to_string()
    } else {
        spot.spot_key.clone()
    }
}

fn format_tree_action(action: TreeAction) -> String {
    match action {
        TreeAction::Fold => "Fold".to_string(),
        TreeAction::Check => "Check".to_string(),
        TreeAction::Call => "Call".to_string(),
        TreeAction::Bet(amount) => format!("Bet({amount:.2})"),
        TreeAction::Raise(amount) => format!("Raise({amount:.2})"),
        TreeAction::AllIn => "AllIn".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedProvider {
        preflop_buckets: u16,
        rows: BTreeMap<u16, Vec<f64>>,
        default: Vec<f64>,
    }

    impl BaselineStrategyProvider for FixedProvider {
        fn preflop_bucket_count(&self) -> u16 {
            self.preflop_buckets
        }

        fn average_strategy(&self, _node_idx: u32, bucket: u16) -> Vec<f64> {
            self.rows
                .get(&bucket)
                .cloned()
                .unwrap_or_else(|| self.default.clone())
        }
    }

    fn target_tree(allow_preflop_limp: bool) -> GameTree {
        GameTree::build_with_options(
            40.0,
            1.0,
            2.0,
            &[vec!["2.5bb".to_string()], vec!["5bb".to_string()]],
            &[],
            &[],
            &[],
            allow_preflop_limp,
        )
    }

    fn tree_with_stack(stack_depth: f64) -> GameTree {
        GameTree::build_with_options(
            stack_depth,
            1.0,
            2.0,
            &[vec!["2.5bb".to_string()], vec!["5bb".to_string()]],
            &[],
            &[],
            &[],
            false,
        )
    }

    fn spot(spot_key: &str, path: &[&str], actions: &[&str]) -> BaselineSpot {
        BaselineSpot {
            spot_key: spot_key.to_string(),
            path: path.iter().map(|item| (*item).to_string()).collect(),
            street: "PREFLOP".to_string(),
            actions: actions.iter().map(|item| (*item).to_string()).collect(),
            ..BaselineSpot::default()
        }
    }

    fn strict_baseline() -> BaselineDocument {
        BaselineDocument {
            schema_version: Some(1),
            site: Some("gtowizard".to_string()),
            game: BaselineGameMetadata {
                id: Some("cash_hu_20bb_cev".to_string()),
                stack_depth_bb: Some(20.0),
                opening_size: Some("25x".to_string()),
                ..BaselineGameMetadata::default()
            },
            actions: BTreeMap::from([
                (
                    "F".to_string(),
                    BaselineActionMetadata {
                        amount_bb: Some(0.0),
                        action_type: Some("FOLD".to_string()),
                        ..BaselineActionMetadata::default()
                    },
                ),
                (
                    "R2.5".to_string(),
                    BaselineActionMetadata {
                        amount_bb: Some(2.5),
                        action_type: Some("RAISE".to_string()),
                        ..BaselineActionMetadata::default()
                    },
                ),
                (
                    "R5".to_string(),
                    BaselineActionMetadata {
                        amount_bb: Some(5.0),
                        action_type: Some("RAISE".to_string()),
                        ..BaselineActionMetadata::default()
                    },
                ),
                (
                    "RAI".to_string(),
                    BaselineActionMetadata {
                        amount_bb: Some(20.0),
                        action_type: Some("RAISE".to_string()),
                        ..BaselineActionMetadata::default()
                    },
                ),
                (
                    "C".to_string(),
                    BaselineActionMetadata {
                        amount_bb: Some(20.0),
                        action_type: Some("CALL".to_string()),
                        ..BaselineActionMetadata::default()
                    },
                ),
            ]),
            spots: BTreeMap::from([
                ("root".to_string(), spot("root", &[], &["F", "R2.5", "RAI"])),
                (
                    "SB:r2.5".to_string(),
                    spot("SB:r2.5", &["R2.5"], &["F", "C", "R5", "RAI"]),
                ),
                ("SB:rai".to_string(), spot("SB:rai", &["RAI"], &["F", "C"])),
                (
                    "SB:r2.5, BB:r5".to_string(),
                    spot("SB:r2.5, BB:r5", &["R2.5", "R5"], &["F", "C", "RAI"]),
                ),
                (
                    "SB:r2.5, BB:rai".to_string(),
                    spot("SB:r2.5, BB:rai", &["R2.5", "RAI"], &["F", "C"]),
                ),
                (
                    "SB:r2.5, BB:r5, SB:rai".to_string(),
                    spot(
                        "SB:r2.5, BB:r5, SB:rai",
                        &["R2.5", "R5", "RAI"],
                        &["F", "C"],
                    ),
                ),
            ]),
            ..BaselineDocument::default()
        }
    }

    fn fixed_provider(preflop_buckets: u16, num_actions: usize) -> FixedProvider {
        FixedProvider {
            preflop_buckets,
            rows: BTreeMap::new(),
            default: vec![1.0 / num_actions as f64; num_actions],
        }
    }

    #[test]
    fn parse_representative_baseline_json_shape() {
        let json = r#"
        {
          "schema_version": 1,
          "site": "gtowizard",
          "game": {
            "id": "cash_hu_20bb_cev",
            "stackDepthBb": 20,
            "rawGametype": "CashHuGeneral_cEVR25"
          },
          "actions": {
            "F": { "label": "FOLD", "type": "FOLD", "isTerminal": true },
            "R2.5": { "label": "RAISE 2.5", "amountBb": 2.5, "type": "RAISE" }
          },
          "spots": {
            "root": {
              "spot_key": "root",
              "path": [],
              "street": "PREFLOP",
              "position_to_act": "SB",
              "actions": ["F", "R2.5"],
              "action_summary": { "F": 0.4, "R2.5": 0.6 },
              "strategy": {
                "AA": {
                  "ev": 9.5,
                  "action_frequencies": { "F": 0.0, "R2.5": 1.0 }
                }
              },
              "unknownSpotField": 17
            }
          },
          "unknownDocumentField": true
        }
        "#;

        let baseline = parse_baseline_json(json).expect("fixture should parse");
        assert_eq!(baseline.schema_version, Some(1));
        assert_eq!(baseline.game.id.as_deref(), Some("cash_hu_20bb_cev"));
        assert_eq!(baseline.game.stack_depth_bb, Some(20.0));
        assert!(baseline.actions.contains_key("R2.5"));
        let root = baseline.spots.get("root").expect("root spot");
        assert_eq!(root.strategy["AA"].action_frequencies["R2.5"], 1.0);
        assert!(root.extra.contains_key("unknownSpotField"));
        assert!(baseline.extra.contains_key("unknownDocumentField"));
    }

    #[test]
    fn six_baseline_spot_paths_resolve_under_exact_config() {
        let tree = target_tree(false);
        let spots = [
            spot("root", &[], &["F", "R2.5", "RAI"]),
            spot("SB:r2.5", &["R2.5"], &["F", "C", "R5", "RAI"]),
            spot("SB:rai", &["RAI"], &["F", "C"]),
            spot("SB:r2.5, BB:r5", &["R2.5", "R5"], &["F", "C", "RAI"]),
            spot("SB:r2.5, BB:rai", &["R2.5", "RAI"], &["F", "C"]),
            spot(
                "SB:r2.5, BB:r5, SB:rai",
                &["R2.5", "R5", "RAI"],
                &["F", "C"],
            ),
        ];

        for baseline_spot in spots {
            let node_idx =
                resolve_spot_path(&tree, &baseline_spot, 2.0).expect("spot should resolve");
            assert!(matches!(
                tree.nodes[node_idx as usize],
                GameNode::Decision {
                    street: Street::Preflop,
                    ..
                }
            ));
        }
    }

    #[test]
    fn root_action_schema_maps_exactly() {
        let tree = target_tree(false);
        let root = spot("root", &[], &["F", "R2.5", "RAI"]);
        let node_idx = resolve_spot_path(&tree, &root, 2.0).expect("root resolves");
        let mapping = build_action_mapping(&tree, node_idx, &root.actions, 2.0)
            .expect("mapping should build");

        assert!(mapping.is_supported());
        assert_eq!(
            mapping
                .entries
                .iter()
                .map(|entry| (&entry.label, entry.action))
                .collect::<Vec<_>>(),
            vec![
                (&"F".to_string(), TreeAction::Fold),
                (&"R2.5".to_string(), TreeAction::Raise(5.0)),
                (&"RAI".to_string(), TreeAction::AllIn),
            ]
        );
    }

    #[test]
    fn all_in_call_response_maps_to_baseline_call() {
        let tree = target_tree(false);
        let all_in_response = spot("SB:rai", &["RAI"], &["F", "C"]);
        let node_idx =
            resolve_spot_path(&tree, &all_in_response, 2.0).expect("all-in response resolves");
        let mapping = build_action_mapping(&tree, node_idx, &all_in_response.actions, 2.0)
            .expect("mapping should build");

        let call = mapping
            .entries
            .iter()
            .find(|entry| entry.label == "C")
            .expect("baseline call should map");
        assert_eq!(call.action, TreeAction::AllIn);
    }

    #[test]
    fn zero_mass_rows_are_skipped_from_scoring() {
        let tree = target_tree(false);
        let mut root = spot("root", &[], &["F", "R2.5", "RAI"]);
        root.strategy.insert(
            "22".to_string(),
            BaselineComboStrategy {
                action_frequencies: BTreeMap::from([
                    ("F".to_string(), 0.0),
                    ("R2.5".to_string(), 0.0),
                    ("RAI".to_string(), 0.0),
                ]),
                ..BaselineComboStrategy::default()
            },
        );
        root.strategy.insert(
            "AA".to_string(),
            BaselineComboStrategy {
                action_frequencies: BTreeMap::from([
                    ("F".to_string(), 0.0),
                    ("R2.5".to_string(), 1.0),
                    ("RAI".to_string(), 0.0),
                ]),
                ..BaselineComboStrategy::default()
            },
        );

        let node_idx = resolve_spot_path(&tree, &root, 2.0).expect("root resolves");
        let mapping =
            build_action_mapping(&tree, node_idx, &root.actions, 2.0).expect("mapping builds");
        let provider = FixedProvider {
            preflop_buckets: 169,
            rows: BTreeMap::from([(
                CanonicalHand::parse("AA").unwrap().index() as u16,
                vec![0.0, 1.0, 0.0],
            )]),
            default: vec![1.0 / 3.0; 3],
        };

        let (metrics, rows, invalid_hands) = score_spot(&tree, &provider, "root", &root, &mapping);
        assert_eq!(metrics.combo_rows_scored, 1);
        assert_eq!(metrics.combo_rows_skipped_zero_mass, 1);
        assert_eq!(metrics.combo_rows_invalid_hand, 0);
        assert_eq!(metrics.total_combo_weight, 6.0);
        assert_eq!(rows.len(), 1);
        assert!(invalid_hands.is_empty());
        assert_eq!(rows[0].hand, "AA");
    }

    #[test]
    fn limp_enabled_config_reports_unsupported_root_schema() {
        let tree = target_tree(true);
        let root = spot("root", &[], &["F", "R2.5", "RAI"]);
        let node_idx = resolve_spot_path(&tree, &root, 2.0).expect("root resolves");
        let mapping = build_action_mapping(&tree, node_idx, &root.actions, 2.0)
            .expect("mapping should build");

        assert!(!mapping.is_supported());
        assert_eq!(mapping.unsupported_baseline_actions, Vec::<String>::new());
        assert_eq!(mapping.unmapped_candidate_actions, vec!["Call".to_string()]);
    }

    #[test]
    fn non_169_provider_is_refused_before_scoring() {
        let baseline = strict_baseline();
        let tree = target_tree(false);
        let provider = fixed_provider(168, 3);

        let report = validate_baseline(
            &baseline,
            &tree,
            &provider,
            BaselineValidationConfig::default(),
        );

        assert_eq!(report.aggregate.precondition_failures, 1);
        assert_eq!(report.aggregate.spots_scored, 0);
        assert_eq!(report.aggregate.combo_rows_scored, 0);
        assert_eq!(
            report.precondition_failures[0].field,
            "provider.preflop_bucket_count"
        );
    }

    #[test]
    fn wrong_stack_tree_is_refused_even_when_actions_map() {
        let baseline = strict_baseline();
        let tree = tree_with_stack(60.0);
        let provider = fixed_provider(169, 3);

        let root = baseline.spots.get("root").expect("root spot");
        let root_node = resolve_spot_path(&tree, root, 2.0).expect("wrong-stack root resolves");
        let root_mapping = build_action_mapping(&tree, root_node, &root.actions, 2.0)
            .expect("wrong-stack root mapping builds");
        assert!(root_mapping.is_supported());

        let report = validate_baseline(
            &baseline,
            &tree,
            &provider,
            BaselineValidationConfig::default(),
        );

        assert_eq!(report.aggregate.spots_scored, 0);
        assert!(
            report
                .precondition_failures
                .iter()
                .any(|failure| failure.field == "tree.starting_stack")
        );
    }

    #[test]
    fn baseline_metadata_is_validated_before_scoring() {
        let mut baseline = strict_baseline();
        baseline.game.stack_depth_bb = Some(25.0);
        baseline.game.opening_size = Some("30x".to_string());
        let tree = target_tree(false);
        let provider = fixed_provider(169, 3);

        let report = validate_baseline(
            &baseline,
            &tree,
            &provider,
            BaselineValidationConfig::default(),
        );

        assert_eq!(report.aggregate.spots_scored, 0);
        assert!(
            report
                .precondition_failures
                .iter()
                .any(|failure| failure.field == "baseline.game.stackDepthBb")
        );
        assert!(
            report
                .precondition_failures
                .iter()
                .any(|failure| failure.field == "baseline.game.openingSize")
        );
    }

    #[test]
    fn malformed_hand_rows_are_reported() {
        let mut baseline = strict_baseline();
        baseline.spots.get_mut("root").unwrap().strategy.insert(
            "not-a-hand".to_string(),
            BaselineComboStrategy {
                action_frequencies: BTreeMap::from([
                    ("F".to_string(), 1.0),
                    ("R2.5".to_string(), 0.0),
                    ("RAI".to_string(), 0.0),
                ]),
                ..BaselineComboStrategy::default()
            },
        );
        baseline.spots.get_mut("root").unwrap().strategy.insert(
            "AA".to_string(),
            BaselineComboStrategy {
                action_frequencies: BTreeMap::from([
                    ("F".to_string(), 0.0),
                    ("R2.5".to_string(), 1.0),
                    ("RAI".to_string(), 0.0),
                ]),
                ..BaselineComboStrategy::default()
            },
        );
        let tree = target_tree(false);
        let provider = fixed_provider(169, 3);

        let report = validate_baseline(
            &baseline,
            &tree,
            &provider,
            BaselineValidationConfig::default(),
        );

        assert_eq!(report.aggregate.precondition_failures, 0);
        assert_eq!(report.aggregate.combo_rows_invalid_hand, 1);
        assert_eq!(report.invalid_hand_rows.len(), 1);
        assert_eq!(report.invalid_hand_rows[0].spot_key, "root");
        assert_eq!(report.invalid_hand_rows[0].hand_label, "not-a-hand");
    }
}
