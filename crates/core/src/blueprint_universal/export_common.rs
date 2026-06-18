//! Shared row-collection and flattening machinery for universal export.
//!
//! Both `hu_export` and `mp_eager_export` collect intermediate [`RowEntry`]
//! values, sort them by spec order, and flatten into contiguous
//! `(RowDescriptor, ActionDescriptor, f32)` arrays.  This module
//! provides the shared types and functions so neither exporter
//! duplicates the plumbing.

#![allow(clippy::cast_possible_truncation)]

use std::path::Path;

use super::descriptors::{ActionDescriptor, RowDescriptor};
use super::hash::Fnv1aHasher;
use super::manifest::Manifest;

// ---------------------------------------------------------------------------
// Namespace constants
// ---------------------------------------------------------------------------

/// Namespace for HU arena rows.
pub(crate) const NS_HU_ARENA: u16 = 0;
/// Namespace for MP eager arena rows.
pub(crate) const NS_MP_ARENA: u16 = 1;
/// Namespace for MP lazy semantic rows.
pub(crate) const NS_MP_SEMANTIC: u16 = 2;

// ---------------------------------------------------------------------------
// Semantic key kind constants
// ---------------------------------------------------------------------------

/// No semantic key side table (HU and MP eager rows).
pub(crate) const SEMANTIC_KEY_NONE: u16 = 0;
/// MP history semantic key v1 (side table record index).
pub(crate) const SEMANTIC_KEY_MP_HISTORY_V1: u16 = 1;

// ---------------------------------------------------------------------------
// RowEntry: intermediate row before sorting and offset assignment
// ---------------------------------------------------------------------------

/// Intermediate row collected by an exporter before sorting.
pub(crate) struct RowEntry {
    pub namespace: u16,
    pub seat: u8,
    pub street: u8,
    pub local_bucket: u16,
    pub source_node_idx: u32,
    pub global_bucket: u32,
    pub row_key_fp: u64,
    pub action_schema_fp: u64,
    pub actions: Vec<ActionDescriptor>,
    pub probs: Vec<f32>,
    /// Semantic key kind (0 = none, 1 = mp_history_v1).
    pub semantic_key_kind: u16,
    /// Index into the semantic key side table (record index, not byte offset).
    pub semantic_key_index: u64,
}

impl RowEntry {
    /// Sort key per spec: `(namespace, seat, street, source_node_idx,
    /// global_bucket, row_key_fingerprint)`.
    pub fn sort_key(&self) -> (u16, u8, u8, u32, u32, u64) {
        (
            self.namespace,
            self.seat,
            self.street,
            self.source_node_idx,
            self.global_bucket,
            self.row_key_fp,
        )
    }
}

// ---------------------------------------------------------------------------
// Row key fingerprint (version 1)
// ---------------------------------------------------------------------------

/// Compute a stable FNV-1a row key fingerprint over the identity tuple.
///
/// Inputs (version 1): `(namespace: u16, seat: u8, street: u8,
/// source_node_idx: u32, global_bucket: u32)`.
pub(crate) fn row_key_fingerprint(
    namespace: u16,
    seat: u8,
    street: u8,
    source_node_idx: u32,
    global_bucket: u32,
) -> u64 {
    let mut h = Fnv1aHasher::new();
    h.mix_u16(namespace);
    h.mix_u8(seat);
    h.mix_u8(street);
    h.mix_u32(source_node_idx);
    h.mix_u32(global_bucket);
    h.finish()
}

// ---------------------------------------------------------------------------
// Flatten sorted entries into contiguous arrays
// ---------------------------------------------------------------------------

/// Assign contiguous offsets to sorted entries and flatten into arrays.
pub(crate) fn flatten_entries(
    entries: &[RowEntry],
) -> (Vec<RowDescriptor>, Vec<ActionDescriptor>, Vec<f32>) {
    let mut rows = Vec::with_capacity(entries.len());
    let mut all_actions = Vec::new();
    let mut all_probs = Vec::new();

    for (row_id, entry) in entries.iter().enumerate() {
        let action_offset = all_actions.len() as u64;
        let prob_offset = all_probs.len() as u64;

        rows.push(RowDescriptor {
            row_id: row_id as u64,
            namespace: entry.namespace,
            seat: entry.seat,
            street: entry.street,
            local_bucket: entry.local_bucket,
            global_bucket: entry.global_bucket,
            source_node_idx: entry.source_node_idx,
            action_offset,
            action_count: entry.actions.len() as u16,
            prob_offset,
            row_key_fingerprint: entry.row_key_fp,
            action_schema_fingerprint: entry.action_schema_fp,
            semantic_key_kind: entry.semantic_key_kind,
            semantic_key_offset: entry.semantic_key_index,
        });

        all_actions.extend_from_slice(&entry.actions);
        all_probs.extend_from_slice(&entry.probs);
    }

    (rows, all_actions, all_probs)
}

// ---------------------------------------------------------------------------
// Bucket semantic fingerprint
// ---------------------------------------------------------------------------

/// Compute a bucket semantic fingerprint from bucket counts.
pub(crate) fn bucket_semantic_fingerprint(bucket_counts: [u16; 4]) -> u64 {
    let mut h = Fnv1aHasher::new();
    for bc in bucket_counts {
        h.mix_u16(bc);
    }
    h.finish()
}

// ---------------------------------------------------------------------------
// Timestamp helpers
// ---------------------------------------------------------------------------

/// Current UTC timestamp in RFC 3339 format.
pub(crate) fn now_rfc3339_approx() -> String {
    use std::time::SystemTime;
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    epoch_secs_to_rfc3339(dur.as_secs())
}

/// Convert Unix epoch seconds to an RFC 3339 UTC timestamp string.
fn epoch_secs_to_rfc3339(epoch: u64) -> String {
    let (days, day_secs) = (epoch / 86_400, epoch % 86_400);
    let (hour, min, sec) = (
        day_secs / 3600,
        (day_secs % 3600) / 60,
        day_secs % 60,
    );
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// Convert days since 1970-01-01 to (year, month, day).
fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    let mut year = 1970u64;
    loop {
        let year_days = if is_leap(year) { 366 } else { 365 };
        if days < year_days {
            break;
        }
        days -= year_days;
        year += 1;
    }
    let leap = is_leap(year);
    let month_days: [u64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ];
    let mut month = 1u64;
    for &md in &month_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    (year, month, days + 1)
}

/// True if `year` is a Gregorian leap year.
fn is_leap(year: u64) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100))
        || year.is_multiple_of(400)
}

// ---------------------------------------------------------------------------
// Config file retention
// ---------------------------------------------------------------------------

/// Copy a source `config.yaml` into the output bundle directory and
/// record the relative path in `manifest.training.config_path`.
///
/// Creates `out_dir` (including parents) if it does not already exist.
pub fn retain_config_yaml(
    source: &Path,
    out_dir: &Path,
    manifest: &mut Manifest,
) -> Result<(), std::io::Error> {
    std::fs::create_dir_all(out_dir)?;
    std::fs::copy(source, out_dir.join("config.yaml"))?;
    manifest.training.config_path = Some("config.yaml".to_string());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_key_fingerprint_is_deterministic() {
        let fp1 = row_key_fingerprint(1, 0, 1, 42, 7);
        let fp2 = row_key_fingerprint(1, 0, 1, 42, 7);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn row_key_fingerprint_varies_with_seat() {
        let fp0 = row_key_fingerprint(1, 0, 0, 10, 5);
        let fp1 = row_key_fingerprint(1, 1, 0, 10, 5);
        assert_ne!(fp0, fp1);
    }

    #[test]
    fn row_key_fingerprint_varies_with_namespace() {
        let fp_hu = row_key_fingerprint(0, 0, 1, 42, 7);
        let fp_mp = row_key_fingerprint(1, 0, 1, 42, 7);
        assert_ne!(fp_hu, fp_mp);
    }

    #[test]
    fn flatten_entries_assigns_contiguous_offsets() {
        use super::super::descriptors::{ActionDescriptor, ActionKind};

        let ad = ActionDescriptor {
            kind: ActionKind::Fold,
            amount_chips: 0,
            size_key: String::new(),
            label: "Fold".to_string(),
            is_aggressive: false,
            source_action_index: 0,
        };
        let entries = vec![
            RowEntry {
                namespace: 1,
                seat: 0,
                street: 0,
                local_bucket: 0,
                source_node_idx: 0,
                global_bucket: 0,
                row_key_fp: 100,
                action_schema_fp: 200,
                actions: vec![ad.clone(), ad.clone()],
                probs: vec![0.5, 0.5],
                semantic_key_kind: 0,
                semantic_key_index: 0,
            },
            RowEntry {
                namespace: 1,
                seat: 0,
                street: 0,
                local_bucket: 1,
                source_node_idx: 0,
                global_bucket: 1,
                row_key_fp: 101,
                action_schema_fp: 200,
                actions: vec![ad.clone(), ad.clone(), ad.clone()],
                probs: vec![0.33, 0.33, 0.34],
                semantic_key_kind: 0,
                semantic_key_index: 0,
            },
        ];
        let (rows, actions, probs) = flatten_entries(&entries);
        assert_eq!(rows.len(), 2);
        assert_eq!(actions.len(), 5);
        assert_eq!(probs.len(), 5);

        assert_eq!(rows[0].action_offset, 0);
        assert_eq!(rows[0].action_count, 2);
        assert_eq!(rows[0].prob_offset, 0);

        assert_eq!(rows[1].action_offset, 2);
        assert_eq!(rows[1].action_count, 3);
        assert_eq!(rows[1].prob_offset, 2);
    }

    #[test]
    fn bucket_semantic_fingerprint_deterministic() {
        let fp1 = bucket_semantic_fingerprint([10, 20, 30, 40]);
        let fp2 = bucket_semantic_fingerprint([10, 20, 30, 40]);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn bucket_semantic_fingerprint_differs() {
        let fp1 = bucket_semantic_fingerprint([10, 20, 30, 40]);
        let fp2 = bucket_semantic_fingerprint([10, 20, 30, 41]);
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn epoch_to_rfc3339_known_value() {
        let ts = epoch_secs_to_rfc3339(1_735_689_600);
        assert_eq!(ts, "2025-01-01T00:00:00Z");
    }

    #[test]
    fn epoch_to_rfc3339_leap_year() {
        let ts = epoch_secs_to_rfc3339(1_709_208_000);
        assert_eq!(ts, "2024-02-29T12:00:00Z");
    }

    #[test]
    fn epoch_to_rfc3339_unix_epoch() {
        let ts = epoch_secs_to_rfc3339(0);
        assert_eq!(ts, "1970-01-01T00:00:00Z");
    }

    /// Build a minimal [`Manifest`] for tests that only inspect the
    /// training.config_path field.
    fn stub_manifest() -> super::super::manifest::Manifest {
        use super::super::manifest::*;
        use std::collections::BTreeMap;
        Manifest {
            format_name: "universal_dense".into(),
            format_version: 1,
            compat_min_reader: 1,
            created_at: String::new(),
            producer: String::new(),
            producer_git: String::new(),
            required_features: vec![],
            optional_features: vec![],
            game_fingerprint: 0,
            source_config_fingerprint: None,
            game: GameMetadata {
                game_kind: "nlhe".into(),
                num_players: 2,
                seats: vec![],
                button_seat: 0,
                small_blind: 0.5,
                big_blind: 1.0,
                antes: vec![],
                straddles: vec![],
                stack_units: "chips".into(),
                rake: RakeConfig { rate: 0.0, cap: 0.0 },
                max_flop_players: None,
            },
            training: TrainingMetadata {
                source_backend: String::new(),
                unit_kind: "iterations".into(),
                units_completed: 0,
                elapsed_minutes: 0.0,
                strategy_unit: "average".into(),
                command: None,
                config_path: None,
                config_fingerprint: None,
            },
            strategy: StrategyMetadata {
                normalization_tolerance: 1e-4,
            },
            layout: LayoutMetadata {
                row_count: 0,
                action_descriptor_count: 0,
                probability_count: 0,
                row_sort_order: "spec_v1".into(),
                row_namespace: vec![],
                missing_row_policy: "uniform".into(),
            },
            actions: ActionsMetadata {
                action_abstraction_fingerprint: 0,
            },
            buckets: BucketsMetadata {
                bucket_mode: "fixed".into(),
                bucket_semantic_fingerprint: 0,
                street_bucket_counts: BTreeMap::new(),
                bucket_files: vec![],
                bucket_generator_version: String::new(),
                per_flop_bucket_config: None,
            },
            compatibility: CompatibilityMetadata {
                legacy_fallback: false,
                missing_row_policy: "uniform".into(),
                resumable: false,
            },
            files: BTreeMap::new(),
        }
    }

    #[test]
    fn retain_config_yaml_copies_file_and_sets_manifest_path() {
        let tmp = std::env::temp_dir()
            .join("retain_config_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let source = tmp.join("source_config.yaml");
        std::fs::write(&source, "game: test\n").unwrap();

        let out_dir = tmp.join("output_bundle");
        let mut manifest = stub_manifest();

        retain_config_yaml(&source, &out_dir, &mut manifest)
            .unwrap();

        assert!(out_dir.join("config.yaml").exists());
        assert_eq!(
            std::fs::read_to_string(out_dir.join("config.yaml"))
                .unwrap(),
            "game: test\n"
        );
        assert_eq!(
            manifest.training.config_path.as_deref(),
            Some("config.yaml")
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn retain_config_yaml_creates_output_dir() {
        let tmp = std::env::temp_dir()
            .join("retain_config_mkdir_test");
        let _ = std::fs::remove_dir_all(&tmp);

        let source = tmp.join("src.yaml");
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(&source, "x: 1\n").unwrap();

        let nested = tmp.join("a").join("b").join("c");
        let mut manifest = stub_manifest();

        retain_config_yaml(&source, &nested, &mut manifest)
            .unwrap();

        assert!(nested.join("config.yaml").exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
