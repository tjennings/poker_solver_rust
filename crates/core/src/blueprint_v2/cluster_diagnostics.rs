//! Diagnostics for analyzing bucket files produced by the clustering pipeline.
//!
//! Reads `.buckets` files from a directory and produces per-street reports on
//! bucket count, entry distribution, and size uniformity.
//!
//! The [`audit_bucket_equity`] function samples boards, computes showdown
//! equity for every combo, and reports per-bucket equity statistics to
//! verify that hands within the same bucket share similar equity profiles.

use std::path::Path;

use rayon::prelude::*;

use super::bucket_file::BucketFile;
use super::cluster_pipeline::{build_deck, compute_board_equities, enumerate_combos, sample_boards};

/// Size distribution statistics for bucket assignments.
#[derive(Debug)]
pub struct SizeStats {
    pub min: usize,
    pub max: usize,
    pub mean: f64,
    pub std_dev: f64,
}

/// Diagnostics report for a single `BucketFile`.
#[derive(Debug)]
pub struct ClusterReport {
    pub street: String,
    pub bucket_count: u16,
    pub board_count: u32,
    pub combos_per_board: u16,
    pub total_entries: usize,
    /// Number of entries assigned to each bucket, indexed by bucket id.
    pub bucket_sizes: Vec<usize>,
    pub size_stats: SizeStats,
}

impl ClusterReport {
    /// Build a report by scanning every entry in the bucket file.
    #[must_use]
    pub fn from_bucket_file(bf: &BucketFile) -> Self {
        let bucket_count = bf.header.bucket_count;
        let mut counts = vec![0_usize; bucket_count as usize];
        for &b in &bf.buckets {
            if (b as usize) < counts.len() {
                counts[b as usize] += 1;
            }
        }

        let total = bf.buckets.len();
        let n = f64::from(bucket_count);
        #[allow(clippy::cast_precision_loss)]
        let mean = total as f64 / n;
        #[allow(clippy::cast_precision_loss)]
        let variance = counts
            .iter()
            .map(|&c| (c as f64 - mean).powi(2))
            .sum::<f64>()
            / n;

        let street_name = format!("{:?}", bf.header.street);

        Self {
            street: street_name,
            bucket_count,
            board_count: bf.header.board_count,
            combos_per_board: bf.header.combos_per_board,
            total_entries: total,
            bucket_sizes: counts.clone(),
            size_stats: SizeStats {
                min: counts.iter().copied().min().unwrap_or(0),
                max: counts.iter().copied().max().unwrap_or(0),
                mean,
                std_dev: variance.sqrt(),
            },
        }
    }

    /// Format as a human-readable summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{street}: {bc} buckets, {boards} boards x {cpb} combos = {total} entries\n  \
             Bucket sizes: min={min}, max={max}, mean={mean:.1}, std={std:.1}",
            street = self.street,
            bc = self.bucket_count,
            boards = self.board_count,
            cpb = self.combos_per_board,
            total = self.total_entries,
            min = self.size_stats.min,
            max = self.size_stats.max,
            mean = self.size_stats.mean,
            std = self.size_stats.std_dev,
        )
    }
}

/// Scan a directory for `.buckets` files and produce a report for each.
///
/// Looks for files named `river.buckets`, `turn.buckets`, `flop.buckets`,
/// and `preflop.buckets` (in that order).
///
/// # Errors
///
/// Returns an error if any existing `.buckets` file cannot be loaded.
pub fn diagnose_cluster_dir(
    dir: &Path,
) -> Result<Vec<ClusterReport>, Box<dyn std::error::Error>> {
    let mut reports = Vec::new();

    for street_name in &["river", "turn", "flop", "preflop"] {
        let path = dir.join(format!("{street_name}.buckets"));
        if path.exists() {
            let bf = BucketFile::load(&path)?;
            reports.push(ClusterReport::from_bucket_file(&bf));
        }
    }

    Ok(reports)
}

// ---------------------------------------------------------------------------
// Equity audit
// ---------------------------------------------------------------------------

/// Per-bucket equity statistics from the audit.
#[derive(Debug, Clone)]
pub struct BucketEquityStats {
    pub bucket_id: u16,
    pub count: usize,
    pub mean_equity: f64,
    pub std_dev: f64,
    pub min_equity: f64,
    pub max_equity: f64,
}

/// Full audit report for a single street's bucket file.
#[derive(Debug)]
pub struct EquityAuditReport {
    pub street: String,
    pub bucket_count: u16,
    pub sample_boards: usize,
    pub buckets: Vec<BucketEquityStats>,
    /// Mean of per-bucket std deviations (lower = better clustering).
    pub mean_intra_bucket_std: f64,
    /// Worst (highest) intra-bucket std deviation.
    pub max_intra_bucket_std: f64,
}

impl EquityAuditReport {
    /// Format as a human-readable summary.
    #[must_use]
    pub fn summary(&self) -> String {
        use std::fmt::Write;
        let mut s = format!(
            "{street}: {bc} buckets, {nb} sample boards\n  \
             Mean intra-bucket std: {mean:.4}, max: {max:.4}\n  \
             Per-bucket (sorted by mean equity):",
            street = self.street,
            bc = self.bucket_count,
            nb = self.sample_boards,
            mean = self.mean_intra_bucket_std,
            max = self.max_intra_bucket_std,
        );
        let mut sorted: Vec<_> = self.buckets.iter().collect();
        sorted.sort_by(|a, b| {
            a.mean_equity
                .partial_cmp(&b.mean_equity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for b in &sorted {
            let _ = write!(
                s,
                "\n    bucket {:>3}: n={:<6} eq={:.3}..{:.3}  mean={:.3}  std={:.4}",
                b.bucket_id, b.count, b.min_equity, b.max_equity, b.mean_equity, b.std_dev,
            );
        }
        s
    }
}

/// Audit a river bucket file by sampling boards and computing showdown equity.
///
/// For each sampled 5-card board, computes equity for all 1326 combos and
/// groups by bucket assignment. Reports per-bucket equity distribution.
///
/// # Arguments
/// * `bf` — The bucket file to audit (should be river street).
/// * `num_sample_boards` — How many boards to sample for equity computation.
/// * `seed` — RNG seed for board sampling.
#[must_use]
pub fn audit_bucket_equity(bf: &BucketFile, num_sample_boards: usize, seed: u64) -> EquityAuditReport {
    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    let boards = sample_boards(&deck, num_sample_boards, seed);
    let bucket_count = bf.header.bucket_count;
    let board_count = bf.header.board_count as usize;

    // Collect (equity, bucket_id) pairs across all sampled boards.
    let equity_bucket_pairs: Vec<(f64, u16)> = boards
        .par_iter()
        .enumerate()
        .flat_map_iter(|(sample_idx, &board)| {
            let equities = compute_board_equities(board, &combos);
            // Map sample board index to the bucket file's board dimension.
            // If the bucket file has fewer boards than our sample, wrap around.
            let board_idx = sample_idx % board_count;
            equities
                .into_iter()
                .enumerate()
                .filter_map(move |(combo_idx, eq_opt)| {
                    let eq = eq_opt?;
                    #[allow(clippy::cast_possible_truncation)]
                    let bucket = bf.get_bucket(board_idx as u32, combo_idx as u16);
                    Some((eq, bucket))
                })
        })
        .collect();

    // Group equities by bucket.
    let mut bucket_equities: Vec<Vec<f64>> = vec![Vec::new(); bucket_count as usize];
    for &(eq, bucket) in &equity_bucket_pairs {
        if (bucket as usize) < bucket_equities.len() {
            bucket_equities[bucket as usize].push(eq);
        }
    }

    // Compute per-bucket statistics.
    let bucket_stats: Vec<BucketEquityStats> = bucket_equities
        .iter()
        .enumerate()
        .map(|(id, eqs)| {
            if eqs.is_empty() {
                #[allow(clippy::cast_possible_truncation)]
                return BucketEquityStats {
                    bucket_id: id as u16,
                    count: 0,
                    mean_equity: 0.0,
                    std_dev: 0.0,
                    min_equity: 0.0,
                    max_equity: 0.0,
                };
            }
            #[allow(clippy::cast_precision_loss)]
            let n = eqs.len() as f64;
            let sum: f64 = eqs.iter().sum();
            let mean = sum / n;
            let variance = eqs.iter().map(|&e| (e - mean).powi(2)).sum::<f64>() / n;
            let min = eqs.iter().copied().fold(f64::INFINITY, f64::min);
            let max = eqs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            #[allow(clippy::cast_possible_truncation)]
            BucketEquityStats {
                bucket_id: id as u16,
                count: eqs.len(),
                mean_equity: mean,
                std_dev: variance.sqrt(),
                min_equity: min,
                max_equity: max,
            }
        })
        .collect();

    let non_empty: Vec<_> = bucket_stats.iter().filter(|b| b.count > 0).collect();
    let mean_std = if non_empty.is_empty() {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        { non_empty.iter().map(|b| b.std_dev).sum::<f64>() / non_empty.len() as f64 }
    };
    let max_std = non_empty
        .iter()
        .map(|b| b.std_dev)
        .fold(0.0_f64, f64::max);

    let street_name = format!("{:?}", bf.header.street);

    EquityAuditReport {
        street: street_name,
        bucket_count,
        sample_boards: num_sample_boards,
        buckets: bucket_stats,
        mean_intra_bucket_std: mean_std,
        max_intra_bucket_std: max_std,
    }
}

/// Audit all bucket files in a directory.
///
/// Samples `num_boards` random 5-card boards, computes equity, and reports
/// per-bucket equity stats for each street.
///
/// # Errors
///
/// Returns an error if any `.buckets` file cannot be loaded.
pub fn audit_cluster_dir(
    dir: &Path,
    num_boards: usize,
    seed: u64,
) -> Result<Vec<EquityAuditReport>, Box<dyn std::error::Error>> {
    let mut reports = Vec::new();

    for street_name in &["river", "turn", "flop", "preflop"] {
        let path = dir.join(format!("{street_name}.buckets"));
        if path.exists() {
            let bf = BucketFile::load(&path)?;
            reports.push(audit_bucket_equity(&bf, num_boards, seed));
        }
    }

    Ok(reports)
}

/// Transition matrix between two adjacent streets.
#[derive(Debug)]
pub struct TransitionMatrix {
    pub from_street: String,
    pub to_street: String,
    /// `matrix[from_bucket][to_bucket]` = count of (board, combo) pairs
    pub matrix: Vec<Vec<u64>>,
}

impl TransitionMatrix {
    #[must_use]
    pub fn summary(&self) -> String {
        use std::fmt::Write;
        let num_to = self.matrix.first().map_or(0, Vec::len);
        let num_from = self.matrix.len();
        let mut s = format!(
            "Transition: {} → {} ({num_from} × {num_to} buckets)\n",
            self.from_street, self.to_street,
        );
        let _ = write!(s, "  {:>6}", "");
        for j in 0..num_to {
            let _ = write!(s, " {j:>6}");
        }
        s.push('\n');
        for (i, row) in self.matrix.iter().enumerate() {
            let _ = write!(s, "  {i:>6}");
            let row_total: u64 = row.iter().sum();
            for &count in row {
                if row_total > 0 {
                    #[allow(clippy::cast_precision_loss)]
                    let pct = count as f64 / row_total as f64 * 100.0;
                    let _ = write!(s, " {pct:>5.1}%");
                } else {
                    let _ = write!(s, " {:>6}", 0);
                }
            }
            s.push('\n');
        }
        s
    }
}

#[must_use]
pub fn cross_street_transition_matrix(
    from_bf: &BucketFile,
    to_bf: &BucketFile,
) -> TransitionMatrix {
    let from_k = from_bf.header.bucket_count as usize;
    let to_k = to_bf.header.bucket_count as usize;
    let mut matrix = vec![vec![0_u64; to_k]; from_k];
    let from_boards = from_bf.header.board_count as usize;
    let to_boards = to_bf.header.board_count as usize;
    let combos = from_bf.header.combos_per_board as usize;

    #[allow(clippy::cast_possible_truncation)]
    if from_boards == 1 && to_boards > 1 {
        for combo_idx in 0..combos {
            let from_bucket = from_bf.get_bucket(0, combo_idx as u16) as usize;
            for board_idx in 0..to_boards {
                let to_bucket = to_bf.get_bucket(board_idx as u32, combo_idx as u16) as usize;
                if from_bucket < from_k && to_bucket < to_k {
                    matrix[from_bucket][to_bucket] += 1;
                }
            }
        }
    } else {
        let boards = from_boards.min(to_boards);
        #[allow(clippy::cast_possible_truncation)]
        for board_idx in 0..boards {
            for combo_idx in 0..combos {
                let from_bucket = from_bf.get_bucket(board_idx as u32, combo_idx as u16) as usize;
                let to_bucket = to_bf.get_bucket(board_idx as u32, combo_idx as u16) as usize;
                if from_bucket < from_k && to_bucket < to_k {
                    matrix[from_bucket][to_bucket] += 1;
                }
            }
        }
    }

    TransitionMatrix {
        from_street: format!("{:?}", from_bf.header.street),
        to_street: format!("{:?}", to_bf.header.street),
        matrix,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_v2::bucket_file::{BucketFileHeader, PackedBoard};
    use crate::blueprint_v2::Street;

    fn make_test_bucket_file(bucket_count: u16, buckets: Vec<u16>) -> BucketFile {
        #[allow(clippy::cast_possible_truncation)]
        BucketFile {
            header: BucketFileHeader {
                street: Street::River,
                bucket_count,
                board_count: 1,
                combos_per_board: buckets.len() as u16,
                version: 2,
            },
            boards: vec![PackedBoard(0)],
            buckets,
        }
    }

    #[test]
    fn report_basic() {
        let bf = make_test_bucket_file(3, vec![0, 0, 1, 1, 1, 2]);
        let report = ClusterReport::from_bucket_file(&bf);
        assert_eq!(report.bucket_count, 3);
        assert_eq!(report.total_entries, 6);
        assert_eq!(report.bucket_sizes, vec![2, 3, 1]);
        assert_eq!(report.size_stats.min, 1);
        assert_eq!(report.size_stats.max, 3);
    }

    #[test]
    fn report_uniform() {
        let bf = make_test_bucket_file(2, vec![0, 1, 0, 1]);
        let report = ClusterReport::from_bucket_file(&bf);
        assert_eq!(report.bucket_sizes, vec![2, 2]);
        assert!(
            report.size_stats.std_dev.abs() < 1e-10,
            "expected zero std_dev for uniform distribution, got {}",
            report.size_stats.std_dev,
        );
    }

    #[test]
    fn report_summary_contains_street() {
        let bf = make_test_bucket_file(5, vec![0, 1, 2, 3, 4]);
        let report = ClusterReport::from_bucket_file(&bf);
        let s = report.summary();
        assert!(s.contains("River"), "summary missing street name: {s}");
        assert!(s.contains("5 buckets"), "summary missing bucket count: {s}");
    }

    #[test]
    fn diagnose_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let reports = diagnose_cluster_dir(dir.path()).unwrap();
        assert!(reports.is_empty());
    }

    #[test]
    fn transition_matrix_basic() {
        let river = BucketFile {
            header: BucketFileHeader {
                street: Street::River, bucket_count: 3, board_count: 1, combos_per_board: 4, version: 2,
            },
            boards: vec![PackedBoard(0)],
            buckets: vec![0, 1, 2, 0],
        };
        let turn = BucketFile {
            header: BucketFileHeader {
                street: Street::Turn, bucket_count: 2, board_count: 1, combos_per_board: 4, version: 2,
            },
            boards: vec![PackedBoard(0)],
            buckets: vec![0, 0, 1, 1],
        };
        let matrix = cross_street_transition_matrix(&turn, &river);
        assert_eq!(matrix.from_street, "Turn");
        assert_eq!(matrix.to_street, "River");
        assert_eq!(matrix.matrix.len(), 2);
        assert_eq!(matrix.matrix[0].len(), 3);
        assert_eq!(matrix.matrix[0][0], 1);
        assert_eq!(matrix.matrix[0][1], 1);
        assert_eq!(matrix.matrix[0][2], 0);
        assert_eq!(matrix.matrix[1][0], 1);
        assert_eq!(matrix.matrix[1][1], 0);
        assert_eq!(matrix.matrix[1][2], 1);
    }

    #[test]
    fn transition_matrix_summary_format() {
        let a = BucketFile {
            header: BucketFileHeader {
                street: Street::Turn, bucket_count: 2, board_count: 1, combos_per_board: 2, version: 2,
            },
            boards: vec![PackedBoard(0)],
            buckets: vec![0, 1],
        };
        let b = BucketFile {
            header: BucketFileHeader {
                street: Street::River, bucket_count: 2, board_count: 1, combos_per_board: 2, version: 2,
            },
            boards: vec![PackedBoard(0)],
            buckets: vec![0, 1],
        };
        let matrix = cross_street_transition_matrix(&a, &b);
        let s = matrix.summary();
        assert!(s.contains("Turn"), "summary missing from_street: {s}");
        assert!(s.contains("River"), "summary missing to_street: {s}");
        assert!(s.contains("2 × 2"), "summary missing dimensions: {s}");
        assert!(s.contains("100.0%"), "summary missing percentage: {s}");
    }

    #[test]
    fn transition_matrix_preflop_to_multi_board() {
        // Preflop has 1 board, flop has 2 boards: tests the fan-out branch.
        let preflop = BucketFile {
            header: BucketFileHeader {
                street: Street::Preflop, bucket_count: 2, board_count: 1, combos_per_board: 3, version: 2,
            },
            boards: vec![PackedBoard(0)],
            buckets: vec![0, 0, 1],
        };
        let flop = BucketFile {
            header: BucketFileHeader {
                street: Street::Flop, bucket_count: 2, board_count: 2, combos_per_board: 3, version: 2,
            },
            boards: vec![PackedBoard(0), PackedBoard(1)],
            // board0: combos -> [0, 1, 0], board1: combos -> [1, 0, 1]
            buckets: vec![0, 1, 0, 1, 0, 1],
        };
        let matrix = cross_street_transition_matrix(&preflop, &flop);
        assert_eq!(matrix.matrix.len(), 2);
        // preflop bucket 0 (combos 0, 1): flop board0 -> [0,1], flop board1 -> [1,0]
        // so bucket0 -> flop_bucket0: 2, flop_bucket1: 2
        assert_eq!(matrix.matrix[0][0], 2);
        assert_eq!(matrix.matrix[0][1], 2);
        // preflop bucket 1 (combo 2): flop board0 -> [0], flop board1 -> [1]
        assert_eq!(matrix.matrix[1][0], 1);
        assert_eq!(matrix.matrix[1][1], 1);
    }

    #[test]
    fn transition_matrix_empty_row_shows_zero() {
        // One bucket has no entries, should show 0 not percentage.
        let from = BucketFile {
            header: BucketFileHeader {
                street: Street::Turn, bucket_count: 2, board_count: 1, combos_per_board: 2, version: 2,
            },
            boards: vec![PackedBoard(0)],
            buckets: vec![0, 0], // bucket 1 is empty
        };
        let to = BucketFile {
            header: BucketFileHeader {
                street: Street::River, bucket_count: 2, board_count: 1, combos_per_board: 2, version: 2,
            },
            boards: vec![PackedBoard(0)],
            buckets: vec![0, 1],
        };
        let matrix = cross_street_transition_matrix(&from, &to);
        assert_eq!(matrix.matrix[1][0], 0);
        assert_eq!(matrix.matrix[1][1], 0);
        let s = matrix.summary();
        // Row 1 should show 0s, not percentages
        assert!(s.contains("     0"), "empty row should show 0: {s}");
    }

    #[test]
    fn diagnose_dir_with_files() {
        let dir = tempfile::tempdir().unwrap();
        let bf = make_test_bucket_file(3, vec![0, 1, 2, 0, 1, 2]);
        bf.save(&dir.path().join("river.buckets")).unwrap();
        let reports = diagnose_cluster_dir(dir.path()).unwrap();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].bucket_count, 3);
    }
}
