//! Diagnostics for analyzing bucket files produced by the clustering pipeline.
//!
//! Reads `.buckets` files from a directory and produces per-street reports on
//! bucket count, entry distribution, and size uniformity.

use std::path::Path;

use super::bucket_file::BucketFile;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_v2::bucket_file::BucketFileHeader;
    use crate::blueprint_v2::Street;

    fn make_test_bucket_file(bucket_count: u16, buckets: Vec<u16>) -> BucketFile {
        #[allow(clippy::cast_possible_truncation)]
        BucketFile {
            header: BucketFileHeader {
                street: Street::River,
                bucket_count,
                board_count: 1,
                combos_per_board: buckets.len() as u16,
            },
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
    fn diagnose_dir_with_files() {
        let dir = tempfile::tempdir().unwrap();
        let bf = make_test_bucket_file(3, vec![0, 1, 2, 0, 1, 2]);
        bf.save(&dir.path().join("river.buckets")).unwrap();
        let reports = diagnose_cluster_dir(dir.path()).unwrap();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].bucket_count, 3);
    }
}
