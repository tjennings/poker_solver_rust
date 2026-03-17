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

use crate::poker::Card;

use super::bucket_file::BucketFile;
use super::cluster_pipeline::{
    build_deck, canonical_key, compute_board_equities, enumerate_combos, sample_boards,
    sample_n_card_boards,
};
use super::Street;

use crate::abstraction::isomorphism::CanonicalBoard;
use crate::showdown_equity;

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

/// Audit a bucket file by sampling boards and computing showdown equity.
///
/// For each sampled board, computes equity for all 1326 combos and
/// groups by bucket assignment. Reports per-bucket equity distribution.
///
/// Handles all streets: river (5 cards), turn (4 cards), flop (3 cards),
/// and preflop (0 cards — samples 5-card runouts for equity).
///
/// # Arguments
/// * `bf` — The bucket file to audit.
/// * `num_sample_boards` — How many boards to sample for equity computation.
/// * `seed` — RNG seed for board sampling.
#[must_use]
pub fn audit_bucket_equity(bf: &BucketFile, num_sample_boards: usize, seed: u64) -> EquityAuditReport {
    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    let bucket_count = bf.header.bucket_count;
    let board_map = bf.board_index_map();

    let card_count = match bf.header.street {
        Street::River => 5,
        Street::Turn => 4,
        Street::Flop => 3,
        Street::Preflop => 5, // sample full runouts for equity
    };

    // Collect (equity, bucket_id) pairs across all sampled boards.
    // Canonicalize each sampled board and look up its index in the bucket file.
    let equity_bucket_pairs: Vec<(f64, u16)> = if card_count == 5 {
        // River (or preflop with full runout): use optimized batch equity
        let boards = sample_boards(&deck, num_sample_boards, seed);
        boards
            .par_iter()
            .flat_map_iter(|&board| {
                let board_idx = canonicalize_and_lookup(&board, &board_map);
                let Some(idx) = board_idx else {
                    return Vec::new();
                };
                compute_board_equities(board, &combos)
                    .into_iter()
                    .enumerate()
                    .filter_map(move |(combo_idx, eq_opt)| {
                        let eq = eq_opt?;
                        #[allow(clippy::cast_possible_truncation)]
                        let bucket = bf.get_bucket(idx, combo_idx as u16);
                        Some((eq, bucket))
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    } else {
        // Turn/Flop: sample N-card boards, compute equity per combo
        let boards = sample_n_card_boards(&deck, card_count, num_sample_boards, seed);
        boards
            .par_iter()
            .flat_map_iter(|board| {
                let board_arr: Vec<Card> = board.clone();
                let board_idx = canonicalize_and_lookup_slice(&board_arr, &board_map);
                let Some(idx) = board_idx else {
                    return Vec::new();
                };
                combos
                    .iter()
                    .enumerate()
                    .filter_map(move |(combo_idx, &combo)| {
                        // Skip combos that share cards with the board
                        if board_arr.iter().any(|&bc| bc == combo[0] || bc == combo[1]) {
                            return None;
                        }
                        let eq = showdown_equity::compute_equity(combo, &board_arr);
                        #[allow(clippy::cast_possible_truncation)]
                        let bucket = bf.get_bucket(idx, combo_idx as u16);
                        Some((eq, bucket))
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    };

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

/// Canonicalize a 5-card board and look up its index in the board map.
fn canonicalize_and_lookup(
    board: &[Card; 5],
    board_map: &rustc_hash::FxHashMap<super::bucket_file::PackedBoard, u32>,
) -> Option<u32> {
    CanonicalBoard::from_cards(&board.to_vec())
        .ok()
        .map(|cb| canonical_key(&cb.cards))
        .and_then(|key| board_map.get(&key).copied())
}

/// Canonicalize a board of any size and look up its index in the board map.
fn canonicalize_and_lookup_slice(
    board: &[Card],
    board_map: &rustc_hash::FxHashMap<super::bucket_file::PackedBoard, u32>,
) -> Option<u32> {
    CanonicalBoard::from_cards(board)
        .ok()
        .map(|cb| canonical_key(&cb.cards))
        .and_then(|key| board_map.get(&key).copied())
}

/// Per-bucket transition consistency statistics.
#[derive(Debug)]
pub struct BucketTransitionStats {
    pub bucket_id: u16,
    /// Number of combos observed in this bucket.
    pub count: usize,
    /// Mean EMD from each combo's turn-bucket histogram to the bucket centroid.
    pub mean_emd_to_centroid: f64,
    /// Max EMD from any combo to the centroid.
    pub max_emd_to_centroid: f64,
    /// Number of distinct turn buckets that combos in this bucket transition to.
    pub distinct_turn_buckets: usize,
    /// Entropy of the aggregated turn-bucket distribution (bits).
    pub transition_entropy: f64,
}

/// Report on transition consistency for a street's clustering.
#[derive(Debug)]
pub struct TransitionConsistencyReport {
    pub from_street: String,
    pub to_street: String,
    pub bucket_count: u16,
    pub sample_boards: usize,
    pub buckets: Vec<BucketTransitionStats>,
    pub mean_emd: f64,
    pub max_emd: f64,
}

impl TransitionConsistencyReport {
    /// Format the report as a human-readable string.
    #[must_use]
    pub fn summary(&self) -> String {
        use std::fmt::Write;
        let non_empty: Vec<_> = self.buckets.iter().filter(|b| b.count > 0).collect();
        let mut s = format!(
            "{} → {}: {} buckets, {} sample boards\n  Mean EMD to centroid: {:.4}, max: {:.4}\n  Per-bucket (sorted by mean EMD):",
            self.from_street, self.to_street, non_empty.len(), self.sample_boards,
            self.mean_emd, self.max_emd,
        );
        let mut sorted: Vec<_> = non_empty.iter().collect();
        sorted.sort_by(|a, b| {
            a.mean_emd_to_centroid
                .partial_cmp(&b.mean_emd_to_centroid)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for b in &sorted {
            let _ = write!(
                s,
                "\n    bucket {:>3}: n={:<6} emd={:.4}  max_emd={:.4}  distinct={:<4} entropy={:.2}",
                b.bucket_id, b.count, b.mean_emd_to_centroid, b.max_emd_to_centroid,
                b.distinct_turn_buckets, b.transition_entropy,
            );
        }
        s
    }
}

/// Audit transition consistency: do combos in the same bucket produce similar
/// next-street bucket distributions?
///
/// For each sampled board, deals all possible next-street cards, looks up the
/// next-street bucket for each combo, and builds a per-combo histogram. Then
/// measures how similar the histograms are within each bucket using EMD to
/// the bucket's centroid histogram.
///
/// Works for flop→turn and turn→river transitions.
#[must_use]
pub fn audit_transition_consistency(
    current_bf: &BucketFile,
    next_bf: &BucketFile,
    num_sample_boards: usize,
    seed: u64,
) -> TransitionConsistencyReport {
    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    let current_map = current_bf.board_index_map();
    let next_map = next_bf.board_index_map();
    let current_k = current_bf.header.bucket_count as usize;
    let next_k = next_bf.header.bucket_count as usize;

    let card_count = match current_bf.header.street {
        Street::Flop => 3,
        Street::Turn => 4,
        _ => panic!("transition consistency audit requires flop or turn"),
    };

    let boards = sample_n_card_boards(&deck, card_count, num_sample_boards, seed);

    // For each bucket, collect per-combo histograms (normalized).
    // Each histogram is a Vec<f64> of length next_k.
    let mut bucket_histograms: Vec<Vec<Vec<f64>>> = vec![Vec::new(); current_k];

    for board in &boards {
        // Look up this board in the current bucket file
        let Some(current_board_idx) = canonicalize_and_lookup_slice(board, &current_map) else {
            continue;
        };

        // Find remaining cards (not on the board)
        let remaining: Vec<Card> = deck
            .iter()
            .filter(|c| !board.contains(c))
            .copied()
            .collect();

        // For each combo, build histogram over next-street buckets
        for (combo_idx, combo) in combos.iter().enumerate() {
            // Skip combos blocked by the board
            if board.iter().any(|&bc| bc == combo[0] || bc == combo[1]) {
                continue;
            }

            #[allow(clippy::cast_possible_truncation)]
            let current_bucket = current_bf.get_bucket(current_board_idx, combo_idx as u16) as usize;
            if current_bucket >= current_k {
                continue;
            }

            // Deal each possible next card and look up the next-street bucket
            let mut histogram = vec![0_u32; next_k];
            let mut total = 0_u32;

            for &next_card in &remaining {
                // Skip if next card overlaps with combo
                if next_card == combo[0] || next_card == combo[1] {
                    continue;
                }

                let mut next_board: Vec<Card> = board.clone();
                next_board.push(next_card);

                if let Some(next_board_idx) = canonicalize_and_lookup_slice(&next_board, &next_map) {
                    #[allow(clippy::cast_possible_truncation)]
                    let next_bucket = next_bf.get_bucket(next_board_idx, combo_idx as u16) as usize;
                    if next_bucket < next_k {
                        histogram[next_bucket] += 1;
                        total += 1;
                    }
                }
            }

            if total == 0 {
                continue;
            }

            // Normalize to probability distribution
            #[allow(clippy::cast_precision_loss)]
            let norm_hist: Vec<f64> = histogram.iter().map(|&c| c as f64 / total as f64).collect();
            bucket_histograms[current_bucket].push(norm_hist);
        }
    }

    // Compute per-bucket statistics
    let bucket_stats: Vec<BucketTransitionStats> = bucket_histograms
        .iter()
        .enumerate()
        .map(|(id, hists)| {
            if hists.is_empty() {
                #[allow(clippy::cast_possible_truncation)]
                return BucketTransitionStats {
                    bucket_id: id as u16,
                    count: 0,
                    mean_emd_to_centroid: 0.0,
                    max_emd_to_centroid: 0.0,
                    distinct_turn_buckets: 0,
                    transition_entropy: 0.0,
                };
            }

            // Compute centroid (mean histogram)
            let dim = next_k;
            let mut centroid = vec![0.0_f64; dim];
            for h in hists {
                for (j, &v) in h.iter().enumerate() {
                    centroid[j] += v;
                }
            }
            #[allow(clippy::cast_precision_loss)]
            let inv = 1.0 / hists.len() as f64;
            for v in &mut centroid {
                *v *= inv;
            }

            // EMD from each histogram to centroid
            let emds: Vec<f64> = hists
                .iter()
                .map(|h| emd_1d(h, &centroid))
                .collect();
            let mean_emd = emds.iter().sum::<f64>() / emds.len() as f64;
            let max_emd = emds.iter().fold(0.0_f64, |a, &b| a.max(b));

            // Distinct turn buckets and entropy from aggregated centroid
            let distinct = centroid.iter().filter(|&&v| v > 0.0).count();
            let entropy: f64 = centroid
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| -p * p.log2())
                .sum();

            #[allow(clippy::cast_possible_truncation)]
            BucketTransitionStats {
                bucket_id: id as u16,
                count: hists.len(),
                mean_emd_to_centroid: mean_emd,
                max_emd_to_centroid: max_emd,
                distinct_turn_buckets: distinct,
                transition_entropy: entropy,
            }
        })
        .collect();

    let non_empty: Vec<_> = bucket_stats.iter().filter(|b| b.count > 0).collect();
    let mean_emd = if non_empty.is_empty() {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        { non_empty.iter().map(|b| b.mean_emd_to_centroid).sum::<f64>() / non_empty.len() as f64 }
    };
    let max_emd = non_empty
        .iter()
        .map(|b| b.mean_emd_to_centroid)
        .fold(0.0_f64, f64::max);

    let from_street = format!("{:?}", current_bf.header.street);
    let to_street = format!("{:?}", next_bf.header.street);

    TransitionConsistencyReport {
        from_street,
        to_street,
        bucket_count: current_bf.header.bucket_count,
        sample_boards: num_sample_boards,
        buckets: bucket_stats,
        mean_emd: mean_emd,
        max_emd: max_emd,
    }
}

/// 1-D Earth Mover's Distance between two probability distributions.
///
/// For 1-D histograms, EMD equals the L1 norm of the cumulative difference.
fn emd_1d(a: &[f64], b: &[f64]) -> f64 {
    let mut cum = 0.0_f64;
    let mut total = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        cum += x - y;
        total += cum.abs();
    }
    total
}

/// Audit all bucket files in a directory.
///
/// Samples `num_boards` random boards, computes equity, and reports
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

/// A pair of buckets and their inter-centroid EMD.
#[derive(Debug, Clone)]
pub struct CentroidPairEmd {
    pub bucket_a: u16,
    pub bucket_b: u16,
    pub emd: f64,
}

/// Report on inter-centroid EMD distances.
#[derive(Debug)]
pub struct CentroidEmdReport {
    pub num_buckets: usize,
    pub pairwise_emd: Vec<CentroidPairEmd>,
    pub min_emd: f64,
    pub max_emd: f64,
    pub mean_emd: f64,
}

impl CentroidEmdReport {
    #[must_use]
    pub fn summary(&self) -> String {
        use std::fmt::Write;
        let mut s = format!(
            "Centroid EMD: {} buckets, {} pairs\n  min={:.4}, max={:.4}, mean={:.4}\n  Closest pairs:",
            self.num_buckets, self.pairwise_emd.len(), self.min_emd, self.max_emd, self.mean_emd,
        );
        let mut sorted: Vec<_> = self.pairwise_emd.iter().collect();
        sorted.sort_by(|a, b| a.emd.partial_cmp(&b.emd).unwrap_or(std::cmp::Ordering::Equal));
        for pair in sorted.iter().take(5) {
            let _ = write!(s, "\n    bucket {} <-> {}: EMD={:.4}", pair.bucket_a, pair.bucket_b, pair.emd);
        }
        s
    }
}

/// Compute pairwise EMD between reconstructed centroids.
///
/// Reconstructs centroids by averaging feature vectors within each bucket,
/// then computes EMD between all pairs.
#[must_use]
pub fn centroid_emd_report(
    features: &[Vec<f64>],
    assignments: &[u16],
    k: usize,
) -> CentroidEmdReport {
    use super::clustering::emd;

    let dim = features.first().map_or(0, Vec::len);
    let mut centroids = vec![vec![0.0_f64; dim]; k];
    let mut counts = vec![0_usize; k];

    for (i, feat) in features.iter().enumerate() {
        let ci = assignments[i] as usize;
        if ci < k {
            counts[ci] += 1;
            for (j, &val) in feat.iter().enumerate() {
                centroids[ci][j] += val;
            }
        }
    }

    for ci in 0..k {
        if counts[ci] > 0 {
            #[allow(clippy::cast_precision_loss)]
            let inv = 1.0 / counts[ci] as f64;
            for v in &mut centroids[ci] {
                *v *= inv;
            }
        }
    }

    let mut pairwise = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            let d = emd(&centroids[i], &centroids[j]);
            #[allow(clippy::cast_possible_truncation)]
            pairwise.push(CentroidPairEmd {
                bucket_a: i as u16,
                bucket_b: j as u16,
                emd: d,
            });
        }
    }

    let min = pairwise.iter().map(|p| p.emd).fold(f64::INFINITY, f64::min);
    let max = pairwise.iter().map(|p| p.emd).fold(0.0_f64, f64::max);
    #[allow(clippy::cast_precision_loss)]
    let mean = if pairwise.is_empty() {
        0.0
    } else {
        pairwise.iter().map(|p| p.emd).sum::<f64>() / pairwise.len() as f64
    };

    CentroidEmdReport {
        num_buckets: k,
        pairwise_emd: pairwise,
        min_emd: if min.is_infinite() { 0.0 } else { min },
        max_emd: max,
        mean_emd: mean,
    }
}

/// Audit river bucket assignments against cfvnet training data.
///
/// Streams cfvnet bin files, computes equity from CFVs, looks up the assigned
/// bucket, and reports per-bucket equity statistics. Samples every `sample_rate`-th
/// record for speed.
///
/// # Errors
///
/// Returns an error if bin files cannot be read or the bucket file has no matching boards.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn audit_cfvnet_buckets(
    cfvnet_dir: &Path,
    bf: &BucketFile,
    sample_rate: usize,
    progress: impl Fn(f64),
) -> Result<EquityAuditReport, Box<dyn std::error::Error>> {
    use super::cluster_pipeline::{
        build_cfvnet_to_core_combo_map, cfvnet_card_to_core, canonical_key,
        collect_bin_files, read_cfv_as_equity, record_size_for_board,
        CFVNET_RIVER_RECORD_SIZE, CFV_FIELD_OFFSET,
    };

    let combo_map = build_cfvnet_to_core_combo_map();
    let bin_files = collect_bin_files(cfvnet_dir)?;
    let board_map = bf.board_index_map();
    let bucket_count = bf.header.bucket_count as usize;
    let combos_per_board = bf.header.combos_per_board as usize;

    // Accumulators per bucket
    let mut sum = vec![0.0f64; bucket_count];
    let mut sq_sum = vec![0.0f64; bucket_count];
    let mut min_eq = vec![f64::INFINITY; bucket_count];
    let mut max_eq = vec![f64::NEG_INFINITY; bucket_count];
    let mut count = vec![0u64; bucket_count];

    let mut records_seen = 0u64;

    for (fi, path) in bin_files.iter().enumerate() {
        let data = std::fs::read(path)?;
        let mut offset = 0;
        while offset + CFVNET_RIVER_RECORD_SIZE <= data.len() {
            if data[offset] != 5 {
                offset += record_size_for_board(data[offset]);
                continue;
            }

            records_seen += 1;
            if records_seen % sample_rate as u64 != 0 {
                offset += CFVNET_RIVER_RECORD_SIZE;
                continue;
            }

            let board_cards: Vec<Card> = (0..5)
                .map(|i| cfvnet_card_to_core(data[offset + 1 + i]))
                .collect();
            let packed = canonical_key(&board_cards);

            if let Some(&board_idx) = board_map.get(&packed) {
                let cfv_offset = offset + CFV_FIELD_OFFSET;
                let mask_offset = cfv_offset + 1326 * 4;

                for i in 0..1326 {
                    if data[mask_offset + i] != 0 {
                        let equity = read_cfv_as_equity(&data, cfv_offset, i);
                        let core_combo = combo_map[i] as usize;
                        let bucket = bf.buckets[board_idx as usize * combos_per_board + core_combo] as usize;
                        if bucket < bucket_count {
                            sum[bucket] += equity;
                            sq_sum[bucket] += equity * equity;
                            if equity < min_eq[bucket] { min_eq[bucket] = equity; }
                            if equity > max_eq[bucket] { max_eq[bucket] = equity; }
                            count[bucket] += 1;
                        }
                    }
                }
            }

            offset += CFVNET_RIVER_RECORD_SIZE;
        }
        progress((fi + 1) as f64 / bin_files.len() as f64);
    }

    let bucket_stats: Vec<BucketEquityStats> = (0..bucket_count)
        .map(|b| {
            let n = count[b];
            if n == 0 {
                return BucketEquityStats {
                    bucket_id: b as u16,
                    count: 0,
                    mean_equity: 0.0,
                    std_dev: 0.0,
                    min_equity: 0.0,
                    max_equity: 0.0,
                };
            }
            let mean = sum[b] / n as f64;
            let variance = (sq_sum[b] / n as f64 - mean * mean).max(0.0);
            BucketEquityStats {
                bucket_id: b as u16,
                count: n as usize,
                mean_equity: mean,
                std_dev: variance.sqrt(),
                min_equity: min_eq[b],
                max_equity: max_eq[b],
            }
        })
        .collect();

    let non_empty: Vec<_> = bucket_stats.iter().filter(|b| b.count > 0).collect();
    let mean_std = if non_empty.is_empty() {
        0.0
    } else {
        non_empty.iter().map(|b| b.std_dev).sum::<f64>() / non_empty.len() as f64
    };
    let max_std = non_empty.iter().map(|b| b.std_dev).fold(0.0_f64, f64::max);

    Ok(EquityAuditReport {
        street: "River (cfvnet)".to_string(),
        bucket_count: bucket_count as u16,
        sample_boards: records_seen as usize / sample_rate.max(1),
        buckets: bucket_stats,
        mean_intra_bucket_std: mean_std,
        max_intra_bucket_std: max_std,
    })
}

/// A sample hand from a specific bucket.
#[derive(Debug, Clone)]
pub struct BucketHandSample {
    pub bucket: u16,
    pub board_idx: u32,
    pub combo_idx: u16,
    pub board_cards: Vec<Card>,
    pub hole_cards: [Card; 2],
}

impl BucketHandSample {
    #[must_use]
    pub fn display(&self) -> String {
        let board_str: Vec<String> = self.board_cards.iter().map(|c| format!("{c}")).collect();
        format!(
            "  [{} {}] on [{}]",
            self.hole_cards[0], self.hole_cards[1], board_str.join(" "),
        )
    }
}

/// Sample up to `max_samples` hands from a specific bucket.
///
/// Scans the bucket file for entries matching `target_bucket`, collects
/// them, and returns a random subset (seeded for reproducibility).
#[must_use]
pub fn sample_hands_for_bucket(
    bf: &BucketFile,
    target_bucket: u16,
    max_samples: usize,
    seed: u64,
) -> Vec<BucketHandSample> {
    use rand::prelude::*;
    use rand::rngs::StdRng;

    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    let board_count = bf.header.board_count;
    let combos_per_board = bf.header.combos_per_board as usize;

    let mut candidates: Vec<(u32, u16)> = Vec::new();
    for board_idx in 0..board_count {
        for combo_idx in 0..combos_per_board {
            #[allow(clippy::cast_possible_truncation)]
            let bucket = bf.get_bucket(board_idx, combo_idx as u16);
            if bucket == target_bucket {
                #[allow(clippy::cast_possible_truncation)]
                candidates.push((board_idx, combo_idx as u16));
            }
        }
    }

    let mut rng = StdRng::seed_from_u64(seed);
    candidates.shuffle(&mut rng);
    candidates.truncate(max_samples);

    candidates
        .into_iter()
        .map(|(board_idx, combo_idx)| {
            let board_cards = if (board_idx as usize) < bf.boards.len() {
                let num_cards = match bf.header.street {
                    Street::Preflop => 0,
                    Street::Flop => 3,
                    Street::Turn => 4,
                    Street::River => 5,
                };
                bf.boards[board_idx as usize].to_cards(num_cards)
            } else {
                Vec::new()
            };

            let hole_cards = if (combo_idx as usize) < combos.len() {
                combos[combo_idx as usize]
            } else {
                [deck[0], deck[1]]
            };

            BucketHandSample {
                bucket: target_bucket,
                board_idx,
                combo_idx,
                board_cards,
                hole_cards,
            }
        })
        .collect()
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

    #[test]
    fn centroid_emd_matrix_basic() {
        let features = vec![
            vec![0.9, 0.1, 0.0],
            vec![0.8, 0.2, 0.0],
            vec![0.0, 0.9, 0.1],
            vec![0.0, 0.8, 0.2],
            vec![0.0, 0.1, 0.9],
            vec![0.0, 0.0, 1.0],
        ];
        let assignments: Vec<u16> = vec![0, 0, 1, 1, 2, 2];
        let report = centroid_emd_report(&features, &assignments, 3);
        assert_eq!(report.num_buckets, 3);
        assert_eq!(report.pairwise_emd.len(), 3);
        for pair in &report.pairwise_emd {
            assert!(pair.emd > 0.0, "EMD should be positive: {:?}", pair);
        }
        assert!(report.min_emd > 0.0);
        assert!(report.max_emd > report.min_emd);
    }

    #[test]
    fn centroid_emd_single_bucket() {
        let features = vec![vec![0.5, 0.5], vec![0.6, 0.4]];
        let assignments: Vec<u16> = vec![0, 0];
        let report = centroid_emd_report(&features, &assignments, 1);
        assert_eq!(report.num_buckets, 1);
        assert!(report.pairwise_emd.is_empty());
        assert_eq!(report.min_emd, 0.0);
        assert_eq!(report.max_emd, 0.0);
        assert_eq!(report.mean_emd, 0.0);
    }

    #[test]
    fn centroid_emd_empty_features() {
        let features: Vec<Vec<f64>> = vec![];
        let assignments: Vec<u16> = vec![];
        let report = centroid_emd_report(&features, &assignments, 2);
        assert_eq!(report.num_buckets, 2);
        // Still produces pairs but EMD between zero-vectors is 0
        assert_eq!(report.pairwise_emd.len(), 1);
    }

    #[test]
    fn centroid_emd_summary_format() {
        let features = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let assignments: Vec<u16> = vec![0, 1];
        let report = centroid_emd_report(&features, &assignments, 2);
        let s = report.summary();
        assert!(s.contains("2 buckets"), "summary missing bucket count: {s}");
        assert!(s.contains("1 pairs"), "summary missing pair count: {s}");
        assert!(s.contains("Closest pairs:"), "summary missing header: {s}");
        assert!(s.contains("bucket 0 <-> 1"), "summary missing pair: {s}");
    }

    #[test]
    fn sample_hands_for_bucket_basic() {
        let deck = build_deck();

        let mut buckets = vec![1_u16; 1326];
        for i in 0..10 {
            buckets[i] = 0;
        }
        let board_cards = [deck[10], deck[20], deck[30], deck[40], deck[50]];
        let bf = BucketFile {
            header: BucketFileHeader {
                street: Street::River,
                bucket_count: 2,
                board_count: 1,
                combos_per_board: 1326,
                version: 2,
            },
            boards: vec![PackedBoard::from_cards(&board_cards)],
            buckets,
        };

        let samples = sample_hands_for_bucket(&bf, 0, 5, 42);
        assert_eq!(samples.len(), 5);
        for sample in &samples {
            assert_eq!(sample.bucket, 0);
        }
    }
}
