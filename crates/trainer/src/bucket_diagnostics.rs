//! Standalone EHS bucket diagnostic tool.
//!
//! Computes (or loads from cache) hand buckets for a postflop abstraction config,
//! runs 8 quality checks, and prints a pass/fail report.

use std::path::Path;

use poker_solver_core::hands::{CanonicalHand, all_hands};
use poker_solver_core::preflop::board_abstraction::{BoardAbstraction, BoardAbstractionConfig};
use poker_solver_core::preflop::hand_buckets::{
    BucketEquity, HandBucketMapping, StreetEquity, compute_all_flop_features, cluster_per_texture,
    bucket_ehs_centroids, build_bucket_equity_from_centroids,
};
use poker_solver_core::preflop::postflop_model::PostflopModelConfig;
use poker_solver_core::preflop::{abstraction_cache, ehs::EhsFeatures};

/// Status of a single diagnostic check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    Pass,
    Fail,
    Warn,
    Info,
}

/// Result of a single diagnostic check.
#[derive(Debug)]
pub struct CheckResult {
    pub name: &'static str,
    pub status: CheckStatus,
    pub score: f64,
    pub threshold: f64,
    pub detail: String,
}

/// Run all bucket diagnostics, returning true if all non-info checks pass.
pub fn run(config: &PostflopModelConfig, cache_dir: &Path, json: bool) -> bool {
    let num_buckets = config.num_hand_buckets_flop;
    let num_textures = config.num_flop_textures;

    eprintln!("Loading or building abstraction...");
    let (board, buckets, street_equity) = load_or_build(config, cache_dir);
    let equity = &street_equity.flop;

    let hands: Vec<CanonicalHand> = all_hands().collect();
    let flop_samples: Vec<Vec<_>> = board.prototype_flops.iter().map(|f| vec![*f]).collect();

    eprintln!("Computing EHS features ({} hands x {} textures)...", hands.len(), flop_samples.len());
    let features = compute_all_flop_features(&hands, &flop_samples, &|_| {});
    let assignments = &buckets.flop_buckets;
    let centroids = bucket_ehs_centroids(&features, assignments, num_buckets as usize);

    let nb = num_buckets as usize;
    let nt = num_textures as usize;
    let results = vec![
        check_bucket_sizes(assignments, num_buckets, nt),
        check_silhouette(&features, assignments, &centroids, nb),
        check_ehs_overlap(&features, assignments, &centroids, nb),
        check_monotonicity(&centroids, &equity),
        check_known_hands(&hands, assignments, &centroids, nb),
        check_cross_texture(&hands, assignments, &centroids, nb, nt),
        check_equity_coherence(&equity),
        check_draw_buckets(&features, assignments, nb),
    ];

    if json {
        print_json(&results);
    } else {
        print_text(&results, num_buckets, num_textures);
    }

    results.iter().all(|r| r.status != CheckStatus::Fail)
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

fn load_or_build(
    config: &PostflopModelConfig,
    cache_base: &Path,
) -> (BoardAbstraction, HandBucketMapping, StreetEquity) {
    let key = abstraction_cache::cache_key(config, false);
    if let Some(cached) = abstraction_cache::load(cache_base, &key) {
        eprintln!("Cache hit: {}", abstraction_cache::cache_dir(cache_base, &key).display());
        return cached;
    }

    eprintln!("Cache miss — building board abstraction...");
    let ba_config = BoardAbstractionConfig {
        num_flop_textures: config.num_flop_textures,
        num_turn_transitions: config.num_turn_transitions,
        num_river_transitions: config.num_river_transitions,
        kmeans_max_iter: 50,
    };
    let board = BoardAbstraction::build(&ba_config).expect("board abstraction build failed");

    let hands: Vec<CanonicalHand> = all_hands().collect();
    let flop_samples: Vec<Vec<_>> = board.prototype_flops.iter().map(|f| vec![*f]).collect();

    eprintln!("Computing EHS features...");
    let features = compute_all_flop_features(&hands, &flop_samples, &|_| {});
    let num_buckets = config.num_hand_buckets_flop;
    let num_textures = flop_samples.len();

    eprintln!("Clustering into {} buckets x {} textures...", num_buckets, num_textures);
    let flop_assignments = cluster_per_texture(&features, num_buckets, num_textures);

    let centroids = bucket_ehs_centroids(&features, &flop_assignments, num_buckets as usize);
    let flop_equity = build_bucket_equity_from_centroids(&centroids);
    let street_equity = StreetEquity {
        flop: flop_equity,
        turn: build_bucket_equity_from_centroids(&centroids),
        river: build_bucket_equity_from_centroids(&centroids),
    };

    let buckets = HandBucketMapping {
        num_flop_buckets: num_buckets,
        num_turn_buckets: num_buckets,
        num_river_buckets: num_buckets,
        flop_buckets: flop_assignments,
        turn_buckets: vec![],
        river_buckets: vec![],
    };

    if let Err(e) = abstraction_cache::save(cache_base, &key, &board, &buckets, &street_equity) {
        eprintln!("Warning: failed to save cache: {e}");
    }

    (board, buckets, street_equity)
}

// ---------------------------------------------------------------------------
// Check 1: Bucket sizes
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn check_bucket_sizes(
    assignments: &[Vec<u16>],
    num_buckets: u16,
    num_textures: usize,
) -> CheckResult {
    let mut counts = vec![0usize; num_buckets as usize];
    for hand_a in assignments {
        for &bucket in hand_a {
            counts[bucket as usize] += 1;
        }
    }

    let total = counts.iter().sum::<usize>() as f64;
    let avg = total / counts.len() as f64;
    let variance = counts.iter().map(|&c| (c as f64 - avg).powi(2)).sum::<f64>() / counts.len() as f64;
    let stdev = variance.sqrt();
    let cv = if avg > 0.0 { stdev / avg } else { 0.0 };
    let empty_count = counts.iter().filter(|&&c| c == 0).count();
    let empty_pct = empty_count as f64 / counts.len() as f64 * 100.0;

    let pass = empty_pct < 10.0 && cv < 1.5;
    CheckResult {
        name: "Bucket sizes",
        status: if pass { CheckStatus::Pass } else { CheckStatus::Fail },
        score: cv,
        threshold: 1.5,
        detail: format!(
            "avg {:.1}, stdev {:.1} (cv={:.2}), empty {:.1}% ({} textures)",
            avg, stdev, cv, empty_pct, num_textures
        ),
    }
}

// ---------------------------------------------------------------------------
// Check 2: Silhouette score (per-texture, since k-means is independent per texture)
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn check_silhouette(
    features: &[Vec<EhsFeatures>],
    assignments: &[Vec<u16>],
    _centroids: &[f64],
    num_buckets: usize,
) -> CheckResult {
    let num_textures = features.first().map_or(0, Vec::len);
    let mut silhouette_sum = 0.0f64;
    let mut count = 0u64;

    for tex_id in 0..num_textures {
        // Compute 3D centroids for this texture only.
        let tex_centroids = compute_texture_centroids(features, assignments, num_buckets, tex_id);

        for (hand_idx, hand_feats) in features.iter().enumerate() {
            let feat = &hand_feats[tex_id];
            let bucket = assignments[hand_idx][tex_id] as usize;
            let a = sq_dist_3d(feat, &tex_centroids[bucket]);

            let b = (0..num_buckets)
                .filter(|&bi| bi != bucket)
                .map(|bi| sq_dist_3d(feat, &tex_centroids[bi]))
                .fold(f64::MAX, f64::min);

            let max_ab = a.max(b);
            let s = if max_ab > 1e-15 { (b - a) / max_ab } else { 0.0 };
            silhouette_sum += s;
            count += 1;
        }
    }

    let mean = if count > 0 { silhouette_sum / count as f64 } else { 0.0 };
    let threshold = 0.10;
    CheckResult {
        name: "Silhouette score",
        status: if mean > threshold { CheckStatus::Pass } else if mean > 0.0 { CheckStatus::Warn } else { CheckStatus::Fail },
        score: mean,
        threshold,
        detail: format!("{:.3} (threshold: {:.2})", mean, threshold),
    }
}

/// Compute 3D feature centroids for a single texture.
#[allow(clippy::cast_precision_loss)]
fn compute_texture_centroids(
    features: &[Vec<EhsFeatures>],
    assignments: &[Vec<u16>],
    num_buckets: usize,
    tex_id: usize,
) -> Vec<[f64; 3]> {
    let mut sums = vec![[0.0f64; 3]; num_buckets];
    let mut counts = vec![0u64; num_buckets];

    for (hand_idx, hand_feats) in features.iter().enumerate() {
        let b = assignments[hand_idx][tex_id] as usize;
        let feat = &hand_feats[tex_id];
        for d in 0..3 {
            sums[b][d] += feat[d];
        }
        counts[b] += 1;
    }

    sums.iter()
        .zip(&counts)
        .map(|(s, &c)| {
            if c > 0 {
                [s[0] / c as f64, s[1] / c as f64, s[2] / c as f64]
            } else {
                [0.5, 0.0, 0.0]
            }
        })
        .collect()
}

fn sq_dist_3d(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

// ---------------------------------------------------------------------------
// Check 3: EHS overlap (per-texture, averaged)
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn check_ehs_overlap(
    features: &[Vec<EhsFeatures>],
    assignments: &[Vec<u16>],
    _centroids: &[f64],
    num_buckets: usize,
) -> CheckResult {
    let num_textures = features.first().map_or(0, Vec::len);
    let mut total_overlap = 0.0f64;
    let mut total_pairs = 0usize;

    for tex_id in 0..num_textures {
        // Compute per-bucket EHS range and centroid for this texture.
        let (mins, maxs, tex_centroids) =
            bucket_ehs_ranges_for_texture(features, assignments, num_buckets, tex_id);

        // Sort buckets by centroid EHS.
        let mut sorted: Vec<usize> = (0..num_buckets).collect();
        sorted.sort_by(|&a, &b| tex_centroids[a].partial_cmp(&tex_centroids[b]).unwrap());

        for w in sorted.windows(2) {
            let (a, b) = (w[0], w[1]);
            let ov = (maxs[a].min(maxs[b]) - mins[a].max(mins[b])).max(0.0);
            total_overlap += ov;
            total_pairs += 1;
        }
    }

    let mean_overlap = if total_pairs > 0 { total_overlap / total_pairs as f64 } else { 0.0 };

    // Buckets separate on 3D features (EHS, ppot, npot) so 1D EHS overlap is expected.
    let threshold = 0.30;
    CheckResult {
        name: "EHS overlap",
        status: if mean_overlap < threshold { CheckStatus::Pass } else { CheckStatus::Warn },
        score: mean_overlap,
        threshold,
        detail: format!("{:.4} (threshold: {:.2})", mean_overlap, threshold),
    }
}

/// Compute per-bucket EHS min/max/centroid for a single texture.
fn bucket_ehs_ranges_for_texture(
    features: &[Vec<EhsFeatures>],
    assignments: &[Vec<u16>],
    num_buckets: usize,
    tex_id: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut mins = vec![f64::MAX; num_buckets];
    let mut maxs = vec![f64::MIN; num_buckets];
    let mut sums = vec![0.0f64; num_buckets];
    let mut counts = vec![0u32; num_buckets];

    for (hand_idx, hand_feats) in features.iter().enumerate() {
        let bucket = assignments[hand_idx][tex_id] as usize;
        let ehs = hand_feats[tex_id][0];
        mins[bucket] = mins[bucket].min(ehs);
        maxs[bucket] = maxs[bucket].max(ehs);
        sums[bucket] += ehs;
        counts[bucket] += 1;
    }

    let centroids: Vec<f64> = sums.iter().zip(&counts)
        .map(|(&s, &c)| if c > 0 { s / f64::from(c) } else { 0.5 })
        .collect();

    // Fix empty buckets.
    for i in 0..num_buckets {
        if mins[i] > maxs[i] {
            mins[i] = 0.5;
            maxs[i] = 0.5;
        }
    }

    (mins, maxs, centroids)
}

// ---------------------------------------------------------------------------
// Check 4: Monotonicity
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn check_monotonicity(centroids: &[f64], equity: &BucketEquity) -> CheckResult {
    let n = centroids.len();
    let mut correct = 0u64;
    let mut total = 0u64;

    for i in 0..n {
        for j in (i + 1)..n {
            if (centroids[i] - centroids[j]).abs() < 1e-12 {
                continue;
            }
            total += 1;
            let stronger = if centroids[i] > centroids[j] { i } else { j };
            let weaker = if stronger == i { j } else { i };
            let eq = equity.get(stronger, weaker) as f64;
            if eq >= 0.5 {
                correct += 1;
            }
        }
    }

    let pct = if total > 0 { correct as f64 / total as f64 * 100.0 } else { 100.0 };
    let threshold = 95.0;
    CheckResult {
        name: "Monotonicity",
        status: if pct >= threshold { CheckStatus::Pass } else { CheckStatus::Fail },
        score: pct,
        threshold,
        detail: format!("{:.1}% (threshold: {:.0}%)", pct, threshold),
    }
}

// ---------------------------------------------------------------------------
// Check 5: Known hands
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn check_known_hands(
    hands: &[CanonicalHand],
    assignments: &[Vec<u16>],
    centroids: &[f64],
    num_buckets: usize,
) -> CheckResult {
    // Sort buckets by centroid EHS descending.
    let mut sorted_indices: Vec<usize> = (0..num_buckets).collect();
    sorted_indices.sort_by(|&a, &b| centroids[b].partial_cmp(&centroids[a]).unwrap());
    let bucket_rank: Vec<usize> = {
        let mut rank = vec![0usize; num_buckets];
        for (r, &b) in sorted_indices.iter().enumerate() {
            rank[b] = r;
        }
        rank
    };

    let find_hand_idx = |name: &str| -> Option<usize> {
        let target = CanonicalHand::parse(name).ok()?;
        hands.iter().position(|h| *h == target)
    };

    let avg_rank_pct = |hand_idx: usize| -> f64 {
        let tex_assignments = &assignments[hand_idx];
        let avg_rank: f64 = tex_assignments.iter()
            .map(|&b| bucket_rank[b as usize] as f64)
            .sum::<f64>() / tex_assignments.len() as f64;
        avg_rank / num_buckets as f64 * 100.0
    };

    let aa_idx = find_hand_idx("AA");
    let sevtwo_idx = find_hand_idx("72o");

    let (aa_pct, aa_ok) = match aa_idx {
        Some(idx) => {
            let pct = avg_rank_pct(idx);
            (pct, pct < 20.0)
        }
        None => (f64::NAN, false),
    };

    let (sevtwo_pct, sevtwo_ok) = match sevtwo_idx {
        Some(idx) => {
            let pct = avg_rank_pct(idx);
            (pct, pct > 70.0) // bottom 30% means rank% > 70%
        }
        None => (f64::NAN, false),
    };

    let pass = aa_ok && sevtwo_ok;
    CheckResult {
        name: "Known hands",
        status: if pass { CheckStatus::Pass } else { CheckStatus::Fail },
        score: aa_pct,
        threshold: 20.0,
        detail: format!(
            "AA avg rank {:.1}% (top 20%: {}), 72o avg rank {:.1}% (bottom 30%: {})",
            aa_pct, if aa_ok { "yes" } else { "no" },
            sevtwo_pct, if sevtwo_ok { "yes" } else { "no" }
        ),
    }
}

// ---------------------------------------------------------------------------
// Check 6: Cross-texture consistency
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn check_cross_texture(
    hands: &[CanonicalHand],
    assignments: &[Vec<u16>],
    centroids: &[f64],
    num_buckets: usize,
    num_textures: usize,
) -> CheckResult {
    let _ = hands;
    // For each bucket, determine its rank by centroid EHS (higher EHS = lower rank number).
    let mut sorted_indices: Vec<usize> = (0..num_buckets).collect();
    sorted_indices.sort_by(|&a, &b| centroids[b].partial_cmp(&centroids[a]).unwrap());
    let bucket_rank: Vec<f64> = {
        let mut rank = vec![0.0f64; num_buckets];
        for (r, &b) in sorted_indices.iter().enumerate() {
            rank[b] = r as f64;
        }
        rank
    };

    // For each hand, compute its centroid rank across textures and measure variance.
    let mut total_variance = 0.0f64;
    let mut hand_count = 0usize;

    for hand_assignments in assignments.iter() {
        if hand_assignments.is_empty() {
            continue;
        }
        let ranks: Vec<f64> = hand_assignments.iter()
            .map(|&b| bucket_rank[b as usize])
            .collect();
        let mean = ranks.iter().sum::<f64>() / ranks.len() as f64;
        let var = ranks.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / ranks.len() as f64;
        total_variance += var.sqrt();
        hand_count += 1;
    }

    let mean_stdev = if hand_count > 0 { total_variance / hand_count as f64 } else { 0.0 };
    let normalized = mean_stdev / num_buckets as f64 * 100.0;
    let _ = num_textures;

    let threshold = 20.0;
    let status = if normalized < threshold {
        if normalized > threshold * 0.9 { CheckStatus::Warn } else { CheckStatus::Pass }
    } else {
        CheckStatus::Fail
    };

    CheckResult {
        name: "Cross-texture",
        status,
        score: normalized,
        threshold,
        detail: format!("rank variance {:.1}% (threshold: {:.0}%)", normalized, threshold),
    }
}

// ---------------------------------------------------------------------------
// Check 7: Equity coherence
// ---------------------------------------------------------------------------

fn check_equity_coherence(equity: &BucketEquity) -> CheckResult {
    let n = equity.num_buckets;
    let mut max_symmetry_err = 0.0f64;
    let mut max_diag_err = 0.0f64;

    for a in 0..n {
        let diag = equity.get(a, a) as f64;
        max_diag_err = max_diag_err.max((diag - 0.5).abs());
        for b in (a + 1)..n {
            let sum = equity.get(a, b) as f64 + equity.get(b, a) as f64;
            max_symmetry_err = max_symmetry_err.max((sum - 1.0).abs());
        }
    }

    let sym_ok = max_symmetry_err < 0.02;
    let diag_ok = max_diag_err < 0.05;
    let pass = sym_ok && diag_ok;
    CheckResult {
        name: "Equity coherence",
        status: if pass { CheckStatus::Pass } else { CheckStatus::Fail },
        score: max_symmetry_err,
        threshold: 0.02,
        detail: format!(
            "symmetry err {:.4}, diag err {:.4}",
            max_symmetry_err, max_diag_err
        ),
    }
}

// ---------------------------------------------------------------------------
// Check 8: Draw buckets (info only)
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn check_draw_buckets(
    features: &[Vec<EhsFeatures>],
    assignments: &[Vec<u16>],
    num_buckets: usize,
) -> CheckResult {
    // EhsFeatures = [ehs, positive_potential, negative_potential]
    // A "draw bucket" has high positive potential (> 0.15 avg).
    let mut ppot_sums = vec![0.0f64; num_buckets];
    let mut ehs_sums = vec![0.0f64; num_buckets];
    let mut counts = vec![0u64; num_buckets];

    for (hand_idx, hand_feats) in features.iter().enumerate() {
        for (tex_id, feat) in hand_feats.iter().enumerate() {
            let bucket = assignments[hand_idx][tex_id] as usize;
            ppot_sums[bucket] += feat[1]; // positive potential
            ehs_sums[bucket] += feat[0];
            counts[bucket] += 1;
        }
    }

    let draw_threshold = 0.15;
    let mut draw_count = 0usize;
    let mut draw_ppot_sum = 0.0f64;
    let mut draw_ehs_sum = 0.0f64;

    for b in 0..num_buckets {
        if counts[b] == 0 {
            continue;
        }
        let avg_ppot = ppot_sums[b] / counts[b] as f64;
        if avg_ppot > draw_threshold {
            draw_count += 1;
            draw_ppot_sum += avg_ppot;
            draw_ehs_sum += ehs_sums[b] / counts[b] as f64;
        }
    }

    let avg_ppot = if draw_count > 0 { draw_ppot_sum / draw_count as f64 } else { 0.0 };
    let avg_ehs = if draw_count > 0 { draw_ehs_sum / draw_count as f64 } else { 0.0 };

    CheckResult {
        name: "Draw buckets",
        status: CheckStatus::Info,
        score: draw_count as f64,
        threshold: 0.0,
        detail: format!(
            "{} identified (avg ppot {:.2}, avg EHS {:.2})",
            draw_count, avg_ppot, avg_ehs
        ),
    }
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

fn status_tag(status: CheckStatus) -> &'static str {
    match status {
        CheckStatus::Pass => "[PASS]",
        CheckStatus::Fail => "[FAIL]",
        CheckStatus::Warn => "[WARN]",
        CheckStatus::Info => "[INFO]",
    }
}

fn print_text(results: &[CheckResult], num_buckets: u16, num_textures: u16) {
    println!("EHS Bucket Diagnostics ({} buckets x {} textures)", num_buckets, num_textures);
    println!("{}", "=".repeat(64));
    for r in results {
        println!("{} {:<20} {}", status_tag(r.status), r.name, r.detail);
    }
    println!("{}", "=".repeat(64));

    let checks = results.iter().filter(|r| r.status != CheckStatus::Info);
    let passed = checks.clone().filter(|r| r.status == CheckStatus::Pass).count();
    let warnings = checks.clone().filter(|r| r.status == CheckStatus::Warn).count();
    let total = checks.count();
    println!("{}/{} checks passed, {} warnings", passed, total, warnings);
}

fn print_json(results: &[CheckResult]) {
    print!("[");
    for (i, r) in results.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        let status_str = match r.status {
            CheckStatus::Pass => "pass",
            CheckStatus::Fail => "fail",
            CheckStatus::Warn => "warn",
            CheckStatus::Info => "info",
        };
        print!(
            r#"{{"name":"{}","status":"{}","score":{:.6},"threshold":{:.6},"detail":"{}"}}"#,
            r.name,
            status_str,
            r.score,
            r.threshold,
            r.detail.replace('"', "\\\"")
        );
    }
    println!("]");
}
