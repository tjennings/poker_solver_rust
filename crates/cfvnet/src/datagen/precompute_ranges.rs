//! Precompute flop entry ranges from a blueprint strategy.
//!
//! Enumerates all preflop action paths that reach the flop and computes
//! exact reach-weighted ranges for each. Each path (e.g., "raise/call",
//! "raise/3bet/call") produces a unique set of OOP/IP ranges.

use std::path::Path;

use serde::{Deserialize, Serialize};

use poker_solver_core::blueprint_v2::game_tree::{GameNode, GameTree, TreeAction};
use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::full_depth_solver::rs_poker_card_to_id;
use poker_solver_core::blueprint_v2::Street;
use poker_solver_core::hands::all_hands;
use range_solver::card::card_pair_to_index;

use super::blueprint_ranges::NUM_COMBOS;

/// All preflop terminal paths and their ranges.
#[derive(Serialize, Deserialize, Debug)]
pub struct PrecomputedRanges {
    pub paths: Vec<PreflopPath>,
}

/// A single preflop action path that reaches the flop.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PreflopPath {
    /// Human-readable label, e.g. "raise/call", "raise/3bet/call"
    pub label: String,
    /// OOP (BB) reach weights at the flop entry (1326 combos).
    pub oop_range: Vec<f32>,
    /// IP (SB/dealer) reach weights at the flop entry (1326 combos).
    pub ip_range: Vec<f32>,
    /// Number of non-zero OOP combos (> 0.01).
    pub oop_nonzero: usize,
    /// Number of non-zero IP combos (> 0.01).
    pub ip_nonzero: usize,
    /// Relative weight of this path (product of action probabilities,
    /// averaged across all combos). Higher = more common line.
    pub frequency: f64,
}

impl PrecomputedRanges {
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let file = std::fs::File::create(path).map_err(|e| format!("create: {e}"))?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, self).map_err(|e| format!("serialize: {e}"))
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
        let reader = std::io::BufReader::new(file);
        bincode::deserialize_from(reader).map_err(|e| format!("deserialize: {e}"))
    }

    /// Pick a random path weighted by frequency.
    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> Option<&PreflopPath> {
        if self.paths.is_empty() {
            return None;
        }
        let total: f64 = self.paths.iter().map(|p| p.frequency).sum();
        if total <= 0.0 {
            return Some(&self.paths[0]);
        }
        let mut draw = rng.gen_range(0.0..total);
        for path in &self.paths {
            draw -= path.frequency;
            if draw <= 0.0 {
                return Some(path);
            }
        }
        self.paths.last()
    }
}

fn action_label(action: &TreeAction) -> String {
    match action {
        TreeAction::Fold => "fold".into(),
        TreeAction::Check => "check".into(),
        TreeAction::Call => "call".into(),
        TreeAction::AllIn => "allin".into(),
        TreeAction::Bet(v) | TreeAction::Raise(v) => {
            let bb = (v / 2.0).round() as u32;
            format!("{bb}bb")
        }
    }
}

/// Enumerate all preflop paths to the flop and compute exact ranges.
pub fn compute_preflop_paths(
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    decision_map: &[u32],
) -> PrecomputedRanges {
    let mut paths = Vec::new();
    let init_oop = [1.0f64; NUM_COMBOS];
    let init_ip = [1.0f64; NUM_COMBOS];

    walk_preflop(
        strategy,
        tree,
        decision_map,
        tree.root,
        &init_oop,
        &init_ip,
        1.0,
        &mut Vec::new(),
        &mut paths,
    );

    // Sort by frequency descending.
    paths.sort_by(|a, b| b.frequency.partial_cmp(&a.frequency).unwrap_or(std::cmp::Ordering::Equal));

    PrecomputedRanges { paths }
}

fn walk_preflop(
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    decision_map: &[u32],
    node_idx: u32,
    oop_weights: &[f64; NUM_COMBOS],
    ip_weights: &[f64; NUM_COMBOS],
    path_freq: f64,
    action_labels: &mut Vec<String>,
    results: &mut Vec<PreflopPath>,
) {
    match &tree.nodes[node_idx as usize] {
        GameNode::Terminal { .. } => {
            // Hand ended (fold/all-in) before flop — skip.
        }
        GameNode::Chance { child, .. } => {
            // Reached flop — record this path.
            let label = if action_labels.is_empty() {
                "limp".to_string()
            } else {
                action_labels.join("/")
            };

            let oop_range: Vec<f32> = oop_weights.iter().map(|&w| w as f32).collect();
            let ip_range: Vec<f32> = ip_weights.iter().map(|&w| w as f32).collect();
            let oop_nonzero = oop_range.iter().filter(|&&w| w > 0.01).count();
            let ip_nonzero = ip_range.iter().filter(|&&w| w > 0.01).count();

            results.push(PreflopPath {
                label,
                oop_range,
                ip_range,
                oop_nonzero,
                ip_nonzero,
                frequency: path_freq,
            });
        }
        GameNode::Decision {
            player,
            street,
            actions,
            children,
            ..
        } => {
            if *street != Street::Preflop {
                // Somehow reached a postflop decision without a Chance node — record anyway.
                let label = action_labels.join("/");
                let oop_range: Vec<f32> = oop_weights.iter().map(|&w| w as f32).collect();
                let ip_range: Vec<f32> = ip_weights.iter().map(|&w| w as f32).collect();
                let oop_nonzero = oop_range.iter().filter(|&&w| w > 0.01).count();
                let ip_nonzero = ip_range.iter().filter(|&&w| w > 0.01).count();
                results.push(PreflopPath {
                    label,
                    oop_range,
                    ip_range,
                    oop_nonzero,
                    ip_nonzero,
                    frequency: path_freq,
                });
                return;
            }

            let dec_idx = decision_map[node_idx as usize];
            if dec_idx == u32::MAX {
                return;
            }

            for (action_idx, &child_idx) in children.iter().enumerate() {
                let mut new_oop = *oop_weights;
                let mut new_ip = *ip_weights;

                let weights = if *player == tree.dealer {
                    &mut new_ip
                } else {
                    &mut new_oop
                };

                // Compute per-combo action probability and multiply weights.
                // Also track average action frequency for path weighting.
                let mut action_freq_sum = 0.0f64;
                let mut combo_count = 0u32;

                for hand in all_hands() {
                    let bucket = if strategy.bucket_counts[0] == 169 {
                        hand.index() as u16
                    } else {
                        (hand.index() % strategy.bucket_counts[0] as usize) as u16
                    };

                    let probs = strategy.get_action_probs(dec_idx as usize, bucket);
                    let p = probs.get(action_idx).copied().unwrap_or(0.0) as f64;

                    for (c0, c1) in hand.combos() {
                        let ci = card_pair_to_index(
                            rs_poker_card_to_id(c0),
                            rs_poker_card_to_id(c1),
                        );
                        weights[ci] *= p;
                        action_freq_sum += p;
                        combo_count += 1;
                    }
                }

                let avg_freq = if combo_count > 0 {
                    action_freq_sum / combo_count as f64
                } else {
                    0.0
                };

                action_labels.push(action_label(&actions[action_idx]));
                walk_preflop(
                    strategy,
                    tree,
                    decision_map,
                    child_idx,
                    &new_oop,
                    &new_ip,
                    path_freq * avg_freq,
                    action_labels,
                    results,
                );
                action_labels.pop();
            }
        }
    }
}

/// Parse SPR boundary string (kept for backward compat).
pub fn parse_spr_boundaries(s: &str) -> Vec<f64> {
    s.split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precomputed_ranges_save_load_roundtrip() {
        let ranges = PrecomputedRanges {
            paths: vec![PreflopPath {
                label: "raise/call".into(),
                oop_range: vec![0.5; NUM_COMBOS],
                ip_range: vec![0.3; NUM_COMBOS],
                oop_nonzero: 1326,
                ip_nonzero: 1326,
                frequency: 0.45,
            }],
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_ranges.bin");
        ranges.save(&path).unwrap();
        let loaded = PrecomputedRanges::load(&path).unwrap();
        assert_eq!(loaded.paths.len(), 1);
        assert_eq!(loaded.paths[0].label, "raise/call");
        assert!((loaded.paths[0].frequency - 0.45).abs() < 1e-6);
    }

    #[test]
    fn sample_weighted_by_frequency() {
        let ranges = PrecomputedRanges {
            paths: vec![
                PreflopPath {
                    label: "rare".into(),
                    oop_range: vec![],
                    ip_range: vec![],
                    oop_nonzero: 0,
                    ip_nonzero: 0,
                    frequency: 0.01,
                },
                PreflopPath {
                    label: "common".into(),
                    oop_range: vec![],
                    ip_range: vec![],
                    oop_nonzero: 0,
                    ip_nonzero: 0,
                    frequency: 0.99,
                },
            ],
        };

        let mut rng = rand::thread_rng();
        let mut common_count = 0;
        for _ in 0..1000 {
            if ranges.sample(&mut rng).unwrap().label == "common" {
                common_count += 1;
            }
        }
        assert!(common_count > 900, "common should be sampled ~99% of the time, got {common_count}");
    }
}
