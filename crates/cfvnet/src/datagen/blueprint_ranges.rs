//! Compute turn entry ranges from a saved blueprint strategy.
//!
//! Given a blueprint bundle (config.yaml + strategy.bin + bucket files), walk
//! preflop action sequences and compute reach-weighted ranges for both players
//! at the flop boundary. This produces ranges with ~400-600 non-zero combos
//! (after preflop fold removes ~40% of hands), a large improvement over RSP's
//! ~1000 combos.

use std::path::{Path, PathBuf};

use poker_solver_core::blueprint_v2::bundle::{load_config, BlueprintV2Strategy};
use poker_solver_core::blueprint_v2::full_depth_solver::rs_poker_card_to_id;
use poker_solver_core::blueprint_v2::game_tree::{GameNode, GameTree};
use poker_solver_core::blueprint_v2::Street;
use poker_solver_core::hands::all_hands;
use range_solver::card::card_pair_to_index;

pub const NUM_COMBOS: usize = 1326;

/// Precomputed turn entry ranges from a blueprint strategy.
///
/// Loads a blueprint bundle once and provides `compute_preflop_ranges()`
/// to propagate the average strategy through all preflop action sequences,
/// yielding realistic OOP/IP ranges at the flop boundary.
pub struct BlueprintRangeGenerator {
    strategy: BlueprintV2Strategy,
    tree: GameTree,
    decision_map: Vec<u32>,
}

/// A sampled turn situation with blueprint-derived ranges.
#[derive(Debug, Clone)]
pub struct BlueprintSituation {
    pub oop_range: [f32; NUM_COMBOS],
    pub ip_range: [f32; NUM_COMBOS],
}

/// Find the latest snapshot directory (highest numbered snapshot_NNNN).
fn find_latest_snapshot(bundle_dir: &Path) -> Result<PathBuf, String> {
    let mut snapshots: Vec<(u32, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(bundle_dir).map_err(|e| format!("read dir: {e}"))? {
        let entry = entry.map_err(|e| format!("entry: {e}"))?;
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(num_str) = name.strip_prefix("snapshot_") {
            if let Ok(num) = num_str.parse::<u32>() {
                snapshots.push((num, entry.path()));
            }
        }
    }
    snapshots.sort_by_key(|(n, _)| *n);
    snapshots
        .last()
        .map(|(_, p)| p.clone())
        .ok_or_else(|| "no snapshots found".to_string())
}

impl BlueprintRangeGenerator {
    /// Load a blueprint bundle from disk.
    ///
    /// Reads config.yaml, the latest snapshot's strategy.bin, and builds
    /// the game tree with the same action abstraction used during training.
    pub fn load(bundle_dir: &Path) -> Result<Self, String> {
        let config = load_config(bundle_dir).map_err(|e| format!("load config: {e}"))?;

        let snap_dir = find_latest_snapshot(bundle_dir)?;

        let mut strategy = BlueprintV2Strategy::load(&snap_dir.join("strategy.bin"))
            .map_err(|e| format!("load strategy: {e}"))?;
        strategy.post_deserialize();

        let tree = GameTree::build_with_options(
            config.game.stack_depth,
            config.game.small_blind,
            config.game.big_blind,
            &config.action_abstraction.preflop,
            &config.action_abstraction.flop,
            &config.action_abstraction.turn,
            &config.action_abstraction.river,
            config.game.allow_preflop_limp,
        );
        let decision_map = tree.decision_index_map();

        eprintln!("[blueprint ranges] loaded from {}", bundle_dir.display());
        eprintln!(
            "[blueprint ranges] tree: {} nodes, {} decision nodes",
            tree.nodes.len(),
            decision_map.iter().filter(|&&d| d != u32::MAX).count(),
        );

        Ok(Self {
            strategy,
            tree,
            decision_map,
        })
    }

    /// Compute preflop-propagated ranges.
    ///
    /// Walks all preflop action sequences in the blueprint tree. For each
    /// sequence that reaches the flop (via a Chance node), accumulates the
    /// reach-weighted range for both OOP and IP players. Folds and all-in
    /// terminals are excluded.
    ///
    /// Returns `(oop_range, ip_range)` where each is a 1326-element array
    /// of reach probabilities. The ranges are NOT normalized to sum to 1;
    /// each combo's weight represents the probability of reaching the flop
    /// with that combo under the blueprint's average strategy.
    pub fn compute_preflop_ranges(&self) -> BlueprintSituation {
        let mut total_oop = [0.0f64; NUM_COMBOS];
        let mut total_ip = [0.0f64; NUM_COMBOS];
        let mut total_weight = 0.0f64;

        let init_oop = [1.0f64; NUM_COMBOS];
        let init_ip = [1.0f64; NUM_COMBOS];

        self.propagate_preflop(
            self.tree.root,
            &init_oop,
            &init_ip,
            1.0,
            &mut total_oop,
            &mut total_ip,
            &mut total_weight,
        );

        let mut oop_range = [0.0f32; NUM_COMBOS];
        let mut ip_range = [0.0f32; NUM_COMBOS];
        if total_weight > 0.0 {
            for i in 0..NUM_COMBOS {
                oop_range[i] = (total_oop[i] / total_weight) as f32;
                ip_range[i] = (total_ip[i] / total_weight) as f32;
            }
        }

        BlueprintSituation {
            oop_range,
            ip_range,
        }
    }

    /// Recursively walk the preflop portion of the game tree.
    ///
    /// At each Decision node on the preflop street, look up the strategy
    /// probabilities for each action and each canonical hand. For each combo
    /// of the acting player, multiply its weight by the action probability,
    /// then recurse into the child.
    ///
    /// When we reach a Chance node (transition to flop) or a Decision node
    /// on a postflop street, accumulate the current weights into the totals.
    /// Terminal nodes (folds) are ignored -- those hands are removed from
    /// the range.
    fn propagate_preflop(
        &self,
        node_idx: u32,
        oop_weights: &[f64; NUM_COMBOS],
        ip_weights: &[f64; NUM_COMBOS],
        path_prob: f64,
        total_oop: &mut [f64; NUM_COMBOS],
        total_ip: &mut [f64; NUM_COMBOS],
        total_weight: &mut f64,
    ) {
        match &self.tree.nodes[node_idx as usize] {
            GameNode::Terminal { .. } => {
                // Hand ended before reaching the flop -- do not accumulate.
            }
            GameNode::Chance { .. } => {
                // Reached a street boundary (preflop -> flop).
                // Accumulate the current ranges.
                for i in 0..NUM_COMBOS {
                    total_oop[i] += oop_weights[i] * path_prob;
                    total_ip[i] += ip_weights[i] * path_prob;
                }
                *total_weight += path_prob;
            }
            GameNode::Decision {
                player,
                street,
                children,
                ..
            } => {
                if *street != Street::Preflop {
                    // Reached a postflop decision -- accumulate.
                    for i in 0..NUM_COMBOS {
                        total_oop[i] += oop_weights[i] * path_prob;
                        total_ip[i] += ip_weights[i] * path_prob;
                    }
                    *total_weight += path_prob;
                    return;
                }

                let dec_idx = self.decision_map[node_idx as usize];
                if dec_idx == u32::MAX {
                    return;
                }

                // For each action, compute updated weights and recurse.
                for (action_idx, &child_idx) in children.iter().enumerate() {
                    let mut new_oop = *oop_weights;
                    let mut new_ip = *ip_weights;

                    // Determine which player acts: dealer = IP (SB), other = OOP (BB).
                    let weights = if *player == self.tree.dealer {
                        &mut new_ip
                    } else {
                        &mut new_oop
                    };

                    // For each canonical hand, get the bucket and strategy probability,
                    // then multiply each combo's weight.
                    for hand in all_hands() {
                        let bucket = if self.strategy.bucket_counts[0] == 169 {
                            hand.index() as u16
                        } else {
                            (hand.index() % self.strategy.bucket_counts[0] as usize) as u16
                        };

                        let probs =
                            self.strategy.get_action_probs(dec_idx as usize, bucket);
                        let p = probs.get(action_idx).copied().unwrap_or(0.0) as f64;

                        for (c0, c1) in hand.combos() {
                            let id0 = rs_poker_card_to_id(c0);
                            let id1 = rs_poker_card_to_id(c1);
                            let ci = card_pair_to_index(id0, id1);
                            weights[ci] *= p;
                        }
                    }

                    self.propagate_preflop(
                        child_idx,
                        &new_oop,
                        &new_ip,
                        path_prob,
                        total_oop,
                        total_ip,
                        total_weight,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
    use poker_solver_core::blueprint_v2::game_tree::GameTree;

    use poker_solver_core::blueprint_v2::game_tree::TreeAction;

    const BUCKET_COUNTS: [u16; 4] = [169, 10, 10, 10];

    fn test_tree() -> GameTree {
        GameTree::build(
            20.0,
            1.0,
            2.0,
            &[vec!["5bb".into()]],
            &[vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
        )
    }

    /// Build strategy action probs by applying `prob_fn` to each action
    /// at each decision node. `prob_fn(actions, action_idx)` returns the
    /// probability for that action.
    fn build_strategy_with<F>(tree: &GameTree, prob_fn: F) -> BlueprintV2Strategy
    where
        F: Fn(&[TreeAction], usize) -> f32,
    {
        let mut node_action_counts: Vec<u16> = Vec::new();
        let mut node_street_indices: Vec<u8> = Vec::new();
        let mut action_probs = Vec::new();

        for node in &tree.nodes {
            if let GameNode::Decision {
                actions, street, ..
            } = node
            {
                let n_act = actions.len();
                node_action_counts.push(n_act as u16);
                node_street_indices.push(*street as u8);

                let st = *street as usize;
                let n_buckets = BUCKET_COUNTS[st] as usize;
                for _ in 0..n_buckets {
                    for ai in 0..n_act {
                        action_probs.push(prob_fn(actions, ai));
                    }
                }
            }
        }

        let mut strategy = BlueprintV2Strategy {
            action_probs,
            node_action_counts,
            node_street_indices,
            bucket_counts: BUCKET_COUNTS,
            iterations: 0,
            elapsed_minutes: 0,
            node_offsets: Vec::new(),
        };
        strategy.post_deserialize();
        strategy
    }

    fn build_generator(tree: GameTree, strategy: BlueprintV2Strategy) -> BlueprintRangeGenerator {
        let decision_map = tree.decision_index_map();
        BlueprintRangeGenerator {
            strategy,
            tree,
            decision_map,
        }
    }

    /// Build a generator with a uniform strategy (equal probability per action).
    fn build_test_generator() -> BlueprintRangeGenerator {
        let tree = test_tree();
        let strategy = build_strategy_with(&tree, |actions, _| 1.0 / actions.len() as f32);
        build_generator(tree, strategy)
    }

    #[test]
    fn preflop_ranges_have_nonzero_entries() {
        let generator = build_test_generator();
        let result = generator.compute_preflop_ranges();

        let oop_nonzero = result.oop_range.iter().filter(|&&w| w > 0.0).count();
        let ip_nonzero = result.ip_range.iter().filter(|&&w| w > 0.0).count();

        assert!(
            oop_nonzero > 0,
            "OOP range should have non-zero combos, got {oop_nonzero}"
        );
        assert!(
            ip_nonzero > 0,
            "IP range should have non-zero combos, got {ip_nonzero}"
        );
    }

    #[test]
    fn preflop_ranges_weights_reduced_by_folding() {
        let generator = build_test_generator();
        let result = generator.compute_preflop_ranges();

        // With a uniform strategy that includes fold, the average weight
        // per combo should be less than 1.0 because some probability mass
        // goes to fold paths that never reach the flop.
        let oop_sum: f64 = result.oop_range.iter().map(|&w| w as f64).sum();
        let ip_sum: f64 = result.ip_range.iter().map(|&w| w as f64).sum();

        let oop_avg = oop_sum / NUM_COMBOS as f64;
        let ip_avg = ip_sum / NUM_COMBOS as f64;

        assert!(
            oop_avg < 1.0,
            "OOP average weight {oop_avg} should be < 1.0 due to folding"
        );
        assert!(
            ip_avg < 1.0,
            "IP average weight {ip_avg} should be < 1.0 due to folding"
        );
    }

    #[test]
    fn preflop_ranges_all_non_negative() {
        let generator = build_test_generator();
        let result = generator.compute_preflop_ranges();

        for (i, &w) in result.oop_range.iter().enumerate() {
            assert!(w >= 0.0, "OOP range[{i}] = {w} is negative");
        }
        for (i, &w) in result.ip_range.iter().enumerate() {
            assert!(w >= 0.0, "IP range[{i}] = {w} is negative");
        }
    }

    #[test]
    fn preflop_ranges_max_weight_at_most_one() {
        let generator = build_test_generator();
        let result = generator.compute_preflop_ranges();

        let oop_max = result
            .oop_range
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        let ip_max = result
            .ip_range
            .iter()
            .copied()
            .fold(0.0f32, f32::max);

        assert!(
            oop_max <= 1.0 + 1e-6,
            "OOP max weight {oop_max} should be <= 1.0"
        );
        assert!(
            ip_max <= 1.0 + 1e-6,
            "IP max weight {ip_max} should be <= 1.0"
        );
    }

    #[test]
    fn preflop_ranges_deterministic() {
        let generator = build_test_generator();
        let r1 = generator.compute_preflop_ranges();
        let r2 = generator.compute_preflop_ranges();

        assert_eq!(r1.oop_range, r2.oop_range, "OOP ranges should be deterministic");
        assert_eq!(r1.ip_range, r2.ip_range, "IP ranges should be deterministic");
    }

    /// With a purely fold-only strategy (fold prob = 1.0 for all actions
    /// except fold), both ranges should be empty at the flop boundary.
    #[test]
    fn fold_only_strategy_yields_empty_ranges() {
        let tree = test_tree();
        let strategy = build_strategy_with(&tree, |actions, ai| {
            if matches!(actions[ai], TreeAction::Fold) {
                1.0
            } else {
                0.0
            }
        });
        let generator = build_generator(tree, strategy);
        let result = generator.compute_preflop_ranges();

        let oop_nonzero = result.oop_range.iter().filter(|&&w| w > 0.0).count();
        let ip_nonzero = result.ip_range.iter().filter(|&&w| w > 0.0).count();

        // When everyone always folds preflop, no action sequence reaches
        // the flop, so both ranges should be all zeros.
        assert_eq!(
            oop_nonzero, 0,
            "fold-only strategy: OOP should have 0 non-zero combos, got {oop_nonzero}"
        );
        assert_eq!(
            ip_nonzero, 0,
            "fold-only strategy: IP should have 0 non-zero combos, got {ip_nonzero}"
        );
    }

    /// With a strategy where everyone always calls, both players should
    /// have all 1326 combos with equal weight at the flop boundary.
    #[test]
    fn all_call_strategy_preserves_full_range() {
        let tree = test_tree();
        let strategy = build_strategy_with(&tree, |actions, ai| {
            if matches!(actions[ai], TreeAction::Call | TreeAction::Check) {
                1.0
            } else {
                0.0
            }
        });
        let generator = build_generator(tree, strategy);
        let result = generator.compute_preflop_ranges();

        let oop_nonzero = result.oop_range.iter().filter(|&&w| w > 0.0).count();
        let ip_nonzero = result.ip_range.iter().filter(|&&w| w > 0.0).count();

        // When everyone always calls/checks, all combos should reach the flop.
        assert_eq!(
            oop_nonzero, NUM_COMBOS,
            "all-call strategy: OOP should have {NUM_COMBOS} non-zero combos, got {oop_nonzero}"
        );
        assert_eq!(
            ip_nonzero, NUM_COMBOS,
            "all-call strategy: IP should have {NUM_COMBOS} non-zero combos, got {ip_nonzero}"
        );

        // All weights should be equal (exactly 1.0 since everyone calls).
        let first_oop = result.oop_range.iter().find(|&&w| w > 0.0).unwrap();
        for (i, &w) in result.oop_range.iter().enumerate() {
            if w > 0.0 {
                assert!(
                    (w - first_oop).abs() < 1e-5,
                    "OOP range[{i}] = {w}, expected {first_oop}"
                );
            }
        }
    }

    #[test]
    fn find_latest_snapshot_picks_highest_number() {
        let dir = tempfile::tempdir().unwrap();
        // Create snapshot directories in non-sorted order.
        std::fs::create_dir(dir.path().join("snapshot_0002")).unwrap();
        std::fs::create_dir(dir.path().join("snapshot_0000")).unwrap();
        std::fs::create_dir(dir.path().join("snapshot_0005")).unwrap();
        std::fs::create_dir(dir.path().join("snapshot_0001")).unwrap();

        let latest = find_latest_snapshot(dir.path()).unwrap();
        assert!(
            latest.ends_with("snapshot_0005"),
            "expected snapshot_0005, got {}",
            latest.display()
        );
    }

    #[test]
    fn find_latest_snapshot_no_snapshots_is_error() {
        let dir = tempfile::tempdir().unwrap();
        let result = find_latest_snapshot(dir.path());
        assert!(result.is_err(), "should error when no snapshots exist");
    }

    #[test]
    fn find_latest_snapshot_ignores_non_snapshot_dirs() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("other_dir")).unwrap();
        std::fs::create_dir(dir.path().join("final")).unwrap();
        std::fs::create_dir(dir.path().join("snapshot_0003")).unwrap();

        let latest = find_latest_snapshot(dir.path()).unwrap();
        assert!(
            latest.ends_with("snapshot_0003"),
            "expected snapshot_0003, got {}",
            latest.display()
        );
    }
}
