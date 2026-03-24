use std::collections::BTreeMap;

use crate::solver_trait::{ComboEvMap, StrategyMap};

/// Compute L1 strategy distance between baseline and candidate.
/// L1 at a node = average over hands of sum_a |p(a) - q(a)|.
/// Returns (per_node_distances, overall_weighted_average).
pub fn l1_strategy_distance(
    baseline: &StrategyMap,
    candidate: &StrategyMap,
    num_hands: usize,
) -> (BTreeMap<u64, f64>, f64) {
    let mut per_node = BTreeMap::new();
    let mut total_distance = 0.0;
    let mut node_count = 0;

    for (node_id, base_strat) in baseline {
        if let Some(cand_strat) = candidate.get(node_id) {
            if base_strat.len() != cand_strat.len() {
                continue;
            }

            let num_actions = base_strat.len() / num_hands;
            let mut node_l1_sum = 0.0;
            let mut hand_count = 0;

            for h in 0..num_hands {
                let mut hand_l1 = 0.0;
                for a in 0..num_actions {
                    let idx = a * num_hands + h;
                    hand_l1 += (base_strat[idx] - cand_strat[idx]).abs() as f64;
                }
                node_l1_sum += hand_l1;
                hand_count += 1;
            }

            let avg_l1 = if hand_count > 0 {
                node_l1_sum / hand_count as f64
            } else {
                0.0
            };
            per_node.insert(*node_id, avg_l1);
            total_distance += avg_l1;
            node_count += 1;
        }
    }

    let overall = if node_count > 0 {
        total_distance / node_count as f64
    } else {
        0.0
    };
    (per_node, overall)
}

/// Compute combo EV differences.
/// Returns (per_node_max_abs_diff, overall_average_max_diff).
pub fn combo_ev_diff(
    baseline: &ComboEvMap,
    candidate: &ComboEvMap,
) -> (BTreeMap<u64, f64>, f64) {
    let mut per_node = BTreeMap::new();
    let mut total_max_diff = 0.0;
    let mut node_count = 0;

    for (node_id, [base_oop, base_ip]) in baseline {
        if let Some([cand_oop, cand_ip]) = candidate.get(node_id) {
            let max_diff_oop = base_oop
                .iter()
                .zip(cand_oop.iter())
                .map(|(b, c)| (b - c).abs() as f64)
                .fold(0.0f64, f64::max);
            let max_diff_ip = base_ip
                .iter()
                .zip(cand_ip.iter())
                .map(|(b, c)| (b - c).abs() as f64)
                .fold(0.0f64, f64::max);
            let max_diff = max_diff_oop.max(max_diff_ip);

            per_node.insert(*node_id, max_diff);
            total_max_diff += max_diff;
            node_count += 1;
        }
    }

    let overall = if node_count > 0 {
        total_max_diff / node_count as f64
    } else {
        0.0
    };
    (per_node, overall)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1_distance_identical() {
        let mut strat = StrategyMap::new();
        // 2 actions, 3 hands: [0.6, 0.4, 0.8, 0.4, 0.6, 0.2]
        strat.insert(0, vec![0.6, 0.4, 0.8, 0.4, 0.6, 0.2]);

        let (per_node, overall) = l1_strategy_distance(&strat, &strat, 3);
        assert!(overall.abs() < 1e-9, "Identical strategies should have 0 L1 distance");
        assert!(per_node[&0].abs() < 1e-9);
    }

    #[test]
    fn test_l1_distance_opposite() {
        let mut base = StrategyMap::new();
        let mut cand = StrategyMap::new();
        // 2 actions, 2 hands
        base.insert(0, vec![1.0, 0.0, 0.0, 1.0]); // hand0: always action0, hand1: always action1
        cand.insert(0, vec![0.0, 1.0, 1.0, 0.0]); // opposite

        let (per_node, overall) = l1_strategy_distance(&base, &cand, 2);
        // Each hand has L1 = |1-0| + |0-1| = 2.0. Average over 2 hands = 2.0. Average over 1 node = 2.0.
        assert!((overall - 2.0).abs() < 1e-9);
        assert!((per_node[&0] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_l1_distance_empty_maps() {
        let base = StrategyMap::new();
        let cand = StrategyMap::new();

        let (per_node, overall) = l1_strategy_distance(&base, &cand, 3);
        assert!(per_node.is_empty());
        assert!(overall.abs() < 1e-9);
    }

    #[test]
    fn test_l1_distance_mismatched_nodes_skipped() {
        let mut base = StrategyMap::new();
        let cand = StrategyMap::new();
        // baseline has node 0, candidate does not
        base.insert(0, vec![0.5, 0.5]);

        let (per_node, overall) = l1_strategy_distance(&base, &cand, 1);
        assert!(per_node.is_empty());
        assert!(overall.abs() < 1e-9);
    }

    #[test]
    fn test_l1_distance_mismatched_lengths_skipped() {
        let mut base = StrategyMap::new();
        let mut cand = StrategyMap::new();
        base.insert(0, vec![0.5, 0.5]);
        cand.insert(0, vec![0.3, 0.3, 0.4]); // different length

        let (per_node, overall) = l1_strategy_distance(&base, &cand, 1);
        assert!(per_node.is_empty());
        assert!(overall.abs() < 1e-9);
    }

    #[test]
    fn test_l1_distance_multiple_nodes() {
        let mut base = StrategyMap::new();
        let mut cand = StrategyMap::new();
        // Node 0: 2 actions, 1 hand. base=[1.0, 0.0], cand=[0.5, 0.5] => L1 = 0.5 + 0.5 = 1.0
        base.insert(0, vec![1.0, 0.0]);
        cand.insert(0, vec![0.5, 0.5]);
        // Node 1: 2 actions, 1 hand. base=[0.5, 0.5], cand=[0.5, 0.5] => L1 = 0.0
        base.insert(1, vec![0.5, 0.5]);
        cand.insert(1, vec![0.5, 0.5]);

        let (per_node, overall) = l1_strategy_distance(&base, &cand, 1);
        assert!((per_node[&0] - 1.0).abs() < 1e-9);
        assert!(per_node[&1].abs() < 1e-9);
        // overall = (1.0 + 0.0) / 2 = 0.5
        assert!((overall - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_combo_ev_diff_identical() {
        let mut evs = ComboEvMap::new();
        evs.insert(0, [vec![1.5, -0.5], vec![-1.5, 0.5]]);

        let (per_node, overall) = combo_ev_diff(&evs, &evs);
        assert!(per_node[&0].abs() < 1e-9);
        assert!(overall.abs() < 1e-9);
    }

    #[test]
    fn test_combo_ev_diff_different() {
        let mut base = ComboEvMap::new();
        let mut cand = ComboEvMap::new();
        // OOP: base=[1.0, 2.0], cand=[1.5, 2.0] => max diff = 0.5
        // IP:  base=[3.0, 4.0], cand=[3.0, 3.0] => max diff = 1.0
        // Node max = 1.0
        base.insert(0, [vec![1.0, 2.0], vec![3.0, 4.0]]);
        cand.insert(0, [vec![1.5, 2.0], vec![3.0, 3.0]]);

        let (per_node, overall) = combo_ev_diff(&base, &cand);
        assert!((per_node[&0] - 1.0).abs() < 1e-9);
        assert!((overall - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_combo_ev_diff_empty_maps() {
        let base = ComboEvMap::new();
        let cand = ComboEvMap::new();

        let (per_node, overall) = combo_ev_diff(&base, &cand);
        assert!(per_node.is_empty());
        assert!(overall.abs() < 1e-9);
    }

    #[test]
    fn test_combo_ev_diff_mismatched_nodes_skipped() {
        let mut base = ComboEvMap::new();
        let cand = ComboEvMap::new();
        base.insert(0, [vec![1.0], vec![2.0]]);

        let (per_node, overall) = combo_ev_diff(&base, &cand);
        assert!(per_node.is_empty());
        assert!(overall.abs() < 1e-9);
    }

    #[test]
    fn test_combo_ev_diff_multiple_nodes() {
        let mut base = ComboEvMap::new();
        let mut cand = ComboEvMap::new();
        // Node 0: max diff = 0.5 (OOP)
        base.insert(0, [vec![1.0], vec![2.0]]);
        cand.insert(0, [vec![1.5], vec![2.0]]);
        // Node 1: max diff = 0.25 (IP)
        base.insert(1, [vec![0.0], vec![1.0]]);
        cand.insert(1, [vec![0.0], vec![0.75]]);

        let (per_node, overall) = combo_ev_diff(&base, &cand);
        assert!((per_node[&0] - 0.5).abs() < 1e-9);
        assert!((per_node[&1] - 0.25).abs() < 1e-9);
        // overall = (0.5 + 0.25) / 2 = 0.375
        assert!((overall - 0.375).abs() < 1e-9);
    }
}
