use crate::baseline::ConvergenceSample;
use std::collections::BTreeMap;
use std::path::Path;

/// Comparison results from running a solver against the baseline.
#[derive(Debug, serde::Serialize)]
pub struct ComparisonResult {
    pub solver_name: String,
    pub total_iterations: u64,
    pub total_time_ms: u64,
    pub final_exploitability: f64,
    pub baseline_exploitability: f64,
    pub overall_l1_distance: f64,
    pub overall_max_ev_diff: f64,
    pub convergence_curve: Vec<ConvergenceSample>,
    pub per_node_l1: BTreeMap<u64, f64>,
    pub per_node_ev_diff: BTreeMap<u64, f64>,
}

impl ComparisonResult {
    /// Save all comparison artifacts to a directory.
    ///
    /// Creates the directory if it does not exist. Writes:
    /// - `summary.json` (machine-readable summary)
    /// - `convergence.csv` (convergence curve)
    /// - `strategy_distance.csv` (per-node L1 distances)
    /// - `combo_ev_diff.csv` (per-node max EV diffs)
    /// - `report.txt` (human-readable summary)
    pub fn save(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(dir)?;

        let summary_json = serde_json::to_string_pretty(&serde_json::json!({
            "solver_name": self.solver_name,
            "total_iterations": self.total_iterations,
            "total_time_ms": self.total_time_ms,
            "final_exploitability": self.final_exploitability,
            "baseline_exploitability": self.baseline_exploitability,
            "overall_l1_distance": self.overall_l1_distance,
            "overall_max_ev_diff": self.overall_max_ev_diff,
        }))?;
        std::fs::write(dir.join("summary.json"), summary_json)?;

        let mut wtr = csv::Writer::from_path(dir.join("convergence.csv"))?;
        for sample in &self.convergence_curve {
            wtr.serialize(sample)?;
        }
        wtr.flush()?;

        let mut wtr = csv::Writer::from_path(dir.join("strategy_distance.csv"))?;
        wtr.write_record(["node_id", "l1_distance"])?;
        for (node_id, dist) in &self.per_node_l1 {
            wtr.write_record([node_id.to_string(), format!("{:.6}", dist)])?;
        }
        wtr.flush()?;

        let mut wtr = csv::Writer::from_path(dir.join("combo_ev_diff.csv"))?;
        wtr.write_record(["node_id", "max_ev_diff"])?;
        for (node_id, diff) in &self.per_node_ev_diff {
            wtr.write_record([node_id.to_string(), format!("{:.6}", diff)])?;
        }
        wtr.flush()?;

        let report = self.human_summary();
        std::fs::write(dir.join("report.txt"), &report)?;

        Ok(())
    }

    /// Generate a human-readable summary report.
    pub fn human_summary(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Convergence Harness Report ===\n\n");

        s.push_str(&format!("Solver: {}\n", self.solver_name));
        s.push_str(&format!("Iterations: {}\n", self.total_iterations));
        s.push_str(&format!(
            "Time: {:.1}s\n\n",
            self.total_time_ms as f64 / 1000.0
        ));

        s.push_str("--- Exploitability ---\n");
        s.push_str(&format!("Baseline: {:.1e}\n", self.baseline_exploitability));
        s.push_str(&format!("Solver:   {:.1e}\n", self.final_exploitability));
        let gap = self.final_exploitability - self.baseline_exploitability;
        s.push_str(&format!("Gap:      {:.1e}\n\n", gap));

        s.push_str("--- Strategy Distance (L1) ---\n");
        s.push_str(&format!("Overall average: {:.4}\n", self.overall_l1_distance));
        if self.overall_l1_distance < 0.05 {
            s.push_str("Assessment: Excellent — abstraction is faithful.\n\n");
        } else if self.overall_l1_distance < 0.1 {
            s.push_str("Assessment: Good — minor strategy deviations.\n\n");
        } else if self.overall_l1_distance < 0.3 {
            s.push_str("Assessment: Concerning — noticeable strategy deviations.\n\n");
        } else {
            s.push_str("Assessment: Poor — abstraction substantially alters strategy.\n\n");
        }

        s.push_str("--- Combo EV Difference ---\n");
        s.push_str(&format!(
            "Overall average max diff: {:.4} bb\n",
            self.overall_max_ev_diff
        ));

        if self.convergence_curve.len() >= 2 {
            let first = &self.convergence_curve[0];
            let last = self.convergence_curve.last().unwrap();
            if last.iteration > first.iteration
                && last.exploitability > 0.0
                && first.exploitability > 0.0
            {
                let log_ratio = (first.exploitability / last.exploitability).ln();
                let iter_ratio =
                    (last.iteration as f64 / first.iteration.max(1) as f64).ln();
                if iter_ratio > 0.0 {
                    let rate = log_ratio / iter_ratio;
                    s.push_str(&format!("\n--- Convergence Rate ---\n"));
                    s.push_str(&format!("Approximate rate: O(1/T^{:.2})\n", rate));
                    s.push_str("(DCFR typically converges at ~O(1/T^0.5))\n");
                }
            }
        }

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample_result() -> ComparisonResult {
        ComparisonResult {
            solver_name: "Test Solver".into(),
            total_iterations: 100,
            total_time_ms: 5000,
            final_exploitability: 0.05,
            baseline_exploitability: 0.001,
            overall_l1_distance: 0.12,
            overall_max_ev_diff: 0.5,
            convergence_curve: vec![
                ConvergenceSample {
                    iteration: 1,
                    exploitability: 1.0,
                    elapsed_ms: 50,
                },
                ConvergenceSample {
                    iteration: 100,
                    exploitability: 0.05,
                    elapsed_ms: 5000,
                },
            ],
            per_node_l1: {
                let mut m = BTreeMap::new();
                m.insert(0, 0.1);
                m.insert(1, 0.15);
                m
            },
            per_node_ev_diff: {
                let mut m = BTreeMap::new();
                m.insert(0, 0.3);
                m
            },
        }
    }

    #[test]
    fn test_comparison_result_save() {
        let result = sample_result();
        let dir = TempDir::new().unwrap();
        result.save(dir.path()).unwrap();

        assert!(dir.path().join("summary.json").exists());
        assert!(dir.path().join("convergence.csv").exists());
        assert!(dir.path().join("strategy_distance.csv").exists());
        assert!(dir.path().join("combo_ev_diff.csv").exists());
        assert!(dir.path().join("report.txt").exists());
    }

    #[test]
    fn test_comparison_result_save_creates_directory() {
        let result = sample_result();
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("nested").join("deep");
        result.save(&nested).unwrap();

        assert!(nested.join("summary.json").exists());
        assert!(nested.join("report.txt").exists());
    }

    #[test]
    fn test_human_summary_contains_key_info() {
        let result = sample_result();
        let summary = result.human_summary();

        assert!(summary.contains("Test Solver"), "Should contain solver name");
        assert!(summary.contains("100"), "Should contain iteration count");
        assert!(summary.contains("5.0e-2"), "Should contain solver exploitability");
        assert!(summary.contains("1.0e-3"), "Should contain baseline exploitability");
        assert!(summary.contains("Concerning"), "L1=0.12 should be assessed as Concerning");
    }

    #[test]
    fn test_human_summary_excellent_assessment() {
        let mut result = sample_result();
        result.overall_l1_distance = 0.03;
        let summary = result.human_summary();
        assert!(summary.contains("Excellent"), "L1=0.03 should be assessed as Excellent");
    }

    #[test]
    fn test_human_summary_good_assessment() {
        let mut result = sample_result();
        result.overall_l1_distance = 0.07;
        let summary = result.human_summary();
        assert!(summary.contains("Good"), "L1=0.07 should be assessed as Good");
    }

    #[test]
    fn test_human_summary_poor_assessment() {
        let mut result = sample_result();
        result.overall_l1_distance = 0.5;
        let summary = result.human_summary();
        assert!(summary.contains("Poor"), "L1=0.5 should be assessed as Poor");
    }

    #[test]
    fn test_human_summary_convergence_rate() {
        let result = sample_result();
        let summary = result.human_summary();
        // With iteration 1->100, exploitability 1.0->0.05, should compute a convergence rate
        assert!(
            summary.contains("Convergence Rate"),
            "Should contain convergence rate section"
        );
        assert!(
            summary.contains("O(1/T^"),
            "Should contain convergence rate estimate"
        );
    }

    #[test]
    fn test_human_summary_no_convergence_rate_single_sample() {
        let mut result = sample_result();
        result.convergence_curve = vec![ConvergenceSample {
            iteration: 50,
            exploitability: 0.1,
            elapsed_ms: 1000,
        }];
        let summary = result.human_summary();
        assert!(
            !summary.contains("Convergence Rate"),
            "Single sample should not produce convergence rate"
        );
    }

    #[test]
    fn test_save_summary_json_contains_fields() {
        let result = sample_result();
        let dir = TempDir::new().unwrap();
        result.save(dir.path()).unwrap();

        let json_str = std::fs::read_to_string(dir.path().join("summary.json")).unwrap();
        let json: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(json["solver_name"], "Test Solver");
        assert_eq!(json["total_iterations"], 100);
        assert_eq!(json["total_time_ms"], 5000);
    }

    #[test]
    fn test_save_convergence_csv_has_rows() {
        let result = sample_result();
        let dir = TempDir::new().unwrap();
        result.save(dir.path()).unwrap();

        let csv_str = std::fs::read_to_string(dir.path().join("convergence.csv")).unwrap();
        let lines: Vec<&str> = csv_str.lines().collect();
        // header + 2 data rows
        assert_eq!(lines.len(), 3, "convergence.csv should have header + 2 rows");
    }

    #[test]
    fn test_save_strategy_distance_csv_has_rows() {
        let result = sample_result();
        let dir = TempDir::new().unwrap();
        result.save(dir.path()).unwrap();

        let csv_str =
            std::fs::read_to_string(dir.path().join("strategy_distance.csv")).unwrap();
        let lines: Vec<&str> = csv_str.lines().collect();
        // header + 2 nodes
        assert_eq!(
            lines.len(),
            3,
            "strategy_distance.csv should have header + 2 rows"
        );
    }
}
