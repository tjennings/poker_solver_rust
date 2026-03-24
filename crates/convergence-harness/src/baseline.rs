use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// A single convergence sample: iteration number, exploitability, elapsed time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceSample {
    pub iteration: u64,
    pub exploitability: f64,
    pub elapsed_ms: u64,
}

/// Summary of the baseline solve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineSummary {
    pub solver_name: String,
    pub total_iterations: u64,
    pub final_exploitability: f64,
    pub total_time_ms: u64,
    pub num_info_sets: usize,
    pub num_combos_per_player: usize,
    pub game_description: String,
}

/// The full baseline artifact.
#[derive(Debug, Serialize, Deserialize)]
pub struct Baseline {
    pub summary: BaselineSummary,
    pub convergence_curve: Vec<ConvergenceSample>,
    /// Per-node strategy: node_id -> flat Vec<f32> (action_idx * num_hands + hand_idx)
    pub strategy: BTreeMap<u64, Vec<f32>>,
    /// Per-node combo EVs: node_id -> [oop_evs, ip_evs]
    pub combo_evs: BTreeMap<u64, [Vec<f32>; 2]>,
}

impl Baseline {
    /// Save baseline to a directory.
    ///
    /// Creates the directory if it does not exist. Writes:
    /// - `summary.json` (human-readable JSON)
    /// - `convergence.csv` (CSV with headers)
    /// - `strategy.bin` (bincode)
    /// - `combo_ev.bin` (bincode)
    pub fn save(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(dir)?;

        let summary_json = serde_json::to_string_pretty(&self.summary)?;
        std::fs::write(dir.join("summary.json"), summary_json)?;

        let mut wtr = csv::Writer::from_path(dir.join("convergence.csv"))?;
        for sample in &self.convergence_curve {
            wtr.serialize(sample)?;
        }
        wtr.flush()?;

        let strategy_bytes = bincode::serialize(&self.strategy)?;
        std::fs::write(dir.join("strategy.bin"), strategy_bytes)?;

        let ev_bytes = bincode::serialize(&self.combo_evs)?;
        std::fs::write(dir.join("combo_ev.bin"), ev_bytes)?;

        Ok(())
    }

    /// Load baseline from a directory.
    ///
    /// Reads back all four files written by `save`.
    pub fn load(dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let summary: BaselineSummary =
            serde_json::from_str(&std::fs::read_to_string(dir.join("summary.json"))?)?;

        let mut convergence_curve = Vec::new();
        let mut rdr = csv::Reader::from_path(dir.join("convergence.csv"))?;
        for result in rdr.deserialize() {
            convergence_curve.push(result?);
        }

        let strategy: BTreeMap<u64, Vec<f32>> =
            bincode::deserialize(&std::fs::read(dir.join("strategy.bin"))?)?;

        let combo_evs: BTreeMap<u64, [Vec<f32>; 2]> =
            bincode::deserialize(&std::fs::read(dir.join("combo_ev.bin"))?)?;

        Ok(Self {
            summary,
            convergence_curve,
            strategy,
            combo_evs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Helper to create a sample baseline for testing.
    fn sample_baseline() -> Baseline {
        Baseline {
            summary: BaselineSummary {
                solver_name: "Exhaustive DCFR".into(),
                total_iterations: 100,
                final_exploitability: 0.001,
                total_time_ms: 5000,
                num_info_sets: 42,
                num_combos_per_player: 1176,
                game_description: "Flop Poker: QhJdTh, 50bb/2bb".into(),
            },
            convergence_curve: vec![
                ConvergenceSample {
                    iteration: 0,
                    exploitability: 1.0,
                    elapsed_ms: 0,
                },
                ConvergenceSample {
                    iteration: 50,
                    exploitability: 0.01,
                    elapsed_ms: 2500,
                },
                ConvergenceSample {
                    iteration: 100,
                    exploitability: 0.001,
                    elapsed_ms: 5000,
                },
            ],
            strategy: {
                let mut m = BTreeMap::new();
                m.insert(0, vec![0.5, 0.3, 0.2]);
                m
            },
            combo_evs: {
                let mut m = BTreeMap::new();
                m.insert(0, [vec![1.5, -0.5], vec![-1.5, 0.5]]);
                m
            },
        }
    }

    #[test]
    fn test_baseline_round_trip() {
        let baseline = sample_baseline();
        let dir = TempDir::new().unwrap();
        baseline.save(dir.path()).unwrap();

        let loaded = Baseline::load(dir.path()).unwrap();

        // Verify summary fields
        assert_eq!(loaded.summary.solver_name, "Exhaustive DCFR");
        assert_eq!(loaded.summary.total_iterations, 100);
        assert!((loaded.summary.final_exploitability - 0.001).abs() < 1e-9);
        assert_eq!(loaded.summary.total_time_ms, 5000);
        assert_eq!(loaded.summary.num_info_sets, 42);
        assert_eq!(loaded.summary.num_combos_per_player, 1176);
        assert_eq!(
            loaded.summary.game_description,
            "Flop Poker: QhJdTh, 50bb/2bb"
        );

        // Verify convergence curve
        assert_eq!(loaded.convergence_curve.len(), 3);
        assert_eq!(loaded.convergence_curve[0].iteration, 0);
        assert!((loaded.convergence_curve[0].exploitability - 1.0).abs() < 1e-9);
        assert_eq!(loaded.convergence_curve[1].iteration, 50);
        assert!((loaded.convergence_curve[2].exploitability - 0.001).abs() < 1e-9);

        // Verify strategy
        assert_eq!(loaded.strategy.len(), 1);
        let strat = loaded.strategy.get(&0).unwrap();
        assert_eq!(strat.len(), 3);
        assert!((strat[0] - 0.5).abs() < 1e-6);
        assert!((strat[1] - 0.3).abs() < 1e-6);
        assert!((strat[2] - 0.2).abs() < 1e-6);

        // Verify combo EVs
        assert_eq!(loaded.combo_evs.len(), 1);
        let evs = loaded.combo_evs.get(&0).unwrap();
        assert_eq!(evs[0].len(), 2);
        assert!((evs[0][0] - 1.5).abs() < 1e-6);
        assert!((evs[1][1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_baseline_files_exist() {
        let baseline = sample_baseline();
        let dir = TempDir::new().unwrap();
        baseline.save(dir.path()).unwrap();

        assert!(dir.path().join("summary.json").exists());
        assert!(dir.path().join("convergence.csv").exists());
        assert!(dir.path().join("strategy.bin").exists());
        assert!(dir.path().join("combo_ev.bin").exists());
    }

    #[test]
    fn test_baseline_save_creates_directory() {
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("nested").join("deep");
        let baseline = sample_baseline();
        baseline.save(&nested).unwrap();

        assert!(nested.join("summary.json").exists());
    }

    #[test]
    fn test_baseline_load_missing_dir_returns_error() {
        let result = Baseline::load(Path::new("/nonexistent/path/baseline"));
        assert!(result.is_err());
    }

    #[test]
    fn test_baseline_empty_collections_round_trip() {
        let baseline = Baseline {
            summary: BaselineSummary {
                solver_name: "empty".into(),
                total_iterations: 0,
                final_exploitability: 0.0,
                total_time_ms: 0,
                num_info_sets: 0,
                num_combos_per_player: 0,
                game_description: "empty".into(),
            },
            convergence_curve: vec![],
            strategy: BTreeMap::new(),
            combo_evs: BTreeMap::new(),
        };

        let dir = TempDir::new().unwrap();
        baseline.save(dir.path()).unwrap();
        let loaded = Baseline::load(dir.path()).unwrap();

        assert_eq!(loaded.summary.solver_name, "empty");
        assert!(loaded.convergence_curve.is_empty());
        assert!(loaded.strategy.is_empty());
        assert!(loaded.combo_evs.is_empty());
    }

    #[test]
    fn test_baseline_multiple_strategy_nodes_round_trip() {
        let mut baseline = sample_baseline();
        baseline.strategy.insert(1, vec![0.1, 0.9]);
        baseline.strategy.insert(42, vec![0.25, 0.25, 0.25, 0.25]);
        baseline
            .combo_evs
            .insert(1, [vec![2.0, 3.0], vec![-2.0, -3.0]]);

        let dir = TempDir::new().unwrap();
        baseline.save(dir.path()).unwrap();
        let loaded = Baseline::load(dir.path()).unwrap();

        assert_eq!(loaded.strategy.len(), 3);
        assert_eq!(loaded.strategy.get(&1).unwrap(), &vec![0.1, 0.9]);
        assert_eq!(loaded.combo_evs.len(), 2);
    }
}
