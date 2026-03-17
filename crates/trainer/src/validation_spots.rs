//! Validation spot definitions for comparing blueprint strategies against exact solutions.

use serde::Deserialize;
use std::path::Path;

/// A single poker spot to validate blueprint strategy against exact solution.
#[derive(Debug, Deserialize)]
pub struct ValidationSpot {
    /// Human-readable description of the spot.
    pub name: String,
    /// Board cards in string format (e.g. ["Ks", "7d", "2c"]).
    pub board: Vec<String>,
    /// Out-of-position player range in PioSolver format.
    pub oop_range: String,
    /// In-position player range in PioSolver format.
    pub ip_range: String,
    /// Pot size in big blinds.
    pub pot: f64,
    /// Remaining effective stack in big blinds.
    pub effective_stack: f64,
}

/// Top-level container for a validation spots YAML file.
#[derive(Debug, Deserialize)]
pub struct ValidationSpotsFile {
    pub spots: Vec<ValidationSpot>,
}

impl ValidationSpotsFile {
    /// Load validation spots from a YAML file.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let yaml = std::fs::read_to_string(path)?;
        let file: Self = serde_yaml::from_str(&yaml)?;
        Ok(file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_single_spot() {
        let yaml = r#"
spots:
  - name: "SB open, dry flop cbet"
    board: ["Ks", "7d", "2c"]
    oop_range: "22+,A2s+,K2s+"
    ip_range: "22+,A2s+,K2s+,Q2s+"
    pot: 6.0
    effective_stack: 97.0
"#;
        let file: ValidationSpotsFile = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(file.spots.len(), 1);
        let spot = &file.spots[0];
        assert_eq!(spot.name, "SB open, dry flop cbet");
        assert_eq!(spot.board, vec!["Ks", "7d", "2c"]);
        assert_eq!(spot.oop_range, "22+,A2s+,K2s+");
        assert_eq!(spot.ip_range, "22+,A2s+,K2s+,Q2s+");
        assert!((spot.pot - 6.0).abs() < f64::EPSILON);
        assert!((spot.effective_stack - 97.0).abs() < f64::EPSILON);
    }

    #[test]
    fn deserialize_multiple_spots() {
        let yaml = r#"
spots:
  - name: "Spot A"
    board: ["Ks", "7d", "2c"]
    oop_range: "AA"
    ip_range: "KK"
    pot: 6.0
    effective_stack: 97.0
  - name: "Spot B"
    board: ["Jh", "Ts", "9d", "8c"]
    oop_range: "QQ"
    ip_range: "JJ"
    pot: 10.0
    effective_stack: 90.0
"#;
        let file: ValidationSpotsFile = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(file.spots.len(), 2);
        assert_eq!(file.spots[0].name, "Spot A");
        assert_eq!(file.spots[1].name, "Spot B");
        assert_eq!(file.spots[1].board.len(), 4);
    }

    #[test]
    fn deserialize_empty_spots_list() {
        let yaml = "spots: []\n";
        let file: ValidationSpotsFile = serde_yaml::from_str(yaml).unwrap();
        assert!(file.spots.is_empty());
    }

    #[test]
    fn spot_board_accepts_turn_card() {
        let yaml = r#"
spots:
  - name: "Turn spot"
    board: ["Ks", "7d", "2c", "Ah"]
    oop_range: "AA"
    ip_range: "KK"
    pot: 10.0
    effective_stack: 90.0
"#;
        let file: ValidationSpotsFile = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(file.spots[0].board.len(), 4);
    }

    #[test]
    fn spot_board_accepts_river_card() {
        let yaml = r#"
spots:
  - name: "River spot"
    board: ["Ks", "7d", "2c", "Ah", "3s"]
    oop_range: "AA"
    ip_range: "KK"
    pot: 10.0
    effective_stack: 90.0
"#;
        let file: ValidationSpotsFile = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(file.spots[0].board.len(), 5);
    }

    #[test]
    fn load_canonical_validation_spots_yaml() {
        // Integration test: ensure the shipped validation_spots.yaml parses correctly
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let yaml_path = manifest_dir
            .parent().unwrap()  // crates/
            .parent().unwrap()  // repo root
            .join("sample_configurations/validation_spots.yaml");
        let file = ValidationSpotsFile::load(&yaml_path)
            .expect("validation_spots.yaml should parse");
        assert!(file.spots.len() >= 15, "expected at least 15 spots, got {}", file.spots.len());
        // Verify all spots have required fields with non-empty values
        for spot in &file.spots {
            assert!(!spot.name.is_empty(), "spot name must not be empty");
            assert!(spot.board.len() >= 3 && spot.board.len() <= 5,
                "spot '{}' board must have 3-5 cards, got {}", spot.name, spot.board.len());
            assert!(!spot.oop_range.is_empty(), "spot '{}' oop_range must not be empty", spot.name);
            assert!(!spot.ip_range.is_empty(), "spot '{}' ip_range must not be empty", spot.name);
            assert!(spot.pot > 0.0, "spot '{}' pot must be positive", spot.name);
            assert!(spot.effective_stack > 0.0, "spot '{}' stack must be positive", spot.name);
        }
    }

    #[test]
    fn load_from_file() {
        let yaml = r#"
spots:
  - name: "Test spot"
    board: ["As", "Kd", "Qc"]
    oop_range: "AA"
    ip_range: "KK"
    pot: 100.0
    effective_stack: 200.0
"#;
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("spots.yaml");
        std::fs::write(&path, yaml).unwrap();

        let file = ValidationSpotsFile::load(&path).unwrap();
        assert_eq!(file.spots.len(), 1);
        assert_eq!(file.spots[0].name, "Test spot");
    }
}
