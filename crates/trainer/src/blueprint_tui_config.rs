use serde::Deserialize;

/// Which street a scenario or random-pool entry refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StreetLabel {
    Preflop,
    Flop,
    Turn,
    River,
}

/// Which player position a regret-audit entry refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum PlayerLabel {
    Sb,
    Bb,
}

/// A hand whose regret accumulation is tracked live in the TUI.
#[derive(Debug, Clone, Deserialize)]
pub struct RegretAuditConfig {
    pub name: String,
    #[serde(default)]
    pub spot: String,
    pub hand: String,
    pub player: PlayerLabel,
}

/// A single scenario to display live strategy evolution in the TUI.
#[derive(Debug, Clone, Deserialize)]
pub struct ScenarioConfig {
    pub name: String,
    #[serde(default)]
    pub spot: String,
}

/// Controls how often telemetry signals are sampled.
#[derive(Debug, Clone, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default = "default_delta_interval")]
    pub strategy_delta_interval_seconds: u64,
    #[serde(default = "default_sparkline_window")]
    pub sparkline_window: usize,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            strategy_delta_interval_seconds: default_delta_interval(),
            sparkline_window: default_sparkline_window(),
        }
    }
}

/// Configuration for the random-scenario carousel.
#[derive(Debug, Clone, Deserialize)]
pub struct RandomScenarioConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_hold_minutes")]
    pub hold_minutes: u64,
    #[serde(default = "default_pool")]
    pub pool: Vec<StreetLabel>,
}

impl Default for RandomScenarioConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            hold_minutes: default_hold_minutes(),
            pool: default_pool(),
        }
    }
}

/// Top-level TUI configuration, parsed from the `tui:` key in a training YAML.
#[derive(Debug, Clone, Deserialize)]
pub struct BlueprintTuiConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_refresh_rate_ms")]
    pub refresh_rate_ms: u64,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub scenarios: Vec<ScenarioConfig>,
    #[serde(default)]
    pub random_scenario: RandomScenarioConfig,
    #[serde(default)]
    pub regret_audits: Vec<RegretAuditConfig>,
}

impl Default for BlueprintTuiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            refresh_rate_ms: default_refresh_rate_ms(),
            telemetry: TelemetryConfig::default(),
            scenarios: Vec::new(),
            random_scenario: RandomScenarioConfig::default(),
            regret_audits: Vec::new(),
        }
    }
}

// --- serde default helpers ---

fn default_refresh_rate_ms() -> u64 {
    250
}
fn default_delta_interval() -> u64 {
    30
}
fn default_sparkline_window() -> usize {
    60
}
fn default_hold_minutes() -> u64 {
    3
}
fn default_pool() -> Vec<StreetLabel> {
    vec![StreetLabel::Preflop, StreetLabel::Flop, StreetLabel::Turn]
}

// --- Extraction from full YAML ---

/// Wrapper that picks out just the optional `tui` key from a full config YAML.
#[derive(Deserialize)]
struct TuiWrapper {
    #[serde(default)]
    tui: Option<BlueprintTuiConfig>,
}

/// Parse the `tui:` section from a full training YAML string.
///
/// Returns `BlueprintTuiConfig::default()` when the key is absent.
pub fn parse_tui_config(yaml: &str) -> BlueprintTuiConfig {
    let wrapper: TuiWrapper =
        serde_yaml::from_str(yaml).unwrap_or(TuiWrapper { tui: None });
    wrapper.tui.unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test(10)]
    fn parse_complete_tui_config() {
        let yaml = r#"
tui:
  enabled: true
  refresh_rate_ms: 100
  telemetry:
    strategy_delta_interval_seconds: 15
    sparkline_window: 120
  scenarios:
    - name: "SB open"
      spot: ""
    - name: "BB vs raise"
      spot: "sb:5bb"
  random_scenario:
    enabled: true
    hold_minutes: 5
    pool: [preflop, flop, turn, river]
"#;
        let cfg = parse_tui_config(yaml);
        assert!(cfg.enabled);
        assert_eq!(cfg.refresh_rate_ms, 100);
        assert_eq!(cfg.telemetry.strategy_delta_interval_seconds, 15);
        assert_eq!(cfg.telemetry.sparkline_window, 120);
        assert_eq!(cfg.scenarios.len(), 2);

        let s0 = &cfg.scenarios[0];
        assert_eq!(s0.name, "SB open");
        assert_eq!(s0.spot, "");

        let s1 = &cfg.scenarios[1];
        assert_eq!(s1.name, "BB vs raise");
        assert_eq!(s1.spot, "sb:5bb");

        assert!(cfg.random_scenario.enabled);
        assert_eq!(cfg.random_scenario.hold_minutes, 5);
        assert_eq!(cfg.random_scenario.pool.len(), 4);
        assert_eq!(cfg.random_scenario.pool[3], StreetLabel::River);
    }

    #[timed_test(10)]
    fn defaults_when_tui_absent() {
        let cfg = parse_tui_config("");
        assert!(!cfg.enabled);
        assert_eq!(cfg.refresh_rate_ms, 250);
        assert_eq!(cfg.telemetry.strategy_delta_interval_seconds, 30);
        assert_eq!(cfg.telemetry.sparkline_window, 60);
        assert!(cfg.scenarios.is_empty());
        assert!(!cfg.random_scenario.enabled);
        assert_eq!(cfg.random_scenario.hold_minutes, 3);
        assert_eq!(cfg.random_scenario.pool, vec![
            StreetLabel::Preflop,
            StreetLabel::Flop,
            StreetLabel::Turn,
        ]);
    }

    #[timed_test(10)]
    fn parse_regret_audits() {
        let yaml = r#"
tui:
  enabled: true
  regret_audits:
    - name: "AKo SB open"
      spot: ""
      hand: "AKo"
      player: SB
    - name: "T9s flop cbet"
      spot: "sb:2bb,bb:call|AsTd9d"
      hand: "Ts9s"
      player: SB
"#;
        let cfg = parse_tui_config(yaml);
        assert_eq!(cfg.regret_audits.len(), 2);
        assert_eq!(cfg.regret_audits[0].name, "AKo SB open");
        assert_eq!(cfg.regret_audits[0].hand, "AKo");
        assert_eq!(cfg.regret_audits[0].player, PlayerLabel::Sb);
        assert_eq!(cfg.regret_audits[1].spot, "sb:2bb,bb:call|AsTd9d");
        assert_eq!(cfg.regret_audits[1].hand, "Ts9s");
    }

    #[timed_test(10)]
    fn regret_audits_default_empty() {
        let cfg = parse_tui_config("");
        assert!(cfg.regret_audits.is_empty());
    }

    #[timed_test(10)]
    fn extracts_tui_from_full_config() {
        let yaml = r#"
game:
  players: 2
  stack_depth: 200.0
  small_blind: 1
  big_blind: 2
training:
  cluster_path: "/tmp/clusters"
tui:
  enabled: true
  refresh_rate_ms: 500
  scenarios:
    - name: "Check spot"
      spot: ""
"#;
        let cfg = parse_tui_config(yaml);
        assert!(cfg.enabled);
        assert_eq!(cfg.refresh_rate_ms, 500);
        assert_eq!(cfg.scenarios.len(), 1);
        assert_eq!(cfg.scenarios[0].name, "Check spot");
        assert_eq!(cfg.scenarios[0].spot, "");
        assert_eq!(cfg.telemetry.sparkline_window, 60);
        assert!(!cfg.random_scenario.enabled);
    }
}
