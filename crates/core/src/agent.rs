//! Configurable rule-based poker agents.
//!
//! Agents are defined in TOML files that map `HandClass` to action frequencies,
//! with optional preflop range filtering. They produce the same `StrategyMatrix`
//! output as trained blueprint strategies, enabling UI development without training.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;
use std::str::FromStr;

use rs_poker::holdem::RangeParser;

use crate::hand_class::{HandClass, HandClassification};
use crate::hands::CanonicalHand;

/// Error type for agent configuration loading and validation.
#[derive(Debug)]
pub enum AgentError {
    /// I/O error reading the TOML file.
    Io(std::io::Error),
    /// TOML parse error.
    Parse(toml::de::Error),
    /// Invalid hand class name in `[classes.*]`.
    InvalidClass(String),
    /// Frequency value is negative.
    InvalidFrequency(String),
    /// Range string failed to parse.
    InvalidRange(String),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Parse(e) => write!(f, "TOML parse error: {e}"),
            Self::InvalidClass(s) => write!(f, "invalid hand class: '{s}'"),
            Self::InvalidFrequency(s) => write!(f, "invalid frequency: {s}"),
            Self::InvalidRange(s) => write!(f, "invalid range: {s}"),
        }
    }
}

impl std::error::Error for AgentError {}

impl From<std::io::Error> for AgentError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<toml::de::Error> for AgentError {
    fn from(e: toml::de::Error) -> Self {
        Self::Parse(e)
    }
}

/// Game settings from the agent config.
#[derive(Debug, Clone)]
pub struct GameSettings {
    pub name: Option<String>,
    pub stack_depth: u32,
    pub bet_sizes: Vec<f32>,
}

/// Abstract action frequencies (fold/call/raise), normalized to sum to 1.0.
#[derive(Debug, Clone)]
pub struct FrequencyMap {
    pub fold: f32,
    pub call: f32,
    pub raise: f32,
}

impl FrequencyMap {
    /// The all-fold frequency map.
    pub const FOLD: Self = Self {
        fold: 1.0,
        call: 0.0,
        raise: 0.0,
    };
}

/// A fully loaded and validated agent configuration.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub game: GameSettings,
    /// Position name -> set of playable canonical hands.
    pub ranges: HashMap<String, HashSet<CanonicalHand>>,
    /// Default frequencies for unclassified / preflop in-range hands.
    pub default: FrequencyMap,
    /// Per-HandClass frequency overrides.
    pub classes: HashMap<HandClass, FrequencyMap>,
}

impl AgentConfig {
    /// Load an agent configuration from a TOML file.
    ///
    /// # Errors
    ///
    /// Returns `AgentError` if the file can't be read, parsed, or contains
    /// invalid class names, frequencies, or range strings.
    pub fn load(path: &Path) -> Result<Self, AgentError> {
        let content = std::fs::read_to_string(path)?;
        Self::from_toml(&content)
    }

    /// Parse an agent configuration from a TOML string.
    ///
    /// # Errors
    ///
    /// Returns `AgentError` if the TOML is invalid or contains
    /// invalid class names, frequencies, or range strings.
    pub fn from_toml(content: &str) -> Result<Self, AgentError> {
        let raw: RawConfig = toml::from_str(content)?;
        validate_and_build(raw)
    }

    /// Check if a hand is in the opening range for a position.
    #[must_use]
    pub fn in_range(&self, position: &str, hand: &CanonicalHand) -> bool {
        self.ranges
            .get(position)
            .is_some_and(|set| set.contains(hand))
    }

    /// Resolve frequencies for a classified hand.
    ///
    /// Walks active classes in enum order (strongest first).
    /// First class with a config entry wins. Falls back to `default`.
    #[must_use]
    pub fn resolve(&self, classification: &HandClassification) -> &FrequencyMap {
        for class in classification.iter() {
            if let Some(freq) = self.classes.get(&class) {
                return freq;
            }
        }
        &self.default
    }
}

// ============================================================================
// Raw TOML deserialization types
// ============================================================================

#[derive(serde::Deserialize)]
struct RawConfig {
    game: RawGameSettings,
    #[serde(default)]
    ranges: HashMap<String, String>,
    default: RawFrequency,
    #[serde(default)]
    classes: HashMap<String, RawFrequency>,
}

#[derive(serde::Deserialize)]
struct RawGameSettings {
    name: Option<String>,
    stack_depth: u32,
    bet_sizes: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct RawFrequency {
    fold: f32,
    call: f32,
    raise: f32,
}

// ============================================================================
// Validation and building
// ============================================================================

fn validate_and_build(raw: RawConfig) -> Result<AgentConfig, AgentError> {
    let game = GameSettings {
        name: raw.game.name,
        stack_depth: raw.game.stack_depth,
        bet_sizes: raw.game.bet_sizes,
    };

    let default = validate_frequency(&raw.default, "default")?;

    let classes = parse_class_overrides(&raw.classes)?;
    let ranges = parse_all_ranges(&raw.ranges)?;

    Ok(AgentConfig {
        game,
        ranges,
        default,
        classes,
    })
}

fn validate_frequency(raw: &RawFrequency, context: &str) -> Result<FrequencyMap, AgentError> {
    if raw.fold < 0.0 || raw.call < 0.0 || raw.raise < 0.0 {
        return Err(AgentError::InvalidFrequency(format!(
            "{context}: frequencies must be non-negative"
        )));
    }
    let sum = raw.fold + raw.call + raw.raise;
    if sum <= 0.0 {
        return Err(AgentError::InvalidFrequency(format!(
            "{context}: frequencies must sum to a positive value"
        )));
    }
    Ok(FrequencyMap {
        fold: raw.fold / sum,
        call: raw.call / sum,
        raise: raw.raise / sum,
    })
}

fn parse_class_overrides(
    raw_classes: &HashMap<String, RawFrequency>,
) -> Result<HashMap<HandClass, FrequencyMap>, AgentError> {
    let mut result = HashMap::new();
    for (name, raw_freq) in raw_classes {
        let class =
            HandClass::from_str(name).map_err(|_| AgentError::InvalidClass(name.clone()))?;
        let freq = validate_frequency(raw_freq, name)?;
        result.insert(class, freq);
    }
    Ok(result)
}

fn parse_all_ranges(
    raw_ranges: &HashMap<String, String>,
) -> Result<HashMap<String, HashSet<CanonicalHand>>, AgentError> {
    let mut result = HashMap::new();
    for (position, range_str) in raw_ranges {
        let hands = parse_range(range_str)?;
        result.insert(position.clone(), hands);
    }
    Ok(result)
}

fn parse_range(range_str: &str) -> Result<HashSet<CanonicalHand>, AgentError> {
    let flat_hands = RangeParser::parse_many(range_str)
        .map_err(|e| AgentError::InvalidRange(format!("{range_str}: {e}")))?;

    let mut canonical_set = HashSet::new();
    for flat_hand in &flat_hands {
        let cards: Vec<_> = flat_hand.iter().collect();
        if cards.len() >= 2 {
            let canonical = CanonicalHand::from_cards(*cards[0], *cards[1]);
            canonical_set.insert(canonical);
        }
    }
    Ok(canonical_set)
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    const MINIMAL_TOML: &str = r#"
[game]
stack_depth = 100
bet_sizes = [0.5, 1.0]

[default]
fold = 0.33
call = 0.34
raise = 0.33

[classes.TopPair]
fold = 0.1
call = 0.5
raise = 0.4
"#;

    const TOML_WITH_RANGES: &str = r#"
[game]
stack_depth = 100
bet_sizes = [0.5, 1.0]

[ranges]
btn = "AA,KK,QQ,AKs"
bb  = "22+,A2s+,K2s+"

[default]
fold = 0.33
call = 0.34
raise = 0.33

[classes.TopPair]
fold = 0.1
call = 0.5
raise = 0.4
"#;

    #[timed_test]
    fn load_minimal_toml() {
        let config = AgentConfig::from_toml(MINIMAL_TOML).unwrap();
        assert_eq!(config.game.stack_depth, 100);
        assert_eq!(config.game.bet_sizes, vec![0.5, 1.0]);
    }

    #[timed_test]
    fn default_frequencies_normalized() {
        let config = AgentConfig::from_toml(MINIMAL_TOML).unwrap();
        let sum = config.default.fold + config.default.call + config.default.raise;
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[timed_test]
    fn class_override_parsed() {
        let config = AgentConfig::from_toml(MINIMAL_TOML).unwrap();
        assert!(config.classes.contains_key(&HandClass::TopPair));
        let tp = &config.classes[&HandClass::TopPair];
        assert!((tp.fold - 0.1).abs() < 1e-5);
        assert!((tp.call - 0.5).abs() < 1e-5);
        assert!((tp.raise - 0.4).abs() < 1e-5);
    }

    #[timed_test]
    fn invalid_class_name_rejected() {
        let toml = r#"
[game]
stack_depth = 100
bet_sizes = [0.5]

[default]
fold = 0.5
call = 0.3
raise = 0.2

[classes.NotARealClass]
fold = 0.5
call = 0.3
raise = 0.2
"#;
        let result = AgentConfig::from_toml(toml);
        assert!(matches!(result, Err(AgentError::InvalidClass(_))));
    }

    #[timed_test]
    fn negative_frequency_rejected() {
        let toml = r#"
[game]
stack_depth = 100
bet_sizes = [0.5]

[default]
fold = -0.1
call = 0.6
raise = 0.5
"#;
        let result = AgentConfig::from_toml(toml);
        assert!(matches!(result, Err(AgentError::InvalidFrequency(_))));
    }

    #[timed_test]
    fn zero_sum_frequency_rejected() {
        let toml = r#"
[game]
stack_depth = 100
bet_sizes = [0.5]

[default]
fold = 0.0
call = 0.0
raise = 0.0
"#;
        let result = AgentConfig::from_toml(toml);
        assert!(matches!(result, Err(AgentError::InvalidFrequency(_))));
    }

    #[timed_test]
    fn unnormalized_frequencies_get_normalized() {
        let toml = r#"
[game]
stack_depth = 100
bet_sizes = [0.5]

[default]
fold = 1.0
call = 1.0
raise = 2.0
"#;
        let config = AgentConfig::from_toml(toml).unwrap();
        assert!((config.default.fold - 0.25).abs() < 1e-5);
        assert!((config.default.call - 0.25).abs() < 1e-5);
        assert!((config.default.raise - 0.5).abs() < 1e-5);
    }

    #[timed_test]
    fn range_parsing_basic() {
        let config = AgentConfig::from_toml(TOML_WITH_RANGES).unwrap();
        let btn_range = config.ranges.get("btn").unwrap();

        // AA, KK, QQ, AKs should be in the range
        assert!(btn_range.contains(&CanonicalHand::parse("AA").unwrap()));
        assert!(btn_range.contains(&CanonicalHand::parse("KK").unwrap()));
        assert!(btn_range.contains(&CanonicalHand::parse("QQ").unwrap()));
        assert!(btn_range.contains(&CanonicalHand::parse("AKs").unwrap()));

        // AKo, JJ should not
        assert!(!btn_range.contains(&CanonicalHand::parse("AKo").unwrap()));
        assert!(!btn_range.contains(&CanonicalHand::parse("JJ").unwrap()));
    }

    #[timed_test]
    fn range_plus_notation() {
        let config = AgentConfig::from_toml(TOML_WITH_RANGES).unwrap();
        let bb_range = config.ranges.get("bb").unwrap();

        // 22+ means all pairs
        assert!(bb_range.contains(&CanonicalHand::parse("AA").unwrap()));
        assert!(bb_range.contains(&CanonicalHand::parse("22").unwrap()));

        // A2s+ means all Axs
        assert!(bb_range.contains(&CanonicalHand::parse("AKs").unwrap()));
        assert!(bb_range.contains(&CanonicalHand::parse("A2s").unwrap()));

        // K2s+ means all Kxs
        assert!(bb_range.contains(&CanonicalHand::parse("KQs").unwrap()));
        assert!(bb_range.contains(&CanonicalHand::parse("K2s").unwrap()));

        // Offsuit hands not in range
        assert!(!bb_range.contains(&CanonicalHand::parse("A2o").unwrap()));
    }

    #[timed_test]
    fn in_range_helper() {
        let config = AgentConfig::from_toml(TOML_WITH_RANGES).unwrap();
        let aa = CanonicalHand::parse("AA").unwrap();
        let seven_two = CanonicalHand::parse("72o").unwrap();

        assert!(config.in_range("btn", &aa));
        assert!(!config.in_range("btn", &seven_two));
        // Non-existent position returns false
        assert!(!config.in_range("utg", &aa));
    }

    #[timed_test]
    fn resolve_first_matching_class_wins() {
        let config = AgentConfig::from_toml(MINIMAL_TOML).unwrap();

        // Classification with TopPair should resolve to the TopPair override
        let mut classification = HandClassification::new();
        classification.add(HandClass::TopPair);
        classification.add(HandClass::FlushDraw);

        let freq = config.resolve(&classification);
        assert!((freq.fold - 0.1).abs() < 1e-5);
        assert!((freq.call - 0.5).abs() < 1e-5);
        assert!((freq.raise - 0.4).abs() < 1e-5);
    }

    #[timed_test]
    fn resolve_falls_back_to_default() {
        let config = AgentConfig::from_toml(MINIMAL_TOML).unwrap();

        // Classification with only FlushDraw (no override) → default
        let mut classification = HandClassification::new();
        classification.add(HandClass::FlushDraw);

        let freq = config.resolve(&classification);
        let sum = freq.fold + freq.call + freq.raise;
        assert!((sum - 1.0).abs() < 1e-5);
        // Should be the default frequencies
        assert!((freq.fold - 0.33).abs() < 0.01);
    }

    #[timed_test]
    fn resolve_empty_classification_uses_default() {
        let config = AgentConfig::from_toml(MINIMAL_TOML).unwrap();
        let classification = HandClassification::new();
        let freq = config.resolve(&classification);
        assert!((freq.fold - 0.33).abs() < 0.01);
    }

    #[timed_test]
    fn resolve_enum_order_strongest_first() {
        // If a hand has both Set and FlushDraw, and config overrides both,
        // Set (index 5) should win over FlushDraw (index 16) since it comes first
        let toml = r#"
[game]
stack_depth = 100
bet_sizes = [0.5]

[default]
fold = 0.5
call = 0.3
raise = 0.2

[classes.Set]
fold = 0.0
call = 0.2
raise = 0.8

[classes.FlushDraw]
fold = 0.1
call = 0.6
raise = 0.3
"#;
        let config = AgentConfig::from_toml(toml).unwrap();
        let mut classification = HandClassification::new();
        classification.add(HandClass::Set);
        classification.add(HandClass::FlushDraw);

        let freq = config.resolve(&classification);
        // Set should win (raise = 0.8)
        assert!((freq.raise - 0.8).abs() < 1e-5);
    }

    #[timed_test]
    fn invalid_range_rejected() {
        let toml = r#"
[game]
stack_depth = 100
bet_sizes = [0.5]

[ranges]
btn = "ZZZ_INVALID"

[default]
fold = 0.5
call = 0.3
raise = 0.2
"#;
        let result = AgentConfig::from_toml(toml);
        assert!(matches!(result, Err(AgentError::InvalidRange(_))));
    }

    #[timed_test]
    fn multiple_class_overrides() {
        let toml = r#"
[game]
stack_depth = 100
bet_sizes = [0.5, 1.0]

[default]
fold = 0.5
call = 0.3
raise = 0.2

[classes.StraightFlush]
fold = 0.0
call = 0.1
raise = 0.9

[classes.FourOfAKind]
fold = 0.0
call = 0.15
raise = 0.85

[classes.TopPair]
fold = 0.05
call = 0.4
raise = 0.55

[classes.Gutshot]
fold = 0.35
call = 0.3
raise = 0.35
"#;
        let config = AgentConfig::from_toml(toml).unwrap();
        assert_eq!(config.classes.len(), 4);
        assert!(config.classes.contains_key(&HandClass::StraightFlush));
        assert!(config.classes.contains_key(&HandClass::FourOfAKind));
        assert!(config.classes.contains_key(&HandClass::TopPair));
        assert!(config.classes.contains_key(&HandClass::Gutshot));
    }

    #[timed_test]
    fn no_ranges_section_ok() {
        let config = AgentConfig::from_toml(MINIMAL_TOML).unwrap();
        assert!(config.ranges.is_empty());
    }

    #[timed_test]
    fn load_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_agent.toml");
        std::fs::write(&path, MINIMAL_TOML).unwrap();

        let config = AgentConfig::load(&path).unwrap();
        assert_eq!(config.game.stack_depth, 100);
    }

    #[timed_test]
    fn load_nonexistent_file_errors() {
        let result = AgentConfig::load(Path::new("/tmp/nonexistent_agent_xyz.toml"));
        assert!(matches!(result, Err(AgentError::Io(_))));
    }

    #[timed_test]
    fn load_tight_weak_agent() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("agents/tight_weak.toml");
        let config = AgentConfig::load(&path).unwrap();
        assert_eq!(config.game.stack_depth, 100);
        assert_eq!(config.classes.len(), 20);
        assert!(config.ranges.contains_key("btn"));
        assert!(config.ranges.contains_key("bb"));
    }

    #[timed_test]
    fn load_tight_aggressive_agent() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("agents/tight_aggressive.toml");
        let config = AgentConfig::load(&path).unwrap();
        assert_eq!(config.game.stack_depth, 100);
        assert_eq!(config.classes.len(), 20);
        // TAG should have wider ranges than tight-weak
        let tag_btn = config.ranges.get("btn").unwrap().len();
        assert!(tag_btn > 50, "TAG btn range should be wide, got {tag_btn}");
    }

    #[timed_test]
    fn load_loose_aggressive_agent() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("agents/loose_aggressive.toml");
        let config = AgentConfig::load(&path).unwrap();
        assert_eq!(config.game.stack_depth, 100);
        assert_eq!(config.classes.len(), 20);
        // LAG default should be raise-heavy
        assert!(config.default.raise > config.default.fold);
    }
}
