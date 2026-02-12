//! Configurable rule-based poker agents.
//!
//! Agents are defined in TOML files that map `HandClass` to action frequencies,
//! with optional preflop range filtering. They produce the same `StrategyMatrix`
//! output as trained blueprint strategies, enabling UI development without training.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::str::FromStr;

use rs_poker::holdem::RangeParser;
use thiserror::Error;

use crate::hand_class::{HandClass, HandClassification};
use crate::hands::CanonicalHand;

/// Error type for agent configuration loading and validation.
#[derive(Debug, Error)]
pub enum AgentError {
    /// I/O error reading the TOML file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// TOML parse error.
    #[error("TOML parse error: {0}")]
    Parse(#[from] toml::de::Error),
    /// Invalid hand class name in `[classes.*]`.
    #[error("invalid hand class: '{0}'")]
    InvalidClass(String),
    /// Frequency value is negative.
    #[error("invalid frequency: {0}")]
    InvalidFrequency(String),
    /// Range string failed to parse.
    #[error("invalid range: {0}")]
    InvalidRange(String),
}

/// Game settings from the agent config.
#[derive(Debug, Clone)]
pub struct GameSettings {
    pub name: Option<String>,
    pub stack_depth: u32,
    pub bet_sizes: Vec<f32>,
}

/// Abstract action frequencies (fold/call/raise), normalized to sum to 1.0.
///
/// When `raise_sizes` is `Some`, it maps bet-size keys (e.g. `"0.5"`, `"1.0"`,
/// `"allin"`) to individual frequencies that sum to `raise`.
/// When `None`, `raise` is split evenly across available raise actions.
#[derive(Debug, Clone)]
pub struct FrequencyMap {
    pub fold: f32,
    pub call: f32,
    pub raise: f32,
    /// Per-size raise distribution. Keys are pot-fraction strings or `"allin"`.
    pub raise_sizes: Option<HashMap<String, f32>>,
}

impl FrequencyMap {
    pub const FOLD: Self = Self {
        fold: 1.0,
        call: 0.0,
        raise: 0.0,
        raise_sizes: None,
    };
    pub const CALL: Self = Self {
        fold: 0.0,
        call: 1.0,
        raise: 0.0,
        raise_sizes: None,
    };
    pub const RAISE: Self = Self {
        fold: 0.0,
        call: 0.0,
        raise: 1.0,
        raise_sizes: None,
    };
}

/// Preflop action ranges for a single position.
///
/// Hands in `raise` get raise action, hands in `call` get call action,
/// everything else folds. Raise takes priority over call if a hand
/// appears in both.
#[derive(Debug, Clone)]
pub struct PositionRanges {
    pub raise: HashSet<CanonicalHand>,
    pub call: HashSet<CanonicalHand>,
    /// Frequency distribution for hands in the raise range.
    /// Defaults to `FrequencyMap::RAISE` (100% raise, split evenly) when
    /// no `raise_sizes` are configured.
    pub raise_freq: FrequencyMap,
}

/// A fully loaded and validated agent configuration.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub game: GameSettings,
    /// Position name -> preflop action ranges (raise / call / fold).
    pub ranges: HashMap<String, PositionRanges>,
    /// Default frequencies for unclassified postflop hands.
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

    /// Resolve preflop frequency for a hand at a position.
    ///
    /// Returns `RAISE` if the hand is in the raise range, `CALL` if in the
    /// call range, `FOLD` otherwise. Raise takes priority over call.
    #[must_use]
    pub fn preflop_frequency(&self, position: &str, hand: &CanonicalHand) -> &FrequencyMap {
        match self.ranges.get(position) {
            Some(pos) if pos.raise.contains(hand) => &pos.raise_freq,
            Some(pos) if pos.call.contains(hand) => &FrequencyMap::CALL,
            _ => &FrequencyMap::FOLD,
        }
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
    ranges: HashMap<String, RawPositionRanges>,
    default: RawFrequency,
    #[serde(default)]
    classes: HashMap<String, RawFrequency>,
}

#[derive(serde::Deserialize, Default)]
struct RawPositionRanges {
    raise: Option<String>,
    call: Option<String>,
    /// Per-size raise distribution for the raise range (e.g. `{"0.5" = 0.3, "1.0" = 0.6, allin = 0.1}`).
    #[serde(default)]
    raise_sizes: Option<HashMap<String, f32>>,
}

#[derive(serde::Deserialize)]
struct RawGameSettings {
    name: Option<String>,
    stack_depth: u32,
    bet_sizes: Vec<f32>,
}

/// Raw frequency from TOML. Supports two formats:
/// - Legacy: `raise = 0.55` (split evenly across sizes)
/// - Per-size: `[raises]` table with `"0.5" = 0.3, "1.0" = 0.2, allin = 0.05`
#[derive(serde::Deserialize)]
struct RawFrequency {
    fold: f32,
    call: f32,
    /// Legacy single-value raise (split evenly). Ignored when `raises` is present.
    #[serde(default)]
    raise: Option<f32>,
    /// Per-size raise frequencies. Keys are pot fraction strings or `"allin"`.
    #[serde(default)]
    raises: Option<HashMap<String, f32>>,
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
    if raw.fold < 0.0 || raw.call < 0.0 {
        return Err(AgentError::InvalidFrequency(format!(
            "{context}: frequencies must be non-negative"
        )));
    }

    let (raise_total, raise_sizes) = if let Some(ref sizes) = raw.raises {
        validate_raise_sizes(sizes, context)?
    } else {
        let r = raw.raise.unwrap_or(0.0);
        if r < 0.0 {
            return Err(AgentError::InvalidFrequency(format!(
                "{context}: raise frequency must be non-negative"
            )));
        }
        (r, None)
    };

    let sum = raw.fold + raw.call + raise_total;
    if sum <= 0.0 {
        return Err(AgentError::InvalidFrequency(format!(
            "{context}: frequencies must sum to a positive value"
        )));
    }

    Ok(FrequencyMap {
        fold: raw.fold / sum,
        call: raw.call / sum,
        raise: raise_total / sum,
        raise_sizes: raise_sizes
            .map(|sizes| sizes.into_iter().map(|(k, v)| (k, v / sum)).collect()),
    })
}

fn validate_raise_sizes(
    sizes: &HashMap<String, f32>,
    context: &str,
) -> Result<(f32, Option<HashMap<String, f32>>), AgentError> {
    for (key, &val) in sizes {
        if val < 0.0 {
            return Err(AgentError::InvalidFrequency(format!(
                "{context}: raise size '{key}' must be non-negative"
            )));
        }
    }
    let total: f32 = sizes.values().sum();
    Ok((total, Some(sizes.clone())))
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
    raw_ranges: &HashMap<String, RawPositionRanges>,
) -> Result<HashMap<String, PositionRanges>, AgentError> {
    let mut result = HashMap::new();
    for (position, raw_pos) in raw_ranges {
        let raise = match &raw_pos.raise {
            Some(s) if !s.is_empty() => parse_range(s)?,
            _ => HashSet::new(),
        };
        let call = match &raw_pos.call {
            Some(s) if !s.is_empty() => parse_range(s)?,
            _ => HashSet::new(),
        };
        let raise_freq = build_raise_freq(raw_pos.raise_sizes.as_ref(), position)?;
        result.insert(position.clone(), PositionRanges { raise, call, raise_freq });
    }
    Ok(result)
}

fn build_raise_freq(
    raise_sizes: Option<&HashMap<String, f32>>,
    context: &str,
) -> Result<FrequencyMap, AgentError> {
    let Some(sizes) = raise_sizes else {
        return Ok(FrequencyMap {
            fold: 0.0,
            call: 0.0,
            raise: 1.0,
            raise_sizes: None,
        });
    };

    let (total, validated) = validate_raise_sizes(sizes, context)?;
    if total <= 0.0 {
        return Err(AgentError::InvalidFrequency(format!(
            "{context}: raise sizes must sum to a positive value"
        )));
    }

    Ok(FrequencyMap {
        fold: 0.0,
        call: 0.0,
        raise: 1.0,
        raise_sizes: validated
            .map(|s| s.into_iter().map(|(k, v)| (k, v / total)).collect()),
    })
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

[classes.Pair]
fold = 0.1
call = 0.5
raise = 0.4
"#;

    const TOML_WITH_RANGES: &str = r#"
[game]
stack_depth = 100
bet_sizes = [0.5, 1.0]

[ranges.btn]
raise = "AA,KK,QQ,AKs"
call = "JJ,TT,AQs,AJs"

[ranges.btn.raise_sizes]
"0.5" = 0.3
"1.0" = 0.6
allin = 0.1

[ranges.bb]
raise = "QQ+,AKs"
call = "22+,A2s+,K2s+"

[default]
fold = 0.33
call = 0.34
raise = 0.33

[classes.Pair]
fold = 0.1
call = 0.5

[classes.Pair.raises]
"0.5" = 0.3
"1.0" = 0.08
allin = 0.02
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
        assert!(config.classes.contains_key(&HandClass::Pair));
        let tp = &config.classes[&HandClass::Pair];
        assert!((tp.fold - 0.1).abs() < 1e-5);
        assert!((tp.call - 0.5).abs() < 1e-5);
        assert!((tp.raise - 0.4).abs() < 1e-5);
        assert!(tp.raise_sizes.is_none()); // legacy single-value raise
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
    fn range_parsing_raise_and_call() {
        let config = AgentConfig::from_toml(TOML_WITH_RANGES).unwrap();
        let btn = config.ranges.get("btn").unwrap();

        // AA, KK, QQ, AKs in raise range
        assert!(btn.raise.contains(&CanonicalHand::parse("AA").unwrap()));
        assert!(btn.raise.contains(&CanonicalHand::parse("KK").unwrap()));
        assert!(btn.raise.contains(&CanonicalHand::parse("AKs").unwrap()));

        // JJ, TT, AQs in call range
        assert!(btn.call.contains(&CanonicalHand::parse("JJ").unwrap()));
        assert!(btn.call.contains(&CanonicalHand::parse("TT").unwrap()));
        assert!(btn.call.contains(&CanonicalHand::parse("AQs").unwrap()));

        // AKo, 99 not in either
        assert!(!btn.raise.contains(&CanonicalHand::parse("AKo").unwrap()));
        assert!(!btn.call.contains(&CanonicalHand::parse("AKo").unwrap()));
    }

    #[timed_test]
    fn range_plus_notation() {
        let config = AgentConfig::from_toml(TOML_WITH_RANGES).unwrap();
        let bb = config.ranges.get("bb").unwrap();

        // QQ+, AKs in raise range
        assert!(bb.raise.contains(&CanonicalHand::parse("AA").unwrap()));
        assert!(bb.raise.contains(&CanonicalHand::parse("QQ").unwrap()));
        assert!(bb.raise.contains(&CanonicalHand::parse("AKs").unwrap()));

        // 22+ in call range (overlaps with raise for QQ+)
        assert!(bb.call.contains(&CanonicalHand::parse("22").unwrap()));
        assert!(bb.call.contains(&CanonicalHand::parse("TT").unwrap()));

        // Offsuit hands not in call range
        assert!(!bb.call.contains(&CanonicalHand::parse("A2o").unwrap()));
    }

    #[timed_test]
    fn preflop_frequency_raise_priority() {
        let config = AgentConfig::from_toml(TOML_WITH_RANGES).unwrap();
        let aa = CanonicalHand::parse("AA").unwrap();
        let jj = CanonicalHand::parse("JJ").unwrap();
        let seven_two = CanonicalHand::parse("72o").unwrap();

        // AA in btn raise range → raise with per-size distribution
        let freq = config.preflop_frequency("btn", &aa);
        assert!((freq.raise - 1.0).abs() < 1e-5);
        let sizes = freq.raise_sizes.as_ref().unwrap();
        assert!((sizes["0.5"] - 0.3).abs() < 1e-5);
        assert!((sizes["1.0"] - 0.6).abs() < 1e-5);
        assert!((sizes["allin"] - 0.1).abs() < 1e-5);

        // JJ in call range → call
        let freq = config.preflop_frequency("btn", &jj);
        assert!((freq.call - 1.0).abs() < 1e-5);

        // 72o in neither → fold
        let freq = config.preflop_frequency("btn", &seven_two);
        assert!((freq.fold - 1.0).abs() < 1e-5);

        // Non-existent position → fold
        let freq = config.preflop_frequency("utg", &aa);
        assert!((freq.fold - 1.0).abs() < 1e-5);
    }

    #[timed_test]
    fn preflop_frequency_raise_beats_call_overlap() {
        let config = AgentConfig::from_toml(TOML_WITH_RANGES).unwrap();
        // BB: AA is in both raise (QQ+) and call (22+). Raise wins.
        // BB has no raise_sizes → raise_sizes is None (split evenly).
        let aa = CanonicalHand::parse("AA").unwrap();
        let freq = config.preflop_frequency("bb", &aa);
        assert!((freq.raise - 1.0).abs() < 1e-5);
        assert!(freq.raise_sizes.is_none());
    }

    #[timed_test]
    fn class_override_with_per_size_raises() {
        let config = AgentConfig::from_toml(TOML_WITH_RANGES).unwrap();
        let tp = &config.classes[&HandClass::Pair];
        assert!((tp.fold - 0.1).abs() < 1e-5);
        assert!((tp.call - 0.5).abs() < 1e-5);
        assert!((tp.raise - 0.4).abs() < 1e-5);
        let sizes = tp.raise_sizes.as_ref().unwrap();
        assert!((sizes["0.5"] - 0.3).abs() < 1e-5);
        assert!((sizes["1.0"] - 0.08).abs() < 1e-5);
        assert!((sizes["allin"] - 0.02).abs() < 1e-5);
    }

    #[timed_test]
    fn legacy_single_raise_still_works() {
        let toml = r#"
[game]
stack_depth = 100
bet_sizes = [0.5, 1.0]

[default]
fold = 0.3
call = 0.3
raise = 0.4
"#;
        let config = AgentConfig::from_toml(toml).unwrap();
        assert!((config.default.raise - 0.4).abs() < 1e-5);
        assert!(config.default.raise_sizes.is_none());
    }

    #[timed_test]
    fn resolve_first_matching_class_wins() {
        let config = AgentConfig::from_toml(MINIMAL_TOML).unwrap();

        // Classification with Pair should resolve to the Pair override
        let mut classification = HandClassification::new();
        classification.add(HandClass::Pair);
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

[ranges.btn]
raise = "ZZZ_INVALID"

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

[classes.Pair]
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
        assert!(config.classes.contains_key(&HandClass::Pair));
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
        assert_eq!(config.classes.len(), 19);
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
        assert_eq!(config.classes.len(), 19);
        // TAG should have wide playable range and non-empty raise range
        let btn = config.ranges.get("btn").unwrap();
        assert!(btn.call.len() > 50, "TAG btn call range should be wide, got {}", btn.call.len());
        assert!(!btn.raise.is_empty(), "TAG btn should have a raise range");
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
        assert_eq!(config.classes.len(), 19);
        // LAG default should be raise-heavy
        assert!(config.default.raise > config.default.fold);
    }
}
