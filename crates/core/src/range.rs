//! Player range tracking for strategy exploration and subgame solving.
//!
//! `PlayerRange` holds reaching probabilities for 169 canonical starting hands.
//! Shared between the explorer UI and the future real-time postflop solver.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

pub const NUM_HANDS: usize = 169;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RangeSource {
    Computed,
    Edited,
    Manual,
}

/// Serde helper for `[f64; NUM_HANDS]` since serde's derive only supports
/// arrays up to 32 elements.
mod serde_hands {
    use super::NUM_HANDS;
    use serde::de::{self, SeqAccess, Visitor};
    use serde::ser::SerializeSeq;
    use serde::{Deserializer, Serializer};
    use std::fmt;

    pub fn serialize<S: Serializer>(hands: &[f64; NUM_HANDS], ser: S) -> Result<S::Ok, S::Error> {
        let mut seq = ser.serialize_seq(Some(NUM_HANDS))?;
        for &v in hands {
            seq.serialize_element(&v)?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<[f64; NUM_HANDS], D::Error> {
        struct HandsVisitor;

        impl<'de> Visitor<'de> for HandsVisitor {
            type Value = [f64; NUM_HANDS];

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "an array of {NUM_HANDS} floats")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut arr = [0.0f64; NUM_HANDS];
                for (i, slot) in arr.iter_mut().enumerate() {
                    *slot = seq
                        .next_element()?
                        .ok_or_else(|| de::Error::invalid_length(i, &self))?;
                }
                Ok(arr)
            }
        }

        de.deserialize_seq(HandsVisitor)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerRange {
    #[serde(with = "serde_hands")]
    pub hands: [f64; NUM_HANDS],
    pub source: RangeSource,
    #[serde(default, skip_serializing_if = "HashSet::is_empty")]
    pub overrides: HashSet<usize>,
}

impl PlayerRange {
    #[must_use]
    pub fn new() -> Self {
        Self {
            hands: [1.0; NUM_HANDS],
            source: RangeSource::Computed,
            overrides: HashSet::new(),
        }
    }

    /// Multiply each hand's reaching probability by the action probability.
    /// Hands with manual overrides are not modified.
    pub fn multiply_action(&mut self, action_probs: &[f64; NUM_HANDS]) {
        for (i, (hand, &prob)) in self.hands.iter_mut().zip(action_probs).enumerate() {
            if !self.overrides.contains(&i) {
                *hand *= prob;
            }
        }
    }

    /// Manually set a hand's reaching probability.
    ///
    /// # Panics
    ///
    /// Panics if `index >= NUM_HANDS`.
    pub fn set_hand(&mut self, index: usize, weight: f64) {
        assert!(index < NUM_HANDS, "hand index out of range");
        self.hands[index] = weight.clamp(0.0, 1.0);
        self.overrides.insert(index);
        if self.source == RangeSource::Computed {
            self.source = RangeSource::Edited;
        }
    }
}

impl Default for PlayerRange {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_range_is_full() {
        let r = PlayerRange::new();
        assert_eq!(r.hands.len(), 169);
        assert!(r.hands.iter().all(|&h| (h - 1.0).abs() < f64::EPSILON));
        assert_eq!(r.source, RangeSource::Computed);
        assert!(r.overrides.is_empty());
    }

    #[test]
    fn multiply_action_narrows_range() {
        let mut r = PlayerRange::new();
        let mut action_probs = [0.0f64; 169];
        action_probs[0] = 0.8;
        action_probs[1] = 0.0;
        action_probs[2] = 1.0;
        r.multiply_action(&action_probs);
        assert!((r.hands[0] - 0.8).abs() < 1e-9);
        assert!((r.hands[1] - 0.0).abs() < 1e-9);
        assert!((r.hands[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cumulative_multiply() {
        let mut r = PlayerRange::new();
        let mut probs1 = [1.0f64; 169];
        probs1[0] = 0.5;
        r.multiply_action(&probs1);
        let mut probs2 = [1.0f64; 169];
        probs2[0] = 0.4;
        r.multiply_action(&probs2);
        assert!((r.hands[0] - 0.2).abs() < 1e-9);
    }

    #[test]
    fn manual_override_sets_value() {
        let mut r = PlayerRange::new();
        r.set_hand(0, 0.75);
        assert!((r.hands[0] - 0.75).abs() < 1e-9);
        assert!(r.overrides.contains(&0));
        assert_eq!(r.source, RangeSource::Edited);
    }

    #[test]
    fn override_survives_multiply() {
        let mut r = PlayerRange::new();
        r.set_hand(0, 0.75);
        let action_probs = [0.5f64; 169];
        r.multiply_action(&action_probs);
        // Hand 0 was overridden, so multiply should NOT change it
        assert!((r.hands[0] - 0.75).abs() < 1e-9);
        // Hand 1 was not overridden, so it should be multiplied
        assert!((r.hands[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn serialization_roundtrip() {
        let mut r = PlayerRange::new();
        r.set_hand(5, 0.33);
        let json = serde_json::to_string(&r).unwrap();
        let r2: PlayerRange = serde_json::from_str(&json).unwrap();
        assert!((r2.hands[5] - 0.33).abs() < 1e-9);
        assert!(r2.overrides.contains(&5));
    }
}
