//! Numeric u64 info set key encoding for fast hashing and zero-allocation lookups.
//!
//! ## Bit Layout
//!
//! ```text
//! Bits 63-36: hand/bucket   (28 bits)
//! Bits 35-34: street         (2 bits)
//! Bits 33-29: spr_bucket     (5 bits) — min(eff_stack*2/pot, 31), half-SPR units
//! Bits 28-24: (reserved)     (5 bits)
//! Bits 23-0:  action slots  (24 bits) — up to 6 actions × 4 bits
//! ```
//!
//! Action encoding (4 bits): 0=empty, 1=fold, 2=check, 3=call,
//! 4-8=bet idx 0-4, 9-13=raise idx 0-4, 14=bet all-in, 15=raise all-in.

use crate::blueprint::{AbstractionModeConfig, BlueprintStrategy};
use crate::game::{ALL_IN, Action};
use crate::hand_class::{HandClass, classify, intra_class_strength};
use crate::poker::{Card, Suit, Value};
use crate::showdown_equity;

const HAND_SHIFT: u32 = 36;
const STREET_SHIFT: u32 = 34;
const SPR_SHIFT: u32 = 29;

/// Compute SPR bucket: `min(eff_stack * 2 / pot, 31)`.
///
/// Half-SPR units give fine resolution at low SPR where decisions matter most.
/// Returns 31 when pot is zero (infinite SPR).
#[must_use]
pub fn spr_bucket(pot: u32, eff_stack: u32) -> u32 {
    if pot == 0 {
        return 31;
    }
    (eff_stack * 2 / pot).min(31)
}

/// A packed u64 information set key.
///
/// Encodes hand/bucket, street, SPR/depth buckets, and up to 6 actions
/// in a single 64-bit integer for allocation-free hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InfoKey(u64);

impl InfoKey {
    /// Build a key from its components.
    ///
    /// # Arguments
    /// * `hand_or_bucket` - Canonical hand index (0-168) or classification bits (up to 28 bits)
    /// * `street` - 0=Preflop, 1=Flop, 2=Turn, 3=River
    /// * `spr_bucket` - `min(eff_stack * 2 / pot, 31)` (5 bits, max 31)
    /// * `actions` - Slice of encoded action codes (from `encode_action`)
    #[must_use]
    pub fn new(hand_or_bucket: u32, street: u8, spr_bucket: u32, actions: &[u8]) -> Self {
        let mut key: u64 = 0;
        key |= (u64::from(hand_or_bucket) & 0xFFF_FFFF) << HAND_SHIFT;
        key |= (u64::from(street) & 0x3) << STREET_SHIFT;
        key |= (u64::from(spr_bucket) & 0x1F) << SPR_SHIFT;

        // Pack up to 6 actions into bits 23..0 (4 bits each, MSB-first)
        // Action 0 → bits 23-20, action 1 → bits 19-16, ..., action 5 → bits 3-0
        for (i, &code) in actions.iter().take(6).enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let shift = 20 - (i as u32) * 4;
            key |= (u64::from(code) & 0xF) << shift;
        }

        Self(key)
    }

    /// Wrap a raw u64 as an `InfoKey`.
    #[must_use]
    #[inline]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Extract the raw u64 value.
    #[must_use]
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Extract SPR bucket (5 bits).
    #[must_use]
    pub const fn spr_bucket(self) -> u32 {
        ((self.0 >> SPR_SHIFT) & 0x1F) as u32
    }

    /// Extract the street (2 bits): 0=Preflop, 1=Flop, 2=Turn, 3=River.
    #[must_use]
    pub const fn street(self) -> u8 {
        ((self.0 >> STREET_SHIFT) & 0x3) as u8
    }

    /// Extract the hand/bucket bits (28 bits).
    #[must_use]
    pub const fn hand_bits(self) -> u32 {
        ((self.0 >> HAND_SHIFT) & 0xFFF_FFFF) as u32
    }

    /// Extract the action slots (lower 24 bits).
    #[must_use]
    pub const fn actions_bits(self) -> u32 {
        (self.0 & 0xFF_FFFF) as u32
    }

    /// Return a new key with a modified SPR bucket.
    #[must_use]
    pub const fn with_spr(self, spr_bucket: u32) -> Self {
        let mask = !(0x1F << SPR_SHIFT);
        let cleared = self.0 & mask;
        let new_bits = (spr_bucket as u64 & 0x1F) << SPR_SHIFT;
        Self(cleared | new_bits)
    }
}

/// Encode an [`Action`] into a 4-bit code.
///
/// 0=empty, 1=fold, 2=check, 3=call, 4-8=bet idx 0-4,
/// 9-13=raise idx 0-4, 14=bet all-in, 15=raise all-in.
#[must_use]
pub fn encode_action(action: Action) -> u8 {
    match action {
        Action::Fold => 1,
        Action::Check => 2,
        Action::Call => 3,
        Action::Bet(idx) if idx == ALL_IN => 14,
        Action::Bet(idx) => 4 + idx.min(4) as u8,
        Action::Raise(idx) if idx == ALL_IN => 15,
        Action::Raise(idx) => 9 + idx.min(4) as u8,
    }
}

/// Encode the 28-bit hand field for `HandClassV2` mode.
///
/// Packs made-hand class ID, quantized strength, quantized equity, and draw
/// flags into 28 bits:
///
/// ```text
/// Bits 27-23: Made hand class ID     (5 bits, 0-12, or 13 if draw-only)
/// Bits 22-19: Strength               (4 bits, max — quantized from 1-14)
/// Bits 18-15: Equity bin             (4 bits, max — quantized from 0-15)
/// Bits 14-8:  Draw flags             (6 bits used, 1 spare)
/// Bits 7-0:   Spare                  (8 bits, zero)
/// ```
///
/// When `strength_bits` or `equity_bits` is less than 4, the value is
/// right-shifted to quantize into fewer bins.
#[must_use]
pub fn encode_hand_v2(
    class_id: u8,
    strength: u8,
    equity_bin: u8,
    draw_flags: u8,
    strength_bits: u8,
    equity_bits: u8,
) -> u32 {
    // Quantize strength (1-14, stored as 0-13) into strength_bits
    let s = if strength_bits == 0 {
        0u32
    } else {
        let val = strength.saturating_sub(1).min(13); // 0-13
        let quantized = u32::from(val) >> (4 - strength_bits);
        quantized & ((1 << strength_bits) - 1)
    };

    // Quantize equity bin (0-15) into equity_bits
    let e = if equity_bits == 0 {
        0u32
    } else {
        let quantized = u32::from(equity_bin) >> (4 - equity_bits);
        quantized & ((1 << equity_bits) - 1)
    };

    let mut bits: u32 = 0;
    bits |= (u32::from(class_id) & 0x1F) << 23; // 5 bits at 27-23
    bits |= (s & 0xF) << 19; // 4 bits at 22-19
    bits |= (e & 0xF) << 15; // 4 bits at 18-15
    let draw_mask = (1u32 << HandClass::NUM_DRAWS) - 1;
    bits |= (u32::from(draw_flags) & draw_mask) << 8; // draw bits at 13-8
    bits
}

/// Compute the 28-bit hand field for `HandClassV2` mode from scratch.
///
/// Classifies the hand, computes intra-class strength and equity, then
/// packs everything via [`encode_hand_v2`]. Returns 0 if classification fails.
#[must_use]
pub fn compute_hand_bits_v2(
    hole: [Card; 2],
    board: &[Card],
    strength_bits: u8,
    equity_bits: u8,
) -> u32 {
    let Ok(classification) = classify(hole, board) else {
        return 0;
    };
    let made_id = classification.strongest_made_id();
    let draw_flags = classification.draw_flags();
    let strength = if HandClass::is_made_hand_id(made_id) {
        intra_class_strength(hole, board, HandClass::ALL[made_id as usize])
    } else {
        1
    };
    let equity = showdown_equity::compute_equity(hole, board);
    let eq_bin = showdown_equity::equity_bin(equity, 1u8 << equity_bits);
    encode_hand_v2(
        made_id,
        strength,
        eq_bin,
        draw_flags,
        strength_bits,
        equity_bits,
    )
}

/// Map a canonical hand to a unique index in 0..169.
///
/// Canonical hands: 13 pairs + 78 suited combos + 78 offsuit combos.
/// Index layout: pairs first (AA=0..22=12), then suited (AKs=13..),
/// then offsuit (...=91..168).
#[must_use]
pub fn canonical_hand_index(holding: [Card; 2]) -> u16 {
    let r1 = rank_ordinal(holding[0].value);
    let r2 = rank_ordinal(holding[1].value);
    let (high, low) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };
    let suited = holding[0].suit == holding[1].suit;

    if high == low {
        // Pair: index 0..12 (A=0, K=1, ..., 2=12)
        u16::from(high)
    } else if suited {
        // Suited: 13 pairs already used, then upper triangle
        // For (high, low) with high > low: index = 13 + triangle_offset(high, low)
        13 + triangle_index(high, low)
    } else {
        // Offsuit: 13 + 78 suited = 91, then same triangle
        91 + triangle_index(high, low)
    }
}

/// Map a canonical hand string (e.g. "AKs", "QQ", "72o") to its index.
///
/// Returns `None` if the string is not a valid canonical hand.
#[must_use]
pub fn canonical_hand_index_from_str(hand: &str) -> Option<u16> {
    let chars: Vec<char> = hand.chars().collect();
    if chars.len() < 2 || chars.len() > 3 {
        return None;
    }

    let r1 = rank_ordinal_from_char(chars[0])?;
    let r2 = rank_ordinal_from_char(chars[1])?;
    let (high, low) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };

    if high == low {
        Some(u16::from(high))
    } else {
        let suited = chars.get(2) == Some(&'s');
        if suited {
            Some(13 + triangle_index(high, low))
        } else {
            Some(91 + triangle_index(high, low))
        }
    }
}

/// Upper-triangle index for (high, low) where high > low.
///
/// Enumerates pairs (high, low) with high in 0..12, low in 0..high.
/// For high=1,low=0 → 0; high=2,low=0 → 1; high=2,low=1 → 2; ...
fn triangle_index(high: u8, low: u8) -> u16 {
    // Number of pairs before row `high` = high*(high-1)/2
    let base = u16::from(high) * (u16::from(high) - 1) / 2;
    base + u16::from(low)
}

/// Map card rank to ordinal: A=0, K=1, Q=2, ..., 2=12.
fn rank_ordinal(value: Value) -> u8 {
    match value {
        Value::Ace => 0,
        Value::King => 1,
        Value::Queen => 2,
        Value::Jack => 3,
        Value::Ten => 4,
        Value::Nine => 5,
        Value::Eight => 6,
        Value::Seven => 7,
        Value::Six => 8,
        Value::Five => 9,
        Value::Four => 10,
        Value::Three => 11,
        Value::Two => 12,
    }
}

/// Map a rank character to its ordinal.
fn rank_ordinal_from_char(c: char) -> Option<u8> {
    match c {
        'A' => Some(0),
        'K' => Some(1),
        'Q' => Some(2),
        'J' => Some(3),
        'T' => Some(4),
        '9' => Some(5),
        '8' => Some(6),
        '7' => Some(7),
        '6' => Some(8),
        '5' => Some(9),
        '4' => Some(10),
        '3' => Some(11),
        '2' => Some(12),
        _ => None,
    }
}

/// Map a rank character and suited flag to representative cards.
///
/// Uses Spade for first card; second card is same suit if suited,
/// Heart if offsuit. For pairs, uses Spade+Heart.
#[must_use]
pub fn cards_from_rank_chars(rank1: char, rank2: char, suited: bool) -> Option<[Card; 2]> {
    let v1 = value_from_char(rank1)?;
    let v2 = value_from_char(rank2)?;
    let suit2 = if suited { Suit::Spade } else { Suit::Heart };
    Some([Card::new(v1, Suit::Spade), Card::new(v2, suit2)])
}

fn value_from_char(c: char) -> Option<Value> {
    match c {
        'A' => Some(Value::Ace),
        'K' => Some(Value::King),
        'Q' => Some(Value::Queen),
        'J' => Some(Value::Jack),
        'T' => Some(Value::Ten),
        '9' => Some(Value::Nine),
        '8' => Some(Value::Eight),
        '7' => Some(Value::Seven),
        '6' => Some(Value::Six),
        '5' => Some(Value::Five),
        '4' => Some(Value::Four),
        '3' => Some(Value::Three),
        '2' => Some(Value::Two),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Key translation API
// ---------------------------------------------------------------------------

const STREET_NAMES: [&str; 4] = ["preflop", "flop", "turn", "river"];

const ACTION_CODE_NAMES: [&str; 16] = [
    "empty",
    "fold",
    "check",
    "call",
    "bet0",
    "bet1",
    "bet2",
    "bet3",
    "bet4",
    "raise0",
    "raise1",
    "raise2",
    "raise3",
    "raise4",
    "bet-allin",
    "raise-allin",
];

/// Decoded components of an info set key with both hex and human-readable forms.
#[derive(Debug, Clone)]
pub struct KeyDescription {
    /// Raw u64 key value.
    pub raw: u64,
    /// 28-bit hand field.
    pub hand_bits: u32,
    /// Human-readable hand label (e.g. "AKs", "`Pair`", "`Pair`:5:8").
    pub hand_label: String,
    /// Street index (0-3).
    pub street: u8,
    /// Street name ("preflop", "flop", "turn", "river").
    pub street_label: &'static str,
    /// SPR bucket (0-31).
    pub spr_bucket: u32,
    /// Non-zero action codes extracted from the key.
    pub action_codes: Vec<u8>,
    /// Human-readable action labels (e.g. "Check", "Bet 67%").
    pub action_labels: Vec<String>,
    /// Strategy probabilities from the blueprint, if found.
    pub strategy: Option<Vec<f32>>,
}

impl KeyDescription {
    /// Produce the human-readable compose string that recreates this key.
    #[must_use]
    pub fn compose_string(&self) -> String {
        let actions_str = self
            .action_codes
            .iter()
            .map(|&c| ACTION_CODE_NAMES[c as usize])
            .collect::<Vec<_>>()
            .join(",");

        if actions_str.is_empty() {
            format!(
                "hand={} street={} spr={}",
                self.hand_label, self.street_label, self.spr_bucket
            )
        } else {
            format!(
                "hand={} street={} spr={} actions={}",
                self.hand_label, self.street_label, self.spr_bucket, actions_str
            )
        }
    }
}

#[cfg(test)]
/// Decode a 4-bit action code to its string label.
#[must_use]
pub(crate) fn decode_action_code(code: u8) -> &'static str {
    ACTION_CODE_NAMES.get(code as usize).unwrap_or(&"unknown")
}

/// Format an action with resolved bet sizes as a human-readable string.
///
/// Uses `bet_sizes` to convert `Bet(idx)` / `Raise(idx)` into percentage labels.
#[must_use]
pub fn format_action_label(action: Action, bet_sizes: &[f32]) -> String {
    match action {
        Action::Fold => "Fold".into(),
        Action::Check => "Check".into(),
        Action::Call => "Call".into(),
        Action::Bet(idx) if idx == ALL_IN => "Bet All-In".into(),
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Action::Bet(idx) => {
            let pct = (bet_sizes.get(idx as usize).unwrap_or(&0.0) * 100.0) as u32;
            format!("Bet {pct}%")
        }
        Action::Raise(idx) if idx == ALL_IN => "Raise All-In".into(),
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Action::Raise(idx) => {
            let pct = (bet_sizes.get(idx as usize).unwrap_or(&0.0) * 100.0) as u32;
            format!("Raise {pct}%")
        }
    }
}

/// Format a 4-bit action code with resolved bet sizes as a human-readable string.
#[must_use]
pub fn format_action_code_label(code: u8, bet_sizes: &[f32]) -> String {
    match code {
        0 => String::new(),
        1 => "Fold".into(),
        2 => "Check".into(),
        3 => "Call".into(),
        c @ 4..=8 => {
            let idx = (c - 4) as usize;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let pct = (bet_sizes.get(idx).unwrap_or(&0.0) * 100.0) as u32;
            format!("Bet {pct}%")
        }
        c @ 9..=13 => {
            let idx = (c - 9) as usize;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let pct = (bet_sizes.get(idx).unwrap_or(&0.0) * 100.0) as u32;
            format!("Raise {pct}%")
        }
        14 => "Bet All-In".into(),
        15 => "Raise All-In".into(),
        _ => "Unknown".into(),
    }
}

/// Map a canonical hand index (0..168) back to its string representation.
///
/// Returns labels like `"AA"`, `"AKs"`, `"72o"`.
#[must_use]
pub fn reverse_canonical_index(index: u16) -> &'static str {
    const RANKS: [char; 13] = [
        'A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2',
    ];

    static HANDS: std::sync::LazyLock<[String; 169]> = std::sync::LazyLock::new(|| {
        let mut table: [String; 169] = std::array::from_fn(|_| String::new());
        let mut idx = 0usize;

        // Pairs: AA=0, KK=1, ..., 22=12
        for r in RANKS {
            table[idx] = format!("{r}{r}");
            idx += 1;
        }

        // Suited: triangle_index(high, low) = high*(high-1)/2 + low
        for (high, &h) in RANKS.iter().enumerate().skip(1) {
            for &l in &RANKS[..high] {
                table[idx] = format!("{l}{h}s");
                idx += 1;
            }
        }

        // Offsuit: same triangle order
        for (high, &h) in RANKS.iter().enumerate().skip(1) {
            for &l in &RANKS[..high] {
                table[idx] = format!("{l}{h}o");
                idx += 1;
            }
        }

        table
    });

    &HANDS[index as usize]
}

/// Describe a key from either hex or compose format.
///
/// Accepts:
/// - Hex: `"0x002A400000300000"` or raw hex digits
/// - Compose: `"hand=Pair street=flop spr=12 spr=7 actions=check,bet1"`
///
/// # Errors
///
/// Returns an error if the input cannot be parsed.
pub fn describe_key(
    input: &str,
    bet_sizes: &[f32],
    abstraction_mode: AbstractionModeConfig,
    blueprint: Option<&BlueprintStrategy>,
) -> Result<KeyDescription, String> {
    let trimmed = input.trim();

    if is_hex_input(trimmed) {
        describe_from_hex(trimmed, bet_sizes, abstraction_mode, blueprint)
    } else {
        describe_from_compose(trimmed, bet_sizes, abstraction_mode, blueprint)
    }
}

fn is_hex_input(s: &str) -> bool {
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    !s.is_empty() && s.chars().all(|c| c.is_ascii_hexdigit() || c == '_')
}

fn describe_from_hex(
    input: &str,
    bet_sizes: &[f32],
    abstraction_mode: AbstractionModeConfig,
    blueprint: Option<&BlueprintStrategy>,
) -> Result<KeyDescription, String> {
    let hex_str = input
        .strip_prefix("0x")
        .or_else(|| input.strip_prefix("0X"))
        .unwrap_or(input)
        .replace('_', "");

    let raw = u64::from_str_radix(&hex_str, 16).map_err(|e| format!("invalid hex: {e}"))?;

    let key = InfoKey::from_raw(raw);
    let hand_bits = key.hand_bits();
    let street = key.street();
    let spr = key.spr_bucket();

    let action_codes = extract_action_codes(key.actions_bits());
    let action_labels = action_codes
        .iter()
        .map(|&c| format_action_code_label(c, bet_sizes))
        .collect();

    let hand_label = hand_label_from_bits(hand_bits, street, abstraction_mode);
    let strategy = blueprint.and_then(|bp| bp.lookup(raw).map(<[f32]>::to_vec));

    Ok(KeyDescription {
        raw,
        hand_bits,
        hand_label,
        street,
        street_label: STREET_NAMES[street as usize],
        spr_bucket: spr,
        action_codes,
        action_labels,
        strategy,
    })
}

fn describe_from_compose(
    input: &str,
    bet_sizes: &[f32],
    abstraction_mode: AbstractionModeConfig,
    blueprint: Option<&BlueprintStrategy>,
) -> Result<KeyDescription, String> {
    let mut hand_str: Option<&str> = None;
    let mut street_str: Option<&str> = None;
    let mut spr_val: Option<u32> = None;
    let mut actions_str: Option<&str> = None;

    for part in input.split_whitespace() {
        if let Some(val) = part.strip_prefix("hand=") {
            hand_str = Some(val);
        } else if let Some(val) = part.strip_prefix("street=") {
            street_str = Some(val);
        } else if let Some(val) = part.strip_prefix("spr=") {
            spr_val = Some(val.parse().map_err(|_| format!("invalid spr: {val}"))?);
        } else if let Some(val) = part.strip_prefix("actions=") {
            actions_str = Some(val);
        } else if part.starts_with("depth=") {
            // Silently ignore legacy depth= field for backwards compatibility
        } else {
            return Err(format!("unknown field: {part}"));
        }
    }

    let hand_raw = hand_str.ok_or("missing hand= field")?;
    let street = parse_street(street_str.ok_or("missing street= field")?)?;
    let spr = spr_val.ok_or("missing spr= field")?;

    let hand_bits = parse_hand_bits(hand_raw, abstraction_mode)?;

    let action_codes = match actions_str {
        Some(s) if !s.is_empty() => parse_action_codes(s)?,
        _ => Vec::new(),
    };

    let action_labels = action_codes
        .iter()
        .map(|&c| format_action_code_label(c, bet_sizes))
        .collect();

    let raw = InfoKey::new(hand_bits, street, spr, &action_codes).as_u64();
    let strategy = blueprint.and_then(|bp| bp.lookup(raw).map(<[f32]>::to_vec));

    let hand_label = hand_raw.to_string();

    Ok(KeyDescription {
        raw,
        hand_bits,
        hand_label,
        street,
        street_label: STREET_NAMES[street as usize],
        spr_bucket: spr,
        action_codes,
        action_labels,
        strategy,
    })
}

/// Extract non-zero action codes from the 24-bit action field.
fn extract_action_codes(actions_bits: u32) -> Vec<u8> {
    let mut codes = Vec::new();
    for i in 0..6 {
        let code = ((actions_bits >> (20 - i * 4)) & 0xF) as u8;
        if code == 0 {
            break;
        }
        codes.push(code);
    }
    codes
}

fn parse_street(s: &str) -> Result<u8, String> {
    match s.to_ascii_lowercase().as_str() {
        "preflop" | "0" => Ok(0),
        "flop" | "1" => Ok(1),
        "turn" | "2" => Ok(2),
        "river" | "3" => Ok(3),
        _ => Err(format!("invalid street: {s}")),
    }
}

fn parse_hand_bits(hand: &str, mode: AbstractionModeConfig) -> Result<u32, String> {
    // Try raw numeric first (decimal or hex)
    if let Ok(v) = hand.parse::<u32>() {
        return Ok(v);
    }
    if let Some(hex) = hand.strip_prefix("0x") {
        return u32::from_str_radix(hex, 16).map_err(|e| format!("invalid hex hand: {e}"));
    }

    match mode {
        AbstractionModeConfig::Ehs2 => canonical_hand_index_from_str(hand)
            .map(u32::from)
            .ok_or_else(|| format!("invalid canonical hand: {hand}")),
        AbstractionModeConfig::HandClassV2 => parse_hand_class_v2(hand),
    }
}

fn parse_hand_class_v2(hand: &str) -> Result<u32, String> {
    let parts: Vec<&str> = hand.split(':').collect();
    let class_name = parts.first().ok_or("empty hand class")?;

    let class = HandClass::from_name(class_name)
        .ok_or_else(|| format!("unknown hand class: {class_name}"))?;
    let class_id = class as u8;

    let strength: u8 = parts
        .get(1)
        .unwrap_or(&"1")
        .parse()
        .map_err(|_| "invalid strength")?;
    let equity: u8 = parts
        .get(2)
        .unwrap_or(&"0")
        .parse()
        .map_err(|_| "invalid equity")?;

    // Draw flags from additional parts like "+FlushDraw"
    let draw_flags = 0u8; // Simplified — could extend later

    Ok(encode_hand_v2(class_id, strength, equity, draw_flags, 4, 4))
}

fn parse_action_codes(s: &str) -> Result<Vec<u8>, String> {
    s.split(',')
        .map(|a| {
            let lower = a.trim().to_ascii_lowercase();
            match lower.as_str() {
                "fold" => Ok(1),
                "check" => Ok(2),
                "call" => Ok(3),
                "bet0" => Ok(4),
                "bet1" => Ok(5),
                "bet2" => Ok(6),
                "bet3" => Ok(7),
                "bet4" => Ok(8),
                "raise0" => Ok(9),
                "raise1" => Ok(10),
                "raise2" => Ok(11),
                "raise3" => Ok(12),
                "raise4" => Ok(13),
                "bet-allin" | "betallin" => Ok(14),
                "raise-allin" | "raiseallin" => Ok(15),
                _ => Err(format!("unknown action: {a}")),
            }
        })
        .collect()
}

/// Generate a human-readable hand label from the raw 28-bit hand field.
pub fn hand_label_from_bits(hand_bits: u32, street: u8, mode: AbstractionModeConfig) -> String {
    match mode {
        AbstractionModeConfig::Ehs2 => {
            if street == 0 && hand_bits < 169 {
                #[allow(clippy::cast_possible_truncation)]
                reverse_canonical_index(hand_bits as u16).to_string()
            } else {
                format!("bucket:{hand_bits}")
            }
        }
        AbstractionModeConfig::HandClassV2 => {
            let class_id = (hand_bits >> 23) & 0x1F;
            let strength = ((hand_bits >> 19) & 0xF) as u8;
            let equity = ((hand_bits >> 15) & 0xF) as u8;
            let draw_mask = (1u32 << HandClass::NUM_DRAWS) - 1;
            #[allow(clippy::cast_possible_truncation)]
            let draw_flags = ((hand_bits >> 8) & draw_mask) as u8;

            let class_name = HandClass::ALL.get(class_id as usize).map_or_else(
                || format!("class{class_id}"),
                std::string::ToString::to_string,
            );

            let mut label = format!("{class_name}:{strength}:{equity}");
            if draw_flags != 0 {
                use std::fmt::Write;
                let _ = write!(label, ":d{draw_flags:#04x}");
            }
            label
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn round_trip_key_components() {
        let key = InfoKey::new(42, 2, 15, &[1, 3, 5]);
        assert_eq!(key.spr_bucket(), 15);
        assert_eq!(key.street(), 2);
        assert_eq!(key.hand_bits(), 42);
    }

    #[timed_test]
    fn street_extractor() {
        for street in 0..=3u8 {
            let key = InfoKey::new(0, street, 0, &[]);
            assert_eq!(key.street(), street);
        }
    }

    #[timed_test]
    fn hand_bits_extractor() {
        let key = InfoKey::new(0xABC_DEF, 1, 5, &[2]);
        assert_eq!(key.hand_bits(), 0xABC_DEF);
    }

    #[timed_test]
    fn actions_bits_extractor() {
        // Actions packed: [4, 7] → action 0 at bits 23-20 = 4, action 1 at bits 19-16 = 7
        let key = InfoKey::new(0, 0, 0, &[4, 7]);
        let expected = (4u32 << 20) | (7u32 << 16);
        assert_eq!(key.actions_bits(), expected);
    }

    #[timed_test]
    fn actions_bits_empty() {
        let key = InfoKey::new(100, 3, 10, &[]);
        assert_eq!(key.actions_bits(), 0);
    }

    #[timed_test]
    fn with_spr_replaces_correctly() {
        let key = InfoKey::new(42, 1, 10, &[2, 3]);
        let modified = key.with_spr(20);
        assert_eq!(modified.spr_bucket(), 20);
        // Hand and street bits should be unchanged
        assert_eq!(key.as_u64() >> HAND_SHIFT, modified.as_u64() >> HAND_SHIFT,);
    }

    #[timed_test]
    fn all_169_canonical_hands_unique() {
        let mut seen = std::collections::HashSet::new();
        let values = [
            Value::Ace,
            Value::King,
            Value::Queen,
            Value::Jack,
            Value::Ten,
            Value::Nine,
            Value::Eight,
            Value::Seven,
            Value::Six,
            Value::Five,
            Value::Four,
            Value::Three,
            Value::Two,
        ];

        let mut count = 0;
        for (i, &v1) in values.iter().enumerate() {
            for &v2 in &values[i..] {
                if v1 == v2 {
                    // Pair
                    let idx = canonical_hand_index([
                        Card::new(v1, Suit::Spade),
                        Card::new(v2, Suit::Heart),
                    ]);
                    assert!(seen.insert(idx), "Duplicate index {idx} for pair {v1:?}");
                    count += 1;
                } else {
                    // Suited
                    let idx_s = canonical_hand_index([
                        Card::new(v1, Suit::Spade),
                        Card::new(v2, Suit::Spade),
                    ]);
                    assert!(
                        seen.insert(idx_s),
                        "Duplicate index {idx_s} for suited {v1:?}{v2:?}"
                    );
                    count += 1;

                    // Offsuit
                    let idx_o = canonical_hand_index([
                        Card::new(v1, Suit::Spade),
                        Card::new(v2, Suit::Heart),
                    ]);
                    assert!(
                        seen.insert(idx_o),
                        "Duplicate index {idx_o} for offsuit {v1:?}{v2:?}"
                    );
                    count += 1;
                }
            }
        }

        assert_eq!(count, 169);
        assert_eq!(seen.len(), 169);
        // All indices should be in 0..169
        for &idx in &seen {
            assert!(idx < 169, "Index {idx} out of range");
        }
    }

    #[timed_test]
    fn canonical_hand_index_symmetry() {
        // AKs should give the same index regardless of card order
        let idx1 = canonical_hand_index([
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ]);
        let idx2 = canonical_hand_index([
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Ace, Suit::Spade),
        ]);
        assert_eq!(idx1, idx2);
    }

    #[timed_test]
    fn canonical_hand_index_from_str_examples() {
        assert_eq!(canonical_hand_index_from_str("AA"), Some(0));
        assert_eq!(canonical_hand_index_from_str("KK"), Some(1));
        assert_eq!(canonical_hand_index_from_str("22"), Some(12));
        assert_eq!(canonical_hand_index_from_str("AKs"), Some(13));
        assert_eq!(canonical_hand_index_from_str("AKo"), Some(91));
    }

    #[timed_test]
    fn canonical_hand_index_str_matches_card_index() {
        // AKs from string should match AKs from cards
        let from_str = canonical_hand_index_from_str("AKs").unwrap();
        let from_cards = canonical_hand_index([
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ]);
        assert_eq!(from_str, from_cards);

        // QQ from string should match QQ from cards
        let from_str = canonical_hand_index_from_str("QQ").unwrap();
        let from_cards = canonical_hand_index([
            Card::new(Value::Queen, Suit::Spade),
            Card::new(Value::Queen, Suit::Heart),
        ]);
        assert_eq!(from_str, from_cards);
    }

    #[timed_test]
    fn encode_action_covers_all_variants() {
        assert_eq!(encode_action(Action::Fold), 1);
        assert_eq!(encode_action(Action::Check), 2);
        assert_eq!(encode_action(Action::Call), 3);
        assert_eq!(encode_action(Action::Bet(0)), 4);
        assert_eq!(encode_action(Action::Bet(1)), 5);
        assert_eq!(encode_action(Action::Bet(2)), 6);
        assert_eq!(encode_action(Action::Bet(3)), 7);
        assert_eq!(encode_action(Action::Bet(4)), 8);
        assert_eq!(encode_action(Action::Raise(0)), 9);
        assert_eq!(encode_action(Action::Raise(1)), 10);
        assert_eq!(encode_action(Action::Raise(2)), 11);
        assert_eq!(encode_action(Action::Raise(3)), 12);
        assert_eq!(encode_action(Action::Raise(4)), 13);
        assert_eq!(encode_action(Action::Bet(ALL_IN)), 14);
        assert_eq!(encode_action(Action::Raise(ALL_IN)), 15);
    }

    #[timed_test]
    fn different_streets_produce_different_keys() {
        let k1 = InfoKey::new(0, 0, 0, &[]);
        let k2 = InfoKey::new(0, 1, 0, &[]);
        assert_ne!(k1.as_u64(), k2.as_u64());
    }

    #[timed_test]
    fn different_actions_produce_different_keys() {
        let k1 = InfoKey::new(0, 0, 0, &[1]);
        let k2 = InfoKey::new(0, 0, 0, &[2]);
        assert_ne!(k1.as_u64(), k2.as_u64());
    }

    #[timed_test]
    fn action_packing_order_matters() {
        let k1 = InfoKey::new(0, 0, 0, &[1, 2]);
        let k2 = InfoKey::new(0, 0, 0, &[2, 1]);
        assert_ne!(k1.as_u64(), k2.as_u64());
    }

    #[timed_test]
    fn actions_do_not_overlap_with_spr_bucket() {
        // Verify that setting all action bits to max doesn't corrupt spr_bucket.
        let key = InfoKey::new(0, 0, 31, &[15, 15, 15, 15, 15, 15]);
        assert_eq!(key.spr_bucket(), 31, "SPR bucket corrupted by action bits");

        // Verify that max spr_bucket doesn't corrupt first action
        let k1 = InfoKey::new(0, 0, 31, &[1]);
        let k2 = InfoKey::new(0, 0, 31, &[2]);
        assert_ne!(
            k1.as_u64(),
            k2.as_u64(),
            "Actions indistinguishable with max spr_bucket"
        );
    }

    #[timed_test]
    fn spr_bucket_edge_cases() {
        // Zero pot → infinite SPR → capped at 31
        assert_eq!(spr_bucket(0, 100), 31);
        // Small pot, large stack → high SPR capped at 31
        assert_eq!(spr_bucket(1, 200), 31);
        // Equal pot and stack → SPR = 2 (half-SPR units)
        assert_eq!(spr_bucket(10, 10), 2);
        // Pot larger than stack → low SPR
        assert_eq!(spr_bucket(20, 5), 0);
        // Typical flop after limp: pot=4, eff_stack=18 → 36/4 = 9
        assert_eq!(spr_bucket(4, 18), 9);
        // Typical raised pot: pot=12, eff_stack=14 → 28/12 = 2
        assert_eq!(spr_bucket(12, 14), 2);
    }

    #[timed_test]
    fn invalid_str_returns_none() {
        assert_eq!(canonical_hand_index_from_str(""), None);
        assert_eq!(canonical_hand_index_from_str("X"), None);
        assert_eq!(canonical_hand_index_from_str("AAAA"), None);
    }

    // === encode_hand_v2 tests ===

    #[timed_test]
    fn encode_hand_v2_class_id_round_trips() {
        for id in 0..=21u8 {
            let bits = encode_hand_v2(id, 1, 0, 0, 4, 4);
            let extracted = (bits >> 23) & 0x1F;
            assert_eq!(extracted, u32::from(id), "class_id {id}");
        }
    }

    #[timed_test]
    fn encode_hand_v2_strength_round_trips() {
        for strength in 1..=14u8 {
            let bits = encode_hand_v2(0, strength, 0, 0, 4, 4);
            let extracted = ((bits >> 19) & 0xF) as u8;
            // strength 1-14 stored as 0-13
            assert_eq!(extracted, strength - 1, "strength {strength}");
        }
    }

    #[timed_test]
    fn encode_hand_v2_equity_round_trips() {
        for eq in 0..=15u8 {
            let bits = encode_hand_v2(0, 1, eq, 0, 4, 4);
            let extracted = ((bits >> 15) & 0xF) as u8;
            assert_eq!(extracted, eq, "equity_bin {eq}");
        }
    }

    #[timed_test]
    fn encode_hand_v2_draw_flags_round_trips() {
        for flags in [0u8, 0x3F, 0b10_1010, 0b01_0101] {
            let bits = encode_hand_v2(0, 1, 0, flags, 0, 0);
            let extracted = ((bits >> 8) & 0x3F) as u8;
            assert_eq!(extracted, flags, "draw_flags {flags:#b}");
        }
    }

    #[timed_test]
    fn encode_hand_v2_quantization_strength_bits_2() {
        // strength=14 → val=13 → quantized = 13 >> 2 = 3 (2-bit: 0-3)
        let bits = encode_hand_v2(0, 14, 0, 0, 2, 0);
        let extracted = ((bits >> 19) & 0xF) as u8;
        assert_eq!(extracted, 3);

        // strength=1 → val=0 → quantized = 0 >> 2 = 0
        let bits = encode_hand_v2(0, 1, 0, 0, 2, 0);
        let extracted = ((bits >> 19) & 0xF) as u8;
        assert_eq!(extracted, 0);
    }

    #[timed_test]
    fn encode_hand_v2_zero_bits_means_omitted() {
        let bits = encode_hand_v2(5, 14, 15, 0x3F, 0, 0);
        // strength and equity should be zero
        let s = ((bits >> 19) & 0xF) as u8;
        let e = ((bits >> 15) & 0xF) as u8;
        assert_eq!(s, 0, "strength should be 0 when strength_bits=0");
        assert_eq!(e, 0, "equity should be 0 when equity_bits=0");
        // class_id and draw_flags should still be present
        assert_eq!((bits >> 23) & 0x1F, 5);
        assert_eq!((bits >> 8) & 0x3F, 0x3F);
    }

    #[timed_test]
    fn encode_hand_v2_different_strengths_different_bits() {
        let b1 = encode_hand_v2(5, 1, 8, 0, 4, 4);
        let b2 = encode_hand_v2(5, 14, 8, 0, 4, 4);
        assert_ne!(
            b1, b2,
            "Different strengths should produce different encodings"
        );
    }

    #[timed_test]
    fn encode_hand_v2_different_equity_different_bits() {
        let b1 = encode_hand_v2(5, 7, 0, 0, 4, 4);
        let b2 = encode_hand_v2(5, 7, 15, 0, 4, 4);
        assert_ne!(
            b1, b2,
            "Different equity bins should produce different encodings"
        );
    }

    #[timed_test]
    fn encode_hand_v2_fits_in_28_bits() {
        // Max values for all fields
        let bits = encode_hand_v2(31, 14, 15, 0x3F, 4, 4);
        assert!(
            bits < (1 << 28),
            "Encoded value {bits:#010x} exceeds 28 bits"
        );
    }

    // === Key translation API tests ===

    #[timed_test]
    fn decode_action_code_all_16() {
        assert_eq!(decode_action_code(0), "empty");
        assert_eq!(decode_action_code(1), "fold");
        assert_eq!(decode_action_code(2), "check");
        assert_eq!(decode_action_code(3), "call");
        assert_eq!(decode_action_code(4), "bet0");
        assert_eq!(decode_action_code(8), "bet4");
        assert_eq!(decode_action_code(9), "raise0");
        assert_eq!(decode_action_code(13), "raise4");
        assert_eq!(decode_action_code(14), "bet-allin");
        assert_eq!(decode_action_code(15), "raise-allin");
    }

    #[timed_test]
    fn format_action_label_all_variants() {
        let sizes = vec![0.33, 0.67, 1.0, 2.0, 3.0];
        assert_eq!(format_action_label(Action::Fold, &sizes), "Fold");
        assert_eq!(format_action_label(Action::Check, &sizes), "Check");
        assert_eq!(format_action_label(Action::Call, &sizes), "Call");
        assert_eq!(format_action_label(Action::Bet(0), &sizes), "Bet 33%");
        assert_eq!(format_action_label(Action::Bet(1), &sizes), "Bet 67%");
        assert_eq!(format_action_label(Action::Bet(2), &sizes), "Bet 100%");
        assert_eq!(
            format_action_label(Action::Bet(ALL_IN), &sizes),
            "Bet All-In"
        );
        assert_eq!(format_action_label(Action::Raise(0), &sizes), "Raise 33%");
        assert_eq!(
            format_action_label(Action::Raise(ALL_IN), &sizes),
            "Raise All-In"
        );
    }

    #[timed_test]
    fn reverse_canonical_index_roundtrip() {
        for idx in 0..169u16 {
            let name = reverse_canonical_index(idx);
            let back = canonical_hand_index_from_str(name);
            assert_eq!(
                back,
                Some(idx),
                "roundtrip failed for index {idx} -> {name}"
            );
        }
    }

    #[timed_test]
    fn reverse_canonical_index_known_values() {
        assert_eq!(reverse_canonical_index(0), "AA");
        assert_eq!(reverse_canonical_index(1), "KK");
        assert_eq!(reverse_canonical_index(12), "22");
        assert_eq!(reverse_canonical_index(13), "AKs");
        assert_eq!(reverse_canonical_index(91), "AKo");
    }

    #[timed_test]
    fn describe_key_hex_roundtrip() {
        let sizes = vec![0.33, 0.67, 1.0, 2.0, 3.0];
        let key = InfoKey::new(42, 1, 10, &[2, 5]).as_u64();
        let hex = format!("0x{key:016X}");

        let desc = describe_key(&hex, &sizes, AbstractionModeConfig::Ehs2, None).unwrap();
        assert_eq!(desc.raw, key);
        assert_eq!(desc.street, 1);
        assert_eq!(desc.spr_bucket, 10);
        assert_eq!(desc.action_codes, vec![2, 5]);
    }

    #[timed_test]
    fn describe_key_compose_format() {
        let sizes = vec![0.33, 0.67, 1.0, 2.0, 3.0];
        let input = "hand=AKs street=preflop spr=31";

        let desc = describe_key(input, &sizes, AbstractionModeConfig::Ehs2, None).unwrap();
        assert_eq!(desc.street, 0);
        assert_eq!(desc.spr_bucket, 31);
        assert_eq!(desc.hand_bits, 13); // AKs = index 13
    }

    #[timed_test]
    fn describe_key_compose_with_actions() {
        let sizes = vec![0.33, 0.67, 1.0, 2.0, 3.0];
        let input = "hand=QQ street=flop spr=5 actions=check,bet1";

        let desc = describe_key(input, &sizes, AbstractionModeConfig::Ehs2, None).unwrap();
        assert_eq!(desc.action_codes, vec![2, 5]); // check=2, bet1=5
        assert_eq!(desc.action_labels, vec!["Check", "Bet 67%"]);
    }

    #[timed_test]
    fn describe_key_hex_then_compose_roundtrip() {
        let sizes = vec![0.33, 0.67, 1.0, 2.0, 3.0];
        let input = "hand=AKs street=preflop spr=20 actions=call";

        let desc1 = describe_key(input, &sizes, AbstractionModeConfig::Ehs2, None).unwrap();
        let hex = format!("0x{:016X}", desc1.raw);
        let desc2 = describe_key(&hex, &sizes, AbstractionModeConfig::Ehs2, None).unwrap();

        assert_eq!(desc1.raw, desc2.raw);
        assert_eq!(desc1.street, desc2.street);
        assert_eq!(desc1.spr_bucket, desc2.spr_bucket);
        assert_eq!(desc1.action_codes, desc2.action_codes);
    }

    #[timed_test]
    fn compose_string_output() {
        let sizes = vec![0.33, 0.67, 1.0, 2.0, 3.0];
        let input = "hand=AKs street=flop spr=10 actions=check,bet1";

        let desc = describe_key(input, &sizes, AbstractionModeConfig::Ehs2, None).unwrap();
        let compose = desc.compose_string();
        assert!(compose.contains("hand=AKs"), "compose: {compose}");
        assert!(compose.contains("street=flop"), "compose: {compose}");
        assert!(compose.contains("spr=10"), "compose: {compose}");
        assert!(compose.contains("actions=check,bet1"), "compose: {compose}");
    }

    #[timed_test]
    fn extract_action_codes_stops_at_zero() {
        // Actions [3, 5, 0, 0, 0, 0] → should return [3, 5]
        let bits = (3u32 << 20) | (5u32 << 16);
        let codes = extract_action_codes(bits);
        assert_eq!(codes, vec![3, 5]);
    }

    #[timed_test]
    fn describe_key_hand_class_v2_mode() {
        let sizes = vec![0.33, 0.67, 1.0];
        let input = "hand=Pair:1:0 street=flop spr=5";

        let desc = describe_key(input, &sizes, AbstractionModeConfig::HandClassV2, None).unwrap();
        // Pair = discriminant 9.
        // parse_hand_class_v2 passes raw strength=1, equity=0 to encode_hand_v2(bits=4,4).
        // encode_hand_v2 quantizes strength: (1-1).min(13) = 0, so encoded strength = 0.
        // class_id(5 bits) at bits 23..28 = 9 << 23
        let expected_bits = 9u32 << 23;
        assert_eq!(desc.hand_bits, expected_bits);
    }
}
