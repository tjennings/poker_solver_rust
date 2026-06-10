//! Shared FNV-1a 64-bit hash utility.
//!
//! Used for stable fingerprints across the universal blueprint format:
//! row key fingerprints, game fingerprints, action abstraction fingerprints,
//! bucket semantic fingerprints, and action schema fingerprints.

/// FNV-1a 64-bit hasher for stable fingerprints.
///
/// All fingerprint sites in the universal format share the same
/// FNV-1a byte-mixing logic.  This struct consolidates that into
/// a single reusable primitive.
pub struct Fnv1aHasher {
    state: u64,
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

impl Fnv1aHasher {
    /// Create a new hasher with the standard FNV-1a offset basis.
    #[must_use]
    pub const fn new() -> Self {
        Self { state: FNV_OFFSET }
    }

    /// Mix a byte slice into the hash state.
    pub fn mix_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= u64::from(byte);
            self.state = self.state.wrapping_mul(FNV_PRIME);
        }
    }

    /// Mix a single `u8` value.
    pub fn mix_u8(&mut self, v: u8) {
        self.state ^= u64::from(v);
        self.state = self.state.wrapping_mul(FNV_PRIME);
    }

    /// Mix a `u16` value (little-endian bytes).
    pub fn mix_u16(&mut self, v: u16) {
        self.mix_bytes(&v.to_le_bytes());
    }

    /// Mix a `u32` value (little-endian bytes).
    pub fn mix_u32(&mut self, v: u32) {
        self.mix_bytes(&v.to_le_bytes());
    }

    /// Mix a `u64` value (little-endian bytes).
    pub fn mix_u64(&mut self, v: u64) {
        self.mix_bytes(&v.to_le_bytes());
    }

    /// Finalize and return the hash value.
    #[must_use]
    pub const fn finish(self) -> u64 {
        self.state
    }
}

impl Default for Fnv1aHasher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_input() {
        let mut h1 = Fnv1aHasher::new();
        h1.mix_bytes(b"hello");
        let mut h2 = Fnv1aHasher::new();
        h2.mix_bytes(b"hello");
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn different_inputs_differ() {
        let mut h1 = Fnv1aHasher::new();
        h1.mix_bytes(b"hello");
        let mut h2 = Fnv1aHasher::new();
        h2.mix_bytes(b"world");
        assert_ne!(h1.finish(), h2.finish());
    }

    #[test]
    fn empty_input_returns_offset_basis() {
        let h = Fnv1aHasher::new();
        assert_eq!(h.finish(), FNV_OFFSET);
    }

    #[test]
    fn mix_u8_matches_mix_bytes() {
        let mut h1 = Fnv1aHasher::new();
        h1.mix_u8(42);
        let mut h2 = Fnv1aHasher::new();
        h2.mix_bytes(&[42]);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn mix_u16_matches_le_bytes() {
        let mut h1 = Fnv1aHasher::new();
        h1.mix_u16(0x1234);
        let mut h2 = Fnv1aHasher::new();
        h2.mix_bytes(&0x1234u16.to_le_bytes());
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn mix_u32_matches_le_bytes() {
        let mut h1 = Fnv1aHasher::new();
        h1.mix_u32(0xDEAD_BEEF);
        let mut h2 = Fnv1aHasher::new();
        h2.mix_bytes(&0xDEAD_BEEFu32.to_le_bytes());
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn mix_u64_matches_le_bytes() {
        let mut h1 = Fnv1aHasher::new();
        h1.mix_u64(0x0123_4567_89AB_CDEF);
        let mut h2 = Fnv1aHasher::new();
        h2.mix_bytes(&0x0123_4567_89AB_CDEFu64.to_le_bytes());
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn order_matters() {
        let mut h1 = Fnv1aHasher::new();
        h1.mix_u8(1);
        h1.mix_u8(2);
        let mut h2 = Fnv1aHasher::new();
        h2.mix_u8(2);
        h2.mix_u8(1);
        assert_ne!(h1.finish(), h2.finish());
    }
}
