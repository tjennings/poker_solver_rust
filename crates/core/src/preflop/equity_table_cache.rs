//! Disk cache for precomputed flop equity tables.
//!
//! Each canonical flop requires an expensive O(169² × combos² × 990 runouts) equity
//! computation. Since equity depends only on flop cards (not SPR, bet sizing, or solver
//! parameters), these tables can be computed once and reused across all training runs.
//!
//! File format: bincode-serialized `CacheData` with a 4-byte magic header and version.
//! All 1755 canonical flops are stored in deterministic order (`canonical_flops()`).
//! Flops are encoded as `[u8; 3]` using `value * 4 + suit` per card.

use std::io;
use std::path::Path;

use super::postflop_exhaustive::compute_equity_table;
use super::postflop_hands::{build_combo_map, canonical_flops, NUM_CANONICAL_HANDS};
use rs_poker::core::Card;
use serde::{Deserialize, Serialize};

const MAGIC: [u8; 4] = *b"EQTC";
const VERSION: u32 = 1;

/// On-disk representation (bincode-serialized).
#[derive(Serialize, Deserialize)]
struct CacheData {
    magic: [u8; 4],
    version: u32,
    num_flops: u32,
    /// Flops encoded as [value*4+suit; 3] per flop, flattened.
    flop_bytes: Vec<u8>,
    /// Equity tables flattened: num_flops × 169 × 169 f64s.
    tables_flat: Vec<f64>,
}

/// In-memory equity table cache.
pub struct EquityTableCache {
    /// Canonical flops in deterministic order.
    flops: Vec<[Card; 3]>,
    /// Flat array: flops × 169 × 169.
    tables_flat: Vec<f64>,
}

fn encode_card(c: Card) -> u8 {
    c.value as u8 * 4 + c.suit as u8
}

fn decode_card(b: u8) -> Card {
    use rs_poker::core::{Suit, Value};
    // SAFETY: Value is repr(u8) with contiguous variants 0..=12 (Two..Ace).
    // b/4 is in range 0..=12 because encode_card produces value*4+suit where
    // value <= 12 and suit <= 3, so max encoded = 12*4+3 = 51, and 51/4 = 12.
    let value = unsafe { std::mem::transmute::<u8, Value>(b / 4) };
    // SAFETY: Suit is repr(u8) with contiguous variants 0..=3 (Spade..Club).
    // b%4 is always in range 0..=3.
    let suit = unsafe { std::mem::transmute::<u8, Suit>(b % 4) };
    Card { value, suit }
}

impl EquityTableCache {
    /// Number of entries per equity table (169 × 169).
    const TABLE_SIZE: usize = NUM_CANONICAL_HANDS * NUM_CANONICAL_HANDS;

    /// Build the cache by computing equity tables for all canonical flops.
    ///
    /// Calls `on_progress(completed, total)` after each flop finishes.
    pub fn build(on_progress: impl Fn(usize, usize) + Sync) -> Self {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let flops = canonical_flops();
        let total = flops.len();
        let completed = AtomicUsize::new(0);

        let tables: Vec<Vec<f64>> = flops
            .par_iter()
            .map(|flop| {
                let combo_map = build_combo_map(flop);
                let table = compute_equity_table(&combo_map, *flop);
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                on_progress(done, total);
                table
            })
            .collect();

        let mut tables_flat = Vec::with_capacity(total * Self::TABLE_SIZE);
        for table in tables {
            tables_flat.extend_from_slice(&table);
        }

        Self { flops, tables_flat }
    }

    /// Save the cache to a binary file.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut flop_bytes = Vec::with_capacity(self.flops.len() * 3);
        for flop in &self.flops {
            flop_bytes.push(encode_card(flop[0]));
            flop_bytes.push(encode_card(flop[1]));
            flop_bytes.push(encode_card(flop[2]));
        }
        let data = CacheData {
            magic: MAGIC,
            version: VERSION,
            num_flops: self.flops.len() as u32,
            flop_bytes,
            tables_flat: self.tables_flat.clone(),
        };
        let bytes =
            bincode::serialize(&data).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        std::fs::write(path, bytes)
    }

    /// Load a cache from a binary file.
    ///
    /// Returns `None` if the file doesn't exist, has an invalid magic/version,
    /// or fails to deserialize.
    pub fn load(path: &Path) -> Option<Self> {
        let bytes = std::fs::read(path).ok()?;
        let data: CacheData = bincode::deserialize(&bytes).ok()?;
        if data.magic != MAGIC || data.version != VERSION {
            return None;
        }
        let num_flops = data.num_flops as usize;
        if data.flop_bytes.len() != num_flops * 3 {
            return None;
        }
        if data.tables_flat.len() != num_flops * Self::TABLE_SIZE {
            return None;
        }
        let mut flops = Vec::with_capacity(num_flops);
        for i in 0..num_flops {
            let base = i * 3;
            flops.push([
                decode_card(data.flop_bytes[base]),
                decode_card(data.flop_bytes[base + 1]),
                decode_card(data.flop_bytes[base + 2]),
            ]);
        }
        Some(Self {
            flops,
            tables_flat: data.tables_flat,
        })
    }

    /// Number of flops in the cache.
    pub fn num_flops(&self) -> usize {
        self.flops.len()
    }

    /// Extract equity tables for a specific set of flops (in the order given).
    ///
    /// For each requested flop, looks up its index in the cache and copies
    /// the corresponding 169x169 table. Returns `None` for any flop not
    /// found in the cache.
    pub fn extract_tables(&self, requested_flops: &[[Card; 3]]) -> Option<Vec<Vec<f64>>> {
        use rustc_hash::FxHashMap;
        let mut index_map: FxHashMap<[u8; 3], usize> = FxHashMap::default();
        for (i, flop) in self.flops.iter().enumerate() {
            let key = [
                encode_card(flop[0]),
                encode_card(flop[1]),
                encode_card(flop[2]),
            ];
            index_map.insert(key, i);
        }

        let mut tables = Vec::with_capacity(requested_flops.len());
        for flop in requested_flops {
            let key = [
                encode_card(flop[0]),
                encode_card(flop[1]),
                encode_card(flop[2]),
            ];
            let idx = index_map.get(&key)?;
            let start = idx * Self::TABLE_SIZE;
            let end = start + Self::TABLE_SIZE;
            tables.push(self.tables_flat[start..end].to_vec());
        }
        Some(tables)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_round_trip() {
        use rs_poker::core::{Suit, Value};
        let card = Card {
            value: Value::Ace,
            suit: Suit::Spade,
        };
        let encoded = encode_card(card);
        let decoded = decode_card(encoded);
        assert_eq!(card, decoded);
    }

    #[test]
    #[ignore] // compute_equity_table is expensive (~seconds per flop)
    fn save_load_round_trip() {
        // Build a tiny cache with just 1 flop to test serialization
        let flops = canonical_flops();
        let flop = flops[0];
        let combo_map = build_combo_map(&flop);
        let table = compute_equity_table(&combo_map, flop);

        let cache = EquityTableCache {
            flops: vec![flop],
            tables_flat: table.clone(),
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_equity.bin");
        cache.save(&path).unwrap();

        let loaded = EquityTableCache::load(&path).unwrap();
        assert_eq!(loaded.num_flops(), 1);
        assert_eq!(loaded.flops[0], flop);
        assert_eq!(loaded.tables_flat.len(), table.len());

        // Verify values match (NaN-aware)
        for (a, b) in loaded.tables_flat.iter().zip(table.iter()) {
            if a.is_nan() {
                assert!(b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-15);
            }
        }
    }

    #[test]
    #[ignore] // compute_equity_table is expensive (~seconds per flop)
    fn extract_tables_finds_cached_flop() {
        let flops = canonical_flops();
        let flop = flops[0];
        let combo_map = build_combo_map(&flop);
        let table = compute_equity_table(&combo_map, flop);

        let cache = EquityTableCache {
            flops: vec![flop],
            tables_flat: table,
        };

        let extracted = cache.extract_tables(&[flop]).unwrap();
        assert_eq!(extracted.len(), 1);
        assert_eq!(
            extracted[0].len(),
            NUM_CANONICAL_HANDS * NUM_CANONICAL_HANDS
        );
    }

    #[test]
    fn extract_tables_returns_none_for_missing_flop() {
        let flops = canonical_flops();
        let cache = EquityTableCache {
            flops: vec![flops[0]],
            tables_flat: vec![0.0; NUM_CANONICAL_HANDS * NUM_CANONICAL_HANDS],
        };
        // Ask for a different flop
        assert!(cache.extract_tables(&[flops[1]]).is_none());
    }

    #[test]
    fn load_returns_none_for_missing_file() {
        assert!(
            EquityTableCache::load(Path::new("/tmp/nonexistent_equity_cache_xyz.bin")).is_none()
        );
    }

    #[test]
    fn load_returns_none_for_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_magic.bin");
        let data = CacheData {
            magic: *b"XXXX",
            version: VERSION,
            num_flops: 0,
            flop_bytes: vec![],
            tables_flat: vec![],
        };
        let bytes = bincode::serialize(&data).unwrap();
        std::fs::write(&path, bytes).unwrap();
        assert!(EquityTableCache::load(&path).is_none());
    }

    #[test]
    #[ignore] // compute_equity_table is expensive (~seconds per flop)
    fn build_small_save_load_extract() {
        let flops = canonical_flops();
        let first_three: Vec<[Card; 3]> = flops[..3].to_vec();

        // Compute tables for 3 flops
        let mut tables_flat = Vec::new();
        for flop in &first_three {
            let combo_map = build_combo_map(flop);
            let table = compute_equity_table(&combo_map, *flop);
            tables_flat.extend_from_slice(&table);
        }

        let cache = EquityTableCache {
            flops: first_three.clone(),
            tables_flat,
        };

        // Save and reload
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("equity.bin");
        cache.save(&path).unwrap();

        let loaded = EquityTableCache::load(&path).unwrap();
        assert_eq!(loaded.num_flops(), 3);

        // Extract in different order
        let reversed: Vec<[Card; 3]> = vec![first_three[2], first_three[0]];
        let extracted = loaded.extract_tables(&reversed).unwrap();
        assert_eq!(extracted.len(), 2);

        // Verify second extracted table matches first flop's original table
        let combo_map = build_combo_map(&first_three[0]);
        let original = compute_equity_table(&combo_map, first_three[0]);
        for (a, b) in extracted[1].iter().zip(original.iter()) {
            if a.is_nan() {
                assert!(b.is_nan());
            } else {
                assert!((a - b).abs() < 1e-15);
            }
        }
    }
}
