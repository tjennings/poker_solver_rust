//! Disk cache for per-board rank arrays.
//!
//! For each canonical flop, stores a `u32` rank ordinal for every concrete
//! combo on every (turn, river) board. The equity table can then be derived
//! from cached ranks via pure integer comparison -- no hand evaluation needed.
//!
//! File format: `b"RKAC"` magic, version `1u32`, bincode payload compressed
//! with zstd at level 3.

use std::io;
use std::path::Path;

use rayon::prelude::*;
use rs_poker::core::Card;
use serde::{Deserialize, Serialize};

use super::postflop_hands::{all_cards_vec, NUM_CANONICAL_HANDS};
use crate::showdown_equity::{rank_hand, rank_to_ordinal};

const MAGIC: [u8; 4] = *b"RKAC";
const VERSION: u32 = 1;

// ──────────────────────────────────────────────────────────────────────────────
// Card encoding helpers (local copies -- same scheme as equity_table_cache)
// ──────────────────────────────────────────────────────────────────────────────

#[inline]
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

// ──────────────────────────────────────────────────────────────────────────────
// Data types
// ──────────────────────────────────────────────────────────────────────────────

/// Rank data for one canonical flop: all boards x all combos.
#[derive(Clone, Serialize, Deserialize)]
pub struct FlopRankData {
    /// (turn, river) pairs as encoded card bytes.
    pub board_cards: Vec<(u8, u8)>,
    /// Flat combo cards: `(c1_encoded, c2_encoded)` for each combo in
    /// `combo_map` order.
    pub combo_cards: Vec<(u8, u8)>,
    /// Canonical hand index (0..168) for each combo.
    pub combo_canonical: Vec<u16>,
    /// Rank ordinals: `board_ranks[board_idx * num_combos + combo_idx]`.
    /// `u32::MAX` = combo conflicts with board.
    pub board_ranks: Vec<u32>,
}

/// On-disk representation (bincode-serialized, then zstd-compressed).
#[derive(Serialize, Deserialize)]
struct CacheData {
    magic: [u8; 4],
    version: u32,
    num_flops: u32,
    /// Flops encoded as `[value*4+suit; 3]` per flop, flattened.
    flop_bytes: Vec<u8>,
    entries: Vec<FlopRankData>,
}

/// In-memory rank array cache.
pub struct RankArrayCache {
    /// Canonical flops in deterministic order.
    pub flops: Vec<[Card; 3]>,
    /// One `FlopRankData` per flop.
    pub entries: Vec<FlopRankData>,
}

impl RankArrayCache {
    /// Save the cache to a binary file (bincode + zstd level 3).
    ///
    /// # Errors
    ///
    /// Returns an error if directory creation, serialization, compression,
    /// or file writing fails.
    #[allow(clippy::cast_possible_truncation)]
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
            entries: self.entries.clone(),
        };
        let raw = bincode::serialize(&data).map_err(io::Error::other)?;
        // Compress with zstd level 3 (good speed/ratio balance).
        let compressed =
            zstd::encode_all(std::io::Cursor::new(&raw), 3).map_err(io::Error::other)?;
        std::fs::write(path, compressed)
    }

    /// Load a cache from a binary file.
    ///
    /// Returns `None` if the file doesn't exist, has invalid magic/version,
    /// or fails to decompress/deserialize.
    #[must_use]
    pub fn load(path: &Path) -> Option<Self> {
        let compressed = std::fs::read(path).ok()?;
        let raw = zstd::decode_all(std::io::Cursor::new(&compressed)).ok()?;
        let data: CacheData = bincode::deserialize(&raw).ok()?;
        if data.magic != MAGIC || data.version != VERSION {
            return None;
        }
        let num_flops = data.num_flops as usize;
        if data.flop_bytes.len() != num_flops * 3 || data.entries.len() != num_flops {
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
            entries: data.entries,
        })
    }

    /// Number of flops in the cache.
    #[must_use]
    pub fn num_flops(&self) -> usize {
        self.flops.len()
    }

    /// Look up rank data for a specific flop.
    #[must_use]
    pub fn get_flop_data(&self, flop: &[Card; 3]) -> Option<&FlopRankData> {
        let key = [
            encode_card(flop[0]),
            encode_card(flop[1]),
            encode_card(flop[2]),
        ];
        self.flops.iter().enumerate().find_map(|(i, f)| {
            let fk = [encode_card(f[0]), encode_card(f[1]), encode_card(f[2])];
            if fk == key {
                Some(&self.entries[i])
            } else {
                None
            }
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Computation
// ──────────────────────────────────────────────────────────────────────────────

/// Flattened combo data extracted from a combo map.
struct FlatComboList {
    cards: Vec<(Card, Card)>,
    canonical: Vec<u16>,
    masks: Vec<u64>,
}

/// Build a flat combo list from the hierarchical combo map.
#[allow(clippy::cast_possible_truncation)]
fn build_combo_list(combo_map: &[Vec<(Card, Card)>]) -> FlatComboList {
    let mut cards = Vec::new();
    let mut canonical = Vec::new();
    let mut masks = Vec::new();
    for (idx, combos) in combo_map.iter().enumerate() {
        for &(c1, c2) in combos {
            cards.push((c1, c2));
            canonical.push(idx as u16);
            masks.push((1u64 << encode_card(c1)) | (1u64 << encode_card(c2)));
        }
    }
    FlatComboList {
        cards,
        canonical,
        masks,
    }
}

/// Evaluate rank ordinals for all combos on each board in parallel.
fn evaluate_combo_ranks(
    flop: [Card; 3],
    boards: &[(Card, Card, u64)],
    combos: &FlatComboList,
) -> Vec<u32> {
    let num_combos = combos.cards.len();
    let nested: Vec<Vec<u32>> = boards
        .par_iter()
        .map(|&(turn, river, board_mask)| {
            let board = [flop[0], flop[1], flop[2], turn, river];
            let mut ranks = vec![u32::MAX; num_combos];
            for (ci, &(c1, c2)) in combos.cards.iter().enumerate() {
                if board_mask & combos.masks[ci] != 0 {
                    continue;
                }
                ranks[ci] = rank_to_ordinal(rank_hand([c1, c2], &board));
            }
            ranks
        })
        .collect();

    let mut flat = Vec::with_capacity(boards.len() * num_combos);
    for board_rank in &nested {
        flat.extend_from_slice(board_rank);
    }
    flat
}

/// Compute rank arrays for a single flop.
///
/// For every (turn, river) board, evaluates every concrete combo's rank
/// ordinal. Parallelised over boards via rayon.
#[must_use]
pub fn compute_rank_arrays(combo_map: &[Vec<(Card, Card)>], flop: [Card; 3]) -> FlopRankData {
    let deck = all_cards_vec();
    let flop_mask: u64 =
        (1u64 << encode_card(flop[0])) | (1u64 << encode_card(flop[1])) | (1u64 << encode_card(flop[2]));

    let combos = build_combo_list(combo_map);

    let mut boards: Vec<(Card, Card, u64)> = Vec::with_capacity(1176);
    for (ti, &turn) in deck.iter().enumerate() {
        let tb = 1u64 << encode_card(turn);
        if flop_mask & tb != 0 {
            continue;
        }
        for &river in &deck[ti + 1..] {
            let rb = 1u64 << encode_card(river);
            if flop_mask & rb != 0 {
                continue;
            }
            boards.push((turn, river, flop_mask | tb | rb));
        }
    }

    let flat_ranks = evaluate_combo_ranks(flop, &boards, &combos);

    FlopRankData {
        board_cards: boards
            .iter()
            .map(|&(t, r, _)| (encode_card(t), encode_card(r)))
            .collect(),
        combo_cards: combos
            .cards
            .iter()
            .map(|&(c1, c2)| (encode_card(c1), encode_card(c2)))
            .collect(),
        combo_canonical: combos.canonical,
        board_ranks: flat_ranks,
    }
}

/// Build canonical hand index ranges from the combo map.
///
/// Returns a vec where `ranges[h]` is the start..end of combo indices
/// belonging to canonical hand `h`.
fn build_canonical_ranges(combo_map: &[Vec<(Card, Card)>]) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::with_capacity(combo_map.len());
    let mut ci = 0;
    for combos in combo_map {
        let start = ci;
        ci += combos.len();
        ranges.push(start..ci);
    }
    ranges
}

/// Accumulate equity for one board into the running totals.
///
/// For each (hero, opponent) canonical hand pair, compares rank ordinals
/// across all concrete combo matchups and adds to the (`equity_sum`, count) cell.
#[inline]
fn accumulate_board_equity(
    accum: &mut [(f64, u64)],
    ranks: &[u32],
    combo_masks: &[u64],
    canonical_range: &[std::ops::Range<usize>],
    n: usize,
) {
    for hero_idx in 0..n {
        let hero_range = &canonical_range[hero_idx];
        if hero_range.is_empty() {
            continue;
        }
        for opp_idx in 0..n {
            let opp_range = &canonical_range[opp_idx];
            if opp_range.is_empty() {
                continue;
            }
            let cell = &mut accum[hero_idx * n + opp_idx];

            for hi in hero_range.clone() {
                let hero_rank = ranks[hi];
                if hero_rank == u32::MAX {
                    continue;
                }
                let hm = combo_masks[hi];

                for oi in opp_range.clone() {
                    let opp_rank = ranks[oi];
                    if opp_rank == u32::MAX {
                        continue;
                    }
                    if hm & combo_masks[oi] != 0 {
                        continue;
                    }
                    cell.0 += match hero_rank.cmp(&opp_rank) {
                        std::cmp::Ordering::Greater => 1.0,
                        std::cmp::Ordering::Equal => 0.5,
                        std::cmp::Ordering::Less => 0.0,
                    };
                    cell.1 += 1;
                }
            }
        }
    }
}

/// Derive a 169x169 equity table from precomputed rank arrays.
///
/// Pure integer comparison -- no hand evaluation. Uses rayon `fold`/`reduce`
/// over boards with local accumulators for each thread.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn derive_equity_table(data: &FlopRankData, combo_map: &[Vec<(Card, Card)>]) -> Vec<f64> {
    let n = NUM_CANONICAL_HANDS;
    let num_combos = data.combo_cards.len();
    let num_boards = data.board_cards.len();

    let combo_masks: Vec<u64> = data
        .combo_cards
        .iter()
        .map(|&(c1, c2)| (1u64 << c1) | (1u64 << c2))
        .collect();

    let canonical_range = build_canonical_ranges(combo_map);

    let accum = (0..num_boards)
        .into_par_iter()
        .fold(
            || vec![(0.0f64, 0u64); n * n],
            |mut accum, board_idx| {
                let ranks =
                    &data.board_ranks[board_idx * num_combos..(board_idx + 1) * num_combos];
                accumulate_board_equity(&mut accum, ranks, &combo_masks, &canonical_range, n);
                accum
            },
        )
        .reduce(
            || vec![(0.0f64, 0u64); n * n],
            |mut a, b| {
                for i in 0..a.len() {
                    a[i].0 += b[i].0;
                    a[i].1 += b[i].1;
                }
                a
            },
        );

    let mut table = vec![f64::NAN; n * n];
    for i in 0..n * n {
        let (eq_sum, count) = accum[i];
        if count > 0 {
            table[i] = eq_sum / count as f64;
        }
    }
    table
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preflop::postflop_hands::{build_combo_map, canonical_flops};
    use test_macros::timed_test;

    #[timed_test(10)]
    fn save_load_round_trip() {
        let flops = canonical_flops();
        let flop = flops[0];
        let combo_map = build_combo_map(&flop);
        let data = compute_rank_arrays(&combo_map, flop);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_ranks.bin");

        let cache = RankArrayCache {
            flops: vec![flop],
            entries: vec![data],
        };
        cache.save(&path).unwrap();

        let loaded = RankArrayCache::load(&path).unwrap();
        assert_eq!(loaded.flops.len(), 1);
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.flops[0], flop);
        assert_eq!(
            loaded.entries[0].board_ranks.len(),
            cache.entries[0].board_ranks.len()
        );
        // Verify rank data matches exactly.
        assert_eq!(loaded.entries[0].board_ranks, cache.entries[0].board_ranks);
        assert_eq!(loaded.entries[0].combo_cards, cache.entries[0].combo_cards);
        assert_eq!(
            loaded.entries[0].combo_canonical,
            cache.entries[0].combo_canonical
        );
        assert_eq!(loaded.entries[0].board_cards, cache.entries[0].board_cards);
    }

    #[timed_test(10)]
    fn load_returns_none_for_missing_file() {
        assert!(
            RankArrayCache::load(std::path::Path::new("/tmp/nonexistent_rank_cache_xyz.bin"))
                .is_none()
        );
    }

    #[timed_test(30)]
    #[ignore = "slow: compares derived equity table against compute_equity_table"]
    fn derive_equity_matches_direct_computation() {
        use crate::preflop::postflop_exhaustive::compute_equity_table;
        use crate::preflop::postflop_hands::NUM_CANONICAL_HANDS;

        let flops = canonical_flops();
        let flop = flops[0];
        let combo_map = build_combo_map(&flop);

        let rank_data = compute_rank_arrays(&combo_map, flop);
        let derived = derive_equity_table(&rank_data, &combo_map);
        let direct = compute_equity_table(&combo_map, flop);

        let n = NUM_CANONICAL_HANDS;
        for h in 0..n {
            for o in 0..n {
                let idx = h * n + o;
                let a = direct[idx];
                let b = derived[idx];
                if a.is_nan() {
                    assert!(b.is_nan(), "mismatch at [{h}][{o}]: direct=NaN, derived={b}");
                } else {
                    assert!(
                        (a - b).abs() < 1e-10,
                        "mismatch at [{h}][{o}]: direct={a}, derived={b}, diff={}",
                        (a - b).abs()
                    );
                }
            }
        }
    }
}
