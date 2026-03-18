//! Per-flop regret store with background preloader and writer using
//! crossbeam channels for true multi-consumer support.
//!
//! Manages loading/saving [`CompactStorage`] instances per canonical flop,
//! using a reader thread for disk preloading and a writer thread for async
//! persistence.

// Arena indices are u32; flop indices fit in u16. Truncation is safe.
#![allow(clippy::cast_possible_truncation)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::{Sender, Receiver, bounded};
use rs_poker::core::Card;

use super::compact_storage::CompactStorage;
use super::epoch_schedule::EpochSchedule;
use super::game_tree::GameTree;

/// A ready-to-process flop work unit.
pub struct FlopWork {
    pub flop_index: u16,
    pub flop_cards: [Card; 3],
    pub weight: u32,
    pub storage: CompactStorage,
}

/// Manages loading/saving per-flop [`CompactStorage`] with a background
/// preloader and writer using crossbeam channels for true multi-consumer
/// support.
pub struct PerFlopRegretStore {
    dir: PathBuf,
    tree: Arc<GameTree>,
    bucket_counts: [u16; 4],
}

/// Return type for [`PerFlopRegretStore::start_epoch`]: the ready channel
/// receiver, dirty channel sender, and join handles for background threads.
pub type EpochChannels = (Receiver<FlopWork>, Sender<(u16, CompactStorage)>, Vec<JoinHandle<()>>);

impl PerFlopRegretStore {
    /// Create a new store, ensuring the directory exists.
    ///
    /// # Panics
    ///
    /// Panics if the regret directory cannot be created.
    #[must_use]
    pub fn new(dir: PathBuf, tree: Arc<GameTree>, bucket_counts: [u16; 4]) -> Self {
        std::fs::create_dir_all(&dir).expect("create regret dir");
        Self { dir, tree, bucket_counts }
    }

    /// Start an epoch. Spawns reader + writer threads.
    ///
    /// Returns:
    /// - `ready_rx`: workers recv [`FlopWork`] from this (multi-consumer safe)
    /// - `dirty_tx`: workers send `(flop_index, CompactStorage)` for async write
    /// - `handles`: join handles for reader + writer threads
    ///
    /// crossbeam's [`Receiver`] is `Clone` -- each worker thread clones it and
    /// calls `recv()` independently. No Mutex needed.
    #[must_use]
    pub fn start_epoch(
        &self,
        schedule: &EpochSchedule,
        buffer_size: usize,
    ) -> EpochChannels {
        let (ready_tx, ready_rx) = bounded::<FlopWork>(buffer_size);
        let (dirty_tx, dirty_rx) = bounded::<(u16, CompactStorage)>(buffer_size);

        // Reader thread: walks schedule, loads each flop's storage
        let entries = schedule.entries.clone();
        let dir = self.dir.clone();
        let tree = Arc::clone(&self.tree);
        let bucket_counts = self.bucket_counts;
        let reader = std::thread::spawn(move || {
            for entry in &entries {
                let storage = load_or_create(&dir, entry.flop_index, &tree, bucket_counts);
                let work = FlopWork {
                    flop_index: entry.flop_index,
                    flop_cards: entry.flop_cards,
                    weight: entry.weight,
                    storage,
                };
                if ready_tx.send(work).is_err() {
                    break; // workers dropped their receivers
                }
            }
            // ready_tx dropped here -- workers' recv() returns Err
        });

        // Writer thread: receives dirty storages, saves to disk
        let dir_w = self.dir.clone();
        let writer = std::thread::spawn(move || {
            while let Ok((flop_index, storage)) = dirty_rx.recv() {
                save_to_disk(&dir_w, flop_index, &storage);
            }
        });

        (ready_rx, dirty_tx, vec![reader, writer])
    }

    /// The directory where per-flop regret files are stored.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

/// Load a flop's storage from disk, or create a fresh zeroed one.
fn load_or_create(
    dir: &Path,
    flop_index: u16,
    tree: &GameTree,
    bucket_counts: [u16; 4],
) -> CompactStorage {
    let path = dir.join(format!("flop_{flop_index:04}.regrets"));
    if path.exists()
        && let Ok(storage) = CompactStorage::load_regrets(&path, tree, bucket_counts)
    {
        return storage;
    }
    CompactStorage::new(tree, bucket_counts)
}

/// Save a flop's storage to disk.
fn save_to_disk(dir: &Path, flop_index: u16, storage: &CompactStorage) {
    let path = dir.join(format!("flop_{flop_index:04}.regrets"));
    if let Err(e) = storage.save_regrets(&path) {
        eprintln!("Warning: failed to save {}: {e}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_v2::compact_storage::CompactStorage;
    use crate::blueprint_v2::epoch_schedule::{EpochSchedule, ScheduleEntry};
    use crate::blueprint_v2::game_tree::GameTree;
    use rs_poker::core::{Card, Suit, Value};
    use std::collections::HashSet;
    use std::sync::Arc;

    fn toy_tree() -> GameTree {
        GameTree::build(
            10.0,
            0.5,
            1.0,
            &[vec!["2.5bb".into()]],
            &[vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
        )
    }

    /// Build a test schedule with `n` entries using distinct flop cards.
    fn small_schedule(n: usize) -> EpochSchedule {
        let values = [
            Value::Two, Value::Three, Value::Four, Value::Five, Value::Six,
            Value::Seven, Value::Eight, Value::Nine, Value::Ten, Value::Jack,
            Value::Queen, Value::King, Value::Ace,
        ];
        let entries: Vec<ScheduleEntry> = (0..n)
            .map(|i| ScheduleEntry {
                flop_index: i as u16,
                flop_cards: [
                    Card::new(values[i % 13], Suit::Spade),
                    Card::new(values[i % 13], Suit::Heart),
                    Card::new(values[i % 13], Suit::Diamond),
                ],
                weight: (i as u32) + 1,
            })
            .collect();
        EpochSchedule { entries }
    }

    #[test]
    fn load_creates_fresh_when_no_file() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let tree = Arc::new(toy_tree());
        let bucket_counts = [50, 50, 50, 50];

        let storage = load_or_create(dir.path(), 9999, &tree, bucket_counts);

        // All regrets should be zero in a freshly created storage
        assert!(storage.num_slots() > 0);
        assert_eq!(storage.get_regret(0, 0, 0), 0);
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let tree = Arc::new(toy_tree());
        let bucket_counts = [50, 50, 50, 50];
        let flop_index: u16 = 42;

        // Create storage and modify a regret
        let storage = CompactStorage::new(&tree, bucket_counts);
        let node_idx = tree
            .nodes
            .iter()
            .position(|n| {
                matches!(n, crate::blueprint_v2::game_tree::GameNode::Decision { .. })
            })
            .expect("need a decision node") as u32;
        storage.add_regret(node_idx, 3, 0, 777);

        // Save to disk
        save_to_disk(dir.path(), flop_index, &storage);

        // Load it back
        let loaded = load_or_create(dir.path(), flop_index, &tree, bucket_counts);
        assert_eq!(loaded.get_regret(node_idx, 3, 0), 777);
    }

    #[test]
    fn start_epoch_delivers_all_entries() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let tree = Arc::new(toy_tree());
        let bucket_counts = [50, 50, 50, 50];

        let store = PerFlopRegretStore::new(dir.path().to_path_buf(), tree, bucket_counts);
        let schedule = small_schedule(5);

        let (ready_rx, _dirty_tx, handles) = store.start_epoch(&schedule, 4);

        let mut received = Vec::new();
        while let Ok(work) = ready_rx.recv() {
            received.push((work.flop_index, work.weight));
        }

        assert_eq!(received.len(), 5);
        for i in 0..5 {
            assert_eq!(received[i].0, i as u16, "flop_index mismatch at {i}");
            assert_eq!(
                received[i].1,
                (i as u32) + 1,
                "weight mismatch at {i}"
            );
        }

        drop(_dirty_tx);
        for h in handles {
            h.join().expect("thread should join cleanly");
        }
    }

    #[test]
    fn multiple_consumers_work() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let tree = Arc::new(toy_tree());
        let bucket_counts = [50, 50, 50, 50];

        let store = PerFlopRegretStore::new(dir.path().to_path_buf(), tree, bucket_counts);
        let schedule = small_schedule(10);

        let (ready_rx, dirty_tx, handles) = store.start_epoch(&schedule, 4);

        // Spawn 2 consumer threads that recv concurrently
        let collected = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut consumers = Vec::new();
        for _ in 0..2 {
            let rx = ready_rx.clone();
            let tx = dirty_tx.clone();
            let collected = Arc::clone(&collected);
            consumers.push(std::thread::spawn(move || {
                while let Ok(work) = rx.recv() {
                    collected
                        .lock()
                        .unwrap()
                        .push(work.flop_index);
                    tx.send((work.flop_index, work.storage)).ok();
                }
            }));
        }
        // Drop our copies of the channels so writer sees EOF
        drop(ready_rx);
        drop(dirty_tx);

        for c in consumers {
            c.join().expect("consumer thread should join");
        }
        for h in handles {
            h.join().expect("background thread should join");
        }

        let mut indices: Vec<u16> = collected.lock().unwrap().clone();
        indices.sort();
        let unique: HashSet<u16> = indices.iter().copied().collect();

        // All 10 entries consumed exactly once
        assert_eq!(indices.len(), 10, "should have 10 items total");
        assert_eq!(unique.len(), 10, "each flop should be consumed exactly once");
        for i in 0..10 {
            assert!(
                unique.contains(&(i as u16)),
                "missing flop_index {i}"
            );
        }
    }
}
