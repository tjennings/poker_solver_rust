use rand::prelude::*;
use rs_poker::core::Card;

/// A single entry in the epoch schedule: one canonical flop with its
/// combinatorial weight (the number of raw deals it represents).
#[derive(Clone)]
pub struct ScheduleEntry {
    pub flop_index: u16,
    pub flop_cards: [Card; 3],
    /// Combinatorial weight -- how many raw deals this canonical flop represents.
    pub weight: u32,
}

/// Weighted round-robin schedule over all 1,755 canonical flops.
///
/// Each epoch iterates over every canonical flop once, processing
/// `weight` deals per flop to match the natural deal distribution.
pub struct EpochSchedule {
    pub entries: Vec<ScheduleEntry>,
}

impl EpochSchedule {
    /// Build a schedule covering all canonical flops from `enumerate_canonical_flops`.
    #[must_use]
    pub fn new() -> Self {
        use super::cluster_pipeline::enumerate_canonical_flops;
        let flops = enumerate_canonical_flops();
        let entries = flops
            .iter()
            .enumerate()
            .map(|(i, wb)| ScheduleEntry {
                flop_index: i as u16,
                flop_cards: wb.cards,
                weight: wb.weight,
            })
            .collect();
        Self { entries }
    }

    /// Randomly reorder the schedule entries.
    pub fn shuffle(&mut self, rng: &mut impl Rng) {
        self.entries.shuffle(rng);
    }

    /// Number of canonical flops in the schedule.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the schedule is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total number of raw deals across all entries (sum of weights).
    #[must_use]
    pub fn total_deals(&self) -> u64 {
        self.entries.iter().map(|e| u64::from(e.weight)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn schedule_has_1755_entries() {
        let schedule = EpochSchedule::new();
        assert_eq!(schedule.len(), 1755);
    }

    #[test]
    fn total_deals_is_22100() {
        let schedule = EpochSchedule::new();
        assert_eq!(schedule.total_deals(), 22100);
    }

    #[test]
    fn shuffle_changes_order() {
        let mut schedule = EpochSchedule::new();
        let original_first = schedule.entries[0].flop_index;
        let original_last = schedule.entries[schedule.len() - 1].flop_index;

        let mut rng = StdRng::seed_from_u64(42);
        schedule.shuffle(&mut rng);

        // After shuffle, at least one of the first or last entry should differ.
        let changed = schedule.entries[0].flop_index != original_first
            || schedule.entries[schedule.len() - 1].flop_index != original_last;
        assert!(changed, "shuffle should change the order of entries");
    }

    #[test]
    fn all_entries_have_positive_weight() {
        let schedule = EpochSchedule::new();
        for (i, entry) in schedule.entries.iter().enumerate() {
            assert!(
                entry.weight > 0,
                "entry {} (flop_index={}) has zero weight",
                i,
                entry.flop_index
            );
        }
    }

    #[test]
    fn flop_indices_are_sequential() {
        let schedule = EpochSchedule::new();
        for (i, entry) in schedule.entries.iter().enumerate() {
            assert_eq!(
                entry.flop_index,
                i as u16,
                "entry at position {} should have flop_index={}, got {}",
                i,
                i,
                entry.flop_index
            );
        }
    }
}
