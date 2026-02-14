//! Reservoir buffer for accumulating training samples across CFR iterations.
//!
//! Uses Algorithm R (Vitter 1985) for weighted reservoir sampling: once the
//! buffer reaches capacity, each new sample replaces a random existing entry
//! with probability `capacity / total_seen`, ensuring a uniform sample of
//! everything offered.

use rand::Rng;

/// Fixed-capacity buffer that maintains a uniform random sample of all items
/// ever pushed, using reservoir sampling (Vitter 1985).
///
/// One buffer is used per player to accumulate advantage network training
/// samples across the entire SD-CFR run.
pub struct ReservoirBuffer<T> {
    data: Vec<T>,
    capacity: usize,
    total_seen: u64,
}

impl<T: Clone + Send> ReservoirBuffer<T> {
    /// Create an empty buffer that will hold at most `capacity` samples.
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
            total_seen: 0,
        }
    }

    /// Offer a sample to the buffer.
    ///
    /// If the buffer is not yet full the sample is appended directly.
    /// Otherwise it replaces a uniformly random existing entry with
    /// probability `capacity / total_seen` (Algorithm R).
    pub fn push(&mut self, sample: T, rng: &mut impl Rng) {
        self.total_seen += 1;

        if self.data.len() < self.capacity {
            self.data.push(sample);
            return;
        }

        // Reservoir replacement: accept with probability capacity / total_seen
        let j = rng.random_range(0..self.total_seen);
        if let Some(slot) = self.data.get_mut(j as usize) {
            // j < capacity, so replace
            *slot = sample;
        }
    }

    /// Number of samples currently stored.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the buffer contains no samples.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Draw `batch_size` samples uniformly at random (with replacement).
    ///
    /// Returns at most `self.len()` references if `batch_size` exceeds the
    /// stored count, and an empty vec when the buffer is empty.
    pub fn sample_batch(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<&T> {
        let n = batch_size.min(self.data.len());
        (0..n)
            .map(|_| {
                let idx = rng.random_range(0..self.data.len());
                &self.data[idx]
            })
            .collect()
    }

    /// Total number of samples ever offered via [`push`](Self::push),
    /// including those that were rejected by reservoir sampling.
    pub fn total_seen(&self) -> u64 {
        self.total_seen
    }

    /// Maximum number of samples this buffer can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn seeded_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    #[test]
    fn push_fills_to_capacity() {
        let mut buf = ReservoirBuffer::new(5);
        let mut rng = seeded_rng(42);

        for i in 0..5 {
            buf.push(i, &mut rng);
        }

        assert_eq!(buf.len(), 5);
        assert_eq!(buf.total_seen(), 5);
    }

    #[test]
    fn push_beyond_capacity_maintains_size() {
        let mut buf = ReservoirBuffer::new(5);
        let mut rng = seeded_rng(42);

        for i in 0..100 {
            buf.push(i, &mut rng);
        }

        assert_eq!(buf.len(), 5, "buffer should not grow beyond capacity");
        assert_eq!(buf.total_seen(), 100);
    }

    #[test]
    fn reservoir_sampling_uniformity() {
        // Push values 0..1000 into a buffer of capacity 100.
        // Repeat many trials and count how often each value lands in the buffer.
        // Under uniform reservoir sampling every value has equal probability
        // (capacity / total = 100 / 1000 = 0.10).
        let capacity = 100;
        let total = 1_000usize;
        let trials = 5_000;
        let mut counts = vec![0u32; total];

        for trial in 0..trials {
            let mut rng = seeded_rng(trial as u64);
            let mut buf = ReservoirBuffer::new(capacity);
            for v in 0..total {
                buf.push(v, &mut rng);
            }
            for &v in &buf.data {
                counts[v] += 1;
            }
        }

        let expected = (capacity as f64 / total as f64) * trials as f64; // 500
        let tolerance = expected * 0.15; // 15% relative tolerance

        for (value, &count) in counts.iter().enumerate() {
            let diff = (count as f64 - expected).abs();
            assert!(
                diff < tolerance,
                "value {value} appeared {count} times, expected ~{expected} (tolerance {tolerance})"
            );
        }
    }

    #[test]
    fn empty_buffer_returns_empty_batch() {
        let buf: ReservoirBuffer<i32> = ReservoirBuffer::new(10);
        let mut rng = seeded_rng(42);

        let batch = buf.sample_batch(5, &mut rng);
        assert!(batch.is_empty());
    }

    #[test]
    fn batch_size_capped_at_buffer_len() {
        let mut buf = ReservoirBuffer::new(10);
        let mut rng = seeded_rng(42);

        for i in 0..3 {
            buf.push(i, &mut rng);
        }

        let batch = buf.sample_batch(100, &mut rng);
        assert_eq!(batch.len(), 3, "batch should not exceed buffer len");
    }

    #[test]
    fn total_seen_tracks_all_pushes() {
        let mut buf = ReservoirBuffer::new(2);
        let mut rng = seeded_rng(42);

        assert_eq!(buf.total_seen(), 0);

        buf.push(1, &mut rng);
        assert_eq!(buf.total_seen(), 1);

        buf.push(2, &mut rng);
        assert_eq!(buf.total_seen(), 2);

        buf.push(3, &mut rng);
        assert_eq!(buf.total_seen(), 3);

        buf.push(4, &mut rng);
        assert_eq!(buf.total_seen(), 4);
    }

    #[test]
    fn new_buffer_is_empty() {
        let buf: ReservoirBuffer<f64> = ReservoirBuffer::new(100);

        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), 100);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let run = |seed| {
            let mut rng = seeded_rng(seed);
            let mut buf = ReservoirBuffer::new(5);
            for i in 0..50 {
                buf.push(i, &mut rng);
            }
            buf.data.clone()
        };

        assert_eq!(run(99), run(99), "same seed must produce identical buffers");
    }
}
