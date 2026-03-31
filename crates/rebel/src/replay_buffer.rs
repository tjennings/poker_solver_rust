use std::collections::VecDeque;
use std::sync::Mutex;

use rand::Rng;

/// A single training example: encoded PBS input and target CFVs.
#[derive(Clone)]
pub struct ReplayEntry {
    pub input: Vec<f32>,  // 2720 elements
    pub target: Vec<f32>, // 1326 elements
}

/// Thread-safe circular replay buffer.
///
/// Workers push training examples, the inference server samples batches
/// for training. Evicts oldest entries when full.
pub struct ReplayBuffer {
    entries: Mutex<VecDeque<ReplayEntry>>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        }
    }

    pub fn push(&self, entry: ReplayEntry) {
        let mut entries = self.entries.lock().unwrap();
        if entries.len() >= self.capacity {
            entries.pop_front();
        }
        entries.push_back(entry);
    }

    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sample `n` random entries. Returns fewer if buffer has < n entries.
    pub fn sample(&self, n: usize) -> Vec<ReplayEntry> {
        let entries = self.entries.lock().unwrap();
        let len = entries.len();
        if len == 0 {
            return Vec::new();
        }
        let mut rng = rand::rng();
        (0..n.min(len))
            .map(|_| {
                let idx = rng.random_range(0..len);
                entries[idx].clone()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_buffer_push_and_len() {
        let buf = ReplayBuffer::new(100);
        assert_eq!(buf.len(), 0);
        buf.push(ReplayEntry {
            input: vec![0.0; 2720],
            target: vec![0.0; 1326],
        });
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_replay_buffer_sample() {
        let buf = ReplayBuffer::new(100);
        for i in 0..50 {
            buf.push(ReplayEntry {
                input: vec![i as f32; 2720],
                target: vec![0.0; 1326],
            });
        }
        let samples = buf.sample(10);
        assert_eq!(samples.len(), 10);
        // Each sample's input[0] should be in [0, 50)
        for s in &samples {
            assert!(s.input[0] >= 0.0 && s.input[0] < 50.0);
        }
    }

    #[test]
    fn test_replay_buffer_evicts_oldest() {
        let buf = ReplayBuffer::new(5);
        for i in 0..10 {
            buf.push(ReplayEntry {
                input: vec![i as f32; 2720],
                target: vec![0.0; 1326],
            });
        }
        assert_eq!(buf.len(), 5);
        // Oldest entries (0-4) should be evicted, only 5-9 remain
        let samples = buf.sample(5);
        for s in &samples {
            assert!(s.input[0] >= 5.0);
        }
    }

    #[test]
    fn test_replay_buffer_thread_safe() {
        use std::sync::Arc;
        let buf = Arc::new(ReplayBuffer::new(1000));
        let buf2 = Arc::clone(&buf);
        let handle = std::thread::spawn(move || {
            for i in 0..100 {
                buf2.push(ReplayEntry {
                    input: vec![i as f32; 2720],
                    target: vec![0.0; 1326],
                });
            }
        });
        handle.join().unwrap();
        assert_eq!(buf.len(), 100);
    }
}
