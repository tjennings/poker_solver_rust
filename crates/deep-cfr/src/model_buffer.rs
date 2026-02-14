//! Stores trained value network weights per CFR iteration.
//!
//! Each entry holds safetensors-serialized weights alongside an iteration number
//! used as the linear CFR weight. Weighted sampling selects networks proportional
//! to their iteration, giving more recent (and presumably better) networks higher
//! probability.

use crate::SdCfrError;
use crate::network::AdvantageNet;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use rand::Rng;

/// A single stored network snapshot.
pub struct ModelEntry {
    /// Serialized VarMap in safetensors binary format.
    weights: Vec<u8>,
    /// CFR iteration when this network was trained.
    iteration: u32,
    /// Linear CFR weight (equal to iteration number).
    weight: f64,
}

/// Stores one trained value net per CFR iteration.
///
/// Grows by one entry per player per iteration. No eviction policy --
/// all networks are retained for weighted averaging.
pub struct ModelBuffer {
    entries: Vec<ModelEntry>,
}

impl ModelBuffer {
    /// Create an empty model buffer.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Serialize a VarMap to safetensors bytes and store it with the given iteration.
    pub fn push(&mut self, varmap: &VarMap, iteration: u32) -> Result<(), SdCfrError> {
        let bytes = serialize_varmap(varmap)?;
        self.entries.push(ModelEntry {
            weights: bytes,
            iteration,
            weight: f64::from(iteration),
        });
        Ok(())
    }

    /// Number of stored network snapshots.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the buffer contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Sample an entry index proportional to its weight (iteration number).
    ///
    /// Returns an error if the buffer is empty.
    pub fn sample_weighted(&self, rng: &mut impl Rng) -> Result<usize, SdCfrError> {
        let total = self.total_weight();
        if total <= 0.0 {
            return Err(SdCfrError::EmptyModelBuffer);
        }
        let threshold = rng.random::<f64>() * total;
        select_by_cumulative_weight(&self.entries, threshold)
    }

    /// Deserialize the entry at `index` into a fresh `AdvantageNet`.
    pub fn load_net(
        &self,
        index: usize,
        num_actions: usize,
        hidden_dim: usize,
        device: &Device,
    ) -> Result<AdvantageNet, SdCfrError> {
        let entry = &self.entries[index];
        let vs = VarBuilder::from_buffered_safetensors(entry.weights.clone(), DType::F32, device)?;
        let net = AdvantageNet::new(num_actions, hidden_dim, &vs)?;
        Ok(net)
    }

    /// Iterate over all stored entries.
    pub fn iter(&self) -> impl Iterator<Item = &ModelEntry> {
        self.entries.iter()
    }

    /// Sum of all entry weights.
    pub fn total_weight(&self) -> f64 {
        self.entries.iter().map(|e| e.weight).sum()
    }

    /// Weight of the entry at `index`.
    pub fn entry_weight(&self, index: usize) -> f64 {
        self.entries[index].weight
    }

    /// Iteration number of the entry at `index`.
    pub fn entry_iteration(&self, index: usize) -> u32 {
        self.entries[index].iteration
    }

    /// Iterate over `(index, weight)` pairs for all stored entries.
    pub fn iter_weights(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.entries.iter().enumerate().map(|(i, e)| (i, e.weight))
    }
}

impl Default for ModelBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelEntry {
    /// The CFR iteration that produced this network.
    pub fn iteration(&self) -> u32 {
        self.iteration
    }

    /// Linear CFR weight for this entry.
    pub fn weight(&self) -> f64 {
        self.weight
    }
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Serialize a `VarMap` to safetensors binary format in memory.
///
/// Extracts all named tensors from the VarMap and writes them using
/// the safetensors wire format.
fn serialize_varmap(varmap: &VarMap) -> Result<Vec<u8>, SdCfrError> {
    let data = varmap
        .data()
        .lock()
        .map_err(|e| SdCfrError::Config(format!("failed to lock VarMap: {e}")))?;
    let tensor_map: Vec<(String, candle_core::Tensor)> = data
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
        .collect();
    let refs: Vec<(&str, &candle_core::Tensor)> = tensor_map
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    let bytes = safetensors::tensor::serialize(refs, None)?;
    Ok(bytes)
}

/// Select an index from entries by cumulative weight, given a random threshold.
fn select_by_cumulative_weight(
    entries: &[ModelEntry],
    threshold: f64,
) -> Result<usize, SdCfrError> {
    let mut cumulative = 0.0;
    for (i, entry) in entries.iter().enumerate() {
        cumulative += entry.weight;
        if cumulative >= threshold {
            return Ok(i);
        }
    }
    // Floating-point rounding: return the last entry.
    entries
        .len()
        .checked_sub(1)
        .ok_or(SdCfrError::EmptyModelBuffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarMap;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Helper: create a VarMap + AdvantageNet for testing.
    fn make_varmap_and_net(num_actions: usize, hidden_dim: usize) -> (VarMap, AdvantageNet) {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let net = AdvantageNet::new(num_actions, hidden_dim, &vs).unwrap();
        (varmap, net)
    }

    /// Helper: create dummy card indices `[B, 7]`.
    fn dummy_cards(batch: usize) -> Tensor {
        let data: Vec<i64> = (0..batch)
            .flat_map(|b| {
                let offset = (b as i64 * 7) % 46;
                (offset..offset + 7).collect::<Vec<_>>()
            })
            .collect();
        Tensor::from_vec(data, (batch, 7), &Device::Cpu).unwrap()
    }

    /// Helper: create dummy bet features `[B, 48]`.
    fn dummy_bets(batch: usize) -> Tensor {
        Tensor::zeros((batch, 48), DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn empty_buffer() {
        let buf = ModelBuffer::new();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.total_weight(), 0.0);
    }

    #[test]
    fn push_and_len() {
        let mut buf = ModelBuffer::new();
        let (varmap, _net) = make_varmap_and_net(5, 16);

        buf.push(&varmap, 1).unwrap();
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());

        buf.push(&varmap, 2).unwrap();
        assert_eq!(buf.len(), 2);

        buf.push(&varmap, 3).unwrap();
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn sample_weighted_distribution() {
        let mut buf = ModelBuffer::new();
        let (varmap, _net) = make_varmap_and_net(5, 16);

        // Push iterations 1, 2, 3 -> weights 1, 2, 3 -> total 6
        // Expected probabilities: 1/6, 2/6, 3/6
        buf.push(&varmap, 1).unwrap();
        buf.push(&varmap, 2).unwrap();
        buf.push(&varmap, 3).unwrap();

        let mut rng = StdRng::seed_from_u64(42);
        let num_samples = 60_000;
        let mut counts = [0u32; 3];

        for _ in 0..num_samples {
            let idx = buf.sample_weighted(&mut rng).unwrap();
            counts[idx] += 1;
        }

        let total = num_samples as f64;
        let p0 = counts[0] as f64 / total;
        let p1 = counts[1] as f64 / total;
        let p2 = counts[2] as f64 / total;

        // Expected: ~0.167, ~0.333, ~0.500. Allow 2% tolerance.
        assert!((p0 - 1.0 / 6.0).abs() < 0.02, "p0={p0:.4}, expected ~0.167");
        assert!((p1 - 2.0 / 6.0).abs() < 0.02, "p1={p1:.4}, expected ~0.333");
        assert!((p2 - 3.0 / 6.0).abs() < 0.02, "p2={p2:.4}, expected ~0.500");
    }

    #[test]
    fn save_load_round_trip() {
        let num_actions = 5;
        let hidden_dim = 16;
        let (varmap, net) = make_varmap_and_net(num_actions, hidden_dim);

        let cards = dummy_cards(2);
        let bets = dummy_bets(2);
        let original_output = net.forward(&cards, &bets).unwrap();

        let mut buf = ModelBuffer::new();
        buf.push(&varmap, 1).unwrap();

        let loaded_net = buf
            .load_net(0, num_actions, hidden_dim, &Device::Cpu)
            .unwrap();
        let loaded_output = loaded_net.forward(&cards, &bets).unwrap();

        let diff = original_output
            .sub(&loaded_output)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(
            diff < 1e-10,
            "Round-trip outputs should match, got diff={diff}"
        );
    }

    #[test]
    fn total_weight_is_sum_of_iterations() {
        let mut buf = ModelBuffer::new();
        let (varmap, _net) = make_varmap_and_net(5, 16);

        buf.push(&varmap, 3).unwrap();
        buf.push(&varmap, 7).unwrap();
        buf.push(&varmap, 11).unwrap();

        // Weights equal iteration numbers: 3 + 7 + 11 = 21
        let expected = 3.0 + 7.0 + 11.0;
        assert!(
            (buf.total_weight() - expected).abs() < 1e-10,
            "total_weight={}, expected={expected}",
            buf.total_weight()
        );
    }

    #[test]
    fn entry_accessors_return_correct_values() {
        let mut buf = ModelBuffer::new();
        let (varmap, _net) = make_varmap_and_net(5, 16);

        buf.push(&varmap, 5).unwrap();
        buf.push(&varmap, 10).unwrap();

        assert_eq!(buf.entry_iteration(0), 5);
        assert_eq!(buf.entry_iteration(1), 10);
        assert!((buf.entry_weight(0) - 5.0).abs() < 1e-10);
        assert!((buf.entry_weight(1) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn sample_weighted_errors_on_empty_buffer() {
        let buf = ModelBuffer::new();
        let mut rng = StdRng::seed_from_u64(0);
        let result = buf.sample_weighted(&mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn iter_yields_all_entries() {
        let mut buf = ModelBuffer::new();
        let (varmap, _net) = make_varmap_and_net(5, 16);

        buf.push(&varmap, 1).unwrap();
        buf.push(&varmap, 2).unwrap();
        buf.push(&varmap, 3).unwrap();

        let iterations: Vec<u32> = buf.iter().map(|e| e.iteration()).collect();
        assert_eq!(iterations, vec![1, 2, 3]);
    }
}
