//! GPU-resident reservoir buffer for CFVNet training examples.
//!
//! A fixed-capacity circular buffer storing training records in GPU memory.
//! Each record consists of:
//!   - `input`:  `[INPUT_SIZE]` f32 — encoded situation features
//!   - `target`: `[OUTPUT_SIZE]` f32 — CFVs (training targets)
//!   - `mask`:   `[OUTPUT_SIZE]` f32 — valid combo mask (1.0 or 0.0)
//!   - `range`:  `[OUTPUT_SIZE]` f32 — acting player's range (for auxiliary loss)
//!   - `game_value`: f32 — scalar game value
//!
//! Records are inserted after each solve batch and randomly sampled to form
//! mini-batches for neural network training.

/// Input feature size for the CFVNet encoder.
///
/// Layout (2720 floats):
///   - `[0..1326]`:     OOP range weights
///   - `[1326..2652]`:  IP range weights
///   - `[2652..2704]`:  Board one-hot (52 cards)
///   - `[2704..2717]`:  Rank presence (13 ranks)
///   - `[2717]`:        Pot (normalised by 400)
///   - `[2718]`:        Effective stack (normalised by 400)
///   - `[2719]`:        Player indicator (0.0 = OOP, 1.0 = IP)
pub const INPUT_SIZE: usize = 2720;

/// Output (target) size — one CFV per combo.
pub const OUTPUT_SIZE: usize = 1326;

/// A CPU-side training record ready for reservoir insertion.
#[derive(Debug, Clone)]
pub struct TrainingRecord {
    pub input: Vec<f32>,
    pub target: Vec<f32>,
    pub mask: Vec<f32>,
    pub range: Vec<f32>,
    pub game_value: f32,
}

/// CPU-side reservoir buffer for training examples.
///
/// This is a fixed-capacity circular buffer. When full, new records
/// overwrite the oldest entries. The buffer supports random mini-batch
/// sampling for training.
///
/// While the plan calls for a GPU-resident reservoir, we implement a
/// CPU-side version first because:
///   1. It avoids the complexity of custom CUDA gather/insert kernels.
///   2. The mini-batch transfer to burn-cuda tensors would require a
///      GPU→GPU copy anyway (cudarc ↔ burn share different allocators).
///   3. The reservoir bottleneck is solve time, not data transfer.
///
/// A future optimisation can move storage to GPU memory.
pub struct Reservoir {
    inputs: Vec<f32>,
    targets: Vec<f32>,
    masks: Vec<f32>,
    ranges: Vec<f32>,
    game_values: Vec<f32>,
    capacity: usize,
    write_idx: usize,
    size: usize,
}

/// A sampled mini-batch of training data.
pub struct MiniBatch {
    /// `[batch_size * INPUT_SIZE]` input features.
    pub inputs: Vec<f32>,
    /// `[batch_size * OUTPUT_SIZE]` target CFVs.
    pub targets: Vec<f32>,
    /// `[batch_size * OUTPUT_SIZE]` valid-combo masks.
    pub masks: Vec<f32>,
    /// `[batch_size * OUTPUT_SIZE]` acting player's range.
    pub ranges: Vec<f32>,
    /// `[batch_size]` scalar game values.
    pub game_values: Vec<f32>,
    /// Number of records in this batch.
    pub batch_size: usize,
}

impl Reservoir {
    /// Create a new reservoir with the given capacity (maximum number of
    /// training records).
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Reservoir capacity must be > 0");
        Self {
            inputs: vec![0.0; capacity * INPUT_SIZE],
            targets: vec![0.0; capacity * OUTPUT_SIZE],
            masks: vec![0.0; capacity * OUTPUT_SIZE],
            ranges: vec![0.0; capacity * OUTPUT_SIZE],
            game_values: vec![0.0; capacity],
            capacity,
            write_idx: 0,
            size: 0,
        }
    }

    /// Number of records currently stored.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Maximum capacity of the reservoir.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Whether the reservoir is full.
    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    /// Insert a single training record into the reservoir.
    ///
    /// The record is written at `write_idx` (circular). If the reservoir is
    /// full, the oldest record is overwritten.
    pub fn insert(&mut self, record: &TrainingRecord) {
        assert_eq!(record.input.len(), INPUT_SIZE);
        assert_eq!(record.target.len(), OUTPUT_SIZE);
        assert_eq!(record.mask.len(), OUTPUT_SIZE);
        assert_eq!(record.range.len(), OUTPUT_SIZE);

        let idx = self.write_idx;

        // Copy input features
        let in_start = idx * INPUT_SIZE;
        self.inputs[in_start..in_start + INPUT_SIZE]
            .copy_from_slice(&record.input);

        // Copy target CFVs
        let out_start = idx * OUTPUT_SIZE;
        self.targets[out_start..out_start + OUTPUT_SIZE]
            .copy_from_slice(&record.target);

        // Copy mask
        self.masks[out_start..out_start + OUTPUT_SIZE]
            .copy_from_slice(&record.mask);

        // Copy range
        self.ranges[out_start..out_start + OUTPUT_SIZE]
            .copy_from_slice(&record.range);

        // Copy game value
        self.game_values[idx] = record.game_value;

        // Advance write pointer
        self.write_idx = (self.write_idx + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
    }

    /// Insert multiple training records from a batch of solved situations.
    ///
    /// Each situation produces two records: one from OOP's perspective and
    /// one from IP's perspective.
    ///
    /// # Arguments
    ///
    /// * `boards` — `[num_situations * 5]` board card indices.
    /// * `ranges_oop` — `[num_situations * 1326]` OOP range weights.
    /// * `ranges_ip` — `[num_situations * 1326]` IP range weights.
    /// * `pots` — `[num_situations]` pot sizes.
    /// * `stacks` — `[num_situations]` effective stacks.
    /// * `cfvs_oop` — `[num_situations * 1326]` OOP CFV targets.
    /// * `cfvs_ip` — `[num_situations * 1326]` IP CFV targets.
    /// * `num_situations` — Number of situations.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_batch(
        &mut self,
        boards: &[u32],
        ranges_oop: &[f32],
        ranges_ip: &[f32],
        pots: &[f32],
        stacks: &[f32],
        cfvs_oop: &[f32],
        cfvs_ip: &[f32],
        num_situations: usize,
    ) {
        assert_eq!(boards.len(), num_situations * 5);
        assert_eq!(ranges_oop.len(), num_situations * OUTPUT_SIZE);
        assert_eq!(ranges_ip.len(), num_situations * OUTPUT_SIZE);
        assert_eq!(pots.len(), num_situations);
        assert_eq!(stacks.len(), num_situations);
        assert_eq!(cfvs_oop.len(), num_situations * OUTPUT_SIZE);
        assert_eq!(cfvs_ip.len(), num_situations * OUTPUT_SIZE);

        for sit in 0..num_situations {
            let board = &boards[sit * 5..(sit + 1) * 5];
            let r_oop = &ranges_oop[sit * OUTPUT_SIZE..(sit + 1) * OUTPUT_SIZE];
            let r_ip = &ranges_ip[sit * OUTPUT_SIZE..(sit + 1) * OUTPUT_SIZE];
            let pot = pots[sit];
            let stack = stacks[sit];

            // OOP perspective (player = 0)
            let (input_oop, mask_oop, game_value_oop) =
                encode_input(r_oop, r_ip, board, pot, stack, 0);
            let cfv_oop = &cfvs_oop[sit * OUTPUT_SIZE..(sit + 1) * OUTPUT_SIZE];

            self.insert(&TrainingRecord {
                input: input_oop,
                target: cfv_oop.to_vec(),
                mask: mask_oop.clone(),
                range: r_oop.to_vec(),
                game_value: game_value_oop,
            });

            // IP perspective (player = 1)
            let (input_ip, mask_ip, game_value_ip) =
                encode_input(r_oop, r_ip, board, pot, stack, 1);
            let cfv_ip = &cfvs_ip[sit * OUTPUT_SIZE..(sit + 1) * OUTPUT_SIZE];

            self.insert(&TrainingRecord {
                input: input_ip,
                target: cfv_ip.to_vec(),
                mask: mask_ip,
                range: r_ip.to_vec(),
                game_value: game_value_ip,
            });
        }
    }

    /// Sample a random mini-batch of `batch_size` records from the reservoir.
    ///
    /// Uses the provided `rng` for index selection. Returns `Err` if the
    /// reservoir has fewer records than `batch_size`.
    pub fn sample_minibatch(
        &self,
        batch_size: usize,
        rng: &mut impl FnMut(usize) -> usize,
    ) -> Result<MiniBatch, String> {
        if self.size < batch_size {
            return Err(format!(
                "Reservoir has {} records but batch_size is {}",
                self.size, batch_size
            ));
        }

        let mut batch_inputs = vec![0.0f32; batch_size * INPUT_SIZE];
        let mut batch_targets = vec![0.0f32; batch_size * OUTPUT_SIZE];
        let mut batch_masks = vec![0.0f32; batch_size * OUTPUT_SIZE];
        let mut batch_ranges = vec![0.0f32; batch_size * OUTPUT_SIZE];
        let mut batch_game_values = vec![0.0f32; batch_size];

        for (b, gv) in batch_game_values.iter_mut().enumerate() {
            let idx = rng(self.size);

            // Gather input
            let in_src = idx * INPUT_SIZE;
            let in_dst = b * INPUT_SIZE;
            batch_inputs[in_dst..in_dst + INPUT_SIZE]
                .copy_from_slice(&self.inputs[in_src..in_src + INPUT_SIZE]);

            // Gather target
            let out_src = idx * OUTPUT_SIZE;
            let out_dst = b * OUTPUT_SIZE;
            batch_targets[out_dst..out_dst + OUTPUT_SIZE]
                .copy_from_slice(&self.targets[out_src..out_src + OUTPUT_SIZE]);

            // Gather mask
            batch_masks[out_dst..out_dst + OUTPUT_SIZE]
                .copy_from_slice(&self.masks[out_src..out_src + OUTPUT_SIZE]);

            // Gather range
            batch_ranges[out_dst..out_dst + OUTPUT_SIZE]
                .copy_from_slice(&self.ranges[out_src..out_src + OUTPUT_SIZE]);

            // Gather game value
            *gv = self.game_values[idx];
        }

        Ok(MiniBatch {
            inputs: batch_inputs,
            targets: batch_targets,
            masks: batch_masks,
            ranges: batch_ranges,
            game_values: batch_game_values,
            batch_size,
        })
    }
}

/// Encode a situation into the 2720-dimensional input feature vector.
///
/// Returns `(input, mask, game_value)` where:
///   - `input` is `[INPUT_SIZE]` f32
///   - `mask` is `[OUTPUT_SIZE]` f32 (1.0 for valid combos, 0.0 for blocked)
///   - `game_value` is a scalar (mean CFV over valid combos, computed later)
fn encode_input(
    range_oop: &[f32],
    range_ip: &[f32],
    board: &[u32],
    pot: f32,
    stack: f32,
    player: u8,
) -> (Vec<f32>, Vec<f32>, f32) {
    let mut input = vec![0.0f32; INPUT_SIZE];

    // [0..1326]: OOP range
    input[..OUTPUT_SIZE].copy_from_slice(range_oop);

    // [1326..2652]: IP range
    input[OUTPUT_SIZE..2 * OUTPUT_SIZE].copy_from_slice(range_ip);

    // [2652..2704]: Board one-hot (52 cards)
    for &card_val in board {
        let card = card_val as usize;
        if card < 52 {
            input[2652 + card] = 1.0;
        }
    }

    // [2704..2717]: Rank presence (13 ranks)
    for &card_val in board {
        let card = card_val as usize;
        if card < 52 {
            input[2704 + card / 4] = 1.0;
        }
    }

    // [2717]: Pot normalised
    input[2717] = pot / 400.0;

    // [2718]: Stack normalised
    input[2718] = stack / 400.0;

    // [2719]: Player indicator
    input[2719] = f32::from(player);

    // Build combo mask: valid if neither card is on the board
    let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
    let mut mask = vec![0.0f32; OUTPUT_SIZE];
    for (i, m) in mask.iter_mut().enumerate() {
        let (c1, c2) = range_solver::card::index_to_card_pair(i);
        if !board_set.contains(&c1) && !board_set.contains(&c2) {
            *m = 1.0;
        }
    }

    // Game value: will be computed by the caller from CFVs; placeholder 0.0
    (input, mask, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_solver::card::card_from_str;

    fn card(s: &str) -> u32 {
        u32::from(card_from_str(s).unwrap())
    }

    fn test_board() -> [u32; 5] {
        [card("2c"), card("5d"), card("8h"), card("Js"), card("Ac")]
    }

    fn dummy_range(board: &[u32; 5]) -> Vec<f32> {
        let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
        let mut range = vec![0.0f32; OUTPUT_SIZE];
        for (i, r) in range.iter_mut().enumerate() {
            let (c1, c2) = range_solver::card::index_to_card_pair(i);
            if !board_set.contains(&c1) && !board_set.contains(&c2) {
                *r = 0.5;
            }
        }
        range
    }

    fn make_record(board: &[u32; 5]) -> TrainingRecord {
        let range = dummy_range(board);
        let (input, mask, gv) =
            encode_input(&range, &range, board, 200.0, 400.0, 0);
        TrainingRecord {
            input,
            target: vec![0.1; OUTPUT_SIZE],
            mask,
            range,
            game_value: gv,
        }
    }

    #[test]
    fn test_reservoir_new() {
        let res = Reservoir::new(100);
        assert_eq!(res.size(), 0);
        assert_eq!(res.capacity(), 100);
        assert!(!res.is_full());
    }

    #[test]
    fn test_reservoir_insert_and_size() {
        let mut res = Reservoir::new(100);
        let board = test_board();
        let record = make_record(&board);

        for _ in 0..50 {
            res.insert(&record);
        }
        assert_eq!(res.size(), 50);
        assert!(!res.is_full());

        for _ in 0..50 {
            res.insert(&record);
        }
        assert_eq!(res.size(), 100);
        assert!(res.is_full());

        // Inserting beyond capacity wraps around
        res.insert(&record);
        assert_eq!(res.size(), 100); // still at capacity
    }

    #[test]
    fn test_reservoir_circular_overwrite() {
        let mut res = Reservoir::new(5);
        let board = test_board();

        // Insert 5 records with different game values
        for i in 0..5 {
            let range = dummy_range(&board);
            let (input, mask, _) =
                encode_input(&range, &range, &board, 200.0, 400.0, 0);
            res.insert(&TrainingRecord {
                input,
                target: vec![i as f32; OUTPUT_SIZE],
                mask,
                range,
                game_value: i as f32,
            });
        }
        assert_eq!(res.size(), 5);
        assert_eq!(res.game_values[0], 0.0);
        assert_eq!(res.game_values[4], 4.0);

        // Insert one more — overwrites position 0
        let range = dummy_range(&board);
        let (input, mask, _) =
            encode_input(&range, &range, &board, 200.0, 400.0, 0);
        res.insert(&TrainingRecord {
            input,
            target: vec![99.0; OUTPUT_SIZE],
            mask,
            range,
            game_value: 99.0,
        });
        assert_eq!(res.size(), 5);
        assert_eq!(res.game_values[0], 99.0); // overwritten
        assert_eq!(res.game_values[1], 1.0); // still original
    }

    #[test]
    fn test_reservoir_sample_minibatch() {
        let mut res = Reservoir::new(100);
        let board = test_board();

        // Insert 50 records
        for i in 0..50 {
            let range = dummy_range(&board);
            let (input, mask, _) =
                encode_input(&range, &range, &board, 200.0, 400.0, 0);
            res.insert(&TrainingRecord {
                input,
                target: vec![i as f32 * 0.01; OUTPUT_SIZE],
                mask,
                range,
                game_value: i as f32,
            });
        }

        // Sample a mini-batch of 10
        let mut counter = 0usize;
        let batch = res
            .sample_minibatch(10, &mut |bound| {
                counter += 1;
                // Deterministic: cycle through indices
                (counter * 7) % bound
            })
            .unwrap();

        assert_eq!(batch.batch_size, 10);
        assert_eq!(batch.inputs.len(), 10 * INPUT_SIZE);
        assert_eq!(batch.targets.len(), 10 * OUTPUT_SIZE);
        assert_eq!(batch.masks.len(), 10 * OUTPUT_SIZE);
        assert_eq!(batch.ranges.len(), 10 * OUTPUT_SIZE);
        assert_eq!(batch.game_values.len(), 10);
    }

    #[test]
    fn test_reservoir_sample_too_large() {
        let res = Reservoir::new(100);
        let mut rng = |_bound: usize| 0usize;
        let result = res.sample_minibatch(10, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_input_structure() {
        let board = test_board();
        let range = dummy_range(&board);
        let (input, mask, _) =
            encode_input(&range, &range, &board, 200.0, 400.0, 0);

        assert_eq!(input.len(), INPUT_SIZE);
        assert_eq!(mask.len(), OUTPUT_SIZE);

        // Check OOP range copied correctly
        assert_eq!(&input[..OUTPUT_SIZE], &range[..]);

        // Check IP range copied correctly
        assert_eq!(&input[OUTPUT_SIZE..2 * OUTPUT_SIZE], &range[..]);

        // Check board one-hot
        let ac_card = card("Ac") as usize;
        assert_eq!(input[2652 + ac_card], 1.0);

        // Check pot normalisation
        assert!((input[2717] - 0.5).abs() < 1e-6); // 200/400 = 0.5

        // Check stack normalisation
        assert!((input[2718] - 1.0).abs() < 1e-6); // 400/400 = 1.0

        // Check player indicator
        assert_eq!(input[2719], 0.0);

        // Check mask: blocked combos should be 0.0
        let board_set: Vec<u8> = board.iter().map(|&c| c as u8).collect();
        let valid_count = mask.iter().filter(|&&m| m == 1.0).count();
        // 5 board cards block C(5,2)=10 two-card combos plus 5*47=235 one-card combos
        // Valid = 1326 - 245 = 1081
        assert_eq!(valid_count, 1081);

        for (i, &m) in mask.iter().enumerate() {
            let (c1, c2) = range_solver::card::index_to_card_pair(i);
            if board_set.contains(&c1) || board_set.contains(&c2) {
                assert_eq!(m, 0.0, "Blocked combo {i} should have mask 0.0");
            } else {
                assert_eq!(m, 1.0, "Valid combo {i} should have mask 1.0");
            }
        }
    }

    #[test]
    fn test_encode_input_player_indicator() {
        let board = test_board();
        let range = dummy_range(&board);

        let (input_oop, _, _) =
            encode_input(&range, &range, &board, 200.0, 400.0, 0);
        assert_eq!(input_oop[2719], 0.0);

        let (input_ip, _, _) =
            encode_input(&range, &range, &board, 200.0, 400.0, 1);
        assert_eq!(input_ip[2719], 1.0);
    }

    #[test]
    fn test_insert_batch() {
        let mut res = Reservoir::new(100);
        let board = test_board();
        let range = dummy_range(&board);

        let num_situations = 3;
        let mut boards = Vec::new();
        let mut ranges_oop = Vec::new();
        let mut ranges_ip = Vec::new();
        let mut pots = Vec::new();
        let mut stacks = Vec::new();
        let cfvs = vec![0.01f32; OUTPUT_SIZE];
        let mut cfvs_oop = Vec::new();
        let mut cfvs_ip = Vec::new();

        for _ in 0..num_situations {
            boards.extend_from_slice(&board);
            ranges_oop.extend_from_slice(&range);
            ranges_ip.extend_from_slice(&range);
            pots.push(200.0f32);
            stacks.push(400.0f32);
            cfvs_oop.extend_from_slice(&cfvs);
            cfvs_ip.extend_from_slice(&cfvs);
        }

        res.insert_batch(
            &boards,
            &ranges_oop,
            &ranges_ip,
            &pots,
            &stacks,
            &cfvs_oop,
            &cfvs_ip,
            num_situations,
        );

        // Each situation produces 2 records (OOP + IP)
        assert_eq!(res.size(), num_situations * 2);
    }
}
