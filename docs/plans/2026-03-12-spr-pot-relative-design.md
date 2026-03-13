# SPR Input + Pot-Relative Output Design

**Goal:** Match Supremus input/output normalization — single SPR feature, pot-relative CFV targets.

**Motivation:** Large-pot spots have 3-5x worse MAE. Supremus avoids this by predicting CFVs as fractions of pot, making the network scale-invariant. Their single scalar input is `pot / starting_chips` (effectively 1/SPR). We adapt this for variable-stack situations.

---

## 1. Input Change

**Current:** `[OOP_range(1326), IP_range(1326), board(52), pot(1), stack(1)] = 2706`
**New:** `[OOP_range(1326), IP_range(1326), board(52), spr(1)] = 2705`

SPR feature: `pot / effective_stack` (1/SPR, matching Supremus convention). No further normalization — value is naturally bounded and meaningful.

INPUT_SIZE: 2706 → 2705

**Files:** `dataset.rs`, `network.rs`

## 2. Output Change

**Current:** Network predicts raw CFVs in chip units.
**New:** Network predicts CFVs as fractions of pot.

During encoding (training): `target = cfvs / pot`
During inference: `cfvs = prediction * pot`

**Files:** `dataset.rs`, `river_net_evaluator.rs`, `main.rs`, `compare.rs`, `compare_turn.rs`

## 3. Loss Change

Remove pot-weighted loss. With pot-relative targets, errors are already comparable across pot sizes.

Remove `pot_weighted_loss` config option and `pot` field from training pipeline (`CfvItem`, `PreEncoded`, `ChunkTensors`, `MiniBatch`).

**Files:** `training.rs`, `loss.rs`, `config.rs`

## 4. What We're NOT Changing

- 52-dim one-hot board encoding
- Zero-sum constraint network (dual output)
- Hidden layer count/size (7x500)
- Huber delta (1.0)
- Training record binary format (pot is still stored as raw value — normalization happens at encode time)
