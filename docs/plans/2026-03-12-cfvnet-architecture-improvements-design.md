# CFVnet Architecture Improvements Design

**Goal:** Improve large-pot / complex-board accuracy by switching to 52-dim one-hot board encoding and adding a hard zero-sum constraint network (matching Supremus architecture).

**Motivation:** After retraining with pot-weighted loss and log-uniform sampling, large-pot spots still show 3-5x worse MAE than small pots. The ml-researcher identified two root causes: (1) scalar card encoding (5 features) gives the network no structured access to board texture, and (2) the soft aux loss doesn't guarantee zero-sum consistency.

---

## 1. Board Encoding (52-dim one-hot)

**Current:** 5 scalars (`card_id / 51.0`)
**New:** 52-dim binary vector, 1.0 at each board card's index, 0.0 elsewhere

Input layout changes from:
```
[OOP_range(1326), IP_range(1326), board(5), pot(1), stack(1), player(1)] = 2660
```
to:
```
[OOP_range(1326), IP_range(1326), board(52), pot(1), stack(1)] = 2706
```

Player indicator removed (see zero-sum section). Board encoding is always 52 dims regardless of street (turn has 4 ones, river has 5 ones). `input_size()` becomes a constant.

**Files:** `dataset.rs`, `network.rs`, `sampler.rs`

## 2. Zero-Sum Network Architecture

**Current:** Single-player output (1326), soft aux loss, player indicator in input
**New:** Dual-player output (2x1326), hard zero-sum constraint in forward pass

Network structure:
```
Input(2706) -> [Linear->BatchNorm->PReLU] x 7 -> Linear(2652)
                                                     |
                                             split into OOP(1326) + IP(1326)
                                                     |
                                             zero-sum correction
                                                     |
                                             output: [OOP_cfv(1326), IP_cfv(1326)]
```

Zero-sum correction (differentiable, applied in forward pass):
```
gv_oop = dot(range_oop, cfv_oop)
gv_ip  = dot(range_ip, cfv_ip)
error  = (gv_oop + gv_ip) / 2

cfv_oop_corrected = cfv_oop - error / sum(range_oop)
cfv_ip_corrected  = cfv_ip  - error / sum(range_ip)
```

Guarantees `dot(range_oop, cfv_oop) + dot(range_ip, cfv_ip) == 0` exactly on every forward pass.

Network signature changes — ranges must be passed into forward:
```rust
fn forward(&self, x: Tensor<B, 2>, range_oop: Tensor<B, 2>, range_ip: Tensor<B, 2>) -> Tensor<B, 2>
```

**Removes:** `player` input feature, `aux_game_value_loss`, `aux_loss_weight` config, `game_value` from training records

## 3. Training Record Format

**Current:** One record per player per situation (2 records per solve)
**New:** One record per situation with both players' CFVs

```rust
pub struct TrainingRecord {
    pub board: Vec<u8>,
    pub pot: f32,
    pub effective_stack: f32,
    pub oop_range: [f32; 1326],
    pub ip_range: [f32; 1326],
    pub oop_cfvs: [f32; 1326],    // was: cfvs (single player)
    pub ip_cfvs: [f32; 1326],     // new
    pub valid_mask: [u8; 1326],
}
```

Removed fields: `player`, `game_value`

Breaking change — requires data regeneration. Datagen writes one record per situation instead of two. Disk size roughly the same (fewer duplicated headers).

## 4. Loss Function

**Current:** `cfvnet_loss = masked_huber_loss + aux_weight * aux_game_value_loss`
**New:** `cfvnet_loss = masked_huber_loss(oop_cfvs) + masked_huber_loss(ip_cfvs)`

Both players' losses summed with equal weight. Pot-weighting applies to both. Zero-sum is enforced architecturally, not in the loss.

```rust
pub fn cfvnet_loss<B: Backend>(
    pred: Tensor<B, 2>,        // [batch, 2652]
    target: Tensor<B, 2>,      // [batch, 2652]
    mask: Tensor<B, 2>,        // [batch, 1326] — same mask for both
    huber_delta: f64,
    pot_weight: Option<Tensor<B, 1>>,
) -> Tensor<B, 1> {
    // split pred and target at dim 1326
    // loss_oop + loss_ip
}
```

Removed: `aux_game_value_loss`, `aux_loss_weight` config, `game_value` from CfvItem, `range` from loss

## 5. Inference / Evaluation Changes

- Forward pass requires `range_oop` and `range_ip` tensors (for zero-sum correction)
- Returns 2652 values — caller slices out the player they need
- `encode_situation_for_inference` no longer takes a `player` parameter
- One forward pass gives both players (was two passes before)

Files: `compare.rs`, `river_net_evaluator.rs`, `main.rs`

## 6. Testing Strategy

| Test | Validates |
|---|---|
| `encode_board_one_hot` | 52-dim vector has 1s at correct card positions |
| `encode_board_4_cards` | Turn boards produce 4 ones, 48 zeros |
| `input_size_constant` | Always 2706 regardless of street |
| `zero_sum_holds_exactly` | `dot(r_oop, cfv_oop) + dot(r_ip, cfv_ip) < 1e-6` |
| `zero_sum_gradients_finite` | Backward pass produces no NaN/Inf |
| `dual_output_shape` | Forward returns `[batch, 2652]` |
| `training_record_roundtrip` | New format serializes/deserializes correctly |
| `loss_both_players` | Loss computed over both OOP and IP CFVs |
| `full_pipeline_smoke_test` | Generate -> train -> verify loss finite |

## 7. What We're NOT Changing

- Hidden layer count/size (7x500)
- BatchNorm + PReLU activation
- Pot/stack normalization (`/400.0`)
- Huber delta (1.0)
- Pot-weighted loss
