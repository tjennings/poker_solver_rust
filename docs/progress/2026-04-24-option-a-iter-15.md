# Iteration 15 — Option A gadget tree E2E

**Spot:** `sb:2bb,bb:10bb,sb:22bb,bb:call|JhTh9h|bb:15bb,sb:call|7d` (same as iter-14)
**Branch at time of run:** `feat/option-a-gadget-tree`
**Comparison baselines:**
- iter-10 no-gadget baseline: `subgame_exp = 20932.49 mbb`
- iter-14 post-clamp gadget: `subgame_exp ≈ 40000 mbb` (2× worse than baseline)

---

## Reproduce

```bash
cargo build --release -p poker-solver-trainer
./target/release/poker-solver-trainer compare-solve \
    --bundle ./local_data/blueprints/1k_100bb_brdcfr_v2 \
    --snapshot snapshot_0013 \
    --spot 'sb:2bb,bb:10bb,sb:22bb,bb:call|JhTh9h|bb:15bb,sb:call|7d' \
    --river-boundary cfvnet \
    --river-model ./local_data/models/cfvnet_river_py_v2/checkpoint_epoch675.onnx \
    --gadget --iters 40 --tolerance 0.001 \
    2>&1 | tee /tmp/option_a_iter15.log
```

## Results

> [!NOTE] To be filled in after running the harness.

| Metric | iter-10 baseline | iter-14 post-clamp | **iter-15 Option A** |
|-|-|-|-|
| `exact_exp` (mbb) | 77.67 | — | **TBD** |
| `subgame_exp` (mbb) | 20932.49 | ~40000 | **TBD** |
| `worst_delta` | 1.0000 | — | **TBD** |
| gadget mode printed at startup | off | clamp | **tree** (expected) |
| Burch §3 safety invariant (test 4) | N/A | not checked | **PASSES** (offline verified at commit `5fc8153e`, test `gadget_safety_invariant_realized_cfv_geq_opt_out`) |

## Verdict (informational only — goal (a) is safety, not exploitability)

**TBD** — options:

- **Gadget ≈ baseline or better:** Option A's bucketed opt-out approximation is tight enough for this spot. Consider scrapping the rescoped `akg3` (un-abstracted CBVs) or deferring indefinitely.
- **Gadget better than iter-14 (post-clamp) but still worse than iter-10 (no-gadget):** expected outcome given MVP uses the same bucketed CBV source as iter-14. Safety holds by construction (test 4); looseness costs exploitability. Promotes `akg3` to active: tighter opt-outs via un-abstracted CBVs (Brown & Sandholm 2017 §6) or decision-node backward induction.
- **Gadget worse than iter-14:** unexpected. Would indicate a wiring issue between the gadget tree and cfvnet dispatch. Investigate before shipping.

## Commentary

**Safety is the shipped guarantee, not exploitability.** Test (4) in
`crates/tauri-app/tests/gadget_integration.rs` proves the Burch §3
sufficiency condition holds: `avg_realized_CFV[h] ≥ opt_out[h] − 0.01`
for every gadget-player hand. The iter-15 exploitability number is
informational and feeds the decision about whether to activate `akg3`
(opt-out tightness work) as follow-up.

## Next actions based on result

- [ ] If baseline-matching: close `akg3` as not-needed.
- [ ] If looser: promote `akg3` from blocked → todo with priority per product need.
- [ ] Regardless: retire the post-clamp `GadgetEvaluator` path via the follow-up bean (Task 18).
