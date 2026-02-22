//! Hand trace diagnostic: traces hands through EHS → buckets → postflop EV.

use serde::Serialize;

use poker_solver_core::hands::{all_hands, CanonicalHand};
use poker_solver_core::preflop::ehs::EhsFeatures;
use poker_solver_core::preflop::hand_buckets::{bucket_ehs_centroids, compute_all_flop_features};
use poker_solver_core::preflop::postflop_abstraction::PostflopAbstraction;
use poker_solver_core::preflop::postflop_model::PostflopModelConfig;
use poker_solver_core::poker::{Card, Suit, Value};

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct TraceOutput {
    hands: Vec<HandTrace>,
}

#[derive(Serialize)]
struct HandTrace {
    hand: String,
    canonical_index: usize,
    textures: Vec<TextureTrace>,
    summary: HandSummary,
}

#[derive(Serialize)]
struct TextureTrace {
    texture_id: usize,
    prototype_flop: String,
    ehs_features: [f64; 3],
    blocked: bool,
    flop_bucket: u16,
    bucket_centroid_ehs: f64,
    postflop_ev: PostflopEvTrace,
}

#[derive(Serialize)]
struct PostflopEvTrace {
    by_spr: Vec<SprEv>,
}

#[derive(Serialize)]
struct SprEv {
    spr: f64,
    sb: f64,
    bb: f64,
}

#[derive(Serialize)]
struct PositionEv {
    sb: f64,
    bb: f64,
}

#[derive(Serialize)]
struct HandSummary {
    avg_ehs: f64,
    bucket_range: [u16; 2],
    blocked_texture_count: usize,
    avg_postflop_ev_mid_spr: PositionEv,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run the hand trace diagnostic with a pre-built postflop abstraction.
pub fn run_with_abstraction(
    config: &PostflopModelConfig,
    abstraction: &PostflopAbstraction,
) -> Result<(), Box<dyn std::error::Error>> {
    let all_hands_vec: Vec<CanonicalHand> = all_hands().collect();

    eprintln!("Tracing all {} canonical hands...", all_hands_vec.len());

    // Compute EHS features for all hands
    eprintln!("Computing EHS features...");
    let flop_samples: Vec<Vec<[Card; 3]>> = abstraction
        .board
        .prototype_flops
        .iter()
        .map(|f| vec![*f])
        .collect();
    let features = compute_all_flop_features(&all_hands_vec, &flop_samples, &|_| {});
    let num_buckets = config.num_hand_buckets_flop as usize;
    let centroids =
        bucket_ehs_centroids(&features, &abstraction.buckets.flop_buckets, num_buckets);

    // Build and print JSON output
    let output = build_trace_output(&all_hands_vec, &features, abstraction, &centroids);
    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}

// ---------------------------------------------------------------------------
// Trace builder
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn build_trace_output(
    all_hands: &[CanonicalHand],
    features: &[Vec<EhsFeatures>],
    abstraction: &PostflopAbstraction,
    centroids: &[f64],
) -> TraceOutput {
    let num_textures = abstraction.board.prototype_flops.len();
    let num_buckets = centroids.len();

    let hands: Vec<HandTrace> = all_hands
        .iter()
        .enumerate()
        .map(|(hand_idx, hand)| {
            let mut textures = Vec::with_capacity(num_textures);
            let mut ehs_sum = 0.0_f64;
            let mut ehs_count = 0_usize;
            let mut blocked_count = 0_usize;
            let mut min_bucket = u16::MAX;
            let mut max_bucket = 0_u16;
            let mid_spr_idx = abstraction.canonical_sprs.len() / 2;
            let mut ev_mid_sb_sum = 0.0_f64;
            let mut ev_mid_bb_sum = 0.0_f64;
            let mut ev_count = 0_usize;

            #[allow(clippy::needless_range_loop)]
            for tex_id in 0..num_textures {
                let feat = features[hand_idx][tex_id];
                let blocked = feat[0].is_nan();
                let bucket = abstraction.buckets.flop_buckets[hand_idx][tex_id];
                let centroid_ehs = centroids.get(bucket as usize).copied().unwrap_or(0.0);

                let postflop_ev = compute_postflop_ev(abstraction, bucket, num_buckets);

                if blocked {
                    blocked_count += 1;
                } else {
                    ehs_sum += feat[0];
                    ehs_count += 1;
                }
                min_bucket = min_bucket.min(bucket);
                max_bucket = max_bucket.max(bucket);
                ev_mid_sb_sum += postflop_ev.by_spr[mid_spr_idx].sb;
                ev_mid_bb_sum += postflop_ev.by_spr[mid_spr_idx].bb;
                ev_count += 1;

                let flop_str = format_flop(&abstraction.board.prototype_flops[tex_id]);

                textures.push(TextureTrace {
                    texture_id: tex_id,
                    prototype_flop: flop_str,
                    ehs_features: if blocked {
                        [f64::NAN, f64::NAN, f64::NAN]
                    } else {
                        feat
                    },
                    blocked,
                    flop_bucket: bucket,
                    bucket_centroid_ehs: centroid_ehs,
                    postflop_ev,
                });
            }

            let avg_ehs = if ehs_count > 0 {
                ehs_sum / ehs_count as f64
            } else {
                f64::NAN
            };
            let avg_ev_mid_spr = if ev_count > 0 {
                PositionEv {
                    sb: ev_mid_sb_sum / ev_count as f64,
                    bb: ev_mid_bb_sum / ev_count as f64,
                }
            } else {
                PositionEv { sb: 0.0, bb: 0.0 }
            };

            HandTrace {
                hand: hand.to_string(),
                canonical_index: hand_idx,
                textures,
                summary: HandSummary {
                    avg_ehs,
                    bucket_range: [min_bucket, max_bucket],
                    blocked_texture_count: blocked_count,
                    avg_postflop_ev_mid_spr: avg_ev_mid_spr,
                },
            }
        })
        .collect();

    TraceOutput { hands }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Average postflop EV for a hero bucket across all opponent buckets (uniform weighting).
#[allow(clippy::cast_precision_loss)]
fn compute_postflop_ev(
    abstraction: &PostflopAbstraction,
    hero_bucket: u16,
    num_buckets: usize,
) -> PostflopEvTrace {
    let by_spr: Vec<SprEv> = abstraction
        .canonical_sprs
        .iter()
        .enumerate()
        .map(|(spr_idx, &spr)| {
            let mut evs = [0.0_f64; 2];
            for pos in 0..2_u8 {
                let sum: f64 = (0..num_buckets as u16)
                    .map(|opp| abstraction.values.get_by_spr(spr_idx, pos, hero_bucket, opp))
                    .sum();
                evs[pos as usize] = sum / num_buckets as f64;
            }
            SprEv {
                spr,
                sb: evs[0],
                bb: evs[1],
            }
        })
        .collect();
    PostflopEvTrace { by_spr }
}

fn format_card(card: Card) -> String {
    let rank = match card.value {
        Value::Ace => 'A',
        Value::King => 'K',
        Value::Queen => 'Q',
        Value::Jack => 'J',
        Value::Ten => 'T',
        Value::Nine => '9',
        Value::Eight => '8',
        Value::Seven => '7',
        Value::Six => '6',
        Value::Five => '5',
        Value::Four => '4',
        Value::Three => '3',
        Value::Two => '2',
    };
    let suit = match card.suit {
        Suit::Spade => 's',
        Suit::Heart => 'h',
        Suit::Diamond => 'd',
        Suit::Club => 'c',
    };
    format!("{rank}{suit}")
}

fn format_flop(flop: &[Card; 3]) -> String {
    flop.iter().map(|c| format_card(*c)).collect::<Vec<_>>().join(" ")
}
