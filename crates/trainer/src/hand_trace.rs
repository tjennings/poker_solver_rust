//! Hand trace diagnostic: traces hands through EHS → buckets → postflop EV.

use std::path::Path;

use serde::Serialize;

use poker_solver_core::hands::{all_hands, CanonicalHand};
use poker_solver_core::preflop::ehs::EhsFeatures;
use poker_solver_core::preflop::hand_buckets::{bucket_ehs_centroids, compute_all_flop_features};
use poker_solver_core::preflop::postflop_abstraction::PostflopAbstraction;
use poker_solver_core::preflop::postflop_model::{GameStructure, PostflopModelConfig};
use poker_solver_core::preflop::postflop_tree::PotType;
use poker_solver_core::poker::{Card, Suit, Value};

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct TraceOutput {
    spr: SprTrace,
    hands: Vec<HandTrace>,
}

#[derive(Serialize)]
struct SprTrace {
    limped: Option<f64>,
    raised: Option<f64>,
    three_bet: Option<f64>,
    four_bet_plus: Option<f64>,
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
    limped: PositionEv,
    raised: PositionEv,
    three_bet: PositionEv,
    four_bet_plus: PositionEv,
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
    avg_postflop_ev_raised: PositionEv,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run the hand trace diagnostic for all 169 canonical hands.
pub fn run(
    config: &PostflopModelConfig,
    cache_dir: &Path,
    game_structure: Option<&GameStructure>,
) -> Result<(), Box<dyn std::error::Error>> {
    let all_hands_vec: Vec<CanonicalHand> = all_hands().collect();

    eprintln!("Tracing all {} canonical hands...", all_hands_vec.len());

    // Build the full postflop abstraction (loads board/buckets/equity from cache,
    // rebuilds trees + solves postflop)
    eprintln!("Building postflop abstraction...");
    let abstraction = PostflopAbstraction::build_with_game_structure(
        config, None, Some(cache_dir), game_structure, |phase| {
            eprintln!("  {phase:?}");
        },
    )?;

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

    // Build SPR trace from abstraction
    let spr = if let Some(ref sprs) = abstraction.spr_values {
        SprTrace {
            limped: sprs.get(&PotType::Limped).copied(),
            raised: sprs.get(&PotType::Raised).copied(),
            three_bet: sprs.get(&PotType::ThreeBet).copied(),
            four_bet_plus: sprs.get(&PotType::FourBetPlus).copied(),
        }
    } else {
        SprTrace {
            limped: None,
            raised: None,
            three_bet: None,
            four_bet_plus: None,
        }
    };

    // Build and print JSON output
    let output = build_trace_output(&all_hands_vec, &features, &abstraction, &centroids, spr);
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
    spr: SprTrace,
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
            let mut ev_raised_sb_sum = 0.0_f64;
            let mut ev_raised_bb_sum = 0.0_f64;
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
                ev_raised_sb_sum += postflop_ev.raised.sb;
                ev_raised_bb_sum += postflop_ev.raised.bb;
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
            let avg_ev_raised = if ev_count > 0 {
                PositionEv {
                    sb: ev_raised_sb_sum / ev_count as f64,
                    bb: ev_raised_bb_sum / ev_count as f64,
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
                    avg_postflop_ev_raised: avg_ev_raised,
                },
            }
        })
        .collect();

    TraceOutput { spr, hands }
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
    let pot_types = [
        PotType::Limped,
        PotType::Raised,
        PotType::ThreeBet,
        PotType::FourBetPlus,
    ];

    let mut evs = [[0.0_f64; 2]; 4];
    for (pt_idx, &pot_type) in pot_types.iter().enumerate() {
        for pos in 0..2_u8 {
            let sum: f64 = (0..num_buckets as u16)
                .map(|opp| abstraction.values.get(pot_type, pos, hero_bucket, opp))
                .sum();
            evs[pt_idx][pos as usize] = sum / num_buckets as f64;
        }
    }

    PostflopEvTrace {
        limped: PositionEv {
            sb: evs[0][0],
            bb: evs[0][1],
        },
        raised: PositionEv {
            sb: evs[1][0],
            bb: evs[1][1],
        },
        three_bet: PositionEv {
            sb: evs[2][0],
            bb: evs[2][1],
        },
        four_bet_plus: PositionEv {
            sb: evs[3][0],
            bb: evs[3][1],
        },
    }
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
