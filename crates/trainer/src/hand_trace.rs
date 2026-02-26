//! Hand trace diagnostic: traces hands through canonical index → postflop EV.

use serde::Serialize;

use poker_solver_core::hands::{all_hands, CanonicalHand};
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
    hand_index: u16,
    postflop_ev: PostflopEvTrace,
}

#[derive(Serialize)]
struct PostflopEvTrace {
    avg_sb: f64,
    avg_bb: f64,
}

#[derive(Serialize)]
struct PositionEv {
    sb: f64,
    bb: f64,
}

#[derive(Serialize)]
struct HandSummary {
    canonical_index: usize,
    avg_postflop_ev_mid_spr: PositionEv,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run the hand trace diagnostic with a pre-built postflop abstraction.
pub fn run_with_abstraction(
    _config: &PostflopModelConfig,
    abstraction: &PostflopAbstraction,
) -> Result<(), Box<dyn std::error::Error>> {
    let all_hands_vec: Vec<CanonicalHand> = all_hands().collect();
    let flops = &abstraction.flops;
    let num_hands = 169;

    eprintln!("Tracing all {} canonical hands...", all_hands_vec.len());

    let output = build_trace_output(&all_hands_vec, flops, abstraction, num_hands);
    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}

// ---------------------------------------------------------------------------
// Trace builder
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn build_trace_output(
    all_hands: &[CanonicalHand],
    flops: &[[Card; 3]],
    abstraction: &PostflopAbstraction,
    num_hands: usize,
) -> TraceOutput {
    let num_textures = flops.len();

    let hands: Vec<HandTrace> = all_hands
        .iter()
        .enumerate()
        .map(|(hand_idx, hand)| {
            let mut textures = Vec::with_capacity(num_textures);
            let mut ev_sb_sum = 0.0_f64;
            let mut ev_bb_sum = 0.0_f64;
            let mut ev_count = 0_usize;

            for tex_id in 0..num_textures {
                let postflop_ev = compute_postflop_ev(abstraction, hand_idx as u16, num_hands);

                ev_sb_sum += postflop_ev.avg_sb;
                ev_bb_sum += postflop_ev.avg_bb;
                ev_count += 1;

                let flop_str = format_flop(&flops[tex_id]);

                textures.push(TextureTrace {
                    texture_id: tex_id,
                    prototype_flop: flop_str,
                    hand_index: hand_idx as u16,
                    postflop_ev,
                });
            }

            let avg_ev_mid_spr = if ev_count > 0 {
                PositionEv {
                    sb: ev_sb_sum / ev_count as f64,
                    bb: ev_bb_sum / ev_count as f64,
                }
            } else {
                PositionEv { sb: 0.0, bb: 0.0 }
            };

            HandTrace {
                hand: hand.to_string(),
                canonical_index: hand_idx,
                textures,
                summary: HandSummary {
                    canonical_index: hand_idx,
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

/// Average postflop EV for a hero hand across all opponent hands (uniform weighting).
#[allow(clippy::cast_precision_loss)]
fn compute_postflop_ev(
    abstraction: &PostflopAbstraction,
    hero_hand: u16,
    num_hands: usize,
) -> PostflopEvTrace {
    let num_flops = abstraction.values.num_flops();
    let mut evs = [0.0_f64; 2];
    for pos in 0..2_u8 {
        let sum: f64 = (0..num_flops)
            .flat_map(|flop_idx| {
                (0..num_hands as u16)
                    .map(move |opp| abstraction.values.get_by_flop(flop_idx, pos, hero_hand, opp))
            })
            .sum();
        #[allow(clippy::cast_precision_loss)]
        let denom = (num_flops * num_hands) as f64;
        evs[pos as usize] = sum / denom;
    }
    PostflopEvTrace {
        avg_sb: evs[0],
        avg_bb: evs[1],
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
