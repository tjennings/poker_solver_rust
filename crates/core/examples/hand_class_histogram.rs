//! Generate a histogram of hand class frequencies across random deals.
//!
//! Classifies both players' hands on flop, turn, and river for N random
//! deals and prints per-street frequency tables. Useful for evaluating
//! whether rare but strategically important classes (Flush, StraightFlush,
//! etc.) appear often enough in the training deal pool.
//!
//! When run with `--stratified`, also generates a stratified pool for comparison.
//!
//! Usage:
//!   cargo run --release --example hand_class_histogram                        # 50K deals
//!   cargo run --release --example hand_class_histogram -- 200000              # 200K deals
//!   cargo run --release --example hand_class_histogram -- 50000 42            # 50K deals, seed 42
//!   cargo run --release --example hand_class_histogram -- 20000 1 --stratified

use std::time::Instant;

use poker_solver_core::game::{AbstractionMode, HunlPostflop, PostflopConfig};
use poker_solver_core::hand_class::{HandClass, classify};
use poker_solver_core::Game;

/// All hand classes in display order (made hands strongest-first, then draws).
const ALL_CLASSES: [HandClass; HandClass::COUNT] = [
    HandClass::StraightFlush,
    HandClass::FourOfAKind,
    HandClass::FullHouse,
    HandClass::Flush,
    HandClass::Straight,
    HandClass::Set,
    HandClass::Trips,
    HandClass::TwoPair,
    HandClass::Overpair,
    HandClass::Pair,
    HandClass::Underpair,
    HandClass::Overcards,
    HandClass::HighCard,
    HandClass::ComboDraw,
    HandClass::FlushDraw,
    HandClass::BackdoorFlushDraw,
    HandClass::Oesd,
    HandClass::Gutshot,
    HandClass::BackdoorStraightDraw,
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let deal_count: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let stratified = args.iter().any(|a| a == "--stratified");

    println!("=== Hand Class Histogram ===");
    println!("Deals: {deal_count}, Seed: {seed}\n");

    // --- Base pool ---
    let config = PostflopConfig {
        stack_depth: 100,
        bet_sizes: vec![1.0],
        ..PostflopConfig::default()
    };

    let start = Instant::now();
    let game = HunlPostflop::new(config.clone(), Some(AbstractionMode::HandClassV2 { strength_bits: 0, equity_bits: 0 }), deal_count);
    let deals = game.initial_states();
    println!("Generated {} base deals in {:?}\n", deals.len(), start.elapsed());

    let (counts, no_class) = count_all_streets(&deals);
    let total_per_street = (deals.len() as u64) * 2;

    println!("--- Base Pool ---");
    print_header();
    for class in &ALL_CLASSES {
        print_class_row(*class, &counts, total_per_street);
    }
    print_no_class_row(&no_class, total_per_street);
    print_separator();

    print_rare_analysis(&counts, total_per_street);

    // --- Stratified pool ---
    if stratified {
        println!("\n\n--- Stratified Pool (min 50 per class) ---\n");

        let start = Instant::now();
        let strat_game = HunlPostflop::new(config, Some(AbstractionMode::HandClassV2 { strength_bits: 0, equity_bits: 0 }), deal_count)
            .with_stratification(50, 500_000);
        let strat_deals = strat_game.initial_states();
        println!("Generated {} stratified deals in {:?}\n", strat_deals.len(), start.elapsed());

        let (strat_counts, strat_no_class) = count_all_streets(&strat_deals);
        let strat_total = (strat_deals.len() as u64) * 2;

        print_comparison_header();
        for class in &ALL_CLASSES {
            print_comparison_row(*class, &counts, total_per_street, &strat_counts, strat_total);
        }
        print_separator();
    }
}

fn count_all_streets(
    deals: &[poker_solver_core::game::PostflopState],
) -> ([[u64; HandClass::COUNT]; 3], [u64; 3]) {
    let mut counts = [[0u64; HandClass::COUNT]; 3];
    let mut no_class = [0u64; 3];

    for deal in deals {
        let board = deal.full_board.expect("deals should have full boards");
        let p1 = deal.p1_holding;
        let p2 = deal.p2_holding;

        no_class[0] += u64::from(classify_and_count(p1, &board[..3], &mut counts[0]));
        no_class[0] += u64::from(classify_and_count(p2, &board[..3], &mut counts[0]));

        no_class[1] += u64::from(classify_and_count(p1, &board[..4], &mut counts[1]));
        no_class[1] += u64::from(classify_and_count(p2, &board[..4], &mut counts[1]));

        no_class[2] += u64::from(classify_and_count(p1, &board[..5], &mut counts[2]));
        no_class[2] += u64::from(classify_and_count(p2, &board[..5], &mut counts[2]));
    }

    (counts, no_class)
}

/// Classify a hand and increment counters. Returns `true` if no class matched.
fn classify_and_count(
    hole: [poker_solver_core::poker::Card; 2],
    board: &[poker_solver_core::poker::Card],
    counts: &mut [u64; HandClass::COUNT],
) -> bool {
    if let Ok(classification) = classify(hole, board) {
        for class in classification.iter() {
            counts[class as usize] += 1;
        }
        classification.is_empty()
    } else {
        true
    }
}

fn print_rare_analysis(counts: &[[u64; HandClass::COUNT]; 3], total_per_street: u64) {
    println!("\nRare class analysis (classes appearing < 1% on river):");
    print_separator();
    println!(
        "{:<22} {:>8} {:>10}  Est. per 1K iters (500 samples)",
        "Class", "River #", "River %"
    );
    print_separator();

    for class in &ALL_CLASSES {
        let d = *class as usize;
        let river_count = counts[2][d];
        let river_pct = 100.0 * river_count as f64 / total_per_street as f64;
        if river_pct < 1.0 && river_count > 0 {
            let est_per_iter = river_pct / 100.0 * 1000.0;
            let est_per_1k = est_per_iter * 1000.0;
            println!(
                "{:<22} {:>8} {:>9.4}%  {:.0}",
                class, river_count, river_pct, est_per_1k
            );
        }
    }
    print_separator();
}

fn print_comparison_header() {
    println!(
        "{:<22} {:>10} {:>10}  {:>6}",
        "Class", "Base %", "Strat %", "Boost"
    );
    print_separator();
}

fn print_comparison_row(
    class: HandClass,
    base_counts: &[[u64; HandClass::COUNT]; 3],
    base_total: u64,
    strat_counts: &[[u64; HandClass::COUNT]; 3],
    strat_total: u64,
) {
    let d = class as usize;
    let base_pct = 100.0 * base_counts[2][d] as f64 / base_total as f64;
    let strat_pct = 100.0 * strat_counts[2][d] as f64 / strat_total as f64;

    if base_counts[2][d] == 0 && strat_counts[2][d] == 0 {
        return;
    }

    let boost = if base_pct > 0.0 {
        format!("{:.1}x", strat_pct / base_pct)
    } else {
        "new".to_string()
    };

    println!(
        "{:<22} {:>9.4}% {:>9.4}%  {:>6}",
        class, base_pct, strat_pct, boost
    );
}

fn print_no_class_row(no_class: &[u64; 3], total: u64) {
    let pcts: Vec<String> = (0..3)
        .map(|s| {
            let pct = 100.0 * no_class[s] as f64 / total as f64;
            if no_class[s] == 0 {
                format!("{:>10}", "-")
            } else {
                format!("{:>9.4}%", pct)
            }
        })
        .collect();
    println!("{:<22} {} {} {}", "(no class)", pcts[0], pcts[1], pcts[2]);
}

fn print_header() {
    println!(
        "{:<22} {:>10} {:>10} {:>10}",
        "Class", "Flop %", "Turn %", "River %"
    );
    print_separator();
}

fn print_separator() {
    println!("{}", "-".repeat(56));
}

fn print_class_row(class: HandClass, counts: &[[u64; HandClass::COUNT]; 3], total: u64) {
    let d = class as usize;
    let pcts: Vec<String> = (0..3)
        .map(|s| {
            let pct = 100.0 * counts[s][d] as f64 / total as f64;
            if counts[s][d] == 0 {
                format!("{:>10}", "-")
            } else {
                format!("{:>9.4}%", pct)
            }
        })
        .collect();
    println!("{:<22} {} {} {}", class, pcts[0], pcts[1], pcts[2]);
}
