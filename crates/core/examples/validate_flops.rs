//! Validates that every flop in a reference dataset exists in the generated flops.csv.
//!
//! Usage: validate_flops <reference.csv> <generated.csv>
//!
//! The reference file contains comma-separated canonical flops like `2c3d5h`
//! (cards concatenated, no header). The generated file is the output of
//! `poker-solver-trainer flops --format csv` with columns card1,card2,card3,...

use std::collections::HashSet;
use std::process;

use poker_solver_core::abstraction::CanonicalBoard;
use poker_solver_core::poker::{Card, Suit, Value};

fn parse_value(c: char) -> Option<Value> {
    Value::from_char(c)
}

fn parse_suit(c: char) -> Option<Suit> {
    Suit::from_char(c)
}

/// Parse a card string like "Ah" into a Card.
fn parse_card(s: &str) -> Option<Card> {
    let mut chars = s.chars();
    let value = parse_value(chars.next()?)?;
    let suit = parse_suit(chars.next()?)?;
    if chars.next().is_some() {
        return None; // trailing chars
    }
    Some(Card::new(value, suit))
}

/// Parse a 6-char flop like "2c3d5h" into three cards.
fn parse_concatenated_flop(s: &str) -> Option<[Card; 3]> {
    if s.len() != 6 {
        return None;
    }
    let c0 = parse_card(&s[0..2])?;
    let c1 = parse_card(&s[2..4])?;
    let c2 = parse_card(&s[4..6])?;
    Some([c0, c1, c2])
}

/// Canonicalize 3 cards and return a sorted key for set comparison.
fn canonical_key(cards: &[Card; 3]) -> Option<[Card; 3]> {
    let board = CanonicalBoard::from_cards(cards).ok()?;
    let mut key = [board.cards[0], board.cards[1], board.cards[2]];
    key.sort_by(|a, b| b.value.cmp(&a.value).then(a.suit.cmp(&b.suit)));
    Some(key)
}

/// Parse reference file: comma-separated concatenated flops across multiple lines.
fn parse_reference(content: &str) -> Vec<(String, [Card; 3])> {
    content
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter_map(|token| {
            let cards = parse_concatenated_flop(token)?;
            Some((token.to_string(), cards))
        })
        .collect()
}

/// Parse generated flops.csv: header row, then card1,card2,card3,... columns.
fn parse_generated(content: &str) -> Vec<(String, [Card; 3])> {
    content
        .lines()
        .skip(1) // header
        .filter(|line| !line.is_empty())
        .filter_map(|line| {
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() < 3 {
                return None;
            }
            let c0 = parse_card(cols[0])?;
            let c1 = parse_card(cols[1])?;
            let c2 = parse_card(cols[2])?;
            let label = format!("{},{},{}", cols[0], cols[1], cols[2]);
            Some((label, [c0, c1, c2]))
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: validate_flops <reference.csv> <generated.csv>");
        process::exit(1);
    }

    let ref_content = std::fs::read_to_string(&args[1]).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {e}", args[1]);
        process::exit(1);
    });
    let gen_content = std::fs::read_to_string(&args[2]).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {e}", args[2]);
        process::exit(1);
    });

    let ref_flops = parse_reference(&ref_content);
    let gen_flops = parse_generated(&gen_content);

    println!("Reference flops: {}", ref_flops.len());
    println!("Generated flops: {}", gen_flops.len());

    // Build set of canonical keys from generated file
    let gen_keys: HashSet<[Card; 3]> = gen_flops
        .iter()
        .filter_map(|(_, cards)| canonical_key(cards))
        .collect();

    println!("Unique canonical keys in generated: {}", gen_keys.len());

    // Validate each reference flop exists in generated
    let mut missing = Vec::new();
    let mut parse_errors = Vec::new();

    for (label, cards) in &ref_flops {
        match canonical_key(cards) {
            Some(key) => {
                if !gen_keys.contains(&key) {
                    missing.push(label.clone());
                }
            }
            None => {
                parse_errors.push(label.clone());
            }
        }
    }

    if !parse_errors.is_empty() {
        println!("\nFailed to canonicalize {} flops:", parse_errors.len());
        for label in &parse_errors {
            println!("  {label}");
        }
    }

    if missing.is_empty() && parse_errors.is_empty() {
        println!(
            "\nAll {} reference flops found in generated dataset.",
            ref_flops.len()
        );
    } else if !missing.is_empty() {
        println!("\nMISSING {} flops from generated dataset:", missing.len());
        for label in &missing {
            println!("  {label}");
        }
        process::exit(1);
    }
}
