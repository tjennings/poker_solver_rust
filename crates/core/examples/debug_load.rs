//! Debug script: load a strategy bundle and test key lookups.
//!
//! Usage: cargo run -p poker-solver-core --example debug_load -- ./handclass_100bb

use std::collections::HashMap;
use std::path::PathBuf;

use poker_solver_core::blueprint::{BlueprintStrategy, BundleConfig, StrategyBundle};
use poker_solver_core::info_key::{InfoKey, canonical_hand_index_from_str, spr_bucket};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./handclass_100bb".to_string());
    let dir = PathBuf::from(&path);

    println!("=== Debug Bundle Load ===\n");
    println!("Path: {}\n", dir.display());

    // --- Step 1: Check files ---
    let config_path = dir.join("config.yaml");
    let blueprint_path = dir.join("blueprint.bin");
    let boundaries_path = dir.join("boundaries.bin");
    for (name, p) in [
        ("config.yaml", &config_path),
        ("blueprint.bin", &blueprint_path),
        ("boundaries.bin", &boundaries_path),
    ] {
        let (exists, size) = match std::fs::metadata(p) {
            Ok(m) => (true, m.len()),
            Err(_) => (false, 0),
        };
        println!("  {name:<18} exists={exists:<5} size={size}");
    }
    println!();

    // --- Step 2: Deserialize config.yaml ---
    println!("--- config.yaml ---");
    let config_yaml = match std::fs::read_to_string(&config_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FAILED to read config.yaml: {e}");
            return;
        }
    };
    println!("{config_yaml}");

    let config: BundleConfig = match serde_yaml::from_str(&config_yaml) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAILED to deserialize config.yaml: {e}");
            return;
        }
    };
    println!("Parsed OK:");
    println!("  stack_depth:          {}", config.game.stack_depth);
    println!("  bet_sizes:            {:?}", config.game.bet_sizes);
    println!(
        "  max_raises_per_street:{}",
        config.game.max_raises_per_street
    );
    println!("  abstraction_mode:     {:?}", config.abstraction_mode);
    println!("  abstraction:          {:?}", config.abstraction);
    println!();

    // --- Step 3: Load blueprint.bin ---
    println!("--- blueprint.bin ---");
    let start = std::time::Instant::now();
    let blueprint = match BlueprintStrategy::load(&blueprint_path) {
        Ok(b) => {
            println!("Loaded in {:?}", start.elapsed());
            println!("  info_sets:  {}", b.len());
            println!("  iterations: {}", b.iterations_trained());
            b
        }
        Err(e) => {
            eprintln!("FAILED to load blueprint.bin: {e}");
            return;
        }
    };
    println!();

    // --- Step 4: Raw key inspection ---
    println!("--- Raw key inspection ---");
    let start = std::time::Instant::now();
    let raw: HashMap<u64, Vec<f32>> = match std::fs::File::open(&blueprint_path)
        .map_err(|e| e.to_string())
        .and_then(|f| {
            let reader = std::io::BufReader::new(f);
            bincode::deserialize_from::<_, (HashMap<u64, Vec<f32>>, u64)>(reader)
                .map(|(m, _)| m)
                .map_err(|e| e.to_string())
        }) {
        Ok(m) => {
            println!("Raw load in {:?}, {} keys", start.elapsed(), m.len());
            m
        }
        Err(e) => {
            eprintln!("Raw bincode failed: {e}");
            eprintln!("Falling back to key probing only.\n");
            HashMap::new()
        }
    };

    if !raw.is_empty() {
        let mut keys: Vec<u64> = raw.keys().copied().collect();
        keys.sort();

        // Street breakdown by extracting street bits (bits 43-42)
        let street_of = |k: u64| (k >> 42) & 0x3;
        let preflop_count = keys.iter().filter(|&&k| street_of(k) == 0).count();
        let flop_count = keys.iter().filter(|&&k| street_of(k) == 1).count();
        let turn_count = keys.iter().filter(|&&k| street_of(k) == 2).count();
        let river_count = keys.iter().filter(|&&k| street_of(k) == 3).count();

        println!("\nStreet breakdown:");
        println!("  Preflop: {preflop_count}");
        println!("  Flop:    {flop_count}");
        println!("  Turn:    {turn_count}");
        println!("  River:   {river_count}");
        println!("  Total:   {}", keys.len());

        // Sample preflop keys
        println!("\nSample preflop keys:");
        for key in keys.iter().filter(|&&k| street_of(k) == 0).take(20) {
            let n = raw[key].len();
            println!("  {key:#018x}  ({n} actions)");
        }
    }
    println!();

    // --- Step 5: Probe specific preflop keys ---
    println!("--- Preflop key probes ---");
    let probe_hands = ["AA", "KK", "AKs", "AKo", "AQs", "JTs", "76s", "72o", "32o"];
    // Initial SB opening: SPR and depth buckets vary by config
    let preflop_eff_stack = config.game.stack_depth * 2 - 2;
    let spr_b = spr_bucket(3, preflop_eff_stack);

    for hand in &probe_hands {
        if let Some(idx) = canonical_hand_index_from_str(hand) {
            let key = InfoKey::new(u32::from(idx), 0, spr_b, &[]).as_u64();
            match blueprint.lookup(key) {
                Some(probs) => {
                    let s: Vec<String> = probs.iter().map(|p| format!("{p:.3}")).collect();
                    println!("  {hand:<6} (key={key:#018x}) => [{}]", s.join(", "));
                }
                None => {
                    println!("  {hand:<6} (key={key:#018x}) => NOT FOUND");
                }
            }
        } else {
            println!("  {hand:<6} => INVALID HAND");
        }
    }
    println!();

    // --- Step 6: Simulated 13x13 matrix lookup ---
    println!("--- Simulated matrix lookup (preflop) ---");
    let ranks = [
        'A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2',
    ];
    let mut found = 0u32;
    let mut missing = 0u32;

    for (row, &rank1) in ranks.iter().enumerate() {
        for (col, &rank2) in ranks.iter().enumerate() {
            let suited = row < col;
            let rank_order = |c: char| -> u32 {
                match c {
                    'A' => 14,
                    'K' => 13,
                    'Q' => 12,
                    'J' => 11,
                    'T' => 10,
                    _ => c.to_digit(10).unwrap_or(0),
                }
            };
            let (high, low) = if rank_order(rank1) >= rank_order(rank2) {
                (rank1, rank2)
            } else {
                (rank2, rank1)
            };
            let hand = if high == low {
                format!("{high}{low}")
            } else if suited {
                format!("{high}{low}s")
            } else {
                format!("{high}{low}o")
            };

            if let Some(idx) = canonical_hand_index_from_str(&hand) {
                let key = InfoKey::new(u32::from(idx), 0, spr_b, &[]).as_u64();
                if blueprint.lookup(key).is_some() {
                    found += 1;
                } else {
                    missing += 1;
                }
            }
        }
    }

    println!("  Found:   {found}/169");
    println!("  Missing: {missing}/169");

    // --- Step 7: Full bundle load test ---
    println!("\n--- Full StrategyBundle::load ---");
    let start = std::time::Instant::now();
    match StrategyBundle::load(&dir) {
        Ok(bundle) => {
            println!("OK in {:?}", start.elapsed());
            println!(
                "  config.abstraction_mode: {:?}",
                bundle.config.abstraction_mode
            );
            println!("  blueprint info sets:     {}", bundle.blueprint.len());
            println!("  boundaries present:      {}", bundle.boundaries.is_some());
        }
        Err(e) => {
            eprintln!("FAILED: {e}");
        }
    }

    // --- Step 8: Scan test (simulates scan_node_classes) ---
    if config.abstraction_mode.is_hand_class() {
        println!("\n--- Scan test (hand_class node scan) ---");
        // Simulate exploring a flop after preflop limp:
        // pot = 4 (SB 1 + BB 2 + SB call 1), stacks = [198, 198] for 100BB
        let stack_depth = config.game.stack_depth;
        let limp_pot = 4u32; // After limp
        let limp_stacks = stack_depth * 2 - 2; // Both players have same stack after limp

        let spr_b = spr_bucket(limp_pot, limp_stacks);
        let street_num = 1u8; // Flop

        println!("  pot={limp_pot} spr_bucket={spr_b}");

        let position_key = InfoKey::new(0, street_num, spr_b, &[]).as_u64();
        let position_mask: u64 = (1u64 << 44) - 1;

        println!("  position_key: {position_key:#018x}");
        println!("  position_mask: {position_mask:#018x}");

        let matches: Vec<(u64, u32)> = blueprint
            .iter()
            .filter(|(k, _)| (**k & position_mask) == position_key)
            .map(|(k, _)| {
                let hand_bits = (*k >> 44) as u32;
                (*k, hand_bits)
            })
            .collect();

        println!("  Matches: {}", matches.len());

        if matches.is_empty() {
            // Try all pot/stack combos for flop street
            let street_mask: u64 = 0x3u64 << 42;
            let street_key: u64 = (u64::from(street_num) & 0x3) << 42;
            let mut bucket_combos: Vec<u32> = blueprint
                .iter()
                .filter(|(k, _)| (**k & street_mask) == street_key)
                .map(|(k, _)| {
                    let decoded = InfoKey::from_raw(*k);
                    decoded.spr_bucket()
                })
                .collect();
            bucket_combos.sort();
            bucket_combos.dedup();

            println!("  No exact match! Flop SPR buckets in blueprint:");
            for pb in &bucket_combos {
                println!("    spr_bucket={pb}");
            }
        } else {
            // Show some matched hand classifications
            for (key, bits) in matches.iter().take(10) {
                let class = poker_solver_core::hand_class::HandClassification::from_bits(*bits);
                let names = class.to_strings();
                println!("    key={key:#018x} bits={bits:>5} classes={names:?}");
            }
            if matches.len() > 10 {
                println!("    ... and {} more", matches.len() - 10);
            }
        }
    }

    println!("\n=== Done ===");
}
