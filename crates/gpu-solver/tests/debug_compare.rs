//! Debug comparison test: runs GPU and CPU solvers on the same positions
//! and prints detailed side-by-side strategy comparisons.
//!
//! Run with: cargo test -p poker-solver-gpu --features cuda --release --test debug_compare -- --nocapture

#![cfg(feature = "cuda")]

use poker_solver_gpu::gpu::GpuContext;
use poker_solver_gpu::solver::GpuSolver;
use poker_solver_gpu::tree::FlatTree;
use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
use range_solver::bet_size::BetSizeOptions;
use range_solver::card::{card_from_str, flop_from_str};
use range_solver::interface::Game;
use range_solver::range::Range;
use range_solver::{solve, CardConfig, PostFlopGame};

/// Build a river game with configurable pot and stack.
fn build_game(
    oop_range_str: &str,
    ip_range_str: &str,
    flop: &str,
    turn: &str,
    river: &str,
    starting_pot: i32,
    effective_stack: i32,
) -> PostFlopGame {
    let oop_range: Range = oop_range_str.parse().unwrap();
    let ip_range: Range = ip_range_str.parse().unwrap();
    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: flop_from_str(flop).unwrap(),
        turn: card_from_str(turn).unwrap(),
        river: card_from_str(river).unwrap(),
    };
    let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot,
        effective_stack,
        river_bet_sizes: [sizes.clone(), sizes],
        ..Default::default()
    };
    let tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
    game.allocate_memory(false);
    game
}

struct Position {
    name: &'static str,
    oop_range: &'static str,
    ip_range: &'static str,
    flop: &'static str,
    turn: &'static str,
    river: &'static str,
    pot: i32,
    stack: i32,
}

#[test]
fn debug_compare_gpu_vs_cpu() {
    let positions = [
        Position {
            name: "Position 1: QQ+,AKs vs QQ-JJ,AQs,AJs on Qs Jh 2c 8d 3s",
            oop_range: "QQ+,AKs",
            ip_range: "QQ-JJ,AQs,AJs",
            flop: "Qs Jh 2c",
            turn: "8d",
            river: "3s",
            pot: 100,
            stack: 100,
        },
        Position {
            name: "Position 2: AA,KK vs 22+,AK on Kd 7h 3c 9s 2d",
            oop_range: "AA,KK",
            ip_range: "22+,AK",
            flop: "Kd 7h 3c",
            turn: "9s",
            river: "2d",
            pot: 200,
            stack: 150,
        },
        Position {
            name: "Position 3: TT+,AQ+ vs 77+,AJ+ on Tc 8s 4h Jd 2c",
            oop_range: "TT+,AQ+",
            ip_range: "77+,AJ+",
            flop: "Tc 8s 4h",
            turn: "Jd",
            river: "2c",
            pot: 80,
            stack: 200,
        },
    ];

    let iterations = 1000u32;

    for pos in &positions {
        eprintln!("\n{}", "=".repeat(80));
        eprintln!("{}", pos.name);
        eprintln!("{}", "=".repeat(80));

        // --- CPU solve ---
        let mut cpu_game = build_game(
            pos.oop_range,
            pos.ip_range,
            pos.flop,
            pos.turn,
            pos.river,
            pos.pot,
            pos.stack,
        );
        let cpu_exploit = solve(&mut cpu_game, iterations, 0.0, false);
        eprintln!("CPU exploitability: {:.6}", cpu_exploit);

        cpu_game.back_to_root();
        let cpu_strategy = cpu_game.strategy();
        let cpu_player = cpu_game.current_player();
        let cpu_num_hands = cpu_game.num_private_hands(cpu_player);
        let cpu_actions = cpu_game.available_actions();
        let cpu_num_actions = cpu_actions.len();

        // Get private card names for the acting player
        let cpu_cards = cpu_game.private_cards(cpu_player);

        // --- GPU solve ---
        let mut gpu_game = build_game(
            pos.oop_range,
            pos.ip_range,
            pos.flop,
            pos.turn,
            pos.river,
            pos.pot,
            pos.stack,
        );
        let flat_tree = FlatTree::from_postflop_game(&mut gpu_game);
        let gpu_ctx = GpuContext::new(0).unwrap();
        let gpu_result = GpuSolver::new(&gpu_ctx, &flat_tree)
            .unwrap()
            .solve(iterations, None)
            .unwrap();
        let gpu_strategy = &gpu_result.strategy;

        let gpu_num_hands = flat_tree.num_hands;
        let max_actions = flat_tree.max_actions();
        let root_n_actions = flat_tree.infoset_num_actions[0] as usize;

        eprintln!(
            "CPU: {} actions, {} hands | GPU: {} actions, {} hands (max={})",
            cpu_num_actions, cpu_num_hands, root_n_actions, gpu_num_hands, max_actions
        );
        eprintln!("Actions: {:?}", cpu_actions);

        // --- Print per-hand strategy comparison ---
        eprintln!("\n{:<20} | {:>30} | {:>30}", "Hand", "CPU strategy", "GPU strategy");
        eprintln!("{:-<85}", "");

        let num_compare = cpu_num_hands.min(gpu_num_hands);
        let mut max_abs_diff = 0.0f32;
        let mut dominant_agree = 0u32;
        let mut dominant_total = 0u32;

        for h in 0..num_compare {
            // Format hand name
            let (c1, c2) = cpu_cards[h];
            let hand_name = format!("{}{}", card_name(c1), card_name(c2));

            // CPU strategy for this hand
            let mut cpu_probs = Vec::new();
            for a in 0..cpu_num_actions {
                cpu_probs.push(cpu_strategy[a * cpu_num_hands + h]);
            }

            // GPU strategy for this hand (root = infoset 0)
            let mut gpu_probs = Vec::new();
            for a in 0..root_n_actions {
                let idx = (0 * max_actions + a) * gpu_num_hands + h;
                gpu_probs.push(gpu_strategy[idx]);
            }

            // Format probabilities
            let cpu_str: Vec<String> = cpu_probs.iter().map(|p| format!("{:.3}", p)).collect();
            let gpu_str: Vec<String> = gpu_probs.iter().map(|p| format!("{:.3}", p)).collect();

            eprintln!(
                "{:<20} | {:>30} | {:>30}",
                hand_name,
                cpu_str.join(" "),
                gpu_str.join(" ")
            );

            // Track max absolute difference
            let min_actions = cpu_probs.len().min(gpu_probs.len());
            for a in 0..min_actions {
                let diff = (cpu_probs[a] - gpu_probs[a]).abs();
                if diff > max_abs_diff {
                    max_abs_diff = diff;
                }
            }

            // Check dominant action agreement
            if min_actions > 0 {
                let cpu_dominant = cpu_probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                let gpu_dominant = gpu_probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                dominant_total += 1;
                if cpu_dominant == gpu_dominant {
                    dominant_agree += 1;
                }
            }
        }

        eprintln!("\nMax absolute difference: {:.6}", max_abs_diff);
        eprintln!(
            "Dominant action agreement: {}/{} ({:.1}%)",
            dominant_agree,
            dominant_total,
            if dominant_total > 0 {
                dominant_agree as f64 / dominant_total as f64 * 100.0
            } else {
                0.0
            }
        );
    }
}

/// Convert a card byte to a human-readable name like "As", "Kh", etc.
fn card_name(card: u8) -> String {
    let rank = card / 4;
    let suit = card % 4;
    let rank_ch = match rank {
        0 => '2',
        1 => '3',
        2 => '4',
        3 => '5',
        4 => '6',
        5 => '7',
        6 => '8',
        7 => '9',
        8 => 'T',
        9 => 'J',
        10 => 'Q',
        11 => 'K',
        12 => 'A',
        _ => '?',
    };
    let suit_ch = match suit {
        0 => 'c',
        1 => 'd',
        2 => 'h',
        3 => 's',
        _ => '?',
    };
    format!("{}{}", rank_ch, suit_ch)
}
