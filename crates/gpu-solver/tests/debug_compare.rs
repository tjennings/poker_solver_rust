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
use range_solver::{solve, solve_step, CardConfig, PostFlopGame};
use std::time::Instant;

/// Build a river game with configurable pot, stack, and bet sizes.
fn build_game(
    oop_range_str: &str,
    ip_range_str: &str,
    flop: &str,
    turn: &str,
    river: &str,
    starting_pot: i32,
    effective_stack: i32,
    oop_bet: &str,
    ip_bet: &str,
) -> PostFlopGame {
    let oop_range: Range = oop_range_str.parse().unwrap();
    let ip_range: Range = ip_range_str.parse().unwrap();
    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: flop_from_str(flop).unwrap(),
        turn: card_from_str(turn).unwrap(),
        river: card_from_str(river).unwrap(),
    };
    let oop_sizes = BetSizeOptions::try_from((oop_bet, "")).unwrap();
    let ip_sizes = BetSizeOptions::try_from((ip_bet, "")).unwrap();
    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot,
        effective_stack,
        river_bet_sizes: [oop_sizes, ip_sizes],
        ..Default::default()
    };
    let tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
    game.allocate_memory(false);
    game
}

/// Parse a board string like "7c 2d 5s Kh 9c" into (flop, turn, river).
fn parse_board(board: &str) -> (&str, &str, &str) {
    let parts: Vec<&str> = board.split_whitespace().collect();
    assert_eq!(parts.len(), 5, "Board must have exactly 5 cards");
    let flop_end = board
        .match_indices(' ')
        .nth(2)
        .map(|(i, _)| i)
        .unwrap();
    let flop = &board[..flop_end];
    let turn = parts[3];
    let river = parts[4];
    (flop, turn, river)
}

struct Position {
    name: &'static str,
    oop_range: &'static str,
    ip_range: &'static str,
    board: &'static str, // "Fc Fd Fh Tc Rc" — 5 cards: flop[3] turn river
    pot: i32,
    stack: i32,
    oop_bet: &'static str,
    ip_bet: &'static str,
}

#[test]
fn debug_compare_gpu_vs_cpu() {
    let positions = [
        // --- Original 3 positions ---
        Position {
            name: "Pos 1: QQ+,AKs vs QQ-JJ,AQs,AJs — Qs Jh 2c 8d 3s",
            oop_range: "QQ+,AKs",
            ip_range: "QQ-JJ,AQs,AJs",
            board: "Qs Jh 2c 8d 3s",
            pot: 100,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        Position {
            name: "Pos 2: AA,KK vs 22+,AK — Kd 7h 3c 9s 2d",
            oop_range: "AA,KK",
            ip_range: "22+,AK",
            board: "Kd 7h 3c 9s 2d",
            pot: 200,
            stack: 150,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        Position {
            name: "Pos 3: TT+,AQ+ vs 77+,AJ+ — Tc 8s 4h Jd 2c",
            oop_range: "TT+,AQ+",
            ip_range: "77+,AJ+",
            board: "Tc 8s 4h Jd 2c",
            pot: 80,
            stack: 200,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // --- 27 new positions ---
        // 4. BTN vs BB SRP, dry board
        Position {
            name: "Pos 4: dry board — 7c 2d 5s Kh 9d",
            oop_range: "22+,A2s+",
            ip_range: "QQ+,AK",
            board: "7c 2d 5s Kh 9d",
            pot: 60,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 5. 3bet pot, paired board
        Position {
            name: "Pos 5: 3bet paired — Jd Jc 3h 8s Tc",
            oop_range: "QQ+,AKs",
            ip_range: "TT+,AQs+",
            board: "Jd Jc 3h 8s Tc",
            pot: 180,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 6. Monotone flop (4 hearts on board)
        Position {
            name: "Pos 6: monotone — 9h 6h 2h Th 4c",
            oop_range: "AA,KK,AKs",
            ip_range: "TT+,AJs+",
            board: "9h 6h 2h Th 4c",
            pot: 100,
            stack: 150,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 7. Broadway heavy
        Position {
            name: "Pos 7: broadway — Kc Qd Ts 3h 7d",
            oop_range: "TT+,AJ+",
            ip_range: "99+,AT+",
            board: "Kc Qd Ts 3h 7d",
            pot: 100,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 8. Low board
        Position {
            name: "Pos 8: low board — 4d 3c 2s 8h 6c",
            oop_range: "AA-TT",
            ip_range: "77+,A9s+",
            board: "4d 3c 2s 8h 6c",
            pot: 80,
            stack: 120,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 9. Shallow stack
        Position {
            name: "Pos 9: shallow — As Kd 7c 4h 2d",
            oop_range: "JJ+,AK",
            ip_range: "TT+,AQ+",
            board: "As Kd 7c 4h 2d",
            pot: 100,
            stack: 50,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 10. Deep stack
        Position {
            name: "Pos 10: deep — 8c 5d 3h Jc 9s",
            oop_range: "QQ+,AKs",
            ip_range: "JJ-99,AQs",
            board: "8c 5d 3h Jc 9s",
            pot: 60,
            stack: 300,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 11. Paired board 2
        Position {
            name: "Pos 11: paired 2 — 6c 6d Ks Ah 3s",
            oop_range: "AA,KK",
            ip_range: "QQ,JJ,TT",
            board: "6c 6d Ks Ah 3s",
            pot: 100,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 12. Connected board
        Position {
            name: "Pos 12: connected — 9c 8d 7s Td 2h",
            oop_range: "TT+,AQ+",
            ip_range: "88+,AJ+",
            board: "9c 8d 7s Td 2h",
            pot: 120,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 13. Ace-high
        Position {
            name: "Pos 13: ace-high — Ad 9c 5h 3s 7c",
            oop_range: "KK,QQ",
            ip_range: "AA,AKs",
            board: "Ad 9c 5h 3s 7c",
            pot: 100,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 14. Two-pair board
        Position {
            name: "Pos 14: two-pair board — 5c 5d 9s 9h Kd",
            oop_range: "JJ+,AK",
            ip_range: "TT+,AQ",
            board: "5c 5d 9s 9h Kd",
            pot: 100,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 15. Flush possible (3 spades)
        Position {
            name: "Pos 15: flush possible — 7s 4s 2c Js 8d",
            oop_range: "AA,KK,AKs",
            ip_range: "QQ-TT,AQs",
            board: "7s 4s 2c Js 8d",
            pot: 80,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 16. Wide vs narrow (OOP wide, IP narrow)
        Position {
            name: "Pos 16: wide vs narrow — 6c 3d 8h Ts 2s",
            oop_range: "22+,A2s+,K9s+",
            ip_range: "AA",
            board: "6c 3d 8h Ts 2s",
            pot: 100,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 17. Narrow vs wide (OOP narrow, IP wide)
        Position {
            name: "Pos 17: narrow vs wide — Qc 7d 4s 9h 3d",
            oop_range: "AA",
            ip_range: "22+,A5s+,KTs+",
            board: "Qc 7d 4s 9h 3d",
            pot: 100,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 18. High card board
        Position {
            name: "Pos 18: high card board — Ac Kd Qh 5s 3c",
            oop_range: "TT+,AJ+",
            ip_range: "99+,AT+",
            board: "Ac Kd Qh 5s 3c",
            pot: 150,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 19. Gutshot board
        Position {
            name: "Pos 19: gutshot — 9c 7d 3s Th 6h",
            oop_range: "JJ+,AK",
            ip_range: "TT+,AQ",
            board: "9c 7d 3s Th 6h",
            pot: 100,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 20. Overcard river
        Position {
            name: "Pos 20: overcard river — 8c 5d 2h 4s Ad",
            oop_range: "QQ,JJ,TT",
            ip_range: "99-77,ATs+",
            board: "8c 5d 2h 4s Ad",
            pot: 100,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 21. Blank river
        Position {
            name: "Pos 21: blank river — Td 7c 3s 5h 2c",
            oop_range: "AA,KK,QQ",
            ip_range: "JJ,TT,99",
            board: "Td 7c 3s 5h 2c",
            pot: 100,
            stack: 150,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 22. Trips board
        Position {
            name: "Pos 22: trips board — 4c 4d 4h 9s Kd",
            oop_range: "KK,QQ,AK",
            ip_range: "JJ,TT,AQ",
            board: "4c 4d 4h 9s Kd",
            pot: 100,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 23. Small pot, deep effective
        Position {
            name: "Pos 23: small pot — Jc 8d 2s 5h Kc",
            oop_range: "TT+,AQ+",
            ip_range: "99+,AJ+",
            board: "Jc 8d 2s 5h Kc",
            pot: 40,
            stack: 200,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 24. Large pot
        Position {
            name: "Pos 24: large pot — 7c 3d 9h 2s Td",
            oop_range: "QQ+,AKs",
            ip_range: "JJ+,AQs",
            board: "7c 3d 9h 2s Td",
            pot: 300,
            stack: 150,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 25. SB vs BB
        Position {
            name: "Pos 25: SB vs BB — 6d 4c 9h Ks 2h",
            oop_range: "33+,A2s+,K8s+",
            ip_range: "55+,A7s+,KTs+",
            board: "6d 4c 9h Ks 2h",
            pot: 50,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 26. Squeeze pot (big pot, shallow SPR)
        Position {
            name: "Pos 26: squeeze pot — 8c 7d 3h As 5d",
            oop_range: "KK+,AKs",
            ip_range: "QQ-TT,AQs",
            board: "8c 7d 3h As 5d",
            pot: 250,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 27. Limped pot (tiny pot, deep)
        Position {
            name: "Pos 27: limped pot — Ts 6c 3d 8h Kd",
            oop_range: "22+,A2s+",
            ip_range: "22+,A2s+",
            board: "Ts 6c 3d 8h Kd",
            pot: 20,
            stack: 200,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 28. 4bet pot (huge pot, shallow SPR)
        Position {
            name: "Pos 28: 4bet pot — 7c 4d 2s Jh 9d",
            oop_range: "KK+,AKs",
            ip_range: "QQ+,AKs",
            board: "7c 4d 2s Jh 9d",
            pot: 400,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 29. Medium-wide ranges
        Position {
            name: "Pos 29: medium ranges — 5c 3d 2h 9s Qd",
            oop_range: "TT+,AJs+",
            ip_range: "88+,ATs+",
            board: "5c 3d 2h 9s Qd",
            pot: 100,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
        // 30. Very dry, pocket pair matchup
        Position {
            name: "Pos 30: very dry — 2c 3d 7s Kh 4s",
            oop_range: "AA-JJ",
            ip_range: "TT-77",
            board: "2c 3d 7s Kh 4s",
            pot: 100,
            stack: 100,
            oop_bet: "50%, a",
            ip_bet: "50%, a",
        },
    ];

    // Track per-position results for the summary.
    struct PositionResult {
        name: String,
        num_hands: usize,
        tree_nodes: usize,
        iterations: u32,
        gpu_time_ms: f64,
        cpu_time_ms: f64,
        max_diff: f32,
        dominant_agree: u32,
        dominant_total: u32,
        passed: bool,
    }
    let mut results: Vec<PositionResult> = Vec::new();

    for (pos_idx, pos) in positions.iter().enumerate() {
        eprintln!("\n{}", "=".repeat(80));
        eprintln!("[{}/{}] {}", pos_idx + 1, positions.len(), pos.name);
        eprintln!("{}", "=".repeat(80));

        let (flop, turn, river) = parse_board(pos.board);

        // --- Build games for both solvers (not timed) ---
        let mut cpu_game = build_game(
            pos.oop_range,
            pos.ip_range,
            flop,
            turn,
            river,
            pos.pot,
            pos.stack,
            pos.oop_bet,
            pos.ip_bet,
        );
        let mut gpu_game = build_game(
            pos.oop_range,
            pos.ip_range,
            flop,
            turn,
            river,
            pos.pot,
            pos.stack,
            pos.oop_bet,
            pos.ip_bet,
        );

        // Determine hand counts for iteration heuristic
        let oop_hands = cpu_game.num_private_hands(0);
        let ip_hands = cpu_game.num_private_hands(1);
        let max_hands = oop_hands.max(ip_hands);

        // Bump iterations for wide ranges: >80 hands => 2000, else 1000
        let iterations: u32 = if max_hands > 80 { 2000 } else { 1000 };

        // --- CPU solve (timed) ---
        let cpu_start = Instant::now();
        let cpu_exploit = solve(&mut cpu_game, iterations, 0.0, false);
        let cpu_time = cpu_start.elapsed();
        let cpu_time_ms = cpu_time.as_secs_f64() * 1000.0;
        eprintln!("CPU exploitability: {:.6} ({:.1} ms, {} iters)", cpu_exploit, cpu_time_ms, iterations);

        cpu_game.back_to_root();
        let cpu_strategy = cpu_game.strategy();
        let cpu_player = cpu_game.current_player();
        let cpu_num_hands = cpu_game.num_private_hands(cpu_player);
        let cpu_actions = cpu_game.available_actions();
        let cpu_num_actions = cpu_actions.len();

        // Get private card names for the acting player
        let cpu_cards = cpu_game.private_cards(cpu_player);

        // --- GPU solve (setup not timed, only solve step) ---
        let flat_tree = FlatTree::from_postflop_game(&mut gpu_game);
        let tree_nodes = flat_tree.num_nodes();
        let gpu_ctx = GpuContext::new(0).unwrap();
        let mut gpu_solver = GpuSolver::new(&gpu_ctx, &flat_tree).unwrap();

        let gpu_start = Instant::now();
        let gpu_result = gpu_solver.solve(iterations, None).unwrap();
        let gpu_time = gpu_start.elapsed();
        let gpu_time_ms = gpu_time.as_secs_f64() * 1000.0;

        let speedup = cpu_time_ms / gpu_time_ms;
        eprintln!("GPU solve: {:.1} ms | CPU solve: {:.1} ms | Speedup: {:.2}x", gpu_time_ms, cpu_time_ms, speedup);

        let gpu_strategy = &gpu_result.strategy;

        let gpu_num_hands = flat_tree.num_hands;
        let max_actions = flat_tree.max_actions();
        let root_n_actions = flat_tree.infoset_num_actions[0] as usize;

        eprintln!(
            "CPU: {} actions, {} hands | GPU: {} actions, {} hands (max={}) | Tree nodes: {}",
            cpu_num_actions, cpu_num_hands, root_n_actions, gpu_num_hands, max_actions, tree_nodes
        );
        eprintln!("Actions: {:?}", cpu_actions);

        // --- Print per-hand strategy comparison (first 15 hands to keep output manageable) ---
        eprintln!(
            "\n{:<20} | {:>30} | {:>30}",
            "Hand", "CPU strategy", "GPU strategy"
        );
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

            // Print first 15 hands for readability
            if h < 15 {
                let cpu_str: Vec<String> =
                    cpu_probs.iter().map(|p| format!("{:.3}", p)).collect();
                let gpu_str: Vec<String> =
                    gpu_probs.iter().map(|p| format!("{:.3}", p)).collect();

                eprintln!(
                    "{:<20} | {:>30} | {:>30}",
                    hand_name,
                    cpu_str.join(" "),
                    gpu_str.join(" ")
                );
            }

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

        if num_compare > 15 {
            eprintln!("... ({} more hands omitted)", num_compare - 15);
        }

        let dominant_pct = if dominant_total > 0 {
            dominant_agree as f64 / dominant_total as f64 * 100.0
        } else {
            0.0
        };

        eprintln!("\nMax absolute difference: {:.6}", max_abs_diff);
        eprintln!(
            "Dominant action agreement: {}/{} ({:.1}%)",
            dominant_agree, dominant_total, dominant_pct
        );

        let passed = max_abs_diff < 0.01 && dominant_pct >= 95.0;
        eprintln!(
            "Result: {}",
            if passed { "PASS" } else { "FAIL" }
        );

        results.push(PositionResult {
            name: pos.name.to_string(),
            num_hands: max_hands,
            tree_nodes,
            iterations,
            gpu_time_ms,
            cpu_time_ms,
            max_diff: max_abs_diff,
            dominant_agree,
            dominant_total,
            passed,
        });
    }

    // === Performance Summary Table ===
    eprintln!("\n{}", "=".repeat(140));
    eprintln!(
        "PERFORMANCE SUMMARY: GPU vs CPU comparison across {} positions",
        results.len()
    );
    eprintln!("{}", "=".repeat(140));
    eprintln!(
        "{:<45} {:>6} {:>7} {:>6} {:>10} {:>10} {:>9} {:>10} {:>6}",
        "Position", "Hands", "Nodes", "Iters", "GPU (ms)", "CPU (ms)", "Speedup", "Max Diff", "Result"
    );
    eprintln!("{:-<115}", "");

    let mut worst_diff = 0.0f32;
    let mut num_passed = 0u32;
    let mut total_gpu_ms = 0.0f64;
    let mut total_cpu_ms = 0.0f64;
    let mut speedups: Vec<(f64, String)> = Vec::new();

    for r in &results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        let speedup = r.cpu_time_ms / r.gpu_time_ms;
        eprintln!(
            "{:<45} {:>6} {:>7} {:>6} {:>10.1} {:>10.1} {:>8.2}x {:>10.6} {:>6}",
            r.name, r.num_hands, r.tree_nodes, r.iterations,
            r.gpu_time_ms, r.cpu_time_ms, speedup, r.max_diff, status
        );
        if r.max_diff > worst_diff {
            worst_diff = r.max_diff;
        }
        if r.passed {
            num_passed += 1;
        }
        total_gpu_ms += r.gpu_time_ms;
        total_cpu_ms += r.cpu_time_ms;
        speedups.push((speedup, r.name.clone()));
    }

    eprintln!("{:-<115}", "");
    eprintln!(
        "Positions passed: {}/{} | Worst max diff: {:.6}",
        num_passed,
        results.len(),
        worst_diff
    );

    // === Aggregate Performance Stats ===
    eprintln!("\n{}", "=".repeat(80));
    eprintln!("AGGREGATE PERFORMANCE STATS");
    eprintln!("{}", "=".repeat(80));

    let avg_speedup = if !speedups.is_empty() {
        speedups.iter().map(|(s, _)| s).sum::<f64>() / speedups.len() as f64
    } else {
        0.0
    };
    let overall_speedup = total_cpu_ms / total_gpu_ms;

    eprintln!("Average GPU speedup:   {:.2}x", avg_speedup);
    eprintln!("Overall GPU speedup:   {:.2}x (total GPU: {:.1} ms, total CPU: {:.1} ms)", overall_speedup, total_gpu_ms, total_cpu_ms);

    if let Some((fastest_speed, fastest_name)) = speedups.iter().max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()) {
        eprintln!("Fastest GPU speedup:   {:.2}x — {}", fastest_speed, fastest_name);
    }
    if let Some((slowest_speed, slowest_name)) = speedups.iter().min_by(|a, b| a.0.partial_cmp(&b.0).unwrap()) {
        eprintln!("Slowest GPU speedup:   {:.2}x — {}", slowest_speed, slowest_name);
    }

    // Assert all positions pass
    for r in &results {
        assert!(
            r.max_diff < 0.01,
            "Position '{}' exceeded max diff threshold: {:.6} >= 0.01",
            r.name,
            r.max_diff
        );
        let pct = if r.dominant_total > 0 {
            r.dominant_agree as f64 / r.dominant_total as f64 * 100.0
        } else {
            100.0 // no hands = vacuously true
        };
        assert!(
            pct >= 95.0,
            "Position '{}' dominant action agreement too low: {:.1}% < 95%",
            r.name,
            pct
        );
    }
}

/// Diagnostic: Run 1 iteration and check terminal fold values match.
#[test]
fn debug_terminal_values() {
    // Use position 2: simple case, OOP has 9 hands
    let mut game = build_game(
        "AA,KK", "22+,AK", "Kd 7h 3c", "9s", "2d", 200, 150, "50%, a", "50%, a",
    );
    let flat = FlatTree::from_postflop_game(&mut game);

    eprintln!("num_hands_oop={}, num_hands_ip={}, num_hands={}",
        flat.num_hands_oop, flat.num_hands_ip, flat.num_hands);
    eprintln!("num_fold_terminals: {}",
        flat.terminal_indices.iter()
            .filter(|&&i| flat.node_types[i as usize] == poker_solver_gpu::tree::NodeType::TerminalFold)
            .count());
    eprintln!("num_showdown_terminals: {}",
        flat.terminal_indices.iter()
            .filter(|&&i| flat.node_types[i as usize] == poker_solver_gpu::tree::NodeType::TerminalShowdown)
            .count());

    // Print tree structure
    for (i, nt) in flat.node_types.iter().enumerate() {
        let parent = flat.parent_nodes[i];
        let parent_action = flat.parent_actions[i];
        eprintln!("Node {}: {:?}, pot={:.0}, parent={}, action={}",
            i, nt, flat.pots[i],
            if parent == u32::MAX { "root".to_string() } else { parent.to_string() },
            if parent_action == u32::MAX { "N/A".to_string() } else { parent_action.to_string() });
    }

    // Print fold payoffs
    eprintln!("\nFold payoffs:");
    for (term_i, &node_id) in flat.terminal_indices.iter().enumerate() {
        if flat.node_types[node_id as usize] == poker_solver_gpu::tree::NodeType::TerminalFold {
            let payoff = &flat.fold_payoffs[term_i];
            eprintln!("  node={}: amount_win={:.6}, amount_lose={:.6}, fold_player={}",
                node_id, payoff[0], payoff[1], payoff[2] as u32);
        }
    }

    // Print showdown info
    eprintln!("\nShowdown payoffs:");
    for (term_i, &node_id) in flat.terminal_indices.iter().enumerate() {
        if flat.node_types[node_id as usize] == poker_solver_gpu::tree::NodeType::TerminalShowdown {
            let eq_id = flat.showdown_equity_ids[term_i] as usize;
            let eq = &flat.equity_tables[eq_id];
            eprintln!("  node={}: amount_win={:.6}, amount_lose={:.6}",
                node_id, eq[0], eq[1]);
        }
    }

    // Print some hand strengths
    eprintln!("\nOOP hand strengths (first 9):");
    for (i, &s) in flat.hand_strengths_oop.iter().take(9).enumerate() {
        let cards = game.private_cards(0);
        let (c1, c2) = cards[i];
        eprintln!("  hand {}: {} {} -> strength {}", i, card_name(c1), card_name(c2), s);
    }
    eprintln!("IP hand strengths (first 10):");
    for (i, &s) in flat.hand_strengths_ip.iter().take(10).enumerate() {
        let cards = game.private_cards(1);
        let (c1, c2) = cards[i];
        eprintln!("  hand {}: {} {} -> strength {}", i, card_name(c1), card_name(c2), s);
    }

    // Print valid_matchups for OOP hand 0 vs first 10 IP hands
    eprintln!("\nValid matchups (OOP hand 0 vs IP hands 0..10):");
    for ip_h in 0..10.min(flat.num_hands_ip) {
        let cards_ip = game.private_cards(1);
        let (c1, c2) = cards_ip[ip_h];
        let valid = flat.valid_matchups_oop[0 * flat.num_hands + ip_h];
        eprintln!("  OOP[0] vs IP[{}] ({} {}): valid={}", ip_h, card_name(c1), card_name(c2), valid);
    }

    // Print initial reaches
    eprintln!("\nInitial reach OOP (first 9):");
    for i in 0..9 {
        eprintln!("  hand {}: {}", i, flat.initial_reach_oop[i]);
    }
    eprintln!("Initial reach IP (first 10):");
    for i in 0..10 {
        eprintln!("  hand {}: {}", i, flat.initial_reach_ip[i]);
    }

    // Run GPU solver for 1 iteration and check cfvalues at root
    let gpu_ctx = GpuContext::new(0).unwrap();
    let gpu_result = GpuSolver::new(&gpu_ctx, &flat)
        .unwrap()
        .solve(1, None)
        .unwrap();

    eprintln!("\nGPU strategy after 1 iteration (root, OOP hands 0..9):");
    let _max_actions = flat.max_actions();
    let root_n_actions = flat.infoset_num_actions[0] as usize;
    for h in 0..9 {
        let mut probs = Vec::new();
        for a in 0..root_n_actions {
            let idx = a * flat.num_hands + h;
            probs.push(gpu_result.strategy[idx]);
        }
        let cards = game.private_cards(0);
        let (c1, c2) = cards[h];
        eprintln!("  {} {}: {:?}", card_name(c1), card_name(c2), probs);
    }
}

/// Minimal iteration trace: runs 1 iteration on both GPU and CPU,
/// dumps all intermediate values, and identifies the first divergence.
#[test]
fn debug_minimal_iteration_trace() {
    eprintln!("\n{}", "=".repeat(80));
    eprintln!("MINIMAL ITERATION TRACE: 1 iteration, narrow ranges");
    eprintln!("{}", "=".repeat(80));

    // Use a very simple setup: narrow ranges on a board where both have few combos.
    // OOP: KK (6 combos minus board blockers)
    // IP: AA (6 combos minus board blockers)
    // Board: 2c 3d 4h 5s 6c — no blockers for either range
    // Wait, 2c and 6c share suit — let's use a board that doesn't block KK or AA.
    // Board: 2c 3d 4h 5s 7h — KK has 6 combos, AA has 6 combos
    let mut cpu_game = build_game(
        "KK",
        "AA",
        "2c 3d 4h",
        "5s",
        "7h",
        100,
        100,
        "50%, a",
        "50%, a",
    );

    let num_hands_oop = cpu_game.num_private_hands(0);
    let num_hands_ip = cpu_game.num_private_hands(1);
    eprintln!("OOP hands: {}, IP hands: {}", num_hands_oop, num_hands_ip);

    // Print the hand cards
    let oop_cards = cpu_game.private_cards(0);
    let ip_cards = cpu_game.private_cards(1);
    eprintln!("OOP hands:");
    for (i, &(c1, c2)) in oop_cards.iter().enumerate() {
        eprintln!("  {}: {} {}", i, card_name(c1), card_name(c2));
    }
    eprintln!("IP hands:");
    for (i, &(c1, c2)) in ip_cards.iter().enumerate() {
        eprintln!("  {}: {} {}", i, card_name(c1), card_name(c2));
    }

    // Run 1 iteration on CPU
    solve_step(&cpu_game, 0);

    // Read CPU strategy at root after 1 iteration
    cpu_game.back_to_root();
    let cpu_root_player = cpu_game.current_player();
    let cpu_root_actions = cpu_game.available_actions();
    eprintln!("\nCPU root player: {} ({})", cpu_root_player,
        if cpu_root_player == 0 { "OOP" } else { "IP" });
    eprintln!("CPU root actions: {:?}", cpu_root_actions);

    // Walk the CPU tree and print strategy at each decision node
    eprintln!("\n--- CPU strategy at each node after 1 iteration ---");
    print_cpu_tree_strategies(&mut cpu_game, &[], 0);

    // Build FlatTree and run GPU
    let mut gpu_game = build_game(
        "KK", "AA", "2c 3d 4h", "5s", "7h", 100, 100, "50%, a", "50%, a",
    );
    let flat = FlatTree::from_postflop_game(&mut gpu_game);

    eprintln!("\n--- FlatTree structure ---");
    eprintln!("num_nodes={}, num_infosets={}, num_hands={}, max_actions={}",
        flat.num_nodes(), flat.num_infosets, flat.num_hands, flat.max_actions());
    for (i, nt) in flat.node_types.iter().enumerate() {
        let parent = flat.parent_nodes[i];
        let iset = flat.infoset_ids[i];
        eprintln!("  Node {}: {:?}, pot={:.0}, parent={}, iset={}, n_actions={}",
            i, nt, flat.pots[i],
            if parent == u32::MAX { "root".to_string() } else { parent.to_string() },
            if iset == u32::MAX { "N/A".to_string() } else { iset.to_string() },
            if iset != u32::MAX { flat.infoset_num_actions[iset as usize] } else { 0 }
        );
    }

    // Run GPU debug iteration
    let gpu_ctx = GpuContext::new(0).unwrap();
    let mut gpu_solver = GpuSolver::new(&gpu_ctx, &flat).unwrap();
    let debug = gpu_solver.debug_iteration().unwrap();

    let nh = flat.num_hands;
    let ma = flat.max_actions();

    // Print GPU intermediate state
    eprintln!("\n--- GPU reach after forward pass (traverser=0) ---");
    for node_id in 0..flat.num_nodes() {
        let oop_reach: Vec<f32> = (0..nh).map(|h| debug.reach_oop[node_id * nh + h]).collect();
        let ip_reach: Vec<f32> = (0..nh).map(|h| debug.reach_ip[node_id * nh + h]).collect();
        // Only print if any non-zero
        let has_data = oop_reach.iter().any(|&v| v != 0.0) || ip_reach.iter().any(|&v| v != 0.0);
        if has_data {
            eprintln!("  Node {}: OOP_reach={:?}, IP_reach={:?}", node_id,
                &oop_reach[..num_hands_oop.min(nh)],
                &ip_reach[..num_hands_ip.min(nh)]);
        }
    }

    eprintln!("\n--- GPU CFVs (OOP as traverser) ---");
    for node_id in 0..flat.num_nodes() {
        let cfvs: Vec<f32> = (0..nh).map(|h| debug.cfvalues_oop_traverser[node_id * nh + h]).collect();
        let has_data = cfvs.iter().any(|&v| v != 0.0);
        if has_data {
            eprintln!("  Node {} ({:?}): {:?}", node_id, flat.node_types[node_id],
                &cfvs[..num_hands_oop.min(nh)]);
        }
    }

    eprintln!("\n--- GPU CFVs (IP as traverser) ---");
    for node_id in 0..flat.num_nodes() {
        let cfvs: Vec<f32> = (0..nh).map(|h| debug.cfvalues_ip_traverser[node_id * nh + h]).collect();
        let has_data = cfvs.iter().any(|&v| v != 0.0);
        if has_data {
            eprintln!("  Node {} ({:?}): {:?}", node_id, flat.node_types[node_id],
                &cfvs[..num_hands_ip.min(nh)]);
        }
    }

    // Print GPU regrets
    eprintln!("\n--- GPU regrets after 1 iteration ---");
    for iset in 0..flat.num_infosets {
        let n_actions = flat.infoset_num_actions[iset] as usize;
        let _player = flat.player(
            flat.node_types.iter().enumerate()
                .find(|(_, nt)| **nt == poker_solver_gpu::tree::NodeType::DecisionOop ||
                    **nt == poker_solver_gpu::tree::NodeType::DecisionIp)
                .map(|(i, _)| i)
                .unwrap_or(0)
        );
        // Find which node this infoset belongs to
        let node_id = flat.infoset_ids.iter().position(|&id| id == iset as u32).unwrap();
        let acting_player = flat.player(node_id);
        let acting_num_hands = if acting_player == 0 { num_hands_oop } else { num_hands_ip };

        eprintln!("  Infoset {} (node {}, player={}):", iset, node_id,
            if acting_player == 0 { "OOP" } else { "IP" });
        for a in 0..n_actions {
            let regrets: Vec<f32> = (0..acting_num_hands.min(nh))
                .map(|h| debug.regrets[(iset * ma + a) * nh + h])
                .collect();
            eprintln!("    action {}: regrets={:?}", a, regrets);
        }
    }

    // Print GPU strategy (current, from regret matching)
    eprintln!("\n--- GPU strategy after 1 iteration ---");
    for iset in 0..flat.num_infosets {
        let n_actions = flat.infoset_num_actions[iset] as usize;
        let node_id = flat.infoset_ids.iter().position(|&id| id == iset as u32).unwrap();
        let acting_player = flat.player(node_id);
        let acting_num_hands = if acting_player == 0 { num_hands_oop } else { num_hands_ip };

        eprintln!("  Infoset {} (node {}, player={}):", iset, node_id,
            if acting_player == 0 { "OOP" } else { "IP" });
        for a in 0..n_actions {
            let strat: Vec<f32> = (0..acting_num_hands.min(nh))
                .map(|h| debug.strategy[(iset * ma + a) * nh + h])
                .collect();
            eprintln!("    action {}: strategy={:?}", a, strat);
        }
    }

    // Compare GPU and CPU strategies after 1 iteration at root
    eprintln!("\n--- COMPARISON: Root strategy after 1 iteration ---");
    cpu_game.back_to_root();
    // Note: CPU strategy() returns the *normalized* strategy from strategy_sum,
    // but after only 1 iteration, strategy_sum = current_strategy (gamma=0),
    // so normalized strategy_sum = regret-matched strategy = uniform.
    // What we really want to compare is the raw regrets.
    // After 1 iter, we can look at the strategy that would be used in iteration 2.
    // For that, we'd need to do regret matching on the new regrets.
    // The GPU debug_iteration returns the strategy AFTER the last regret_match
    // (which was for traverser=1's pass). But it also returns regrets.
    //
    // Better: run 2 iterations on both and compare the final strategies.
    eprintln!("(See full solve comparison in debug_compare_gpu_vs_cpu test)");

    // Also run the full solve for a few hundred iterations and compare
    eprintln!("\n--- Full solve comparison (200 iterations) ---");
    let mut cpu_game2 = build_game(
        "KK", "AA", "2c 3d 4h", "5s", "7h", 100, 100, "50%, a", "50%, a",
    );
    let cpu_exploit = solve(&mut cpu_game2, 200, 0.0, false);
    eprintln!("CPU exploitability: {:.6}", cpu_exploit);

    cpu_game2.back_to_root();
    let cpu_strat = cpu_game2.strategy();
    let cpu_pl = cpu_game2.current_player();
    let cpu_nh = cpu_game2.num_private_hands(cpu_pl);
    let cpu_na = cpu_game2.available_actions().len();

    let mut gpu_game2 = build_game(
        "KK", "AA", "2c 3d 4h", "5s", "7h", 100, 100, "50%, a", "50%, a",
    );
    let flat2 = FlatTree::from_postflop_game(&mut gpu_game2);
    let gpu_ctx2 = GpuContext::new(0).unwrap();
    let gpu_result2 = GpuSolver::new(&gpu_ctx2, &flat2)
        .unwrap()
        .solve(200, None)
        .unwrap();
    let gpu_strat = &gpu_result2.strategy;
    let gpu_nh = flat2.num_hands;
    let gpu_ma = flat2.max_actions();
    let gpu_na = flat2.infoset_num_actions[0] as usize;

    let oop_cards2 = cpu_game2.private_cards(cpu_pl);
    eprintln!("{:<20} | {:>30} | {:>30}", "Hand", "CPU", "GPU");
    eprintln!("{:-<85}", "");
    let mut max_diff = 0.0f32;
    for h in 0..cpu_nh.min(gpu_nh) {
        let (c1, c2) = oop_cards2[h];
        let hand_name = format!("{}{}", card_name(c1), card_name(c2));

        let mut cpu_probs = Vec::new();
        for a in 0..cpu_na {
            cpu_probs.push(cpu_strat[a * cpu_nh + h]);
        }
        let mut gpu_probs = Vec::new();
        for a in 0..gpu_na {
            let idx = (0 * gpu_ma + a) * gpu_nh + h;
            gpu_probs.push(gpu_strat[idx]);
        }

        let cpu_str: Vec<String> = cpu_probs.iter().map(|p| format!("{:.3}", p)).collect();
        let gpu_str: Vec<String> = gpu_probs.iter().map(|p| format!("{:.3}", p)).collect();
        eprintln!("{:<20} | {:>30} | {:>30}", hand_name, cpu_str.join(" "), gpu_str.join(" "));

        for a in 0..cpu_na.min(gpu_na) {
            let diff = (cpu_probs[a] - gpu_probs[a]).abs();
            if diff > max_diff { max_diff = diff; }
        }
    }
    eprintln!("Max difference: {:.6}", max_diff);

    // Assert the strategies are reasonably close
    assert!(
        max_diff < 0.05,
        "GPU and CPU strategies diverge too much at root: max_diff={:.6}",
        max_diff
    );
}

/// Recursively print CPU strategies at each decision node.
fn print_cpu_tree_strategies(game: &mut PostFlopGame, history: &[usize], depth: usize) {
    game.apply_history(history);
    if game.is_terminal_node() {
        return;
    }

    let player = game.current_player();
    let actions = game.available_actions();
    let n_actions = actions.len();
    let _num_hands = game.num_private_hands(player);

    let indent = "  ".repeat(depth);
    eprintln!("{}Node (history={:?}, player={}, actions={:?}):",
        indent, history,
        if player == 0 { "OOP" } else { "IP" },
        actions);

    // Print strategy — but note: after 0 iterations with solve_step(0),
    // the strategy is computed from regrets. Since all regrets are initially zero,
    // the strategy should be uniform. After 1 call to solve_step, regrets are updated.
    // The strategy() method normalizes strategy_sum, which after 1 iteration = current strategy.
    // This won't be meaningful to compare directly — we need the raw regrets.
    // Let's just print what we can.

    // Strategy is from strategy_sum normalization. After 1 iteration with gamma=0,
    // strategy_sum = just the current strategy from that iteration.
    // For the root (OOP), the strategy used was uniform (regrets were zero).
    // After the update, regrets changed. strategy_sum = uniform.
    // So game.strategy() would give us the normalized strategy_sum = uniform for all nodes.
    // This is not very useful for per-iteration comparison.

    // Instead, let's compare after many iterations.

    for ai in 0..n_actions {
        let mut child_history = history.to_vec();
        child_history.push(ai);
        print_cpu_tree_strategies(game, &child_history, depth + 1);
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
