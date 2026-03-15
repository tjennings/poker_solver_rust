//! Large-tree GPU scaling benchmarks.
//!
//! Exercises the GPU solver on progressively larger trees (more hand combos,
//! more bet sizes) to find the crossover point where GPU starts outperforming
//! CPU. Each position increases tree width via wider ranges and more bet sizes.
//!
//! Run with:
//!   cargo test -p poker-solver-gpu --features cuda --release --test bench_large_trees -- --ignored --nocapture

#![cfg(feature = "cuda")]

use poker_solver_gpu::gpu::GpuContext;
use poker_solver_gpu::solver::GpuSolver;
use poker_solver_gpu::tree::FlatTree;
use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
use range_solver::bet_size::BetSizeOptions;
use range_solver::card::{card_from_str, flop_from_str};
use range_solver::range::Range;
use range_solver::{solve, CardConfig, PostFlopGame};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct BenchPosition {
    name: &'static str,
    oop_range: &'static str,
    ip_range: &'static str,
    board: &'static str, // "Fc Fd Fh Tc Rc" — 5 cards: flop[3] turn river
    pot: i32,
    stack: i32,
    oop_bet: &'static str,
    ip_bet: &'static str,
    iterations: u32,
}

/// Parse a 5-card board string like "7c 2d 5s Kh 9c" into (flop, turn, river).
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

/// Build a river PostFlopGame from benchmark position parameters.
fn build_game(pos: &BenchPosition) -> PostFlopGame {
    let (flop_str, turn_str, river_str) = parse_board(pos.board);
    let oop_range: Range = pos.oop_range.parse().unwrap();
    let ip_range: Range = pos.ip_range.parse().unwrap();
    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: flop_from_str(flop_str).unwrap(),
        turn: card_from_str(turn_str).unwrap(),
        river: card_from_str(river_str).unwrap(),
    };
    let oop_sizes = BetSizeOptions::try_from((pos.oop_bet, "")).unwrap();
    let ip_sizes = BetSizeOptions::try_from((pos.ip_bet, "")).unwrap();
    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: pos.pot,
        effective_stack: pos.stack,
        river_bet_sizes: [oop_sizes, ip_sizes],
        ..Default::default()
    };
    let tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
    game.allocate_memory(false);
    game
}

// ---------------------------------------------------------------------------
// Benchmark test
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn bench_gpu_scaling() {
    let gpu = GpuContext::new(0).unwrap();

    let positions = vec![
        // Size 1: Tiny (baseline, ~15 nodes)
        BenchPosition {
            name: "tiny",
            oop_range: "AA,KK",
            ip_range: "QQ,JJ",
            board: "2c 3d 4h 5s 6c",
            pot: 100,
            stack: 100,
            oop_bet: "50%,a",
            ip_bet: "50%,a",
            iterations: 500,
        },
        // Size 2: Small (~30-50 nodes)
        BenchPosition {
            name: "small",
            oop_range: "TT+,AQs+",
            ip_range: "99+,AJs+",
            board: "7c 2d 9s Kh 3c",
            pot: 100,
            stack: 100,
            oop_bet: "33%,66%,a",
            ip_bet: "33%,66%,a",
            iterations: 500,
        },
        // Size 3: Medium (~100-200 nodes)
        BenchPosition {
            name: "medium",
            oop_range: "22+,A2s+,KTs+",
            ip_range: "22+,A5s+,KJs+",
            board: "8c 4d 2s Jh 9c",
            pot: 100,
            stack: 100,
            oop_bet: "33%,66%,100%,a",
            ip_bet: "33%,66%,100%,a",
            iterations: 500,
        },
        // Size 4: Large (~500+ nodes)
        BenchPosition {
            name: "large",
            oop_range: "22+,A2s+,K2s+,Q8s+,J9s+,T9s,A2o+,K9o+,QTo+,JTo",
            ip_range: "22+,A2s+,K5s+,Q9s+,JTs,A5o+,KTo+,QJo",
            board: "Qs 7d 3c 5h Kc",
            pot: 100,
            stack: 100,
            oop_bet: "25%,50%,75%,100%,150%,a",
            ip_bet: "25%,50%,75%,100%,150%,a",
            iterations: 500,
        },
        // Size 5: Very large (widest possible tree)
        BenchPosition {
            name: "xlarge",
            oop_range: "22+,A2s+,K2s+,Q8s+,J9s+,T9s,A2o+,K9o+,QTo+,JTo",
            ip_range: "22+,A2s+,K5s+,Q9s+,JTs,A5o+,KTo+,QJo",
            board: "Qs 7d 3c 5h Kc",
            pot: 100,
            stack: 100,
            oop_bet: "25%,33%,50%,66%,75%,100%,150%,a",
            ip_bet: "25%,33%,50%,66%,75%,100%,150%,a",
            iterations: 500,
        },
    ];

    println!();
    println!(
        "{:<12} {:>8} {:>8} {:>6} {:>5} {:>10} {:>10} {:>8}",
        "Name", "Nodes", "Infosets", "Hands", "Iters", "GPU (ms)", "CPU (ms)", "Speedup"
    );
    println!("{}", "=".repeat(78));

    for pos in &positions {
        // Build game + FlatTree for GPU
        let mut gpu_game = build_game(pos);
        let flat_tree = FlatTree::from_postflop_game(&mut gpu_game);

        let num_nodes = flat_tree.num_nodes();
        let num_infosets = flat_tree.num_infosets;
        let num_hands = flat_tree.num_hands;

        // Determine iterations: cap at configured value, but reduce if tree
        // is very large to keep total benchmark time reasonable (<30s per pos).
        let iterations = pos.iterations;

        // ---- GPU solve ----
        let mut gpu_solver = GpuSolver::new(&gpu, &flat_tree).unwrap();
        let gpu_start = Instant::now();
        let gpu_result = gpu_solver.solve(iterations, None).unwrap();
        let gpu_time = gpu_start.elapsed();

        // ---- CPU solve ----
        let mut cpu_game = build_game(pos);
        let cpu_start = Instant::now();
        let _cpu_exploit = solve(&mut cpu_game, iterations, 0.0, false);
        let cpu_time = cpu_start.elapsed();

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!(
            "{:<12} {:>8} {:>8} {:>6} {:>5} {:>10.1} {:>10.1} {:>7.2}x",
            pos.name,
            num_nodes,
            num_infosets,
            num_hands,
            iterations,
            gpu_time.as_secs_f64() * 1000.0,
            cpu_time.as_secs_f64() * 1000.0,
            speedup
        );

        // Sanity check: GPU strategy should be valid
        let max_actions = flat_tree.max_actions();
        for iset in 0..num_infosets {
            let n_actions = flat_tree.infoset_num_actions[iset] as usize;
            for h in 0..num_hands {
                let mut total = 0.0f32;
                for a in 0..n_actions {
                    let idx = (iset * max_actions + a) * num_hands + h;
                    let prob = gpu_result.strategy[idx];
                    assert!(
                        prob.is_finite(),
                        "NaN/Inf in GPU strategy: pos={}, iset={iset}, action={a}, hand={h}",
                        pos.name
                    );
                    total += prob;
                }
                assert!(
                    (total - 1.0).abs() < 1e-2,
                    "GPU strategy doesn't sum to 1.0: pos={}, iset={iset}, hand={h}, sum={total}",
                    pos.name
                );
            }
        }
    }

    println!("{}", "=".repeat(78));
    println!("GPU advantage expected at larger trees where kernel parallelism");
    println!("amortizes launch overhead. Speedup > 1.0 = GPU wins.");
}
