use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Configuration for a single comparison test.
pub struct TestConfig {
    pub oop_range: String,
    pub ip_range: String,
    pub flop: [u8; 3],
    pub turn: Option<u8>,
    pub river: Option<u8>,
    pub pot: i32,
    pub stack: i32,
    pub bet_pcts: Vec<f64>,
    pub raise_pcts: Vec<f64>,
}

/// Collected results from a solve for comparison.
pub struct SolveResult {
    pub exploitability: f32,
    pub root_strategy: Vec<f32>,
    pub ev_oop: Vec<f32>,
    pub ev_ip: Vec<f32>,
    pub equity_oop: Vec<f32>,
    pub equity_ip: Vec<f32>,
}

const RANGE_POOL: &[&str] = &[
    "AA,KK,QQ,JJ,TT,99,AKs,AQs,AJs,ATs,KQs",
    "AA,KK,QQ,JJ,TT,99,88,77,AKs,AQs,AJs,ATs,A9s,KQs,KJs,QJs,JTs",
    "22+,A2s+,K9s+,Q9s+,J9s+,T9s,98s,87s,76s,ATo+,KTo+,QTo+,JTo",
    "AA,KK,QQ,AKs,AKo",
    "22+,A2s+,K2s+,Q2s+,J8s+,T8s+,97s+,87s,76s,65s,A2o+,K9o+,Q9o+,J9o+,T9o",
    "TT-66,AJs-A9s,KQs,KJs,QJs,JTs,T9s,AJo-ATo,KQo",
];

const BET_OPTIONS: &[f64] = &[0.33, 0.5, 0.67, 1.0, 1.5];
const RAISE_OPTIONS: &[f64] = &[2.0, 2.5, 3.0];

/// Generate `n` random but valid test configurations from a deterministic seed.
pub fn generate_configs(n: usize, seed: u64) -> Vec<TestConfig> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut configs = Vec::with_capacity(n);

    for _ in 0..n {
        let oop_range = RANGE_POOL[rng.gen_range(0..RANGE_POOL.len())].to_string();
        let ip_range = RANGE_POOL[rng.gen_range(0..RANGE_POOL.len())].to_string();

        // Generate 3 unique flop cards from 0..52
        let flop = random_unique_cards(&mut rng, 3, &[]);
        let flop_arr = [flop[0], flop[1], flop[2]];

        // Decide street depth: ~40% flop-only, ~30% flop+turn, ~30% all streets
        let street_roll: f64 = rng.gen();
        let (turn, river) = if street_roll < 0.4 {
            (None, None)
        } else if street_roll < 0.7 {
            let t = random_unique_cards(&mut rng, 1, &flop);
            (Some(t[0]), None)
        } else {
            let t = random_unique_cards(&mut rng, 1, &flop);
            let mut excluded = flop.clone();
            excluded.push(t[0]);
            let r = random_unique_cards(&mut rng, 1, &excluded);
            (Some(t[0]), Some(r[0]))
        };

        let pot = rng.gen_range(20..=500);
        let stack = rng.gen_range(100..=2000);

        // Pick 1-3 bet sizes
        let num_bets = rng.gen_range(1..=3);
        let mut bet_pcts: Vec<f64> = (0..num_bets)
            .map(|_| BET_OPTIONS[rng.gen_range(0..BET_OPTIONS.len())])
            .collect();
        bet_pcts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        bet_pcts.dedup();

        // Pick 1-2 raise sizes
        let num_raises = rng.gen_range(1..=2);
        let mut raise_pcts: Vec<f64> = (0..num_raises)
            .map(|_| RAISE_OPTIONS[rng.gen_range(0..RAISE_OPTIONS.len())])
            .collect();
        raise_pcts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        raise_pcts.dedup();

        configs.push(TestConfig {
            oop_range,
            ip_range,
            flop: flop_arr,
            turn,
            river,
            pot,
            stack,
            bet_pcts,
            raise_pcts,
        });
    }

    configs
}

/// Generate `n` river-only test configurations (fastest to solve).
pub fn generate_river_configs(n: usize, seed: u64) -> Vec<TestConfig> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut configs = Vec::with_capacity(n);

    for _ in 0..n {
        let oop_range = RANGE_POOL[rng.gen_range(0..RANGE_POOL.len())].to_string();
        let ip_range = RANGE_POOL[rng.gen_range(0..RANGE_POOL.len())].to_string();

        let board = random_unique_cards(&mut rng, 5, &[]);
        let flop = [board[0], board[1], board[2]];
        let turn = Some(board[3]);
        let river = Some(board[4]);

        let pot = rng.gen_range(20..=500);
        let stack = rng.gen_range(100..=2000);

        let num_bets = rng.gen_range(1..=3);
        let mut bet_pcts: Vec<f64> = (0..num_bets)
            .map(|_| BET_OPTIONS[rng.gen_range(0..BET_OPTIONS.len())])
            .collect();
        bet_pcts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        bet_pcts.dedup();

        let num_raises = rng.gen_range(1..=2);
        let mut raise_pcts: Vec<f64> = (0..num_raises)
            .map(|_| RAISE_OPTIONS[rng.gen_range(0..RAISE_OPTIONS.len())])
            .collect();
        raise_pcts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        raise_pcts.dedup();

        configs.push(TestConfig {
            oop_range,
            ip_range,
            flop,
            turn,
            river,
            pot,
            stack,
            bet_pcts,
            raise_pcts,
        });
    }

    configs
}

fn random_unique_cards(rng: &mut ChaCha8Rng, count: usize, exclude: &[u8]) -> Vec<u8> {
    let mut cards = Vec::with_capacity(count);
    while cards.len() < count {
        let card: u8 = rng.gen_range(0..52);
        if !exclude.contains(&card) && !cards.contains(&card) {
            cards.push(card);
        }
    }
    cards.sort_unstable();
    cards
}

// ---------------------------------------------------------------------------
// Bet-size string builders
// ---------------------------------------------------------------------------

fn bet_str(pcts: &[f64]) -> String {
    pcts.iter()
        .map(|p| format!("{}%", p * 100.0))
        .collect::<Vec<_>>()
        .join(", ")
}

fn raise_str(pcts: &[f64]) -> String {
    pcts.iter()
        .map(|p| format!("{}x", p))
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// Run our solver
// ---------------------------------------------------------------------------

pub fn run_ours(config: &TestConfig, iterations: u32) -> SolveResult {
    use range_solver::action_tree::*;
    use range_solver::bet_size::*;
    use range_solver::card::*;
    use range_solver::*;

    let initial_state = match (config.turn, config.river) {
        (None, _) => BoardState::Flop,
        (Some(_), None) => BoardState::Turn,
        (Some(_), Some(_)) => BoardState::River,
    };

    let bet_sizes = BetSizeOptions::try_from((bet_str(&config.bet_pcts).as_str(), raise_str(&config.raise_pcts).as_str())).unwrap();

    let card_config = CardConfig {
        range: [
            config.oop_range.parse().unwrap(),
            config.ip_range.parse().unwrap(),
        ],
        flop: config.flop,
        turn: config.turn.unwrap_or(NOT_DEALT),
        river: config.river.unwrap_or(NOT_DEALT),
    };

    let tree_config = TreeConfig {
        initial_state,
        starting_pot: config.pot,
        effective_stack: config.stack,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game.allocate_memory(false);

    let exploitability = solve(&mut game, iterations, 0.0, false);

    game.cache_normalized_weights();
    let root_strategy = game.strategy();
    let ev_oop = game.expected_values(0);
    let ev_ip = game.expected_values(1);
    let equity_oop = game.equity(0);
    let equity_ip = game.equity(1);

    SolveResult {
        exploitability,
        root_strategy,
        ev_oop,
        ev_ip,
        equity_oop,
        equity_ip,
    }
}

// ---------------------------------------------------------------------------
// Run the original solver
// ---------------------------------------------------------------------------

pub fn run_original(config: &TestConfig, iterations: u32) -> SolveResult {
    use postflop_solver::*;

    let initial_state = match (config.turn, config.river) {
        (None, _) => BoardState::Flop,
        (Some(_), None) => BoardState::Turn,
        (Some(_), Some(_)) => BoardState::River,
    };

    let bet_sizes = BetSizeOptions::try_from((bet_str(&config.bet_pcts).as_str(), raise_str(&config.raise_pcts).as_str())).unwrap();

    let card_config = CardConfig {
        range: [
            config.oop_range.parse().unwrap(),
            config.ip_range.parse().unwrap(),
        ],
        flop: config.flop,
        turn: config.turn.unwrap_or(NOT_DEALT),
        river: config.river.unwrap_or(NOT_DEALT),
    };

    let tree_config = TreeConfig {
        initial_state,
        starting_pot: config.pot,
        effective_stack: config.stack,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game.allocate_memory(false);

    let exploitability = solve(&mut game, iterations, 0.0, false);

    game.cache_normalized_weights();
    let root_strategy = game.strategy();
    let ev_oop = game.expected_values(0);
    let ev_ip = game.expected_values(1);
    let equity_oop = game.equity(0);
    let equity_ip = game.equity(1);

    SolveResult {
        exploitability,
        root_strategy,
        ev_oop,
        ev_ip,
        equity_oop,
        equity_ip,
    }
}
