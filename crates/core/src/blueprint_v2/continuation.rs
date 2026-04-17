use rand::Rng;

use crate::blueprint_v2::bundle::BlueprintV2Strategy;
use crate::blueprint_v2::game_tree::{GameNode, GameTree, TerminalKind, TreeAction};
use crate::blueprint_v2::mccfr::AllBuckets;
use crate::poker::Card;

/// Classification of poker actions for biasing purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionClass {
    Fold,  // includes Check (passive)
    Call,
    Raise, // includes Bet, Raise, AllIn (aggressive)
}

/// Which bias to apply to a continuation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiasType {
    Unbiased,
    Fold,
    Call,
    Raise,
}

/// Classify a tree action into one of three biasing categories.
///
/// Fold and Check are passive (grouped as `Fold`).
/// Call is its own class.
/// Bet, Raise, and `AllIn` are aggressive (grouped as `Raise`).
#[must_use]
pub fn classify_action(action: &TreeAction) -> ActionClass {
    match action {
        TreeAction::Fold | TreeAction::Check => ActionClass::Fold,
        TreeAction::Call => ActionClass::Call,
        TreeAction::Bet(_) | TreeAction::Raise(_) | TreeAction::AllIn => ActionClass::Raise,
    }
}

/// Bias a probability distribution toward a target action class.
///
/// Multiplies probabilities of the target `ActionClass` (determined by `bias`)
/// by `factor`, then renormalizes so the result sums to 1.
/// If `bias` is `Unbiased`, returns the original probabilities unchanged.
#[must_use]
pub fn bias_strategy(
    probs: &[f32],
    actions: &[ActionClass],
    bias: BiasType,
    factor: f64,
) -> Vec<f32> {
    if bias == BiasType::Unbiased {
        return probs.to_vec();
    }

    let target = match bias {
        BiasType::Fold => ActionClass::Fold,
        BiasType::Call => ActionClass::Call,
        BiasType::Raise => ActionClass::Raise,
        BiasType::Unbiased => unreachable!(),
    };

    #[allow(clippy::cast_possible_truncation)]
    let factor = factor as f32;

    let biased: Vec<f32> = probs
        .iter()
        .zip(actions.iter())
        .map(|(&p, &a)| if a == target { p * factor } else { p })
        .collect();

    let sum: f32 = biased.iter().sum();
    if sum == 0.0 {
        return biased;
    }

    biased.iter().map(|&p| p / sum).collect()
}

/// Map a card to a bit index (0..51) for use in a 52-card bitset.
fn card_bit(card: Card) -> u32 {
    card.value as u32 * 4 + card.suit as u32
}

/// Build a deck of remaining cards excluding both hands and the board.
fn remaining_deck(hero: [Card; 2], opponent: [Card; 2], board: &[Card]) -> Vec<Card> {
    use crate::poker::{ALL_VALUES, ALL_SUITS};

    let mut used = 0u64;
    for &c in &hero {
        used |= 1u64 << card_bit(c);
    }
    for &c in &opponent {
        used |= 1u64 << card_bit(c);
    }
    for &c in board {
        used |= 1u64 << card_bit(c);
    }

    let mut remaining = Vec::new();
    for &v in &ALL_VALUES {
        for &s in &ALL_SUITS {
            let c = Card::new(v, s);
            if used & (1u64 << card_bit(c)) == 0 {
                remaining.push(c);
            }
        }
    }
    remaining
}

/// Context for rollout evaluation -- groups invariant parameters that don't
/// change across recursive calls.
pub struct RolloutContext<'a> {
    pub abstract_tree: &'a GameTree,
    pub decision_idx_map: &'a [u32],
    pub strategy: &'a BlueprintV2Strategy,
    pub buckets: &'a AllBuckets,
    pub bias: BiasType,
    pub bias_factor: f64,
    pub player: u8,
    pub num_rollouts: u32,
    /// Starting stack per player (in the same units as pot/invested).
    /// Used for all-in calculations in the rollout.
    pub starting_stack: f64,
    /// Optional counter incremented once per rollout terminal reached.
    /// Used for hands/sec telemetry. `None` disables the counter (no overhead).
    pub hand_counter: Option<&'a std::sync::atomic::AtomicU64>,
}

/// Walk the abstract game tree using a (biased) blueprint strategy with
/// two concrete hands, computing the expected value for `player`.
///
/// At decision nodes, the acting player's strategy is looked up from
/// the blueprint, optionally biased, and the EV is the probability-
/// weighted sum over all children. At terminal nodes, payoffs are
/// computed directly. At chance nodes, `num_rollouts` random cards
/// are sampled and the results averaged.
///
/// # Panics
///
/// Panics if a `DepthBoundary` terminal is encountered (should not
/// appear in rollout trees).
/// Rollout from a depth boundary using the subgame's actual pot and
/// invested amounts. Terminal payoffs are computed from the carried
/// state, not from the abstract tree's stored values.
pub fn rollout_from_boundary(
    hero_hand: [Card; 2],
    opponent_hand: [Card; 2],
    board: &[Card],
    ctx: &RolloutContext<'_>,
    abstract_node: u32,
    rng: &mut impl Rng,
    pot: f64,
    invested: [f64; 2],
) -> f64 {
    rollout_inner(hero_hand, opponent_hand, board, ctx, abstract_node, rng, None, pot, invested)
}

/// Inner recursive function with bucket caching.
///
/// `cached_buckets` holds `[hero_bucket, opponent_bucket]` for the current
/// street. Computed on the first Decision node of each street, invalidated
/// (set to `None`) at Chance nodes when the board changes.
///
/// `pot` and `invested` are carried from the subgame's boundary state and
/// updated at each action so terminal payoffs reflect the actual chip amounts.
#[allow(clippy::too_many_arguments)]
fn rollout_inner(
    hero_hand: [Card; 2],
    opponent_hand: [Card; 2],
    board: &[Card],
    ctx: &RolloutContext<'_>,
    abstract_node: u32,
    rng: &mut impl Rng,
    cached_buckets: Option<[u16; 2]>,
    pot: f64,
    invested: [f64; 2],
) -> f64 {
    match &ctx.abstract_tree.nodes[abstract_node as usize] {
        GameNode::Terminal { kind, .. } => {
            if let Some(counter) = ctx.hand_counter {
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            // Use carried pot/invested, NOT the abstract tree's stored values.
            let p = ctx.player as usize;
            match kind {
                TerminalKind::Fold { winner } => {
                    if ctx.player == *winner {
                        // Winner gains pot minus own investment
                        pot - invested[p]
                    } else {
                        // Folder loses own investment
                        -invested[p]
                    }
                }
                TerminalKind::Showdown => {
                    let player_rank = crate::showdown_equity::rank_hand(hero_hand, board);
                    let opponent_rank = crate::showdown_equity::rank_hand(opponent_hand, board);
                    match player_rank.cmp(&opponent_rank) {
                        std::cmp::Ordering::Greater => pot - invested[p],
                        std::cmp::Ordering::Less => -invested[p],
                        std::cmp::Ordering::Equal => pot / 2.0 - invested[p],
                    }
                }
                TerminalKind::DepthBoundary => {
                    panic!("DepthBoundary should not appear during rollout");
                }
            }
        }
        GameNode::Decision { player: acting_player, street, actions, children, .. } => {
            let acting_player = *acting_player;
            let actor = acting_player as usize;
            let street = *street;
            let actions = actions.clone();
            let children = children.clone();

            // Compute buckets for both players once per street, reuse
            // across all Decision nodes until a Chance node changes the board.
            let buckets = cached_buckets.unwrap_or_else(|| [
                ctx.buckets.get_bucket(street, hero_hand, board),
                ctx.buckets.get_bucket(street, opponent_hand, board),
            ]);

            let acting_bucket = if acting_player == ctx.player {
                buckets[0] // hero
            } else {
                buckets[1] // opponent
            };

            let decision_idx = ctx.decision_idx_map[abstract_node as usize];
            let probs = ctx.strategy.get_action_probs(decision_idx as usize, acting_bucket);

            let action_classes: Vec<ActionClass> = actions.iter().map(classify_action).collect();
            let biased = bias_strategy(probs, &action_classes, ctx.bias, ctx.bias_factor);

            let mut ev = 0.0;
            for (i, &child) in children.iter().enumerate() {
                // Compute new invested/pot for this action.
                let (child_pot, child_invested) = apply_action(
                    &actions[i], pot, invested, actor, ctx.starting_stack,
                );
                let child_ev = rollout_inner(
                    hero_hand, opponent_hand, board, ctx, child, rng,
                    Some(buckets), child_pot, child_invested,
                );
                ev += f64::from(biased[i]) * child_ev;
            }
            ev
        }
        GameNode::Chance { next_street: _, child } => {
            let child = *child;
            let deck = remaining_deck(hero_hand, opponent_hand, board);
            #[allow(clippy::cast_possible_truncation)]
            let deck_len = deck.len() as u32;
            let n = ctx.num_rollouts.min(deck_len);
            if n == 0 {
                return 0.0;
            }

            let mut total = 0.0;
            for _ in 0..n {
                let idx = rng.random_range(0..deck.len());
                let card = deck[idx];
                let mut new_board = board.to_vec();
                new_board.push(card);
                let child_ev = rollout_inner(
                    hero_hand, opponent_hand, &new_board, ctx, child, rng,
                    None, // invalidate cache — board changed
                    pot, invested, // pot/invested unchanged through chance
                );
                total += child_ev;
            }
            total / f64::from(n)
        }
    }
}

/// Compute new pot and invested amounts after an action.
/// Mirrors the logic in `GameTree::build_child`.
fn apply_action(
    action: &TreeAction,
    pot: f64,
    invested: [f64; 2],
    actor: usize,
    starting_stack: f64,
) -> (f64, [f64; 2]) {
    let opponent = 1 - actor;
    let mut new_invested = invested;

    match action {
        TreeAction::Fold | TreeAction::Check => {
            // No chip movement
            (pot, new_invested)
        }
        TreeAction::Call => {
            // Match opponent's investment, capped to stack.
            new_invested[actor] = invested[opponent].min(starting_stack);
            let new_pot = pot + (new_invested[actor] - invested[actor]);
            (new_pot, new_invested)
        }
        TreeAction::Bet(amount) | TreeAction::Raise(amount) => {
            // Bet/raise TO this amount (absolute, not increment).
            // Cap to starting_stack — the abstract tree's amounts may exceed
            // the unit game's stack when SPR is small.
            let old = invested[actor];
            new_invested[actor] = (*amount).min(starting_stack);
            let new_pot = pot + (new_invested[actor] - old);
            (new_pot, new_invested)
        }
        TreeAction::AllIn => {
            let old = invested[actor];
            new_invested[actor] = starting_stack;
            let new_pot = pot + (new_invested[actor] - old);
            (new_pot, new_invested)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_fold_multiplies_fold_actions() {
        let probs = vec![0.2_f32, 0.5, 0.3];
        let actions = vec![ActionClass::Fold, ActionClass::Call, ActionClass::Raise];
        let biased = bias_strategy(&probs, &actions, BiasType::Fold, 10.0);
        assert!((biased[0] - 0.714).abs() < 0.01);
        assert!((biased[1] - 0.179).abs() < 0.01);
        assert!((biased[2] - 0.107).abs() < 0.01);
    }

    #[test]
    fn bias_unbiased_returns_original() {
        let probs = vec![0.2_f32, 0.5, 0.3];
        let actions = vec![ActionClass::Fold, ActionClass::Call, ActionClass::Raise];
        let biased = bias_strategy(&probs, &actions, BiasType::Unbiased, 10.0);
        for (a, b) in probs.iter().zip(biased.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn classify_tree_actions() {
        use crate::blueprint_v2::game_tree::TreeAction;
        assert_eq!(classify_action(&TreeAction::Fold), ActionClass::Fold);
        assert_eq!(classify_action(&TreeAction::Check), ActionClass::Fold);
        assert_eq!(classify_action(&TreeAction::Call), ActionClass::Call);
        assert_eq!(classify_action(&TreeAction::Bet(5.0)), ActionClass::Raise);
        assert_eq!(classify_action(&TreeAction::Raise(10.0)), ActionClass::Raise);
        assert_eq!(classify_action(&TreeAction::AllIn), ActionClass::Raise);
    }

    #[test]
    fn bias_call_multiplies_call_actions() {
        let probs = vec![0.2_f32, 0.5, 0.3];
        let actions = vec![ActionClass::Fold, ActionClass::Call, ActionClass::Raise];
        let biased = bias_strategy(&probs, &actions, BiasType::Call, 10.0);
        // call: 0.5*10=5.0, fold: 0.2, raise: 0.3 => sum=5.5
        // call: 5.0/5.5 = 0.909, fold: 0.2/5.5 = 0.0364, raise: 0.3/5.5 = 0.0545
        assert!((biased[1] - 0.909).abs() < 0.01);
        assert!((biased[0] - 0.0364).abs() < 0.01);
        assert!((biased[2] - 0.0545).abs() < 0.01);
    }

    #[test]
    fn bias_raise_multiplies_raise_actions() {
        let probs = vec![0.2_f32, 0.5, 0.3];
        let actions = vec![ActionClass::Fold, ActionClass::Call, ActionClass::Raise];
        let biased = bias_strategy(&probs, &actions, BiasType::Raise, 10.0);
        // raise: 0.3*10=3.0, fold: 0.2, call: 0.5 => sum=3.7
        // raise: 3.0/3.7 = 0.8108, fold: 0.2/3.7 = 0.0541, call: 0.5/3.7 = 0.1351
        assert!((biased[2] - 0.8108).abs() < 0.01);
        assert!((biased[0] - 0.0541).abs() < 0.01);
        assert!((biased[1] - 0.1351).abs() < 0.01);
    }

    #[test]
    fn bias_with_factor_zero_zeroes_target_class() {
        let probs = vec![0.2_f32, 0.5, 0.3];
        let actions = vec![ActionClass::Fold, ActionClass::Call, ActionClass::Raise];
        let biased = bias_strategy(&probs, &actions, BiasType::Fold, 0.0);
        // fold: 0.2*0=0, call: 0.5, raise: 0.3 => sum=0.8
        // fold: 0.0, call: 0.5/0.8=0.625, raise: 0.3/0.8=0.375
        assert!((biased[0]).abs() < 1e-6);
        assert!((biased[1] - 0.625).abs() < 0.01);
        assert!((biased[2] - 0.375).abs() < 0.01);
    }

    #[test]
    fn bias_preserves_sum_to_one() {
        let probs = vec![0.1_f32, 0.3, 0.4, 0.2];
        let actions = vec![ActionClass::Fold, ActionClass::Call, ActionClass::Raise, ActionClass::Raise];
        let biased = bias_strategy(&probs, &actions, BiasType::Raise, 5.0);
        let sum: f32 = biased.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn bias_with_factor_one_returns_original() {
        let probs = vec![0.2_f32, 0.5, 0.3];
        let actions = vec![ActionClass::Fold, ActionClass::Call, ActionClass::Raise];
        let biased = bias_strategy(&probs, &actions, BiasType::Fold, 1.0);
        for (a, b) in probs.iter().zip(biased.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    // ---- rollout_from_boundary tests ----

    use crate::blueprint_v2::game_tree::{GameNode, GameTree, TerminalKind};
    use crate::blueprint_v2::bundle::BlueprintV2Strategy;
    use crate::blueprint_v2::mccfr::AllBuckets;
    use crate::blueprint_v2::Street;
    use crate::poker::{Card, Value, Suit};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Build a minimal tree with a single fold terminal.
    /// Player 0 folds, player 1 wins. Pot=10.
    fn fold_terminal_tree() -> (GameTree, Vec<u32>) {
        let nodes = vec![
            GameNode::Terminal {
                kind: TerminalKind::Fold { winner: 1 },
                pot: 10.0,
                stacks: [97.0, 95.0],
            },
        ];
        let tree = GameTree { nodes, root: 0, dealer: 0, starting_stack: 100.0 };
        let decision_idx_map = vec![u32::MAX]; // no decision nodes
        (tree, decision_idx_map)
    }

    fn dummy_hands() -> ([Card; 2], [Card; 2]) {
        let hero = [Card::new(Value::Ace, Suit::Spade), Card::new(Value::King, Suit::Spade)];
        let opp = [Card::new(Value::Two, Suit::Heart), Card::new(Value::Three, Suit::Heart)];
        (hero, opp)
    }

    fn dummy_buckets() -> AllBuckets {
        AllBuckets::new([169, 10, 10, 10], [None, None, None, None])
    }

    fn dummy_strategy() -> BlueprintV2Strategy {
        BlueprintV2Strategy::empty()
    }

    fn make_ctx<'a>(
        tree: &'a GameTree,
        dim: &'a [u32],
        strategy: &'a BlueprintV2Strategy,
        buckets: &'a AllBuckets,
        bias: BiasType,
        bias_factor: f64,
        player: u8,
        num_rollouts: u32,
    ) -> RolloutContext<'a> {
        RolloutContext {
            abstract_tree: tree,
            decision_idx_map: dim,
            strategy,
            buckets,
            bias,
            bias_factor,
            player,
            num_rollouts,
            starting_stack: 100.0,
            hand_counter: None,
        }
    }

    #[test]
    fn rollout_fold_terminal_winner_gets_pot_minus_invested() {
        let (tree, dim) = fold_terminal_tree();
        let (hero, opp) = dummy_hands();
        let buckets = dummy_buckets();
        let strategy = dummy_strategy();
        let mut rng = StdRng::seed_from_u64(42);

        // player=1 (winner): payoff = pot - invested[1] = 10 - 5 = 5
        let ctx = make_ctx(&tree, &dim, &strategy, &buckets, BiasType::Unbiased, 1.0, 1, 1);
        let ev = rollout_from_boundary(hero, opp, &[], &ctx, 0, &mut rng, 10.0, [3.0, 5.0]);
        assert!((ev - 5.0).abs() < 1e-9, "Winner EV should be 5.0, got {ev}");
    }

    #[test]
    fn rollout_fold_terminal_loser_loses_invested() {
        let (tree, dim) = fold_terminal_tree();
        let (hero, opp) = dummy_hands();
        let buckets = dummy_buckets();
        let strategy = dummy_strategy();
        let mut rng = StdRng::seed_from_u64(42);

        // player=0 (loser): payoff = -invested[0] = -3
        let ctx = make_ctx(&tree, &dim, &strategy, &buckets, BiasType::Unbiased, 1.0, 0, 1);
        let ev = rollout_from_boundary(hero, opp, &[], &ctx, 0, &mut rng, 10.0, [3.0, 5.0]);
        assert!((ev - (-3.0)).abs() < 1e-9, "Loser EV should be -3.0, got {ev}");
    }

    /// Build a showdown terminal with pot=20.
    fn showdown_terminal_tree() -> (GameTree, Vec<u32>) {
        let nodes = vec![
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 20.0,
                stacks: [92.0, 92.0],
            },
        ];
        let tree = GameTree { nodes, root: 0, dealer: 0, starting_stack: 100.0 };
        let decision_idx_map = vec![u32::MAX];
        (tree, decision_idx_map)
    }

    #[test]
    fn rollout_showdown_hero_wins_with_better_hand() {
        let (tree, dim) = showdown_terminal_tree();
        let buckets = dummy_buckets();
        let strategy = dummy_strategy();
        let mut rng = StdRng::seed_from_u64(42);

        // AKs vs 23o on a board that gives AK the best hand
        let hero = [Card::new(Value::Ace, Suit::Spade), Card::new(Value::King, Suit::Spade)];
        let opp = [Card::new(Value::Two, Suit::Heart), Card::new(Value::Three, Suit::Heart)];
        // Board: A K Q 7 4 (different suits to avoid flush)
        let board = [
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Diamond),
            Card::new(Value::Queen, Suit::Club),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Four, Suit::Club),
        ];

        // hero (player=0) has trip aces with K kicker, opponent has nothing
        let ctx = make_ctx(&tree, &dim, &strategy, &buckets, BiasType::Unbiased, 1.0, 0, 1);
        let ev = rollout_from_boundary(hero, opp, &board, &ctx, 0, &mut rng, 20.0, [8.0, 8.0]);
        // Hero wins: pot - invested[0] = 20 - 8 = 12
        assert!((ev - 12.0).abs() < 1e-9, "Hero should win 12.0, got {ev}");
    }

    #[test]
    fn rollout_showdown_hero_loses() {
        let (tree, dim) = showdown_terminal_tree();
        let buckets = dummy_buckets();
        let strategy = dummy_strategy();
        let mut rng = StdRng::seed_from_u64(42);

        // hero has 23, opp has AK
        let hero = [Card::new(Value::Two, Suit::Heart), Card::new(Value::Three, Suit::Heart)];
        let opp = [Card::new(Value::Ace, Suit::Spade), Card::new(Value::King, Suit::Spade)];
        let board = [
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Diamond),
            Card::new(Value::Queen, Suit::Club),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Four, Suit::Club),
        ];

        // hero (player=0) loses
        let ctx = make_ctx(&tree, &dim, &strategy, &buckets, BiasType::Unbiased, 1.0, 0, 1);
        let ev = rollout_from_boundary(hero, opp, &board, &ctx, 0, &mut rng, 20.0, [8.0, 8.0]);
        assert!((ev - (-8.0)).abs() < 1e-9, "Hero should lose 8.0, got {ev}");
    }

    #[test]
    fn rollout_showdown_tie_splits_pot() {
        // Both have same kickers - tie scenario
        let nodes = vec![
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 20.0,
                stacks: [90.0, 90.0],
            },
        ];
        let tree = GameTree { nodes, root: 0, dealer: 0, starting_stack: 100.0 };
        let dim = vec![u32::MAX];
        let buckets = dummy_buckets();
        let strategy = dummy_strategy();
        let mut rng = StdRng::seed_from_u64(42);

        // Both have AK same kickers, board gives same best 5-card hand
        let hero = [Card::new(Value::Ace, Suit::Spade), Card::new(Value::King, Suit::Spade)];
        let opp = [Card::new(Value::Ace, Suit::Heart), Card::new(Value::King, Suit::Heart)];
        let board = [
            Card::new(Value::Queen, Suit::Diamond),
            Card::new(Value::Jack, Suit::Club),
            Card::new(Value::Ten, Suit::Diamond),
            Card::new(Value::Four, Suit::Club),
            Card::new(Value::Five, Suit::Diamond),
        ];

        let ctx = make_ctx(&tree, &dim, &strategy, &buckets, BiasType::Unbiased, 1.0, 0, 1);
        let ev = rollout_from_boundary(hero, opp, &board, &ctx, 0, &mut rng, 20.0, [10.0, 10.0]);
        // Tie: pot/2 - invested[0] = 10 - 10 = 0
        assert!((ev).abs() < 1e-9, "Tie should give EV 0.0, got {ev}");
    }

    /// Build a 1-decision tree: player 0 can Fold or Call.
    /// Fold -> player 1 wins, pot=10, invested=[3,5].
    /// Call -> showdown, pot=14, invested=[7,7].
    /// Uses a hand-crafted strategy where bucket 0 plays [0.6 fold, 0.4 call].
    fn decision_with_fold_call_tree() -> (GameTree, Vec<u32>, BlueprintV2Strategy) {
        use crate::blueprint_v2::game_tree::TreeAction;

        let nodes = vec![
            // Node 0: Decision, player 0, river
            GameNode::Decision {
                player: 0,
                street: Street::River,
                actions: vec![TreeAction::Fold, TreeAction::Call],
                children: vec![1, 2],
                blueprint_decision_idx: None,
            },
            // Node 1: Fold terminal, winner=1
            GameNode::Terminal {
                kind: TerminalKind::Fold { winner: 1 },
                pot: 10.0,
                stacks: [97.0, 95.0],
            },
            // Node 2: Showdown
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 14.0,
                stacks: [93.0, 93.0],
            },
        ];
        let tree = GameTree { nodes, root: 0, dealer: 0, starting_stack: 100.0 };
        let decision_idx_map = vec![0, u32::MAX, u32::MAX];

        // Build a strategy with 1 decision node, 2 actions, 10 river buckets.
        // All buckets get [0.6, 0.4] (fold 60%, call 40%).
        let bucket_counts: [u16; 4] = [169, 10, 10, 10];
        let river_buckets = bucket_counts[3] as usize;
        let num_actions = 2;
        let mut action_probs = Vec::with_capacity(river_buckets * num_actions);
        for _ in 0..river_buckets {
            action_probs.push(0.6_f32); // fold
            action_probs.push(0.4_f32); // call
        }
        let strategy = BlueprintV2Strategy::from_parts(
            action_probs,
            vec![num_actions as u16],
            vec![Street::River as u8],
            bucket_counts,
        );

        (tree, decision_idx_map, strategy)
    }

    #[test]
    fn rollout_decision_fold_call_weighted_ev() {
        let (tree, dim, strategy) = decision_with_fold_call_tree();
        let buckets = dummy_buckets();
        let mut rng = StdRng::seed_from_u64(42);

        // Hero = player 0 = AK, Opp = player 1 = 23
        // Board where hero (AK) wins at showdown
        let hero = [Card::new(Value::Ace, Suit::Spade), Card::new(Value::King, Suit::Spade)];
        let opp = [Card::new(Value::Two, Suit::Heart), Card::new(Value::Three, Suit::Heart)];
        let board = [
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Diamond),
            Card::new(Value::Queen, Suit::Club),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Four, Suit::Club),
        ];

        // EV for player 0:
        // Fold: -invested[0] = -3 (player 0 is loser)
        // Call (showdown): hero wins => pot - invested[0] = 14 - 7 = 7
        // EV = 0.6 * (-3) + 0.4 * 7 = -1.8 + 2.8 = 1.0
        // Initial state: pot=10, invested=[3,7]. Player 0 facing a bet.
        // Fold: pot=10, invested=[3,7] → loser payoff = -3
        // Call: invested[0] matches invested[1]=7, pot=10+(7-3)=14, invested=[7,7] → winner payoff = 14-7=7
        let ctx = make_ctx(&tree, &dim, &strategy, &buckets, BiasType::Unbiased, 1.0, 0, 1);
        let ev = rollout_from_boundary(hero, opp, &board, &ctx, 0, &mut rng, 10.0, [3.0, 7.0]);
        assert!((ev - 1.0).abs() < 0.01, "Expected EV ~1.0, got {ev}");
    }

    #[test]
    fn rollout_decision_with_bias_shifts_ev() {
        let (tree, dim, strategy) = decision_with_fold_call_tree();
        let buckets = dummy_buckets();
        let mut rng = StdRng::seed_from_u64(42);

        let hero = [Card::new(Value::Ace, Suit::Spade), Card::new(Value::King, Suit::Spade)];
        let opp = [Card::new(Value::Two, Suit::Heart), Card::new(Value::Three, Suit::Heart)];
        let board = [
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Diamond),
            Card::new(Value::Queen, Suit::Club),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Four, Suit::Club),
        ];

        // Unbiased EV = 0.6 * (-3) + 0.4 * 7 = 1.0
        let ctx = make_ctx(&tree, &dim, &strategy, &buckets, BiasType::Unbiased, 1.0, 0, 1);
        let unbiased_ev = rollout_from_boundary(hero, opp, &board, &ctx, 0, &mut rng, 10.0, [3.0, 7.0]);

        // Bias toward Call (factor 10): call gets more weight, fold less
        // So EV should increase (call=+7 is more favored than fold=-3)
        let ctx_biased = make_ctx(&tree, &dim, &strategy, &buckets, BiasType::Call, 10.0, 0, 1);
        let biased_ev = rollout_from_boundary(hero, opp, &board, &ctx_biased, 0, &mut rng, 10.0, [3.0, 7.0]);
        assert!(biased_ev > unbiased_ev, "Call bias should increase EV when call is +7");
    }

    #[test]
    fn rollout_chance_node_samples_cards() {
        // Chance -> Showdown terminal. Hero has a dominant hand.
        // The EV should approximate the showdown value across sampled runouts.
        let nodes = vec![
            // Node 0: Chance node to turn
            GameNode::Chance { next_street: Street::Turn, child: 1 },
            // Node 1: Showdown terminal (pot=10)
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 10.0,
                stacks: [95.0, 95.0],
            },
        ];
        let tree = GameTree { nodes, root: 0, dealer: 0, starting_stack: 100.0 };
        let dim = vec![u32::MAX, u32::MAX];
        let buckets = dummy_buckets();
        let strategy = dummy_strategy();
        let mut rng = StdRng::seed_from_u64(42);

        // Hero AA vs Opp 72o on a flop of K84 rainbow
        // Hero is strongly favored; most runouts hero wins
        let hero = [Card::new(Value::Ace, Suit::Spade), Card::new(Value::Ace, Suit::Heart)];
        let opp = [Card::new(Value::Seven, Suit::Club), Card::new(Value::Two, Suit::Diamond)];
        let board = vec![
            Card::new(Value::King, Suit::Diamond),
            Card::new(Value::Eight, Suit::Club),
            Card::new(Value::Four, Suit::Spade),
        ];

        // With many rollouts, hero should have positive EV
        let ctx = make_ctx(&tree, &dim, &strategy, &buckets, BiasType::Unbiased, 1.0, 0, 100);
        let ev = rollout_from_boundary(hero, opp, &board, &ctx, 0, &mut rng, 10.0, [5.0, 5.0]);
        // AA vs 72o on K84: hero is ~90%+ equity
        // EV should be positive (win 5 most of the time)
        assert!(ev > 0.0, "AA vs 72o should have positive EV, got {ev}");
    }

    #[test]
    fn remaining_deck_excludes_all_known_cards() {
        let hero = [Card::new(Value::Ace, Suit::Spade), Card::new(Value::King, Suit::Spade)];
        let opp = [Card::new(Value::Two, Suit::Heart), Card::new(Value::Three, Suit::Heart)];
        let board = [
            Card::new(Value::Queen, Suit::Diamond),
            Card::new(Value::Jack, Suit::Club),
            Card::new(Value::Ten, Suit::Diamond),
        ];
        let deck = remaining_deck(hero, opp, &board);
        // 52 - 4 (hands) - 3 (board) = 45
        assert_eq!(deck.len(), 45, "Deck should have 45 remaining cards");

        // Verify none of the excluded cards are in the deck
        for &c in &hero {
            assert!(!deck.contains(&c), "Hero card should not be in deck");
        }
        for &c in &opp {
            assert!(!deck.contains(&c), "Opponent card should not be in deck");
        }
        for &c in &board {
            assert!(!deck.contains(&c), "Board card should not be in deck");
        }
    }

    #[test]
    fn remaining_deck_empty_board() {
        let hero = [Card::new(Value::Ace, Suit::Spade), Card::new(Value::King, Suit::Spade)];
        let opp = [Card::new(Value::Two, Suit::Heart), Card::new(Value::Three, Suit::Heart)];
        let deck = remaining_deck(hero, opp, &[]);
        assert_eq!(deck.len(), 48, "52 - 4 = 48 remaining cards");
    }

    #[test]
    fn remaining_deck_full_board() {
        let hero = [Card::new(Value::Ace, Suit::Spade), Card::new(Value::King, Suit::Spade)];
        let opp = [Card::new(Value::Two, Suit::Heart), Card::new(Value::Three, Suit::Heart)];
        let board = [
            Card::new(Value::Queen, Suit::Diamond),
            Card::new(Value::Jack, Suit::Club),
            Card::new(Value::Ten, Suit::Diamond),
            Card::new(Value::Nine, Suit::Spade),
            Card::new(Value::Eight, Suit::Heart),
        ];
        let deck = remaining_deck(hero, opp, &board);
        // 52 - 4 - 5 = 43
        assert_eq!(deck.len(), 43, "52 - 9 = 43 remaining cards");
    }

    #[test]
    fn rollout_opponent_decision_uses_opponent_hand() {
        use crate::blueprint_v2::game_tree::TreeAction;

        // Player 1 decides. We are computing EV for player 0.
        // Player 1's hand should be used for bucket lookup.
        let nodes = vec![
            // Node 0: Decision by player 1 (opponent if player=0)
            GameNode::Decision {
                player: 1,
                street: Street::River,
                actions: vec![TreeAction::Fold, TreeAction::Call],
                children: vec![1, 2],
                blueprint_decision_idx: None,
            },
            // Fold -> player 0 wins
            GameNode::Terminal {
                kind: TerminalKind::Fold { winner: 0 },
                pot: 10.0,
                stacks: [97.0, 97.0],
            },
            // Call -> showdown
            GameNode::Terminal {
                kind: TerminalKind::Showdown,
                pot: 14.0,
                stacks: [93.0, 93.0],
            },
        ];
        let tree = GameTree { nodes, root: 0, dealer: 0, starting_stack: 100.0 };
        let dim = vec![0, u32::MAX, u32::MAX];

        // Strategy: all buckets play [0.5, 0.5]
        let bucket_counts: [u16; 4] = [169, 10, 10, 10];
        let river_buckets = bucket_counts[3] as usize;
        let mut action_probs = Vec::new();
        for _ in 0..river_buckets {
            action_probs.push(0.5_f32);
            action_probs.push(0.5_f32);
        }
        let strategy = BlueprintV2Strategy::from_parts(
            action_probs,
            vec![2],
            vec![Street::River as u8],
            bucket_counts,
        );
        let buckets = dummy_buckets();
        let mut rng = StdRng::seed_from_u64(42);

        // Hero AK, Opp 23. Board where hero wins showdown.
        let hero = [Card::new(Value::Ace, Suit::Spade), Card::new(Value::King, Suit::Spade)];
        let opp = [Card::new(Value::Two, Suit::Heart), Card::new(Value::Three, Suit::Heart)];
        let board = [
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Diamond),
            Card::new(Value::Queen, Suit::Club),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Four, Suit::Club),
        ];

        // For player 0:
        // Initial: pot=10, invested=[3,3]. Player 1 deciding.
        // Fold by opp: pot=10, invested=[3,3] → player 0 wins: 10 - 3 = 7
        // Call by opp: invested[1]=invested[0]=3 (no change), pot=10 → showdown hero wins: 10 - 3 = 7
        // EV = 0.5 * 7 + 0.5 * 7 = 7
        let ctx = make_ctx(&tree, &dim, &strategy, &buckets, BiasType::Unbiased, 1.0, 0, 1);
        let ev = rollout_from_boundary(hero, opp, &board, &ctx, 0, &mut rng, 10.0, [3.0, 3.0]);
        assert!((ev - 7.0).abs() < 0.01, "EV should be 7.0, got {ev}");
    }

    #[test]
    #[should_panic(expected = "DepthBoundary should not appear during rollout")]
    fn rollout_depth_boundary_panics() {
        let nodes = vec![
            GameNode::Terminal {
                kind: TerminalKind::DepthBoundary,
                pot: 10.0,
                stacks: [95.0, 95.0],
            },
        ];
        let tree = GameTree { nodes, root: 0, dealer: 0, starting_stack: 100.0 };
        let dim = vec![u32::MAX];
        let (hero, opp) = dummy_hands();
        let buckets = dummy_buckets();
        let strategy = dummy_strategy();
        let mut rng = StdRng::seed_from_u64(42);

        let ctx = make_ctx(&tree, &dim, &strategy, &buckets, BiasType::Unbiased, 1.0, 0, 1);
        rollout_from_boundary(hero, opp, &[], &ctx, 0, &mut rng, 10.0, [5.0, 5.0]);
    }
}
