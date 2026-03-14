use crate::tree::{FlatTree, NodeType};
use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
use range_solver::bet_size::BetSizeOptions;
use range_solver::card::{card_from_str, flop_from_str};
use range_solver::interface::Game;
use range_solver::range::Range;
use range_solver::{CardConfig, PostFlopGame};

/// Helper: build a simple river game, allocate memory, return the game.
fn make_river_game() -> PostFlopGame {
    let oop_range: Range = "AA,KK,QQ,AKs".parse().unwrap();
    let ip_range: Range = "QQ-JJ,AQs,AJs".parse().unwrap();
    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: flop_from_str("Qs Jh 2c").unwrap(),
        turn: card_from_str("8d").unwrap(),
        river: card_from_str("3s").unwrap(),
    };
    let sizes = BetSizeOptions::try_from(("50%, a", "")).unwrap();
    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 100,
        effective_stack: 100,
        river_bet_sizes: [sizes.clone(), sizes],
        ..Default::default()
    };
    let tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
    game.allocate_memory(false);
    game
}

#[test]
fn test_build_from_postflop_game() {
    let mut game = make_river_game();

    let num_hands_oop = game.num_private_hands(0);
    let num_hands_ip = game.num_private_hands(1);

    let flat = FlatTree::from_postflop_game(&mut game);

    // The tree should have at least some nodes
    assert!(flat.num_nodes() > 0, "tree should have nodes");
    assert!(flat.num_levels() > 0, "tree should have levels");
    assert!(flat.num_infosets > 0, "tree should have infosets");
    assert_eq!(flat.num_hands_oop, num_hands_oop);
    assert_eq!(flat.num_hands_ip, num_hands_ip);

    // Root should be a decision node (OOP acts first on river)
    assert_eq!(flat.node_types[0], NodeType::DecisionOop);
    assert_eq!(flat.parent_nodes[0], u32::MAX);
    assert_eq!(flat.parent_actions[0], u32::MAX);

    // Root pot should be the starting pot (no bets yet)
    assert_eq!(flat.pots[0], 100.0);

    // Root should have at least 2 children (fold, check, and possibly bets)
    assert!(
        flat.num_children(0) >= 2,
        "root should have >= 2 children, got {}",
        flat.num_children(0)
    );

    // Level 0 should contain exactly the root
    assert_eq!(flat.level_node_count(0), 1);

    // Level 1 should have as many nodes as root has children
    assert_eq!(flat.level_node_count(1), flat.num_children(0));

    // Every non-root node should have a valid parent
    for i in 1..flat.num_nodes() {
        assert_ne!(
            flat.parent_nodes[i], u32::MAX,
            "non-root node {i} should have a parent"
        );
        let parent = flat.parent_nodes[i] as usize;
        assert!(parent < i, "parent {parent} should come before child {i}");
    }

    // All terminal nodes should be in terminal_indices
    let mut expected_terminals: Vec<u32> = (0..flat.num_nodes() as u32)
        .filter(|&i| flat.is_terminal(i as usize))
        .collect();
    let mut actual_terminals = flat.terminal_indices.clone();
    expected_terminals.sort();
    actual_terminals.sort();
    assert_eq!(expected_terminals, actual_terminals);

    // There should be both fold and showdown terminals
    let has_fold = flat
        .terminal_indices
        .iter()
        .any(|&i| flat.node_types[i as usize] == NodeType::TerminalFold);
    let has_showdown = flat
        .terminal_indices
        .iter()
        .any(|&i| flat.node_types[i as usize] == NodeType::TerminalShowdown);
    assert!(has_fold, "should have fold terminals");
    assert!(has_showdown, "should have showdown terminals");

    // Fold payoffs should have entries for fold terminals
    let num_fold_terminals = flat
        .terminal_indices
        .iter()
        .filter(|&&i| flat.node_types[i as usize] == NodeType::TerminalFold)
        .count();
    assert_eq!(
        flat.fold_payoffs.iter().filter(|v| !v.is_empty()).count(),
        num_fold_terminals,
        "should have fold payoff entries for each fold terminal"
    );

    // Showdown equity tables should have entries for showdown terminals
    let num_showdown_terminals = flat
        .terminal_indices
        .iter()
        .filter(|&&i| flat.node_types[i as usize] == NodeType::TerminalShowdown)
        .count();
    assert_eq!(
        flat.equity_tables.len(),
        num_showdown_terminals,
        "should have equity table entries for each showdown terminal"
    );

    // CSR child offsets should be consistent
    assert_eq!(flat.child_offsets.len(), flat.num_nodes() + 1);
    for i in 0..flat.num_nodes() {
        let n_children = flat.num_children(i);
        if flat.is_terminal(i) {
            assert_eq!(n_children, 0, "terminal node {i} should have 0 children");
        } else {
            assert!(n_children > 0, "decision node {i} should have children");
        }
    }

    // Verify infoset consistency
    for i in 0..flat.num_nodes() {
        if flat.is_terminal(i) {
            assert_eq!(flat.infoset_ids[i], u32::MAX);
        } else {
            let iset = flat.infoset_ids[i] as usize;
            assert!(iset < flat.num_infosets);
            assert_eq!(
                flat.infoset_num_actions[iset] as usize,
                flat.num_children(i),
                "infoset {iset} action count mismatch at node {i}"
            );
        }
    }
}

#[test]
fn test_flat_tree_pot_sizes() {
    let mut game = make_river_game();
    let flat = FlatTree::from_postflop_game(&mut game);

    // All pot sizes should be >= starting pot
    for (i, &pot) in flat.pots.iter().enumerate() {
        assert!(
            pot >= 100.0,
            "node {i} pot {pot} should be >= starting pot 100"
        );
    }

    // Children should have pot >= parent's pot (bets only increase the pot)
    for i in 0..flat.num_nodes() {
        let start = flat.child_offsets[i] as usize;
        let end = flat.child_offsets[i + 1] as usize;
        for &child_id in &flat.children[start..end] {
            assert!(
                flat.pots[child_id as usize] >= flat.pots[i],
                "child {} pot {} < parent {} pot {}",
                child_id,
                flat.pots[child_id as usize],
                i,
                flat.pots[i]
            );
        }
    }
}

#[test]
fn test_flat_tree_fold_payoffs_sign() {
    let mut game = make_river_game();
    let flat = FlatTree::from_postflop_game(&mut game);

    // Each fold payoff entry should have [amount_win > 0, amount_lose < 0, player in {0, 1}]
    for payoff in &flat.fold_payoffs {
        if payoff.is_empty() {
            continue; // placeholder for showdown terminals
        }
        assert_eq!(payoff.len(), 3, "fold payoff should have 3 elements");
        assert!(
            payoff[0] > 0.0,
            "amount_win should be positive, got {}",
            payoff[0]
        );
        assert!(
            payoff[1] < 0.0,
            "amount_lose should be negative, got {}",
            payoff[1]
        );
        let folded = payoff[2] as usize;
        assert!(
            folded == 0 || folded == 1,
            "folded_player should be 0 or 1, got {folded}"
        );
    }
}

#[test]
fn test_flat_tree_showdown_payoffs() {
    let mut game = make_river_game();
    let flat = FlatTree::from_postflop_game(&mut game);

    // Each equity table entry should have [amount_win > 0, amount_lose < 0]
    for eq in &flat.equity_tables {
        assert_eq!(eq.len(), 2);
        assert!(eq[0] > 0.0, "showdown amount_win should be positive");
        assert!(eq[1] < 0.0, "showdown amount_lose should be negative");
    }
}
