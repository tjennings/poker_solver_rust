//! Spot resolution for multiplayer game trees.
//!
//! Resolves position-aware spot notation strings (e.g. `"utg:5bb,hj:fold"`)
//! by walking an `MpGameTree`.

use poker_solver_core::blueprint_mp::game_tree::{MpGameNode, MpGameTree, TreeAction};
use poker_solver_core::poker::{self, Card};

/// Walk the MP game tree following a position-aware spot string.
/// Returns (node_idx, board_cards) or None if any action fails to match.
pub fn resolve_mp_spot(
    tree: &MpGameTree,
    spot: &str,
    num_players: u8,
) -> Option<(u32, Vec<Card>)> {
    let spot = spot.trim();
    if spot.is_empty() {
        return Some((tree.root, vec![]));
    }
    let mut node_idx = tree.root;
    let mut board = Vec::new();
    for segment in spot.split('|') {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        if segment.contains(':') {
            node_idx = resolve_action_segment(tree, node_idx, segment, num_players)?;
        } else {
            parse_board_segment(segment, &mut board)?;
        }
    }
    node_idx = skip_chance_mp(tree, node_idx);
    Some((node_idx, board))
}

fn resolve_action_segment(
    tree: &MpGameTree,
    mut node_idx: u32,
    segment: &str,
    num_players: u8,
) -> Option<u32> {
    for pair in segment.split(',') {
        let pair = pair.trim();
        let (pos_name, label) = pair.split_once(':')?;
        let _expected_seat = position_to_seat(pos_name.trim(), num_players)?;
        node_idx = skip_chance_mp(tree, node_idx);
        let MpGameNode::Decision {
            ref actions,
            ref children,
            ..
        } = tree.nodes[node_idx as usize]
        else {
            return None;
        };
        let matched = match_action_label_mp(label.trim(), actions)?;
        node_idx = children[matched];
    }
    Some(node_idx)
}

fn match_action_label_mp(label: &str, actions: &[TreeAction]) -> Option<usize> {
    let lower = label.to_ascii_lowercase();
    match lower.as_str() {
        "fold" => return actions.iter().position(|a| matches!(a, TreeAction::Fold)),
        "check" => return actions.iter().position(|a| matches!(a, TreeAction::Check)),
        "call" => return actions.iter().position(|a| matches!(a, TreeAction::Call)),
        "all-in" => return actions.iter().position(|a| matches!(a, TreeAction::AllIn)),
        _ => {}
    }
    if let Some(bb_str) = lower.strip_suffix("bb") {
        let bb_val: f64 = bb_str.parse().ok()?;
        let chips = bb_val * 2.0;
        return find_closest_sized_action(actions, chips);
    }
    if let Some(x_str) = lower.strip_suffix('x') {
        let _mult: f64 = x_str.parse().ok()?;
        return find_closest_multiplier(actions, _mult);
    }
    None
}

/// Find the index of the Lead/Raise action closest to `target_chips`.
/// Returns None if no sized action is within tolerance (20% + 1 chip).
fn find_closest_sized_action(actions: &[TreeAction], target_chips: f64) -> Option<usize> {
    let (idx, chips) = closest_sized_to(actions, target_chips)?;
    let diff = (chips - target_chips).abs();
    if diff <= target_chips * 0.2 + 1.0 { Some(idx) } else { None }
}

/// Find the closest Lead/Raise by multiplier label. Without game-state
/// context, picks the single closest sized action if one exists.
fn find_closest_multiplier(actions: &[TreeAction], _mult: f64) -> Option<usize> {
    closest_sized_to(actions, 0.0).map(|(idx, _)| idx)
}

/// Return (index, chip_value) of the Lead/Raise closest to `target`.
fn closest_sized_to(actions: &[TreeAction], target: f64) -> Option<(usize, f64)> {
    actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| match a {
            TreeAction::Lead(v) | TreeAction::Raise(v) => Some((i, *v)),
            _ => None,
        })
        .min_by(|(_, a), (_, b)| {
            let da = (*a - target).abs();
            let db = (*b - target).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn parse_board_segment(segment: &str, board: &mut Vec<Card>) -> Option<()> {
    let chars: Vec<char> = segment.chars().collect();
    for pair in chars.chunks(2) {
        if pair.len() == 2 {
            let card_str: String = pair.iter().collect();
            let card = poker::parse_card(&card_str)?;
            board.push(card);
        }
    }
    Some(())
}

fn skip_chance_mp(tree: &MpGameTree, mut node_idx: u32) -> u32 {
    while let MpGameNode::Chance { child, .. } = tree.nodes[node_idx as usize] {
        node_idx = child;
    }
    node_idx
}

/// Map a position name to a seat index for the given number of players.
///
/// Standard 6-max layout (SB=4, BB=5, BTN=3):
///   seat 0: UTG, seat 1: HJ, seat 2: CO, seat 3: BTN, seat 4: SB, seat 5: BB
fn position_to_seat(name: &str, num_players: u8) -> Option<u8> {
    let lower = name.to_ascii_lowercase();
    let n = num_players;
    match lower.as_str() {
        "sb" => Some(n - 2),
        "bb" => Some(n - 1),
        "btn" | "bu" => {
            if n >= 3 { Some(n - 3) } else { None }
        }
        "co" => {
            if n >= 4 { Some(n - 4) } else { None }
        }
        "hj" => {
            if n >= 5 { Some(n - 5) } else { None }
        }
        "utg" => Some(0),
        "utg1" | "utg+1" => {
            if n >= 7 { Some(1) } else { None }
        }
        "utg2" | "utg+2" => {
            if n >= 8 { Some(2) } else { None }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_solver_core::blueprint_mp::config::*;
    use poker_solver_core::blueprint_mp::game_tree::MpGameTree;
    use test_macros::timed_test;

    fn yaml_f64(v: f64) -> serde_yaml::Value {
        serde_yaml::Value::Number(serde_yaml::Number::from(v))
    }

    /// Build a 6-player tree with a 5bb preflop raise and 0.67x postflop lead.
    fn test_6p_tree() -> MpGameTree {
        let game = MpGameConfig {
            name: "test".into(),
            num_players: 6,
            stack_depth: 200.0,
            blinds: vec![
                ForcedBet { seat: 4, kind: ForcedBetKind::SmallBlind, amount: 1.0 },
                ForcedBet { seat: 5, kind: ForcedBetKind::BigBlind, amount: 2.0 },
            ],
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        let preflop = MpStreetSizes {
            lead: vec![serde_yaml::Value::String("5bb".into())],
            raise: vec![vec![serde_yaml::Value::String("5bb".into())]],
        };
        let postflop = MpStreetSizes {
            lead: vec![yaml_f64(0.67)],
            raise: vec![vec![yaml_f64(1.0)]],
        };
        let action = MpActionAbstractionConfig {
            preflop,
            flop: postflop.clone(),
            turn: postflop.clone(),
            river: postflop,
        };
        MpGameTree::build(&game, &action)
    }

    // -- resolve_mp_spot tests --

    #[timed_test(10)]
    fn resolve_empty_returns_root() {
        let tree = test_6p_tree();
        let (idx, board) = resolve_mp_spot(&tree, "", 6).unwrap();
        assert_eq!(idx, tree.root);
        assert!(board.is_empty());
    }

    #[timed_test(10)]
    fn resolve_single_fold() {
        let tree = test_6p_tree();
        let result = resolve_mp_spot(&tree, "utg:fold", 6);
        assert!(result.is_some());
        let (idx, _) = result.unwrap();
        assert_ne!(idx, tree.root);
        if let MpGameNode::Decision { seat, .. } = &tree.nodes[idx as usize] {
            assert_eq!(seat.index(), 1, "HJ is seat 1");
        } else {
            panic!("expected Decision node for HJ");
        }
    }

    #[timed_test(10)]
    fn resolve_utg_open() {
        let tree = test_6p_tree();
        let result = resolve_mp_spot(&tree, "utg:5bb", 6);
        assert!(result.is_some(), "utg:5bb should resolve");
        let (idx, _) = result.unwrap();
        assert_ne!(idx, tree.root);
        if let MpGameNode::Decision { seat, .. } = &tree.nodes[idx as usize] {
            assert_eq!(seat.index(), 1, "HJ is seat 1");
        } else {
            panic!("expected Decision node for HJ");
        }
    }

    #[timed_test(10)]
    fn resolve_full_fold_sequence() {
        // 4 folds (UTG, HJ, CO, BTN) leave SB and BB active.
        let tree = test_6p_tree();
        let result = resolve_mp_spot(
            &tree,
            "utg:fold,hj:fold,co:fold,btn:fold",
            6,
        );
        assert!(result.is_some());
        let (idx, _) = result.unwrap();
        if let MpGameNode::Decision { seat, .. } = &tree.nodes[idx as usize] {
            assert_eq!(seat.index(), 4, "SB is seat 4");
        } else {
            panic!("expected Decision node for SB after 4 folds");
        }
    }

    #[timed_test(10)]
    fn resolve_invalid_position_returns_none() {
        let tree = test_6p_tree();
        let result = resolve_mp_spot(&tree, "xyz:5bb", 6);
        assert!(result.is_none());
    }

    #[timed_test(10)]
    fn resolve_invalid_action_returns_none() {
        let tree = test_6p_tree();
        let result = resolve_mp_spot(&tree, "utg:999bb", 6);
        assert!(result.is_none());
    }

    // -- match_action_label_mp tests --

    #[timed_test]
    fn match_action_fold() {
        let actions = vec![TreeAction::Fold, TreeAction::Call, TreeAction::Lead(10.0)];
        assert_eq!(match_action_label_mp("fold", &actions), Some(0));
    }

    #[timed_test]
    fn match_action_check() {
        let actions = vec![TreeAction::Check, TreeAction::Lead(10.0)];
        assert_eq!(match_action_label_mp("check", &actions), Some(0));
    }

    #[timed_test]
    fn match_action_bb_label() {
        // 5bb = 10 chips (5 * 2), should match Lead(10.0)
        let actions = vec![
            TreeAction::Fold,
            TreeAction::Call,
            TreeAction::Lead(10.0),
        ];
        assert_eq!(match_action_label_mp("5bb", &actions), Some(2));
    }

    #[timed_test]
    fn match_action_call() {
        let actions = vec![TreeAction::Fold, TreeAction::Call, TreeAction::AllIn];
        assert_eq!(match_action_label_mp("call", &actions), Some(1));
    }

    #[timed_test]
    fn match_action_all_in() {
        let actions = vec![TreeAction::Fold, TreeAction::Call, TreeAction::AllIn];
        assert_eq!(match_action_label_mp("all-in", &actions), Some(2));
    }

    #[timed_test]
    fn match_action_no_match() {
        let actions = vec![TreeAction::Fold, TreeAction::Call];
        assert_eq!(match_action_label_mp("check", &actions), None);
    }

    #[timed_test]
    fn match_action_bb_raise() {
        // 5bb = 10 chips, should also match Raise(10.0)
        let actions = vec![TreeAction::Fold, TreeAction::Call, TreeAction::Raise(10.0)];
        assert_eq!(match_action_label_mp("5bb", &actions), Some(2));
    }

    #[timed_test]
    fn match_action_bb_out_of_tolerance() {
        // 999bb = 1998 chips, no action even close
        let actions = vec![TreeAction::Fold, TreeAction::Lead(10.0)];
        assert_eq!(match_action_label_mp("999bb", &actions), None);
    }

    // -- position_to_seat tests --

    #[timed_test]
    fn position_to_seat_6max() {
        assert_eq!(position_to_seat("utg", 6), Some(0));
        assert_eq!(position_to_seat("hj", 6), Some(1));
        assert_eq!(position_to_seat("co", 6), Some(2));
        assert_eq!(position_to_seat("btn", 6), Some(3));
        assert_eq!(position_to_seat("sb", 6), Some(4));
        assert_eq!(position_to_seat("bb", 6), Some(5));
    }

    #[timed_test]
    fn position_to_seat_case_insensitive() {
        assert_eq!(position_to_seat("UTG", 6), Some(0));
        assert_eq!(position_to_seat("Sb", 6), Some(4));
        assert_eq!(position_to_seat("BB", 6), Some(5));
    }

    #[timed_test]
    fn position_to_seat_invalid() {
        assert_eq!(position_to_seat("xyz", 6), None);
        assert_eq!(position_to_seat("", 6), None);
    }

    #[timed_test]
    fn position_to_seat_hj_requires_5_players() {
        assert_eq!(position_to_seat("hj", 4), None);
        assert_eq!(position_to_seat("hj", 5), Some(0));
    }

    #[timed_test]
    fn position_to_seat_co_requires_4_players() {
        assert_eq!(position_to_seat("co", 3), None);
        assert_eq!(position_to_seat("co", 4), Some(0));
    }

    // -- parse_board_segment tests --

    #[timed_test]
    fn parse_board_segment_flop() {
        let mut board = Vec::new();
        parse_board_segment("Kh7s2d", &mut board).unwrap();
        assert_eq!(board.len(), 3);
    }

    #[timed_test]
    fn parse_board_segment_invalid_card() {
        let mut board = Vec::new();
        let result = parse_board_segment("Xh7s2d", &mut board);
        assert!(result.is_none());
    }

    #[timed_test]
    fn parse_board_segment_empty() {
        let mut board = Vec::new();
        parse_board_segment("", &mut board).unwrap();
        assert!(board.is_empty());
    }
}
