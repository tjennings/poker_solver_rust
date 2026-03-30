//! Regret audit resolution: resolve a hand+spot config entry into the
//! concrete storage coordinates needed to read per-action regret values.

use std::collections::VecDeque;

use poker_solver_core::blueprint_v2::game_tree::{GameNode, GameTree};
use poker_solver_core::blueprint_v2::storage::BlueprintStorage;
use poker_solver_core::blueprint_v2::Street;
use poker_solver_core::hands::CanonicalHand;
use poker_solver_core::poker::{self, Card};
use poker_solver_core::showdown_equity::compute_equity;

use crate::blueprint_tui_config::PlayerLabel;
use crate::blueprint_tui_scenarios::{format_tree_action, resolve_spot};

/// Direction of a regret trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    Up,
    Down,
    Flat,
}

/// Snapshot for transfer to TUI thread.
#[derive(Debug, Clone)]
pub struct AuditSnapshot {
    pub regrets: Vec<f64>,
    pub deltas: Vec<f64>,
    pub trends: Vec<Trend>,
    /// Current strategy from regret matching (instantaneous).
    pub strategy: Vec<f64>,
    /// Average strategy from strategy sums (converged output, matches matrix).
    pub avg_strategy: Vec<f64>,
}

/// Resolved regret audit state for a single hand+spot.
pub struct ResolvedRegretAudit {
    #[allow(dead_code)]
    pub name: String,
    pub node_idx: u32,
    #[allow(dead_code)]
    pub player: u8,
    pub bucket: u16,
    pub bucket_trail: Vec<(Street, u16)>,
    pub action_labels: Vec<String>,
    pub num_actions: usize,
    pub regrets: Vec<f64>,
    pub prev_regrets: Vec<f64>,
    pub avg_strategy: Vec<f64>,
    pub trend_buffers: Vec<VecDeque<f64>>,
    pub error: Option<String>,
    trend_window: usize,
}

/// Build an error-state audit result.
fn error_audit(name: &str, msg: String, trend_window: usize) -> ResolvedRegretAudit {
    ResolvedRegretAudit {
        name: name.to_string(),
        node_idx: 0,
        player: 0,
        bucket: 0,
        bucket_trail: Vec::new(),
        action_labels: Vec::new(),
        num_actions: 0,
        regrets: Vec::new(),
        prev_regrets: Vec::new(),
        avg_strategy: Vec::new(),
        trend_buffers: Vec::new(),
        error: Some(msg),
        trend_window,
    }
}

/// Parse a hand string into two hole cards.
///
/// If the string is 4 characters, parse as a specific combo (e.g. "Ts9s").
/// Otherwise, parse as a canonical hand (e.g. "AKo") and pick the first
/// combo not blocked by the board.
fn parse_hand(
    hand_str: &str,
    board: &[Card],
) -> Result<(Card, Card, CanonicalHand), String> {
    let hand_str = hand_str.trim();
    if hand_str.len() == 4 {
        // Specific combo like "Ts9s"
        let c1 = poker::parse_card(&hand_str[0..2])
            .ok_or_else(|| format!("bad card: {}", &hand_str[0..2]))?;
        let c2 = poker::parse_card(&hand_str[2..4])
            .ok_or_else(|| format!("bad card: {}", &hand_str[2..4]))?;
        if board.contains(&c1) || board.contains(&c2) {
            return Err(format!("hand {hand_str} blocked by board"));
        }
        let canonical = CanonicalHand::from_cards(c1, c2);
        Ok((c1, c2, canonical))
    } else {
        // Canonical hand like "AKo"
        let canonical = CanonicalHand::parse(hand_str)
            .map_err(|e| format!("bad hand '{hand_str}': {e}"))?;
        let combo = canonical
            .combos()
            .into_iter()
            .find(|(c1, c2)| !board.contains(c1) && !board.contains(c2))
            .ok_or_else(|| format!("all combos of {hand_str} blocked by board"))?;
        Ok((combo.0, combo.1, canonical))
    }
}

/// Compute the equity-based bucket for a hand on a given street.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn equity_bucket(c1: Card, c2: Card, board: &[Card], num_buckets: u16) -> u16 {
    let equity = compute_equity([c1, c2], board);
    ((equity * f64::from(num_buckets)) as u16).min(num_buckets - 1)
}

/// Compute a bucket trail from preflop up to (and including) the target street.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn compute_bucket_trail(
    canonical: CanonicalHand,
    c1: Card,
    c2: Card,
    board: &[Card],
    target_street: Street,
    bucket_counts: &[u16; 4],
) -> Vec<(Street, u16)> {
    let mut trail = Vec::new();

    // Preflop bucket
    let preflop_buckets = bucket_counts[0];
    let preflop_bucket = (canonical.index() as u16) % preflop_buckets;
    trail.push((Street::Preflop, preflop_bucket));

    if target_street == Street::Preflop {
        return trail;
    }

    // Flop bucket (board[0..3])
    if board.len() >= 3 {
        let flop_bucket = equity_bucket(c1, c2, &board[..3], bucket_counts[1]);
        trail.push((Street::Flop, flop_bucket));
    }
    if target_street == Street::Flop {
        return trail;
    }

    // Turn bucket (board[0..4])
    if board.len() >= 4 {
        let turn_bucket = equity_bucket(c1, c2, &board[..4], bucket_counts[2]);
        trail.push((Street::Turn, turn_bucket));
    }
    if target_street == Street::Turn {
        return trail;
    }

    // River bucket (board[0..5])
    if board.len() >= 5 {
        let river_bucket = equity_bucket(c1, c2, &board[..5], bucket_counts[3]);
        trail.push((Street::River, river_bucket));
    }

    trail
}

/// Resolve a regret audit configuration into concrete storage coordinates.
///
/// On failure (bad spot, bad hand, blocked by board), returns a
/// `ResolvedRegretAudit` with the `error` field set.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub fn resolve_regret_audit(
    tree: &GameTree,
    storage: &BlueprintStorage,
    name: &str,
    spot: &str,
    hand_str: &str,
    player_label: PlayerLabel,
    trend_window: usize,
) -> ResolvedRegretAudit {
    // 1. Resolve spot to node + board
    let (node_idx, board) = match resolve_spot(tree, spot) {
        Some(result) => result,
        None => return error_audit(name, format!("bad spot: '{spot}'"), trend_window),
    };

    // 2. Get node info
    let (street, actions, node_player) = match &tree.nodes[node_idx as usize] {
        GameNode::Decision {
            street,
            actions,
            player,
            ..
        } => (*street, actions.clone(), *player),
        _ => {
            return error_audit(
                name,
                format!("spot '{spot}' does not resolve to a decision node"),
                trend_window,
            )
        }
    };

    // 3. Check player matches
    let expected_player = match player_label {
        PlayerLabel::Sb => tree.sb_seat(),
        PlayerLabel::Bb => tree.bb_seat(),
    };
    if node_player != expected_player {
        // Still proceed -- the user specified a player label but the node
        // belongs to the other player. Use the node's actual player.
    }

    // 4. Parse hand and compute bucket
    let (c1, c2, canonical) = match parse_hand(hand_str, &board) {
        Ok(result) => result,
        Err(msg) => return error_audit(name, msg, trend_window),
    };

    let street_idx = street as u8 as usize;
    let num_buckets = storage.bucket_counts[street_idx];
    let bucket = if street == Street::Preflop {
        (canonical.index() as u16) % num_buckets
    } else {
        equity_bucket(c1, c2, &board, num_buckets)
    };

    // 5. Compute bucket trail
    let bucket_trail = compute_bucket_trail(
        canonical,
        c1,
        c2,
        &board,
        street,
        &storage.bucket_counts,
    );

    // 6. Build action labels and read initial regrets
    let num_actions = actions.len();
    let action_labels: Vec<String> = actions.iter().map(format_tree_action).collect();
    let regrets: Vec<f64> = (0..num_actions)
        .map(|a| storage.get_regret(node_idx, bucket, a) as f64 / poker_solver_core::blueprint_v2::storage::REGRET_SCALE)
        .collect();
    let avg_strategy = storage.average_strategy(node_idx, bucket);

    ResolvedRegretAudit {
        name: name.to_string(),
        node_idx,
        player: node_player,
        bucket,
        bucket_trail,
        action_labels,
        num_actions,
        prev_regrets: regrets.clone(),
        regrets,
        avg_strategy,
        trend_buffers: vec![VecDeque::with_capacity(trend_window); num_actions],
        error: None,
        trend_window,
    }
}

impl ResolvedRegretAudit {
    /// Read current regrets from storage and update deltas and trend buffers.
    pub fn tick(&mut self, storage: &BlueprintStorage) {
        self.prev_regrets.clone_from(&self.regrets);
        for a in 0..self.num_actions {
            let raw = storage.get_regret(self.node_idx, self.bucket, a);
            self.regrets[a] = raw as f64 / poker_solver_core::blueprint_v2::storage::REGRET_SCALE;
        }
        self.avg_strategy = storage.average_strategy(self.node_idx, self.bucket);
        // Push magnitude deltas into trend ring buffers.
        // Positive = moving away from zero (strengthening), negative = decaying toward zero.
        for a in 0..self.num_actions {
            let mag_delta = self.regrets[a].abs() - self.prev_regrets[a].abs();
            let buf = &mut self.trend_buffers[a];
            if buf.len() == self.trend_window {
                buf.pop_front();
            }
            buf.push_back(mag_delta);
        }
    }

    /// Compute the delta for action `a` (current - previous regret).
    #[must_use]
    pub fn delta(&self, a: usize) -> f64 {
        self.regrets[a] - self.prev_regrets[a]
    }

    /// Regret-matching strategy from current regrets.
    ///
    /// Positive regrets normalized; uniform if all non-positive.
    #[must_use]
    pub fn strategy(&self) -> Vec<f64> {
        let positives: Vec<f64> = self.regrets.iter().map(|&r| r.max(0.0)).collect();
        let sum: f64 = positives.iter().sum();
        if sum > 0.0 {
            positives.iter().map(|&p| p / sum).collect()
        } else {
            vec![1.0 / self.num_actions as f64; self.num_actions]
        }
    }

    /// Trend direction for action `a` based on the ring buffer average.
    #[must_use]
    pub fn trend(&self, a: usize) -> Trend {
        let buf = &self.trend_buffers[a];
        if buf.is_empty() {
            return Trend::Flat;
        }
        let avg: f64 = buf.iter().sum::<f64>() / buf.len() as f64;
        if avg > 0.001 {
            Trend::Up
        } else if avg < -0.001 {
            Trend::Down
        } else {
            Trend::Flat
        }
    }

    /// Create a snapshot of the current state for transfer to the TUI thread.
    #[must_use]
    pub fn snapshot(&self) -> AuditSnapshot {
        AuditSnapshot {
            regrets: self.regrets.clone(),
            deltas: (0..self.num_actions).map(|a| self.delta(a)).collect(),
            trends: (0..self.num_actions).map(|a| self.trend(a)).collect(),
            strategy: self.strategy(),
            avg_strategy: self.avg_strategy.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    fn toy_tree() -> GameTree {
        GameTree::build(
            20.0,
            1.0,
            2.0,
            &[vec!["5bb".into()]],
            &[vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
        )
    }

    #[timed_test(10)]
    fn resolve_preflop_audit() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let audit = resolve_regret_audit(
            &tree, &storage, "AKo SB open", "", "AKo", PlayerLabel::Sb, 10,
        );
        assert!(audit.error.is_none(), "got: {:?}", audit.error);
        assert_eq!(audit.node_idx, tree.root);
        assert_eq!(audit.player, 0);
        assert!(!audit.action_labels.is_empty());
        assert_eq!(audit.num_actions, audit.action_labels.len());
        assert_eq!(audit.regrets.len(), audit.num_actions);
        assert_eq!(audit.bucket_trail.len(), 1);
        assert_eq!(audit.bucket_trail[0].0, Street::Preflop);
    }

    #[timed_test(10)]
    fn resolve_invalid_spot_returns_error() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let audit = resolve_regret_audit(
            &tree, &storage, "bad", "sb:999bb", "AKo", PlayerLabel::Sb, 10,
        );
        assert!(audit.error.is_some());
    }

    #[timed_test(10)]
    fn resolve_invalid_hand_returns_error() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let audit = resolve_regret_audit(
            &tree, &storage, "bad", "", "ZZo", PlayerLabel::Sb, 10,
        );
        assert!(audit.error.is_some());
    }

    #[timed_test(10)]
    fn tick_updates_regrets_and_deltas() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let mut audit = resolve_regret_audit(
            &tree, &storage, "AKo", "", "AKo", PlayerLabel::Sb, 10,
        );
        assert!(audit.error.is_none());
        // All regrets start at zero
        for &r in &audit.regrets {
            assert_eq!(r, 0.0);
        }
        // Simulate regret accumulation (values in scaled units: chip_value * REGRET_SCALE)
        let scale = poker_solver_core::blueprint_v2::storage::REGRET_SCALE as i32;
        storage.add_regret(audit.node_idx, audit.bucket, 0, 500 * scale);
        storage.add_regret(audit.node_idx, audit.bucket, 1, -300 * scale);
        audit.tick(&storage);
        assert!((audit.regrets[0] - 500.0).abs() < 0.01);
        assert!((audit.regrets[1] - (-300.0)).abs() < 0.01);
        assert!((audit.delta(0) - 500.0).abs() < 0.01);
    }

    #[timed_test(10)]
    fn strategy_from_regrets() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let mut audit = resolve_regret_audit(
            &tree, &storage, "AKo", "", "AKo", PlayerLabel::Sb, 10,
        );
        assert!(audit.error.is_none());
        let scale = poker_solver_core::blueprint_v2::storage::REGRET_SCALE as i32;
        storage.add_regret(audit.node_idx, audit.bucket, 0, 0);
        storage.add_regret(audit.node_idx, audit.bucket, 1, 100 * scale);
        storage.add_regret(audit.node_idx, audit.bucket, 2, 300 * scale);
        audit.tick(&storage);
        let strat = audit.strategy();
        assert!(strat[0] < 0.01); // fold=0
        assert!((strat[1] - 0.25).abs() < 0.01); // call=25%
        assert!((strat[2] - 0.75).abs() < 0.01); // raise=75%
    }

    #[timed_test(10)]
    fn trend_detection() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let mut audit = resolve_regret_audit(
            &tree, &storage, "AKo", "", "AKo", PlayerLabel::Sb, 3,
        );
        assert!(audit.error.is_none());
        let scale = poker_solver_core::blueprint_v2::storage::REGRET_SCALE as i32;
        for _ in 0..3 {
            storage.add_regret(audit.node_idx, audit.bucket, 0, 100 * scale);
            audit.tick(&storage);
        }
        assert_eq!(audit.trend(0), Trend::Up);
        assert_eq!(audit.trend(1), Trend::Flat);
    }

    #[timed_test(10)]
    fn snapshot_captures_state() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        let mut audit = resolve_regret_audit(
            &tree, &storage, "AKo", "", "AKo", PlayerLabel::Sb, 10,
        );
        let scale = poker_solver_core::blueprint_v2::storage::REGRET_SCALE as i32;
        storage.add_regret(audit.node_idx, audit.bucket, 0, 100 * scale);
        audit.tick(&storage);
        let snap = audit.snapshot();
        assert_eq!(snap.regrets.len(), audit.num_actions);
        assert_eq!(snap.deltas.len(), audit.num_actions);
        assert_eq!(snap.trends.len(), audit.num_actions);
        assert_eq!(snap.strategy.len(), audit.num_actions);
    }
}
