//! Compute turn entry ranges from a saved blueprint strategy.
//!
//! Given a blueprint bundle (config.yaml + strategy.bin + bucket files),
//! sample a random action path through preflop+flop and compute reach-weighted
//! ranges at the turn entry. Produces ranges with ~200-400 non-zero combos
//! (realistic) instead of RSP's ~1000.

use std::path::{Path, PathBuf};

use poker_solver_core::blueprint_v2::Street;
use poker_solver_core::blueprint_v2::bundle::{BlueprintV2Strategy, load_config};
use poker_solver_core::blueprint_v2::full_depth_solver::rs_poker_card_to_id;
use poker_solver_core::blueprint_v2::game_tree::{GameNode, GameTree, TreeAction};
use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
use poker_solver_core::hands::{CanonicalHand, all_hands};
use poker_solver_core::poker::Card;
use rand::Rng;
use range_solver::card::card_pair_to_index;

pub const NUM_COMBOS: usize = 1326;

/// Blueprint-based range generator for datagen.
///
/// Loads a blueprint bundle once, then for each sample, walks a random
/// action path through preflop+flop to produce realistic turn entry ranges.
pub struct BlueprintRangeGenerator {
    strategy: BlueprintV2Strategy,
    tree: GameTree,
    decision_map: Vec<u32>,
    buckets: AllBuckets,
    starting_stack: f64,
    small_blind: f64,
    big_blind: f64,
}

/// A sampled turn situation with blueprint-derived ranges.
#[derive(Debug, Clone)]
pub struct BlueprintSituation {
    pub oop_range: [f32; NUM_COMBOS],
    pub ip_range: [f32; NUM_COMBOS],
}

/// A reached river spot sampled from a full blueprint path.
#[derive(Debug, Clone)]
pub struct BlueprintRiverSpot {
    pub oop_range: [f32; NUM_COMBOS],
    pub ip_range: [f32; NUM_COMBOS],
    pub pot: f64,
    pub effective_stack: f64,
    pub line: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
struct PathState {
    pot: f64,
    stacks: [f64; 2],
    street_bets: [f64; 2],
}

/// Find the latest snapshot directory (highest numbered snapshot_NNNN).
fn find_latest_snapshot(bundle_dir: &Path) -> Result<PathBuf, String> {
    let mut snapshots: Vec<(u32, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(bundle_dir).map_err(|e| format!("read dir: {e}"))? {
        let entry = entry.map_err(|e| format!("entry: {e}"))?;
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(num_str) = name.strip_prefix("snapshot_") {
            if let Ok(num) = num_str.parse::<u32>() {
                snapshots.push((num, entry.path()));
            }
        }
    }
    snapshots.sort_by_key(|(n, _)| *n);
    snapshots
        .last()
        .map(|(_, p)| p.clone())
        .ok_or_else(|| "no snapshots found".to_string())
}

/// Convert a u8 card ID to an rs_poker Card.
fn u8_to_card(id: u8) -> Card {
    use poker_solver_core::poker::{Suit, Value};
    let rank = id / 4;
    let suit_id = id % 4;
    let value = Value::from(rank);
    let suit = match suit_id {
        0 => Suit::Club,
        1 => Suit::Diamond,
        2 => Suit::Heart,
        3 => Suit::Spade,
        _ => unreachable!(),
    };
    Card::new(value, suit)
}

impl BlueprintRangeGenerator {
    /// Load a blueprint bundle from disk.
    ///
    /// Reads config.yaml, the latest snapshot's strategy.bin, bucket files,
    /// and builds the game tree.
    pub fn load(bundle_dir: &Path) -> Result<Self, String> {
        let config = load_config(bundle_dir).map_err(|e| format!("load config: {e}"))?;

        let snap_dir = find_latest_snapshot(bundle_dir)?;

        let mut strategy = BlueprintV2Strategy::load(&snap_dir.join("strategy.bin"))
            .map_err(|e| format!("load strategy: {e}"))?;
        strategy.post_deserialize();

        let tree = GameTree::build_with_options(
            config.game.stack_depth,
            config.game.small_blind,
            config.game.big_blind,
            &config.action_abstraction.preflop,
            &config.action_abstraction.flop,
            &config.action_abstraction.turn,
            &config.action_abstraction.river,
            config.game.allow_preflop_limp,
        );
        let decision_map = tree.decision_index_map();

        // Load bucket files — try multiple locations in priority order.
        let candidates = [
            snap_dir.join("buckets"),   // snapshot/buckets (copied during training)
            bundle_dir.join("buckets"), // bundle/buckets
            config
                .training
                .cluster_path
                .as_ref() // cluster_path from config (relative to CWD)
                .map(PathBuf::from)
                .unwrap_or_default(),
            config
                .training
                .cluster_path
                .as_ref() // cluster_path relative to bundle_dir
                .map(|cp| bundle_dir.join(cp))
                .unwrap_or_default(),
        ];
        let actual_buckets_dir = candidates
            .iter()
            .find(|p| p.join("flop.buckets").exists())
            .cloned()
            .unwrap_or_else(|| {
                eprintln!(
                    "[blueprint ranges] warning: no bucket files found in any candidate directory"
                );
                bundle_dir.join("buckets")
            });

        let bucket_files =
            poker_solver_core::blueprint_v2::trainer::load_bucket_files(&actual_buckets_dir);
        let bucket_counts = [
            config.clustering.preflop.buckets,
            config.clustering.flop.buckets,
            config.clustering.turn.buckets,
            config.clustering.river.buckets,
        ];
        let buckets = AllBuckets::new(bucket_counts, bucket_files);

        let loaded_streets = (0..4)
            .filter(|&i| buckets.bucket_files[i].is_some())
            .count();
        eprintln!("[blueprint ranges] loaded from {}", bundle_dir.display());
        eprintln!(
            "[blueprint ranges] tree: {} nodes, buckets: {}/4 streets loaded",
            tree.nodes.len(),
            loaded_streets,
        );

        Ok(Self {
            strategy,
            tree,
            decision_map,
            buckets,
            starting_stack: config.game.stack_depth,
            small_blind: config.game.small_blind,
            big_blind: config.game.big_blind,
        })
    }

    pub fn strategy(&self) -> &BlueprintV2Strategy {
        &self.strategy
    }
    pub fn tree(&self) -> &GameTree {
        &self.tree
    }
    pub fn decision_map(&self) -> &[u32] {
        &self.decision_map
    }

    pub fn starting_stack(&self) -> f64 {
        self.starting_stack
    }

    /// Sample a reached river spot by walking one blueprint action path through
    /// preflop, flop, and turn. The returned ranges are board-blocked and
    /// normalized to unit mass for each player.
    pub fn sample_river_spot<R: Rng>(
        &self,
        board: &[u8; 5],
        rng: &mut R,
    ) -> Option<BlueprintRiverSpot> {
        let board_cards: Vec<Card> = board.iter().map(|&c| u8_to_card(c)).collect();

        let mut oop_weights = [1.0f32; NUM_COMBOS];
        let mut ip_weights = [1.0f32; NUM_COMBOS];
        let mut node_idx = self.tree.root;
        let mut line = Vec::new();
        let mut state = PathState {
            pot: self.small_blind + self.big_blind,
            stacks: [
                self.starting_stack - self.small_blind,
                self.starting_stack - self.big_blind,
            ],
            street_bets: [self.small_blind, self.big_blind],
        };

        loop {
            match &self.tree.nodes[node_idx as usize] {
                GameNode::Terminal { .. } => return None,
                GameNode::Chance { next_street, child } => {
                    state.street_bets = [0.0; 2];
                    line.push(format!("{next_street:?}"));
                    node_idx = *child;
                }
                GameNode::Decision {
                    player,
                    street,
                    actions,
                    children,
                    ..
                } => {
                    if *street == Street::River {
                        let effective_stack = state.stacks[0].min(state.stacks[1]);
                        if effective_stack <= 0.0 {
                            return None;
                        }
                        normalize_ranges_for_board(&mut oop_weights, &mut ip_weights, board)?;
                        return Some(BlueprintRiverSpot {
                            oop_range: oop_weights,
                            ip_range: ip_weights,
                            pot: state.pot,
                            effective_stack,
                            line,
                        });
                    }

                    let chosen = self.sample_action_for_node(
                        node_idx as usize,
                        *player,
                        *street,
                        actions,
                        &oop_weights,
                        &ip_weights,
                        &board_cards,
                        rng,
                    )?;

                    let action = actions[chosen];
                    self.apply_action_weights(
                        node_idx as usize,
                        *player,
                        *street,
                        chosen,
                        &mut oop_weights,
                        &mut ip_weights,
                        &board_cards,
                    );
                    apply_path_action(&mut state, *player as usize, action);
                    line.push(action_label(action));
                    node_idx = children[chosen];
                }
            }
        }
    }

    /// Sample turn entry ranges by walking ONE random action path through
    /// preflop and flop, weighted by the blueprint's average strategy.
    ///
    /// `board` is a 4-card turn board as u8 IDs (needed for flop bucket lookups).
    ///
    /// Returns `None` if the sampled path leads to a terminal (fold/all-in)
    /// before reaching the turn.
    pub fn sample_turn_ranges<R: Rng>(
        &self,
        board: &[u8],
        rng: &mut R,
    ) -> Option<BlueprintSituation> {
        let board_cards: Vec<Card> = board.iter().map(|&c| u8_to_card(c)).collect();

        let mut oop_weights = [1.0f32; NUM_COMBOS];
        let mut ip_weights = [1.0f32; NUM_COMBOS];
        let mut node_idx = self.tree.root;

        loop {
            match &self.tree.nodes[node_idx as usize] {
                GameNode::Terminal { .. } => {
                    // Hand ended before turn — retry
                    return None;
                }
                GameNode::Chance { child, .. } => {
                    // Street transition — just pass through
                    node_idx = *child;
                }
                GameNode::Decision {
                    player,
                    street,
                    actions,
                    children,
                    ..
                } => {
                    if *street == Street::Turn || *street == Street::River {
                        // Reached the turn — done
                        break;
                    }

                    let dec_idx = self.decision_map[node_idx as usize];
                    if dec_idx == u32::MAX {
                        return None;
                    }

                    let board_slice: &[Card] = match street {
                        Street::Preflop => &[],
                        Street::Flop => &board_cards[..3.min(board_cards.len())],
                        _ => &board_cards,
                    };

                    // Compute average action probability across all live combos
                    // to weight-sample which action to take.
                    let num_actions = actions.len();
                    let mut action_weights = vec![0.0f64; num_actions];

                    let weights = if *player == self.tree.dealer {
                        &ip_weights
                    } else {
                        &oop_weights
                    };

                    for hand in all_hands() {
                        let bucket = if *street == Street::Preflop {
                            if self.strategy.bucket_counts[0] == 169 {
                                hand.index() as u16
                            } else {
                                (hand.index() % self.strategy.bucket_counts[0] as usize) as u16
                            }
                        } else {
                            // Need a representative combo for this hand's bucket lookup.
                            // Use the first non-blocked combo.
                            let mut bucket = 0u16;
                            for (c0, c1) in hand.combos() {
                                if board_slice.iter().any(|b| *b == c0 || *b == c1) {
                                    continue;
                                }
                                bucket = self.buckets.get_bucket(*street, [c0, c1], board_slice);
                                break;
                            }
                            bucket
                        };

                        let probs = self.strategy.get_action_probs(dec_idx as usize, bucket);

                        // Weight by the sum of this hand's combo weights
                        let mut hand_weight = 0.0f64;
                        for (c0, c1) in hand.combos() {
                            if board_slice.iter().any(|b| *b == c0 || *b == c1) {
                                continue;
                            }
                            let ci = card_pair_to_index(
                                rs_poker_card_to_id(c0),
                                rs_poker_card_to_id(c1),
                            );
                            hand_weight += weights[ci] as f64;
                        }

                        for (a, &p) in probs.iter().enumerate().take(num_actions) {
                            action_weights[a] += p as f64 * hand_weight;
                        }
                    }

                    // Weighted sample an action
                    let total: f64 = action_weights.iter().sum();
                    if total <= 0.0 {
                        return None;
                    }
                    let mut draw = rng.gen_range(0.0..total);
                    let mut chosen = 0;
                    for (a, &w) in action_weights.iter().enumerate() {
                        draw -= w;
                        if draw <= 0.0 {
                            chosen = a;
                            break;
                        }
                    }

                    // Multiply acting player's weights by the action probability
                    let weights_mut = if *player == self.tree.dealer {
                        &mut ip_weights
                    } else {
                        &mut oop_weights
                    };

                    for hand in all_hands() {
                        let bucket = if *street == Street::Preflop {
                            if self.strategy.bucket_counts[0] == 169 {
                                hand.index() as u16
                            } else {
                                (hand.index() % self.strategy.bucket_counts[0] as usize) as u16
                            }
                        } else {
                            let mut bucket = 0u16;
                            for (c0, c1) in hand.combos() {
                                if board_slice.iter().any(|b| *b == c0 || *b == c1) {
                                    continue;
                                }
                                bucket = self.buckets.get_bucket(*street, [c0, c1], board_slice);
                                break;
                            }
                            bucket
                        };

                        let probs = self.strategy.get_action_probs(dec_idx as usize, bucket);
                        let p = probs.get(chosen).copied().unwrap_or(0.0);

                        for (c0, c1) in hand.combos() {
                            if board_slice.iter().any(|b| *b == c0 || *b == c1) {
                                continue;
                            }
                            let ci = card_pair_to_index(
                                rs_poker_card_to_id(c0),
                                rs_poker_card_to_id(c1),
                            );
                            weights_mut[ci] *= p;
                        }
                    }

                    // Log the first few propagation steps.
                    use std::sync::atomic::{AtomicU32, Ordering as AO};
                    static LOG_COUNT: AtomicU32 = AtomicU32::new(0);
                    let lc = LOG_COUNT.fetch_add(1, AO::Relaxed);
                    if lc < 20 {
                        let oop_nz = oop_weights.iter().filter(|&&w| w > 0.01).count();
                        let ip_nz = ip_weights.iter().filter(|&&w| w > 0.01).count();
                        eprintln!(
                            "[propagate] street={street:?} player={player} action={chosen}/{num_actions} oop_nz={oop_nz} ip_nz={ip_nz}"
                        );
                    }

                    node_idx = children[chosen];
                }
            }
        }

        Some(BlueprintSituation {
            oop_range: oop_weights,
            ip_range: ip_weights,
        })
    }

    fn sample_action_for_node<R: Rng>(
        &self,
        node_idx: usize,
        player: u8,
        street: Street,
        actions: &[TreeAction],
        oop_weights: &[f32; NUM_COMBOS],
        ip_weights: &[f32; NUM_COMBOS],
        board_cards: &[Card],
        rng: &mut R,
    ) -> Option<usize> {
        let dec_idx = self.decision_map.get(node_idx)?;
        if *dec_idx == u32::MAX {
            return None;
        }
        self.sample_action_with_decision(
            *dec_idx as usize,
            player,
            street,
            actions,
            oop_weights,
            ip_weights,
            board_cards,
            rng,
        )
    }

    fn sample_action_with_decision<R: Rng>(
        &self,
        dec_idx: usize,
        player: u8,
        street: Street,
        actions: &[TreeAction],
        oop_weights: &[f32; NUM_COMBOS],
        ip_weights: &[f32; NUM_COMBOS],
        board_cards: &[Card],
        rng: &mut R,
    ) -> Option<usize> {
        let num_actions = actions.len();
        let mut action_weights = vec![0.0f64; num_actions];
        let weights = if player == self.tree.dealer {
            ip_weights
        } else {
            oop_weights
        };
        let board_slice = board_slice_for_street(street, board_cards);

        for hand in all_hands() {
            let bucket = self.bucket_for_hand(street, hand, board_slice);
            let probs = self.strategy.get_action_probs(dec_idx, bucket);
            let mut hand_weight = 0.0f64;
            for (c0, c1) in hand.combos() {
                if board_slice.iter().any(|b| *b == c0 || *b == c1) {
                    continue;
                }
                let ci = card_pair_to_index(rs_poker_card_to_id(c0), rs_poker_card_to_id(c1));
                hand_weight += f64::from(weights[ci]);
            }
            for (a, &p) in probs.iter().enumerate().take(num_actions) {
                action_weights[a] += f64::from(p) * hand_weight;
            }
        }

        let total: f64 = action_weights.iter().sum();
        if total <= 0.0 {
            return None;
        }
        let mut draw = rng.gen_range(0.0..total);
        for (a, &weight) in action_weights.iter().enumerate() {
            draw -= weight;
            if draw <= 0.0 {
                return Some(a);
            }
        }
        Some(num_actions.saturating_sub(1))
    }

    fn apply_action_weights(
        &self,
        node_idx: usize,
        player: u8,
        street: Street,
        chosen: usize,
        oop_weights: &mut [f32; NUM_COMBOS],
        ip_weights: &mut [f32; NUM_COMBOS],
        board_cards: &[Card],
    ) {
        let dec_idx = self.decision_map[node_idx] as usize;
        let weights_mut = if player == self.tree.dealer {
            ip_weights
        } else {
            oop_weights
        };
        let board_slice = board_slice_for_street(street, board_cards);
        for hand in all_hands() {
            let bucket = self.bucket_for_hand(street, hand, board_slice);
            let probs = self.strategy.get_action_probs(dec_idx, bucket);
            let p = probs.get(chosen).copied().unwrap_or(0.0);
            for (c0, c1) in hand.combos() {
                if board_slice.iter().any(|b| *b == c0 || *b == c1) {
                    continue;
                }
                let ci = card_pair_to_index(rs_poker_card_to_id(c0), rs_poker_card_to_id(c1));
                weights_mut[ci] *= p;
            }
        }
    }

    fn bucket_for_hand(&self, street: Street, hand: CanonicalHand, board_slice: &[Card]) -> u16 {
        if street == Street::Preflop {
            if self.strategy.bucket_counts[0] == 169 {
                hand.index() as u16
            } else {
                (hand.index() % self.strategy.bucket_counts[0] as usize) as u16
            }
        } else {
            for (c0, c1) in hand.combos() {
                if board_slice.iter().any(|b| *b == c0 || *b == c1) {
                    continue;
                }
                return self.buckets.get_bucket(street, [c0, c1], board_slice);
            }
            0
        }
    }
}

fn board_slice_for_street(street: Street, board_cards: &[Card]) -> &[Card] {
    match street {
        Street::Preflop => &[],
        Street::Flop => &board_cards[..3.min(board_cards.len())],
        Street::Turn => &board_cards[..4.min(board_cards.len())],
        Street::River => board_cards,
    }
}

fn action_label(action: TreeAction) -> String {
    match action {
        TreeAction::Fold => "fold".into(),
        TreeAction::Check => "check".into(),
        TreeAction::Call => "call".into(),
        TreeAction::AllIn => "allin".into(),
        TreeAction::Bet(v) => format!("bet:{v:.2}"),
        TreeAction::Raise(v) => format!("raise:{v:.2}"),
    }
}

fn apply_path_action(state: &mut PathState, actor: usize, action: TreeAction) {
    let opponent = 1 - actor;
    match action {
        TreeAction::Fold | TreeAction::Check => {}
        TreeAction::Call => {
            let call_amount = (state.street_bets[opponent] - state.street_bets[actor]).max(0.0);
            state.stacks[actor] -= call_amount;
            state.pot += call_amount;
            state.street_bets[actor] = state.street_bets[opponent];
        }
        TreeAction::AllIn => {
            let additional = state.stacks[actor].max(0.0);
            state.stacks[actor] = 0.0;
            state.pot += additional;
            state.street_bets[actor] += additional;
        }
        TreeAction::Bet(amount) | TreeAction::Raise(amount) => {
            let additional = (amount - state.street_bets[actor]).max(0.0);
            state.stacks[actor] -= additional;
            state.pot += additional;
            state.street_bets[actor] = amount;
        }
    }
}

fn normalize_ranges_for_board(
    oop: &mut [f32; NUM_COMBOS],
    ip: &mut [f32; NUM_COMBOS],
    board: &[u8; 5],
) -> Option<()> {
    let mut oop_sum = 0.0f32;
    let mut ip_sum = 0.0f32;
    for idx in 0..NUM_COMBOS {
        let (c0, c1) = range_solver::card::index_to_card_pair(idx);
        if board.contains(&c0) || board.contains(&c1) {
            oop[idx] = 0.0;
            ip[idx] = 0.0;
        }
        oop_sum += oop[idx];
        ip_sum += ip[idx];
    }
    if oop_sum <= 0.0 || ip_sum <= 0.0 {
        return None;
    }
    for idx in 0..NUM_COMBOS {
        oop[idx] /= oop_sum;
        ip[idx] /= ip_sum;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_latest_snapshot_picks_highest() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("snapshot_0002")).unwrap();
        std::fs::create_dir(dir.path().join("snapshot_0005")).unwrap();
        std::fs::create_dir(dir.path().join("snapshot_0001")).unwrap();
        let result = find_latest_snapshot(dir.path()).unwrap();
        assert!(result.ends_with("snapshot_0005"));
    }

    #[test]
    fn find_latest_snapshot_empty_dir_errors() {
        let dir = tempfile::tempdir().unwrap();
        assert!(find_latest_snapshot(dir.path()).is_err());
    }

    #[test]
    fn u8_to_card_roundtrip() {
        for id in 0u8..52 {
            let card = u8_to_card(id);
            let back = rs_poker_card_to_id(card);
            assert_eq!(id, back, "roundtrip failed for id {id}");
        }
    }
}
