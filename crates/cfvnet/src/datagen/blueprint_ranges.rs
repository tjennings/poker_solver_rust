//! Compute turn entry ranges from a saved blueprint strategy.
//!
//! Given a blueprint bundle (config.yaml + strategy.bin + bucket files),
//! sample a random action path through preflop+flop and compute reach-weighted
//! ranges at the turn entry. Produces ranges with ~200-400 non-zero combos
//! (realistic) instead of RSP's ~1000.

use std::path::{Path, PathBuf};

use poker_solver_core::blueprint_v2::bundle::{load_config, BlueprintV2Strategy};
use poker_solver_core::blueprint_v2::full_depth_solver::rs_poker_card_to_id;
use poker_solver_core::blueprint_v2::game_tree::{GameNode, GameTree};
use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
use poker_solver_core::blueprint_v2::Street;
use poker_solver_core::hands::all_hands;
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
}

/// A sampled turn situation with blueprint-derived ranges.
#[derive(Debug, Clone)]
pub struct BlueprintSituation {
    pub oop_range: [f32; NUM_COMBOS],
    pub ip_range: [f32; NUM_COMBOS],
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
            snap_dir.join("buckets"),                    // snapshot/buckets (copied during training)
            bundle_dir.join("buckets"),                  // bundle/buckets
            config.training.cluster_path.as_ref()        // cluster_path from config (relative to CWD)
                .map(PathBuf::from)
                .unwrap_or_default(),
            config.training.cluster_path.as_ref()        // cluster_path relative to bundle_dir
                .map(|cp| bundle_dir.join(cp))
                .unwrap_or_default(),
        ];
        let actual_buckets_dir = candidates.iter()
            .find(|p| p.join("flop.buckets").exists())
            .cloned()
            .unwrap_or_else(|| {
                eprintln!("[blueprint ranges] warning: no bucket files found in any candidate directory");
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
        })
    }

    pub fn strategy(&self) -> &BlueprintV2Strategy { &self.strategy }
    pub fn tree(&self) -> &GameTree { &self.tree }
    pub fn decision_map(&self) -> &[u32] { &self.decision_map }

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
                                bucket =
                                    self.buckets.get_bucket(*street, [c0, c1], board_slice);
                                break;
                            }
                            bucket
                        };

                        let probs =
                            self.strategy.get_action_probs(dec_idx as usize, bucket);

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
                                bucket =
                                    self.buckets.get_bucket(*street, [c0, c1], board_slice);
                                break;
                            }
                            bucket
                        };

                        let probs =
                            self.strategy.get_action_probs(dec_idx as usize, bucket);
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
                        eprintln!("[propagate] street={street:?} player={player} action={chosen}/{num_actions} oop_nz={oop_nz} ip_nz={ip_nz}");
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
