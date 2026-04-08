use super::types::{Chips, PlayerSet, Seat};
use super::MAX_PLAYERS;

/// Resolve payoffs when all but one player has folded.
///
/// Returns net payoff per seat: winner gets the total pot minus their contribution,
/// losers get negative their contribution.
#[must_use]
pub fn resolve_fold(
    contributions: [Chips; MAX_PLAYERS],
    winner: Seat,
    num_players: u8,
) -> [Chips; MAX_PLAYERS] {
    let total_pot: Chips = contributions.iter().take(num_players as usize).sum();
    let mut payoffs = [Chips::ZERO; MAX_PLAYERS];
    for i in 0..num_players as usize {
        payoffs[i] = -contributions[i];
    }
    payoffs[winner.index() as usize] += total_pot;
    payoffs
}

// ---------------------------------------------------------------------------
// Side-pot helpers
// ---------------------------------------------------------------------------

/// Collect active players' (seat, contribution) pairs, sorted ascending by contribution.
fn collect_active_contributions(
    contributions: &[Chips; MAX_PLAYERS],
    active: PlayerSet,
) -> Vec<(Seat, Chips)> {
    let mut result: Vec<(Seat, Chips)> = active
        .iter()
        .map(|s| (s, contributions[s.index() as usize]))
        .collect();
    result.sort_by(|a, b| a.1 .0.total_cmp(&b.1 .0));
    result
}

/// Build side pots layer-by-layer from sorted active contributions.
///
/// Returns `Vec<(pot_amount, eligible_seats)>` where eligible seats are
/// active players who contributed at least up to that layer.
fn build_side_pots(
    sorted: &[(Seat, Chips)],
    contributions: &[Chips; MAX_PLAYERS],
    num_players: u8,
) -> Vec<(Chips, Vec<Seat>)> {
    let mut pots = Vec::new();
    let mut prev_level = Chips::ZERO;

    for (idx, &(_seat, contrib)) in sorted.iter().enumerate() {
        let layer = contrib - prev_level;
        if layer <= Chips::ZERO {
            continue;
        }
        let pot = compute_layer_pot(contributions, num_players, prev_level, contrib);
        let eligible: Vec<Seat> = sorted[idx..].iter().map(|&(s, _)| s).collect();
        pots.push((pot, eligible));
        prev_level = contrib;
    }
    pots
}

/// Compute the total chips contributed by ALL players within a single layer.
fn compute_layer_pot(
    contributions: &[Chips; MAX_PLAYERS],
    num_players: u8,
    floor: Chips,
    cap: Chips,
) -> Chips {
    let layer = cap - floor;
    contributions
        .iter()
        .take(num_players as usize)
        .map(|&c| (c - floor).clamp(Chips::ZERO, layer))
        .sum()
}

/// Apply rake to a pot, respecting the cumulative cap.
/// Returns `(net_pot_after_rake, new_cumulative_rake)`.
fn apply_rake(pot: Chips, rate: f64, total_rake: Chips, cap: Chips) -> (Chips, Chips) {
    if rate <= 0.0 {
        return (pot, total_rake);
    }
    let raw_rake = pot * rate;
    let headroom = (cap - total_rake).max(Chips::ZERO);
    let actual_rake = raw_rake.min(headroom);
    (pot - actual_rake, total_rake + actual_rake)
}

/// Award a pot to the best hand(s) among eligible players, splitting ties evenly.
fn award_pot(
    net_pot: Chips,
    eligible: &[Seat],
    hand_ranks: &[u32; MAX_PLAYERS],
) -> Vec<(Seat, Chips)> {
    let best = eligible
        .iter()
        .map(|&s| hand_ranks[s.index() as usize])
        .max()
        .unwrap_or(0);
    let winners: Vec<Seat> = eligible
        .iter()
        .copied()
        .filter(|&s| hand_ranks[s.index() as usize] == best)
        .collect();
    #[allow(clippy::cast_precision_loss)] // winner count is at most 8
    let share = net_pot / winners.len() as f64;
    winners.into_iter().map(|s| (s, share)).collect()
}

/// Resolve payoffs at showdown with full side-pot resolution and rake.
///
/// `hand_ranks`: higher value = better hand.
/// `active`: bitset of players who reached showdown (didn't fold).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn resolve_showdown(
    contributions: &[Chips; MAX_PLAYERS],
    hand_ranks: &[u32; MAX_PLAYERS],
    active: PlayerSet,
    num_players: u8,
    rake_rate: f64,
    rake_cap: Chips,
) -> [Chips; MAX_PLAYERS] {
    let mut payoffs = [Chips::ZERO; MAX_PLAYERS];
    for i in 0..num_players as usize {
        payoffs[i] = -contributions[i];
    }

    let sorted = collect_active_contributions(contributions, active);
    let pots = build_side_pots(&sorted, contributions, num_players);
    let mut total_rake = Chips::ZERO;

    for (pot_amount, eligible) in &pots {
        let (net_pot, new_rake) = apply_rake(*pot_amount, rake_rate, total_rake, rake_cap);
        total_rake = new_rake;
        for (seat, share) in award_pot(net_pot, eligible, hand_ranks) {
            payoffs[seat.index() as usize] += share;
        }
    }
    payoffs
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;
    use test_macros::timed_test;

    // ---------- resolve_fold tests ----------

    #[timed_test]
    fn fold_last_standing_3_player() {
        // 3 players each put in 10, P2 wins
        let contribs = [
            Chips(10.0), Chips(10.0), Chips(10.0),
            Chips::ZERO, Chips::ZERO, Chips::ZERO,
            Chips::ZERO, Chips::ZERO,
        ];
        let payoffs = resolve_fold(contribs, Seat::from_raw(2), 3);
        assert_eq!(payoffs[0], Chips(-10.0));
        assert_eq!(payoffs[1], Chips(-10.0));
        assert_eq!(payoffs[2], Chips(20.0));
        // Unused seats are zero
        assert_eq!(payoffs[3], Chips::ZERO);
        assert_eq!(payoffs[4], Chips::ZERO);
        assert_eq!(payoffs[5], Chips::ZERO);
    }

    #[timed_test]
    fn fold_payoffs_sum_to_zero() {
        // 4 players, arbitrary contributions
        let contribs = [
            Chips(15.0), Chips(25.0), Chips(5.0), Chips(55.0),
            Chips::ZERO, Chips::ZERO, Chips::ZERO, Chips::ZERO,
        ];
        let payoffs = resolve_fold(contribs, Seat::from_raw(1), 4);
        let sum: f64 = payoffs.iter().map(|c| c.0).sum();
        assert!(
            sum.abs() < 1e-9,
            "fold payoffs must sum to zero, got {sum}"
        );
        // Winner gets total pot minus own contribution
        assert_eq!(payoffs[1], Chips(75.0));
    }

    // ---------- resolve_showdown tests ----------

    #[timed_test]
    fn showdown_no_side_pots() {
        // 3 players, equal contributions, P0 has best hand
        let contribs = [
            Chips(50.0), Chips(50.0), Chips(50.0),
            Chips::ZERO, Chips::ZERO, Chips::ZERO,
            Chips::ZERO, Chips::ZERO,
        ];
        let ranks = [100, 50, 30, 0, 0, 0, 0, 0];
        let active = PlayerSet::from_bits(0b0000_0111);
        let payoffs = resolve_showdown(&contribs, &ranks, active, 3, 0.0, Chips::ZERO);
        assert_eq!(payoffs[0], Chips(100.0));
        assert_eq!(payoffs[1], Chips(-50.0));
        assert_eq!(payoffs[2], Chips(-50.0));
    }

    #[timed_test]
    fn showdown_with_side_pot() {
        // P0 all-in 30, P1/P2 put in 100. P0 best hand, P1 second best.
        let contribs = [
            Chips(30.0), Chips(100.0), Chips(100.0),
            Chips::ZERO, Chips::ZERO, Chips::ZERO,
            Chips::ZERO, Chips::ZERO,
        ];
        let ranks = [300, 200, 100, 0, 0, 0, 0, 0];
        let active = PlayerSet::from_bits(0b0000_0111);
        let payoffs = resolve_showdown(&contribs, &ranks, active, 3, 0.0, Chips::ZERO);
        assert_eq!(payoffs[0], Chips(60.0));
        assert_eq!(payoffs[1], Chips(40.0));
        assert_eq!(payoffs[2], Chips(-100.0));
    }

    #[timed_test]
    fn showdown_three_way_side_pots() {
        // P0=20, P1=50, P2=100. P2 has best hand, wins everything.
        let contribs = [
            Chips(20.0), Chips(50.0), Chips(100.0),
            Chips::ZERO, Chips::ZERO, Chips::ZERO,
            Chips::ZERO, Chips::ZERO,
        ];
        let ranks = [10, 20, 30, 0, 0, 0, 0, 0];
        let active = PlayerSet::from_bits(0b0000_0111);
        let payoffs = resolve_showdown(&contribs, &ranks, active, 3, 0.0, Chips::ZERO);
        assert_eq!(payoffs[0], Chips(-20.0));
        assert_eq!(payoffs[1], Chips(-50.0));
        assert_eq!(payoffs[2], Chips(70.0));
    }

    #[timed_test]
    fn showdown_with_rake() {
        // 2 players, 50 each, 5% rake cap 10. P0 wins.
        let contribs = [
            Chips(50.0), Chips(50.0), Chips::ZERO, Chips::ZERO,
            Chips::ZERO, Chips::ZERO, Chips::ZERO, Chips::ZERO,
        ];
        let ranks = [200, 100, 0, 0, 0, 0, 0, 0];
        let active = PlayerSet::from_bits(0b0000_0011);
        let payoffs = resolve_showdown(&contribs, &ranks, active, 2, 0.05, Chips(10.0));
        assert!((payoffs[0].0 - 45.0).abs() < 1e-9, "P0 net: {:?}", payoffs[0]);
        assert!((payoffs[1].0 - (-50.0)).abs() < 1e-9, "P1 net: {:?}", payoffs[1]);
    }

    #[timed_test]
    fn showdown_payoffs_sum_to_negative_rake() {
        // With rake, sum of payoffs = -(total rake taken)
        let contribs = [
            Chips(50.0), Chips(50.0), Chips::ZERO, Chips::ZERO,
            Chips::ZERO, Chips::ZERO, Chips::ZERO, Chips::ZERO,
        ];
        let ranks = [200, 100, 0, 0, 0, 0, 0, 0];
        let active = PlayerSet::from_bits(0b0000_0011);
        let payoffs = resolve_showdown(&contribs, &ranks, active, 2, 0.05, Chips(10.0));
        let sum: f64 = payoffs.iter().map(|c| c.0).sum();
        assert!(
            (sum + 5.0).abs() < 1e-9,
            "payoffs should sum to -rake (-5.0), got {sum}"
        );
    }

    #[timed_test]
    fn showdown_tie_splits_pot() {
        // 2 players equal hands, each put 50. Both break even.
        let contribs = [
            Chips(50.0), Chips(50.0), Chips::ZERO, Chips::ZERO,
            Chips::ZERO, Chips::ZERO, Chips::ZERO, Chips::ZERO,
        ];
        let ranks = [100, 100, 0, 0, 0, 0, 0, 0];
        let active = PlayerSet::from_bits(0b0000_0011);
        let payoffs = resolve_showdown(&contribs, &ranks, active, 2, 0.0, Chips::ZERO);
        assert!(
            payoffs[0].0.abs() < 1e-9,
            "P0 should break even, got {:?}",
            payoffs[0]
        );
        assert!(
            payoffs[1].0.abs() < 1e-9,
            "P1 should break even, got {:?}",
            payoffs[1]
        );
    }

    #[timed_test]
    fn showdown_tie_three_way() {
        // 3 players equal hands, equal contributions. All break even.
        let contribs = [
            Chips(30.0), Chips(30.0), Chips(30.0),
            Chips::ZERO, Chips::ZERO, Chips::ZERO,
            Chips::ZERO, Chips::ZERO,
        ];
        let ranks = [50, 50, 50, 0, 0, 0, 0, 0];
        let active = PlayerSet::from_bits(0b0000_0111);
        let payoffs = resolve_showdown(&contribs, &ranks, active, 3, 0.0, Chips::ZERO);
        for i in 0..3 {
            assert!(
                payoffs[i].0.abs() < 1e-9,
                "P{i} should break even, got {:?}",
                payoffs[i]
            );
        }
    }

    #[timed_test]
    fn showdown_short_stack_best_hand_wins_main_pot_only() {
        // Short-stack with best hand can only win the main pot.
        let contribs = [
            Chips(20.0), Chips(100.0), Chips(100.0),
            Chips::ZERO, Chips::ZERO, Chips::ZERO,
            Chips::ZERO, Chips::ZERO,
        ];
        let ranks = [999, 500, 100, 0, 0, 0, 0, 0];
        let active = PlayerSet::from_bits(0b0000_0111);
        let payoffs = resolve_showdown(&contribs, &ranks, active, 3, 0.0, Chips::ZERO);
        assert_eq!(payoffs[0], Chips(40.0));
        assert_eq!(payoffs[1], Chips(60.0));
        assert_eq!(payoffs[2], Chips(-100.0));
    }

    #[timed_test]
    fn showdown_folded_player_dead_money_included() {
        // P0 folds after putting in 40, P1 all-in 30, P2 calls 100.
        // P0 is NOT active (folded). P1 and P2 are active.
        // Main pot: min(30,40) + 30 + 30 = 90 (P0 contributes 30 at P1's level)
        // Side pot: (100-30)*1 = 70 from P2 only, plus P0's extra 10 above P1's level
        //   Actually: P0 contributed 40, so at the 30-100 layer: min(40-30, 100-30) = 10 from P0
        //   Side pot = 10 + 70 = 80, only P2 is eligible (P1 is all-in at 30)
        // P1 wins main pot (best hand): 90 - 30 = +60
        // P2 wins side pot (only eligible): 80 - 100 = net -20
        // P0 folded: -40
        let contribs = [
            Chips(40.0), Chips(30.0), Chips(100.0),
            Chips::ZERO, Chips::ZERO, Chips::ZERO,
            Chips::ZERO, Chips::ZERO,
        ];
        let ranks = [0, 999, 500, 0, 0, 0, 0, 0]; // P0 rank irrelevant (folded)
        let active = PlayerSet::from_bits(0b0000_0110); // P1, P2 active
        let payoffs = resolve_showdown(&contribs, &ranks, active, 3, 0.0, Chips::ZERO);
        // Total pot = 40 + 30 + 100 = 170. Must all be distributed.
        assert_eq!(payoffs[0], Chips(-40.0), "P0 (folded) loses contribution");
        // P1 wins main pot (30*3 players = 90), net = 90 - 30 = +60
        assert_eq!(payoffs[1], Chips(60.0), "P1 wins main pot");
        // P2 wins side pot (170 - 90 = 80), net = 80 - 100 = -20
        assert_eq!(payoffs[2], Chips(-20.0), "P2 wins side pot");
        // Verify zero-sum
        let sum: Chips = payoffs.iter().take(3).copied().sum();
        assert!(sum.0.abs() < 1e-9, "payoffs must sum to zero, got {}", sum.0);
    }

    #[timed_test]
    fn showdown_rake_cap_across_multiple_pots() {
        // P0 all-in 30, P1 puts in 100. P0 has best hand.
        // Rake 10%, cap 5 chips.
        // Main pot: 30*2 = 60, rake = min(6, 5) = 5, net = 55
        // Side pot: 70*1 = 70, rake = min(7, 0 remaining cap) = 0, net = 70
        let contribs = [
            Chips(30.0), Chips(100.0),
            Chips::ZERO, Chips::ZERO, Chips::ZERO, Chips::ZERO,
            Chips::ZERO, Chips::ZERO,
        ];
        let ranks = [200, 100, 0, 0, 0, 0, 0, 0];
        let active = PlayerSet::from_bits(0b0000_0011);
        let payoffs = resolve_showdown(&contribs, &ranks, active, 2, 0.10, Chips(5.0));
        // P0 wins main pot net: 55 - 30 = +25
        assert!((payoffs[0].0 - 25.0).abs() < 1e-9, "P0: {}", payoffs[0].0);
        // P1 wins side pot (uncontested, no more rake room): 70 - 100 = -30
        assert!((payoffs[1].0 - (-30.0)).abs() < 1e-9, "P1: {}", payoffs[1].0);
        // Total rake should be exactly 5 (cap hit on first pot)
        let sum: f64 = payoffs.iter().take(2).map(|c| c.0).sum();
        assert!((sum + 5.0).abs() < 1e-9, "sum should be -5 (rake), got {sum}");
    }
}
