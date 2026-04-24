//! Structural CFR-D gadget: two nested Decision(player, [Terminate, Follow])
//! nodes prepended at the PostFlopGame::node_arena root, using existing
//! depth-boundary terminals for per-hand opt-out CFVs.
//!
//! See docs/plans/2026-04-24-option-a-deepstack-gadget-design.md and
//! docs/plans/2026-04-24-option-a-deepstack-gadget-plan.md.

#[allow(unused_imports)]
use crate::action_tree::*;

/// Gadget configuration passed to PostFlopGame::with_config_and_gadget.
#[derive(Debug, Clone)]
pub struct GadgetConfig {
    /// Per-hand OOP opt-out in bcfv units; len = num_private_hands(OOP).
    pub opt_out_oop: Vec<f32>,
    /// Per-hand IP opt-out in bcfv units; len = num_private_hands(IP).
    pub opt_out_ip: Vec<f32>,
    /// Which player's Decision is the outer gadget layer (0 = OOP, 1 = IP).
    /// Default: 1 (IP outer). Affects convergence dynamics but not safety
    /// guarantee -- Burch S3 sufficiency holds at each gadget decision node
    /// independently of nesting order.
    pub outer_player: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gadget_config_constructs() {
        let cfg = GadgetConfig {
            opt_out_oop: vec![0.1; 3],
            opt_out_ip:  vec![0.2; 2],
            outer_player: 1,
        };
        assert_eq!(cfg.opt_out_oop.len(), 3);
        assert_eq!(cfg.opt_out_ip.len(), 2);
        assert_eq!(cfg.outer_player, 1);
    }
}
