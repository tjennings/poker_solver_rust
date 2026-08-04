pub mod config;
pub mod exploitability;
pub mod game_tree;
pub mod info_key;
pub mod lazy_mccfr;
pub mod mccfr;
pub mod mmap_buffer;
pub mod sparse_storage;
pub mod storage;
pub mod terminal;
pub mod trainer;
pub mod training_runtime_adapter;
pub mod types;

/// Maximum number of players supported in multiplayer blueprints.
pub const MAX_PLAYERS: usize = 10;

pub use config::*;
pub use info_key::InfoKey128;
pub use types::{
    Bucket, Chips, Deal, DealWithBuckets, PlayerSet, Seat, Street, parse_position, position_label,
};

/// Discount a signed integer regret, truncating fractional values toward zero.
///
/// Keeping the bounds explicit documents the conversion contract and protects
/// both eager and sparse MP storage if a discount factor falls outside its
/// expected range.
#[allow(clippy::cast_possible_truncation)]
pub(super) fn discount_signed_regret(value: i32, factor: f64) -> i32 {
    (f64::from(value) * factor).clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
}

#[cfg(test)]
mod tests {
    use super::discount_signed_regret;
    use test_macros::timed_test;

    #[timed_test]
    fn signed_regret_discount_truncates_fractional_results_toward_zero() {
        assert_eq!(discount_signed_regret(1, 0.9), 0);
        assert_eq!(discount_signed_regret(-1, 0.9), 0);
        assert_eq!(discount_signed_regret(1, 0.5), 0);
        assert_eq!(discount_signed_regret(-1, 0.5), 0);
    }

    #[timed_test]
    fn signed_regret_discount_is_symmetric_for_larger_values() {
        assert_eq!(discount_signed_regret(101, 0.5), 50);
        assert_eq!(discount_signed_regret(-101, 0.5), -50);
        assert_eq!(discount_signed_regret(7, 0.4), 2);
        assert_eq!(discount_signed_regret(-7, 0.4), -2);
    }

    #[timed_test]
    fn signed_regret_discount_eliminates_sticky_endpoints() {
        let mut positive = 3;
        let mut negative = -3;

        for _ in 0..3 {
            positive = discount_signed_regret(positive, 0.5);
            negative = discount_signed_regret(negative, 0.5);
        }

        assert_eq!(positive, 0);
        assert_eq!(negative, 0);
    }

    #[timed_test]
    fn signed_regret_discount_clamps_to_integer_bounds() {
        assert_eq!(discount_signed_regret(i32::MAX, 2.0), i32::MAX);
        assert_eq!(discount_signed_regret(i32::MIN, 2.0), i32::MIN);
    }
}
