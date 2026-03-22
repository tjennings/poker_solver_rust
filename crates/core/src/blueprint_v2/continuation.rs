use crate::blueprint_v2::game_tree::TreeAction;

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
}
