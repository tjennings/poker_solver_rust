# Rake Support + Blueprint Name — Design

## Overview

Add rake (percentage of pot) and rake cap (absolute chip amount) to the blueprint MCCFR solver and range solver Tauri integration. Add a required `name` field to blueprint config for selection and display in the Tauri explorer.

## Blueprint Config Changes

Three new **required** fields in `BlueprintV2Config` / `GameConfig`:

```yaml
name: "NL50 HU 5% rake"
rake_rate: 0.05      # 0.0–1.0 fraction of pot
rake_cap: 3.0        # chips (min bet units), 0.0 = no cap
```

Validation:
- `name`: non-empty string
- `rake_rate`: 0.0 <= rate <= 1.0
- `rake_cap`: >= 0.0

## Blueprint MCCFR Terminal Payoffs

In `terminal_value()`, deduct rake from the winner's share:

```
rake = min(pot * rake_rate, rake_cap)
winner_payoff -= rake
```

Applied at both fold and showdown terminals, matching the range solver's existing behavior in `evaluation.rs`.

## Range Solver (Tauri Integration)

The range solver already supports `rake_rate` / `rake_cap` in `TreeConfig` and correctly applies rake at terminal nodes. Changes:

- Replace hardcoded `rake_rate: 0.0, rake_cap: 0.0` in `postflop.rs` with values from the loaded blueprint config
- Add rake rate/cap input fields to the range solver config panel in the frontend
- Fields pre-populate from blueprint, user can override

## Tauri Explorer UX

- **Blueprint selection**: `name` field shown in the blueprint picker dropdown
- **Header/sidebar**: display the blueprint name as the loaded model label
- **Blueprint info panel**: show rake rate and rake cap as read-only fields alongside stack depth, bet sizes, etc.

## Out of Scope

- Preflop solver rake (standard "no flop no drop" convention)
- Abstraction pipeline changes (rake only affects terminal payoffs)
- Convergence is unaffected — CFR works identically on non-zero-sum games
