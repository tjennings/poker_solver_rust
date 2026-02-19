# Strategy Explorer

The desktop app lets you browse trained strategies interactively using a 13x13 hand matrix.

## Starting the App

```bash
cd frontend && npm install && cd ..
cd crates/tauri-app && cargo tauri dev
```

## Loading a Strategy

Click **Load Strategy Bundle** and select the output directory from training (e.g. `my_strategy/`). The app displays the bundle metadata: stack depth, bet sizes, info set count, and training iterations.

## Browsing the Game Tree

### Preflop

The 13x13 hand matrix shows action probabilities for every starting hand. Each cell displays a color-coded bar:
- Blue = fold
- Green = call/check
- Red = bet/raise
- Purple = all-in

Click an action button (fold, call, raise) to advance down the game tree.

### Postflop

When the game reaches the flop, enter board cards (e.g. `AcTh4d`). The app computes EHS2 buckets for all 169 canonical hands (progress bar shown), then displays the strategy matrix for that board.

Continue navigating through turn and river by entering additional cards.

### History

The action strip at the top shows the full history. Click any point to rewind.
