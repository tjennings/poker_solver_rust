#!/usr/bin/env python3
"""Extract poker strategy data from GTO Wizard HTML exports.

Usage:
    python3 scripts/extract_gto_wizard.py <input.html> [output.csv]

The input HTML must contain the rendered DOM from GTO Wizard's range table.
To capture it:
  1. Open GTO Wizard in Chrome, navigate to the strategy view
  2. Right-click the range table -> Inspect
  3. Find the outermost <div> containing the tables (class="sttranmain")
  4. Right-click -> Copy -> Copy outerHTML
  5. Paste into a file, wrap with: <html><body>PASTE</body></html>

GTO Wizard color palette:
  rgb(125, 31, 31)   = darkest red  -> all-in
  rgb(202, 50, 50)   = medium red   -> raise (smaller) / 3bet (smaller)
  rgb(240, 60, 60)   = bright red   -> raise (larger)  / 3bet (larger)
  rgb(90, 185, 102)  = green        -> check / call / limp
  rgb(61, 124, 184)  = blue         -> fold

Auto-detected scenarios by color set:
  {raise_2, check, fold}                    -> SB open:    raise, limp, fold
  {allin, raise_1, raise_2, check}          -> BB vs limp: allin, raise_sm, raise_lg, check
  {allin, raise_1, raise_2, check, fold}    -> BB vs raise: allin, 3bet_sm, 3bet_lg, call, fold
"""

import re
import sys
from pathlib import Path

# GTO Wizard RGB -> internal name, ordered most aggressive to least
PALETTE = [
    ((125, 31, 31), "allin"),
    ((202, 50, 50), "raise_1"),
    ((240, 60, 60), "raise_2"),
    ((90, 185, 102), "check"),
    ((61, 124, 184), "fold"),
]

# Scenario-specific column renames keyed by frozenset of internal color names
SCENARIO_NAMES = {
    frozenset(["raise_2", "check", "fold"]): {
        "raise_2": "raise",
        "check": "limp",
    },
    frozenset(["allin", "raise_1", "raise_2", "check"]): {
        "raise_1": "raise_sm",
        "raise_2": "raise_lg",
    },
    frozenset(["allin", "raise_1", "raise_2", "check", "fold"]): {
        "raise_1": "3bet_sm",
        "raise_2": "3bet_lg",
        "check": "call",
    },
}

RGB_TOLERANCE = 15


def match_color(r: int, g: int, b: int) -> str:
    for (pr, pg, pb), name in PALETTE:
        if abs(r - pr) <= RGB_TOLERANCE and abs(g - pg) <= RGB_TOLERANCE and abs(b - pb) <= RGB_TOLERANCE:
            return name
    return f"unknown_{r}_{g}_{b}"


def parse_cell_style(style: str) -> list[tuple[str, float]]:
    """Parse a cell's CSS gradient layers into (action, percentage) pairs."""
    colors_raw = re.findall(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", style)
    sizes = [float(s) for s in re.findall(r"([\d.]+)%\s*[\d.]+%", style)]

    # Deduplicate colors (each gradient repeats the color twice for solid fill)
    actions = []
    seen = set()
    for r_s, g_s, b_s in colors_raw:
        key = (int(r_s), int(g_s), int(b_s))
        if key not in seen:
            seen.add(key)
            actions.append(match_color(*key))

    # Compute per-action pct from cumulative background-size widths
    result = []
    prev = 0.0
    for i, action in enumerate(actions):
        cur = min(sizes[i], 100.0) if i < len(sizes) else 100.0
        result.append((action, round(max(cur - prev, 0.0), 2)))
        prev = cur
    return result


def extract(html: str) -> tuple[list[dict], list[str]]:
    """Extract strategy table (table 0) from GTO Wizard HTML.

    Returns (rows, column_names).
    """
    cells = re.findall(r'id="0_(\w+)"[^>]*style="([^"]*)"', html)
    evs = re.findall(r'class="rtc_value"><span>([^<]+)</span>', html)

    if not cells:
        return [], []
    if len(cells) != 169:
        print(f"Warning: found {len(cells)} cells (expected 169)", file=sys.stderr)

    # First pass: parse cells and discover actions
    all_actions = set()
    parsed = []
    for i, (hand, style) in enumerate(cells):
        ev = float(evs[i]) if i < len(evs) else 0.0
        actions = parse_cell_style(style)
        parsed.append((hand, ev, actions))
        for name, _ in actions:
            all_actions.add(name)

    # Determine column order (palette order) and display names
    palette_order = [name for _, name in PALETTE]
    action_cols = [a for a in palette_order if a in all_actions]
    action_cols += sorted(a for a in all_actions if a not in palette_order)

    renames = SCENARIO_NAMES.get(frozenset(action_cols), {})
    display_cols = [renames.get(c, c) for c in action_cols]

    # Build rows
    rows = []
    for hand, ev, actions in parsed:
        row = {"hand": hand, "ev": ev}
        for col, disp in zip(action_cols, display_cols):
            row[disp] = 0.0
        for name, pct in actions:
            row[renames.get(name, name)] = pct
        rows.append(row)

    return rows, display_cols


def to_csv(rows: list[dict], columns: list[str]) -> str:
    header = "hand,ev," + ",".join(columns)
    lines = [header]
    for r in rows:
        vals = ",".join(str(r[c]) for c in columns)
        lines.append(f"{r['hand']},{r['ev']},{vals}")
    return "\n".join(lines) + "\n"


def print_summary(rows: list[dict], columns: list[str]):
    n = len(rows)
    for col in columns:
        avg = sum(r[col] for r in rows) / n
        majority = sum(1 for r in rows if r[col] > 50)
        pure = sum(1 for r in rows if r[col] >= 99.9)
        print(f"  {col:>10s}: avg {avg:5.1f}%  majority(>50%) {majority:3d}  pure(~100%) {pure:2d}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.html> [output.csv]", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    html = input_path.read_text()

    rows, columns = extract(html)
    if not rows:
        print("Error: no strategy data found.", file=sys.stderr)
        print("Make sure the file contains the rendered DOM, not the SPA shell.", file=sys.stderr)
        sys.exit(1)

    output_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else input_path.with_suffix(".csv")
    output_path.write_text(to_csv(rows, columns))

    print(f"Wrote {len(rows)} hands to {output_path}")
    print(f"Columns: {columns}")
    print_summary(rows, columns)


if __name__ == "__main__":
    main()
