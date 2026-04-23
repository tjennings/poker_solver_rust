#!/usr/bin/env python3
"""
Diff two boundary-trace directories (subgame vs exact) and report the largest
per-hand-class CFV and strategy biases, one section per boundary ordinal.

Usage:
    scripts/trace_diff.py \
        --subgame local_data/logs/subgame \
        --exact   local_data/logs/exact \
        [--top 20] [--boundary 0]

The game_solve Tauri command writes traces to `<trace_dir>/subgame/cfvnet/`,
`<trace_dir>/subgame/exact_subtree/`, or `<trace_dir>/exact/` automatically
based on the solve mode, so after running both modes on the same spot you can
diff them without moving files (e.g. subgame/cfvnet vs subgame/exact_subtree).

Both directories must contain `boundary_<N>.txt` files produced by the
BoundaryTracer. The script reads the LAST `[iter=... FINAL ...]` record from
each file (post-finalize strategy, matches the UI).

Output (per boundary):
    - Top-N biggest CFV deltas  per side (OOP/IP)
    - Top-N biggest strategy deltas at the preceding decision (by max |delta|
      across actions). The action contributing the max delta is named.
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Record:
    iter: int
    is_final: bool
    ordinal: int
    board: str
    pot: int
    stack: int
    spr: float
    spot: Optional[str] = None
    oop_range: Dict[str, float] = field(default_factory=dict)  # hand -> mean weight
    ip_range: Dict[str, float] = field(default_factory=dict)
    oop_cfvs: Dict[str, float] = field(default_factory=dict)  # hand -> mean CFV (chips)
    ip_cfvs: Dict[str, float] = field(default_factory=dict)
    preceding_player: Optional[str] = None  # "OOP" or "IP"
    preceding_actions: List[str] = field(default_factory=list)
    preceding_strategy: Dict[str, List[float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


HEADER_RE = re.compile(
    r"\[iter=(\d+)(?: FINAL)? boundary=(\d+) board=(\S+) pot=(-?\d+) stack=(-?\d+) spr=(-?[\d.]+)\]"
)
RANGE_TOKEN_RE = re.compile(r"(\S+?):(\d+)/(\d+):(-?[\d.]+)")
CFV_TOKEN_RE = re.compile(r"(\S+?):([+\-][\d.]+)")
STRAT_HEADER_RE = re.compile(
    r"Strategy at preceding decision \(node #(\d+), (OOP|IP) to act\):"
)
HAND_STRAT_RE = re.compile(r"^\s{2,}(\S+?):\s+\[([-\d.,\s]+)\]\s*$")


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_records(path: str) -> List[Record]:
    """Parse every record in a single boundary_*.txt file.

    Records are separated by `---` lines.
    """
    records: List[Record] = []
    with open(path) as f:
        text = f.read()

    blocks = [b for b in text.split("\n---") if b.strip()]
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        m = HEADER_RE.search(lines[0])
        if not m:
            continue
        rec = Record(
            iter=int(m.group(1)),
            is_final="FINAL" in lines[0],
            ordinal=int(m.group(2)),
            board=m.group(3),
            pot=int(m.group(4)),
            stack=int(m.group(5)),
            spr=float(m.group(6)),
        )

        in_strat = False
        for ln in lines[1:]:
            if ln.startswith("Spot: "):
                rec.spot = ln[len("Spot: "):].strip()
                continue
            if ln.startswith("OOP range @ boundary"):
                for tm in RANGE_TOKEN_RE.finditer(ln):
                    rec.oop_range[tm.group(1)] = float(tm.group(4))
                continue
            if ln.startswith("IP range  @ boundary") or ln.startswith("IP range @ boundary"):
                for tm in RANGE_TOKEN_RE.finditer(ln):
                    rec.ip_range[tm.group(1)] = float(tm.group(4))
                continue
            if ln.startswith("OOP CFVs @ boundary"):
                for tm in CFV_TOKEN_RE.finditer(ln):
                    rec.oop_cfvs[tm.group(1)] = float(tm.group(2))
                continue
            if ln.startswith("IP CFVs  @ boundary") or ln.startswith("IP CFVs @ boundary"):
                for tm in CFV_TOKEN_RE.finditer(ln):
                    rec.ip_cfvs[tm.group(1)] = float(tm.group(2))
                continue
            sm = STRAT_HEADER_RE.search(ln)
            if sm:
                rec.preceding_player = sm.group(2)
                in_strat = True
                continue
            if in_strat:
                if ln.lstrip().startswith("Actions:"):
                    inner = ln[ln.index("[") + 1 : ln.rindex("]")]
                    rec.preceding_actions = [a.strip() for a in inner.split(",")]
                    continue
                if ln.startswith("Per-combo strategy:"):
                    in_strat = False
                    continue
                hm = HAND_STRAT_RE.match(ln)
                if hm:
                    rec.preceding_strategy[hm.group(1)] = parse_float_list(hm.group(2))

        records.append(rec)
    return records


def last_final_record(path: str) -> Optional[Record]:
    recs = parse_records(path)
    finals = [r for r in recs if r.is_final]
    if finals:
        return finals[-1]
    return recs[-1] if recs else None


def list_trace_files(dirpath: str) -> List[Tuple[int, str]]:
    """Return sorted list of (ordinal, path) for every boundary_N.txt in dir."""
    out = []
    for name in os.listdir(dirpath):
        m = re.fullmatch(r"boundary_(\d+)\.txt", name)
        if m:
            out.append((int(m.group(1)), os.path.join(dirpath, name)))
    out.sort()
    return out


# ---------------------------------------------------------------------------
# Diff logic
# ---------------------------------------------------------------------------


def top_cfv_deltas(
    a: Dict[str, float], b: Dict[str, float], top_n: int
) -> List[Tuple[str, float, float, float]]:
    """Return [(hand, a_cfv, b_cfv, delta)] sorted by |delta| desc."""
    rows = []
    for hand in sorted(set(a) | set(b)):
        va = a.get(hand, 0.0)
        vb = b.get(hand, 0.0)
        rows.append((hand, va, vb, va - vb))
    rows.sort(key=lambda r: -abs(r[3]))
    return rows[:top_n]


def top_strategy_deltas(
    a: Record, b: Record, top_n: int
) -> List[Tuple[str, str, List[float], List[float], float]]:
    """Return [(hand, action_name, a_probs, b_probs, max_abs_delta)] sorted."""
    if not a.preceding_strategy or not b.preceding_strategy:
        return []
    if a.preceding_actions != b.preceding_actions:
        print(
            f"  (warning: action labels differ; a={a.preceding_actions} b={b.preceding_actions})",
            file=sys.stderr,
        )
    actions = a.preceding_actions
    rows = []
    for hand in sorted(set(a.preceding_strategy) | set(b.preceding_strategy)):
        pa = a.preceding_strategy.get(hand, [0.0] * len(actions))
        pb = b.preceding_strategy.get(hand, [0.0] * len(actions))
        # Pad to equal length defensively
        n = max(len(pa), len(pb), len(actions))
        pa = pa + [0.0] * (n - len(pa))
        pb = pb + [0.0] * (n - len(pb))
        diffs = [x - y for x, y in zip(pa, pb)]
        max_i = max(range(len(diffs)), key=lambda i: abs(diffs[i]))
        action_label = actions[max_i] if max_i < len(actions) else f"action[{max_i}]"
        rows.append((hand, action_label, pa, pb, diffs[max_i]))
    rows.sort(key=lambda r: -abs(r[4]))
    return rows[:top_n]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def fmt_probs(probs: List[float]) -> str:
    return "[" + ", ".join(f"{p:.2f}" for p in probs) + "]"


def report_boundary(sub: Record, exc: Record, top_n: int) -> str:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(
        f"Boundary {sub.ordinal}  board={sub.board}  pot={sub.pot}  spr={sub.spr:.2f}"
    )
    if sub.spot:
        lines.append(f"Spot: {sub.spot}")
    if sub.board != exc.board:
        lines.append(f"(!) board differs exact={exc.board}")

    # CFV deltas
    for side in ("oop", "ip"):
        a_cfv = getattr(sub, f"{side}_cfvs")
        b_cfv = getattr(exc, f"{side}_cfvs")
        deltas = top_cfv_deltas(a_cfv, b_cfv, top_n)
        lines.append(f"\n[{side.upper()} CFV bias]   sub − exact   (chips)")
        lines.append(f"  {'hand':<6} {'sub':>8} {'exact':>8} {'Δ':>8}")
        for hand, va, vb, d in deltas:
            marker = "  <<<" if abs(d) > 1.0 else ""
            lines.append(f"  {hand:<6} {va:+8.2f} {vb:+8.2f} {d:+8.2f}{marker}")

    # Strategy deltas at preceding decision
    if sub.preceding_strategy and exc.preceding_strategy:
        side = sub.preceding_player or "?"
        lines.append(
            f"\n[{side} strategy bias at preceding decision]   sub − exact   "
            f"(by max |Δ| action)"
        )
        lines.append(f"  actions: {sub.preceding_actions}")
        rows = top_strategy_deltas(sub, exc, top_n)
        for hand, action, pa, pb, d in rows:
            marker = "  <<<" if abs(d) > 0.10 else ""
            lines.append(
                f"  {hand:<6} sub={fmt_probs(pa)} exact={fmt_probs(pb)} "
                f"Δ{action}={d:+.2f}{marker}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subgame", required=True, help="dir with subgame traces")
    ap.add_argument("--exact", required=True, help="dir with exact reference traces")
    ap.add_argument("--top", type=int, default=15, help="top N rows per section")
    ap.add_argument(
        "--boundary",
        type=int,
        default=None,
        help="report only this boundary ordinal (default: all)",
    )
    args = ap.parse_args()

    sub_files = dict(list_trace_files(args.subgame))
    exc_files = dict(list_trace_files(args.exact))

    common = sorted(set(sub_files) & set(exc_files))
    if not common:
        print("No matching boundary_*.txt files in both dirs.", file=sys.stderr)
        return 1

    only_sub = sorted(set(sub_files) - set(exc_files))
    only_exc = sorted(set(exc_files) - set(sub_files))
    if only_sub:
        print(f"(subgame-only ordinals: {only_sub})", file=sys.stderr)
    if only_exc:
        print(f"(exact-only ordinals:   {only_exc})", file=sys.stderr)

    for ord_idx in common:
        if args.boundary is not None and ord_idx != args.boundary:
            continue
        sub_rec = last_final_record(sub_files[ord_idx])
        exc_rec = last_final_record(exc_files[ord_idx])
        if sub_rec is None or exc_rec is None:
            print(f"(skipping ord={ord_idx}: empty trace)", file=sys.stderr)
            continue
        print(report_boundary(sub_rec, exc_rec, args.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
