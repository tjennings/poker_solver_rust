#!/usr/bin/env python3
"""Summarize bucket sweep scorecards into a compact markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def by_street(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("street", "")).lower(): item for item in items}


def num(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def get_path(data: dict[str, Any], path: str, default: float = 0.0) -> float:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return num(cur, default)


def scorecard_metrics(card: dict[str, Any]) -> dict[str, float]:
    clusters = by_street(card.get("cluster_reports", []))
    audits = by_street(card.get("hand_class_audits", []))
    river_nut = card.get("river_nut_distance_audit") or {}
    river_summary = river_nut.get("summary", {}) if isinstance(river_nut, dict) else {}

    inversion_total = sum(
        get_path(audit, "summary.strength_inversion_count") for audit in audits.values()
    )
    skipped_total = sum(get_path(audit, "skipped_lookups") for audit in audits.values())
    max_equity_span = max(
        [get_path(audit, "summary.max_equity_span") for audit in audits.values()] or [0.0]
    )
    max_class_entropy = max(
        [get_path(audit, "summary.max_class_entropy") for audit in audits.values()] or [0.0]
    )

    return {
        "flop_size_skew": get_path(clusters.get("flop", {}), "bucket_size.max_to_mean"),
        "turn_size_skew": get_path(clusters.get("turn", {}), "bucket_size.max_to_mean"),
        "river_size_skew": get_path(clusters.get("river", {}), "bucket_size.max_to_mean"),
        "inversions": inversion_total,
        "skipped": skipped_total,
        "max_equity_span": max_equity_span,
        "max_class_entropy": max_class_entropy,
        "river_class_gap_span": get_path(river_summary, "max_class_gap_span"),
        "river_dominance_span": get_path(river_summary, "max_dominance_margin_span"),
        "river_global_rank_span": get_path(river_summary, "max_global_rank_percentile_span"),
    }


def metric_scale_summary(card: dict[str, Any]) -> str:
    scales = card.get("metric_scales")
    if not isinstance(scales, list) or not scales:
        return "none"
    parts = []
    for item in scales:
        street = str(item.get("street", "?")).lower()
        nw = num(item.get("nut_distance_weight"))
        ew = num(item.get("equity_weight"))
        nt = item.get("nut_distance_transform", "linear")
        nc = item.get("nut_distance_cap")
        nut_div = get_path(item, "nut_distance.divisor")
        eq_div = get_path(item, "equity.divisor")
        cap = "" if nc is None else f",cap={num(nc):.2f}"
        parts.append(
            f"{street}:e={ew:.2f},n={nw:.2f},{nt}{cap},eq_div={eq_div:.4g},nut_div={nut_div:.4g}"
        )
    return "; ".join(parts)


def fmt(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.4f}"


def delta(value: float, baseline: float) -> str:
    diff = value - baseline
    sign = "+" if diff >= 0 else ""
    return f"{fmt(value)} ({sign}{fmt(diff)})"


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME=DIR")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("candidate name must not be empty")
    return name, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", action="append", required=True, type=parse_candidate)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline_card = load_json(args.baseline / "scorecard.json")
    baseline = scorecard_metrics(baseline_card)
    candidates = []
    for name, directory in args.candidate:
        scorecard_path = directory / "scorecard.json"
        if not scorecard_path.exists():
            candidates.append((name, directory, None, None))
            continue
        card = load_json(scorecard_path)
        candidates.append((name, directory, card, scorecard_metrics(card)))

    lines = [
        "# Bucket Metric Sweep Analysis",
        "",
        f"Baseline: `{args.baseline}`",
        "",
        "## Summary",
        "",
        "| candidate | river class gap span | river dominance span | inversions | max equity span | class entropy | size skew f/t/r | skipped |",
        "|-|-:|-:|-:|-:|-:|-:|-:|",
    ]
    for name, directory, card, metrics in candidates:
        if metrics is None:
            lines.append(f"| {name} | missing `{directory / 'scorecard.json'}` | | | | | | |")
            continue
        lines.append(
            "| {name} | {class_gap} | {dominance} | {inversions} | {equity_span} | {entropy} | {size_skew} | {skipped} |".format(
                name=name,
                class_gap=delta(metrics["river_class_gap_span"], baseline["river_class_gap_span"]),
                dominance=delta(metrics["river_dominance_span"], baseline["river_dominance_span"]),
                inversions=delta(metrics["inversions"], baseline["inversions"]),
                equity_span=delta(metrics["max_equity_span"], baseline["max_equity_span"]),
                entropy=delta(metrics["max_class_entropy"], baseline["max_class_entropy"]),
                size_skew="/".join(
                    [
                        delta(metrics["flop_size_skew"], baseline["flop_size_skew"]),
                        delta(metrics["turn_size_skew"], baseline["turn_size_skew"]),
                        delta(metrics["river_size_skew"], baseline["river_size_skew"]),
                    ]
                ),
                skipped=delta(metrics["skipped"], baseline["skipped"]),
            )
        )

    lines.extend(["", "## Metric Scales", ""])
    for name, directory, card, _metrics in candidates:
        if card is None:
            continue
        lines.append(f"- `{name}`: {metric_scale_summary(card)}")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Lower river nut-distance spans, fewer strength inversions, lower equity span, and controlled size skew are preferred.",
            "- Size skew regressions are acceptable only if nut-distance and hand-class metrics improve enough to justify retesting in training.",
            "- This report is a first-pass abstraction audit; final promotion still needs short-run strategy sanity checks.",
            "",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
