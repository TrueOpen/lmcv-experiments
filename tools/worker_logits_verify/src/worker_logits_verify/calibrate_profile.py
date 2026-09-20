from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .metrics import quantile, safe_float
from .schema import read_json, runtime_metadata, write_json


PROFILE_METRICS = [
    "selected_abs_logprob_diff",
    "selected_abs_raw_logit_diff",
    "selected_abs_rank_delta",
    "union_js_divergence",
    "union_max_abs_logprob_diff",
    "probe_max_abs_logprob_diff",
    "probe_rms_logprob_diff",
]


LOWER_BOUND_METRICS = [
    "topk_jaccard",
    "common_prob_cosine",
    "union_prob_cosine",
    "probe_logprob_cosine",
]


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def group_key(row: dict, group_by_margin: bool) -> Tuple[str, str]:
    depth_bucket = str(row.get("depth_bucket", "unknown"))
    margin_bucket = str(row.get("margin_bucket", "all")) if group_by_margin else "all"
    return depth_bucket, margin_bucket


def summarize_group(rows: Sequence[dict],
                    depth_bucket: str,
                    margin_bucket: str,
                    quantiles: Sequence[float]) -> dict:
    out = {
        "depth_bucket": depth_bucket,
        "margin_bucket": margin_bucket,
        "count": len(rows),
        "top1_mismatch_rate": (
            sum(1 for row in rows if str(row.get("top1_equal")) != "True") / len(rows)
            if rows else float("nan")
        ),
    }
    for metric in PROFILE_METRICS:
        values = [safe_float(row.get(metric)) for row in rows]
        out[f"{metric}_max"] = max(
            (value for value in values if math.isfinite(value)),
            default=float("nan"),
        )
        for q in quantiles:
            out[f"{metric}_p{int(q * 1000):03d}"] = quantile(values, q)
    for metric in LOWER_BOUND_METRICS:
        values = [safe_float(row.get(metric)) for row in rows]
        out[f"{metric}_min"] = min(
            (value for value in values if math.isfinite(value)),
            default=float("nan"),
        )
        for q in quantiles:
            # Lower tail is the risky side for similarity metrics.
            out[f"{metric}_p{int((1.0 - q) * 1000):03d}_lower"] = quantile(values, 1.0 - q)
    return out


def load_profile_rows(path: Path) -> List[dict]:
    if path.is_dir():
        path = path / "depth_metrics.csv"
    return read_csv(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate an accepted drift envelope from a same-weight cross-GPU comparison."
    )
    parser.add_argument("--comparison", required=True,
                        help="Comparison directory or depth_metrics.csv from compare-traces.")
    parser.add_argument("--output-dir", default="results/profile")
    parser.add_argument("--profile-id", default="qwen3-32b-bf16-h100-vs-6000")
    parser.add_argument("--variant-pair", default="")
    parser.add_argument("--quantiles", default="0.95,0.99,0.999")
    parser.add_argument("--group-by-margin", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--safety-multiplier", type=float, default=1.25,
                        help="Multiplier applied to upper-bound metrics in thresholds.")
    args = parser.parse_args(argv)

    rows = load_profile_rows(Path(args.comparison))
    quantiles = [
        float(item.strip())
        for item in args.quantiles.split(",")
        if item.strip()
    ]
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row, args.group_by_margin)].append(row)

    summary_rows = [
        summarize_group(group_rows, depth_bucket, margin_bucket, quantiles)
        for (depth_bucket, margin_bucket), group_rows in sorted(grouped.items())
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in summary_rows for field in row})
    write_csv(output_dir / "profile_summary.csv", summary_rows, fields)

    threshold_quantile = max(quantiles) if quantiles else 0.99
    threshold_key = f"p{int(threshold_quantile * 1000):03d}"
    thresholds = {}
    for row in summary_rows:
        bucket_key = f"{row['depth_bucket']}|{row['margin_bucket']}"
        thresholds[bucket_key] = {}
        for metric in PROFILE_METRICS:
            value = safe_float(row.get(f"{metric}_{threshold_key}"))
            thresholds[bucket_key][metric] = (
                value * args.safety_multiplier
                if math.isfinite(value) else value
            )
        for metric in LOWER_BOUND_METRICS:
            # Similarity metrics are lower-bound checks. Use the observed minimum
            # for the accepted envelope so the calibration pair can replay through
            # its own profile. The quantile columns are still emitted for analysis.
            value = safe_float(row.get(f"{metric}_min"))
            thresholds[bucket_key][metric] = value
        thresholds[bucket_key]["max_top1_mismatch_rate"] = safe_float(
            row.get("top1_mismatch_rate")
        )
        thresholds[bucket_key]["calibration_count"] = row.get("count")

    write_json(output_dir / "profile.json", {
        "metadata": runtime_metadata(sys.argv),
        "profile_id": args.profile_id,
        "variant_pair": args.variant_pair,
        "source_comparison": args.comparison,
        "quantiles": quantiles,
        "threshold_quantile": threshold_quantile,
        "safety_multiplier": args.safety_multiplier,
        "group_by_margin": args.group_by_margin,
        "metrics": {
            "upper_bound": PROFILE_METRICS,
            "lower_bound": LOWER_BOUND_METRICS,
        },
        "thresholds": thresholds,
    })
    print(f"Wrote {output_dir / 'profile_summary.csv'}")
    print(f"Wrote {output_dir / 'profile.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
