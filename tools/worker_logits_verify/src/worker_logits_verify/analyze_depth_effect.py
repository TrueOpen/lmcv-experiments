from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .metrics import depth_bucket, quantile, safe_float
from .schema import runtime_metadata, write_json


DEFAULT_METRICS = [
    "selected_abs_logprob_diff",
    "selected_abs_raw_logit_diff",
    "selected_abs_rank_delta",
    "union_js_divergence",
    "union_max_abs_logprob_diff",
    "probe_max_abs_logprob_diff",
    "probe_rms_logprob_diff",
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


def comparison_path(path: Path) -> Path:
    if path.is_dir():
        return path / "depth_metrics.csv"
    return path


def parse_metrics(raw: str) -> List[str]:
    if not raw:
        return DEFAULT_METRICS
    return [item.strip() for item in raw.split(",") if item.strip()]


def finite_pairs(xs: Sequence[float], ys: Sequence[float]) -> Tuple[List[float], List[float]]:
    clean_xs: List[float] = []
    clean_ys: List[float] = []
    for x, y in zip(xs, ys):
        if math.isfinite(x) and math.isfinite(y):
            clean_xs.append(x)
            clean_ys.append(y)
    return clean_xs, clean_ys


def linear_regression(xs: Sequence[float], ys: Sequence[float]) -> dict:
    xs, ys = finite_pairs(xs, ys)
    n = len(xs)
    if n < 2:
        return {
            "count": n,
            "slope": float("nan"),
            "intercept": float("nan"),
            "pearson_r": float("nan"),
            "r_squared": float("nan"),
        }
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx == 0.0:
        slope = float("nan")
        intercept = float("nan")
    else:
        slope = sxy / sxx
        intercept = mean_y - slope * mean_x
    if sxx == 0.0 or syy == 0.0:
        pearson = float("nan")
    else:
        pearson = sxy / math.sqrt(sxx * syy)
    return {
        "count": n,
        "slope": slope,
        "intercept": intercept,
        "pearson_r": pearson,
        "r_squared": pearson * pearson if math.isfinite(pearson) else float("nan"),
    }


def add_relative_depth(rows: Sequence[dict]) -> List[dict]:
    by_sample: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_sample[str(row.get("sample_id"))].append(dict(row))
    out: List[dict] = []
    for sample_rows in by_sample.values():
        max_depth = max((int(row.get("depth", 0)) for row in sample_rows), default=0)
        denom = max(max_depth, 1)
        for row in sample_rows:
            row["relative_depth"] = int(row.get("depth", 0)) / denom
            out.append(row)
    return out


def bucket_summary(rows: Sequence[dict], metrics: Sequence[str]) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("depth_bucket") or depth_bucket(int(row.get("depth", 0))))].append(row)

    out = []
    for bucket, bucket_rows in sorted(grouped.items()):
        summary = {
            "depth_bucket": bucket,
            "count": len(bucket_rows),
            "top1_mismatch_rate": (
                sum(1 for row in bucket_rows if str(row.get("top1_equal")) != "True")
                / len(bucket_rows)
                if bucket_rows else float("nan")
            ),
        }
        for metric in metrics:
            values = [safe_float(row.get(metric)) for row in bucket_rows]
            finite = [value for value in values if math.isfinite(value)]
            summary[f"{metric}_mean"] = sum(finite) / len(finite) if finite else float("nan")
            summary[f"{metric}_p50"] = quantile(finite, 0.50)
            summary[f"{metric}_p95"] = quantile(finite, 0.95)
            summary[f"{metric}_p99"] = quantile(finite, 0.99)
            summary[f"{metric}_max"] = max(finite, default=float("nan"))
        out.append(summary)
    return out


def trend_summary(rows: Sequence[dict], metrics: Sequence[str]) -> List[dict]:
    out = []
    depth_values = [float(int(row.get("depth", 0))) for row in rows]
    relative_depth_values = [safe_float(row.get("relative_depth")) for row in rows]
    for metric in metrics:
        metric_values = [safe_float(row.get(metric)) for row in rows]
        by_depth = linear_regression(depth_values, metric_values)
        by_relative = linear_regression(relative_depth_values, metric_values)
        out.append({
            "metric": metric,
            "count": by_depth["count"],
            "slope_per_token": by_depth["slope"],
            "slope_per_100_tokens": (
                by_depth["slope"] * 100.0
                if math.isfinite(by_depth["slope"]) else float("nan")
            ),
            "pearson_r_depth": by_depth["pearson_r"],
            "r_squared_depth": by_depth["r_squared"],
            "slope_per_relative_output": by_relative["slope"],
            "pearson_r_relative_depth": by_relative["pearson_r"],
            "r_squared_relative_depth": by_relative["r_squared"],
        })
    return out


def cumulative_by_sample(rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sample_id"))].append(row)
    out = []
    for sample_id, sample_rows in grouped.items():
        sample_rows = sorted(sample_rows, key=lambda row: int(row.get("depth", 0)))
        cumulative_abs_lp = 0.0
        cumulative_signed_nll_delta = 0.0
        max_abs_lp = 0.0
        max_abs_raw = 0.0
        for row in sample_rows:
            depth = int(row.get("depth", 0))
            lp_diff = safe_float(row.get("selected_logprob_diff_right_minus_left"), 0.0)
            abs_lp = safe_float(row.get("selected_abs_logprob_diff"), 0.0)
            abs_raw = safe_float(row.get("selected_abs_raw_logit_diff"), 0.0)
            cumulative_abs_lp += abs_lp
            # NLL = -logprob, so right-left NLL delta is -logprob_diff.
            cumulative_signed_nll_delta += -lp_diff
            max_abs_lp = max(max_abs_lp, abs_lp)
            max_abs_raw = max(max_abs_raw, abs_raw)
            out.append({
                "sample_id": sample_id,
                "depth": depth,
                "depth_bucket": row.get("depth_bucket"),
                "relative_depth": row.get("relative_depth"),
                "cumulative_abs_selected_logprob_diff": cumulative_abs_lp,
                "cumulative_nll_delta_right_minus_left": cumulative_signed_nll_delta,
                "cumulative_abs_nll_delta": abs(cumulative_signed_nll_delta),
                "max_selected_abs_logprob_diff_so_far": max_abs_lp,
                "max_selected_abs_raw_logit_diff_so_far": max_abs_raw,
            })
    return out


def write_markdown(path: Path,
                   comparison: str,
                   bucket_rows: Sequence[dict],
                   trend_rows: Sequence[dict]) -> None:
    lines = [
        "# depth effect analysis",
        "",
        f"- source: `{comparison}`",
        "",
        "## trend",
        "",
        "| metric | slope / 100 tokens | pearson_r | r2 | relative slope | relative_r |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in trend_rows:
        lines.append(
            "| {metric} | {slope:.8g} | {r:.8g} | {r2:.8g} | {rel_slope:.8g} | {rel_r:.8g} |".format(
                metric=row.get("metric"),
                slope=safe_float(row.get("slope_per_100_tokens")),
                r=safe_float(row.get("pearson_r_depth")),
                r2=safe_float(row.get("r_squared_depth")),
                rel_slope=safe_float(row.get("slope_per_relative_output")),
                rel_r=safe_float(row.get("pearson_r_relative_depth")),
            )
        )
    lines.extend([
        "",
        "## depth buckets",
        "",
        "| bucket | count | top1 mismatch rate | selected lp p95 | selected raw p95 | union JS p95 | top-k union lp p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in bucket_rows:
        lines.append(
            "| {bucket} | {count} | {flip:.8g} | {lp:.8g} | {raw:.8g} | {js:.8g} | {union_lp:.8g} |".format(
                bucket=row.get("depth_bucket"),
                count=row.get("count"),
                flip=safe_float(row.get("top1_mismatch_rate")),
                lp=safe_float(row.get("selected_abs_logprob_diff_p95")),
                raw=safe_float(row.get("selected_abs_raw_logit_diff_p95")),
                js=safe_float(row.get("union_js_divergence_p95")),
                union_lp=safe_float(row.get("union_max_abs_logprob_diff_p95")),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze whether same-context drift grows with output token depth."
    )
    parser.add_argument("--comparison", required=True,
                        help="Comparison dir or depth_metrics.csv from compare-traces.")
    parser.add_argument("--output-dir", default="results/depth_effect")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                        help="Comma-separated metrics to analyze.")
    args = parser.parse_args(argv)

    comparison = comparison_path(Path(args.comparison))
    rows = add_relative_depth(read_csv(comparison))
    metrics = parse_metrics(args.metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bucket_rows = bucket_summary(rows, metrics)
    trend_rows = trend_summary(rows, metrics)
    cumulative_rows = cumulative_by_sample(rows)

    bucket_fields = sorted({field for row in bucket_rows for field in row})
    trend_fields = [
        "metric",
        "count",
        "slope_per_token",
        "slope_per_100_tokens",
        "pearson_r_depth",
        "r_squared_depth",
        "slope_per_relative_output",
        "pearson_r_relative_depth",
        "r_squared_relative_depth",
    ]
    cumulative_fields = [
        "sample_id",
        "depth",
        "depth_bucket",
        "relative_depth",
        "cumulative_abs_selected_logprob_diff",
        "cumulative_nll_delta_right_minus_left",
        "cumulative_abs_nll_delta",
        "max_selected_abs_logprob_diff_so_far",
        "max_selected_abs_raw_logit_diff_so_far",
    ]
    write_csv(output_dir / "depth_bucket_summary.csv", bucket_rows, bucket_fields)
    write_csv(output_dir / "depth_trends.csv", trend_rows, trend_fields)
    write_csv(output_dir / "cumulative_by_sample.csv", cumulative_rows, cumulative_fields)
    write_markdown(output_dir / "summary.md", str(args.comparison), bucket_rows, trend_rows)
    write_json(output_dir / "metadata.json", {
        "metadata": runtime_metadata(sys.argv),
        "args": vars(args),
        "row_count": len(rows),
        "metrics": metrics,
    })

    print(f"Wrote {output_dir / 'depth_bucket_summary.csv'}")
    print(f"Wrote {output_dir / 'depth_trends.csv'}")
    print(f"Wrote {output_dir / 'cumulative_by_sample.csv'}")
    print(f"Wrote {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
