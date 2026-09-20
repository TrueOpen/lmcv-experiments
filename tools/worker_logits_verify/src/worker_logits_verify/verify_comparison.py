from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .metrics import safe_float
from .schema import read_json, runtime_metadata, write_json


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


def threshold_key(row: dict) -> str:
    return f"{row.get('depth_bucket')}|{row.get('margin_bucket')}"


def fallback_threshold_key(row: dict) -> str:
    return f"{row.get('depth_bucket')}|all"


def metric_failed(metric: str, value: float, threshold: float, lower_bound: bool) -> bool:
    if not math.isfinite(value) or not math.isfinite(threshold):
        return False
    if lower_bound:
        return value < threshold
    return value > threshold


def evaluate_row(row: dict,
                 thresholds: Dict[str, dict],
                 upper_metrics: Sequence[str],
                 lower_metrics: Sequence[str]) -> List[dict]:
    key = threshold_key(row)
    bucket = thresholds.get(key)
    if bucket is None:
        bucket = thresholds.get(fallback_threshold_key(row))
    if bucket is None:
        return [{
            "sample_id": row.get("sample_id"),
            "depth": row.get("depth"),
            "depth_bucket": row.get("depth_bucket"),
            "margin_bucket": row.get("margin_bucket"),
            "metric": "__bucket__",
            "value": "",
            "threshold": "",
            "direction": "missing_threshold",
            "failed": True,
        }]

    failures = []
    for metric in upper_metrics:
        value = safe_float(row.get(metric))
        threshold = safe_float(bucket.get(metric))
        failed = metric_failed(metric, value, threshold, lower_bound=False)
        if failed:
            failures.append({
                "sample_id": row.get("sample_id"),
                "depth": row.get("depth"),
                "depth_bucket": row.get("depth_bucket"),
                "margin_bucket": row.get("margin_bucket"),
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "direction": "<=",
                "failed": True,
            })
    for metric in lower_metrics:
        value = safe_float(row.get(metric))
        threshold = safe_float(bucket.get(metric))
        failed = metric_failed(metric, value, threshold, lower_bound=True)
        if failed:
            failures.append({
                "sample_id": row.get("sample_id"),
                "depth": row.get("depth"),
                "depth_bucket": row.get("depth_bucket"),
                "margin_bucket": row.get("margin_bucket"),
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "direction": ">=",
                "failed": True,
            })
    return failures


def write_summary(path: Path,
                  comparison: str,
                  profile: dict,
                  total_rows: int,
                  failed_rows: int,
                  failures: Sequence[dict],
                  fail_ratio: float,
                  max_fail_ratio: float) -> None:
    by_metric = Counter(str(row["metric"]) for row in failures)
    by_bucket = Counter(
        f"{row.get('depth_bucket')}|{row.get('margin_bucket')}"
        for row in failures
    )
    lines = [
        "# profile verification",
        "",
        f"- comparison: `{comparison}`",
        f"- profile: `{profile.get('profile_id')}`",
        f"- total rows: {total_rows}",
        f"- rows with failures: {failed_rows}",
        f"- fail ratio: {fail_ratio:.8g}",
        f"- allowed fail ratio: {max_fail_ratio:.8g}",
        f"- verdict: {'PASS' if fail_ratio <= max_fail_ratio else 'FAIL'}",
        "",
        "## failure metrics",
        "",
        "| metric | failures |",
        "|---|---:|",
    ]
    for metric, count in by_metric.most_common():
        lines.append(f"| {metric} | {count} |")
    lines.extend([
        "",
        "## failure buckets",
        "",
        "| bucket | failures |",
        "|---|---:|",
    ])
    for bucket, count in by_bucket.most_common():
        lines.append(f"| {bucket} | {count} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a trace comparison against a calibrated accepted envelope."
    )
    parser.add_argument("--comparison", required=True,
                        help="Comparison dir or depth_metrics.csv to verify.")
    parser.add_argument("--profile", required=True,
                        help="profile.json from calibrate-profile.")
    parser.add_argument("--output-dir", default="results/verification")
    parser.add_argument("--max-fail-ratio", type=float, default=0.0,
                        help="Allow this fraction of depth rows to exceed thresholds.")
    parser.add_argument("--fail-on-reject", action=argparse.BooleanOptionalAction,
                        default=False)
    args = parser.parse_args(argv)

    rows = read_csv(comparison_path(Path(args.comparison)))
    profile = read_json(Path(args.profile))
    thresholds = profile.get("thresholds", {})
    metrics = profile.get("metrics", {})
    upper_metrics = metrics.get("upper_bound", [])
    lower_metrics = metrics.get("lower_bound", [])

    all_failures = []
    failed_row_keys = set()
    for row in rows:
        failures = evaluate_row(row, thresholds, upper_metrics, lower_metrics)
        if failures:
            failed_row_keys.add((row.get("sample_id"), row.get("depth")))
            all_failures.extend(failures)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "depth",
        "depth_bucket",
        "margin_bucket",
        "metric",
        "value",
        "threshold",
        "direction",
        "failed",
    ]
    write_csv(output_dir / "failures.csv", all_failures, fields)

    total_rows = len(rows)
    failed_rows = len(failed_row_keys)
    fail_ratio = failed_rows / total_rows if total_rows else 1.0
    passed = fail_ratio <= args.max_fail_ratio
    write_json(output_dir / "verification.json", {
        "metadata": runtime_metadata(sys.argv),
        "args": vars(args),
        "profile_id": profile.get("profile_id"),
        "total_rows": total_rows,
        "failed_rows": failed_rows,
        "failure_count": len(all_failures),
        "fail_ratio": fail_ratio,
        "max_fail_ratio": args.max_fail_ratio,
        "passed": passed,
    })
    write_summary(
        output_dir / "summary.md",
        comparison=args.comparison,
        profile=profile,
        total_rows=total_rows,
        failed_rows=failed_rows,
        failures=all_failures,
        fail_ratio=fail_ratio,
        max_fail_ratio=args.max_fail_ratio,
    )
    print(f"Wrote {output_dir / 'failures.csv'}")
    print(f"Wrote {output_dir / 'verification.json'}")
    print(f"Wrote {output_dir / 'summary.md'}")
    print(f"Verdict: {'PASS' if passed else 'FAIL'}")
    return 1 if (not passed and args.fail_on_reject) else 0


if __name__ == "__main__":
    raise SystemExit(main())
