from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .metrics import safe_float
from .schema import runtime_metadata, write_json
from .summarize_worker_verify import summarize_rows


Threshold = Tuple[str, str, float, str]


PRESETS: Dict[str, dict] = {
    "qwen3-8b-bf16-4090-v1": {
        "profile_id": "qwen3-8b-bf16-4090-v1",
        "target": {
            "model": "Qwen/Qwen3-8B",
            "quantization": "bf16",
            "dtype": "bfloat16",
            "worker_gpu": "4090",
            "verifier_trace_mode": "vllm_prompt_logprobs_full_prefill",
            "worker_trace_mode": "vllm_decode_generate",
        },
        "requirements": {
            "min_single_output_tokens": 64,
            "min_strict_total_tokens": 512,
            "min_strict_samples": 4,
            "recommended_output_tokens_per_probe": 128,
            "recommended_probe_count": 8,
            "recommended_logprobs": 32,
            "require_topk_for_strict": True,
        },
        "sample_accept_thresholds": [
            ("missing_selected_count", "==", 0.0, "selected token must be present in verifier top-k"),
            ("mean_abs_logprob_diff", "<=", 0.020, "sample mean selected-token drift"),
            ("abs_logprob_diff_p95", "<=", 0.120, "sample p95 selected-token drift"),
            ("abs_logprob_diff_p99", "<=", 0.250, "sample p99 selected-token drift"),
            ("rank_delta_nonzero_rate", "<=", 0.030, "sample selected-token rank drift rate"),
            ("topk_jaccard_mean", ">=", 0.920, "sample top-k token-set overlap"),
            ("union_js_p99", "<=", 0.020, "sample p99 top-k union JS divergence"),
        ],
        "single_reject_thresholds": [
            ("missing_selected_count", ">", 0.0, "selected token missing from verifier top-k"),
            ("mean_abs_logprob_diff", ">", 0.030, "single-sample mean selected-token drift too high"),
            ("abs_logprob_diff_p95", ">", 0.200, "single-sample p95 selected-token drift too high"),
            ("abs_logprob_diff_p99", ">", 0.500, "single-sample p99 selected-token drift too high"),
            ("rank_delta_nonzero_rate", ">", 0.060, "single-sample selected-token rank drift too high"),
            ("topk_jaccard_mean", "<", 0.880, "single-sample top-k overlap too low"),
            ("union_js_p99", ">", 0.060, "single-sample top-k JS divergence too high"),
        ],
        "batch_accept_thresholds": [
            ("missing_selected_count", "==", 0.0, "selected token must be present in verifier top-k"),
            ("abs_logprob_diff_p50", "<=", 0.00010, "batch median selected-token drift"),
            ("mean_abs_logprob_diff", "<=", 0.015, "batch mean selected-token drift"),
            ("abs_logprob_diff_p90", "<=", 0.060, "batch p90 selected-token drift"),
            ("abs_logprob_diff_p95", "<=", 0.090, "batch p95 selected-token drift"),
            ("abs_logprob_diff_p99", "<=", 0.200, "batch p99 selected-token drift"),
            ("rank_delta_nonzero_rate", "<=", 0.012, "batch selected-token rank drift rate"),
            ("topk_jaccard_mean", ">=", 0.940, "batch mean top-k token-set overlap"),
            ("topk_jaccard_p05", ">=", 0.820, "batch p05 top-k token-set overlap"),
            ("union_js_p95", "<=", 0.004, "batch p95 top-k union JS divergence"),
            ("union_js_p99", "<=", 0.012, "batch p99 top-k union JS divergence"),
        ],
        "batch_reject_thresholds": [
            ("missing_selected_count", ">", 0.0, "selected token missing from verifier top-k"),
            ("mean_abs_logprob_diff", ">", 0.020, "batch mean selected-token drift too high"),
            ("abs_logprob_diff_p95", ">", 0.120, "batch p95 selected-token drift too high"),
            ("abs_logprob_diff_p99", ">", 0.250, "batch p99 selected-token drift too high"),
            ("rank_delta_nonzero_rate", ">", 0.030, "batch selected-token rank drift too high"),
            ("topk_jaccard_mean", "<", 0.920, "batch top-k overlap too low"),
            ("union_js_p99", ">", 0.020, "batch top-k JS divergence too high"),
        ],
        "sample_rate_thresholds": {
            "min_strict_sample_accept_rate": 0.90,
            "max_strict_sample_reject_rate": 0.02,
        },
        "calibration_note": (
            "Calibrated from qwen3_8b_4090_data_results: BF16 worker-vs-BF16 verifier "
            "overall mean ~=0.0081, p95 ~=0.0499, p99 ~=0.1018; FP8 overall mean "
            "~=0.0290, p95 ~=0.1351, p99 ~=0.3123."
        ),
    }
}


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


def metrics_path(path: Path) -> Path:
    if path.is_dir():
        direct = path / "worker_vs_verifier_depth_metrics.csv"
        if direct.exists():
            return direct
        nested = path / "distribution" / "overall.csv"
        if nested.exists():
            return nested
    return path


def is_distribution_summary(path: Path) -> bool:
    return path.name in {"overall.csv", "by_sample.csv"} and path.parent.name == "distribution"


def distribution_dir(path: Path) -> Path:
    if path.is_dir():
        return path / "distribution"
    if path.parent.name == "distribution":
        return path.parent
    return path.parent / "distribution"


def group_by_sample(rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("sample_id", ""))].append(row)
    return [
        summarize_rows(group_rows, sample_id)
        for sample_id, group_rows in sorted(grouped.items())
    ]


def load_summaries(path: Path) -> Tuple[dict, List[dict], str]:
    resolved = metrics_path(path)
    if is_distribution_summary(resolved):
        dist = distribution_dir(resolved)
        overall_rows = read_csv(dist / "overall.csv")
        sample_path = dist / "by_sample.csv"
        sample_rows = read_csv(sample_path) if sample_path.exists() else []
        if not overall_rows:
            raise ValueError(f"{dist / 'overall.csv'} is empty")
        return overall_rows[0], sample_rows, str(dist)

    rows = read_csv(resolved)
    if not rows:
        raise ValueError(f"{resolved} is empty")
    overall = summarize_rows(rows, "overall")
    return overall, group_by_sample(rows), str(resolved)


def finite(value: object) -> bool:
    return math.isfinite(safe_float(value))


def compare(value: float, op: str, threshold: float) -> bool:
    if not math.isfinite(value):
        return False
    if op == "<=":
        return value <= threshold
    if op == "<":
        return value < threshold
    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    if op == "==":
        return value == threshold
    raise ValueError(f"unsupported threshold operator: {op}")


def evaluate_thresholds(summary: dict, thresholds: Sequence[Threshold]) -> List[dict]:
    checks = []
    for metric, op, threshold, reason in thresholds:
        value = safe_float(summary.get(metric))
        passed = compare(value, op, threshold)
        checks.append({
            "metric": metric,
            "op": op,
            "threshold": threshold,
            "value": value if math.isfinite(value) else "",
            "passed": passed,
            "reason": reason,
        })
    return checks


def all_passed(checks: Sequence[dict]) -> bool:
    return all(bool(check.get("passed")) for check in checks)


def any_passed(checks: Sequence[dict]) -> bool:
    return any(bool(check.get("passed")) for check in checks)


def has_topk(summary: dict) -> bool:
    required = ["topk_jaccard_mean", "union_js_p99"]
    return all(finite(summary.get(metric)) for metric in required)


def classify_sample(summary: dict, preset: dict) -> dict:
    count = int(safe_float(summary.get("finite_count", summary.get("count", 0)), 0.0))
    accept_checks = evaluate_thresholds(summary, preset["sample_accept_thresholds"])
    reject_checks = evaluate_thresholds(summary, preset["single_reject_thresholds"])
    min_tokens = int(preset["requirements"]["min_single_output_tokens"])
    if count < min_tokens:
        verdict = "INCONCLUSIVE"
        reason = f"sample has {count} finite tokens; need at least {min_tokens}"
    elif all_passed(accept_checks):
        verdict = "PASS_PROVISIONAL"
        reason = "single sample matches the BF16 dense fingerprint"
    elif any_passed(reject_checks):
        verdict = "REJECT"
        reason = "single sample violates one or more reject thresholds"
    else:
        verdict = "INCONCLUSIVE"
        reason = "single sample is between accept and reject thresholds"
    return {
        **summary,
        "verdict": verdict,
        "reason": reason,
        "accept_failures": "; ".join(
            check["metric"] for check in accept_checks if not check["passed"]
        ),
        "reject_hits": "; ".join(
            check["metric"] for check in reject_checks if check["passed"]
        ),
    }


def verdict_for_batch(overall: dict,
                      sample_decisions: Sequence[dict],
                      preset: dict) -> Tuple[str, str, List[dict], List[dict]]:
    requirements = preset["requirements"]
    total_tokens = int(safe_float(overall.get("finite_count", overall.get("count", 0)), 0.0))
    sample_count = len(sample_decisions)
    pass_count = sum(1 for row in sample_decisions if row.get("verdict") == "PASS_PROVISIONAL")
    reject_count = sum(1 for row in sample_decisions if row.get("verdict") == "REJECT")
    pass_rate = pass_count / sample_count if sample_count else 0.0
    reject_rate = reject_count / sample_count if sample_count else 1.0
    has_required_topk = has_topk(overall)

    accept_checks = evaluate_thresholds(overall, preset["batch_accept_thresholds"])
    reject_checks = evaluate_thresholds(overall, preset["batch_reject_thresholds"])
    sample_rate = preset["sample_rate_thresholds"]
    eligibility_failures = []
    if total_tokens < int(requirements["min_strict_total_tokens"]):
        eligibility_failures.append(
            f"total finite tokens {total_tokens} < {requirements['min_strict_total_tokens']}"
        )
    if sample_count < int(requirements["min_strict_samples"]):
        eligibility_failures.append(
            f"samples {sample_count} < {requirements['min_strict_samples']}"
        )
    if requirements.get("require_topk_for_strict", False) and not has_required_topk:
        eligibility_failures.append("top-k metrics are required for strict BF16 verification")

    sample_rate_failed = []
    if pass_rate < float(sample_rate["min_strict_sample_accept_rate"]):
        sample_rate_failed.append(
            f"sample accept rate {pass_rate:.6g} < {sample_rate['min_strict_sample_accept_rate']}"
        )
    if reject_rate > float(sample_rate["max_strict_sample_reject_rate"]):
        sample_rate_failed.append(
            f"sample reject rate {reject_rate:.6g} > {sample_rate['max_strict_sample_reject_rate']}"
        )

    strict_eligible = not eligibility_failures
    if strict_eligible and all_passed(accept_checks) and not sample_rate_failed:
        return "PASS_STRICT", "batch matches the Qwen3-8B BF16 dense fingerprint", accept_checks, reject_checks
    if any_passed(reject_checks):
        return "REJECT", "batch violates one or more exact-BF16 reject thresholds", accept_checks, reject_checks
    if not strict_eligible:
        return "INCONCLUSIVE", "; ".join(eligibility_failures), accept_checks, reject_checks
    if sample_rate_failed:
        return "INCONCLUSIVE", "; ".join(sample_rate_failed), accept_checks, reject_checks
    return "INCONCLUSIVE", "batch is between accept and reject thresholds", accept_checks, reject_checks


def format_value(value: object) -> str:
    number = safe_float(value)
    if not math.isfinite(number):
        return ""
    if abs(number) < 1e-4 and number != 0.0:
        return f"{number:.3e}"
    return f"{number:.6g}"


def write_markdown(path: Path,
                   source: str,
                   preset: dict,
                   verdict: str,
                   reason: str,
                   overall: dict,
                   accept_checks: Sequence[dict],
                   reject_checks: Sequence[dict],
                   sample_decisions: Sequence[dict]) -> None:
    counts = Counter(str(row.get("verdict")) for row in sample_decisions)
    sample_count = len(sample_decisions)
    pass_rate = counts.get("PASS_PROVISIONAL", 0) / sample_count if sample_count else 0.0
    reject_rate = counts.get("REJECT", 0) / sample_count if sample_count else 0.0
    lines = [
        "# Dense BF16 verifier",
        "",
        f"- source: `{source}`",
        f"- preset: `{preset['profile_id']}`",
        f"- verdict: **{verdict}**",
        f"- reason: {reason}",
        f"- finite tokens: {int(safe_float(overall.get('finite_count', overall.get('count', 0)), 0.0))}",
        f"- samples: {sample_count}",
        f"- sample provisional pass rate: {pass_rate:.6g}",
        f"- sample reject rate: {reject_rate:.6g}",
        "",
        "## Overall metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for metric in [
        "mean_abs_logprob_diff",
        "abs_logprob_diff_p50",
        "abs_logprob_diff_p90",
        "abs_logprob_diff_p95",
        "abs_logprob_diff_p99",
        "abs_logprob_diff_p999",
        "rank_delta_nonzero_rate",
        "topk_jaccard_mean",
        "topk_jaccard_p05",
        "union_js_p95",
        "union_js_p99",
        "max_abs_logprob_diff",
        "missing_selected_count",
    ]:
        lines.append(f"| `{metric}` | {format_value(overall.get(metric))} |")

    lines.extend([
        "",
        "## Batch accept checks",
        "",
        "| metric | op | threshold | value | passed |",
        "|---|---:|---:|---:|---|",
    ])
    for check in accept_checks:
        lines.append(
            f"| `{check['metric']}` | {check['op']} | {format_value(check['threshold'])} | "
            f"{format_value(check['value'])} | {check['passed']} |"
        )

    lines.extend([
        "",
        "## Reject checks",
        "",
        "| metric | op | threshold | value | hit |",
        "|---|---:|---:|---:|---|",
    ])
    for check in reject_checks:
        lines.append(
            f"| `{check['metric']}` | {check['op']} | {format_value(check['threshold'])} | "
            f"{format_value(check['value'])} | {check['passed']} |"
        )

    lines.extend([
        "",
        "## Sample verdicts",
        "",
        "| verdict | count |",
        "|---|---:|",
    ])
    for item, count in counts.most_common():
        lines.append(f"| {item} | {count} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply dense Qwen3-8B BF16 identity thresholds to worker-vs-verifier metrics."
    )
    parser.add_argument("--metrics", required=True,
                        help="worker_vs_verifier_depth_metrics.csv, its parent dir, or distribution/overall.csv.")
    parser.add_argument("--output-dir", default="results/dense_bf16_verification")
    parser.add_argument("--preset", default="qwen3-8b-bf16-4090-v1",
                        choices=sorted(PRESETS))
    parser.add_argument("--fail-on-reject", action=argparse.BooleanOptionalAction,
                        default=False)
    args = parser.parse_args(argv)

    preset = PRESETS[args.preset]
    overall, by_sample, source = load_summaries(Path(args.metrics))
    sample_decisions = [classify_sample(row, preset) for row in by_sample]
    verdict, reason, accept_checks, reject_checks = verdict_for_batch(
        overall,
        sample_decisions,
        preset,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_fields = [
        "group",
        "verdict",
        "reason",
        "count",
        "finite_count",
        "mean_abs_logprob_diff",
        "abs_logprob_diff_p50",
        "abs_logprob_diff_p90",
        "abs_logprob_diff_p95",
        "abs_logprob_diff_p99",
        "rank_delta_nonzero_rate",
        "topk_jaccard_mean",
        "topk_jaccard_p05",
        "union_js_p95",
        "union_js_p99",
        "missing_selected_count",
        "accept_failures",
        "reject_hits",
    ]
    write_csv(output_dir / "sample_decisions.csv", sample_decisions, sample_fields)
    write_json(output_dir / "parameters.json", preset)
    counts = Counter(str(row.get("verdict")) for row in sample_decisions)
    result = {
        "metadata": runtime_metadata(sys.argv),
        "args": vars(args),
        "source": source,
        "preset": args.preset,
        "verdict": verdict,
        "passed": verdict == "PASS_STRICT",
        "reason": reason,
        "overall": overall,
        "sample_verdict_counts": dict(counts),
        "sample_count": len(sample_decisions),
        "sample_pass_rate": (
            counts.get("PASS_PROVISIONAL", 0) / len(sample_decisions)
            if sample_decisions else 0.0
        ),
        "sample_reject_rate": (
            counts.get("REJECT", 0) / len(sample_decisions)
            if sample_decisions else 1.0
        ),
        "batch_accept_checks": accept_checks,
        "batch_reject_checks": reject_checks,
    }
    write_json(output_dir / "dense_verification.json", result)
    write_markdown(
        output_dir / "summary.md",
        source=source,
        preset=preset,
        verdict=verdict,
        reason=reason,
        overall=overall,
        accept_checks=accept_checks,
        reject_checks=reject_checks,
        sample_decisions=sample_decisions,
    )
    print(f"Wrote {output_dir / 'summary.md'}")
    print(f"Wrote {output_dir / 'dense_verification.json'}")
    print(f"Wrote {output_dir / 'sample_decisions.csv'}")
    print(f"Wrote {output_dir / 'parameters.json'}")
    print(f"Verdict: {verdict}")
    return 1 if (verdict == "REJECT" and args.fail_on_reject) else 0


if __name__ == "__main__":
    raise SystemExit(main())
