from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .metrics import quantile, safe_float
from .schema import runtime_metadata, write_json
from .summarize_worker_verify import summarize_rows


MetricDirection = str


METRIC_DIRECTIONS: Dict[str, MetricDirection] = {
    "missing_selected_count": "low",
    "mean_abs_logprob_diff": "low",
    "abs_logprob_diff_p50": "low",
    "abs_logprob_diff_p90": "low",
    "abs_logprob_diff_p95": "low",
    "abs_logprob_diff_p99": "low",
    "abs_logprob_diff_p999": "low",
    "rank_delta_nonzero_rate": "low",
    "union_js_p95": "low",
    "union_js_p99": "low",
    "union_js_max": "low",
    "topk_jaccard_mean": "high",
    "topk_jaccard_p01": "high",
    "topk_jaccard_p05": "high",
}

DEFAULT_METRICS = [
    "missing_selected_count",
    "mean_abs_logprob_diff",
    "abs_logprob_diff_p95",
    "abs_logprob_diff_p99",
    "rank_delta_nonzero_rate",
    "topk_jaccard_mean",
    "union_js_p99",
]


@dataclass
class CaseData:
    label: str
    path: Path
    groups: Dict[str, List[dict]]

    @property
    def sample_ids(self) -> List[str]:
        return sorted(self.groups)

    @property
    def sample_count(self) -> int:
        return len(self.groups)

    @property
    def row_count(self) -> int:
        return sum(len(rows) for rows in self.groups.values())


@dataclass
class ResampleResult:
    summary: dict
    actual_samples: int
    actual_tokens: int


@dataclass
class Threshold:
    metric: str
    direction: MetricDirection
    value: float


def log(message: str) -> None:
    print(message, flush=True)


def parse_int_list(raw: str) -> List[int]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"budget values must be positive: {value}")
        out.append(value)
    return sorted(set(out))


def parse_metrics(raw: str) -> List[str]:
    metrics = [item.strip() for item in raw.split(",") if item.strip()]
    if not metrics:
        raise ValueError("at least one metric is required")
    unknown = [metric for metric in metrics if metric not in METRIC_DIRECTIONS]
    if unknown:
        known = ", ".join(sorted(METRIC_DIRECTIONS))
        raise ValueError(f"unknown metric(s): {', '.join(unknown)}. Known metrics: {known}")
    return metrics


def parse_case_spec(raw: str) -> Tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"empty case label in {raw!r}")
        return label, Path(path.strip())
    path = Path(raw.strip())
    label = path.parent.name if path.name == "worker_vs_verifier_depth_metrics.csv" else path.name
    return label or "case", path


def metrics_path(path: Path) -> Path:
    if path.is_dir():
        direct = path / "worker_vs_verifier_depth_metrics.csv"
        if direct.exists():
            return direct
    return path


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_case(raw: str) -> CaseData:
    label, raw_path = parse_case_spec(raw)
    path = metrics_path(raw_path)
    rows = read_csv(path)
    groups: Dict[str, List[dict]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"{path} has a row without sample_id")
        groups.setdefault(sample_id, []).append(row)
    if not groups:
        raise ValueError(f"{path} has no sample rows")
    return CaseData(label=label, path=path, groups=groups)


def finite_token_count(rows: Sequence[dict]) -> int:
    return sum(
        1
        for row in rows
        if math.isfinite(safe_float(row.get("abs_logprob_diff")))
    )


def choose_sample_ids(case: CaseData,
                      rng: random.Random,
                      *,
                      budget_type: str,
                      budget_target: int,
                      with_replacement: bool) -> List[str]:
    sample_ids = case.sample_ids
    if budget_type == "sample":
        if with_replacement:
            return [rng.choice(sample_ids) for _ in range(budget_target)]
        if budget_target > len(sample_ids):
            raise ValueError(
                f"sample budget {budget_target} exceeds available samples "
                f"{len(sample_ids)} for {case.label}; use --with-replacement"
            )
        return rng.sample(sample_ids, budget_target)

    if budget_type != "token":
        raise ValueError(f"unsupported budget type: {budget_type}")

    chosen: List[str] = []
    total_tokens = 0
    if with_replacement:
        while total_tokens < budget_target:
            sample_id = rng.choice(sample_ids)
            chosen.append(sample_id)
            total_tokens += finite_token_count(case.groups[sample_id])
        return chosen

    shuffled = sample_ids[:]
    rng.shuffle(shuffled)
    for sample_id in shuffled:
        chosen.append(sample_id)
        total_tokens += finite_token_count(case.groups[sample_id])
        if total_tokens >= budget_target:
            break
    if total_tokens < budget_target:
        raise ValueError(
            f"token budget {budget_target} exceeds available finite tokens "
            f"{total_tokens} for {case.label}; use --with-replacement"
        )
    return chosen


def resample_case(case: CaseData,
                  rng: random.Random,
                  *,
                  budget_type: str,
                  budget_target: int,
                  with_replacement: bool) -> ResampleResult:
    sample_ids = choose_sample_ids(
        case,
        rng,
        budget_type=budget_type,
        budget_target=budget_target,
        with_replacement=with_replacement,
    )
    rows: List[dict] = []
    for sample_id in sample_ids:
        rows.extend(case.groups[sample_id])
    summary = summarize_rows(rows, case.label)
    return ResampleResult(
        summary=summary,
        actual_samples=len(sample_ids),
        actual_tokens=int(safe_float(summary.get("finite_count"), 0.0)),
    )


def build_resamples(case: CaseData,
                    *,
                    budget_type: str,
                    budget_target: int,
                    trials: int,
                    seed: int,
                    with_replacement: bool) -> List[ResampleResult]:
    rng = random.Random(seed)
    return [
        resample_case(
            case,
            rng,
            budget_type=budget_type,
            budget_target=budget_target,
            with_replacement=with_replacement,
        )
        for _ in range(trials)
    ]


def values_for_metric(resamples: Sequence[ResampleResult], metric: str) -> List[float]:
    values = [safe_float(item.summary.get(metric)) for item in resamples]
    return [value for value in values if math.isfinite(value)]


def derive_thresholds(positive: Sequence[ResampleResult],
                      metrics: Sequence[str],
                      *,
                      max_false_reject_rate: float,
                      bonferroni: bool) -> List[Threshold]:
    if max_false_reject_rate <= 0.0 or max_false_reject_rate >= 1.0:
        raise ValueError("--max-false-reject-rate must be in (0, 1)")
    alpha = (
        max_false_reject_rate / max(1, len(metrics))
        if bonferroni else max_false_reject_rate
    )
    thresholds: List[Threshold] = []
    for metric in metrics:
        values = values_for_metric(positive, metric)
        if not values:
            continue
        direction = METRIC_DIRECTIONS[metric]
        if direction == "low":
            value = quantile(values, 1.0 - alpha)
        elif direction == "high":
            value = quantile(values, alpha)
        else:
            raise ValueError(f"unsupported metric direction: {direction}")
        thresholds.append(Threshold(metric=metric, direction=direction, value=value))
    return thresholds


def is_rejected(summary: Mapping[str, object], thresholds: Sequence[Threshold]) -> bool:
    for threshold in thresholds:
        value = safe_float(summary.get(threshold.metric))
        if not math.isfinite(value):
            return True
        if threshold.direction == "low" and value > threshold.value:
            return True
        if threshold.direction == "high" and value < threshold.value:
            return True
    return False


def rejection_rate(resamples: Sequence[ResampleResult], thresholds: Sequence[Threshold]) -> float:
    if not resamples:
        return float("nan")
    rejected = sum(1 for item in resamples if is_rejected(item.summary, thresholds))
    return rejected / len(resamples)


def summarize_actuals(resamples: Sequence[ResampleResult], attr: str) -> dict:
    values = [float(getattr(item, attr)) for item in resamples]
    return {
        f"{attr}_mean": sum(values) / len(values) if values else float("nan"),
        f"{attr}_p05": quantile(values, 0.05),
        f"{attr}_p95": quantile(values, 0.95),
    }


def metric_curve_rows(case_label: str,
                      budget_type: str,
                      budget_target: int,
                      resamples: Sequence[ResampleResult],
                      metrics: Sequence[str]) -> List[dict]:
    rows = []
    for metric in metrics:
        values = values_for_metric(resamples, metric)
        if not values:
            rows.append({
                "case": case_label,
                "budget_type": budget_type,
                "budget_target": budget_target,
                "metric": metric,
                "direction": METRIC_DIRECTIONS[metric],
                "finite_trials": 0,
            })
            continue
        q01 = quantile(values, 0.01)
        q05 = quantile(values, 0.05)
        q50 = quantile(values, 0.50)
        q95 = quantile(values, 0.95)
        q99 = quantile(values, 0.99)
        rows.append({
            "case": case_label,
            "budget_type": budget_type,
            "budget_target": budget_target,
            "metric": metric,
            "direction": METRIC_DIRECTIONS[metric],
            "finite_trials": len(values),
            "mean": sum(values) / len(values),
            "q01": q01,
            "q05": q05,
            "q50": q50,
            "q95": q95,
            "q99": q99,
            "ci90_width": q95 - q05,
            "ci98_width": q99 - q01,
        })
    return rows


def threshold_rows(budget_type: str,
                   budget_target: int,
                   thresholds: Sequence[Threshold],
                   positive: Sequence[ResampleResult],
                   metrics: Sequence[str]) -> List[dict]:
    by_metric = {threshold.metric: threshold for threshold in thresholds}
    rows = []
    for metric in metrics:
        values = values_for_metric(positive, metric)
        threshold = by_metric.get(metric)
        rows.append({
            "budget_type": budget_type,
            "budget_target": budget_target,
            "metric": metric,
            "direction": METRIC_DIRECTIONS[metric],
            "threshold": threshold.value if threshold else "",
            "positive_q01": quantile(values, 0.01),
            "positive_q05": quantile(values, 0.05),
            "positive_q50": quantile(values, 0.50),
            "positive_q95": quantile(values, 0.95),
            "positive_q99": quantile(values, 0.99),
        })
    return rows


def analyze_budget(positive_case: CaseData,
                   negative_cases: Sequence[CaseData],
                   *,
                   budget_type: str,
                   budget_target: int,
                   metrics: Sequence[str],
                   trials: int,
                   seed: int,
                   with_replacement: bool,
                   max_false_reject_rate: float,
                   max_false_accept_rate: float,
                   bonferroni: bool) -> Tuple[dict, List[dict], List[dict]]:
    positive = build_resamples(
        positive_case,
        budget_type=budget_type,
        budget_target=budget_target,
        trials=trials,
        seed=seed,
        with_replacement=with_replacement,
    )
    thresholds = derive_thresholds(
        positive,
        metrics,
        max_false_reject_rate=max_false_reject_rate,
        bonferroni=bonferroni,
    )
    positive_reject_rate = rejection_rate(positive, thresholds)

    metric_rows = metric_curve_rows(
        positive_case.label,
        budget_type,
        budget_target,
        positive,
        metrics,
    )

    worst_negative_accept_rate = float("nan")
    worst_negative_label = ""
    negative_rates: Dict[str, float] = {}
    for index, negative_case in enumerate(negative_cases, start=1):
        negative = build_resamples(
            negative_case,
            budget_type=budget_type,
            budget_target=budget_target,
            trials=trials,
            seed=seed + index * 100_003,
            with_replacement=with_replacement,
        )
        negative_reject_rate = rejection_rate(negative, thresholds)
        negative_accept_rate = 1.0 - negative_reject_rate
        negative_rates[negative_case.label] = negative_accept_rate
        if not math.isfinite(worst_negative_accept_rate) or negative_accept_rate > worst_negative_accept_rate:
            worst_negative_accept_rate = negative_accept_rate
            worst_negative_label = negative_case.label
        metric_rows.extend(
            metric_curve_rows(
                negative_case.label,
                budget_type,
                budget_target,
                negative,
                metrics,
            )
        )

    actual_sample_stats = summarize_actuals(positive, "actual_samples")
    actual_token_stats = summarize_actuals(positive, "actual_tokens")
    meets_targets = (
        positive_reject_rate <= max_false_reject_rate
        and (
            not negative_cases
            or worst_negative_accept_rate <= max_false_accept_rate
        )
    )
    summary = {
        "budget_type": budget_type,
        "budget_target": budget_target,
        "trials": trials,
        "threshold_metric_count": len(thresholds),
        "positive_label": positive_case.label,
        "positive_reject_rate": positive_reject_rate,
        "positive_accept_rate": 1.0 - positive_reject_rate,
        "worst_negative_accept_rate": worst_negative_accept_rate,
        "worst_negative_label": worst_negative_label,
        "meets_targets": meets_targets,
        **actual_sample_stats,
        **actual_token_stats,
    }
    for label, rate in sorted(negative_rates.items()):
        summary[f"negative_accept_rate__{label}"] = rate

    return summary, metric_rows, threshold_rows(
        budget_type,
        budget_target,
        thresholds,
        positive,
        metrics,
    )


def format_number(value: object) -> str:
    number = safe_float(value)
    if not math.isfinite(number):
        return ""
    if number != 0.0 and abs(number) < 1e-4:
        return f"{number:.3e}"
    return f"{number:.6g}"


def first_recommendation(rows: Sequence[dict], budget_type: str) -> Optional[dict]:
    candidates = [
        row for row in rows
        if row.get("budget_type") == budget_type and bool(row.get("meets_targets"))
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: int(row["budget_target"]))[0]


def write_markdown(path: Path,
                   *,
                   positive_case: CaseData,
                   negative_cases: Sequence[CaseData],
                   metrics: Sequence[str],
                   budget_rows: Sequence[dict],
                   max_false_reject_rate: float,
                   max_false_accept_rate: float,
                   with_replacement: bool,
                   bonferroni: bool) -> None:
    sample_rec = first_recommendation(budget_rows, "sample")
    token_rec = first_recommendation(budget_rows, "token")
    lines = [
        "# Worker Verify Sample Size Estimate",
        "",
        f"- positive case: `{positive_case.label}` ({positive_case.sample_count} samples, {positive_case.row_count} rows)",
        f"- negative cases: {', '.join(case.label for case in negative_cases) or 'none'}",
        f"- metrics: {', '.join(f'`{metric}`' for metric in metrics)}",
        f"- max false reject rate: {max_false_reject_rate:.6g}",
        f"- max false accept rate: {max_false_accept_rate:.6g}",
        f"- resampling: {'bootstrap with replacement' if with_replacement else 'subsample without replacement'}",
        f"- per-metric threshold correction: {'Bonferroni' if bonferroni else 'none'}",
        "",
        "## Recommendation",
        "",
    ]
    if sample_rec:
        lines.append(
            "- minimum sample budget: `{}` samples "
            "(positive reject `{}`, worst negative accept `{}` from `{}`)".format(
                sample_rec["budget_target"],
                format_number(sample_rec.get("positive_reject_rate")),
                format_number(sample_rec.get("worst_negative_accept_rate")),
                sample_rec.get("worst_negative_label", ""),
            )
        )
    else:
        lines.append("- minimum sample budget: not found in the evaluated grid")
    if token_rec:
        lines.append(
            "- minimum token budget: `{}` output-token rows "
            "(mean samples `{}`, positive reject `{}`, worst negative accept `{}` from `{}`)".format(
                token_rec["budget_target"],
                format_number(token_rec.get("actual_samples_mean")),
                format_number(token_rec.get("positive_reject_rate")),
                format_number(token_rec.get("worst_negative_accept_rate")),
                token_rec.get("worst_negative_label", ""),
            )
        )
    else:
        lines.append("- minimum token budget: not found in the evaluated grid")

    lines.extend([
        "",
        "Use the recommendation as a calibration estimate, then validate the chosen budget on a held-out prompt set.",
        "",
        "## Budget Curve",
        "",
        "| budget | positive reject | worst negative accept | worst negative | mean samples | mean tokens | meets |",
        "|---|---:|---:|---|---:|---:|---|",
    ])
    for row in sorted(budget_rows, key=lambda item: (str(item["budget_type"]), int(item["budget_target"]))):
        lines.append(
            "| {kind}={target} | {pos} | {neg} | {label} | {samples} | {tokens} | {meets} |".format(
                kind=row.get("budget_type"),
                target=row.get("budget_target"),
                pos=format_number(row.get("positive_reject_rate")),
                neg=format_number(row.get("worst_negative_accept_rate")),
                label=row.get("worst_negative_label", ""),
                samples=format_number(row.get("actual_samples_mean")),
                tokens=format_number(row.get("actual_tokens_mean")),
                meets=row.get("meets_targets"),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate how many worker-verifier samples or output-token rows are "
            "needed for stable model-identity decisions using block bootstrap by sample_id."
        )
    )
    parser.add_argument("--positive", required=True,
                        help="Positive/same-model metrics CSV or directory. Accepts label=path.")
    parser.add_argument("--negative", action="append", default=[],
                        help="Negative/challenger metrics CSV or directory. Accepts label=path. Repeatable.")
    parser.add_argument("--output-dir", default="results/sample_size_estimate")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                        help="Comma-separated summary metrics used by the combined reject rule.")
    parser.add_argument("--sample-counts", default="1,2,4,8,16,32,64,128",
                        help="Comma-separated whole-sample budgets to evaluate.")
    parser.add_argument("--token-counts", default="128,256,512,1024,2048,4096,8192,16384",
                        help="Comma-separated output-token row budgets to evaluate. Use empty string to disable.")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-false-reject-rate", type=float, default=0.01,
                        help="Allowed same-model rejection rate.")
    parser.add_argument("--max-false-accept-rate", type=float, default=0.01,
                        help="Allowed challenger acceptance rate.")
    parser.add_argument("--with-replacement", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Bootstrap whole samples with replacement. Disable for finite-pool subsampling.")
    parser.add_argument("--bonferroni", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Divide the positive false-reject budget across metrics when setting thresholds.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.trials <= 0:
        parser.error("--trials must be positive")

    metrics = parse_metrics(args.metrics)
    sample_counts = parse_int_list(args.sample_counts) if args.sample_counts.strip() else []
    token_counts = parse_int_list(args.token_counts) if args.token_counts.strip() else []
    if not sample_counts and not token_counts:
        parser.error("at least one of --sample-counts or --token-counts must be non-empty")

    positive_case = load_case(args.positive)
    negative_cases = [load_case(raw) for raw in args.negative]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    budget_rows: List[dict] = []
    metric_rows: List[dict] = []
    thresholds: List[dict] = []
    for budget_type, budgets in [("sample", sample_counts), ("token", token_counts)]:
        for budget_target in budgets:
            log(f"Analyzing {budget_type} budget {budget_target} with {args.trials} trials")
            summary, metric_curve, threshold_curve = analyze_budget(
                positive_case,
                negative_cases,
                budget_type=budget_type,
                budget_target=budget_target,
                metrics=metrics,
                trials=args.trials,
                seed=args.seed + budget_target * 1009 + (0 if budget_type == "sample" else 1_000_000),
                with_replacement=args.with_replacement,
                max_false_reject_rate=args.max_false_reject_rate,
                max_false_accept_rate=args.max_false_accept_rate,
                bonferroni=args.bonferroni,
            )
            budget_rows.append(summary)
            metric_rows.extend(metric_curve)
            thresholds.extend(threshold_curve)

    budget_fields = sorted({field for row in budget_rows for field in row})
    metric_fields = [
        "case",
        "budget_type",
        "budget_target",
        "metric",
        "direction",
        "finite_trials",
        "mean",
        "q01",
        "q05",
        "q50",
        "q95",
        "q99",
        "ci90_width",
        "ci98_width",
    ]
    threshold_fields = [
        "budget_type",
        "budget_target",
        "metric",
        "direction",
        "threshold",
        "positive_q01",
        "positive_q05",
        "positive_q50",
        "positive_q95",
        "positive_q99",
    ]
    write_csv(output_dir / "budget_curve.csv", budget_rows, budget_fields)
    write_csv(output_dir / "metric_curve.csv", metric_rows, metric_fields)
    write_csv(output_dir / "thresholds_by_budget.csv", thresholds, threshold_fields)
    write_markdown(
        output_dir / "summary.md",
        positive_case=positive_case,
        negative_cases=negative_cases,
        metrics=metrics,
        budget_rows=budget_rows,
        max_false_reject_rate=args.max_false_reject_rate,
        max_false_accept_rate=args.max_false_accept_rate,
        with_replacement=args.with_replacement,
        bonferroni=args.bonferroni,
    )
    write_json(output_dir / "metadata.json", {
        "metadata": runtime_metadata(sys.argv),
        "args": vars(args),
        "positive": {
            "label": positive_case.label,
            "path": str(positive_case.path),
            "sample_count": positive_case.sample_count,
            "row_count": positive_case.row_count,
        },
        "negative": [
            {
                "label": case.label,
                "path": str(case.path),
                "sample_count": case.sample_count,
                "row_count": case.row_count,
            }
            for case in negative_cases
        ],
        "metrics": metrics,
    })
    log(f"Wrote {output_dir / 'summary.md'}")
    log(f"Wrote {output_dir / 'budget_curve.csv'}")
    log(f"Wrote {output_dir / 'metric_curve.csv'}")
    log(f"Wrote {output_dir / 'thresholds_by_budget.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
