from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .metrics import (
    cosine,
    depth_bucket,
    js_divergence,
    margin_bucket,
    normalize_logprobs,
    probability_metrics,
    safe_float,
    sorted_token_ids,
)
from .schema import read_json, read_jsonl, runtime_metadata, write_json


TraceKey = Tuple[str, int]


def load_trace_dir(path: Path) -> Tuple[dict, List[dict]]:
    if path.is_dir():
        metadata_path = path / "metadata.json"
        trace_path = path / "trace.jsonl"
    else:
        metadata_path = path.with_name("metadata.json")
        trace_path = path
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    rows = list(read_jsonl(trace_path))
    return metadata, rows


def key_row(row: dict) -> TraceKey:
    return str(row["sample_id"]), int(row["depth"])


def top_map(row: dict) -> Dict[str, dict]:
    out = {}
    for item in row.get("top_logprobs", []):
        token_id = str(item.get("token_id"))
        out[token_id] = item
    return out


def numeric_delta(right: object, left: object) -> float:
    return safe_float(right) - safe_float(left)


def abs_delta(right: object, left: object) -> float:
    return abs(numeric_delta(right, left))


def selected_probe_metrics(left: dict, right: dict) -> dict:
    left_probe = left.get("probe_logprobs", {}) or {}
    right_probe = right.get("probe_logprobs", {}) or {}
    keys = sorted_token_ids(set(left_probe) & set(right_probe))
    if not keys:
        return {
            "probe_common_count": 0,
            "probe_logprob_cosine": float("nan"),
            "probe_max_abs_logprob_diff": float("nan"),
            "probe_mean_abs_logprob_diff": float("nan"),
            "probe_rms_logprob_diff": float("nan"),
        }
    left_values = [safe_float(left_probe[key]) for key in keys]
    right_values = [safe_float(right_probe[key]) for key in keys]
    diffs = [right_value - left_value
             for left_value, right_value in zip(left_values, right_values)]
    abs_diffs = [abs(value) for value in diffs]
    return {
        "probe_common_count": len(keys),
        "probe_logprob_cosine": cosine(left_values, right_values),
        "probe_max_abs_logprob_diff": max(abs_diffs),
        "probe_mean_abs_logprob_diff": sum(abs_diffs) / len(abs_diffs),
        "probe_rms_logprob_diff": math.sqrt(sum(value * value for value in diffs) / len(diffs)),
    }


def topk_logprob_vector_metrics(left_top: Mapping[str, Mapping[str, object]],
                                right_top: Mapping[str, Mapping[str, object]],
                                token_ids: Sequence[str],
                                missing_logprob: float) -> dict:
    left_values = [
        safe_float(left_top[token_id].get("logprob"))
        if token_id in left_top else missing_logprob
        for token_id in token_ids
    ]
    right_values = [
        safe_float(right_top[token_id].get("logprob"))
        if token_id in right_top else missing_logprob
        for token_id in token_ids
    ]
    diffs = [right_value - left_value
             for left_value, right_value in zip(left_values, right_values)]
    abs_diffs = [abs(value) for value in diffs]
    if not diffs:
        return {
            "logprob_cosine": float("nan"),
            "max_abs_logprob_diff": float("nan"),
            "mean_abs_logprob_diff": float("nan"),
            "rms_logprob_diff": float("nan"),
        }
    return {
        "logprob_cosine": cosine(left_values, right_values),
        "max_abs_logprob_diff": max(abs_diffs),
        "mean_abs_logprob_diff": sum(abs_diffs) / len(abs_diffs),
        "rms_logprob_diff": math.sqrt(sum(value * value for value in diffs) / len(diffs)),
    }


def compare_rows(left: dict,
                 right: dict,
                 missing_logprob: float,
                 left_label: str,
                 right_label: str) -> dict:
    left_top = top_map(left)
    right_top = top_map(right)
    left_tokens = set(left_top)
    right_tokens = set(right_top)
    common_tokens = sorted_token_ids(left_tokens & right_tokens)
    union_tokens = sorted_token_ids(left_tokens | right_tokens)

    common_prob = probability_metrics(
        left_top,
        right_top,
        common_tokens,
        missing_logprob,
    )
    union_prob = probability_metrics(
        left_top,
        right_top,
        union_tokens,
        missing_logprob,
    )
    common_logprob = topk_logprob_vector_metrics(
        left_top,
        right_top,
        common_tokens,
        missing_logprob,
    )
    union_logprob = topk_logprob_vector_metrics(
        left_top,
        right_top,
        union_tokens,
        missing_logprob,
    )
    probe_metrics = selected_probe_metrics(left, right)

    selected_lp_diff = numeric_delta(
        right.get("selected_logprob"),
        left.get("selected_logprob"),
    )
    selected_raw_logit_diff = numeric_delta(
        right.get("selected_raw_logit"),
        left.get("selected_raw_logit"),
    )
    rank_delta = int(right.get("selected_rank", 0)) - int(left.get("selected_rank", 0))
    left_margin = safe_float(left.get("top1_top2_margin_logprob"))
    right_margin = safe_float(right.get("top1_top2_margin_logprob"))

    return {
        "sample_id": left.get("sample_id"),
        "depth": int(left.get("depth")),
        "depth_bucket": depth_bucket(int(left.get("depth"))),
        "input_token_count": left.get("input_token_count"),
        "output_token_count": left.get("output_token_count"),
        "target_token_id": left.get("target_token_id"),
        "target_decoded_token": left.get("target_decoded_token"),
        "left_variant_id": left.get("variant_id", left_label),
        "right_variant_id": right.get("variant_id", right_label),
        "same_target_token": left.get("target_token_id") == right.get("target_token_id"),
        "selected_logprob_left": left.get("selected_logprob"),
        "selected_logprob_right": right.get("selected_logprob"),
        "selected_logprob_diff_right_minus_left": selected_lp_diff,
        "selected_abs_logprob_diff": abs(selected_lp_diff),
        "selected_raw_logit_left": left.get("selected_raw_logit"),
        "selected_raw_logit_right": right.get("selected_raw_logit"),
        "selected_raw_logit_diff_right_minus_left": selected_raw_logit_diff,
        "selected_abs_raw_logit_diff": abs(selected_raw_logit_diff),
        "selected_rank_left": left.get("selected_rank"),
        "selected_rank_right": right.get("selected_rank"),
        "selected_rank_delta_right_minus_left": rank_delta,
        "selected_abs_rank_delta": abs(rank_delta),
        "left_top1_token_id": left.get("top1_token_id"),
        "right_top1_token_id": right.get("top1_token_id"),
        "top1_equal": left.get("top1_token_id") == right.get("top1_token_id"),
        "left_top1_logprob": left.get("top1_logprob"),
        "right_top1_logprob": right.get("top1_logprob"),
        "left_margin": left_margin,
        "right_margin": right_margin,
        "min_margin": min(left_margin, right_margin),
        "margin_bucket": margin_bucket(min(left_margin, right_margin)),
        "left_top_k_mass": left.get("top_k_mass"),
        "right_top_k_mass": right.get("top_k_mass"),
        "top_k_mass_diff_right_minus_left": numeric_delta(
            right.get("top_k_mass"),
            left.get("top_k_mass"),
        ),
        "left_top_count": len(left_tokens),
        "right_top_count": len(right_tokens),
        "common_top_count": len(common_tokens),
        "union_top_count": len(union_tokens),
        "topk_jaccard": (
            len(common_tokens) / len(union_tokens)
            if union_tokens else float("nan")
        ),
        "missing_from_right_count": len(left_tokens - right_tokens),
        "extra_in_right_count": len(right_tokens - left_tokens),
        "common_prob_cosine": common_prob["prob_cosine"],
        "common_kl_left_to_right": common_prob["kl_left_to_right"],
        "common_kl_right_to_left": common_prob["kl_right_to_left"],
        "common_js_divergence": common_prob["js_divergence"],
        "union_prob_cosine": union_prob["prob_cosine"],
        "union_kl_left_to_right": union_prob["kl_left_to_right"],
        "union_kl_right_to_left": union_prob["kl_right_to_left"],
        "union_js_divergence": union_prob["js_divergence"],
        "common_logprob_cosine": common_logprob["logprob_cosine"],
        "common_max_abs_logprob_diff": common_logprob["max_abs_logprob_diff"],
        "common_mean_abs_logprob_diff": common_logprob["mean_abs_logprob_diff"],
        "common_rms_logprob_diff": common_logprob["rms_logprob_diff"],
        "union_logprob_cosine": union_logprob["logprob_cosine"],
        "union_max_abs_logprob_diff": union_logprob["max_abs_logprob_diff"],
        "union_mean_abs_logprob_diff": union_logprob["mean_abs_logprob_diff"],
        "union_rms_logprob_diff": union_logprob["rms_logprob_diff"],
        **probe_metrics,
    }


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def cumulative_rows(rows: Sequence[dict]) -> List[dict]:
    by_sample: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_sample[str(row["sample_id"])].append(row)
    out = []
    for sample_id, sample_rows in by_sample.items():
        sample_rows = sorted(sample_rows, key=lambda row: int(row["depth"]))
        cumulative_left = 0.0
        cumulative_right = 0.0
        max_abs_lp = 0.0
        max_abs_raw = 0.0
        top1_mismatches = 0
        for row in sample_rows:
            left_lp = safe_float(row.get("selected_logprob_left"))
            right_lp = safe_float(row.get("selected_logprob_right"))
            cumulative_left += -left_lp
            cumulative_right += -right_lp
            max_abs_lp = max(max_abs_lp, safe_float(row.get("selected_abs_logprob_diff"), 0.0))
            max_abs_raw = max(max_abs_raw, safe_float(row.get("selected_abs_raw_logit_diff"), 0.0))
            if not row.get("top1_equal"):
                top1_mismatches += 1
            out.append({
                "sample_id": sample_id,
                "depth": row["depth"],
                "depth_bucket": row["depth_bucket"],
                "cumulative_nll_left": cumulative_left,
                "cumulative_nll_right": cumulative_right,
                "cumulative_nll_delta_right_minus_left": cumulative_right - cumulative_left,
                "cumulative_abs_nll_delta": abs(cumulative_right - cumulative_left),
                "max_selected_abs_logprob_diff_so_far": max_abs_lp,
                "max_selected_abs_raw_logit_diff_so_far": max_abs_raw,
                "top1_mismatches_so_far": top1_mismatches,
            })
    return out


def markdown_summary(rows: Sequence[dict],
                     left_label: str,
                     right_label: str) -> str:
    if not rows:
        return "# trace comparison\n\nNo matched rows.\n"
    top1_mismatches = sum(1 for row in rows if not row.get("top1_equal"))
    max_lp = max(safe_float(row.get("selected_abs_logprob_diff"), 0.0) for row in rows)
    max_raw = max(safe_float(row.get("selected_abs_raw_logit_diff"), 0.0) for row in rows)
    min_jaccard = min(safe_float(row.get("topk_jaccard"), 1.0) for row in rows)
    max_union_js = max(safe_float(row.get("union_js_divergence"), 0.0) for row in rows)

    sorted_rows = sorted(
        rows,
        key=lambda row: safe_float(row.get("selected_abs_logprob_diff"), 0.0),
        reverse=True,
    )
    lines = [
        "# trace comparison",
        "",
        f"- left: `{left_label}`",
        f"- right: `{right_label}`",
        f"- matched depth rows: {len(rows)}",
        f"- top1 mismatches: {top1_mismatches}",
        f"- max selected abs logprob diff: {max_lp:.8g}",
        f"- max selected abs raw logit diff: {max_raw:.8g}",
        f"- min top-k Jaccard: {min_jaccard:.8g}",
        f"- max top-k observed union JS: {max_union_js:.8g}",
        "",
        "| sample | depth | target | abs_lp_diff | abs_raw_diff | rank_l/r | top1_equal | jaccard | union_js | margin |",
        "|---|---:|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in sorted_rows[:50]:
        lines.append(
            "| {sample} | {depth} | `{target}` | {lp:.8g} | {raw:.8g} | {rank_l}/{rank_r} | "
            "{top1} | {jaccard:.8g} | {js:.8g} | {margin:.8g} |".format(
                sample=row.get("sample_id"),
                depth=row.get("depth"),
                target=str(row.get("target_decoded_token", "")).replace("\n", "\\n"),
                lp=safe_float(row.get("selected_abs_logprob_diff")),
                raw=safe_float(row.get("selected_abs_raw_logit_diff")),
                rank_l=row.get("selected_rank_left"),
                rank_r=row.get("selected_rank_right"),
                top1=row.get("top1_equal"),
                jaccard=safe_float(row.get("topk_jaccard")),
                js=safe_float(row.get("union_js_divergence")),
                margin=safe_float(row.get("min_margin")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two depth-aware trace directories on the same canonical output path."
    )
    parser.add_argument("--left", required=True, help="Left trace dir or trace.jsonl")
    parser.add_argument("--right", required=True, help="Right trace dir or trace.jsonl")
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--output-dir", default="results/compare")
    parser.add_argument("--missing-logprob", type=float, default=-100.0)
    args = parser.parse_args(argv)

    left_meta, left_rows = load_trace_dir(Path(args.left))
    right_meta, right_rows = load_trace_dir(Path(args.right))
    left_by_key = {key_row(row): row for row in left_rows}
    right_by_key = {key_row(row): row for row in right_rows}
    keys = sorted(set(left_by_key) & set(right_by_key))

    rows = [
        compare_rows(
            left_by_key[key],
            right_by_key[key],
            missing_logprob=args.missing_logprob,
            left_label=args.left_label,
            right_label=args.right_label,
        )
        for key in keys
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "depth",
        "depth_bucket",
        "input_token_count",
        "output_token_count",
        "target_token_id",
        "target_decoded_token",
        "left_variant_id",
        "right_variant_id",
        "same_target_token",
        "selected_logprob_left",
        "selected_logprob_right",
        "selected_logprob_diff_right_minus_left",
        "selected_abs_logprob_diff",
        "selected_raw_logit_left",
        "selected_raw_logit_right",
        "selected_raw_logit_diff_right_minus_left",
        "selected_abs_raw_logit_diff",
        "selected_rank_left",
        "selected_rank_right",
        "selected_rank_delta_right_minus_left",
        "selected_abs_rank_delta",
        "left_top1_token_id",
        "right_top1_token_id",
        "top1_equal",
        "left_top1_logprob",
        "right_top1_logprob",
        "left_margin",
        "right_margin",
        "min_margin",
        "margin_bucket",
        "left_top_k_mass",
        "right_top_k_mass",
        "top_k_mass_diff_right_minus_left",
        "left_top_count",
        "right_top_count",
        "common_top_count",
        "union_top_count",
        "topk_jaccard",
        "missing_from_right_count",
        "extra_in_right_count",
        "common_prob_cosine",
        "common_kl_left_to_right",
        "common_kl_right_to_left",
        "common_js_divergence",
        "union_prob_cosine",
        "union_kl_left_to_right",
        "union_kl_right_to_left",
        "union_js_divergence",
        "common_logprob_cosine",
        "common_max_abs_logprob_diff",
        "common_mean_abs_logprob_diff",
        "common_rms_logprob_diff",
        "union_logprob_cosine",
        "union_max_abs_logprob_diff",
        "union_mean_abs_logprob_diff",
        "union_rms_logprob_diff",
        "probe_common_count",
        "probe_logprob_cosine",
        "probe_max_abs_logprob_diff",
        "probe_mean_abs_logprob_diff",
        "probe_rms_logprob_diff",
    ]
    write_csv(output_dir / "depth_metrics.csv", rows, fields)

    cumulative = cumulative_rows(rows)
    cumulative_fields = [
        "sample_id",
        "depth",
        "depth_bucket",
        "cumulative_nll_left",
        "cumulative_nll_right",
        "cumulative_nll_delta_right_minus_left",
        "cumulative_abs_nll_delta",
        "max_selected_abs_logprob_diff_so_far",
        "max_selected_abs_raw_logit_diff_so_far",
        "top1_mismatches_so_far",
    ]
    write_csv(output_dir / "cumulative_metrics.csv", cumulative, cumulative_fields)

    (output_dir / "summary.md").write_text(
        markdown_summary(rows, args.left_label, args.right_label),
        encoding="utf-8",
    )
    missing_left = sorted(set(right_by_key) - set(left_by_key))
    missing_right = sorted(set(left_by_key) - set(right_by_key))
    write_json(output_dir / "comparison_metadata.json", {
        "metadata": runtime_metadata(sys.argv),
        "args": vars(args),
        "left_metadata": left_meta,
        "right_metadata": right_meta,
        "matched_rows": len(keys),
        "missing_from_left": [list(key) for key in missing_left],
        "missing_from_right": [list(key) for key in missing_right],
    })
    print(f"Wrote {output_dir / 'depth_metrics.csv'}")
    print(f"Wrote {output_dir / 'cumulative_metrics.csv'}")
    print(f"Wrote {output_dir / 'summary.md'}")
    print(f"Wrote {output_dir / 'comparison_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
