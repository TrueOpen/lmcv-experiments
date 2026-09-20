#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from routed_common import (
    decode_routed_experts,
    normalize_route_array,
    print_progress,
    read_jsonl,
    require_numpy,
)


def route_array(row: Dict[str, Any]) -> Optional[np.ndarray]:
    arr = decode_routed_experts(row.get("routed_experts"))
    if arr is None:
        return None
    return normalize_route_array(arr)


def load_by_sample(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id"))
        rows[sample_id] = row
    return rows


def safe_div(numer: float, denom: float) -> Optional[float]:
    if denom == 0:
        return None
    return numer / denom


def parse_layer_spec(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or raw.strip() == "":
        return None
    layers = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid layer range: {part}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    out = sorted(set(layers))
    if any(layer < 0 for layer in out):
        raise ValueError(f"Layer indices must be non-negative: {raw}")
    return out


def topk_set_stats(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    np = require_numpy()
    overlaps = []
    jaccards = []
    set_match_count = 0
    for token_idx in range(a.shape[0]):
        for layer_idx in range(a.shape[1]):
            if not mask[token_idx, layer_idx].any():
                continue
            left = set(int(x) for x in a[token_idx, layer_idx][mask[token_idx, layer_idx]])
            right = set(int(x) for x in b[token_idx, layer_idx][mask[token_idx, layer_idx]])
            intersection = left.intersection(right)
            union = left.union(right)
            denom = max(len(left), len(right), 1)
            overlaps.append(len(intersection) / denom)
            jaccards.append(len(intersection) / len(union) if union else 1.0)
            if left == right:
                set_match_count += 1
    count = len(overlaps)
    return {
        "token_layer_set_count": count,
        "token_layer_set_match_count": set_match_count,
        "token_layer_set_match_rate": safe_div(set_match_count, count),
        "topk_overlap_sum": float(sum(overlaps)),
        "topk_overlap_mean": float(np.mean(overlaps)) if overlaps else None,
        "topk_jaccard_sum": float(sum(jaccards)),
        "topk_jaccard_mean": float(np.mean(jaccards)) if jaccards else None,
    }


def infer_shape_status(meta: Dict[str, Any], gen_shape: List[int], prefill_shape: List[int]) -> Dict[str, Any]:
    input_len = int(meta.get("input_token_count") or 0)
    output_len = int(meta.get("output_token_count") or 0)
    gen_start = int(meta.get("gen_route_start") or 0)
    pre_start = int(meta.get("prefill_route_start") or 0)

    expected_gen_positions = max(0, input_len + output_len - 1)
    expected_prefill_positions = input_len + output_len
    expected_gen_len = max(0, expected_gen_positions - gen_start)
    expected_prefill_len = max(0, expected_prefill_positions - pre_start)
    expected_overlap_start = max(gen_start, pre_start)
    expected_overlap_end = min(expected_gen_positions, expected_prefill_positions)
    expected_overlap_len = max(0, expected_overlap_end - expected_overlap_start)

    gen_len = int(gen_shape[0]) if gen_shape else 0
    prefill_len = int(prefill_shape[0]) if prefill_shape else 0
    gen_layers = int(gen_shape[1]) if len(gen_shape) > 1 else 0
    prefill_layers = int(prefill_shape[1]) if len(prefill_shape) > 1 else 0
    gen_topk = int(gen_shape[2]) if len(gen_shape) > 2 else 0
    prefill_topk = int(prefill_shape[2]) if len(prefill_shape) > 2 else 0

    layer_topk_match = gen_layers == prefill_layers and gen_topk == prefill_topk
    expected_length_match = (
        gen_len == expected_gen_len
        and prefill_len == expected_prefill_len
        and int(meta.get("common_len") or 0) == expected_overlap_len
    )
    inferred_shape_match = layer_topk_match and expected_length_match

    return {
        "raw_shape_match": gen_shape == prefill_shape,
        "expected_gen_route_len": expected_gen_len,
        "expected_prefill_route_len": expected_prefill_len,
        "expected_overlap_len": expected_overlap_len,
        "expected_length_match": expected_length_match,
        "layer_topk_shape_match": layer_topk_match,
        "inferred_shape_match": inferred_shape_match,
        "shape_status": "expected_aligned_shape" if inferred_shape_match else "unexpected_shape",
    }


def aligned_route_window(gen_row: Dict[str, Any],
                         prefill_row: Dict[str, Any],
                         *,
                         ignore_value: Optional[int],
                         layer_indices: Optional[List[int]]) -> Tuple[Dict[str, Any], Any, Any, Any]:
    np = require_numpy()
    sample_id = str(gen_row.get("sample_id"))
    gen = route_array(gen_row)
    pre = route_array(prefill_row)
    if gen is None or pre is None:
        return {"sample_id": sample_id, "status": "missing_routes"}, None, None, None

    gen_start = int(gen_row.get("route_prompt_start", 0) or 0)
    pre_start = int(prefill_row.get("route_prompt_start", 0) or 0)
    gen_request = gen_row.get("request") or {}
    source_model = str(gen_request.get("model") or prefill_row.get("source_model") or "")
    verifier_model = str(prefill_row.get("verifier_model") or "")
    gen_len, pre_len = int(gen.shape[0]), int(pre.shape[0])
    overlap_start = max(gen_start, pre_start)
    overlap_end = min(gen_start + gen_len, pre_start + pre_len)
    common_len = max(0, overlap_end - overlap_start)
    common_layers = min(int(gen.shape[1]), int(pre.shape[1]))
    common_topk = min(int(gen.shape[2]), int(pre.shape[2]))
    meta = {
        "sample_id": sample_id,
        "source_model": source_model,
        "verifier_model": verifier_model,
        "comparison_kind": (
            "same_model"
            if source_model and verifier_model and source_model == verifier_model
            else "cross_model"
        ),
        "category": gen_row.get("category", prefill_row.get("category", "")),
        "target_input_tokens": gen_row.get(
            "target_input_tokens",
            prefill_row.get("target_input_tokens"),
        ),
        "input_token_count": len(gen_row.get("input_token_ids") or []),
        "output_token_count": len(gen_row.get("output_token_ids") or []),
        "gen_shape": list(gen.shape),
        "prefill_shape": list(pre.shape),
        "gen_route_start": gen_start,
        "prefill_route_start": pre_start,
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
    }
    if common_len == 0 or common_layers == 0 or common_topk == 0:
        meta.update(infer_shape_status(meta, list(gen.shape), list(pre.shape)))
        meta["status"] = "empty_common_window"
        return meta, None, None, None

    gen_offset = overlap_start - gen_start
    pre_offset = overlap_start - pre_start
    raw_common_layers = common_layers
    selected_layer_indices = (
        [layer for layer in layer_indices if layer < common_layers]
        if layer_indices is not None
        else list(range(common_layers))
    )
    if not selected_layer_indices:
        meta.update(infer_shape_status(meta, list(gen.shape), list(pre.shape)))
        meta["status"] = "no_selected_layers"
        meta["requested_layers"] = layer_indices
        return meta, None, None, None
    gen_cmp = gen[gen_offset:gen_offset + common_len, :common_layers, :common_topk]
    pre_cmp = pre[pre_offset:pre_offset + common_len, :common_layers, :common_topk]
    gen_cmp = gen_cmp[:, selected_layer_indices, :]
    pre_cmp = pre_cmp[:, selected_layer_indices, :]
    valid = np.ones(gen_cmp.shape, dtype=bool)
    if ignore_value is not None:
        valid &= gen_cmp != ignore_value
        valid &= pre_cmp != ignore_value

    meta.update({
        "status": "ok",
        "gen_overlap_offset": gen_offset,
        "prefill_overlap_offset": pre_offset,
        "common_len": common_len,
        "common_layers": len(selected_layer_indices),
        "raw_common_layers": raw_common_layers,
        "selected_layers": ",".join(str(layer) for layer in selected_layer_indices),
        "common_topk": common_topk,
        "gen_prefix_ignored": gen_offset,
        "prefill_prefix_ignored": pre_offset,
        "gen_tail_ignored": max(0, gen_len - gen_offset - common_len),
        "prefill_tail_ignored": max(0, pre_len - pre_offset - common_len),
    })
    meta.update(infer_shape_status(meta, list(gen.shape), list(pre.shape)))
    return meta, gen_cmp, pre_cmp, valid


def compare_one(gen_row: Dict[str, Any],
                prefill_row: Dict[str, Any],
                *,
                ignore_value: Optional[int],
                layer_indices: Optional[List[int]]) -> Dict[str, Any]:
    np = require_numpy()
    meta, gen_cmp, pre_cmp, valid = aligned_route_window(
        gen_row,
        prefill_row,
        ignore_value=ignore_value,
        layer_indices=layer_indices,
    )
    if meta.get("status") != "ok":
        return meta

    compared_entries = int(valid.sum())
    entry_matches = int(((gen_cmp == pre_cmp) & valid).sum())
    entry_mismatches = compared_entries - entry_matches

    token_layer_valid = valid.any(axis=2)
    token_layer_exact = ((gen_cmp == pre_cmp) | ~valid).all(axis=2) & token_layer_valid
    token_layer_count = int(token_layer_valid.sum())
    token_layer_match_count = int(token_layer_exact.sum())

    token_valid = token_layer_valid.any(axis=1)
    token_exact = ((gen_cmp == pre_cmp) | ~valid).all(axis=(1, 2)) & token_valid
    token_count = int(token_valid.sum())
    token_match_count = int(token_exact.sum())
    set_stats = topk_set_stats(gen_cmp, pre_cmp, valid)

    meta.update({
        "compared_entries": compared_entries,
        "entry_match_count": entry_matches,
        "entry_mismatch_count": entry_mismatches,
        "entry_match_rate": safe_div(entry_matches, compared_entries),
        "entry_mismatch_rate": safe_div(entry_mismatches, compared_entries),
        "token_layer_count": token_layer_count,
        "token_layer_match_count": token_layer_match_count,
        "token_layer_match_rate": safe_div(token_layer_match_count, token_layer_count),
        "token_count": token_count,
        "token_match_count": token_match_count,
        "token_match_rate": safe_div(token_match_count, token_count),
        **set_stats,
    })
    return meta


def compare_layer_rows(gen_row: Dict[str, Any],
                       prefill_row: Dict[str, Any],
                       *,
                       ignore_value: Optional[int],
                       layer_indices: Optional[List[int]]) -> List[Dict[str, Any]]:
    np = require_numpy()
    meta, gen_cmp, pre_cmp, valid = aligned_route_window(
        gen_row,
        prefill_row,
        ignore_value=ignore_value,
        layer_indices=layer_indices,
    )
    if meta.get("status") != "ok":
        return []

    rows = []
    selected_layer_indices = [
        int(layer)
        for layer in str(meta.get("selected_layers", "")).split(",")
        if layer != ""
    ]
    for local_layer_idx, original_layer_idx in enumerate(selected_layer_indices):
        layer_gen = gen_cmp[:, local_layer_idx:local_layer_idx + 1, :]
        layer_pre = pre_cmp[:, local_layer_idx:local_layer_idx + 1, :]
        layer_valid = valid[:, local_layer_idx:local_layer_idx + 1, :]
        compared_entries = int(layer_valid.sum())
        entry_matches = int(((layer_gen == layer_pre) & layer_valid).sum())
        token_layer_valid = layer_valid.any(axis=2)
        token_layer_exact = ((layer_gen == layer_pre) | ~layer_valid).all(axis=2) & token_layer_valid
        token_layer_count = int(token_layer_valid.sum())
        token_layer_match_count = int(token_layer_exact.sum())
        set_stats = topk_set_stats(layer_gen, layer_pre, layer_valid)
        rows.append({
            "sample_id": meta["sample_id"],
            "layer_idx": original_layer_idx,
            "common_len": meta["common_len"],
            "common_topk": meta["common_topk"],
            "compared_entries": compared_entries,
            "entry_match_count": entry_matches,
            "entry_mismatch_count": compared_entries - entry_matches,
            "entry_match_rate": safe_div(entry_matches, compared_entries),
            "entry_mismatch_rate": safe_div(compared_entries - entry_matches, compared_entries),
            "token_layer_count": token_layer_count,
            "token_layer_match_count": token_layer_match_count,
            "token_layer_match_rate": safe_div(token_layer_match_count, token_layer_count),
            **set_stats,
        })
    return rows


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    np = require_numpy()
    rows = list(rows)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    selected_layers = sorted({
        str(row.get("selected_layers"))
        for row in ok_rows
        if row.get("selected_layers") is not None
    })
    selected_layer_counts = sorted({
        int(row.get("common_layers") or 0)
        for row in ok_rows
        if row.get("common_layers") is not None
    })
    compared_entries = sum(int(row.get("compared_entries") or 0) for row in ok_rows)
    entry_matches = sum(int(row.get("entry_match_count") or 0) for row in ok_rows)
    token_count = sum(int(row.get("token_count") or 0) for row in ok_rows)
    token_matches = sum(int(row.get("token_match_count") or 0) for row in ok_rows)
    token_layer_set_count = sum(int(row.get("token_layer_set_count") or 0) for row in ok_rows)
    token_layer_set_matches = sum(
        int(row.get("token_layer_set_match_count") or 0)
        for row in ok_rows
    )
    topk_values = [
        float(row["topk_overlap_mean"])
        for row in ok_rows
        if row.get("topk_overlap_mean") is not None
    ]
    topk_jaccard_values = [
        float(row["topk_jaccard_mean"])
        for row in ok_rows
        if row.get("topk_jaccard_mean") is not None
    ]
    sample_metric_summary = summarize_sample_metrics(ok_rows)
    return {
        "sample_count": len(rows),
        "ok_sample_count": len(ok_rows),
        "non_ok_sample_count": len(rows) - len(ok_rows),
        "source_models": sorted({
            str(row.get("source_model"))
            for row in ok_rows
            if row.get("source_model")
        }),
        "verifier_models": sorted({
            str(row.get("verifier_model"))
            for row in ok_rows
            if row.get("verifier_model")
        }),
        "comparison_kind_counts": {
            "same_model": sum(1 for row in ok_rows if row.get("comparison_kind") == "same_model"),
            "cross_model": sum(1 for row in ok_rows if row.get("comparison_kind") == "cross_model"),
        },
        "selected_layers": selected_layers,
        "selected_layer_counts": selected_layer_counts,
        "weighted_entry_match_rate": safe_div(entry_matches, compared_entries),
        "weighted_entry_mismatch_rate": safe_div(compared_entries - entry_matches, compared_entries),
        "weighted_token_match_rate": safe_div(token_matches, token_count),
        "weighted_token_layer_set_match_rate": safe_div(
            token_layer_set_matches,
            token_layer_set_count,
        ),
        "mean_sample_entry_mismatch_rate": (
            float(np.mean([row["entry_mismatch_rate"] for row in ok_rows if row.get("entry_mismatch_rate") is not None]))
            if ok_rows else None
        ),
        "mean_topk_overlap": float(np.mean(topk_values)) if topk_values else None,
        "mean_topk_jaccard": float(np.mean(topk_jaccard_values)) if topk_jaccard_values else None,
        "sample_metric_summary": sample_metric_summary,
        "total_compared_entries": compared_entries,
        "total_compared_tokens": token_count,
        "total_compared_token_layers": token_layer_set_count,
        "raw_shape_mismatch_sample_count": sum(
            1 for row in ok_rows if not row.get("raw_shape_match")
        ),
        "expected_length_mismatch_sample_count": sum(
            1 for row in ok_rows if not row.get("expected_length_match")
        ),
        "layer_topk_shape_mismatch_sample_count": sum(
            1 for row in ok_rows if not row.get("layer_topk_shape_match")
        ),
        "shape_mismatch_sample_count": sum(
            1 for row in ok_rows if not row.get("inferred_shape_match")
        ),
    }


def summarize_sample_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    np = require_numpy()
    metric_names = (
        "entry_mismatch_rate",
        "token_layer_match_rate",
        "token_layer_set_match_rate",
        "token_match_rate",
        "topk_overlap_mean",
        "topk_jaccard_mean",
    )
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for metric_name in metric_names:
        values = [
            float(row[metric_name])
            for row in rows
            if row.get(metric_name) is not None
        ]
        if not values:
            summary[metric_name] = {
                "count": 0,
                "mean": None,
                "min": None,
                "p50": None,
                "p90": None,
                "max": None,
            }
            continue
        arr = np.asarray(values, dtype=float)
        summary[metric_name] = {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(np.max(arr)),
        }
    return summary


def summarize_layer_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = defaultdict(lambda: {
        "sample_layer_count": 0,
        "compared_entries": 0,
        "entry_match_count": 0,
        "token_layer_count": 0,
        "token_layer_match_count": 0,
        "token_layer_set_count": 0,
        "token_layer_set_match_count": 0,
        "topk_overlap_sum": 0.0,
        "topk_jaccard_sum": 0.0,
    })
    for row in rows:
        group = groups[int(row["layer_idx"])]
        group["sample_layer_count"] += 1
        for key in (
            "compared_entries",
            "entry_match_count",
            "token_layer_count",
            "token_layer_match_count",
            "token_layer_set_count",
            "token_layer_set_match_count",
        ):
            group[key] += int(row.get(key) or 0)
        group["topk_overlap_sum"] += float(row.get("topk_overlap_sum") or 0.0)
        group["topk_jaccard_sum"] += float(row.get("topk_jaccard_sum") or 0.0)

    out = []
    for layer_idx in sorted(groups):
        group = groups[layer_idx]
        compared_entries = group["compared_entries"]
        entry_matches = group["entry_match_count"]
        token_layer_count = group["token_layer_count"]
        token_layer_matches = group["token_layer_match_count"]
        set_count = group["token_layer_set_count"]
        set_matches = group["token_layer_set_match_count"]
        out.append({
            "layer_idx": layer_idx,
            "sample_layer_count": group["sample_layer_count"],
            "compared_entries": compared_entries,
            "entry_match_count": entry_matches,
            "entry_mismatch_count": compared_entries - entry_matches,
            "entry_match_rate": safe_div(entry_matches, compared_entries),
            "entry_mismatch_rate": safe_div(compared_entries - entry_matches, compared_entries),
            "token_layer_count": token_layer_count,
            "token_layer_match_count": token_layer_matches,
            "token_layer_match_rate": safe_div(token_layer_matches, token_layer_count),
            "token_layer_set_count": set_count,
            "token_layer_set_match_count": set_matches,
            "token_layer_set_match_rate": safe_div(set_matches, set_count),
            "topk_overlap_mean": safe_div(group["topk_overlap_sum"], set_count),
            "topk_jaccard_mean": safe_div(group["topk_jaccard_sum"], set_count),
        })
    return out


def write_summary_md(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Routed Experts Diff Summary",
        "",
        f"- samples: {summary['sample_count']}",
        f"- ok samples: {summary['ok_sample_count']}",
        f"- selected layers: {summary.get('selected_layers')}",
        f"- selected layer counts: {summary.get('selected_layer_counts')}",
        f"- weighted entry mismatch rate: {summary['weighted_entry_mismatch_rate']}",
        f"- weighted token match rate: {summary['weighted_token_match_rate']}",
        f"- weighted token-layer set match rate: {summary['weighted_token_layer_set_match_rate']}",
        f"- mean top-k overlap: {summary['mean_topk_overlap']}",
        f"- mean top-k jaccard: {summary['mean_topk_jaccard']}",
        f"- total compared entries: {summary['total_compared_entries']}",
        f"- total compared token-layers: {summary['total_compared_token_layers']}",
        f"- shape mismatch samples: {summary['shape_mismatch_sample_count']}",
        f"- raw shape mismatch samples: {summary['raw_shape_mismatch_sample_count']}",
        f"- expected length mismatch samples: {summary['expected_length_mismatch_sample_count']}",
        f"- layer/top-k shape mismatch samples: {summary['layer_topk_shape_mismatch_sample_count']}",
        "",
    ]
    sample_summary = summary.get("sample_metric_summary") or {}
    if sample_summary:
        lines.extend([
            "## Sample Metric Summary",
            "",
            "| metric | count | mean | min | p50 | p90 | max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for metric_name, stats in sample_summary.items():
            lines.append(
                f"| {metric_name} | {stats.get('count')} | "
                f"{stats.get('mean')} | {stats.get('min')} | "
                f"{stats.get('p50')} | {stats.get('p90')} | {stats.get('max')} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare generated routed_experts with prefill routed_experts."
    )
    parser.add_argument("--generated", default="data/generated_routes.jsonl")
    parser.add_argument("--prefill", default="data/prefill_routes.jsonl")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--ignore-value", type=int, default=-1)
    parser.add_argument(
        "--layers",
        default=None,
        help="Optional layer subset, for example '0', '0-4', or '0-4,7,12-15'.",
    )
    parser.add_argument(
        "--do-not-ignore-sentinel",
        action="store_true",
        help="Compare -1 sentinel entries instead of ignoring them.",
    )
    args = parser.parse_args()

    generated = load_by_sample(Path(args.generated))
    prefill = load_by_sample(Path(args.prefill))
    sample_ids = sorted(set(generated).intersection(prefill))
    if not sample_ids:
        raise SystemExit("No overlapping sample_id values between generated and prefill files.")

    ignore_value = None if args.do_not_ignore_sentinel else args.ignore_value
    layer_indices = parse_layer_spec(args.layers)
    rows = [
        compare_one(
            generated[sample_id],
            prefill[sample_id],
            ignore_value=ignore_value,
            layer_indices=layer_indices,
        )
        for sample_id in sample_ids
    ]
    layer_rows = []
    for sample_id in sample_ids:
        layer_rows.extend(compare_layer_rows(
            generated[sample_id],
            prefill[sample_id],
            ignore_value=ignore_value,
            layer_indices=layer_indices,
        ))
    summary = summarize(rows)
    layer_summary_rows = summarize_layer_rows(layer_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "route_diff_by_sample.csv"
    layer_csv_path = output_dir / "route_diff_by_sample_layer.csv"
    layer_summary_csv_path = output_dir / "route_diff_by_layer.csv"
    json_path = output_dir / "route_diff_summary.json"
    md_path = output_dir / "route_diff_summary.md"

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if layer_rows:
        layer_fieldnames = sorted({key for row in layer_rows for key in row.keys()})
        with layer_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=layer_fieldnames)
            writer.writeheader()
            writer.writerows(layer_rows)
    if layer_summary_rows:
        layer_summary_fieldnames = sorted({key for row in layer_summary_rows for key in row.keys()})
        with layer_summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=layer_summary_fieldnames)
            writer.writeheader()
            writer.writerows(layer_summary_rows)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_summary_md(md_path, summary)

    print_progress(f"Wrote {csv_path}")
    if layer_rows:
        print_progress(f"Wrote {layer_csv_path}")
    if layer_summary_rows:
        print_progress(f"Wrote {layer_summary_csv_path}")
    print_progress(f"Wrote {json_path}")
    print_progress(f"Wrote {md_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
