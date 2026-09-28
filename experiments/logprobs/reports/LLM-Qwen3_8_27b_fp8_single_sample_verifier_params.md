# Qwen3.8-27B-FP8 Single-Sample Verifier Design

Goal: given a single `input + output + worker logprobs`, decide whether the evidence looks like it came from a `Qwen/Qwen3.8-27B-FP8` worker. This report follows the same metric set and three-state decision shape as `LLM-QWen3_8b_bf16_single_sample_verifier_params.md`, but recalibrates on the current `qwen3.6_27b_fp8/` result directory.

This version uses `finite_count > 5` for sample inclusion and excludes BF16 from the practical FP8-vs-other-model comparison. BF16 is intentionally excluded because BF16 pretending to be FP8 is not a cost-effective cheat; the production boundary should focus on lower-cost or wrong-model fingerprints such as AWQ4 and Qwen3.6-FP8.

## Metrics

Use the same per-sample metrics as the 8B BF16 document:

| Metric | Meaning |
|---|---|
| `finite_count` | Number of valid output tokens |
| `missing_selected_count` | Number of selected worker tokens missing from verifier top-k |
| `mean_abs_logprob_diff` | Mean selected-token logprob drift |
| `abs_logprob_diff_p95` | Selected-token p95 drift |
| `abs_logprob_diff_p99` | Selected-token p99 drift |
| `rank_delta_nonzero_rate` | Fraction of selected-token rank disagreements |
| `topk_jaccard_mean` | Mean top-k token-set overlap |
| `union_js_p99` | p99 JS divergence of the top-k union distribution |

## Data Summary

Main sample counts under `finite_count > 5`:

| comparison | sample meaning | samples |
|---|---|---:|
| `3.8-FP8 worker -> 3.8-FP8 verifier` | normal baseline | 714 |
| `3.8-AWQ4 worker -> 3.8-FP8 verifier` | AWQ4 forward wrong-worker check | 725 |
| `3.6-FP8 worker -> 3.8-FP8 verifier` | version forward wrong-worker check | 917 |
| `3.8-FP8 worker -> 3.8-AWQ4 verifier` | AWQ4 paired wrong-verifier check | 714 |
| `3.8-FP8 worker -> 3.6-FP8 verifier` | version paired wrong-verifier check | 714 |

Token-level distribution:

| comparison | `mean_abs` | `p95_abs` | `p99_abs` | rank mismatch | top-k Jaccard | `union_js_p99` |
|---|---:|---:|---:|---:|---:|---:|
| normal FP8 baseline | `0.05285` | `0.15122` | `0.39477` | `0.02592` | `0.90006` | `0.04717` |
| AWQ4 worker -> FP8 verifier | `0.09720` | `0.32163` | `0.92599` | `0.04714` | `0.82717` | `0.13440` |
| 3.6 worker -> FP8 verifier | `0.61603` | `2.60702` | `14.01559` | `0.14879` | `0.48905` | `0.69077` |
| FP8 worker -> AWQ4 verifier | `0.09314` | `0.30517` | `0.82282` | `0.04760` | `0.83013` | `0.10963` |
| FP8 worker -> 3.6 verifier | `0.64767` | `3.34233` | `7.84530` | `0.21067` | `0.47039` | `0.66847` |

## Single-Sample Decision Rule

This keeps the same three-state shape as `LLM-QWen3_8b_bf16_single_sample_verifier_params.md`:

```text
if finite_count <= 5:
    verdict = INCONCLUSIVE_SINGLE
elif all PASS_SINGLE_STRICT conditions:
    verdict = PASS_SINGLE_STRICT
elif any REJECT_SINGLE_LOW_FR condition:
    verdict = REJECT_SINGLE
else:
    verdict = INCONCLUSIVE_SINGLE
```

### PASS_SINGLE_STRICT

All must hold:

| Metric | Condition |
|---|---:|
| `finite_count` | `> 5` |
| `missing_selected_count` | `== 0` |
| `mean_abs_logprob_diff` | `<= 0.080` |
| `abs_logprob_diff_p95` | `<= 0.220` |
| `abs_logprob_diff_p99` | `<= 0.550` |
| `rank_delta_nonzero_rate` | `<= 0.047` |
| `topk_jaccard_mean` | `>= 0.865` |
| `union_js_p99` | `<= 0.100` |

### REJECT_SINGLE_LOW_FR

After `finite_count > 5` and after the strict-pass check, any one clause below triggers reject/review. This is intentionally a multi-metric rule, matching the BF16 document's "any reject condition" logic. Some guardrail clauses are redundant on the current data, but they keep the same metric coverage without increasing the observed normal FP8 false reject rate.

| Clause | Condition |
|---|---|
| missing selected token guardrail | `missing_selected_count > 0` |
| mean logprob drift guardrail | `mean_abs_logprob_diff > 3.240` |
| p95 logprob drift guardrail | `abs_logprob_diff_p95 > 13.337` |
| p99 logprob drift guardrail | `abs_logprob_diff_p99 > 16.663` |
| rank disagreement guardrail | `rank_delta_nonzero_rate > 0.500` |
| Qwen3.6 low-overlap detector | `topk_jaccard_mean < 0.650` |
| Qwen3.6 high-divergence detector | `union_js_p99 > 0.635` |
| AWQ4 low-JS branch | `topk_jaccard_mean < 0.870` AND `union_js_p99 <= 0.100` AND `abs_logprob_diff_p99 > 0.200` AND (`abs_logprob_diff_p99 <= 0.320` OR `topk_jaccard_mean > 0.780`) |
| AWQ4 low-Jaccard branch | `topk_jaccard_mean < 0.870` AND `union_js_p99 > 0.100` AND `topk_jaccard_mean <= 0.800` AND `abs_logprob_diff_p95 <= 1.630` |
| AWQ4 mid-Jaccard branch | `topk_jaccard_mean < 0.870` AND `union_js_p99 > 0.100` AND `topk_jaccard_mean > 0.800` AND `union_js_p99 <= 0.160` |

The first five guardrails complete the same metric coverage as the BF16 single-sample verifier. They do not add normal FP8 false rejects on the current data; `abs_logprob_diff_p99 > 16.663` adds two AWQ4 detections and one paired AWQ4-verifier mismatch detection.

Three-state results under `finite_count > 5`:

| comparison | pass | reject | inconclusive |
|---|---:|---:|---:|
| normal FP8 baseline | `65.97%` (`471/714`) | `9.66%` (`69/714`) | `24.37%` (`174/714`) |
| AWQ4 worker -> FP8 verifier | `0.83%` (`6/725`) | `90.90%` (`659/725`) | `8.28%` (`60/725`) |
| 3.6 worker -> FP8 verifier | `0.00%` (`0/917`) | `100.00%` (`917/917`) | `0.00%` (`0/917`) |
| FP8 worker -> AWQ4 verifier | `2.80%` (`20/714`) | `83.47%` (`596/714`) | `13.73%` (`98/714`) |
| FP8 worker -> 3.6 verifier | `0.00%` (`0/714`) | `99.86%` (`713/714`) | `0.14%` (`1/714`) |

BF16 remains excluded from this practical comparison. The operating goal here is normal FP8 false reject below `10%` while keeping both AWQ4 and Qwen3.6-FP8 detection above `80%`.

## Required Summary Table

The following table uses `REJECT_SINGLE_LOW_FR` and excludes BF16 from comparison.

| 指标 / 判定阈值 | 作弊类型 | 样本数（作弊 / 正常） | 作弊检出率 | 正常结果误拒率 |
|---|---|---:|---:|---:|
| `finite_count > 5` 且严格通过优先；否则任一 `REJECT_SINGLE_LOW_FR` clause | `3.8_27b_awq4_h100_worker` 冒充 `3.8_27b_fp8_h100_worker` | `725 / 714` | `90.90%` (`659/725`) | `9.66%` (`69/714`) |
| same as above | `3.6_27b_fp8_h100_worker` 冒充 `3.8_27b_fp8_h100_worker` | `917 / 714` | `100.00%` (`917/917`) | `9.66%` (`69/714`) |

## Recommendation

For this dataset, use the BF16-style three-state rule above with `REJECT_SINGLE_LOW_FR`.

```text
if finite_count <= 5:
    verdict = INCONCLUSIVE_SINGLE
elif all PASS_SINGLE_STRICT conditions:
    verdict = PASS_SINGLE_STRICT
elif any REJECT_SINGLE_LOW_FR clause:
    verdict = REJECT_SINGLE
else:
    verdict = INCONCLUSIVE_SINGLE
```

This keeps normal FP8 false reject below `10%` while keeping both positive classes high: AWQ4 detection is `90.90%`, and Qwen3.6-FP8 detection is `100.00%`.
