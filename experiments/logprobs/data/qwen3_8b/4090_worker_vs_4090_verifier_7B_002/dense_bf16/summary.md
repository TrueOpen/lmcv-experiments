# Dense BF16 verifier

- source: `qwen3_8b_4090_data_results/results/qwen3_8b_identity_v1/worker_verify/4090_worker_vs_4090_verifier_7B_002/worker_vs_verifier_depth_metrics.csv`
- preset: `qwen3-8b-bf16-4090-v1`
- verdict: **REJECT**
- reason: batch violates one or more exact-BF16 reject thresholds
- finite tokens: 25600
- samples: 200
- sample provisional pass rate: 0
- sample reject rate: 1

## Overall metrics

| metric | value |
|---|---:|
| `mean_abs_logprob_diff` | 0.570281 |
| `abs_logprob_diff_p50` | 0.0413289 |
| `abs_logprob_diff_p90` | 1.80396 |
| `abs_logprob_diff_p95` | 2.89393 |
| `abs_logprob_diff_p99` | 5.82915 |
| `abs_logprob_diff_p999` | 9.4282 |
| `rank_delta_nonzero_rate` | 0.195156 |
| `topk_jaccard_mean` | 0.409895 |
| `topk_jaccard_p05` | 0.207547 |
| `union_js_p95` | 0.477591 |
| `union_js_p99` | 0.627882 |
| `max_abs_logprob_diff` | 24.0941 |
| `missing_selected_count` | 0 |

## Batch accept checks

| metric | op | threshold | value | passed |
|---|---:|---:|---:|---|
| `missing_selected_count` | == | 0 | 0 | True |
| `abs_logprob_diff_p50` | <= | 0.0001 | 0.0413289 | False |
| `mean_abs_logprob_diff` | <= | 0.015 | 0.570281 | False |
| `abs_logprob_diff_p90` | <= | 0.06 | 1.80396 | False |
| `abs_logprob_diff_p95` | <= | 0.09 | 2.89393 | False |
| `abs_logprob_diff_p99` | <= | 0.2 | 5.82915 | False |
| `rank_delta_nonzero_rate` | <= | 0.012 | 0.195156 | False |
| `topk_jaccard_mean` | >= | 0.94 | 0.409895 | False |
| `topk_jaccard_p05` | >= | 0.82 | 0.207547 | False |
| `union_js_p95` | <= | 0.004 | 0.477591 | False |
| `union_js_p99` | <= | 0.012 | 0.627882 | False |

## Reject checks

| metric | op | threshold | value | hit |
|---|---:|---:|---:|---|
| `missing_selected_count` | > | 0 | 0 | False |
| `mean_abs_logprob_diff` | > | 0.02 | 0.570281 | True |
| `abs_logprob_diff_p95` | > | 0.12 | 2.89393 | True |
| `abs_logprob_diff_p99` | > | 0.25 | 5.82915 | True |
| `rank_delta_nonzero_rate` | > | 0.03 | 0.195156 | True |
| `topk_jaccard_mean` | < | 0.92 | 0.409895 | True |
| `union_js_p99` | > | 0.02 | 0.627882 | True |

## Sample verdicts

| verdict | count |
|---|---:|
| REJECT | 200 |
