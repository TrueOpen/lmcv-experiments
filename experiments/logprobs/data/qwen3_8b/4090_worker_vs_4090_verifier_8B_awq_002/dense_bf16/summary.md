# Dense BF16 verifier

- source: `qwen3_8b_4090_data_results/results/qwen3_8b_identity_v1/worker_verify/4090_worker_vs_4090_verifier_8B_awq_002/worker_vs_verifier_depth_metrics.csv`
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
| `mean_abs_logprob_diff` | 0.0799436 |
| `abs_logprob_diff_p50` | 0.000638794 |
| `abs_logprob_diff_p90` | 0.211595 |
| `abs_logprob_diff_p95` | 0.388427 |
| `abs_logprob_diff_p99` | 1.13349 |
| `abs_logprob_diff_p999` | 2.8117 |
| `rank_delta_nonzero_rate` | 0.0494531 |
| `topk_jaccard_mean` | 0.792892 |
| `topk_jaccard_p05` | 0.560976 |
| `union_js_p95` | 0.0451176 |
| `union_js_p99` | 0.143801 |
| `max_abs_logprob_diff` | 31.0372 |
| `missing_selected_count` | 0 |

## Batch accept checks

| metric | op | threshold | value | passed |
|---|---:|---:|---:|---|
| `missing_selected_count` | == | 0 | 0 | True |
| `abs_logprob_diff_p50` | <= | 0.0001 | 0.000638794 | False |
| `mean_abs_logprob_diff` | <= | 0.015 | 0.0799436 | False |
| `abs_logprob_diff_p90` | <= | 0.06 | 0.211595 | False |
| `abs_logprob_diff_p95` | <= | 0.09 | 0.388427 | False |
| `abs_logprob_diff_p99` | <= | 0.2 | 1.13349 | False |
| `rank_delta_nonzero_rate` | <= | 0.012 | 0.0494531 | False |
| `topk_jaccard_mean` | >= | 0.94 | 0.792892 | False |
| `topk_jaccard_p05` | >= | 0.82 | 0.560976 | False |
| `union_js_p95` | <= | 0.004 | 0.0451176 | False |
| `union_js_p99` | <= | 0.012 | 0.143801 | False |

## Reject checks

| metric | op | threshold | value | hit |
|---|---:|---:|---:|---|
| `missing_selected_count` | > | 0 | 0 | False |
| `mean_abs_logprob_diff` | > | 0.02 | 0.0799436 | True |
| `abs_logprob_diff_p95` | > | 0.12 | 0.388427 | True |
| `abs_logprob_diff_p99` | > | 0.25 | 1.13349 | True |
| `rank_delta_nonzero_rate` | > | 0.03 | 0.0494531 | True |
| `topk_jaccard_mean` | < | 0.92 | 0.792892 | True |
| `union_js_p99` | > | 0.02 | 0.143801 | True |

## Sample verdicts

| verdict | count |
|---|---:|
| REJECT | 200 |
