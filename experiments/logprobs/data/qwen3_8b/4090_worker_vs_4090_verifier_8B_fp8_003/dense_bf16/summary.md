# Dense BF16 verifier

- source: `qwen3_8b_4090_data_results/results/qwen3_8b_identity_v1/worker_verify/4090_worker_vs_4090_verifier_8B_fp8_003/worker_vs_verifier_depth_metrics.csv`
- preset: `qwen3-8b-bf16-4090-v1`
- verdict: **REJECT**
- reason: batch violates one or more exact-BF16 reject thresholds
- finite tokens: 25600
- samples: 200
- sample provisional pass rate: 0.05
- sample reject rate: 0.695

## Overall metrics

| metric | value |
|---|---:|
| `mean_abs_logprob_diff` | 0.0290245 |
| `abs_logprob_diff_p50` | 0.000144825 |
| `abs_logprob_diff_p90` | 0.0794408 |
| `abs_logprob_diff_p95` | 0.1351 |
| `abs_logprob_diff_p99` | 0.312511 |
| `abs_logprob_diff_p999` | 0.836101 |
| `rank_delta_nonzero_rate` | 0.0189844 |
| `topk_jaccard_mean` | 0.899532 |
| `topk_jaccard_p05` | 0.72973 |
| `union_js_p95` | 0.00751844 |
| `union_js_p99` | 0.0228632 |
| `max_abs_logprob_diff` | 30.526 |
| `missing_selected_count` | 0 |

## Batch accept checks

| metric | op | threshold | value | passed |
|---|---:|---:|---:|---|
| `missing_selected_count` | == | 0 | 0 | True |
| `abs_logprob_diff_p50` | <= | 0.0001 | 0.000144825 | False |
| `mean_abs_logprob_diff` | <= | 0.015 | 0.0290245 | False |
| `abs_logprob_diff_p90` | <= | 0.06 | 0.0794408 | False |
| `abs_logprob_diff_p95` | <= | 0.09 | 0.1351 | False |
| `abs_logprob_diff_p99` | <= | 0.2 | 0.312511 | False |
| `rank_delta_nonzero_rate` | <= | 0.012 | 0.0189844 | False |
| `topk_jaccard_mean` | >= | 0.94 | 0.899532 | False |
| `topk_jaccard_p05` | >= | 0.82 | 0.72973 | False |
| `union_js_p95` | <= | 0.004 | 0.00751844 | False |
| `union_js_p99` | <= | 0.012 | 0.0228632 | False |

## Reject checks

| metric | op | threshold | value | hit |
|---|---:|---:|---:|---|
| `missing_selected_count` | > | 0 | 0 | False |
| `mean_abs_logprob_diff` | > | 0.02 | 0.0290245 | True |
| `abs_logprob_diff_p95` | > | 0.12 | 0.1351 | True |
| `abs_logprob_diff_p99` | > | 0.25 | 0.312511 | True |
| `rank_delta_nonzero_rate` | > | 0.03 | 0.0189844 | False |
| `topk_jaccard_mean` | < | 0.92 | 0.899532 | True |
| `union_js_p99` | > | 0.02 | 0.0228632 | True |

## Sample verdicts

| verdict | count |
|---|---:|
| REJECT | 139 |
| INCONCLUSIVE | 51 |
| PASS_PROVISIONAL | 10 |
