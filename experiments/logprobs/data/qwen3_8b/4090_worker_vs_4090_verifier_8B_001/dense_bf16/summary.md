# Dense BF16 verifier

- source: `qwen3_8b_4090_data_results/results/qwen3_8b_identity_v1/worker_verify/4090_worker_vs_4090_verifier_8B_001/worker_vs_verifier_depth_metrics.csv`
- preset: `qwen3-8b-bf16-4090-v1`
- verdict: **PASS_STRICT**
- reason: batch matches the Qwen3-8B BF16 dense fingerprint
- finite tokens: 25600
- samples: 200
- sample provisional pass rate: 0.955
- sample reject rate: 0.005

## Overall metrics

| metric | value |
|---|---:|
| `mean_abs_logprob_diff` | 0.00851067 |
| `abs_logprob_diff_p50` | 1.000e-05 |
| `abs_logprob_diff_p90` | 0.0261561 |
| `abs_logprob_diff_p95` | 0.0498562 |
| `abs_logprob_diff_p99` | 0.101925 |
| `abs_logprob_diff_p999` | 0.199487 |
| `rank_delta_nonzero_rate` | 0.00605469 |
| `topk_jaccard_mean` | 0.968556 |
| `topk_jaccard_p05` | 0.882353 |
| `union_js_p95` | 0.0010323 |
| `union_js_p99` | 0.00271656 |
| `max_abs_logprob_diff` | 18.7481 |
| `missing_selected_count` | 0 |

## Batch accept checks

| metric | op | threshold | value | passed |
|---|---:|---:|---:|---|
| `missing_selected_count` | == | 0 | 0 | True |
| `abs_logprob_diff_p50` | <= | 0.0001 | 1.000e-05 | True |
| `mean_abs_logprob_diff` | <= | 0.015 | 0.00851067 | True |
| `abs_logprob_diff_p90` | <= | 0.06 | 0.0261561 | True |
| `abs_logprob_diff_p95` | <= | 0.09 | 0.0498562 | True |
| `abs_logprob_diff_p99` | <= | 0.2 | 0.101925 | True |
| `rank_delta_nonzero_rate` | <= | 0.012 | 0.00605469 | True |
| `topk_jaccard_mean` | >= | 0.94 | 0.968556 | True |
| `topk_jaccard_p05` | >= | 0.82 | 0.882353 | True |
| `union_js_p95` | <= | 0.004 | 0.0010323 | True |
| `union_js_p99` | <= | 0.012 | 0.00271656 | True |

## Reject checks

| metric | op | threshold | value | hit |
|---|---:|---:|---:|---|
| `missing_selected_count` | > | 0 | 0 | False |
| `mean_abs_logprob_diff` | > | 0.02 | 0.00851067 | False |
| `abs_logprob_diff_p95` | > | 0.12 | 0.0498562 | False |
| `abs_logprob_diff_p99` | > | 0.25 | 0.101925 | False |
| `rank_delta_nonzero_rate` | > | 0.03 | 0.00605469 | False |
| `topk_jaccard_mean` | < | 0.92 | 0.968556 | False |
| `union_js_p99` | > | 0.02 | 0.00271656 | False |

## Sample verdicts

| verdict | count |
|---|---:|
| PASS_PROVISIONAL | 191 |
| INCONCLUSIVE | 8 |
| REJECT | 1 |
