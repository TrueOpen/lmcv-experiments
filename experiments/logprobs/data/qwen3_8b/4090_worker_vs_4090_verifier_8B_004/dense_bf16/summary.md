# Dense BF16 verifier

- source: `qwen3_8b_4090_data_results/results/qwen3_8b_identity_v1/worker_verify/4090_worker_vs_4090_verifier_8B_004/worker_vs_verifier_depth_metrics.csv`
- preset: `qwen3-8b-bf16-4090-v1`
- verdict: **PASS_STRICT**
- reason: batch matches the Qwen3-8B BF16 dense fingerprint
- finite tokens: 25600
- samples: 200
- sample provisional pass rate: 0.96
- sample reject rate: 0

## Overall metrics

| metric | value |
|---|---:|
| `mean_abs_logprob_diff` | 0.00772444 |
| `abs_logprob_diff_p50` | 9.296e-06 |
| `abs_logprob_diff_p90` | 0.0257429 |
| `abs_logprob_diff_p95` | 0.0501133 |
| `abs_logprob_diff_p99` | 0.101863 |
| `abs_logprob_diff_p999` | 0.189096 |
| `rank_delta_nonzero_rate` | 0.00609375 |
| `topk_jaccard_mean` | 0.968965 |
| `topk_jaccard_p05` | 0.882353 |
| `union_js_p95` | 0.00102758 |
| `union_js_p99` | 0.00282292 |
| `max_abs_logprob_diff` | 1.41611 |
| `missing_selected_count` | 0 |

## Batch accept checks

| metric | op | threshold | value | passed |
|---|---:|---:|---:|---|
| `missing_selected_count` | == | 0 | 0 | True |
| `abs_logprob_diff_p50` | <= | 0.0001 | 9.296e-06 | True |
| `mean_abs_logprob_diff` | <= | 0.015 | 0.00772444 | True |
| `abs_logprob_diff_p90` | <= | 0.06 | 0.0257429 | True |
| `abs_logprob_diff_p95` | <= | 0.09 | 0.0501133 | True |
| `abs_logprob_diff_p99` | <= | 0.2 | 0.101863 | True |
| `rank_delta_nonzero_rate` | <= | 0.012 | 0.00609375 | True |
| `topk_jaccard_mean` | >= | 0.94 | 0.968965 | True |
| `topk_jaccard_p05` | >= | 0.82 | 0.882353 | True |
| `union_js_p95` | <= | 0.004 | 0.00102758 | True |
| `union_js_p99` | <= | 0.012 | 0.00282292 | True |

## Reject checks

| metric | op | threshold | value | hit |
|---|---:|---:|---:|---|
| `missing_selected_count` | > | 0 | 0 | False |
| `mean_abs_logprob_diff` | > | 0.02 | 0.00772444 | False |
| `abs_logprob_diff_p95` | > | 0.12 | 0.0501133 | False |
| `abs_logprob_diff_p99` | > | 0.25 | 0.101863 | False |
| `rank_delta_nonzero_rate` | > | 0.03 | 0.00609375 | False |
| `topk_jaccard_mean` | < | 0.92 | 0.968965 | False |
| `union_js_p99` | > | 0.02 | 0.00282292 | False |

## Sample verdicts

| verdict | count |
|---|---:|
| PASS_PROVISIONAL | 192 |
| INCONCLUSIVE | 8 |
