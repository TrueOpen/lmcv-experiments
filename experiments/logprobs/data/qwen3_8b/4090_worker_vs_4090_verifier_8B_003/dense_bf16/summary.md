# Dense BF16 verifier

- source: `qwen3_8b_4090_data_results/results/qwen3_8b_identity_v1/worker_verify/4090_worker_vs_4090_verifier_8B_003/worker_vs_verifier_depth_metrics.csv`
- preset: `qwen3-8b-bf16-4090-v1`
- verdict: **PASS_STRICT**
- reason: batch matches the Qwen3-8B BF16 dense fingerprint
- finite tokens: 25600
- samples: 200
- sample provisional pass rate: 0.96
- sample reject rate: 0.005

## Overall metrics

| metric | value |
|---|---:|
| `mean_abs_logprob_diff` | 0.00856894 |
| `abs_logprob_diff_p50` | 1.001e-05 |
| `abs_logprob_diff_p90` | 0.0260685 |
| `abs_logprob_diff_p95` | 0.0504999 |
| `abs_logprob_diff_p99` | 0.101958 |
| `abs_logprob_diff_p999` | 0.222131 |
| `rank_delta_nonzero_rate` | 0.00613281 |
| `topk_jaccard_mean` | 0.968735 |
| `topk_jaccard_p05` | 0.882353 |
| `union_js_p95` | 0.00106358 |
| `union_js_p99` | 0.00281771 |
| `max_abs_logprob_diff` | 18.7481 |
| `missing_selected_count` | 0 |

## Batch accept checks

| metric | op | threshold | value | passed |
|---|---:|---:|---:|---|
| `missing_selected_count` | == | 0 | 0 | True |
| `abs_logprob_diff_p50` | <= | 0.0001 | 1.001e-05 | True |
| `mean_abs_logprob_diff` | <= | 0.015 | 0.00856894 | True |
| `abs_logprob_diff_p90` | <= | 0.06 | 0.0260685 | True |
| `abs_logprob_diff_p95` | <= | 0.09 | 0.0504999 | True |
| `abs_logprob_diff_p99` | <= | 0.2 | 0.101958 | True |
| `rank_delta_nonzero_rate` | <= | 0.012 | 0.00613281 | True |
| `topk_jaccard_mean` | >= | 0.94 | 0.968735 | True |
| `topk_jaccard_p05` | >= | 0.82 | 0.882353 | True |
| `union_js_p95` | <= | 0.004 | 0.00106358 | True |
| `union_js_p99` | <= | 0.012 | 0.00281771 | True |

## Reject checks

| metric | op | threshold | value | hit |
|---|---:|---:|---:|---|
| `missing_selected_count` | > | 0 | 0 | False |
| `mean_abs_logprob_diff` | > | 0.02 | 0.00856894 | False |
| `abs_logprob_diff_p95` | > | 0.12 | 0.0504999 | False |
| `abs_logprob_diff_p99` | > | 0.25 | 0.101958 | False |
| `rank_delta_nonzero_rate` | > | 0.03 | 0.00613281 | False |
| `topk_jaccard_mean` | < | 0.92 | 0.968735 | False |
| `union_js_p99` | > | 0.02 | 0.00281771 | False |

## Sample verdicts

| verdict | count |
|---|---:|
| PASS_PROVISIONAL | 192 |
| INCONCLUSIVE | 7 |
| REJECT | 1 |
