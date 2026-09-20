# MoE Worker-Verifier Spike Hypothesis Testing Report

This report is compiled from two sets of results under `experiments/logprobs/data/qwen3.6-35b-a3b/`:

- BF16 same-model verifier: 1 original BF16 result (`...-bf16`) plus 3 replicate runs
  (`...-bf16_0001` / `_0002` / `_0003`), for a total of 4 BF16 baselines.
- FP8 verifier result: `6000ws_worker_vs_6000ws_verifier_qwen3.6-35b-a3b-fp8/`

The goal is to test the following hypotheses:

```text
For MoE models, the diff between same-model BF16 worker decode and BF16 verifier prefill
mainly manifests as sparse, position-sensitive token-level spikes;

while the diff between the FP8 verifier and the BF16 worker, beyond inheriting these
sensitive token spikes, also exhibits a more pervasive right shift in the distribution.

Therefore, replacing the single-token decision with window-level robust aggregation is reasonable.
```

## 1. Summary of Conclusions

The current experimental results support the hypotheses above. Once the original BF16 result is included, there are 4 BF16 baselines in total, making the conclusions more robust than the three-replicate version.

1. **The large diffs in the BF16 same-model baseline are primarily token-level spikes.**
   - Across the four BF16 verifier runs, the `abs_diff > 1` token rate is only `0.53% - 0.62%`.
   - Of these spikes, `92.8% - 94.3%` are isolated single-point runs.
   - Using the pooled BF16 `p999 = 11.123` as the extreme-spike threshold, `96.9% - 100%` of the spike runs across the four experiments are single points.

2. **These spikes are not pure random noise but token/context sensitive points.**
   - Among the `abs_diff > 1` spike locations, `65.1%` appear in at least 2 BF16 runs, `43.6%` in at least 3 runs, and `21.5%` in all 4 runs.
   - Among the `p999`-level extreme spikes, `56.1%` still appear in at least 2 runs.
   - This indicates that spikes behave more like tokens that are sensitive under the MoE routing / decode-prefill path, rather than run noise at completely random positions.

3. **The window median substantially suppresses the BF16 baseline tail.**
   - The BF16 token p99 is roughly `0.366 - 0.433`.
   - The `window=5 median` p99 drops to `0.097 - 0.113`.
   - The `window=16 median` p99 drops further to `0.044 - 0.054`.
   - This matches the expectation that "single-point spikes are filtered out by the window median".

4. **FP8 also has spikes, but it is not pure spike; it also exhibits a more pervasive right shift.**
   - Using the BF16 pooled p99 `0.408` as the threshold, the FP8 token rate is `2.00%`, clearly higher than the BF16 baseline's `0.89% - 1.07%`.
   - Using the BF16 window median p99 as the threshold, the FP8 bad-window rate remains clearly higher than BF16 as the window grows:
     - `window=5`: FP8 `3.41%` vs BF16 baseline about `1%`
     - `window=16`: FP8 `4.66%` vs BF16 baseline about `1%`
     - `window=32`: FP8 `5.07%` vs BF16 baseline about `1%`

5. **The paired comparison further supports that FP8 is a pervasive shift.**
   - At the same sample/depth, the fraction where `FP8 diff > BF16 4-run mean diff` is `63.9%`.
   - The larger the window, the higher this fraction:
     - `window=5`: `68.7%`
     - `window=16`: `78.8%`
     - `window=32`: `84.8%`
   - If FP8 were only random spikes, this monotonic strengthening should not appear as the window grows.

One-sentence conclusion:

```text
The large tail of the BF16 same-model diff mainly comes from sparse, position-sensitive token spikes;
FP8 inherits these sensitive spikes while superimposing a more pervasive quantization distribution shift.
Therefore, window-level median / bad-window-rate is a more reasonable verifier decision method than single-token diff.
```

## 2. Data and Experimental Subjects

### 2.1 BF16 Baseline Runs

Four BF16 same-model verifier results:

All result directories are under `experiments/logprobs/data/qwen3.6-35b-a3b/`:

| run | result dir |
|---|---|
| `0000_results` | `6000ws_worker_vs_6000ws_verifier_qwen3.6-35b-a3b-bf16/` |
| `0001` | `6000ws_worker_vs_6000ws_verifier_qwen3.6-35b-a3b-bf16_0001/` |
| `0002` | `6000ws_worker_vs_6000ws_verifier_qwen3.6-35b-a3b-bf16_0002/` |
| `0003` | `6000ws_worker_vs_6000ws_verifier_qwen3.6-35b-a3b-bf16_0003/` |

Shared configuration:

| item | value |
|---|---|
| worker evidence | `worker_evidence.6000ws_long_first.jsonl` |
| worker model | `Qwen/Qwen3.6-35B-A3B` |
| verifier model | `Qwen/Qwen3.6-35B-A3B` |
| quantization | `bf16` |
| dtype | `bfloat16` |
| gpu | `6000ws` |
| backend | `vLLM 0.24.0` |
| prompt_logprobs / max_logprobs | `32 / 32` |
| batch size | `1` |
| depth rows | `31411` per run |

### 2.2 FP8 Result

FP8 verifier result:

```text
experiments/logprobs/data/qwen3.6-35b-a3b/6000ws_worker_vs_6000ws_verifier_qwen3.6-35b-a3b-fp8/
```

Configuration:

| item | value |
|---|---|
| worker model | `Qwen/Qwen3.6-35B-A3B` BF16 |
| verifier model | `Qwen/Qwen3.6-35B-A3B-FP8` |
| verifier quantization | `fp8` |
| dtype | `auto` |
| gpu | `6000ws` |
| prompt_logprobs / max_logprobs | `32 / 32` |
| depth rows | `31411` |

## 3. Overall BF16 Baseline Distribution

Overall metrics for the four BF16 baselines:

| run | mean | p50 | p95 | p99 | p999 | max |
|---|---:|---:|---:|---:|---:|---:|
| `0000_results` | 0.0591 | 0.000606 | 0.1432 | 0.4331 | 11.1895 | 24.1029 |
| `0001` | 0.0540 | 0.000525 | 0.1196 | 0.3665 | 10.1542 | 25.4778 |
| `0002` | 0.0572 | 0.000568 | 0.1362 | 0.4208 | 10.5622 | 25.5559 |
| `0003` | 0.0622 | 0.000619 | 0.1423 | 0.4181 | 11.8944 | 24.1498 |

Pooled BF16 baseline thresholds:

| threshold | value |
|---|---:|
| pooled p95 | 0.1350 |
| pooled p99 | 0.4082 |
| pooled p999 | 11.1230 |

As can be seen, the bulk of the BF16 baseline error is very small, but the tail is very long:

```text
p50 ~5e-4
p99 ~0.4
p999 ~11
max ~24-26
```

This already suggests that the large diffs in the same-model baseline are not an overall drift, but rather extreme amplification at a small number of tokens.

## 4. BF16 Spike Morphology Analysis

### 4.1 Using abs_diff > 1

| run | spike count | token rate | sample any | run length=1 | run >=2 | spike tokens in run>=2 | max run |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0000_results` | 183 | 0.583% | 38.7% | 92.9% | 7.1% | 13.7% | 3 |
| `0001` | 166 | 0.528% | 34.3% | 94.3% | 5.7% | 10.8% | 2 |
| `0002` | 174 | 0.554% | 33.3% | 93.9% | 6.1% | 11.5% | 2 |
| `0003` | 195 | 0.621% | 38.3% | 92.8% | 7.2% | 14.4% | 3 |

Conclusion:

```text
Tokens with abs_diff > 1 are rare, and the vast majority are isolated tokens.
```

### 4.2 Using pooled BF16 p99

Threshold:

```text
abs_diff > 0.408211
```

| run | spike count | token rate | sample any | run length=1 | run >=2 | spike tokens in run>=2 | max run |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0000_results` | 337 | 1.073% | 56.0% | 90.2% | 9.8% | 18.4% | 3 |
| `0001` | 279 | 0.888% | 46.0% | 91.9% | 8.1% | 15.1% | 2 |
| `0002` | 320 | 1.019% | 50.3% | 90.7% | 9.3% | 17.8% | 3 |
| `0003` | 321 | 1.022% | 52.0% | 90.3% | 9.7% | 18.7% | 3 |

Conclusion:

```text
When defining spikes by the p99 threshold, the four BF16 baselines stabilize at ~1% token rate;
~90% or more of the spike runs are still single points.
```

### 4.3 Using pooled BF16 p999

Threshold:

```text
abs_diff > 11.123
```

| run | spike count | token rate | sample any | run length=1 | run >=2 | max run |
|---|---:|---:|---:|---:|---:|---:|
| `0000_results` | 33 | 0.105% | 9.3% | 96.9% | 3.1% | 2 |
| `0001` | 27 | 0.086% | 8.3% | 100.0% | 0.0% | 1 |
| `0002` | 30 | 0.096% | 9.3% | 100.0% | 0.0% | 1 |
| `0003` | 36 | 0.115% | 9.3% | 97.1% | 2.9% | 3 |

Conclusion:

```text
The p999-level extreme diffs are almost entirely isolated token spikes.
```

This is very important for the window median: as long as fewer than half of the tokens within a window spike, the median will not be dominated by these extreme values.

## 5. Whether BF16 Spike Locations Are Stable

If spikes were pure random run noise, the spike locations across different runs should be highly unstable. The actual results show that a substantial fraction of spike locations recur.

Here `(sample_id, depth)` is treated as the spike location.

### 5.1 Location Overlap for abs_diff > 1

Distribution of spike location occurrence counts:

| BF16 run count | locations |
|---|---:|
| 1 | 109 |
| 2 | 67 |
| 3 | 69 |
| 4 | 67 |

In other words:

```text
65.1% of spike locations appear in at least 2 runs;
43.6% of spike locations appear in at least 3 runs;
21.5% of spike locations appear in all 4 BF16 baselines.
```

### 5.2 Location Overlap for pooled p999 Extreme Spikes

Occurrence counts of extreme spike locations:

| BF16 run count | locations |
|---|---:|
| 1 | 29 |
| 2 | 19 |
| 3 | 13 |
| 4 | 5 |

In other words:

```text
56.1% of extreme spike locations appear in at least 2 runs;
27.3% of extreme spike locations appear in at least 3 runs;
7.6% of extreme spike locations appear in all 4 BF16 baselines.
```

### 5.3 Examples of Stable High-Spike Locations

The following locations repeatedly exhibit large diffs across all 4 BF16 baselines:

| sample | depth | run count | mean diff | max diff |
|---|---:|---:|---:|---:|
| `ef63ae0b76a298deb773` | 13 | 4 | 22.65 | 25.56 |
| `aa2d2f1274b0d99eee0e` | 7 | 4 | 21.92 | 23.29 |
| `21e9f9d...` | 6 | 4 | 18.40 | 24.15 |
| `039de85a378ae7082fd2` | 8 | 4 | 18.34 | 18.94 |
| `3d4f88e2f300c1aa21b8` | 8 | 4 | 16.80 | 18.26 |
| `5ade7d...` | 7 | 4 | 16.17 | 21.20 |
| `23cb4621d0cc79754f3c` | 5 | 4 | 15.58 | 16.36 |
| `ecdb9497228e5a267c55` | 9 | 4 | 13.76 | 15.12 |

Interpretation:

```text
These spikes are not randomly scattered points, but specific sample/depth locations sensitive to the decode-prefill path.
For MoE models, this matches the expectation of tokens sensitive to routing boundaries / expert selection.
```

## 6. Suppression of the BF16 Tail by the Window Median

| run | token p99 | window=5 median p99 | window=16 median p99 | token max | window=5 max | window=16 max |
|---|---:|---:|---:|---:|---:|---:|
| `0000_results` | 0.4331 | 0.1101 | 0.0513 | 24.1029 | 1.9280 | 0.1561 |
| `0001` | 0.3665 | 0.0972 | 0.0437 | 25.4778 | 2.5547 | 0.1618 |
| `0002` | 0.4208 | 0.1131 | 0.0540 | 25.5559 | 2.2055 | 0.1050 |
| `0003` | 0.4181 | 0.1124 | 0.0537 | 24.1498 | 14.1659 | 0.1185 |

Conclusion:

```text
window=5 median can already significantly reduce the p99 tail of BF16 same-model;
window=16 median further compresses p99 to ~0.05.
```

The `window=5 max = 14.1659` for `0003` indicates there is still a local segment within a short window where most tokens are anomalous, but the `window=16 max = 0.1185` shows that such a local segment is not long.

## 7. FP8 Overall and Spike Analysis

FP8 verifier overall metrics:

| metric | BF16 `0000_results` | FP8 |
|---|---:|---:|
| mean | 0.0591 | 0.0848 |
| p50 | 0.000606 | 0.001088 |
| p95 | 0.1432 | 0.2278 |
| p99 | 0.4331 | 0.6729 |
| p999 | 11.1895 | 12.6798 |
| max | 24.1029 | 27.5559 |

Both the bulk and the tail of FP8 are higher than the BF16 original baseline.

### 7.1 FP8 Using BF16 pooled thresholds

| threshold | value | FP8 spike count | FP8 token rate | FP8 sample any | FP8 run length=1 | FP8 max run |
|---|---:|---:|---:|---:|---:|---:|
| BF16 pooled p95 | 0.1350 | 3105 | 9.89% | 94.7% | 82.7% | 5 |
| BF16 pooled p99 | 0.4082 | 628 | 2.00% | 75.3% | 92.1% | 3 |
| BF16 pooled p999 | 11.1230 | 46 | 0.146% | 13.3% | 97.8% | 2 |
| abs > 1 | 1.0 | 229 | 0.729% | 41.3% | 92.5% | 3 |

Comparison with the BF16 baseline:

| threshold | BF16 token rate | FP8 token rate |
|---|---:|---:|
| BF16 pooled p95 | 4.17% - 5.39% | 9.89% |
| BF16 pooled p99 | 0.89% - 1.07% | 2.00% |
| abs > 1 | 0.53% - 0.62% | 0.729% |
| BF16 pooled p999 | 0.086% - 0.115% | 0.146% |

Conclusion:

```text
The extreme tail of FP8 is still mainly spikes;
but the number of tokens exceeding the BF16 baseline envelope is clearly larger.
```

### 7.2 FP8 Using Its Own thresholds

| threshold | value | token rate | run length=1 | max run |
|---|---:|---:|---:|---:|
| FP8 p95 | 0.2278 | 5.00% | 88.8% | 4 |
| FP8 p99 | 0.6729 | 1.00% | 93.2% | 3 |
| FP8 p999 | 12.6798 | 0.102% | 96.8% | 2 |

Conclusion:

```text
As for FP8's own extreme tail, it also exhibits a spike-like morphology.
```

This indicates that FP8 is not a completely smooth distribution shift; it simultaneously contains:

1. Spikes at MoE/decode-prefill sensitive tokens;
2. A more pervasive rightward diff shift introduced by FP8 quantization.

## 8. FP8 Window-Level Separation

### 8.1 Window p99

| variant | token p99 | window=5 p99 | window=16 p99 | window=32 p99 |
|---|---:|---:|---:|---:|
| BF16 `0000_results` | 0.4331 | 0.1101 | 0.0513 | 0.0376 |
| BF16 `0001` | 0.3665 | 0.0972 | 0.0437 | 0.0312 |
| BF16 `0002` | 0.4208 | 0.1131 | 0.0540 | 0.0378 |
| BF16 `0003` | 0.4181 | 0.1124 | 0.0537 | 0.0384 |
| FP8 | 0.6729 | 0.1685 | 0.0802 | 0.0567 |

FP8 is higher than the BF16 baseline at both the token level and the window level.

### 8.2 Bad Window Rate

Here the BF16 pooled window median p99 is used as the threshold, so the BF16 baseline bad window rate is expected to be about `1%`.

| window | BF16 pooled window p99 threshold | BF16 `0000_results` bad rate | BF16 4-run bad-rate range | FP8 bad rate |
|---|---:|---:|---:|---:|
| 3 | 0.1600 | 1.08% | 0.79% - 1.08% | 2.99% |
| 5 | 0.1085 | 1.02% | 0.78% - 1.13% | 3.41% |
| 16 | 0.0513 | 1.00% | 0.65% - 1.17% | 4.66% |
| 32 | 0.0365 | 1.21% | 0.50% - 1.22% | 5.07% |

Conclusion:

```text
The larger the window, the more BF16 spikes are suppressed;
but the FP8 bad window rate remains in the 3%-5% range, and is clearly higher relative to BF16.
```

This is precisely the evidence that "FP8 exhibits a more pervasive shift".

## 9. Paired Comparison of FP8 and BF16

Directly comparing the same sample/depth:

| comparison | FP8 > BF16 rate | delta mean | delta p90 | delta p95 | delta p99 |
|---|---:|---:|---:|---:|---:|
| token vs BF16 `0000_results` | 64.0% | 0.0258 | 0.0670 | 0.1405 | 0.4296 |
| token vs BF16 4-run mean | 63.9% | 0.0267 | 0.0662 | 0.1364 | 0.4180 |

Window-level paired comparison against BF16 4-run mean:

| window | FP8 > BF16 rate | delta mean | delta p90 | delta p99 |
|---|---:|---:|---:|---:|
| 3 | 66.1% | 0.0117 | 0.0376 | 0.1843 |
| 5 | 68.7% | 0.00677 | 0.0287 | 0.1193 |
| 16 | 78.8% | 0.00418 | 0.0162 | 0.0490 |
| 32 | 84.8% | 0.00312 | 0.0116 | 0.0315 |

This result is highly significant:

```text
If FP8 were only random spikes, the fraction of FP8 > BF16 should not keep rising as the window grows;
in reality, the larger the window, the higher this fraction, indicating that FP8's shift is more pervasive.
```

## 10. The Relationship Between FP8 Spikes and BF16 Sensitive Locations

Many of the FP8 spikes occur at locations where the BF16 baseline already exhibits spikes.

| threshold | FP8 spikes | overlap with BF16 >=1 run | overlap with BF16 >=2 runs | overlap with BF16 >=3 runs | overlap with BF16 all 4 runs |
|---|---:|---:|---:|---:|---:|
| abs > 1 | 229 | 72.1% | 55.0% | 41.0% | 24.9% |
| BF16 pooled p99 | 628 | 48.6% | 37.3% | 27.2% | 17.8% |
| BF16 pooled p999 | 46 | 54.3% | 39.1% | 28.3% | 8.7% |

Interpretation:

```text
Many of FP8's large spikes fall at locations where BF16 same-model is already sensitive.
This indicates that MoE/decode-prefill sensitive tokens are still one of the main spike sources.

But FP8 also has a large number of additional locations exceeding the BF16 baseline threshold,
and its window-level metrics remain consistently higher than BF16, so FP8 is not purely reusing BF16 spikes.
```

## 11. Final Determination

The current experiments support the following judgments:

### 11.1 Hypothesis 1 Holds

```text
The large diffs of BF16 same-model worker decode vs verifier prefill are mainly sparse token-level spikes.
```

Evidence:

- Low token rate;
- The vast majority of run lengths are 1;
- p999 extreme spikes are almost entirely isolated points;
- The window median substantially suppresses the BF16 tail.

### 11.2 Hypothesis 2 Holds

```text
These spikes are not pure random noise, but are sensitive to specific token/context.
```

Evidence:

- Spikes at the same sample/depth show clear overlap across the 4 BF16 runs;
- Multiple extreme spike locations recur across the 4 BF16 runs;
- This matches the expectation of tokens sensitive to MoE routing boundaries or the decode-prefill path.

### 11.3 Hypothesis 3 Holds

```text
The FP8 verifier both inherits the MoE/decode-prefill sensitive spikes and exhibits a more pervasive distribution right shift.
```

Evidence:

- The extreme tail of FP8 is still spike-like;
- But FP8 has more tokens exceeding the BF16 baseline threshold;
- The FP8 window-level bad rate is clearly higher than the BF16 baseline;
- In the paired window comparison, the fraction of `FP8 > BF16` rises as the window grows.

### 11.4 Methodological Conclusion

Therefore, for the current MoE verifier scenario, we recommend against using the single-token `abs_logprob_diff` as the primary decision metric. The most reasonable minimal scheme is:

```text
metric = window=5 median(abs_logprob_diff)
threshold = window=5 median p99 of the BF16 same-model baseline
score = bad_window_rate
reject if score is clearly higher than the BF16 baseline envelope
```

Under the current data, the corresponding values are:

```text
BF16 pooled window=5 median p99 threshold ≈ 0.1085
BF16 4-run bad-window-rate range ≈ 0.78% - 1.13%
FP8 bad-window rate ≈ 3.41%
```

If greater robustness is needed, one can additionally monitor:

```text
window=16 median bad-window rate
window=32 median bad-window rate
```

But for the first version of the decision rule, using only `window=5 median + bad_window_rate` is already sufficient to express the core idea.

## 12. One-Sentence Summary

```text
The four BF16 baselines prove that the large diffs of same-model are mainly sparse, position-sensitive token spikes;
FP8 proves that a differently-quantized verifier still has a more pervasive window-level shift beyond these spikes.
Therefore, the window-level verifier hypothesis proposed by the user holds in the current MoE experiments.
```
