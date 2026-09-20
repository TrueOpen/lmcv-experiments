# vLLM Worker Verify Pipeline

This folder holds the standalone pipeline scripts. The main script chains together the local implementations in the same directory:

1. `generate_prompts.py`
2. `collect_vllm_worker_evidence.py`
3. `verify_vllm_worker_evidence.py`
4. `summarize_worker_verify.py`

When running the vLLM stages, the current Python environment still needs `transformers`, `vllm`, and the GPU runtime required by the model.

```bash
python3 -m pip install transformers vllm
```

Then run:

```bash
./run_vllm_worker_verify.py \
  --baseline-model /models/Qwen3-32B \
  --comparison-model /models/Qwen3-32B-FP8 \
  --baseline-variant-id qwen3-32b-bf16-worker \
  --comparison-variant-id qwen3-32b-fp8-verifier \
  --baseline-dtype bfloat16 \
  --comparison-quantization fp8 \
  --comparison-dtype auto \
  --trust-remote-code
```

Common options:

- `--dry-run`: only print the commands, don't write any output.
- `--prompts path/to/prompts.jsonl`: reuse existing prompts and skip prompt generation.
- `--resume`: reuse existing stage artifacts.
- `--force`: overwrite existing stage artifacts.
- `--python /path/to/python`: specify the Python interpreter used by the local sub-scripts.

By default, output goes under the current directory:

```text
results/vllm_worker_verify_runs/<run-id>/
  data/prompts.jsonl
  data/worker_evidence.<baseline>.jsonl
  worker_verify/<baseline>_worker_vs_<comparison>_verifier/
    worker_vs_verifier_depth_metrics.csv
    summary.md
    distribution/summary.md
```

## Parameter Calibration and Statistical Analysis

The recommended workflow has three steps:

1. `run_vllm_worker_verify.py` generates the base / challenger experiment data.
2. `calibrate_vllm_worker_verify_params.py` automatically derives recommended parameters from the experiment data.
3. `analyze_vllm_worker_verify.py` uses the parameters from step 2 to verify whether the base model can be distinguished from other models.

### 2. Automatic Parameter Calibration

Based on the `worker_vs_verifier_depth_metrics.csv` of the base model and challenger models, automatically derive the PASS / REJECT boundaries for each metric:

```bash
./calibrate_vllm_worker_verify_params.py \
  --input results/vllm_worker_verify_runs \
  --base-model Qwen/Qwen3-32B \
  --output-dir calibration
```

Output files:

```text
calibration/
  recommended_thresholds.json   # recommended parameters, can be fed directly to analyze
  metric_calibration.csv        # per-metric base/other distribution, recommended boundaries, separation
  group_verdicts.csv            # model-level verdict stats after replaying with the recommended parameters
  sample_verdicts.csv           # sample-level verdicts after replaying with the recommended parameters
  summary.md                    # quick report
```

Common calibration strategy options:

```bash
--target-base-pass-rate 0.99          # how many base samples the PASS boundary should cover
--max-base-false-reject-rate 0.01     # how many base false rejects the REJECT boundary allows
--pass-safety-multiplier 1.10         # PASS safety margin for upper-bound metrics
--lower-margin 0.02                   # PASS safety margin for lower-bound metrics such as jaccard
```

### 3. Using the Parameters for Statistics / Validation

Based on the `recommended_thresholds.json` from step 2, generate the protocol-aligned statistics table:

```bash
./analyze_vllm_worker_verify.py \
  --input results/vllm_worker_verify_runs \
  --base-model Qwen/Qwen3-32B \
  --thresholds calibration/recommended_thresholds.json \
  --output-dir analysis
```

Output files:

```text
analysis/
  run_summary.csv              # overall metrics for each run/model
  model_summary.csv            # aggregated by verifier model
  sample_summary.csv           # per-sample metric summary + PASS/REJECT/INCONCLUSIVE
  by_input_bucket.csv          # aggregated by input token bucket
  by_output_depth_bucket.csv   # aggregated by generated token depth bucket
  top_outlier_tokens.csv       # token-level detail with the largest abs_logprob_diff
  thresholds.json              # local analysis thresholds used or derived from base
  summary.md                   # report for quick reading
```

The statistics script follows `PREFILL_GENERATED_TOKEN_METRICS_V1`, covering the following over all generated output tokens:

- `mean_abs_logprob_diff`
- `abs_logprob_diff_p95 / p99`
- `rank_delta_nonzero_rate`
- `topk_jaccard_mean`
- `union_js_p99`
- `missing_compared_count`

Note: the calibration script outputs recommended parameters derived from experiment data, suitable for the initial profile values and backtesting; the final protocol thresholds should still go into `ProfileState.verification_thresholds` and be confirmed through review / governance.

## Worker vs Verifier Time-Cost Benchmark

`benchmark_worker_verifier_latency.py` is used to measure and compare the time cost of the two paths in isolation, sweeping the impact of different input / output lengths:

- **worker**: normal generation, `prefill(input) + decode(output_len)`, `SamplingParams(max_tokens=output_len, logprobs=K)`.
- **verifier**: based on the worker's `input_ids + output_ids`, does one prefill to obtain top-k logprobs, `SamplingParams(max_tokens=1, prompt_logprobs=K)`.

The two paths **share the same offline `vllm.LLM` instance** (same model, same GPU state, for a fair comparison); the verifier uses the `output_ids` actually generated by the worker. The output length is strictly fixed via `ignore_eos=True` + `min_tokens=max_tokens`, keeping the sweep axis clean. Each grid cell first runs `--warmup` times (not timed) and then `--repeats` times, reporting median / mean / p10 / p90; it also splits out TTFT (≈prefill) and decode from `request_output.metrics`.

```bash
./benchmark_worker_verifier_latency.py \
  --model /models/Qwen3-8B \
  --dtype bfloat16 --trust-remote-code \
  --input-lengths "128 512 1024 2048 4096" \
  --output-lengths "1 16 64 256 1024" \
  --top-k 20 --warmup 2 --repeats 5 \
  --gpu "L4" \
  --output-dir results/worker_verifier_latency/qwen3_8b
```

Common options:

- `--input-lengths` / `--output-lengths`: comma- or space-separated sweep grid.
- `--top-k`: depth of the worker's `logprobs` and the verifier's `prompt_logprobs` (`--max-logprobs` defaults to this value).
- `--warmup` / `--repeats`: number of warmup and timed iterations.
- `--skip-verifier` / `--skip-worker`: measure only one of the two paths.
- `--max-total-tokens`: skip cells where input+output exceeds this value (0 disables).

Output (under `--output-dir`):

```text
results/worker_verifier_latency/<name>/
  latency_summary.csv   # per-cell median/mean/p10/p90, TTFT, decode, tok/s, worker/verifier ratio
  latency_raw.jsonl     # raw data for each timed iteration
  summary.md            # Markdown table
  metadata.json         # parameters, vLLM version, environment
```

How to read the results: rows with `output_len=1` reflect the growth of pure prefill as input length increases (worker≈verifier); with input fixed and output increasing, the worker's decode grows linearly while the verifier only adds one extra prefill, so `worker_over_verifier_median` widens as output length grows.
