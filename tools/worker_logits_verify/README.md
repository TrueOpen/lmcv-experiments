# Worker Logits Verify

This project is a redesigned worker-verify experiment that does not rely on the old first-token batch/fork logic. The core goal is to build a **depth-aware logits/logprob fingerprint along the same context path**, used to distinguish:

- Should pass: legitimate numerical drift of `Qwen3-32B BF16` between `H100` and `RTX Pro 6000`.
- Should fail: challenges to `Qwen3-32B BF16` identity from `Qwen3-32B FP8/AWQ`, `Qwen3-14B`, and `Qwen3-8B`.

## Experiment definition

First, fix a canonical input-output pair:

```text
anchor = Qwen3-32B BF16 on H100
sample = input_text + canonical_output_text
```

After that, every variant performs a teacher-forced replay:

```text
context_t = input_ids + output_ids[:t]
target_t = output_ids[t]
logits_t = model(context_t).logits[-1]
```

This compares the numerical drift along the same output path, not free-generation branching.

## Why not use vLLM logprobs

The vLLM API usually returns logprobs, i.e. the values after `log_softmax(logits)`, not raw logits. This project's collection backend uses `transformers` first, taking directly from the forward output:

- `raw_logits`
- `logprobs`
- selected token rank
- top-k logprob fingerprint
- probe token logprob fingerprint

Raw logits are used for identity fingerprint research; logprob/rank/NLL are used as the main verification metrics.

## Data scale

By default, 6 input token length buckets are generated:

```text
32, 128, 512, 2048, 8192, 16384
```

50 samples per bucket, 300 total. The default anchor output generates 128 tokens, so the first version is roughly:

```text
300 * 128 = 38,400 depth rows / variant
```

If the H100 vs 6000 envelope is very narrow and the negative samples clearly fail, expand to 100 samples per bucket or a longer output.

## Installation

```bash
cd worker_logits_verify
python3 -m pip install -e ".[transformers]"
```

If you already have a GPU environment, you can also run directly:

```bash
PYTHONPATH=src python3 -m worker_logits_verify.generate_prompts --help
```

## 1. Generate calibration inputs

```bash
worker-logits-generate-prompts \
  --model /models/Qwen3-32B \
  --buckets 32,128,512,2048,8192,16384 \
  --per-bucket 50 \
  --output data/qwen3_32b_identity_v1/prompts.jsonl \
  --trust-remote-code
```

## 2. Generate the canonical output

Recommended to fix:

```text
Qwen3-32B BF16 on H100
```

```bash
worker-logits-generate-anchor \
  --model /models/Qwen3-32B \
  --variant-id qwen3-32b-bf16-h100 \
  --gpu H100 \
  --quantization bf16 \
  --dtype bfloat16 \
  --input data/qwen3_32b_identity_v1/prompts.jsonl \
  --output data/qwen3_32b_identity_v1/anchor_samples.jsonl \
  --max-new-tokens 128 \
  --trust-remote-code
```

## 3. Collect H100 and 6000 traces

H100:

```bash
worker-logits-collect-trace \
  --samples data/qwen3_32b_identity_v1/anchor_samples.jsonl \
  --model /models/Qwen3-32B \
  --variant-id qwen3-32b-bf16-h100 \
  --gpu H100 \
  --quantization bf16 \
  --dtype bfloat16 \
  --output-dir results/qwen3_32b_identity_v1/traces/qwen3-32b-bf16-h100 \
  --top-k 64 \
  --trust-remote-code
```

6000:

```bash
worker-logits-collect-trace \
  --samples data/qwen3_32b_identity_v1/anchor_samples.jsonl \
  --model /models/Qwen3-32B \
  --variant-id qwen3-32b-bf16-6000 \
  --gpu RTXPro6000 \
  --quantization bf16 \
  --dtype bfloat16 \
  --output-dir results/qwen3_32b_identity_v1/traces/qwen3-32b-bf16-6000 \
  --top-k 64 \
  --trust-remote-code
```

## 4. Build the legitimate cross-GPU envelope

```bash
worker-logits-compare-traces \
  --left results/qwen3_32b_identity_v1/traces/qwen3-32b-bf16-h100 \
  --right results/qwen3_32b_identity_v1/traces/qwen3-32b-bf16-6000 \
  --left-label H100-BF16 \
  --right-label 6000-BF16 \
  --output-dir results/qwen3_32b_identity_v1/comparisons/h100_vs_6000_bf16
```

First analyze whether token depth actually changes drift:

```bash
worker-logits-analyze-depth-effect \
  --comparison results/qwen3_32b_identity_v1/comparisons/h100_vs_6000_bf16 \
  --output-dir results/qwen3_32b_identity_v1/depth_effect/h100_vs_6000_bf16
```

Focus on:

- `depth_bucket_summary.csv`: p50/p95/p99/max per depth bucket.
- `depth_trends.csv`: slope / Pearson r / R2 of each metric against absolute depth and relative depth.
- `cumulative_by_sample.csv`: cumulative NLL delta and cumulative abs drift within each sample.
- `summary.md`: a quick-read summary.

If the slope is close to 0 and R2 is very low, depth is not a major factor; if the slope is clearly positive and bucket p95/p99 rise with depth, then depth buckets need to be kept in the profile.

Then build the legitimate envelope:

```bash

worker-logits-calibrate-profile \
  --comparison results/qwen3_32b_identity_v1/comparisons/h100_vs_6000_bf16 \
  --profile-id qwen3-32b-bf16-h100-6000-envelope \
  --variant-pair "Qwen3-32B BF16 H100 vs RTXPro6000" \
  --output-dir results/qwen3_32b_identity_v1/profiles/h100_6000_bf16
```

## 5. Run the negative-sample challenges

For example FP8:

```bash
worker-logits-collect-trace \
  --samples data/qwen3_32b_identity_v1/anchor_samples.jsonl \
  --model /models/Qwen3-32B-FP8 \
  --variant-id qwen3-32b-fp8-h100 \
  --gpu H100 \
  --quantization fp8 \
  --dtype auto \
  --output-dir results/qwen3_32b_identity_v1/traces/qwen3-32b-fp8-h100 \
  --top-k 64 \
  --trust-remote-code \
  --allow-retokenize

worker-logits-compare-traces \
  --left results/qwen3_32b_identity_v1/traces/qwen3-32b-bf16-h100 \
  --right results/qwen3_32b_identity_v1/traces/qwen3-32b-fp8-h100 \
  --left-label H100-BF16 \
  --right-label H100-FP8 \
  --output-dir results/qwen3_32b_identity_v1/comparisons/bf16_vs_fp8

worker-logits-verify-comparison \
  --comparison results/qwen3_32b_identity_v1/comparisons/bf16_vs_fp8 \
  --profile results/qwen3_32b_identity_v1/profiles/h100_6000_bf16/profile.json \
  --output-dir results/qwen3_32b_identity_v1/verifications/bf16_vs_fp8
```

The expected verification verdict for FP8/AWQ/14B/8B is `FAIL`.

## Worker Evidence vs Verifier Logprob

If you already have `prompts.jsonl` and want to verify directly:

```text
the logprob produced during worker decode
vs
the logprob the verifier recomputes for the same input+output
```

you can go through the vLLM evidence pipeline.

Step one: generate output in the worker environment, and at the same time save each output token's decode-time logprob/top-k:

```bash
worker-logits-collect-vllm-evidence \
  --input data/qwen3_32b_identity_v1/prompts.jsonl \
  --output data/qwen3_32b_identity_v1/worker_evidence.h100.jsonl \
  --model Qwen/Qwen3-32B \
  --variant-id qwen3-32b-bf16-h100 \
  --gpu H100 \
  --quantization bf16 \
  --dtype bfloat16 \
  --max-new-tokens 128 \
  --logprobs 64 \
  --max-logprobs 64 \
  --batch-size 8 \
  --trust-remote-code
```

Step two: in the verifier environment, run vLLM `prompt_logprobs` on the same `input_ids + output_ids`, and compare against the worker evidence token by token:

```bash
worker-logits-verify-vllm-evidence \
  --evidence data/qwen3_32b_identity_v1/worker_evidence.h100.jsonl \
  --output-dir results/qwen3_32b_identity_v1/worker_verify/h100_worker_vs_6000_verifier \
  --model Qwen/Qwen3-32B \
  --variant-id qwen3-32b-bf16-6000-verifier \
  --gpu RTXPro6000 \
  --quantization bf16 \
  --dtype bfloat16 \
  --prompt-logprobs 64 \
  --max-logprobs 64 \
  --batch-size 4 \
  --trust-remote-code
```

You can also use a single-command script to chain prompt generation, worker evidence collection, verifier comparison, and distribution summarization:

```bash
worker-logits-run-vllm-worker-verify \
  --baseline-model /models/Qwen3-32B \
  --comparison-model /models/Qwen3-32B-FP8 \
  --baseline-variant-id qwen3-32b-bf16-worker \
  --comparison-variant-id qwen3-32b-fp8-verifier \
  --baseline-quantization bf16 \
  --comparison-quantization fp8 \
  --baseline-dtype bfloat16 \
  --comparison-dtype auto \
  --buckets 32,128,512,2048,8192,16384 \
  --per-bucket 50 \
  --max-new-tokens 128 \
  --logprobs 64 \
  --prompt-logprobs 64 \
  --max-logprobs 64 \
  --worker-batch-size 8 \
  --verifier-batch-size 4 \
  --output-root results/vllm_worker_verify_runs \
  --trust-remote-code
```

Common helper flags:

- `--dry-run`: only print the pipeline commands that would run, to check paths, dtype, quantization, and batch size first.
- `--prompts path/to/prompts.jsonl`: reuse existing prompts and skip `worker-logits-generate-prompts`.
- `--resume`: reuse stage artifacts that already exist; `--force`: overwrite stage artifacts.
- `--baseline-tensor-parallel-size` / `--comparison-tensor-parallel-size`: set separately when the two models need different TP.

The default output directory is:

```text
results/vllm_worker_verify_runs/<run-id>/
  data/prompts.jsonl
  data/worker_evidence.<baseline>.jsonl
  worker_verify/<baseline>_worker_vs_<comparison>_verifier/
    worker_vs_verifier_depth_metrics.csv
    summary.md
    distribution/summary.md
```

This output tells you:

- `worker_selected_logprob`
- `verifier_selected_logprob`
- `abs_logprob_diff`
- `worker_selected_rank` / `verifier_selected_rank`
- the overlap / JS / cosine between the worker top-k and the verifier top-k

Note that in this pipeline the verifier is `full-prefill prompt_logprobs` while the worker is `decode generate logprobs`. This also lets it measure exactly:

```text
how large the difference is between decode-time worker logprob and full-prefill verifier logprob
```

If this mode delta is large, strict verification requires the verifier to also do a decode replay rather than using a full-prefill approximation.

## Core metrics

For each depth, record:

- selected token raw logit / logprob / rank
- top1/top2 token and margin
- top-k token/logprob/raw_logit map
- top-k mass
- fixed probe token logprobs

When comparing, output:

- `selected_abs_logprob_diff`
- `selected_abs_raw_logit_diff`
- `selected_abs_rank_delta`
- `top1_equal`
- `topk_jaccard`
- top-k observed common/union cosine, KL, JS
- probe logprob cosine / max abs diff
- cumulative NLL delta

Note: top-k KL/JS are only engineering metrics over the observed support, not full-vocab KL.

## File structure

```text
src/worker_logits_verify/
  generate_prompts.py         # generate calibration inputs across input length buckets
  generate_anchor_outputs.py  # fix the canonical output using the anchor model
  collect_trace.py            # teacher-forced replay to collect the depth trace
  compare_traces.py           # same-depth drift comparison between two variants
  calibrate_profile.py        # build the accepted envelope from H100 vs 6000
  verify_comparison.py        # use the profile to decide whether a negative sample is out of bounds
  analyze_depth_effect.py     # specifically analyze whether drift grows with token depth
  collect_vllm_worker_evidence.py
  verify_vllm_worker_evidence.py
  schema.py
  metrics.py
  tokenizer_utils.py
```
