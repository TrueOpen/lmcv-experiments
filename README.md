# lmcv-experiments

Research code, tooling, and empirical data for **Logprob Majority-Consensus
Verification (LMCV)** — a low-cost execution-verification protocol for open,
decentralized inference networks.

## Background

In open inference markets, jobs are dispatched to untrusted worker nodes. A core
integrity question is: *how can the network confirm a worker actually executed
the user-specified reference model and runtime configuration, without re-running
the entire generation?* Full recomputation is as expensive as regenerating the
answer, and comparing only the final text cannot distinguish same-family models,
quantized variants, or a small model masquerading as a large one.

LMCV combines a **token-level probability fingerprint** with **independent
multi-node consensus**. Once generation is complete, a verifier runs a single
parallel *teacher-forced prefill* over the frozen `input_ids + output_ids` path,
reconstructs each position's predictive distribution, and checks the logprob,
rank, and top-*k* candidate set of each returned token — without decoding token
by token. For Mixture-of-Experts (MoE) models it adds a *routed-experts joint
check* using the layer-0 routing fingerprint.

See [`paper/LMCV.md`](paper/LMCV.md) for the full write-up.

## Repository layout

```
.
├── paper/         # The LMCV paper (Markdown)
├── tools/         # Runnable pipelines and scripts
└── experiments/   # Experiment reports and collected data
```

### `paper/`

- **`LMCV.md`** — The complete paper: protocol design (VRF verifier selection,
  commit–reveal, multi-metric decision, two-of-three majority consensus),
  MoE routed-experts extension, and experimental results across Qwen3-32B,
  Qwen3-8B (dense), and Qwen3.6-35B-A3B (MoE).

### `tools/`

- **`worker_logits_verify/`** — The main, redesigned worker-verify project. Builds
  a *depth-aware logits/logprob fingerprint* along a fixed context path using a
  `transformers` backend (raw logits, logprobs, token rank, top-*k* and probe-token
  fingerprints) to separate legitimate cross-GPU numerical drift from cross-model /
  quantization challenges. Packaged under `src/worker_logits_verify/` with example
  traces.
- **`vllm_worker_verify_pipeline/`** — Standalone end-to-end pipeline
  (`generate_prompts` → `collect_vllm_worker_evidence` → `verify_vllm_worker_evidence`
  → `summarize_worker_verify`) driven by vLLM, with calibration, analysis, and
  latency benchmarking scripts.
- **`moe_routed_experts/`** — Tests whether vLLM decode-time and prefill-time
  `routed_experts` agree for the same `input + output_token_ids` (requires
  `vllm serve <MODEL> --enable-return-routed-experts`).

### `experiments/`

Reports and collected data for three experiment tracks. Each is described in
detail in [Experiments in detail](#experiments-in-detail) below.

- **`moe-routed-experts/`** — Routed-experts fingerprint differences for a large
  MoE model.
- **`logprobs/`** — Logprob-fingerprint differences for large dense/MoE LLMs.
- **`worker-verifier-latency/`** — The cost gap between the worker and the verifier.

## Experiments in detail

### 1. MoE routed-experts difference (`experiments/moe-routed-experts/`)

**Question:** For a Mixture-of-Experts model, can the *routing fingerprint* tell
the reference model apart from quantized or cross-GPU variants?

The verifier runs one teacher-forced prefill over `input_token_ids +
output_token_ids`, extracts vLLM's `routed_experts`, and compares them against the
routing recorded in the worker's decode trace. The primary score is the **layer-0
`entry_mismatch_rate`** (`sum(entry_mismatch_count) / sum(compared_entries)`),
alongside a `registered_logit_tokens_top_k` legality check that each output token
falls inside the model's registered top-*k* candidate set.

Compared variants are `Qwen/Qwen3.6-35B-A3B` (BF16) vs. `Qwen/Qwen3.6-35B-A3B-FP8`
(FP8), across GPU pairings (6000ws, H100). Per-pairing route-diff summaries are
stored as CSV/JSON/Markdown (by sample, by layer, and by sample×layer). The report
is [`report/moe_model_routed_experts_verifier.md`](experiments/moe-routed-experts/report/moe_model_routed_experts_verifier.md).

**Finding:** Where per-token logprob is too noisy for MoE, the layer-0 routed-experts
fingerprint yields a structurally stable, cross-GPU-reproducible gap between
same-model and cross-model cases — clean enough to set a production threshold.

### 2. LLM logprob difference (`experiments/logprobs/`)

**Question:** Can a same-model verifier reconstruct the worker's per-token logprob
fingerprint well enough to separate legitimate numerical drift (same model, different
GPU) from an impostor (FP8/AWQ or a smaller model)?

The verifier teacher-force replays `input_ids + output_ids` and compares its
per-position distribution against the worker's `selected_logprob`, `selected_rank`,
and `top_logprobs`. Rather than a single-token binary test, it uses window-level
robust aggregation (`mean/p95/p99 abs_logprob_diff`, `rank_delta_rate`, top-*k*
Jaccard, union-JS) — motivated by the finding that same-model BF16 diffs appear as
sparse, position-sensitive token spikes, while FP8 adds a broader distribution
shift on top. Three reports cover this track:

- [`LLM-QWen3_8b_bf16_single_sample_verifier_params.md`](experiments/logprobs/reports/LLM-QWen3_8b_bf16_single_sample_verifier_params.md)
  — a tri-state (`PASS` / `REJECT` / `INCONCLUSIVE`) single-sample verifier design
  for dense Qwen3-8B BF16.
- [`qwen3_6_35b_a3b_6000ws_verifier_testing_report.md`](experiments/logprobs/reports/qwen3_6_35b_a3b_6000ws_verifier_testing_report.md)
  — Qwen3.6-35B-A3B BF16-vs-FP8 logprob comparison (FP8 shows a systematic ~1.4×–1.6×
  bulk-to-p99 shift and worse top-*k* overlap).
- [`moe_replicates_bf16_vs_fp8_spike_testing_report.md`](experiments/logprobs/reports/moe_replicates_bf16_vs_fp8_spike_testing_report.md)
  — the spike hypothesis test across BF16 replicates justifying window-level median
  aggregation over single-token thresholds.

**Finding:** For dense models the same-model BF16 baseline is narrow and stable while
FP8/AWQ/smaller models separate sharply; for MoE the same-model baseline itself has a
heavy tail, so window-aggregated metrics (not single-token max) must drive the decision.

### 3. Worker-vs-verifier cost difference (`experiments/worker-verifier-latency/`)

**Question:** Is verification actually cheaper than generation, and by how much?

A vLLM offline benchmark (`benchmark_worker_verifier_latency.py`, Qwen3-8B BF16 on a
single RTX 4090) times both paths on one shared `LLM` instance across an input×output
length grid: the **worker** does normal `prefill + serial decode`, the **verifier**
does a single parallel prefill with `prompt_logprobs=20`. Raw data and summaries live
under [`data/qwen3_8b/`](experiments/worker-verifier-latency/data/qwen3_8b/); the
analysis is [`reports/worker_verifier_latency_benchmark_report.md`](experiments/worker-verifier-latency/reports/worker_verifier_latency_benchmark_report.md).

**Finding:** Worker cost ≈ `output_len × 17.5 ms` (serial, memory-bandwidth-bound);
verifier cost ≈ `total_tokens / ~5200 tok/s` (one parallel prefill). For typical
multi-token outputs the verifier costs about **1%** of a generation
(worker/verifier ≈ **80×–114×**, peaking at output 1024–2048). The only exception is
`output_len = 1`, where the verifier's fixed per-position logprob overhead makes it
slightly slower. Three verifiers together therefore cost roughly **1.9%–14.8%** of a
single generation.

## Key findings

- The same-model BF16 replay baseline is narrow and stable across GPUs
  (6000ws / H100 / 4090 / L4), while FP8, AWQ, and heterogeneous smaller models
  separate sharply on p99, rank-delta rate, top-*k* Jaccard, and union-JS.
- For MoE, per-token logprob is confounded by sparse heavy-tailed spikes, but the
  layer-0 routed-experts fingerprint forms a structurally stable, cross-GPU
  reproducible gap (a BF16 verifier attains TPR = 1.000 with 0.000 false-pass at
  threshold 0.060 under a layer-0 random-5 aggregate).
- A single verification prefill costs about **1%** of one generation
  (worker/verifier ≈ 80×–114×), so three verifiers together consume roughly
  **1.9%–14.8%** of the GPU time of a single generation.

## Getting started

Each tool directory carries its own `README.md` with concrete run commands and
dependencies. The vLLM-based stages generally require `transformers`, `vllm`, and
a GPU runtime:

```bash
python3 -m pip install transformers vllm
```
