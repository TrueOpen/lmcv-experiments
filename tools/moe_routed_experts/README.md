# MoE Routed Experts Test

This folder tests whether vLLM decode-time `routed_experts` and prefill-time
`routed_experts` agree for the same `input + output_token_ids`.

Start vLLM with routed experts enabled:

```bash
vllm serve <MODEL> --enable-return-routed-experts
```

Only this flag is required for these scripts:

```text
--enable-return-routed-experts
```

These scripts do not use trace replay, so this flag is not required:

```text
--enable-trace-replay
```

The HTTP requests also use these vLLM extra parameters:

```json
{
  "return_token_ids": true
}
```

`return_token_ids` lets the generation step save the exact output token ids.
The prefill step additionally sends:

```json
{
  "routed_experts_prompt_start": "len(input_token_ids)"
}
```

`routed_experts_prompt_start` asks vLLM to skip early prompt route entries. vLLM
requires this value to be strictly smaller than the prompt length. During prefill
the prompt is `input_token_ids + output_token_ids`, so `len(input_token_ids)` is
valid when the output is non-empty.

Do not set `routed_experts_prompt_start` to exactly `len(input_token_ids)` during
normal generation, because the generation prompt contains only the input tokens
and some vLLM versions hit this assertion:

```text
assert prompt_start < request.num_prompt_tokens
```

If your vLLM build rejects either field, check the installed vLLM version and the
OpenAI-compatible server's extra parameter support.

Install the tiny local dependency used to decode base64 `.npy` route payloads:

```bash
cd tools/moe_routed_experts
python3 -m pip install -r requirements.txt
```

For the cleanest comparison, avoid prefix-cache reuse while debugging. Some vLLM
versions can return `-1` sentinel values for cached prompt routes; the compare
script ignores `-1` by default.

If same-model prefill comparison unexpectedly returns many `-1` entries or a
large route mismatch, rerun once with prefix caching disabled. The exact CLI
spelling can vary by vLLM version, so confirm it with:

```bash
vllm serve --help | grep -i prefix
```

## Input Format

The default input is the existing `worker_logits_verify` prompt file:

```text
../data/qwen3_32b_identity_v1/prompts.jsonl
```

Expected row shape:

```json
{"sample_id":"7a613643598c07a5fd54","input_text":"...","category":"math_numeric","target_input_tokens":32,"metadata":{},"tokenizations":{}}
```

The scripts use `input_text` as the prompt and preserve `category`,
`target_input_tokens`, `anchor_variant`, `metadata`, and `tokenizations` in the
generated evidence rows.

## Command 1: Normal Prompt Generation

Generate output text, output token ids, and decode routed experts:

```bash
cd tools/moe_routed_experts
make generate MODEL=<MODEL> BASE_URL=http://localhost:8000
```

To test only the first `N` prompt rows:

```bash
make generate MODEL=<MODEL> BASE_URL=http://localhost:8000 LIMIT=10
```

Equivalent direct command:

```bash
python3 scripts/generate_decode_routes.py \
  --base-url http://localhost:8000 \
  --model <MODEL> \
  --prompts-file ../data/qwen3_32b_identity_v1/prompts.jsonl \
  --output data/generated_routes.jsonl \
  --max-tokens 128 \
  --temperature 0 \
  --seed 0 \
  --limit 10
```

Output:

```text
data/generated_routes.jsonl
```

Each row contains `input_token_ids`, `output_token_ids`, `output_text`, and
`routed_experts`. The generation request does not pass
`routed_experts_prompt_start`; it records the route window returned by vLLM and
saves `route_prompt_start = 0` for alignment.

## Command 2: Prefill Routes From Generated Output Tokens

Use the generated `output_token_ids` as the suffix of a full prompt and collect
prefill routed experts:

```bash
cd tools/moe_routed_experts
make prefill MODEL=<MODEL> BASE_URL=http://localhost:8000
```

Equivalent direct command:

```bash
python3 scripts/prefill_routes.py \
  --base-url http://localhost:8000 \
  --model <MODEL> \
  --input data/generated_routes.jsonl \
  --output data/prefill_routes.jsonl
```

Internally this sends:

```text
prompt = input_token_ids + output_token_ids
max_tokens = 0
echo = true
routed_experts_prompt_start = len(input_token_ids)
```

Output:

```text
data/prefill_routes.jsonl
```

## Command 3: Compare Decode Routes vs Prefill Routes

Compute route mismatch statistics:

```bash
cd tools/moe_routed_experts
make compare
```

Equivalent direct command:

```bash
python3 scripts/compare_routes.py \
  --generated data/generated_routes.jsonl \
  --prefill data/prefill_routes.jsonl \
  --output-dir reports
```

To compare only selected MoE layers, add `--layers`. The value can be a single
layer, a range, or a comma-separated mix:

```bash
python3 scripts/compare_routes.py \
  --generated data/generated_routes.jsonl \
  --prefill data/prefill_routes.jsonl \
  --output-dir reports/layers_0_4 \
  --layers 0-4
```

Examples:

```text
--layers 0
--layers 0-4
--layers 0-4,7,12-15
```

Outputs:

```text
reports/route_diff_by_sample.csv
reports/route_diff_by_sample_layer.csv
reports/route_diff_by_layer.csv
reports/route_diff_summary.json
reports/route_diff_summary.md
```

Key metrics:

- `entry_mismatch_rate`: order-sensitive mismatch over token, MoE layer, top-k slot.
- `token_layer_set_match_rate`: order-insensitive exact top-k expert set match for each token/layer.
- `token_match_rate`: fraction of output route positions whose full layer/top-k route matches.
- `topk_overlap_mean`: order-insensitive top-k expert overlap per token/layer.
- `topk_jaccard_mean`: order-insensitive Jaccard similarity for top-k expert sets.
- `gen_tail_ignored` and `prefill_tail_ignored`: suffix lengths ignored because decode and prefill can expose one different final route position.
- `shape_mismatch_sample_count`: samples whose aligned route shape does not match the shape inferred from `input_token_ids` and `output_token_ids`.
- `raw_shape_mismatch_sample_count`: samples whose raw generation and prefill route arrays have different shapes. This can be expected because generation usually returns `input + output - 1` positions while prefill returns the output suffix.

`route_diff_by_sample.csv` is the sample-level report. It includes category,
input/output token counts, route alignment offsets, and every metric above for
each `sample_id`. The JSON/Markdown summary also includes sample-level
mean/min/p50/p90/max so one bad sample is not hidden by a global weighted average.

`route_diff_by_layer.csv` aggregates the same metrics by MoE layer index, which
is the most direct report for checking whether the same token's per-layer top-k
experts differ between normal generation and prefill.

## One-Vs-Rest Verification

The final identity test should generate worker traces from every candidate
source model, then prefill those same `input + output_token_ids` sequences with
each verifier model.

Example with `M1` and `M2`:

```bash
# 1. Generate claimed-source traces.
python3 scripts/generate_decode_routes.py \
  --base-url http://localhost:8000 \
  --model M1 \
  --prompts-file ../data/qwen3.6_35b_a3b_identity_v1/prompts.jsonl \
  --limit 5 \
  --output data/generated_routes_m1.jsonl \
  --max-tokens 128 \
  --temperature 0

python3 scripts/generate_decode_routes.py \
  --base-url http://localhost:8000 \
  --model M2 \
  --prompts-file ../data/qwen3.6_35b_a3b_identity_v1/prompts.jsonl \
  --limit 5 \
  --output data/generated_routes_m2.jsonl \
  --max-tokens 128 \
  --temperature 0

# 2. Use M1 as verifier against M1 and non-M1 outputs.
python3 scripts/prefill_routes.py \
  --base-url http://localhost:8000 \
  --model M1 \
  --input data/generated_routes_m1.jsonl \
  --output data/prefill_m1_generated_by_m1.jsonl

python3 scripts/prefill_routes.py \
  --base-url http://localhost:8000 \
  --model M1 \
  --input data/generated_routes_m2.jsonl \
  --output data/prefill_m1_generated_by_m2.jsonl

# 3. Use M2 as verifier against M2 and non-M2 outputs.
python3 scripts/prefill_routes.py \
  --base-url http://localhost:8000 \
  --model M2 \
  --input data/generated_routes_m2.jsonl \
  --output data/prefill_m2_generated_by_m2.jsonl

python3 scripts/prefill_routes.py \
  --base-url http://localhost:8000 \
  --model M2 \
  --input data/generated_routes_m1.jsonl \
  --output data/prefill_m2_generated_by_m1.jsonl
```

Then compare the positive and negative baselines:

```bash
# Positive for M1.
python3 scripts/compare_routes.py \
  --generated data/generated_routes_m1.jsonl \
  --prefill data/prefill_m1_generated_by_m1.jsonl \
  --output-dir reports/m1_positive

# Negative for M1.
python3 scripts/compare_routes.py \
  --generated data/generated_routes_m2.jsonl \
  --prefill data/prefill_m1_generated_by_m2.jsonl \
  --output-dir reports/m1_negative_m2

# Positive for M2.
python3 scripts/compare_routes.py \
  --generated data/generated_routes_m2.jsonl \
  --prefill data/prefill_m2_generated_by_m2.jsonl \
  --output-dir reports/m2_positive

# Negative for M2.
python3 scripts/compare_routes.py \
  --generated data/generated_routes_m1.jsonl \
  --prefill data/prefill_m2_generated_by_m1.jsonl \
  --output-dir reports/m2_negative_m1
```

For each verifier, the positive distribution should separate from the negative
distribution. For example, M1 is useful as a verifier only if `reports/m1_positive`
has much higher `topk_overlap_mean` / `token_layer_set_match_rate` and much lower
`entry_mismatch_rate` than `reports/m1_negative_m2`.

## Notes

- This path does not use trace replay. It compares normal decode generation
  against a teacher-forcing prefill pass.
- If same-model mismatch is high, first check token alignment, prefix cache,
  vLLM version, MoE kernel differences, and whether the server actually returns
  output-only routes after `routed_experts_prompt_start`.
