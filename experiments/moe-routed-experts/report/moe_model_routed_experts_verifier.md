# MoE Routed Experts Verifier Design

## TL;DR

Current recommended protocol: use the verifier to prefill `input_token_ids + output_token_ids`,
compare only the routed experts of MoE `layer 0`, and use
`entry_mismatch_rate = sum(entry_mismatch_count) / sum(compared_entries)` as the primary score.
By default, use at least `5` samples with `total_compared_tokens >= 512`; production thresholds
are governed by "Threshold Details". A `registered_logit_tokens_top_k` legality check is also
mandatory: use the verifier prefill logits to check whether each actual output token falls within
the model's registered logit-token top-k candidate set.

## Table of Contents

- [Goal](#goal)
- [Evidence Data Format](#evidence-data-format)
- [vLLM Requirements](#vllm-requirements)
- [Security Boundaries and Mitigations](#security-boundaries-and-mitigations)
- [Comparison Unit](#comparison-unit)
- [Layer Selection](#layer-selection)
- [Primary Metric](#primary-metric)
- [Auxiliary Metrics](#auxiliary-metrics)
- [Sample and Token Requirements](#sample-and-token-requirements)
- [Decision Rule](#decision-rule)
- [Current Experimental Results](#current-experimental-results)
- [Verifier Workflow](#verifier-workflow)
- [Threshold Details](#threshold-details)
- [Multi-Verifier Decision](#multi-verifier-decision)
- [Pre-Run Checks](#pre-run-checks)
- [Command Examples](#command-examples)
- [Conditions for Re-Calibrating Thresholds](#conditions-for-re-calibrating-thresholds)
- [Current Recommended Design](#current-recommended-design)

## Goal

Use the `routed_experts` returned by vLLM as a routing fingerprint for MoE models to verify whether a batch of generations claimed to come from a particular model really does look more like output produced by that model.

The verifier inputs include:

- The original `input_token_ids`
- The claimed `output_token_ids`
- The `routed_experts` returned by the claimed source model during normal decode inference

Then the verifier model runs one prefill teacher-forcing pass over:

```text
input_token_ids + output_token_ids
```

to obtain the verifier's own prefill `routed_experts`, and then aligns and compares them against the decode `routed_experts` in the claimed trace.

The current experiments compare two related Qwen MoE variants:

- `Qwen/Qwen3.6-35B-A3B`, denoted BF16
- `Qwen/Qwen3.6-35B-A3B-FP8`, denoted FP8

When a model is registered, the generation protocol it is permitted to use must also be registered, for example:

```text
registered_logit_tokens_top_k = 20
temperature
top_p
whether greedy is allowed
whether sampling is allowed
```

The verifier then uses the generation protocol from registration time for its checks, and does not accept a worker temporarily declaring or modifying `registered_logit_tokens_top_k` after submitting results.

Throughout the rest of this document, we consistently use two distinct terms:

| Name | Meaning | Current Value |
| --- | --- | ---: |
| `moe_route_experts_top_k` | Number of experts selected per token, per layer during MoE routing | 8 |
| `registered_logit_tokens_top_k` | Number of candidate tokens allowed to be sampled from the logits in the generation protocol | 20 |

Avoid using the bare term `top-k` without a qualifier.
The `N` in `top-N logprobs` refers only to the number of candidate tokens returned by the API; it is neither of the two top-k values above.

The core question is:

```text
If a batch of generations is claimed to come from model M,
and model M itself is used as the verifier, can it distinguish M from other models?
```

To emphasize: this design verifies the compatibility of the routed experts with the verifier model. It does not rigorously prove that the tokens were selected step by step by the verifier model during the decode phase.

## Evidence Data Format

Each piece of generate evidence should contain at least:

- `sample_id`
- `input_token_ids`
- `output_token_ids`
- The `routed_experts` returned during the decode phase
- The model name used for generation
- The generation policy id bound at model registration time
- Route alignment metadata, e.g. `route_prompt_start`

Each verifier/prefill result should contain at least:

- The same `sample_id`
- The same `input_token_ids`
- The same `output_token_ids`
- The `routed_experts` returned during the prefill phase
- The logits, or sufficiently large top-N logprobs, at each output token position during the prefill phase
- The model name used by the verifier
- `routed_experts_prompt_start = len(input_token_ids)`

The prefill request uses teacher-forcing:

```text
prompt = input_token_ids + output_token_ids
max_tokens = 0
echo = true
routed_experts_prompt_start = len(input_token_ids)
```

Do not pass the following during the normal generate phase:

```text
routed_experts_prompt_start = len(input_token_ids)
```

because the prompt for a normal generate consists only of the input part, and some vLLM versions require `routed_experts_prompt_start` to be strictly less than the prompt length, otherwise an assertion like the following may trigger:

```text
assert prompt_start < request.num_prompt_tokens
```

## vLLM Requirements

When starting vLLM, the following must be enabled:

```bash
vllm serve <MODEL> --enable-return-routed-experts
```

HTTP requests must be non-streaming:

```json
{
  "stream": false
}
```

The scripts also depend on:

```json
{
  "return_token_ids": true
}
```

If the registered logit-token top-k legality check is enabled, the verifier prefill must also obtain the logits or top-N logprobs at each output position:

```text
N >= registered_logit_tokens_top_k
```

For example, if the model is registered as:

```text
registered_logit_tokens_top_k = 20
```

then the verifier needs at least the large-model logit-token top-20 logprobs at each output token position in order to determine whether the worker's output token falls within the large model's logit-token top-20.

If an output token does not appear in the returned top-N logprobs, and `N >= registered_logit_tokens_top_k`, then it can be treated as:

```text
rank(output_token) > registered_logit_tokens_top_k
```

unless that position must be marked `uncertain` under the drift boundary rules.

The current verifier protocol does not require trace replay. The current design is:

```text
normal decode generate
        +
prefill replay based on the known output_token_ids
```

## Security Boundaries and Mitigations

This design cannot, on its own, defend against one important form of cheating:

```text
The worker first generates all output tokens with a small model,
then runs prefill over the input + output tokens with the claimed large model,
and submits the routed_experts obtained from the large model's prefill as "legitimate evidence".
```

The reason is that under teacher-forcing, the large model has well-defined hidden states, router logits, and routed experts for any token sequence. In other words, as long as an attacker can access the claimed large model and run prefill, they can fabricate activation/routing evidence for a "small-model-generated token sequence" that looks like it came from the large model.

Therefore, what this design can verify is:

```text
Given input_token_ids + output_token_ids,
whether these routed_experts are consistent with the verifier model's prefill routing.
```

It cannot, on its own, prove:

```text
That these output_token_ids were actually selected by the verifier model during decode.
```

This is the core blind spot of TOPLOC/speculative-decoding-style attacks: activation consistency can only show that "the target model can explain this token string", not that "the target model was responsible for selecting this token string".

So the threat model must distinguish two situations:

| Scenario | Is this design sufficient |
| --- | --- |
| Non-adversarial statistical analysis, comparing the distinguishability of routed experts across different models | Largely sufficient |
| The worker cannot access the verifier model and can only submit existing decode evidence | Useful |
| The worker can access the claimed large model and run prefill offline | Insufficient |
| Need to prove that the tokens were genuinely produced by decode of a particular model | Insufficient, additional mechanisms required |

If the worker can access the claimed large model offline, then a purely post-hoc `input/output/routed_experts` check cannot fully resolve this problem. To defeat the attack, at least one of the following two classes of constraints must be added:

```text
1. Verify the token selection policy
2. Bind the evidence to the online decode process
```

### Mitigation 1: Verify the Token Selection Policy

This is the most readily deployable enhancement under the current vLLM HTTP/prefill system.

In addition to verifying the routed experts, also verify whether the output tokens conform to the claimed model's generation policy.

#### Registered Logit-Token Top-k Legality Check

If, at registration time, the model declares and binds:

```text
registered_logit_tokens_top_k = 20
```

then the verifier can check whether each output token satisfies:

```text
rank_verifier_model(output_token_t | prefix_t) <= registered_logit_tokens_top_k
```

where `rank_verifier_model` is computed from the logits/logprobs obtained after the verifier runs prefill over `input_token_ids + output_token_ids`.

For `registered_logit_tokens_top_k = 20`, each step requires:

```text
output_token_t must be within the verifier model's logit-token top-20 tokens under that prefix
```

If, at some step:

```text
rank_verifier_model(output_token_t | prefix_t) > registered_logit_tokens_top_k
```

and this is not a boundary case caused by numerical drift, then under the registered generation protocol in this example (`registered_logit_tokens_top_k = 20`), this token could not have been sampled by the verifier model and should be rejected outright.

This check can identify one major class of cheating:

```text
A small model first generates an arbitrary token sequence,
then a large model's prefill fills in the routed_experts/logprobs.
```

because a portion of the tokens generated by the small model will typically fall outside the large model's logit-token top-20. The more tokens there are, the higher the probability that such an illegal token appears.

Note: the logit-token top-k legality check is a necessary condition, not a sufficient proof. What it proves is:

```text
This token string does not violate the registered generation candidate token top-k constraint.
```

It cannot, on its own, prove:

```text
That these tokens were necessarily genuinely sampled by the large model according to its probability distribution.
```

If an attacker can access the large model online and use the large model's logit-token top-k results to filter small-model tokens, then they can still construct a sequence in which every step falls within the large model's logit-token top-20. To continue defending against this class of attack, stronger constraints are needed: exact replay with a pre-bound seed/nonce in an identical-hardware, identical-version environment, or online commitment / trusted server-side records. Across GPUs and across vLLM versions, exact replay should not be used as the default pass condition.

Drift-handling recommendation:

```text
rank <= registered_logit_tokens_top_k:
    pass

rank > registered_logit_tokens_top_k
and logprob(logit_tokens_top_k_boundary) - logprob(output_token) is very small:
    uncertain

rank > registered_logit_tokens_top_k
and the margin is clear:
    fail
```

`uncertain` should not be treated directly as pass; increase samples/tokens, or switch to more trustworthy online generation evidence.

For deterministic/greedy decode:

```text
temperature = 0
registered_logit_tokens_top_k = 1
```

Verification method:

```text
Run prefill over input + output_token_ids
Obtain the logits/logprobs at each output position
Check whether output_token_id is the logit top-1 token at the corresponding position
```

Greedy also requires drift boundary handling; do not hard-fail every non-top-1:

```text
output_token == verifier logit top-1:
    pass

output_token != verifier logit top-1
and logprob(top1) - logprob(output_token) is very small:
    uncertain

output_token != verifier logit top-1
and logprob(top1) - logprob(output_token) is clear:
    fail
```

In other words, the greedy check can be seen as the special case of `registered_logit_tokens_top_k = 1`, but it still retains the logic of marking `uncertain` when the margin is very small. Otherwise, decode/prefill drift at low-margin positions could incorrectly reject genuine greedy positives.

If a small model first generates a token string and then a large model prefills in the routed experts, this token string will typically not step-by-step equal the large model's greedy argmax; tokens clearly below the verifier top-1 will be rejected.

For sampling decode:

```text
By default, only verify whether the output token belongs to the registered logit-token top-k candidate set.
```

Verification method:

```text
1. The worker generates the output using the sampling params bound at model registration time
2. The verifier prefills the target model to obtain logits/logprobs at each step
3. The verifier computes the rank of output_token_t under that prefix
4. Require rank <= registered_logit_tokens_top_k
5. Boundary positions where rank is exceeded but the margin is very small are marked uncertain
```

Note:

- In sampling scenarios, exact sampled token replay is not recommended across GPUs, across vLLM versions, or across kernels
- The reason is that even very small logits drift can change the nucleus boundary, candidate probability normalization, or the sampled token
- If top-p is used, a sufficiently complete logits distribution must be obtainable; otherwise it is recommended to disable top-p and use only the registered logit-token top-k
- If the HTTP API can only return a very small `logprobs` top-N, then it is recommended to first restrict to greedy or a fixed logit-token top-k
- With a fixed logit-token top-k, the `N` returned by `logprobs` must cover `registered_logit_tokens_top_k`
- Only when the worker and verifier are forced to use identical hardware, identical vLLM version, identical kernel, and identical sampler implementation, and the seed/nonce is bound before generation, can exact sampled token replay be used as a stronger check
- If the seed is supplied by the worker after the fact, the exact replay check is meaningless

What this mitigation proves is:

```text
The output token sequence does not violate the target model's registered logit-token top-k candidate set constraint
```

It is still not a rigorous proof that the worker actually decoded with the target model at the time, nor a proof that the tokens were genuinely sampled according to the target model's probability distribution; but it will largely defeat the "generate arbitrary tokens with a small model then fill in evidence via large-model prefill" attack.

### Mitigation 2: Bind Evidence Online, Step by Step

If the goal is to prove "attribution of the generation process", then the evidence must be bound to the moment the token is selected.

Basic idea:

```text
For each generated token, or each short span of tokens,
the worker must immediately submit a tamper-proof commitment.
```

A commitment can include:

```text
request_id
verifier nonce
step index
previous token
sampled token
layer0 routed experts hash
logits/logit-token top-k logits hash
sampling params
previous commitment hash
timestamp
```

forming a hash chain:

```text
C_t = H(C_{t-1}, request_id, nonce, step, prev_token, token_t, route_hash_t, logits_hash_t)
```

Finally, the verifier randomly spot-checks some steps and requires the worker to reveal the routes/logits at the corresponding positions.

But note: if this commitment is generated entirely locally by the worker after the fact, it also cannot prevent cheating. It must satisfy at least one of the following conditions:

- The commitment is sent to the verifier online in real time
- The commitment is signed by a trusted service
- The commitment is generated inside a TEE/trusted execution environment
- The commitment has an external timestamp or a non-rollback log

The goal of this mitigation is to prevent the worker from waiting until the full output is known and then batch-filling evidence via large-model prefill.

### Mitigation 3: Trusted Server-Side Records

The strongest and cleanest mitigation is to not accept the routed experts self-reported by the worker.

Instead:

```text
The verifier/platform controls the inference service
The server runs decode directly
The server records routed experts/logits/token directly
The server signs the evidence
```

The worker can only submit evidence that the server has signed.

The signed content binds at least:

```text
model_id
model checkpoint hash
tokenizer hash
request_id
prompt hash
sampling params
seed/nonce
output_token_ids
decode routed experts proof/hash
generation time
```

This mitigation directly prevents "filling in evidence via post-hoc prefill", because legitimate evidence can only be produced during a genuine server-side decode.

### Mitigation 4: TEE or Remote Auditable Inference

If a third-party worker must run the model themselves, a trusted execution environment or a remote auditable service can be used.

The TEE must attest:

```text
That the loaded model is the specified one
That the running inference code is the specified one
That the sampling parameters were not altered
That the evidence was produced within the decode path
That the signing key is only available inside the trusted environment
```

This class of mitigation has the highest engineering cost, but in a decentralized-worker scenario it is the direction closest to "proving genuine execution".

## Comparison Unit

The shape of `routed_experts` is:

```text
[token_position, moe_layer, moe_route_expert_slot]
```

In the current Qwen3.6-35B-A3B data:

```text
moe layers = 40
moe_route_experts_top_k = 8
```

The route length returned during the generate phase is typically about:

```text
input_tokens + output_tokens - 1
```

The prefill phase sets:

```text
routed_experts_prompt_start = len(input_token_ids)
```

and then returns the route window corresponding to the output suffix.

Therefore the compare script first aligns the overlapping token window between the generate route and the prefill route, and then computes the score.

### Route Window Alignment Example

Assume:

```text
input_token_ids  = [I0, I1, I2]          # input_len = 3
output_token_ids = [O0, O1, O2, O3]      # output_len = 4
full sequence    = [I0, I1, I2, O0, O1, O2, O3]
global position  =  0   1   2   3   4   5   6
```

During the normal generate phase, the prompt consists only of the input, and the routed experts returned by vLLM typically cover:

```text
global position: 0   1   2   3   4   5
token:           I0  I1  I2  O0  O1  O2
route index:     0   1   2   3   4   5
route_prompt_start = 0
route length = input_len + output_len - 1 = 6
```

That is, the next-step route corresponding to the last output token `O3` is typically not in the generate return window.

The prefill verifier phase uses the full prompt:

```text
prompt = [I0, I1, I2, O0, O1, O2, O3]
routed_experts_prompt_start = len(input_token_ids) = 3
```

Therefore the prefill routed experts return the output suffix:

```text
global position: 3   4   5   6
token:           O0  O1  O2  O3
route index:     0   1   2   3
route_prompt_start = 3
route length = output_len = 4
```

The global windows for generate and prefill are, respectively:

```text
generate window: [0, 6)   # 0,1,2,3,4,5
prefill window:  [3, 7)   # 3,4,5,6
overlap window:  [3, 6)   # 3,4,5
```

Ultimately only the overlap window is compared:

```text
global position:      3   4   5
token:                O0  O1  O2
generate route index: 3   4   5
prefill route index:  0   1   2
```

So in this example:

```text
compared_tokens = output_len - 1 = 3
```

Looking only at layer 0 with `moe_route_experts_top_k = 8`:

```text
compared_entries = compared_tokens * 1 * 8 = 24
```

This is also why the raw shapes are often different, but the aligned shape can be correct.

Note:

```text
A non-zero raw_shape_mismatch_sample_count is normal
A non-zero shape_mismatch_sample_count is what warrants attention
```

The reason is that the raw route windows returned by generate and prefill may naturally differ in length, but the aligned comparison window should be reasonable.

## Layer Selection

The production verifier score uses only MoE `layer 0`.

Reasons:

- The FP8 verifier's distinguishability weakens noticeably when averaging over all 40 layers
- Later layers accumulate numerical differences between decode and prefill
- The `moe_route_experts_top_k` of MoE routing is a hard discrete selection, and even tiny hidden state differences can flip an expert slot
- The same-model routing signal at layer 0 is the cleanest

The FP8 verifier's distinguishability is weak when using all layers:

```text
FP8 verifier, all 40 layers:
positive fp8_vs_fp8 entry mismatch ~= 0.2725
negative bf16_vs_fp8 entry mismatch ~= 0.2922
gap ~= 0.0197
```

Looking only at layer 0, the FP8 verifier's distinguishability is markedly stronger:

```text
FP8 verifier, layer 0:
positive fp8_vs_fp8 entry mismatch ~= 0.0491
negative bf16_vs_fp8 entry mismatch ~= 0.1131
gap ~= 0.0640
```

The BF16 verifier is also very strong at layer 0:

```text
BF16 verifier, layer 0:
positive bf16_vs_bf16 entry mismatch ~= 0.0294
negative fp8_vs_bf16 entry mismatch ~= 0.1147
gap ~= 0.0853
```

So the current unified convention is:

```text
Compare only layer 0
```

## Primary Metric

The primary metric uses the `entry_mismatch_rate` of layer 0:

```text
score =
  sum(entry_mismatch_count)
  / sum(compared_entries)
```

For the case of layer 0 with `moe_route_experts_top_k = 8`:

```text
compared_entries = compared_output_tokens * moe_route_experts_top_k
```

This metric is order-sensitive; it compares item by item every:

```text
[token, layer, moe_route_expert_slot]
```

that is:

```text
generate[token, 0, moe_route_expert_slot] == prefill[token, 0, moe_route_expert_slot]
```

The lower the score, the more the claimed trace looks like it was produced by the verifier model itself.

Do not use `token_layer_set_match_rate` as the primary metric. It ignores the MoE route expert slot/order, dropping part of the model identity signal, and its discriminative power is noticeably weaker especially on the FP8 verifier.

## Auxiliary Metrics

The `token_layer_match_rate` of layer 0 can also be reported as an auxiliary diagnostic:

```text
token_layer_match_rate =
  number of tokens whose full `moe_route_experts_top_k` vector matches exactly
  / total number of compared tokens
```

It is stricter than `entry_mismatch_rate`, but not as smooth as `entry_mismatch_rate`.

Recommendation:

```text
Use the weighted entry_mismatch_rate for the final decision
Look at token_layer_match_rate alongside it for debugging and interpretation
```

## Sample and Token Requirements

Multi-sample aggregation is required, and it must be weighted by entry; you cannot simply average the per-sample rates.

Minimum requirements:

```text
sample_count >= 5
total_compared_tokens >= 512
```

Recommended defaults:

```text
sample_count = 5
target total_compared_tokens ~= 600+
```

More conservative settings:

```text
sample_count >= 8
total_compared_tokens >= 768
```

Why both a sample count and a token count are required:

- Different samples may have different output token counts
- A sample with especially short output makes the single-sample score very unstable
- In the current data, the median effective compared tokens per sample is about 127
- Therefore 5 samples usually equal roughly 635 effective compared tokens
- If future data produces shorter outputs, keep sampling until the token count requirement is met

## Decision Rule

For each verifier model, compute:

```text
group_score =
  sum(layer0_entry_mismatch_count across selected samples)
  / sum(layer0_compared_entries across selected samples)
```

then apply a verifier-specific threshold. The production decision threshold has "Threshold Details" as its sole authoritative source; the threshold column in the experimental results tables is a historical snapshot and is not a configuration source.

General decision:

```text
group_score <= verifier_threshold: accept
group_score > verifier_threshold: reject
```

The FP8 verifier's production default should prioritize controlling false accepts; if a more recall-oriented threshold is needed, it must be re-evaluated in conjunction with the cross-GPU negative tail.

## Current Experimental Results

Layer 0 single-sample score distribution:

| case | n | mean | p05 | p50 | p95 | max | token p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 verifier positive | 1000 | 0.029385 | 0.016683 | 0.028543 | 0.045276 | 0.101378 | 127 |
| BF16 verifier negative | 1000 | 0.114704 | 0.085630 | 0.109252 | 0.139764 | 0.416667 | 127 |
| FP8 verifier positive | 1000 | 0.049098 | 0.032480 | 0.047244 | 0.066929 | 0.171260 | 127 |
| FP8 verifier negative | 1000 | 0.113092 | 0.086614 | 0.109252 | 0.142717 | 0.336614 | 127 |

Single-sample threshold check; the following is an experimental snapshot:

| verifier | source | threshold | pass count | pass rate |
| --- | --- | ---: | ---: | ---: |
| BF16 | BF16 | 0.066 | 995 / 1000 | 99.5% |
| BF16 | FP8 | 0.066 | 2 / 1000 | 0.2% false accept |
| FP8 | FP8 | 0.074 | 976 / 1000 | 97.6% |
| FP8 | BF16 | 0.074 | 9 / 1000 | 0.9% false accept |

A single sample already has distinguishability, but the final verifier decision is not recommended to use a single sample; multi-sample aggregation should be used.

## Random 5-Sample Results

Protocol:

```text
layer = 0
sample_count = 5
score = sum(entry_mismatch_count) / sum(compared_entries)
```

The tables below are experimental snapshots computed from the candidate thresholds at that time, and may be out of sync with future production thresholds;
production thresholds are governed by "Threshold Details".

Original A6000-style report results:

| verifier | threshold | positive pass rate | false reject | negative false pass | token p50 | token p05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.060 | 99.9992% | 0.0008% | 0.0000% | 635 | 541 |
| FP8 | 0.080 | 99.8320% | 0.1680% | 0.0000% | 635 | 519 |
| FP8 | 0.085 | 99.9335% | 0.0665% | 0.0027% | 635 | 519 |

Results after merging cross-GPU reports:

| verifier | threshold | positive pass TPR | false reject FNR | negative false pass FPR |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 0.060 | 100.0000% | 0.0000% | 0.0000% |
| FP8 | 0.080 | 99.9453% | 0.0547% | 0.0000% |
| FP8 | 0.085 | 99.9850% | 0.0150% | 0.0047% |

The current cross-GPU results do not overturn the existing thresholds.

## Cross-GPU Result Interpretation

The newly added cross-GPU report shows that the layer 0 score distribution remains stable.

BF16 positive layer 0 mean:

```text
6000_bf16_vs_bf16       ~= 0.0294
6000_bf16_vs_h100_bf16  ~= 0.0242
h100_bf16_vs_bf16       ~= 0.0241
```

FP8 positive layer 0 mean:

```text
6000_fp8_vs_fp8         ~= 0.0491
6000_fp8_vs_h100_fp8    ~= 0.0481
h100_fp8_vs_fp8         ~= 0.0475
```

The negative layer 0 mean still lies roughly at:

```text
0.112 - 0.117
```

This indicates that the current layer 0 score is fairly stable against the GPU differences tested so far.

## Verifier Workflow

For a claimed source model `M`:

1. Collect at least 5 samples from the claimed generation trace.
2. Ensure the total effective compared output tokens is at least 512.
3. Use the verifier model `M` to prefill each `input_token_ids + output_token_ids`.
4. Compare only the routed experts of layer 0.
5. Compute the weighted `entry_mismatch_rate`.
6. Read `registered_logit_tokens_top_k` from the model registration info.
7. Use the verifier prefill logits/logprobs to check whether each output token falls within the verifier model's registered logit-token top-k.
8. Accept only when both the routed experts score and the logit-token top-k legality pass; otherwise reject or mark uncertain.

Pseudocode:

```python
def verify(samples, verifier_model):
    total_mismatch = 0
    total_entries = 0
    total_tokens = 0
    illegal_token_count = 0
    uncertain_token_count = 0
    registered_logit_tokens_top_k = get_registered_logit_tokens_top_k(verifier_model)

    for sample in samples:
        prefill_result = run_prefill_verifier(
            model=verifier_model,
            input_token_ids=sample.input_token_ids,
            output_token_ids=sample.output_token_ids,
        )
        gen_layer0, prefill_layer0 = align_layer0(
            sample.decode_routes,
            prefill_result.routed_experts,
        )
        total_mismatch += count_entries_not_equal(gen_layer0, prefill_layer0)
        total_entries += count_compared_entries(gen_layer0, prefill_layer0)
        total_tokens += count_compared_tokens(gen_layer0, prefill_layer0)

        legality = check_output_tokens_in_registered_logit_tokens_topk(
            output_token_ids=sample.output_token_ids,
            verifier_logits=prefill_result.logits,
            registered_logit_tokens_top_k=registered_logit_tokens_top_k,
        )
        illegal_token_count += legality.illegal_token_count
        uncertain_token_count += legality.uncertain_token_count

    if len(samples) < 5:
        return "insufficient_samples"
    if total_tokens < 512:
        return "insufficient_tokens"
    if illegal_token_count > 0:
        return "reject"
    if uncertain_token_count > 0:
        return "uncertain"

    score = total_mismatch / total_entries
    threshold = threshold_for(verifier_model)

    return "accept" if score <= threshold else "reject"
```

## Threshold Details

Current default thresholds:

| verifier model | threshold | decision |
| --- | ---: | --- |
| BF16 | 0.060 | accept when score <= 0.060 |
| FP8 | 0.080 | accept when score <= 0.080 |

Thresholds for single-sample debugging only:

| verifier model | single-sample threshold |
| --- | ---: |
| BF16 | 0.066 |
| FP8 | 0.074 |

Single-sample thresholds are not recommended for the final verifier decision.

## Multi-Verifier Decision

If the same claim was run through both the BF16 verifier and the FP8 verifier:

1. Compute the BF16 verifier score.
2. Compute the FP8 verifier score.
3. Apply each of their respective thresholds.

Decision table:

| BF16 result | FP8 result | final decision |
| --- | --- | --- |
| accept | reject | BF16 |
| reject | accept | FP8 |
| reject | reject | unknown or neither |
| accept | accept | ambiguous, need more samples |

For ambiguous:

- Increase to at least 10 samples
- Require at least 1024 total compared tokens
- Still use only layer 0
- Compare the relative threshold margins:

```text
bf16_margin = bf16_threshold - bf16_score
fp8_margin = fp8_threshold - fp8_score
```

If only one verifier clearly exceeds the threshold, you can choose the model with the larger margin.  
If both verifiers clearly pass, it should be marked ambiguous and the test sample set expanded.

## Pre-Run Checks

Before trusting a single verifier result, check at least:

- `shape_mismatch_sample_count` should be 0
- `expected_length_mismatch_sample_count` should be 0
- `layer_topk_shape_mismatch_sample_count` should be 0. The `topk` in this legacy field name refers to the MoE routing expert top-k, i.e. `moe_route_experts_top_k`
- `selected_layers` should be `0`
- `total_compared_tokens` should be at least 512
- `registered_logit_tokens_top_k` must exist in the model registration info
- The logits/top-N logprobs returned by the verifier prefill must be able to cover `registered_logit_tokens_top_k`
- The output tokens must pass the registered logit-token top-k legality check
- All samples should use the same tokenizer/model family
- HTTP requests should be non-streaming

`raw_shape_mismatch_sample_count` may be non-zero; this is expected behavior, because the raw route windows returned by generate and prefill may differ in length.

## Command Examples

The script is located at `tools/moe_routed_experts/scripts/compare_routes.py`; comparison results are output to
`experiments/moe-routed-experts/data/<variant>/`. The raw route dumps fed to `--generated` / `--prefill`
(`generated_routes_*.jsonl` / `prefill_routes_*.jsonl`) are produced by
`generate_decode_routes.py` and `prefill_routes.py`, are intermediate artifacts, and are not shipped with the repo;
they are represented below with `<...>` placeholders. All commands below are executed from the repository root.

Compare only layer 0:

```bash
python3 tools/moe_routed_experts/scripts/compare_routes.py \
  --generated <generated_routes.jsonl> \
  --prefill <prefill_routes.jsonl> \
  --output-dir <report_dir> \
  --layers 0
```

FP8 positive (output corresponds to `experiments/moe-routed-experts/data/6000_fp8_vs_fp8_layers_0/`):

```bash
python3 tools/moe_routed_experts/scripts/compare_routes.py \
  --generated <generated_routes_6000_fp8.jsonl> \
  --prefill <prefill_routes_6000_fp8_vs_fp8.jsonl> \
  --output-dir experiments/moe-routed-experts/data/6000_fp8_vs_fp8_layers_0 \
  --layers 0
```

FP8 negative (output corresponds to `experiments/moe-routed-experts/data/6000_bf16_vs_fp8_layers_0/`):

```bash
python3 tools/moe_routed_experts/scripts/compare_routes.py \
  --generated <generated_routes_6000_bf16.jsonl> \
  --prefill <prefill_routes_6000_bf16_vs_fp8.jsonl> \
  --output-dir experiments/moe-routed-experts/data/6000_bf16_vs_fp8_layers_0 \
  --layers 0
```

## Conditions for Re-Calibrating Thresholds

Re-calibration of the thresholds is recommended when any of the following factors change:

- vLLM version
- CUDA/Triton kernel
- GPU architecture
- Quantization recipe
- Model checkpoint
- tokenizer
- prompt distribution
- max output length
- routing output format

The re-calibration procedure for each verifier model:

1. Collect same-model positive traces.
2. Collect negative traces from all possible competing models.
3. Use only layer 0.
4. Compute the random 5-sample grouped weighted score.
5. Prioritize choosing a threshold that controls false accepts.

Recommended target:

```text
FPR <= 0.1% on random 5-sample groups
```

If the verifier is used in a stricter scenario:

```text
sample_count >= 10
total_compared_tokens >= 1024
```

## Current Recommended Design

Only this section is retained as the final deployment convention. The verifier is recommended to use two layers of checks:

1. Route compatibility check: compare only the routed experts of layer 0, and compute
   `route_score = sum(entry_mismatch_count) / sum(compared_entries)`.
2. Logit-token legality check: use the verifier prefill logits to compute the rank of the actual
   `output_token`, and check whether it conforms to the model's registered
   `registered_logit_tokens_top_k`.

Fixed parameters:

```text
selected_layers = [0]
sample_count >= 5
total_compared_tokens >= 512

route_score threshold = use the verifier-specific threshold in "Threshold Details"

registered_logit_tokens_top_k = provided at model registration time, currently recommended to be 20
```

Decision convention:

- When `route_score` is above the corresponding verifier threshold, judge as `reject`
- When the verifier-prefill rank of the `output_token` exceeds `registered_logit_tokens_top_k`,
  and this is not a numerical boundary drift, judge as `reject`
- Greedy is the special case of `registered_logit_tokens_top_k = 1`, but low-margin
  positions should still be marked `uncertain` rather than hard-rejected
- Sampling by default only performs logit-token top-k set membership checks; only with identical hardware, identical vLLM
  version, identical kernel, and identical sampler, and with the seed/nonce bound before generation, is exact
  sampled-token replay used as a stronger check
- Judge as `accept` only when both route compatibility and logit-token legality pass, and there are no `uncertain` tokens

This design is supported jointly by the current original experiments and the cross-GPU experiments.
