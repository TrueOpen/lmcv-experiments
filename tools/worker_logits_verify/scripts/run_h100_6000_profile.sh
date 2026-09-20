#!/usr/bin/env bash
set -euo pipefail

# Edit these paths for your machines. Run the collection commands on the matching GPU host,
# then copy trace directories to one analysis host before compare/profile steps.

MODEL_32B_BF16="${MODEL_32B_BF16:-/models/Qwen3-32B}"
MODEL_32B_FP8="${MODEL_32B_FP8:-/models/Qwen3-32B-FP8}"
MODEL_32B_AWQ="${MODEL_32B_AWQ:-/models/Qwen3-32B-AWQ}"
MODEL_14B="${MODEL_14B:-/models/Qwen3-14B}"
MODEL_8B="${MODEL_8B:-/models/Qwen3-8B}"

DATA_ROOT="${DATA_ROOT:-data/qwen3_32b_identity_v1}"
RESULT_ROOT="${RESULT_ROOT:-results/qwen3_32b_identity_v1}"
PYTHONPATH="${PYTHONPATH:-src}"

mkdir -p "${DATA_ROOT}" "${RESULT_ROOT}"

worker-logits-generate-prompts \
  --model "${MODEL_32B_BF16}" \
  --buckets "${BUCKETS:-32,128,512,2048,8192,16384}" \
  --per-bucket "${PER_BUCKET:-50}" \
  --output "${DATA_ROOT}/prompts.jsonl" \
  --trust-remote-code

worker-logits-generate-anchor \
  --model "${MODEL_32B_BF16}" \
  --variant-id qwen3-32b-bf16-h100 \
  --gpu H100 \
  --quantization bf16 \
  --dtype bfloat16 \
  --input "${DATA_ROOT}/prompts.jsonl" \
  --output "${DATA_ROOT}/anchor_samples.jsonl" \
  --max-new-tokens "${MAX_NEW_TOKENS:-128}" \
  --trust-remote-code

echo
echo "Now collect traces on each target host. Examples:"
echo
cat <<EOF
worker-logits-collect-trace \\
  --samples ${DATA_ROOT}/anchor_samples.jsonl \\
  --model ${MODEL_32B_BF16} \\
  --variant-id qwen3-32b-bf16-h100 \\
  --gpu H100 \\
  --quantization bf16 \\
  --dtype bfloat16 \\
  --output-dir ${RESULT_ROOT}/traces/qwen3-32b-bf16-h100 \\
  --top-k 64 \\
  --trust-remote-code

worker-logits-collect-trace \\
  --samples ${DATA_ROOT}/anchor_samples.jsonl \\
  --model ${MODEL_32B_BF16} \\
  --variant-id qwen3-32b-bf16-6000 \\
  --gpu RTXPro6000 \\
  --quantization bf16 \\
  --dtype bfloat16 \\
  --output-dir ${RESULT_ROOT}/traces/qwen3-32b-bf16-6000 \\
  --top-k 64 \\
  --trust-remote-code

# Negative controls, run where they fit:
worker-logits-collect-trace --samples ${DATA_ROOT}/anchor_samples.jsonl --model ${MODEL_32B_FP8} --variant-id qwen3-32b-fp8-h100 --gpu H100 --quantization fp8 --dtype auto --output-dir ${RESULT_ROOT}/traces/qwen3-32b-fp8-h100 --top-k 64 --trust-remote-code --allow-retokenize
worker-logits-collect-trace --samples ${DATA_ROOT}/anchor_samples.jsonl --model ${MODEL_32B_AWQ} --variant-id qwen3-32b-awq-h100 --gpu H100 --quantization awq --dtype auto --output-dir ${RESULT_ROOT}/traces/qwen3-32b-awq-h100 --top-k 64 --trust-remote-code --allow-retokenize
worker-logits-collect-trace --samples ${DATA_ROOT}/anchor_samples.jsonl --model ${MODEL_14B} --variant-id qwen3-14b-bf16-h100 --gpu H100 --quantization bf16 --dtype bfloat16 --output-dir ${RESULT_ROOT}/traces/qwen3-14b-bf16-h100 --top-k 64 --trust-remote-code --allow-retokenize
worker-logits-collect-trace --samples ${DATA_ROOT}/anchor_samples.jsonl --model ${MODEL_8B} --variant-id qwen3-8b-bf16-h100 --gpu H100 --quantization bf16 --dtype bfloat16 --output-dir ${RESULT_ROOT}/traces/qwen3-8b-bf16-h100 --top-k 64 --trust-remote-code --allow-retokenize
EOF

echo
echo "After traces are available on this host, run:"
echo
cat <<EOF
worker-logits-compare-traces \\
  --left ${RESULT_ROOT}/traces/qwen3-32b-bf16-h100 \\
  --right ${RESULT_ROOT}/traces/qwen3-32b-bf16-6000 \\
  --left-label H100-BF16 \\
  --right-label 6000-BF16 \\
  --output-dir ${RESULT_ROOT}/comparisons/h100_vs_6000_bf16

worker-logits-analyze-depth-effect \\
  --comparison ${RESULT_ROOT}/comparisons/h100_vs_6000_bf16 \\
  --output-dir ${RESULT_ROOT}/depth_effect/h100_vs_6000_bf16

worker-logits-calibrate-profile \\
  --comparison ${RESULT_ROOT}/comparisons/h100_vs_6000_bf16 \\
  --profile-id qwen3-32b-bf16-h100-6000-envelope \\
  --variant-pair "Qwen3-32B BF16 H100 vs RTXPro6000" \\
  --output-dir ${RESULT_ROOT}/profiles/h100_6000_bf16

worker-logits-compare-traces --left ${RESULT_ROOT}/traces/qwen3-32b-bf16-h100 --right ${RESULT_ROOT}/traces/qwen3-32b-fp8-h100 --left-label H100-BF16 --right-label H100-FP8 --output-dir ${RESULT_ROOT}/comparisons/bf16_vs_fp8
worker-logits-verify-comparison --comparison ${RESULT_ROOT}/comparisons/bf16_vs_fp8 --profile ${RESULT_ROOT}/profiles/h100_6000_bf16/profile.json --output-dir ${RESULT_ROOT}/verifications/bf16_vs_fp8 --no-fail-on-reject
EOF
