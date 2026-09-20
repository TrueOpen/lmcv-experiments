BASE_URL ?= http://localhost:8000
MODEL ?=
PROMPTS ?= ../data/qwen3_32b_identity_v1/prompts.jsonl
OUT_DIR ?= data
REPORT_DIR ?= reports
MAX_TOKENS ?= 128
TEMPERATURE ?= 0
SEED ?= 0
PYTHON ?= python3
LIMIT ?= 0

.PHONY: generate prefill compare guard-model

guard-model:
	@test -n "$(MODEL)" || { echo "Set MODEL=<MODEL> when running this target." >&2; exit 2; }

generate: guard-model
	$(PYTHON) scripts/generate_decode_routes.py \
		--base-url "$(BASE_URL)" \
		--model "$(MODEL)" \
		--prompts-file "$(PROMPTS)" \
		--output "$(OUT_DIR)/generated_routes.jsonl" \
		--max-tokens "$(MAX_TOKENS)" \
		--temperature "$(TEMPERATURE)" \
		--seed "$(SEED)" \
		--limit "$(LIMIT)"

prefill:
	$(PYTHON) scripts/prefill_routes.py \
		--base-url "$(BASE_URL)" \
		--model "$(MODEL)" \
		--input "$(OUT_DIR)/generated_routes.jsonl" \
		--output "$(OUT_DIR)/prefill_routes.jsonl"

compare:
	$(PYTHON) scripts/compare_routes.py \
		--generated "$(OUT_DIR)/generated_routes.jsonl" \
		--prefill "$(OUT_DIR)/prefill_routes.jsonl" \
		--output-dir "$(REPORT_DIR)"
