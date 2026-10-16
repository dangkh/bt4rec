#!/usr/bin/env bash
# Train-only behaviour windows (dual view) for the Tree AND for inference.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/02_precompute_behavior.sh CDs [plan|run]
#        BACKEND=qwen (default gemma), LLM_BATCH_SIZE=..., EXTRA_ARGS="--overwrite"
source "$(dirname "$0")/_common.sh" "${1:-CDs}"
STAGE="${2:-plan}"
args=(--backend "${BACKEND:-gemma}" --dataset "$DS" --data-root "$DATA_ROOT"
      --user-ids-file "$EVAL_FILE" --history-size "$HIST" --window-size "$WINDOW")
[[ -z "${LLM_BATCH_SIZE:-}" ]] || args+=(--llm-batch-size "$LLM_BATCH_SIZE")
case "$STAGE" in
  plan) run "$PYTHON" -m bt4rec.cli.precompute "${args[@]}" --plan-only ${EXTRA_ARGS:-} ;;
  run)  run "$PYTHON" -m bt4rec.cli.precompute "${args[@]}" ${EXTRA_ARGS:-} ;;
  *) echo "Unknown stage $STAGE (plan|run)" >&2; exit 2 ;;
esac
