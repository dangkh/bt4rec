#!/usr/bin/env bash
# Usage: CUDA_VISIBLE_DEVICES=0 bash run_pmn_protocol.sh CDs all
# Modes: validate, summaries, pmn-validate, prompts, rank, all
# Source memories must contain LOCAL dual-view behavior windows generated from
# TRAIN interactions only (not future-aware oracle/test behaviors).
set -euo pipefail
DS="${1:-CDs}"
MODE="${2:-all}"
case "$DS" in
  CDs|CDs_and_Vinyl) DS=CDs; HIST=20;;
  Video_Games|VGs|Video) DS=Video_Games; HIST=20;;
  ML|ML1M|MovieLens) DS=ML; HIST=30;;
  *) echo "Unknown dataset $DS" >&2; exit 2;;
esac
BASE="data/$DS"
if [[ ! -f "$BASE/protocol/eval_users_1000.json" || ! -f "$BASE/protocol/user_candidates_testpool1000_seed42.json" ]]; then
  echo "Missing frozen protocol/ files under $BASE" >&2; exit 2
fi
if [[ -z "${TRAIN_BEHAVIORS:-}" ]]; then
  for path in "precomputed/$DS/train_shared.jsonl" \
              "precomputed/$DS/local_memories_gemma_dual_1000.jsonl" \
              "precomputed/$DS/local_memories_gemma_dual_0.jsonl"; do
    [[ ! -f "$path" ]] || { TRAIN_BEHAVIORS="$path"; break; }
  done
fi
if [[ -z "${TEST_BEHAVIORS:-}" ]]; then
  for path in "precomputed/$DS/test_shared.jsonl" \
              "precomputed2/$DS/test_user_behaviors_gemma_0.jsonl" \
              "precomputed/$DS/test_user_behaviors_gemma_0.jsonl"; do
    [[ ! -f "$path" ]] || { TEST_BEHAVIORS="$path"; break; }
  done
fi
[[ -f "${TRAIN_BEHAVIORS:-}" ]] || { echo 'Set TRAIN_BEHAVIORS=... training dual-view JSONL' >&2; exit 2; }
[[ -f "${TEST_BEHAVIORS:-}" ]] || { echo 'Set TEST_BEHAVIORS=... target train-only behavior cache' >&2; exit 2; }
TRAIN_IDS="${TRAIN_USER_IDS:-}"
if [[ -z "$TRAIN_IDS" ]]; then
  for path in "$BASE/protocol/train_users_1000.json" "$BASE/train_users_1000.json"; do
    [[ ! -f "$path" ]] || { TRAIN_IDS="$path"; break; }
  done
fi
SUMMARIES="${SUMMARIES:-precomputed/$DS/pmn_preference_summaries_gemma_v3.jsonl}"
MAX_USERS="${MAX_USERS:-1000}"
MAX_SOURCE_USERS="${MAX_SOURCE_USERS:-0}"
WINDOW_SIZE="${WINDOW_SIZE:-3}"
WINDOW_INDEX_OFFSET="${WINDOW_INDEX_OFFSET:-auto}"
SUM_BATCH="${SUM_BATCH:-8}"
RANK_BATCH="${RANK_BATCH:-8}"
RANK_TOKENS="${RANK_TOKENS:-1024}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/pmn}"
SRC_ARGS=()
[[ -z "$TRAIN_IDS" ]] || SRC_ARGS+=(--train-user-ids "$TRAIN_IDS")
COMMON=(--dataset "$DS" --train-behaviors "$TRAIN_BEHAVIORS" \
  --precomputed-behaviors "$TEST_BEHAVIORS" --history-size "$HIST" \
  --window-size "$WINDOW_SIZE" --window-index-offset "$WINDOW_INDEX_OFFSET" \
  "${SRC_ARGS[@]}")
SUM_ARGS=("${COMMON[@]}" --max-target-users "$MAX_USERS" \
  --max-source-users "$MAX_SOURCE_USERS" --batch-size "$SUM_BATCH" \
  --seed "$SEED" --output "$SUMMARIES")
RANK_ARGS=("${COMMON[@]}" --max-users "$MAX_USERS" \
  --max-source-users "$MAX_SOURCE_USERS" --summaries "$SUMMARIES" \
  --llm-batch-size "$RANK_BATCH" --rank-max-new-tokens "$RANK_TOKENS" \
  --seed "$SEED" --output-root "$OUTPUT_ROOT")
[[ "${RERUN_LATEST:-0}" != 1 ]] || RANK_ARGS+=(--run-latest-only)
[[ "${STRICT_USER_DISJOINT:-0}" != 1 ]] || RANK_ARGS+=(--exclude-all-eval-users-from-sources)
[[ -z "${RUN_DIR:-}" ]] || RANK_ARGS+=(--run-dir "$RUN_DIR")
[[ "${DEBUG_TFIDF:-0}" != 1 ]] || RANK_ARGS+=(--embedding-backend tfidf_debug --allow-debug-profiles)
[[ "${DEBUG_EXTRACTIVE:-0}" != 1 ]] || SUM_ARGS+=(--debug-extractive-summary)
[[ "${DEBUG_EXTRACTIVE:-0}" != 1 ]] || RANK_ARGS+=(--allow-debug-profiles)

echo "[PMN] dataset=$DS history=$HIST train=$TRAIN_BEHAVIORS test=$TEST_BEHAVIORS"
echo "[PMN] summaries=$SUMMARIES eval=$BASE/protocol/eval_users_1000.json"
case "$MODE" in
  validate)
    python prepare_pmn_summaries.py "${SUM_ARGS[@]}" --validate-only;;
  summaries)
    python prepare_pmn_summaries.py "${SUM_ARGS[@]}" --resume;;
  pmn-validate)
    python inference_pmn_qwen.py "${RANK_ARGS[@]}" --validate-only;;
  prompts)
    python inference_pmn_qwen.py "${RANK_ARGS[@]}" --prompts-only;;
  rank)
    python inference_pmn_qwen.py "${RANK_ARGS[@]}";;
  all)
    python prepare_pmn_summaries.py "${SUM_ARGS[@]}" --validate-only
    python prepare_pmn_summaries.py "${SUM_ARGS[@]}" --resume
    python inference_pmn_qwen.py "${RANK_ARGS[@]}" --validate-only
    python inference_pmn_qwen.py "${RANK_ARGS[@]}";;
  *) echo "Unknown mode $MODE. Use validate|summaries|pmn-validate|prompts|rank|all" >&2; exit 2;;
esac
