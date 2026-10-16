#!/usr/bin/env bash
set -euo pipefail

# Run on the standardized frozen protocol in data/<dataset>/protocol/.
# Usage: CUDA_VISIBLE_DEVICES=0 bash run_recent3_tree_protocol.sh CDs [main|test|prompts|validate]
# Set COMPARE_RECENT3=0 to run only the NEW tree_recent3 condition (no baseline replay).
# Optionally override TREE_DIR, TEST_BEHAVIORS, OUTPUT_ROOT, MAX_USERS, GPU.

DATASET="${1:-CDs}"
MODE="${2:-main}"
case "${DATASET,,}" in
  cds|cd) DATASET="CDs"; tree_default="trees/CDs_1000_20"; history_default=20 ;;
  video_games|videogames|vgs|vg) DATASET="Video_Games"; tree_default="trees/VGs_1000_20"; history_default=20 ;;
  ml|ml1m|movielens) DATASET="ML"; tree_default="trees/ML_1000_20"; history_default=30 ;;
  *) echo "Invalid DATASET: $DATASET (CDs, Video_Games, ML)" >&2; exit 2 ;;
esac

TREE_DIR="${TREE_DIR:-${tree_default}}"
HISTORY_SIZE="${HISTORY_SIZE:-${history_default}}"
MAX_USERS="${MAX_USERS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RANK_MAX_NEW_TOKENS="${RANK_MAX_NEW_TOKENS:-1024}"
TREE_MAX_QUERY_ORDER="${TREE_MAX_QUERY_ORDER:-0}"
TOP_NEXT="${TOP_NEXT:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/comemtree_recent3_tree}"
COMPARE_RECENT3="${COMPARE_RECENT3:-1}"
ALLOW_TREE_SOURCE_MISMATCH="${ALLOW_TREE_SOURCE_MISMATCH:-1}"
SEED="${SEED:-42}"

if [[ -n "${TEST_BEHAVIORS:-}" ]]; then
    CACHE="$TEST_BEHAVIORS"
elif [[ -f "precomputed/${DATASET}/test_shared.jsonl" ]]; then
    CACHE="precomputed/${DATASET}/test_shared.jsonl"
elif [[ -f "precomputed2/${DATASET}/test_user_behaviors_gemma_0.jsonl" ]]; then
    CACHE="precomputed2/${DATASET}/test_user_behaviors_gemma_0.jsonl"
else
    echo "Missing test behavior cache; set TEST_BEHAVIORS=/path/to/test.jsonl" >&2
    exit 2
fi

if [[ ! -d "$TREE_DIR" ]]; then
    echo "Tree folder not found: $TREE_DIR" >&2
    exit 2
fi

args=(
  --dataset "$DATASET"
  --tree "$TREE_DIR"
  --precomputed-behaviors "$CACHE"
  --output-root "$OUTPUT_ROOT"
  --max-users "$MAX_USERS"
  --history-size "$HISTORY_SIZE"
  --llm-batch-size "$BATCH_SIZE"
  --rank-max-new-tokens "$RANK_MAX_NEW_TOKENS"
  --tree-max-query-order "$TREE_MAX_QUERY_ORDER"
  --top-next "$TOP_NEXT"
  --seed "$SEED"
  --run-tree-recent3
  --no-run-tree
  --no-run-native
  --no-run-latest-only
)
if [[ "$COMPARE_RECENT3" == 1 ]]; then
    args+=(--run-recent3)
else
    args+=(--no-run-recent3)
fi
if [[ "$ALLOW_TREE_SOURCE_MISMATCH" == 1 ]]; then
    args+=(--allow-tree-source-mismatch)
fi

case "$MODE" in
  main) ;;
  test) args+=(--max-users 10) ;;
  prompts) args+=(--max-users 10 --prompts-only) ;;
  validate) args+=(--validate-only) ;;
  *) echo "Mode must be main|test|prompts|validate" >&2; exit 2 ;;
esac

printf '[CONFIG] dataset=%s tree=%s test_cache=%s history=%s mode=%s compare_recent3=%s\n' \
  "$DATASET" "$TREE_DIR" "$CACHE" "$HISTORY_SIZE" "$MODE" "$COMPARE_RECENT3"
exec python inference_comemtree_recent3_tree.py "${args[@]}"
