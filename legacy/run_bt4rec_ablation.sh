#!/usr/bin/env bash
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_bt4rec_ablation.sh CDs validate
#   CUDA_VISIBLE_DEVICES=0 bash run_bt4rec_ablation.sh CDs summaries
#   CUDA_VISIBLE_DEVICES=0 bash run_bt4rec_ablation.sh CDs oracle
#   CUDA_VISIBLE_DEVICES=0 bash run_bt4rec_ablation.sh CDs test
#   CUDA_VISIBLE_DEVICES=0 bash run_bt4rec_ablation.sh CDs main
#   CUDA_VISIBLE_DEVICES=0 bash run_bt4rec_ablation.sh CDs all
# Modes: latest_only,latest_how,random_how,global_continuation,pmn,tree,oracle,
#        recent3,recent3_tree,tree_original,native,
#        stay_only,tree_stay,tree_original_stay,recent3_tree_stay.
# For matched prompt comparisons without PMN or oracle caches:
#   MODES=tree,tree_stay bash run_bt4rec_ablation.sh CDs main
#   MODES=tree_original,tree_original_stay bash run_bt4rec_ablation.sh CDs main
set -euo pipefail
DS="${1:-CDs}"; STAGE="${2:-validate}"
case "${DS,,}" in
  cds|cd) DS=CDs; HIST=20; TREE_DEFAULT=trees/CDs_1000_20 ;;
  video_games|vg|vgs|video|videogames) DS=Video_Games; HIST=20; TREE_DEFAULT=trees/VGs_1000_20 ;;
  ml|ml1m|movielens) DS=ML; HIST=30; TREE_DEFAULT=trees/ML_1000_20 ;;
  *) echo "Unknown dataset: $DS" >&2; exit 2;;
esac
PYTHON="${PYTHON:-python}"
TREE="${TREE_DIR:-$TREE_DEFAULT}"
DATA_ROOT="${DATA_ROOT:-data}"
BASE="$DATA_ROOT/$DS"
TEST_BEHAVIORS="${TEST_BEHAVIORS:-precomputed/$DS/test_shared.jsonl}"
TRAIN_BEHAVIORS="${TRAIN_BEHAVIORS:-precomputed/$DS/train_shared.jsonl}"
PMN_SUMMARIES="${PMN_SUMMARIES:-precomputed/$DS/pmn_preference_summaries_gemma_v4.jsonl}"
ORACLE_BEHAVIORS="${ORACLE_BEHAVIORS:-precomputed/$DS/target_user_oracle_protocol_seed42.jsonl}"
MODES="${MODES:-latest_only,latest_how,random_how,global_continuation,pmn,tree,oracle}"
MAX_USERS="${MAX_USERS:-1000}"
TEST_USERS="${TEST_USERS:-10}"
HISTORY_SIZE="${HISTORY_SIZE:-$HIST}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RANK_TOKENS="${RANK_TOKENS:-1024}"
TREE_ORDER="${TREE_ORDER:-0}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/bt4rec_ablation}"
EVAL_FILE="$BASE/protocol/eval_users_1000.json"
CAND_FILE="$BASE/protocol/user_candidates_testpool1000_seed42.json"
TRAIN_IDS="${TRAIN_IDS:-}"
if [[ -z "$TRAIN_IDS" ]]; then
  for v in "$BASE/protocol/train_users_1000.json" "$BASE/train_users_1000.json"; do
    if [[ -f "$v" ]]; then TRAIN_IDS="$v"; break; fi
  done
fi
COMMON=(--dataset "$DS" --tree "$TREE" --data-root "$DATA_ROOT"
        --precomputed-behaviors "$TEST_BEHAVIORS" --train-behaviors "$TRAIN_BEHAVIORS"
        --pmn-summaries "$PMN_SUMMARIES" --oracle-behaviors "$ORACLE_BEHAVIORS"
        --history-size "$HISTORY_SIZE" --user-ids-file "$EVAL_FILE"
        --candidate-file "$CAND_FILE" --tree-max-query-order "$TREE_ORDER"
        --top-next 1 --rank-max-new-tokens "$RANK_TOKENS"
        --llm-batch-size "$BATCH_SIZE" --seed "$SEED" --modes "$MODES"
        --output-root "$OUTPUT_ROOT")
if [[ -n "$TRAIN_IDS" ]]; then COMMON+=(--source-user-ids-file "$TRAIN_IDS"); fi
[[ "${EXCLUDE_ALL_EVAL_SOURCES:-0}" != "1" ]] || COMMON+=(--exclude-all-eval-sources)
[[ "${REQUIRE_USER_DISJOINT_TREE:-0}" != "1" ]] || COMMON+=(--require-user-disjoint-tree)
[[ "${ALLOW_TREE_SOURCE_MISMATCH:-0}" != "1" ]] || COMMON+=(--allow-tree-source-mismatch)
[[ "${SAVE_ALL_PROMPTS:-0}" != "1" ]] || COMMON+=(--save-all-prompts)
[[ "${ALLOW_DEBUG_PMN:-0}" != "1" ]] || COMMON+=(--allow-debug-pmn)
[[ -z "${RUN_DIR:-}" ]] || COMMON+=(--run-dir "$RUN_DIR")
run() { printf '[CMD]'; printf ' %q' "$@"; printf '\n'; "$@"; }
gen_summaries(){
    local extra=()
    [[ -z "$TRAIN_IDS" ]] || extra+=(--train-user-ids "$TRAIN_IDS")
    run "$PYTHON" prepare_pmn_summaries.py --dataset "$DS" --data-root "$DATA_ROOT" \
      --train-behaviors "$TRAIN_BEHAVIORS" --precomputed-behaviors "$TEST_BEHAVIORS" \
      --history-size "$HISTORY_SIZE" --max-target-users 1000 \
      --window-size 3 --window-index-offset "${WINDOW_INDEX_OFFSET:-auto}" \
      --output "$PMN_SUMMARIES" --seed "$SEED" --resume "${extra[@]}"
}
gen_oracle(){
    local extra=()
    [[ ! -f "$ORACLE_BEHAVIORS" ]] || extra+=(--resume)
    run "$PYTHON" prepare_target_user_oracle.py --dataset "$DS" \
      --items "$BASE/items.json" --sequences "$BASE/user_sequences.json" \
      --candidate-file "$CAND_FILE" --user-ids-file "$EVAL_FILE" \
      --max-users 1000 --window-size 3 --seed "$SEED" \
      --output "$ORACLE_BEHAVIORS" "${extra[@]}"
}
case "$STAGE" in
  summaries) gen_summaries;;
  oracle) gen_oracle;;
  validate) run "$PYTHON" inference_bt4rec_ablation.py "${COMMON[@]}" --max-users "$MAX_USERS" --validate-only;;
  prompts) run "$PYTHON" inference_bt4rec_ablation.py "${COMMON[@]}" --max-users "$TEST_USERS" --prompts-only;;
  test) run "$PYTHON" inference_bt4rec_ablation.py "${COMMON[@]}" --max-users "$TEST_USERS";;
  main) run "$PYTHON" inference_bt4rec_ablation.py "${COMMON[@]}" --max-users "$MAX_USERS";;
  all)
    if [[ ",$MODES," == *,pmn,* ]]; then gen_summaries; fi
    if [[ ",$MODES," == *,oracle,* ]]; then gen_oracle; fi
    run "$PYTHON" inference_bt4rec_ablation.py "${COMMON[@]}" --max-users "$MAX_USERS" --validate-only
    run "$PYTHON" inference_bt4rec_ablation.py "${COMMON[@]}" --max-users "$MAX_USERS";;
  *) echo "Unknown stage $STAGE (summaries|oracle|validate|prompts|test|main|all)" >&2; exit 2;;
esac
