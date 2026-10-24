#!/usr/bin/env bash
# CoMemTree ranking conditions.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/05_run_ablation.sh CDs {validate|prompts|test|main}
# Modes (MODES=comma list)
#   main:     native, latest_what_only, latest_how_only, tree, tree_v2      (default)
#   ablation: random_how_only, global_continuation, pmn, oracle            (pmn/oracle: scripts/04_prepare_baselines.sh)
# Examples:
#   bash scripts/05_run_ablation.sh CDs main
#   MODES=tree,random_how_only,global_continuation,pmn,oracle bash scripts/05_run_ablation.sh CDs main
source "$(dirname "$0")/_common.sh" "${1:-CDs}"
STAGE="${2:-validate}"
SUB=""; [[ "${BACKEND:-gemma}" == qwen ]] && SUB="/qwen"
TEST_BEHAVIORS="${TEST_BEHAVIORS:-$PRECOMP_DIR$SUB/test_shared.jsonl}"
TRAIN_BEHAVIORS="${TRAIN_BEHAVIORS:-$PRECOMP_DIR$SUB/train_shared.jsonl}"
PMN_SUMMARIES="${PMN_SUMMARIES:-$PRECOMP_DIR/pmn_preference_summaries_gemma_v4.jsonl}"
ORACLE_BEHAVIORS="${ORACLE_BEHAVIORS:-$PRECOMP_DIR/target_user_oracle_protocol_seed$SEED.jsonl}"
MODES="${MODES:-native,latest_what_only,latest_how_only,tree,tree_v2}"
args=(--dataset "$DS" --tree "${TREE_DIR:-$TREE_DEFAULT}" --data-root "$DATA_ROOT"
      --precomputed-behaviors "$TEST_BEHAVIORS" --train-behaviors "$TRAIN_BEHAVIORS"
      --history-size "${HISTORY_SIZE:-$HIST}" --user-ids-file "$EVAL_FILE" --candidate-file "$CAND_FILE"
      --tree-max-query-order "${TREE_ORDER:-0}" --top-next "${TOP_NEXT:-1}"
      --rank-max-new-tokens "${RANK_TOKENS:-1024}" --llm-batch-size "${BATCH_SIZE:-8}"
      --seed "$SEED" --modes "$MODES" --output-root "${OUTPUT_ROOT:-results/comemtree}")
[[ ",$MODES," != *,pmn,* ]] || args+=(--pmn-summaries "$PMN_SUMMARIES")
[[ ",$MODES," != *,oracle,* ]] || args+=(--oracle-behaviors "$ORACLE_BEHAVIORS")
[[ ! -f "$TRAIN_IDS_DEFAULT" ]] || args+=(--source-user-ids-file "$TRAIN_IDS_DEFAULT")
[[ "${EXCLUDE_ALL_EVAL_SOURCES:-0}" != 1 ]] || args+=(--exclude-all-eval-sources)
[[ "${REQUIRE_USER_DISJOINT_TREE:-0}" != 1 ]] || args+=(--require-user-disjoint-tree)
[[ "${ALLOW_TREE_SOURCE_MISMATCH:-0}" != 1 ]] || args+=(--allow-tree-source-mismatch)
[[ "${SAVE_ALL_PROMPTS:-0}" != 1 ]] || args+=(--save-all-prompts)
[[ "${ALLOW_DEBUG_PMN:-0}" != 1 ]] || args+=(--allow-debug-pmn)
[[ -z "${RUN_DIR:-}" ]] || args+=(--run-dir "$RUN_DIR")
case "$STAGE" in
  validate) run "$PYTHON" -m bt4rec.cli.run "${args[@]}" --max-users "${MAX_USERS:-1000}" --validate-only ;;
  prompts)  run "$PYTHON" -m bt4rec.cli.run "${args[@]}" --max-users "${TEST_USERS:-10}" --prompts-only ;;
  test)     run "$PYTHON" -m bt4rec.cli.run "${args[@]}" --max-users "${TEST_USERS:-10}" ;;
  main)     run "$PYTHON" -m bt4rec.cli.run "${args[@]}" --max-users "${MAX_USERS:-1000}" ;;
  *) echo "Unknown stage $STAGE (validate|prompts|test|main)" >&2; exit 2 ;;
esac
