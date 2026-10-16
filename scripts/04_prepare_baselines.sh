#!/usr/bin/env bash
# Inputs for the PMN and ORACLE ablation modes.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/04_prepare_baselines.sh CDs {pmn|oracle|all} [validate]
source "$(dirname "$0")/_common.sh" "${1:-CDs}"
WHAT="${2:-all}"; VALIDATE="${3:-}"
SUB=""; [[ "${BACKEND:-gemma}" == qwen ]] && SUB="/qwen"
TRAIN_BEHAVIORS="${TRAIN_BEHAVIORS:-$PRECOMP_DIR$SUB/train_shared.jsonl}"
TEST_BEHAVIORS="${TEST_BEHAVIORS:-$PRECOMP_DIR$SUB/test_shared.jsonl}"
PMN_SUMMARIES="${PMN_SUMMARIES:-$PRECOMP_DIR/pmn_preference_summaries_gemma_v4.jsonl}"
ORACLE_BEHAVIORS="${ORACLE_BEHAVIORS:-$PRECOMP_DIR/target_user_oracle_protocol_seed$SEED.jsonl}"
pmn() {
  local extra=(--resume); [[ "$VALIDATE" != validate ]] || extra=(--validate-only)
  [[ ! -f "$TRAIN_IDS_DEFAULT" ]] || extra+=(--train-user-ids "$TRAIN_IDS_DEFAULT")
  run "$PYTHON" -m bt4rec.cli.prepare_pmn --dataset "$DS" --data-root "$DATA_ROOT" \
    --train-behaviors "$TRAIN_BEHAVIORS" --precomputed-behaviors "$TEST_BEHAVIORS" \
    --history-size "$HIST" --max-target-users "${MAX_USERS:-1000}" --window-size "$WINDOW" \
    --window-index-offset "${WINDOW_INDEX_OFFSET:-auto}" --output "$PMN_SUMMARIES" \
    --batch-size "${SUM_BATCH:-8}" --max-source-users "${MAX_SOURCE_USERS:-0}" \
    --seed "$SEED" "${extra[@]}" ${PMN_EXTRA:-}
}
oracle() {
  local extra=(); [[ "$VALIDATE" != validate ]] || extra=(--validate-only)
  [[ ! -f "$ORACLE_BEHAVIORS" || "$VALIDATE" == validate ]] || extra+=(--resume)
  run "$PYTHON" -m bt4rec.cli.prepare_oracle --dataset "$DS" \
    --items "$DATA_ROOT/$DS/items.json" --sequences "$DATA_ROOT/$DS/user_sequences.json" \
    --candidate-file "$CAND_FILE" --user-ids-file "$EVAL_FILE" --max-users "${MAX_USERS:-1000}" \
    --window-size "$WINDOW" --seed "$SEED" --output "$ORACLE_BEHAVIORS" "${extra[@]}"
}
case "$WHAT" in
  pmn) pmn ;; oracle) oracle ;; all) pmn; oracle ;;
  *) echo "Unknown target $WHAT (pmn|oracle|all)" >&2; exit 2 ;;
esac
