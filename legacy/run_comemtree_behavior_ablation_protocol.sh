#!/usr/bin/env bash
# Run the five-condition CoMemTree behavior-source ablation on the FROZEN protocol.
# Usage: bash run_comemtree_behavior_ablation_protocol.sh {CDs|Video_Games|ML} {oracle|validate|prompts|test|main|all}
# Supports CUDA_VISIBLE_DEVICES, TRAIN_BEHAVIORS, TEST_BEHAVIORS, ORACLE_BEHAVIORS,
# TREE_DIR, MAX_USERS, HISTORY_SIZE, TREE_ORDER, LLM_BATCH_SIZE,
# RANK_MAX_NEW_TOKENS, ALLOW_TREE_SOURCE_MISMATCH, STRICT_MATCHED_SOURCE,
# SAVE_ALL_PROMPTS, RUN_DIR, OUTPUT_ROOT, DRY_RUN, PYTHON.
set -euo pipefail

DS_RAW="${1:-Video_Games}"
MODE="${2:-validate}"
case "${DS_RAW,,}" in
  cds|cd) DATASET="CDs"; TREE_DEFAULT="trees/CDs_1000_20"; HISTORY_DEFAULT=20 ;;
  video_games|videogames|video|vg|vgs) DATASET="Video_Games"; TREE_DEFAULT="trees/VGs_1000_20"; HISTORY_DEFAULT=20 ;;
  ml|ml1m|movielens) DATASET="ML"; TREE_DEFAULT="trees/ML_1000_20"; HISTORY_DEFAULT=30 ;;
  *) echo "ERROR: unsupported dataset '$DS_RAW'. Use CDs, Video_Games or ML." >&2; exit 2 ;;
esac
case "$MODE" in
  oracle|validate|prompts|test|main|all) ;;
  *) echo "ERROR: unknown mode '$MODE'. Use oracle|validate|prompts|test|main|all." >&2; exit 2 ;;
esac

PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATA_ROOT="${DATA_ROOT:-data}"
TREE_DIR="${TREE_DIR:-$TREE_DEFAULT}"
DATA_DIR="$DATA_ROOT/$DATASET"
ITEMS="${ITEMS:-$DATA_DIR/items.json}"
SEQUENCES="${SEQUENCES:-$DATA_DIR/user_sequences.json}"
CANDIDATE_FILE="${CANDIDATE_FILE:-$DATA_DIR/protocol/user_candidates_testpool1000_seed42.json}"
EVAL_USERS_FILE="${EVAL_USERS_FILE:-$DATA_DIR/protocol/eval_users_1000.json}"
HISTORY_SIZE="${HISTORY_SIZE:-$HISTORY_DEFAULT}"
MAX_USERS="${MAX_USERS:-1000}"
TEST_USERS="${TEST_USERS:-10}"
TREE_ORDER="${TREE_ORDER:-0}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-8}"
RANK_MAX_NEW_TOKENS="${RANK_MAX_NEW_TOKENS:-1024}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/comemtree_behavior_ablation}"
ORACLE_BEHAVIORS="${ORACLE_BEHAVIORS:-precomputed/$DATASET/target_user_oracle_protocol_seed42.jsonl}"

# Select the test-behavior cache (prefer protocol-standardized cache, then legacy Gemma cache).
if [[ -z "${TEST_BEHAVIORS:-}" ]]; then
  for x in \
      "precomputed/$DATASET/test_shared.jsonl" \
      "precomputed/$DATASET/gemma/test_shared.jsonl" \
      "precomputed2/$DATASET/test_user_behaviors_gemma_0.jsonl"; do
    if [[ -f "$x" ]]; then TEST_BEHAVIORS="$x"; break; fi
  done
  TEST_BEHAVIORS="${TEST_BEHAVIORS:-precomputed/$DATASET/test_shared.jsonl}"
fi

# Training behavior memory must match the *Tree-building* stream, not the test cache.
# Prefer the input path stored in behavior_states.json; if the Tree has moved,
# find a file with the same basename locally. Never silently substitute an
# unrelated memory when Tree metadata points to a different stream.
if [[ -z "${TRAIN_BEHAVIORS:-}" ]]; then
  TREE_SOURCE=""
  if [[ -f "$TREE_DIR/behavior_states.json" ]]; then
    TREE_SOURCE="$($PYTHON - "$TREE_DIR/behavior_states.json" <<'PY'
import json,sys
with open(sys.argv[1],encoding='utf-8') as f: obj=json.load(f)
print(obj.get('metadata',{}).get('input') or '')
PY
)"
  fi
  if [[ -n "$TREE_SOURCE" ]]; then
    if [[ -f "$TREE_SOURCE" ]]; then
      TRAIN_BEHAVIORS="$TREE_SOURCE"
    else
      BN="${TREE_SOURCE##*/}"
      for x in "precomputed/$DATASET/$BN" "precomputed2/$DATASET/$BN"; do
        if [[ -f "$x" ]]; then TRAIN_BEHAVIORS="$x"; break; fi
      done
    fi
  else
    for x in \
        "precomputed/$DATASET/train_shared.jsonl" \
        "precomputed/$DATASET/local_memories_gemma_dual_1000.jsonl" \
        "precomputed/$DATASET/local_memories_gemma_dual_0.jsonl"; do
      if [[ -f "$x" ]]; then TRAIN_BEHAVIORS="$x"; break; fi
    done
  fi
  if [[ -z "${TRAIN_BEHAVIORS:-}" ]]; then
    if [[ "$MODE" == "oracle" ]]; then
      TRAIN_BEHAVIORS="not_required_for_oracle_generation"
    else
      echo "ERROR: Cannot locate the exact training memory used to build '$TREE_DIR'." >&2
      echo "Tree metadata input: '${TREE_SOURCE:-not recorded}'" >&2
      echo "Set TRAIN_BEHAVIORS=/path/to/exact_train_memory.jsonl explicitly." >&2
      exit 1
    fi
  fi
fi

if [[ "$TRAIN_BEHAVIORS" == "$TEST_BEHAVIORS" ]]; then
  echo "ERROR: train-source and test-behavior paths must differ." >&2; exit 1
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  for f in "$ITEMS" "$SEQUENCES" "$CANDIDATE_FILE" "$EVAL_USERS_FILE"; do
    [[ -f "$f" ]] || { echo "ERROR: Missing input file: $f" >&2; exit 1; }
  done
  if [[ "$MODE" != "oracle" ]]; then
    for f in "$TEST_BEHAVIORS" "$TRAIN_BEHAVIORS" "$TREE_DIR/behavior_states.json" \
             "$TREE_DIR/behavior_state_embeddings.npy" "$TREE_DIR/behavior_tree.json"; do
      [[ -f "$f" ]] || { echo "ERROR: Missing input file: $f" >&2; exit 1; }
    done
    if [[ "$MODE" != "all" && ! -f "$ORACLE_BEHAVIORS" ]]; then
      echo "ERROR: Oracle cache not found: $ORACLE_BEHAVIORS" >&2
      echo "Create it first: bash $0 $DATASET oracle" >&2
      exit 1
    fi
  fi
fi

COMMON=(
  --dataset "$DATASET"
  --items "$ITEMS"
  --sequences "$SEQUENCES"
  --tree "$TREE_DIR"
  --precomputed-behaviors "$TEST_BEHAVIORS"
  --train-behaviors "$TRAIN_BEHAVIORS"
  --oracle-behaviors "$ORACLE_BEHAVIORS"
  --candidate-file "$CANDIDATE_FILE"
  --user-ids-file "$EVAL_USERS_FILE"
  --history-size "$HISTORY_SIZE"
  --tree-max-query-order "$TREE_ORDER"
  --top-next 1
  --llm-batch-size "$LLM_BATCH_SIZE"
  --rank-max-new-tokens "$RANK_MAX_NEW_TOKENS"
  --embedding-batch-size "$EMBEDDING_BATCH_SIZE"
  --output-root "$OUTPUT_ROOT"
  --seed 42
)
[[ "${ALLOW_TREE_SOURCE_MISMATCH:-0}" == "1" ]] && COMMON+=(--allow-tree-source-mismatch)
[[ "${STRICT_MATCHED_SOURCE:-0}" == "1" ]] && COMMON+=(--strict-matched-source)
[[ "${SAVE_ALL_PROMPTS:-0}" == "1" ]] && COMMON+=(--save-all-prompts)
[[ -n "${RUN_DIR:-}" ]] && COMMON+=(--run-dir "$RUN_DIR")

run_cmd() {
  printf '\n[COMMAND] '; printf '%q ' "$@"; printf '\n'
  if [[ "${DRY_RUN:-0}" != "1" ]]; then "$@"; fi
}

oracle() {
  # Do NOT resume a cache built for a different split. The oracle preparer only
  # checks user IDs on resume; the ablation validator checks target IDs separately.
  local extra=()
  if [[ -f "$ORACLE_BEHAVIORS" ]]; then
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
      # Avoid resuming a future-aware cache from a different frozen test split.
      "$PYTHON" - "$ORACLE_BEHAVIORS" "$SEQUENCES" "$EVAL_USERS_FILE" "$MAX_USERS" <<'ORACLE_RESUME_CHECK'
import json, sys
oracle_path, seq_path, eval_path, max_users = sys.argv[1:]
with open(seq_path, encoding='utf-8') as f:
    sequences = json.load(f)
with open(eval_path, encoding='utf-8') as f:
    payload = json.load(f)
if isinstance(payload, dict):
    users = payload['users'] if isinstance(payload.get('users'), list) else list(payload)
else:
    users = payload
users = {str(u) for u in (users[:int(max_users)] if int(max_users) > 0 else users)}
seen = set()
with open(oracle_path, encoding='utf-8') as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        uid = str(row.get('user_id'))
        if uid not in users:
            continue
        target = [str(x) for x in sequences[uid].get('test', [])]
        if len(target) != 1 or str(row.get('target_item_id')) != target[0]:
            raise SystemExit(
                f'ERROR: Stale oracle line {line_no}, user={uid}; '
                f'saved_target={row.get("target_item_id")!r}, current_test={target}. '
                'Use a new ORACLE_BEHAVIORS filename for the new protocol.'
            )
        if uid in seen:
            raise SystemExit(f'ERROR: Duplicate saved oracle user {uid} at line {line_no}')
        seen.add(uid)
print(f'[ORACLE] Resume provenance OK for {len(seen)} previously saved eval users')
ORACLE_RESUME_CHECK
    fi
    extra+=(--resume)
  fi
  run_cmd "$PYTHON" prepare_target_user_oracle.py \
    --dataset "$DATASET" --items "$ITEMS" --sequences "$SEQUENCES" \
    --candidate-file "$CANDIDATE_FILE" --user-ids-file "$EVAL_USERS_FILE" \
    --output "$ORACLE_BEHAVIORS" --max-users "$MAX_USERS" --window-size 3 \
    --seed 42 "${extra[@]}"
}
validate() {
  run_cmd "$PYTHON" inference_comemtree_behavior_ablation.py \
    "${COMMON[@]}" --max-users "$MAX_USERS" --validate-only
}
case "$MODE" in
  oracle) oracle ;;
  validate) validate ;;
  prompts) run_cmd "$PYTHON" inference_comemtree_behavior_ablation.py \
             "${COMMON[@]}" --max-users "$TEST_USERS" --prompts-only ;;
  test) run_cmd "$PYTHON" inference_comemtree_behavior_ablation.py \
          "${COMMON[@]}" --max-users "$TEST_USERS" ;;
  main) run_cmd "$PYTHON" inference_comemtree_behavior_ablation.py \
          "${COMMON[@]}" --max-users "$MAX_USERS" ;;
  all) oracle; validate; run_cmd "$PYTHON" inference_comemtree_behavior_ablation.py \
           "${COMMON[@]}" --max-users "$MAX_USERS" ;;
esac
