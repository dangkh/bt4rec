#!/usr/bin/env bash
# Collaborative reverse behaviour Tree from train_shared.jsonl (hyper-parameters: configs/tree.yaml).
# Usage: bash scripts/03_build_tree.sh CDs      (TRAIN_BEHAVIORS=..., TREE_DIR=..., LEGACY=1)
source "$(dirname "$0")/_common.sh" "${1:-CDs}"
SUB=""; [[ "${BACKEND:-gemma}" == qwen ]] && SUB="/qwen"
TRAIN_BEHAVIORS="${TRAIN_BEHAVIORS:-$PRECOMP_DIR$SUB/train_shared.jsonl}"
TREE_DIR="${TREE_DIR:-$TREE_DEFAULT}"
args=(--input "$TRAIN_BEHAVIORS" --output-dir "$TREE_DIR" --seed "$SEED"
      --num-clusters "${NUM_CLUSTERS:-$TREECFG_NUM_CLUSTERS}"
      --cluster-mode "$TREECFG_CLUSTER_MODE" --constraint-level "$TREECFG_CONSTRAINT_LEVEL"
      --cluster-text-field "$TREECFG_CLUSTER_TEXT_FIELD" --embedding-model "$TREECFG_EMBEDDING_MODEL"
      --min-cluster-similarity "$TREECFG_MIN_CLUSTER_SIMILARITY"
      --min-cluster-margin "$TREECFG_MIN_CLUSTER_MARGIN" --max-order "$TREECFG_MAX_ORDER"
      --count-mode "$TREECFG_COUNT_MODE" --smoothing-kappa "$TREECFG_SMOOTHING_KAPPA"
      --min-support-users "$TREECFG_MIN_SUPPORT_USERS"
      --min-support-occurrences "$TREECFG_MIN_SUPPORT_OCCURRENCES" --min-jsd "$TREECFG_MIN_JSD"
      --label-assignment "$TREECFG_LABEL_ASSIGNMENT" --state-signature "$TREECFG_STATE_SIGNATURE")
# LEGACY=1 reproduces the old builder (raw KMeans labels + most-frequent signature).
[[ "${LEGACY:-0}" != 1 ]] || args+=(--label-assignment kmeans --state-signature mode)
[[ "$TREECFG_COLLAPSE_CONSECUTIVE_STATES" != true ]] || args+=(--collapse-consecutive-states)
[[ -z "${EXCLUDE_USERS:-}" ]] || args+=(--exclude-users "$EXCLUDE_USERS")
run "$PYTHON" -m bt4rec.cli.build_tree "${args[@]}" ${EXTRA_ARGS:-}
