#!/usr/bin/env bash
# Shared setup for scripts/*.sh. Source it:  source "$(dirname "$0")/_common.sh" "$DATASET"
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Dataset + Tree presets from configs/*.yaml -> DS, HIST, TREE_DEFAULT, EVAL_FILE, CAND_FILE, ...
eval "$("$PYTHON" -m bt4rec.cli.config "${1:-CDs}")"
DATA_ROOT="${DATA_ROOT:-data}"
SEED="${SEED:-42}"
run() { printf '[CMD]'; printf ' %q' "$@"; printf '\n'; [[ "${DRY_RUN:-0}" == 1 ]] || "$@"; }
