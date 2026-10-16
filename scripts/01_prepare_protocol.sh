#!/usr/bin/env bash
# Frozen no-validation protocol: nested eval/train cohorts + 20 title-safe candidates.
# Usage: bash scripts/01_prepare_protocol.sh CDs        (NEG_FALLBACK=pooled SIZES=100,300,500,1000)
source "$(dirname "$0")/_common.sh" "${1:-CDs}"
run "$PYTHON" -m bt4rec.cli.prepare_protocol --dataset "$DS" \
  --sizes "${SIZES:-100,300,500,1000}" --negative-fallback "${NEG_FALLBACK:-none}" --seed "$SEED" \
  ${EXTRA_ARGS:-}
