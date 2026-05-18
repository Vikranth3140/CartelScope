#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"$PYTHON_BIN" -m pip install -r requirements.txt

PYTHONPATH=src "$PYTHON_BIN" -m citation_cartels.annotation.train_student \
  --config configs/cikm_middle.yaml \
  --labels data/labels/teacher_labels_final.parquet \
  --output-dir models/scibert_citation_intent_final \
  --report-dir reports \
  --epochs 4 \
  --learning-rate 2e-5 \
  --train-batch-size "${TRAIN_BATCH_SIZE:-16}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-32}"
