#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

PYTHONPATH=src "$PYTHON_BIN" -m citation_cartels.annotation.train_student \
  --config configs/cikm_middle.yaml \
  --labels data/labels/teacher_labels_final.parquet \
  --output-dir models/scibert_citation_intent_local_mps \
  --report-dir reports/scibert_local_mps \
  --epochs "${EPOCHS:-3}" \
  --learning-rate 2e-5 \
  --train-batch-size "${TRAIN_BATCH_SIZE:-4}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-8}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --save-steps "${SAVE_STEPS:-2000}" \
  --logging-steps "${LOGGING_STEPS:-50}" \
  --eval-strategy steps \
  ${RESUME_FROM_CHECKPOINT:+--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT"}
