#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
LOG_DIR="${LOG_DIR:-reports/cci_seed_stability_cpu_vm}"
STABILITY_OUTPUT_DIR="${STABILITY_OUTPUT_DIR:-graph_analysis_outputs/seed_stability}"
STABILITY_SEEDS="${STABILITY_SEEDS:-11 23 37 42 101}"
mkdir -p "${LOG_DIR}" "${STABILITY_OUTPUT_DIR}" data/processed

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

export PYTHONPATH=src
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(python3 - <<'PY'
import os
print(os.cpu_count() or 1)
PY
)}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-${OMP_NUM_THREADS}}"

"${PYTHON_BIN}" -m citation_cartels.graph.cci_seed_stability \
  --config configs/cikm_middle.yaml \
  --edges data/interim/edges.parquet \
  --papers data/interim/paper_metadata.parquet \
  --semantic-edges data/processed/semantic_edges_combined.parquet \
  --weighted-edges-output data/processed/weighted_edges_stability.parquet \
  --output-dir "${STABILITY_OUTPUT_DIR}" \
  --reference-seed 42 \
  --seeds ${STABILITY_SEEDS} \
  2>&1 | tee "${LOG_DIR}/cci_seed_stability.log"
