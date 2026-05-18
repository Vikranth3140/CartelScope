#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
LOG_DIR="${LOG_DIR:-reports/graph_analysis_cpu_vm}"
mkdir -p "${LOG_DIR}" graph_analysis_outputs data/processed
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

"${PYTHON_BIN}" -m citation_cartels.graph.final_graph_analysis \
  --config configs/cikm_middle.yaml \
  --edges data/interim/edges.parquet \
  --papers data/interim/paper_metadata.parquet \
  --communities data/interim/pre_annotation_node_communities.parquet \
  --semantic-edges data/processed/semantic_edges_combined.parquet \
  --weighted-edges-output data/processed/weighted_edges.parquet \
  --output-dir graph_analysis_outputs \
  --top-k-excision-communities 5 \
  --sample-path-nodes "${SAMPLE_PATH_NODES:-32}" \
  2>&1 | tee "${LOG_DIR}/final_graph_analysis.log"

"${PYTHON_BIN}" -m citation_cartels.reporting.package_graph_outputs \
  --graph-output-dir graph_analysis_outputs \
  --reports-dir "${LOG_DIR}" \
  --artifacts-dir artifacts \
  2>&1 | tee "${LOG_DIR}/package_graph_outputs.log"
