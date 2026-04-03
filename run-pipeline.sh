#!/usr/bin/env bash
# ==========================================================
# 🚀 CiteGraphLens Full Pipeline Runner (WSL / Linux + GPU)
# Author: Vikranth Udandarao
# ==========================================================

set -e  # Exit immediately if any command fails
set -o pipefail

# ---- CONFIG ----
BASE_DIR="/mnt/c/Users/vikra/OneDrive/Desktop/CiteGraphLens"
VENV_PATH="$BASE_DIR/venv"
LOG_DIR="$BASE_DIR/logs"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

mkdir -p "$LOG_DIR"
cd "$BASE_DIR" || exit

echo "============================================"
echo "🧠 CiteGraphLens Pipeline Starting ($TIMESTAMP)"
echo "Base Directory: $BASE_DIR"
echo "============================================"

# ---- ACTIVATE VENV ----
echo "🔹 Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# ---- GPU CHECK ----
echo "🔍 Checking CUDA availability..."
python -c "import torch; print('✅ CUDA available' if torch.cuda.is_available() else '⚠️ CUDA NOT available')"

# ---- YEARLY FETCH ----
for YEAR in 2020 2021 2022 2023; do
  echo "--------------------------------------------------"
  echo "📅 Fetching papers for year $YEAR"
  python T1-cite_graph_lens.py \
    --cs-papers 500 \
    --bio-papers 500 \
    --med-papers 500 \
    --include-references \
    --email "vikranth22570@iiitd.ac.in" \
    --year $YEAR \
    > "$LOG_DIR/T1_${YEAR}_$TIMESTAMP.log" 2>&1

  echo "✅ Completed year $YEAR — Log: $LOG_DIR/T1_${YEAR}_$TIMESTAMP.log"
done

# ---- SANITY CHECK ----
echo "--------------------------------------------------"
echo "🧪 Running T1c Sanity Check..."
python T1c-sanitycheck.py > "$LOG_DIR/T1c_$TIMESTAMP.log" 2>&1
echo "✅ T1c complete."

# ---- CLEAN METADATA ----
echo "--------------------------------------------------"
echo "🧹 Running T1b Clean Metadata + Edges..."
python T1b-clean_metadata_edges.py > "$LOG_DIR/T1b_$TIMESTAMP.log" 2>&1
echo "✅ T1b complete."

# ---- GRAPH ANALYSIS ----
echo "--------------------------------------------------"
echo "📊 Running T2 Graph Analysis..."
python T2-graph_analysis.py > "$LOG_DIR/T2_$TIMESTAMP.log" 2>&1
echo "✅ T2 complete."

# ---- CITATION INTENT ----
echo "--------------------------------------------------"
echo "🧬 Running T3 Citation Intent Classifier (no-train)..."
python T3-citation_intent_classifier.py --no-train > "$LOG_DIR/T3_$TIMESTAMP.log" 2>&1
echo "✅ T3 complete."

# ---- BIAS DETECTION ----
echo "--------------------------------------------------"
echo "⚖️ Running T4 Bias Detection..."
python T4-bias_detection.py > "$LOG_DIR/T4_$TIMESTAMP.log" 2>&1
echo "✅ T4 complete."

# ---- SUMMARY ----
echo "=================================================="
echo "🎯 CiteGraphLens Pipeline Finished Successfully!"
echo "📂 Logs saved in: $LOG_DIR"
echo "🕒 Completed at: $(date +"%Y-%m-%d_%H-%M-%S")"
echo "=================================================="
