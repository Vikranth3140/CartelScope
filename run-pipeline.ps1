# ==========================================================
# CiteGraphLens Full Pipeline Runner (Windows + RTX 3060)
# Author: Vikranth Udandarao
# ==========================================================

$ErrorActionPreference = "Stop"

# ---- CONFIG ----
$BASE_DIR = "C:\Users\vikra\OneDrive\Desktop\CiteGraphLens"
$VENV_PATH = "$BASE_DIR\venv"
$LOG_DIR = "$BASE_DIR\logs"
$TIMESTAMP = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

# ---- SETUP ----
if (!(Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR | Out-Null }

Write-Host "============================================"
Write-Host "CiteGraphLens Pipeline Starting ($TIMESTAMP)"
Write-Host "Base Directory: $BASE_DIR"
Write-Host "============================================"

# ---- ACTIVATE VENV ----
Write-Host "Activating virtual environment..."
& "$VENV_PATH\Scripts\Activate.ps1"
Set-Location $BASE_DIR

# ---- GPU CHECK ----
try {
    $gpuCheck = python -c "import torch; print('CUDA available' if torch.cuda.is_available() else 'CUDA NOT available')"
    Write-Host $gpuCheck
} catch {
    Write-Host "Could not check CUDA availability."
}

# ---- YEARLY FETCH ----
$years = 2020, 2021, 2022, 2023
foreach ($year in $years) {
    Write-Host "--------------------------------------------------"
    Write-Host "Fetching papers for year $year"
    python T1-cite_graph_lens.py `
        --cs-papers 500 `
        --bio-papers 500 `
        --med-papers 500 `
        --include-references `
        --email "vikranth22570@iiitd.ac.in" `
        --year $year `
        *> "$LOG_DIR\T1_${year}_$TIMESTAMP.log"

    Write-Host "Completed year $year — Log: $LOG_DIR\T1_${year}_$TIMESTAMP.log"
}

# ---- SANITY CHECK ----
Write-Host "--------------------------------------------------"
Write-Host "Running T1c Sanity Check..."
python .\T1c-sanitycheck.py *> "$LOG_DIR\T1c_$TIMESTAMP.log"
Write-Host "T1c complete."

# ---- CLEAN METADATA ----
Write-Host "--------------------------------------------------"
Write-Host "Running T1b Clean Metadata + Edges..."
python .\T1b-clean_metadata_edges.py *> "$LOG_DIR\T1b_$TIMESTAMP.log"
Write-Host "T1b complete."

# ---- GRAPH ANALYSIS ----
Write-Host "--------------------------------------------------"
Write-Host "Running T2 Graph Analysis..."
python .\T2-graph_analysis.py *> "$LOG_DIR\T2_$TIMESTAMP.log"
Write-Host "T2 complete."

# ---- CITATION INTENT ----
Write-Host "--------------------------------------------------"
Write-Host "Running T3 Citation Intent Classifier (no-train)..."
python .\T3-citation_intent_classifier.py --no-train *> "$LOG_DIR\T3_$TIMESTAMP.log"
Write-Host "T3 complete."

# ---- BIAS DETECTION ----
Write-Host "--------------------------------------------------"
Write-Host "Running T4 Bias Detection..."
python .\T4-bias_detection.py *> "$LOG_DIR\T4_$TIMESTAMP.log"
Write-Host "T4 complete."

# ---- SUMMARY ----
Write-Host "=================================================="
Write-Host "CiteGraphLens Pipeline Finished Successfully!"
Write-Host "Logs saved in: $LOG_DIR"
Write-Host "Completed at: $(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')"
Write-Host "=================================================="