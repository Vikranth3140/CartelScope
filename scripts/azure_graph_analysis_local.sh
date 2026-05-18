#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-help}"

RG="${RG:-ns-cikm-graph-rg}"
LOC="${LOC:-centralindia}"
VM="${VM:-ns-graph-e64}"
SIZE="${SIZE:-Standard_E64ds_v5}"
ADMIN="${ADMIN:-azureuser}"
PROJECT_DIR="${PROJECT_DIR:-/Users/pratyushgupta/Desktop/NS/NS-Project}"
REMOTE_DIR="${REMOTE_DIR:-~/NS-Project}"
JOB_SCRIPT="${JOB_SCRIPT:-scripts/run_graph_analysis_cpu_vm.sh}"
TMUX_SESSION="${TMUX_SESSION:-graph-analysis}"

vm_ip() {
  az vm show -d -g "${RG}" -n "${VM}" --query publicIps -o tsv
}

usage() {
  cat <<'EOF'
Usage:
  scripts/azure_graph_analysis_local.sh create
  scripts/azure_graph_analysis_local.sh upload
  scripts/azure_graph_analysis_local.sh start
  scripts/azure_graph_analysis_local.sh status
  scripts/azure_graph_analysis_local.sh logs
  scripts/azure_graph_analysis_local.sh download
  scripts/azure_graph_analysis_local.sh deallocate
  scripts/azure_graph_analysis_local.sh destroy
  scripts/azure_graph_analysis_local.sh all

Optional environment overrides:
  RG, LOC, VM, SIZE, ADMIN, PROJECT_DIR, JOB_SCRIPT, TMUX_SESSION

Recommended default:
  SIZE=Standard_E64ds_v5 LOC=centralindia

Fallback examples:
  SIZE=Standard_E48ds_v5 VM=ns-graph-e48 scripts/azure_graph_analysis_local.sh all
  SIZE=Standard_E32ds_v5 VM=ns-graph-e32 scripts/azure_graph_analysis_local.sh all

Parallel stability example:
  RG=ns-cikm-stability-rg LOC=southindia VM=ns-stability-e64 JOB_SCRIPT=scripts/run_cci_seed_stability_cpu_vm.sh TMUX_SESSION=seed-stability scripts/azure_graph_analysis_local.sh all
EOF
}

create_vm() {
  az group create --name "${RG}" --location "${LOC}"
  az vm create \
    --resource-group "${RG}" \
    --name "${VM}" \
    --location "${LOC}" \
    --image Ubuntu2204 \
    --size "${SIZE}" \
    --admin-username "${ADMIN}" \
    --generate-ssh-keys \
    --os-disk-size-gb 512 \
    --storage-sku Premium_LRS \
    --public-ip-sku Standard
  echo "VM IP: $(vm_ip)"
}

upload_inputs() {
  cd "${PROJECT_DIR}"
  local ip
  ip="${IP:-$(vm_ip)}"
  ssh "${ADMIN}@${ip}" "mkdir -p ${REMOTE_DIR}/data/processed ${REMOTE_DIR}/data/interim ${REMOTE_DIR}/artifacts ${REMOTE_DIR}/reports"
  rsync -avP configs requirements.txt src scripts "${ADMIN}@${ip}:${REMOTE_DIR}/"
  rsync -avP data/interim/ "${ADMIN}@${ip}:${REMOTE_DIR}/data/interim/"
  rsync -avP data/processed/semantic_edges_combined.parquet "${ADMIN}@${ip}:${REMOTE_DIR}/data/processed/"
}

start_job() {
  local ip
  ip="${IP:-$(vm_ip)}"
  ssh "${ADMIN}@${ip}" "JOB_SCRIPT='${JOB_SCRIPT}' TMUX_SESSION='${TMUX_SESSION}' bash -s" <<'EOF'
set -euo pipefail
cd ~/NS-Project
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip tmux htop
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install pandas pyarrow networkx matplotlib numpy PyYAML tqdm scikit-learn scipy
if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "${TMUX_SESSION} tmux session already exists. Attach with: tmux attach -t ${TMUX_SESSION}"
else
  tmux new -d -s "${TMUX_SESSION}" "bash ${JOB_SCRIPT}"
fi
tmux ls
EOF
}

status_job() {
  local ip
  ip="${IP:-$(vm_ip)}"
  ssh "${ADMIN}@${ip}" "tmux ls || true; for log in ~/NS-Project/reports/*/*.log; do test -f \"\$log\" && echo \"--- \$log\" && tail -n 30 \"\$log\"; done"
}

logs_job() {
  local ip
  ip="${IP:-$(vm_ip)}"
  ssh "${ADMIN}@${ip}" "tail -f ~/NS-Project/reports/*/*.log"
}

download_outputs() {
  cd "${PROJECT_DIR}"
  mkdir -p artifacts graph_analysis_outputs data/processed reports/graph_analysis_cpu_vm
  local ip
  ip="${IP:-$(vm_ip)}"
  ssh "${ADMIN}@${ip}" 'mkdir -p ~/NS-Project/artifacts ~/NS-Project/graph_analysis_outputs ~/NS-Project/data/processed ~/NS-Project/reports'
  rsync -avP "${ADMIN}@${ip}:~/NS-Project/artifacts/" artifacts/
  rsync -avP "${ADMIN}@${ip}:~/NS-Project/graph_analysis_outputs/" graph_analysis_outputs/
  rsync -avP "${ADMIN}@${ip}:~/NS-Project/data/processed/" data/processed/
  rsync -avP "${ADMIN}@${ip}:~/NS-Project/reports/" reports/
}

case "${ACTION}" in
  create)
    create_vm
    ;;
  upload)
    upload_inputs
    ;;
  start)
    start_job
    ;;
  status)
    status_job
    ;;
  logs)
    logs_job
    ;;
  download)
    download_outputs
    ;;
  deallocate)
    az vm deallocate --resource-group "${RG}" --name "${VM}"
    ;;
  destroy)
    az group delete --name "${RG}" --yes --no-wait
    ;;
  all)
    create_vm
    upload_inputs
    start_job
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
