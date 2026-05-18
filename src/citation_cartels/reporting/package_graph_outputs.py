"""Package final graph-analysis outputs for paper writing and download."""

from __future__ import annotations

import argparse
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_OUTPUTS = [
    "baseline_comparison.csv",
    "baseline_precision_at_k.csv",
    "cartel_community_scores_final.csv",
    "cartel_edge_excision_final.png",
    "cci_ablation.csv",
    "cci_distribution_final.png",
    "excision_results.csv",
    "excision_target_edges_manifest.json",
    "final_graph_analysis_manifest.json",
    "intent_distribution_final.csv",
    "intent_distribution_final.png",
    "top_communities_final.csv",
    "trust_pagerank_examples.json",
    "trust_pagerank_shifts.csv",
]


def file_record(path: Path, base_dir: Path) -> dict[str, Any]:
    path = path.resolve()
    base_dir = base_dir.resolve()
    return {
        "path": str(path.relative_to(base_dir)),
        "bytes": path.stat().st_size,
        "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-output-dir", default="graph_analysis_outputs", type=Path)
    parser.add_argument("--reports-dir", default="reports/graph_analysis_cpu_vm", type=Path)
    parser.add_argument("--artifacts-dir", default="artifacts", type=Path)
    parser.add_argument("--package-name", default="cikm_graph_analysis_outputs.tar.gz")
    args = parser.parse_args()

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    package_path = args.artifacts_dir / args.package_name
    manifest_path = args.artifacts_dir / "cikm_graph_analysis_artifact_manifest.json"

    missing = [name for name in EXPECTED_OUTPUTS if not (args.graph_output_dir / name).exists()]
    files = []
    if args.graph_output_dir.exists():
        files.extend(path for path in sorted(args.graph_output_dir.rglob("*")) if path.is_file())
    if args.reports_dir.exists():
        files.extend(path for path in sorted(args.reports_dir.rglob("*")) if path.is_file())

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package_path": str(package_path),
        "graph_output_dir": str(args.graph_output_dir),
        "reports_dir": str(args.reports_dir),
        "missing_expected_outputs": missing,
        "files": [file_record(path, Path.cwd()) for path in files],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    files.append(manifest_path)

    with tarfile.open(package_path, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.resolve().relative_to(Path.cwd().resolve())))

    print(json.dumps(payload, indent=2))
    print(f"Wrote artifact package to {package_path}")
    if missing:
        raise SystemExit(f"Missing expected outputs: {missing}")


if __name__ == "__main__":
    main()
