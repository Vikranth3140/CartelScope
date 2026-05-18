"""Combine teacher labels and student inference shards into one semantic edge table.

Teacher labels are treated as the highest-priority source. Student inference
shards are then added in the order supplied on the command line. The output is a
canonical table for downstream weighted graph, CCI, baseline, and excision
analysis.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml


OUTPUT_COLUMNS = [
    "edge_id",
    "citing_id",
    "cited_id",
    "predicted_label",
    "label_confidence",
    "semantic_weight",
    "semantic_source",
    "model_version",
    "source_path",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_teacher_labels(paths: list[Path], weights: dict[str, float], priority: int) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        required = ["edge_id", "citing_id", "cited_id", "label", "confidence"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        normalized = frame[required].rename(
            columns={
                "label": "predicted_label",
                "confidence": "label_confidence",
            }
        )
        normalized["semantic_weight"] = normalized["predicted_label"].map(weights).astype(float)
        normalized["semantic_source"] = "teacher"
        normalized["model_version"] = "azure_openai_teacher"
        normalized["source_path"] = str(path)
        normalized["_priority"] = priority
        frames.append(normalized)
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS + ["_priority"])
    return pd.concat(frames, ignore_index=True)


def load_student_dir(directory: Path, source_name: str, priority: int) -> pd.DataFrame:
    frames = []
    for path in sorted(directory.glob("part_*.parquet")):
        frame = pd.read_parquet(path)
        required = ["edge_id", "citing_id", "cited_id", "predicted_label", "label_confidence", "semantic_weight"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        normalized = frame[required].copy()
        normalized["semantic_source"] = source_name
        normalized["model_version"] = frame["model_version"].iloc[0] if "model_version" in frame.columns and len(frame) else ""
        normalized["source_path"] = str(path)
        normalized["_priority"] = priority
        frames.append(normalized)
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS + ["_priority"])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cikm_middle.yaml", type=Path)
    parser.add_argument("--teacher-labels", nargs="*", default=[], type=Path)
    parser.add_argument("--student-dir", action="append", default=[], type=Path)
    parser.add_argument("--student-source", action="append", default=[])
    parser.add_argument("--output", default="data/processed/semantic_edges_combined.parquet", type=Path)
    args = parser.parse_args()

    if args.student_source and len(args.student_source) != len(args.student_dir):
        raise ValueError("--student-source must be supplied once per --student-dir, or not at all.")
    student_sources = args.student_source or [f"student_{idx}" for idx, _ in enumerate(args.student_dir, start=1)]

    config = load_config(args.config)
    weights = {str(label): float(weight) for label, weight in config["annotation"]["weights"].items()}
    manifest_dir = Path(config["paths"]["manifest_dir"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for idx, (directory, source_name) in enumerate(zip(args.student_dir, student_sources), start=1):
        frames.append(load_student_dir(directory, source_name, idx))
    # Teacher labels win over any duplicate student predictions.
    frames.append(load_teacher_labels(args.teacher_labels, weights, priority=10_000))

    combined_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLUMNS + ["_priority"])
    input_rows = int(len(combined_raw))
    combined = (
        combined_raw.sort_values(["edge_id", "_priority"])
        .drop_duplicates("edge_id", keep="last")
        .drop(columns=["_priority"])
    )
    combined = combined[OUTPUT_COLUMNS]
    combined.to_parquet(args.output, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(args.output),
        "input_rows": input_rows,
        "output_rows": int(len(combined)),
        "duplicates_removed": int(input_rows - len(combined)),
        "teacher_label_paths": [str(path) for path in args.teacher_labels],
        "student_dirs": [str(path) for path in args.student_dir],
        "student_sources": student_sources,
        "label_counts": combined["predicted_label"].value_counts().to_dict() if len(combined) else {},
        "source_counts": combined["semantic_source"].value_counts().to_dict() if len(combined) else {},
    }
    manifest_path = manifest_dir / f"{args.output.stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"Wrote combined semantic edges to {args.output}")


if __name__ == "__main__":
    main()
