"""Merge multiple parsed teacher-label Parquet files.

Later batches may intentionally overlap earlier pilots. This helper keeps one
row per edge_id and prefers rows from later files, so higher-quality calibration
or full-batch labels can replace pilot labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cikm_middle.yaml", type=Path)
    parser.add_argument("--labels", nargs="+", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    labels_dir = Path(config["paths"]["labels_dir"])
    manifest_dir = Path(config["paths"]["manifest_dir"])
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for order, labels_path in enumerate(args.labels):
        frame = pd.read_parquet(labels_path).copy()
        frame["_source_file"] = str(labels_path)
        frame["_source_order"] = order
        frames.append(frame)

    merged = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["edge_id", "_source_order"])
        .drop_duplicates("edge_id", keep="last")
        .drop(columns=["_source_order"])
    )

    output_path = args.output or labels_dir / "teacher_labels_merged.parquet"
    merged.to_parquet(output_path, index=False)

    manifest: dict[str, Any] = {
        "output": str(output_path),
        "input_files": [str(path) for path in args.labels],
        "input_rows": int(sum(len(frame) for frame in frames)),
        "output_rows": int(len(merged)),
        "duplicates_removed": int(sum(len(frame) for frame in frames) - len(merged)),
        "label_counts": merged["label"].value_counts().to_dict(),
    }
    manifest_path = manifest_dir / f"{output_path.stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"Wrote merged labels to {output_path}")


if __name__ == "__main__":
    main()
