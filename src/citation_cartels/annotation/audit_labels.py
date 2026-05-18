"""Audit teacher labels before student training.

This is intentionally lightweight: it checks class coverage, parse usability,
confidence, and whether the label set is large enough for stratified
train/validation/test splits.
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


def audit(labels_path: Path, allowed_labels: list[str], min_test_per_class: int) -> dict[str, Any]:
    frame = pd.read_parquet(labels_path)
    counts = frame["label"].value_counts().reindex(allowed_labels, fill_value=0)
    shares = (counts / len(frame)).fillna(0.0)

    confidence = {}
    if "confidence" in frame.columns:
        confidence = {
            "mean": float(frame["confidence"].mean()),
            "median": float(frame["confidence"].median()),
            "min": float(frame["confidence"].min()),
            "max": float(frame["confidence"].max()),
        }

    missing = [label for label, count in counts.items() if int(count) == 0]
    weak = [label for label, count in counts.items() if 0 < int(count) < min_test_per_class]

    return {
        "labels_path": str(labels_path),
        "rows": int(len(frame)),
        "label_counts": {label: int(count) for label, count in counts.items()},
        "label_shares": {label: round(float(share), 4) for label, share in shares.items()},
        "confidence": confidence,
        "missing_labels": missing,
        "underpowered_labels": weak,
        "ready_for_student_training": not missing and not weak,
        "minimum_recommended_count_per_label": min_test_per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cikm_middle.yaml", type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--min-test-per-class", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    payload = audit(args.labels, list(config["annotation"]["labels"]), args.min_test_per_class)

    output_path = args.output or Path("reports") / f"{args.labels.stem}_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Wrote audit to {output_path}")


if __name__ == "__main__":
    main()
