"""Parse teacher batch JSONL responses into a clean label table.

Expected input is an OpenAI/Azure Batch-style JSONL where each line includes
`custom_id` and a response body containing a JSON object with `label` and
`confidence`.

Usage:
    PYTHONPATH=src python -m citation_cartels.annotation.parse_teacher_batch \
        --config configs/cikm_middle.yaml \
        --responses data/labels/teacher_batch_2000_results.jsonl \
        --selected-edges data/labels/teacher_selected_edges_2000.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_message_content(record: dict[str, Any]) -> Optional[str]:
    try:
        choices = record["response"]["body"]["choices"]
        return choices[0]["message"]["content"]
    except Exception:
        return None


def normalize_label(label: str, allowed_labels: list[str]) -> Optional[str]:
    label_lower = label.strip().lower()
    for allowed in allowed_labels:
        if allowed.lower() == label_lower:
            return allowed
    for allowed in allowed_labels:
        if allowed.lower() in label_lower:
            return allowed
    return None


def parse_content(content: str, allowed_labels: list[str]) -> tuple[Optional[str], Optional[float], Optional[str]]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, None, f"json_decode_error: {exc}"

    label_raw = str(payload.get("label", ""))
    label = normalize_label(label_raw, allowed_labels)
    if label is None:
        return None, None, f"unknown_label: {label_raw}"

    confidence_raw = payload.get("confidence", None)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))

    return label, confidence, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cikm_middle.yaml", type=Path)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--selected-edges", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--failures-output", default=None, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    allowed_labels = list(config["annotation"]["labels"])
    labels_dir = Path(config["paths"]["labels_dir"])
    labels_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or labels_dir / "teacher_labels_parsed.parquet"

    rows = []
    failures = []
    with args.responses.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            edge_id = record.get("custom_id")
            content = extract_message_content(record)
            if content is None:
                failures.append({"line_no": line_no, "edge_id": edge_id, "error": "missing_message_content"})
                continue
            label, confidence, error = parse_content(content, allowed_labels)
            if error:
                failures.append({"line_no": line_no, "edge_id": edge_id, "error": error, "content": content})
                continue
            rows.append({"edge_id": edge_id, "label": label, "confidence": confidence})

    selected_edges = pd.read_parquet(args.selected_edges)
    labels = pd.DataFrame(rows)
    merged = selected_edges.merge(labels, on="edge_id", how="inner")
    merged.to_parquet(output_path, index=False)

    failure_path = args.failures_output or labels_dir / f"{output_path.stem}_parse_failures.json"
    failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    print(f"Parsed {len(merged):,} labels to {output_path}")
    print(f"Parse failures: {len(failures):,} written to {failure_path}")


if __name__ == "__main__":
    main()
