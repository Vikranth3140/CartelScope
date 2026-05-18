"""Run SciBERT citation-intent inference over graph edges in shards.

This is Phase 6 of the CIKM plan. It assumes `train_student.py` has already
saved a fine-tuned model directory with tokenizer files.

Usage:
    PYTHONPATH=src python -m citation_cartels.annotation.infer_student \
        --config configs/cikm_middle.yaml \
        --model-dir models/scibert_citation_intent_final/best_model
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def join_text(edges: pd.DataFrame, papers: pd.DataFrame) -> pd.DataFrame:
    paper_text = papers[["paper_id", "title", "abstract"]].copy()
    merged = edges.merge(
        paper_text.rename(
            columns={
                "paper_id": "citing_id",
                "title": "citing_title",
                "abstract": "citing_abstract",
            }
        ),
        on="citing_id",
        how="left",
    ).merge(
        paper_text.rename(
            columns={
                "paper_id": "cited_id",
                "title": "cited_title",
                "abstract": "cited_abstract",
            }
        ),
        on="cited_id",
        how="left",
    )
    for column in ["citing_title", "citing_abstract", "cited_title", "cited_abstract"]:
        merged[column] = merged[column].fillna("")
    return merged


def predict_chunk(
    frame: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    labels: list[str],
    weights: dict[str, float],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> pd.DataFrame:
    citing_text = (frame["citing_title"] + ". " + frame["citing_abstract"]).tolist()
    cited_text = (frame["cited_title"] + ". " + frame["cited_abstract"]).tolist()

    pred_ids: list[int] = []
    confidences: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            encoded = tokenizer(
                citing_text[start : start + batch_size],
                cited_text[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted = probabilities.max(dim=-1)
            pred_ids.extend(predicted.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())

    predicted_labels = [labels[pred_id] for pred_id in pred_ids]
    result = frame[["edge_id", "citing_id", "cited_id"]].copy()
    result["predicted_label"] = predicted_labels
    result["label_confidence"] = confidences
    result["semantic_weight"] = [float(weights[label]) for label in predicted_labels]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cikm_middle.yaml", type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--edges", default=None, type=Path)
    parser.add_argument("--papers", default=None, type=Path)
    parser.add_argument("--output-dir", default="data/processed/typed_edges_shards", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--edge-chunk-size", type=int, default=100_000)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    labels = list(config["annotation"]["labels"])
    weights = {str(label): float(weight) for label, weight in config["annotation"]["weights"].items()}
    max_length = int(config["student_model"]["max_length"])
    interim_dir = Path(config["paths"]["interim_dir"])
    manifest_dir = Path(config["paths"]["manifest_dir"])
    edges_path = args.edges or interim_dir / "edges.parquet"
    papers_path = args.papers or interim_dir / "papers.parquet"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    papers = pd.read_parquet(papers_path, columns=["paper_id", "title", "abstract"])

    outputs = []
    parquet_file = pq.ParquetFile(edges_path)
    for shard_idx, batch in enumerate(parquet_file.iter_batches(batch_size=args.edge_chunk_size)):
        if args.max_shards is not None and shard_idx >= args.max_shards:
            break
        shard_path = args.output_dir / f"part_{shard_idx:04d}.parquet"
        if args.skip_existing and shard_path.exists():
            row_count = pq.read_metadata(shard_path).num_rows
            outputs.append({"shard": shard_idx, "path": str(shard_path), "rows": int(row_count), "skipped": True})
            print(f"Skipping existing shard {shard_path}")
            continue
        edges = batch.to_pandas()
        with_text = join_text(edges, papers)
        predictions = predict_chunk(
            with_text,
            tokenizer=tokenizer,
            model=model,
            labels=labels,
            weights=weights,
            batch_size=args.batch_size,
            max_length=max_length,
            device=device,
        )
        predictions["model_version"] = str(args.model_dir)
        predictions.to_parquet(shard_path, index=False)
        outputs.append({"shard": shard_idx, "path": str(shard_path), "rows": int(len(predictions)), "skipped": False})
        print(f"Wrote {len(predictions):,} typed edges to {shard_path}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(args.model_dir),
        "edges_path": str(edges_path),
        "papers_path": str(papers_path),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "batch_size": args.batch_size,
        "edge_chunk_size": args.edge_chunk_size,
        "outputs": outputs,
        "total_rows": int(sum(item["rows"] for item in outputs)),
    }
    manifest_path = manifest_dir / "student_inference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote inference manifest to {manifest_path}")


if __name__ == "__main__":
    main()
