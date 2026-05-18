"""Prepare a large teacher-label batch as Azure-safe chunks.

Azure rejects batch input files above 200 MB. This script builds one large,
unique set of unlabeled citation edges and writes several smaller JSONL chunks,
each with a matching selected-edge Parquet file for later parsing.

Usage:
    PYTHONPATH=src python -m citation_cartels.annotation.prepare_teacher_batch_chunks \
        --config configs/cikm_middle.yaml \
        --target-labels 125000 \
        --chunk-size 25000 \
        --deployment gpt-4-1-mini-batch \
        --exclude-labels data/labels/teacher_labels_pilot_plus_calibration.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from citation_cartels.annotation.prepare_calibration_batch import low_similarity_edges
from citation_cartels.annotation.prepare_teacher_batch import (
    batch_request,
    join_paper_text,
    load_excluded_edge_ids,
    sample_pool,
)


FINAL_POOL_FRACTIONS = {
    "low_similarity_candidate": 0.15,
    "random_global": 0.20,
    "internal_dense_communities": 0.20,
    "internal_high_inflation_communities": 0.15,
    "reciprocal_edges": 0.15,
    "touches_top_candidate_communities": 0.15,
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_final_edges(pools: pd.DataFrame, papers: pd.DataFrame, target_labels: int, seed: int) -> pd.DataFrame:
    selected = []
    low_similarity_count = int(target_labels * FINAL_POOL_FRACTIONS["low_similarity_candidate"])
    selected.append(low_similarity_edges(pools, papers, low_similarity_count, seed).assign(pool="low_similarity_candidate"))

    for pool_name, fraction in FINAL_POOL_FRACTIONS.items():
        if pool_name == "low_similarity_candidate":
            continue
        pool_frame = pools[pools["pool"] == pool_name]
        selected.append(sample_pool(pool_frame, int(target_labels * fraction), seed))

    combined = pd.concat(selected, ignore_index=True).drop_duplicates("edge_id")
    if len(combined) < target_labels:
        remaining = pools[~pools["edge_id"].isin(set(combined["edge_id"]))].drop_duplicates("edge_id")
        filler = sample_pool(remaining, target_labels - len(combined), seed + 1).assign(pool="final_filler")
        combined = pd.concat([combined, filler], ignore_index=True).drop_duplicates("edge_id")

    if len(combined) < target_labels:
        raise RuntimeError(f"Only found {len(combined):,} unique candidate edges for target {target_labels:,}.")

    selected_edges = combined.head(target_labels).copy()
    text_columns = [column for column in selected_edges.columns if column.endswith("_title") or column.endswith("_abstract")]
    if text_columns:
        selected_edges = selected_edges.drop(columns=text_columns)
    return join_paper_text(selected_edges, papers)


def flush_chunk(
    rows: list[dict[str, Any]],
    lines: list[str],
    labels_dir: Path,
    prefix: str,
    chunk_id: int,
) -> dict[str, Any]:
    selected_path = labels_dir / f"{prefix}_selected_edges_chunk_{chunk_id:03d}.parquet"
    batch_path = labels_dir / f"{prefix}_batch_chunk_{chunk_id:03d}.jsonl"

    pd.DataFrame(rows).to_parquet(selected_path, index=False)
    batch_path.write_text("".join(lines), encoding="utf-8")

    return {
        "chunk_id": chunk_id,
        "num_requests": len(rows),
        "selected_edges_path": str(selected_path),
        "batch_jsonl_path": str(batch_path),
        "bytes": batch_path.stat().st_size,
    }


def write_chunks(
    selected: pd.DataFrame,
    labels_dir: Path,
    prefix: str,
    deployment: str,
    chunk_size: int,
    max_bytes: int,
) -> list[dict[str, Any]]:
    chunks = []
    rows: list[dict[str, Any]] = []
    lines: list[str] = []
    current_bytes = 0
    chunk_id = 1

    for _, edge in selected.iterrows():
        line = json.dumps(batch_request(edge, deployment=deployment)) + "\n"
        line_bytes = len(line.encode("utf-8"))
        if rows and (len(rows) >= chunk_size or current_bytes + line_bytes > max_bytes):
            chunks.append(flush_chunk(rows, lines, labels_dir, prefix, chunk_id))
            rows = []
            lines = []
            current_bytes = 0
            chunk_id += 1

        rows.append(edge.to_dict())
        lines.append(line)
        current_bytes += line_bytes

    if rows:
        chunks.append(flush_chunk(rows, lines, labels_dir, prefix, chunk_id))

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cikm_middle.yaml", type=Path)
    parser.add_argument("--target-labels", type=int, default=125000)
    parser.add_argument("--chunk-size", type=int, default=25000)
    parser.add_argument("--max-bytes", type=int, default=180_000_000)
    parser.add_argument("--deployment", default="gpt-4-1-mini-batch")
    parser.add_argument("--exclude-labels", nargs="*", default=[], type=Path)
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    interim_dir = Path(config["paths"]["interim_dir"])
    labels_dir = Path(config["paths"]["labels_dir"])
    manifest_dir = Path(config["paths"]["manifest_dir"])
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    pools = pd.read_parquet(interim_dir / "annotation_sampling_pools.parquet")
    papers = pd.read_parquet(interim_dir / "papers.parquet")

    excluded_edge_ids = load_excluded_edge_ids(args.exclude_labels)
    if excluded_edge_ids:
        pools = pools[~pools["edge_id"].isin(excluded_edge_ids)].copy()

    selected = select_final_edges(pools, papers, args.target_labels, seed)
    prefix = args.prefix or f"teacher_final_{args.target_labels}"
    selected_path = labels_dir / f"{prefix}_selected_edges.parquet"
    selected.to_parquet(selected_path, index=False)

    chunks = write_chunks(
        selected=selected,
        labels_dir=labels_dir,
        prefix=prefix,
        deployment=args.deployment,
        chunk_size=args.chunk_size,
        max_bytes=args.max_bytes,
    )

    manifest = {
        "target_labels": args.target_labels,
        "actual_labels": int(len(selected)),
        "deployment": args.deployment,
        "chunk_size": args.chunk_size,
        "max_bytes": args.max_bytes,
        "selected_edges_path": str(selected_path),
        "pool_counts": selected["pool"].value_counts().to_dict(),
        "excluded_label_paths": [str(path) for path in args.exclude_labels],
        "excluded_edge_count": len(excluded_edge_ids),
        "chunks": chunks,
        "seed": seed,
    }
    manifest_path = manifest_dir / f"{prefix}_chunks_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"Wrote chunk manifest to {manifest_path}")


if __name__ == "__main__":
    main()
