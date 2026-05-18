"""Build canonical paper and edge tables from the snowball-sampled JSON.

Usage:
    PYTHONPATH=src python -m citation_cartels.data.build_tables \
        --config configs/cikm_middle.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import ijson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm


EDGE_CHUNK_SIZE = 500_000


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruct_abstract(indexed_abstract: Any) -> str:
    """Reconstruct plain-text abstract from DBLP/OpenAlex-style InvertedIndex."""
    if not isinstance(indexed_abstract, dict) or "InvertedIndex" not in indexed_abstract:
        return ""

    inv_idx = indexed_abstract["InvertedIndex"]
    length = indexed_abstract.get("IndexLength", 0)
    if length == 0:
        max_idx = max((idx for indices in inv_idx.values() for idx in indices), default=-1)
        length = max_idx + 1
    if length == 0:
        return ""

    words = [""] * length
    for word, indices in inv_idx.items():
        for idx in indices:
            if 0 <= idx < length:
                words[idx] = word
    return " ".join(words).strip()


def top_field_of_study(fos: Any) -> str:
    if not isinstance(fos, list) or not fos:
        return "Unknown"
    try:
        best = max(fos, key=lambda item: item.get("w", 0))
        return str(best.get("name") or "Unknown")
    except Exception:
        return "Unknown"


def edge_id(source: str, target: str) -> str:
    return hashlib.sha1(f"{source}->{target}".encode("utf-8")).hexdigest()


def flush_edges(
    edge_rows: list[dict[str, Any]],
    writer: Optional[pq.ParquetWriter],
    path: Path,
) -> pq.ParquetWriter:
    table = pa.Table.from_pylist(edge_rows)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    edge_rows.clear()
    return writer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cikm_middle.yaml", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]

    raw_dataset = Path(paths["raw_dataset"])
    interim_dir = Path(paths["interim_dir"])
    manifest_dir = Path(paths["manifest_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dataset.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_dataset}. "
            "Place the 500K snowball JSON there before running Phase 1."
        )

    print(f"Reading papers from {raw_dataset}")
    paper_rows: list[dict[str, Any]] = []
    references_by_id: dict[str, list[str]] = {}

    with raw_dataset.open("rb") as f:
        for paper in tqdm(ijson.items(f, "item"), desc="Parsing papers", unit="paper"):
            pid_raw = paper.get("id")
            if pid_raw is None:
                continue
            pid = str(pid_raw)
            refs = [str(ref) for ref in paper.get("references", []) if ref is not None]
            references_by_id[pid] = refs

            paper_rows.append(
                {
                    "paper_id": pid,
                    "title": paper.get("title", "") or "",
                    "abstract": reconstruct_abstract(paper.get("indexed_abstract")),
                    "year": paper.get("year"),
                    "n_citation": paper.get("n_citation", 0),
                    "top_fos": top_field_of_study(paper.get("fos")),
                    "out_reference_count_raw": len(refs),
                }
            )

    paper_ids = {row["paper_id"] for row in paper_rows}
    papers_path = interim_dir / "papers.parquet"
    metadata_path = interim_dir / "paper_metadata.parquet"
    edges_path = interim_dir / "edges.parquet"

    papers_df = pd.DataFrame(paper_rows)
    papers_df.to_parquet(papers_path, index=False)
    papers_df[["paper_id", "title", "year", "n_citation", "top_fos"]].to_parquet(metadata_path, index=False)

    print(f"Wrote {len(papers_df):,} papers to {papers_path}")
    print("Building closed-world edge table")

    writer: Optional[pq.ParquetWriter] = None
    edge_rows: list[dict[str, Any]] = []
    edge_count = 0

    for source, refs in tqdm(references_by_id.items(), desc="Closed-world edges", unit="paper"):
        for target in refs:
            if target not in paper_ids:
                continue
            edge_rows.append(
                {
                    "edge_id": edge_id(source, target),
                    "citing_id": source,
                    "cited_id": target,
                }
            )
            edge_count += 1
            if len(edge_rows) >= EDGE_CHUNK_SIZE:
                writer = flush_edges(edge_rows, writer, edges_path)

    if edge_rows:
        writer = flush_edges(edge_rows, writer, edges_path)
    if writer is not None:
        writer.close()

    node_count = len(paper_ids)
    edge_node_ratio = edge_count / node_count if node_count else 0.0
    manifest = {
        "raw_dataset": str(raw_dataset),
        "raw_dataset_sha256": sha256_file(raw_dataset),
        "papers_path": str(papers_path),
        "metadata_path": str(metadata_path),
        "edges_path": str(edges_path),
        "node_count": node_count,
        "closed_world_edge_count": edge_count,
        "edge_node_ratio": edge_node_ratio,
        "expected_nodes": config.get("dataset", {}).get("expected_nodes"),
        "expected_edges": config.get("dataset", {}).get("expected_edges"),
        "seed": config.get("project", {}).get("seed"),
    }

    manifest_path = manifest_dir / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {edge_count:,} edges to {edges_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
