"""Prepare a rare-class calibration teacher batch.

The first 2K pilot showed very few Contrast/Criticism examples and no
Perfunctory/Ceremonial examples. This script prepares a smaller calibration
batch that oversamples weak lexical-overlap edges and suspicious structural
contexts before scaling to a full teacher-label run.

Usage:
    PYTHONPATH=src python -m citation_cartels.annotation.prepare_calibration_batch \
        --config configs/cikm_middle.yaml \
        --target-labels 5000 \
        --deployment gpt-4-1-mini-batch
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from citation_cartels.annotation.prepare_teacher_batch import batch_request, join_paper_text


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "using",
    "based",
    "paper",
    "method",
    "approach",
    "results",
    "data",
    "model",
    "models",
    "study",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def token_set(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text or "") if token.lower() not in STOPWORDS}


def jaccard(a: str, b: str) -> float:
    left = token_set(a)
    right = token_set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def sample(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if frame.empty or count <= 0:
        return frame.head(0)
    return frame.sample(min(count, len(frame)), random_state=seed)


def low_similarity_edges(pools: pd.DataFrame, papers: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    candidates = pools[pools["pool"].isin(["random_global", "touches_top_candidate_communities"])].drop_duplicates("edge_id")
    candidates = sample(candidates, min(100_000, max(count * 20, count)), seed)
    with_text = join_paper_text(candidates, papers)
    citing_text = (with_text["citing_title"].fillna("") + " " + with_text["citing_abstract"].fillna("")).str.slice(0, 1200)
    cited_text = (with_text["cited_title"].fillna("") + " " + with_text["cited_abstract"].fillna("")).str.slice(0, 1200)
    with_text["lexical_jaccard"] = [jaccard(a, b) for a, b in zip(citing_text, cited_text)]
    return with_text.sort_values("lexical_jaccard").head(count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cikm_middle.yaml", type=Path)
    parser.add_argument("--target-labels", type=int, default=5000)
    parser.add_argument("--deployment", default="gpt-4-1-mini-batch")
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

    target = args.target_labels
    frames = [
        low_similarity_edges(pools, papers, int(target * 0.30), seed).assign(pool="low_similarity_candidate"),
        sample(pools[pools["pool"] == "reciprocal_edges"], int(target * 0.20), seed),
        sample(pools[pools["pool"] == "internal_high_inflation_communities"], int(target * 0.20), seed),
        sample(pools[pools["pool"] == "internal_dense_communities"], int(target * 0.15), seed),
        sample(pools[pools["pool"] == "random_global"], int(target * 0.15), seed),
    ]
    selected = pd.concat(frames, ignore_index=True).drop_duplicates("edge_id").head(target)
    if len(selected) < target:
        remaining = pools[~pools["edge_id"].isin(set(selected["edge_id"]))].drop_duplicates("edge_id")
        filler = sample(remaining, target - len(selected), seed + 1).assign(pool="calibration_filler")
        selected = pd.concat([selected, filler], ignore_index=True).drop_duplicates("edge_id").head(target)
    if "citing_title" not in selected.columns:
        selected = join_paper_text(selected, papers)
    else:
        missing_text = selected["citing_title"].isna() | selected["cited_title"].isna()
        if missing_text.any():
            selected = selected.drop(columns=[c for c in selected.columns if c.endswith("_title") or c.endswith("_abstract")])
            selected = join_paper_text(selected, papers)

    selected_path = labels_dir / f"teacher_calibration_selected_edges_{len(selected)}.parquet"
    batch_path = labels_dir / f"teacher_calibration_batch_{len(selected)}.jsonl"
    selected.to_parquet(selected_path, index=False)

    with batch_path.open("w", encoding="utf-8") as f:
        for _, edge in selected.iterrows():
            f.write(json.dumps(batch_request(edge, deployment=args.deployment)) + "\n")

    manifest = {
        "target_labels": target,
        "actual_labels": int(len(selected)),
        "deployment": args.deployment,
        "selected_edges_path": str(selected_path),
        "batch_jsonl_path": str(batch_path),
        "pool_counts": selected["pool"].value_counts().to_dict(),
        "seed": seed,
    }
    manifest_path = manifest_dir / "teacher_calibration_batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote selected edges to {selected_path}")
    print(f"Wrote batch JSONL to {batch_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
