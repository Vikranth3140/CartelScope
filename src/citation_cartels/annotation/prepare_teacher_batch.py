"""Prepare Azure/OpenAI Batch API JSONL for teacher citation-intent labels.

This script does not call any API. It samples from Phase 2 structural pools,
joins paper text, and writes a JSONL request file plus the selected edge table.

Usage:
    PYTHONPATH=src python -m citation_cartels.annotation.prepare_teacher_batch \
        --config configs/cikm_middle.yaml --target-labels 50000 \
        --exclude-labels data/labels/teacher_labels_pilot_plus_calibration.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from citation_cartels.annotation.prompt_templates import citation_intent_prompt


POOL_FRACTIONS = {
    "random_global": 0.25,
    "internal_dense_communities": 0.20,
    "internal_high_inflation_communities": 0.15,
    "reciprocal_edges": 0.10,
    "touches_top_candidate_communities": 0.10,
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_pool(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if frame.empty or count <= 0:
        return frame.head(0)
    return frame.sample(min(count, len(frame)), random_state=seed)


def load_excluded_edge_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        frame = pd.read_parquet(path, columns=["edge_id"])
        excluded.update(frame["edge_id"].dropna().astype(str))
    return excluded


def select_edges(pools: pd.DataFrame, target_labels: int, seed: int) -> pd.DataFrame:
    selected = []
    for pool_name, fraction in POOL_FRACTIONS.items():
        pool_frame = pools[pools["pool"] == pool_name]
        selected.append(sample_pool(pool_frame, int(target_labels * fraction), seed))

    combined = pd.concat(selected, ignore_index=True) if selected else pools.head(0)
    combined = combined.drop_duplicates("edge_id")

    if len(combined) < target_labels:
        remaining = pools[~pools["edge_id"].isin(set(combined["edge_id"]))].drop_duplicates("edge_id")
        fill = sample_pool(remaining, target_labels - len(combined), seed)
        combined = pd.concat([combined, fill], ignore_index=True).drop_duplicates("edge_id")

    return combined.head(target_labels)


def join_paper_text(selected: pd.DataFrame, papers: pd.DataFrame) -> pd.DataFrame:
    paper_text = papers[["paper_id", "title", "abstract"]].copy()
    result = selected.merge(
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
        result[column] = result[column].fillna("")
    return result


def batch_request(edge: pd.Series, deployment: str) -> dict[str, Any]:
    prompt = citation_intent_prompt(
        citing_title=edge["citing_title"],
        citing_abstract=edge["citing_abstract"],
        cited_title=edge["cited_title"],
        cited_abstract=edge["cited_abstract"],
    )
    return {
        "custom_id": edge["edge_id"],
        "method": "POST",
        "url": "/chat/completions",
        "body": {
            "model": deployment,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful scientific citation-intent annotator.",
                },
                {"role": "user", "content": prompt},
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cikm_middle.yaml", type=Path)
    parser.add_argument("--target-labels", type=int, default=None)
    parser.add_argument("--deployment", default=None)
    parser.add_argument("--exclude-labels", nargs="*", default=[], type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    target_labels = args.target_labels or int(config["annotation"]["target_teacher_labels"])
    deployment = args.deployment or "gpt-4.1-mini"

    interim_dir = Path(config["paths"]["interim_dir"])
    labels_dir = Path(config["paths"]["labels_dir"])
    manifest_dir = Path(config["paths"]["manifest_dir"])
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    pools_path = interim_dir / "annotation_sampling_pools.parquet"
    papers_path = interim_dir / "papers.parquet"
    if not pools_path.exists():
        raise FileNotFoundError(f"Missing {pools_path}. Run Phase 2 sampling pools first.")
    if not papers_path.exists():
        raise FileNotFoundError(f"Missing {papers_path}. Run Phase 1 table building first.")

    pools = pd.read_parquet(pools_path)
    papers = pd.read_parquet(papers_path)
    excluded_edge_ids = load_excluded_edge_ids(args.exclude_labels)
    if excluded_edge_ids:
        pools = pools[~pools["edge_id"].isin(excluded_edge_ids)].copy()
    selected = select_edges(pools, target_labels=target_labels, seed=seed)
    selected_with_text = join_paper_text(selected, papers)

    selected_path = labels_dir / f"teacher_selected_edges_{len(selected_with_text)}.parquet"
    batch_path = labels_dir / f"teacher_batch_{len(selected_with_text)}.jsonl"
    selected_with_text.to_parquet(selected_path, index=False)

    with batch_path.open("w", encoding="utf-8") as f:
        for _, edge in selected_with_text.iterrows():
            f.write(json.dumps(batch_request(edge, deployment=deployment)) + "\n")

    manifest = {
        "target_labels": target_labels,
        "actual_labels": int(len(selected_with_text)),
        "deployment": deployment,
        "selected_edges_path": str(selected_path),
        "batch_jsonl_path": str(batch_path),
        "pool_counts": selected_with_text["pool"].value_counts().to_dict(),
        "excluded_label_paths": [str(path) for path in args.exclude_labels],
        "excluded_edge_count": len(excluded_edge_ids),
        "seed": seed,
    }
    manifest_path = manifest_dir / "teacher_batch_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote selected edges to {selected_path}")
    print(f"Wrote batch JSONL to {batch_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
