"""Prepare a targeted semantic-inference edge set.

This is the fallback/accelerated Phase 6 path when full-graph inference is too
slow for a local laptop. The goal is not random subsampling; it is to type the
edges that matter most for citation-cartel detection plus enough controls to
make comparisons credible.

The selected set prioritizes:
- reciprocal edges,
- internal edges in high-inflation communities,
- internal edges in dense communities,
- edges touching top candidate communities,
- low lexical-overlap candidate edges,
- global random controls.

The output can be passed directly to `citation_cartels.annotation.infer_student`
with `--edges`.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml


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


DEFAULT_FRACTIONS = {
    "reciprocal_edges": 0.05,
    "internal_high_inflation_communities": 0.25,
    "internal_dense_communities": 0.20,
    "touches_top_candidate_communities": 0.20,
    "low_similarity_candidate": 0.15,
    "random_global_control": 0.15,
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def token_set(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text or "") if token.lower() not in STOPWORDS}


def jaccard(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def sample(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if frame.empty or count <= 0:
        return frame.head(0)
    return frame.sample(min(count, len(frame)), random_state=seed)


def typed_edge_ids(typed_dir: Path) -> set[str]:
    if not typed_dir.exists():
        return set()
    ids: set[str] = set()
    for path in sorted(typed_dir.glob("part_*.parquet")):
        try:
            frame = pd.read_parquet(path, columns=["edge_id"])
        except Exception:
            continue
        ids.update(frame["edge_id"].dropna().astype(str))
    return ids


def label_edge_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=["edge_id"])
        ids.update(frame["edge_id"].dropna().astype(str))
    return ids


def select_from_pool(
    pools: pd.DataFrame,
    pool_name: str,
    count: int,
    seed: int,
    used_ids: set[str],
) -> pd.DataFrame:
    frame = pools[(pools["pool"] == pool_name) & ~pools["edge_id"].isin(used_ids)].drop_duplicates("edge_id")
    selected = sample(frame, count, seed).copy()
    used_ids.update(selected["edge_id"].astype(str))
    return selected.assign(selection_group=pool_name)


def join_text(edges: pd.DataFrame, papers: pd.DataFrame) -> pd.DataFrame:
    paper_text = papers[["paper_id", "title", "abstract"]].copy()
    result = edges.merge(
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


def select_low_similarity(
    pools: pd.DataFrame,
    papers: pd.DataFrame,
    count: int,
    seed: int,
    used_ids: set[str],
    candidate_multiplier: int,
) -> pd.DataFrame:
    candidate_pool_names = ["touches_top_candidate_communities", "internal_high_inflation_communities"]
    candidates = pools[pools["pool"].isin(candidate_pool_names) & ~pools["edge_id"].isin(used_ids)]
    candidates = candidates.drop_duplicates("edge_id")
    candidates = sample(candidates, min(len(candidates), max(count * candidate_multiplier, count)), seed + 17)
    with_text = join_text(candidates, papers)
    citing_text = (with_text["citing_title"] + " " + with_text["citing_abstract"]).str.slice(0, 1200)
    cited_text = (with_text["cited_title"] + " " + with_text["cited_abstract"]).str.slice(0, 1200)
    with_text["lexical_jaccard"] = [jaccard(a, b) for a, b in zip(citing_text, cited_text)]
    selected = with_text.sort_values("lexical_jaccard").head(count).copy()
    used_ids.update(selected["edge_id"].astype(str))
    keep = ["edge_id", "citing_id", "cited_id", "pool", "lexical_jaccard"]
    return selected[keep].assign(selection_group="low_similarity_candidate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cikm_middle.yaml", type=Path)
    parser.add_argument("--target-edges", type=int, default=800_000)
    parser.add_argument("--output", default="data/processed/targeted_inference_edges_800k.parquet", type=Path)
    parser.add_argument("--typed-dir", default="data/processed/typed_edges_shards", type=Path)
    parser.add_argument("--exclude-labels", nargs="*", default=[], type=Path)
    parser.add_argument("--low-similarity-candidate-multiplier", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    interim_dir = Path(config["paths"]["interim_dir"])
    manifest_dir = Path(config["paths"]["manifest_dir"])
    manifest_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pools = pd.read_parquet(interim_dir / "annotation_sampling_pools.parquet")
    edges = pd.read_parquet(interim_dir / "edges.parquet")
    papers = pd.read_parquet(interim_dir / "papers.parquet", columns=["paper_id", "title", "abstract"])

    already_typed_ids = typed_edge_ids(args.typed_dir)
    already_label_ids = label_edge_ids(args.exclude_labels)
    used_ids = set(already_typed_ids) | set(already_label_ids)
    already_typed_count = len(already_typed_ids)
    already_labeled_count = len(already_label_ids)

    target = args.target_edges
    selected_frames = []
    for pool_name in ["reciprocal_edges", "internal_high_inflation_communities", "internal_dense_communities", "touches_top_candidate_communities"]:
        count = int(target * DEFAULT_FRACTIONS[pool_name])
        selected_frames.append(select_from_pool(pools, pool_name, count, seed, used_ids))

    selected_frames.append(
        select_low_similarity(
            pools,
            papers,
            int(target * DEFAULT_FRACTIONS["low_similarity_candidate"]),
            seed,
            used_ids,
            args.low_similarity_candidate_multiplier,
        )
    )

    random_count = int(target * DEFAULT_FRACTIONS["random_global_control"])
    remaining_edges = edges[~edges["edge_id"].isin(used_ids)].drop_duplicates("edge_id")
    random_controls = sample(remaining_edges, random_count, seed + 101).copy()
    used_ids.update(random_controls["edge_id"].astype(str))
    selected_frames.append(random_controls.assign(pool="global_edges", selection_group="random_global_control"))

    selected = pd.concat(selected_frames, ignore_index=True).drop_duplicates("edge_id")
    if len(selected) < target:
        remaining_edges = edges[~edges["edge_id"].isin(set(selected["edge_id"].astype(str)) | typed_edge_ids(args.typed_dir))]
        filler = sample(remaining_edges.drop_duplicates("edge_id"), target - len(selected), seed + 202)
        selected = pd.concat(
            [selected, filler.assign(pool="global_edges", selection_group="filler_global_control")],
            ignore_index=True,
        ).drop_duplicates("edge_id")

    selected = selected.head(target)
    selected[["edge_id", "citing_id", "cited_id", "selection_group", "pool"]].to_parquet(args.output, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_edges": target,
        "actual_edges": int(len(selected)),
        "output": str(args.output),
        "typed_dir_excluded": str(args.typed_dir),
        "already_typed_edges_excluded": already_typed_count,
        "label_files_excluded": [str(path) for path in args.exclude_labels],
        "already_labeled_edges_excluded": already_labeled_count,
        "total_excluded_edge_ids": len(used_ids),
        "selection_counts": selected["selection_group"].value_counts().to_dict(),
        "method": {
            "reciprocal_edges": "quid-pro-quo structural signal",
            "internal_high_inflation_communities": "communities with more internal citation than expected",
            "internal_dense_communities": "locally dense citation regions",
            "touches_top_candidate_communities": "boundary and internal edges around suspicious communities",
            "low_similarity_candidate": "weak topical-overlap edges likely to expose superficial citations",
            "random_global_control": "background/control distribution for comparison",
        },
        "seed": seed,
    }
    manifest_path = manifest_dir / f"{args.output.stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"Wrote targeted inference edges to {args.output}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
