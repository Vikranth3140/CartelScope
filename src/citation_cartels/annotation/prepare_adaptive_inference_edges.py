"""Prepare an adaptive semantic-inference expansion set.

This script is meant to run after an initial targeted inference pass has already
typed some suspicious-region edges. It uses those partial semantic predictions
to identify communities with the strongest combined structural and semantic
risk, then selects additional untyped edges around those communities plus
matched controls.

The output can be passed to `citation_cartels.annotation.infer_student --edges`.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
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


SELECTION_FRACTIONS = {
    "adaptive_top_internal": 0.35,
    "adaptive_top_boundary": 0.20,
    "high_inflation_internal": 0.15,
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


def edge_ids_from_dirs(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for directory in paths:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("part_*.parquet")):
            frame = pd.read_parquet(path, columns=["edge_id"])
            ids.update(frame["edge_id"].dropna().astype(str))
    return ids


def edge_ids_from_labels(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=["edge_id"])
        ids.update(frame["edge_id"].dropna().astype(str))
    return ids


def load_typed_edges(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for directory in paths:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("part_*.parquet")):
            frame = pd.read_parquet(path, columns=["edge_id", "predicted_label", "semantic_weight", "label_confidence"])
            frames.append(frame.assign(source=str(path)))
    if not frames:
        return pd.DataFrame(columns=["edge_id", "predicted_label", "semantic_weight", "label_confidence", "source"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("edge_id")


def load_teacher_labels(paths: list[Path], weights: dict[str, float]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["edge_id", "label", "confidence"])
        frame = frame.rename(columns={"label": "predicted_label", "confidence": "label_confidence"})
        frame["semantic_weight"] = frame["predicted_label"].map(weights).astype(float)
        frames.append(frame.assign(source=str(path)))
    if not frames:
        return pd.DataFrame(columns=["edge_id", "predicted_label", "semantic_weight", "label_confidence", "source"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("edge_id")


def attach_communities(edges: pd.DataFrame, node_communities: pd.DataFrame) -> pd.DataFrame:
    return edges.merge(
        node_communities.rename(columns={"paper_id": "citing_id", "community_id": "citing_community"}),
        on="citing_id",
        how="left",
    ).merge(
        node_communities.rename(columns={"paper_id": "cited_id", "community_id": "cited_community"}),
        on="cited_id",
        how="left",
    )


def rank_adaptive_communities(
    typed: pd.DataFrame,
    edges_with_communities: pd.DataFrame,
    community_scores: pd.DataFrame,
    min_semantic_edges: int,
    top_k: int,
) -> pd.DataFrame:
    typed_edges = edges_with_communities[["edge_id", "citing_community", "cited_community"]].merge(
        typed,
        on="edge_id",
        how="inner",
    )
    typed_edges["is_internal"] = typed_edges["citing_community"] == typed_edges["cited_community"]
    internal = typed_edges[typed_edges["is_internal"]].copy()
    if internal.empty:
        ranked = community_scores.copy()
        ranked["semantic_superficiality"] = 0.0
        ranked["semantic_coverage"] = 0
    else:
        internal["is_superficial"] = internal["predicted_label"].isin(["Background", "Perfunctory/Ceremonial"])
        semantic = (
            internal.groupby("citing_community")
            .agg(
                semantic_coverage=("edge_id", "size"),
                semantic_superficiality=("is_superficial", "mean"),
                mean_semantic_weight=("semantic_weight", "mean"),
            )
            .reset_index()
            .rename(columns={"citing_community": "community_id"})
        )
        ranked = community_scores.merge(semantic, on="community_id", how="left")
        ranked["semantic_coverage"] = ranked["semantic_coverage"].fillna(0).astype(int)
        ranked["semantic_superficiality"] = ranked["semantic_superficiality"].fillna(0.0)
        ranked["mean_semantic_weight"] = ranked["mean_semantic_weight"].fillna(ranked["mean_semantic_weight"].median())

    eligible = ranked[ranked["semantic_coverage"] >= min_semantic_edges].copy()
    if eligible.empty:
        eligible = ranked.copy()
    eligible["adaptive_score"] = (
        eligible["inflation"].rank(pct=True)
        + eligible["density"].rank(pct=True)
        + eligible["reciprocity"].rank(pct=True)
        + eligible["semantic_superficiality"].rank(pct=True)
    )
    return eligible.sort_values("adaptive_score", ascending=False).head(top_k)


def select_edges(frame: pd.DataFrame, count: int, seed: int, used_ids: set[str], group: str) -> pd.DataFrame:
    candidates = frame[~frame["edge_id"].isin(used_ids)].drop_duplicates("edge_id")
    selected = sample(candidates, count, seed).copy()
    used_ids.update(selected["edge_id"].astype(str))
    return selected.assign(selection_group=group)


def join_text(edges: pd.DataFrame, papers: pd.DataFrame) -> pd.DataFrame:
    paper_text = papers[["paper_id", "title", "abstract"]].copy()
    result = edges.merge(
        paper_text.rename(columns={"paper_id": "citing_id", "title": "citing_title", "abstract": "citing_abstract"}),
        on="citing_id",
        how="left",
    ).merge(
        paper_text.rename(columns={"paper_id": "cited_id", "title": "cited_title", "abstract": "cited_abstract"}),
        on="cited_id",
        how="left",
    )
    for column in ["citing_title", "citing_abstract", "cited_title", "cited_abstract"]:
        result[column] = result[column].fillna("")
    return result


def select_low_similarity(
    candidates: pd.DataFrame,
    papers: pd.DataFrame,
    count: int,
    seed: int,
    used_ids: set[str],
    multiplier: int,
) -> pd.DataFrame:
    candidates = candidates[~candidates["edge_id"].isin(used_ids)].drop_duplicates("edge_id")
    candidates = sample(candidates, min(len(candidates), max(count * multiplier, count)), seed)
    with_text = join_text(candidates, papers)
    citing_text = (with_text["citing_title"] + " " + with_text["citing_abstract"]).str.slice(0, 1200)
    cited_text = (with_text["cited_title"] + " " + with_text["cited_abstract"]).str.slice(0, 1200)
    with_text["lexical_jaccard"] = [jaccard(a, b) for a, b in zip(citing_text, cited_text)]
    selected = with_text.sort_values("lexical_jaccard").head(count).copy()
    used_ids.update(selected["edge_id"].astype(str))
    keep = ["edge_id", "citing_id", "cited_id", "citing_community", "cited_community", "lexical_jaccard"]
    return selected[keep].assign(selection_group="low_similarity_candidate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cikm_middle.yaml", type=Path)
    parser.add_argument("--target-edges", type=int, default=1_050_000)
    parser.add_argument("--typed-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--exclude-labels", nargs="*", default=[], type=Path)
    parser.add_argument("--output", default="data/processed/adaptive_inference_edges_1050k.parquet", type=Path)
    parser.add_argument("--top-communities", type=int, default=100)
    parser.add_argument("--min-semantic-edges-per-community", type=int, default=50)
    parser.add_argument("--low-similarity-candidate-multiplier", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    weights = {str(label): float(weight) for label, weight in config["annotation"]["weights"].items()}
    interim_dir = Path(config["paths"]["interim_dir"])
    manifest_dir = Path(config["paths"]["manifest_dir"])
    manifest_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    edges = pd.read_parquet(interim_dir / "edges.parquet")
    papers = pd.read_parquet(interim_dir / "papers.parquet", columns=["paper_id", "title", "abstract"])
    node_communities = pd.read_parquet(interim_dir / "pre_annotation_node_communities.parquet")
    community_scores = pd.read_parquet(interim_dir / "pre_annotation_communities.parquet")
    edges_with_communities = attach_communities(edges, node_communities)
    edges_with_communities["is_internal"] = edges_with_communities["citing_community"] == edges_with_communities["cited_community"]

    typed_ids = edge_ids_from_dirs(args.typed_dirs)
    label_ids = edge_ids_from_labels(args.exclude_labels)
    used_ids = set(typed_ids) | set(label_ids)
    typed_semantics = pd.concat(
        [load_typed_edges(args.typed_dirs), load_teacher_labels(args.exclude_labels, weights)],
        ignore_index=True,
    ).drop_duplicates("edge_id")

    ranked_communities = rank_adaptive_communities(
        typed_semantics,
        edges_with_communities,
        community_scores,
        args.min_semantic_edges_per_community,
        args.top_communities,
    )
    top_ids = set(ranked_communities["community_id"])
    high_inflation_ids = set(community_scores.sort_values("inflation", ascending=False).head(args.top_communities)["community_id"])

    target = args.target_edges
    frames = []
    internal_top = edges_with_communities[edges_with_communities["is_internal"] & edges_with_communities["citing_community"].isin(top_ids)]
    frames.append(select_edges(internal_top, int(target * SELECTION_FRACTIONS["adaptive_top_internal"]), seed, used_ids, "adaptive_top_internal"))

    boundary_top = edges_with_communities[
        (edges_with_communities["citing_community"].isin(top_ids) | edges_with_communities["cited_community"].isin(top_ids))
        & ~edges_with_communities["is_internal"]
    ]
    frames.append(select_edges(boundary_top, int(target * SELECTION_FRACTIONS["adaptive_top_boundary"]), seed + 1, used_ids, "adaptive_top_boundary"))

    high_inflation = edges_with_communities[
        edges_with_communities["is_internal"] & edges_with_communities["citing_community"].isin(high_inflation_ids)
    ]
    frames.append(select_edges(high_inflation, int(target * SELECTION_FRACTIONS["high_inflation_internal"]), seed + 2, used_ids, "high_inflation_internal"))

    low_similarity_candidates = edges_with_communities[
        edges_with_communities["citing_community"].isin(top_ids) | edges_with_communities["cited_community"].isin(top_ids)
    ]
    frames.append(
        select_low_similarity(
            low_similarity_candidates,
            papers,
            int(target * SELECTION_FRACTIONS["low_similarity_candidate"]),
            seed + 3,
            used_ids,
            args.low_similarity_candidate_multiplier,
        )
    )

    frames.append(
        select_edges(
            edges_with_communities,
            int(target * SELECTION_FRACTIONS["random_global_control"]),
            seed + 4,
            used_ids,
            "random_global_control",
        )
    )

    selected = pd.concat(frames, ignore_index=True).drop_duplicates("edge_id")
    if len(selected) < target:
        filler = select_edges(edges_with_communities, target - len(selected), seed + 5, used_ids, "filler_global_control")
        selected = pd.concat([selected, filler], ignore_index=True).drop_duplicates("edge_id")

    selected = selected.head(target)
    keep_cols = ["edge_id", "citing_id", "cited_id", "selection_group", "citing_community", "cited_community"]
    selected[keep_cols].to_parquet(args.output, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_edges": target,
        "actual_edges": int(len(selected)),
        "output": str(args.output),
        "typed_dirs_excluded": [str(path) for path in args.typed_dirs],
        "typed_edges_excluded": len(typed_ids),
        "label_files_excluded": [str(path) for path in args.exclude_labels],
        "labeled_edges_excluded": len(label_ids),
        "total_excluded_edge_ids": len(set(typed_ids) | set(label_ids)),
        "top_communities": int(args.top_communities),
        "ranked_community_count": int(len(ranked_communities)),
        "selection_counts": selected["selection_group"].value_counts().to_dict(),
        "top_ranked_communities": ranked_communities.head(20).to_dict(orient="records"),
        "method": {
            "adaptive_top_internal": "remaining internal edges in communities that are structurally suspicious and semantically superficial in the partial run",
            "adaptive_top_boundary": "cross-boundary edges touching those adaptive high-risk communities",
            "high_inflation_internal": "remaining internal edges in structurally high-inflation communities",
            "low_similarity_candidate": "weak topical-overlap edges around adaptive high-risk communities",
            "random_global_control": "matched global controls for comparison",
        },
        "seed": seed,
    }
    manifest_path = manifest_dir / f"{args.output.stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "top_ranked_communities"}, indent=2))
    print(f"Wrote adaptive inference edges to {args.output}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
