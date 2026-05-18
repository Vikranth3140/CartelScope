"""Build structural edge pools for cost-effective teacher annotation.

This script runs after Phase 1 has produced `papers.parquet` and
`edges.parquet`. It computes preliminary communities and writes labeled
edge pools that the teacher-labeling job can sample from.

Usage:
    PYTHONPATH=src python -m citation_cartels.annotation.build_sampling_pools \
        --config configs/cikm_middle.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_reciprocal_flag(edges: pd.DataFrame) -> pd.DataFrame:
    reverse_edges = edges.rename(columns={"citing_id": "cited_id", "cited_id": "citing_id"})
    reverse_edges = reverse_edges[["citing_id", "cited_id"]].drop_duplicates()
    reverse_edges["is_reciprocal"] = True
    return edges.merge(reverse_edges, on=["citing_id", "cited_id"], how="left").assign(
        is_reciprocal=lambda df: df["is_reciprocal"].fillna(False)
    )


def build_graph(edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(edges[["citing_id", "cited_id"]].itertuples(index=False, name=None))
    return graph


def community_dataframe(graph: nx.Graph, seed: int) -> pd.DataFrame:
    communities = nx.community.louvain_communities(graph, seed=seed)
    rows = []
    for community_id, nodes in enumerate(communities):
        for node in nodes:
            rows.append({"paper_id": node, "community_id": community_id})
    return pd.DataFrame(rows)


def score_communities(edges: pd.DataFrame, node_communities: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
    edges_with_com = edges.merge(
        node_communities.rename(columns={"paper_id": "citing_id", "community_id": "citing_community"}),
        on="citing_id",
        how="left",
    ).merge(
        node_communities.rename(columns={"paper_id": "cited_id", "community_id": "cited_community"}),
        on="cited_id",
        how="left",
    )
    edges_with_com["is_internal"] = edges_with_com["citing_community"] == edges_with_com["cited_community"]

    community_sizes = node_communities.groupby("community_id").size().rename("size")
    internal_edges = (
        edges_with_com[edges_with_com["is_internal"]]
        .groupby("citing_community")
        .size()
        .rename("internal_edges")
    )
    reciprocal_internal = (
        edges_with_com[edges_with_com["is_internal"] & edges_with_com["is_reciprocal"]]
        .groupby("citing_community")
        .size()
        .rename("reciprocal_internal_edges")
    )

    rows = pd.concat([community_sizes, internal_edges, reciprocal_internal], axis=1).fillna(0)
    rows.index.name = "community_id"
    rows = rows.reset_index()
    rows["internal_edges"] = rows["internal_edges"].astype(int)
    rows["reciprocal_internal_edges"] = rows["reciprocal_internal_edges"].astype(int)
    rows["possible_directed_edges"] = rows["size"] * (rows["size"] - 1)
    rows["density"] = rows["internal_edges"] / rows["possible_directed_edges"].clip(lower=1)
    rows["reciprocity"] = rows["reciprocal_internal_edges"] / rows["internal_edges"].clip(lower=1)

    degree = Counter(dict(graph.degree()))
    total_undirected_edges = max(graph.number_of_edges(), 1)
    volumes = []
    for community_id, group in node_communities.groupby("community_id"):
        volume = sum(degree[node] for node in group["paper_id"])
        expected_edges = (volume * volume) / (2 * total_undirected_edges)
        actual_edges = rows.loc[rows["community_id"] == community_id, "internal_edges"].iloc[0]
        volumes.append(
            {
                "community_id": community_id,
                "volume": volume,
                "expected_internal_edges": expected_edges,
                "inflation": actual_edges / max(expected_edges, 1e-9),
            }
        )

    return rows.merge(pd.DataFrame(volumes), on="community_id", how="left")


def attach_communities(edges: pd.DataFrame, node_communities: pd.DataFrame) -> pd.DataFrame:
    result = edges.merge(
        node_communities.rename(columns={"paper_id": "citing_id", "community_id": "citing_community"}),
        on="citing_id",
        how="left",
    ).merge(
        node_communities.rename(columns={"paper_id": "cited_id", "community_id": "cited_community"}),
        on="cited_id",
        how="left",
    )
    result["is_internal"] = result["citing_community"] == result["cited_community"]
    return result


def make_sampling_pools(edges: pd.DataFrame, community_scores: pd.DataFrame) -> pd.DataFrame:
    pools = []

    def add_pool(name: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        pools.append(frame[["edge_id", "citing_id", "cited_id"]].assign(pool=name))

    dense_ids = set(community_scores.sort_values(["density", "size"], ascending=False).head(100)["community_id"])
    inflation_ids = set(community_scores.sort_values("inflation", ascending=False).head(100)["community_id"])
    candidate_ids = set(
        community_scores.assign(
            rough_score=lambda df: df["density"].rank(pct=True)
            + df["inflation"].rank(pct=True)
            + df["reciprocity"].rank(pct=True)
        )
        .sort_values("rough_score", ascending=False)
        .head(100)["community_id"]
    )

    add_pool("random_global", edges.sample(min(25_000, len(edges)), random_state=42))
    add_pool("reciprocal_edges", edges[edges["is_reciprocal"]])
    add_pool("internal_dense_communities", edges[edges["is_internal"] & edges["citing_community"].isin(dense_ids)])
    add_pool(
        "internal_high_inflation_communities",
        edges[edges["is_internal"] & edges["citing_community"].isin(inflation_ids)],
    )
    add_pool(
        "touches_top_candidate_communities",
        edges[edges["citing_community"].isin(candidate_ids) | edges["cited_community"].isin(candidate_ids)],
    )

    if not pools:
        return pd.DataFrame(columns=["edge_id", "citing_id", "cited_id", "pool"])
    return pd.concat(pools, ignore_index=True).drop_duplicates(["edge_id", "pool"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cikm_middle.yaml", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    interim_dir = Path(config["paths"]["interim_dir"])
    manifest_dir = Path(config["paths"]["manifest_dir"])
    manifest_dir.mkdir(parents=True, exist_ok=True)

    edges_path = interim_dir / "edges.parquet"
    if not edges_path.exists():
        raise FileNotFoundError(
            f"Missing {edges_path}. Run Phase 1 first: "
            "PYTHONPATH=src python -m citation_cartels.data.build_tables"
        )

    edges = pd.read_parquet(edges_path)
    edges = add_reciprocal_flag(edges)

    print(f"Building graph from {len(edges):,} edges")
    graph = build_graph(edges)
    print(f"Graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} undirected edges")

    print("Running preliminary Louvain communities")
    node_communities = community_dataframe(graph, seed=seed)
    community_scores = score_communities(edges, node_communities, graph)
    edges_with_com = attach_communities(edges, node_communities)

    community_path = interim_dir / "pre_annotation_communities.parquet"
    pools_path = interim_dir / "annotation_sampling_pools.parquet"
    node_communities_path = interim_dir / "pre_annotation_node_communities.parquet"

    community_scores.to_parquet(community_path, index=False)
    node_communities.to_parquet(node_communities_path, index=False)
    pools = make_sampling_pools(edges_with_com, community_scores)
    pools.to_parquet(pools_path, index=False)

    manifest = {
        "edges_path": str(edges_path),
        "community_scores_path": str(community_path),
        "node_communities_path": str(node_communities_path),
        "sampling_pools_path": str(pools_path),
        "num_communities": int(community_scores.shape[0]),
        "num_pool_rows": int(pools.shape[0]),
        "pool_counts": pools["pool"].value_counts().to_dict() if not pools.empty else {},
        "seed": seed,
    }
    manifest_path = manifest_dir / "annotation_sampling_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote community scores to {community_path}")
    print(f"Wrote sampling pools to {pools_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
