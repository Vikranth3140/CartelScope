"""Run final semantic graph, CCI, baseline, and excision analyses.

This script starts after teacher labels and student inference shards have been
combined into `semantic_edges_combined.parquet`. It creates the paper-facing
graph artifacts for Phases 7-11 of the CIKM plan.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml


FEATURE_COLUMNS = [
    "density",
    "log_inflation",
    "reciprocity",
    "semantic_superficiality",
    "degree_assortativity",
    "mean_pagerank_drop",
]
SUPERFICIAL_LABELS = {"Background", "Perfunctory/Ceremonial"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def zscore(series: pd.Series) -> pd.Series:
    values = series.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    std = values.std(ddof=0)
    if std == 0 or math.isnan(std):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / std


def top_overlap(full: pd.Series, other: pd.Series, k: int) -> float:
    full_top = set(full.sort_values(ascending=False).head(k).index)
    other_top = set(other.sort_values(ascending=False).head(k).index)
    if not full_top:
        return 0.0
    return len(full_top & other_top) / len(full_top)


def safe_spearman(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left, right], axis=1).dropna()
    if aligned.shape[0] < 2:
        return 0.0
    value = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
    return 0.0 if pd.isna(value) else float(value)


def build_weighted_edges(
    edges_path: Path,
    semantic_edges_path: Path,
    output_path: Path,
    neutral_weight: float,
) -> pd.DataFrame:
    print(f"Loading full edge table from {edges_path}")
    edges = pd.read_parquet(edges_path, columns=["edge_id", "citing_id", "cited_id"])
    print(f"Loading semantic edge table from {semantic_edges_path}")
    semantic = pd.read_parquet(
        semantic_edges_path,
        columns=[
            "edge_id",
            "predicted_label",
            "label_confidence",
            "semantic_weight",
            "semantic_source",
            "model_version",
        ],
    )
    semantic = semantic.drop_duplicates("edge_id", keep="last")

    weighted = edges.merge(semantic, on="edge_id", how="left")
    weighted["is_semantically_typed"] = weighted["predicted_label"].notna()
    weighted["predicted_label"] = weighted["predicted_label"].fillna("Untyped")
    weighted["label_confidence"] = weighted["label_confidence"].fillna(0.0).astype(float)
    weighted["semantic_weight"] = weighted["semantic_weight"].fillna(neutral_weight).astype(float)
    weighted["semantic_source"] = weighted["semantic_source"].fillna("untyped_neutral")
    weighted["model_version"] = weighted["model_version"].fillna("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    weighted.to_parquet(output_path, index=False)
    print(f"Wrote weighted edge table with {len(weighted):,} rows to {output_path}")
    return weighted


def load_or_build_weighted_edges(
    edges_path: Path,
    semantic_edges_path: Path,
    output_path: Path,
    neutral_weight: float,
    reuse_existing: bool,
) -> pd.DataFrame:
    if reuse_existing and output_path.exists():
        print(f"Reusing existing weighted edge table at {output_path}")
        return pd.read_parquet(output_path)
    return build_weighted_edges(edges_path, semantic_edges_path, output_path, neutral_weight)


def build_directed_graph(weighted_edges: pd.DataFrame, papers: pd.DataFrame) -> nx.DiGraph:
    print("Building directed graph for PageRank")
    graph = nx.DiGraph()
    graph.add_nodes_from(papers["paper_id"].astype(str).tolist())
    graph.add_weighted_edges_from(
        (
            str(row.citing_id),
            str(row.cited_id),
            float(row.semantic_weight),
        )
        for row in weighted_edges[["citing_id", "cited_id", "semantic_weight"]].itertuples(index=False)
    )
    print(f"Directed graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    return graph


def compute_pagerank_outputs(
    graph: nx.DiGraph,
    papers: pd.DataFrame,
    output_dir: Path,
    seed: int,
) -> pd.DataFrame:
    print("Computing unweighted PageRank")
    unweighted = nx.pagerank(graph, alpha=0.85, weight=None, max_iter=100, tol=1.0e-6)
    print("Computing trust-weighted PageRank")
    weighted = nx.pagerank(graph, alpha=0.85, weight="weight", max_iter=100, tol=1.0e-6)

    rows = pd.DataFrame(
        {
            "paper_id": list(graph.nodes()),
            "pagerank_unweighted": [unweighted[node] for node in graph.nodes()],
            "pagerank_weighted": [weighted[node] for node in graph.nodes()],
        }
    )
    rows["pagerank_delta"] = rows["pagerank_weighted"] - rows["pagerank_unweighted"]
    rows["pagerank_drop"] = rows["pagerank_unweighted"] - rows["pagerank_weighted"]
    rows["rank_unweighted"] = rows["pagerank_unweighted"].rank(ascending=False, method="min").astype(int)
    rows["rank_weighted"] = rows["pagerank_weighted"].rank(ascending=False, method="min").astype(int)
    rows["rank_drop"] = rows["rank_weighted"] - rows["rank_unweighted"]

    metadata_cols = [column for column in ["paper_id", "title", "year", "n_citation", "top_fos"] if column in papers.columns]
    rows = rows.merge(papers[metadata_cols], on="paper_id", how="left")
    rows = rows.sort_values("rank_drop", ascending=False)
    output_path = output_dir / "trust_pagerank_shifts.csv"
    rows.to_csv(output_path, index=False)
    print(f"Wrote PageRank shifts to {output_path}")

    examples = {
        "largest_rank_drops": rows.head(25).to_dict(orient="records"),
        "largest_rank_gains": rows.sort_values("rank_drop").head(25).to_dict(orient="records"),
        "seed": seed,
    }
    write_json(output_dir / "trust_pagerank_examples.json", examples)
    return rows


def attach_communities(weighted_edges: pd.DataFrame, node_communities: pd.DataFrame) -> pd.DataFrame:
    edges = weighted_edges.merge(
        node_communities.rename(columns={"paper_id": "citing_id", "community_id": "citing_community"}),
        on="citing_id",
        how="left",
    ).merge(
        node_communities.rename(columns={"paper_id": "cited_id", "community_id": "cited_community"}),
        on="cited_id",
        how="left",
    )
    edges["is_internal"] = edges["citing_community"] == edges["cited_community"]
    return edges


def load_or_compute_communities(
    communities_path: Optional[Path],
    graph: nx.DiGraph,
    output_path: Path,
    seed: int,
) -> pd.DataFrame:
    if communities_path is not None and communities_path.exists():
        print(f"Loading communities from {communities_path}")
        communities = pd.read_parquet(communities_path)
    else:
        print("Computing Louvain communities on undirected graph")
        groups = nx.community.louvain_communities(graph.to_undirected(), seed=seed)
        rows = []
        for community_id, nodes in enumerate(groups):
            for node in nodes:
                rows.append({"paper_id": str(node), "community_id": int(community_id)})
        communities = pd.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        communities.to_parquet(output_path, index=False)
        print(f"Wrote communities to {output_path}")

    communities["paper_id"] = communities["paper_id"].astype(str)
    communities["community_id"] = communities["community_id"].astype(int)
    return communities[["paper_id", "community_id"]].drop_duplicates("paper_id")


def add_reciprocal_flag(edges: pd.DataFrame) -> pd.DataFrame:
    print("Computing reciprocal edge flags")
    reverse_edges = edges.rename(columns={"citing_id": "cited_id", "cited_id": "citing_id"})
    reverse_edges = reverse_edges[["citing_id", "cited_id"]].drop_duplicates()
    reverse_edges["is_reciprocal"] = True
    return edges.merge(reverse_edges, on=["citing_id", "cited_id"], how="left").assign(
        is_reciprocal=lambda df: df["is_reciprocal"].fillna(False)
    )


def degree_correlation(group: pd.DataFrame) -> float:
    if len(group) < 2:
        return 0.0
    left = np.log1p(group["citing_out_degree"].to_numpy(dtype=float))
    right = np.log1p(group["cited_in_degree"].to_numpy(dtype=float))
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return 0.0 if math.isnan(value) else value


def compute_community_scores(
    weighted_edges: pd.DataFrame,
    pagerank_shifts: pd.DataFrame,
    node_communities: pd.DataFrame,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Attaching communities to weighted edges")
    edges = attach_communities(weighted_edges, node_communities)
    edges = add_reciprocal_flag(edges)

    out_degree = weighted_edges.groupby("citing_id").size().rename("out_degree")
    in_degree = weighted_edges.groupby("cited_id").size().rename("in_degree")
    node_stats = (
        node_communities.merge(out_degree, left_on="paper_id", right_index=True, how="left")
        .merge(in_degree, left_on="paper_id", right_index=True, how="left")
        .fillna({"out_degree": 0, "in_degree": 0})
    )

    community_sizes = node_stats.groupby("community_id").size().rename("size")
    community_degree = node_stats.groupby("community_id").agg(
        community_out_degree=("out_degree", "sum"),
        community_in_degree=("in_degree", "sum"),
    )

    internal = edges[edges["is_internal"]].copy()
    internal["is_superficial"] = internal["predicted_label"].isin(SUPERFICIAL_LABELS)

    print("Aggregating internal community features")
    internal_stats = internal.groupby("citing_community").agg(
        internal_edges=("edge_id", "size"),
        reciprocal_internal_edges=("is_reciprocal", "sum"),
        typed_internal_edges=("is_semantically_typed", "sum"),
        superficial_internal_edges=("is_superficial", "sum"),
        mean_internal_semantic_weight=("semantic_weight", "mean"),
        mean_internal_label_confidence=("label_confidence", "mean"),
    )
    internal_stats.index.name = "community_id"

    citing_out = out_degree.rename("citing_out_degree")
    cited_in = in_degree.rename("cited_in_degree")
    internal_for_corr = internal[["citing_id", "cited_id", "citing_community"]].copy()
    internal_for_corr = internal_for_corr.merge(citing_out, left_on="citing_id", right_index=True, how="left")
    internal_for_corr = internal_for_corr.merge(cited_in, left_on="cited_id", right_index=True, how="left")
    assortativity = internal_for_corr.groupby("citing_community").apply(degree_correlation)
    assortativity.index.name = "community_id"
    assortativity = assortativity.rename("degree_assortativity")

    pagerank_by_community = pagerank_shifts.merge(node_communities, on="paper_id", how="left")
    pagerank_stats = pagerank_by_community.groupby("community_id").agg(
        mean_pagerank_drop=("pagerank_drop", "mean"),
        mean_rank_drop=("rank_drop", "mean"),
        mean_weighted_pagerank=("pagerank_weighted", "mean"),
        mean_unweighted_pagerank=("pagerank_unweighted", "mean"),
    )

    scores = pd.concat([community_sizes, community_degree, internal_stats, assortativity, pagerank_stats], axis=1).reset_index()
    fill_zero = [
        "internal_edges",
        "reciprocal_internal_edges",
        "typed_internal_edges",
        "superficial_internal_edges",
        "degree_assortativity",
        "mean_pagerank_drop",
        "mean_rank_drop",
        "mean_weighted_pagerank",
        "mean_unweighted_pagerank",
    ]
    scores[fill_zero] = scores[fill_zero].fillna(0)
    scores["mean_internal_semantic_weight"] = scores["mean_internal_semantic_weight"].fillna(1.0)
    scores["mean_internal_label_confidence"] = scores["mean_internal_label_confidence"].fillna(0.0)

    scores["possible_directed_edges"] = scores["size"] * (scores["size"] - 1)
    scores["density"] = scores["internal_edges"] / scores["possible_directed_edges"].clip(lower=1)
    scores["reciprocity"] = scores["reciprocal_internal_edges"] / scores["internal_edges"].clip(lower=1)
    total_edges = max(float(len(weighted_edges)), 1.0)
    scores["expected_internal_edges"] = (
        scores["community_out_degree"].astype(float) * scores["community_in_degree"].astype(float)
    ) / total_edges
    scores["inflation"] = scores["internal_edges"] / scores["expected_internal_edges"].clip(lower=1.0e-9)
    scores["log_inflation"] = np.log1p(scores["inflation"])
    scores["semantic_coverage"] = scores["typed_internal_edges"] / scores["internal_edges"].clip(lower=1)
    scores["semantic_superficiality"] = (
        scores["superficial_internal_edges"] / scores["typed_internal_edges"].clip(lower=1)
    )

    for feature in FEATURE_COLUMNS:
        scores[f"{feature}_z"] = zscore(scores[feature])
    z_columns = [f"{feature}_z" for feature in FEATURE_COLUMNS]
    scores["cci_score"] = scores[z_columns].mean(axis=1)

    scores["structural_cci_score"] = scores[
        ["density_z", "log_inflation_z", "reciprocity_z", "degree_assortativity_z", "mean_pagerank_drop_z"]
    ].mean(axis=1)
    scores["rank_cci"] = scores["cci_score"].rank(ascending=False, method="min").astype(int)
    scores = scores.sort_values("cci_score", ascending=False)

    final_path = output_dir / "cartel_community_scores_final.csv"
    top_path = output_dir / "top_communities_final.csv"
    scores.to_csv(final_path, index=False)
    scores.head(25).to_csv(top_path, index=False)
    print(f"Wrote final CCI scores to {final_path}")

    plt.figure(figsize=(9, 5))
    plt.hist(scores["cci_score"], bins=50, color="#4C78A8", alpha=0.85)
    plt.axvline(scores["cci_score"].quantile(0.95), color="#E45756", linestyle="--", label="95th percentile")
    plt.xlabel("Composite Cartel Index")
    plt.ylabel("Community count")
    plt.title("Final CCI Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cci_distribution_final.png", dpi=250)
    plt.close()

    return scores, edges


def plot_intent_distribution(weighted_edges: pd.DataFrame, output_dir: Path) -> None:
    typed = weighted_edges[weighted_edges["is_semantically_typed"]].copy()
    counts = typed["predicted_label"].value_counts().sort_values(ascending=True)
    counts.to_csv(output_dir / "intent_distribution_final.csv", header=["edge_count"])

    plt.figure(figsize=(9, 5))
    counts.plot(kind="barh", color="#59A14F")
    plt.xlabel("Typed edge count")
    plt.ylabel("Predicted citation intent")
    plt.title("Semantic Intent Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "intent_distribution_final.png", dpi=250)
    plt.close()


def write_baselines(scores: pd.DataFrame, output_dir: Path, seed: int) -> None:
    print("Writing baseline and ablation comparisons")
    indexed = scores.set_index("community_id").copy()
    rng = np.random.default_rng(seed)
    indexed["random_score"] = rng.random(len(indexed))
    baseline_scores = {
        "density_only": "density",
        "inflation_only": "log_inflation",
        "reciprocity_only": "reciprocity",
        "structural_only_without_semantic": "structural_cci_score",
        "semantic_only_superficiality": "semantic_superficiality",
        "random": "random_score",
    }

    rows = []
    precision_rows = []
    full = indexed["cci_score"]
    for name, column in baseline_scores.items():
        score = indexed[column]
        rows.append(
            {
                "baseline": name,
                "spearman_with_final_cci": safe_spearman(full, score),
                "top5_overlap_with_final_cci": top_overlap(full, score, 5),
                "top10_overlap_with_final_cci": top_overlap(full, score, 10),
                "top25_overlap_with_final_cci": top_overlap(full, score, 25),
            }
        )
        for k in [5, 10, 25]:
            precision_rows.append(
                {
                    "baseline": name,
                    "k": k,
                    "proxy_precision_at_k_vs_final_cci": top_overlap(full, score, k),
                }
            )

    pd.DataFrame(rows).to_csv(output_dir / "baseline_comparison.csv", index=False)
    pd.DataFrame(precision_rows).to_csv(output_dir / "baseline_precision_at_k.csv", index=False)

    ablation_rows = []
    z_columns = [f"{feature}_z" for feature in FEATURE_COLUMNS]
    for removed in FEATURE_COLUMNS:
        remaining = [column for column in z_columns if column != f"{removed}_z"]
        ablated = indexed[remaining].mean(axis=1)
        ablation_rows.append(
            {
                "removed_feature": removed,
                "spearman_with_full_cci": safe_spearman(full, ablated),
                "top5_overlap_with_full_cci": top_overlap(full, ablated, 5),
                "top10_overlap_with_full_cci": top_overlap(full, ablated, 10),
                "top25_overlap_with_full_cci": top_overlap(full, ablated, 25),
            }
        )
    pd.DataFrame(ablation_rows).to_csv(output_dir / "cci_ablation.csv", index=False)


def sampled_average_shortest_path(graph: nx.Graph, sample_size: int, seed: int) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    largest = max(nx.connected_components(graph), key=len)
    if len(largest) <= 1:
        return 0.0
    rng = random.Random(seed)
    sources = rng.sample(list(largest), min(sample_size, len(largest)))
    total_distance = 0
    pair_count = 0
    for source in sources:
        lengths = nx.single_source_shortest_path_length(graph, source)
        for target, distance in lengths.items():
            if target != source and target in largest:
                total_distance += distance
                pair_count += 1
    return total_distance / pair_count if pair_count else 0.0


def graph_health(graph: nx.Graph, sample_path_nodes: int, seed: int) -> dict[str, Any]:
    if graph.number_of_nodes() == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "connected_components": 0,
            "giant_component_nodes": 0,
            "giant_component_fraction": 0.0,
            "sampled_average_shortest_path": 0.0,
        }
    components = list(nx.connected_components(graph))
    giant = max((len(component) for component in components), default=0)
    return {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "connected_components": int(len(components)),
        "giant_component_nodes": int(giant),
        "giant_component_fraction": giant / max(graph.number_of_nodes(), 1),
        "sampled_average_shortest_path": sampled_average_shortest_path(graph, sample_path_nodes, seed),
    }


def edge_pairs(frame: pd.DataFrame) -> List[Tuple[str, str]]:
    return [(str(row.citing_id), str(row.cited_id)) for row in frame[["citing_id", "cited_id"]].itertuples(index=False)]


def run_excision(
    graph: nx.DiGraph,
    edges_with_communities: pd.DataFrame,
    scores: pd.DataFrame,
    output_dir: Path,
    top_k: int,
    seeds: Sequence[int],
    sample_path_nodes: int,
) -> None:
    print("Running CCI-based edge excision validation")
    undirected = graph.to_undirected()
    top_communities = set(scores.sort_values("cci_score", ascending=False).head(top_k)["community_id"].astype(int))
    target_frame = edges_with_communities[
        edges_with_communities["is_internal"] & edges_with_communities["citing_community"].isin(top_communities)
    ]
    target_edges = edge_pairs(target_frame)

    rows = []
    baseline = graph_health(undirected, sample_path_nodes, int(seeds[0]) if seeds else 42)
    baseline.update({"strategy": "baseline", "seed": None, "removed_edges": 0, "top_k_communities": top_k})
    rows.append(baseline)

    cci_graph = undirected.copy()
    cci_graph.remove_edges_from(target_edges)
    cci_result = graph_health(cci_graph, sample_path_nodes, int(seeds[0]) if seeds else 42)
    cci_result.update(
        {
            "strategy": "cci_top_community_internal_edges",
            "seed": None,
            "removed_edges": len(target_edges),
            "top_k_communities": top_k,
        }
    )
    rows.append(cci_result)

    all_edges = list(undirected.edges())
    for seed in seeds:
        rng = random.Random(int(seed))
        sample_size = min(len(target_edges), len(all_edges))
        random_edges = rng.sample(all_edges, sample_size)
        control = undirected.copy()
        control.remove_edges_from(random_edges)
        result = graph_health(control, sample_path_nodes, int(seed))
        result.update(
            {
                "strategy": "random_matched_edges",
                "seed": int(seed),
                "removed_edges": sample_size,
                "top_k_communities": top_k,
            }
        )
        rows.append(result)

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "excision_results.csv", index=False)

    plt.figure(figsize=(8, 5))
    plot_rows = results[results["strategy"] != "baseline"].copy()
    labels = [
        row.strategy if pd.isna(row.seed) else f"{row.strategy}_{int(row.seed)}"
        for row in plot_rows.itertuples(index=False)
    ]
    plt.bar(range(len(plot_rows)), plot_rows["giant_component_fraction"], color="#F28E2B")
    plt.axhline(baseline["giant_component_fraction"], color="#4C78A8", linestyle="--", label="baseline")
    plt.xticks(range(len(plot_rows)), labels, rotation=35, ha="right")
    plt.ylabel("Giant component fraction")
    plt.title("CCI Edge Excision vs Random Controls")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cartel_edge_excision_final.png", dpi=250)
    plt.close()

    write_json(
        output_dir / "excision_target_edges_manifest.json",
        {
            "top_k_communities": top_k,
            "top_community_ids": sorted(int(value) for value in top_communities),
            "target_internal_edge_count": len(target_edges),
            "random_control_seeds": [int(seed) for seed in seeds],
            "sample_path_nodes": sample_path_nodes,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cikm_middle.yaml", type=Path)
    parser.add_argument("--edges", default=None, type=Path)
    parser.add_argument("--papers", default=None, type=Path)
    parser.add_argument("--communities", default=None, type=Path)
    parser.add_argument("--semantic-edges", default="data/processed/semantic_edges_combined.parquet", type=Path)
    parser.add_argument("--weighted-edges-output", default="data/processed/weighted_edges.parquet", type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--neutral-weight", type=float, default=1.0)
    parser.add_argument("--reuse-weighted-edges", action="store_true")
    parser.add_argument("--top-k-excision-communities", type=int, default=None)
    parser.add_argument("--sample-path-nodes", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    paths = config["paths"]
    graph_config = config.get("graph", {})

    interim_dir = Path(paths["interim_dir"])
    output_dir = args.output_dir or Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    edges_path = args.edges or interim_dir / "edges.parquet"
    papers_path = args.papers or interim_dir / "paper_metadata.parquet"
    communities_path = args.communities or interim_dir / "pre_annotation_node_communities.parquet"
    top_k = args.top_k_excision_communities or int(graph_config.get("top_k_excision_communities", 5))
    stability_seeds = [int(value) for value in graph_config.get("stability_seeds", [seed])]

    print("Starting final graph analysis")
    weighted_edges = load_or_build_weighted_edges(
        edges_path=edges_path,
        semantic_edges_path=args.semantic_edges,
        output_path=args.weighted_edges_output,
        neutral_weight=args.neutral_weight,
        reuse_existing=args.reuse_weighted_edges,
    )
    papers = pd.read_parquet(papers_path)
    papers["paper_id"] = papers["paper_id"].astype(str)

    graph = build_directed_graph(weighted_edges, papers)
    node_communities = load_or_compute_communities(
        communities_path=communities_path,
        graph=graph,
        output_path=interim_dir / "final_node_communities.parquet",
        seed=seed,
    )
    pagerank_shifts = compute_pagerank_outputs(graph, papers, output_dir, seed)
    scores, edges_with_communities = compute_community_scores(weighted_edges, pagerank_shifts, node_communities, output_dir)
    plot_intent_distribution(weighted_edges, output_dir)
    write_baselines(scores, output_dir, seed)
    run_excision(
        graph=graph,
        edges_with_communities=edges_with_communities,
        scores=scores,
        output_dir=output_dir,
        top_k=top_k,
        seeds=stability_seeds,
        sample_path_nodes=args.sample_path_nodes,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "edges_path": str(edges_path),
        "semantic_edges_path": str(args.semantic_edges),
        "weighted_edges_path": str(args.weighted_edges_output),
        "papers_path": str(papers_path),
        "communities_path": str(communities_path),
        "output_dir": str(output_dir),
        "full_edge_rows": int(len(weighted_edges)),
        "typed_edge_rows": int(weighted_edges["is_semantically_typed"].sum()),
        "community_count": int(scores.shape[0]),
        "top_k_excision_communities": int(top_k),
        "stability_seeds": stability_seeds,
    }
    write_json(output_dir / "final_graph_analysis_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    print("Final graph analysis complete")


if __name__ == "__main__":
    main()
