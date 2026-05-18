"""Run multi-seed CCI stability analysis for the CIKM paper."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from citation_cartels.graph.final_graph_analysis import (
    build_directed_graph,
    compute_community_scores,
    compute_pagerank_outputs,
    load_config,
    load_or_build_weighted_edges,
    safe_spearman,
    top_overlap,
    write_json,
)


def communities_for_seed(graph: nx.DiGraph, seed: int, output_path: Path) -> pd.DataFrame:
    print(f"Running Louvain community detection for seed={seed}")
    groups = nx.community.louvain_communities(graph.to_undirected(), seed=seed)
    rows = []
    for community_id, nodes in enumerate(groups):
        for node in nodes:
            rows.append({"paper_id": str(node), "community_id": int(community_id)})
    communities = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    communities.to_parquet(output_path, index=False)
    print(f"Wrote {len(groups):,} communities for seed={seed} to {output_path}")
    return communities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cikm_middle.yaml", type=Path)
    parser.add_argument("--edges", default=None, type=Path)
    parser.add_argument("--papers", default=None, type=Path)
    parser.add_argument("--semantic-edges", default="data/processed/semantic_edges_combined.parquet", type=Path)
    parser.add_argument("--weighted-edges-output", default="data/processed/weighted_edges_stability.parquet", type=Path)
    parser.add_argument("--output-dir", default="graph_analysis_outputs/seed_stability", type=Path)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--reference-seed", type=int, default=None)
    parser.add_argument("--neutral-weight", type=float, default=1.0)
    parser.add_argument("--reuse-weighted-edges", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    seed = int(config["project"]["seed"])
    seeds = args.seeds or [int(value) for value in config.get("graph", {}).get("stability_seeds", [seed])]
    reference_seed = args.reference_seed if args.reference_seed is not None else seed
    if reference_seed not in seeds:
        seeds = [reference_seed] + seeds

    interim_dir = Path(paths["interim_dir"])
    edges_path = args.edges or interim_dir / "edges.parquet"
    papers_path = args.papers or interim_dir / "paper_metadata.parquet"
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    pagerank_shifts = compute_pagerank_outputs(graph, papers, args.output_dir, seed)

    ranking_by_seed: dict[int, pd.Series] = {}
    summary_rows: list[dict[str, Any]] = []
    for current_seed in seeds:
        seed_dir = args.output_dir / f"seed_{current_seed}"
        communities = communities_for_seed(graph, current_seed, seed_dir / "node_communities.parquet")
        scores, _ = compute_community_scores(weighted_edges, pagerank_shifts, communities, seed_dir)
        indexed = scores.set_index("community_id")["cci_score"]
        ranking_by_seed[current_seed] = indexed
        scores.assign(seed=current_seed).to_parquet(seed_dir / "community_scores.parquet", index=False)
        summary_rows.append(
            {
                "seed": current_seed,
                "community_count": int(scores.shape[0]),
                "top_community_id": int(scores.iloc[0]["community_id"]) if len(scores) else None,
                "top_cci_score": float(scores.iloc[0]["cci_score"]) if len(scores) else None,
            }
        )

    reference = ranking_by_seed[reference_seed]
    stability_rows = []
    for current_seed, ranking in ranking_by_seed.items():
        stability_rows.append(
            {
                "reference_seed": reference_seed,
                "seed": current_seed,
                "spearman_with_reference": safe_spearman(reference, ranking),
                "top5_overlap_with_reference": top_overlap(reference, ranking, 5),
                "top10_overlap_with_reference": top_overlap(reference, ranking, 10),
                "top25_overlap_with_reference": top_overlap(reference, ranking, 25),
            }
        )

    summary = pd.DataFrame(summary_rows)
    stability = pd.DataFrame(stability_rows)
    summary.to_csv(args.output_dir / "cci_seed_summary.csv", index=False)
    stability.to_csv(args.output_dir / "cci_seed_stability.csv", index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "edges_path": str(edges_path),
        "semantic_edges_path": str(args.semantic_edges),
        "weighted_edges_path": str(args.weighted_edges_output),
        "papers_path": str(papers_path),
        "output_dir": str(args.output_dir),
        "seeds": seeds,
        "reference_seed": reference_seed,
        "typed_edge_rows": int(weighted_edges["is_semantically_typed"].sum()),
    }
    write_json(args.output_dir / "cci_seed_stability_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    print("CCI seed stability complete")


if __name__ == "__main__":
    main()
