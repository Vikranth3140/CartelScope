#!/usr/bin/env python3
"""
CiteGraphLens: T2 - Citation Graph Analysis
-------------------------------------------
Builds a directed citation graph from cleaned data,
computes centrality and structural metrics,
analyzes community structure, and visualizes results.
"""

import os
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import community as community_louvain
import json
from datetime import datetime
import logging

# ============ Logging Setup ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============ Utility Functions ============

def load_data(data_dir):
    """
    Load edges and metadata from cleaned CSV files
    """
    edges_path = os.path.join(data_dir, "edges_cleaned.csv")
    meta_path = os.path.join(data_dir, "metadata_extended.csv")

    logger.info(f"📂 Loading edges from {edges_path}")
    edges_df = pd.read_csv(edges_path)

    logger.info(f"📘 Loading metadata from {meta_path}")
    metadata_df = pd.read_csv(meta_path)

    if metadata_df["paper_id"].duplicated().any():
        dup_count = metadata_df["paper_id"].duplicated().sum()
        logger.warning(f"⚠️ Found {dup_count} duplicate paper_ids. Keeping first occurrence.")
        metadata_df = metadata_df.drop_duplicates(subset="paper_id", keep="first")

    metadata_df.set_index("paper_id", inplace=True)
    logger.info(f"✅ Loaded {len(metadata_df):,} papers and {len(edges_df):,} edges.")
    return edges_df, metadata_df


def build_graph(edges_df, metadata_df):
    """
    Build directed citation graph with node attributes from metadata
    
    Parameters:
    -----------
    edges_df : pandas.DataFrame
        DataFrame containing citation edges (source, target)
    metadata_df : pandas.DataFrame
        DataFrame containing paper metadata with paper_id as index
    
    Returns:
    --------
    networkx.DiGraph
        Directed graph with node attributes
    """
    logger.info("🧠 Building directed citation graph...")
    G = nx.DiGraph()
    metadata_dict = metadata_df.to_dict("index")

    for _, row in edges_df.iterrows():
        G.add_edge(row["source"], row["target"])

    for node, data in metadata_dict.items():
        if node in G:
            G.nodes[node].update({
                "title": data.get("title", ""),
                "domain": data.get("domain", ""),
                "institution": data.get("institution", ""),
                "country": data.get("country", ""),
                "year": data.get("year", None)
            })

    logger.info(f"✅ Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges.")
    return G


def compute_self_citation_ratio(G):
    """
    Compute self-citation ratio for each node
    
    Parameters:
    -----------
    G : networkx.DiGraph
        Directed citation graph with node attributes
    
    Returns:
    --------
    dict
        Dictionary with node_id as key and self_citation_ratio as value
    """
    ratios = {}
    for node in G.nodes():
        outs = list(G.successors(node))
        if not outs:
            ratios[node] = 0.0
            continue
        inst = G.nodes[node].get("institution", "")
        self_cites = sum(
            1 for tgt in outs if G.nodes[tgt].get("institution", "") == inst and inst
        )
        ratios[node] = self_cites / len(outs)
    return ratios


def compute_metrics(G):
    """
    Compute various graph metrics for each node
    
    Parameters:
    -----------
    G : networkx.DiGraph
        Directed citation graph with node attributes
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with computed metrics for each node
    """
    logger.info("📊 Computing graph metrics...")

    # Louvain communities
    logger.info("🧩 Detecting communities via Louvain method...")
    G_undirected = G.to_undirected()
    try:
        communities = community_louvain.best_partition(G_undirected, random_state=42)
    except Exception as e:
        logger.warning(f"Louvain failed: {e}")
        communities = {n: 0 for n in G.nodes()}

    # Self-citation ratio
    self_ratios = compute_self_citation_ratio(G)

    # Global metrics
    try:
        reciprocity = nx.reciprocity(G)
    except:
        reciprocity = 0
    density = nx.density(G)
    avg_in_deg = np.mean([d for _, d in G.in_degree()])
    avg_out_deg = np.mean([d for _, d in G.out_degree()])

    logger.info(f"🔹 Density: {density:.6f}")
    logger.info(f"🔹 Reciprocity: {reciprocity:.4f}")
    logger.info(f"🔹 Avg In-degree: {avg_in_deg:.2f}, Avg Out-degree: {avg_out_deg:.2f}")

    # Clustering coefficients
    try:
        cluster_vals = nx.clustering(G_undirected)
    except Exception as e:
        logger.warning(f"Error computing clustering: {e}")
        cluster_vals = {n: 0 for n in G.nodes()}

    # Assemble node metrics
    metrics = []
    for n in G.nodes():
        indeg = G.in_degree(n)
        outdeg = G.out_degree(n)
        successors = set(G.successors(n))
        predecessors = set(G.predecessors(n))
        recips = len(successors & predecessors) / len(successors) if successors else 0
        metrics.append({
            "node_id": n,
            "in_deg": indeg,
            "out_deg": outdeg,
            "recip": recips,
            "cluster": cluster_vals.get(n, 0),
            "self_ratio": self_ratios.get(n, 0),
            "community": communities.get(n, -1)
        })

    df = pd.DataFrame(metrics)
    return df, {"density": density, "reciprocity": reciprocity,
                "avg_in_deg": avg_in_deg, "avg_out_deg": avg_out_deg}


def analyze_communities(df):
    """
    Analyze community structure and identify top communities
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        DataFrame with node metrics including community assignments
    
    Returns:
    --------
    list
        List of top 5 communities by size
    """
    if df.empty or "community" not in df.columns:
        logger.warning("No community data to analyze.")
        return []

    counts = df["community"].value_counts()
    logger.info(f"🔸 Detected {len(counts)} communities. Largest = {counts.iloc[0]}")
    densities = df.groupby("community")["cluster"].mean()
    top5 = densities.sort_values(ascending=False).head(5)

    logger.info("🏅 Top 5 densest communities:")
    for cid, dens in top5.items():
        logger.info(f"   • Community {cid} — density {dens:.4f} ({counts[cid]} nodes)")
    return top5.index.tolist()


def plot_histograms(G, metrics_df, output_dir):
    """
    Plot histogram of reciprocity values
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        DataFrame with node metrics
    output_path : str
        Path to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)

    # In-degree histogram
    plt.figure(figsize=(8, 5))
    plt.hist([d for _, d in G.in_degree()], bins=40, color="orange", alpha=0.7)
    plt.xlabel("In-degree (citations received)")
    plt.ylabel("Frequency")
    plt.title("In-degree Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    in_path = os.path.join(output_dir, "in_degree_distribution.png")
    plt.savefig(in_path)
    logger.info(f"📊 Saved → {in_path}")

    # Out-degree histogram
    plt.figure(figsize=(8, 5))
    plt.hist([d for _, d in G.out_degree()], bins=40, color="green", alpha=0.7)
    plt.xlabel("Out-degree (citations made)")
    plt.ylabel("Frequency")
    plt.title("Out-degree Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "out_degree_distribution.png")
    plt.savefig(out_path)
    logger.info(f"📊 Saved → {out_path}")

    # Reciprocity histogram
    plt.figure(figsize=(8, 5))
    plt.hist(metrics_df["recip"], bins=25, color="blue", alpha=0.7)
    plt.xlabel("Reciprocity")
    plt.ylabel("Frequency")
    plt.title("Reciprocity Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    rec_path = os.path.join(output_dir, "reciprocity_histogram.png")
    plt.savefig(rec_path)
    logger.info(f"📊 Saved → {rec_path}")


def main():
    logger.info("🚀 Starting T2 Graph Analysis")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    edges_df, metadata_df = load_data(data_dir)
    G = build_graph(edges_df, metadata_df)
    metrics_df, global_stats = compute_metrics(G)
    analyze_communities(metrics_df)

    # Save node metrics
    metrics_path = os.path.join(output_dir, "graph_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"💾 Saved node metrics → {metrics_path}")

    # Save global summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        **global_stats
    }
    with open(os.path.join(output_dir, "graph_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("📘 Saved graph_summary.json")

    # Identify top cited and citing nodes
    top_in = metrics_df.nlargest(10, "in_deg")[["node_id", "in_deg"]]
    top_out = metrics_df.nlargest(10, "out_deg")[["node_id", "out_deg"]]
    logger.info("🏆 Top 10 most cited papers:")
    for _, r in top_in.iterrows():
        title = metadata_df.loc[r.node_id, "title"] if r.node_id in metadata_df.index else "—"
        logger.info(f"   • {r.node_id} — {r.in_deg} cites — {title[:80]}")
    logger.info("🏆 Top 10 most citing papers:")
    for _, r in top_out.iterrows():
        title = metadata_df.loc[r.node_id, "title"] if r.node_id in metadata_df.index else "—"
        logger.info(f"   • {r.node_id} — {r.out_deg} refs — {title[:80]}")

    # Plot degree & reciprocity histograms
    plot_histograms(G, metrics_df, output_dir)

    logger.info("🎯 Graph Analysis Complete. Proceed to T3 (interpretation/reporting).")


if __name__ == "__main__":
    main()