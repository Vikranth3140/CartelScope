#!/usr/bin/env python3
# T5-visual_dashboard.py
# CiteGraphLens – Interactive Visual Analytics Dashboard
#
# Run:
#   streamlit run T5-visual_dashboard.py
#
# Requires:
#   pip install streamlit plotly pyvis networkx pandas numpy

import os
import json
import tempfile
import webbrowser
from typing import List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from pyvis.network import Network

import streamlit as st

# -----------------------------
# Paths (relative to this script)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "output")

PATH_EDGES = os.path.join(DATA_DIR, "edges_cleaned.csv")        # from T1b
PATH_META_EXT = os.path.join(DATA_DIR, "metadata_extended.csv") # from T1b
PATH_METRICS = os.path.join(OUT_DIR, "graph_metrics.csv")       # from T2
PATH_INTENTS = os.path.join(OUT_DIR, "intent_labels.csv")       # from T3
PATH_BIAS = os.path.join(OUT_DIR, "bias_summary.csv")           # from T4
PATH_SUMMARY = os.path.join(OUT_DIR, "graph_summary.json")      # from T2

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str, dtype_map: dict = None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype_map)
    except Exception:
        # mixed types are common in these files; fallback
        return pd.read_csv(path, low_memory=False)

@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def kpi_card(label: str, value, help_text: str = ""):
    st.metric(label, value, help=help_text)

def compute_domain_flow(edges: pd.DataFrame, meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (source_domain -> target_domain) counts and row-normalized shares.
    Assumes metadata_extended includes 'paper_id' and 'domain' (may be empty).
    """
    if edges.empty or meta.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Domain mapping
    dom = meta[["paper_id", "domain"]].copy()
    dom["domain"] = dom["domain"].fillna("").replace("", "External")
    dom = dom.drop_duplicates("paper_id")

    df = edges.merge(dom.rename(columns={"paper_id": "source", "domain": "source_domain"}), on="source", how="left")
    df = df.merge(dom.rename(columns={"paper_id": "target", "domain": "target_domain"}), on="target", how="left")
    df["source_domain"] = df["source_domain"].fillna("External")
    df["target_domain"] = df["target_domain"].fillna("External")

    count = df.groupby(["source_domain", "target_domain"]).size().unstack(fill_value=0).sort_index(axis=0).sort_index(axis=1)

    norm = count.div(count.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    # order columns
    desired = ["Biology", "Computer Science", "Medicine", "External"]
    cols = [c for c in desired if c in norm.columns] + [c for c in norm.columns if c not in desired]
    norm = norm[cols]
    count = count[cols]
    return count, norm

def build_pyvis_subgraph(G: nx.DiGraph, nodes: List[str], meta: pd.DataFrame, metrics: pd.DataFrame, max_nodes: int = 200) -> str:
    """
    Create a PyVis network HTML (returns file path). Limits to max_nodes to keep it responsive.
    """
    # Sample large communities for visualization responsiveness
    nodes_sub = nodes[:max_nodes]

    H = G.subgraph(nodes_sub).copy()
    # decorate nodes
    meta_idx = meta.set_index("paper_id") if "paper_id" in meta.columns else pd.DataFrame()
    metrics_idx = metrics.set_index("node_id") if "node_id" in metrics.columns else pd.DataFrame()

    net = Network(height="720px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222")
    net.toggle_physics(True)
    net.barnes_hut(gravity=-20000, central_gravity=0.1, spring_length=120, spring_strength=0.01, damping=0.9)

    for n in H.nodes():
        title = n
        dom = ""
        year = ""
        inst = ""
        if not meta_idx.empty and n in meta_idx.index:
            mrow = meta_idx.loc[n]
            dom = str(mrow.get("domain", ""))
            year = str(mrow.get("year", ""))
            inst = str(mrow.get("institution", ""))
        deg_in = H.in_degree(n)
        deg_out = H.out_degree(n)
        size = 8 + 2 * (deg_in + deg_out)**0.5
        color = {"Biology": "#4e79a7", "Computer Science": "#59a14f", "Medicine": "#e15759"}.get(dom, "#9d9da1")
        hint = f"<b>{n}</b><br>domain: {dom or '—'}<br>year: {year or '—'}<br>inst: {inst or '—'}<br>in:{deg_in} out:{deg_out}"
        net.add_node(n, label=n, title=hint, color=color, size=size)

    for u, v in H.edges():
        net.add_edge(u, v, arrows="to", color="#bbbbbb")

    # Write to a temp file
    tmpdir = tempfile.mkdtemp(prefix="citelens_")
    html_path = os.path.join(tmpdir, "community_graph.html")
    net.show(html_path)
    return html_path

@st.cache_data(show_spinner=False)
def build_graph_from_edges(edges: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    if not edges.empty:
        G.add_edges_from(edges[["source", "target"]].itertuples(index=False, name=None))
    return G

def df_download_button(df: pd.DataFrame, filename: str, label: str):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )

# -----------------------------
# UI – Sidebar
# -----------------------------
st.set_page_config(page_title="CiteGraphLens – Visual Dashboard", layout="wide")
st.title("CiteGraphLens – Visual Analytics Dashboard")

with st.sidebar:
    st.header("Data Sources")
    st.caption("Make sure you ran T1b, T2, T3, and T4 first.")
    st.write(f"**Edges:** `{os.path.relpath(PATH_EDGES, BASE_DIR)}`")
    st.write(f"**Metadata (extended):** `{os.path.relpath(PATH_META_EXT, BASE_DIR)}`")
    st.write(f"**Graph metrics:** `{os.path.relpath(PATH_METRICS, BASE_DIR)}`")
    st.write(f"**Intent labels:** `{os.path.relpath(PATH_INTENTS, BASE_DIR)}`")
    st.write(f"**Bias summary:** `{os.path.relpath(PATH_BIAS, BASE_DIR)}`")

# -----------------------------
# Load data
# -----------------------------
edges = load_csv(PATH_EDGES, dtype_map={"source": str, "target": str})
meta_ext = load_csv(PATH_META_EXT, dtype_map={"paper_id": str})
metrics = load_csv(PATH_METRICS, dtype_map={"node_id": str})
intents = load_csv(PATH_INTENTS)
bias = load_csv(PATH_BIAS)
summary = load_json(PATH_SUMMARY)

# Fallbacks/renames
if "node_id" not in metrics.columns and "paper_id" in metrics.columns:
    metrics = metrics.rename(columns={"paper_id": "node_id"})
if "paper_id" not in meta_ext.columns and "id" in meta_ext.columns:
    meta_ext = meta_ext.rename(columns={"id": "paper_id"})

# -----------------------------
# KPI Row
# -----------------------------
st.subheader("Overview")

col1, col2, col3, col4, col5 = st.columns(5)
nodes_kpi = summary.get("nodes", len(metrics)) if summary else len(metrics)
edges_kpi = summary.get("edges", len(edges)) if summary else len(edges)
density_kpi = summary.get("density", np.nan)
recip_kpi = summary.get("reciprocity", np.nan)

with col1: kpi_card("Nodes", f"{nodes_kpi:,}")
with col2: kpi_card("Edges", f"{edges_kpi:,}")
with col3: kpi_card("Density", f"{density_kpi:.6f}" if pd.notna(density_kpi) else "—")
with col4: kpi_card("Reciprocity", f"{recip_kpi:.5f}" if pd.notna(recip_kpi) else "—")
with col5:
    dom_counts = meta_ext["domain"].fillna("").replace("", "External").value_counts().to_dict() if not meta_ext.empty else {}
    kpi_card("Domains (seed)", ", ".join([f"{k}:{v}" for k, v in dom_counts.items()][:3]) if dom_counts else "—", "Top counts")

st.divider()

# -----------------------------
# Bias plots & table
# -----------------------------
st.subheader("Potential Citation Cartels (from T4)")
if bias.empty:
    st.info("No `bias_summary.csv` found yet.")
else:
    # Filters
    left, right = st.columns([2, 1])
    with right:
        min_size = st.slider("Minimum community size", 2, int(bias["size"].max()), 3, step=1)
        max_show = st.slider("Show top N by bias score", 5, 50, 15, step=5)

    bias_f = bias.copy()
    bias_f = bias_f[bias_f["size"] >= min_size].sort_values("bias_score", ascending=False)

    with left:
        fig_bar = px.bar(
            bias_f.head(max_show),
            x="community_id",
            y="bias_score",
            color="potential_cartel",
            color_discrete_map={True: "crimson", False: "steelblue"},
            hover_data=["size", "avg_self_ratio", "reciprocity_index"],
            labels={"community_id": "Community", "bias_score": "Bias Score"},
            title="Bias Scores (top by score)"
        )
        fig_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter
    c1, c2 = st.columns(2)
    with c1:
        fig_sc = px.scatter(
            bias_f,
            x="reciprocity_index",
            y="avg_self_ratio",
            color="potential_cartel",
            size="size",
            hover_name="community_id",
            labels={"reciprocity_index": "Reciprocity", "avg_self_ratio": "Self-citation"},
            title="Reciprocity vs Self-citation by Community"
        )
        fig_sc.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_sc, use_container_width=True)

    with c2:
        st.dataframe(
            bias_f.head(100).reset_index(drop=True),
            use_container_width=True,
            height=420
        )
        df_download_button(bias_f, "bias_summary_filtered.csv", "⬇️ Download filtered bias table")

st.divider()

# -----------------------------
# Domain ↔ Domain flows (recompute on-demand)
# -----------------------------
st.subheader("Domain → Domain Citation Flow")
if edges.empty or meta_ext.empty:
    st.info("Need `edges_cleaned.csv` and `metadata_extended.csv` to compute domain flows.")
else:
    count_flow, norm_flow = compute_domain_flow(edges, meta_ext)
    if norm_flow.empty:
        st.info("No flow matrix computed.")
    else:
        st.caption("Row-normalized shares (each row sums to 1).")
        fig_hm = px.imshow(
            norm_flow.values,
            x=list(norm_flow.columns),
            y=list(norm_flow.index),
            color_continuous_scale="YlGnBu",
            text_auto=".2f",
            aspect="auto",
        )
        fig_hm.update_layout(
            xaxis_title="Cited Domain →",
            yaxis_title="Citing Domain ↓",
            margin=dict(l=10, r=10, t=30, b=10),
            coloraxis_colorbar=dict(title="Share"),
            title="Normalized Domain–Domain Citation Flow"
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        st.caption("Raw counts")
        st.dataframe(count_flow, use_container_width=True)
        df_download_button(count_flow.reset_index(), "domain_flow_counts.csv", "⬇️ Download counts")

st.divider()

# -----------------------------
# Community Explorer
# -----------------------------
st.subheader("Community Explorer")
if metrics.empty:
    st.info("Need `graph_metrics.csv` from T2.")
else:
    # Build graph (once)
    G = build_graph_from_edges(edges)

    # Sidebar filters for communities
    all_comms = metrics["community"].dropna().unique().tolist() if "community" in metrics.columns else []
    if not all_comms:
        st.info("No community assignments present in graph_metrics.csv.")
    else:
        # Selection controls
        colA, colB, colC = st.columns([2, 1, 1])
        with colA:
            # Suggest a few suspicious defaults if bias file exists
            default_comm = None
            if not bias.empty:
                # handle both 'community' and 'community_id' column names
                if "community" in bias.columns:
                    default_comm = int(bias.sort_values("bias_score", ascending=False).iloc[0]["community"])
                elif "community_id" in bias.columns:
                    default_comm = int(bias.sort_values("bias_score", ascending=False).iloc[0]["community_id"])
                else:
                    default_comm = None
            selected_comm = st.selectbox(
                "Select community",
                sorted(all_comms),
                index=(sorted(all_comms).index(default_comm) if default_comm in all_comms else 0)
            )
        with colB:
            max_nodes_show = st.number_input("Max nodes to visualize", 50, 1000, 200, step=50)
        with colC:
            deg_min = st.number_input("Min in+out degree (filter table)", 0, 1000, 0, step=1)

        comm_nodes = metrics.loc[metrics["community"] == selected_comm, "node_id"].astype(str).tolist()
        st.write(f"**Community {selected_comm}** — {len(comm_nodes):,} nodes")

        # Table of members with metadata + metrics
        sub_metrics = metrics[metrics["community"] == selected_comm].copy()
        if deg_min > 0:
            sub_metrics = sub_metrics[(sub_metrics["in_deg"] + sub_metrics["out_deg"]) >= deg_min]

        # Attach metadata where possible
        if not meta_ext.empty and "paper_id" in meta_ext.columns:
            sub = sub_metrics.merge(
                meta_ext[["paper_id", "title", "year", "domain", "institution", "country"]]
                .rename(columns={"paper_id": "node_id"}),
                on="node_id",
                how="left"
            )
        else:
            sub = sub_metrics.copy()

        # Sort by in-degree (citations received) as default
        if "in_deg" in sub.columns:
            sub = sub.sort_values("in_deg", ascending=False)

        st.dataframe(sub.reset_index(drop=True), use_container_width=True, height=420)
        df_download_button(sub, f"community_{selected_comm}_members.csv", "⬇️ Download members (filtered)")

        # Build & show PyVis network for selected community
        st.markdown("**Interactive Graph**")
        try:
            html_path = build_pyvis_subgraph(G, comm_nodes, meta_ext, metrics, max_nodes=int(max_nodes_show))
            # Show as iframe
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=740, scrolling=True)
        except Exception as e:
            st.warning(f"Could not render interactive graph: {e}")

        # Intra-community suspicious pairs (mutual citations)
        st.markdown("**Mutual Citation Pairs (within community)**")
        if not edges.empty:
            comm_edges = edges[edges["source"].isin(comm_nodes) & edges["target"].isin(comm_nodes)]
            # find reciprocals
            pairs = set()
            for u, v in comm_edges[["source", "target"]].itertuples(index=False, name=None):
                if (v, u) in pairs or u == v:
                    continue
                pairs.add((u, v))
            reciprocals = [(u, v) for (u, v) in pairs if ((v, u) in pairs)]
            rec_df = pd.DataFrame(reciprocals, columns=["u", "v"])
            if not rec_df.empty:
                st.dataframe(rec_df, use_container_width=True)
                df_download_button(rec_df, f"community_{selected_comm}_mutual_pairs.csv", "⬇️ Download mutual pairs")
            else:
                st.caption("No mutual pairs detected in this community.")

st.divider()

# -----------------------------
# Intent distribution (from T3)
# -----------------------------
st.subheader("Citation Intent Distribution (sampled)")
if intents.empty:
    st.info("No `intent_labels.csv` found from T3.")
else:
    # If T3 saved columns source_paper, intent; otherwise fallback on citation_id parsing
    intents_df = intents.copy()
    if "source_paper" not in intents_df.columns and "citation_id" in intents_df.columns:
        intents_df["source_paper"] = intents_df["citation_id"].astype(str).str.split("_cites_").str[0]

    # Attach domain if we can
    if not meta_ext.empty:
        dom_map = meta_ext[["paper_id", "domain"]].rename(columns={"paper_id": "source_paper"})
        intents_df = intents_df.merge(dom_map, on="source_paper", how="left")
        intents_df["domain"] = intents_df["domain"].fillna("External")
    else:
        intents_df["domain"] = "External"

    colL, colR = st.columns(2)
    with colL:
        fig_int = px.histogram(
            intents_df,
            x="intent",
            color="domain",
            barmode="group",
            title="Citation Intent by Domain (sample)",
        )
        fig_int.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_int, use_container_width=True)

    with colR:
        top_contexts = intents_df.sort_values("probability", ascending=False).head(25)[
            ["citation_id", "source_paper", "cited_paper", "intent", "probability", "domain"]
        ]
        st.dataframe(top_contexts, use_container_width=True, height=420)
        df_download_button(top_contexts, "top_intent_contexts.csv", "⬇️ Download top contexts")

st.caption("© CiteGraphLens – T5 Visual Dashboard")
