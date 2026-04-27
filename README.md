<p align="center">
  <h1 align="center">🔍 Topological Signatures of Citation Cartels<br/>in Scale-Free Academic Networks</h1>
  <p align="center">
    <em>A multi-modal pipeline for detecting organized citation fraud<br/>using Network Science, Large Language Models, and Graph Theory</em>
  </p>
  <p align="center">
    <a href="#-overview">Overview</a> •
    <a href="#-key-findings-at-a-glance">Findings</a> •
    <a href="#-the-pipeline">Pipeline</a> •
    <a href="#-figure-gallery">Gallery</a> •
    <a href="#-deep-dives">Deep Dives</a> •
    <a href="#-getting-started">Setup</a> •
    <a href="#-project-structure">Structure</a> •
    <a href="#-authors">Authors</a>
  </p>
</p>

---

## 📌 Overview

**Citation cartels** — covert groups of researchers who systematically inflate each other's h-indexes through superficial, reciprocal, or ceremonial citations — are one of the most insidious threats to scientific integrity. They distort rankings, misdirect funding, and erode trust in the academic enterprise.

This project constructs a **complete, end-to-end detection pipeline** applied to a macroscopic citation network of **500,000 academic papers** and **4.87 million citation edges**, sampled via topological snowball BFS from the [DBLP v12 dataset](https://www.kaggle.com/datasets/mathurinache/citation-network-dataset). The pipeline proceeds through five integrated stages:

1. **Structural Baseline** — Verify the network is scale-free and small-world; decompose its bow-tie architecture
2. **Scale-Free Proof** — MLE power-law fitting confirms γ ∈ (2, 3) across all degree types
3. **Semantic Annotation** — A Teacher-Student LLM framework classifies every citation edge into 6 intent categories and assigns semantic weights
4. **Cartel Detection** — Louvain community detection + a 6-metric Composite Cartel Index flags anomalous communities
5. **Resilience Surgery** — Targeted attack simulations and a novel cartel edge excision experiment prove cartels are non-load-bearing

### 🎯 The Core Research Question

> *Can we detect citation cartels purely from the topological signatures they leave in the citation graph — and what happens to the network when we surgically remove them?*

**Answer:** Yes. Cartel communities exhibit a characteristic fingerprint — extreme internal citation inflation (49.6× random expectation), abnormal reciprocity, and overwhelmingly shallow citation intent. Most critically, **their complete removal produces zero degradation** of the giant component. They are parasitic appendages, not load-bearing structures.

---

## 📊 Key Findings at a Glance

### Network Properties

| Property | Our Citation Network | Erdős-Rényi Null | Barabási-Albert Null |
|----------|---------------------|------------------|---------------------|
| **Nodes / Edges** | 500,000 / 4,871,544 | Same N, ⟨k⟩ | Same N, m |
| **Clustering Coefficient** | **0.265** | 0.0004 | 0.0025 |
| **Avg. Path Length** | **3.92** | — | — |
| **Degree Exponent (γ)** | **2.215** (in) · 2.367 (total) · 2.956 (out) | Poisson | γ = 3 |
| **Reciprocity** | 0.0025 | ≈ 0 | ≈ 0 |
| **Degree Assortativity** | −0.112 (disassortative) | ≈ 0 | < 0 |

### Bow-Tie Architecture

| Component | Nodes | % of Network |
|-----------|-------|-------------|
| Giant SCC Core | 12,872 | 2.57% |
| IN-Component | 387,670 | 77.53% |
| OUT-Component | 18,100 | 3.62% |
| Tendrils & Tubes | 81,358 | 16.27% |

### Cartel Detection Results

| Community | Nodes | Cartel Index (Z) | Inflation vs Random | Reciprocity |
|-----------|-------|-------------------|--------------------:|-------------|
| **10** | 11 | **1.235** | **49.63×** | 0.14 |
| 30 | 31 | 0.771 | 2.00× | 0.00 |
| 25 | 26 | 0.539 | 3.00× | 0.00 |

> Community 10 cites itself **50× more than random chance predicts**, with 14% reciprocal edges — a smoking gun for organized citation manipulation.

### The Definitive Verdict

| Experiment | Giant Component S(f) | Avg Path Length ⟨d⟩ |
|------------|:--------------------:|:-------------------:|
| Cartel edges removed | **No change** | **No change** |
| Random edges removed (same count) | **No change** | **No change** |
| Hub nodes removed (targeted attack) | **Rapid collapse** | **Diverges** |

**Cartel citation edges are structurally indistinguishable from random noise.** Their removal restores meritocratic citation dynamics without collateral damage to legitimate science.

---

## 🧬 The Pipeline

```
 ╔══════════════════════════════════════════════════════════════════════════════╗
 ║  STAGE 1 · TOPOLOGICAL BASELINE & STRUCTURAL DECOMPOSITION                   ║
 ║  ──────────────────────────────────────────────────────────────────────      ║
 ║  Bidirectional snowball BFS from 200 mega-hubs → 500K papers                 ║
 ║  Clustering C = 0.265 (618× ER null) · ⟨d⟩ = 3.92 · Diameter = 6             ║
 ║  Bow-tie decomposition via Tarjan's algorithm                                ║
 ║  Reciprocity analysis · Degree assortativity · K_nn correlation              ║
 ╠══════════════════════════════════════════════════════════════════════════════╣
 ║  STAGE 2 · SCALE-FREE VERIFICATION & PREFERENTIAL ATTACHMENT                 ║
 ║  ──────────────────────────────────────────────────────────────────────      ║
 ║  Log-log degree distributions (in / out / total)                             ║
 ║  MLE power-law fitting via powerlaw library → γ ∈ (2.2, 3.0)                 ║
 ║  KS goodness-of-fit: power-law vs exponential vs log-normal                  ║
 ║  Hexbin in-degree vs out-degree landscape visualization                      ║
 ╠══════════════════════════════════════════════════════════════════════════════╣
 ║  STAGE 3 · SEMANTIC EDGE ANNOTATION (TEACHER-STUDENT LLM)                    ║
 ║  ──────────────────────────────────────────────────────────────────────      ║
 ║  Teacher: 5× Gemma-3 LLMs label 3,000 citation edges                         ║
 ║  Taxonomy: Background · Method · Result · Support · Contrast · Perfunctory   ║
 ║  Student: Fine-tuned SciBERT (3 epochs, 2e-5 LR, FP16 precision)             ║
 ║  Binary graph → Semantically weighted network (w ∈ [0.1, 1.0])               ║
 ║  Weighted vs unweighted PageRank → papers inflated by padding exposed        ║
 ╠══════════════════════════════════════════════════════════════════════════════╣
 ║  STAGE 4 · COMMUNITY DETECTION & COMPOSITE CARTEL SCORING                    ║
 ║  ──────────────────────────────────────────────────────────────────────      ║
 ║  Louvain algorithm → 32 communities, M = 0.813                               ║
 ║  Configuration model null (10 instances) → Z = 59.61                         ║
 ║  NMI domain alignment = 0.514                                                ║
 ║  Composite Cartel Index = mean Z-score across 6 orthogonal metrics:          ║
 ║    Density · Inflation · Reciprocity · Superficiality · Assortativity ·      ║
 ║    PageRank Anomaly                                                          ║
 ║  Triad census: 16 directed MAN motif types vs null expectation               ║
 ╠══════════════════════════════════════════════════════════════════════════════╣
 ║  STAGE 5 · NETWORK RESILIENCE & CARTEL SURGERY                               ║
 ║  ──────────────────────────────────────────────────────────────────────      ║
 ║  Random failure · Degree-based attack · Betweenness-based attack             ║
 ║  ER & BA null model resilience comparison                                    ║
 ║  Novel experiment: surgical cartel edge excision vs random edge removal      ║
 ║  Result: S(f) unchanged → cartels are hollow, non-load-bearing structures    ║
 ╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 🖼️ Figure Gallery

The pipeline generates **22 publication-ready figures** across two output directories. Below is a categorized index:

### Topological Baseline (`graph_analysis_outputs/`)

| Figure | Description |
|--------|-------------|
| `graph_preview_hubs.png` | Force-directed layout of the full network highlighting mega-citation hubs |
| `bowtie_structure.png` | Bow-tie macroscopic decomposition: SCC core, IN/OUT components, tendrils |
| `path_length_distribution.png` | Shortest path length distribution confirming small-world connectivity (⟨d⟩ ≈ 4) |
| `year_distribution.png` | Publication year distribution showing cumulative temporal growth of the literature |
| `n_citation_distribution.png` | Citation count frequency distribution revealing heavy-tailed accumulation |
| `degree_correlation_knn.png` | K_nn scatter plot proving disassortative mixing (hubs connect to low-degree nodes) |

### Scale-Free Verification (`graph_analysis_outputs/`)

| Figure | Description |
|--------|-------------|
| `in_degree_hist_loglog.png` | In-degree histogram on log-log scale — straight line confirms power-law |
| `total_degree_hist_loglog.png` | Total degree histogram on log-log scale |
| `powerlaw_fit_in-degree.png` | MLE-fitted CCDF for in-degree (γ = 2.215) |
| `powerlaw_fit_out-degree.png` | MLE-fitted CCDF for out-degree (γ = 2.956) |
| `powerlaw_fit_total_degree.png` | MLE-fitted CCDF for total degree (γ = 2.367) |
| `in_vs_out_degree_hexbin.png` | Hexbin density: writing more references ≠ receiving more citations |

### Semantic Annotation (`data/`)

| Figure | Description |
|--------|-------------|
| `citation_subgraph.png` | Force-directed visualization of the annotated citation subgraph colored by community, showing the dense local clusters where cartel-like behavior is analyzed |
| `intent_distribution.png` | Bar chart of citation intent class distribution from the Teacher-Student pipeline — reveals that ~51% of all citations are shallow "Background" references with only ~12% classified as deep "Method" dependencies |

### Cartel Detection & Resilience (`graph_analysis_outputs/`)

| Figure | Description |
|--------|-------------|
| `cartel_index_distribution.png` | Histogram of Composite Cartel Z-scores — distinct right tail reveals outlier communities |
| `network_cartels.png` | Full network force-directed layout with suspected cartel communities highlighted in **red** |
| `triad_profile.png` | Triad Significance Profile: z-scores for all 16 directed MAN motif types vs null |
| `resilience_nodes_comparison.png` | S(f) decay curves: random failure vs degree attack vs betweenness attack |
| `resilience_null_models.png` | Side-by-side resilience comparison: Real network vs ER null vs BA null |
| `cartel_edge_removal_impact.png` | The novel experiment — cartel edge excision vs random edge removal, proving zero structural impact |
| `cartel_community_scores.csv` | Full scoring table for all 32 detected communities across 6 anomaly metrics |

---

## 🔬 Deep Dives

### Why Snowball Sampling? The Density Problem

A naive random sample of 500K papers from a 5M-paper corpus produces a **catastrophically sparse** graph. Our **bidirectional snowball BFS** from the top-200 cited hubs completely solves this:

| | Random 10% Sample | Snowball BFS |
|---|:---:|:---:|
| **Isolates** | 212,421 (43.3%) | **0 (0.0%)** |
| **Edge / Node Ratio** | 0.94 | **9.74** |
| **Largest SCC** | 8 nodes | **12,872 nodes** |
| **Scale-Free Preserved?** | ✗ Destroyed | **✓ Fully intact** |

By expanding in both directions — forward through references (preserving out-degree) and backward through citers (preserving in-degree) — the BFS honors the rich-get-richer attachment process that generated the original power law. See [`sample_dataset_explanation.md`](sample_dataset_explanation.md) for the complete 19-page technical breakdown of the algorithm.

### The Teacher-Student LLM Framework

Not all citations are equal. A paper citing another as *"the method we directly build upon"* has a fundamentally different relationship than one padding a perfunctory background mention. We built a two-stage NLP pipeline:

| Stage | Model | Purpose | Scale |
|-------|-------|---------|-------|
| **Teacher** | 5× Gemma-3 variants | Generate ground-truth intent labels by analyzing title-abstract pairs | 3,000 edges |
| **Student** | Fine-tuned SciBERT | Scale annotation to the full graph with memory-efficient inference | Full network |

Each citation is classified into one of **6 intent categories** and assigned a semantic edge weight:

| Intent | Weight | Cartel Signal |
|--------|:------:|:-------------:|
| Method | 1.0 | Low risk — genuine dependency |
| Result/Comparison | 0.7 | Medium — legitimate |
| Support | 0.5 | Medium-high — can be gamed |
| Contrast/Criticism | 0.3 | Low risk — adversarial |
| Background | 0.2 | High — general context |
| **Perfunctory/Ceremonial** | **0.1** | **Highest — padding** |

**Impact:** When PageRank is re-computed on the weighted graph, papers propped up by shallow citations **plummet in rank** while genuinely impactful work rises — a direct, quantitative separation of real influence from artificially inflated metrics.

### The Composite Cartel Index

Each of the 32 detected Louvain communities is scored across **6 orthogonal structural anomaly metrics**, Z-score normalized, and averaged:

```
Cartel Index(C) = mean( Z_density, Z_inflation, Z_reciprocity,
                        Z_superficiality, Z_assortativity, Z_pagerank_anomaly )
```

| Metric | What It Catches |
|--------|----------------|
| **Internal Citation Density** | How tightly the group cites itself |
| **Citation Inflation Score** | Self-citation exceeding random expectation given the degree sequence |
| **Reciprocity ρ(C)** | Quid-pro-quo mutual back-scratching (A↔B) |
| **Superficiality Ratio** | Fraction of internal citations classified as Background/Perfunctory |
| **Degree Assortativity** | Whether hubs selectively cite other hubs within the clique |
| **PageRank Anomaly** | Papers being boosted by strategically placed citation links |

Communities exceeding Z > 2 are flagged. Community 10 scored the highest with an inflation of **49.6× random expectation** and 14% reciprocal edges — statistically indistinguishable from an organized citation ring.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **~6 GB RAM** for graph processing
- **GPU recommended** for SciBERT fine-tuning (Stage 3)

### Installation

```bash
git clone https://github.com/Vikranth3140/NS-Project.git
cd NS-Project

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dataset

Download the **DBLP v12** citation dataset (~12.5 GB, ~4.9M papers) from [AMiner](https://www.aminer.org/citation). Place `dblp.v12.json` inside a `dataset/` directory at the project root.

### Running the Pipeline

```bash
# Stage 1: Snowball sampling → 500K-paper subgraph (~30 min)
python sample_dataset.py

# Stages 1-2: Topological profiling & scale-free verification
jupyter notebook data_analysis.ipynb

# Stage 3: Semantic edge annotation (Teacher-Student LLM)
jupyter notebook deliverable_2.ipynb

# Stage 4: Community detection & cartel scoring
python deliverable_3.py

# Stage 5: Network resilience & cartel surgery
python deliverable_4.py
```

---

## 📁 Project Structure

```
NS-Project/
│
├── sample_dataset.py                 # Bidirectional snowball BFS sampler (500K from DBLP)
├── sample_dataset_explanation.md     # 19-page technical deep-dive of the sampling algorithm
│
├── data_analysis.ipynb               # Stage 1-2: Topological profiling & scale-free verification
├── deliverable_2.ipynb               # Stage 3: Teacher-Student LLM semantic annotation
├── deliverable_3.py                  # Stage 4: Louvain communities & composite cartel scoring
├── deliverable_3a.ipynb              # Stage 4: Girvan-Newman hierarchical analysis
├── deliverable_3b.py                 # Stage 4: Triad census & motif significance profiling
├── deliverable_3b.ipynb              # Stage 4: Triad analysis (notebook version)
├── deliverable_4.py                  # Stage 5: Resilience simulations & cartel edge excision
├── deliverable_4.ipynb               # Stage 5: Resilience analysis (notebook version)
│
├── data/                             # Intermediate graph artifacts & annotation visuals
│   ├── weighted_citation_graph.graphml    # Semantically weighted directed graph
│   ├── unweighted_citation_graph.graphml  # Binary citation graph
│   ├── weighted_edges.json                # Edge-level intent labels + weights
│   ├── citation_subgraph.png              # Annotated subgraph visualization
│   └── intent_distribution.png            # Citation intent class distribution
│
├── graph_analysis_outputs/           # All generated figures & analysis results (20 files)
│   ├── bowtie_structure.png               # Bow-tie macroscopic decomposition
│   ├── graph_preview_hubs.png             # Full network force-directed layout
│   ├── network_cartels.png                # Cartel communities highlighted in red
│   ├── cartel_index_distribution.png      # Composite Cartel Z-score histogram
│   ├── cartel_edge_removal_impact.png     # Novel cartel surgery experiment
│   ├── resilience_nodes_comparison.png    # Random vs targeted attack S(f) curves
│   ├── resilience_null_models.png         # ER & BA null model comparison
│   ├── triad_profile.png                  # 16-motif triad significance profile
│   ├── powerlaw_fit_*.png                 # MLE fits (in/out/total degree)
│   ├── *_hist_loglog.png                  # Log-log degree distributions
│   ├── degree_correlation_knn.png         # Degree-degree correlation scatter
│   ├── in_vs_out_degree_hexbin.png        # In vs out degree density landscape
│   ├── path_length_distribution.png       # Shortest path distribution
│   ├── year_distribution.png              # Temporal publication growth
│   ├── n_citation_distribution.png        # Citation count frequency
│   └── cartel_community_scores.csv        # Full scoring table (all 32 communities)
│
├── PPT/                              # PPT presentation for the project
│   └──Topological-Signatures-of-Citation-Cartels-in-Scale-Free-Networks.pdf              
│   
├── report/                           # Final academic deliverable
│   ├── NS Project report.tex              # Full LaTeX source (~800 lines)
│   └── NS_Project_Report.pdf              # Compiled PDF report
│
├── deliverables_analysis.md          # Comprehensive deliverable specifications
├── Citation_Network_Project_Plan.md  # Initial project planning document
├── dataset_analysis.txt              # Raw snowball sampling pipeline output log
├── requirements.txt                  # Python dependencies
├── .gitignore
└── LICENSE                           # MIT
```

---

## 👥 Authors

<table>
  <tr>
    <td align="center"><strong>Pratyush Gupta</strong><br/>(2022375)</td>
    <td align="center"><strong>Syam Sai Santosh Bandi</strong><br/>(2022528)</td>
    <td align="center"><strong>Vikranth Udandarao</strong><br/>(2022570)</td>
  </tr>
</table>

*Network Science Course Project*

## License

This project is licensed under the [MIT License](LICENSE).