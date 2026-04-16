# Topological Signatures of Citation Cartels in Scale-Free Academic Networks

## Project Deliverables

---

## Deliverable 1: Large-Scale Graph Construction & Comprehensive Topological Characterization

### Motivation
Every network science study begins with constructing a faithful representation of the system and computing its complete topological fingerprint. Before we can detect anomalies like citation cartels, we must first characterize what the "normal" structure of the citation network looks like — and rigorously prove that its properties deviate from what random chance alone would produce. This deliverable builds the graph, verifies its scale-free and small-world nature, establishes null model baselines, and measures whether the network's growth follows preferential attachment. It is the foundation upon which every subsequent analysis depends.

### Methodology

#### 1.1 Data Acquisition & Closed-World Graph Construction

We ingest our multi-domain academic citation dataset spanning three scientific domains — Computer Science, Biology, and Medicine — sourced from the OpenAlex API. Each paper becomes a node, and each citation relationship becomes a directed edge from the citing paper to the cited paper.

Critically, we construct a **closed-world network**: we only retain citation edges where **both** the source and target papers exist within our curated dataset. This ensures every node can have both incoming and outgoing edges, enabling meaningful computation of reciprocity, cycles, and community structure. Open-ended "dangling" references to external papers are excluded to preserve topological integrity.

**Target scale**: ~30,000 papers (~10,000 per domain) yielding an estimated 100,000–200,000 internal citation edges.

#### 1.2 Baseline Topological Profiling

We compute the complete topological profile of the constructed graph:

| Metric | Description | Formula / Method |
|--------|-------------|------------------|
| N (nodes), L (edges) | Network size | Direct count |
| ⟨k_in⟩, ⟨k_out⟩ | Average in-degree and out-degree | ⟨k⟩ = L/N |
| Network density | Fraction of possible edges realized | d = L / N(N-1) |
| Diameter | Longest shortest path in the network | BFS over sampled pairs |
| Average path length ⟨d⟩ | Mean shortest path between reachable node pairs | BFS / Dijkstra |
| Global clustering coefficient C | Tendency of nodes to form triangles | C_i = 2T_i / k_i(k_i - 1) |
| Degree distribution P(k) | Probability distribution of node degrees | Histogram of k_in, k_out |

#### 1.3 Connected Component Decomposition (Bow-Tie Structure)

Since our graph is directed, we decompose it into its fundamental structural layers:

- **Strongly Connected Component (SCC)**: The largest subset of nodes where every node can reach every other node following edge directions. This is the "core" of the citation network where mutual reachability — and therefore citation loops — can exist.
- **Weakly Connected Component (WCC)**: The largest connected subset when edge directions are ignored.
- **In-Component**: Nodes that can reach the SCC but cannot be reached from it (recent papers citing the core, not yet cited back).
- **Out-Component**: Nodes reachable from the SCC but that cannot reach it (foundational older papers cited by the core).
- **Tendrils and Tubes**: Remaining peripheral structures.

This "bow-tie" decomposition is essential because citation cartels can only exist within the SCC — they require mutual citation cycles.

#### 1.4 Reciprocity Analysis

We compute the **reciprocity** ρ of the directed network: the fraction of edges (A→B) for which the reverse edge (B→A) also exists.

$$\rho = \frac{|\{(i,j) : A_{ij} = 1 \text{ AND } A_{ji} = 1\}|}{L}$$

In normal citation networks, reciprocity is near zero because Paper B is typically published before Paper A, so B cannot cite A. **Abnormally high reciprocity is the most direct topological signature of citation cartels** — groups of authors who systematically cite each other's concurrent or future work. We report both the global reciprocity and per-community reciprocity (computed in Deliverable 3).

#### 1.5 Scale-Free Verification (Power-Law Degree Distribution)

We plot the in-degree and out-degree distributions P(k_in) and P(k_out) on a log-log scale. If the network is scale-free, we expect:

$$P(k) \sim k^{-\gamma}$$

We estimate the degree exponent γ using **Maximum Likelihood Estimation** (MLE) via the `powerlaw` Python library, rather than naive linear regression on log-log plots (which produces biased estimates). We perform a **Kolmogorov-Smirnov goodness-of-fit test** to verify that the power-law fit is statistically better than alternative distributions (exponential, log-normal, stretched exponential).

We identify the **hub papers** — the nodes in the heavy tail — and report:
- The observed maximum degree k_max
- The theoretical prediction: k_max = k_min · N^(1/(γ-1))
- Whether observed k_max matches the prediction (pure scale-free) or deviates (cutoffs or anomalies)

#### 1.6 Erdős-Rényi Null Model Comparison

We generate an Erdős-Rényi (ER) random directed graph G(N, p) with the same number of nodes N and the same average degree ⟨k⟩ as our real network. We compute all baseline metrics on the ER graph and systematically compare:

| Property | Citation Network | ER Random Graph | What the Difference Means |
|----------|-----------------|-----------------|--------------------------|
| P(k) | Power-law (heavy tail) | Poisson (narrow bell) | Real network has hubs; random does not |
| Clustering C | High | ⟨k⟩/N ≈ 0 | Real network has local structure |
| ⟨d⟩ | Short | log(N)/log(⟨k⟩) | Both are "small world" but for different reasons |
| Reciprocity ρ | Measured | p (near zero) | Any excess reciprocity is structurally meaningful |
| Giant component | Measured | Predicted by p·N > 1 threshold | Real network may be more/less connected |

This comparison is the methodological backbone of the entire project. Every subsequent finding (communities, cartels, resilience) gains credibility only because we can show it deviates from the random null.

#### 1.7 Small-World Property Verification

We verify the Watts-Strogatz "small-world" property by showing our network simultaneously exhibits:
1. **High clustering**: C >> C_random = ⟨k⟩/N (like a regular lattice)
2. **Short path lengths**: ⟨d⟩ ≈ log(N)/log(⟨k⟩) (like a random graph)

This dual property means our network has tight local neighborhoods (research groups that co-cite) connected by occasional long-range shortcuts (interdisciplinary citations) — exactly the topology where cartels could hide within dense local clusters.

#### 1.8 Temporal Growth Dynamics & Preferential Attachment Measurement

Using the `year` field from our metadata, we reconstruct the growth of the network over time:
- Plot N(t) (number of papers) and L(t) (number of edges) as a function of publication year
- Plot ⟨k⟩(t) — does the average degree increase, stay constant, or decrease over time?

**Measuring Preferential Attachment** (following Jeong, Néda, Barabási, Europhys. Lett. 2003):

1. Divide the timeline into time windows (e.g., 2-year intervals)
2. For each window, compute Δk_i = the number of new citations paper i received
3. Plot Δk vs k (current degree at the start of the window)
4. If the relationship is **linear** (Δk ∝ k): preferential attachment is confirmed
5. If **sublinear** (Δk ∝ k^α, α < 1): attachment is weaker than BA predicts
6. If specific papers show Δk >> expected for their k: they may be receiving artificially inflated citations

To reduce noise, we also plot the cumulative: κ(k) = ∫₀ᵏ Π(k') dk', which should be quadratic (κ ~ k²) for linear preferential attachment.

#### 1.9 Barabási-Albert Model Comparison

We generate a BA model graph with the same final N and parameter m chosen so that ⟨k⟩ matches our real network, and compare:

| Property | Citation Network | BA Model | Interpretation |
|----------|-----------------|----------|----------------|
| γ (degree exponent) | Measured | 3.0 | How close to pure BA? |
| Clustering C | Measured | (m/8)(lnN)²/N | BA has low C; real networks have high C |
| ⟨d⟩ | Measured | lnN/lnlnN (ultra-small) | BA is ultra-small world |

#### 1.10 Preferential Attachment Anomaly Detection

For each paper, we compute a **PA residual**: the difference between its actual citation growth and the growth predicted by preferential attachment. Papers with consistently large positive residuals are receiving "unearned" citations — either groundbreaking works or beneficiaries of citation manipulation. We aggregate PA residuals by community (from Deliverable 3) to identify entire communities that systematically deviate from preferential attachment — a topological signature of organized citation inflation.

### Expected Outputs
- Complete topological profile table
- Bow-tie diagram (SCC, in/out-component sizes)
- Reciprocity analysis (global and comparison to ER null)
- Log-log degree distribution with MLE power-law fit and γ estimate
- KS test results: power-law vs exponential vs log-normal
- Comprehensive comparison table: Real network vs ER vs BA model
- Small-world ratios: C/C_random and ⟨d⟩/⟨d⟩_random
- k_max: observed vs theoretical
- Temporal growth curves: N(t), L(t), ⟨k⟩(t)
- Preferential attachment plot: Δk vs k and cumulative κ(k)
- PA residual distribution; list of top anomalous papers

### Course Concepts Demonstrated
- **Week 2**: Directed graphs, in-degree/out-degree, adjacency matrix, paths, BFS, clustering coefficient, connected components (SCC/WCC)
- **Week 3**: Erdős-Rényi model, Poisson distribution, small-world property, Watts-Strogatz
- **Week 4-5**: Power-law distributions, hubs, k_max, scale-free property, 80/20 rule
- **Week 5 (Flavor Network)**: Null model methodology — comparing real data to randomized baseline
- **Week 6-7**: Barabási-Albert model, growth, preferential attachment, Π(k) measurement, configuration model, γ = 3

---

## Deliverable 2: Semantic Edge Annotation via LLM-Based Citation Intent Classification

### Motivation
A raw citation graph treats all edges as identical, but in reality citations carry very different semantic weight. A paper that cites another as "foundational background reading" has a fundamentally different relationship than one citing it as "the method we directly build upon" or "a result we dispute." By classifying citation intent, we upgrade our unweighted graph into a **semantically weighted graph**, enabling us to distinguish genuine intellectual dependency from superficial or strategic citation — which is precisely what citation cartels exploit. A cartel doesn't need deep intellectual engagement; it just needs the citation count. This deliverable gives us the tools to detect that shallowness.

### Methodology

#### 2.1 Citation Intent Taxonomy

We adopt a standard citation intent classification scheme:

| Intent Category | Description | Edge Weight | Cartel Relevance |
|----------------|-------------|-------------|------------------|
| **Background** | General context, broad area reference | 0.2 | Low — normal, expected |
| **Method** | The cited work's method is used or adapted | 1.0 | High weight — genuine dependency |
| **Result/Comparison** | Citing specific findings for comparison | 0.7 | Medium — legitimate |
| **Support** | Citing to bolster or validate a claim | 0.5 | Medium-high — can be gamed |
| **Contrast/Criticism** | Citing to disagree or present alternatives | 0.3 | Low cartel risk — adversarial |
| **Perfunctory/Ceremonial** | Superficial mention, no real engagement | 0.1 | **Highest cartel risk — padding** |

#### 2.2 LLM Classification Pipeline

For each citation edge (Paper A → Paper B), we construct a prompt containing:
- Title and abstract of Paper A (the citing paper)
- Title and abstract of Paper B (the cited paper)
- The classification taxonomy with definitions and examples

We use a prompted LLM to classify the citation intent into one of the categories above. Since we lack the full text (and therefore the exact citation sentence), the model infers intent from the thematic relationship between the two abstracts — whether the papers are thematically similar or distant, share methodology, or address competing claims.

**Scalability**: For a network with ~150,000 edges, we process in batches. At ~0.5 seconds per edge, this requires approximately 20 hours of compute time. We implement checkpointing so the pipeline can be interrupted and resumed.

#### 2.3 Weighted Graph Construction

Based on classified intent, we assign semantic weights to edges, creating a **weighted directed graph** W alongside the original unweighted graph G. This weighted graph is used in parallel with the unweighted graph in Deliverables 3 and 4, allowing us to compare: do results change when we account for citation depth?

#### 2.4 Superficiality Ratio (Cartel Signal)

For each community C detected in Deliverable 3, we compute:

$$\text{Superficiality Ratio}(C) = \frac{|\text{Perfunctory + Background edges within } C|}{|\text{All edges within } C|}$$

A legitimate research cluster should have deep methodological interdependencies (high proportion of Method and Result citations). A citation cartel will have shallow, ceremonial citations (high proportion of Perfunctory and Background). Communities with a high Superficiality Ratio AND high internal citation density are flagged as potential cartels.

#### 2.5 Weighted vs Unweighted Comparison

We run key analyses (community detection, PageRank) on both the weighted and unweighted graphs:
- Do the same communities emerge?
- Does PageRank ranking change when "cheap" citations are downweighted?
- Papers whose ranking drops significantly in the weighted graph were propped up by shallow citations

### Expected Outputs
- Distribution of citation intent categories across the entire network
- Per-community Superficiality Ratio (feeds into Deliverable 3's Cartel Score)
- Weighted adjacency matrix for parallel analysis in Deliverables 3 and 4
- Comparison: community detection and PageRank results on weighted vs unweighted graphs
- Case studies: specific paper pairs with interesting intent classifications

### Course Concepts Demonstrated
- **Week 2**: Weighted networks (extending the binary adjacency matrix to weighted A_ij ∈ [0, 1])
- This deliverable transforms the network representation, enabling semantically enriched versions of all subsequent analyses

---

## Deliverable 3: Community Detection, Modularity Analysis & Comprehensive Cartel Scoring

### Motivation
This is the central deliverable of the project. Citation cartels are, by definition, **communities** — groups of authors or papers that disproportionately cite each other to inflate citation counts. Community detection algorithms identify dense subgroups, and modularity quantifies how "real" these communities are versus what random wiring would produce. By combining community structure with metrics from Deliverables 1 and 2 (reciprocity, PA anomalies, citation intent), we construct a comprehensive, multi-dimensional **Cartel Score** that mathematically flags suspicious communities.

### Methodology

#### 3.1 Community Detection Algorithms

We apply multiple community detection methods and compare their partitions:

**Louvain Algorithm (Primary method):**
Greedy modularity optimization that iteratively moves nodes between communities to maximize modularity M:

$$M = \sum_{c=1}^{n_c} \left[ \frac{L_c}{L} - \left( \frac{d_c}{2L} \right)^2 \right]$$

where L_c = number of internal edges of community c, d_c = sum of degrees of nodes in c, and L = total edges. M ranges from -0.5 to 1, with higher values indicating stronger community structure.

The Louvain method is fast and scalable to our 30K-node network, typically completing in seconds.

**Girvan-Newman Algorithm (Secondary method for validation):**
A divisive method that iteratively removes the edge with the highest **betweenness centrality** — the edge through which the most shortest paths pass. This peels away the "bridges" between communities, causing the network to fragment into natural clusters. The process produces a **dendrogram** showing the full hierarchical community structure, from one giant cluster down to individual nodes.

Though slower than Louvain, Girvan-Newman provides richer structural insight and a complementary perspective on community boundaries.

#### 3.2 Modularity Null Model (Configuration Model)

To verify that the detected community structure is real and not an artifact of the degree distribution, we compare our measured modularity M against the modularity of **degree-preserving randomized networks** (configuration model). We:

1. Generate 100 configuration model instances (random graphs with the exact same degree sequence as our real network)
2. Run Louvain on each instance
3. Compute the distribution of M_random
4. If our real M is significantly higher than M_random (e.g., more than 3 standard deviations above the mean), the community structure is genuine

This is the network science equivalent of a statistical significance test.

#### 3.3 Community-Domain Alignment

Since our papers belong to three known domains (CS, Biology, Medicine), we check whether detected communities align with these disciplinary boundaries. We compute:

- **Normalized Mutual Information (NMI)** between the Louvain partition and the domain labels
- High NMI → communities correspond to disciplines (expected, natural)
- Low NMI → communities cut across disciplines (potentially interesting)

A community containing papers from multiple domains that heavily cite each other could be either legitimate interdisciplinary research or a cross-domain citation ring. The Cartel Score (below) disambiguates these.

#### 3.4 Comprehensive Cartel Scoring

For each detected community C, we compute a battery of cartel indicators that collectively form a multi-dimensional "cartel fingerprint":

| Metric | Definition | What It Detects |
|--------|-----------|-----------------|
| **Internal Citation Density** | (actual internal edges) / (max possible internal edges) | How tightly the community cites itself |
| **Citation Inflation Score** | (actual internal edges) / (expected internal edges under configuration model) | Whether self-citation *exceeds random expectation* given the degree sequence |
| **Reciprocity ρ(C)** | Fraction of mutual citation pairs (A↔B) within C | Quid-pro-quo mutual back-scratching |
| **Superficiality Ratio** | Fraction of internal citations classified as Perfunctory/Background by the LLM (from Deliverable 2) | Shallow, strategic citations vs genuine intellectual engagement |
| **Assortativity r(C)** | Pearson correlation between degrees of connected node pairs within C | Whether hubs selectively cite other hubs — "rich-get-richer" within a clique |
| **Rich-Club Coefficient φ(k)** | Fraction of edges among the top-k degree nodes within C, normalized against random expectation | Whether the most-cited papers form an exclusive mutual citation club |
| **PA Residual** | Average preferential attachment residual for papers in C (from Deliverable 1) | Community-level "unearned" citations beyond what popularity predicts |
| **PageRank Anomaly** | Average of (PageRank_i − k_in_i / L) for nodes in C | Papers being boosted by strategically placed citation links |

#### 3.5 Composite Cartel Index

We compute a normalized composite index by z-scoring each metric across all communities and averaging:

$$\text{Cartel Index}(C) = \frac{1}{n} \sum_{m=1}^{n} \frac{X_m(C) - \mu_m}{\sigma_m}$$

Communities with Cartel Index > 2 standard deviations above the mean are flagged as **suspected citation cartels**. We report the top-K flagged communities with detailed breakdowns of each sub-metric.

#### 3.6 Triad Census & Motif Analysis

We count all 16 directed triad types (MAN classification: Mutual, Asymmetric, Null) in our network and compare their frequencies to those expected under the configuration model null. Over-represented triads reveal structural signatures:

| Triad Type | Pattern | Cartel Significance |
|-----------|---------|-------------------|
| **003** (all null) | No connections | Baseline — no relationship |
| **012** (single edge) | A→B only | Normal one-way citation |
| **102** (mutual) | A↔B | Mutual citation — suspicious if over-represented |
| **030C** (cycle) | A→B→C→A | **Direct citation loop — strongest cartel signal** |
| **120C** (mixed cycle) | Complex with cycle | Organized mutual citation structure |
| **300** (complete) | A↔B↔C↔A | **All three cite each other — maximum cartel signal** |

We report the **Triad Significance Profile (TSP)**: a vector of z-scores for each triad type relative to the null model. Citation cartels produce a characteristic TSP with over-representation of mutual and cyclic triads.

### Expected Outputs
- Community partition with modularity M and comparison to M_random (100 configuration model samples)
- Dendrogram from Girvan-Newman showing hierarchical community structure
- Community-domain alignment: NMI score and confusion matrix
- Full cartel scoring table for all detected communities (all 8 metrics)
- Ranked list of communities by Composite Cartel Index
- Network visualization colored by community, with suspected cartels highlighted in red
- Triad Significance Profile bar chart
- Inter-community citation flow heatmap (which communities cite which, and how much)

### Course Concepts Demonstrated
- **Week 10-11**: Modularity M (definition and optimization), Louvain algorithm, Girvan-Newman (edge betweenness divisive method), hierarchical clustering, dendrogram, strong vs weak communities, configuration model as null, overlapping communities, clique percolation

---

## Deliverable 4: Network Resilience Under Random Failures, Targeted Attacks & Cartel Removal

### Motivation
Scale-free networks exhibit a striking duality: they are remarkably robust to the random failure of nodes, yet catastrophically vulnerable to targeted attacks on their highest-degree hubs. This was demonstrated in the seminal paper by Albert, Jeong, and Barabási ("Error and Attack Tolerance of Complex Networks," Nature, 2000) — covered extensively in the course lectures. We replicate this analysis on our citation network and then introduce a **novel experiment**: what happens when we specifically remove the edges identified as "cartel" citations in Deliverable 3? This experiment answers a question unique to our project: are citation cartels structurally load-bearing, or are they hollow shortcuts that can be removed without consequence?

### Methodology

#### 4.1 Random Failure Simulation

We iteratively remove nodes uniformly at random (simulating accidental paper retractions, database errors, or natural attrition) and after each removal step measure:

- **S(f)**: The relative size of the largest connected component (giant component) as a fraction of original N
- **⟨s⟩(f)**: The average size of isolated clusters (excluding the giant component)
- **⟨d⟩(f)**: The average shortest path length in the remaining network

We plot S(f) as a function of f (fraction of nodes removed), from f = 0 to f = 1. For a scale-free network, we expect S(f) to decline gradually — the network is **robust** because removing random nodes is statistically unlikely to hit the rare hubs. We average over 50 independent random removal sequences to reduce variance.

#### 4.2 Targeted Attack Simulation (Degree-Based)

We remove nodes in strict descending order of degree (highest-degree hubs first, simulating a deliberate attack on the most influential papers) and measure the same metrics. For a scale-free network, we expect a dramatically different response:

- S(f) drops sharply at a small critical threshold f_c
- The network fragments rapidly into small, disconnected clusters
- This is the "Achilles' heel" of scale-free networks — removing just 5-15% of hubs can shatter the entire structure

We identify the critical percolation threshold f_c for both random and targeted scenarios and compare.

#### 4.3 Targeted Attack Simulation (Betweenness-Based)

In addition to degree-based attacks, we simulate removal of nodes with highest **betweenness centrality** — the bridge nodes connecting different communities. These are often even more devastating targets than pure degree-hubs because they sever the pathways between communities. We compare f_c across all three attack strategies.

#### 4.4 Novel Experiment: Cartel Edge Removal

Using the cartel-flagged communities from Deliverable 3, we perform a targeted experiment: remove only the **internal edges** of suspected cartel communities (not the nodes, just the cartel-inflated citation links) and measure:

| Metric After Cartel Removal | If It Increases | If It Decreases | Interpretation |
|----------------------------|-----------------|-----------------|----------------|
| Giant component S | Network holds together | Network fragments | Cartel edges were/weren't load-bearing |
| Average path length ⟨d⟩ | Paths become longer | Paths shorten | Cartel edges were/weren't shortcuts |
| Modularity M | Community structure sharpens | Structure blurs | Cartel edges were/weren't blurring boundaries |
| Degree exponent γ | Power-law steepens (fewer hubs) | γ unchanged | Cartels were/weren't distorting the degree distribution |

This experiment directly answers: **"If we could magically remove all cartel citations from the scientific literature, what would happen to the structure of knowledge?"**

#### 4.5 Comparison Against ER and BA Null Models

We run identical failure/attack simulations on our ER and BA null model graphs (from Deliverable 1) and overlay all curves on the same plot:

| Scenario | ER Prediction | BA Prediction | Our Network |
|----------|--------------|---------------|-------------|
| Random failure f_c | ~1/⟨k⟩ (fragile) | Very high (robust) | Measured |
| Targeted attack f_c | ~1/⟨k⟩ (same as random) | Very low (fragile) | Measured |
| Response asymmetry | None (symmetric) | Massive (asymmetric) | Measured |

The BA model prediction is that random failure and targeted attack should produce vastly different results. Observing this asymmetry in our real network confirms its scale-free nature from a completely different angle than the degree distribution analysis.

### Expected Outputs
- **Primary plot**: S(f) resilience curves for random failure, degree-based attack, betweenness-based attack — all overlaid on the same figure with ER and BA baselines
- Critical thresholds f_c for each strategy (table + annotated on plot)
- **Novel plot**: S(f) curve for cartel edge removal, compared against random edge removal of the same number of edges (to control for the effect of simply removing edges)
- Analysis of how cartel removal changes modularity M, average path length ⟨d⟩, clustering C, and degree exponent γ
- Written interpretation: are cartels structurally integrated or superficial?

### Course Concepts Demonstrated
- **Week 4-5**: Albert et al. "Error and Attack Tolerance of Complex Networks" (Nature 2000), robustness of scale-free networks, targeted attacks on hubs, giant component fragmentation, percolation threshold

---

## Deliverable 5: Comprehensive Visualization, Synthesis & Final Academic Report

### Motivation
The culmination of the project transforms complex graph metrics into clear, compelling, publication-ready visualizations and a coherent narrative that ties all findings together into a unified answer to our research question: **Can we detect citation cartels using topological signatures in scale-free academic networks?** The professor evaluates not just whether we computed the metrics, but whether we can interpret them, compare them to theoretical predictions, and synthesize them into meaningful conclusions.

### Methodology

#### 5.1 Publication-Ready Visualizations

| # | Visualization | Type | Source Deliverable |
|---|--------------|------|-------------------|
| 1 | **Degree distribution P(k)** | Log-log scatter with MLE fit line + Poisson overlay | D1 |
| 2 | **Full network graph** | Force-directed layout (ForceAtlas2), nodes colored by community, suspected cartels in red | D3 |
| 3 | **Bow-tie diagram** | Schematic showing SCC, in-component, out-component, tendrils with sizes | D1 |
| 4 | **Resilience decay curves** | S(f) vs f: random failure, targeted attack, cartel removal — all overlaid with ER/BA | D4 |
| 5 | **Community cartel scatter** | x = community size, y = Composite Cartel Index, color = dominant domain, size = internal density | D3 |
| 6 | **Inter-community citation heatmap** | N_communities × N_communities matrix, color intensity = citation flow volume | D3 |
| 7 | **Preferential attachment verification** | Δk vs k scatter + linear fit + cumulative κ(k) | D1 |
| 8 | **Temporal growth curves** | N(t) and L(t) over publication years | D1 |
| 9 | **Triad significance profile** | Bar chart of z-scores for each of 16 directed triad types | D3 |
| 10 | **Citation intent distribution** | Stacked bar chart showing intent breakdown per community, sorted by Superficiality Ratio | D2 |
| 11 | **Dendrogram** | Hierarchical tree from Girvan-Newman showing nested community structure | D3 |

#### 5.2 Master Comparison Table

The single most important table in the report — showing at a glance how our real citation network compares to theoretical models:

| Property | Our Citation Network | ER Model (same N, ⟨k⟩) | BA Model (same N, m) | Interpretation |
|----------|---------------------|------------------------|---------------------|----------------|
| Degree distribution | Power-law, γ = ? | Poisson | Power-law, γ = 3 | Scale-free confirmed |
| Average path length ⟨d⟩ | ? | logN/log⟨k⟩ | lnN/lnlnN (ultra-small) | Small-world confirmed |
| Clustering coefficient C | ? | ⟨k⟩/N ≈ 0 | (m/8)(lnN)²/N | High clustering confirmed |
| Reciprocity ρ | ? | p ≈ 0 | ≈ 0 | Excess = cartel signal |
| Modularity M | ? | M_random | M_random | Community structure real |
| f_c (random failure) | ? | ~1/⟨k⟩ | Very high (robust) | Robust to failure |
| f_c (targeted attack) | ? | ~1/⟨k⟩ (same) | Very low (fragile) | Fragile to attack |

#### 5.3 Final Academic Report Structure

1. **Abstract**: One-paragraph summary of the entire project and key findings
2. **Introduction**: Citation cartels as a threat to scientific integrity; network science as a principled detection methodology; research question and hypotheses
3. **Data & Methods**: Dataset description (OpenAlex, 3 domains, 30K papers), graph construction methodology, all algorithms and metrics defined mathematically
4. **Results**:
   - Section 4.1: Topological characterization (baseline, scale-free, small-world)
   - Section 4.2: Null model comparison (ER and BA)
   - Section 4.3: Growth dynamics and preferential attachment
   - Section 4.4: Semantic edge annotation and weighted graph analysis
   - Section 4.5: Community structure and cartel detection
   - Section 4.6: Network resilience and cartel removal experiment
5. **Discussion**: Interpretation of findings — were cartels detected? Which domains are most affected? How do cartels distort the network? Limitations and future work
6. **Conclusion**: Summary of contributions to the field

### Expected Outputs
- All 11 visualizations listed above as high-resolution exportable figures
- Master comparison table
- Complete academic report (10-15 pages)
- Presentation slides (if required for oral defense)

### Course Concepts Demonstrated
- **CO3**: "Students are able to analyze and visualize networks"
- **CO4**: "Students are able to tweak and design algorithms to answer specific questions"
- **Week 13-14**: Applications of Network Science — applying all theoretical concepts to a real-world problem with societal implications

---

## Appendix: Course Coverage Map

| Course Week | Topic | Deliverable |
|-------------|-------|------------|
| Week 1 | Introduction to complex systems | Project framing |
| Week 2 | Graph theory, degree, clustering, SCC/WCC, directed graphs | **D1** (§1.2–1.4) |
| Week 3 | Erdős-Rényi, Poisson, small-world, Watts-Strogatz | **D1** (§1.6–1.7) |
| Week 4-5 | Scale-free, power-law, hubs, robustness vs attacks | **D1** (§1.5) + **D4** (all) |
| Week 5 | Flavor Network (null model methodology) | **D1** (§1.6) |
| Week 6-7 | BA model, growth, preferential attachment | **D1** (§1.8–1.10) |
| Week 10-11 | Community detection, modularity, Louvain, Girvan-Newman | **D3** (all) |
| Week 13-14 | Applications of network science | **D5** (synthesis) |
| *Bonus* | Weighted networks, NLP/LLM | **D2** (all) |

**Every single course week is explicitly addressed in at least one deliverable.**
