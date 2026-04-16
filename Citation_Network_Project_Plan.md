# Phase-Wise Execution Plan: Citation Network Analysis
Adapted for the 10% Kaggle Citation Network Dataset Sample.

## Phase 1: Data Ingestion & Coherent Sub-sampling
Instead of pulling from OpenAlex, this phase adapts the Kaggle data into a valid closed-world network.
* **Ingest the Dataset:** Download the Kaggle dataset and load the node metadata and edge list into your environment.
* **Topological Sampling:** Start from a highly cited 'seed' paper and use a breadth-first search (Snowball Sampling) to gather neighbors until you reach your target size of ~10% of the original dataset.
* **Closed-World Enforcement:** Filter the edge list to only retain citation edges where both the source and target papers exist within your curated 10% dataset.
* **Sanity Check:** Ensure this closed-world graph allows every node to have incoming and outgoing edges, which is necessary to compute reciprocity, cycles, and community structure.

## Phase 2: Topological Profiling & Baselines (Deliverable 1)
Establishing what the 'normal' structure of your sub-sampled citation network looks like.
* **Baseline Metrics:** Compute the global clustering coefficient, network density, diameter, and average path length.
* **Structural Decomposition:** Decompose the graph into its fundamental bow-tie structure, identifying the Strongly Connected Component (SCC), Weakly Connected Component (WCC), In-Component, and Out-Component.
* **Scale-Free Verification:** Plot the degree distributions on a $log-log$ scale and use Maximum Likelihood Estimation (MLE) to estimate the degree exponent y.
* **Null Models:** Generate an Erdős-Rényi random graph and a Barabási-Albert model with the same parameters as your real network to serve as baseline comparisons.

## Phase 3: Semantic Edge Annotation (Deliverable 2)
Upgrading the unweighted graph into a semantically weighted graph to distinguish genuine dependency from superficial citations.
* **Prompt Engineering:** Build an LLM prompt that contains the classification taxonomy along with the titles and abstracts of citing and cited papers.
* **Batch Processing:** Run the citation edges through the LLM to classify intents into categories such as Method, Result, Background, or Perfunctory.
* **Weighted Graph Generation:** Assign semantic weights based on intent to create a weighted directed graph for use in later phases.

## Phase 4: Community Detection & Cartel Scoring (Deliverable 3)
The core analysis phase where suspicious groups are mathematically flagged.
* **Community Detection:** Apply the Louvain Algorithm to optimize modularity and identify dense subgroups within the network.
* **Cartel Indexing:** For each detected community, compute the comprehensive Cartel Score using metrics like Internal Citation Density, Reciprocity, and the Superficiality Ratio.
* **Triad Census:** Count all 16 directed triad types and compare them to a configuration model to detect over-represented citation loops.

## Phase 5: Resilience Testing & Simulations (Deliverable 4)
Testing whether the identified cartels are load-bearing structures or hollow shortcuts.
* **Standard Resilience Tests:** Iteratively remove nodes randomly, and then specifically target high-degree hubs, measuring the size of the giant component at each step.
* **Cartel Edge Removal:** Conduct the novel experiment by removing only the internal edges of suspected cartel communities.
* **Impact Measurement:** Observe how this removal alters network metrics like modularity, average path length, and the degree exponent.

## Phase 6: Synthesis, Visualization & Reporting (Deliverable 5)
Transforming the graph metrics into a coherent academic narrative.
* **Visual Generation:** Create the 11 required visualizations, including the force-directed full network graph, resilience decay curves, and the triad significance profile.
* **Comparison Matrix:** Compile the master comparison table to show how the real citation network stacks up against the theoretical ER and BA models.
* **Final Documentation:** Draft the final academic report detailing the methodology, results, and whether the topological signatures successfully detected citation cartels.
