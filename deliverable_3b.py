import os
import json
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score
from collections import defaultdict
import random
from typing import Dict, List, Tuple
from tqdm import tqdm
from networkx.algorithms.community import girvan_newman
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.colors import ListedColormap

os.makedirs('graph_analysis_outputs', exist_ok=True)

print("Starting Deliverable 3: Community Detection & Cartel Analysis")

print("1. Loading graphs...")
G = nx.read_graphml('data/unweighted_citation_graph.graphml')

# Make sure nodes are strings
G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})

print(f"Loaded Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print("2. Enhancing with Node Metadata from raw dataset...")
# Load fos to infer domains
valid_nodes = set(G.nodes())
import ijson
node_metadata = {}
try:
    with open('sample_snowball_500k.json', 'rb') as f:
        # Since reading standard json is memory intensive for 1.6GB, we can do it iteratively
        parser = ijson.items(f, 'item')
        for record in parser:
            pid = str(record.get('id', ''))
            if pid in valid_nodes:
                fos = record.get('fos', [])
                if isinstance(fos, list) and len(fos) > 0:
                    try:
                        # Extract the top weighted fos
                        fos = sorted(fos, key=lambda x: x.get('w', 0), reverse=True)
                        domain = fos[0].get('name', 'Unknown')
                    except Exception:
                        domain = 'Unknown'
                else:
                    domain = 'Unknown'
                
                year = record.get('year', 0)
                node_metadata[pid] = {'domain': domain, 'year': year}
                
                # Check if we got all needed
                if len(node_metadata) == len(valid_nodes):
                    break
except Exception as e:
    print("Warning: ijson extraction failed or slow, using basic approach.", e)

nx.set_node_attributes(G, node_metadata)

domains = [data.get('domain', 'Unknown') for n, data in G.nodes(data=True)]
print(f"Sample domains: {domains[:5]}")

print("3. Louvain Community Detection...")
# Louvain needs undirected graph
G_undirected = G.to_undirected()
louvain_groups = nx.community.louvain_communities(G_undirected)
partition = {}
for i, com in enumerate(louvain_groups):
    for n in com:
        partition[n] = i
        
nx.set_node_attributes(G, partition, 'community')

modularity = nx.community.modularity(G_undirected, louvain_groups)
print(f"Louvain Modularity M = {modularity:.4f}")
num_communities = len(set(partition.values()))
print(f"Found {num_communities} communities.")

print("4. Modularity Null Model (Configuration Model)...")
M_randoms = []
degrees = [d for n, d in G_undirected.degree()]
for _ in tqdm(range(10), desc="Config Models"):  # Reduced to 10 for python script speed
    # Configuration model can have parallel edges and self loops
    G_null = nx.configuration_model(degrees)
    G_null = nx.Graph(G_null) # convert to simple graph
    G_null.remove_edges_from(nx.selfloop_edges(G_null))
    
    # We must ensure components matching, but for Louvain we can run directly
    try:
        part_random_groups = nx.community.louvain_communities(G_null)
        m_rand = nx.community.modularity(G_null, part_random_groups)
        M_randoms.append(m_rand)
    except:
        pass

if M_randoms:
    M_rand_mean = np.mean(M_randoms)
    M_rand_std = np.std(M_randoms)
    z_score_M = (modularity - M_rand_mean) / (M_rand_std + 1e-9)
    print(f"Random Modularity: M_random = {M_rand_mean:.4f} ± {M_rand_std:.4f}. Z = {z_score_M:.2f}")

print("5. Community-Domain Alignment (NMI)...")
# Map domain text strings to label IDs
domain_labels = {d: i for i, d in enumerate(set(domains))}
true_labels = [domain_labels[G.nodes[n].get('domain', 'Unknown')] for n in G.nodes()]
louvain_labels = [partition[n] for n in G.nodes()]
if len(set(true_labels)) > 1:
    nmi = normalized_mutual_info_score(true_labels, louvain_labels)
    print(f"Normalized Mutual Information (NMI) with Domains: {nmi:.4f}")
else:
    print("NMI could not be calculated (no variance in domains).")

print("6. Cartel Scoring...")
community_metrics = []
# Create a mapping of communities to nodes
com_to_nodes = defaultdict(list)
for n, c in partition.items():
    com_to_nodes[c].append(n)

# Global metrics for PageRank and PA
print("Computing Centralities")
pagerank = nx.pagerank(G)
# Expected preferential attachment
total_citations = G.number_of_edges()
avg_k_in = total_citations / max(1, G.number_of_nodes())

for c, nodes in tqdm(com_to_nodes.items(), desc="Scoring communities"):
    if len(nodes) < 3: # Skip very small communities
        continue
        
    subG = G.subgraph(nodes)
    
    Nc = subG.number_of_nodes()
    Ec = subG.number_of_edges()
    
    # Internal citation density
    density = nx.density(subG)
    
    # Reciprocity (fraction of mutual edges)
    # in directed graph, A->B & B->A
    reciprocal_edges = sum(1 for u, v in subG.edges() if subG.has_edge(v, u))
    reciprocity = reciprocal_edges / max(1, Ec)
    
    # Assortativity
    try:
        assortativity = nx.degree_assortativity_coefficient(subG)
        if np.isnan(assortativity): assortativity = 0
    except:
        assortativity = 0
        
    # PageRank Anomaly
    pr_anomaly = 0
    for n in nodes:
        kin = G.in_degree(n)
        expected_pr = kin / max(1, total_citations)
        pr_anomaly += (pagerank[n] - expected_pr)
    pr_anomaly /= Nc
    
    # Expected edges random (inflation)
    vol_c = sum(G_undirected.degree(n) for n in nodes)
    expected_edges = (vol_c * vol_c) / (2 * max(1, G_undirected.number_of_edges()))
    inflation = Ec / max(1, expected_edges)
    
    community_metrics.append({
        'community': c,
        'size': Nc,
        'density': density,
        'inflation': inflation,
        'reciprocity': reciprocity,
        'assortativity': assortativity,
        'pr_anomaly': pr_anomaly
    })

df_metrics = pd.DataFrame(community_metrics)

print("7. Composite Cartel Index...")
# Z-score normalization for selected features
features_to_z = ['density', 'inflation', 'reciprocity', 'assortativity', 'pr_anomaly']
for f in features_to_z:
    mean_val = df_metrics[f].mean()
    std_val = df_metrics[f].std()
    if std_val > 0:
        df_metrics[f"{f}_z"] = (df_metrics[f] - mean_val) / std_val
    else:
        df_metrics[f"{f}_z"] = 0

df_metrics['Cartel_Index'] = df_metrics[[f"{f}_z" for f in features_to_z]].mean(axis=1)

# Sort and get top cartels
df_metrics = df_metrics.sort_values(by='Cartel_Index', ascending=False)
suspected_cartel_ids = df_metrics[df_metrics['Cartel_Index'] > 2]['community'].tolist()

print("\n--- Top 5 Suspected Communities ---")
print(df_metrics.head(5)[['community', 'size', 'Cartel_Index', 'inflation', 'reciprocity']])
df_metrics.to_csv('graph_analysis_outputs/cartel_community_scores.csv', index=False)

print("8. Saving Visualizations...")

# Plot Cartel Score distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_metrics['Cartel_Index'], bins=30, kde=True)
plt.axvline(2.0, color='r', linestyle='--', label='threshold (z=2)')
plt.title("Distribution of Composite Cartel Index")
plt.xlabel("Cartel Index (Z-score composite)")
plt.legend()
plt.savefig('graph_analysis_outputs/cartel_index_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Network Visualization highlighting cartels
print("Generating network plot...")
try:
    plt.figure(figsize=(12, 12))
    # We might not plot all 3300 if it takes too long, but spring layout on 3300 takes ~20 seconds
    pos = nx.spring_layout(G_undirected, seed=42, k=0.1)
    
    node_colors = []
    for n in G.nodes():
        c = partition.get(n, -1)
        if c in suspected_cartel_ids:
            node_colors.append('red')
        else:
            node_colors.append('lightgray')
            
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    plt.axis('off')
    plt.title("Network Plot (Suspected Cartels in Red)")
    plt.savefig('graph_analysis_outputs/network_cartels.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print("Network plot failed:", e)

print("Finished Deliverable 3 Louvain Analysis!")
