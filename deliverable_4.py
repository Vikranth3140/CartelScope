import os
import json
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from networkx.algorithms import community

os.makedirs('graph_analysis_outputs', exist_ok=True)

print("Starting Deliverable 4: Network Resilience Analysis")

# Load graph
G_dir = nx.read_graphml('data/unweighted_citation_graph.graphml')
# Convert node IDs to strings for consistency
G_dir = nx.relabel_nodes(G_dir, {n: str(n) for n in G_dir.nodes()})
G = G_dir.to_undirected()
N = G.number_of_nodes()
E = G.number_of_edges()

# -----------------------------
# 1. Resilience Function
# -----------------------------
def analyze_resilience(graph, strategy='random', fractions=None):
    if fractions is None:
        fractions = np.linspace(0, 0.95, 20)
    
    # Pre-compute rankings if targeted
    if strategy == 'degree':
        node_rank = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
        nodes_ordered = [n for n, d in node_rank]
    elif strategy == 'betweenness':
        print(f"Computing betweenness for {graph.number_of_nodes()} nodes...")
        betw = nx.betweenness_centrality(graph)
        node_rank = sorted(betw.items(), key=lambda x: x[1], reverse=True)
        nodes_ordered = [n for n, c in node_rank]
    else:
        nodes_ordered = list(graph.nodes())
        np.random.shuffle(nodes_ordered)
        
    s_f = []
    d_f = []
    
    total_nodes = graph.number_of_nodes()
    
    for f in tqdm(fractions, desc=f"{strategy} removal"):
        num_remove = int(f * total_nodes)
        
        # We need a fresh graph each iteration if we do it independently, 
        # or we can do it cumulatively. Culmulative is faster!
        G_temp = graph.copy()
        to_remove = nodes_ordered[:num_remove]
        G_temp.remove_nodes_from(to_remove)
        
        # S(f)
        if G_temp.number_of_nodes() == 0:
            s_f.append(0)
            d_f.append(0)
            continue
            
        largest_cc = max(nx.connected_components(G_temp), key=len)
        s_f.append(len(largest_cc) / total_nodes)
        
        # exact average path length on largest connected component
        try:
            LCC_graph = G_temp.subgraph(largest_cc)
            avg_d = nx.average_shortest_path_length(LCC_graph)
            d_f.append(avg_d)
        except:
            d_f.append(0)
            
    return fractions, s_f, d_f

# -----------------------------
# 2. Node Attacks on Main Graph
# -----------------------------
print("Analyzing Random Failures (Real Network)...")
f_rand, s_rand, d_rand = analyze_resilience(G, strategy='random')

print("Analyzing Degree-Based Attacks (Real Network)...")
f_deg, s_deg, d_deg = analyze_resilience(G, strategy='degree')

print("Analyzing Betweenness-Based Attacks (Real Network)...")
f_bet, s_bet, d_bet = analyze_resilience(G, strategy='betweenness')

plt.figure(figsize=(10, 6))
plt.plot(f_rand, s_rand, marker='o', label='Random Failure')
plt.plot(f_deg, s_deg, marker='s', label='Degree Attack')
plt.plot(f_bet, s_bet, marker='^', label='Betweenness Attack')
plt.title("Citation Network Robustness vs Targeted Attacks")
plt.xlabel("Fraction of nodes removed (f)")
plt.ylabel("Fraction of nodes in Giant Component S(f)")
plt.legend()
plt.grid(True)
plt.savefig('graph_analysis_outputs/resilience_nodes_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# 3. Null Models Validation
# -----------------------------
print("Building ER and BA Null Models...")
# ER Model
avg_k = 2 * E / N
prob_er = avg_k / (N - 1)
G_er = nx.erdos_renyi_graph(N, prob_er, seed=42)

# BA Model
# m edges to attach per new node. 
m_ba = max(1, int(round(E / N)))
G_ba = nx.barabasi_albert_graph(N, m_ba, seed=42)

print("Analyzing ER Model Resilience...")
_, s_er_rand, _ = analyze_resilience(G_er, strategy='random')
_, s_er_deg, _ = analyze_resilience(G_er, strategy='degree')

print("Analyzing BA Model Resilience...")
_, s_ba_rand, _ = analyze_resilience(G_ba, strategy='random')
_, s_ba_deg, _ = analyze_resilience(G_ba, strategy='degree')

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
axs[0].plot(f_rand, s_rand, 'b-o', label='Random')
axs[0].plot(f_deg, s_deg, 'r-s', label='Targeted')
axs[0].set_title('Real Citation Network')
axs[0].set_ylabel('S(f)')
axs[0].legend()

axs[1].plot(f_rand, s_er_rand, 'b-o', label='Random')
axs[1].plot(f_deg, s_er_deg, 'r-s', label='Targeted')
axs[1].set_title('Erdős-Rényi (ER) Null Model')
axs[1].legend()

axs[2].plot(f_rand, s_ba_rand, 'b-o', label='Random')
axs[2].plot(f_deg, s_ba_deg, 'r-s', label='Targeted')
axs[2].set_title('Barabási-Albert (BA) Null Model')
axs[2].legend()

for ax in axs:
    ax.set_xlabel('Fraction of Nodes Removed (f)')
    ax.grid(True)

plt.tight_layout()
plt.savefig('graph_analysis_outputs/resilience_null_models.png', dpi=300)
plt.show()

# -----------------------------
# 4. Cartel Edge Removal
# -----------------------------
print("Running Cartel Louvain Profiler for Edge Removal experiment...")
louvain_groups = nx.community.louvain_communities(G)

com_scores = []
for i, com in enumerate(louvain_groups):
    sub = G.subgraph(com)
    if sub.number_of_nodes() > 5:
        density = nx.density(sub)
        size = sub.number_of_nodes()
        internal_edges = sub.number_of_edges()
        com_scores.append((i, com, density * size, internal_edges))

com_scores.sort(key=lambda x: x[2], reverse=True)
top_cartels = com_scores[:3] # Pick top 3 for removal

cartel_internal_edges = []
for _, com, _, _ in top_cartels:
    sub = G.subgraph(com)
    cartel_internal_edges.extend(list(sub.edges()))

print(f"Identified {len(cartel_internal_edges)} Cartel internal edges to delete.")

def analyze_edge_resilience(graph, edges_to_remove, steps=20):
    fractions = np.linspace(0, 1.0, steps) # 0 to 100% of the target edges
    s_f = []
    d_f = []
    
    total_nodes = graph.number_of_nodes()
    
    for f in tqdm(fractions, desc="Removing Edges"):
        num_remove = int(f * len(edges_to_remove))
        
        G_temp = graph.copy()
        G_temp.remove_edges_from(edges_to_remove[:num_remove])
        
        largest_cc = max(nx.connected_components(G_temp), key=len)
        s_f.append(len(largest_cc) / total_nodes)
        
        try:
            LCC_graph = G_temp.subgraph(largest_cc)
            avg_d = nx.average_shortest_path_length(LCC_graph)
            d_f.append(avg_d)
        except:
            d_f.append(0)
            
    return fractions, s_f, d_f

print("Simulating Cartel Edge Removals...")
fc, sc_cartel, dc_cartel = analyze_edge_resilience(G, cartel_internal_edges)

print("Simulating Random Edge Removals (Control)...")
all_edges = list(G.edges())
np.random.shuffle(all_edges)
control_edges = all_edges[:len(cartel_internal_edges)]
fc, sc_rand, dc_rand = analyze_edge_resilience(G, control_edges)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(fc, sc_cartel, marker='o', label='Cartel Edges Removed')
axs[0].plot(fc, sc_rand, marker='s', label='Random Edges Removed')
axs[0].set_title('Giant Component vs Target Edge Deletion')
axs[0].set_xlabel('Fraction of target edges removed')
axs[0].set_ylabel('S(f)')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(fc, dc_cartel, marker='o', label='Cartel Edges Removed')
axs[1].plot(fc, dc_rand, marker='s', label='Random Edges Removed')
axs[1].set_title('Avg Path Length vs Target Edge Deletion')
axs[1].set_xlabel('Fraction of target edges removed')
axs[1].set_ylabel('<d>')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig('graph_analysis_outputs/cartel_edge_removal_impact.png', dpi=300)
plt.show()

print("Deliverable 4 Simulation Complete!")
