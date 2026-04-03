import pandas as pd

# Load files
metrics = pd.read_csv("output/graph_metrics.csv")
meta = pd.read_csv("data/metadata.csv")

# Find nodes in community 1689
community_nodes = metrics[metrics["community"] == 1689]["node_id"].tolist()

# Get their metadata
community_meta = meta[meta["paper_id"].isin(community_nodes)][["paper_id", "title", "institution", "domain"]]

print(community_meta)
