import pandas as pd
import matplotlib.pyplot as plt

# Load the graph metrics file
df = pd.read_csv("output/graph_metrics.csv")

# Ensure domain column exists (if not, merge from metadata.csv)
if "domain" not in df.columns:
    meta = pd.read_csv("data/metadata.csv")[["paper_id", "domain"]]
    # Rename node_id to paper_id for the merge
    df = df.rename(columns={"node_id": "paper_id"})
    df = df.merge(meta, on="paper_id", how="left")

domains = ["Computer Science", "Biology", "Medicine"]
metrics = {
    "Reciprocity": "recip",
    "Clustering Coefficient": "cluster",
    "Self-Citation Ratio": "self_ratio"
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (title, col) in enumerate(metrics.items()):
    ax = axes[i]
    for d in domains:
        subset = df[df["domain"] == d][col].dropna()
        ax.hist(subset, bins=30, alpha=0.6, label=d)
    ax.set_title(title)
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

plt.suptitle("Domain-wise Citation Metrics: Reciprocity, Clustering, and Self-Citation", fontsize=14)
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig("output/domainwise_metrics.png", dpi=300)
plt.show()
