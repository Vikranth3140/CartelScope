import json

with open("sample_10pct.json", "r", encoding="utf-8") as f:
    data = json.load(f)

n_nodes = len(data)

n_edges = sum(len(p.get("references", [])) for p in data)

isolated = sum(1 for p in data if len(p.get("references", [])) == 0)

print("Nodes:", n_nodes)
print("Edges:", n_edges)
print("Isolated nodes:", isolated)
print("Average out-degree:", n_edges / n_nodes if n_nodes else 0)