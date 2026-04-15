import ijson
import json
import random
from tqdm import tqdm

INPUT_FILE = "dataset\\dblp.v12.json"
OUTPUT_FILE = "sample_10pct.json"
SAMPLE_FRAC = 0.10
SEED = 42

random.seed(SEED)

sampled = []
sampled_ids = set()

print("Sampling papers...")

with open(INPUT_FILE, "rb") as f:
    for paper in tqdm(ijson.items(f, "item"), desc="Pass 1", unit="papers"):
        if random.random() < SAMPLE_FRAC:
            pid = paper["id"]
            sampled_ids.add(pid)
            sampled.append(paper)

print(f"Sampled papers: {len(sampled)}")

print("Filtering references...")

for paper in tqdm(sampled, desc="Pass 2", unit="papers"):
    paper["references"] = [
        ref for ref in paper.get("references", [])
        if ref in sampled_ids
    ]

print("Saving dataset...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(sampled, f, default=float)

print("Done")