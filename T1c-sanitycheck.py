#!/usr/bin/env python3
"""
CiteGraphLens - T1 Sanity Check Script
--------------------------------------
Validates the integrity and consistency of:
- edges.csv
- metadata.csv
- openalex_raw.jsonl

Checks:
  ✅ Schema validity
  ✅ Duplicate or missing values
  ✅ Edge consistency (source/target in metadata)
  ✅ Cross-file integrity
  ✅ Summary statistics for papers, citations, domains
"""

import pandas as pd
import json
from tqdm import tqdm
import os

def check_files_exist():
    required = ["data/edges.csv", "data/metadata.csv", "data/openalex_raw.jsonl"]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing required file(s): {', '.join(missing)}")
        return False
    print("✅ All required files found.\n")
    return True

def check_metadata(metadata):
    print("🔍 Checking metadata.csv ...")
    expected_cols = {"paper_id", "title", "year", "doi", "domain", "institution", "country"}
    missing_cols = expected_cols - set(metadata.columns)
    if missing_cols:
        print(f"❌ Missing columns in metadata.csv: {missing_cols}")
    else:
        print("✅ All expected columns present.")
    
    duplicates = metadata['paper_id'].duplicated().sum()
    if duplicates:
        print(f"⚠️ Found {duplicates} duplicate paper_id entries in metadata.")
    else:
        print("✅ No duplicate paper_id entries found.")
    
    missing_titles = metadata['title'].isna().sum()
    missing_years = metadata['year'].isna().sum()
    print(f"ℹ️ Missing titles: {missing_titles}, Missing years: {missing_years}")
    
    domain_counts = metadata['domain'].value_counts(dropna=False)
    print("\n📊 Domain distribution:")
    print(domain_counts)
    
    return metadata

def check_edges(edges, metadata):
    print("\n🔍 Checking edges.csv ...")
    expected_cols = {"source", "target"}
    if not expected_cols.issubset(edges.columns):
        print(f"❌ Missing required columns in edges.csv. Found: {edges.columns.tolist()}")
        return
    
    print(f"✅ Edges file has {len(edges):,} total citations.")
    
    duplicates = edges.duplicated().sum()
    if duplicates:
        print(f"⚠️ Found {duplicates} duplicate edges.")
    else:
        print("✅ No duplicate edges.")
    
    # Check if all sources and targets exist in metadata
    all_ids = set(metadata['paper_id'])
    missing_sources = edges.loc[~edges['source'].isin(all_ids)]
    missing_targets = edges.loc[~edges['target'].isin(all_ids)]
    
    print(f"🔹 Missing sources in metadata: {len(missing_sources)}")
    print(f"🔹 Missing targets in metadata: {len(missing_targets)}")
    
    if len(missing_targets) > 0:
        pct_missing = len(missing_targets) / len(edges) * 100
        print(f"⚠️ {pct_missing:.2f}% of target papers not in metadata.")
        print("   Consider running T1b-clean_metadata_edges.py (Option A) to keep external references.")
    else:
        print("✅ All targets found in metadata.")
    
    # Self-citations check
    self_cites = (edges['source'] == edges['target']).sum()
    if self_cites > 0:
        print(f"⚠️ Found {self_cites} self-citation edges.")
    else:
        print("✅ No self-citations detected.")
    
    return edges

def check_openalex_raw(filepath="data/openalex_raw.jsonl"):
    print("\n🔍 Checking openalex_raw.jsonl ...")
    count = 0
    sample = None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            try:
                rec = json.loads(line)
                count += 1
                if sample is None:
                    sample = rec
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON line detected at line {count+1}")
                continue
    
    print(f"✅ {count:,} records successfully parsed from openalex_raw.jsonl")
    print("📘 Sample record keys:", list(sample.keys()) if sample else "N/A")

def summarize_graph(edges, metadata):
    print("\n📈 Graph Summary")
    nodes = len(metadata)
    edges_count = len(edges)
    print(f"🧩 Nodes (papers): {nodes:,}")
    print(f"🔗 Edges (citations): {edges_count:,}")
    print(f"📚 Average out-degree: {edges_count / nodes:.2f}")
    
    # Top 5 most cited (in-degree)
    top_cited = edges['target'].value_counts().head(5)
    print("\n🏅 Top 5 most cited papers:")
    for pid, count in top_cited.items():
        title = metadata.loc[metadata['paper_id'] == pid, 'title'].values
        title = title[0][:80] + "..." if len(title) else "(Unknown)"
        print(f"  • {pid} — {count} citations — {title}")
    
    # Top 5 most citing (out-degree)
    top_citing = edges['source'].value_counts().head(5)
    print("\n📖 Top 5 most citing papers:")
    for pid, count in top_citing.items():
        title = metadata.loc[metadata['paper_id'] == pid, 'title'].values
        title = title[0][:80] + "..." if len(title) else "(Unknown)"
        print(f"  • {pid} — cites {count} papers — {title}")

def main():
    print("🔬 CiteGraphLens T1 Sanity Check\n" + "-" * 50)
    
    if not check_files_exist():
        return
    
    # Load files
    metadata = pd.read_csv("data/metadata.csv")
    edges = pd.read_csv("data/edges.csv")
    
    # Perform checks
    metadata = check_metadata(metadata)
    edges = check_edges(edges, metadata)
    check_openalex_raw("data/openalex_raw.jsonl")
    summarize_graph(edges, metadata)

    print("\n✅ Sanity check complete. If all sections show green ticks, T1 output is consistent.\n")

if __name__ == "__main__":
    main()
