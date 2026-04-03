#!/usr/bin/env python3
"""
CiteGraphLens - T1b: Clean Metadata & Edges (Option A)
------------------------------------------------------
Performs safe cleaning on T1 outputs:
- Deduplicates metadata by paper_id
- Deduplicates edges (source, target)
- Adds stub entries for missing target papers
- Generates a detailed cleaning report

Expected input files:
  data/metadata.csv
  data/edges.csv
Outputs:
  data/metadata_extended.csv
  data/edges_cleaned.csv
  data/cleaning_report.txt
"""

import pandas as pd
import os
from datetime import datetime

def main():
    print("🧹 CiteGraphLens - Cleaning Metadata & Edges (Option A)\n" + "-"*60)

    # === Load input files ===
    meta_path = "data/metadata.csv"
    edges_path = "data/edges.csv"

    if not os.path.exists(meta_path) or not os.path.exists(edges_path):
        print("❌ Required files not found in data/. Please run T1 first.")
        return

    metadata = pd.read_csv(meta_path)
    edges = pd.read_csv(edges_path)
    print(f"📄 Loaded metadata: {len(metadata):,} records")
    print(f"🔗 Loaded edges: {len(edges):,} records")

    # === Step 1: Deduplicate metadata ===
    before_meta = len(metadata)
    metadata = metadata.drop_duplicates(subset="paper_id", keep="first")
    after_meta = len(metadata)
    print(f"✅ Deduplicated metadata: {after_meta:,} (removed {before_meta - after_meta:,})")

    # === Step 2: Deduplicate edges ===
    before_edges = len(edges)
    edges = edges.drop_duplicates(subset=["source", "target"])
    after_edges = len(edges)
    print(f"✅ Deduplicated edges: {after_edges:,} (removed {before_edges - after_edges:,})")

    # === Step 3: Identify missing nodes (external citations) ===
    meta_ids = set(metadata["paper_id"])
    missing_sources = edges.loc[~edges["source"].isin(meta_ids), "source"].unique()
    missing_targets = edges.loc[~edges["target"].isin(meta_ids), "target"].unique()

    print(f"🟡 Missing source papers: {len(missing_sources):,}")
    print(f"🟡 Missing target papers: {len(missing_targets):,}")

    # === Step 4: Create stub metadata entries for missing nodes ===
    def make_stub(paper_id):
        return {
            "paper_id": paper_id,
            "title": "(External Paper)",
            "year": None,
            "doi": "",
            "domain": "External",
            "institution": "External",
            "country": ""
        }

    stub_papers = [make_stub(pid) for pid in missing_targets]
    stubs_df = pd.DataFrame(stub_papers)

    extended_meta = pd.concat([metadata, stubs_df], ignore_index=True)
    print(f"✅ Added {len(stubs_df):,} external stub papers → total metadata: {len(extended_meta):,}")

    # === Step 5: Verify coverage ===
    missing_after = edges.loc[
        ~edges["source"].isin(extended_meta["paper_id"]) |
        ~edges["target"].isin(extended_meta["paper_id"])
    ]
    print(f"🔍 Remaining edges with missing endpoints: {len(missing_after):,}")

    # === Step 6: Save cleaned outputs ===
    os.makedirs("data", exist_ok=True)
    extended_meta.to_csv("data/metadata_extended.csv", index=False)
    edges.to_csv("data/edges_cleaned.csv", index=False)

    print("💾 Saved cleaned files:")
    print("   • data/metadata_extended.csv")
    print("   • data/edges_cleaned.csv")

    # === Step 7: Generate cleaning report ===
    report_path = "data/cleaning_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== CiteGraphLens Cleaning Report (Option A) ===\n")
        f.write(f"Run timestamp: {datetime.now()}\n\n")
        f.write(f"Metadata before: {before_meta:,}\n")
        f.write(f"Metadata after (deduped): {after_meta:,}\n")
        f.write(f"Added external stubs: {len(stubs_df):,}\n")
        f.write(f"Final metadata: {len(extended_meta):,}\n\n")
        f.write(f"Edges before: {before_edges:,}\n")
        f.write(f"Edges after (deduped): {after_edges:,}\n\n")
        f.write(f"Missing sources before: {len(missing_sources):,}\n")
        f.write(f"Missing targets before: {len(missing_targets):,}\n")
        f.write(f"Missing endpoints after: {len(missing_after):,}\n")
        f.write("------------------------------------------------------\n")
        f.write("Cleaning complete. Use metadata_extended.csv and edges_cleaned.csv for T2 onwards.\n")

    print(f"📘 Report saved → {report_path}")
    print("🎯 Cleaning complete. Proceed with T2-graph_analysis.py.\n")

if __name__ == "__main__":
    main()
