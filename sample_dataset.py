#!/usr/bin/env python3
"""
Phase 1: Topological Snowball Sampling for Citation Network Analysis
====================================================================
,
Creates a dense, scale-free citation subgraph of ~500K papers from the
full DBLP v12 dataset using multi-seed BFS-based snowball sampling.

WHY SNOWBALL SAMPLING instead of random sampling?
──────────────────────────────────────────────────
Random 10% sampling destroyed the citation network:
  → 43% isolates, edge/node ratio < 1, SCC of only 8 nodes
  → Power-law shape was mangled because hub neighborhoods were scattered

Snowball sampling from highly-cited hubs preserves topology:
  → Starts from the densest region of the network and expands outward
  → Connected papers stay connected (BFS guarantees adjacency)
  → BOTH in-degree and out-degree retain scale-free structure:
       FORWARD  expansion (A→refs)   preserves OUT-degree
       BACKWARD expansion (A←citers) preserves IN-degree
  → Minimal isolates, dense connected component spanning most of the graph

ALGORITHM
─────────
  PASS 1 ─ Stream 12.5 GB file → build lightweight adjacency index
            (paper_id → references, paper_id → citers, paper_id → n_citation)
  SEEDS  ─ Select the top-N most cited papers as BFS starting points
  BFS    ─ Bidirectional multi-seed BFS until oversample target reached
  PASS 2 ─ Stream file again → extract full records for sampled IDs
  CLEAN  ─ Closed-world enforcement → iterative isolate removal → trim
  VERIFY ─ Report degree stats confirming scale-free & density
"""

import json
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import ijson
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────

INPUT_FILE = Path("dataset") / "dblp.v12.json"
OUTPUT_FILE = Path("sample_snowball_500k.json")

TARGET_SIZE = 500_000          # desired papers in the final output
OVERSAMPLE_FACTOR = 1.05       # BFS collects 5% extra to absorb isolate removal
NUM_SEEDS = 200                # highly-cited seed papers for multi-seed BFS
RNG_SEED = 42                  # reproducibility

random.seed(RNG_SEED)


# ═══════════════════════════════════════════════════════════════════════════════
#  PASS 1 — Build Lightweight Graph Index
# ═══════════════════════════════════════════════════════════════════════════════

def pass1_build_index(input_path: Path):
    """
    Stream the entire DBLP JSON once and extract only the information
    needed for snowball sampling:

        fwd[pid] = [ref1, ref2, ...]   outgoing references
        rev[ref] = [citer1, citer2, ...] incoming citations
        ncite[pid] = int               citation count (metadata field)
        all_ids = {pid, ...}           every paper ID present as a record

    Memory estimate for ~5 M papers / ~40 M edges:
        forward adj  ~ 1.0-1.5 GB (dict of int lists)
        reverse adj  ~ 1.0-1.5 GB
        all_ids      ~ 0.2 GB
    Total ≈ 2-3 GB, well within reach of a 16 GB machine.
    """
    print()
    print("─" * 62)
    print("  PASS 1 │ Building graph index from full DBLP v12 dataset")
    print("─" * 62)

    fwd: dict[int, list[int]] = {}
    ncite: dict[int, int] = {}
    all_ids: set[int] = set()

    t0 = time.time()
    paper_count = 0
    raw_edge_count = 0

    with open(input_path, "rb") as f:
        for paper in tqdm(ijson.items(f, "item"),
                          desc="  Indexing", unit=" papers"):
            pid = paper.get("id")
            if pid is None:
                continue

            all_ids.add(pid)

            refs = paper.get("references")
            if refs:
                fwd[pid] = refs
                raw_edge_count += len(refs)

            nc = paper.get("n_citation", 0)
            if isinstance(nc, (int, float)) and nc > 0:
                ncite[pid] = int(nc)

            paper_count += 1

    elapsed = time.time() - t0
    print(f"\n  Indexed {paper_count:,} papers in {elapsed:.0f}s "
          f"({elapsed / 60:.1f} min)")
    print(f"  Raw edges in file:       {raw_edge_count:,}")
    print(f"  Papers with references:  {len(fwd):,}")
    print(f"  Papers with citation ct: {len(ncite):,}")

    # ── reverse adjacency ─────────────────────────────────────────────────
    print("\n  Building reverse adjacency (cited → citers) ...")
    t1 = time.time()
    rev: dict[int, list[int]] = defaultdict(list)
    valid_edge_count = 0

    for src, refs in tqdm(fwd.items(), desc="  Rev-index", unit=" src"):
        for ref in refs:
            if ref in all_ids:                # edge target must exist as record
                rev[ref].append(src)
                valid_edge_count += 1

    rev = dict(rev)                           # save ~20% memory vs defaultdict
    elapsed_r = time.time() - t1
    print(f"  Reverse index built in {elapsed_r:.0f}s")
    print(f"  Valid internal edges: {valid_edge_count:,}")
    print(f"  Papers cited by ≥1 other paper: {len(rev):,}")

    return fwd, rev, ncite, all_ids


# ═══════════════════════════════════════════════════════════════════════════════
#  SEED SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def select_seeds(ncite, all_ids, num_seeds=NUM_SEEDS):
    """
    Pick the most-cited papers as BFS origins.

    Starting from high-citation hubs guarantees that the first BFS levels
    capture the dense "core" of the citation graph and that the resulting
    subgraph inherits its hub-dominated, scale-free topology.
    """
    print(f"\n  Selecting top-{num_seeds} most-cited papers as BFS seeds …")

    ranked = sorted(
        ((pid, c) for pid, c in ncite.items() if pid in all_ids),
        key=lambda x: x[1],
        reverse=True,
    )

    seeds = [pid for pid, _ in ranked[:num_seeds]]

    print("  Top-10 seeds:")
    for i, (pid, c) in enumerate(ranked[:10], 1):
        print(f"    [{i:>3d}] paper {pid}  —  {c:,} citations")
    if len(ranked) >= num_seeds:
        print(f"    …")
        pid_last, c_last = ranked[num_seeds - 1]
        print(f"    [{num_seeds}] paper {pid_last}  —  {c_last:,} citations")

    return seeds


# ═══════════════════════════════════════════════════════════════════════════════
#  SNOWBALL BFS  —  Bidirectional, Multi-Seed
# ═══════════════════════════════════════════════════════════════════════════════

def snowball_bfs(seeds, fwd, rev, all_ids, target):
    """
    Level-synchronous BFS expanding in **both** directions from every
    seed simultaneously.

    Why bidirectional?
    ──────────────────
    • FORWARD  (paper → its references):
        Captures the papers that the hub depends on.  Because we include
        the hub *and* its references, the hub's out-degree is preserved
        in the subgraph.

    • BACKWARD (paper ← its citers):
        Captures the papers that depend on the hub.  Because we include
        the hub *and* its citers, the hub's in-degree is preserved in
        the subgraph.

    Combined, the sampling honours the *rich-get-richer* attachment
    process that generated the original power law:
        few papers with very high in-degree  (citation magnets)
        few papers with very high out-degree  (survey / review papers)
        most papers with moderate degrees     (rank-and-file publications)
    """
    print()
    print("─" * 62)
    print(f"  SNOWBALL BFS │ target {target:,} papers, "
          f"{len(seeds)} seeds")
    print("─" * 62)

    sampled: set[int] = set()
    queue: deque[int] = deque()

    for s in seeds:
        if s in all_ids:
            sampled.add(s)
            queue.append(s)

    print(f"  Seeded with {len(sampled)} papers")

    t0 = time.time()
    level = 0
    pbar = tqdm(total=target, initial=len(sampled),
                desc="  BFS progress", unit=" papers")

    while queue and len(sampled) < target:
        level += 1
        level_size = len(queue)
        added = 0
        next_q: deque[int] = deque()

        for _ in range(level_size):
            if len(sampled) >= target:
                break

            node = queue.popleft()

            # Collect all unseen neighbours in both directions
            neighbours: list[int] = []
            neighbours.extend(fwd.get(node, []))     # outgoing
            neighbours.extend(rev.get(node, []))      # incoming

            # Shuffle so neither direction is systematically favoured
            # at the frontier cutoff
            random.shuffle(neighbours)

            for nbr in neighbours:
                if nbr not in sampled and nbr in all_ids:
                    sampled.add(nbr)
                    next_q.append(nbr)
                    added += 1
                    pbar.update(1)
                    if len(sampled) >= target:
                        break

        queue = next_q
        elapsed = time.time() - t0
        tqdm.write(
            f"    Level {level:>2d}:  +{added:>9,} papers │ "
            f"total {len(sampled):>9,} │ "
            f"queue {len(queue):>9,} │ "
            f"{elapsed:>6.1f}s"
        )

    pbar.close()

    elapsed = time.time() - t0
    print(f"\n  BFS complete: {len(sampled):,} papers in {level} levels, "
          f"{elapsed:.0f}s ({elapsed / 60:.1f} min)")
    return sampled


# ═══════════════════════════════════════════════════════════════════════════════
#  PASS 2 — Extract Full Paper Records
# ═══════════════════════════════════════════════════════════════════════════════

def pass2_extract_records(input_path: Path, sampled_ids: set):
    """
    Stream the dataset a second time.  For every paper whose ID is in
    `sampled_ids`, keep its full record **but** truncate `references`
    to those that also appear in the sample (closed-world enforcement).
    """
    print()
    print("─" * 62)
    print(f"  PASS 2 │ Extracting {len(sampled_ids):,} paper records")
    print("─" * 62)

    records: list[dict] = []
    found = 0
    t0 = time.time()

    with open(input_path, "rb") as f:
        for paper in tqdm(ijson.items(f, "item"),
                          desc="  Extracting", unit=" papers"):
            pid = paper.get("id")
            if pid in sampled_ids:
                # ── closed-world: keep only intra-sample references ──
                raw_refs = paper.get("references", [])
                paper["references"] = [r for r in raw_refs if r in sampled_ids]
                records.append(paper)
                found += 1

    elapsed = time.time() - t0
    print(f"\n  Extracted {found:,} full records in {elapsed:.0f}s "
          f"({elapsed / 60:.1f} min)")

    if found < len(sampled_ids):
        missing = len(sampled_ids) - found
        print(f"  ⚠  {missing:,} IDs had no full record "
              f"(exist only as dangling reference targets)")

    return records


# ═══════════════════════════════════════════════════════════════════════════════
#  POST-PROCESSING — Isolate Removal & Trim
# ═══════════════════════════════════════════════════════════════════════════════

def remove_isolates(records):
    """
    Iteratively drop nodes with zero in-degree AND zero out-degree in
    the closed-world graph, then re-enforce closed-world constraints.

    With snowball sampling, isolate counts are already very low (each
    BFS-added node had ≥ 1 edge to an already-sampled node), but a
    small number can appear when reference-only IDs had no record.
    """
    print("\n  Removing isolates …")
    original = len(records)

    for it in range(5):          # converges quickly
        current_ids = {p["id"] for p in records}

        # re-enforce closed world after any removals
        for p in records:
            p["references"] = [r for r in p.get("references", [])
                               if r in current_ids]

        # find nodes that participate in at least one edge
        connected: set[int] = set()
        for p in records:
            pid = p["id"]
            refs = p.get("references", [])
            if refs:
                connected.add(pid)
                connected.update(refs)

        before = len(records)
        records = [p for p in records if p["id"] in connected]
        removed = before - len(records)

        print(f"    iter {it + 1}: −{removed:,} isolates "
              f"→ {len(records):,} papers")
        if removed == 0:
            break

    print(f"  Net removed: {original - len(records):,}")
    return records


def trim_to_target(records, target):
    """
    If oversampling left us above target, drop the least-connected
    papers (by total degree in the closed-world graph).

    This selectively prunes the *frontier* of the BFS tree, where
    papers have the weakest connection to the core, so it has minimal
    impact on the degree distribution's power-law shape.
    """
    if len(records) <= target:
        return records

    print(f"\n  Trimming {len(records):,} → {target:,} papers "
          f"(removing least-connected frontier nodes) …")

    current_ids = {p["id"] for p in records}

    # compute total degree = out-degree + in-degree
    in_deg: dict[int, int] = defaultdict(int)
    out_deg: dict[int, int] = {}
    for p in records:
        pid = p["id"]
        refs = [r for r in p.get("references", []) if r in current_ids]
        out_deg[pid] = len(refs)
        for r in refs:
            in_deg[r] += 1

    total_deg = {p["id"]: out_deg.get(p["id"], 0) + in_deg.get(p["id"], 0)
                 for p in records}

    # keep the highest-degree papers
    records.sort(key=lambda p: total_deg.get(p["id"], 0), reverse=True)
    records = records[:target]

    # re-enforce closed world
    keep_ids = {p["id"] for p in records}
    for p in records:
        p["references"] = [r for r in p.get("references", [])
                           if r in keep_ids]

    # one more isolate pass after trimming
    records = remove_isolates(records)
    print(f"  Final count after trim + clean: {len(records):,}")
    return records


# ═══════════════════════════════════════════════════════════════════════════════
#  VERIFICATION REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def verify_and_report(records):
    """
    Print a dashboard confirming that the dataset meets Phase 1 criteria:

    ✓ ~500 K papers
    ✓ Dense  (edge/node ratio ≫ 1, negligible isolates)
    ✓ In-degree  follows power-law (few mega-hubs, long tail)
    ✓ Out-degree follows power-law (few surveys, long tail)
    ✓ Large connected component
    """
    print()
    print("━" * 62)
    print("  DATASET VERIFICATION REPORT")
    print("━" * 62)

    n = len(records)

    out_deg: dict[int, int] = {}
    in_deg: dict[int, int] = defaultdict(int)
    for p in records:
        pid = p["id"]
        refs = p.get("references", [])
        out_deg[pid] = len(refs)
        for r in refs:
            in_deg[r] += 1

    total_edges = sum(out_deg.values())
    out_vals = sorted(out_deg.values())
    in_vals = sorted([in_deg.get(p["id"], 0) for p in records])

    isolates = sum(1 for p in records
                   if out_deg[p["id"]] == 0
                   and in_deg.get(p["id"], 0) == 0)
    sinks = sum(1 for p in records
                if out_deg[p["id"]] == 0
                and in_deg.get(p["id"], 0) > 0)
    sources = sum(1 for p in records
                  if out_deg[p["id"]] > 0
                  and in_deg.get(p["id"], 0) == 0)
    both = n - isolates - sinks - sources

    print(f"""
  SIZE & DENSITY
  ──────────────
    Papers  (nodes):   {n:>10,}
    Citations (edges): {total_edges:>10,}
    Edge / node ratio: {total_edges / n:>10.2f}
    Network density:   {total_edges / (n * (n - 1)):>14.2e}

  DEGREE STATISTICS
  ─────────────────
    OUT-DEGREE  (references made)
      mean {sum(out_vals) / n:>8.2f}   median {out_vals[n // 2]:>5d}   max {out_vals[-1]:>6d}
    IN-DEGREE   (citations received)
      mean {sum(in_vals) / n:>8.2f}   median {in_vals[n // 2]:>5d}   max {in_vals[-1]:>6d}

  NODE CONNECTIVITY
  ─────────────────
    Both in & out edges: {both:>9,}  ({100 * both / n:>5.1f}%)
    Sink  (in only):     {sinks:>9,}  ({100 * sinks / n:>5.1f}%)
    Source (out only):   {sources:>9,}  ({100 * sources / n:>5.1f}%)
    Isolate (none):      {isolates:>9,}  ({100 * isolates / n:>5.1f}%)
""")

    # ── decade-binned degree histogram (quick visual sanity check) ──
    print("  SCALE-FREE PREVIEW  (decade bins, papers with degree ≥ 1)")
    print("  ─────────────────")
    for label, vals in [("IN ", in_vals), ("OUT", out_vals)]:
        pos = [v for v in vals if v > 0]
        if not pos:
            continue
        max_d = max(pos)
        lo = 1
        while lo <= max_d:
            hi = lo * 10 - 1
            cnt = sum(1 for v in pos if lo <= v <= hi)
            if cnt > 0:
                bar_len = max(1, int(40 * cnt / len(pos)))
                bar = "█" * bar_len
                print(f"    {label} [{lo:>6,}–{min(hi, max_d):>6,}]  "
                      f"{cnt:>8,}  {bar}")
            lo *= 10
    print()

    print("━" * 62)
    return {
        "n_papers": n,
        "n_edges": total_edges,
        "edge_per_node": total_edges / n,
        "isolates": isolates,
        "pct_both": 100 * both / n,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════════════════

def save_dataset(records, output_path: Path):
    """Serialize the final curated dataset to JSON."""
    print(f"  Saving {len(records):,} papers → {output_path} …")
    t0 = time.time()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, default=float)
    elapsed = time.time() - t0
    size_gb = output_path.stat().st_size / (1 << 30)
    print(f"  Done in {elapsed:.0f}s  ({size_gb:.2f} GB)")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Citation Network — Phase 1                              ║")
    print("║  Topological Snowball Sampling for Dense Scale-Free Graph ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"  Input:   {INPUT_FILE}")
    print(f"  Output:  {OUTPUT_FILE}")
    print(f"  Target:  ~{TARGET_SIZE:,} papers")
    print(f"  Seeds:   {NUM_SEEDS} most-cited papers")
    print(f"  Oversample: {OVERSAMPLE_FACTOR:.0%}")

    t_global = time.time()

    # ── 1. Build adjacency index ──────────────────────────────────────────
    fwd, rev, ncite, all_ids = pass1_build_index(INPUT_FILE)

    # ── 2. Pick seeds ─────────────────────────────────────────────────────
    seeds = select_seeds(ncite, all_ids)

    # ── 3. Snowball BFS ───────────────────────────────────────────────────
    bfs_target = int(TARGET_SIZE * OVERSAMPLE_FACTOR)
    print(f"\n  BFS oversample target: {bfs_target:,}  "
          f"(+{(OVERSAMPLE_FACTOR - 1) * 100:.0f}% headroom for cleanup)")

    sampled_ids = snowball_bfs(seeds, fwd, rev, all_ids, bfs_target)

    # free heavy structures before second file scan
    del fwd, rev, ncite
    import gc; gc.collect()

    # ── 4. Extract full records (second file scan) ────────────────────────
    records = pass2_extract_records(INPUT_FILE, sampled_ids)
    del sampled_ids
    gc.collect()

    # ── 5. Closed-world cleanup ───────────────────────────────────────────
    records = remove_isolates(records)

    # ── 6. Trim to target if still over ───────────────────────────────────
    records = trim_to_target(records, TARGET_SIZE)

    # ── 7. Verification ──────────────────────────────────────────────────
    stats = verify_and_report(records)

    # ── 8. Save ──────────────────────────────────────────────────────────
    save_dataset(records, OUTPUT_FILE)

    total = time.time() - t_global
    print(f"\n  Total pipeline: {total:.0f}s ({total / 60:.1f} min)")

    # ── Summary verdict ──────────────────────────────────────────────────
    ratio = stats["edge_per_node"]
    iso = stats["isolates"]
    pct = stats["pct_both"]
    ok = ratio > 2.0 and iso == 0 and pct > 50.0
    verdict = "✓ PASS" if ok else "⚠ REVIEW"
    print(f"\n  {verdict}  │  {stats['n_papers']:,} papers, "
          f"{stats['n_edges']:,} edges, "
          f"edge/node={ratio:.1f}, "
          f"isolates={iso}, "
          f"{pct:.1f}% bidirectional")
    print(f"\n  Dataset ready for Phase 2: Topological Profiling & Baselines")


if __name__ == "__main__":
    main()