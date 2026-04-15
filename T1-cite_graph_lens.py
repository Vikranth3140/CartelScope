#!/usr/bin/env python3
# CiteGraphLens: Data Acquisition & Preprocessing
# Script to fetch, process, and analyze citation data from OpenAlex API

import requests
import pandas as pd
import jsonlines
import os
import argparse
from rapidfuzz import fuzz, process
from tqdm import tqdm
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("citegraphlens.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def reconstruct_abstract(inverted_index):
    """
    Reconstructs the abstract from OpenAlex's inverted index format into a raw text string.
    Note: The output will lack punctuation, which is expected and acceptable for LLM zero-shot classification.
    """
    if not inverted_index:
        return ""
    
    try:
        # Find the maximum index to size our array
        max_idx = max([max(positions) for positions in inverted_index.values()])
        words = [""] * (max_idx + 1)
        
        # Place each word at its specified indices
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
                
        return " ".join(words).strip()
    except Exception as e:
        logger.warning(f"Failed to reconstruct abstract: {e}")
        return ""

def fetch_openalex(concept_id, max_results=100000, email="user@example.com", start_year=2018, end_year=2022):
    """
    Query the OpenAlex API for papers in a specific concept (e.g., Artificial Intelligence)
    within a strict time window to ensure network density.
    """
    base_url = "https://api.openalex.org/works"
    
    # Enforce time window for density
    year_filter = f",publication_year:{start_year}-{end_year}"

    params = {
        'filter': f"concepts.id:{concept_id}{year_filter}",
        'per_page': 200,
        # CRITICAL UPDATE: Added abstract_inverted_index to the select parameter
        'select': "id,doi,title,publication_year,authorships,referenced_works,abstract_inverted_index",
        'sort': "cited_by_count:desc", # Sort by highly cited to anchor the core of the graph
        'mailto': email
    }
    
    headers = {'User-Agent': f'CiteGraphLens/1.0 ({email})'}
    all_papers = []
    params['cursor'] = '*'
    
    with tqdm(total=max_results, desc="Fetching AI papers") as pbar:
        while len(all_papers) < max_results:
            try:
                response = requests.get(base_url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                papers = data.get('results', [])
                if not papers:
                    logger.warning("No papers returned in response, breaking pagination loop")
                    break
                
                meta = data.get('meta', {})
                next_cursor = meta.get('next_cursor')
                
                all_papers.extend(papers)
                pbar.update(len(papers))
                
                time.sleep(0.3) # Respect API rate limits
                
                if len(all_papers) >= max_results or not next_cursor:
                    break
                
                params['cursor'] = next_cursor
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit hit. Waiting for {retry_after} seconds.")
                    time.sleep(retry_after)
                    continue
                else:
                    logger.error(f"HTTP error: {e}")
                    time.sleep(2)
                    continue
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {e}")
                time.sleep(2)
                continue
    
    logger.info(f"Successfully collected {len(all_papers)} papers.")
    # Trim to exactly max_results if we overshot due to pagination chunks
    return all_papers[:max_results]


def fetch_openalex_by_ids(work_ids, email="user@example.com", batch_size=50):
    """Fetch OpenAlex works by a list of work IDs to flesh out referenced targets."""
    base_url = "https://api.openalex.org/works"
    headers = {'User-Agent': f'CiteGraphLens/1.0 ({email})'}
    fetched = []
    
    for i in tqdm(range(0, len(work_ids), batch_size), desc="Fetching references"):
        batch = work_ids[i:i+batch_size]
        ids_filter = ','.join([f"https://openalex.org/{w}" if not w.startswith('https://') else w for w in batch])
        params = {
            'filter': f"id:{ids_filter}",
            'per_page': 200,
            'mailto': email,
            'select': "id,doi,title,publication_year,authorships,referenced_works,abstract_inverted_index"
        }

        try:
            resp = requests.get(base_url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            fetched.extend(data.get('results', []))
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                time.sleep(int(e.response.headers.get('Retry-After', 60)))
            continue
        except requests.exceptions.RequestException:
            continue
        time.sleep(0.3)

    return fetched

def clean_metadata(raw_data):
    """Process raw data, rebuild abstracts, and enforce closed-world graph."""
    papers = []
    logger.info("Processing and cleaning paper metadata...")
    
    # 1. First pass: Build the metadata and establish the closed world
    for paper in tqdm(raw_data, desc="Building nodes"):
        paper_id = paper.get('id', '').replace('https://openalex.org/W', 'W')
        if not paper_id:
            continue
            
        # Reconstruct abstract for LLM processing
        raw_abstract = reconstruct_abstract(paper.get('abstract_inverted_index', {}))
        
        metadata = {
            'paper_id': paper_id,
            'title': paper.get('title', ''),
            'year': paper.get('publication_year'),
            'doi': paper.get('doi', ''),
            'abstract': raw_abstract
        }
        
        authorships = paper.get('authorships', [])
        if authorships:
            first_author = authorships[0]
            institutions = first_author.get('institutions', [])
            if institutions:
                metadata['institution'] = institutions[0].get('display_name', '')
                metadata['country'] = institutions[0].get('country_code', '')
            else:
                metadata['institution'] = metadata['country'] = ''
        else:
            metadata['institution'] = metadata['country'] = ''
            
        papers.append(metadata)
    
    metadata_df = pd.DataFrame(papers)
    metadata_df = normalize_institutions(metadata_df)
    
    # 2. Establish the Closed World constraint
    valid_paper_ids = set(metadata_df['paper_id'].tolist())
    logger.info(f"Closed world established with {len(valid_paper_ids)} unique nodes.")
    
    # 3. Second pass: Build edges ONLY if target is inside the closed world
    valid_edges = []
    for paper in tqdm(raw_data, desc="Building closed-world edges"):
        source_id = paper.get('id', '').replace('https://openalex.org/W', 'W')
        
        if source_id not in valid_paper_ids:
            continue
            
        for ref in paper.get('referenced_works', []):
            if ref:
                target_id = ref.replace('https://openalex.org/W', 'W')
                # CRITICAL: Only add edge if the target paper is actually in our dataset
                if target_id in valid_paper_ids:
                    valid_edges.append({
                        'source': source_id,
                        'target': target_id
                    })
                    
    edges_df = pd.DataFrame(valid_edges)
    return edges_df, metadata_df

def normalize_institutions(df):
    """Normalize institution names using rapidfuzz"""
    institutions = df.loc[df['institution'] != '', 'institution'].unique()
    if len(institutions) <= 1:
        return df
    
    norm_map = {}
    processed = set()
    logger.info(f"Normalizing {len(institutions)} unique institution names...")
    
    for inst in tqdm(institutions, desc="Normalizing institutions"):
        if inst in processed:
            continue
        similar = process.extract(inst, [i for i in institutions if i not in processed], scorer=fuzz.token_sort_ratio, limit=10)
        
        matches = [res[0] for res in similar if len(res) >= 2 and res[1] > 90]
        if matches:
            canonical = inst
            for match in matches:
                norm_map[match] = canonical
                processed.add(match)
                
    df['institution'] = df['institution'].map(lambda x: norm_map.get(x, x))
    return df

def save_outputs(edges_df, metadata_df, output_dir='data'):
    """Save processed data to output files"""
    os.makedirs(output_dir, exist_ok=True)
    edges_path = os.path.join(output_dir, 'edges.csv')
    meta_path = os.path.join(output_dir, 'metadata.csv')

    if os.path.exists(edges_path):
        try:
            edges_df = pd.concat([pd.read_csv(edges_path), edges_df], ignore_index=True)
            edges_df = edges_df.drop_duplicates(subset=['source', 'target']).reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Could not merge edges: {e}")

    if os.path.exists(meta_path):
        try:
            metadata_df = pd.concat([pd.read_csv(meta_path), metadata_df], ignore_index=True)
            metadata_df = metadata_df.drop_duplicates(subset=['paper_id']).reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Could not merge metadata: {e}")

    edges_df.to_csv(edges_path, index=False)
    metadata_df.to_csv(meta_path, index=False)
    
    return len(metadata_df), len(edges_df)

def main():
    parser = argparse.ArgumentParser(description='CiteGraphLens: Citation Network Analysis')
    # Default is now 100,000 papers 
    parser.add_argument('--papers', type=int, default=100000, help='Number of papers to fetch (default: 100000)')
    parser.add_argument('--concept-id', type=str, default="C154945302", help='OpenAlex Concept ID (default: Artificial Intelligence)')
    parser.add_argument('--start-year', type=int, default=2018, help='Start year for citation window')
    parser.add_argument('--end-year', type=int, default=2022, help='End year for citation window')
    parser.add_argument('--email', type=str, default="user@example.com", help='Email for polite API pool')
    parser.add_argument('--output-dir', type=str, default="data", help='Directory to save output files')
    parser.add_argument('--include-references', action='store_true', help='Also fetch metadata for referenced works')
    parser.add_argument('--max-total-refs', type=int, default=50000, help='Global cap on referenced works to fetch (default: 50000)')
    
    args = parser.parse_args()
    
    logger.info(f"Initializing fetch for {args.papers} papers in concept {args.concept_id} ({args.start_year}-{args.end_year})")
    
    # 1. Fetch core papers
    all_papers = fetch_openalex(
        concept_id=args.concept_id, 
        max_results=args.papers, 
        email=args.email, 
        start_year=args.start_year, 
        end_year=args.end_year
    )

    # 2. Optionally fetch references to bulk out the graph
    if args.include_references:
        logger.info("Collecting referenced work IDs...")
        all_targets = []
        for paper in all_papers:
            refs = paper.get('referenced_works', []) or []
            all_targets.extend([r.replace('https://openalex.org/', '') for r in refs if r])

        unique_targets = list(dict.fromkeys(all_targets))
        if args.max_total_refs and len(unique_targets) > args.max_total_refs:
            unique_targets = unique_targets[:args.max_total_refs]

        logger.info(f"Fetching metadata for {len(unique_targets)} referenced works...")
        if unique_targets:
            ref_records = fetch_openalex_by_ids(unique_targets, email=args.email, batch_size=50)
            
            existing_ids = {p.get('id', '').replace('https://openalex.org/W', 'W') for p in all_papers}
            for rec in ref_records:
                rid = rec.get('id', '').replace('https://openalex.org/W', 'W')
                if rid and rid not in existing_ids:
                    all_papers.append(rec)
                    existing_ids.add(rid)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save raw JSONL backup
    meta_path = f'{args.output_dir}/metadata.csv'
    existing_ids = set()
    if os.path.exists(meta_path):
        try:
            existing_ids = set(pd.read_csv(meta_path)['paper_id'].astype(str).tolist())
        except Exception:
            pass

    to_write_raw = [p for p in all_papers if p.get('id', '').replace('https://openalex.org/W', 'W') not in existing_ids]
    if to_write_raw:
        with jsonlines.open(f'{args.output_dir}/openalex_raw.jsonl', mode='a') as writer:
            writer.write_all(to_write_raw)
    
    # 3. Clean metadata, reconstruct abstracts, and enforce closed-world edges
    edges_df, metadata_df = clean_metadata(all_papers)
    
    # 4. Export
    paper_count, citation_count = save_outputs(edges_df, metadata_df, output_dir=args.output_dir)
    
    print(f"\n✅ Graph Construction Complete:")
    print(f"   - Total Nodes (Papers): {paper_count}")
    print(f"   - Total Edges (Internal Citations): {citation_count}")
    print(f"   - Time Window: {args.start_year} - {args.end_year}")
    print(f"✅ Data saved to {args.output_dir}/ directory")

if __name__ == "__main__":
    main()