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
    level=logging.DEBUG,  # Change to DEBUG to see more information
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("citegraphlens.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_openalex(domain, max_results=5000, email="user@example.com", year=None):
    """
    Query the OpenAlex API for papers in a specific domain
    
    Parameters:
    -----------
    domain : str
        Academic domain to query (e.g., 'Computer Science', 'Biology', 'Medicine')
    max_results : int
        Maximum number of results to fetch
    email : str
        Email to include in API requests for better rate limits
    
    Returns:
    --------
    list
        List of paper records from OpenAlex API
    """
    base_url = "https://api.openalex.org/works"
    
    # Map domains to OpenAlex concept IDs
    domain_ids = {
        'Computer Science': 'C41008148',  # Computer Science concept ID
        'Biology': 'C86803240',          # Biology concept ID
        'Medicine': 'C71924100'          # Medicine concept ID
    }
    
    if domain not in domain_ids:
        raise ValueError(f"Domain must be one of {list(domain_ids.keys())}")
    
    # Prepare query parameters
    # Build publication year filter. If year is provided, fetch that year only.
    year_filter = ''
    if year is not None:
        year_filter = f",publication_year:{int(year)}"
    else:
        # Keep previous default behavior (all years before 2023)
        year_filter = ",publication_year:<2023"

    params = {
        'filter': f"concepts.id:{domain_ids[domain]}{year_filter}",
        'per_page': 200,  # Maximum allowed by the API
        'select': "id,doi,title,publication_year,authorships,referenced_works",
        'sort': "publication_date:desc",
        'mailto': email  # Use polite pool by including email
    }
    
    headers = {'User-Agent': f'CiteGraphLens/1.0 ({email})'}
    
    all_papers = []
    
    # Add cursor parameter to enable cursor-based pagination
    params['cursor'] = '*'
    
    with tqdm(total=max_results, desc=f"Fetching {domain} papers") as pbar:
        while len(all_papers) < max_results:
            try:
                response = requests.get(base_url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                # Extract papers from response
                papers = data.get('results', [])
                if not papers:
                    logger.warning(f"No papers returned in response, breaking pagination loop")
                    break
                
                # Get response metadata
                meta = data.get('meta', {})
                count = meta.get('count', 0)
                per_page = meta.get('per_page', 0)
                next_cursor = meta.get('next_cursor')
                
                logger.debug(f"API response: count={count}, per_page={per_page}, papers={len(papers)}, next_cursor={next_cursor}")
                
                # Add papers to our collection
                all_papers.extend(papers)
                pbar.update(len(papers))
                
                # Respect API rate limits - stay well under 10 requests/second
                time.sleep(0.3)  # Delay to avoid rate limiting (approximately 3.33 requests per second)
                
                # Check if we need to continue pagination
                if len(all_papers) >= max_results or not next_cursor:
                    logger.debug(f"Stopping pagination: collected={len(all_papers)}, max_results={max_results}, has_next_cursor={bool(next_cursor)}")
                    break
                
                # Update the cursor for the next request
                params['cursor'] = next_cursor
                    
            except requests.exceptions.HTTPError as e:
                # Handle rate limit errors specifically
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limit hit. Waiting for {retry_after} seconds before retrying.")
                    time.sleep(retry_after)
                    continue
                else:
                    logger.error(f"HTTP error: {e}")
                    time.sleep(2)  # Wait before retrying
                    continue
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {e}")
                time.sleep(2)  # Wait before retrying
                continue
    
    logger.info(f"Successfully collected {len(all_papers)} papers from {domain}")
    return all_papers


def fetch_openalex_by_ids(work_ids, email="user@example.com", batch_size=50):
    """
    Fetch OpenAlex works by a list of work IDs (e.g., 'W12345').

    Parameters:
    -----------
    work_ids : list
        List of OpenAlex work IDs (without the https://openalex.org/ prefix)
    email : str
        Email to include in API requests
    batch_size : int
        Number of IDs to request per API call (OpenAlex supports filtering by comma-separated ids)

    Returns:
    --------
    list
        List of work records fetched from OpenAlex
    """
    base_url = "https://api.openalex.org/works"
    headers = {'User-Agent': f'CiteGraphLens/1.0 ({email})'}

    fetched = []
    # OpenAlex allows filtering by ids:W1,W2,...; keep batches reasonable to avoid URL length issues
    for i in range(0, len(work_ids), batch_size):
        batch = work_ids[i:i+batch_size]
        ids_filter = ','.join([f"https://openalex.org/{w}" if not w.startswith('https://') else w for w in batch])
        params = {
            'filter': f"id:{ids_filter}",
            'per_page': 200,
            'mailto': email,
            'select': "id,doi,title,publication_year,authorships,referenced_works"
        }

        try:
            resp = requests.get(base_url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            results = data.get('results', [])
            fetched.extend(results)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit hit when fetching by ids. Waiting {retry_after}s")
                time.sleep(retry_after)
                # retry this batch once
                try:
                    resp = requests.get(base_url, params=params, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    results = data.get('results', [])
                    fetched.extend(results)
                except Exception:
                    logger.error(f"Failed to fetch batch starting at index {i}")
                    continue
            else:
                logger.error(f"HTTP error fetching ids batch: {e}")
                continue
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed when fetching ids batch: {e}")
            continue

        # polite pause
        time.sleep(0.3)

    return fetched

def clean_metadata(raw_data, domain_papers=None):
    """
    Process and clean the raw data from OpenAlex
    
    Parameters:
    -----------
    raw_data : list
        List of paper records from OpenAlex API
    domain_papers : dict, optional
        Dictionary mapping paper IDs to their domains
    
    Returns:
    --------
    tuple
        (edges_df, metadata_df) - Dataframes for citation edges and paper metadata
    """
    papers = []
    edges = []
    
    logger.info("Processing and cleaning paper data...")
    
    for paper in tqdm(raw_data, desc="Processing papers"):
        paper_id = paper.get('id', '').replace('https://openalex.org/W', 'W')
        
        # Skip if no paper ID
        if not paper_id:
            continue
        
        # Extract basic metadata
        metadata = {
            'paper_id': paper_id,
            'title': paper.get('title', ''),
            'year': paper.get('publication_year'),
            'doi': paper.get('doi', ''),
            'domain': domain_papers.get(paper_id, '') if domain_papers else ''
        }
        
        # Process author information
        authorships = paper.get('authorships', [])
        
        # Use the first author's institution if available
        if authorships:
            first_author = authorships[0]
            institutions = first_author.get('institutions', [])
            
            if institutions:
                institution = institutions[0]
                metadata['institution'] = institution.get('display_name', '')
                metadata['country'] = institution.get('country_code', '')
            else:
                metadata['institution'] = ''
                metadata['country'] = ''
        else:
            metadata['institution'] = ''
            metadata['country'] = ''
        
        # Add to papers list
        papers.append(metadata)
        
        # Process citation edges
        references = paper.get('referenced_works', [])
        for ref in references:
            if ref:
                target_id = ref.replace('https://openalex.org/W', 'W')
                edges.append({
                    'source': paper_id,
                    'target': target_id
                })
    
    # Create DataFrames
    metadata_df = pd.DataFrame(papers)
    edges_df = pd.DataFrame(edges)
    
    # Clean institution names using rapidfuzz
    metadata_df = normalize_institutions(metadata_df)
    
    return edges_df, metadata_df

def normalize_institutions(df):
    """
    Normalize institution names using rapidfuzz
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with institution column
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with normalized institution names
    """
    # Only process rows with non-empty institution names
    institutions = df.loc[df['institution'] != '', 'institution'].unique()
    
    if len(institutions) <= 1:
        return df
    
    # Create a mapping of similar institution names
    norm_map = {}
    processed = set()
    
    logger.info(f"Normalizing {len(institutions)} unique institution names...")
    
    for inst in tqdm(institutions, desc="Normalizing institutions"):
        if inst in processed:
            continue
            
        # Find similar institution names
        similar = process.extract(
            inst, 
            [i for i in institutions if i not in processed], 
            scorer=fuzz.token_sort_ratio, 
            limit=10
        )
        
        # Group institutions with similarity score > 90
        matches = []
        for result in similar:
            # Each result is (match, score, index) or just (match, score)
            if len(result) >= 2 and result[1] > 90:
                matches.append(result[0])
        
        if matches:
            # Use the most common or first as canonical form
            canonical = inst
            for match in matches:
                norm_map[match] = canonical
                processed.add(match)
    
    # Apply normalization
    df['institution'] = df['institution'].map(lambda x: norm_map.get(x, x))
    
    logger.info(f"Reduced to {len(df['institution'].unique())} unique institutions after normalization")
    return df

def save_outputs(edges_df, metadata_df, output_dir='data'):
    """
    Save processed data to output files
    
    Parameters:
    -----------
    edges_df : pandas.DataFrame
        DataFrame with citation edges
    metadata_df : pandas.DataFrame
        DataFrame with paper metadata
    output_dir : str, optional
        Directory to save output files (default: 'data')
    
    Returns:
    --------
    tuple
        (paper_count, citation_count) - Counts of papers and citations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    edges_path = os.path.join(output_dir, 'edges.csv')
    meta_path = os.path.join(output_dir, 'metadata.csv')

    # Merge with existing edges if present
    if os.path.exists(edges_path):
        try:
            existing_edges = pd.read_csv(edges_path)
            edges_df = pd.concat([existing_edges, edges_df], ignore_index=True)
            if not edges_df.empty:
                edges_df = edges_df.drop_duplicates(subset=['source', 'target']).reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Could not merge existing edges.csv: {e}")

    # Merge with existing metadata if present
    if os.path.exists(meta_path):
        try:
            existing_meta = pd.read_csv(meta_path)
            metadata_df = pd.concat([existing_meta, metadata_df], ignore_index=True)
            if not metadata_df.empty:
                metadata_df = metadata_df.drop_duplicates(subset=['paper_id']).reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Could not merge existing metadata.csv: {e}")

    # Save processed data as CSV (merged)
    edges_df.to_csv(edges_path, index=False)
    metadata_df.to_csv(meta_path, index=False)

    logger.info(f"Saved {len(metadata_df)} papers and {len(edges_df)} citations (merged with existing files if present)")
    return len(metadata_df), len(edges_df)

def main():
    """Main function to run the data collection and processing pipeline"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CiteGraphLens: Citation Network Analysis')
    parser.add_argument('--cs-papers', type=int, default=5000,
                        help='Number of Computer Science papers to fetch (default: 5000)')
    parser.add_argument('--bio-papers', type=int, default=5000,
                        help='Number of Biology papers to fetch (default: 5000)')
    parser.add_argument('--med-papers', type=int, default=5000,
                        help='Number of Medicine papers to fetch (default: 5000)')
    parser.add_argument('--email', type=str, default="user@example.com",
                        help='Email to include in API requests')
    parser.add_argument('--output-dir', type=str, default="data",
                        help='Directory to save output files (default: data)')
    parser.add_argument('--include-references', action='store_true',
                        help='Also fetch metadata for referenced works and include them in the dataset')
    parser.add_argument('--max-refs-per-paper', type=int, default=200,
                        help='Max number of referenced works to consider per paper when include-references is set (default: 200)')
    parser.add_argument('--max-total-refs', type=int, default=10000,
                        help='Global cap on number of referenced works to fetch (default: 10000)')
    parser.add_argument('--year', type=int, default=None,
                        help='Publication year to filter papers by (e.g., 2020). If omitted, previous default (<2023) is used')
    
    args = parser.parse_args()
    
    all_papers = []
    domain_papers = {}  # Track which papers belong to which domain
    
    # Fetch papers from all domains
    domain_counts = {
        'Computer Science': args.cs_papers,
        'Biology': args.bio_papers,
        'Medicine': args.med_papers
    }
    
    for domain, count in domain_counts.items():
        papers = fetch_openalex(domain, max_results=count, email=args.email, year=args.year)
        
        # Track domain for each paper
        for paper in papers:
            paper_id = paper.get('id', '').replace('https://openalex.org/W', 'W')
            domain_papers[paper_id] = domain
        
        all_papers.extend(papers)

    # Optionally fetch referenced works' metadata to enrich the dataset
    if args.include_references:
        logger.info("Collecting referenced work IDs from fetched papers...")
        all_targets = []
        for paper in all_papers:
            refs = paper.get('referenced_works', []) or []
            # Limit per paper to avoid exploding the request size
            if args.max_refs_per_paper and len(refs) > args.max_refs_per_paper:
                refs = refs[:args.max_refs_per_paper]

            # Normalize ids (strip prefix)
            normalized = [r.replace('https://openalex.org/', '') for r in refs if r]
            all_targets.extend(normalized)

        # Deduplicate and enforce a global cap
        unique_targets = list(dict.fromkeys(all_targets))  # preserves order
        if args.max_total_refs and len(unique_targets) > args.max_total_refs:
            unique_targets = unique_targets[:args.max_total_refs]

        logger.info(f"Fetching metadata for {len(unique_targets)} referenced works (capped)")
        if unique_targets:
            ref_records = fetch_openalex_by_ids(unique_targets, email=args.email, batch_size=50)

            # Append fetched reference records to all_papers if not already present
            existing_ids = {p.get('id', '').replace('https://openalex.org/W', 'W') for p in all_papers}
            new_count = 0
            for rec in ref_records:
                rid = rec.get('id', '').replace('https://openalex.org/W', 'W')
                if rid and rid not in existing_ids:
                    # Mark domain as empty for referenced works (they're not in the original seed domains)
                    all_papers.append(rec)
                    domain_papers[rid] = ''
                    existing_ids.add(rid)
                    new_count += 1

            logger.info(f"Appended {new_count} referenced works to dataset")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save raw data for later reference
    # Append raw JSONL entries only for papers not already present in existing metadata
    meta_path = f'{args.output_dir}/metadata.csv'
    existing_ids = set()
    if os.path.exists(meta_path):
        try:
            existing_meta = pd.read_csv(meta_path)
            existing_ids = set(existing_meta['paper_id'].astype(str).tolist())
        except Exception:
            existing_ids = set()

    to_write_raw = []
    for p in all_papers:
        pid = p.get('id', '').replace('https://openalex.org/W', 'W')
        if pid not in existing_ids:
            to_write_raw.append(p)

    if to_write_raw:
        with jsonlines.open(f'{args.output_dir}/openalex_raw.jsonl', mode='a') as writer:
            writer.write_all(to_write_raw)
    
    # Clean and process the data
    edges_df, metadata_df = clean_metadata(all_papers, domain_papers)
    
    # Save outputs to CSV files
    paper_count, citation_count = save_outputs(edges_df, metadata_df, output_dir=args.output_dir)
    
    # Print summary
    domain_counts_actual = metadata_df['domain'].value_counts().to_dict()
    cs_papers = domain_counts_actual.get('Computer Science', 0)
    bio_papers = domain_counts_actual.get('Biology', 0)
    med_papers = domain_counts_actual.get('Medicine', 0)
    
    print(f"\n✅ Collected {paper_count} papers:")
    print(f"   - Computer Science: {cs_papers} papers (requested: {domain_counts['Computer Science']})")
    print(f"   - Biology: {bio_papers} papers (requested: {domain_counts['Biology']})")
    print(f"   - Medicine: {med_papers} papers (requested: {domain_counts['Medicine']})")
    print(f"✅ Found {citation_count} citations between these papers")
    print(f"✅ Data saved to {args.output_dir}/ directory")

if __name__ == "__main__":
    main()