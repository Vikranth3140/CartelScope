#!/usr/bin/env python
"""
CiteGraphLens: Citation Bias & Manipulation Detection
----------------------------------------------------
This script integrates graph metrics, citation intent labels, and paper metadata
to detect patterns of potential citation bias and manipulation.

The workflow includes:
1. Loading and merging graph_metrics.csv, intent_labels.csv, and metadata.csv
2. Computing various bias metrics including self-support ratio and reciprocity index
3. Performing statistical tests to identify unusual citation patterns by region
4. Flagging outlier communities using z-scores
5. Generating a summary report of potential citation cartels
"""

import os
import logging
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define constants
OUTLIER_THRESHOLD = 2.5  # Z-score threshold for flagging outliers

def load_and_merge_data(data_dir, output_dir):
    """
    Load and merge the necessary CSV files for analysis
    
    Parameters:
    -----------
    data_dir : str
        Directory path containing the data files
    output_dir : str
        Directory path for output files
    
    Returns:
    --------
    tuple
        (merged_df, graph_metrics_df, intent_labels_df, metadata_df)
    """
    # Define file paths
    graph_metrics_path = os.path.join(output_dir, 'graph_metrics.csv')
    intent_labels_path = os.path.join(output_dir, 'intent_labels.csv')
    intent_domain_path = os.path.join(output_dir, 'intent_labels_with_domain.csv')
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    
    # Load datasets
    logger.info(f"Loading graph metrics from {graph_metrics_path}")
    try:
        graph_metrics_df = pd.read_csv(graph_metrics_path)
        logger.info(f"Loaded {len(graph_metrics_df)} rows from graph_metrics.csv")
    except FileNotFoundError:
        logger.error(f"File not found: {graph_metrics_path}")
        graph_metrics_df = pd.DataFrame()
    
    logger.info(f"Loading citation intent labels from {intent_labels_path}")
    try:
        intent_labels_df = pd.read_csv(intent_labels_path)
        logger.info(f"Loaded {len(intent_labels_df)} rows from intent_labels.csv")
    except FileNotFoundError:
        logger.error(f"File not found: {intent_labels_path}")
        # Try to load the version with domain information
        try:
            intent_labels_df = pd.read_csv(intent_domain_path)
            logger.info(f"Loaded {len(intent_labels_df)} rows from intent_labels_with_domain.csv")
        except FileNotFoundError:
            logger.error(f"File not found: {intent_domain_path}")
            intent_labels_df = pd.DataFrame()
    
    logger.info(f"Loading paper metadata from {metadata_path}")
    try:
        metadata_df = pd.read_csv(metadata_path)
        logger.info(f"Loaded {len(metadata_df)} rows from metadata.csv")
    except FileNotFoundError:
        logger.error(f"File not found: {metadata_path}")
        metadata_df = pd.DataFrame()
    
    # Check if we have all required data
    if graph_metrics_df.empty or intent_labels_df.empty or metadata_df.empty:
        logger.error("One or more required datasets are missing or empty")
        return None, graph_metrics_df, intent_labels_df, metadata_df
    
    # Rename columns for consistency if needed
    if 'node_id' in graph_metrics_df.columns:
        graph_metrics_df = graph_metrics_df.rename(columns={'node_id': 'paper_id'})
    
    if 'citation_id' in intent_labels_df.columns and 'source_paper' in intent_labels_df.columns:
        # We have source_paper field - use it instead of citation_id
        logger.info("Using source_paper as the paper_id for intent labels")
    elif 'citation_id' in intent_labels_df.columns:
        # Extract paper_id from citation_id if possible
        logger.info("Extracting paper_id from citation_id field")
        intent_labels_df['paper_id'] = intent_labels_df['citation_id'].apply(
            lambda x: x.split('_cites_')[0] if '_cites_' in str(x) else x
        )
    
    # Check if paper_id exists in all dataframes
    required_columns = {
        'graph_metrics_df': 'paper_id' in graph_metrics_df.columns,
        'intent_labels_df': 'paper_id' in intent_labels_df.columns or 'source_paper' in intent_labels_df.columns,
        'metadata_df': 'paper_id' in metadata_df.columns
    }
    
    if not all(required_columns.values()):
        missing = [k for k, v in required_columns.items() if not v]
        logger.error(f"Missing paper_id column in: {', '.join(missing)}")
        return None, graph_metrics_df, intent_labels_df, metadata_df
    
    # If intent labels has source_paper instead of paper_id, rename it
    if 'source_paper' in intent_labels_df.columns and 'paper_id' not in intent_labels_df.columns:
        intent_labels_df = intent_labels_df.rename(columns={'source_paper': 'paper_id'})
    
    # Merge datasets
    logger.info("Merging datasets on paper_id")
    
    # First merge graph_metrics with metadata
    merged_df = pd.merge(
        graph_metrics_df,
        metadata_df,
        on='paper_id',
        how='left'
    )
    
    # Then merge with intent labels
    merged_df = pd.merge(
        merged_df,
        intent_labels_df,
        on='paper_id',
        how='left'
    )
    
    logger.info(f"Merged dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns")
    return merged_df, graph_metrics_df, intent_labels_df, metadata_df

def compute_bias_metrics(merged_df):
    """
    Compute bias metrics from the merged dataset
    
    Parameters:
    -----------
    merged_df : pandas.DataFrame
        The merged dataset containing graph metrics, intent labels, and metadata
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with computed bias metrics
    """
    logger.info("Computing bias metrics")
    
    # Check if we have the necessary columns for computing bias metrics
    required_columns = ['paper_id', 'community', 'self_ratio', 'recip', 'intent']
    missing_columns = [col for col in required_columns if col not in merged_df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {', '.join(missing_columns)}")
        logger.info(f"Available columns: {', '.join(merged_df.columns)}")
        return None
    
    # Group by paper_id to handle potential duplicate rows
    paper_df = merged_df.groupby('paper_id').first().reset_index()
    
    # Calculate community-level metrics
    logger.info("Calculating community-level metrics")
    community_metrics = defaultdict(dict)
    
    for community_id, group in paper_df.groupby('community'):
        # Skip invalid community IDs
        if pd.isna(community_id) or community_id == -1:
            continue
            
        community = int(community_id)  # Convert to int for consistent keys
        community_size = len(group)
        
        # Only analyze communities with sufficient size
        if community_size < 3:
            logger.debug(f"Skipping community {community} with only {community_size} members")
            continue
            
        # Calculate average self-citation ratio
        avg_self_ratio = group['self_ratio'].mean()
        community_metrics[community]['avg_self_ratio'] = avg_self_ratio
        community_metrics[community]['size'] = community_size
        
        # Calculate reciprocity index (average reciprocity within community)
        recip_values = group['recip'].dropna()
        if len(recip_values) > 0:
            reciprocity_index = recip_values.mean()
        else:
            reciprocity_index = 0.0
        community_metrics[community]['reciprocity_index'] = reciprocity_index
    
    # Calculate supportive self-citations by analyzing intent labels
    logger.info("Calculating self-support ratio")
    
    # Build citation_df from merged
    need_cols = {'paper_id','cited_paper','intent','community'}
    if need_cols.issubset(merged_df.columns):
        # Check if institution column exists, if not, create an empty one
        if 'institution' not in merged_df.columns:
            logger.warning("Institution column not found - using empty values")
            merged_df['institution'] = np.nan
            
        citation_df = (merged_df[list(need_cols) + ['institution']]
                   .dropna(subset=['paper_id','cited_paper'])
                   .rename(columns={'paper_id':'source','cited_paper':'target'}))
        # Map institutions
        try:
            inst_map = (merged_df[['paper_id','institution']]
                    .drop_duplicates().rename(columns={'paper_id':'id'}))
            
            # Perform merge operations safely
            citation_df = citation_df.merge(inst_map.rename(columns={'id':'source'}),
                                        on='source', how='left')
            
            # Only rename if the column exists after merge
            if 'institution' in citation_df.columns:
                citation_df = citation_df.rename(columns={'institution':'source_inst'})
            else:
                logger.warning("Institution column not found after source merge - using empty values")
                citation_df['source_inst'] = np.nan
                
            citation_df = citation_df.merge(inst_map.rename(columns={'id':'target'}),
                                        on='target', how='left')
                                        
            # Only rename if the column exists after merge
            if 'institution' in citation_df.columns:
                citation_df = citation_df.rename(columns={'institution':'target_inst'})
            else:
                logger.warning("Institution column not found after target merge - using empty values")
                citation_df['target_inst'] = np.nan
        except Exception as e:
            logger.warning(f"Error during institution mapping: {e}")
            citation_df['source_inst'] = np.nan
            citation_df['target_inst'] = np.nan
                             
        # Check if columns exist before using them
        if 'source_inst' in citation_df.columns and 'target_inst' in citation_df.columns:
            citation_df['is_self_cite'] = (citation_df['source_inst'].notna() &
                                       citation_df['source_inst'].eq(citation_df['target_inst']))
        else:
            logger.warning("Missing source_inst or target_inst columns - cannot calculate self-citation metrics")
            citation_df['is_self_cite'] = False

        supportive = {'method','result'}
        tmp = (citation_df[citation_df['is_self_cite']]
           .assign(is_support=lambda d: d['intent'].isin(supportive)))

        comm = (tmp.groupby('community')
              .agg(total_self=('is_self_cite','size'),
                   support_self=('is_support','sum'))
              .assign(self_support_ratio=lambda d: 
                      np.where(d['total_self']>0, d['support_self']/d['total_self'], np.nan)))

        # inject into community_metrics dict
        for cid, row in comm.reset_index().dropna(subset=['community']).iterrows():
            try:
                cid_int = int(row['community'])
                if cid_int in community_metrics:
                    community_metrics[cid_int]['self_support_ratio'] = row['self_support_ratio']
                    community_metrics[cid_int]['total_self_citations'] = int(row['total_self'])
            except (ValueError, TypeError):
                logger.warning(f"Could not convert community ID {row['community']} to integer")
    else:
        logger.warning("Missing one of required columns for self-support calculation; skipping.")
    
    # Convert to DataFrame
    community_df = pd.DataFrame.from_dict(community_metrics, orient='index').reset_index()
    community_df = community_df.rename(columns={'index': 'community_id'})
    
    # Fill missing values
    if 'self_support_ratio' not in community_df.columns:
        community_df['self_support_ratio'] = np.nan
    
    # Calculate z-scores for each metric to identify outliers
    logger.info("Calculating z-scores for outlier detection")
    
    from scipy.stats import median_abs_deviation
    def robust_z(x):
        # Check if input array is all NaN
        if np.isnan(x).all():
            return np.full_like(x, np.nan, dtype=float)
        
        med = np.nanmedian(x)
        mad = median_abs_deviation(x, nan_policy='omit')
        if not np.isfinite(mad) or mad == 0:
            return np.full_like(x, np.nan, dtype=float)
        return (x - med) / (1.4826 * mad)

    for metric in ['avg_self_ratio', 'reciprocity_index', 'self_support_ratio']:
        if metric in community_df.columns:
            z_col = f"{metric}_zscore"
            
            # Check if we have valid data first
            values = community_df[metric].to_numpy(dtype=float)
            if np.isnan(values).all() or np.nanstd(values) == 0:
                # All NaN or zero variance
                community_df[z_col] = np.nan
            else:
                # Try standard z-score first
                try:
                    z = stats.zscore(community_df[metric], nan_policy='omit')
                    if np.isnan(z).all():
                        # Fall back to robust z-score
                        z = robust_z(values)
                except Exception:
                    # In case of errors, use robust method
                    z = robust_z(values)
                
                community_df[z_col] = z
    
    return community_df

def perform_regional_analysis(merged_df):
    """
    Perform statistical analysis of citation patterns by region
    
    Parameters:
    -----------
    merged_df : pandas.DataFrame
        The merged dataset containing graph metrics, intent labels, and metadata
    
    Returns:
    --------
    dict
        Dictionary containing regional analysis results
    """
    logger.info("Performing regional analysis")
    
    # Check if we have the necessary columns for regional analysis
    if 'country' not in merged_df.columns:
        logger.error("Missing 'country' column, cannot perform regional analysis")
        return None
    
    # Fill missing countries with 'Unknown'
    merged_df['country'] = merged_df['country'].fillna('Unknown')
    
    # Count papers by country
    country_counts = merged_df.groupby('country').size().reset_index(name='paper_count')
    country_counts = country_counts.sort_values('paper_count', ascending=False)
    
    # Calculate expected citation distribution based on paper count
    total_papers = country_counts['paper_count'].sum()
    country_counts['expected_prop'] = country_counts['paper_count'] / total_papers
    
    # Count citations by source country - using a cleaner approach from metadata
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    try:
        # Get a clean source of citation edges from edges.csv
        edges_path = os.path.join(data_dir, 'edges.csv')
        if os.path.exists(edges_path):
            edges = pd.read_csv(edges_path)
            
            # Build country maps directly from metadata
            metadata_df = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
            src_country = metadata_df[['paper_id','country']].rename(columns={'paper_id':'source','country':'source_country'})
            tgt_country = metadata_df[['paper_id','country']].rename(columns={'paper_id':'target','country':'target_country'})
            
            # Merge edges with country information
            citation_df = (edges.merge(src_country, on='source', how='left')
                              .merge(tgt_country, on='target', how='left'))
            
            # Fill missing countries with 'Unknown'
            citation_df['source_country'] = citation_df['source_country'].fillna('Unknown')
            citation_df['target_country'] = citation_df['target_country'].fillna('Unknown')
            
            logger.info(f"Using {len(citation_df)} edges from edges.csv for regional analysis")
        else:
            # Fall back to using citation data from the merged dataset
            logger.warning(f"edges.csv not found at {edges_path}, using intent labels for regional analysis")
            citation_df = merged_df[['paper_id', 'cited_paper', 'country']].dropna(subset=['paper_id', 'cited_paper'])
            citation_df = citation_df.rename(columns={'paper_id': 'source', 'cited_paper': 'target', 'country': 'source_country'})
            
            # Merge with target country information
            target_country_df = merged_df[['paper_id', 'country']].rename(columns={'paper_id': 'target', 'country': 'target_country'})
            citation_df = pd.merge(citation_df, target_country_df, on='target', how='left')
    except Exception as e:
        logger.warning(f"Error loading edges.csv: {e}, falling back to intent labels")
        citation_df = merged_df[['paper_id', 'cited_paper', 'country']].dropna(subset=['paper_id', 'cited_paper'])
        citation_df = citation_df.rename(columns={'paper_id': 'source', 'cited_paper': 'target', 'country': 'source_country'})
        
        # Merge with target country information
        target_country_df = merged_df[['paper_id', 'country']].rename(columns={'paper_id': 'target', 'country': 'target_country'})
        citation_df = pd.merge(citation_df, target_country_df, on='target', how='left')
        
        # Count citations by source->target country
        country_citations = citation_df.groupby(['source_country', 'target_country']).size().reset_index(name='citation_count')
        
        # Pivot to create a citation matrix
        try:
            citation_matrix = country_citations.pivot(index='source_country', columns='target_country', values='citation_count').fillna(0)
            
            # Calculate chi-square test for each source country
            chi_square_results = {}
            for country in citation_matrix.index:
                if country == 'Unknown' or pd.isna(country):
                    continue
                    
                observed = citation_matrix.loc[country]
                # Expected proportions based on overall paper distribution
                expected_props = country_counts.set_index('country')['expected_prop']
                expected = observed.sum() * expected_props
                
                # Remove zeros to avoid division by zero in chi-square test
                valid_cols = [col for col in observed.index if col in expected_props.index and expected_props[col] > 0]
                if len(valid_cols) > 1:  # Need at least two categories for chi-square
                    obs_valid = observed[valid_cols]
                    exp_valid = expected[valid_cols]
                    
                    # Chi-square test
                    chi2, p_value = stats.chisquare(obs_valid, exp_valid)
                    chi_square_results[country] = {
                        'chi2': chi2,
                        'p_value': p_value,
                        'total_citations': observed.sum(),
                        'self_country_ratio': observed[country] / observed.sum() if country in observed and observed.sum() > 0 else 0
                    }
            
            # Convert chi-square results to DataFrame
            chi_df = pd.DataFrame.from_dict(chi_square_results, orient='index').reset_index()
            chi_df = chi_df.rename(columns={'index': 'country'})
            chi_df = chi_df.sort_values('p_value')
            
            return {
                'country_counts': country_counts,
                'citation_matrix': citation_matrix,
                'chi_square_results': chi_df
            }
        except Exception as e:
            logger.error(f"Error creating citation matrix: {e}")
            return {
                'country_counts': country_counts,
                'citation_matrix': pd.DataFrame(),
                'chi_square_results': pd.DataFrame({'country': [], 'chi2': [], 'p_value': [], 'total_citations': [], 'self_country_ratio': []}),
                'error': str(e)
            }
    else:
        logger.warning("Missing 'cited_paper' column - cannot perform detailed regional analysis")
        return {
            'country_counts': country_counts
        }

def flag_outliers_and_cartels(community_df):
    """
    Flag outlier communities and identify potential citation cartels
    
    Parameters:
    -----------
    community_df : pandas.DataFrame
        DataFrame with community metrics
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with outlier flags and bias scores
    """
    logger.info("Flagging outlier communities and identifying potential cartels")
    
    # Create a bias score based on a combination of z-scores
    bias_metrics = []
    
    if 'avg_self_ratio_zscore' in community_df.columns:
        bias_metrics.append('avg_self_ratio_zscore')
    
    if 'reciprocity_index_zscore' in community_df.columns:
        bias_metrics.append('reciprocity_index_zscore')
    
    if 'self_support_ratio_zscore' in community_df.columns:
        bias_metrics.append('self_support_ratio_zscore')
    
    if bias_metrics:
        community_df['bias_score'] = community_df[bias_metrics].mean(axis=1)
        
        # Flag communities with z-score above threshold for any metric
        for metric in bias_metrics:
            flag_col = f"{metric.replace('_zscore', '')}_flag"
            community_df[flag_col] = community_df[metric] > OUTLIER_THRESHOLD
        
        # Flag as potential cartel if multiple metrics are flagged
        community_df['flags_count'] = community_df[[col for col in community_df.columns if col.endswith('_flag')]].sum(axis=1)
        community_df['potential_cartel'] = community_df['flags_count'] >= 2
    else:
        logger.warning("No z-score columns available - cannot create bias score")
        community_df['bias_score'] = np.nan
        community_df['potential_cartel'] = False
    
    # Ensure we have a size column
    if 'size' not in community_df.columns:
        community_df['size'] = 0
    
    # Sort communities by bias score
    community_df = community_df.sort_values('bias_score', ascending=False)
    
    return community_df

def visualize_bias_patterns(community_df, output_dir):
    """
    Create visualizations of bias metrics and patterns
    
    Parameters:
    -----------
    community_df : pandas.DataFrame
        DataFrame with community metrics
    output_dir : str
        Directory to save visualizations
    """
    logger.info("Creating bias pattern visualizations")
    
    if community_df is None or len(community_df) == 0:
        logger.error("No community data available for visualization")
        return
    
    # Set up plot style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot of reciprocity vs self-citation ratio
    if 'reciprocity_index' in community_df.columns and 'avg_self_ratio' in community_df.columns:
        plt.figure(figsize=(10, 8))
        
        # Extract data directly to avoid categorical issues
        x_vals = pd.to_numeric(community_df['reciprocity_index'], errors='coerce')
        y_vals = pd.to_numeric(community_df['avg_self_ratio'], errors='coerce')
        sizes = pd.to_numeric(community_df['size'], errors='coerce')
        is_cartel = community_df['potential_cartel'].fillna(False)
        
        # Normalize sizes for scatter plot
        if sizes.max() > sizes.min():
            size_scaled = 20 + 180 * (sizes - sizes.min()) / (sizes.max() - sizes.min())
        else:
            size_scaled = 100 * np.ones(len(sizes))
        
        # Split data by potential cartel flag
        x_normal = [x for x, c in zip(x_vals, is_cartel) if not c]
        y_normal = [y for y, c in zip(y_vals, is_cartel) if not c]
        size_normal = [s for s, c in zip(size_scaled, is_cartel) if not c]
        
        x_cartel = [x for x, c in zip(x_vals, is_cartel) if c]
        y_cartel = [y for y, c in zip(y_vals, is_cartel) if c]
        size_cartel = [s for s, c in zip(size_scaled, is_cartel) if c]
        
        # Plot using matplotlib directly
        plt.scatter(x_normal, y_normal, s=size_normal, c='blue', alpha=0.7, label='Regular')
        
        if x_cartel:  # Only plot if there are cartel points
            plt.scatter(x_cartel, y_cartel, s=size_cartel, c='red', alpha=0.7, label='Potential Cartel')
            
        plt.legend()
        
        # Add community IDs as labels for potential cartels
        for idx, (x, y, is_c) in enumerate(zip(x_vals, y_vals, is_cartel)):
            if is_c and not pd.isna(x) and not pd.isna(y):
                try:
                    comm_id = int(float(community_df.iloc[idx]['community_id']))
                    label = f"C{comm_id}"
                except (ValueError, TypeError):
                    label = f"C{community_df.iloc[idx]['community_id']}"
                    
                plt.annotate(
                    label,
                    (x, y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontweight='bold'
                )
        
        plt.title('Citation Patterns by Community: Reciprocity vs Self-Citation')
        plt.xlabel('Reciprocity Index')
        plt.ylabel('Self-Citation Ratio')
        plt.tight_layout()
        
        # Save plot
        viz_path = os.path.join(output_dir, 'citation_patterns.png')
        plt.savefig(viz_path)
        logger.info(f"Saved citation pattern visualization to {viz_path}")
        plt.close()
    
    # Create bar plot of bias scores for top communities
    plt.figure(figsize=(12, 6))
    top_n = min(15, len(community_df))  # Show top 15 or fewer if less data
    top_communities = community_df.head(top_n)
    
    # Create a copy and convert types properly
    top_communities = top_communities.copy()
    
    # Create a categorical x-variable for proper plotting
    community_ids = top_communities['community_id'].tolist()
    bias_scores = pd.to_numeric(top_communities['bias_score'], errors='coerce').tolist()
    is_cartel = top_communities['potential_cartel'].tolist()
    
    # Create numeric x positions for the bars
    x_pos = range(len(community_ids))
    
    # Create the bar plot manually to avoid categorical warnings
    cartel_mask = [c for c, pc in zip(x_pos, is_cartel) if pc]
    non_cartel_mask = [c for c, pc in zip(x_pos, is_cartel) if not pc]
    
    # Plot bars directly using matplotlib instead of seaborn
    plt.bar([x for x in x_pos if x in non_cartel_mask], 
            [bias_scores[i] for i in range(len(bias_scores)) if i in non_cartel_mask],
            color='blue', label='Regular')
            
    plt.bar([x for x in x_pos if x in cartel_mask], 
            [bias_scores[i] for i in range(len(bias_scores)) if i in cartel_mask],
            color='red', label='Potential Cartel')
    
    # Set x-ticks with community IDs as labels
    plt.xticks(x_pos, [str(cid) for cid in community_ids], rotation=45)
    
    plt.title('Bias Scores of Top Communities')
    plt.xlabel('Community ID')
    plt.ylabel('Bias Score')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    viz_path = os.path.join(output_dir, 'bias_scores.png')
    plt.savefig(viz_path)
    logger.info(f"Saved bias scores visualization to {viz_path}")
    plt.close()

def main():
    """Main function to execute the bias detection workflow"""
    # Set up directories
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Load and merge data
    merged_df, graph_df, intent_df, metadata_df = load_and_merge_data(data_dir, output_dir)
    
    if merged_df is None:
        logger.error("Failed to merge datasets - cannot proceed")
        return
    
    # Compute bias metrics
    community_df = compute_bias_metrics(merged_df)
    
    if community_df is None:
        logger.error("Failed to compute bias metrics - cannot proceed")
        return
    
    # Perform regional analysis
    regional_analysis = perform_regional_analysis(merged_df)
    
    # Flag outliers and identify potential cartels
    community_df = flag_outliers_and_cartels(community_df)
    
    # Create visualizations
    visualize_bias_patterns(community_df, output_dir)
    
    # Save bias summary
    bias_summary_path = os.path.join(output_dir, 'bias_summary.csv')
    community_df.to_csv(bias_summary_path, index=False)
    logger.info(f"Saved bias summary to {bias_summary_path}")
    
    # Print top potential citation cartels
    top_cartels = community_df[community_df['potential_cartel']].head(10)
    
    if len(top_cartels) > 0:
        logger.info("\n===== TOP POTENTIAL CITATION CARTELS =====")
        for i, (_, cartel) in enumerate(top_cartels.iterrows(), 1):
            # Safe conversion of community_id and size to int with error handling
            try:
                comm_id = int(float(cartel['community_id']))
            except (ValueError, TypeError):
                comm_id = cartel['community_id']
            
            try:
                size = int(float(cartel['size']))
            except (ValueError, TypeError):
                size = cartel['size']
                
            bias = cartel['bias_score']
            
            metrics = []
            if 'avg_self_ratio' in cartel:
                metrics.append(f"self-citation: {cartel['avg_self_ratio']:.2f}")
            if 'reciprocity_index' in cartel:
                metrics.append(f"reciprocity: {cartel['reciprocity_index']:.2f}")
            if 'self_support_ratio' in cartel:
                metrics.append(f"self-support: {cartel['self_support_ratio']:.2f}")
                
            metrics_str = ", ".join(metrics)
            logger.info(f"{i}. Community {comm_id} (size: {size}) - Bias Score: {bias:.2f} ({metrics_str})")
    else:
        logger.info("No potential citation cartels identified.")
    
    logger.info("Bias and manipulation detection completed successfully")

if __name__ == "__main__":
    main()