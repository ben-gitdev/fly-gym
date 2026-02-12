"""
Script to create a randomized connectome edge list.

This script loads the original edge list from a parquet file, shuffles the 
Presynaptic_ID and Postsynaptic_ID columns independently, replaces the 
connectivity weights with random numbers, and saves the result to a new file.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
CONNECTOME_DIR = Path(__file__).parent.parent / "connectomes" / "drosophila adult connectome"
# INPUT_FILE = CONNECTOME_DIR / "Connectivity_783.parquet"
# OUTPUT_FILE = CONNECTOME_DIR / "Connectivity_random.parquet"
INPUT_FILE = CONNECTOME_DIR / "connections_princeton.csv"
OUTPUT_FILE = CONNECTOME_DIR / "connections_princeton_random.csv"


def load_edge_list(path: Path) -> pd.DataFrame:
    """Load the edge list from a parquet file."""
    print(f"Loading edge list from {path}...")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} edges.")
    print(f"  Columns: {list(df.columns)}")
    return df


def create_random_edge_list(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Create a randomized edge list by:
    1. Getting all unique synapse IDs from Presynaptic_ID and Postsynaptic_ID
    2. Replacing each element in both columns with randomly sampled IDs from the full list
    3. Ensuring no duplicate edges exist (if duplicate, replace with new edge)
    4. Replacing 'Excitatory x Connectivity' with random numbers
    
    Args:
        df: Original edge list DataFrame
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with randomized connections and weights (no duplicate edges)
    """
    np.random.seed(seed)
    df_random = df.copy()
    
    # Get all unique synapse IDs from both columns
    all_synapse_ids = np.unique(np.concatenate([
        df["pre_root_id"].values,
        df["post_root_id"].values
    ]))
    print(f"Found {len(all_synapse_ids)} unique synapse IDs.")
    
    # Replace each element with randomly sampled ID from the full list
    n_edges = len(df_random)
    random_pre_ids = np.random.choice(all_synapse_ids, size=n_edges, replace=True)
    random_post_ids = np.random.choice(all_synapse_ids, size=n_edges, replace=True)
    
    # Remove duplicate edges by regenerating duplicates
    print("Checking for duplicate edges...")
    seen_edges = set()
    duplicates_replaced = 0
    
    for i in range(n_edges):
        edge = (random_pre_ids[i], random_post_ids[i])
        
        # If duplicate, keep regenerating until unique
        while edge in seen_edges:
            random_pre_ids[i] = np.random.choice(all_synapse_ids)
            random_post_ids[i] = np.random.choice(all_synapse_ids)
            edge = (random_pre_ids[i], random_post_ids[i])
            duplicates_replaced += 1
        
        seen_edges.add(edge)
        
        if (i + 1) % 1000000 == 0:
            print(f"  Processed {i + 1:,}/{n_edges:,} edges...")
    
    print(f"Replaced {duplicates_replaced:,} duplicate edges.")
    
    df_random["pre_root_id"] = random_pre_ids
    df_random["post_root_id"] = random_post_ids
    
    # Replace connectivity weights with random numbers
    # Use uniform distribution in similar range as original data
    original_weights = df["syn_count"].values
    min_w = original_weights.min()
    max_w = original_weights.max()
    
    print(f"Original weight range: [{min_w:.4f}, {max_w:.4f}]")
    
    # Generate random weights in the same range
    random_weights = np.random.uniform(min_w, max_w, size=len(df_random))
    df_random["syn_count"] = random_weights
    
    print(f"Created randomized edge list with {len(df_random)} unique edges.")
    
    return df_random


def save_edge_list(df: pd.DataFrame, path: Path) -> None:
    """Save the edge list to a parquet file."""
    print(f"Saving randomized edge list to {path}...")
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} edges.")


def main():
    # Load original edge list
    df = load_edge_list(INPUT_FILE)
    
    # Create randomized version
    df_random = create_random_edge_list(df)
    
    # Save to new file
    save_edge_list(df_random, OUTPUT_FILE)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
