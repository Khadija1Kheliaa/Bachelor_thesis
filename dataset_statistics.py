import pandas as pd
import numpy as np

def compute_graph_statistics(users_path, edges_path):
   
    # Load datasets
    users_df = pd.read_csv(users_path)
    edges_df = pd.read_csv(edges_path)

    print("==========================================")
    print("    DATASET STATISTICS      ")
    print("==========================================\n")

    # 1. Global Topology Metrics
    n_nodes = len(users_df)
    n_edges = len(edges_df)
    avg_degree = n_edges / n_nodes if n_nodes > 0 else 0
    # Formula for directed graph density: E / (N * (N-1))
    density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0

    print(f"--- [1] GLOBAL TOPOLOGY ---")
    print(f"Total Nodes (Unique Users): {n_nodes}")
    print(f"Total Edges (Interactions): {n_edges}")
    print(f"Average Node Degree:        {avg_degree:.4f}")
    print(f"Graph Density:              {density:.6f}")
    print("-" * 30)

    # 2. Target Label Distribution 
    if 'vandal_label' in users_df.columns:
        counts = users_df['vandal_label'].value_counts()
        safe_count = counts.get(0, 0)
        vandal_count = counts.get(1, 0)
        vandal_pct = (vandal_count / n_nodes) * 100 if n_nodes > 0 else 0

        print(f"\n--- [2] CLASS DISTRIBUTION ---")
        print(f"Safe Users (y=0): {safe_count}")
        print(f"Vandals (y=1):    {vandal_count}")
        print(f"Vandal Prevalence: {vandal_pct:.2f}%")
        print("-" * 30)

    # 3. Edge Type Breakdown
    if 'type' in edges_df.columns:
        edge_types = edges_df['type'].value_counts()
        print(f"\n--- [3] EDGE TYPE BREAKDOWN ---")
        for etype, count in edge_types.items():
            pct = (count / n_edges) * 100
            print(f"{etype:<15}: {count:<8} ({pct:.2f}%)")
        print("-" * 30)

    # 4. Activity Distribution 
    if 'num_edits' in users_df.columns:
        print(f"\n--- [4] SKEWNESS & ACTIVITY ---")
        print(f"Mean Edits per User:   {users_df['num_edits'].mean():.2f}")
        print(f"Median Edits per User: {users_df['num_edits'].median():.2f}")
        print(f"Max Edits (Outlier):   {users_df['num_edits'].max()}")
        print(f"Std Deviation:         {users_df['num_edits'].std():.2f}")
        print("-" * 30)
                    
compute_graph_statistics('data/users_final.csv', 'data/edges_final.csv')