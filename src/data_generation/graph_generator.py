import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.common import config

class GraphGenerator:
    """Generate test graphs of different sizes"""
    
    @staticmethod
    def generate_graph(num_nodes: int, num_edges_multiplier: int = 3, 
                       graph_type: str = "barabasi", seed: int = 42):
        """
        Generate a graph using different models
        
        Args:
            num_nodes: Number of nodes
            num_edges_multiplier: Multiplier for edges (edges ≈ nodes * multiplier)
            graph_type: "barabasi", "erdos", or "watts"
            seed: Random seed
        """
        np.random.seed(seed)
        
        if graph_type == "barabasi":
            G = nx.barabasi_albert_graph(num_nodes, num_edges_multiplier, seed=seed)
        elif graph_type == "erdos":
            p = (num_edges_multiplier * 2) / num_nodes
            G = nx.erdos_renyi_graph(num_nodes, p, seed=seed)
        else:  # watts
            G = nx.watts_strogatz_graph(num_nodes, min(4, num_nodes-1), 0.3, seed=seed)
        
        # Add weights to edges
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.5, 5.0)
        
        return G
    
    @staticmethod
    def save_graph_to_csv(G: nx.Graph, output_dir: Path, graph_name: str):
        """Save graph as CSV files"""
        output_dir = Path(output_dir) / graph_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save nodes
        nodes_data = []
        for node in G.nodes():
            features = np.random.rand(8).tolist()  # 8D random features
            nodes_data.append({
                "node_id": node,
                "node_name": f"Node_{node}",
                "features": str(features)
            })
        df_nodes = pd.DataFrame(nodes_data)
        df_nodes.to_csv(output_dir / "nodes.csv", index=False)
        
        # Save edges
        edges_data = []
        for u, v, data in G.edges(data=True):
            dist = np.random.uniform(50, 200)
            edges_data.append({
                "source": u,
                "target": v,
                "weight": data.get('weight', 1.0),
                "distance": dist
            })
        df_edges = pd.DataFrame(edges_data)
        df_edges.to_csv(output_dir / "edges.csv", index=False)
        
        # Generate ground truth using Dijkstra
        gt_data = []
        num_samples = min(100, len(G.nodes()) // 2)
        nodes_list = list(G.nodes())
        
        for _ in range(num_samples):
            source = np.random.choice(nodes_list)
            target = np.random.choice(nodes_list)
            if source == target:
                continue
            
            try:
                path = nx.shortest_path(G, source, target, weight='weight')
                cost = nx.shortest_path_length(G, source, target, weight='weight')
                num_hops = len(path) - 1
                gt_data.append({
                    "source": source,
                    "target": target,
                    "shortest_path": str(path),
                    "path_cost": float(cost),
                    "num_hops": num_hops
                })
            except:
                pass
        
        df_gt = pd.DataFrame(gt_data)
        df_gt.to_csv(output_dir / "ground_truth.csv", index=False)
        
        print(f"[OK] Generated graph: {graph_name}")
        print(f"     Nodes: {len(df_nodes)}, Edges: {len(df_edges)}, Test cases: {len(df_gt)}")
        
        return output_dir

def main():
    """Generate all test graphs"""
    print("Generating test graphs...")
    
    # Small graph: 100 nodes
    G_small = GraphGenerator.generate_graph(100, 4, "barabasi")
    GraphGenerator.save_graph_to_csv(G_small, config.GRAPHS_DIR, "graph_small")
    
    # Medium graph: 1000 nodes
    G_medium = GraphGenerator.generate_graph(1000, 5, "barabasi")
    GraphGenerator.save_graph_to_csv(G_medium, config.GRAPHS_DIR, "graph_medium")
    
    # Large graph: 5000 nodes
    G_large = GraphGenerator.generate_graph(5000, 3, "barabasi")
    GraphGenerator.save_graph_to_csv(G_large, config.GRAPHS_DIR, "graph_large")
    
    print("[OK] All test graphs generated!")

if __name__ == "__main__":
    main()
