import pandas as pd
import numpy as np
from pathlib import Path
from src.common import config

class GraphLoader:
    """Load graph data from CSV files"""
    
    def __init__(self, graph_dir: Path):
        self.graph_dir = Path(graph_dir)
        self.nodes = None
        self.edges = None
        self.ground_truth = None
        self.load_data()
    
    def load_data(self):
        """Load all CSV files"""
        nodes_file = self.graph_dir / "nodes.csv"
        edges_file = self.graph_dir / "edges.csv"
        gt_file = self.graph_dir / "ground_truth.csv"
        
        if nodes_file.exists():
            self.nodes = pd.read_csv(nodes_file)
        if edges_file.exists():
            self.edges = pd.read_csv(edges_file)
        if gt_file.exists():
            self.ground_truth = pd.read_csv(gt_file)
    
    def get_nodes(self) -> pd.DataFrame:
        return self.nodes
    
    def get_edges(self) -> pd.DataFrame:
        return self.edges
    
    def get_ground_truth(self) -> pd.DataFrame:
        return self.ground_truth
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Convert edges to adjacency matrix"""
        if self.edges is None:
            return None
        
        num_nodes = len(self.nodes)
        adj_matrix = np.zeros((num_nodes, num_nodes))
        
        for _, row in self.edges.iterrows():
            src = int(row['source'])
            tgt = int(row['target'])
            adj_matrix[src, tgt] = row['weight']
            adj_matrix[tgt, src] = row['weight']  # Undirected
        
        return adj_matrix
    
    def get_node_features(self) -> np.ndarray:
        """Extract node features"""
        if self.nodes is None:
            return None
        
        features = []
        for _, row in self.nodes.iterrows():
            feat_str = row['features'].strip('[]').split(',')
            feat = [float(x.strip()) for x in feat_str]
            features.append(feat)
        
        return np.array(features)

def get_default_graph_loader(size: str = "medium") -> GraphLoader:
    """Get default graph loader"""
    graph_dir = config.GRAPHS_DIR / f"graph_{size}"
    if not graph_dir.exists():
        raise FileNotFoundError(f"Graph directory not found: {graph_dir}")
    return GraphLoader(graph_dir)
