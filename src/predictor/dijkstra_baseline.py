import numpy as np
import pandas as pd
import heapq
import torch
import torch.nn as nn
import json
import os
from pathlib import Path

class DijkstraBaseline:
    """Traditional Dijkstra's algorithm for shortest path"""
    
    def __init__(self, adj_matrix: np.ndarray):
        self.adj_matrix = adj_matrix
        self.adjacency_matrix = adj_matrix  # Add alias for compatibility
        self.num_nodes = adj_matrix.shape[0]
    
    def find_shortest_path(self, source: int, target: int):
        """Find shortest path using Dijkstra"""
        distances = [float('inf')] * self.num_nodes
        distances[source] = 0
        parent = [-1] * self.num_nodes
        visited = [False] * self.num_nodes
        
        pq = [(0, source)]
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if visited[current_node]:
                continue
            
            visited[current_node] = True
            
            if current_node == target:
                break
            
            for neighbor in range(self.num_nodes):
                if self.adj_matrix[current_node, neighbor] > 0:
                    weight = self.adj_matrix[current_node, neighbor]
                    distance = current_dist + weight
                    
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        parent[neighbor] = current_node
                        heapq.heappush(pq, (distance, neighbor))
        
        # Reconstruct path
        path = []
        current = target
        while current != -1:
            path.append(current)
            current = parent[current]
        path.reverse()
        
        return path if path[0] == source else [], distances[target]

class MLPredictor:
    """ML-based shortest path predictor using trained GraphSAGE model"""
    
    def __init__(self, adj_matrix: np.ndarray = None, model_path: str = None, 
                 node_features: np.ndarray = None):
        """
        Args:
            adj_matrix: Adjacency matrix (n_nodes x n_nodes)
            model_path: Path to trained model directory
            node_features: Node features for model input
        """
        self.adj_matrix = adj_matrix
        self.num_nodes = adj_matrix.shape[0] if adj_matrix is not None else 0
        self.model = None
        self.node_features = node_features
        self.node_embeddings = None
        self.device = torch.device("cpu")
        
        # Try to load trained model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Fallback to Dijkstra if no model
            self.use_dijkstra = True
    
    def load_model(self, model_path: str):
        """Load trained GraphSAGE model"""
        try:
            # Import model class
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
            from graphsage_model import ShortestPathModel
            
            # Load config
            config_path = os.path.join(model_path, 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create model
            self.model = ShortestPathModel(
                input_dim=config['input_dim'],
                embedding_dim=config['embedding_dim'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers']
            ).to(self.device)
            
            # Load weights
            model_weights_path = os.path.join(model_path, 'best_model.pt')
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
            self.model.eval()
            
            # Prepare input tensors
            adj_tensor = torch.from_numpy(self.adj_matrix).float().to(self.device)
            
            if self.node_features is None:
                # Use degree as features
                degrees = np.array([self.adj_matrix[i].sum() for i in range(self.num_nodes)])
                degrees = (degrees - degrees.mean()) / (degrees.std() + 1e-6)
                self.node_features = degrees.reshape(-1, 1).astype(np.float32)
            
            features_tensor = torch.from_numpy(self.node_features).float().to(self.device)
            
            # Generate node embeddings
            with torch.no_grad():
                self.node_embeddings = self.model.encode(features_tensor, adj_tensor)
            
            self.use_dijkstra = False
            print(f"[OK] GraphSAGE model loaded from {model_path}")
            
        except Exception as e:
            print(f"[WARNING] Failed to load model: {e}")
            print(f"[INFO] Falling back to Dijkstra algorithm")
            self.use_dijkstra = True
    
    def predict_shortest_path(self, source: int, target: int):
        """
        Predict shortest path using ML model
        
        If model available: Uses trained GraphSAGE for path cost estimation
        If model unavailable: Falls back to Dijkstra algorithm
        
        Args:
            source: source node ID
            target: target node ID
        Returns:
            (path, estimated_cost)
        """
        if self.use_dijkstra or self.model is None:
            return self._predict_dijkstra(source, target)
        
        # Use ML model with beam search
        return self._predict_with_model(source, target)
    
    def _predict_dijkstra(self, source: int, target: int):
        """Fallback to Dijkstra algorithm"""
        if self.adj_matrix is None:
            return [source, target], 1.0
        
        distances = [float('inf')] * self.num_nodes
        distances[source] = 0
        parent = [-1] * self.num_nodes
        visited = [False] * self.num_nodes
        
        pq = [(0, source)]
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if visited[current_node]:
                continue
            
            visited[current_node] = True
            
            if current_node == target:
                break
            
            for neighbor in range(self.num_nodes):
                if self.adj_matrix[current_node, neighbor] > 0:
                    weight = self.adj_matrix[current_node, neighbor]
                    distance = current_dist + weight
                    
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        parent[neighbor] = current_node
                        heapq.heappush(pq, (distance, neighbor))
        
        # Reconstruct path
        path = []
        current = target
        while current != -1:
            path.append(current)
            current = parent[current]
        path.reverse()
        
        return path if path[0] == source else [], distances[target]
    
    def _predict_with_model(self, source: int, target: int, beam_width: int = 5):
        """
        Predict path using ML model with beam search
        
        Args:
            source: source node ID
            target: target node ID
            beam_width: width of beam search
        Returns:
            (path, estimated_cost)
        """
        # Beam search: keep top-k candidates
        candidates = [(0.0, [source])]  # (cost, path)
        visited_paths = set()
        
        with torch.no_grad():
            for _ in range(self.num_nodes):  # Maximum depth
                new_candidates = []
                
                for cost, path in candidates:
                    current_node = path[-1]
                    
                    if current_node == target:
                        return path, float(cost)
                    
                    # Get neighbors
                    neighbors = []
                    for next_node in range(self.num_nodes):
                        if (self.adj_matrix[current_node, next_node] > 0 and 
                            next_node not in path):  # Avoid cycles
                            neighbors.append(next_node)
                    
                    # Score each neighbor using model
                    for next_node in neighbors:
                        source_emb = self.node_embeddings[current_node:current_node+1]
                        target_emb = self.node_embeddings[next_node:next_node+1]
                        
                        predicted_cost = self.model.predict_cost(source_emb, target_emb)
                        edge_cost = predicted_cost.item()
                        new_path = path + [next_node]
                        new_cost = cost + edge_cost
                        
                        new_candidates.append((new_cost, new_path))
                
                if not new_candidates:
                    break
                
                # Keep top-k
                candidates = sorted(new_candidates, key=lambda x: x[0])[:beam_width]
        
        # Return best path found
        if candidates:
            best_cost, best_path = candidates[0]
            return best_path, float(best_cost)
        
        # Fallback
        return [source, target], 1.0


def compare_algorithms(adj_matrix: np.ndarray, test_cases: list):
    """Compare ML vs Dijkstra on test cases"""
    dijkstra = DijkstraBaseline(adj_matrix)
    ml = MLPredictor()
    
    results = []
    for source, target in test_cases:
        try:
            # Dijkstra
            dij_path, dij_cost = dijkstra.find_shortest_path(source, target)
            
            # ML
            ml_path, ml_cost = ml.predict_shortest_path(source, target)
            
            results.append({
                "source": source,
                "target": target,
                "dijkstra_path": str(dij_path),
                "dijkstra_cost": dij_cost,
                "ml_path": str(ml_path),
                "ml_cost": ml_cost,
                "accuracy": 1.0 if dij_cost == ml_cost else 0.0
            })
        except Exception as e:
            print(f"Error processing {source}->{target}: {e}")
    
    return pd.DataFrame(results)
