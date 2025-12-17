import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class GraphSAGELayer(nn.Module):
    """Single GraphSAGE layer"""
    
    def __init__(self, in_features: int, out_features: int, aggregator: str = "mean"):
        super(GraphSAGELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator
        
        # MLP for aggregation
        self.agg_mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        
        # MLP for self features
        self.self_mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )
        
        # Combine aggregated + self
        self.combine_mlp = nn.Sequential(
            nn.Linear(out_features * 2, out_features),
            nn.ReLU()
        )
    
    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (n_nodes, in_features)
            adj_matrix: (n_nodes, n_nodes) adjacency matrix
        Returns:
            embeddings: (n_nodes, out_features)
        """
        # Aggregate neighbor features
        # adj_matrix.T @ features: transpose needed for proper matrix multiplication
        # adj_matrix shape: (n_nodes, n_nodes), features shape: (n_nodes, in_features)
        neighbor_features = torch.matmul(adj_matrix.t(), features)  # (n, in_feat)
        
        # Aggregate (mean pooling)
        neighbor_aggregated = self.agg_mlp(neighbor_features)  # (n, out_feat)
        
        # Self transformation
        self_transformed = self.self_mlp(features)  # (n, out_feat)
        
        # Combine neighbor aggregation + self
        combined = torch.cat([neighbor_aggregated, self_transformed], dim=1)  # (n, 2*out_feat)
        embeddings = self.combine_mlp(combined)  # (n, out_feat)
        
        return embeddings


class GraphSAGE(nn.Module):
    """GraphSAGE model for node embeddings"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super(GraphSAGE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(GraphSAGELayer(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(GraphSAGELayer(hidden_dim, output_dim))
    
    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (n_nodes, input_dim) - node features
            adj_matrix: (n_nodes, n_nodes) - adjacency matrix
        Returns:
            embeddings: (n_nodes, output_dim) - node embeddings
        """
        x = features
        
        for layer in self.layers:
            x = layer(x, adj_matrix)
        
        return x


class PathPredictor(nn.Module):
    """Use node embeddings to predict path cost"""
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        super(PathPredictor, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # MLP to predict edge cost from node embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU()  # Cost should be positive
        )
    
    def forward(self, source_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        """
        Predict cost between source and target using embeddings
        
        Args:
            source_emb: (batch_size, embedding_dim)
            target_emb: (batch_size, embedding_dim)
        Returns:
            predicted_cost: (batch_size, 1)
        """
        combined = torch.cat([source_emb, target_emb], dim=1)
        predicted_cost = self.mlp(combined)
        return predicted_cost


class ShortestPathModel(nn.Module):
    """Complete model for shortest path prediction"""
    
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super(ShortestPathModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # GraphSAGE for node embeddings
        self.graph_encoder = GraphSAGE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_layers
        )
        
        # Cost predictor using embeddings
        self.path_predictor = PathPredictor(embedding_dim, hidden_dim)
    
    def encode(self, features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Get node embeddings"""
        return self.graph_encoder(features, adj_matrix)
    
    def predict_cost(self, source_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        """Predict cost between nodes using embeddings"""
        return self.path_predictor(source_emb, target_emb)
    
    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor, 
                source_indices: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (n_nodes, input_dim)
            adj_matrix: (n_nodes, n_nodes)
            source_indices: (batch_size,)
            target_indices: (batch_size,)
        Returns:
            predicted_costs: (batch_size, 1)
        """
        # Encode all nodes
        embeddings = self.encode(features, adj_matrix)  # (n_nodes, embedding_dim)
        
        # Get source and target embeddings
        source_emb = embeddings[source_indices]  # (batch_size, embedding_dim)
        target_emb = embeddings[target_indices]  # (batch_size, embedding_dim)
        
        # Predict costs
        predicted_costs = self.predict_cost(source_emb, target_emb)
        
        return predicted_costs
