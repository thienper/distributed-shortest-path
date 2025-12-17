import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import json
from tqdm import tqdm
from typing import Tuple, Dict
import logging
import sys
import os

# Add paths for imports
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_path)
sys.path.insert(0, os.path.join(src_path, 'model'))
sys.path.insert(0, os.path.join(src_path, 'predictor'))

from graphsage_model import ShortestPathModel
from dijkstra_baseline import DijkstraBaseline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PathDataset:
    """Generate training data for path prediction"""
    
    def __init__(self, nodes_csv: str, edges_csv: str, num_samples: int = 1000, seed: int = 42):
        """
        Args:
            nodes_csv: path to nodes.csv
            edges_csv: path to edges.csv
            num_samples: number of random paths to sample
            seed: random seed
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load graph data
        self.nodes = pd.read_csv(nodes_csv)
        self.edges = pd.read_csv(edges_csv)
        
        self.num_nodes = len(self.nodes)
        self.num_samples = num_samples
        
        # Build adjacency matrix
        self.adjacency_matrix = self._build_adjacency_matrix()
        
        # Also build NetworkX graph for faster shortest path (used in training)
        self.nx_graph = self._build_networkx_graph()
        
        logger.info(f"Loaded graph with {self.num_nodes} nodes and {len(self.edges)} edges")
    
    def _build_networkx_graph(self) -> nx.Graph:
        """Build NetworkX graph for efficient shortest path"""
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))
        
        for _, row in self.edges.iterrows():
            src, dst, weight = int(row['source']), int(row['target']), float(row['weight'])
            if src < self.num_nodes and dst < self.num_nodes:
                G.add_edge(src, dst, weight=weight)
        
        return G
    
    def _build_adjacency_matrix(self) -> np.ndarray:
        """Build adjacency matrix from edges"""
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        
        for _, row in self.edges.iterrows():
            src, dst, weight = int(row['source']), int(row['target']), float(row['weight'])
            if src < self.num_nodes and dst < self.num_nodes:
                adj[src, dst] = weight
                adj[dst, src] = weight  # Undirected graph
        
        return adj
    
    def generate_samples(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate training samples: (source, target, ground_truth_cost)
        Uses NetworkX for fast shortest path computation
        
        Returns:
            sources: (num_samples,)
            targets: (num_samples,)
            costs: (num_samples,) - ground truth shortest path costs
        """
        sources = []
        targets = []
        costs = []
        
        logger.info(f"Generating {self.num_samples} training samples...")
        
        attempts = 0
        max_attempts = self.num_samples * 5
        
        with tqdm(total=self.num_samples, desc="Generating samples") as pbar:
            while len(sources) < self.num_samples and attempts < max_attempts:
                attempts += 1
                
                # Random source and target
                source = np.random.randint(0, self.num_nodes)
                target = np.random.randint(0, self.num_nodes)
                
                if source == target:
                    continue
                
                # Try NetworkX shortest path (much faster)
                try:
                    if nx.has_path(self.nx_graph, source, target):
                        cost = nx.shortest_path_length(self.nx_graph, source, target, weight='weight')
                        sources.append(source)
                        targets.append(target)
                        costs.append(cost)
                        pbar.update(1)
                except:
                    pass
        
        sources = np.array(sources)
        targets = np.array(targets)
        costs = np.array(costs)
        
        logger.info(f"Generated {len(sources)} valid training samples")
        logger.info(f"Cost stats - Mean: {costs.mean():.2f}, Std: {costs.std():.2f}, "
                   f"Min: {costs.min():.0f}, Max: {costs.max():.0f}")
        
        return sources, targets, costs
    
    def create_node_features(self, feature_type: str = "degree") -> np.ndarray:
        """
        Create node features
        
        Args:
            feature_type: "degree", "identity", or "combined"
        Returns:
            features: (num_nodes, feature_dim)
        """
        if feature_type == "degree":
            # Degree centrality
            degrees = np.array([self.adjacency_matrix[i].sum() 
                              for i in range(self.num_nodes)])
            # Normalize
            degrees = (degrees - degrees.mean()) / (degrees.std() + 1e-6)
            features = degrees.reshape(-1, 1).astype(np.float32)
            
        elif feature_type == "identity":
            # One-hot identity
            features = np.eye(self.num_nodes, dtype=np.float32)
            
        elif feature_type == "combined":
            # Degree + normalized node ID
            degrees = np.array([self.adjacency_matrix[i].sum() 
                              for i in range(self.num_nodes)])
            degrees = (degrees - degrees.mean()) / (degrees.std() + 1e-6)
            node_ids = np.arange(self.num_nodes) / self.num_nodes
            features = np.column_stack([degrees, node_ids]).astype(np.float32)
        
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        logger.info(f"Created node features with shape {features.shape}")
        return features


class ShortestPathTrainer:
    """Train GraphSAGE model for shortest path prediction"""
    
    def __init__(self, model_dir: str = "models", device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
    
    def train(self, nodes_csv: str, edges_csv: str, config: Dict = None) -> Dict:
        """
        Train GraphSAGE model
        
        Args:
            nodes_csv: path to nodes CSV
            edges_csv: path to edges CSV
            config: training configuration
        Returns:
            training results dictionary
        """
        # Default config
        if config is None:
            config = {
                'num_samples': 500,  # Number of training samples
                'input_dim': 1,  # Input feature dimension (degree)
                'embedding_dim': 32,  # Node embedding dimension
                'hidden_dim': 64,  # MLP hidden dimension
                'num_layers': 2,  # GraphSAGE layers
                'batch_size': 32,
                'num_epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'feature_type': 'degree',  # degree, identity, combined
                'train_test_split': 0.8,
            }
        
        logger.info(f"Training config: {json.dumps(config, indent=2)}")
        
        # Generate dataset
        dataset = PathDataset(nodes_csv, edges_csv, num_samples=config['num_samples'])
        sources, targets, costs = dataset.generate_samples()
        node_features = dataset.create_node_features(config['feature_type'])
        
        # Convert to tensors
        adjacency_matrix = torch.from_numpy(dataset.adjacency_matrix).float().to(self.device)
        node_features_tensor = torch.from_numpy(node_features).float().to(self.device)
        sources_tensor = torch.from_numpy(sources).long()
        targets_tensor = torch.from_numpy(targets).long()
        costs_tensor = torch.from_numpy(costs).float().unsqueeze(1)  # (n_samples, 1)
        
        logger.info(f"Adjacency matrix shape: {adjacency_matrix.shape}")
        logger.info(f"Node features shape: {node_features_tensor.shape}")
        
        # Split train/test
        n_train = int(len(sources) * config['train_test_split'])
        train_indices = np.arange(n_train)
        test_indices = np.arange(n_train, len(sources))
        
        # Create dataloaders
        train_dataset = TensorDataset(
            sources_tensor[train_indices],
            targets_tensor[train_indices],
            costs_tensor[train_indices]
        )
        test_dataset = TensorDataset(
            sources_tensor[test_indices],
            targets_tensor[test_indices],
            costs_tensor[test_indices]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        
        # Create model - use actual node feature dimension
        actual_input_dim = node_features_tensor.shape[1]
        model = ShortestPathModel(
            input_dim=actual_input_dim,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers']
        ).to(self.device)
        
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                             weight_decay=config['weight_decay'])
        loss_fn = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                         patience=5, verbose=True)
        
        # Training loop
        best_test_loss = float('inf')
        train_losses = []
        test_losses = []
        
        logger.info("Starting training...")
        
        for epoch in range(config['num_epochs']):
            # Train
            model.train()
            train_loss = 0.0
            
            for batch_src, batch_tgt, batch_cost in train_loader:
                batch_src = batch_src.to(self.device)
                batch_tgt = batch_tgt.to(self.device)
                batch_cost = batch_cost.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass - parameter order: (features, adj_matrix, sources, targets)
                predicted_cost = model(node_features_tensor, adjacency_matrix, batch_src, batch_tgt)
                loss = loss_fn(predicted_cost, batch_cost)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * len(batch_src)
            
            train_loss /= len(train_dataset)
            train_losses.append(train_loss)
            
            # Test
            model.eval()
            test_loss = 0.0
            
            with torch.no_grad():
                for batch_src, batch_tgt, batch_cost in test_loader:
                    batch_src = batch_src.to(self.device)
                    batch_tgt = batch_tgt.to(self.device)
                    batch_cost = batch_cost.to(self.device)
                    
                    predicted_cost = model(node_features_tensor, adjacency_matrix, batch_src, batch_tgt)
                    loss = loss_fn(predicted_cost, batch_cost)
                    test_loss += loss.item() * len(batch_src)
            
            test_loss /= len(test_dataset)
            test_losses.append(test_loss)
            
            scheduler.step(test_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{config['num_epochs']} - "
                           f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            
            # Save best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), self.model_dir / "best_model.pt")
        
        # Load best model
        model.load_state_dict(torch.load(self.model_dir / "best_model.pt"))
        
        # Save config
        with open(self.model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"\nTraining completed!")
        logger.info(f"Best test loss: {best_test_loss:.4f}")
        logger.info(f"Model saved to {self.model_dir}")
        
        # Calculate test metrics
        model.eval()
        with torch.no_grad():
            all_pred = []
            all_true = []
            
            for batch_src, batch_tgt, batch_cost in test_loader:
                batch_src = batch_src.to(self.device)
                batch_tgt = batch_tgt.to(self.device)
                
                predicted_cost = model(node_features_tensor, adjacency_matrix, batch_src, batch_tgt)
                all_pred.extend(predicted_cost.cpu().numpy().flatten())
                all_true.extend(batch_cost.numpy().flatten())
            
            all_pred = np.array(all_pred)
            all_true = np.array(all_true)
            
            mape = np.mean(np.abs((all_true - all_pred) / all_true)) * 100
            rmse = np.sqrt(np.mean((all_true - all_pred) ** 2))
            
            results = {
                'best_train_loss': float(train_losses[-1]),
                'best_test_loss': float(best_test_loss),
                'test_rmse': float(rmse),
                'test_mape': float(mape),
                'train_losses': [float(x) for x in train_losses],
                'test_losses': [float(x) for x in test_losses],
                'config': config
            }
            
            logger.info(f"Test RMSE: {rmse:.4f}")
            logger.info(f"Test MAPE: {mape:.2f}%")
            
            # Save results
            with open(self.model_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            return results


if __name__ == "__main__":
    # Example usage
    trainer = ShortestPathTrainer(device="cpu")  # Use "cuda" if GPU available
    
    config = {
        'num_samples': 500,
        'input_dim': 1,
        'embedding_dim': 32,
        'hidden_dim': 64,
        'num_layers': 2,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'feature_type': 'degree',
        'train_test_split': 0.8,
    }
    
    # Train on medium graph (1000 nodes)
    results = trainer.train(
        nodes_csv="../../data/graphs/medium_graph_nodes.csv",
        edges_csv="../../data/graphs/medium_graph_edges.csv",
        config=config
    )
    
    print("\nTraining Results:")
    print(f"Test RMSE: {results['test_rmse']:.4f}")
    print(f"Test MAPE: {results['test_mape']:.2f}%")
