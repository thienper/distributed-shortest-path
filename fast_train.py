#!/usr/bin/env python3
"""
Fast training script - optimized for CPU
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'predictor'))

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import json
import argparse
from graphsage_model import ShortestPathModel

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train GraphSAGE model')
    parser.add_argument('--graph', type=str, default='large', 
                       choices=['small', 'medium', 'large'],
                       help='Graph size (default: large)')
    parser.add_argument('--samples', type=int, default=200,
                       help='Number of training samples (default: 200)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs (default: 20)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"OPTIMIZED GRAPHSAGE TRAINING (CPU) - {args.graph.upper()}")
    print("="*70 + "\n")
    
    # 1. Load graph
    data_dir = Path(f'data/graphs/graph_{args.graph}')
    
    if not data_dir.exists():
        print(f"[ERROR] Graph directory not found: {data_dir}")
        print(f"Available: data/graphs/graph_small, graph_medium, graph_large")
        sys.exit(1)
    
    nodes_df = pd.read_csv(data_dir / 'nodes.csv')
    edges_df = pd.read_csv(data_dir / 'edges.csv')
    
    n_nodes = len(nodes_df)
    print(f"[GRAPH] {n_nodes} nodes, {len(edges_df)} edges")
    
    # 2. Build adjacency matrix
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    
    for _, row in edges_df.iterrows():
        src, dst, weight = int(row['source']), int(row['target']), float(row['weight'])
        if src < n_nodes and dst < n_nodes:
            adj[src, dst] = weight
            adj[dst, src] = weight
            G.add_edge(src, dst, weight=weight)
    
    # 3. Generate training samples using Dijkstra (NetworkX)
    print(f"\n[GEN] Generating training samples...")
    n_samples = args.samples
    sources, targets, costs = [], [], []
    
    np.random.seed(42)
    attempts = 0
    max_attempts = n_samples * 3
    
    while len(sources) < n_samples and attempts < max_attempts:
        attempts += 1
        src = np.random.randint(0, n_nodes)
        tgt = np.random.randint(0, n_nodes)
        
        if src == tgt:
            continue
        
        try:
            if nx.has_path(G, src, tgt):
                cost = nx.shortest_path_length(G, src, tgt, weight='weight')
                sources.append(src)
                targets.append(tgt)
                costs.append(cost)
        except:
            pass
    
    sources = np.array(sources)
    targets = np.array(targets)
    costs = np.array(costs)
    
    print(f"[OK] Generated {len(sources)} samples")
    print(f"     Cost stats: Mean={costs.mean():.2f}, Std={costs.std():.2f}, Min={costs.min():.0f}, Max={costs.max():.0f}")
    
    # Save training data for reproducibility
    training_data_dir = Path(f'training_data/graph_{args.graph}')
    training_data_dir.mkdir(parents=True, exist_ok=True)
    
    train_df = pd.DataFrame({
        'source': sources,
        'target': targets,
        'cost': costs
    })
    train_df.to_csv(training_data_dir / 'training_samples.csv', index=False)
    print(f"\n[OK] Training data saved to: {training_data_dir / 'training_samples.csv'}")
    
    # Also save metadata
    metadata = {
        'graph_size': args.graph,
        'num_samples': len(sources),
        'seed': 42,
        'data_dir': str(data_dir),
        'edge_count': len(edges_df),
        'node_count': n_nodes
    }
    with open(training_data_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Metadata saved to: {training_data_dir / 'metadata.json'}")
    
    # 4. Create node features
    degrees = np.array([adj[i].sum() for i in range(n_nodes)], dtype=np.float32)
    degrees = (degrees - degrees.mean()) / (degrees.std() + 1e-6)
    node_features = degrees.reshape(-1, 1)
    
    # 5. Convert to tensors
    adj_tensor = torch.from_numpy(adj).float()
    features_tensor = torch.from_numpy(node_features).float()
    src_tensor = torch.from_numpy(sources).long()
    tgt_tensor = torch.from_numpy(targets).long()
    cost_tensor = torch.from_numpy(costs).float().unsqueeze(1)
    
    # 6. Split data
    n_train = int(len(sources) * 0.8)
    train_indices = np.arange(n_train)
    test_indices = np.arange(n_train, len(sources))
    
    X_train_src = src_tensor[train_indices]
    X_train_tgt = tgt_tensor[train_indices]
    y_train = cost_tensor[train_indices]
    
    X_test_src = src_tensor[test_indices]
    X_test_tgt = tgt_tensor[test_indices]
    y_test = cost_tensor[test_indices]
    
    print(f"\n[SPLIT] Dataset split:")
    print(f"        Train: {len(train_indices)}, Test: {len(test_indices)}")
    
    # 7. Create model - Ultra-lightweight for fast training
    model = ShortestPathModel(input_dim=1, embedding_dim=8, hidden_dim=16, num_layers=2)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n[MODEL] {param_count} parameters (ultra-fast)")
    
    # 8. Training
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    batch_size = 16
    num_epochs = args.epochs
    
    print(f"\n[TRAIN] Training ({num_epochs} epochs, batch_size={batch_size})... (~2-5 min)")
    print(f"\nEpoch  | Train Loss | Test Loss | Test RMSE | Test MAPE")
    print(f"-------+-----------+-----------+-----------+----------")
    
    best_test_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(num_epochs):
        # Shuffle training data
        perm = torch.randperm(len(train_indices))
        
        # Train
        model.train()
        train_loss = 0.0
        for i in range(0, len(train_indices), batch_size):
            batch_indices = perm[i:i+batch_size]
            
            src_batch = X_train_src[batch_indices]
            tgt_batch = X_train_tgt[batch_indices]
            cost_batch = y_train[batch_indices]
            
            optimizer.zero_grad()
            pred = model(features_tensor, adj_tensor, src_batch, tgt_batch)
            loss = loss_fn(pred, cost_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(batch_indices)
        
        train_loss /= len(train_indices)
        
        # Test
        model.eval()
        with torch.no_grad():
            test_pred = model(features_tensor, adj_tensor, X_test_src, X_test_tgt)
            test_loss = loss_fn(test_pred, y_test).item()
            test_rmse = np.sqrt(test_loss)
            test_mape = (torch.abs(test_pred - y_test).sum() / y_test.sum() * 100).item()
        
        # Track best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            # Save model in graph-specific directory
            model_dir = Path(f'models/graph_{args.graph}')
            model_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save(model.state_dict(), model_dir / 'best_model.pt')
            
            # Save config for model loading
            config_data = {
                'input_dim': 1,
                'embedding_dim': 8,
                'hidden_dim': 16,
                'num_layers': 2,
                'graph_size': args.graph,
                'training_samples': len(sources),
                'training_epochs': epoch + 1
            }
            with open(model_dir / 'config.json', 'w') as f:
                json.dump(config_data, f, indent=2)
        
        # Print
        if (epoch + 1) % max(1, num_epochs // 10) == 0 or epoch < 5:
            print(f"{epoch+1:5d}  | {train_loss:9.4f} | {test_loss:9.4f} | {test_rmse:9.4f} | {test_mape:7.1f}%")
        
        scheduler.step()
    
    print(f"\n[SUCCESS] Training completed!")
    print(f"          Best epoch: {best_epoch+1}")
    print(f"          Best test loss: {best_test_loss:.4f}")
    print(f"          Model saved to: models/graph_{args.graph}/best_model.pt")
    print(f"          Config saved to: models/graph_{args.graph}/config.json")
    print(f"          Training data: training_data/graph_{args.graph}/training_samples.csv\n")

if __name__ == '__main__':
    main()
