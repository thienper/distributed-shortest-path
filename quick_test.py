#!/usr/bin/env python3
"""Quick test after training"""

import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'model'))
from graphsage_model import ShortestPathModel

print("\n" + "="*60)
print("TEST TRAINED MODEL")
print("="*60)

# Load training data
try:
    df = pd.read_csv('training_data/graph_medium/training_samples.csv')
    print(f"\n[DATA] Loaded {len(df)} training samples")
except:
    print("[ERROR] No training data found!")
    sys.exit(1)

# Load graph
data_dir = Path('data/graphs/graph_medium')
nodes_df = pd.read_csv(data_dir / 'nodes.csv')
edges_df = pd.read_csv(data_dir / 'edges.csv')

n_nodes = len(nodes_df)
print(f"[GRAPH] {n_nodes} nodes, {len(edges_df)} edges")

# Build adjacency matrix
adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
for _, row in edges_df.iterrows():
    src, dst, weight = int(row['source']), int(row['target']), float(row['weight'])
    if src < n_nodes and dst < n_nodes:
        adj[src, dst] = weight
        adj[dst, src] = weight

# Load model config
config_path = Path('models/graph_medium/config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Load model
model = ShortestPathModel(
    input_dim=config['input_dim'],
    embedding_dim=config['embedding_dim'],
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers']
)
model.load_state_dict(torch.load('models/graph_medium/best_model.pt', map_location='cpu'))
model.eval()

param_count = sum(p.numel() for p in model.parameters())
epochs = config.get('training_epochs', 'N/A')
print(f"[MODEL] {param_count} parameters, trained {epochs} epochs")

# Node features
degrees = np.array([adj[i].sum() for i in range(n_nodes)], dtype=np.float32)
degrees = (degrees - degrees.mean()) / (degrees.std() + 1e-6)
features = torch.from_numpy(degrees.reshape(-1, 1)).float()
adj_tensor = torch.from_numpy(adj).float()

# Test
print(f"\n[TEST] 10 random samples\n")
print(f"{'Source':<8} {'Target':<8} {'True':<8} {'Pred':<8} {'Error %':<10} {'Status':<10}")
print("="*60)

test_indices = np.random.choice(len(df), min(10, len(df)), replace=False)
total_error = 0
correct = 0

with torch.no_grad():
    for idx in test_indices:
        src = int(df.iloc[idx]['source'])
        tgt = int(df.iloc[idx]['target'])
        ground_truth = float(df.iloc[idx]['cost'])
        
        src_tensor = torch.tensor([src]).long()
        tgt_tensor = torch.tensor([tgt]).long()
        pred = model(features, adj_tensor, src_tensor, tgt_tensor)
        pred_val = pred.item()
        
        error_pct = abs(pred_val - ground_truth) / max(ground_truth, 1e-6) * 100
        is_correct = "OK" if error_pct < 15 else "X"
        
        if error_pct < 15:
            correct += 1
        total_error += error_pct
        
        print(f"{src:<8} {tgt:<8} {ground_truth:<8.2f} {pred_val:<8.2f} {error_pct:<10.1f} {is_correct:<10}")

avg_error = total_error / 10
acc = correct / 10 * 100

print("="*60)
print(f"\n[RESULTS] Average error: {avg_error:.1f}%")
print(f"[RESULTS] Accuracy (error<15%): {acc:.0f}%")
print(f"[STATUS] Model is {'READY' if acc > 50 else 'TRAINING'} for deployment\n")
