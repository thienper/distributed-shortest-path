# GraphSAGE Training & Testing Guide

## 🚀 TRAIN (Đầu tiên - Chỉ chạy 1 lần)

```bash
# Train mặc định trên graph_large, 50 epochs
python train_model.py

# Hoặc tùy chỉnh
python train_model.py --graph large --epochs 100 --batch-size 32 --learning-rate 0.001
```

**Kết quả:**
- Model lưu tại: `models/graph_large/best_model.pt`
- Config lưu tại: `models/graph_large/config.json`
- Training history: `models/results.json`

---

## ✅ TEST (Sau khi train xong)

```bash
# Cách 1: Tạo script test đơn giản
cat > quick_test.py << 'EOF'
import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'model'))

from graphsage_model import ShortestPathModel

# Load training data
df = pd.read_csv('training_data/graph_large/training_samples.csv')

# Load graph
data_dir = Path('data/graphs/graph_large')
nodes_df = pd.read_csv(data_dir / 'nodes.csv')
edges_df = pd.read_csv(data_dir / 'edges.csv')

n_nodes = len(nodes_df)
adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
for _, row in edges_df.iterrows():
    src, dst, weight = int(row['source']), int(row['target']), float(row['weight'])
    if src < n_nodes and dst < n_nodes:
        adj[src, dst] = weight
        adj[dst, src] = weight

# Load model
config_path = Path('models/graph_large/config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

model = ShortestPathModel(
    input_dim=config['input_dim'],
    embedding_dim=config['embedding_dim'],
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers']
)
model.load_state_dict(torch.load('models/graph_large/best_model.pt', map_location='cpu'))
model.eval()

# Node features
degrees = np.array([adj[i].sum() for i in range(n_nodes)], dtype=np.float32)
degrees = (degrees - degrees.mean()) / (degrees.std() + 1e-6)
features = torch.from_numpy(degrees.reshape(-1, 1)).float()
adj_tensor = torch.from_numpy(adj).float()

# Test 10 samples
print(f"\nTesting 10 random samples from {len(df)} training data\n")
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
print(f"\nAverage error: {avg_error:.1f}%")
print(f"Accuracy (error<15%): {acc:.0f}%\n")
EOF

# Chạy test
python quick_test.py
```

---

## 📊 Luồng xử lý

```
1. TRAINING (train_model.py)
   ↓
   Load graph → Generate Dijkstra labels → Train model
   ↓
   Save: best_model.pt + config.json + results.json

2. INFERENCE (Web App)
   ↓
   Load model từ best_model.pt + config.json
   ↓
   User request (src, tgt) → Forward pass → Dự đoán cost
   
⚠️  KHÔNG dùng results.json cho predictions!
    results.json chỉ là HISTORY của training
```

---

## 🔧 Cấu trúc file sau khi train

```
models/
├── graph_large/
│   ├── best_model.pt          ← Model weights
│   └── config.json             ← Model config
├── results.json                ← Training history (tham khảo)
└── [other old folders]

training_data/
└── graph_large/
    ├── training_samples.csv    ← Dijkstra labels
    └── metadata.json
```

---

## ⚡ Quick Commands

```bash
# Train graph_large, 50 epochs
.venv\Scripts\python.exe train_model.py

# Train graph_large, 100 epochs
.venv\Scripts\python.exe train_model.py --epochs 100

# Train other sizes
.venv\Scripts\python.exe train_model.py --graph medium --epochs 50
.venv\Scripts\python.exe train_model.py --graph small --epochs 20

# Check training results
cat models/results.json | python -m json.tool

# Launch web app (sau khi train)
".venv\Scripts\python.exe" -m web_app.app
```

---

**Status**: Model trained ✅ Ready to deploy 🚀
