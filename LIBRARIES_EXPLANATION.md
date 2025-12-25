# 📚 Giải Thích Chi Tiết 7 Thư Viện - File Chính & Cách Dùng

---

## **1️⃣ Flask - Web Server Framework**

### 📌 File Chính:
**[`web_app/app.py`](web_app/app.py)**

### 🔧 Công Dụng:
Tạo web server và xử lý HTTP requests từ client

### 💻 Code Chi Tiết:

```python
# web_app/app.py - Dòng 8, 31, 40-47

from flask import Flask, render_template, jsonify, request

# Khởi tạo Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def index():
    """Trang chính - render file index.html"""
    return render_template("index.html")

@app.route("/api/graph")
def api_graph():
    """API endpoint - lấy dữ liệu đồ thị"""
    try:
        graph_data = {
            "nodes": [...],
            "edges": [...]
        }
        return jsonify(graph_data)  # Trả về JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=['POST'])
def api_predict():
    """API endpoint - dự đoán shortest path"""
    data = request.get_json()  # Lấy JSON từ request
    start = data['start']
    end = data['end']
    
    result = predict(start, end)
    return jsonify(result)  # Trả về JSON result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Khởi động server
    # Truy cập: http://localhost:5000
```

### 🎯 Ứng Dụng Cụ Thể:
- ✅ Phục vụ file HTML/CSS/JS từ `web_app/templates/` và `web_app/static/`
- ✅ Xử lý HTTP POST requests từ frontend
- ✅ Trả về JSON responses cho frontend
- ✅ Route handling: `/`, `/api/graph`, `/api/predict`

### 📊 Data Flow:
```
Browser (Client)
    ↓ (HTTP POST: /api/predict)
Flask Server (app.py)
    ↓ (Lấy start, end node)
Xử lý dự đoán
    ↓ (Trả về JSON: path, distance)
Browser hiển thị kết quả
```

---

## **2️⃣ Flask-CORS - CORS Support**

### 📌 File Chính:
**[`web_app/app.py`](web_app/app.py)** (Dòng 9)

### 🔧 Công Dụng:
Cho phép frontend gọi API từ domain khác

### 💻 Code Chi Tiết:

```python
# web_app/app.py - Dòng 9, 31

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Kích hoạt CORS cho tất cả endpoints

# Nếu không có CORS, khi frontend gọi API sẽ bị lỗi:
# "Access to XMLHttpRequest at 'http://localhost:5000/api/predict' 
#  from origin 'http://localhost:3000' has been blocked by CORS policy"
```

### 🎯 Ứng Dụng Cụ Thể:
- ✅ Frontend (port 5000) gọi API backend (port 5000) mà không bị block
- ✅ Cho phép cross-origin requests
- ✅ Không cần tùy chỉnh CORS headers thủ công

### 📊 Scenario:
```
Frontend (localhost:5000/index.html)
    ↓ fetch('/api/predict')
CORS Checker: "Có cho phép domain localhost:5000 không?"
    ↓
Flask-CORS: "Được, CORS cho phép"
    ↓
API trả về JSON
```

---

## **3️⃣ Pandas - Đọc/Xử Lý CSV**

### 📌 File Chính:
**[`src/data_generation/graph_loader.py`](src/data_generation/graph_loader.py)**

### 🔧 Công Dụng:
Load dữ liệu từ CSV files (nodes.csv, edges.csv)

### 💻 Code Chi Tiết:

```python
# src/data_generation/graph_loader.py - Dòng 1-50

import pandas as pd
import numpy as np
from pathlib import Path

class GraphLoader:
    """Load graph data from CSV files"""
    
    def __init__(self, graph_dir: Path):
        self.graph_dir = Path(graph_dir)
        self.nodes = None
        self.edges = None
        self.ground_truth = None
        self.load_data()
    
    def load_data(self):
        """Load tất cả CSV files"""
        nodes_file = self.graph_dir / "nodes.csv"
        edges_file = self.graph_dir / "edges.csv"
        gt_file = self.graph_dir / "ground_truth.csv"
        
        # 🔑 Pandas đọc CSV
        if nodes_file.exists():
            self.nodes = pd.read_csv(nodes_file)
            # Output: DataFrame
            #    node_id  degree  betweenness
            # 0        0       5          0.12
            # 1        1       8          0.34
            # ...
        
        if edges_file.exists():
            self.edges = pd.read_csv(edges_file)
            # Output: DataFrame
            #    source target  weight
            # 0        0      1     1.5
            # 1        0      2     2.3
            # ...
        
        if gt_file.exists():
            self.ground_truth = pd.read_csv(gt_file)
    
    def get_nodes(self) -> pd.DataFrame:
        """Trả về nodes DataFrame"""
        return self.nodes
    
    def get_edges(self) -> pd.DataFrame:
        """Trả về edges DataFrame"""
        return self.edges
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Convert edges to adjacency matrix"""
        if self.edges is None:
            return None
        
        num_nodes = len(self.nodes)
        adj_matrix = np.zeros((num_nodes, num_nodes))
        
        # 🔑 Pandas iterate rows
        for _, row in self.edges.iterrows():
            src = int(row['source'])
            tgt = int(row['target'])
            adj_matrix[src, tgt] = row['weight']
            adj_matrix[tgt, src] = row['weight']  # Undirected
        
        return adj_matrix
```

### 🎯 Ứng Dụng Cụ Thể:
- ✅ Đọc file `nodes.csv` (5000 rows)
- ✅ Đọc file `edges.csv` (14991 rows)
- ✅ Xử lý dữ liệu: filter, select columns, aggregate
- ✅ Iterate rows để xây dựng adjacency matrix

### 📊 Data Flow:
```
nodes.csv (5000 rows)
    ↓ pd.read_csv()
Pandas DataFrame (1000, 3) ← columns: node_id, degree, betweenness
    ↓ .iterrows()
Convert to NumPy array
```

---

## **4️⃣ NumPy - Tính Toán Số Học**

### 📌 File Chính:
**[`src/model/graphsage_model.py`](src/model/graphsage_model.py)**

### 🔧 Công Dụng:
Tính toán ma trận, vector operations, normalization

### 💻 Code Chi Tiết:

```python
# src/model/graphsage_model.py - Dòng 1-50

import torch
import torch.nn as nn
import numpy as np

class GraphSAGELayer(nn.Module):
    """Single GraphSAGE layer"""
    
    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (n_nodes, in_features) - node features
            adj_matrix: (n_nodes, n_nodes) - adjacency matrix
        Returns:
            embeddings: (n_nodes, out_features)
        """
        
        # 🔑 NumPy: Ma trận nhân (matrix multiplication)
        # adj_matrix.T @ features = neighbor aggregation
        neighbor_features = torch.matmul(adj_matrix.t(), features)
        # Shape: (1000, 1000) @ (1000, 32) = (1000, 32)
        
        # Ứng dụng NumPy:
        # - Chuẩn hóa dữ liệu
        # - Tính thống kê: mean, std, min, max
        # - Linear algebra: eigenvalues, eigenvectors
        
        neighbor_aggregated = self.agg_mlp(neighbor_features)
        self_transformed = self.self_mlp(features)
        combined = torch.cat([neighbor_aggregated, self_transformed], dim=1)
        embeddings = self.combine_mlp(combined)
        
        return embeddings
```

### 🎯 Ứng Dụng Cụ Thể:
- ✅ Tạo adjacency matrix (1000x1000)
- ✅ Chuẩn hóa node features: `(x - mean) / std`
- ✅ Ma trận nhân trong GraphSAGE aggregation
- ✅ Tính toán metrics: RMSE, MAPE
- ✅ Random sampling

### 📊 Ví Dụ:

```python
# src/model/model_trainer.py
import numpy as np

# Tạo adjacency matrix từ edges
adj_matrix = np.zeros((1000, 1000))
for src, tgt in edges:
    adj_matrix[src, tgt] = 1
# Shape: (1000, 1000) ← 1M phần tử

# Chuẩn hóa node degrees
degrees = np.array([adj[i].sum() for i in range(1000)])
degrees_normalized = (degrees - degrees.mean()) / (degrees.std() + 1e-6)
# Trước: [1, 2, 3, 4, 5, ...]
# Sau: [-1.2, -0.8, 0.1, 0.5, 0.9, ...] (mean=0, std=1)

# Tính RMSE
mape = np.mean(np.abs((all_true - all_pred) / all_true)) * 100
rmse = np.sqrt(np.mean((all_true - all_pred) ** 2))
print(f"RMSE: {rmse:.4f}")  # RMSE: 1.1740
```

---

## **5️⃣ NetworkX - Dijkstra Baseline**

### 📌 File Chính:
**[`src/data_generation/graph_generator.py`](src/data_generation/graph_generator.py)**

### 🔧 Công Dụng:
Tạo đồ thị test, Dijkstra, tính graph properties

### 💻 Code Chi Tiết:

```python
# src/data_generation/graph_generator.py - Dòng 1-50

import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path

class GraphGenerator:
    """Generate test graphs of different sizes"""
    
    @staticmethod
    def generate_graph(num_nodes: int, num_edges_multiplier: int = 3, 
                       graph_type: str = "barabasi", seed: int = 42):
        """Generate graph using different models"""
        np.random.seed(seed)
        
        # 🔑 NetworkX: Tạo đồ thị
        if graph_type == "barabasi":
            # Barabási-Albert model (scale-free)
            # Mỗi node mới attach vào 3 existing nodes
            G = nx.barabasi_albert_graph(num_nodes=5000, m=3, seed=42)
            # Output: Graph với 5000 nodes, ~14991 edges
        
        elif graph_type == "erdos":
            # Erdős-Rényi random graph
            p = (num_edges_multiplier * 2) / num_nodes
            G = nx.erdos_renyi_graph(num_nodes, p, seed=seed)
        
        else:  # watts
            # Watts-Strogatz small-world graph
            G = nx.watts_strogatz_graph(num_nodes, k=4, p=0.3, seed=seed)
        
        # Thêm weights vào edges
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
            nodes_data.append({'node_id': node})
        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_csv(output_dir / "nodes.csv", index=False)
        
        # Save edges
        edges_data = []
        for u, v, data in G.edges(data=True):
            edges_data.append({
                'source': u,
                'target': v,
                'weight': data.get('weight', 1.0)
            })
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv(output_dir / "edges.csv", index=False)
```

### 🎯 Ứng Dụng Cụ Thể:
- ✅ Tạo Barabási-Albert graph (5000 nodes, 14991 edges)
- ✅ Dijkstra shortest path (trong model_trainer.py)
- ✅ Tính graph properties: degree, diameter, connected components
- ✅ Visualize đồ thị

### 📊 Ví Dụ từ model_trainer.py:

```python
# src/model/model_trainer.py
import networkx as nx

# Load từ CSV files
edges_df = pd.read_csv('edges.csv')
G = nx.Graph()

for _, row in edges_df.iterrows():
    G.add_edge(int(row['source']), int(row['target']), 
               weight=row['weight'])

# 🔑 Dijkstra: Lấy ground truth
path = nx.shortest_path(G, source=0, target=100, weight='weight')
distance = nx.shortest_path_length(G, source=0, target=100, weight='weight')

print(f"Shortest path: {path}")  # [0, 5, 12, 100]
print(f"Distance: {distance}")    # 5.3
```

---

## **6️⃣ PyTorch - Deep Learning (GraphSAGE)**

### 📌 File Chính:
**[`src/model/model_trainer.py`](src/model/model_trainer.py)**

### 🔧 Công Dụng:
Xây dựng mô hình neural network, training loop, inference

### 💻 Code Chi Tiết:

```python
# src/model/model_trainer.py - Dòng 1-50

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ShortestPathTrainer:
    """Train GraphSAGE model for shortest path prediction"""
    
    def train(self, nodes_csv, edges_csv, config):
        """Training loop"""
        
        # 1. Load data
        nodes_df = pd.read_csv(nodes_csv)
        edges_df = pd.read_csv(edges_csv)
        
        # 2. Build model
        model = ShortestPathModel(
            input_dim=config['input_dim'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers']
        )
        
        # 3. Loss function
        loss_fn = nn.MSELoss()  # Mean Squared Error
        
        # 4. Optimizer
        optimizer = optim.Adam(model.parameters(), 
                              lr=config['learning_rate'],
                              weight_decay=config['weight_decay'])
        
        # 5. Training loop (50 epochs)
        for epoch in range(config['num_epochs']):
            total_loss = 0
            
            for batch_src, batch_tgt, batch_cost in train_loader:
                # Forward pass
                predicted_cost = model(node_features, adj_matrix, 
                                      batch_src, batch_tgt)
                
                # Compute loss
                loss = loss_fn(predicted_cost, batch_cost)
                total_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()  # Clear gradients
                loss.backward()         # Compute gradients
                optimizer.step()        # Update weights
            
            print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {total_loss:.4f}")
        
        # 6. Save model
        torch.save(model.state_dict(), 'models/best_model.pt')
        
        # 7. Inference (test)
        model.eval()
        with torch.no_grad():
            for batch_src, batch_tgt, batch_cost in test_loader:
                predictions = model(node_features, adj_matrix, 
                                   batch_src, batch_tgt)
                error = torch.abs(predictions - batch_cost).mean()
                print(f"Test Error: {error:.4f}")
```

### 🎯 Ứng Dụng Cụ Thể:
- ✅ Xây dựng GraphSAGE model (2 layers, 32→64→1)
- ✅ Training: forward → loss → backward → update
- ✅ Loss function: MSE (Mean Squared Error)
- ✅ Optimizer: Adam
- ✅ Scheduler: ReduceLROnPlateau (điều chỉnh learning rate)
- ✅ Save/load model weights
- ✅ Inference: dự đoán trên test data

### 📊 Training Flow:
```
Input: (node_features, adj_matrix, source, target)
    ↓
GraphSAGE Model
    ├─ GraphSAGELayer 1: (1, 32) → (32)
    ├─ GraphSAGELayer 2: (32) → (64)
    └─ FC Layer: concat(src_emb, tgt_emb) → (1)  ← predicted distance
    ↓
MSE Loss = (predicted - true)²
    ↓
Backward: compute gradients
    ↓
Adam Optimizer: update weights
    ↓
Epoch 1/50: Loss = 2.5
Epoch 2/50: Loss = 2.1
...
Epoch 50/50: Loss = 1.1
```

---

## **7️⃣ TQDM - Progress Bar**

### 📌 File Chính:
**[`src/model/model_trainer.py`](src/model/model_trainer.py)** (Dòng 10)

### 🔧 Công Dụng:
Hiển thị thanh tiến độ khi chạy các vòng lặp dài

### 💻 Code Chi Tiết:

```python
# src/model/model_trainer.py - Dòng 10, 120-140

from tqdm import tqdm

class ShortestPathTrainer:
    
    def generate_training_samples(self, num_samples):
        """Generate random path pairs with progress bar"""
        
        samples = []
        sources = np.random.randint(0, self.num_nodes, num_samples)
        targets = np.random.randint(0, self.num_nodes, num_samples)
        
        # 🔑 TQDM wrapper cho loop
        for src, tgt in tqdm(zip(sources, targets), 
                            total=num_samples,
                            desc="Generating samples"):
            path, cost = self.dijkstra.find_shortest_path(src, tgt)
            if path:
                samples.append({
                    'source': src,
                    'target': tgt,
                    'cost': cost
                })
        
        # Output:
        # Generating samples: 100%|██████████| 500/500 [00:00<00:00, 541.23it/s]
        
        return samples
    
    def train(self, ...):
        """Training loop with progress bar"""
        
        for epoch in tqdm(range(config['num_epochs']), 
                         desc="Training"):
            total_loss = 0
            
            # Inner loop - không hiển thị vì có leave=False
            for batch_src, batch_tgt, batch_cost in tqdm(train_loader, 
                                                         leave=False,
                                                         desc=f"Epoch {epoch+1}"):
                predicted = model(...)
                loss = loss_fn(predicted, batch_cost)
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
        
        # Output:
        # Training: 100%|██████████| 50/50 [05:30<00:00,  6.60s/it]
        #   Epoch 1: 100%|██████████| 13/13 [00:08<00:00,  1.58it/s]
        #   Epoch 2: 100%|██████████| 13/13 [00:08<00:00,  1.59it/s]
        #   ...
```

### 🎯 Ứng Dụng Cụ Thể:
- ✅ Hiển thị progress bar cho training (50 epochs)
- ✅ Hiển thị progress bar cho data generation (500 samples)
- ✅ Hiển thị tốc độ xử lý (it/s = iterations per second)
- ✅ Hiển thị thời gian còn lại (ETA)

### 📊 Output Ví Dụ:

```
Generating samples: 100%|████████████████████████████| 500/500 [00:00<00:00, 541.23it/s]
Training: 50%|██████░░░░░░░░░░░░░░| 25/50 [02:45<02:45,  6.60s/it]
```

**Giải thích:**
- `100%` = Tiến độ 100%
- `|████████████████████████████|` = Thanh tiến độ
- `500/500` = 500 trong tổng cộng 500
- `00:00<00:00` = Thời gian đã chạy < Thời gian còn lại
- `541.23it/s` = 541 iterations per second

---

## 📊 Bảng Tóm Tắt

| Thư Viện | File Chính | Công Dụng | Dòng Code Chính |
|---------|-----------|-----------|-----------|
| **Flask** | web_app/app.py | Web server, HTTP routing | dòng 8, 31-47 |
| **Flask-CORS** | web_app/app.py | CORS support | dòng 9, 31 |
| **Pandas** | src/data_generation/graph_loader.py | Đọc/xử lý CSV | dòng 1-50 |
| **NumPy** | src/model/graphsage_model.py | Ma trận, tính toán | dòng 3, 43 |
| **NetworkX** | src/data_generation/graph_generator.py | Tạo graph, Dijkstra | dòng 1-50 |
| **PyTorch** | src/model/model_trainer.py | Training, inference | dòng 1-50 |
| **TQDM** | src/model/model_trainer.py | Progress bar | dòng 10 |

---

## 🎯 Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│           Web App (Flask + Flask-CORS)              │
│           web_app/app.py                            │
│  ┌───────────────────────────────────────────────┐  │
│  │  GET /       → render index.html              │  │
│  │  GET /api/graph → return graph data (JSON)    │  │
│  │  POST /api/predict → predict shortest path    │  │
│  └───────────────────────────────────────────────┘  │
└──────────────────┬───────────────────────────────────┘
                   ↓
        ┌────────────────────────────┐
        │ Data Generation            │
        ├────────────────────────────┤
        │ graph_generator.py         │
        │ (NetworkX: tạo đồ thị)     │
        │                            │
        │ graph_loader.py            │
        │ (Pandas: đọc CSV)          │
        └───────────┬────────────────┘
                    ↓
        ┌────────────────────────────┐
        │ Model Training             │
        ├────────────────────────────┤
        │ model_trainer.py           │
        │ (PyTorch: train model)     │
        │ (NumPy: tính toán)         │
        │ (TQDM: progress bar)       │
        │                            │
        │ graphsage_model.py         │
        │ (PyTorch: define model)    │
        └───────────┬────────────────┘
                    ↓
        ┌────────────────────────────┐
        │ Prediction Service         │
        ├────────────────────────────┤
        │ predictor_service.py       │
        │ (Load trained model)       │
        │ (Predict using GraphSAGE)  │
        └────────────────────────────┘
```

Mỗi thư viện có vai trò riêng trong quy trình xây dựng và triển khai project! 🚀
