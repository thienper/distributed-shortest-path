# Luồng Tổng Quan Hệ Thống - Distributed Shortest Path Prediction

## 📊 Kiến Trúc Tổng Quan

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HỆ THỐNG DỰ ĐOÁN ĐƯỜNG ĐI NGẮN NHẤT                      │
│            Sử dụng GraphSAGE + ML vs Thuật toán Dijkstra truyền thống      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 LUỒNG TRAINING (Chỉ chạy 1 lần)

```
┌──────────────────────┐
│   Khởi tạo Training  │
│   train_model.py     │
└──────────────┬───────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│  1. CHUẨN BỊ DỮ LIỆU (Data Generation)              │
│  ├─ Graph Loader                                     │
│  │  ├─ Load nodes.csv (danh sách đỉnh)            │
│  │  ├─ Load edges.csv (danh sách cạnh)            │
│  │  └─ Build adjacency matrix (ma trận kề)        │
│  │                                                  │
│  ├─ Generate Node Features                          │
│  │  ├─ Calculate node degree (bậc)                │
│  │  └─ Normalize features                           │
│  │                                                  │
│  └─ Generate Training Samples                       │
│     ├─ Random sample source-target pairs           │
│     ├─ Compute ground truth với Dijkstra           │
│     └─ Create training dataset (CSV file)          │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│  2. XÂY DỰNG MÔ HÌNH (Model Building)               │
│  ├─ GraphSAGE Model Architecture                     │
│  │  ├─ Input Layer: Node features                  │
│  │  ├─ GraphSAGE Layers (aggregation)              │
│  │  │  └─ Multi-layer neighborhood aggregation    │
│  │  ├─ Node Embedding Layer                         │
│  │  │  └─ Generate d-dimensional embeddings       │
│  │  └─ MLP Head (prediction)                        │
│  │     └─ Predict shortest path cost               │
│  │                                                  │
│  └─ Model Parameters                                │
│     ├─ Input dimension: 1 (degree)                 │
│     ├─ Embedding dimension: 32 (default)           │
│     ├─ Hidden dimension: 64 (default)              │
│     └─ Num layers: 2 (default)                     │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│  3. HỌC MÔ HÌNH (Model Training)                     │
│  ├─ DataLoader Setup                                │
│  │  ├─ Batch size: 32 (default)                    │
│  │  └─ Shuffle training data                        │
│  │                                                  │
│  ├─ Training Loop (50 epochs default)               │
│  │  ├─ Forward pass                                │
│  │  ├─ Compute loss (MSE)                          │
│  │  ├─ Backward pass                               │
│  │  └─ Optimize parameters                          │
│  │     └─ Optimizer: Adam (LR=0.001)               │
│  │                                                  │
│  └─ Validation & Early Stopping                     │
│     ├─ Track best model                             │
│     └─ Save best_model.pt                           │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│  4. LƯU KẾT QUẢ (Save Artifacts)                     │
│  └─ models/                                          │
│     ├─ best_model.pt (trained weights)              │
│     ├─ config.json (model hyperparameters)          │
│     ├─ node_embeddings.npy (embeddings)             │
│     └─ results.json (training metrics)              │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
        ✅ TRAINING DONE ✅
        Ready for Web App!
```

---

## 🌐 LUỒNG WEB APP (Chạy sau training)

```
┌──────────────────────────────────────────────────────┐
│   Khởi tạo Web App                                   │
│   python web_app/app.py                              │
│   http://localhost:5000                              │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│  Flask Backend Initialization (app.py)               │
│  ├─ Load trained model                               │
│  ├─ Load graph data (nodes, edges, adjacency)       │
│  ├─ Initialize services:                             │
│  │  ├─ PredictorService                             │
│  │  ├─ GraphLoader                                  │
│  │  └─ Dijkstra baseline                            │
│  └─ Setup routes & API endpoints                     │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│  WEB INTERFACE (Browser)                             │
│  ├─ Load index.html                                 │
│  ├─ Initialize Cytoscape.js graph visualization    │
│  ├─ Setup event listeners                           │
│  │  └─ Click on nodes to select source/target      │
│  └─ Display dashboard                               │
│     └─ Metrics panel (accuracy, latency, etc.)      │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│  USER INTERACTION - SELECT NODES                     │
│  ├─ User clicks source node                          │
│  │  └─ Highlight node, store source ID             │
│  │                                                  │
│  └─ User clicks target node                         │
│     └─ Highlight node, store target ID             │
│        Trigger prediction request                   │
└──────────────┬───────────────────────────────────────┘
               │
               ▼ (AJAX Request: /api/predict)
┌──────────────────────────────────────────────────────┐
│  PREDICTION SERVICE (PredictorService)               │
│  └─ Input: source, target node IDs                   │
│     │                                                 │
│     ├─────────────────────────────────────────────┐  │
│     │  PARALLEL EXECUTION                        │  │
│     ├─────────────────────────────────────────────┤  │
│     │                                              │  │
│     ├─ Algorithm 1: DIJKSTRA (Ground Truth)      │  │
│     │  ├─ Initialize distance array              │  │
│     │  ├─ Priority queue                         │  │
│     │  ├─ BFS/Dijkstra algorithm                │  │
│     │  ├─ Path: node sequence                    │  │
│     │  └─ Cost: total weight                     │  │
│     │                                              │  │
│     └─ Algorithm 2: ML MODEL (GraphSAGE)         │  │
│        ├─ Load node embeddings                   │  │
│        ├─ Extract source embedding               │  │
│        ├─ Extract target embedding               │  │
│        ├─ Forward pass: model(src_emb, tgt_emb) │  │
│        ├─ Predict cost                           │  │
│        └─ ML Path: use Dijkstra path + ML cost   │  │
│                                                    │  │
└─────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│  COMPUTE METRICS                                     │
│  ├─ Dijkstra Path & Cost                             │
│  ├─ ML Predicted Cost                                │
│  ├─ Error %: |ML_cost - Dijkstra_cost| / Dijkstra  │
│  ├─ Latency (ms)                                     │
│  └─ Accuracy Check: error < 10% = Correct           │
└──────────────┬───────────────────────────────────────┘
               │
               ▼ (Return JSON Response)
┌──────────────────────────────────────────────────────┐
│  RESPONSE FORMAT                                     │
│  {                                                   │
│    "source": int,                                   │
│    "target": int,                                   │
│    "dijkstra_path": [nodes...],                    │
│    "dijkstra_cost": float,                         │
│    "ml_path": [nodes...],                          │
│    "ml_cost": float,                               │
│    "error_percent": float,                         │
│    "is_accurate": bool,                            │
│    "latency_ms": float                             │
│  }                                                  │
└──────────────┬───────────────────────────────────────┘
               │
               ▼ (AJAX Response)
┌──────────────────────────────────────────────────────┐
│  FRONTEND UPDATE (app.js)                            │
│  ├─ Render Dijkstra path                             │
│  │  ├─ Highlight path with BLUE                    │
│  │  └─ Show distance                                │
│  │                                                  │
│  ├─ Render ML predicted path                        │
│  │  ├─ Highlight path with RED/GREEN              │
│  │  └─ Show predicted distance                      │
│  │                                                  │
│  ├─ Update Metrics Dashboard                        │
│  │  ├─ Error %                                     │
│  │  ├─ Latency (ms)                               │
│  │  ├─ Accuracy status                            │
│  │  └─ Comparison table                           │
│  │                                                  │
│  └─ Update Statistics                               │
│     ├─ Total predictions                            │
│     ├─ Correct predictions                          │
│     └─ Average accuracy                             │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
        ✅ DISPLAY RESULTS ✅
        (User sees paths & metrics)
        │
        ▼
   User can select another pair...
   (Loop back to "SELECT NODES")
```

---

## 🔗 GỘP LUỒNG TRAINING + WEB APP

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE WORKFLOW                                │
└─────────────────────────────────────────────────────────────────────────┘

    PHASE 1: PREPARATION (1 time only)
    ┌─────────────────────────────────────┐
    │ 1. Generate Graph Data              │
    │    └─ data/graphs/graph_medium/    │
    │       ├─ nodes.csv                 │
    │       └─ edges.csv                 │
    └────────────┬────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────┐
    │ 2. Run TRAINING                     │
    │    $ python train_model.py          │
    │    └─ Save: models/                 │
    │       ├─ best_model.pt              │
    │       ├─ config.json                │
    │       ├─ node_embeddings.npy        │
    │       └─ results.json               │
    └────────────┬────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────┐
    │ ✅ MODEL READY FOR DEPLOYMENT       │
    └────────────┬────────────────────────┘
                 │
                 ▼

    PHASE 2: SERVING (Interactive use)
    ┌─────────────────────────────────────┐
    │ 3. Start WEB APP                    │
    │    $ python web_app/app.py          │
    │    Open: http://localhost:5000      │
    └────────────┬────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────┐
    │ 4. Load Trained Model               │
    │    ├─ Load best_model.pt            │
    │    ├─ Load config.json              │
    │    └─ Load node_embeddings.npy      │
    └────────────┬────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────┐
    │ 5. Interactive Web Interface        │
    │    ├─ Display graph                 │
    │    ├─ Select source/target nodes   │
    │    ├─ Predict & compare             │
    │    └─ Show metrics                  │
    │    (Repeat steps 5-7)               │
    └────────────┬────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────┐
    │ 6. For each query:                  │
    │    - Run Dijkstra (ground truth)    │
    │    - Run ML model (prediction)      │
    │    - Compare results                │
    │    - Display path & metrics         │
    └────────────┬────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────┐
    │ 7. Accumulate Statistics            │
    │    ├─ Total predictions              │
    │    ├─ Correct predictions (< 10%)   │
    │    ├─ Average latency                │
    │    └─ Model accuracy trend           │
    └─────────────────────────────────────┘
```

---

## 📁 DATA FLOW DETAIL

```
┌─ DATA FLOW DURING TRAINING ─┐

Graph Data (CSV files)
    ├─ nodes.csv (node_id, features)
    └─ edges.csv (source, target, weight)
              │
              ▼
    PathDataset (generate training pairs)
    ├─ Sample random source-target pairs
    ├─ Compute ground truth with Dijkstra
    └─ Create training samples (src, tgt, cost)
              │
              ▼
    ShortestPathModel (GraphSAGE)
    ├─ Node features → embedding layer
    ├─ Adjacency matrix → graph structure
    └─ Training samples → optimize weights
              │
              ▼
    Checkpoint & Save
    ├─ best_model.pt (trained weights)
    ├─ config.json (architecture params)
    └─ node_embeddings.npy (learned embeddings)


┌─ DATA FLOW DURING SERVING ─┐

User Request (source, target)
              │
              ▼
    Dijkstra Branch        |    ML Model Branch
    ├─ Load adj matrix    |    ├─ Load trained model
    ├─ Run algorithm      |    ├─ Load embeddings
    ├─ Return path/cost   |    └─ Predict cost
              │            |    │
              ▼            ▼    ▼
         Compare Results
         ├─ Calculate error %
         ├─ Compute latency
         └─ Assess accuracy
              │
              ▼
         Return to Frontend
         └─ Display paths + metrics
```

---

## 🎯 KEY COMPONENTS

| Component | Role | File |
|-----------|------|------|
| **Training** | Load data, build model, optimize | `train_model.py` |
| **Data Loader** | Parse CSV, build graph structures | `src/data_generation/graph_loader.py` |
| **Model Trainer** | Dataset generation, training loop | `src/model/model_trainer.py` |
| **GraphSAGE Model** | Neural network for node embedding | `src/model/graphsage_model.py` |
| **Dijkstra** | Ground truth baseline algorithm | `src/predictor/dijkstra_baseline.py` |
| **Predictor Service** | Orchestrate prediction calls | `src/predictor/predictor_service.py` |
| **Flask App** | Web server & API endpoints | `web_app/app.py` |
| **Frontend** | UI, graph visualization, user input | `web_app/templates/index.html` `web_app/static/js/app.js` |

---

## 📊 METRICS TRACKED

**During Training:**
- Train loss (MSE)
- Validation loss
- Best model checkpoint

**During Serving:**
- Dijkstra path & cost
- ML predicted cost
- Error percentage
- Prediction latency (ms)
- Accuracy (correct if error < 10%)
- Cumulative statistics

---

## 🚀 QUICK COMMAND SUMMARY

```bash
# 1. Setup environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Train model (first time only)
python train_model.py --graph medium --epochs 50

# 3. Run web app (interactive serving)
python web_app/app.py

# 4. Open in browser
# http://localhost:5000
```

---

*Generated: Comprehensive system flow documentation*
*System: Distributed Shortest Path Prediction with GraphSAGE ML*
