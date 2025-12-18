# 🚀 Setup Guide for New Users

Nếu bạn clone project này từ GitHub, hãy làm theo các bước dưới đây.

---

## 📋 Các Bước Setup (First-Time Setup)

### **Step 1: Clone Repository**
```bash
git clone <your-github-url>
cd distributed-shortest-path
```

### **Step 2: Tạo Python Virtual Environment**

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### **Step 3: Cài Đặt Dependencies**
```bash
pip install -r requirements.txt
```

⏱️ **Thời gian:** ~3-5 phút (tùy tốc độ internet)

### **Step 4: Kiểm Tra Data & Models**
```bash
# Kiểm tra dữ liệu đã có chưa
ls data/graphs/graph_large/

# Kiểm tra trained model
ls models/graph_large/
```

**Nếu không có model:**
```bash
# Train mô hình (lần đầu sẽ lâu ~5-10 phút)
python train_model.py

# Hoặc tùy chỉnh
python train_model.py --graph large --epochs 50
```

### **Step 5: Test Model**
```bash
python quick_test.py
```

**Output mong muốn:**
```
[RESULTS] Average error: 12.0%
[RESULTS] Accuracy (error<15%): 60%
[STATUS] Model is READY for deployment
```

### **Step 6: Chạy Web App**
```bash
python web_app/app.py
```

Truy cập: `http://localhost:5000`

---

## 📦 Dependencies (requirements.txt)

```
flask==3.0.0              # Web framework
torch==2.1.0              # Deep learning
pandas==2.2.2             # Data processing
numpy==1.24.3             # Numerical computing
networkx==3.1             # Graph algorithms (Dijkstra)
kafka-python==2.0.2       # Message queue (optional)
scikit-learn==1.3.2       # ML utilities
dgl==1.1.1                # Graph neural networks
scipy==1.11.4             # Scientific computing
matplotlib==3.8.2         # Visualization
```

---

## 🏗️ Project Structure

```
distributed-shortest-path/
├── train_model.py              ← Training script
├── quick_test.py               ← Testing script
├── requirements.txt            ← Dependencies
├── README.md                   ← Project overview
├── RUN_TRAINING.md             ← Detailed guide
│
├── data/
│   └── graphs/
│       ├── graph_small/        (100 nodes)
│       ├── graph_medium/       (1000 nodes)
│       └── graph_large/        (5000 nodes) ← DEFAULT
│
├── models/
│   └── graph_large/
│       ├── best_model.pt       ← Trained model
│       └── config.json         ← Model config
│
├── training_data/
│   └── graph_large/
│       ├── training_samples.csv
│       └── metadata.json
│
├── src/
│   ├── model/
│   │   ├── graphsage_model.py  ← GraphSAGE architecture
│   │   └── model_trainer.py    ← Training utilities
│   ├── predictor/
│   │   ├── dijkstra_baseline.py
│   │   └── predictor_service.py
│   └── common/
│       ├── config.py
│       └── parsing_utils.py
│
├── web_app/
│   ├── app.py                  ← Flask server
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/app.js
│   └── templates/
│       └── index.html
│
└── docker/
    ├── clusterer.Dockerfile
    └── producer.Dockerfile
```

---

## ⚡ Quick Commands

```bash
# 1. Setup (one-time)
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# 2. Train model (first time or to retrain)
python train_model.py

# 3. Test model
python quick_test.py

# 4. Run web app
python web_app/app.py

# 5. Open browser
# http://localhost:5000
```

---

## ❓ FAQ

### **Q: Cần cài Python gì?**
A: Python 3.9+ (khuyến nghị 3.10 hoặc 3.11)

### **Q: Cần GPU không?**
A: Không. Mặc định dùng CPU, model nhỏ (~2KB) nên nhanh.

### **Q: Model đã trained hay phải train lại?**
A: Đã có `models/graph_large/best_model.pt`, không cần train lại.
Nếu muốn train từ đầu: `python train_model.py`

### **Q: Training mất bao lâu?**
A: ~5-10 phút trên CPU (50 epochs, 1000 samples)

### **Q: Dữ liệu ở đâu?**
A: Đã có trong `data/graphs/graph_large/`
(5000 nodes, 14,991 edges - Barabási-Albert model)

### **Q: Web app port nào?**
A: Mặc định port 5000
Thay đổi: `python web_app/app.py --port 8080`

---

## 🔧 Troubleshooting

### **ImportError: No module named 'torch'**
```bash
pip install torch==2.1.0
```

### **CUDA/GPU errors**
Model chạy CPU by default, không cần GPU.
Nếu cần: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### **Port 5000 đã bị dùng**
```bash
# Kill process
lsof -i :5000 | grep LISTEN | awk '{print $2}' | xargs kill -9

# Hoặc chạy port khác
python web_app/app.py --port 8080
```

### **File data không tìm thấy**
Đảm bảo có folder: `data/graphs/graph_large/`
Nếu không: Hãy liên hệ bạn tôi để lấy data

---

## 📖 Hướng dẫn Chi Tiết

Xem file: [RUN_TRAINING.md](RUN_TRAINING.md)

---

**Status:** ✅ Ready to deploy!
