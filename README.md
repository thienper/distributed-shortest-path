# Distributed Shortest Path Prediction with Machine Learning

A Big Data project using GraphSAGE to predict shortest paths in large graphs with ML instead of traditional Dijkstra algorithm.

## 🚀 Quick Start

**New users? Start here:** [SETUP.md](SETUP.md)

**Training & Testing:** [RUN_TRAINING.md](RUN_TRAINING.md)

## Features

- 🔴 **Graph Visualization**: Interactive graph with Cytoscape.js
- 🎯 **Node Selection**: Choose source and target nodes visually
- ⚡ **Path Finding**: Find shortest path with path highlighting
- 📊 **Real-time Dashboard**: Model accuracy, latency, throughput metrics
- 🤖 **ML Model**: GraphSAGE for node embeddings
- 📡 **Streaming**: Kafka + Spark for distributed processing
- 📈 **Comparison**: ML vs Traditional Dijkstra algorithm

## Project Structure

```
distributed-shortest-path/
├── requirements.txt
├── docker-compose.yml
├── data/                          # Data files
│   └── graphs/
│       ├── graph_small/           # Test graphs
│       ├── graph_medium/
│       └── graph_large/
├── src/
│   ├── __init__.py
│   ├── common/
│   │   ├── config.py              # Configuration
│   │   └── utils.py               # Utility functions
│   ├── data_generation/
│   │   ├── graph_generator.py     # Generate test graphs
│   │   └── graph_loader.py        # Load graph data
│   ├── producer/
│   │   ├── producer_service.py    # Kafka producer
│   │   └── graph_producer.py      # Graph data producer
│   ├── processor/
│   │   ├── processor_service.py   # Spark processor
│   │   └── feature_extractor.py   # Extract node features
│   ├── model/
│   │   ├── graphsage_model.py     # GraphSAGE implementation
│   │   └── model_trainer.py       # Model training
│   └── predictor/
│       ├── predictor_service.py   # Path prediction service
│       └── dijkstra_baseline.py   # Traditional algorithm
└── web_app/
    ├── app.py                     # Flask backend
    ├── config.json                # Web config
    ├── templates/
    │   └── index.html             # Main UI
    ├── static/
    │   ├── css/
    │   │   └── style.css          # Beautiful styling
    │   └── js/
    │       ├── app.js             # Frontend logic
    │       └── graph-viz.js       # Cytoscape visualization
    └── __pycache__/
```

## Quick Start

### 1. Setup Python Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### 2. Generate Test Data

```bash
python -m src.data_generation.graph_generator
```

### 3. Run Web App (Local)

```bash
python web_app/app.py
```

Open browser: `http://localhost:5000`

### 4. Run Full System with Docker

```bash
docker-compose up --build
```

## Usage

1. **Select Source Node**: Click on a node in the graph
2. **Select Target Node**: Click on another node
3. **Find Path**: Click "Find Shortest Path" button
4. **View Results**:
   - Path highlighted in graph
   - Path distance displayed
   - Comparison with Dijkstra algorithm

## Technologies

- **Backend**: Flask, Pandas, NumPy
- **Visualization**: Cytoscape.js
- **ML**: PyTorch, DGL, GraphSAGE
- **Streaming**: Kafka, Spark
- **Data**: NetworkX for graph generation
- **Frontend**: HTML5, CSS3, JavaScript

## Model Architecture

GraphSAGE (Graph SAmple and aggreGatE):
- Multi-layer GNN for node embeddings
- Neighborhood sampling for scalability
- Outputs node representations for path prediction

## Performance Metrics

- **Accuracy**: Compared to Dijkstra algorithm
- **Latency**: Path prediction time (ms)
- **Throughput**: Paths predicted per second
- **Model Loss**: Training progress

## Data Format

### nodes.csv
```csv
node_id,node_name,features
0,Node_A,"[0.5, 0.3, 0.7]"
1,Node_B,"[0.2, 0.8, 0.1]"
...
```

### edges.csv
```csv
source,target,weight,distance
0,1,1.5,100
0,2,2.3,200
...
```

### ground_truth.csv
```csv
source,target,shortest_path,path_cost,num_hops
0,4,"[0,2,4]",5.1,2
0,3,"[0,1,3]",2.7,2
...
```

## Author

Big Data Course Project

## License

MIT
