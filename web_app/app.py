import sys
from pathlib import Path

# Add project root to path to resolve imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import pandas as pd
import numpy as np
from src.predictor.predictor_service import get_predictor_service, init_predictor_service
from src.data_generation.graph_loader import get_default_graph_loader
from src.common import config

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.json_encoder = NumpyEncoder
CORS(app)

# Initialize services
try:
    init_predictor_service()
    graph_loader = get_default_graph_loader()
except Exception as e:
    print(f"[WARNING] Could not initialize services: {e}")
    graph_loader = None

@app.route("/")
def index():
    """Main page"""
    return render_template("index.html")

@app.route("/api/graph")
def api_graph():
    """Get graph data for visualization"""
    try:
        if graph_loader is None:
            return jsonify({"error": "Graph not loaded"}), 500
        
        nodes_df = graph_loader.get_nodes()
        edges_df = graph_loader.get_edges()
        
        if nodes_df is None or edges_df is None:
            return jsonify({"error": "Graph data not found"}), 404
        
        # Convert to Cytoscape format
        nodes = []
        for _, row in nodes_df.iterrows():
            nodes.append({
                "data": {
                    "id": str(int(row["node_id"])),
                    "label": row["node_name"]
                },
                "position": {
                    "x": int(row["node_id"]) % 100 * 50,
                    "y": (int(row["node_id"]) // 100) * 50
                }
            })
        
        edges = []
        for _, row in edges_df.iterrows():
            edges.append({
                "data": {
                    "id": f"{int(row['source'])}-{int(row['target'])}",
                    "source": str(int(row["source"])),
                    "target": str(int(row["target"])),
                    "weight": float(row["weight"])
                }
            })
        
        return jsonify({
            "nodes": nodes,
            "edges": edges,
            "num_nodes": len(nodes),
            "num_edges": len(edges)
        })
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict shortest path"""
    try:
        data = request.json
        source = int(data.get("source"))
        target = int(data.get("target"))
        
        predictor = get_predictor_service()
        result = predictor.predict(source, target)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/metrics")
def api_metrics():
    """Get prediction metrics"""
    try:
        predictor = get_predictor_service()
        metrics = predictor.get_metrics()
        
        return jsonify(metrics)
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/summary")
def api_summary():
    """Get summary statistics"""
    try:
        if graph_loader is None:
            return jsonify({"error": "Graph not loaded"}), 500
        
        nodes_df = graph_loader.get_nodes()
        edges_df = graph_loader.get_edges()
        gt_df = graph_loader.get_ground_truth()
        
        predictor = get_predictor_service()
        metrics = predictor.get_metrics()
        
        return jsonify({
            "graph_stats": {
                "num_nodes": len(nodes_df) if nodes_df is not None else 0,
                "num_edges": len(edges_df) if edges_df is not None else 0,
                "num_test_cases": len(gt_df) if gt_df is not None else 0,
            },
            "model_performance": {
                "total_predictions": metrics["total_predictions"],
                "accuracy": metrics["accuracy"],
                "avg_latency_ms": metrics["avg_latency_ms"]
            }
        })
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/samples")
def api_samples():
    """Get sample predictions"""
    try:
        predictor = get_predictor_service()
        predictions = predictor.metrics["predictions"][-20:]  # Last 20
        
        return jsonify({
            "samples": predictions
        })
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
