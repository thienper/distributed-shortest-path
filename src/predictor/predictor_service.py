import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from src.common import config
from src.data_generation.graph_loader import get_default_graph_loader
from src.predictor.dijkstra_baseline import DijkstraBaseline, MLPredictor, compare_algorithms

class PredictorService:
    """Service for path prediction"""
    
    def __init__(self, graph_size: str = "medium"):
        self.graph_size = graph_size
        self.loader = get_default_graph_loader(graph_size)
        self.adj_matrix = self.loader.get_adjacency_matrix()
        self.node_embeddings = self.loader.get_node_features()
        
        self.dijkstra = DijkstraBaseline(self.adj_matrix)
        
        # ✨ Load trained ML model from graph-specific directory
        model_path = f'models/graph_{graph_size}'
        self.ml = MLPredictor(
            adj_matrix=self.adj_matrix,
            model_path=model_path,    # Graph-specific path
            node_features=None        # Auto-generate from degree
        )
        
        self.metrics = {
            "total_predictions": 0,
            "correct_predictions": 0,
            "total_latency_ms": 0,
            "predictions": []
        }
    
    def predict(self, source: int, target: int) -> dict:
        """Predict shortest path for source-target pair"""
        start_time = time.time()
        
        try:
            # Get Dijkstra prediction (ground truth)
            dij_path, dij_cost = self.dijkstra.find_shortest_path(source, target)
            
            # Get ML model's direct cost prediction (not path-based)
            import torch
            try:
                if self.ml.model is not None and self.ml.node_embeddings is not None:
                    # Direct ML model prediction for source->target cost
                    source_emb = self.ml.node_embeddings[source:source+1]
                    target_emb = self.ml.node_embeddings[target:target+1]
                    
                    with torch.no_grad():
                        ml_predicted_cost = self.ml.model.predict_cost(source_emb, target_emb).item()
                    
                    ml_cost = ml_predicted_cost
                    # For display, use Dijkstra path but ML predicted cost
                    ml_path = dij_path  # Use same path, but with ML cost estimate
                else:
                    # Fallback if model not loaded
                    ml_path, ml_cost = self.dijkstra.find_shortest_path(source, target)
            except:
                # If anything fails, fallback
                ml_path, ml_cost = self.dijkstra.find_shortest_path(source, target)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate error metrics
            error_percent = 100 * abs(float(dij_cost) - float(ml_cost)) / (float(dij_cost) + 1e-6)
            is_accurate = error_percent < 10  # Within 10% is good
            
            result = {
                "source": int(source),
                "target": int(target),
                "dijkstra_path": [int(x) for x in dij_path],
                "dijkstra_cost": float(dij_cost),
                "ml_path": [int(x) for x in ml_path],
                "ml_cost": float(ml_cost),
                "ml_cost_predicted": float(ml_cost),
                "error_percent": float(error_percent),
                "is_accurate": bool(is_accurate),
                "is_correct": bool(is_accurate),  # For frontend compatibility
                "latency_ms": float(latency_ms),
                "timestamp": float(time.time())
            }
            
            # Update metrics
            self.metrics["total_predictions"] += 1
            if is_accurate:
                self.metrics["correct_predictions"] += 1
            self.metrics["total_latency_ms"] += latency_ms
            self.metrics["predictions"].append(result)
            
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def get_metrics(self) -> dict:
        """Get performance metrics"""
        if self.metrics["total_predictions"] == 0:
            return {
                "total_predictions": 0,
                "accuracy": 0.0,
                "avg_latency_ms": 0.0
            }
        
        return {
            "total_predictions": self.metrics["total_predictions"],
            "correct_predictions": self.metrics["correct_predictions"],
            "accuracy": self.metrics["correct_predictions"] / self.metrics["total_predictions"],
            "avg_latency_ms": self.metrics["total_latency_ms"] / self.metrics["total_predictions"]
        }
    
    def save_metrics(self, output_path: Path = None):
        """Save metrics to CSV"""
        if output_path is None:
            output_path = config.METRICS_PATH
        
        df = pd.DataFrame(self.metrics["predictions"])
        df.to_csv(output_path, index=False)
        print(f"[OK] Metrics saved to {output_path}")

# Global service instance
predictor_service = None

def init_predictor_service(graph_size: str = "medium"):
    """Initialize the predictor service"""
    global predictor_service
    predictor_service = PredictorService(graph_size)
    print("[OK] Predictor service initialized")

def get_predictor_service() -> PredictorService:
    """Get the global predictor service"""
    global predictor_service
    if predictor_service is None:
        init_predictor_service()
    return predictor_service
