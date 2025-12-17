import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
GRAPHS_DIR = DATA_DIR / "graphs"

# Kafka config
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "graph-paths")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "predictor-group")

# Model config
MODEL_PATH = DATA_DIR / "models" / "graphsage_model.pt"
EMBEDDINGS_PATH = DATA_DIR / "embeddings"

# Output paths
OUTPUT_PREDICTIONS = DATA_DIR / "predictions.csv"
METRICS_PATH = DATA_DIR / "metrics.csv"

# Graph parameters
DEFAULT_GRAPH_SIZE = "medium"  # small, medium, large

# Create directories if not exist
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)
