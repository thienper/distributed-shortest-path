#!/bin/bash
# Quick start training script

echo "╔═══════════════════════════════════════════════════════╗"
echo "║ GraphSAGE Model Training - Quick Start                ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Check if data files exist
DATA_DIR="data/graphs"
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory not found: $DATA_DIR"
    echo "Generating test graphs..."
    python src/data_generation/graph_generator.py
fi

# Check if model directory exists
MODEL_DIR="models"
mkdir -p $MODEL_DIR
echo "✓ Model directory: $MODEL_DIR"
echo ""

# Parse arguments
GRAPH=${1:-"medium"}  # Default to medium
EPOCHS=${2:-"50"}     # Default 50 epochs

echo "📊 Training Configuration:"
echo "  Graph size: $GRAPH"
echo "  Epochs: $EPOCHS"
echo ""

# Run training
echo "🚀 Starting training..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python train_model.py --graph $GRAPH --epochs $EPOCHS

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Training Complete!"
echo ""
echo "📁 Output files:"
echo "  • models/best_model.pt - Trained model weights"
echo "  • models/config.json - Training configuration"
echo "  • models/results.json - Metrics and results"
echo ""
echo "📊 Next step: Analyze results"
echo "  jupyter notebook notebooks/analyze_training_results.ipynb"
echo ""
echo "🚀 Next step: Use in predictor"
echo "  1. Update src/predictor/predictor_service.py"
echo "  2. Uncomment MLPredictor initialization"
echo "  3. Restart Flask: python -m web_app.app"
echo ""
