#!/usr/bin/env python3
"""
Quick training script for GraphSAGE shortest path model

Usage:
    python train_model.py --graph medium --epochs 50 --batch-size 32

Options:
    --graph: small (100), medium (1000), large (5000)
    --epochs: number of training epochs
    --batch-size: batch size
    --learning-rate: learning rate
    --gpu: use GPU if available
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'predictor'))

from model_trainer import ShortestPathTrainer


def main():
    parser = argparse.ArgumentParser(description="Train GraphSAGE model for shortest path prediction")
    parser.add_argument('--graph', type=str, default='large', 
                       choices=['small', 'medium', 'large'],
                       help='Graph size to train on (default: large)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--embedding-dim', type=int, default=32,
                       help='Node embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='MLP hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of GraphSAGE layers')
    parser.add_argument('--num-samples', type=int, default=500,
                       help='Number of training samples')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for trained model')
    
    args = parser.parse_args()
    
    # Map graph size
    graph_sizes = {
        'small': 100,
        'medium': 1000,
        'large': 5000
    }
    graph_size = graph_sizes[args.graph]
    
    # Paths - check both possible locations
    data_dir = Path(__file__).parent / 'data' / 'graphs'
    
    # Try direct path first, then with graph subdirectory
    nodes_csv = data_dir / f'{args.graph}_graph_nodes.csv'
    edges_csv = data_dir / f'{args.graph}_graph_edges.csv'
    
    if not nodes_csv.exists():
        nodes_csv = data_dir / f'graph_{args.graph}' / 'nodes.csv'
        edges_csv = data_dir / f'graph_{args.graph}' / 'edges.csv'
    
    print(f"\n{'='*60}")
    print(f"GraphSAGE Training - Shortest Path Prediction")
    print(f"{'='*60}")
    print(f"Graph size: {args.graph} ({graph_size} nodes)")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"{'='*60}\n")
    
    # Check files exist
    if not nodes_csv.exists() or not edges_csv.exists():
        print(f"[ERROR] Graph files not found:")
        print(f"  {nodes_csv}")
        print(f"  {edges_csv}")
        print(f"\nPlease run: python src/data_generation/graph_generator.py")
        sys.exit(1)
    
    # Device
    device = "cuda" if args.gpu else "cpu"
    print(f"Using device: {device}\n")
    
    # Config
    config = {
        'num_samples': args.num_samples,
        'input_dim': 1,
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-5,
        'feature_type': 'degree',
        'train_test_split': 0.8,
    }
    
    # Train
    trainer = ShortestPathTrainer(model_dir=args.output_dir, device=device)
    results = trainer.train(
        nodes_csv=str(nodes_csv),
        edges_csv=str(edges_csv),
        config=config
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Training Summary")
    print(f"{'='*60}")
    print(f"Final Train Loss: {results['best_train_loss']:.4f}")
    print(f"Final Test Loss: {results['best_test_loss']:.4f}")
    print(f"Test RMSE: {results['test_rmse']:.4f}")
    print(f"Test MAPE: {results['test_mape']:.2f}%")
    print(f"Model saved to: {trainer.model_dir}")
    print(f"{'='*60}\n")
    
    print("[INFO] Next step: Update predictor_service.py to use the trained model")
    print(f"[INFO] Add this to predictor_service.__init__:")
    print(f"""
        self.ml_predictor = MLPredictor(
            adj_matrix=graph_loader.adjacency_matrix,
            model_path='{args.output_dir}',
            node_features=None
        )
    """)


if __name__ == '__main__':
    main()
