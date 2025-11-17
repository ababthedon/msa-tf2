"""
Fair Comparison Training Script for MSA Sequence-Level Model

This script trains the MSA Seq-Level model with standardized hyperparameters
for fair comparison with Deep-HOSeq model.

Changes from original:
- Standardized batch size and learning rate
- Same LR schedule and early stopping settings
"""

import sys
import os

# Add parent directory to import fair_comparison_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fair_comparison_config import get_config, print_config

from models import MSASeqLevelModel
from utils.data_loader import make_dataset
import tensorflow as tf
from datetime import datetime
import json
import argparse

# Additional stability settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def configure_gpu():
    """Configure GPU settings."""
    print("\n" + "="*70)
    print("GPU Configuration")
    print("="*70)
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"\n✓ Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  - {gpu}")
            
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"  ✓ Enabled memory growth for {gpu.name}")
                except RuntimeError as e:
                    print(f"  ⚠ Could not set memory growth: {e}")
            
            print("\n✓ GPU acceleration enabled")
            return True
        else:
            print("\n⚠ No GPU found. Training will use CPU.")
            return False
            
    except Exception as e:
        print(f"\n⚠ GPU configuration failed: {e}")
        return False


gpu_available = configure_gpu()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Sequence-Level MSA Model (Fair Comparison)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Fair comparison config selection
    parser.add_argument('--config', type=str, default='standard',
                       choices=['standard', 'large_batch', 'small_batch'],
                       help='Fair comparison configuration preset')
    
    # Model architecture parameters (keep these as-is)
    parser.add_argument('--seq_len', type=int, default=20,
                       help='Sequence length')
    parser.add_argument('--model_dim', type=int, default=128,
                       help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--ff_dim', type=int, default=256,
                       help='Feed-forward dimension')
    parser.add_argument('--n_layers_mod', type=int, default=2,
                       help='Transformer layers per modality')
    parser.add_argument('--n_layers_fuse', type=int, default=1,
                       help='Cross-attention fusion layers')
    parser.add_argument('--bidirectional_fusion', action='store_true',
                       help='Use bidirectional cross-attention')
    parser.add_argument('--pooling_method', type=str, default='mean',
                       choices=['mean', 'attention'],
                       help='Pooling method for fusion head')
    
    # Device selection
    parser.add_argument('--use_cpu', action='store_true',
                       help='Force CPU execution')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--weights_dir', type=str, default='./weights',
                       help='Weights directory')
    
    return parser.parse_args()


def create_model(args, fair_config):
    """Create and build the model."""
    print("\n" + "="*70)
    print("Creating Sequence-Level MSA Model")
    print("="*70)
    
    # Fixed input dimensions
    text_dim = 300
    audio_dim = 74
    video_dim = 47  # MOSI - change to 713 for MOSEI
    
    print(f"\nArchitecture Configuration:")
    print(f"  Sequence Length: {args.seq_len}")
    print(f"  Text Dim: {text_dim}, Audio Dim: {audio_dim}, Video Dim: {video_dim}")
    print(f"  Model Dimension: {args.model_dim}")
    print(f"  Attention Heads: {args.num_heads}")
    print(f"  Feed-Forward Dim: {args.ff_dim}")
    print(f"  Modality Layers: {args.n_layers_mod}")
    print(f"  Fusion Layers: {args.n_layers_fuse}")
    print(f"  Dropout Rate: {fair_config['dropout_rate']}")
    
    # Create model
    model = MSASeqLevelModel(
        seq_len=args.seq_len,
        text_dim=text_dim,
        audio_dim=audio_dim,
        video_dim=video_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        n_layers_mod=args.n_layers_mod,
        n_layers_fuse=args.n_layers_fuse,
        bidirectional_fusion=args.bidirectional_fusion,
        pooling_method=args.pooling_method,
        dropout_rate=fair_config['dropout_rate']
    )
    
    # Warm up the model
    dummy_t = tf.zeros((1, args.seq_len, text_dim))
    dummy_a = tf.zeros((1, args.seq_len, audio_dim))
    dummy_v = tf.zeros((1, args.seq_len, video_dim))
    _ = model((dummy_t, dummy_a, dummy_v), training=False)
    
    print("\nModel Summary:")
    model.summary()
    
    return model


def compile_model(model, fair_config):
    """Compile the model with optimizer and loss."""
    print("\n" + "="*70)
    print("Compiling Model")
    print("="*70)
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=fair_config['learning_rate'],
        beta_1=fair_config['beta_1'],
        beta_2=fair_config['beta_2']
    )
    
    # Compile
    model.compile(
        optimizer=optimizer,
        loss=fair_config['loss_function'],
        metrics=['mae', 'mse']
    )
    
    print(f"\nOptimizer: Adam (lr={fair_config['learning_rate']})")
    print(f"Loss: {fair_config['loss_function'].upper()}")
    print(f"Metrics: MAE, MSE")
    
    return model


def load_datasets(args, fair_config):
    """Load training, validation, and test datasets."""
    print("\n" + "="*70)
    print("Loading Datasets")
    print("="*70)
    
    train_data = make_dataset(
        args.data_dir,
        split="train",
        batch_size=fair_config['batch_size']
    )
    
    val_data = make_dataset(
        args.data_dir,
        split="valid",
        batch_size=fair_config['batch_size']
    )
    
    test_data = make_dataset(
        args.data_dir,
        split="test",
        batch_size=fair_config['batch_size']
    )
    
    print(f"\nDatasets loaded successfully")
    print(f"  Batch size: {fair_config['batch_size']}")
    
    return train_data, val_data, test_data


def create_callbacks(args, fair_config, timestamp):
    """Create training callbacks."""
    os.makedirs(args.weights_dir, exist_ok=True)
    
    callbacks = [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                args.weights_dir,
                f'seqlevel_fair_{args.config}_best_{timestamp}.h5'
            ),
            monitor='val_mae',
            mode='min',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',
            mode='min',
            patience=fair_config['early_stop_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            mode='min',
            factor=fair_config['reduce_lr_factor'],
            patience=fair_config['reduce_lr_patience'],
            min_lr=fair_config['reduce_lr_min_lr'],
            verbose=1
        ),
        
        # Log training progress
        tf.keras.callbacks.CSVLogger(
            os.path.join(
                args.weights_dir,
                f'seqlevel_fair_{args.config}_log_{timestamp}.csv'
            )
        )
    ]
    
    return callbacks


def train_model(model, train_data, val_data, fair_config, callbacks):
    """Train the model."""
    print("\n" + "="*70)
    print("Training")
    print("="*70)
    print(f"\nMax Epochs: {fair_config['max_epochs']}")
    print(f"Early stopping patience: {fair_config['early_stop_patience']}")
    print("\nStarting training...\n")
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=fair_config['max_epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, test_data):
    """Evaluate the model on test set."""
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    test_results = model.evaluate(test_data, verbose=1)
    
    print(f"\nTest Results:")
    for metric_name, value in zip(model.metrics_names, test_results):
        print(f"  {metric_name}: {value:.4f}")
    
    return test_results


def save_results(model, history, test_results, args, fair_config, timestamp):
    """Save final model and training history."""
    # Save final model
    final_model_path = os.path.join(
        args.weights_dir,
        f'seqlevel_fair_{args.config}_final_{timestamp}.h5'
    )
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save training history
    history_path = os.path.join(
        args.weights_dir,
        f'seqlevel_fair_{args.config}_history_{timestamp}.json'
    )
    with open(history_path, 'w') as f:
        history_dict = {
            k: [float(x) for x in v]
            for k, v in history.history.items()
        }
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Save configuration
    config_path = os.path.join(
        args.weights_dir,
        f'seqlevel_fair_{args.config}_config_{timestamp}.json'
    )
    config = {
        'fair_comparison_config': fair_config,
        'architecture': {
            'seq_len': args.seq_len,
            'model_dim': args.model_dim,
            'num_heads': args.num_heads,
            'ff_dim': args.ff_dim,
            'n_layers_mod': args.n_layers_mod,
            'n_layers_fuse': args.n_layers_fuse,
            'bidirectional_fusion': args.bidirectional_fusion,
            'pooling_method': args.pooling_method,
        },
        'test_results': {
            metric_name: float(value)
            for metric_name, value in zip(model.metrics_names, test_results)
        }
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Load fair comparison configuration
    fair_config = get_config(args.config)
    
    # Override GPU if requested
    if args.use_cpu:
        print("\n⚠ CPU-only mode requested. Disabling GPU...")
        tf.config.set_visible_devices([], 'GPU')
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "#"*70)
    print("#  MSA Sequence-Level Model - Fair Comparison Training")
    print("#"*70)
    print(f"\nTimestamp: {timestamp}")
    
    # Print fair comparison configuration
    print_config(args.config)
    
    # Create model
    model = create_model(args, fair_config)
    
    # Compile model
    model = compile_model(model, fair_config)
    
    # Load datasets
    train_data, val_data, test_data = load_datasets(args, fair_config)
    
    # Create callbacks
    callbacks = create_callbacks(args, fair_config, timestamp)
    
    # Train model
    history = train_model(model, train_data, val_data, fair_config, callbacks)
    
    # Evaluate model
    test_results = evaluate_model(model, test_data)
    
    # Save results
    save_results(model, history, test_results, args, fair_config, timestamp)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == '__main__':
    main()


