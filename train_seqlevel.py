"""
Training Script for Sequence-Level MSA Model

This script trains the new MSASeqLevelModel on CMU-MOSI/MOSEI data.
It maintains compatibility with the existing data loading pipeline while
using the new sequence-level architecture.
"""

from models import MSASeqLevelModel
from utils.data_loader import make_dataset
import tensorflow as tf
import os
from datetime import datetime
import json
import argparse

# Additional stability settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging

# Configure GPU (Apple Silicon MPS backend)
def configure_gpu():
    """Configure GPU settings for Apple Silicon (M1/M2/M3) or other GPUs."""
    print("\n" + "="*70)
    print("GPU Configuration")
    print("="*70)
    
    # Check for Apple Silicon GPU (Metal Performance Shaders)
    try:
        # List all available devices
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"\n✓ Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  - {gpu}")
            
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"  ✓ Enabled memory growth for {gpu.name}")
                except RuntimeError as e:
                    print(f"  ⚠ Could not set memory growth: {e}")
            
            print("\n✓ GPU acceleration enabled (Metal Performance Shaders for Apple Silicon)")
            return True
        else:
            print("\n⚠ No GPU found. Training will use CPU.")
            print("  For Apple Silicon (M2 Pro), ensure tensorflow-metal is installed:")
            print("  pip install tensorflow-metal")
            return False
            
    except Exception as e:
        print(f"\n⚠ GPU configuration failed: {e}")
        print("  Falling back to CPU execution")
        return False

# Configure GPU at startup
gpu_available = configure_gpu()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Sequence-Level MSA Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model architecture parameters
    parser.add_argument('--seq_len', type=int, default=20,
                       help='Sequence length (default: 20)')
    parser.add_argument('--model_dim', type=int, default=128,
                       help='Model dimension (default: 128)')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads (default: 4)')
    parser.add_argument('--ff_dim', type=int, default=256,
                       help='Feed-forward dimension (default: 256)')
    parser.add_argument('--n_layers_mod', type=int, default=2,
                       help='Transformer layers per modality (default: 2)')
    parser.add_argument('--n_layers_fuse', type=int, default=1,
                       help='Cross-attention fusion layers (default: 1)')
    parser.add_argument('--bidirectional_fusion', action='store_true',
                       help='Use bidirectional cross-attention')
    parser.add_argument('--pooling_method', type=str, default='mean',
                       choices=['mean', 'attention'],
                       help='Pooling method for fusion head (default: mean)')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    
    # Mixed precision
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Enable mixed precision training (recommended for GPU)')
    
    # Device selection
    parser.add_argument('--use_cpu', action='store_true',
                       help='Force CPU execution (disables GPU)')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory (default: ./data)')
    parser.add_argument('--weights_dir', type=str, default='./weights',
                       help='Weights directory (default: ./weights)')
    
    return parser.parse_args()


def create_model(args):
    """Create and build the model."""
    print("\n" + "="*70)
    print("Creating Sequence-Level MSA Model")
    print("="*70)
    
    # Fixed input dimensions from CMU-MOSEI data
    # Note: MOSI uses video_dim=47, MOSEI uses video_dim=713
    text_dim = 300
    audio_dim = 74
    video_dim = 713  # MOSEI video features (MOSI uses 47)
    
    print(f"\nArchitecture Configuration:")
    print(f"  Sequence Length: {args.seq_len}")
    print(f"  Text Dim: {text_dim}, Audio Dim: {audio_dim}, Video Dim: {video_dim}")
    print(f"  Model Dimension: {args.model_dim}")
    print(f"  Attention Heads: {args.num_heads}")
    print(f"  Feed-Forward Dim: {args.ff_dim}")
    print(f"  Modality Layers: {args.n_layers_mod}")
    print(f"  Fusion Layers: {args.n_layers_fuse}")
    print(f"  Bidirectional Fusion: {args.bidirectional_fusion}")
    print(f"  Pooling Method: {args.pooling_method}")
    print(f"  Dropout Rate: {args.dropout_rate}")
    
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
        dropout_rate=args.dropout_rate
    )
    
    # Warm up the model with a dummy batch
    dummy_t = tf.zeros((1, args.seq_len, text_dim))
    dummy_a = tf.zeros((1, args.seq_len, audio_dim))
    dummy_v = tf.zeros((1, args.seq_len, video_dim))
    _ = model((dummy_t, dummy_a, dummy_v), training=False)
    
    # Display summary
    print("\nModel Summary:")
    model.summary()
    
    return model


def compile_model(model, args):
    """Compile the model with optimizer and loss."""
    print("\n" + "="*70)
    print("Compiling Model")
    print("="*70)
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        beta_1=0.9,
        beta_2=0.999
    )
    
    # Mixed precision optimizer wrapper
    if args.mixed_precision:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
        print(f"\nMixed precision enabled: {policy.name}")
    
    # Compile
    model.compile(
        optimizer=optimizer,
        loss='mae',
        metrics=['mae', 'mse']
    )
    
    print(f"\nOptimizer: Adam (lr={args.learning_rate})")
    print(f"Loss: MAE")
    print(f"Metrics: MAE, MSE")
    
    return model


def load_datasets(args):
    """Load training, validation, and test datasets."""
    print("\n" + "="*70)
    print("Loading Datasets")
    print("="*70)
    
    train_data = make_dataset(
        args.data_dir,
        split="train",
        batch_size=args.batch_size
    )
    
    val_data = make_dataset(
        args.data_dir,
        split="valid",
        batch_size=args.batch_size
    )
    
    test_data = make_dataset(
        args.data_dir,
        split="test",
        batch_size=args.batch_size
    )
    
    print(f"\nDatasets loaded successfully")
    print(f"  Batch size: {args.batch_size}")
    
    return train_data, val_data, test_data


def create_callbacks(args, timestamp):
    """Create training callbacks."""
    os.makedirs(args.weights_dir, exist_ok=True)
    
    callbacks = [
        # Save best model based on validation MAE
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                args.weights_dir,
                f'seqlevel_best_val_mae_{timestamp}.h5'
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
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            mode='min',
            factor=0.5,
            patience=args.patience // 2,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Log training progress
        tf.keras.callbacks.CSVLogger(
            os.path.join(
                args.weights_dir,
                f'seqlevel_training_log_{timestamp}.csv'
            )
        )
    ]
    
    return callbacks


def train_model(model, train_data, val_data, args, callbacks):
    """Train the model."""
    print("\n" + "="*70)
    print("Training")
    print("="*70)
    print(f"\nEpochs: {args.epochs}")
    print(f"Early stopping patience: {args.patience}")
    print("\nStarting training...\n")
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
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


def save_results(model, history, test_results, args, timestamp):
    """Save final model and training history."""
    # Save final model
    final_model_path = os.path.join(
        args.weights_dir,
        f'seqlevel_final_{timestamp}.h5'
    )
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save training history
    history_path = os.path.join(
        args.weights_dir,
        f'seqlevel_history_{timestamp}.json'
    )
    with open(history_path, 'w') as f:
        history_dict = {
            k: [float(x) for x in v]
            for k, v in history.history.items()
        }
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Save configuration and results
    config_path = os.path.join(
        args.weights_dir,
        f'seqlevel_config_{timestamp}.json'
    )
    config = {
        'architecture': {
            'seq_len': args.seq_len,
            'model_dim': args.model_dim,
            'num_heads': args.num_heads,
            'ff_dim': args.ff_dim,
            'n_layers_mod': args.n_layers_mod,
            'n_layers_fuse': args.n_layers_fuse,
            'bidirectional_fusion': args.bidirectional_fusion,
            'pooling_method': args.pooling_method,
            'dropout_rate': args.dropout_rate
        },
        'training': {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'patience': args.patience,
            'mixed_precision': args.mixed_precision
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
    # Parse arguments
    args = parse_args()
    
    # Override GPU configuration if CPU is requested
    if args.use_cpu:
        print("\n⚠ CPU-only mode requested. Disabling GPU...")
        tf.config.set_visible_devices([], 'GPU')
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "#"*70)
    print("#  Sequence-Level MSA Model Training")
    print("#"*70)
    print(f"\nTimestamp: {timestamp}")
    
    # Display device info
    if args.use_cpu:
        print("Device: CPU (forced)")
    elif gpu_available:
        print("Device: GPU (Apple Silicon Metal)")
    else:
        print("Device: CPU (no GPU detected)")
    
    # Create model
    model = create_model(args)
    
    # Compile model
    model = compile_model(model, args)
    
    # Load datasets
    train_data, val_data, test_data = load_datasets(args)
    
    # Create callbacks
    callbacks = create_callbacks(args, timestamp)
    
    # Train model
    history = train_model(model, train_data, val_data, args, callbacks)
    
    # Evaluate model
    test_results = evaluate_model(model, test_data)
    
    # Save results
    save_results(model, history, test_results, args, timestamp)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == '__main__':
    main()

