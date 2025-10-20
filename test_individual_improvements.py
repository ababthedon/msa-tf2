"""
Script to test individual improvements one at a time.
This helps identify which changes help vs hurt performance.
"""

from models.msamodel import MSAModel
from utils.data_loader import make_dataset
import tensorflow as tf
import os
from datetime import datetime

# Configure TensorFlow
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_visible_devices([], 'GPU')

os.makedirs('./weights', exist_ok=True)

# Base parameters
T = 20
text_dim = 300
audio_dim = 74
video_dim = 47
model_dim = 64
heads = 4
ff_dim = 256
n_layers = 2
n_layers_fuse = 1

def test_configuration(config_name, **kwargs):
    """Test a specific configuration and return results."""
    print(f"\n{'='*60}")
    print(f"TESTING: {config_name}")
    print(f"{'='*60}")
    
    # Extract hyperparameters
    dropout_rate = kwargs.get('dropout_rate', 0.1)
    batch_size = kwargs.get('batch_size', 4)
    learning_rate = kwargs.get('learning_rate', 0.001)
    l2_reg = kwargs.get('l2_reg', 0.0)
    clipnorm = kwargs.get('clipnorm', None)
    patience = kwargs.get('patience', 15)
    
    print(f"Configuration:")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")
    
    # Build model
    model = MSAModel(
        seq_len=T,
        text_dim=text_dim,
        audio_dim=audio_dim,
        video_dim=video_dim,
        model_dim=model_dim,
        num_heads=heads,
        ff_dim=ff_dim,
        n_layers_mod=n_layers,
        n_layers_fuse=n_layers_fuse,
        adaptive_fusion=True
    )
    
    # Warm up
    dummy_t = tf.zeros((1, T, text_dim))
    dummy_a = tf.zeros((1, T, audio_dim))
    dummy_v = tf.zeros((1, T, video_dim))
    _ = model((dummy_t, dummy_a, dummy_v), training=False)
    
    # Configure optimizer
    optimizer_kwargs = {
        'learning_rate': learning_rate,
        'beta_1': 0.9,
        'beta_2': 0.999
    }
    if clipnorm:
        optimizer_kwargs['clipnorm'] = clipnorm
        
    optimizer = tf.keras.optimizers.Adam(**optimizer_kwargs)
    
    model.compile(
        optimizer=optimizer,
        loss='mae',
        metrics=['mae', 'mse']
    )
    
    # Load data
    train_data = make_dataset("./data", split="train", batch_size=batch_size)
    val_data = make_dataset("./data", split="valid", batch_size=batch_size)
    
    # Setup callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'./weights/test_{config_name.lower().replace(" ", "_")}.h5',
            monitor='val_mae',
            mode='min',
            save_best_only=True,
            save_weights_only=False,
            verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',
            mode='min',
            patience=patience,
            restore_best_weights=True,
            verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            mode='min',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=0
        )
    ]
    
    # Train
    try:
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=30,  # Shorter training for quick tests
            callbacks=callbacks,
            verbose=0
        )
        
        # Results
        best_val_mae = min(history.history['val_mae'])
        final_train_mae = history.history['mae'][-1]
        final_val_mae = history.history['val_mae'][-1]
        overfitting_gap = final_val_mae - final_train_mae
        
        results = {
            'config_name': config_name,
            'best_val_mae': best_val_mae,
            'final_train_mae': final_train_mae,
            'final_val_mae': final_val_mae,
            'overfitting_gap': overfitting_gap,
            'epochs_trained': len(history.history['mae'])
        }
        
        print(f"\nResults:")
        print(f"  Best validation MAE: {best_val_mae:.4f}")
        print(f"  Final train MAE: {final_train_mae:.4f}")
        print(f"  Final validation MAE: {final_val_mae:.4f}")
        print(f"  Overfitting gap: {overfitting_gap:.4f}")
        print(f"  Epochs trained: {results['epochs_trained']}")
        
        return results
        
    except Exception as e:
        print(f"Error during training: {e}")
        return None

# Test configurations
configurations = [
    # Baseline (original)
    {
        'name': 'Baseline (Original)',
        'dropout_rate': 0.1,
        'batch_size': 4,
        'learning_rate': 0.001,
        'patience': 15
    },
    
    # Test 1: Only increase dropout
    {
        'name': 'Only Increased Dropout',
        'dropout_rate': 0.2,
        'batch_size': 4,
        'learning_rate': 0.001,
        'patience': 15
    },
    
    # Test 2: Only increase batch size
    {
        'name': 'Only Larger Batch Size',
        'dropout_rate': 0.1,
        'batch_size': 8,
        'learning_rate': 0.001,
        'patience': 15
    },
    
    # Test 3: Only lower learning rate
    {
        'name': 'Only Lower Learning Rate',
        'dropout_rate': 0.1,
        'batch_size': 4,
        'learning_rate': 0.0008,
        'patience': 15
    },
    
    # Test 4: Only add gradient clipping
    {
        'name': 'Only Gradient Clipping',
        'dropout_rate': 0.1,
        'batch_size': 4,
        'learning_rate': 0.001,
        'clipnorm': 0.5,
        'patience': 15
    },
    
    # Test 5: Only reduce early stopping patience
    {
        'name': 'Only Reduced Patience',
        'dropout_rate': 0.1,
        'batch_size': 4,
        'learning_rate': 0.001,
        'patience': 10
    },
    
    # Test 6: Combine best individual improvements
    {
        'name': 'Combined Best',
        'dropout_rate': 0.15,  # Will adjust based on results
        'batch_size': 8,
        'learning_rate': 0.0008,
        'clipnorm': 0.5,
        'patience': 12
    }
]

# Run tests
results = []
baseline_val_mae = 1.6302  # From original training

print("Starting Individual Improvement Tests")
print("Each test runs for max 30 epochs with early stopping")
print("This will help identify which changes help vs hurt performance")

for config in configurations:
    config_name = config.pop('name')
    result = test_configuration(config_name, **config)
    if result:
        results.append(result)

# Summary
print("\n" + "="*80)
print("SUMMARY OF ALL TESTS")
print("="*80)

print(f"{'Configuration':<25} {'Best Val MAE':<12} {'Gap':<8} {'vs Baseline':<12}")
print("-" * 80)

for result in results:
    improvement = baseline_val_mae - result['best_val_mae']
    status = "BETTER" if improvement > 0 else "WORSE"
    
    print(f"{result['config_name']:<25} "
          f"{result['best_val_mae']:<12.4f} "
          f"{result['overfitting_gap']:<8.4f} "
          f"{improvement:>6.4f} ({status})")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

# Find best configuration
best_result = min(results, key=lambda x: x['best_val_mae'])
print(f"Best configuration: {best_result['config_name']}")
print(f"Best validation MAE: {best_result['best_val_mae']:.4f}")
print(f"Improvement over baseline: {baseline_val_mae - best_result['best_val_mae']:.4f}")

# Find configurations that reduce overfitting well
low_gap_results = [r for r in results if r['overfitting_gap'] < 0.5]
if low_gap_results:
    best_gap = min(low_gap_results, key=lambda x: x['overfitting_gap'])
    print(f"\nBest overfitting control: {best_gap['config_name']}")
    print(f"Overfitting gap: {best_gap['overfitting_gap']:.4f}")

print("\nNext steps:")
print("1. Use the best performing configuration for your final model")
print("2. If multiple configs perform similarly, choose the one with lower overfitting")
print("3. Consider combining successful individual improvements")








