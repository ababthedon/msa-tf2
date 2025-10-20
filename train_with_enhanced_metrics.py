"""
Training script for MSA-TF2 with enhanced sentiment analysis metrics.
Demonstrates proper evaluation methodology for multimodal sentiment analysis.
"""

from models.msamodel import MSAModel
from utils.data_loader import make_dataset
from utils.enhanced_metrics import get_core_sentiment_metrics, comprehensive_evaluation
import tensorflow as tf
import os
from datetime import datetime
import numpy as np

# Configure TensorFlow
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_visible_devices([], 'GPU')

os.makedirs('./weights', exist_ok=True)

# Model parameters
T = 20
text_dim = 300
audio_dim = 74
video_dim = 713  # MOSEI: 713, MOSI: 47
model_dim = 64
heads = 4
ff_dim = 256
n_layers = 2
n_layers_fuse = 1

print("="*60)
print("MSA-TF2 TRAINING WITH ENHANCED METRICS")
print("="*60)

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

# Warm up the model
dummy_t = tf.zeros((1, T, text_dim))
dummy_a = tf.zeros((1, T, audio_dim))
dummy_v = tf.zeros((1, T, video_dim))
_ = model((dummy_t, dummy_a, dummy_v), training=False)

print("\nModel Architecture:")
model.summary()

# Compile with enhanced metrics
print("\nCompiling model with enhanced sentiment analysis metrics...")
enhanced_metrics = get_core_sentiment_metrics()

print("Metrics included:")
for i, metric in enumerate(enhanced_metrics):
    if hasattr(metric, '__name__'):
        print(f"  {i+1}. {metric.__name__}")
    else:
        print(f"  {i+1}. {metric}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mae',  # Keep MAE as primary loss
    metrics=enhanced_metrics
)

# Load datasets
batch_size = 8  # Moderate batch size for small dataset
train_data = make_dataset("./data", split="train", batch_size=batch_size)
val_data = make_dataset("./data", split="valid", batch_size=batch_size)
test_data = make_dataset("./data", split="test", batch_size=batch_size)

# Enhanced callbacks
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Custom callback to track multiple metrics
class MultiMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_val_mae = float('inf')
        self.best_val_pearson = -1.0
        self.best_epoch_mae = 0
        self.best_epoch_pearson = 0
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        val_mae = logs.get('val_mae', float('inf'))
        val_pearson = logs.get('val_pearson_correlation', -1.0)
        
        # Track best MAE
        if val_mae < self.best_val_mae:
            self.best_val_mae = val_mae
            self.best_epoch_mae = epoch + 1
        
        # Track best Pearson correlation
        if val_pearson > self.best_val_pearson:
            self.best_val_pearson = val_pearson
            self.best_epoch_pearson = epoch + 1
        
        # Print summary every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Current Val MAE: {val_mae:.4f} | Best: {self.best_val_mae:.4f} (epoch {self.best_epoch_mae})")
            print(f"  Current Val Pearson: {val_pearson:.4f} | Best: {self.best_val_pearson:.4f} (epoch {self.best_epoch_pearson})")

callbacks = [
    # Primary checkpoint on MAE
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./weights/best_enhanced_model_mae.h5',
        monitor='val_mae',
        mode='min',
        save_best_only=True,
        verbose=1
    ),
    
    # Secondary checkpoint on correlation
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./weights/best_enhanced_model_pearson.h5',
        monitor='val_pearson_correlation',
        mode='max',
        save_best_only=True,
        verbose=0
    ),
    
    # Early stopping with more conservative patience for small dataset
    tf.keras.callbacks.EarlyStopping(
        monitor='val_mae',
        mode='min',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Learning rate reduction
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_mae',
        mode='min',
        factor=0.7,
        patience=6,
        min_lr=1e-6,
        verbose=1
    ),
    
    # CSV logger with all metrics
    tf.keras.callbacks.CSVLogger(f'./weights/enhanced_training_log_{timestamp}.csv'),
    
    # Custom multi-metric callback
    MultiMetricCallback()
]

print(f"\nStarting training with enhanced metrics...")
print(f"Dataset sizes: Train={len(list(train_data.unbatch()))}, "
      f"Val={len(list(val_data.unbatch()))}, "
      f"Test={len(list(test_data.unbatch()))}")
print(f"Batch size: {batch_size}")

# Train model
try:
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=50,  # Reduced for small dataset
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    
    # Basic evaluation on test set
    print("\nBasic Test Set Evaluation:")
    test_results = model.evaluate(test_data, verbose=1)
    
    print(f"\nBasic Test Results:")
    for metric_name, value in zip(model.metrics_names, test_results):
        print(f"  {metric_name}: {value:.4f}")
    
    # Comprehensive evaluation
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST SET EVALUATION")
    print("="*60)
    
    comprehensive_metrics = comprehensive_evaluation(model, test_data, verbose=True)
    
    # Analysis and recommendations
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    mae = comprehensive_metrics['mae']
    pearson_r = comprehensive_metrics['pearson_r']
    binary_acc = comprehensive_metrics['binary_accuracy']
    explained_var = comprehensive_metrics['explained_variance']
    
    print(f"Primary Metrics Summary:")
    print(f"  MAE: {mae:.4f} (lower is better)")
    print(f"  Pearson r: {pearson_r:.4f} (higher is better, >0.7 is good)")
    print(f"  Binary Accuracy: {binary_acc:.4f} (>0.8 is good for practical use)")
    print(f"  Explained Variance: {explained_var:.4f} (>0.5 indicates good fit)")
    
    # Performance assessment
    print(f"\nPerformance Assessment:")
    if mae < 1.0:
        print("  ✅ MAE < 1.0: Excellent prediction accuracy")
    elif mae < 1.5:
        print("  🟡 MAE < 1.5: Good prediction accuracy")
    else:
        print("  ❌ MAE ≥ 1.5: Poor prediction accuracy - model needs improvement")
    
    if pearson_r > 0.7:
        print("  ✅ Pearson r > 0.7: Strong correlation with ground truth")
    elif pearson_r > 0.5:
        print("  🟡 Pearson r > 0.5: Moderate correlation")
    else:
        print("  ❌ Pearson r ≤ 0.5: Weak correlation - model struggles to capture sentiment patterns")
    
    if binary_acc > 0.8:
        print("  ✅ Binary Accuracy > 80%: Excellent for positive/negative classification")
    elif binary_acc > 0.7:
        print("  🟡 Binary Accuracy > 70%: Good for basic sentiment classification")
    else:
        print("  ❌ Binary Accuracy ≤ 70%: Poor sentiment direction prediction")
    
    # Overfitting analysis
    if len(history.history['mae']) > 5:
        final_train_mae = history.history['mae'][-1]
        final_val_mae = history.history['val_mae'][-1]
        overfitting_gap = final_val_mae - final_train_mae
        
        print(f"\nOverfitting Analysis:")
        print(f"  Final Train MAE: {final_train_mae:.4f}")
        print(f"  Final Val MAE: {final_val_mae:.4f}")
        print(f"  Overfitting Gap: {overfitting_gap:.4f}")
        
        if overfitting_gap < 0.3:
            print("  ✅ Low overfitting - model generalizes well")
        elif overfitting_gap < 0.7:
            print("  🟡 Moderate overfitting - consider more regularization")
        else:
            print("  ❌ High overfitting - model memorizing training data")
    
    # Save comprehensive results
    import json
    results_path = f'./weights/comprehensive_results_{timestamp}.json'
    
    # Convert numpy types to native Python types for JSON serialization
    json_metrics = {}
    for key, value in comprehensive_metrics.items():
        if isinstance(value, dict):
            json_metrics[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                               for k, v in value.items()}
        elif isinstance(value, (np.floating, np.integer)):
            json_metrics[key] = float(value)
        else:
            json_metrics[key] = value
    
    with open(results_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    print(f"\nComprehensive results saved to: {results_path}")
    
    # Save training history
    history_path = f'./weights/enhanced_training_history_{timestamp}.json'
    with open(history_path, 'w') as f:
        history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
        json.dump(history_dict, f, indent=2)
    
    print(f"Training history saved to: {history_path}")
    
    # Final recommendations
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if mae > 1.5 or pearson_r < 0.5:
        print("🔴 Model Performance Issues Detected:")
        print("   1. Consider reducing model complexity for this small dataset")
        print("   2. Increase regularization (dropout, L2)")
        print("   3. Try simpler baseline models")
        print("   4. Implement cross-validation due to small dataset size")
        print("   5. Consider data augmentation techniques")
    elif overfitting_gap > 0.7:
        print("🟡 Overfitting Detected:")
        print("   1. Increase dropout rate")
        print("   2. Add L2 regularization")
        print("   3. Reduce model complexity")
        print("   4. Implement early stopping with lower patience")
    else:
        print("✅ Model Performance Acceptable:")
        print("   1. Consider fine-tuning hyperparameters for further improvement")
        print("   2. Experiment with different fusion strategies")
        print("   3. Try ensemble methods")
    
except Exception as e:
    print(f"\nError during training: {e}")
    print("Check data files and model configuration.")
    
print(f"\n" + "="*60)
print("TRAINING SESSION COMPLETE")
print("="*60)








