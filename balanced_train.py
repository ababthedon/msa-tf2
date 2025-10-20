from models.balanced_msamodel import BalancedMSAModel
from utils.data_loader import make_dataset
import tensorflow as tf
import os
from datetime import datetime
import numpy as np

# Configure TensorFlow for stability
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_visible_devices([], 'GPU')

# Create weights directory
os.makedirs('./weights', exist_ok=True)

# Model parameters (keep original dimensions)
T = 20
text_dim = 300
audio_dim = 74
video_dim = 713  # MOSEI: 713, MOSI: 47
model_dim = 64
heads = 4
ff_dim = 256
n_layers = 2
n_layers_fuse = 1

# Build balanced model with moderate improvements
model = BalancedMSAModel(
    seq_len=T,
    text_dim=text_dim,
    audio_dim=audio_dim,
    video_dim=video_dim,
    model_dim=model_dim,
    num_heads=heads,
    ff_dim=ff_dim,
    n_layers_mod=n_layers,
    n_layers_fuse=n_layers_fuse,
    dropout_rate=0.2,        # Moderate increase from 0.1
    l2_reg=0.001,           # Light regularization
    adaptive_fusion=True
)

# Warm up the model
dummy_t = tf.zeros((1, T, text_dim))
dummy_a = tf.zeros((1, T, audio_dim))
dummy_v = tf.zeros((1, T, video_dim))
_ = model((dummy_t, dummy_a, dummy_v), training=False)
model.summary()

# Balanced optimizer configuration
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0008,    # Slightly lower than original 0.001
    beta_1=0.9,
    beta_2=0.999,
    clipnorm=0.5            # Light gradient clipping
)

# Keep original loss function (MAE was working!)
model.compile(
    optimizer=optimizer,
    loss='mae',              # Keep original loss
    metrics=['mae', 'mse']
)

# Use moderate batch size
batch_size = 8  # Between original 4 and aggressive 16
train_data = make_dataset("./data", split="train", batch_size=batch_size)
val_data = make_dataset("./data", split="valid", batch_size=batch_size)
test_data = make_dataset("./data", split="test", batch_size=batch_size)

# Setup balanced callbacks
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

callbacks = [
    # Model checkpointing
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./weights/best_balanced_model.h5',
        monitor='val_mae',
        mode='min',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),
    
    # Moderate early stopping
    tf.keras.callbacks.EarlyStopping(
        monitor='val_mae',
        mode='min',
        patience=12,  # Between original 15 and aggressive 8
        restore_best_weights=True,
        verbose=1
    ),
    
    # Keep original learning rate schedule but with lower factor
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_mae',
        mode='min',
        factor=0.7,     # Less aggressive than 0.5
        patience=6,     # Slightly reduced from 8
        min_lr=1e-6,
        verbose=1
    ),
    
    # Log training progress
    tf.keras.callbacks.CSVLogger(f'./weights/balanced_training_log_{timestamp}.csv')
]

print("Starting balanced training...")
print(f"Key changes from original:")
print(f"• Dropout: 0.1 → 0.2 (moderate increase)")
print(f"• L2 regularization: None → 0.001 (light)")
print(f"• Learning rate: 0.001 → 0.0008 (slightly lower)")
print(f"• Batch size: 4 → 8 (moderate increase)")
print(f"• Gradient clipping: None → 0.5 (light)")
print(f"• Early stopping patience: 15 → 12")

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on test set
print("\n" + "="*50)
print("FINAL EVALUATION - BALANCED MODEL")
print("="*50)
test_results = model.evaluate(test_data, verbose=1)
print(f"\nTest Results:")
for metric_name, value in zip(model.metrics_names, test_results):
    print(f"  {metric_name}: {value:.4f}")

# Save final model
final_model_path = f'./weights/final_balanced_model_{timestamp}.h5'
model.save(final_model_path)
print(f"\nFinal balanced model saved to: {final_model_path}")

# Save training history
import json
history_path = f'./weights/balanced_training_history_{timestamp}.json'
with open(history_path, 'w') as f:
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    json.dump(history_dict, f, indent=2)
print(f"Training history saved to: {history_path}")

# Compare with both previous results
print("\n" + "="*50)
print("COMPARISON WITH ALL PREVIOUS RESULTS")
print("="*50)
original_best_val_mae = 1.6302
improved_best_val_mae = 1.7252  # The over-regularized model
current_best_val_mae = min(history.history['val_mae'])

print(f"Original model best validation MAE: {original_best_val_mae:.4f}")
print(f"Over-regularized model validation MAE: {improved_best_val_mae:.4f}")
print(f"Balanced model best validation MAE: {current_best_val_mae:.4f}")

improvement_vs_original = original_best_val_mae - current_best_val_mae
improvement_vs_overreg = improved_best_val_mae - current_best_val_mae

print(f"\nImprovement vs original: {improvement_vs_original:.4f} ({'Better' if improvement_vs_original > 0 else 'Worse'})")
print(f"Improvement vs over-regularized: {improvement_vs_overreg:.4f} ({'Better' if improvement_vs_overreg > 0 else 'Worse'})")

# Overfitting analysis
final_train_mae = history.history['mae'][-1]
final_val_mae = history.history['val_mae'][-1]
overfitting_gap = final_val_mae - final_train_mae

print(f"\nOverfitting Analysis:")
print(f"Final training MAE: {final_train_mae:.4f}")
print(f"Final validation MAE: {final_val_mae:.4f}")
print(f"Overfitting gap: {overfitting_gap:.4f}")

# Compare gaps
original_gap = 1.63 - 0.61  # Approximate from original training
improved_gap = 0.2293       # From over-regularized model
current_gap = overfitting_gap

print(f"\nGap Comparison:")
print(f"Original gap: ~{original_gap:.2f}")
print(f"Over-regularized gap: {improved_gap:.4f}")
print(f"Balanced model gap: {current_gap:.4f}")
print(f"Gap reduction vs original: {original_gap - current_gap:.4f}")








