from models.improved_msamodel import ImprovedMSAModel
from utils.data_loader import make_dataset
import tensorflow as tf
import os
from datetime import datetime
import numpy as np

# Configure TensorFlow for stability
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_visible_devices([], 'GPU')  # Force CPU for stability

# Create weights directory
os.makedirs('./weights', exist_ok=True)

# Improved model parameters
T = 20
text_dim = 300
audio_dim = 74
video_dim = 713  # MOSEI: 713, MOSI: 47
model_dim = 64
heads = 4
ff_dim = 256
n_layers = 2
n_layers_fuse = 1

# Build improved model with better regularization
model = ImprovedMSAModel(
    seq_len=T,
    text_dim=text_dim,
    audio_dim=audio_dim,
    video_dim=video_dim,
    model_dim=model_dim,
    num_heads=heads,
    ff_dim=ff_dim,
    n_layers_mod=n_layers,
    n_layers_fuse=n_layers_fuse,
    dropout_rate=0.4,        # Increased dropout
    l2_reg=0.01,            # Added L2 regularization
    adaptive_fusion=True
)

# Warm up the model
dummy_t = tf.zeros((1, T, text_dim))
dummy_a = tf.zeros((1, T, audio_dim))
dummy_v = tf.zeros((1, T, video_dim))
_ = model((dummy_t, dummy_a, dummy_v), training=False)
model.summary()

# Improved optimizer configuration
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=0.0005,   # Lower initial learning rate
    weight_decay=0.01,      # Weight decay for better generalization
    beta_1=0.9,
    beta_2=0.999,
    clipnorm=1.0           # Gradient clipping
)

# Compile with label smoothing
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.Huber(delta=1.0),  # More robust to outliers than MAE
    metrics=['mae', 'mse']
)

# Load datasets with larger batch size for stability
batch_size = 16  # Increased from 4
train_data = make_dataset("./data", split="train", batch_size=batch_size)
val_data = make_dataset("./data", split="valid", batch_size=batch_size)
test_data = make_dataset("./data", split="test", batch_size=batch_size)

# Improved learning rate schedule
def cosine_decay_with_warmup(epoch, lr):
    """Cosine decay with warmup for better training dynamics."""
    warmup_epochs = 5
    total_epochs = 50
    
    if epoch < warmup_epochs:
        return 0.0001 + (0.0005 - 0.0001) * epoch / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.0005 * 0.5 * (1 + np.cos(np.pi * progress))

# Setup improved callbacks
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

callbacks = [
    # Model checkpointing - track both loss and MAE
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./weights/best_improved_model_loss.h5',
        monitor='val_loss',  # Monitor validation Huber loss
        mode='min',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),
    
    # Additional checkpoint for MAE
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./weights/best_improved_model_mae.h5',
        monitor='val_mae',
        mode='min',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),
    
    # Early stopping based on validation Huber loss
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor validation Huber loss instead of MAE
        mode='min',
        patience=8,  # Reduced from 15
        restore_best_weights=True,
        verbose=1
    ),
    
    # Custom learning rate schedule
    tf.keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup, verbose=1),
    
    # Log training progress
    tf.keras.callbacks.CSVLogger(f'./weights/improved_training_log_{timestamp}.csv'),
    
    # Reduce LR on plateau - also track loss
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # Monitor validation Huber loss
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    
    # TensorBoard logging
    tf.keras.callbacks.TensorBoard(
        log_dir=f'./logs/improved_{timestamp}',
        histogram_freq=1,
        write_graph=True
    )
]

print("Starting improved training...")
print(f"Batch size: {batch_size}")
print(f"Dropout rate: 0.4")
print(f"L2 regularization: 0.01")
print(f"Optimizer: AdamW with weight decay")
print(f"Loss function: Huber loss")
print(f"Early stopping: Monitoring validation Huber loss (val_loss)")
print(f"Model checkpoints: Both val_loss and val_mae")

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,  # Reduced epochs due to better early stopping
    callbacks=callbacks,
    verbose=1
)

# Evaluate on test set
print("\n" + "="*50)
print("FINAL EVALUATION ON IMPROVED MODEL")
print("="*50)
test_results = model.evaluate(test_data, verbose=1)
print(f"\nTest Results:")
for metric_name, value in zip(model.metrics_names, test_results):
    print(f"  {metric_name}: {value:.4f}")

# Save final model
final_model_path = f'./weights/final_improved_model_{timestamp}.h5'
model.save(final_model_path)
print(f"\nFinal improved model saved to: {final_model_path}")

# Save training history
import json
history_path = f'./weights/improved_training_history_{timestamp}.json'
with open(history_path, 'w') as f:
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    json.dump(history_dict, f, indent=2)
print(f"Training history saved to: {history_path}")

# Compare with previous best results
print("\n" + "="*50)
print("COMPARISON WITH PREVIOUS RESULTS")
print("="*50)
previous_best_val_mae = 1.6302  # From previous training
current_best_val_mae = min(history.history['val_mae'])
improvement = previous_best_val_mae - current_best_val_mae

print(f"Previous best validation MAE: {previous_best_val_mae:.4f}")
print(f"Current best validation MAE: {current_best_val_mae:.4f}")
print(f"Improvement: {improvement:.4f} ({'Better' if improvement > 0 else 'Worse'})")

# Overfitting analysis
final_train_mae = history.history['mae'][-1]
final_val_mae = history.history['val_mae'][-1]
overfitting_gap = final_val_mae - final_train_mae

print(f"\nOverfitting Analysis:")
print(f"Final training MAE: {final_train_mae:.4f}")
print(f"Final validation MAE: {final_val_mae:.4f}")
print(f"Overfitting gap: {overfitting_gap:.4f}")
print(f"Gap reduction vs previous: {(1.63 - 0.61) - overfitting_gap:.4f}")





