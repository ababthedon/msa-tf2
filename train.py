from models.msamodel import MSAModel
from utils.data_loader import make_dataset
import tensorflow as tf
import os
from datetime import datetime

# Additional stability settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging

# Configure GPU (Apple Silicon MPS backend)
print("\n" + "="*70)
print("GPU Configuration")
print("="*70)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n✓ Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - {gpu}")
    
    # Enable memory growth
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  ✓ Enabled memory growth for {gpu.name}")
        except RuntimeError as e:
            print(f"  ⚠ Could not set memory growth: {e}")
    
    print("\n✓ GPU acceleration enabled (Metal Performance Shaders)")
else:
    print("\n⚠ No GPU found. Training will use CPU.")
    print("  For Apple Silicon, ensure tensorflow-metal is installed:")
    print("  pip install tensorflow-metal")

# Create weights directory if it doesn't exist
os.makedirs('./weights', exist_ok=True)


# Define model and data parameters (customise as needed)
T = 20                  # sequence length
text_dim = 300         # input feature size for text embeddings
audio_dim = 74
video_dim = 713        # MOSEI: 713, MOSI: 47
model_dim = 64         # transformer hidden size
heads = 4              # number of attention heads
ff_dim = 256           # feed-forward network inner dimension
n_layers = 2           # number of transformer layers
n_layers_fuse = 1      # fused layer

# Build the multimodal model
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
    adaptive_fusion=True  # or True if you want adaptive weighting
)
# Warm up the model with a dummy batch to build its weights
dummy_t = tf.zeros((1, T, text_dim))
dummy_a = tf.zeros((1, T, audio_dim))
dummy_v = tf.zeros((1, T, video_dim))
_ = model((dummy_t, dummy_a, dummy_v), training=False)
# Now the model is built; display summary
model.summary()
# Compile model with better optimizer configuration
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
    loss='mae',
    metrics=['mae', 'mse']
)
# For a quick smoke test you could replace make_dataset with just t and y arrays:
# t_train, y_train = load_text_and_labels_somehow()
# model.fit(t_train, y_train, batch_size=64, epochs=3, validation_split=0.1)
#ds = make_dataset('./data', split='train', batch_size=64)       # yields ((t,a,v), y)

# Load datasets with very small batch sizes for stability
batch_size = 4  # Very small batch size to avoid memory issues
train_data = make_dataset("./data", split="train", batch_size=batch_size)
val_data = make_dataset("./data", split="valid", batch_size=batch_size)
test_data = make_dataset("./data", split="test", batch_size=batch_size)

# Setup callbacks for training
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

callbacks = [
    # Save best model based on validation MAE
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./weights/best_model_val_mae.h5',
        monitor='val_mae',
        mode='min',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),
    # Save model every epoch
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f'./weights/model_epoch_{{epoch:02d}}_val_mae_{{val_mae:.4f}}.h5',
        monitor='val_mae',
        mode='min',
        save_best_only=False,
        save_weights_only=False,
        verbose=0
    ),
    # Early stopping to prevent overfitting
    tf.keras.callbacks.EarlyStopping(
        monitor='val_mae',
        mode='min',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate on plateau
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_mae',
        mode='min',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    ),
    # Log training progress
    tf.keras.callbacks.CSVLogger(f'./weights/training_log_{timestamp}.csv')
]

print("Starting training...")
print(f"Training samples: {len(list(train_data.unbatch()))}")
print(f"Validation samples: {len(list(val_data.unbatch()))}")
print(f"Test samples: {len(list(test_data.unbatch()))}")

# Train model with proper monitoring
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on test set
print("\n" + "="*50)
print("FINAL EVALUATION ON TEST SET")
print("="*50)
test_results = model.evaluate(test_data, verbose=1)
print(f"\nTest Results:")
for metric_name, value in zip(model.metrics_names, test_results):
    print(f"  {metric_name}: {value:.4f}")

# Save final model
final_model_path = f'./weights/final_model_{timestamp}.h5'
model.save(final_model_path)
print(f"\nFinal model saved to: {final_model_path}")

# Save training history
import json
history_path = f'./weights/training_history_{timestamp}.json'
with open(history_path, 'w') as f:
    # Convert numpy arrays to lists for JSON serialization
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    json.dump(history_dict, f, indent=2)
print(f"Training history saved to: {history_path}")
