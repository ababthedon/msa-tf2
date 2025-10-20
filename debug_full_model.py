"""
Test Full Model Step by Step
"""

import tensorflow as tf
import sys
import os

print("Testing full MSASeqLevelModel...")

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU: {gpus[0]}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print("\n" + "="*70)
print("TEST 1: Import Model")
print("="*70)

try:
    from models import MSASeqLevelModel
    print("✓ Model imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("TEST 2: Create Model (Small)")
print("="*70)

try:
    model = MSASeqLevelModel(
        seq_len=20,
        text_dim=300,
        audio_dim=74,
        video_dim=713,
        model_dim=64,
        num_heads=2,
        ff_dim=128,
        n_layers_mod=1,
        n_layers_fuse=1,
        dropout_rate=0.1
    )
    print("✓ Model created")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("TEST 3: Forward Pass (Inference Mode)")
print("="*70)

try:
    t = tf.random.normal((2, 20, 300))
    a = tf.random.normal((2, 20, 74))
    v = tf.random.normal((2, 20, 713))
    
    output = model((t, a, v), training=False)
    print(f"✓ Forward pass works: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("TEST 4: Forward Pass (Training Mode)")
print("="*70)

try:
    output = model((t, a, v), training=True)
    print(f"✓ Training mode forward pass works: {output.shape}")
except Exception as e:
    print(f"✗ Training mode forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("TEST 5: Compile Model")
print("="*70)

try:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mae',
        metrics=['mae']
    )
    print("✓ Model compiled")
except Exception as e:
    print(f"✗ Compilation failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("TEST 6: Create Dummy Dataset")
print("="*70)

try:
    # Create small dummy dataset
    t_data = tf.random.normal((10, 20, 300))
    a_data = tf.random.normal((10, 20, 74))
    v_data = tf.random.normal((10, 20, 713))
    y_data = tf.random.uniform((10, 1), minval=-3, maxval=3)
    
    dataset = tf.data.Dataset.from_tensor_slices(
        ((t_data, a_data, v_data), y_data)
    )
    dataset = dataset.batch(2).prefetch(tf.data.AUTOTUNE)
    
    print("✓ Dataset created: 10 samples, batch_size=2")
except Exception as e:
    print(f"✗ Dataset creation failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("TEST 7: Train for 1 Step (Eager Mode)")
print("="*70)

try:
    # Get one batch
    for batch in dataset.take(1):
        (t_batch, a_batch, v_batch), y_batch = batch
        
        with tf.GradientTape() as tape:
            predictions = model((t_batch, a_batch, v_batch), training=True)
            loss = tf.reduce_mean(tf.abs(predictions - y_batch))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        print(f"✓ One training step (eager) works")
        print(f"  Loss: {loss.numpy():.4f}")
        print(f"  Gradients computed: {len(gradients)} tensors")
        break
except Exception as e:
    print(f"✗ Eager training step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("TEST 8: model.fit() for 1 Epoch on Dummy Data")
print("="*70)

try:
    print("Starting model.fit()...")
    history = model.fit(dataset, epochs=1, verbose=1)
    print("✓ model.fit() completed successfully!")
    print(f"  Final loss: {history.history['loss'][0]:.4f}")
except Exception as e:
    print(f"✗ model.fit() failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n⚠ The crash occurs during model.fit()")
    print("   This suggests an issue with the training loop on Metal backend")
    sys.exit(1)

print("\n" + "="*70)
print("TEST 9: model.fit() for 2 Epochs")
print("="*70)

try:
    history = model.fit(dataset, epochs=2, verbose=1)
    print("✓ Multi-epoch training works!")
except Exception as e:
    print(f"✗ Multi-epoch training failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nThe full model works correctly with dummy data.")
print("The issue might be with:")
print("  1. Real data from data_loader")
print("  2. Larger batch sizes")
print("  3. Longer training runs")





