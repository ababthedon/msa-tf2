"""
Debug Training Script - Progressive Testing

This script tests each component of the model progressively to isolate
where the malloc error occurs.
"""

import tensorflow as tf
import numpy as np
from models import MSASeqLevelModel
from utils.data_loader import make_dataset
import sys

print("\n" + "="*70)
print("Debug Training - Progressive Component Testing")
print("="*70)

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n✓ GPU detected: {gpus[0]}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("\n⚠ No GPU detected, using CPU")

print("\n" + "="*70)
print("TEST 1: Create Model")
print("="*70)

try:
    model = MSASeqLevelModel(
        seq_len=20,
        text_dim=300,
        audio_dim=74,
        video_dim=713,
        model_dim=64,  # Small model for testing
        num_heads=2,
        ff_dim=128,
        n_layers_mod=1,
        n_layers_fuse=1
    )
    print("✓ Model created successfully")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("TEST 2: Build Model with Dummy Data")
print("="*70)

try:
    # Create dummy inputs
    dummy_t = tf.zeros((2, 20, 300), dtype=tf.float32)
    dummy_a = tf.zeros((2, 20, 74), dtype=tf.float32)
    dummy_v = tf.zeros((2, 20, 713), dtype=tf.float32)
    
    # Forward pass
    output = model((dummy_t, dummy_a, dummy_v), training=False)
    print(f"✓ Forward pass successful: output shape = {output.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("TEST 3: Training Mode Forward Pass")
print("="*70)

try:
    output = model((dummy_t, dummy_a, dummy_v), training=True)
    print(f"✓ Training mode forward pass successful")
except Exception as e:
    print(f"✗ Training mode forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("TEST 4: Model Compilation")
print("="*70)

try:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mae',
        metrics=['mae']
    )
    print("✓ Model compiled successfully")
except Exception as e:
    print(f"✗ Model compilation failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("TEST 5: Single Batch Training on Dummy Data")
print("="*70)

try:
    # Create dummy dataset
    dummy_dataset = tf.data.Dataset.from_tensor_slices(
        ((dummy_t, dummy_a, dummy_v), tf.zeros((2, 1), dtype=tf.float32))
    ).batch(2)
    
    # Train for 1 step
    history = model.fit(dummy_dataset, epochs=1, verbose=1)
    print("✓ Single batch training successful")
except Exception as e:
    print(f"✗ Single batch training failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n⚠ ERROR OCCURRED HERE - This is where the malloc error happens")
    sys.exit(1)

print("\n" + "="*70)
print("TEST 6: Load Real Data (First Batch Only)")
print("="*70)

try:
    # Load just one batch of real data
    train_data = make_dataset('./data', split='train', batch_size=2)
    
    # Get first batch
    for batch in train_data.take(1):
        (t, a, v), y = batch
        print(f"✓ Real data loaded: shapes = {t.shape}, {a.shape}, {v.shape}")
        
        # Test forward pass with real data
        output = model((t, a, v), training=False)
        print(f"✓ Forward pass with real data successful: {output.shape}")
        break
except Exception as e:
    print(f"✗ Real data test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("TEST 7: Training on Real Data (3 Steps)")
print("="*70)

try:
    # Create small dataset
    small_train_data = make_dataset('./data', split='train', batch_size=4)
    small_train_data = small_train_data.take(3)  # Only 3 batches
    
    # Train for 1 epoch (3 steps)
    history = model.fit(small_train_data, epochs=1, verbose=1)
    print("✓ Training on real data successful")
except Exception as e:
    print(f"✗ Training on real data failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n⚠ ERROR OCCURRED HERE - Malloc error during training")
    sys.exit(1)

print("\n" + "="*70)
print("TEST 8: Multiple Epochs on Small Dataset")
print("="*70)

try:
    small_train_data = make_dataset('./data', split='train', batch_size=4)
    small_train_data = small_train_data.take(5)  # Only 5 batches
    
    # Train for 2 epochs
    history = model.fit(small_train_data, epochs=2, verbose=1)
    print("✓ Multi-epoch training successful")
except Exception as e:
    print(f"✗ Multi-epoch training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nIf all tests passed, the issue might be:")
print("  1. Large batch size causing memory corruption")
print("  2. Large model size causing Metal backend issues")
print("  3. Long training runs triggering a memory leak")
print("\nTry training with:")
print("  python train_seqlevel.py --batch_size 4 --model_dim 64 --epochs 5")





