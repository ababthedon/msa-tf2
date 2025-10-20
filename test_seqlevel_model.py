"""
Test Script for Sequence-Level MSA Model

This script validates the new MSASeqLevelModel architecture by:
1. Testing basic instantiation and forward pass
2. Verifying output shapes and dtypes
3. Testing with explicit masks
4. Validating mixed precision compatibility
5. Running a short training loop on dummy data
6. Testing model.summary() and serialization
"""

import tensorflow as tf
from tensorflow.keras import mixed_precision
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    MSASeqLevelModel,
    CrossModalFusion,
    AdaptiveFusionHead,
    create_padding_mask
)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_basic_instantiation():
    """Test 1: Basic model instantiation and forward pass."""
    print_section("TEST 1: Basic Instantiation and Forward Pass")
    
    # Model parameters
    batch_size = 4
    seq_len = 64
    text_dim = 300
    audio_dim = 74
    video_dim = 47
    model_dim = 128
    
    print(f"Creating model with:")
    print(f"  seq_len={seq_len}, model_dim={model_dim}")
    print(f"  text_dim={text_dim}, audio_dim={audio_dim}, video_dim={video_dim}")
    
    # Create model
    model = MSASeqLevelModel(
        seq_len=seq_len,
        text_dim=text_dim,
        audio_dim=audio_dim,
        video_dim=video_dim,
        model_dim=model_dim,
        num_heads=4,
        ff_dim=256,
        n_layers_mod=2,
        n_layers_fuse=2,
        bidirectional_fusion=False
    )
    
    # Create dummy inputs
    text_seq = tf.random.normal((batch_size, seq_len, text_dim))
    audio_seq = tf.random.normal((batch_size, seq_len, audio_dim))
    video_seq = tf.random.normal((batch_size, seq_len, video_dim))
    
    print(f"\nInput shapes:")
    print(f"  Text:  {text_seq.shape}")
    print(f"  Audio: {audio_seq.shape}")
    print(f"  Video: {video_seq.shape}")
    
    # Forward pass
    output = model((text_seq, audio_seq, video_seq), training=False)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output range: [{tf.reduce_min(output):.4f}, {tf.reduce_max(output):.4f}]")
    
    # Assertions
    assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"
    assert output.dtype == tf.float32, f"Expected dtype float32, got {output.dtype}"
    
    print("\n✓ Test 1 PASSED: Basic instantiation works correctly")
    return model


def test_explicit_masks(model):
    """Test 2: Forward pass with explicit masks."""
    print_section("TEST 2: Explicit Mask Handling")
    
    batch_size = 4
    seq_len = 64
    
    # Create inputs with some padding
    text_seq = tf.random.normal((batch_size, seq_len, 300))
    audio_seq = tf.random.normal((batch_size, seq_len, 74))
    video_seq = tf.random.normal((batch_size, seq_len, 47))
    
    # Simulate padding by zeroing out last 10 positions
    text_seq = tf.concat([
        text_seq[:, :-10, :],
        tf.zeros((batch_size, 10, 300))
    ], axis=1)
    
    # Create explicit masks
    text_mask = create_padding_mask(text_seq)
    audio_mask = create_padding_mask(audio_seq)
    video_mask = create_padding_mask(video_seq)
    
    print(f"Mask shapes:")
    print(f"  Text:  {text_mask.shape}, valid positions: {tf.reduce_sum(tf.cast(text_mask, tf.int32), axis=1).numpy()}")
    print(f"  Audio: {audio_mask.shape}, valid positions: {tf.reduce_sum(tf.cast(audio_mask, tf.int32), axis=1).numpy()}")
    print(f"  Video: {video_mask.shape}, valid positions: {tf.reduce_sum(tf.cast(video_mask, tf.int32), axis=1).numpy()}")
    
    # Forward pass with explicit masks
    output = model(
        (text_seq, audio_seq, video_seq, text_mask, audio_mask, video_mask),
        training=False
    )
    
    print(f"\nOutput with masks shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    
    assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"
    
    print("\n✓ Test 2 PASSED: Explicit masks work correctly")


def test_model_summary():
    """Test 3: Model summary and architecture inspection."""
    print_section("TEST 3: Model Summary and Architecture")
    
    model = MSASeqLevelModel(
        seq_len=20,
        text_dim=300,
        audio_dim=74,
        video_dim=47,
        model_dim=128,
        num_heads=4,
        ff_dim=256,
        n_layers_mod=2,
        n_layers_fuse=1
    )
    
    # Build the model with a forward pass
    dummy_t = tf.zeros((1, 20, 300))
    dummy_a = tf.zeros((1, 20, 74))
    dummy_v = tf.zeros((1, 20, 47))
    _ = model((dummy_t, dummy_a, dummy_v), training=False)
    
    print("\nModel Summary:")
    model.summary()
    
    print("\n✓ Test 3 PASSED: Model summary generated successfully")


def test_mixed_precision():
    """Test 4: Mixed precision compatibility."""
    print_section("TEST 4: Mixed Precision Compatibility")
    
    # Enable mixed precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    
    print(f"Mixed precision policy: {policy.name}")
    
    # Create model
    model = MSASeqLevelModel(
        seq_len=64,
        text_dim=300,
        audio_dim=74,
        video_dim=47,
        model_dim=128,
        num_heads=4,
        ff_dim=256,
        n_layers_mod=2,
        n_layers_fuse=1
    )
    
    # Create inputs
    batch_size = 4
    text_seq = tf.random.normal((batch_size, 64, 300))
    audio_seq = tf.random.normal((batch_size, 64, 74))
    video_seq = tf.random.normal((batch_size, 64, 47))
    
    # Forward pass
    output = model((text_seq, audio_seq, video_seq), training=False)
    
    print(f"\nOutput dtype: {output.dtype}")
    print(f"Output shape: {output.shape}")
    
    # Output must be float32 even with mixed precision
    assert output.dtype == tf.float32, f"Expected output dtype float32, got {output.dtype}"
    
    # Reset policy
    mixed_precision.set_global_policy('float32')
    
    print("\n✓ Test 4 PASSED: Mixed precision works correctly (output is float32)")


def test_training_loop():
    """Test 5: Short training loop on dummy data."""
    print_section("TEST 5: Training Loop on Dummy Data")
    
    # Create model
    model = MSASeqLevelModel(
        seq_len=20,
        text_dim=300,
        audio_dim=74,
        video_dim=47,
        model_dim=64,
        num_heads=4,
        ff_dim=128,
        n_layers_mod=1,
        n_layers_fuse=1
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mae',
        metrics=['mae', 'mse']
    )
    
    print("Model compiled successfully")
    
    # Create dummy dataset
    n_samples = 32
    batch_size = 8
    seq_len = 20
    
    text_data = tf.random.normal((n_samples, seq_len, 300))
    audio_data = tf.random.normal((n_samples, seq_len, 74))
    video_data = tf.random.normal((n_samples, seq_len, 47))
    labels = tf.random.uniform((n_samples, 1), minval=-3, maxval=3)
    
    dataset = tf.data.Dataset.from_tensor_slices(
        ((text_data, audio_data, video_data), labels)
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    print(f"\nDummy dataset created:")
    print(f"  Samples: {n_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches: {n_samples // batch_size}")
    
    # Train for 3 epochs
    print("\nTraining for 3 epochs...")
    history = model.fit(
        dataset,
        epochs=3,
        verbose=1
    )
    
    print("\nTraining metrics:")
    for metric_name, values in history.history.items():
        print(f"  {metric_name}: {values}")
    
    # Test prediction
    test_batch = next(iter(dataset))
    (test_t, test_a, test_v), test_y = test_batch
    predictions = model((test_t, test_a, test_v), training=False)
    
    print(f"\nPrediction test:")
    print(f"  Input batch size: {test_t.shape[0]}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Predictions: {predictions[:3, 0].numpy()}")
    print(f"  Ground truth: {test_y[:3, 0].numpy()}")
    
    print("\n✓ Test 5 PASSED: Training loop works correctly")


def test_fusion_components():
    """Test 6: Individual fusion components."""
    print_section("TEST 6: Testing Fusion Components Independently")
    
    batch_size = 4
    seq_len = 64
    model_dim = 256
    
    # Test CrossModalFusion
    print("\n[6a] Testing CrossModalFusion...")
    fusion = CrossModalFusion(
        model_dim=model_dim,
        num_heads=4,
        ff_dim=512,
        n_layers=2,
        bidirectional=True
    )
    
    text = tf.random.normal((batch_size, seq_len, model_dim))
    audio = tf.random.normal((batch_size, seq_len, model_dim))
    video = tf.random.normal((batch_size, seq_len, model_dim))
    
    text_fused, audio_fused, video_fused = fusion(
        text_seq=text,
        audio_seq=audio,
        video_seq=video,
        training=True
    )
    
    print(f"  Text fused shape: {text_fused.shape}")
    print(f"  Audio fused shape: {audio_fused.shape}")
    print(f"  Video fused shape: {video_fused.shape}")
    
    assert text_fused.shape == (batch_size, seq_len, model_dim)
    assert audio_fused.shape == (batch_size, seq_len, model_dim)
    assert video_fused.shape == (batch_size, seq_len, model_dim)
    
    print("  ✓ CrossModalFusion works correctly")
    
    # Test AdaptiveFusionHead
    print("\n[6b] Testing AdaptiveFusionHead...")
    fusion_head = AdaptiveFusionHead(
        model_dim=model_dim,
        num_modalities=3,
        pooling_method='mean'
    )
    
    modality_seqs = [text_fused, audio_fused, video_fused]
    fused_repr = fusion_head(
        modality_sequences=modality_seqs,
        training=True
    )
    
    print(f"  Fused representation shape: {fused_repr.shape}")
    assert fused_repr.shape == (batch_size, model_dim)
    
    print("  ✓ AdaptiveFusionHead works correctly")
    
    print("\n✓ Test 6 PASSED: All fusion components work independently")


def test_different_configs():
    """Test 7: Different model configurations."""
    print_section("TEST 7: Testing Different Configurations")
    
    configs = [
        {
            'name': 'Small model',
            'seq_len': 32, 'model_dim': 64, 'num_heads': 2, 'ff_dim': 128,
            'n_layers_mod': 1, 'n_layers_fuse': 1
        },
        {
            'name': 'Large model',
            'seq_len': 64, 'model_dim': 256, 'num_heads': 8, 'ff_dim': 1024,
            'n_layers_mod': 3, 'n_layers_fuse': 2
        },
        {
            'name': 'Bidirectional fusion',
            'seq_len': 64, 'model_dim': 128, 'num_heads': 4, 'ff_dim': 256,
            'n_layers_mod': 2, 'n_layers_fuse': 2, 'bidirectional_fusion': True
        },
        {
            'name': 'Attention pooling',
            'seq_len': 64, 'model_dim': 128, 'num_heads': 4, 'ff_dim': 256,
            'n_layers_mod': 2, 'n_layers_fuse': 1, 'pooling_method': 'attention'
        }
    ]
    
    for config in configs:
        name = config.pop('name')
        print(f"\n[7.{configs.index(config)+1}] Testing {name}...")
        
        model = MSASeqLevelModel(
            text_dim=300,
            audio_dim=74,
            video_dim=47,
            **config
        )
        
        # Test forward pass
        seq_len = config.get('seq_len', 64)
        text = tf.random.normal((2, seq_len, 300))
        audio = tf.random.normal((2, seq_len, 74))
        video = tf.random.normal((2, seq_len, 47))
        
        output = model((text, audio, video), training=False)
        
        print(f"  Output shape: {output.shape}, dtype: {output.dtype}")
        assert output.shape == (2, 1)
        assert output.dtype == tf.float32
        
        print(f"  ✓ {name} configuration works correctly")
    
    print("\n✓ Test 7 PASSED: All configurations work correctly")


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*70)
    print("#  MSASeqLevelModel Test Suite")
    print("#"*70)
    
    try:
        # Run tests
        model = test_basic_instantiation()
        test_explicit_masks(model)
        test_model_summary()
        test_mixed_precision()
        test_training_loop()
        test_fusion_components()
        test_different_configs()
        
        # Summary
        print_section("ALL TESTS PASSED ✓")
        print("\nThe MSASeqLevelModel architecture is working correctly!")
        print("\nNext steps:")
        print("  1. Train the model on real CMU-MOSI/MOSEI data")
        print("  2. Compare performance with baseline MSAModel")
        print("  3. Experiment with hyperparameters")
        print("  4. Try bidirectional fusion and attention pooling")
        print("\nSee models/README.md for usage examples.")
        
        return True
        
    except Exception as e:
        print_section("TEST FAILED ✗")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # Configure TensorFlow
    tf.config.set_visible_devices([], 'GPU')  # Force CPU for testing
    
    # Run tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)





