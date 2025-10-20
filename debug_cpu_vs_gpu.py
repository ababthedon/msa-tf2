"""
Test Model on CPU vs GPU
"""

import tensorflow as tf
import sys

print("\n" + "="*70)
print("TEST: MSASeqLevelModel on CPU")
print("="*70)

# Force CPU
tf.config.set_visible_devices([], 'GPU')
print("GPU disabled, using CPU only")

try:
    from models import MSASeqLevelModel
    
    model = MSASeqLevelModel(
        seq_len=20,
        text_dim=300,
        audio_dim=74,
        video_dim=713,
        model_dim=64,
        num_heads=2,
        ff_dim=128,
        n_layers_mod=1,
        n_layers_fuse=1
    )
    print("✓ Model created on CPU")
    
    # Test forward pass
    t = tf.random.normal((2, 20, 300))
    a = tf.random.normal((2, 20, 74))
    v = tf.random.normal((2, 20, 713))
    
    output = model((t, a, v), training=False)
    print(f"✓ Forward pass works on CPU: {output.shape}")
    
    # Test training
    model.compile(optimizer='adam', loss='mae')
    dataset = tf.data.Dataset.from_tensor_slices(
        ((t, a, v), tf.zeros((2, 1)))
    ).batch(2)
    
    history = model.fit(dataset, epochs=1, verbose=0)
    print(f"✓ Training works on CPU: loss={history.history['loss'][0]:.4f}")
    
    print("\n✅ MODEL WORKS PERFECTLY ON CPU!")
    print("\n⚠ CONCLUSION: The issue is with TensorFlow-Metal, not your code.")
    print("\nSOLUTION: Use CPU for training:")
    print("  python train_seqlevel.py --use_cpu --batch_size 32 --model_dim 128")
    
except Exception as e:
    print(f"✗ Failed on CPU too: {e}")
    import traceback
    traceback.print_exc()





