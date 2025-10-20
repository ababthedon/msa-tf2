"""
Test Individual Components to Find the Culprit
"""

import tensorflow as tf
import sys

print("Testing individual components...")
print("TensorFlow:", tf.__version__)

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU: {gpus[0]}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print("\n" + "="*70)
print("TEST 1: Simple Dense Layer")
print("="*70)

try:
    x = tf.random.normal((2, 20, 300))
    layer = tf.keras.layers.Dense(256)
    y = layer(x)
    print(f"✓ Dense layer works: {y.shape}")
except Exception as e:
    print(f"✗ Dense layer failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("TEST 2: LayerNormalization")
print("="*70)

try:
    x = tf.random.normal((2, 20, 256))
    layer = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    y = layer(x)
    print(f"✓ LayerNormalization works: {y.shape}")
except Exception as e:
    print(f"✗ LayerNormalization failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("TEST 3: MultiHeadAttention (key_dim=64)")
print("="*70)

try:
    x = tf.random.normal((2, 20, 256))
    layer = tf.keras.layers.MultiHeadAttention(
        num_heads=4,
        key_dim=64,  # model_dim // num_heads
        dropout=0.1
    )
    y = layer(query=x, key=x, value=x, training=False)
    print(f"✓ MultiHeadAttention works: {y.shape}")
except Exception as e:
    print(f"✗ MultiHeadAttention failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("TEST 4: MultiHeadAttention with Mask")
print("="*70)

try:
    x = tf.random.normal((2, 20, 256))
    mask = tf.ones((2, 20), dtype=tf.bool)
    mask_expanded = mask[:, tf.newaxis, :]  # (2, 1, 20)
    
    layer = tf.keras.layers.MultiHeadAttention(
        num_heads=4,
        key_dim=64,
        dropout=0.1
    )
    y = layer(query=x, key=x, value=x, attention_mask=mask_expanded, training=False)
    print(f"✓ MultiHeadAttention with mask works: {y.shape}")
except Exception as e:
    print(f"✗ MultiHeadAttention with mask failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("TEST 5: TransformerBlock (from our model)")
print("="*70)

try:
    from models.blocks import TransformerBlock
    
    x = tf.random.normal((2, 20, 256))
    mask = tf.ones((2, 20), dtype=tf.bool)
    
    block = TransformerBlock(model_dim=256, num_heads=4, ff_dim=512)
    y = block(x, mask=mask, training=False)
    print(f"✓ TransformerBlock works: {y.shape}")
except Exception as e:
    print(f"✗ TransformerBlock failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n⚠ TransformerBlock is the culprit!")
    sys.exit(1)

print("\n" + "="*70)
print("TEST 6: TransformerBlock Training Mode")
print("="*70)

try:
    from models.blocks import TransformerBlock
    
    x = tf.random.normal((2, 20, 256))
    mask = tf.ones((2, 20), dtype=tf.bool)
    
    block = TransformerBlock(model_dim=256, num_heads=4, ff_dim=512, dropout_rate=0.1)
    y = block(x, mask=mask, training=True)
    print(f"✓ TransformerBlock training mode works: {y.shape}")
except Exception as e:
    print(f"✗ TransformerBlock training mode failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("TEST 7: ModalityEncoder")
print("="*70)

try:
    from models.msa_seqlevel import ModalityEncoder
    
    x = tf.random.normal((2, 20, 300))
    mask = tf.ones((2, 20), dtype=tf.bool)
    
    encoder = ModalityEncoder(
        input_dim=300,
        model_dim=256,
        seq_len=20,
        num_heads=4,
        ff_dim=512,
        n_layers=1,
        dropout_rate=0.1
    )
    y = encoder(x, mask=mask, training=False)
    print(f"✓ ModalityEncoder works: {y.shape}")
except Exception as e:
    print(f"✗ ModalityEncoder failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("TEST 8: CrossAttentionBlock")
print("="*70)

try:
    from models.blocks import CrossAttentionBlock
    
    query = tf.random.normal((2, 20, 256))
    context = tf.random.normal((2, 20, 256))
    query_mask = tf.ones((2, 20), dtype=tf.bool)
    context_mask = tf.ones((2, 20), dtype=tf.bool)
    
    block = CrossAttentionBlock(model_dim=256, num_heads=4, ff_dim=512)
    y = block(query, context, query_mask, context_mask, training=False)
    print(f"✓ CrossAttentionBlock works: {y.shape}")
except Exception as e:
    print(f"✗ CrossAttentionBlock failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("TEST 9: AdaptiveFusionHead")
print("="*70)

try:
    from models.fusion import AdaptiveFusionHead
    
    seqs = [
        tf.random.normal((2, 20, 256)),
        tf.random.normal((2, 20, 256)),
        tf.random.normal((2, 20, 256))
    ]
    masks = [
        tf.ones((2, 20), dtype=tf.bool),
        tf.ones((2, 20), dtype=tf.bool),
        tf.ones((2, 20), dtype=tf.bool)
    ]
    
    fusion_head = AdaptiveFusionHead(model_dim=256, num_modalities=3)
    y = fusion_head(seqs, masks, training=False)
    print(f"✓ AdaptiveFusionHead works: {y.shape}")
except Exception as e:
    print(f"✗ AdaptiveFusionHead failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("ALL COMPONENT TESTS PASSED!")
print("="*70)
print("\nThe issue must be in how components are combined or during training.")





