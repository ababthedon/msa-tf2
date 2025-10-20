"""
Model Comparison Script

Compare the original MSAModel with the new MSASeqLevelModel on architecture,
memory usage, and forward pass timing.
"""

import tensorflow as tf
import numpy as np
import time
from models import MSAModel, MSASeqLevelModel

# Configure TensorFlow
tf.config.set_visible_devices([], 'GPU')


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def create_models(seq_len=20, model_dim=128):
    """Create both models with similar configurations."""
    text_dim = 300
    audio_dim = 74
    video_dim = 47
    
    print_section("Creating Models")
    
    # Original model
    print("\n[1] Creating Original MSAModel...")
    model_orig = MSAModel(
        seq_len=seq_len,
        text_dim=text_dim,
        audio_dim=audio_dim,
        video_dim=video_dim,
        model_dim=model_dim,
        num_heads=4,
        ff_dim=256,
        n_layers_mod=2,
        n_layers_fuse=1,
        adaptive_fusion=True
    )
    
    # New sequence-level model
    print("[2] Creating New MSASeqLevelModel...")
    model_new = MSASeqLevelModel(
        seq_len=seq_len,
        text_dim=text_dim,
        audio_dim=audio_dim,
        video_dim=video_dim,
        model_dim=model_dim,
        num_heads=4,
        ff_dim=256,
        n_layers_mod=2,
        n_layers_fuse=1,
        bidirectional_fusion=False,
        pooling_method='mean'
    )
    
    # Build models
    dummy_t = tf.zeros((1, seq_len, text_dim))
    dummy_a = tf.zeros((1, seq_len, audio_dim))
    dummy_v = tf.zeros((1, seq_len, video_dim))
    
    _ = model_orig((dummy_t, dummy_a, dummy_v), training=False)
    _ = model_new((dummy_t, dummy_a, dummy_v), training=False)
    
    print("\n✓ Both models created and built successfully")
    
    return model_orig, model_new


def compare_parameters(model_orig, model_new):
    """Compare number of parameters."""
    print_section("Parameter Comparison")
    
    orig_params = model_orig.count_params()
    new_params = model_new.count_params()
    
    print(f"\nOriginal MSAModel:")
    print(f"  Total parameters: {orig_params:,}")
    print(f"  Memory (approx): {orig_params * 4 / 1024 / 1024:.2f} MB")
    
    print(f"\nNew MSASeqLevelModel:")
    print(f"  Total parameters: {new_params:,}")
    print(f"  Memory (approx): {new_params * 4 / 1024 / 1024:.2f} MB")
    
    diff = new_params - orig_params
    percent = (diff / orig_params) * 100
    
    print(f"\nDifference:")
    print(f"  {diff:,} parameters ({percent:+.1f}%)")
    
    if diff > 0:
        print(f"  → New model has {diff:,} more parameters")
    else:
        print(f"  → New model has {abs(diff):,} fewer parameters")


def compare_architectures(model_orig, model_new):
    """Compare architectural differences."""
    print_section("Architecture Comparison")
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│ Feature                    │ Original      │ Sequence-Level    │")
    print("├────────────────────────────┼───────────────┼───────────────────┤")
    print("│ Encoder Pooling            │ Early         │ Late (no pooling) │")
    print("│ Fusion Type                │ Concatenation │ Cross-attention   │")
    print("│ Positional Embeddings      │ No            │ Yes (learnable)   │")
    print("│ Mask-aware Fusion          │ No            │ Yes               │")
    print("│ Bidirectional Option       │ No            │ Yes (optional)    │")
    print("│ Modality Interaction       │ Token-level   │ Sequence-level    │")
    print("│ Fusion Mechanism           │ Stacked       │ Cross-modal attn  │")
    print("│ Adaptive Weighting         │ Optional      │ Always (mask-aware│")
    print("└────────────────────────────┴───────────────┴───────────────────┘")
    
    print("\nKey Differences:")
    print("  1. Original pools sequences immediately after encoding")
    print("     → New model keeps sequences for richer fusion")
    print("  2. Original uses simple concatenation + transformer")
    print("     → New model uses explicit cross-attention between modalities")
    print("  3. New model has learnable positional embeddings")
    print("  4. New model has mask-aware pooling in fusion head")


def compare_forward_pass(model_orig, model_new, batch_sizes=[4, 8, 16, 32]):
    """Compare forward pass timing."""
    print_section("Forward Pass Timing Comparison")
    
    seq_len = 20
    text_dim = 300
    audio_dim = 74
    video_dim = 47
    n_runs = 10
    
    print(f"\nBenchmark settings:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of runs per batch size: {n_runs}")
    print(f"  Averaging over: {n_runs} runs")
    
    print("\nResults:")
    print("  Batch Size │ Original (ms) │ Seq-Level (ms) │ Ratio")
    print("  ───────────┼───────────────┼────────────────┼───────")
    
    for batch_size in batch_sizes:
        # Create test data
        text = tf.random.normal((batch_size, seq_len, text_dim))
        audio = tf.random.normal((batch_size, seq_len, audio_dim))
        video = tf.random.normal((batch_size, seq_len, video_dim))
        
        # Warm up
        _ = model_orig((text, audio, video), training=False)
        _ = model_new((text, audio, video), training=False)
        
        # Benchmark original
        times_orig = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model_orig((text, audio, video), training=False)
            end = time.perf_counter()
            times_orig.append((end - start) * 1000)
        
        # Benchmark new
        times_new = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model_new((text, audio, video), training=False)
            end = time.perf_counter()
            times_new.append((end - start) * 1000)
        
        avg_orig = np.mean(times_orig)
        avg_new = np.mean(times_new)
        ratio = avg_new / avg_orig
        
        print(f"  {batch_size:>10} │ {avg_orig:>13.2f} │ {avg_new:>14.2f} │ {ratio:.2f}x")
    
    print("\nNote: Timing may vary based on system load and TensorFlow optimizations")


def compare_outputs(model_orig, model_new):
    """Compare output distributions."""
    print_section("Output Comparison")
    
    # Create random inputs
    batch_size = 32
    seq_len = 20
    text = tf.random.normal((batch_size, seq_len, 300))
    audio = tf.random.normal((batch_size, seq_len, 74))
    video = tf.random.normal((batch_size, seq_len, 47))
    
    # Get outputs
    output_orig = model_orig((text, audio, video), training=False)
    output_new = model_new((text, audio, video), training=False)
    
    print(f"\nOriginal MSAModel output:")
    print(f"  Shape: {output_orig.shape}")
    print(f"  Mean: {tf.reduce_mean(output_orig).numpy():.4f}")
    print(f"  Std: {tf.math.reduce_std(output_orig).numpy():.4f}")
    print(f"  Range: [{tf.reduce_min(output_orig).numpy():.4f}, "
          f"{tf.reduce_max(output_orig).numpy():.4f}]")
    
    print(f"\nNew MSASeqLevelModel output:")
    print(f"  Shape: {output_new.shape}")
    print(f"  Mean: {tf.reduce_mean(output_new).numpy():.4f}")
    print(f"  Std: {tf.math.reduce_std(output_new).numpy():.4f}")
    print(f"  Range: [{tf.reduce_min(output_new).numpy():.4f}, "
          f"{tf.reduce_max(output_new).numpy():.4f}]")
    
    print("\nNote: Outputs differ because models have different architectures")
    print("      and randomly initialized weights.")


def main():
    """Run all comparisons."""
    print("\n" + "#"*70)
    print("#  MSA Model Comparison: Original vs Sequence-Level")
    print("#"*70)
    
    # Create models
    model_orig, model_new = create_models(seq_len=20, model_dim=128)
    
    # Run comparisons
    compare_parameters(model_orig, model_new)
    compare_architectures(model_orig, model_new)
    compare_forward_pass(model_orig, model_new)
    compare_outputs(model_orig, model_new)
    
    # Summary
    print_section("Summary")
    print("\nRecommendations:")
    print("  • Use Original MSAModel if:")
    print("    - You need faster inference")
    print("    - Memory is constrained")
    print("    - You prefer simpler architectures")
    print("\n  • Use New MSASeqLevelModel if:")
    print("    - You want richer cross-modal interactions")
    print("    - You need sequence-level fusion")
    print("    - You want modular, reusable components")
    print("    - You're building on this for research")
    
    print("\nFor training and evaluation, see:")
    print("  - train.py (original model)")
    print("  - train_seqlevel.py (new model)")
    print("  - models/README.md (detailed documentation)")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()





