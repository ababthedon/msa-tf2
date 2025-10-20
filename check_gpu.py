"""
GPU Detection and Verification Script for Apple Silicon

This script checks if TensorFlow can detect and use the GPU on your
M1/M2/M3 MacBook via Metal Performance Shaders (MPS).
"""

import tensorflow as tf
import os

def check_tensorflow_version():
    """Check TensorFlow version."""
    print("\n" + "="*70)
    print("TensorFlow Version")
    print("="*70)
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check if tensorflow-metal is available
    try:
        import tensorflow_metal
        print(f"tensorflow-metal: installed")
    except ImportError:
        print("tensorflow-metal: NOT INSTALLED")
        print("\n⚠ To enable GPU acceleration on Apple Silicon, install:")
        print("  conda install -c apple tensorflow-deps")
        print("  pip install tensorflow-metal")
        return False
    
    return True


def check_gpu_devices():
    """Check available GPU devices."""
    print("\n" + "="*70)
    print("Available Devices")
    print("="*70)
    
    # List all physical devices
    devices = tf.config.list_physical_devices()
    print(f"\nAll devices ({len(devices)}):")
    for device in devices:
        print(f"  - {device}")
    
    # Check specifically for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"\n✓ GPU devices found: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
        return True
    else:
        print("\n✗ No GPU devices found")
        print("\nPossible reasons:")
        print("  1. tensorflow-metal is not installed")
        print("  2. TensorFlow version incompatibility")
        print("  3. Running on non-Apple Silicon hardware")
        return False


def test_gpu_computation():
    """Test a simple computation on GPU."""
    print("\n" + "="*70)
    print("GPU Computation Test")
    print("="*70)
    
    try:
        # Create tensors and perform computation
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
        
        print("\n✓ Successfully performed matrix multiplication on GPU")
        print(f"  Result shape: {c.shape}")
        print(f"  Result device: {c.device}")
        return True
        
    except Exception as e:
        print(f"\n✗ GPU computation failed: {e}")
        print("\nTrying CPU computation...")
        
        try:
            with tf.device('/CPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
            print("✓ CPU computation successful")
            print("  (GPU acceleration not available)")
        except Exception as e2:
            print(f"✗ CPU computation also failed: {e2}")
        
        return False


def benchmark_performance():
    """Benchmark GPU vs CPU performance."""
    print("\n" + "="*70)
    print("Performance Benchmark")
    print("="*70)
    
    import time
    import numpy as np
    
    # Test parameters
    matrix_size = 2000
    num_iterations = 10
    
    print(f"\nBenchmark: {num_iterations} matrix multiplications ({matrix_size}x{matrix_size})")
    
    # GPU benchmark
    try:
        with tf.device('/GPU:0'):
            # Warm-up
            a = tf.random.normal([matrix_size, matrix_size])
            b = tf.random.normal([matrix_size, matrix_size])
            _ = tf.matmul(a, b)
            
            # Benchmark
            start = time.time()
            for _ in range(num_iterations):
                c = tf.matmul(a, b)
            _ = c.numpy()  # Force execution
            gpu_time = time.time() - start
            
        print(f"  GPU time: {gpu_time:.3f} seconds ({gpu_time/num_iterations*1000:.1f} ms per operation)")
    except:
        gpu_time = None
        print("  GPU: Not available")
    
    # CPU benchmark
    with tf.device('/CPU:0'):
        # Warm-up
        a = tf.random.normal([matrix_size, matrix_size])
        b = tf.random.normal([matrix_size, matrix_size])
        _ = tf.matmul(a, b)
        
        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            c = tf.matmul(a, b)
        _ = c.numpy()  # Force execution
        cpu_time = time.time() - start
    
    print(f"  CPU time: {cpu_time:.3f} seconds ({cpu_time/num_iterations*1000:.1f} ms per operation)")
    
    # Speedup
    if gpu_time:
        speedup = cpu_time / gpu_time
        print(f"\n  GPU Speedup: {speedup:.2f}x faster than CPU")
        
        if speedup > 1.5:
            print("  ✓ GPU acceleration is working well!")
        elif speedup > 1.0:
            print("  ⚠ GPU is faster, but speedup is modest")
        else:
            print("  ⚠ GPU is slower than CPU (unexpected)")
    else:
        print("\n  (GPU not available for comparison)")


def check_memory_growth():
    """Check GPU memory growth configuration."""
    print("\n" + "="*70)
    print("GPU Memory Configuration")
    print("="*70)
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                # Try to enable memory growth
                tf.config.experimental.set_memory_growth(gpu, True)
            print("\n✓ Memory growth enabled for all GPUs")
            print("  (GPU memory will be allocated as needed)")
        except RuntimeError as e:
            print(f"\n⚠ Could not enable memory growth: {e}")
            print("  (This is usually fine if already configured)")
    else:
        print("\nNo GPUs to configure")


def main():
    """Run all GPU checks."""
    print("\n" + "#"*70)
    print("#  TensorFlow GPU Detection for Apple Silicon (M1/M2/M3)")
    print("#"*70)
    
    # Run checks
    has_metal = check_tensorflow_version()
    has_gpu = check_gpu_devices()
    
    if has_gpu:
        check_memory_growth()
        test_gpu_computation()
        benchmark_performance()
        
        # Summary
        print("\n" + "="*70)
        print("Summary")
        print("="*70)
        print("\n✓ GPU acceleration is available and working!")
        print("\nRecommendations for training:")
        print("  1. Use --mixed_precision flag for faster training")
        print("  2. Increase batch size to fully utilize GPU")
        print("  3. Monitor GPU memory usage with Activity Monitor")
        print("\nTrain with GPU:")
        print("  python train_seqlevel.py --mixed_precision --batch_size 64")
        
    else:
        # Troubleshooting
        print("\n" + "="*70)
        print("Troubleshooting")
        print("="*70)
        
        if not has_metal:
            print("\n✗ GPU acceleration is NOT available")
            print("\nTo enable GPU on Apple Silicon:")
            print("\n1. Install tensorflow-metal:")
            print("   conda activate msa-tf2")
            print("   conda install -c apple tensorflow-deps")
            print("   pip install tensorflow-metal")
            print("\n2. Restart Python/terminal")
            print("\n3. Run this script again to verify")
        else:
            print("\n✗ GPU detection failed despite tensorflow-metal being installed")
            print("\nTry:")
            print("  1. Restart terminal/Python")
            print("  2. Check TensorFlow version compatibility")
            print("  3. Reinstall tensorflow and tensorflow-metal")
        
        print("\n(Training will still work on CPU, just slower)")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()





