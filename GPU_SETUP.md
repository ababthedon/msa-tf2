# GPU Setup Guide for Apple Silicon (M1/M2/M3)

This guide helps you enable GPU acceleration for training on your M2 Pro MacBook.

---

## Quick Setup

### 1. Check Current GPU Status

```bash
cd /Users/rizkimuhammad/Honours/msa-tf2
conda activate msa-tf2
python check_gpu.py
```

This will tell you if GPU acceleration is working.

---

## Installing tensorflow-metal

If GPU is not detected, install `tensorflow-metal`:

### Option 1: Using Conda (Recommended)

```bash
conda activate msa-tf2

# Install Apple's TensorFlow dependencies
conda install -c apple tensorflow-deps

# Install tensorflow-metal via pip
pip install tensorflow-metal
```

### Option 2: Using pip only

```bash
conda activate msa-tf2
pip install tensorflow-metal
```

### Verify Installation

```bash
python check_gpu.py
```

You should see:
```
✓ GPU devices found: 1
  GPU 0: PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
```

---

## Training with GPU

Once GPU is enabled, train with GPU acceleration:

### Basic GPU Training

```bash
python train_seqlevel.py
```

The script automatically detects and uses GPU if available.

### Optimized GPU Training (Recommended)

```bash
python train_seqlevel.py \
    --mixed_precision \
    --batch_size 64 \
    --model_dim 128
```

**Why these settings?**
- `--mixed_precision`: Uses float16 for faster computation, ~2x speedup
- `--batch_size 64`: Larger batches utilize GPU better
- `--model_dim 128`: Good balance of performance and memory

### Maximum Performance

```bash
python train_seqlevel.py \
    --mixed_precision \
    --batch_size 128 \
    --model_dim 256 \
    --num_heads 8 \
    --ff_dim 512
```

**Note:** Monitor memory usage with Activity Monitor. If you get OOM errors, reduce `batch_size` or `model_dim`.

---

## Performance Comparison

### Expected Speedup on M2 Pro

| Configuration | CPU Time/Epoch | GPU Time/Epoch | Speedup |
|---------------|----------------|----------------|---------|
| Small (64-dim) | ~300s | ~60s | 5x |
| Medium (128-dim) | ~600s | ~100s | 6x |
| Large (256-dim) | ~1200s | ~180s | 6-7x |

*Times are approximate for CMU-MOSEI dataset (~16K samples)*

### GPU Utilization Tips

1. **Larger Batch Sizes:** GPU benefits from larger batches
   - CPU optimal: 16-32
   - GPU optimal: 64-128

2. **Mixed Precision:** Essential for GPU speedup
   - Enables float16 operations
   - ~2x faster with minimal accuracy loss

3. **Model Size:** Larger models benefit more from GPU
   - Small models: 3-4x speedup
   - Large models: 6-8x speedup

---

## Monitoring GPU Usage

### Activity Monitor

1. Open **Activity Monitor** (Cmd+Space → "Activity Monitor")
2. Go to **Window → GPU History**
3. Watch GPU utilization during training

You should see:
- GPU utilization: 60-90% during training
- GPU memory: Gradually increases, then stabilizes

### Terminal Monitoring

```bash
# Watch GPU memory usage
while true; do
    clear
    date
    ps aux | grep python | grep -v grep
    sleep 2
done
```

---

## Troubleshooting

### Issue 1: No GPU Detected

**Symptom:**
```
⚠ No GPU found. Training will use CPU.
```

**Solution:**
```bash
# Install tensorflow-metal
conda activate msa-tf2
conda install -c apple tensorflow-deps
pip install tensorflow-metal

# Restart terminal and check
python check_gpu.py
```

### Issue 2: Out of Memory (OOM)

**Symptom:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solution:**
```bash
# Reduce batch size
python train_seqlevel.py --batch_size 32

# Or reduce model size
python train_seqlevel.py --model_dim 64 --batch_size 64
```

### Issue 3: GPU Slower Than Expected

**Symptom:**
- GPU speedup < 2x

**Solutions:**
1. Enable mixed precision: `--mixed_precision`
2. Increase batch size: `--batch_size 128`
3. Check Activity Monitor for GPU utilization
4. Ensure no other GPU-intensive apps are running

### Issue 4: Training Crashes

**Symptom:**
```
Metal device exception / kernel panic
```

**Solution:**
```bash
# Update to latest versions
pip install --upgrade tensorflow tensorflow-metal

# If still crashes, use CPU for stability
python train_seqlevel.py --use_cpu
```

---

## Force CPU Execution

If you want to use CPU instead of GPU:

```bash
python train_seqlevel.py --use_cpu
```

This is useful for:
- Debugging
- Comparing CPU vs GPU performance
- When GPU is having issues

---

## Version Compatibility

### Recommended Versions

```
tensorflow >= 2.13.0
tensorflow-metal >= 1.0.0
Python 3.9-3.11
macOS >= 13.0 (Ventura)
```

### Check Your Versions

```bash
python -c "import tensorflow as tf; print(f'TF: {tf.__version__}')"
pip show tensorflow-metal
sw_vers  # macOS version
```

---

## M2 Pro Specific Optimization

Your M2 Pro has:
- **GPU Cores:** 16-19 (depending on variant)
- **Unified Memory:** 16-32 GB
- **Memory Bandwidth:** ~200 GB/s

### Optimal Settings for M2 Pro

```bash
# 16GB RAM variant
python train_seqlevel.py \
    --mixed_precision \
    --batch_size 64 \
    --model_dim 128 \
    --n_layers_mod 2

# 32GB RAM variant
python train_seqlevel.py \
    --mixed_precision \
    --batch_size 128 \
    --model_dim 256 \
    --n_layers_mod 3
```

---

## Performance Benchmarks

Run a quick benchmark:

```bash
python check_gpu.py
```

This will:
1. Detect GPU
2. Run matrix multiplication test
3. Compare GPU vs CPU speed
4. Show expected speedup

---

## FAQ

**Q: Do I need CUDA for Apple Silicon?**  
A: No! Apple Silicon uses Metal Performance Shaders (MPS), not CUDA.

**Q: Will training use GPU automatically?**  
A: Yes, if tensorflow-metal is installed and GPU is detected.

**Q: Can I train overnight on battery?**  
A: Yes, but plug in for best performance. GPU throttles on battery.

**Q: How much speedup should I expect?**  
A: 5-7x for typical models with mixed precision and good batch sizes.

**Q: Does mixed precision affect accuracy?**  
A: Minimal impact (<0.5% difference) for MSA tasks.

---

## Summary Checklist

✅ **Setup:**
- [ ] Install tensorflow-metal
- [ ] Run `python check_gpu.py`
- [ ] Verify GPU detected

✅ **Training:**
- [ ] Use `--mixed_precision`
- [ ] Use batch size 64-128
- [ ] Monitor GPU usage in Activity Monitor

✅ **Troubleshooting:**
- [ ] Check versions (TF >= 2.13)
- [ ] Reduce batch size if OOM
- [ ] Use `--use_cpu` as fallback

---

## Additional Resources

- [TensorFlow Metal Plugin](https://developer.apple.com/metal/tensorflow-plugin/)
- [Apple Silicon TensorFlow Guide](https://developer.apple.com/metal/tensorflow-plugin/)
- [TensorFlow Mixed Precision](https://www.tensorflow.org/guide/mixed_precision)

---

**Ready to train with GPU acceleration! 🚀**

```bash
python check_gpu.py           # Verify GPU
python train_seqlevel.py      # Train with auto GPU detection
```





