# Memory Error Troubleshooting

## Error Description

```
malloc: *** error for object 0x...: pointer being freed was not allocated
zsh: abort
```

This is a memory corruption error on Apple Silicon, typically caused by:
1. TensorFlow-Metal mixed precision issues
2. Memory fragmentation
3. Incompatible layer operations
4. TensorFlow version conflicts

---

## Solutions (Try in Order)

### Solution 1: Disable Mixed Precision ⭐ (Most Likely Fix)

Mixed precision can cause issues with TensorFlow-Metal on some configurations.

```bash
# Train WITHOUT mixed precision
python train_seqlevel.py \
    --batch_size 64 \
    --model_dim 128 \
    --epochs 100
```

**Trade-off:** ~2x slower, but stable.

---

### Solution 2: Reduce Batch Size

Memory issues can be triggered by large batches.

```bash
# Smaller batch size
python train_seqlevel.py \
    --batch_size 32 \
    --model_dim 128 \
    --epochs 100
```

---

### Solution 3: Use CPU (Stable Fallback)

If GPU continues to cause issues, use CPU mode:

```bash
# Force CPU execution
python train_seqlevel.py \
    --use_cpu \
    --batch_size 32 \
    --model_dim 128 \
    --epochs 100
```

**Note:** Slower (5-6x), but guaranteed stable.

---

### Solution 4: Update TensorFlow

Older versions may have Metal backend bugs:

```bash
conda activate msa-tf2

# Update TensorFlow and Metal
pip install --upgrade tensorflow tensorflow-metal

# Check versions
python -c "import tensorflow as tf; print(f'TF: {tf.__version__}')"
pip show tensorflow-metal
```

**Recommended versions:**
- TensorFlow >= 2.15.0
- tensorflow-metal >= 1.1.0

---

### Solution 5: Smaller Model

Try a smaller model to reduce memory pressure:

```bash
python train_seqlevel.py \
    --batch_size 64 \
    --model_dim 64 \
    --num_heads 2 \
    --ff_dim 128 \
    --n_layers_mod 1 \
    --n_layers_fuse 1 \
    --epochs 100
```

---

### Solution 6: Environment Variables

Set these to reduce memory issues:

```bash
# Disable TensorFlow optimizations that can cause issues
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2

# Train
python train_seqlevel.py --batch_size 64 --model_dim 128
```

---

## Quick Test Script

Test if GPU works without crashing:

```bash
python check_gpu.py
```

If this crashes too, it's a TensorFlow-Metal issue.

---

## Recommended Configurations

### Config 1: Stable (No Mixed Precision)
```bash
python train_seqlevel.py \
    --batch_size 64 \
    --model_dim 128 \
    --num_heads 4 \
    --epochs 100
```

**Expected:** Works reliably, ~3x faster than CPU

### Config 2: CPU Fallback
```bash
python train_seqlevel.py \
    --use_cpu \
    --batch_size 32 \
    --model_dim 128 \
    --epochs 100
```

**Expected:** Slower but 100% stable

### Config 3: Small GPU Model
```bash
python train_seqlevel.py \
    --batch_size 32 \
    --model_dim 64 \
    --epochs 100
```

**Expected:** Fast and stable

---

## Root Cause Analysis

The error typically occurs when:

1. **Mixed precision + Metal backend conflict**
   - Some operations aren't properly supported in float16 on Metal
   - Solution: Disable mixed precision

2. **Large tensors on GPU**
   - Video features (713 dims) create large tensors
   - Solution: Reduce batch size or model size

3. **TensorFlow-Metal bugs**
   - Older versions have known issues
   - Solution: Update packages

---

## What to Try First

```bash
# OPTION 1: No mixed precision (recommended)
python train_seqlevel.py --batch_size 64 --model_dim 128

# OPTION 2: Smaller batch
python train_seqlevel.py --batch_size 32 --model_dim 128

# OPTION 3: CPU fallback
python train_seqlevel.py --use_cpu --batch_size 32 --model_dim 128
```

---

## Expected Performance Without Mixed Precision

| Config | Time/Epoch | Total Time | Speedup vs CPU |
|--------|------------|------------|----------------|
| GPU (no MP) | ~150s | 4h | 3-4x |
| GPU + small model | ~80s | 2h | 5-6x |
| CPU | ~500s | 14h | 1x |

**MP = Mixed Precision**

---

## Verification Steps

After choosing a solution:

1. **Start training:**
   ```bash
   python train_seqlevel.py --batch_size 64 --model_dim 128
   ```

2. **Watch first few epochs:** If it completes 3-5 epochs without crashing, you're good!

3. **Monitor in Activity Monitor:** GPU should show 60-80% utilization

---

## If Nothing Works

Last resort - train on CPU:

```bash
# Stable CPU training
python train_seqlevel.py \
    --use_cpu \
    --batch_size 16 \
    --model_dim 128 \
    --epochs 100
```

This will take longer (~14 hours for MOSEI) but will work reliably.

---

## Summary

✅ **Try first:** `python train_seqlevel.py --batch_size 64 --model_dim 128`  
✅ **If fails:** `python train_seqlevel.py --batch_size 32 --model_dim 64`  
✅ **Last resort:** `python train_seqlevel.py --use_cpu --batch_size 32`  

**The issue is likely mixed precision + TensorFlow-Metal incompatibility.**





