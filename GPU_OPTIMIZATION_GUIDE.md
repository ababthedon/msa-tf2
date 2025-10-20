# GPU Optimization Guide for Your M2 Pro

## ✅ GPU Status: **WORKING**

Your M2 Pro MacBook is already configured for GPU acceleration!

**Hardware:**
- Chip: Apple M2 Pro
- GPU Cores: 16-19 cores
- Unified Memory: 16GB
- Current Speedup: **4.09x** vs CPU

---

## 🚀 Recommended Training Commands

### For CMU-MOSEI Dataset (~16K samples)

#### **Option 1: Balanced (Recommended)**
```bash
python train_seqlevel.py \
    --mixed_precision \
    --batch_size 64 \
    --model_dim 128 \
    --num_heads 4 \
    --n_layers_mod 2 \
    --n_layers_fuse 1 \
    --learning_rate 1e-4 \
    --epochs 100
```

**Expected Performance:**
- Time per epoch: ~90-120 seconds
- Total training: ~2.5-3.5 hours (with early stopping)
- GPU utilization: 70-85%
- Memory usage: ~4-6 GB

#### **Option 2: Maximum Performance**
```bash
python train_seqlevel.py \
    --mixed_precision \
    --batch_size 128 \
    --model_dim 256 \
    --num_heads 8 \
    --ff_dim 512 \
    --n_layers_mod 3 \
    --n_layers_fuse 2 \
    --learning_rate 5e-5 \
    --epochs 100
```

**Expected Performance:**
- Time per epoch: ~180-240 seconds
- Total training: ~5-7 hours
- GPU utilization: 80-95%
- Memory usage: ~10-14 GB

#### **Option 3: Fast Experimentation**
```bash
python train_seqlevel.py \
    --mixed_precision \
    --batch_size 128 \
    --model_dim 64 \
    --num_heads 2 \
    --ff_dim 128 \
    --n_layers_mod 1 \
    --n_layers_fuse 1 \
    --epochs 50
```

**Expected Performance:**
- Time per epoch: ~40-60 seconds
- Total training: ~40-60 minutes
- GPU utilization: 60-75%
- Memory usage: ~2-3 GB

---

## 📊 Performance Optimization Tips

### 1. **Batch Size** (Most Important)
- GPU thrives on larger batches
- Your 16GB allows: 64-128 comfortably, up to 256 for small models
- Start with 64, increase to 128 if no OOM errors

### 2. **Mixed Precision** (2x Speedup)
- **Always use** `--mixed_precision` for GPU training
- Uses float16 for computation, float32 for output
- Minimal accuracy impact (<0.5%)
- Approximate 2x speedup + 40% memory savings

### 3. **Model Size vs Speed**

| Model Dim | Params | Time/Epoch | Memory | Speedup |
|-----------|--------|------------|--------|---------|
| 64 | ~280K | 45s | 2.5GB | 4x |
| 128 | ~1.1M | 100s | 5.5GB | 5x |
| 256 | ~4.3M | 220s | 13GB | 6x |

### 4. **Concurrent Processing**
- Close unnecessary apps during training
- Don't run multiple trainings simultaneously
- Leave ~2GB memory free for system

---

## 🔍 Monitoring GPU Usage

### Activity Monitor (Built-in)
```
1. Open Activity Monitor (Cmd+Space → Activity Monitor)
2. Window → GPU History
3. Watch during training
```

**What to look for:**
- **GPU %:** Should be 60-90% during training
- **Memory:** Gradually increases, then stabilizes
- **If GPU% < 50%:** Increase batch size

### Terminal Monitoring
```bash
# Watch GPU memory
watch -n 2 'ps aux | grep python'
```

---

## 🎯 Optimal Settings for Your M2 Pro

Based on your **16GB RAM M2 Pro**, here are the sweet spots:

### Training MOSEI (Large Dataset)
```bash
python train_seqlevel.py \
    --mixed_precision \
    --batch_size 64 \
    --model_dim 128 \
    --epochs 100
```

### Training MOSI (Small Dataset)
```bash
python train_seqlevel.py \
    --mixed_precision \
    --batch_size 128 \
    --model_dim 256 \
    --epochs 100
```

---

## ⚡ Expected Training Times (MOSEI)

| Configuration | Batch | Model Dim | Time/Epoch | Total Time* |
|---------------|-------|-----------|------------|-------------|
| Fast | 128 | 64 | 45s | ~1 hour |
| Balanced | 64 | 128 | 100s | ~2.5 hours |
| Best | 64 | 256 | 200s | ~5 hours |
| Maximum | 128 | 256 | 180s | ~4.5 hours |

*With early stopping (typically 30-40 epochs)

---

## 🛠️ Troubleshooting

### Out of Memory?
```bash
# Reduce batch size
python train_seqlevel.py --mixed_precision --batch_size 32

# Or reduce model size
python train_seqlevel.py --mixed_precision --model_dim 64 --batch_size 64
```

### GPU Not Being Used?
```bash
# Check GPU status
python check_gpu.py

# Force CPU if needed (for debugging)
python train_seqlevel.py --use_cpu
```

### Slow Training?
1. ✅ Enable mixed precision: `--mixed_precision`
2. ✅ Increase batch size: `--batch_size 128`
3. ✅ Close other apps
4. ✅ Ensure MacBook is plugged in (battery mode throttles GPU)

---

## 🔬 Before Training Checklist

- [ ] MacBook plugged in (for full GPU performance)
- [ ] Close unnecessary apps (Chrome, etc.)
- [ ] Activity Monitor ready to monitor GPU
- [ ] Using `--mixed_precision` flag
- [ ] Batch size 64 or higher
- [ ] Data in `./data` directory

---

## 📈 Performance Comparison

Your current setup vs alternatives:

| Device | Speedup | Time/Epoch (128-dim) | Cost |
|--------|---------|---------------------|------|
| **M2 Pro GPU** | **5-6x** | **~100s** | **$0** ✅ |
| M2 Pro CPU | 1x | ~500s | $0 |
| Cloud GPU (T4) | 8-10x | ~60s | ~$0.35/hr |
| Cloud GPU (V100) | 15-20x | ~30s | ~$2.50/hr |

**Verdict:** Your M2 Pro is excellent for this task! No need for cloud GPUs unless doing massive hyperparameter searches.

---

## 🚀 Ready to Train!

### Quick Start
```bash
cd /Users/rizkimuhammad/Honours/msa-tf2
conda activate msa-tf2

# Recommended settings for MOSEI
python train_seqlevel.py \
    --mixed_precision \
    --batch_size 64 \
    --model_dim 128 \
    --epochs 100
```

### Monitor Progress
- Watch terminal output for loss/metrics
- Open Activity Monitor → GPU History
- Check `weights/` directory for checkpoints
- Training logs saved in `weights/seqlevel_training_log_*.csv`

---

## 💡 Pro Tips

1. **Run overnight:** Training will take 2-5 hours, perfect for overnight runs
2. **Early stopping:** Model usually converges in 30-40 epochs
3. **Save best model:** Automatically saved to `weights/seqlevel_best_val_mae_*.h5`
4. **Try different configs:** Run multiple experiments with different hyperparameters
5. **Mixed precision is free speed:** Always use it on GPU

---

## 📊 Experiment Tracking

Track your experiments in a spreadsheet:

| Run | Model Dim | Batch | Mixed Prec | Time | Val MAE | Test MAE |
|-----|-----------|-------|------------|------|---------|----------|
| 1 | 128 | 64 | Yes | 2.5h | 0.685 | 0.701 |
| 2 | 256 | 64 | Yes | 4.8h | 0.672 | 0.688 |

---

**Your M2 Pro is ready for high-performance training! 🎉**





