# CPU Training Parameter Guide

## Recommended Configurations for CPU Training on MOSEI

All configurations optimized for your M2 Pro CPU with 16GB RAM.

---

## 🌟 **Configuration 1: BALANCED (Recommended)**

Best balance between accuracy and training time.

```bash
python train_seqlevel.py \
    --use_cpu \
    --batch_size 32 \
    --model_dim 128 \
    --num_heads 4 \
    --ff_dim 256 \
    --n_layers_mod 2 \
    --n_layers_fuse 1 \
    --learning_rate 1e-4 \
    --epochs 100 \
    --patience 15
```

**Expected Performance:**
- ⏱️ Time per epoch: ~400-450 seconds (~7 minutes)
- 🕐 Total time: ~4-5 hours (with early stopping ~30-40 epochs)
- 📊 Accuracy: Good baseline performance
- 💾 Memory: ~3-4 GB
- 📈 Parameters: ~1.3M

**Why this config:**
- `model_dim=128`: Sweet spot for CPU - not too large, good capacity
- `batch_size=32`: Optimal for CPU memory bandwidth
- `n_layers_mod=2`: Enough capacity without excessive computation
- `n_layers_fuse=1`: Cross-attention is expensive, keep minimal

---

## 🚀 **Configuration 2: FAST (Quick Experiments)**

For rapid iteration and hyperparameter search.

```bash
python train_seqlevel.py \
    --use_cpu \
    --batch_size 32 \
    --model_dim 64 \
    --num_heads 2 \
    --ff_dim 128 \
    --n_layers_mod 1 \
    --n_layers_fuse 1 \
    --learning_rate 1e-4 \
    --epochs 100 \
    --patience 15
```

**Expected Performance:**
- ⏱️ Time per epoch: ~200-250 seconds (~4 minutes)
- 🕐 Total time: ~2-3 hours
- 📊 Accuracy: Decent (slightly lower than balanced)
- 💾 Memory: ~2 GB
- 📈 Parameters: ~350K

**When to use:**
- Initial experiments
- Hyperparameter tuning
- Quick validation of ideas
- Testing different learning rates/optimizers

---

## 💪 **Configuration 3: HIGH PERFORMANCE (Best Accuracy)**

Maximum model capacity, longer training time.

```bash
python train_seqlevel.py \
    --use_cpu \
    --batch_size 24 \
    --model_dim 256 \
    --num_heads 8 \
    --ff_dim 512 \
    --n_layers_mod 3 \
    --n_layers_fuse 2 \
    --learning_rate 5e-5 \
    --epochs 100 \
    --patience 20
```

**Expected Performance:**
- ⏱️ Time per epoch: ~900-1100 seconds (~15-18 minutes)
- 🕐 Total time: ~10-14 hours
- 📊 Accuracy: Best possible
- 💾 Memory: ~6-8 GB
- 📈 Parameters: ~5.2M

**When to use:**
- Final model for paper/thesis
- Best possible results
- Have overnight to train
- After finding good hyperparameters with fast config

---

## 🔬 **Configuration 4: MINIMAL (Debugging)**

Ultra-fast for debugging and testing.

```bash
python train_seqlevel.py \
    --use_cpu \
    --batch_size 16 \
    --model_dim 32 \
    --num_heads 2 \
    --ff_dim 64 \
    --n_layers_mod 1 \
    --n_layers_fuse 1 \
    --learning_rate 1e-4 \
    --epochs 10
```

**Expected Performance:**
- ⏱️ Time per epoch: ~100-120 seconds (~2 minutes)
- 🕐 Total time: ~20 minutes for 10 epochs
- 📊 Accuracy: Poor (not for production)
- 💾 Memory: ~1 GB

**When to use:**
- Testing if training works
- Debugging code changes
- Verifying data pipeline
- Quick sanity checks

---

## 📊 **Parameter Comparison Table**

| Config | model_dim | batch | Time/Epoch | Total Time | Accuracy |
|--------|-----------|-------|------------|------------|----------|
| **Balanced** ⭐ | 128 | 32 | 7 min | 4-5h | Good |
| Fast | 64 | 32 | 4 min | 2-3h | Decent |
| High Perf | 256 | 24 | 16 min | 10-14h | Best |
| Minimal | 32 | 16 | 2 min | 20min | Debug only |

---

## 🎯 **Recommended Workflow**

### Step 1: Quick Test (10 minutes)
```bash
# Verify everything works
python train_seqlevel.py \
    --use_cpu \
    --batch_size 16 \
    --model_dim 64 \
    --epochs 5
```

### Step 2: Fast Baseline (2-3 hours)
```bash
# Get initial results
python train_seqlevel.py \
    --use_cpu \
    --batch_size 32 \
    --model_dim 64 \
    --epochs 100
```

### Step 3: Balanced Model (4-5 hours) ⭐
```bash
# Main model for thesis
python train_seqlevel.py \
    --use_cpu \
    --batch_size 32 \
    --model_dim 128 \
    --epochs 100
```

### Step 4: Best Model (overnight)
```bash
# Final best results
python train_seqlevel.py \
    --use_cpu \
    --batch_size 24 \
    --model_dim 256 \
    --num_heads 8 \
    --epochs 100
```

---

## 💡 **CPU Optimization Tips**

### 1. **Batch Size**
- **Sweet spot: 24-32** for CPU
- Too small (8-16): Underutilizes CPU, slower
- Too large (64+): Marginal speedup, more memory
- **Recommended: 32**

### 2. **Model Dimension**
- **64**: Fast, decent results (~2.5h)
- **128**: Balanced, good results (~4-5h) ⭐
- **256**: Best results, slow (~10-14h)
- **Recommended: 128 for main experiments**

### 3. **Number of Layers**
- **Each additional layer adds ~40% training time**
- `n_layers_mod=1`: Fast but limited capacity
- `n_layers_mod=2`: Good balance ⭐
- `n_layers_mod=3`: Diminishing returns
- **Recommended: 2 for modality, 1 for fusion**

### 4. **Learning Rate**
- Start with **1e-4** (default)
- If training unstable: try **5e-5**
- For large models: **5e-5** or **1e-5**
- Use `ReduceLROnPlateau` callback (already enabled)

### 5. **Early Stopping**
- **Patience=15**: Standard (already set)
- MOSEI typically converges in 30-40 epochs
- Saves ~60% of training time

---

## 🕐 **Training Time Estimates (MOSEI)**

Based on M2 Pro CPU performance:

### By Model Size:
| model_dim | Time/Epoch | Est. Total (40 epochs) |
|-----------|------------|------------------------|
| 32 | 2 min | 1.3h |
| 64 | 4 min | 2.7h |
| **128** | **7 min** | **4.7h** ⭐ |
| 256 | 16 min | 10.7h |
| 512 | 35 min | 23h |

### By Batch Size (model_dim=128):
| batch_size | Time/Epoch | Memory |
|------------|------------|--------|
| 8 | 8 min | 2 GB |
| 16 | 7.5 min | 2.5 GB |
| **32** | **7 min** | **3.5 GB** ⭐ |
| 64 | 6.8 min | 5 GB |

**Optimal: batch_size=32**

---

## 📈 **Expected Results (MOSEI Benchmark)**

Based on similar architectures:

| Config | Test MAE | Test Acc-2 |
|--------|----------|------------|
| Fast (64-dim) | ~0.72 | ~78% |
| **Balanced (128-dim)** | **~0.68** | **~80%** |
| High Perf (256-dim) | ~0.65 | ~82% |

*Acc-2: Binary accuracy (positive/negative)*

---

## 🎓 **For Your Honours Thesis**

### Recommended Strategy:

#### Phase 1: Quick Validation (Day 1)
```bash
# 2-3 hours
python train_seqlevel.py --use_cpu --batch_size 32 --model_dim 64
```

#### Phase 2: Main Results (Day 2-3)
```bash
# 4-5 hours, run overnight
python train_seqlevel.py --use_cpu --batch_size 32 --model_dim 128
```

#### Phase 3: Ablation Studies (Week 1)
Test variations:
- With/without bidirectional fusion
- Different n_layers_fuse (1 vs 2)
- Attention vs mean pooling

#### Phase 4: Best Model (Week 2)
```bash
# Run overnight for final results
python train_seqlevel.py --use_cpu --batch_size 24 --model_dim 256
```

---

## 🔍 **Monitoring CPU Training**

### Terminal 1: Run Training
```bash
python train_seqlevel.py --use_cpu --batch_size 32 --model_dim 128
```

### Terminal 2: Monitor Progress
```bash
# Watch training log
tail -f weights/seqlevel_training_log_*.csv

# Or use watch
watch -n 5 'tail -5 weights/seqlevel_training_log_*.csv'
```

### Activity Monitor:
- CPU usage should be 100-200% (2 cores)
- Memory: 3-5 GB
- If CPU < 100%: increase batch_size

---

## 💾 **Memory Considerations**

Your 16GB RAM allows:

| model_dim | batch_size | Memory | Status |
|-----------|------------|--------|--------|
| 64 | 64 | ~3 GB | ✅ Safe |
| 128 | 32 | ~4 GB | ✅ Safe |
| 128 | 64 | ~6 GB | ✅ Safe |
| 256 | 24 | ~7 GB | ✅ Safe |
| 256 | 32 | ~9 GB | ✅ Safe |
| 512 | 16 | ~10 GB | ⚠️ Tight |

**You have plenty of headroom!**

---

## 🎯 **My Recommendation for You**

### **START WITH THIS:**

```bash
python train_seqlevel.py \
    --use_cpu \
    --batch_size 32 \
    --model_dim 128 \
    --num_heads 4 \
    --ff_dim 256 \
    --n_layers_mod 2 \
    --n_layers_fuse 1 \
    --learning_rate 1e-4 \
    --epochs 100 \
    --patience 15
```

**Why:**
- ✅ Completes in 4-5 hours (perfect for overnight)
- ✅ Good accuracy for thesis results
- ✅ Not too slow, not underpowered
- ✅ Can run multiple experiments per day
- ✅ Leaves room for ablations

**Then experiment with:**
- Bidirectional fusion: `--bidirectional_fusion`
- Attention pooling: `--pooling_method attention`
- More fusion layers: `--n_layers_fuse 2`

---

## 📝 **Quick Reference Commands**

### Daily Work (Fast Iteration):
```bash
python train_seqlevel.py --use_cpu --batch_size 32 --model_dim 64 --epochs 50
```

### Main Results (Overnight):
```bash
python train_seqlevel.py --use_cpu --batch_size 32 --model_dim 128 --epochs 100
```

### Best Model (Weekend Run):
```bash
python train_seqlevel.py --use_cpu --batch_size 24 --model_dim 256 --epochs 100
```

---

## ✅ **Ready to Start!**

**Recommended first command:**

```bash
cd /Users/rizkimuhammad/Honours/msa-tf2
conda activate msa-tf2

python train_seqlevel.py \
    --use_cpu \
    --batch_size 32 \
    --model_dim 128 \
    --num_heads 4 \
    --n_layers_mod 2 \
    --n_layers_fuse 1 \
    --epochs 100
```

**Expected:** 4-5 hours, good results, perfect for your thesis! 🎓






