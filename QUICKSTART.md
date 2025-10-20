# Quick Start Guide - Sequence-Level MSA Model

## 🚀 Get Started in 5 Minutes

### 1. Validate Installation

```bash
cd /Users/rizkimuhammad/Honours/msa-tf2
conda activate msa-tf2
python test_seqlevel_model.py
```

✅ All 7 tests should pass

---

### 2. Check GPU Status (M2 Pro)

```bash
python check_gpu.py
```

✅ **Your GPU is working! 4.09x speedup already!**

---

### 3. Train Your First Model

**GPU Training (Recommended for M2 Pro + MOSEI):**
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

**Expected:** ~2.5-3 hours, GPU utilization 70-85%, ~100s per epoch

**Fast Experimentation:**
```bash
python train_seqlevel.py \
    --mixed_precision \
    --batch_size 128 \
    --model_dim 64 \
    --epochs 50
```

**Expected:** ~40-60 minutes, ~45s per epoch

**Maximum Performance:**
```bash
python train_seqlevel.py \
    --mixed_precision \
    --batch_size 128 \
    --model_dim 256 \
    --num_heads 8 \
    --epochs 100
```

**Expected:** ~4-5 hours, best accuracy

---

### 4. Compare with Original Model

```bash
python compare_models.py
```

This shows:
- Parameter counts
- Architecture differences
- Speed comparison
- Memory usage

---

### 5. Use in Your Code

```python
from models import MSASeqLevelModel
import tensorflow as tf

# Create model
model = MSASeqLevelModel(
    seq_len=20,           # Your sequence length
    text_dim=300,         # GloVe embeddings
    audio_dim=74,         # COVAREP features
    video_dim=47,         # Facet features
    model_dim=128,        # Hidden dimension
    num_heads=4,          # Attention heads
    ff_dim=256,           # FFN dimension
    n_layers_mod=2,       # Encoder layers
    n_layers_fuse=1       # Fusion layers
)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='mae',
    metrics=['mae', 'mse']
)

# Train
# model.fit(train_data, validation_data=val_data, epochs=50)
```

---

## 📊 Quick Reference

### Model Configurations

| Size | model_dim | num_heads | ff_dim | Parameters | Memory | Speed |
|------|-----------|-----------|--------|------------|--------|-------|
| Small | 64 | 2 | 128 | ~280K | 1.1 MB | Fast |
| Medium | 128 | 4 | 256 | ~1.1M | 4.3 MB | Balanced |
| Large | 256 | 8 | 512 | ~4.3M | 16.4 MB | Slower |

### Key Hyperparameters

- `model_dim`: Transformer hidden size (64-256)
- `num_heads`: Must divide `model_dim` evenly (2-8)
- `ff_dim`: Feed-forward dimension (128-1024)
- `n_layers_mod`: Layers per modality encoder (1-3)
- `n_layers_fuse`: Cross-attention fusion layers (1-2)
- `bidirectional_fusion`: Use symmetric cross-attention (True/False)
- `pooling_method`: 'mean' or 'attention'
- `dropout_rate`: Dropout probability (0.0-0.3)

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| `models/README.md` | API reference and examples |
| `SEQLEVEL_ARCHITECTURE.md` | Detailed architecture docs |
| `IMPLEMENTATION_SUMMARY.md` | Implementation overview |
| `QUICKSTART.md` | This guide |

---

## 🔧 Troubleshooting

### Out of Memory?
```bash
python train_seqlevel.py --model_dim 64 --batch_size 16
```

### Training Too Slow?
```bash
python train_seqlevel.py --mixed_precision --batch_size 64
```

### NaN Loss?
```bash
python train_seqlevel.py --learning_rate 1e-5
```

---

## 🎯 Next Steps

1. ✅ Run `test_seqlevel_model.py` to validate
2. ✅ Train baseline: `python train_seqlevel.py`
3. ✅ Compare: `python compare_models.py`
4. ⬜ Experiment with hyperparameters
5. ⬜ Try bidirectional fusion
6. ⬜ Evaluate on test set
7. ⬜ Compare with original MSAModel

---

## 💡 Tips

- Start with medium config (128-dim)
- Use mixed precision for faster training
- Enable bidirectional fusion for better accuracy
- Monitor validation MAE for early stopping
- Check `weights/` for saved models and logs

---

## 📞 Help

- **Tests failing?** Check Python/TensorFlow versions
- **Import errors?** Ensure you're in `msa-tf2` conda environment
- **Data issues?** Verify data files in `data/` directory
- **Other issues?** Check relevant `.md` documentation

---

*Ready to train? Run `python train_seqlevel.py` now!* 🚀

