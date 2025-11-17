# Dataset Configuration Guide

## Feature Dimensions by Dataset

Different MSA datasets use different feature extraction methods, resulting in different input dimensions.

---

## CMU-MOSI vs CMU-MOSEI

### CMU-MOSI
```python
text_dim = 300    # GloVe embeddings
audio_dim = 74    # COVAREP features
video_dim = 47    # Facet features (OpenFace)
seq_len = 20      # Sequence length
samples = ~2,199  # Total videos
```

### CMU-MOSEI
```python
text_dim = 300    # GloVe embeddings
audio_dim = 74    # COVAREP features
video_dim = 713   # Facet features (different version)
seq_len = 20      # Sequence length
samples = ~23,453 # Total videos
```

---

## Key Difference: Video Features

**The main difference is the video feature dimension:**

| Dataset | Video Dim | Video Features | Extraction Method |
|---------|-----------|----------------|-------------------|
| **MOSI** | 47 | Facet (basic) | OpenFace 1.x |
| **MOSEI** | 713 | Facet (extended) | OpenFace 2.x + additional features |

---

## Current Data in msa-tf2/data/

Your current data is **CMU-MOSEI** with dimensions:
- ✅ Text: 300
- ✅ Audio: 74  
- ✅ Video: **713**

---

## Training Script Configuration

### For MOSEI (Current)

**`train_seqlevel.py`** (✅ Already configured):
```python
text_dim = 300
audio_dim = 74
video_dim = 713  # MOSEI
```

**`train.py`** (✅ Already configured):
```python
text_dim = 300
audio_dim = 74
video_dim = 713  # MOSEI
```

### For MOSI (If You Switch)

If you switch to CMU-MOSI data later, update:
```python
text_dim = 300
audio_dim = 74
video_dim = 47   # MOSI
```

---

## Verifying Your Data

Run this to check your data dimensions:

```bash
cd /Users/rizkimuhammad/Honours/msa-tf2

python -c "
import h5py
import numpy as np

splits = ['train', 'valid', 'test']
modalities = ['text_emb', 'audio', 'video', 'y']

print('Data Dimensions:')
print('-' * 60)
for split in splits:
    print(f'\n{split.upper()} split:')
    for mod in modalities:
        if mod == 'text_emb':
            fname = f'data/text_{split}_emb.h5'
        elif mod == 'y':
            fname = f'data/y_{split}.h5'
        else:
            fname = f'data/{mod}_{split}.h5'
        
        with h5py.File(fname, 'r') as f:
            shape = f['d1'].shape
            print(f'  {mod:10s}: {shape}')
"
```

**Expected output for MOSEI:**
```
TRAIN split:
  text_emb  : (16327, 20, 300)
  audio     : (16327, 20, 74)
  video     : (16327, 20, 713)  ← MOSEI
  y         : (16327, 1)
```

**Expected output for MOSI:**
```
TRAIN split:
  text_emb  : (1284, 20, 300)
  audio     : (1284, 20, 74)
  video     : (1284, 20, 47)   ← MOSI
  y         : (1284, 1)
```

---

## Model Configuration

### Automatic Detection (Future Enhancement)

You could add automatic dimension detection:

```python
import h5py

def detect_data_dimensions(data_dir):
    """Auto-detect input dimensions from data."""
    with h5py.File(f'{data_dir}/text_train_emb.h5', 'r') as f:
        text_dim = f['d1'].shape[-1]
        seq_len = f['d1'].shape[1]
    
    with h5py.File(f'{data_dir}/audio_train.h5', 'r') as f:
        audio_dim = f['d1'].shape[-1]
    
    with h5py.File(f'{data_dir}/video_train.h5', 'r') as f:
        video_dim = f['d1'].shape[-1]
    
    return text_dim, audio_dim, video_dim, seq_len

# Use in training
text_dim, audio_dim, video_dim, seq_len = detect_data_dimensions('./data')
print(f"Detected: text={text_dim}, audio={audio_dim}, video={video_dim}")
```

---

## Dataset Statistics

### CMU-MOSI
- **Size:** 2,199 video clips
- **Speakers:** 89 distinct speakers
- **Videos:** 93 YouTube videos
- **Duration:** ~2.5 hours total
- **Labels:** Sentiment [-3, +3]

**Splits:**
- Train: ~1,284 samples
- Valid: ~229 samples  
- Test: ~686 samples

### CMU-MOSEI
- **Size:** 23,453 video clips
- **Speakers:** 1,000 distinct speakers
- **Videos:** 5,000+ YouTube videos
- **Duration:** ~65 hours total
- **Labels:** Sentiment [-3, +3]

**Splits:**
- Train: ~16,326 samples
- Valid: ~1,871 samples
- Test: ~4,659 samples

---

## Common Issues

### Issue 1: Dimension Mismatch Error

**Error:**
```
ValueError: expected axis -1 of input shape to have value 47, 
but received input with shape (None, 20, 713)
```

**Solution:**
- You have MOSEI data (713 dims) but model configured for MOSI (47 dims)
- Update `video_dim = 713` in training script

### Issue 2: Wrong Dataset Loaded

**Symptom:** Model trains but poor performance

**Check:**
1. Verify dimensions match your data
2. Ensure labels are in correct range [-3, +3]
3. Check number of samples matches expected dataset

---

## Best Practices

1. **Always verify dimensions** before training:
   ```bash
   python -c "import h5py; print(h5py.File('data/video_train.h5')['d1'].shape)"
   ```

2. **Document your dataset** in experiment logs:
   ```python
   print(f"Dataset: CMU-MOSEI")
   print(f"Dimensions: text={text_dim}, audio={audio_dim}, video={video_dim}")
   ```

3. **Use dataset-specific configs** when comparing results:
   - MOSI: Standard benchmark, smaller
   - MOSEI: More challenging, larger, better generalization

---

## Summary

✅ **Your current setup:**
- Dataset: **CMU-MOSEI**
- Video features: **713 dimensions**
- Training scripts: **Updated to 713**

✅ **You're ready to train!**

```bash
python train_seqlevel.py --mixed_precision --batch_size 64 --model_dim 128
```

---

## Quick Reference

| Dataset | Text | Audio | Video | Samples |
|---------|------|-------|-------|---------|
| **MOSI** | 300 | 74 | **47** | 2,199 |
| **MOSEI** | 300 | 74 | **713** | 23,453 |

**Current:** MOSEI (713 video features) ✓






