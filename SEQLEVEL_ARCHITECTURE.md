# Sequence-Level MSA Architecture Documentation

## Overview

This document provides comprehensive documentation for the new sequence-level Multimodal Sentiment Analysis (MSA) architecture implemented in `msa-tf2`. The architecture was designed to enhance Deep-HOSeq by maintaining sequence-level information through fusion and using explicit cross-modal attention mechanisms.

## Key Design Principles

### 1. **No Early Pooling**
Unlike traditional approaches that pool sequences immediately after encoding, this architecture maintains full sequence information through the fusion process. Pooling only occurs at the final adaptive fusion head.

**Benefits:**
- Richer cross-modal interactions
- Fine-grained temporal alignment between modalities
- Better capture of sequential dependencies

### 2. **Sequence-Level Cross-Attention**
Uses explicit cross-attention mechanisms where one modality attends to another at the sequence level (e.g., text attends to audio frame-by-frame).

**Benefits:**
- Direct modality-to-modality interactions
- Learnable alignment between modalities
- More expressive than concatenation-based fusion

### 3. **Modular and Reusable Components**
All fusion components (`CrossModalFusion`, `AdaptiveFusionHead`) are designed to be modular and can work with any sequence encoder (transformer, RNN, CNN).

**Benefits:**
- Easy to integrate into existing architectures
- Supports research experimentation
- Can be used beyond MSA tasks

### 4. **Mask-Aware Processing**
All components properly handle padding masks to ensure padded positions don't affect attention and pooling.

**Benefits:**
- Correct handling of variable-length sequences
- Improved training stability
- Better generalization

---

## Architecture Components

### File Structure

```
msa-tf2/models/
├── __init__.py                  # Package exports
├── positional.py                # Positional embeddings and mask utilities
├── blocks.py                    # Transformer and cross-attention blocks
├── fusion.py                    # Fusion components
├── msa_seqlevel.py             # Main model class
└── README.md                    # Usage documentation
```

### 1. Positional Embeddings (`positional.py`)

#### `LearnablePositionalEmbedding`
Adds trainable position vectors to input sequences.

```python
class LearnablePositionalEmbedding(tf.keras.layers.Layer):
    """
    Input: (batch, seq_len, model_dim)
    Output: (batch, seq_len, model_dim) with positions added
    """
```

**Key Features:**
- Learnable position vectors (not sinusoidal)
- Supports variable sequence lengths
- One embedding per position in the sequence

#### Mask Utilities

**`create_padding_mask(sequences, pad_value=0.0)`**
- Creates boolean masks from sequences
- Returns: (batch, seq_len) where True = valid, False = padding

**`create_look_ahead_mask(seq_len)`**
- Creates causal masks for autoregressive models
- Returns: (seq_len, seq_len) lower triangular

**`combine_masks(padding_mask, look_ahead_mask)`**
- Combines multiple mask types
- Useful for decoder-style architectures

---

### 2. Building Blocks (`blocks.py`)

#### `TransformerBlock`
Standard transformer encoder block with self-attention.

```python
class TransformerBlock(tf.keras.layers.Layer):
    """
    Architecture:
      1. Multi-head self-attention + residual
      2. Layer normalization
      3. Feed-forward network + residual
      4. Layer normalization
    """
```

**Parameters:**
- `model_dim`: Model dimension (e.g., 256)
- `num_heads`: Number of attention heads (e.g., 4)
- `ff_dim`: FFN hidden dimension (e.g., 512)
- `dropout_rate`: Dropout probability (default: 0.1)

**Key Implementation Details:**
- Uses `key_dim = model_dim // num_heads` to avoid dimension issues
- GELU activation in FFN
- Mask expansion for proper attention computation
- Pre-norm architecture (LayerNorm before attention)

#### `CrossAttentionBlock`
Cross-modal attention where one modality attends to another.

```python
class CrossAttentionBlock(tf.keras.layers.Layer):
    """
    Allows query modality to attend to context modality.
    Example: Text queries attend to Audio keys/values
    """
```

**Key Features:**
- Separate query and context inputs
- Separate masks for query and context
- Residual connections to query
- Can be used for any modality pair

---

### 3. Fusion Components (`fusion.py`)

#### `CrossModalFusion`
Orchestrates sequence-level cross-attention between all modalities.

```python
class CrossModalFusion(tf.keras.layers.Layer):
    """
    Default (unidirectional):
      - Text ← Audio
      - Text ← Video
    
    Bidirectional (optional):
      - Text ↔ Audio
      - Text ↔ Video
    """
```

**Parameters:**
- `model_dim`: Model dimension
- `num_heads`: Attention heads
- `ff_dim`: FFN dimension
- `n_layers`: Number of fusion layers to stack
- `bidirectional`: Enable symmetric cross-attention
- `dropout_rate`: Dropout probability

**Architecture Flow:**
```
For each fusion layer:
  1. Text ← Audio (text attends to audio)
  2. Text ← Video (text attends to video)
  3. [Optional] Audio ← Text
  4. [Optional] Video ← Text
```

**Why Text as Anchor?**
- Text typically has the most semantic information
- Audio/video provide complementary non-verbal cues
- Empirically works well for sentiment analysis

#### `AdaptiveFusionHead`
Pools sequences and learns optimal modality weighting.

```python
class AdaptiveFusionHead(tf.keras.layers.Layer):
    """
    Steps:
      1. Mask-aware pooling per modality
      2. Compute softmax weights over modalities
      3. Weighted sum to produce fused representation
    """
```

**Pooling Methods:**

1. **Mean Pooling** (`pooling_method='mean'`)
   - Average over valid (non-padded) positions
   - Fast and simple
   - Good default choice

2. **Attention Pooling** (`pooling_method='attention'`)
   - Learns attention weights over sequence
   - More expressive
   - Slightly slower

**Mask-Aware Pooling:**
```python
# Only average over valid positions
masked_sequence = sequence * mask_expanded
pooled = tf.reduce_sum(masked_sequence, axis=1) / count_valid_positions
```

---

### 4. Main Model (`msa_seqlevel.py`)

#### `ModalityEncoder`
Encodes a single modality sequence.

```python
class ModalityEncoder(tf.keras.layers.Layer):
    """
    Pipeline:
      Input → Dense(model_dim) → Positional Embeddings
            → Transformer Blocks → Output
    
    No pooling applied!
    """
```

**Components:**
- Dense projection to model dimension
- Learnable positional embeddings
- Stack of `TransformerBlock` layers
- Dropout after positional embeddings

#### `MSASeqLevelModel`
Full end-to-end model for sentiment prediction.

```python
class MSASeqLevelModel(tf.keras.Model):
    """
    Architecture:
      Inputs: (text, audio, video) sequences
        ↓
      Modality Encoders (3×)
        ↓
      CrossModalFusion
        ↓
      AdaptiveFusionHead
        ↓
      Regression Head
        ↓
      Output: Sentiment score
    """
```

**Key Features:**

1. **Flexible Input Handling:**
   - Auto-generates masks: `model((text, audio, video))`
   - Explicit masks: `model((text, audio, video, t_mask, a_mask, v_mask))`

2. **Mixed Precision Compatible:**
   - Output layer explicitly set to `float32`
   - Works with `mixed_float16` policy

3. **Configurable Architecture:**
   ```python
   model = MSASeqLevelModel(
       seq_len=64,                    # Sequence length
       text_dim=300,                  # Input dimensions
       audio_dim=74,
       video_dim=47,
       model_dim=256,                 # Model dimension
       num_heads=4,                   # Attention heads
       ff_dim=512,                    # FFN dimension
       n_layers_mod=2,                # Encoder layers
       n_layers_fuse=2,               # Fusion layers
       bidirectional_fusion=False,    # Cross-attention type
       pooling_method='mean',         # Pooling strategy
       dropout_rate=0.1               # Dropout rate
   )
   ```

---

## Data Flow Example

Let's trace a batch through the model:

### Input
```
text:  (4, 64, 300)  # batch=4, seq_len=64, text_features=300
audio: (4, 64, 74)   # audio_features=74
video: (4, 64, 47)   # video_features=47
```

### Step 1: Mask Generation
```python
text_mask:  (4, 64)  # True for valid positions
audio_mask: (4, 64)
video_mask: (4, 64)
```

### Step 2: Modality Encoders
```python
# Each encoder:
# Dense projection → Positional embeddings → Transformer blocks

text_encoded:  (4, 64, 256)  # model_dim=256
audio_encoded: (4, 64, 256)
video_encoded: (4, 64, 256)
```

### Step 3: Cross-Modal Fusion
```python
# Text attends to audio and video
# (optionally bidirectional)

text_fused:  (4, 64, 256)
audio_fused: (4, 64, 256)
video_fused: (4, 64, 256)
```

### Step 4: Adaptive Fusion Head
```python
# Mask-aware pooling per modality
text_pooled:  (4, 256)
audio_pooled: (4, 256)
video_pooled: (4, 256)

# Stack and compute adaptive weights
modality_stack: (4, 3, 256)  # 3 modalities
fusion_weights: (4, 3)       # softmax weights

# Weighted sum
fused_repr: (4, 256)
```

### Step 5: Regression Head
```python
output: (4, 1)  # sentiment predictions
```

---

## Comparison with Original MSAModel

| Feature | Original MSAModel | MSASeqLevelModel |
|---------|------------------|------------------|
| **Pooling Strategy** | Early (after encoders) | Late (after fusion) |
| **Fusion Mechanism** | Concatenation + Transformer | Cross-Attention |
| **Modality Interaction** | Token-level (pooled) | Sequence-level |
| **Positional Encoding** | None | Learnable |
| **Mask Handling** | Basic | Comprehensive |
| **Adaptive Weighting** | Optional | Always (mask-aware) |
| **Reusability** | Monolithic | Modular components |
| **Parameters** | ~800K | ~1.1M (varies) |
| **Inference Speed** | Faster | Slightly slower |
| **Expressiveness** | Moderate | High |

---

## Training Guide

### Basic Training

```bash
cd /Users/rizkimuhammad/Honours/msa-tf2

# Activate environment
conda activate msa-tf2

# Train with default settings
python train_seqlevel.py

# Train with custom settings
python train_seqlevel.py \
    --model_dim 256 \
    --num_heads 8 \
    --n_layers_mod 3 \
    --n_layers_fuse 2 \
    --bidirectional_fusion \
    --pooling_method attention \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 100
```

### Hyperparameter Recommendations

**Small Model (Fast, Low Memory):**
```bash
python train_seqlevel.py \
    --model_dim 64 \
    --num_heads 2 \
    --ff_dim 128 \
    --n_layers_mod 1 \
    --n_layers_fuse 1 \
    --batch_size 64
```

**Medium Model (Balanced):**
```bash
python train_seqlevel.py \
    --model_dim 128 \
    --num_heads 4 \
    --ff_dim 256 \
    --n_layers_mod 2 \
    --n_layers_fuse 1 \
    --batch_size 32
```

**Large Model (Best Performance):**
```bash
python train_seqlevel.py \
    --model_dim 256 \
    --num_heads 8 \
    --ff_dim 512 \
    --n_layers_mod 3 \
    --n_layers_fuse 2 \
    --bidirectional_fusion \
    --pooling_method attention \
    --batch_size 16 \
    --learning_rate 5e-5
```

### Mixed Precision Training

```bash
python train_seqlevel.py \
    --mixed_precision \
    --model_dim 256 \
    --batch_size 64
```

Benefits:
- ~2x faster training
- ~50% memory reduction
- Maintains accuracy (output is still float32)

---

## Testing and Validation

### Run Full Test Suite

```bash
python test_seqlevel_model.py
```

Tests include:
1. Basic instantiation and forward pass
2. Explicit mask handling
3. Model summary
4. Mixed precision compatibility
5. Training loop
6. Individual fusion components
7. Different configurations

### Compare Models

```bash
python compare_models.py
```

Compares:
- Parameter counts
- Architecture differences
- Forward pass timing
- Output distributions

---

## Extending the Architecture

### Adding New Modalities

To add a fourth modality (e.g., depth):

1. **Update ModalityEncoder:**
   ```python
   self.depth_encoder = ModalityEncoder(
       input_dim=depth_dim,
       model_dim=model_dim,
       seq_len=seq_len,
       num_heads=num_heads,
       ff_dim=ff_dim,
       n_layers=n_layers_mod,
       name='depth_encoder'
   )
   ```

2. **Update CrossModalFusion:**
   Add cross-attention blocks for the new modality.

3. **Update AdaptiveFusionHead:**
   Change `num_modalities=4`.

### Using with Different Encoders

The fusion components work with any encoder:

```python
# Example: Using RNN encoders instead of transformers
class CustomMSAModel(tf.keras.Model):
    def __init__(self, model_dim=256):
        super().__init__()
        
        # RNN encoders
        self.text_lstm = LSTM(model_dim, return_sequences=True)
        self.audio_lstm = LSTM(model_dim, return_sequences=True)
        self.video_lstm = LSTM(model_dim, return_sequences=True)
        
        # Reuse fusion components!
        self.fusion = CrossModalFusion(model_dim, num_heads=4, ...)
        self.adaptive_fusion = AdaptiveFusionHead(model_dim, ...)
        self.output_layer = Dense(1, dtype='float32')
    
    def call(self, inputs, training=False):
        text, audio, video = inputs
        
        # Encode with LSTMs
        text_enc = self.text_lstm(text, training=training)
        audio_enc = self.audio_lstm(audio, training=training)
        video_enc = self.video_lstm(video, training=training)
        
        # Apply reusable fusion
        text_fused, audio_fused, video_fused = self.fusion(
            text_enc, audio_enc, video_enc, training=training
        )
        
        fused = self.adaptive_fusion(
            [text_fused, audio_fused, video_fused],
            training=training
        )
        
        return self.output_layer(fused)
```

---

## Performance Optimization

### Memory Optimization
1. Reduce `model_dim` (biggest impact)
2. Reduce `ff_dim`
3. Reduce `n_layers_mod` and `n_layers_fuse`
4. Use smaller batch sizes
5. Enable mixed precision

### Speed Optimization
1. Enable mixed precision
2. Reduce `n_layers_fuse`
3. Use unidirectional fusion
4. Use mean pooling instead of attention pooling
5. Reduce `seq_len` if possible

### Accuracy Optimization
1. Increase `model_dim`
2. Enable bidirectional fusion
3. Use attention pooling
4. Increase `n_layers_mod` and `n_layers_fuse`
5. Lower learning rate (1e-5)
6. Longer training with patience

---

## Common Issues and Solutions

### Issue 1: Out of Memory

**Solution:**
- Reduce `model_dim` and `batch_size`
- Enable mixed precision
- Use gradient accumulation

### Issue 2: NaN Loss

**Solution:**
- Lower learning rate (try 1e-5)
- Enable gradient clipping: `optimizer.clipnorm = 1.0`
- Check for NaN in input data
- Reduce `dropout_rate`

### Issue 3: Slow Training

**Solution:**
- Enable mixed precision
- Reduce `n_layers_fuse`
- Use CPU parallelism (already enabled)
- Profile with TensorBoard

### Issue 4: Dimension Mismatch in Attention

**Solution:**
- Ensure `model_dim % num_heads == 0`
- The code uses `key_dim = model_dim // num_heads` correctly

---

## Future Work

### Potential Enhancements

1. **Hierarchical Attention:**
   - Attend at multiple temporal scales
   - Coarse-to-fine fusion

2. **Dynamic Fusion:**
   - Learn when to fuse modalities
   - Gating mechanisms for fusion

3. **Contrastive Learning:**
   - Self-supervised pre-training
   - Cross-modal alignment objectives

4. **Temporal Modeling:**
   - Explicit temporal convolutions
   - Causal transformers for streaming

5. **Multi-Task Learning:**
   - Joint training on multiple MSA datasets
   - Auxiliary tasks (emotion recognition, topic classification)

---

## References

- **Deep-HOSeq:** Deep High-Order Multimodal Sequence Fusion
- **Transformer:** Attention Is All You Need (Vaswani et al., 2017)
- **Cross-Attention:** Used in Vision-Language models (CLIP, BLIP)
- **CMU-MOSI/MOSEI:** Multimodal sentiment analysis benchmarks

---

## Contact and Support

For questions or issues:
1. Check `models/README.md` for usage examples
2. Run `test_seqlevel_model.py` to validate installation
3. See training logs in `weights/` directory

## Changelog

### Version 1.0 (2025-10-18)
- Initial implementation
- Modality encoders with positional embeddings
- Sequence-level cross-modal fusion
- Adaptive fusion head
- Comprehensive test suite
- Training and comparison scripts





