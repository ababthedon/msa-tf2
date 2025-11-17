# MSA-TF2 Models

This directory contains multimodal sentiment analysis models and reusable components for building MSA architectures.

## Overview

### Existing Models

- **MSAModel** (`msamodel.py`): Original transformer-based MSA model with early pooling after modality encoders
- **ImprovedMSAModel** (`improved_msamodel.py`): Enhanced version with various improvements
- **BalancedMSAModel** (`balanced_msamodel.py`): Variant with balanced training strategies

### New Sequence-Level Architecture

The new architecture maintains sequence-level information through fusion (no early pooling) and is designed to be modular and reusable.

**Key Components:**

1. **MSASeqLevelModel** (`msa_seqlevel.py`): Full sequence-level model
2. **ModalityEncoder** (`msa_seqlevel.py`): Modality-specific transformer encoder
3. **CrossModalFusion** (`fusion.py`): Sequence-level cross-attention between modalities
4. **AdaptiveFusionHead** (`fusion.py`): Learnable modality weighting with mask-aware pooling
5. **TransformerBlock** (`blocks.py`): Standard self-attention encoder block
6. **CrossAttentionBlock** (`blocks.py`): Cross-modal attention block
7. **LearnablePositionalEmbedding** (`positional.py`): Positional embeddings for sequences

---

## Quick Start: Using MSASeqLevelModel

### Basic Usage

```python
import tensorflow as tf
from models import MSASeqLevelModel

# Create model instance
model = MSASeqLevelModel(
    seq_len=64,           # Maximum sequence length
    text_dim=300,         # Text feature dimension (e.g., word2vec)
    audio_dim=74,         # Audio feature dimension (e.g., COVAREP)
    video_dim=47,         # Video feature dimension (e.g., Facet)
    model_dim=256,        # Transformer hidden dimension
    num_heads=4,          # Number of attention heads
    ff_dim=512,           # Feed-forward network dimension
    n_layers_mod=2,       # Transformer layers per modality encoder
    n_layers_fuse=2,      # Cross-attention fusion layers
    bidirectional_fusion=False,  # Use bidirectional cross-attention
    pooling_method='mean',       # 'mean' or 'attention'
    dropout_rate=0.1      # Dropout probability
)

# Prepare dummy inputs
batch_size = 4
text_seq = tf.random.normal((batch_size, 64, 300))
audio_seq = tf.random.normal((batch_size, 64, 74))
video_seq = tf.random.normal((batch_size, 64, 47))

# Forward pass (auto-generates padding masks)
output = model((text_seq, audio_seq, video_seq), training=False)
print(f"Output shape: {output.shape}")  # (4, 1)

# Display model architecture
model.summary()
```

### Training Example

```python
from models import MSASeqLevelModel
from utils.data_loader import make_dataset
import tensorflow as tf

# Create model
model = MSASeqLevelModel(
    seq_len=20,
    text_dim=300,
    audio_dim=74,
    video_dim=47,
    model_dim=128,
    num_heads=4,
    ff_dim=256,
    n_layers_mod=2,
    n_layers_fuse=1
)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='mae',
    metrics=['mae', 'mse']
)

# Load datasets
train_data = make_dataset('./data', split='train', batch_size=32)
val_data = make_dataset('./data', split='valid', batch_size=32)

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=5
        )
    ]
)

# Evaluate
test_data = make_dataset('./data', split='test', batch_size=32)
test_results = model.evaluate(test_data)
print(f"Test MAE: {test_results[1]:.4f}")
```

### Using Explicit Masks

```python
from models import MSASeqLevelModel, create_padding_mask
import tensorflow as tf

model = MSASeqLevelModel(seq_len=64, text_dim=300, audio_dim=74, 
                         video_dim=47, model_dim=256, num_heads=4, 
                         ff_dim=512, n_layers_mod=2, n_layers_fuse=2)

# Prepare inputs with padding
text_seq = tf.random.normal((4, 64, 300))
audio_seq = tf.random.normal((4, 64, 74))
video_seq = tf.random.normal((4, 64, 47))

# Create custom masks (True = valid, False = padded)
text_mask = create_padding_mask(text_seq)
audio_mask = create_padding_mask(audio_seq)
video_mask = create_padding_mask(video_seq)

# Forward pass with explicit masks
output = model(
    (text_seq, audio_seq, video_seq, text_mask, audio_mask, video_mask),
    training=False
)
```

### Mixed Precision Training

The model is compatible with mixed precision training. The output layer always returns float32 to avoid precision issues with loss computation.

```python
import tensorflow as tf
from tensorflow.keras import mixed_precision
from models import MSASeqLevelModel

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Create model (output will still be float32)
model = MSASeqLevelModel(
    seq_len=64, text_dim=300, audio_dim=74, video_dim=47,
    model_dim=256, num_heads=4, ff_dim=512,
    n_layers_mod=2, n_layers_fuse=2
)

# Compile with loss scaling for mixed precision
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
optimizer = mixed_precision.LossScaleOptimizer(optimizer)

model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

# Train as usual...
```

---

## Architecture Details

### Sequence-Level Fusion Design

The key innovation is maintaining sequence-level information through fusion:

1. **No Early Pooling**: Each modality encoder outputs full sequences (batch, seq_len, model_dim)
2. **Cross-Modal Attention**: Text attends to audio and video at sequence level
3. **Bidirectional Option**: Audio/video can attend back to text if enabled
4. **Late Pooling**: Only pool sequences in the adaptive fusion head

This allows richer cross-modal interactions compared to early pooling approaches.

### Model Flow

```
Input: (text, audio, video) sequences
  ↓
Modality Encoders (per-modality transformers)
  → Text:  Dense → PosEmbed → Transformer Blocks
  → Audio: Dense → PosEmbed → Transformer Blocks  
  → Video: Dense → PosEmbed → Transformer Blocks
  ↓
Cross-Modal Fusion (sequence-level cross-attention)
  → Text ← Audio (text queries, audio keys/values)
  → Text ← Video (text queries, video keys/values)
  → [Optional] Audio ← Text, Video ← Text
  ↓
Adaptive Fusion Head
  → Mask-aware pooling per modality
  → Learnable modality weights (softmax)
  → Weighted sum of modalities
  ↓
Regression Head (Dense(1))
  ↓
Output: Sentiment prediction (batch, 1)
```

---

## Reusable Components

The fusion components are designed to be modular and can be used with other MSA backbones.

### Using CrossModalFusion Independently

```python
from models import CrossModalFusion
import tensorflow as tf

# Create fusion layer
fusion = CrossModalFusion(
    model_dim=256,
    num_heads=4,
    ff_dim=512,
    n_layers=2,
    bidirectional=True
)

# Prepare sequence inputs (any encoder outputs)
text = tf.random.normal((4, 64, 256))
audio = tf.random.normal((4, 64, 256))
video = tf.random.normal((4, 64, 256))

# Apply cross-modal fusion
text_fused, audio_fused, video_fused = fusion(
    text_seq=text,
    audio_seq=audio,
    video_seq=video,
    training=True
)

# Use fused representations for downstream tasks
```

### Using AdaptiveFusionHead Independently

```python
from models import AdaptiveFusionHead
import tensorflow as tf

# Create fusion head
fusion_head = AdaptiveFusionHead(
    model_dim=256,
    num_modalities=3,
    pooling_method='attention'  # or 'mean'
)

# Prepare modality sequences
modalities = [
    tf.random.normal((4, 64, 256)),  # text
    tf.random.normal((4, 64, 256)),  # audio
    tf.random.normal((4, 64, 256))   # video
]

# Fuse with learnable weights
fused_repr = fusion_head(
    modality_sequences=modalities,
    training=True
)  # shape: (4, 256)
```

### Building Custom Architectures

You can mix and match components to create custom MSA models:

```python
from models import (
    TransformerBlock,
    CrossAttentionBlock,
    CrossModalFusion,
    AdaptiveFusionHead,
    LearnablePositionalEmbedding
)
import tensorflow as tf

class CustomMSAModel(tf.keras.Model):
    def __init__(self, model_dim=256, **kwargs):
        super().__init__(**kwargs)
        
        # Use RNN encoders instead of transformers
        self.text_rnn = tf.keras.layers.LSTM(model_dim, return_sequences=True)
        self.audio_rnn = tf.keras.layers.LSTM(model_dim, return_sequences=True)
        self.video_rnn = tf.keras.layers.LSTM(model_dim, return_sequences=True)
        
        # Reuse sequence-level fusion components
        self.fusion = CrossModalFusion(
            model_dim=model_dim,
            num_heads=4,
            ff_dim=512,
            n_layers=1
        )
        
        self.adaptive_fusion = AdaptiveFusionHead(
            model_dim=model_dim,
            num_modalities=3
        )
        
        self.output_layer = tf.keras.layers.Dense(1, dtype='float32')
    
    def call(self, inputs, training=False):
        text, audio, video = inputs
        
        # Encode with RNNs
        text_enc = self.text_rnn(text, training=training)
        audio_enc = self.audio_rnn(audio, training=training)
        video_enc = self.video_rnn(video, training=training)
        
        # Apply sequence-level fusion (reusable!)
        text_fused, audio_fused, video_fused = self.fusion(
            text_enc, audio_enc, video_enc, training=training
        )
        
        # Adaptive fusion head (reusable!)
        fused = self.adaptive_fusion(
            [text_fused, audio_fused, video_fused],
            training=training
        )
        
        return self.output_layer(fused)
```

---

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Description | Typical Range | Default |
|-----------|-------------|---------------|---------|
| `model_dim` | Transformer hidden size | 64-512 | 256 |
| `num_heads` | Attention heads | 2-8 | 4 |
| `ff_dim` | FFN dimension | 128-1024 | 512 |
| `n_layers_mod` | Encoder layers per modality | 1-4 | 2 |
| `n_layers_fuse` | Cross-attention fusion layers | 1-3 | 1 |
| `dropout_rate` | Dropout probability | 0.0-0.3 | 0.1 |
| `learning_rate` | Adam learning rate | 1e-5 to 1e-3 | 1e-4 |
| `bidirectional_fusion` | Bidirectional cross-attention | True/False | False |
| `pooling_method` | Fusion head pooling | 'mean', 'attention' | 'mean' |

### Memory Considerations

- **model_dim** has the largest impact on memory
- **num_heads** must divide **model_dim** evenly (key_dim = model_dim // num_heads)
- Sequence length (seq_len) affects memory quadratically due to attention
- Use smaller batch sizes for larger models

---

## Model Comparison

| Feature | MSAModel | MSASeqLevelModel |
|---------|----------|------------------|
| Pooling | Early (after each encoder) | Late (after fusion) |
| Fusion Type | Concatenation + transformer | Sequence-level cross-attention |
| Modality Interaction | Limited (pooled vectors) | Rich (full sequences) |
| Positional Embeddings | No | Yes (learnable) |
| Adaptive Fusion | Optional | Yes (mask-aware) |
| Reusable Components | No | Yes |
| Memory Usage | Lower | Higher |
| Expressiveness | Moderate | High |

---

## Testing

Run the provided test script to validate the architecture:

```bash
cd /Users/rizkimuhammad/Honours/msa-tf2
python test_seqlevel_model.py
```

This will:
- Create a model with test hyperparameters
- Run forward pass with dummy data
- Verify output shapes and dtypes
- Test mixed precision compatibility
- Perform a short training run

---

## Troubleshooting

### Common Issues

**1. Dimension Mismatch in MultiHeadAttention**

```
Error: key_dim * num_heads != model_dim
```

**Solution:** Ensure `model_dim` is divisible by `num_heads`. The implementation uses `key_dim = model_dim // num_heads`.

**2. Out of Memory**

**Solution:** 
- Reduce `model_dim` or `ff_dim`
- Decrease batch size
- Reduce `n_layers_mod` or `n_layers_fuse`
- Use gradient checkpointing (advanced)

**3. Slow Training**

**Solution:**
- Enable mixed precision (see example above)
- Reduce sequence length if possible
- Use smaller `n_layers_fuse`

**4. NaN Loss**

**Solution:**
- Lower learning rate (try 1e-5)
- Enable gradient clipping: `optimizer.clipnorm = 1.0`
- Check for NaN in input data
- Ensure masks are correctly generated

---

## Citation

If you use this architecture in your research, please cite:

```
@misc{msa-seqlevel,
  author = {Honours Project},
  title = {Sequence-Level Multimodal Sentiment Analysis with Cross-Attention Fusion},
  year = {2025},
  note = {MSA-TF2 implementation}
}
```

---

## License

See the main repository LICENSE file for details.






