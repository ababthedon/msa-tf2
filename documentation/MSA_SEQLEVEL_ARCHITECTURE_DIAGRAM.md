# MSA Sequence-Level Model Architecture Diagram

## Overview Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT SEQUENCES                                 │
│                                                                              │
│   Text (B, 20, 300)      Audio (B, 20, 74)      Video (B, 20, 47/713)      │
│         │                      │                         │                   │
│         │                      │                         │                   │
│      [mask_t]              [mask_a]                  [mask_v]               │
└─────────┬──────────────────────┬─────────────────────────┬──────────────────┘
          │                      │                         │
          ▼                      ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MODALITY ENCODERS                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Text Encoder   │  │  Audio Encoder  │  │  Video Encoder  │             │
│  │                 │  │                 │  │                 │             │
│  │ Dense(model_dim)│  │ Dense(model_dim)│  │ Dense(model_dim)│             │
│  │       ↓         │  │       ↓         │  │       ↓         │             │
│  │  Positional     │  │  Positional     │  │  Positional     │             │
│  │  Embeddings     │  │  Embeddings     │  │  Embeddings     │             │
│  │       ↓         │  │       ↓         │  │       ↓         │             │
│  │  Transformer    │  │  Transformer    │  │  Transformer    │             │
│  │  Block × n      │  │  Block × n      │  │  Block × n      │             │
│  │  (Self-Attn)    │  │  (Self-Attn)    │  │  (Self-Attn)    │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│         │                      │                         │                   │
│    (B, 20, D)             (B, 20, D)                (B, 20, D)              │
└─────────┬──────────────────────┬─────────────────────────┬──────────────────┘
          │                      │                         │
          │                      │                         │
          ▼                      ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CROSS-MODAL FUSION                                      │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────┐             │
│  │              Fusion Layer 1 (repeated n times)             │             │
│  │                                                            │             │
│  │  Text ← Audio  (Text attends to Audio via Cross-Attn)     │             │
│  │        ↓                                                   │             │
│  │  Text ← Video  (Text attends to Video via Cross-Attn)     │             │
│  │                                                            │             │
│  │  [Optional Bidirectional]                                 │             │
│  │  Audio ← Text  (Audio attends to Text via Cross-Attn)     │             │
│  │  Video ← Text  (Video attends to Text via Cross-Attn)     │             │
│  └────────────────────────────────────────────────────────────┘             │
│         │                      │                         │                   │
│    Text_fused             Audio_fused              Video_fused              │
│    (B, 20, D)             (B, 20, D)               (B, 20, D)               │
└─────────┬──────────────────────┬─────────────────────────┬──────────────────┘
          │                      │                         │
          │                      │                         │
          ▼                      ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ADAPTIVE FUSION HEAD                                    │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │ Mask-Aware Pool  │  │ Mask-Aware Pool  │  │ Mask-Aware Pool  │          │
│  │  (Mean/Attn)     │  │  (Mean/Attn)     │  │  (Mean/Attn)     │          │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│           │                     │                      │                     │
│           │                     │                      │                     │
│      Text_pooled           Audio_pooled           Video_pooled              │
│       (B, D)                 (B, D)                 (B, D)                   │
│           │                     │                      │                     │
│           └─────────────────────┴──────────────────────┘                    │
│                                  │                                           │
│                           Stack modalities                                   │
│                            (B, 3, D)                                         │
│                                  │                                           │
│                                  ▼                                           │
│                    ┌──────────────────────────┐                             │
│                    │  Compute Fusion Weights  │                             │
│                    │  Dense(D/2) → Dropout    │                             │
│                    │  Dense(3, softmax)       │                             │
│                    └──────────┬───────────────┘                             │
│                               │                                              │
│                        Fusion Weights                                        │
│                            (B, 3)                                            │
│                               │                                              │
│                               ▼                                              │
│                    ┌──────────────────────────┐                             │
│                    │   Weighted Sum           │                             │
│                    │   Σ weight_i * modal_i   │                             │
│                    └──────────┬───────────────┘                             │
│                               │                                              │
│                        Fused Vector                                          │
│                            (B, D)                                            │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          REGRESSION HEAD                                     │
│                                                                              │
│                        Dense(1, dtype=float32)                               │
│                                                                              │
│                                  │                                           │
│                                  ▼                                           │
│                           Sentiment Score                                    │
│                               (B, 1)                                         │
└─────────────────────────────────────────────────────────────────────────────┘

Legend:
  B = Batch size
  D = model_dim (e.g., 128 or 256)
  20 = Sequence length
  n = Number of layers (n_layers_mod for encoders, n_layers_fuse for fusion)
```

---

## Detailed Component Breakdown

### 1. Modality Encoder (per modality)

```
Input: (B, seq_len, input_dim)
  │
  ▼
Dense Projection → (B, seq_len, model_dim)
  │
  ▼
Add Positional Embeddings (learnable)
  │
  ▼
Dropout(rate=0.1)
  │
  ▼
┌─────────────────────────────┐
│   Transformer Block 1       │
│   ┌───────────────────┐     │
│   │ Multi-Head        │     │
│   │ Self-Attention    │     │
│   │ (with mask)       │     │
│   └─────────┬─────────┘     │
│             │               │
│        LayerNorm            │
│             │               │
│             ▼               │
│   ┌───────────────────┐     │
│   │ Feed-Forward      │     │
│   │ Dense(ff_dim)     │     │
│   │ GELU              │     │
│   │ Dense(model_dim)  │     │
│   │ Dropout           │     │
│   └─────────┬─────────┘     │
│             │               │
│        LayerNorm            │
└─────────────┬───────────────┘
              │
          (repeat n times)
              │
              ▼
Output: (B, seq_len, model_dim)
```

### 2. Cross-Attention Block

```
Query Modality:    (B, seq_len, model_dim)
Context Modality:  (B, seq_len, model_dim)
  │                    │
  │                    │
  └────────┬───────────┘
           │
           ▼
┌──────────────────────────────┐
│  Multi-Head Cross-Attention  │
│                              │
│  Q = Query                   │
│  K, V = Context              │
│                              │
│  Attention = softmax(QK'/√d) │
│  Output = Attention × V      │
└──────────┬───────────────────┘
           │
      Residual + LayerNorm
           │
           ▼
┌──────────────────────────────┐
│   Feed-Forward Network       │
│   Dense(ff_dim, GELU)        │
│   Dropout                    │
│   Dense(model_dim)           │
└──────────┬───────────────────┘
           │
      Residual + LayerNorm
           │
           ▼
Enhanced Query: (B, seq_len, model_dim)
```

### 3. Mask-Aware Pooling

#### Mean Pooling:
```
Input: (B, seq_len, model_dim)
Mask:  (B, seq_len) [True=valid, False=padding]
  │
  ▼
Expand mask: (B, seq_len, 1)
  │
  ▼
Masked sequence = Input × mask
  │
  ▼
Sum over seq_len
  │
  ▼
Divide by count of valid positions
  │
  ▼
Output: (B, model_dim)
```

#### Attention Pooling:
```
Input: (B, seq_len, model_dim)
Mask:  (B, seq_len)
  │
  ▼
Compute attention scores:
  Dense(model_dim//4, tanh)
  Dense(1)
  → (B, seq_len, 1)
  │
  ▼
Mask out padding: score × mask + (1-mask) × (-1e9)
  │
  ▼
Softmax over seq_len
  │
  ▼
Weighted sum: Input × attention_weights
  │
  ▼
Output: (B, model_dim)
```

---

## Data Flow Example

### Input Dimensions:
```
Text:  (4, 20, 300)   # 4 samples, 20 timesteps, 300 GloVe dims
Audio: (4, 20, 74)    # 4 samples, 20 timesteps, 74 COVAREP dims
Video: (4, 20, 47)    # 4 samples, 20 timesteps, 47 OpenFace dims (MOSI)
                      # or (4, 20, 713) for MOSEI
```

### After Modality Encoders (model_dim=256):
```
Text_enc:  (4, 20, 256)
Audio_enc: (4, 20, 256)
Video_enc: (4, 20, 256)
```

### After Cross-Modal Fusion:
```
Text_fused:  (4, 20, 256)   # Text enhanced with Audio & Video info
Audio_fused: (4, 20, 256)   # Audio enhanced (if bidirectional)
Video_fused: (4, 20, 256)   # Video enhanced (if bidirectional)
```

### After Adaptive Fusion Head Pooling:
```
Text_pooled:  (4, 256)
Audio_pooled: (4, 256)
Video_pooled: (4, 256)

Stack: (4, 3, 256)
```

### After Fusion Weighting:
```
Fusion weights: (4, 3)   # Softmax weights per modality
                         # e.g., [0.5, 0.3, 0.2] for [text, audio, video]

Fused representation: (4, 256)   # Weighted sum
```

### Final Output:
```
Sentiment predictions: (4, 1)   # One score per sample
```

---

## Key Design Choices

### 1. **No Early Pooling**
- Sequences maintain full temporal length (20 timesteps) through encoders and fusion
- Pooling only occurs in the final Adaptive Fusion Head
- **Benefit**: Rich sequence-level cross-modal interactions

### 2. **Text as Anchor**
- Text modality receives information from Audio and Video via cross-attention
- Rationale: Text typically contains most semantic content; Audio/Video provide complementary non-verbal cues
- Optional bidirectional fusion for symmetric interactions

### 3. **Mask-Aware Processing**
- All attention operations respect padding masks
- Pooling averages over valid (non-padded) positions only
- **Benefit**: Correct handling of variable-length sequences

### 4. **Adaptive Modality Weighting**
- Network learns optimal importance of each modality
- Weights computed dynamically per sample via softmax
- **Benefit**: Handles cases where some modalities are more informative than others

### 5. **Learnable Positional Embeddings**
- Unlike fixed sinusoidal encodings, positions are learned
- One embedding vector per position (max_len = seq_len)
- **Benefit**: Flexible representation of temporal structure

---

## Hyperparameter Summary

| Parameter | Typical Value | Range | Notes |
|-----------|---------------|-------|-------|
| `seq_len` | 20 | Fixed | Temporal length after sampling |
| `model_dim` | 128 or 256 | 64–256 | Must be divisible by `num_heads` |
| `num_heads` | 4–8 | 2–8 | Attention heads per layer |
| `ff_dim` | 256–512 | 128–512 | Feed-forward hidden dimension |
| `n_layers_mod` | 2–3 | 1–3 | Transformer layers per encoder |
| `n_layers_fuse` | 1–2 | 1–2 | Cross-attention fusion layers |
| `dropout_rate` | 0.1 | 0.0–0.3 | Applied in all components |
| `bidirectional_fusion` | False | True/False | Enable symmetric cross-attention |
| `pooling_method` | 'mean' | 'mean'/'attention' | Pooling strategy in fusion head |

---

## Computational Complexity

### Per-Sample Complexity (approximate):

**Modality Encoders** (per modality):
- Self-attention: O(S² × D) where S=seq_len, D=model_dim
- Feed-forward: O(S × D × ff_dim)
- Total per encoder: O(n_layers_mod × S × (S×D + D×ff_dim))

**Cross-Modal Fusion**:
- Cross-attention: O(S² × D) per attention block
- Total: O(n_layers_fuse × n_attention_ops × S² × D)
  - n_attention_ops = 2 (unidirectional) or 4 (bidirectional)

**Adaptive Fusion Head**:
- Pooling: O(S × D)
- Weight computation: O(D × num_modalities)

**Total**: Dominated by attention operations, approximately O(S² × D × (n_layers_mod + n_layers_fuse))

---

## Comparison with Early-Pooling Baseline

| Aspect | Early Pooling | Sequence-Level (This Model) |
|--------|---------------|---------------------------|
| **Pooling location** | After each encoder | After fusion |
| **Fusion input** | Pooled vectors (B, D) | Full sequences (B, S, D) |
| **Cross-modal mechanism** | Concatenation + transformer | Explicit cross-attention |
| **Temporal alignment** | Lost after pooling | Maintained through fusion |
| **Parameters** | ~800K | ~1.1M |
| **Expressiveness** | Limited | High |
| **Speed** | Faster | Slightly slower |


