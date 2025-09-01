# Multimodal Sentiment Analysis with Transformers (MSA-TF2)

A TensorFlow 2 implementation of a transformer-based multimodal sentiment analysis model that processes text, audio, and video modalities for sentiment prediction.

## Overview

This project implements a sophisticated multimodal sentiment analysis system using transformer architectures. The model processes three different modalities:
- **Text**: Pre-computed text embeddings 
- **Audio**: Audio feature sequences
- **Video**: Video feature sequences

The architecture features modality-specific transformer encoders followed by cross-modal fusion using transformer layers, with optional adaptive modality weighting.

## Architecture

### Key Components

1. **Modality-Specific Encoders**: Individual transformer encoders for text, audio, and video
2. **Cross-Modal Fusion**: Transformer-based fusion of multimodal representations
3. **Adaptive Fusion** (Optional): Learnable weights for modality importance
4. **Regression Head**: Final dense layer for sentiment score prediction

### Model Structure

```
Input: (Text, Audio, Video) sequences
  ↓
Modality Projection Layers
  ↓
Modality-Specific Transformer Encoders
  ↓
Cross-Modal Transformer Fusion
  ↓
Global Average Pooling
  ↓
Dense Output Layer → Sentiment Score
```

## Requirements

- Python 3.8+
- TensorFlow 2.13+ (macOS optimized)
- See `requirements.txt` for complete dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/msa-tf2.git
cd msa-tf2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

The model expects data in HDF5 format with the following structure:

- `text_{split}_emb.h5`: Text embeddings (shape: [N, T, 300])
- `audio_{split}.h5`: Audio features (shape: [N, T, 74])
- `video_{split}.h5`: Video features (shape: [N, T, 47])
- `y_{split}.h5`: Labels (shape: [N, 1])

Where `{split}` is one of: `train`, `valid`, `test`

## Usage

### Training

Run the training script:

```bash
python train.py
```

### Model Configuration

Key hyperparameters in `train.py`:

- `T = 20`: Sequence length
- `model_dim = 64`: Transformer hidden dimension
- `heads = 4`: Number of attention heads
- `ff_dim = 256`: Feed-forward network dimension
- `n_layers = 2`: Number of transformer layers per modality
- `n_layers_fuse = 1`: Number of fusion layers
- `adaptive_fusion = False`: Enable/disable adaptive modality weighting

### Custom Training

```python
from models.msamodel import MSAModel
from utils.data_loader import make_dataset

# Initialize model
model = MSAModel(
    seq_len=20,
    text_dim=300,
    audio_dim=74,
    video_dim=47,
    model_dim=64,
    num_heads=4,
    ff_dim=256,
    n_layers_mod=2,
    n_layers_fuse=1,
    adaptive_fusion=False
)

# Load data
train_data = make_dataset("./data", split="train", batch_size=64)
val_data = make_dataset("./data", split="valid", batch_size=64)

# Train
model.compile(optimizer='adam', loss='mae', metrics=['mae'])
model.fit(train_data, validation_data=val_data, epochs=100)
```

## Project Structure

```
msa-tf2/
├── models/
│   ├── msamodel.py      # Main MSA model implementation
│   └── transformer.py   # Transformer encoder layer
├── utils/
│   └── data_loader.py   # Data loading utilities
├── data/               # Dataset files (not tracked in git)
├── train.py           # Training script
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## Model Details

### Transformer Encoder

Each transformer encoder layer includes:
- Multi-head self-attention
- Feed-forward network
- Layer normalization
- Residual connections
- Dropout regularization

### Multimodal Fusion

The model uses a cross-modal transformer approach:
1. Each modality is encoded independently
2. Modality representations are pooled over time
3. Pooled representations are stacked and processed by fusion transformers
4. Optional adaptive weighting learns modality importance

## Performance Notes

- The model is configured for CPU execution (GPU disabled in `train.py`)
- Optimized for macOS with TensorFlow Metal support
- Handles missing values in data (NaN → 0)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of an Honours thesis research project.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{msa-tf2,
  title={Multimodal Sentiment Analysis with Transformers},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/msa-tf2}
}
```

## Acknowledgments

- Built with TensorFlow 2
- Inspired by transformer-based multimodal fusion approaches
- Part of Honours thesis research
