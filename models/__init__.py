"""
MSA-TF2 Models Package

This package contains multimodal sentiment analysis models and components:

Existing Models:
- MSAModel: Original transformer-based MSA model with early pooling
- ImprovedMSAModel: Enhanced version with improvements
- BalancedMSAModel: Balanced training variant

New Sequence-Level Architecture:
- MSASeqLevelModel: Full sequence-level model with cross-modal attention
- ModalityEncoder: Modality-specific transformer encoder
- CrossModalFusion: Sequence-level cross-attention fusion
- AdaptiveFusionHead: Learnable modality weighting

Building Blocks:
- TransformerBlock: Standard self-attention encoder block
- CrossAttentionBlock: Cross-modal attention block
- LearnablePositionalEmbedding: Positional embeddings for sequences

Utilities:
- create_padding_mask: Generate padding masks from sequences
- create_look_ahead_mask: Generate causal masks for autoregressive models
"""

# Import existing models
from models.msamodel import MSAModel

# Check if other models exist before importing
try:
    from models.improved_msamodel import ImprovedMSAModel
except ImportError:
    ImprovedMSAModel = None

try:
    from models.balanced_msamodel import BalancedMSAModel
except ImportError:
    BalancedMSAModel = None

# Import new sequence-level architecture components
from models.positional import (
    LearnablePositionalEmbedding,
    create_padding_mask,
    create_look_ahead_mask,
    combine_masks
)

from models.blocks import (
    TransformerBlock,
    CrossAttentionBlock
)

from models.fusion import (
    CrossModalFusion,
    AdaptiveFusionHead
)

from models.msa_seqlevel import (
    MSASeqLevelModel,
    ModalityEncoder
)

# Define public API
__all__ = [
    # Existing models
    'MSAModel',
    'ImprovedMSAModel',
    'BalancedMSAModel',
    
    # New sequence-level model
    'MSASeqLevelModel',
    'ModalityEncoder',
    
    # Fusion components
    'CrossModalFusion',
    'AdaptiveFusionHead',
    
    # Building blocks
    'TransformerBlock',
    'CrossAttentionBlock',
    
    # Positional embeddings
    'LearnablePositionalEmbedding',
    
    # Utilities
    'create_padding_mask',
    'create_look_ahead_mask',
    'combine_masks',
]





