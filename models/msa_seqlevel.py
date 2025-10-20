"""
Sequence-Level Multimodal Sentiment Analysis Model

This module implements a full sequence-level MSA architecture with:
- Modality-specific transformer encoders
- Sequence-level cross-modal attention fusion
- Adaptive fusion head with learnable modality weighting
- Regression head for sentiment prediction

The architecture is designed to be modular so fusion components can be
reused in other MSA backbones.
"""

import tensorflow as tf
from models.positional import LearnablePositionalEmbedding, create_padding_mask
from models.blocks import TransformerBlock
from models.fusion import CrossModalFusion, AdaptiveFusionHead


class ModalityEncoder(tf.keras.layers.Layer):
    """
    Modality-specific transformer encoder.
    
    Encodes a single modality sequence using:
    1. Dense projection to model dimension
    2. Learnable positional embeddings
    3. Stack of transformer encoder blocks
    
    No pooling is applied - output is full sequence.
    
    Args:
        input_dim: Input feature dimension for this modality (int)
        model_dim: Model/embedding dimension (int)
        seq_len: Maximum sequence length (int)
        num_heads: Number of attention heads (int)
        ff_dim: Feed-forward network dimension (int)
        n_layers: Number of transformer layers (int)
        dropout_rate: Dropout probability (float, default: 0.1)
        **kwargs: Additional keras layer arguments
    
    Input shape:
        - x: (batch, seq_len, input_dim)
        - mask: (batch, seq_len) boolean (optional)
    
    Output shape:
        (batch, seq_len, model_dim)
    
    Example:
        >>> encoder = ModalityEncoder(
        ...     input_dim=300, model_dim=256, seq_len=64,
        ...     num_heads=4, ff_dim=512, n_layers=2
        ... )
        >>> x = tf.random.normal((4, 64, 300))
        >>> mask = tf.ones((4, 64), dtype=tf.bool)
        >>> encoded = encoder(x, mask=mask, training=True)  # (4, 64, 256)
    """
    
    def __init__(
        self,
        input_dim,
        model_dim,
        seq_len,
        num_heads,
        ff_dim,
        n_layers,
        dropout_rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        
        # Input projection
        self.projection = tf.keras.layers.Dense(
            model_dim,
            name='projection'
        )
        
        # Positional embeddings
        self.pos_embedding = LearnablePositionalEmbedding(
            seq_len=seq_len,
            model_dim=model_dim,
            name='positional_embedding'
        )
        
        # Transformer encoder stack
        self.transformer_blocks = [
            TransformerBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                name=f'transformer_block_{i}'
            )
            for i in range(n_layers)
        ]
        
        # Dropout after positional embedding
        self.dropout = tf.keras.layers.Dropout(dropout_rate, name='dropout')
    
    def call(self, x, mask=None, training=False):
        """
        Encode modality sequence.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Padding mask (batch, seq_len) boolean
            training: Training mode flag
        
        Returns:
            Encoded tensor (batch, seq_len, model_dim)
        """
        # Project to model dimension
        x = self.projection(x)
        
        # Add positional embeddings
        x = self.pos_embedding(x)
        x = self.dropout(x, training=training)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask=mask, training=training)
        
        return x
    
    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'model_dim': self.model_dim,
            'seq_len': self.seq_len,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'n_layers': self.n_layers,
            'dropout_rate': self.dropout_rate
        })
        return config


class MSASeqLevelModel(tf.keras.Model):
    """
    Sequence-Level Multimodal Sentiment Analysis Model.
    
    Full architecture:
        1. Three modality-specific transformer encoders (text, audio, video)
        2. Cross-modal fusion at sequence level (no early pooling)
        3. Adaptive fusion head with learnable modality weighting
        4. Regression head for sentiment prediction
    
    This model maintains sequence-level information through fusion and only
    pools at the final stage, allowing for richer cross-modal interactions.
    
    Args:
        seq_len: Maximum sequence length (int)
        text_dim: Text input feature dimension (int)
        audio_dim: Audio input feature dimension (int)
        video_dim: Video input feature dimension (int)
        model_dim: Model/embedding dimension (int)
        num_heads: Number of attention heads (int)
        ff_dim: Feed-forward network dimension (int)
        n_layers_mod: Number of transformer layers per modality encoder (int)
        n_layers_fuse: Number of cross-attention fusion layers (int)
        bidirectional_fusion: Use bidirectional cross-attention (bool, default: False)
        pooling_method: Pooling method for fusion head ('mean' or 'attention')
        dropout_rate: Dropout probability (float, default: 0.1)
        **kwargs: Additional keras model arguments
    
    Call signature:
        Option 1 - Auto-generate masks:
            output = model((text_seq, audio_seq, video_seq), training=False)
        
        Option 2 - Provide explicit masks:
            output = model(
                (text_seq, audio_seq, video_seq, text_mask, audio_mask, video_mask),
                training=False
            )
    
    Input shapes:
        - text_seq: (batch, seq_len, text_dim)
        - audio_seq: (batch, seq_len, audio_dim)
        - video_seq: (batch, seq_len, video_dim)
        - Optional masks: (batch, seq_len) boolean each
    
    Output shape:
        (batch, 1) - sentiment prediction (float32)
    
    Example:
        >>> model = MSASeqLevelModel(
        ...     seq_len=64,
        ...     text_dim=300,
        ...     audio_dim=74,
        ...     video_dim=47,
        ...     model_dim=256,
        ...     num_heads=4,
        ...     ff_dim=512,
        ...     n_layers_mod=2,
        ...     n_layers_fuse=2,
        ...     bidirectional_fusion=False
        ... )
        >>> 
        >>> # Prepare inputs
        >>> text = tf.random.normal((4, 64, 300))
        >>> audio = tf.random.normal((4, 64, 74))
        >>> video = tf.random.normal((4, 64, 47))
        >>> 
        >>> # Forward pass
        >>> output = model((text, audio, video), training=False)
        >>> print(output.shape)  # (4, 1)
        >>> 
        >>> # Compile and train
        >>> model.compile(optimizer='adam', loss='mae')
        >>> # model.fit(dataset, epochs=10)
    """
    
    def __init__(
        self,
        seq_len,
        text_dim,
        audio_dim,
        video_dim,
        model_dim,
        num_heads,
        ff_dim,
        n_layers_mod=2,
        n_layers_fuse=1,
        bidirectional_fusion=False,
        pooling_method='mean',
        dropout_rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Store configuration
        self.seq_len = seq_len
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.n_layers_mod = n_layers_mod
        self.n_layers_fuse = n_layers_fuse
        self.bidirectional_fusion = bidirectional_fusion
        self.pooling_method = pooling_method
        self.dropout_rate = dropout_rate
        
        # Modality encoders
        self.text_encoder = ModalityEncoder(
            input_dim=text_dim,
            model_dim=model_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            ff_dim=ff_dim,
            n_layers=n_layers_mod,
            dropout_rate=dropout_rate,
            name='text_encoder'
        )
        
        self.audio_encoder = ModalityEncoder(
            input_dim=audio_dim,
            model_dim=model_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            ff_dim=ff_dim,
            n_layers=n_layers_mod,
            dropout_rate=dropout_rate,
            name='audio_encoder'
        )
        
        self.video_encoder = ModalityEncoder(
            input_dim=video_dim,
            model_dim=model_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            ff_dim=ff_dim,
            n_layers=n_layers_mod,
            dropout_rate=dropout_rate,
            name='video_encoder'
        )
        
        # Cross-modal fusion
        self.cross_modal_fusion = CrossModalFusion(
            model_dim=model_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            n_layers=n_layers_fuse,
            bidirectional=bidirectional_fusion,
            dropout_rate=dropout_rate,
            name='cross_modal_fusion'
        )
        
        # Adaptive fusion head
        self.adaptive_fusion = AdaptiveFusionHead(
            model_dim=model_dim,
            num_modalities=3,
            pooling_method=pooling_method,
            dropout_rate=dropout_rate,
            name='adaptive_fusion'
        )
        
        # Regression head
        # Use float32 dtype explicitly for mixed precision compatibility
        self.regression_head = tf.keras.layers.Dense(
            1,
            dtype='float32',  # Ensure output is always float32
            name='regression_head'
        )
    
    def call(self, inputs, training=False):
        """
        Forward pass through the model.
        
        Args:
            inputs: Tuple of tensors
                    - If length 3: (text_seq, audio_seq, video_seq)
                    - If length 6: (text_seq, audio_seq, video_seq,
                                   text_mask, audio_mask, video_mask)
            training: Training mode flag
        
        Returns:
            Sentiment predictions (batch, 1) in float32
        """
        # Unpack inputs
        if len(inputs) == 3:
            # Auto-generate masks from input sequences
            text_seq, audio_seq, video_seq = inputs
            text_mask = create_padding_mask(text_seq)
            audio_mask = create_padding_mask(audio_seq)
            video_mask = create_padding_mask(video_seq)
        elif len(inputs) == 6:
            # Use provided masks
            text_seq, audio_seq, video_seq, text_mask, audio_mask, video_mask = inputs
        else:
            raise ValueError(
                f"Expected 3 or 6 input tensors, got {len(inputs)}. "
                "Provide either (text, audio, video) or "
                "(text, audio, video, text_mask, audio_mask, video_mask)."
            )
        
        # Encode each modality (sequence-level, no pooling)
        text_encoded = self.text_encoder(
            text_seq, mask=text_mask, training=training
        )
        audio_encoded = self.audio_encoder(
            audio_seq, mask=audio_mask, training=training
        )
        video_encoded = self.video_encoder(
            video_seq, mask=video_mask, training=training
        )
        
        # Cross-modal fusion at sequence level
        text_fused, audio_fused, video_fused = self.cross_modal_fusion(
            text_seq=text_encoded,
            audio_seq=audio_encoded,
            video_seq=video_encoded,
            text_mask=text_mask,
            audio_mask=audio_mask,
            video_mask=video_mask,
            training=training
        )
        
        # Adaptive fusion with mask-aware pooling
        fused_repr = self.adaptive_fusion(
            modality_sequences=[text_fused, audio_fused, video_fused],
            modality_masks=[text_mask, audio_mask, video_mask],
            training=training
        )
        
        # Final regression (always float32)
        output = self.regression_head(fused_repr)
        
        return output
    
    def get_config(self):
        """Return model configuration for serialization."""
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'text_dim': self.text_dim,
            'audio_dim': self.audio_dim,
            'video_dim': self.video_dim,
            'model_dim': self.model_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'n_layers_mod': self.n_layers_mod,
            'n_layers_fuse': self.n_layers_fuse,
            'bidirectional_fusion': self.bidirectional_fusion,
            'pooling_method': self.pooling_method,
            'dropout_rate': self.dropout_rate
        })
        return config





