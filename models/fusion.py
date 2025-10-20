"""
Cross-Modal Fusion Components

This module provides reusable fusion layers for multimodal learning:
- CrossModalFusion: Sequence-level cross-attention between modalities
- AdaptiveFusionHead: Learnable modality weighting with mask-aware pooling

These components are designed to be modular and reusable across different
MSA backbone architectures.
"""

import tensorflow as tf
from models.blocks import CrossAttentionBlock


class CrossModalFusion(tf.keras.layers.Layer):
    """
    Sequence-level cross-modal fusion using cross-attention.
    
    Performs bidirectional or unidirectional cross-attention between modalities
    at the sequence level (no pooling). Text can attend to audio/video and
    optionally vice versa.
    
    Architecture:
        For each fusion layer:
        1. Text attends to Audio (text ← audio)
        2. Text attends to Video (text ← video)
        3. (Optional) Audio attends to Text (audio ← text)
        4. (Optional) Video attends to Text (video ← text)
    
    Args:
        model_dim: Model/embedding dimension (int)
        num_heads: Number of attention heads (int)
        ff_dim: Feed-forward network dimension (int)
        n_layers: Number of cross-attention layers to stack (int, default: 1)
        bidirectional: Whether to do symmetric cross-attention (bool, default: False)
                      If True: text↔audio, text↔video, audio↔text, video↔text
                      If False: only text←audio, text←video
        dropout_rate: Dropout probability (float, default: 0.1)
        **kwargs: Additional keras layer arguments
    
    Input shapes:
        - text_seq: (batch, seq_len, model_dim)
        - audio_seq: (batch, seq_len, model_dim)
        - video_seq: (batch, seq_len, model_dim)
        - text_mask: (batch, seq_len) boolean (optional)
        - audio_mask: (batch, seq_len) boolean (optional)
        - video_mask: (batch, seq_len) boolean (optional)
    
    Output shapes:
        Tuple of (text_seq, audio_seq, video_seq) with cross-modal information
        Each: (batch, seq_len, model_dim)
    
    Example:
        >>> fusion = CrossModalFusion(
        ...     model_dim=256, num_heads=4, ff_dim=512,
        ...     n_layers=2, bidirectional=True
        ... )
        >>> t = tf.random.normal((4, 64, 256))
        >>> a = tf.random.normal((4, 64, 256))
        >>> v = tf.random.normal((4, 64, 256))
        >>> t_fused, a_fused, v_fused = fusion(
        ...     text_seq=t, audio_seq=a, video_seq=v, training=True
        ... )
    """
    
    def __init__(
        self,
        model_dim,
        num_heads,
        ff_dim,
        n_layers=1,
        bidirectional=False,
        dropout_rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        
        # Create cross-attention blocks for each layer
        # Text as anchor (text attends to audio and video)
        self.text_to_audio_blocks = [
            CrossAttentionBlock(
                model_dim, num_heads, ff_dim, dropout_rate,
                name=f'text_to_audio_{i}'
            )
            for i in range(n_layers)
        ]
        
        self.text_to_video_blocks = [
            CrossAttentionBlock(
                model_dim, num_heads, ff_dim, dropout_rate,
                name=f'text_to_video_{i}'
            )
            for i in range(n_layers)
        ]
        
        # Optional bidirectional cross-attention
        if bidirectional:
            self.audio_to_text_blocks = [
                CrossAttentionBlock(
                    model_dim, num_heads, ff_dim, dropout_rate,
                    name=f'audio_to_text_{i}'
                )
                for i in range(n_layers)
            ]
            
            self.video_to_text_blocks = [
                CrossAttentionBlock(
                    model_dim, num_heads, ff_dim, dropout_rate,
                    name=f'video_to_text_{i}'
                )
                for i in range(n_layers)
            ]
        else:
            self.audio_to_text_blocks = None
            self.video_to_text_blocks = None
    
    def call(
        self,
        text_seq,
        audio_seq,
        video_seq,
        text_mask=None,
        audio_mask=None,
        video_mask=None,
        training=False
    ):
        """
        Apply cross-modal fusion at sequence level.
        
        Args:
            text_seq: Text sequence (batch, seq_len, model_dim)
            audio_seq: Audio sequence (batch, seq_len, model_dim)
            video_seq: Video sequence (batch, seq_len, model_dim)
            text_mask: Text padding mask (batch, seq_len) boolean
            audio_mask: Audio padding mask (batch, seq_len) boolean
            video_mask: Video padding mask (batch, seq_len) boolean
            training: Training mode flag
        
        Returns:
            Tuple of (text_fused, audio_fused, video_fused)
            Each: (batch, seq_len, model_dim)
        """
        # Apply cross-attention layers iteratively
        for i in range(self.n_layers):
            # Text attends to audio and video
            text_seq = self.text_to_audio_blocks[i](
                query=text_seq,
                context=audio_seq,
                query_mask=text_mask,
                context_mask=audio_mask,
                training=training
            )
            
            text_seq = self.text_to_video_blocks[i](
                query=text_seq,
                context=video_seq,
                query_mask=text_mask,
                context_mask=video_mask,
                training=training
            )
            
            # Optional bidirectional: audio and video attend to text
            if self.bidirectional:
                audio_seq = self.audio_to_text_blocks[i](
                    query=audio_seq,
                    context=text_seq,
                    query_mask=audio_mask,
                    context_mask=text_mask,
                    training=training
                )
                
                video_seq = self.video_to_text_blocks[i](
                    query=video_seq,
                    context=text_seq,
                    query_mask=video_mask,
                    context_mask=text_mask,
                    training=training
                )
        
        return text_seq, audio_seq, video_seq
    
    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'model_dim': self.model_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'n_layers': self.n_layers,
            'bidirectional': self.bidirectional,
            'dropout_rate': self.dropout_rate
        })
        return config


class AdaptiveFusionHead(tf.keras.layers.Layer):
    """
    Adaptive fusion head with learnable modality weighting.
    
    Takes multiple modality sequences, applies mask-aware pooling to get
    per-modality representations, then learns optimal weighting across modalities.
    
    Architecture:
        1. Mask-aware pooling for each modality (mean pooling over valid positions)
        2. Concatenate modality representations
        3. Dense layer to compute softmax weights over modalities
        4. Weighted sum of modality representations
    
    Args:
        model_dim: Model/embedding dimension (int)
        num_modalities: Number of modalities to fuse (int, default: 3)
        pooling_method: Pooling strategy ('mean' or 'attention')
                       'mean': simple average over valid positions
                       'attention': learnable attention-based pooling
        dropout_rate: Dropout probability (float, default: 0.1)
        **kwargs: Additional keras layer arguments
    
    Input shapes:
        - modality_sequences: List of tensors [(batch, seq_len, model_dim), ...]
        - modality_masks: List of boolean tensors [(batch, seq_len), ...]
    
    Output shape:
        (batch, model_dim) - fused representation
    
    Example:
        >>> fusion_head = AdaptiveFusionHead(model_dim=256, num_modalities=3)
        >>> t = tf.random.normal((4, 64, 256))
        >>> a = tf.random.normal((4, 64, 256))
        >>> v = tf.random.normal((4, 64, 256))
        >>> t_mask = tf.ones((4, 64), dtype=tf.bool)
        >>> a_mask = tf.ones((4, 64), dtype=tf.bool)
        >>> v_mask = tf.ones((4, 64), dtype=tf.bool)
        >>> fused = fusion_head(
        ...     modality_sequences=[t, a, v],
        ...     modality_masks=[t_mask, a_mask, v_mask],
        ...     training=True
        ... )  # shape: (4, 256)
    """
    
    def __init__(
        self,
        model_dim,
        num_modalities=3,
        pooling_method='mean',
        dropout_rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.num_modalities = num_modalities
        self.pooling_method = pooling_method
        self.dropout_rate = dropout_rate
        
        # Fusion weight computation
        self.fusion_weight_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(
                model_dim // 2,
                activation='relu',
                name='fusion_weight_dense1'
            ),
            tf.keras.layers.Dropout(dropout_rate, name='fusion_weight_dropout'),
            tf.keras.layers.Dense(
                num_modalities,
                activation='softmax',
                name='fusion_weight_dense2'
            )
        ], name='fusion_weights')
        
        # Optional attention-based pooling
        if pooling_method == 'attention':
            self.attention_pooling_layers = [
                tf.keras.Sequential([
                    tf.keras.layers.Dense(model_dim // 4, activation='tanh'),
                    tf.keras.layers.Dense(1)
                ], name=f'attention_pool_{i}')
                for i in range(num_modalities)
            ]
        else:
            self.attention_pooling_layers = None
    
    def _mask_aware_pooling(self, sequence, mask):
        """
        Pool sequence considering padding mask.
        
        Args:
            sequence: (batch, seq_len, model_dim)
            mask: (batch, seq_len) boolean, True = valid position
        
        Returns:
            (batch, model_dim) - pooled representation
        """
        if mask is None:
            # No mask, use simple mean pooling
            return tf.reduce_mean(sequence, axis=1)
        
        # Expand mask for broadcasting: (batch, seq_len, 1)
        mask_expanded = tf.expand_dims(tf.cast(mask, sequence.dtype), axis=-1)
        
        # Mask out padded positions
        masked_sequence = sequence * mask_expanded
        
        # Sum over sequence and divide by number of valid positions
        sum_pooled = tf.reduce_sum(masked_sequence, axis=1)
        count_valid = tf.reduce_sum(mask_expanded, axis=1)
        count_valid = tf.maximum(count_valid, 1e-9)  # Avoid division by zero
        
        return sum_pooled / count_valid
    
    def _attention_pooling(self, sequence, mask, attention_layer):
        """
        Attention-based pooling over sequence.
        
        Args:
            sequence: (batch, seq_len, model_dim)
            mask: (batch, seq_len) boolean
            attention_layer: Dense layers to compute attention scores
        
        Returns:
            (batch, model_dim) - attention-weighted representation
        """
        # Compute attention scores: (batch, seq_len, 1)
        attn_scores = attention_layer(sequence)
        
        if mask is not None:
            # Mask out padded positions with large negative values
            mask_expanded = tf.expand_dims(tf.cast(mask, sequence.dtype), axis=-1)
            attn_scores = attn_scores * mask_expanded + (1.0 - mask_expanded) * (-1e9)
        
        # Softmax over sequence length
        attn_weights = tf.nn.softmax(attn_scores, axis=1)
        
        # Weighted sum: (batch, model_dim)
        pooled = tf.reduce_sum(sequence * attn_weights, axis=1)
        
        return pooled
    
    def call(self, modality_sequences, modality_masks=None, training=False):
        """
        Fuse multiple modality sequences with adaptive weighting.
        
        Args:
            modality_sequences: List of tensors [(batch, seq_len, model_dim), ...]
            modality_masks: List of boolean masks [(batch, seq_len), ...] (optional)
            training: Training mode flag
        
        Returns:
            Fused tensor (batch, model_dim)
        """
        # Handle missing masks
        if modality_masks is None:
            modality_masks = [None] * len(modality_sequences)
        
        # Pool each modality sequence
        modality_vectors = []
        for i, (seq, mask) in enumerate(zip(modality_sequences, modality_masks)):
            if self.pooling_method == 'attention' and self.attention_pooling_layers:
                pooled = self._attention_pooling(
                    seq, mask, self.attention_pooling_layers[i]
                )
            else:
                pooled = self._mask_aware_pooling(seq, mask)
            
            modality_vectors.append(pooled)
        
        # Stack modality vectors: (batch, num_modalities, model_dim)
        modality_stack = tf.stack(modality_vectors, axis=1)
        
        # Compute adaptive fusion weights from concatenated modalities
        # First pool the stack to get (batch, model_dim)
        global_repr = tf.reduce_mean(modality_stack, axis=1)
        
        # Compute softmax weights: (batch, num_modalities)
        fusion_weights = self.fusion_weight_layer(global_repr, training=training)
        
        # Apply weights: (batch, num_modalities, 1) * (batch, num_modalities, model_dim)
        fusion_weights_expanded = tf.expand_dims(fusion_weights, axis=-1)
        weighted_modalities = modality_stack * fusion_weights_expanded
        
        # Sum over modalities: (batch, model_dim)
        fused = tf.reduce_sum(weighted_modalities, axis=1)
        
        return fused
    
    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'model_dim': self.model_dim,
            'num_modalities': self.num_modalities,
            'pooling_method': self.pooling_method,
            'dropout_rate': self.dropout_rate
        })
        return config





