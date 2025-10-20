"""
Transformer Building Blocks for Multimodal Attention

This module provides reusable transformer blocks including:
- TransformerBlock: Standard self-attention encoder block
- CrossAttentionBlock: Cross-modal attention between two modalities

These blocks are mask-aware and can be composed to build various
multimodal architectures.
"""

import tensorflow as tf


class TransformerBlock(tf.keras.layers.Layer):
    """
    Standard transformer encoder block with self-attention.
    
    Architecture:
        1. Multi-head self-attention with residual connection
        2. Layer normalization
        3. Feed-forward network (2-layer MLP) with residual connection
        4. Layer normalization
    
    Args:
        model_dim: Model/embedding dimension (int)
        num_heads: Number of attention heads (int)
        ff_dim: Feed-forward network hidden dimension (int)
        dropout_rate: Dropout probability (float, default: 0.1)
        **kwargs: Additional keras layer arguments
    
    Input shape:
        - x: (batch, seq_len, model_dim)
        - mask: (batch, seq_len) boolean mask (optional)
    
    Output shape:
        (batch, seq_len, model_dim)
    
    Example:
        >>> block = TransformerBlock(model_dim=256, num_heads=4, ff_dim=512)
        >>> x = tf.random.normal((4, 64, 256))
        >>> mask = tf.ones((4, 64), dtype=tf.bool)
        >>> out = block(x, mask=mask, training=True)  # shape: (4, 64, 256)
    """
    
    def __init__(self, model_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # Calculate key_dim to avoid dimension mismatch
        # key_dim is the dimension per attention head
        self.key_dim = model_dim // num_heads
        
        # Multi-head self-attention
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.key_dim,
            dropout=dropout_rate,
            name='self_attention'
        )
        
        # Feed-forward network (2-layer MLP with GELU activation)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='gelu', name='ffn_dense1'),
            tf.keras.layers.Dropout(dropout_rate, name='ffn_dropout'),
            tf.keras.layers.Dense(model_dim, name='ffn_dense2')
        ], name='feed_forward')
        
        # Layer normalization
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='ln1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='ln2')
        
        # Dropout for residual connections
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate, name='dropout1')
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate, name='dropout2')
    
    def call(self, x, mask=None, training=False):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor (batch, seq_len, model_dim)
            mask: Padding mask (batch, seq_len) boolean, True = valid position
            training: Training mode flag
        
        Returns:
            Output tensor (batch, seq_len, model_dim)
        """
        # Expand mask for self-attention if provided
        # MultiHeadAttention expects (batch, seq_len_q, seq_len_kv) for self-attention
        attn_mask = None
        if mask is not None:
            # Expand to (batch, 1, seq_len) then broadcast to (batch, seq_len, seq_len)
            attn_mask = mask[:, tf.newaxis, :]  # (batch, 1, seq_len)
        
        # Self-attention with residual connection
        attn_output = self.mha(
            query=x,
            key=x,
            value=x,
            attention_mask=attn_mask,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layernorm1(x + attn_output)  # Residual + LayerNorm
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(x, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.layernorm2(x + ffn_output)  # Residual + LayerNorm
        
        return x
    
    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'model_dim': self.model_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })
        return config


class CrossAttentionBlock(tf.keras.layers.Layer):
    """
    Cross-modal attention block.
    
    Allows one modality (query) to attend to another modality (key/value).
    Useful for fusing information between different modalities.
    
    Architecture:
        1. Multi-head cross-attention (Q from source, K/V from context)
        2. Residual connection (adds to query input)
        3. Layer normalization
        4. Feed-forward network with residual connection
        5. Layer normalization
    
    Args:
        model_dim: Model/embedding dimension (int)
        num_heads: Number of attention heads (int)
        ff_dim: Feed-forward network hidden dimension (int)
        dropout_rate: Dropout probability (float, default: 0.1)
        **kwargs: Additional keras layer arguments
    
    Input shapes:
        - query: (batch, query_seq_len, model_dim) - source modality
        - context: (batch, context_seq_len, model_dim) - target modality
        - query_mask: (batch, query_seq_len) boolean (optional)
        - context_mask: (batch, context_seq_len) boolean (optional)
    
    Output shape:
        (batch, query_seq_len, model_dim) - enhanced query with cross-modal info
    
    Example:
        >>> cross_attn = CrossAttentionBlock(model_dim=256, num_heads=4, ff_dim=512)
        >>> text = tf.random.normal((4, 64, 256))
        >>> audio = tf.random.normal((4, 64, 256))
        >>> # Text attends to audio
        >>> enhanced_text = cross_attn(query=text, context=audio, training=True)
    """
    
    def __init__(self, model_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # Calculate key_dim per head
        self.key_dim = model_dim // num_heads
        
        # Multi-head cross-attention
        self.cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.key_dim,
            dropout=dropout_rate,
            name='cross_attention'
        )
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='gelu', name='ffn_dense1'),
            tf.keras.layers.Dropout(dropout_rate, name='ffn_dropout'),
            tf.keras.layers.Dense(model_dim, name='ffn_dense2')
        ], name='feed_forward')
        
        # Layer normalization
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='ln1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='ln2')
        
        # Dropout
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate, name='dropout1')
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate, name='dropout2')
    
    def call(self, query, context, query_mask=None, context_mask=None, training=False):
        """
        Forward pass with cross-modal attention.
        
        Args:
            query: Query modality tensor (batch, query_len, model_dim)
            context: Context modality tensor (batch, context_len, model_dim)
            query_mask: Padding mask for query (batch, query_len) boolean
            context_mask: Padding mask for context (batch, context_len) boolean
            training: Training mode flag
        
        Returns:
            Enhanced query tensor (batch, query_len, model_dim)
        """
        # Expand context mask for cross-attention if provided
        # MultiHeadAttention expects (batch, query_len, context_len)
        attn_mask = None
        if context_mask is not None:
            # Expand to (batch, 1, context_len) for broadcasting
            attn_mask = context_mask[:, tf.newaxis, :]  # (batch, 1, context_len)
        
        # Cross-attention: query attends to context
        attn_output = self.cross_attn(
            query=query,
            key=context,
            value=context,
            attention_mask=attn_mask,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        
        # Residual connection with query + LayerNorm
        query = self.layernorm1(query + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(query, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        query = self.layernorm2(query + ffn_output)
        
        return query
    
    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'model_dim': self.model_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })
        return config

