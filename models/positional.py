"""
Positional Embedding Layer for Sequence Modeling

This module provides learnable positional embeddings that can be added to
input sequences to encode positional information for transformer-based models.
"""

import tensorflow as tf


class LearnablePositionalEmbedding(tf.keras.layers.Layer):
    """
    Learnable positional embedding layer.
    
    Adds trainable position vectors to input sequences. Each position in the
    sequence has its own learnable embedding vector.
    
    Args:
        seq_len: Maximum sequence length (int)
        model_dim: Embedding dimension (int)
        **kwargs: Additional keras layer arguments
    
    Input shape:
        (batch_size, seq_len, model_dim)
    
    Output shape:
        (batch_size, seq_len, model_dim) - input with positional embeddings added
    
    Example:
        >>> pos_emb = LearnablePositionalEmbedding(seq_len=64, model_dim=256)
        >>> x = tf.random.normal((4, 64, 256))
        >>> x_with_pos = pos_emb(x)  # shape: (4, 64, 256)
    """
    
    def __init__(self, seq_len, model_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.model_dim = model_dim
        
    def build(self, input_shape):
        """Create the positional embedding weights."""
        # Positional embeddings: (seq_len, model_dim)
        self.pos_embedding = self.add_weight(
            name='positional_embedding',
            shape=(self.seq_len, self.model_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs):
        """
        Add positional embeddings to input.
        
        Args:
            inputs: Tensor of shape (batch, seq_len, model_dim)
        
        Returns:
            Tensor of shape (batch, seq_len, model_dim) with positions added
        """
        # Get sequence length from input (may be less than max)
        seq_len = tf.shape(inputs)[1]
        
        # Slice positional embeddings to match actual sequence length
        pos_emb = self.pos_embedding[:seq_len, :]
        
        # Add positional embeddings (broadcasting over batch dimension)
        return inputs + pos_emb
    
    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'model_dim': self.model_dim
        })
        return config


def create_padding_mask(sequences, pad_value=0.0, epsilon=1e-9):
    """
    Create padding mask from input sequences.
    
    Identifies padded positions in sequences (where all features are close to pad_value)
    and creates a boolean mask compatible with Keras MultiHeadAttention.
    
    Args:
        sequences: Input tensor of shape (batch, seq_len, features)
        pad_value: Value used for padding (default: 0.0)
        epsilon: Tolerance for comparing with pad_value (default: 1e-9)
    
    Returns:
        Boolean tensor of shape (batch, seq_len) where True indicates valid (non-padded)
        positions and False indicates padded positions.
        
        Note: Keras MultiHeadAttention expects attention_mask where True means
        "attend to this position" and False means "ignore this position".
    
    Example:
        >>> x = tf.constant([
        ...     [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],  # last position padded
        ...     [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]   # last two positions padded
        ... ])
        >>> mask = create_padding_mask(x)  # shape: (2, 3)
        >>> # mask = [[True, True, False], [True, False, False]]
    """
    # Check if all features in a position are close to pad_value
    # Shape: (batch, seq_len)
    is_padding = tf.reduce_all(
        tf.abs(sequences - pad_value) < epsilon,
        axis=-1
    )
    
    # Return True for valid positions, False for padding
    return tf.logical_not(is_padding)


def create_look_ahead_mask(seq_len):
    """
    Create causal (look-ahead) mask for autoregressive models.
    
    Creates a mask that prevents positions from attending to subsequent positions,
    useful for decoder-style transformers.
    
    Args:
        seq_len: Sequence length (int)
    
    Returns:
        Boolean tensor of shape (seq_len, seq_len) where True allows attention
        and False blocks it. Lower triangular matrix (including diagonal).
    
    Example:
        >>> mask = create_look_ahead_mask(4)
        >>> # mask:
        >>> # [[True, False, False, False],
        >>> #  [True, True,  False, False],
        >>> #  [True, True,  True,  False],
        >>> #  [True, True,  True,  True]]
    """
    # Create lower triangular matrix
    mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return tf.cast(mask, tf.bool)


def combine_masks(padding_mask, look_ahead_mask=None):
    """
    Combine padding mask with optional look-ahead mask.
    
    Args:
        padding_mask: Boolean tensor (batch, seq_len) or (batch, seq_len, seq_len)
        look_ahead_mask: Optional boolean tensor (seq_len, seq_len)
    
    Returns:
        Combined boolean mask suitable for attention operations
    """
    if look_ahead_mask is None:
        return padding_mask
    
    # If padding_mask is 2D, expand to 3D for combination
    if len(padding_mask.shape) == 2:
        # Expand padding mask: (batch, 1, seq_len) for broadcasting
        padding_mask = tf.expand_dims(padding_mask, axis=1)
        
        # Combine with look-ahead mask
        # Broadcasting: (batch, 1, seq_len) & (1, seq_len, seq_len) -> (batch, seq_len, seq_len)
        look_ahead_mask = tf.expand_dims(look_ahead_mask, axis=0)
        combined = tf.logical_and(padding_mask, look_ahead_mask)
        return combined
    
    return tf.logical_and(padding_mask, look_ahead_mask)






