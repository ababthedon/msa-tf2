import tensorflow as tf
from models.transformer import TransformerEncoder


class ImprovedMSAModel(tf.keras.Model):
    """
    Improved Transformer-Enhanced Multimodal Sentiment Analysis Model
    
    Key Improvements:
    - Higher dropout rates for regularization
    - Batch normalization for stability
    - Attention-based pooling
    - Residual connections in fusion
    - L2 regularization
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
        dropout_rate=0.3,  # Increased from 0.1
        l2_reg=0.01,       # Added regularization
        adaptive_fusion=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.adaptive_fusion = adaptive_fusion
        self.dropout_rate = dropout_rate
        
        # Projection layers with regularization and normalization
        self.text_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(model_dim, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(dropout_rate)
        ], name='text_proj')
        
        self.audio_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(model_dim, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(dropout_rate)
        ], name='audio_proj')
        
        self.video_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(model_dim, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(dropout_rate)
        ], name='video_proj')

        # Modality-specific transformer encoders with higher dropout
        self.text_encoders = [
            TransformerEncoder(model_dim, num_heads, ff_dim, dropout=dropout_rate, name=f'text_enc_{i}')
            for i in range(n_layers_mod)
        ]
        self.audio_encoders = [
            TransformerEncoder(model_dim, num_heads, ff_dim, dropout=dropout_rate, name=f'audio_enc_{i}')
            for i in range(n_layers_mod)
        ]
        self.video_encoders = [
            TransformerEncoder(model_dim, num_heads, ff_dim, dropout=dropout_rate, name=f'video_enc_{i}')
            for i in range(n_layers_mod)
        ]

        # Cross-modal transformer layers
        self.fuse_encoders = [
            TransformerEncoder(model_dim, num_heads, ff_dim, dropout=dropout_rate, name=f'fuse_enc_{i}')
            for i in range(n_layers_fuse)
        ]

        # Attention-based pooling instead of simple average
        self.attention_pool = tf.keras.layers.MultiHeadAttention(
            num_heads=1, key_dim=model_dim, name='attention_pool'
        )
        self.pool_query = self.add_weight(
            shape=(1, model_dim), initializer='random_normal', name='pool_query'
        )

        # Adaptive fusion weights
        if self.adaptive_fusion:
            self.fusion_weights = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(3, activation='softmax')
            ], name='fusion_weights')

        # Improved final layers with regularization
        self.final_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1, name='output_dense')
        ], name='final_layers')

    def call(self, inputs, training=False):
        text_seq, audio_seq, video_seq = inputs
        batch_size = tf.shape(text_seq)[0]

        # Encode each modality with residual connections
        h_t = self._encode_modality_improved(text_seq, self.text_proj, self.text_encoders, training)
        h_a = self._encode_modality_improved(audio_seq, self.audio_proj, self.audio_encoders, training)
        h_v = self._encode_modality_improved(video_seq, self.video_proj, self.video_encoders, training)

        # Cross-modal fusion with residual connections
        fusion_seq = tf.stack([h_t, h_v, h_a], axis=1)  # (batch, 3, model_dim)
        
        # Apply fusion transformers with residual connections
        for enc in self.fuse_encoders:
            residual = fusion_seq
            fusion_seq = enc(fusion_seq, training=training)
            fusion_seq = fusion_seq + residual  # Residual connection

        # Adaptive fusion or attention pooling
        if self.adaptive_fusion:
            # Use attention-based pooling for better representation
            query = tf.tile(tf.expand_dims(self.pool_query, 0), [batch_size, 1, 1])
            pooled = self.attention_pool(query, fusion_seq, training=training)
            pooled = tf.squeeze(pooled, axis=1)
            
            # Compute adaptive weights
            w = self.fusion_weights(pooled, training=training)
            weighted = fusion_seq * tf.expand_dims(w, -1)
            fused = tf.reduce_sum(weighted, axis=1)
        else:
            # Simple attention pooling
            query = tf.tile(tf.expand_dims(self.pool_query, 0), [batch_size, 1, 1])
            fused = self.attention_pool(query, fusion_seq, training=training)
            fused = tf.squeeze(fused, axis=1)

        # Final prediction with improved layers
        return self.final_layers(fused, training=training)

    def _encode_modality_improved(self, seq, proj_layer, encoders, training):
        """Improved modality encoding with residual connections."""
        # Project input features
        x = proj_layer(seq, training=training)
        
        # Apply transformer stack with residual connections
        for enc in encoders:
            residual = x
            x = enc(x, training=training)
            x = x + residual  # Residual connection
        
        # Attention-based pooling instead of simple average
        batch_size = tf.shape(x)[0]
        query = tf.tile(tf.expand_dims(self.pool_query, 0), [batch_size, 1, 1])
        pooled = self.attention_pool(query, x, training=training)
        return tf.squeeze(pooled, axis=1)








