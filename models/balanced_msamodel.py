import tensorflow as tf
from models.transformer import TransformerEncoder


class BalancedMSAModel(tf.keras.Model):
    """
    Balanced Multimodal Sentiment Analysis Model
    
    Key Changes from Original:
    - Moderate dropout increase (0.1 → 0.2)
    - Light L2 regularization 
    - Keep original architecture mostly intact
    - Add only essential improvements
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
        dropout_rate=0.2,    # Moderate increase from 0.1
        l2_reg=0.001,        # Light regularization
        adaptive_fusion=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.adaptive_fusion = adaptive_fusion

        # Projection layers with light regularization
        self.text_proj  = tf.keras.layers.Dense(
            model_dim, 
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name='text_proj'
        )
        self.audio_proj = tf.keras.layers.Dense(
            model_dim, 
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name='audio_proj'
        )
        self.video_proj = tf.keras.layers.Dense(
            model_dim, 
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            name='video_proj'
        )

        # Modality-specific transformer encoders with moderate dropout
        self.text_encoders  = [
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

        # Adaptive fusion weights (keep original design)
        if self.adaptive_fusion:
            self.fusion_weights = tf.keras.layers.Dense(3, activation='softmax', name='fusion_weights')

        # Keep original pooling (it was working!)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling1D(name='avg_pool')

        # Simplified final layer with light dropout
        self.dropout_final = tf.keras.layers.Dropout(dropout_rate)
        self.output_layer = tf.keras.layers.Dense(1, name='output_dense')

    def call(self, inputs, training=False):
        # Unpack inputs: (text_seq, audio_seq, video_seq)
        text_seq, audio_seq, video_seq = inputs

        # Encode each modality (keep original method)
        h_t = self._encode_modality(text_seq, self.text_proj, self.text_encoders, training)
        h_a = self._encode_modality(audio_seq, self.audio_proj, self.audio_encoders, training)
        h_v = self._encode_modality(video_seq, self.video_proj, self.video_encoders, training)

        # Cross-modal fusion (keep original approach)
        fusion_seq = tf.stack([h_t, h_v, h_a], axis=1)
        for enc in self.fuse_encoders:
            fusion_seq = enc(fusion_seq, training=training)

        # Adaptive fusion (keep original)
        if self.adaptive_fusion:
            fused_pooled = self.avg_pool(fusion_seq)
            w = self.fusion_weights(fused_pooled)
            weighted = fusion_seq * tf.expand_dims(w, -1)
            fused = tf.reduce_sum(weighted, axis=1)
        else:
            fused = self.avg_pool(fusion_seq)

        # Add light dropout before final layer
        fused = self.dropout_final(fused, training=training)
        
        # Final output
        return self.output_layer(fused)

    def _encode_modality(self, seq, proj_layer, encoders, training):
        # Project input features
        x = proj_layer(seq)
        # Transformer stack
        for enc in encoders:
            x = enc(x, training=training)
        # Pool over sequence dimension
        return self.avg_pool(x)








