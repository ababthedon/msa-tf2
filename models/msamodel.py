import tensorflow as tf
from models.transformer import TransformerEncoder

class MSAModel(tf.keras.Model):
    """
    Transformer-Enhanced Multimodal Sentiment Analysis Model

    Architecture:
      - Three modality-specific transformer encoders (text, audio, video)
      - Cross-modal transformer fusion
      - Optional adaptive modality weighting
      - Final regression head
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
        adaptive_fusion=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.adaptive_fusion = adaptive_fusion

        # Projection layers for each modality
        self.text_proj  = tf.keras.layers.Dense(model_dim, name='text_proj')
        self.audio_proj = tf.keras.layers.Dense(model_dim, name='audio_proj')
        self.video_proj = tf.keras.layers.Dense(model_dim, name='video_proj')

        # Modality-specific transformer encoders
        self.text_encoders  = [
            TransformerEncoder(model_dim, num_heads, ff_dim, name=f'text_enc_{i}')
            for i in range(n_layers_mod)
        ]
        self.audio_encoders = [
            TransformerEncoder(model_dim, num_heads, ff_dim, name=f'audio_enc_{i}')
            for i in range(n_layers_mod)
        ]
        self.video_encoders = [
            TransformerEncoder(model_dim, num_heads, ff_dim, name=f'video_enc_{i}')
            for i in range(n_layers_mod)
        ]

        # Cross-modal transformer layers
        self.fuse_encoders = [
            TransformerEncoder(model_dim, num_heads, ff_dim, name=f'fuse_enc_{i}')
            for i in range(n_layers_fuse)
        ]

        # Adaptive fusion weights (if enabled)
        if self.adaptive_fusion:
            self.fusion_weights = tf.keras.layers.Dense(3, activation='softmax', name='fusion_weights')

        # Pooling layer to collapse sequence dimension
        self.avg_pool = tf.keras.layers.GlobalAveragePooling1D(name='avg_pool')

        # Final regression head
        self.output_layer = tf.keras.layers.Dense(1, name='output_dense')

        # Helpers for stacking and weighted sum as Keras layers
        self.stack_modal = tf.keras.layers.Lambda(
            lambda tensors: tf.stack(tensors, axis=1), name='stack_modal'
        )
        self.adaptive_sum = tf.keras.layers.Lambda(
            lambda inputs: tf.reduce_sum(inputs[0] * tf.expand_dims(inputs[1], -1), axis=1),
            name='adaptive_sum'
        )

    def call(self, inputs, training=False):
        # Unpack inputs: (text_seq, audio_seq, video_seq)
        text_seq, audio_seq, video_seq = inputs

        # Encode each modality
        h_t = self._encode_modality(text_seq, self.text_proj, self.text_encoders, training)
        h_a = self._encode_modality(audio_seq, self.audio_proj, self.audio_encoders, training)
        h_v = self._encode_modality(video_seq, self.video_proj, self.video_encoders, training)

        # Cross-modal fusion sequence
        # Stack along 'time' axis: shape => (batch, 3, model_dim)
        fusion_seq = self.stack_modal([h_t, h_v, h_a])
        for enc in self.fuse_encoders:
            fusion_seq = enc(fusion_seq, training=training)

        # Optional adaptive fusion
        if self.adaptive_fusion:
            fused = self.avg_pool(fusion_seq)
            # Compute modality weights
            w = self.fusion_weights(fused)  # shape (batch, 3)
            # Weighted sum
            fused = self.adaptive_sum([self.stack_modal([h_t, h_v, h_a]), w])
        else:
            # Pool the fused sequence
            fused = self.avg_pool(fusion_seq)

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
