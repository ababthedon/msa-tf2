import tensorflow as tf

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha      = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=model_dim)
        self.norm1    = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn      = tf.keras.Sequential([
                          tf.keras.layers.Dense(ff_dim, activation='relu'),
                          tf.keras.layers.Dense(model_dim),
                      ])
        self.norm2    = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, mask=None):
        attn_output = self.mha(x, x, attention_mask=mask)
        x1 = self.norm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(x1)
        return self.norm2(x1 + self.dropout2(ffn_output, training=training))
