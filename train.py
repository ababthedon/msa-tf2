from models.transformer import TransformerEncoder
from utils.data_loader import make_dataset
import tensorflow as tf

# Disable GPU devices in TensorFlow to force CPU-only execution
tf.config.set_visible_devices([], 'GPU')
print("GPUs available after disable:", tf.config.list_physical_devices('GPU'))


# Define model and data parameters (customise as needed)
T = 20                  # sequence length
text_dim = 300         # input feature size for text embeddings
model_dim = 64         # transformer hidden size
heads = 4              # number of attention heads
ff_dim = 256           # feed-forward network inner dimension
n_layers = 2           # number of transformer layers


# Example: text-only
text_inputs = tf.keras.Input(shape=(T, text_dim), name="text_inputs")
x = tf.keras.layers.Dense(model_dim)(text_inputs)
for _ in range(n_layers): 
    x = TransformerEncoder(model_dim, heads, ff_dim)(x, training=True)
h_text = tf.keras.layers.GlobalAveragePooling1D()(x)
logits = tf.keras.layers.Dense(1)(h_text)
model = tf.keras.Model(text_inputs, logits)
model.summary()

# 6) Compile & train
model.compile(optimizer='adam', loss='mae')

# For a quick smoke test you could replace make_dataset with just t and y arrays:
# t_train, y_train = load_text_and_labels_somehow()
# model.fit(t_train, y_train, batch_size=64, epochs=3, validation_split=0.1)
ds = make_dataset('./data', split='train', batch_size=64)       # yields ((t,a,v), y)

# try text part only
ds_text = ds.map(lambda tup, y: (tup[0], y))                 
val_ds_text = make_dataset('./data','valid',64) \
                 .map(lambda tup, y: (tup[0], y))
print("GPUs available:", tf.config.list_physical_devices('GPU'))
model.fit(ds_text, epochs=100, validation_data=val_ds_text, verbose=1)
