from models.msamodel import MSAModel
from utils.data_loader import make_dataset
import tensorflow as tf

# Disable GPU devices in TensorFlow to force CPU-only execution
tf.config.set_visible_devices([], 'GPU')
print("GPUs available after disable:", tf.config.list_physical_devices('GPU'))


# Define model and data parameters (customise as needed)
T = 20                  # sequence length
text_dim = 300         # input feature size for text embeddings
audio_dim = 74
video_dim = 47
model_dim = 64         # transformer hidden size
heads = 4              # number of attention heads
ff_dim = 256           # feed-forward network inner dimension
n_layers = 2           # number of transformer layers
n_layers_fuse = 1      # fused layer

# Build the multimodal model
model = MSAModel(
    seq_len=T,
    text_dim=text_dim,
    audio_dim=audio_dim,
    video_dim=video_dim,
    model_dim=model_dim,
    num_heads=heads,
    ff_dim=ff_dim,
    n_layers_mod=n_layers,
    n_layers_fuse=n_layers_fuse,
    adaptive_fusion=False  # or True if you want adaptive weighting
)
# Warm up the model with a dummy batch to build its weights
dummy_t = tf.zeros((1, T, text_dim))
dummy_a = tf.zeros((1, T, audio_dim))
dummy_v = tf.zeros((1, T, video_dim))
_ = model((dummy_t, dummy_a, dummy_v), training=False)
# Now the model is built; display summary
model.summary()
# 6) Compile & train
model.compile(
    optimizer='adam',
    loss='mae',
    metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
)
# For a quick smoke test you could replace make_dataset with just t and y arrays:
# t_train, y_train = load_text_and_labels_somehow()
# model.fit(t_train, y_train, batch_size=64, epochs=3, validation_split=0.1)
#ds = make_dataset('./data', split='train', batch_size=64)       # yields ((t,a,v), y)

train_data = make_dataset("./data", split="train", batch_size=64)
val_data = make_dataset("./data", split="valid", batch_size=64)

# Train model on All Modalities
model.fit(train_data, validation_data=val_data, epochs=100, verbose=1)
