import h5py
import numpy as np
import tensorflow as tf
import os

def load_h5(path):
    """Load HDF5 file and handle missing values."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    with h5py.File(path, 'r') as f:
        if 'd1' not in f:
            raise KeyError(f"Dataset 'd1' not found in {path}")
        
        arr = np.ones(f['d1'].shape, dtype=np.float32)
        f['d1'].read_direct(arr)
    
    # Handle NaN values
    arr[np.isnan(arr)] = 0.0
    
    return arr

def validate_data_shapes(t, a, v, y, split):
    """Validate that all modalities have consistent sample dimensions."""
    n_samples = [t.shape[0], a.shape[0], v.shape[0], y.shape[0]]
    
    if len(set(n_samples)) != 1:
        raise ValueError(
            f"Inconsistent number of samples in {split} split: "
            f"text={t.shape[0]}, audio={a.shape[0]}, video={v.shape[0]}, labels={y.shape[0]}"
        )
    
    print(f"✓ {split} data validation passed:")
    print(f"  Samples: {n_samples[0]}")
    print(f"  Text shape: {t.shape}")
    print(f"  Audio shape: {a.shape}")
    print(f"  Video shape: {v.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Label range: [{y.min():.3f}, {y.max():.3f}]")

def make_dataset(data_dir, split='train', batch_size=32, shuffle_buffer=1000):
    """Create TensorFlow dataset from HDF5 files with validation."""
    print(f"Loading {split} data from {data_dir}...")
    
    # Load all modalities
    t = load_h5(f"{data_dir}/text_{split}_emb.h5")
    a = load_h5(f"{data_dir}/audio_{split}.h5")
    v = load_h5(f"{data_dir}/video_{split}.h5")
    y = load_h5(f"{data_dir}/y_{split}.h5").reshape(-1, 1)
    
    # Validate data consistency
    validate_data_shapes(t, a, v, y, split)
    
    # Convert to TensorFlow dataset
    ds = tf.data.Dataset.from_tensor_slices(((t, a, v), y))
    
    # Apply shuffling for training data only
    if split == 'train' and shuffle_buffer > 0:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    
    # Batch the dataset
    ds = ds.batch(batch_size)
    
    # Prefetch for performance
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds