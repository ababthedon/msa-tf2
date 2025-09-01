import h5py, numpy as np, tensorflow as tf

def load_h5(path):
    with h5py.File(path,'r') as f:
        arr = np.ones(f['d1'].shape)
        f['d1'].read_direct(arr)
    arr[np.isnan(arr)] = 0
    return arr

def make_dataset(data_dir, split='train', batch_size=32):
    # load everything once into NumPy
    t = load_h5(f"{data_dir}/text_{split}_emb.h5")
    a = load_h5(f"{data_dir}/audio_{split}.h5")
    v = load_h5(f"{data_dir}/video_{split}.h5")
    y = load_h5(f"{data_dir}/y_{split}.h5").reshape(-1,1)

    ds = tf.data.Dataset.from_tensor_slices(((t,a,v), y))
    # reduce parallelism
    return ds.shuffle(1000).batch(batch_size)