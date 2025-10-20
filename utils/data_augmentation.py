import tensorflow as tf
import numpy as np


class MultimodalDataAugmentation:
    """
    Data augmentation techniques for multimodal sentiment analysis.
    Implements various augmentation strategies to improve model generalization.
    """
    
    def __init__(self, 
                 noise_std=0.05, 
                 mask_prob=0.1, 
                 time_shift_range=0.1,
                 mixup_alpha=0.2):
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.time_shift_range = time_shift_range
        self.mixup_alpha = mixup_alpha
    
    def add_gaussian_noise(self, x, training=True):
        """Add Gaussian noise to input features."""
        if not training:
            return x
        noise = tf.random.normal(tf.shape(x), stddev=self.noise_std)
        return x + noise
    
    def temporal_masking(self, x, training=True):
        """Randomly mask temporal segments."""
        if not training:
            return x
        
        seq_len = tf.shape(x)[1]
        mask_length = tf.cast(tf.random.uniform([], 0, 0.15) * tf.cast(seq_len, tf.float32), tf.int32)
        start_idx = tf.random.uniform([], 0, seq_len - mask_length, dtype=tf.int32)
        
        # Create mask
        mask = tf.ones_like(x)
        indices = tf.range(start_idx, start_idx + mask_length)
        updates = tf.zeros((mask_length, tf.shape(x)[-1]))
        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, 1), updates)
        
        return x * mask
    
    def feature_dropout(self, x, dropout_rate=0.1, training=True):
        """Randomly drop entire feature dimensions."""
        if not training:
            return x
        
        feat_dim = tf.shape(x)[-1]
        keep_prob = 1.0 - dropout_rate
        mask = tf.random.uniform([feat_dim]) < keep_prob
        mask = tf.cast(mask, tf.float32)
        
        return x * mask
    
    def modality_dropout(self, text, audio, video, dropout_prob=0.1, training=True):
        """Randomly drop entire modalities."""
        if not training:
            return text, audio, video
        
        # Randomly decide which modalities to keep
        keep_text = tf.random.uniform([]) > dropout_prob
        keep_audio = tf.random.uniform([]) > dropout_prob
        keep_video = tf.random.uniform([]) > dropout_prob
        
        # Ensure at least one modality is kept
        if not (keep_text or keep_audio or keep_video):
            keep_text = True
        
        text = text if keep_text else tf.zeros_like(text)
        audio = audio if keep_audio else tf.zeros_like(audio)
        video = video if keep_video else tf.zeros_like(video)
        
        return text, audio, video
    
    def time_shift(self, x, training=True):
        """Apply random time shifts to sequences."""
        if not training:
            return x
        
        seq_len = tf.shape(x)[1]
        shift_amount = tf.cast(
            tf.random.uniform([], -self.time_shift_range, self.time_shift_range) * tf.cast(seq_len, tf.float32),
            tf.int32
        )
        
        if shift_amount > 0:
            # Shift right, pad with zeros on left
            padding = tf.zeros([tf.shape(x)[0], shift_amount, tf.shape(x)[2]])
            x_shifted = tf.concat([padding, x[:, :-shift_amount, :]], axis=1)
        elif shift_amount < 0:
            # Shift left, pad with zeros on right
            padding = tf.zeros([tf.shape(x)[0], -shift_amount, tf.shape(x)[2]])
            x_shifted = tf.concat([x[:, -shift_amount:, :], padding], axis=1)
        else:
            x_shifted = x
        
        return x_shifted
    
    def mixup(self, x1, x2, y1, y2, training=True):
        """Apply mixup augmentation between two samples."""
        if not training:
            return x1, y1
        
        lam = tf.random.uniform([], 0, self.mixup_alpha)
        
        # Mix inputs
        if isinstance(x1, tuple):  # Multiple modalities
            mixed_x = tuple(lam * m1 + (1 - lam) * m2 for m1, m2 in zip(x1, x2))
        else:
            mixed_x = lam * x1 + (1 - lam) * x2
        
        # Mix labels
        mixed_y = lam * y1 + (1 - lam) * y2
        
        return mixed_x, mixed_y
    
    def augment_batch(self, batch, training=True):
        """Apply full augmentation pipeline to a batch."""
        (text, audio, video), labels = batch
        
        if training:
            # Apply individual augmentations
            text = self.add_gaussian_noise(text, training)
            audio = self.add_gaussian_noise(audio, training)
            video = self.add_gaussian_noise(video, training)
            
            text = self.temporal_masking(text, training)
            audio = self.temporal_masking(audio, training)
            video = self.temporal_masking(video, training)
            
            text = self.feature_dropout(text, training=training)
            audio = self.feature_dropout(audio, training=training)
            video = self.feature_dropout(video, training=training)
            
            text, audio, video = self.modality_dropout(text, audio, video, training=training)
        
        return (text, audio, video), labels


def create_augmented_dataset(dataset, augmentation_fn, training=True):
    """Create an augmented version of the dataset."""
    def augment_map_fn(batch):
        return augmentation_fn(batch, training=training)
    
    return dataset.map(augment_map_fn, num_parallel_calls=tf.data.AUTOTUNE)


# Usage example function
def get_augmented_datasets(data_dir, batch_size=16):
    """Get augmented datasets with the new pipeline."""
    from data_loader import make_dataset
    
    # Create augmentation object
    augmenter = MultimodalDataAugmentation(
        noise_std=0.03,
        mask_prob=0.1,
        time_shift_range=0.05,
        mixup_alpha=0.2
    )
    
    # Load base datasets
    train_data = make_dataset(data_dir, split="train", batch_size=batch_size)
    val_data = make_dataset(data_dir, split="valid", batch_size=batch_size)
    test_data = make_dataset(data_dir, split="test", batch_size=batch_size)
    
    # Apply augmentation only to training data
    train_data_aug = create_augmented_dataset(
        train_data, 
        augmenter.augment_batch, 
        training=True
    )
    
    return train_data_aug, val_data, test_data








