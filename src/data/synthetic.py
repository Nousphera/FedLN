"""Small deterministic images and aligned features for execution checks only."""
import numpy as np
import tensorflow as tf


def arrays(train=True, seed=42):
    rng = np.random.default_rng(seed if train else seed + 10000)
    labels = np.arange(32 if train else 8) % 2
    images = rng.normal(0, .1, (len(labels), 16, 16, 3)).astype(np.float32)
    images += labels[:, None, None, None]
    return images, labels


def features(seed=42):
    _, labels = arrays(seed=seed)
    # This synthetic teacher is a fixture, not a pretrained feature extractor.
    return np.tile(np.eye(2, dtype=np.float32)[labels], (1, 256))


def load_synthetic(shard_id=0, num_shards=None, batch_size=8, seed=42,
                   noisy_clients_frac=.8, noise_lvl=.4, noise_sparsity=.7,
                   corrected_labels=None, shuffle=True):
    images, labels = arrays(seed=seed)
    rng = np.random.default_rng(seed)
    idxs = rng.permutation(len(labels))
    if num_shards is not None:
        idxs = np.array_split(idxs, num_shards)[shard_id]
    if not len(idxs): raise ValueError('Synthetic fixture requires at least one example per client')
    images, labels = images[idxs], labels[idxs].copy()
    if corrected_labels is None:
        if shard_id < round(noisy_clients_frac * (num_shards or 1)):
            labels[:round(len(labels) * noise_lvl)] = 1 - labels[:round(len(labels) * noise_lvl)]
        labels = np.eye(2, dtype=np.float32)[labels]
    else:
        labels = np.asarray(corrected_labels, dtype=np.float32)
    data = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle: data = data.shuffle(len(labels), seed=seed)
    return data.batch(batch_size), 2, len(labels), labels, idxs


def load_synthetic_test(batch_size=8):
    images, labels = arrays(train=False)
    labels = np.eye(2, dtype=np.float32)[labels]
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch_size), 2, len(labels)
