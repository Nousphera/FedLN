"""Numerical regressions for the maintained research paths."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest


def test_knn_preserves_actual_class_ids_when_a_shard_lacks_classes():
    from utils.knn_relabel import relabel_with_pretrained_knn
    labels = np.eye(4)[[1, 1, 3, 3]]
    features = np.array([[0.], [.1], [10.], [10.1]])
    actual = relabel_with_pretrained_knn(labels, features, 4, num_neighbors=1, noise_theshold=0)
    np.testing.assert_array_equal(actual.argmax(-1), [1, 1, 3, 3])


def test_uint8_cifar_preprocessing_scales_exactly_once():
    import tensorflow as tf
    from utils.image_utils import valid_prep_cifar, MEAN_IMAGENET, STD_IMAGENET
    actual, _ = valid_prep_cifar(tf.fill([32, 32, 3], tf.constant(255, tf.uint8)), 0)
    expected = np.broadcast_to((1 - MEAN_IMAGENET.numpy()) / STD_IMAGENET.numpy(), actual.shape)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6)


def test_client_forwards_seed_independently_of_batch_size():
    import tensorflow as tf
    from utils.flwr_client import _Client
    seen = {}
    def loader(**kwargs):
        seen.update(kwargs)
        data = tf.data.Dataset.from_tensor_slices((np.zeros((2, 4)), np.eye(2))).batch(2)
        return data, 2, 2, np.eye(2), np.arange(2)
    _Client("0", 1, lambda **kw: None, loader, seed=73, batch_size=2,
            noisy_clients_frac=0, noise_lvl=0, noise_sparsity=0)
    assert seen["seed"] == 73


def test_energy_weights_support_noncontiguous_client_ids():
    from utils.energy_noise_estimation import compute_noise_estimation
    # The >= threshold rule is used; IDs are identities, not row offsets.
    scores = np.array([[2, 0, 1], [2, 2, 3], [7, 1, 0], [7, 3, 0]], dtype=float)
    assert compute_noise_estimation(scores, percentile=50) == {2: .2, 7: 2}


def test_distiller_can_train_without_regularizers_and_evaluate():
    import tensorflow as tf
    from utils.distiller import Distiller
    x = tf.keras.Input((4,))
    model = tf.keras.Model(x, [tf.keras.layers.Dense(2)(x), tf.keras.layers.Dense(3)(x)])
    wrapper = Distiller(model, np.ones((2, 3)), np.array([4, 9]))
    wrapper.compile(tf.keras.optimizers.Adam(.01), tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    [tf.keras.metrics.CategoricalAccuracy(name="accuracy")])
    out = wrapper.train_step(((tf.ones((2, 4)), tf.eye(2)), tf.constant([4, 9])))
    assert np.isfinite(float(out["loss"]))
    assert np.isfinite(float(wrapper.test_step((tf.ones((2, 4)), tf.eye(2)))["loss"]))


def test_pruning_excludes_each_selected_index_and_handles_empty_set():
    from utils.prune import prune_fn
    assert not bool(prune_fn(2, None, [2, 7]))
    assert not bool(prune_fn(7, None, [2, 7]))
    assert bool(prune_fn(3, None, [2, 7]))
    assert bool(prune_fn(3, None, []))


def test_cutout_preserves_original_fixed_origin_geometry():
    import tensorflow as tf
    from utils.image_utils import cutout_cifar
    actual, _ = cutout_cifar(tf.ones([2, 32, 32, 3]), 0)
    expected = np.ones([2, 32, 32, 3])
    expected[:, :8, :8, :] = 0
    np.testing.assert_array_equal(actual.numpy(), expected)
