"""Small invariants for unchanged label-noise and sharding utilities."""

import importlib.util
from pathlib import Path
import unittest

try:
    import numpy as np
except ImportError:
    np = None


def load_utility(name):
    path = Path(__file__).resolve().parents[1] / "src" / "utils" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(np is None, "NumPy is needed for data utility checks")
class DataUtilityTests(unittest.TestCase):
    def test_zero_noise_keeps_labels(self):
        labels = np.eye(3)[[0, 2, 1, 2]]
        result, observed = load_utility("noise").noise_from_level_sparsity(labels, 3, 0.0, 0.0, 42)
        np.testing.assert_array_equal(result, labels)
        self.assertEqual(observed, 0.0)

    def test_noise_is_repeatable_and_preserves_one_hot_contract(self):
        noise = load_utility("noise").noise_from_level_sparsity
        labels = np.eye(10)[np.tile(np.arange(10), 100)]
        result, observed = noise(labels, 10, 0.4, 0.7, 42)
        repeated, _ = noise(labels, 10, 0.4, 0.7, 42)
        np.testing.assert_array_equal(result, repeated)
        self.assertEqual(result.shape, (1000, 10))
        self.assertTrue(np.isin(result, [0, 1]).all())
        np.testing.assert_array_equal(result.sum(axis=1), np.ones(1000))
        self.assertAlmostEqual(observed, 0.4, delta=0.05)

    def test_shards_cover_every_example_once_with_labels_aligned(self):
        shard = load_utility("shard").shard_data
        images = np.arange(1000)
        labels = images + 1000
        parts = [shard((images, labels), id=i, num_shards=10) for i in range(10)]
        indices = np.concatenate([index.ravel() for _, index in parts])
        np.testing.assert_array_equal(np.sort(indices), np.arange(1000))
        for (x, y), index in parts:
            np.testing.assert_array_equal(x, index.ravel())
            np.testing.assert_array_equal(y, x + 1000)


if __name__ == "__main__":
    unittest.main()
