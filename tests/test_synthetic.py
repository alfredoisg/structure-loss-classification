import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from datasets.synthetic import make_images, split_indices, predict_centroids, run_analysis


class SyntheticTests(unittest.TestCase):
    def test_images_and_seed(self):
        images, labels = make_images()
        np.testing.assert_array_equal(images, make_images()[0])
        self.assertFalse(np.array_equal(images, make_images(seed=7)[0]))
        self.assertEqual(images.shape, (120, 24, 24, 3))
        self.assertTrue(np.all((images >= 0) & (images <= 1)))
        train, test = split_indices(labels)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(train) | set(test), set(range(120)))
        np.testing.assert_array_equal(np.bincount(labels[test]), [6] * 4)

    def test_centroids_with_nonconsecutive_labels(self):
        prediction = predict_centroids(np.array([[0.], [2.], [8.], [10.]]),
                                       np.array([3, 3, 9, 9]), np.array([[1.], [9.]]))
        np.testing.assert_array_equal(prediction, [3, 9])

    def test_outputs_account_for_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics, confusion = run_analysis(directory)
            self.assertEqual(confusion.sum(), metrics["test_samples"])
            self.assertEqual(metrics["accuracy"], confusion.trace() / confusion.sum())
            self.assertEqual(metrics["accuracy"], 1.0)  # deliberately separated classes
            self.assertEqual(json.loads((Path(directory) / "metrics.json").read_text()), metrics)
            np.testing.assert_array_equal(np.loadtxt(Path(directory) / "confusion.csv", delimiter=","), confusion)

    def test_insufficient_samples(self):
        with self.assertRaises(ValueError):
            make_images(samples_per_class=1)
        with self.assertRaises(ValueError):
            split_indices(np.array([0, 1]))
