"""Artificial RGB images and a small CPU baseline, independent of private data.

These brightness-separated classes are deliberately easy and have no physical
units or claim to represent structure loss. No historical CNN method is changed.
"""
from pathlib import Path
import json

import numpy as np


def make_images(seed=42, samples_per_class=30):
    """Return balanced (N, 24, 24, 3) float images in [0, 1] and labels."""
    if samples_per_class < 2:
        raise ValueError("Need at least two samples per class")
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(4), samples_per_class)
    images = rng.normal(0, 0.08, (len(labels), 24, 24, 3))
    images += (0.15 + 0.23 * labels)[:, None, None, None]
    return np.clip(images, 0, 1), labels


def split_indices(labels, seed=42):
    """Balanced 80/20 split with at least one train and test image per class."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) < 2:
            raise ValueError("Need at least two samples per class")
        n_test = max(1, int(len(indices) * 0.2))
        test.extend(indices[:n_test])
        train.extend(indices[n_test:])
    return np.array(train, dtype=int), np.array(test, dtype=int)


def predict_centroids(train_features, train_labels, test_features):
    """Fit class means on training features only; predict Euclidean nearest mean."""
    classes = np.unique(train_labels)
    centroids = np.stack([train_features[train_labels == c].mean(axis=0) for c in classes])
    distances = ((test_features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return classes[distances.argmin(axis=1)]


def run_analysis(output_dir="results/synthetic", seed=42):
    """Write metrics and confusion counts; rows=true labels, columns=predictions."""
    images, labels = make_images(seed)
    train, test = split_indices(labels, seed)
    features = images.mean(axis=(1, 2))
    predictions = predict_centroids(features[train], labels[train], features[test])
    confusion = np.zeros((4, 4), dtype=int)
    np.add.at(confusion, (labels[test], predictions), 1)
    metrics = {
        "data": "artificial brightness classes; not publication reproduction",
        "seed": seed,
        "train_samples": len(train),
        "test_samples": len(test),
        "accuracy": float((predictions == labels[test]).mean()),
        "classes": ["good_synthetic", "typeA", "typeB", "typeC"],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savetxt(output / "confusion.csv", confusion, delimiter=",", fmt="%d")
    return metrics, confusion
