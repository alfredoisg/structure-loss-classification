# structure-loss-classification

Utilities, models, and notebook pipelines for offline classification of structure loss in Czochralski silicon ingots.

This repository contains the code used to support the experiments and analysis around deep-learning and hybrid-learning approaches for ingot image classification. In practice, the main workflows live in the notebooks under `pipelines/` and in `results.ipynb`.

# Paper

[Machine Learning Methods for Structure Loss Classification in Czochralski Silicon Ingots](https://pubs.acs.org/doi/10.1021/acs.cgd.4c00760)

# Project Overview

The codebase is organized around a few reusable components:

- `datasets/`: dataset wrappers built on top of `torchvision.datasets.ImageFolder`
- `models/`: raw PyTorch model definitions for `LeNet5`, `VGG16`, and `ResNet18`
- `lightning_modules/`: PyTorch Lightning training modules
- `train/`: training helpers, cross-validation, incremental training, and feature extraction
- `hyperparameter_tuning/`: Ray Tune and scikit-learn based tuning utilities
- `visualization/`: plotting helpers for metrics, confusion matrices, and feature maps
- `pipelines/`: notebook workflows for model training and experimentation
- `results.ipynb`: aggregation and visualization of results

# Installation

Install the main dependencies with:

```bash
python -m pip install -r requirements.txt
```

If you need the optional GPU sklearn path used in `hyperparameter_tuning/tune.py`, also install:

```bash
python -m pip install -r rapids-requirements.txt
```

Optional editable install:

```bash
python -m pip install -e .
```

# Recommended Entry Points

The maintained workflows are the notebooks:

- `pipelines/pipeline_1.ipynb`
- `pipelines/pipeline_2.ipynb`
- `pipelines/pipeline_3.ipynb`
- `results.ipynb`

# Data Format

The dataset is expected to be arranged as an `ImageFolder` root, with one subdirectory per class.

The dataset wrapper supports three classification modes:

- `binary`: class index `0` is treated as the good-ingot class, all others are mapped to bad
- `all`: all classes are kept as separate labels
- `only_bad`: classes whose names contain `good` are excluded, and the remaining bad classes are relabeled consecutively

# Models

This repository includes three image-classification model families:

- `LeNet5`
- `VGG16`
- `ResNet18`

The Lightning wrappers used by the training utilities live in `lightning_modules/modular.py`.

# Outputs And Caching

By default, training artifacts and caches are written under `logdir/`.

Examples:

- cached targets: `logdir/cached_targets_<classification_mode>.pkl`
- extracted features: `logdir/features_labels_<ModelName>_<classification_mode>.pth`
- Lightning CSV logs: under fold-specific directories inside `logdir/.../cv/`

Result summaries are written under `results/` by the utility functions and notebooks.

# Notes And Caveats

- The repository is notebook-oriented rather than packaged as a polished CLI application.
- `LeNet5` and `VGG16` currently assume `244x244` RGB inputs because their flattened dimensions are hardcoded.
- Some paths inside notebooks are environment-specific examples and may need to be adapted locally.
- There is currently no formal test or CI setup in the repository.
