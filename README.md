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

Use a dedicated virtual environment. The repository retains the historical Ray
2.8.1 pin; Python 3.10 is a reasonable starting point, but the complete dependency
set is not locked or validated across current platforms.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the main dependencies with:

```bash
python -m pip install -r requirements.txt
```

If you need the optional GPU sklearn path used in `hyperparameter_tuning/tune.py`, also install:

```bash
python -m pip install -r rapids-requirements.txt
```

Optional editable install (dependencies must be installed separately):

```bash
python -m pip install -e .
```

# Recommended Entry Points

The research workflows are the notebooks:

- `pipelines/pipeline_1.ipynb`
- `pipelines/pipeline_2.ipynb`
- `pipelines/pipeline_3.ipynb`
- `results.ipynb`

Launch Jupyter from the repository root after installing a notebook frontend:

```bash
python -m pip install jupyterlab
export SLC_DATA_DIR="/absolute/path/to/imagefolder"
python -m jupyter lab
```

Run cells in order after reviewing their settings. Pipeline 1 performs CNN tuning
and cross-validation; pipeline 2 extracts CNN features for classical classifiers;
pipeline 3 explores incremental training. Pipeline 3 expects hyperparameters from
pipeline 1. `results.ipynb` expects previously generated Lightning CSV logs; it does
not ship with the paper's results. Update log version numbers to match your runs.

The notebooks default to CUDA/GPU settings. Review accelerator, device, worker,
and Ray resource settings for your machine before starting expensive experiments.
Pipeline 2 uses scikit-learn by default; its optional cuML backend requires a
compatible NVIDIA CUDA environment and the RAPIDS extras. Colab users must adapt
the Drive project path in the setup cell.

# Data Format

The dataset is expected to be arranged as an `ImageFolder` root, with one subdirectory per class.

Provide your own dataset; no dataset or pretrained checkpoints are distributed
in the current checkout. Permission to use or redistribute external data is
separate from the source-code license.

Example directory layout (folder names determine sorted class indices):

```text
imagefolder/
  goodIngots/
    image_001.png
  typeA/
    image_002.png
  typeB/
    image_003.png
  typeC/
    image_004.png
```

Check `dataset.dataset.class_to_idx` before training: the good class must sort
first for binary mode. The notebooks assume four original classes and three bad
classes; adapt their class-count dictionaries for another dataset.

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
- Delete or relocate cached targets/features when changing datasets, preprocessing,
  or model weights: cache filenames do not encode those settings.
- The notebooks are historical research workflows, not a guaranteed reproduction
  environment. Dependencies are largely unpinned, and training has not been
  revalidated against current packages.
- Augmentation is shared by training and validation datasets in some workflows.
  Review transforms and split seeds before using metrics for new comparisons.
- `train_model(test=False)` enables checkpointing and early stopping;
  the default `test=True` still trains but disables these callbacks.
- Lightweight CI checks syntax, JSON, notebook hygiene, and tracked artifacts.
  It does not execute training or validate scientific results.

# Contributing and release review

See [CONTRIBUTING.md](CONTRIBUTING.md) for checks and contribution guidance, and
[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) for the outstanding Git-history review
before making this existing repository public.

# Citation and license

If you use this code in research, cite the [associated paper](https://pubs.acs.org/doi/10.1021/acs.cgd.4c00760).
Source code is available under the [MIT license](LICENSE).
