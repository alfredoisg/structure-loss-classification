# Reproducibility

## Lightweight software demonstration

Use the README quick start. Base installation requires only NumPy; Python >=3.10
is declared. CI uses Python 3.10 and installs notebook tools separately. The
synthetic script and `demo.ipynb` import the same code in `datasets/synthetic.py`.
Seed 42 controls image generation and per-class shuffling. Each class contributes
24 training images and 6 held-out images. Mean RGB features feed Euclidean nearest
class centroids fitted only on training samples. Accuracy and a 4×4 confusion
matrix (true rows, predicted columns) are written under `results/synthetic/`.
No hyperparameter selection or preprocessing is fitted on test images.

Repeat with `--seed 7` or `--output-dir results/another-run`. Results are artificial
and deliberately easy. Tests check split separation and metric accounting, not
scientific validity. NumPy versions are not locked; bitwise equivalence across
versions/platforms is not promised.

## Historical research workflows

Install `python -m pip install -r requirements.txt` for research, notebook and dev
extras. The Ray 2.8.1 pin is retained; Python 3.10 is the starting point for that
older stack. This is not a validated lockfile. RAPIDS remains separately optional
in `rapids-requirements.txt` and requires compatible NVIDIA/CUDA hardware. The
lightweight example does not import Torch, Lightning, Ray or cuML.

Pipeline 1 performs CNN tuning and cross-validation; pipeline 2 extracts CNN
features and fits classical classifiers; pipeline 3 explores incremental training.
`results.ipynb` consumes existing Lightning CSV logs. Review GPU, worker, resource,
class-count and log-version settings before executing them. Their Colab Drive
paths are generic examples requiring adaptation. Existing research notebooks
remain historical workflows rather than the small ordered demo.

`train.train_with_cv` uses shuffled StratifiedKFold with random_state=42
by default. Pipeline 2 includes an image-level train/test split with seed 42.
Image-level splits do not guarantee ingot independence. Some workflows share
augmentation between training and validation. Review the full tuning/split order
before reporting unbiased generalization or comparing methods; the synthetic
holdout does not certify historical evaluation design. No causal claims follow.

Training normally writes `logdir/`; summaries go to `results/`. Target caches are
keyed by label mode and features by model name/mode, not dataset, transforms or
weights. Clear or isolate caches when any of those change. Metrics aggregation
expects `cv/fold_N/lightning_logs/version_<n>/metrics.csv`. Current
`train_model(test=False)` enables checkpointing and early stopping; `test=True`
still trains but disables those callbacks.

To reproduce research findings, obtain the authorized dataset and exact splits,
record the commit, package freeze, hardware/CUDA, all seeds, transforms, weights,
and tuning settings, then rerun and compare against reference artifacts. This
reorganization does not change historical model/training methods or establish
publication reproduction. No full historical training run was performed.
