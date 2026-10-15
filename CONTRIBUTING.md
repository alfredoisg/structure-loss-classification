# Contributing

Bug reports and focused pull requests are welcome. Include the classification mode,
model, Python and dependency versions, hardware, and a minimal reproduction. Use
synthetic images or data you have permission to share; do not attach private
production images, logs, checkpoints, or credentials.

Run notebooks from the repository root. Keep generated artifacts in `logdir/` or
`results/`, and clear notebook outputs and execution counts before committing.
Explain changes to experimental settings, data splits, preprocessing, and metrics
so that their effect on comparisons can be reviewed.

Run the lightweight source checks before submitting:

```bash
python -m pip install -e .
python -m unittest discover -s tests
python examples/synthetic_analysis.py
python scripts/check_release.py
```

These checks validate syntax and repository hygiene. Training changes also need a
focused smoke run with an appropriate dataset and environment; describe what you
ran and any limitations in the pull request. There is no end-to-end training test
suite or validated dependency lockfile.
