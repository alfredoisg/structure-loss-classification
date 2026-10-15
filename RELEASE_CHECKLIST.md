# Public release review

The current tree excludes datasets, model checkpoints, cached labels, and notebook
outputs. The MIT license covers the source code; it does not establish permission
to redistribute external datasets or trained artifacts.

## Git history requires review before changing visibility

Historical commits still contain these artifacts (even though they are absent
from the current tree):

- `logdir/cached_targets.pkl`
- `logdir/cached_targets_all.pkl`
- `logdir/cached_targets_struct.pkl`
- `logdir/epoch=3-val_loss=0.41.ckpt`

Making the existing repository public exposes its reachable history too. Review
these artifacts and historical notebook outputs for disclosure permissions. If
history must be removed, publish a new repository from the cleaned tree or perform
a coordinated history rewrite with backups and collaborator coordination. History
has deliberately not been rewritten by the cleanup.

## Final owner checks

- Confirm rights to release the code and any historical data or model artifacts.
- Review all reachable branches and tags with a dedicated secret scanner; current
  source checks are not a comprehensive secret scan. Rotate any exposed secrets.
- Review historical notebook outputs and machine-specific/private data references.
- Run `python scripts/check_release.py` and review the final diff.
- For reproducibility claims, record a successfully tested dependency environment,
  GPU/CUDA details, dataset version, split seeds, and experimental settings.
- Enable GitHub secret scanning and push protection where available, and review
  repository access/settings before changing visibility.

The cleanup does not publish, push commits, or change repository visibility.
