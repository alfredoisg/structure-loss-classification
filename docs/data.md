# Data contract

Historical workflows use `datasets.datasets.CustomDatasetWrapper`, backed by
Torchvision ImageFolder. Supply readable image files under one directory per
class and set `SLC_DATA_DIR` when running the pipelines from the repository root.
Private/local data belongs outside Git; `data/` is ignored. No research images or
trained weights are supplied. Obtain access and redistribution rights separately.

Folder names are sorted by ImageFolder. `binary` maps index 0 to good (0) and all
others to bad (1); verify `dataset.dataset.class_to_idx`. `all` preserves indices.
`only_bad` excludes names containing the case-sensitive substring `good` and
renumbers remaining classes. The `classes` property still exposes original names;
use `label_map` to interpret adjusted labels. Historical notebooks assume four
original classes and three bad classes. Folder renaming changes target semantics.

Images have pixel intensities, not calibrated physical units. Historical notebook
transforms resize to 244×244, convert to RGB tensors and apply configured
augmentation/normalization; inspect each transform cell before analysis. LeNet5
and VGG16 hardcode flattened sizes for 244×244 RGB input. Missing/corrupt images
are not imputed: remove or repair them with a recorded exclusion policy before
training. ImageFolder requires supported files in class directories.

The demo instead generates balanced 24×24 RGB float arrays clipped to [0, 1]
with four artificial brightness distributions and Gaussian noise. Labels are
0–3, with 0 designated synthetic good. It has no missing values or real ingot
provenance; these dimensions are for the centroid demo, not the CNNs.

Record dataset version, folder-to-label mapping, capture conditions, exclusions,
and ingot identifiers. Multiple images from one ingot may be dependent; preserve
identifiers outside pixel arrays so future evaluations can split by ingot.
