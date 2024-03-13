from torchvision.datasets import ImageFolder
import os
from typing import Tuple, List, Dict
from torch.utils.data import Dataset
from torchvision.datasets.folder import has_file_allowed_extension


class CustomDatasetWrapper(Dataset):
    def __init__(self, root_dir, classification_mode, transform=None):
        assert classification_mode in [
            "binary",
            "all",
            "only_bad",
        ], "Unsupported classification mode"
        self.dataset = ImageFolder(root=root_dir, transform=transform)
        self.classification_mode = classification_mode
        self.label_map, self.valid_indices = self._create_label_map()

    @property
    def classes(self):
        return self.dataset.classes

    def _create_label_map(self):
        label_map = {}
        valid_indices = []

        if self.classification_mode == "binary":
            # Assuming 'goodIngots' is index 0, and others are 'bad'
            label_map = {0: 0, 1: 1, 2: 1, 3: 1}
            valid_indices = list(range(len(self.dataset)))

        elif self.classification_mode == "all":
            # Direct mapping for all classes
            label_map = {i: i for i in range(len(self.dataset.classes))}
            valid_indices = list(range(len(self.dataset)))

        elif self.classification_mode == "only_bad":
            # Exclude 'goodIngots' samples and adjust indices for 'bad' types
            for idx, class_name in enumerate(self.dataset.classes):
                if (
                    "good" not in class_name
                ):  # Flexible condition to exclude 'goodIngots'
                    new_index = len(label_map)  # Assign a new consecutive index
                    label_map[idx] = new_index
                    valid_indices.extend(
                        [
                            i
                            for i, (_, label) in enumerate(self.dataset.samples)
                            if label == idx
                        ]
                    )

        return label_map, valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Use valid_indices to filter and access the actual sample
        actual_idx = self.valid_indices[idx]
        image, label = self.dataset[actual_idx]
        # Adjust the label based on the classification mode
        adjusted_label = self.label_map.get(
            label, -1
        )  # Default to -1 for any unexpected label
        return image, adjusted_label
