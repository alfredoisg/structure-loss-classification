from torchvision.datasets import ImageFolder
import os
from typing import Tuple, List, Dict
from torch.utils.data import Dataset
from torchvision.datasets.folder import has_file_allowed_extension


class CustomImageFolder(Dataset):
    def __init__(self, root_dir, classification_mode, transform=None):
        self.dataset = ImageFolder(root=root_dir, transform=transform)
        self.classification_mode = classification_mode
        self.label_map = self._create_label_map()

    def _create_label_map(self):
        if self.classification_mode == "binary":
            # Assuming '0' is the label for 'good' and all others are 'bad'
            return {0: 0, 1: 1, 2: 1, 3: 1}
        elif self.classification_mode == "multi":
            # Direct mapping for multi-class (adjust according to your dataset)
            return {i: i for i in range(len(self.dataset.classes))}
        else:
            raise ValueError("Unsupported classification mode")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Adjust the label based on the classification mode
        adjusted_label = self.label_map[label]
        return image, adjusted_label
