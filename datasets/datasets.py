from torchvision.datasets import ImageFolder
import os
from typing import Tuple, List, Dict

class CustomImageFolder(ImageFolder):
    def __init__(self, root: str, classification_mode: str, **kwargs):
        """
        Custom image folder dataset loader.

        Args:
            root (str): Root directory path.
            classification_mode (str): Classification mode, 'binary', 'multi_class', or 'multi_label'.
            **kwargs: Other keyword arguments for the ImageFolder constructor.
        """
        self.classification_mode = classification_mode
        super().__init__(root, **kwargs)


    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            directory (str): Root directory path.
            classification_mode (str): Either 'binary' for good vs. bad classification, 
                                    or 'multi' for detailed classification including types of bad ingots.

        Returns:
            Tuple[List[str], Dict[str, int]]: The list of class names and a dictionary mapping class names to class indices.
        """
        if self.classification_mode not in ['binary', 'multi']:
            raise ValueError("Invalid classification_mode. Expected 'binary' or 'multi'.")

        classes = []
        class_to_idx = {}

        # For binary classification, aggregate all 'bad' types under a single 'bad' label
        if self.classification_mode == 'binary':
            for entry in os.scandir(directory):
                if entry.is_dir():
                    if entry.name == 'bad':
                        if 'bad' not in classes:
                            classes.append('bad')
                    else:
                        classes.append(entry.name)
            classes.sort()
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        # For multi-class classification, treat each subdirectory in 'bad' as a separate class
        elif self.classification_mode == 'multi':
            for entry in os.scandir(directory):
                if entry.is_dir():
                    # Check if the directory is 'bad' and has subdirectories
                    if entry.name == 'bad':
                        for subentry in os.scandir(entry.path):
                            if subentry.is_dir():
                                classes.append(subentry.name)
                    else:
                        classes.append(entry.name)
            classes.sort()
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        
        return classes, class_to_idx

