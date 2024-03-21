from torch.utils.data import DataLoader, Subset
import os
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from datasets.datasets import CustomDatasetWrapper

class CustomImageDataModule(pl.LightningDataModule):
    def __init__(self, dataset: CustomDatasetWrapper, batch_size: int = 8, num_workers: int = os.cpu_count()):
        super().__init__()
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    
    def setup(self, stage=None):
        valid_indices = self.dataset.valid_indices
        labels = [self.dataset.dataset.samples[i][1] for i in valid_indices]  # Extracting labels based on valid_indices
        adjusted_labels = [self.dataset.label_map[label] for label in labels]  # Adjusting labels based on classification mode
        
        train_idx, val_idx = train_test_split(
            range(len(valid_indices)),
            test_size=0.2, 
            stratify=adjusted_labels, 
            random_state=42
        )
        
        # Mapping back to original indices in the dataset
        train_indices = [valid_indices[i] for i in train_idx]
        val_indices = [valid_indices[i] for i in val_idx]
        
        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
