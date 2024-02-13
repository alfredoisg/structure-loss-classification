from sklearn.model_selection import StratifiedKFold
import pytorch_lightning as pl
import torch.nn as nn

def train_model(model: nn.Module, lightning_module: pl.LightningModule, trainer_config: dict,):
    
    
    pass


def train_with_CV(num_splits: int, ):

    kfold = StratifiedKFold(n_splits=num_splits, shuffle=True,)
    pass
