import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import torch.nn as nn
import torch

from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold

def train_model(
    model: pl.LightningModule,
    trainer_config: dict,
    save_dir: str,
    data_module,
    use_best_model: bool = False,
    fold: int = None,
) -> dict:
    """
    Initializes the PyTorch Lightning Trainer with the given configuration.

    Parameters:
    - trainer_config (dict): Configuration dictionary for the Trainer with the following structure:
        - 'accelerator' (str): Specifies the type of accelerator to use ('gpu', 'cpu', etc.).
        - 'devices' (int): Number of devices to train on (1 for single device, -1 for all available devices).
        - 'max_epochs' (int): Maximum number of epochs to train for.
        - 'precision' (int): Floating point precision (16 for FP16, 32 for FP32, etc.).
        - Additional keys as needed by PyTorch Lightning's Trainer.

    Example:
    trainer_config = {
        'patience': ,
        'accelerator': 'gpu',
        'devices': 1,
        'max_epochs': 100,
        'precision': 32,
        'log_every_n_steps':
    }
    """

    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=trainer_config["patience"], min_delta=0.005
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    csv_logger = CSVLogger(f"{save_dir}/{model.__class__.__name__}")

    # Determine checkpoint path for loading best model if applicable
    ckpt_path = None
    # if use_best_model and fold is not None:
    #     ckpt_path = best_model_paths.get(f"fold_{fold}")

    trainer = pl.Trainer(
        accelerator=trainer_config["accelerator"],
        devices=trainer_config["devices"],
        max_epochs=trainer_config["max_epochs"],
        precision=trainer_config["precision"],
        log_every_n_steps=trainer_config["n_steps"],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=csv_logger,
    )

    trainer.fit(model=model, datamodule=data_module)

    trainer.validate(model=model, datamodule=data_module)
 
 
    cm = model.stored_confusion_matrix
    print("Validation Confusion Matrix:", cm)
    

    # Collect validation metrics and include the confusion matrix
    val_metrics = trainer.callback_metrics
    val_metrics_cpu = {key: val.cpu().item() for key, val in val_metrics.items()}



    return val_metrics_cpu, cm 

def train_with_cv(
    model_class: pl.LightningModule,
    model_params: dict,
    trainer_config: dict,
    data_module_class: pl.LightningDataModule,
    data_module_params: dict,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    save_dir_base: str = "logdir-struct/",
) -> dict:
    """
    Trains a model using k-fold cross-validation.
    """
    
    # Initialize KFold or StratifiedKFold
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # Storage for all validation metrics for aggregation later
    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data_module_params["data"])):
        print(f"Starting Fold {fold+1}/{n_splits}")
        
        # Update data_module_params with current fold's indices
        data_module_params.update({'train_idx': train_idx, 'val_idx': val_idx})
        
        # Instantiate the model and data module for the current fold
        model = model_class(**model_params)
        data_module = data_module_class(**data_module_params)
        
        # Set save directory for the current fold
        save_dir = f"{save_dir_base}/fold_{fold+1}"
        
        # Train the model using the existing train_model function
        val_metrics = train_model(
            model=model,
            trainer_config=trainer_config,
            save_dir=save_dir,
            data_module=data_module,
            use_best_model=True,  # Assuming you want to use the best model; adjust as needed
            fold=fold
        )

        all_metrics.append(val_metrics)
        
    # Calculate and return summary metrics (mean, std) for all folds
    metrics_summary = calculate_metrics_summary(all_metrics)
    return metrics_summary



def get_features(model: nn.Module, layers: list, data_loader: DataLoader, device: str):

    feature_extractor = create_feature_extractor(model, layers)

    features = {layer_name: [] for layer_name in layers}
    labels = []

    feature_extractor.eval()
    with torch.no_grad():
        for image, label in data_loader:
            image, label = image.to(device), label.to(device)
            predicted_dict = feature_extractor(image)

            for layer_name in layers:
                features[layer_name].extend(predicted_dict[layer_name].cpu().numpy())
            labels.extend(label.cpu().numpy())

    return features, labels
