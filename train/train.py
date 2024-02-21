import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import torch.nn as nn
import torch

from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold

from datasets.datasets import CustomDatasetWrapper
from datasets.data_modules import CustomImageDataModule


def train_model(
    model: pl.LightningModule,
    trainer_config: dict,
    save_dir: str,
    data_module,
    test: bool = True,
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
        monitor="val_loss", patience=trainer_config["patience"], min_delta=0.00
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    if test:
        callbacks = [early_stop_callback]

    else:
        callbacks = [checkpoint_callback, early_stop_callback]

    csv_logger = CSVLogger(f"{save_dir}")

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
        callbacks=callbacks,
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
    model: pl.LightningModule,
    model_params: dict,
    trainer_config: dict,
    data: CustomDatasetWrapper,
    targets: list,
    classification_mode: str,
    save_dir_base: str = None,
    n_splits: int = 5,
    shuffle: bool = True,
    batch_size: int = 16,
    num_workers: int = 12,
    random_state: int = 42,
) -> dict:
    """
    Trains a model using k-fold cross-validation.
    """

    if save_dir_base == None:
        save_dir_base = f"logdir/{model.__name__}/{classification_mode}/cv/"

    # Initialize KFold or StratifiedKFold
    kfold = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )

    # Storage for all validation metrics for aggregation later
    all_metrics = {}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data, targets)):
        print(f"Starting Fold {fold+1}/{n_splits}")

        model = model(**model_params)

        # Update data_module_params with current fold's indices
        train_data = torch.utils.data.Subset(data, train_idx)
        val_data = torch.utils.data.Subset(data, val_idx)

        data_module = CustomImageDataModule(
            train_dataset=train_data,
            val_dataset=val_data,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Set save directory for the current fold
        save_dir = f"{save_dir_base}/fold_{fold+1}"

        # Train the model using the existing train_model function
        val_metrics = train_model(
            model=model,
            trainer_config=trainer_config,
            save_dir=save_dir,
            data_module=data_module,
            fold=fold,
        )

        all_metrics[f"Fold {fold+1}"] = val_metrics

    return all_metrics


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
