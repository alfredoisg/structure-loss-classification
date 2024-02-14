import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import torch.nn as nn
import torch

from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader


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

    # Collect validation metrics from the most recent epoch
    val_metrics = trainer.callback_metrics

    # Example: Log or print the validation metrics for this fold
    print(f"Fold {fold} Validation Metrics:", val_metrics)

    # Return the validation metrics for aggregation
    return val_metrics



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
