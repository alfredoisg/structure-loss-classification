import pickle
import json

import os
from datasets.datasets import CustomDatasetWrapper
from lightning_modules.lightning_modules import LitModelBase
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

import numpy as np


def load_targets(data: CustomDatasetWrapper, from_dir: str = None):
    if from_dir is None:
        from_dir = f"logdir/cached_targets_{data.classification_mode}.pkl"

    try:
        with open(from_dir, "rb") as f:
            targets = pickle.load(f)
    except FileNotFoundError:
        targets = [t for _, t in data]
        # Cache the targets for next time
        with open(from_dir, "wb") as f:
            pickle.dump(targets, f)

    return targets


def load_hyperparameter(classification_mode, filename: str = None):

    if filename is None:
        filename = f"hyperparameters_{classification_mode}.json"

    with open(filename, "r") as f:
        test_config = json.load(f)
    return test_config


def get_stat_metrics(
    parent_dir: str,
    version: int,
    save_dir: str,
    file_name: str,
) -> pd.DataFrame:

    dfs = []

    dir_name = parent_dir[parent_dir.index("/") + 1 :]
    # parent_dir = parent_dir + "/cv"

    for fold in range(len(os.listdir(parent_dir))):

        print(f"{parent_dir}/fold_{fold+1}")

        df = pd.read_csv(
            f"{parent_dir}/fold_{fold+1}/lightning_logs/version_{version}/metrics.csv"
        )

        dfs.append(df)

    all_folds = pd.concat(dfs, ignore_index=True)
    
    grouped = (all_folds.groupby("epoch").agg({"val_accuracy": ["mean", "std"], "val_loss": ["mean", "std"]}).reset_index())

    grouped.columns = [
        "epoch",
        "mean_val_accuracy",
        "std_val_accuracy",
        "mean_val_loss",
        "std_val_loss",
    ]

    try:
        grouped.to_csv(f"results/{save_dir}/{file_name}.csv")
    except OSError:
        # Create the directory if it doesn't exist
        os.makedirs(f"results/{save_dir}", exist_ok=True)
        grouped.to_csv(f"results/{save_dir}/{file_name}.csv")

    return grouped


def get_train_val_data(
    data: CustomDatasetWrapper,
    targets: list,
    test_size: float = 0.2,

):
    train_idx, val_idx, _, _ = train_test_split(
        range(len(data)), targets, test_size=test_size, 
    )

    train_data = torch.utils.data.Subset(data, train_idx)
    val_data = torch.utils.data.Subset(data, val_idx)

    return train_data, val_data


def get_category_names(data: CustomDatasetWrapper):

    names_dict = {
        "typeA": "Diameter\nFluctuations",
        "typeB": "Node Cut",
        "typeC": "Particle Hit",
        "goodIngots": "No Structure\nLoss",
    }

    list_of_categories = [data.dataset.classes[i] for i in data.label_map.keys()]

    if len(np.unique(list(data.label_map.values()))) == 2:
        category_names = ["Good Ingots", "Bad Ingots"]

    else:
        category_names = [names_dict[class_name] for class_name in list_of_categories]

    return category_names
