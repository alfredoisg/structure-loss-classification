import os

import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer


from datasets.data_modules import CustomImageDataModule
from lightning_modules.lightning_modules import LitModelBase


class HyperParameterTuner:
    """
    A class for hyperparameter tuning using PyTorch Lightning models and Ray Tune.

    This class encapsulates the setup and execution of hyperparameter tuning experiments,
    allowing for systematic searches across a defined hyperparameter space using Ray Tune's
    optimization algorithms and PyTorch Lightning's training framework.

    Parameters:
    - model (LitModelBase): A PyTorch Lightning model to be tuned.
    - datamodule (CustomImageDataModule): A PyTorch Lightning DataModule providing the dataset for training and validation.
    - search_space (dict): A dictionary defining the search space for hyperparameters.
                            Keys are hyperparameter names, and values are Ray Tune search spaces (e.g., tune.grid_search, tune.uniform).
    - resources (dict, optional): A dictionary specifying the computational resources to use for the tuning process.
                                  Defaults to using all available CPUs and GPUs.
                                  Expected keys are 'num_cpus' and 'num_gpus'.
    - num_samples (int, optional): The number of samples or trials to run in the tuning process. Defaults to 3.
    - num_epochs (int, optional): The number of epochs to train each trial. Defaults to 2.

    Methods:
    - auto_init_ray(): Initializes or resets the Ray environment with the specified resources.
    - train_func(config): The training function that will be passed to Ray Tune. It sets up and runs the PyTorch Lightning Trainer.
    - hypertune(): Executes the hyperparameter tuning process, logging results and returning the best configuration found.

    Example usage:
    ```python
    model = MyLightningModel(num_classes=2,
                             size_layer_1=default_config['layer_1_size'],
                             size_layer_2=default_config['layer_2_size'],
                             learning_rate=default_config['lr'])

    datamodule = MyDataModule()
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128])
    }
    tuner = HyperParameterTuner(model, datamodule, search_space, num_samples=10, num_epochs=10)
    best_config = tuner.hypertune()
    print("Best hyperparameters found were: ", best_config)
    ```
    """

    def __init__(
        self,
        model: LitModelBase,
        datamodule: CustomImageDataModule,
        search_space: dict,
        resources: dict = {
            "num_cpus": os.cpu_count(),
            "num_gpus": torch.cuda.device_count(),
        },
        num_samples: int = 3,
        num_epochs: int = 2,
    ) -> None:

        self.model = model
        self.datamodule = datamodule
        self.search_space = search_space
        self.resources = resources
        self.num_samples = num_samples
        self.num_epochs = num_epochs

    def auto_init_ray(self):

        resources = self.resources

        try:
            ray.init(**resources)
        except RuntimeError:
            ray.shutdown()
            ray.init(**resources)

    def train_func(self, config):

        dm = self.datamodule

        model = self.model

        num_steps_per_epoch = max(
            1, (len(dm.train_dataset) + len(dm.val_dataset)) // config["batch_size"]
        )

        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=False,
            max_epochs=self.num_epochs,
            log_every_n_steps=num_steps_per_epoch,
        )

        trainer = prepare_trainer(trainer)
        trainer.fit(model, datamodule=dm)

    def hypertune(self):

        self.auto_init_ray()

        use_gpu = True if self.resources["num_gpus"] > 0 else False

        scaling_config = ScalingConfig(
            num_workers=1,
            use_gpu=use_gpu,
            trainer_resources={"CPU": self.resources["num_cpus"]},
        )

        run_config = RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="val_accuracy",
                checkpoint_score_order="max",
            ),
        )

        ray_trainer = TorchTrainer(
            self.train_func,
            scaling_config=scaling_config,
            run_config=run_config,
        )

        scheduler = ASHAScheduler(
            max_t=self.num_epochs, grace_period=1, reduction_factor=2
        )

        tuner = tune.Tuner(
            ray_trainer,
            param_space={"train_loop_config": self.search_space},
            tune_config=tune.TuneConfig(
                metric="val_accuracy",
                mode="max",
                num_samples=self.num_samples,
                scheduler=scheduler,
            ),
        )

        results = tuner.fit()
        best_config = results.get_best_result(metric="val_accuracy", mode="max").config
        print(" Best hyperparameter configuration found: ", best_config)

        return best_config


def hypertune_classifier(
    ml_model,
    X: np.array,
    Y: np.array,
    test_size: float,
    param_grid: dict,
    scoring_metric: str = "accuracy",
) -> dict:
    """
    Function to tune hyperparameters for classification models.

    :param ml_model: The machine learning model class to be tuned
    :param X: Feature dataset as a numpy array
    :param Y: Target dataset as a numpy array
    :param test_size: Fraction of the dataset to be used as test set
    :param param_grid: Dictionary with parameters names (str) as keys and lists of parameter settings to try as values
    :param scoring_metric: Metric to be used for evaluating the predictions on the test set. Default is 'accuracy'.
    :return: Dictionary of the best parameters found by GridSearchCV
    """

    model = ml_model()

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring=scoring_metric, n_jobs=-1
    )

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, shuffle=False
    )

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)

    # Evaluation
    score = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {score}")

    print(f"Best Parameters: {grid_search.best_params_}")

    return grid_search.best_params_
