import os

import torch
import numpy as np

import pytorch_lightning as pl

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.tune import JupyterNotebookReporter
from ray.train.torch import TorchTrainer

from ray.tune.sklearn import TuneGridSearchCV
from sklearn.model_selection import cross_val_score

from datasets.data_modules import CustomImageDataModule



class HyperParameterTuner:
    """
    hyperparameter tuning of PyTorch Lightning models using Ray Tune.

    Parameters:
    - model_class (LitModelBase): Class of the PyTorch Lightning model to be tuned.
    - datamodule (CustomImageDataModule): PyTorch Lightning DataModule instance providing the dataset.
    - search_space (dict): Dictionary defining the search space for hyperparameters.
    - resources (dict, optional): Resources for the tuning process (num_cpus, num_gpus).
    - num_samples (int, optional): Number of hyperparameter combinations to sample.
    - num_epochs (int, optional): Number of epochs to train each model configuration.

    Usage:
        >>> model_class = YourModelClass
        >>> datamodule = YourDataModule(...)
        >>> search_space = {'model_params': {...}, 'batch_size': tune.choice([32, 64, 128])}
        >>> tuner = HyperParameterTuner(model_class, datamodule, search_space, num_samples=10, num_epochs=10)
        >>> best_config = tuner.hypertune()
        >>> print("Best hyperparameters found were: ", best_config)
    """

    def __init__(
        self,
        model_class: pl.LightningModule,
        datamodule: CustomImageDataModule,
        search_space: dict,
        resources: dict = {
            "num_cpus": os.cpu_count(),
            "num_gpus": torch.cuda.device_count(),
        },
        num_samples: int = 3,
        num_epochs: int = 2,
    ) -> None:

        self.model_class = model_class
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
        torch.set_float32_matmul_precision('high')
        
        dm = self.datamodule
        model = self.model_class(config)

        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(),
            callbacks=[RayTrainReportCallback()],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=False,
            max_epochs=self.num_epochs,
            # log_every_n_steps=num_steps_per_epoch,
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
            progress_reporter=JupyterNotebookReporter(),
            verbose=1
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

        ray.shutdown()

        return best_config["train_loop_config"]


class SKLearnHyperParameterTuner:
    """
    A class for hyperparameter tuning of scikit-learn models using Ray Tune with
    optional GPU acceleration.

    This class supports the hyperparameter tuning of any scikit-learn-compatible
    models by leveraging Ray Tune for distributed and parallelized tuning sessions.
    It supports using both CPUs and GPUs (if applicable) to potentially accelerate
    the tuning process.

    Parameters:
    - model: The scikit-learn model class or a compatible model with a similar interface
      to be tuned.
    - search_space (dict): A dictionary defining the hyperparameter search space.
    - X (np.array): The feature dataset.
    - y (np.array): The target dataset.
    - use_gpu (bool, optional): Whether to utilize GPUs for the tuning process, if available.
      Defaults to True.
    - cv (int, optional): The number of cross-validation folds. Defaults to 4.
    - scoring (str, optional): The scoring metric to use for evaluation. Defaults to 'accuracy'.
    - resources (dict, optional): A dictionary specifying the computational resources
      to use for the tuning process, with keys 'num_cpus' and 'num_gpus'. Defaults to
      using all available CPUs and GPUs.
    - num_samples (int, optional): The number of hyperparameter combinations to sample
      and evaluate. Defaults to 5.

    Example usage:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> X, y = load_data()
        >>> param_grid = {
        ...     'n_estimators': tune.choice([10, 100, 1000]),
        ...     'max_depth': tune.choice([5, 10, 20, None])
        ... }
        >>> tuner = SKLearnHyperParameterTuner(RandomForestClassifier, param_grid, X, y)
        >>> best_params = tuner.hypertune()
        >>> print("Best hyperparameters found:", best_params)
    """

    def __init__(
        self,
        model,
        search_space: dict,
        X: np.array,
        y: np.array,
        use_gpu: bool = True,
        cv=4,
        scoring="accuracy",
        resources=None,
        num_samples=5,
    ):

        self.model = model
        self.search_space = search_space
        self.X = ray.put(X)
        self.y = ray.put(y)
        self.use_gpu = use_gpu
        self.cv = cv
        self.scoring = scoring
        self.resources = resources or {
            "num_cpus": os.cpu_count(),
            "num_gpus": torch.cuda.device_count() if use_gpu else 0,
        }
        self.num_samples = num_samples

    def auto_init_ray(self):
        try:
            if not ray.is_initialized():
                ray.init(**self.resources)
        except RuntimeError as e:
            print(f"Error initializing Ray: {e}")
            ray.shutdown()
            ray.init(**self.resources)
            print(self.resources)

    def train_model(self, config):
        X = ray.get(self.X)
        y = ray.get(self.y)

        model_params = {k: v for k, v in config.items() if k != "model"}
        model = self.model(**model_params)

        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=self.cv,
        )  # scoring=self.scoring)
        mean_cv_score = np.mean(cv_scores)

        ray.train.report(metrics={"mean_cv_score": mean_cv_score})

    def hypertune(self):
        self.auto_init_ray()
        print(f"------ {self.resources} -----")
        my_trainable = tune.with_resources(
            trainable=self.train_model,
            resources={
                "cpu": self.resources["num_cpus"],
                "gpu": self.resources["num_gpus"],
            },
        )

        my_tune_config = tune.TuneConfig(
            metric="mean_cv_score", mode="max", num_samples=self.num_samples
        )

        analysis = tune.Tuner(
            trainable=my_trainable,
            param_space=self.search_space,
            tune_config=my_tune_config,
        )

        results = analysis.fit()
        best_config = results.get_best_result(metric="mean_cv_score", mode="max").config
        print("Best hyperparameter configuration found:", best_config)

        ray.shutdown()

        return best_config


def hypertune_classifier(
    ml_model,
    X: np.array,
    Y: np.array,
    test_size: float,
    param_grid: dict,
    use_ray: bool = False,
    scoring_metric: str = "accuracy",
) -> dict:
    """
    Simplifies the hyperparameter tuning process for classification models, supporting
    both traditional GridSearchCV and distributed tuning via Ray Tune.

    Parameters:
    - ml_model: The machine learning model class (e.g., from scikit-learn) to be tuned.
    - X (np.array): The feature dataset.
    - Y (np.array): The target dataset.
    - test_size (float): The proportion of the dataset to use as the test set.
    - param_grid (dict): The hyperparameter search space.
    - use_ray (bool, optional): Whether to use Ray Tune for distributed tuning. Defaults to False.
    - scoring_metric (str, optional): The metric to use for scoring the model performance.
      Defaults to "accuracy".

    Returns:
    - dict: The best hyperparameter configuration found during the tuning process.

    Example usage:
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> X, y = load_iris(return_X_y=True)
        >>> param_grid = {
        ...     'n_estimators': [10, 100, 1000],
        ...     'max_depth': [5, 10, None]
        ... }
        >>> best_params = hypertune_classifier(RandomForestClassifier, X, y, 0.2, param_grid, use_ray=True)
        >>> print("Best hyperparameters:", best_params)
    """

    model = ml_model()

    # Set up GridSearchCV

    if use_ray:

        grid_search = TuneGridSearchCV(
            model, param_grid, early_stopping=False, use_gpu=True, max_iters=10
        )

    else:
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring=scoring_metric, n_jobs=-1
        )

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, shuffle=True
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
