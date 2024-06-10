import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scienceplots
import matplotlib.colors as mcolors

from typing import Optional, Union

import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

plt.style.use(["science", "notebook", "grid"])


def process_plot_image(data, x: int, plot: bool = False):
    image_data = np.transpose(data[x][0])
    image_data = np.rot90(image_data, k=-1)  # Rotate 90 degrees counter-clockwise
    image_data = np.flip(image_data, axis=1)

    # Normalize or rescale the image data
    if image_data.dtype == np.float32 or image_data.dtype == np.float64:
        if image_data.min() < 0 or image_data.max() > 1:
            image_data = (image_data - image_data.min()) / (
                image_data.max() - image_data.min()
            )
    elif image_data.dtype == np.uint8:
        if image_data.min() < 0 or image_data.max() > 255:
            image_data = np.clip(image_data, 0, 255)
    else:
        # Rescale to [0, 1] for other data types
        image_data = (image_data - image_data.min()) / (
            image_data.max() - image_data.min()
        )

    # Display the image
    if plot:
        plt.imshow(image_data)
    else:
        return image_data


def compare(dfs: list, labels: list, save: bool = False, filename: str = None):

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for df, label in zip(dfs, labels):

        ax.plot(df.epoch, df.mean_val_loss, label=f"validation loss - {label}")
        ax.plot(df.epoch, df.mean_val_accuracy, label=f"validation accuracy - {label}")

        ax.fill_between(
            df.epoch,
            df.mean_val_loss + df.std_val_loss,
            df.mean_val_loss - df.std_val_loss,
            alpha=0.2,
        )
        ax.fill_between(
            df.epoch,
            df.mean_val_accuracy + df.std_val_accuracy,
            df.mean_val_accuracy - df.std_val_accuracy,
            alpha=0.2,
        )
        ax.set_ylim(0, 1.6)
        ax.set_xlabel("Epoch")
    ax.legend()
    
    if save:
        fig.savefig(
            filename,
        )

    


def display_metrics(
    path_to_data: Optional[Union[pd.DataFrame, str]] = None,
    y_lim: float = None,
    save: bool = False,
    path_to_file: Optional[str] = None,
):

    if isinstance(path_to_data, str):  # assuming 'data' is a path to a CSV file
        df = pd.read_csv(path_to_data)
    elif isinstance(path_to_data, pd.DataFrame):
        df = path_to_data
    else:
        raise ValueError("Invalid input: data must be a DataFrame or a CSV file path")

    plt.plot(df.epoch, df.mean_val_loss, label="validation loss")
    plt.fill_between(
        df.epoch,
        df.mean_val_loss + df.std_val_loss,
        df.mean_val_loss - df.std_val_loss,
        color="tab:blue",
        alpha=0.4,
    )

    plt.plot(df.epoch, df.mean_val_accuracy, label="validation accuracy")
    plt.fill_between(
        df.epoch,
        df.mean_val_accuracy + df.std_val_accuracy,
        df.mean_val_accuracy - df.std_val_accuracy,
        color="tab:green",
        alpha=0.4,
    )

    plt.xlabel("Epoch")
    plt.ylim(0.13, y_lim)

    plt.title("Mean Accuracy and Validation loss with Standard Deviation")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(path_to_file, bbox_inches="tight")


def display_cm(
    cm,
    labels,
    classifier_name: str,
    model: nn.Module,
    classification_mode: str,
    save: bool = True,
):
    """
    Display and save the confusion matrix as a PDF.

    :param cm: The confusion matrix.
    :param labels: The labels for the classes.

    """

    # Create the ConfusionMatrixDisplay object
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # Create a new figure for the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the confusion matrix and customize the appearance
    cm_display.plot(ax=ax, cmap="viridis")
    plt.title(classifier_name)

    # Save the confusion matrix as a PDF
    if save:
        plt.savefig(
            f"results/cm-hybrid-training/{model.__class__.__name__}/{classification_mode}/{classifier_name}.pdf",
            bbox_inches="tight",
        )
        print("Confusion Matrix saved")
    else:
        plt.show()

def plot_metrics(dfs: list, model: str, classification_mode: str, y_lim: Optional[float], save: bool=False):
    
# Assuming 'dfs' is a list of dataframes and each dataframe has 'epoch' and 'val_accuracy' columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    for c, df in enumerate(dfs):
        # Plot the main line with markers
        ax2.plot(df.epoch, df.val_accuracy, '-o',
                            color=list(mcolors.TABLEAU_COLORS.values())[c],
                            label=f'round {str(c+1)}')
        
        ax1.plot(df.epoch, df.val_loss, '-o',
                            color=list(mcolors.TABLEAU_COLORS.values())[c],
                        )


    ax2.legend(loc=(1.005, 0.15))
    ax1.set_title('validation loss')
    ax2.set_title('validation accuracy')
    ax1.set_xlabel('Epoch')
    ax2.set_xlabel('Epoch')
    
    ax1.set_ylim(0, y_lim)
    
    if save:
        plt.savefig(f'results/pipeline_3/{model}-{classification_mode}.pdf', bbox_inches='tight')
    plt.show()
