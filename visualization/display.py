import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from lightning_modules.lightning_modules import LitModelBase


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


# def display_metrics(
#     model_class: LitModelBase,
#     classification_mode: str,
#     version: int,
#     parent_dir: str = None,
#     plot=False,
# ):

#     dfs = []

#     if parent_dir is None:
#         parent_dir = f"logdir/{model_class.__name__}/{classification_mode}/cv"

#     for fold in range(len(os.listdir(parent_dir))):

#         print(f"{parent_dir}/fold_{fold+1}")

#         df = pd.read_csv(
#             f"{parent_dir}/fold_{fold+1}/lightning_logs/version_{version}/metrics.csv"
#         )

#         dfs.append(df)

#     all_folds = pd.concat(dfs, ignore_index=True)
#     grouped = (
#         all_folds.groupby("epoch")
#         .agg({"val_accuracy": ["mean", "std"], "val_loss": ["mean", "std"]})
#         .reset_index()
#     )

#     grouped.columns = [
#         "epoch",
#         "mean_val_accuracy",
#         "std_val_accuracy",
#         "mean_val_loss",
#         "std_val_loss",
#     ]

#     if plot:

#         plt.plot(grouped.epoch, grouped.mean_val_loss, label="validation loss")
#         plt.fill_between(
#             grouped.epoch,
#             grouped.mean_val_loss + grouped.std_val_loss,
#             grouped.mean_val_loss - grouped.std_val_loss,
#             color="tab:blue",
#             alpha=0.4,
#         )

#         plt.plot(grouped.epoch, grouped.mean_val_accuracy, label="validation accuracy")
#         plt.fill_between(
#             grouped.epoch,
#             grouped.mean_val_accuracy + grouped.std_val_accuracy,
#             grouped.mean_val_accuracy - grouped.std_val_accuracy,
#             color="tab:green",
#             alpha=0.4,
#         )

#         plt.xlabel("Epoch")

#         plt.title("Mean Accuracy and Validation loss with Standard Deviation")
#         plt.legend()
#         plt.grid(True)

#     return grouped

def display_metrics(
    csv_file: str,
    save: bool = False,
    save_dir: str = None
):

    df = pd.read_csv(csv_file)

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

    plt.title("Mean Accuracy and Validation loss with Standard Deviation")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig()
