import numpy as np
import matplotlib.pyplot as plt
from .display import process_plot_image
from torch.utils.data import Dataset

def display_filters(
    features: list,
    img_num: int,
    layers: list,
    cmap: str,
    name_layers: list,
    data: Dataset,
    collapse_func = np.mean,
    normalize: bool = True,
    plot: bool = True,
    save: bool = False,
    model_class: str = None
):
    """
    Display the original image and collapsed feature maps for each layer of interest.

    :param features: dictionary containing the extracted features for each layer
    :param img_num: index of the image in the batch
    :param layers: list of layer names to display the feature maps
    :param cmap: colormap for the feature maps
    :param name_layers: list of names for the layers to be displayed
    :param original_images: batch of original images
    :param collapse_func: function to use for collapsing channels (default: np.mean)
    """

    image_data = process_plot_image(data, img_num, False)

    num_layers = len(layers)
    fig, axes = plt.subplots(
        1, num_layers + 1, figsize=(32, 8)
    )  # Add 1 for the original image

    # Display the original image
    axes[0].imshow(image_data)
    axes[0].set_title("Original Image")

    used_axes = [axes[0]]  # Track used axes

    for count, ax in enumerate(axes[1:], 1):  
        feature_map = features[layers[count - 1]][img_num]

        # Check if the shape is invalid for visualization
        if feature_map.shape[-2:] == (1, 1):
            print(f"Skipping visualization for {layers[count - 1]} due to invalid shape.")
            continue  # Skip this iteration

        collapsed_feature_map = collapse_func(
            features[layers[count - 1]][img_num], axis=0
        )

        if normalize:
            collapsed_feature_map = (collapsed_feature_map - collapsed_feature_map.min()) / (collapsed_feature_map.max() - collapsed_feature_map.min())


        if collapsed_feature_map.ndim > 2:
            collapsed_feature_map = collapsed_feature_map.squeeze()
        # Ensure the data is at least 2D
        if collapsed_feature_map.ndim < 2:
            collapsed_feature_map = collapsed_feature_map.reshape(1, -1)
        im = ax.imshow(collapsed_feature_map.squeeze(), cmap=cmap)
        ax.set_title(f"{name_layers[count - 1]}")
        used_axes.append(ax)  # Mark this axis as used

    # Remove unused axes
    for ax in axes:
        if ax not in used_axes:
            fig.delaxes(ax)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.81, 0.15, 0.015, 0.7])
    
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Normalized Filter Activation Intensity')  # Set the title for the color bar

    if save:
        plt.savefig(f'filters_{model_class}.pdf', bbox_inches='tight')
    
    plt.show()


def save_filters(
    features: list,
    img_num: int,
    layers: int,
    cmap: str,
    name_layers: list,
    data,
    model,
    collapse_func=np.mean,
):
    """
    Display the original image and collapsed feature maps for each layer of interest.

    :param features: dictionary containing the extracted features for each layer
    :param img_num: index of the image in the batch
    :param layers: list of layer names to display the feature maps
    :param cmap: colormap for the feature maps
    :param name_layers: list of names for the layers to be displayed
    :param original_images: batch of original images
    :param collapse_func: function to use for collapsing channels (default: np.mean)
    """

    image_data = process_plot_image(data, img_num, False)

    num_layers = len(layers)
    fig, axes = plt.subplots(
        1, num_layers + 1, figsize=(25, 8)
    )  # Add 1 for the original image

    # Display the original image
    axes[0].imshow(image_data)
    axes[0].set_title("Original Image")

    for count, ax in enumerate(
        axes[1:], 1
    ):  # Start from 1 to leave space for the original image
        collapsed_feature_map = collapse_func(
            features[layers[count - 1]][img_num], axis=0
        )
        im = ax.imshow(collapsed_feature_map.squeeze(), cmap=cmap)
        ax.set_title(f"{name_layers[count - 1]}")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.savefig(f"results/{model.__class__.__name__}_filters.pdf", bbox_inches="tight")
