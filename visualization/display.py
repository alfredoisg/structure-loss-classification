import numpy as np
import matplotlib.pyplot as plt

def process_plot_image(data, x: int, plot: bool = False):
    image_data = np.transpose(data[x][0])
    image_data = np.rot90(image_data, k=-1)  # Rotate 90 degrees counter-clockwise
    image_data=np.flip(image_data, axis=1)

    # Normalize or rescale the image data
    if image_data.dtype == np.float32 or image_data.dtype == np.float64:
        if image_data.min() < 0 or image_data.max() > 1:
            image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
    elif image_data.dtype == np.uint8:
        if image_data.min() < 0 or image_data.max() > 255:
            image_data = np.clip(image_data, 0, 255)
    else:
        # Rescale to [0, 1] for other data types
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())

    # Display the image
    if plot:
        plt.imshow(image_data)
    else:
        return image_data