import numpy as np
import plotly.express as px


def create_plotly_figure(img_array, 
                         title="Image Colormap", 
                         cmap='jet', 
                         color_range=None,):
    """
    Create a Plotly figure for an image array.
    """

    if color_range is None:
        color_range = [np.min(img_array), np.max(img_array)]

    fig = px.imshow(img_array, 
                    color_continuous_scale=cmap,
                    range_color=color_range,
                    labels=dict(x="x", y="y", color="value"))

    fig.update_layout(title=title)

    return fig

def plot_histogram(img_array, title="Histogram"):
    """
    Plot a histogram of the image array.
    """
    # flatten the image array into a 1D array
    img_array_1D = img_array.flatten()
    # print(f"img_array_1D shape: {img_array_1D.shape}")
    fig = px.histogram(img_array_1D, nbins=256, title=title)
    return fig

def image_array_statistics(img_array):
    """
    Calculate the statistics of the image array.
    """
    img_array_1D = img_array.flatten()
    return np.mean(img_array_1D), np.std(img_array_1D), np.min(img_array_1D), np.max(img_array_1D)
