import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px
import os
import streamlit as st


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

def save_plotly_figure(fig, filename, save_dir=None):
    """
    Save a Plotly figure to a file.
    """
    if save_dir is not None:
        filename = os.path.join(save_dir, filename)
    if not filename.endswith('.html'):
        filename = filename + '.html'
    fig.write_html(filename)

def colored_pockels_images_matplotlib(images_dict: dict, 
                                      color_range_radio: str, 
                                      color_min: float, 
                                      color_max: float, 
                                      apply_bounding_box: bool, 
                                      bounding_box: tuple,
                                      box_color="white"):
    """
    Plot a set of colored Pockels images using Matplotlib.
    """
    n_rows = len(images_dict)
    mat_fig, axs = plt.subplots(n_rows, 1, figsize=(10, n_rows*2.5))
    plt.subplots_adjust(hspace=0.4)  # Increase vertical spacing between subplots
    mat_fig.tight_layout()
    # mat_fig.suptitle(f"Pockels Images with {color_range_radio}-scale color range", fontsize=15, y=1.05)
    for i, key in enumerate(images_dict):
        if n_rows == 1:
            ax = axs
        else:
            ax = axs[i]
        img_array = images_dict[key]
        im = ax.imshow(img_array, cmap="jet")
        if color_range_radio == "Fixed":
            im.set_clim(color_min, color_max)
        plt.colorbar(im, ax=ax)
        ax.set_title(key, fontsize=8)  # Decrease title font size
        ax.tick_params(axis='both', which='major', labelsize=8)  # Decrease tick label size
        if apply_bounding_box:
            # Draw bounding box on matplotlib subplot
            rect = patches.Rectangle((bounding_box[0], bounding_box[1]), 
                                bounding_box[2]-bounding_box[0], 
                                bounding_box[3]-bounding_box[1],
                                linewidth=2, edgecolor=box_color, facecolor='none', 
                                linestyle='--')
            ax.add_patch(rect)
    return mat_fig


def heatmap_plot_with_bounding_box(
    image_array,
    title,
    color_map,
    color_range,
    fig_height=800,
    fig_width=800,
    bounding_box=None,
):
    
    fig = create_plotly_figure(
        image_array, title=title, cmap=color_map, color_range=color_range
    )
    if bounding_box:
        # Check if bounding box dimensions exceed image array dimensions
        if (bounding_box[2] > image_array.shape[1] or 
            bounding_box[3] > image_array.shape[0]):
            st.warning("Warning: Bounding box dimensions exceed image dimensions")
            bounding_box = [0, 0, image_array.shape[1], image_array.shape[0]]
        fig.add_shape(
            type="rect",
            x0=bounding_box[0],
            y0=bounding_box[1],
            x1=bounding_box[2],
            y1=bounding_box[3],
        line=dict(
            color="white",
            width=3,
            dash="dash"
            )
        )
        
    fig.update_layout(
        height=fig_height,
        width=fig_width
    )
    
    return fig

