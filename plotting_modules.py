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

