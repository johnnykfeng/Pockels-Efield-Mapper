import numpy as np
import plotly.express as px
import os


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


# DEPRECATED
# def plot_image_with_color_slider(
#     image_array,
#     title,
#     color_map,
#     global_range_pctl,
#     fig_height=800,
#     fig_width=800,
#     pctl_color_range=False,
#     manual_color_range=None,
#     bounding_box=None,
# ):
    
#     if manual_color_range:
#         color_range = manual_color_range
#     elif pctl_color_range:
#         col1, col2 = st.columns(2)
#         with col1:
#             min_pctl = st.number_input(
#                 "Min percentile",
#                 min_value=0.0,
#                 max_value=100.0,
#                 value=5.0,
#                 key=f"{title}_min_pctl",
#             )
#         with col2:
#             max_pctl = st.number_input(
#                 "Max percentile",
#                 min_value=0.0,
#                 max_value=100.0,
#                 value=95.0,
#                 key=f"{title}_max_pctl",
#             )
#         if min_pctl < max_pctl:
#             color_range = np.percentile(image_array, q=[min_pctl, max_pctl])
#         else:
#             color_range = [float(np.min(image_array)), float(np.max(image_array))]
#     else:
#         vmin, vmax = np.percentile(image_array, q=global_range_pctl)

#         color_range = st.slider(
#             "Color range",
#             min_value=0.0,
#             max_value=65500.0,
#             value=[vmin, vmax],
#             key=f"{title}_color_range",
#         )

#     fig = create_plotly_figure(
#         image_array, title=title, cmap=color_map, color_range=color_range
#     )
#     if bounding_box:
#         # Check if bounding box dimensions exceed image array dimensions
#         if (bounding_box[2] > image_array.shape[1] or 
#             bounding_box[3] > image_array.shape[0]):
#             st.warning("Warning: Bounding box dimensions exceed image dimensions")
#             bounding_box = [0, 0, image_array.shape[1], image_array.shape[0]]
#         fig.add_shape(
#             type="rect",
#             x0=bounding_box[0],
#             y0=bounding_box[1],
#             x1=bounding_box[2],
#             y1=bounding_box[3],
#         line=dict(
#             color="white",
#             width=3,
#             dash="dash"
#             )
#         )
        
#     fig.update_layout(
#         height=fig_height,
#         width=fig_width
#     )
    
#     return fig