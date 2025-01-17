import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import time


def png_to_array(image_path):
    """
    Convert a PNG image to a numpy array.

    Args:
        image_path (str): Path to the PNG image file

    Returns:
        numpy.ndarray: Image data as a numpy array with shape (height, width) for grayscale
                      or (height, width, channels) for RGB/RGBA
    """
    # Open image using PIL
    img = Image.open(image_path)

    # Convert to numpy array
    img_array = np.array(img)

    return img_array


def save_array_to_png(img_array, filename, save_dir=None):
    """
    Convert a numpy array to a PNG image.
    Args:
        img_array (numpy.ndarray): Input array to save as PNG
        filename (str): Name of output file
        save_dir (str, optional): Directory to save the file in
    """
    # Convert floating point arrays to uint16
    if img_array.dtype in [np.float32, np.float64]:
        # Scale to full uint16 range
        scaled = (img_array - np.min(img_array)) / (
            np.max(img_array) - np.min(img_array)
        )
        img_array = (scaled * 65535).astype(np.uint16)
        
    img = Image.fromarray(img_array)
    if save_dir is not None:
        filename = os.path.join(save_dir, filename)
    if not filename.endswith('.png'):
        filename = filename + '.png'
    img.save(filename)

def save_array_to_csv(img_array, filename, save_dir=None):
    """
    Save a numpy array to a CSV file.
    """
    if save_dir is not None:
        filename = os.path.join(save_dir, filename)
    if not filename.endswith('.csv'):
        filename = filename + '.csv'
    np.savetxt(filename, img_array, delimiter=',')

def csv_to_array(filename):
    """
    Convert a CSV file to a numpy array.
    """
    return np.genfromtxt(filename, delimiter=',')


def plot_image_colormap(
    img_array,
    title="Image Colormap",
    cmap="jet",
    color_range=None,
    auto_color_range=False,
):
    """
    Plot a colormap visualization of an image array.

    Args:
        img_array (numpy.ndarray): Image data as numpy array
        title (str): Title for the plot
        cmap (str): Matplotlib colormap to use (default: 'viridis')
    """

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # If image is RGB/RGBA, convert to grayscale by taking mean across color channels
    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=2)

    if color_range is None:
        # Calculate 5th and 95th percentiles for color range
        vmin = np.percentile(img_array, 1)
        vmax = np.percentile(img_array, 99)
    else:
        vmin = color_range[0]
        vmax = color_range[1]

    if auto_color_range:
        # Plot colormap with percentile-based color range
        im = ax.imshow(img_array, cmap=cmap)
    else:
        # Plot colormap with fixed color range
        im = ax.imshow(img_array, cmap=cmap, vmin=vmin, vmax=vmax)

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Intensity")

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    plt.show()


def plot_image_plotly(
    img_array, title="Image Colormap", cmap="viridis", z_range=(5, 95)
):
    """
    Plot a colormap visualization of an image array using Plotly.
    The color range is set between the 5th and 95th percentiles of the image values.
    """

    # Calculate 5th and 95th percentiles
    vmin = np.percentile(img_array, z_range[0])
    vmax = np.percentile(img_array, z_range[1])

    fig = px.imshow(
        img_array,
        color_continuous_scale=cmap,
        zmin=vmin,  # Set minimum of color range
        zmax=vmax,
    )  # Set maximum of color range
    fig.update_layout(title=title)
    fig.show()


def crop_image(img_array, crop_range_x, crop_range_y):
    """
    Crop an image array to a specified range.
    """
    return img_array[
        crop_range_y[0] : crop_range_y[1], crop_range_x[0] : crop_range_x[1]
    ]


def impute_bad_pixels(img_array, bad_pixels):
    """
    Impute bad pixels in an image array.
    """
    for pixel in bad_pixels:
        x, y = pixel
        # Get values of surrounding pixels in a 3x3 grid, excluding the center pixel
        surrounding_values = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                # Skip the center pixel
                if i == 0 and j == 0:
                    continue
                # Check bounds to avoid index errors
                if 0 <= y + i < img_array.shape[0] and 0 <= x + j < img_array.shape[1]:
                    surrounding_values.append(img_array[y + i, x + j])
        # Calculate mean of surrounding pixels and assign to dead pixel
        if surrounding_values:
            img_array[y, x] = np.mean(surrounding_values)
    return img_array


def find_dead_pixels(img_array, threshold=100):
    """
    Find dead pixels in an image array.
    """
    dead_pixels = np.where(img_array < threshold)
    # Convert array indices to list of (x,y) tuples
    dead_pixel_coords = list(zip(dead_pixels[1], dead_pixels[0]))
    # return dead_pixel_coords
    return dead_pixel_coords


def find_bright_pixels(img_array, threshold=20e3):
    """
    Find bright pixels in an image array.
    """
    bright_pixels = np.where(img_array > threshold)
    # Convert array indices to list of (x,y) tuples
    bright_pixel_coords = list(zip(bright_pixels[1], bright_pixels[0]))
    return bright_pixel_coords


def cap_array(img_array, min_value, max_value):
    """
    Cap the values of an image array to a minimum and maximum value.
    """
    img_array[img_array < min_value] = min_value
    img_array[img_array > max_value] = max_value
    return img_array


def remove_low_value_pixels(img_array):
    """
    Remove zero and low value pixels from an image array.
    """
    img_array[(img_array > 0) & (img_array < 1.0)] = 1.0
    img_array[(img_array < 0) & (img_array > -1.0)] = -1.0
    img_array[img_array == 0] = 1.0
    return img_array


# if __name__ == "__main__":
#     image_dir = r"R:\Pockels_data\NEXT GEN POCKELS\pockels_run_2025-01-13"
#     crop_range_y = (190, 320)
#     crop_range_x = (5, 635)

#     calib_parallel_on = crop_image(png_to_array(os.path.join(image_dir, "calib_parallel_on.png")),
#                                    crop_range_x, crop_range_y)
#     vmin = np.percentile(calib_parallel_on, 10)
#     vmax = np.percentile(calib_parallel_on, 90)
#     print("Number of dead pixels: ", len(find_dead_pixels(calib_parallel_on)))
#     print(f"Dead pixels: {find_dead_pixels(calib_parallel_on)}")
#     plot_image_colormap(calib_parallel_on, title="Calibrated Parallel On", color_range=(vmin, vmax))

#     calib_parallel_on = impute_dead_pixels(calib_parallel_on, find_dead_pixels(calib_parallel_on))
#     print("Number of dead pixels: ", len(find_dead_pixels(calib_parallel_on)))
#     print(f"Dead pixels: {find_dead_pixels(calib_parallel_on)}")
#     plot_image_colormap(calib_parallel_on, title="Calibrated Parallel On", color_range=(vmin, vmax))

#     calib_parallel_off = crop_image(png_to_array(os.path.join(image_dir, "calib_parallel_off.png")),
#                                    crop_range_x, crop_range_y)
#     print(f"Dead pixels: {find_dead_pixels(calib_parallel_off)}")
#     plot_image_colormap(calib_parallel_off, title="Calibrated Parallel Off")


#     calib_cross_on = crop_image(png_to_array(os.path.join(image_dir, "calib_cross_on.png")),
#                                    crop_range_x, crop_range_y)
#     print(f"Dead pixels: {find_dead_pixels(calib_cross_on)}")
#     plot_image_colormap(calib_cross_on, title="Calibrated Cross On")

# # Find and process HV files
# hv_files = [f for f in os.listdir(image_dir) if f.startswith('hv_')]
# print(f"\nFound {len(hv_files)} HV files:")
# for hv_file in sorted(hv_files[0:2]):
#     print(f"Processing {hv_file}")
#     hv_array = png_to_array(os.path.join(image_dir, hv_file))
#     plot_image_colormap(hv_array, title=f"High Voltage Bias - {hv_file}", color_range=(vmin, vmax))

#     numerator = hv_array - calib_cross_on
#     denominator = calib_parallel_on - calib_parallel_off
#     plot_image_colormap(numerator, title=f"Numerator - {hv_file}")
#     plot_image_colormap(denominator, title=f"Denominator - {hv_file}")
#     T_array = numerator / denominator
#     # Clip transmission values to valid arcsin range
#     T_array[T_array > 1.0] = 0.99
#     T_array[T_array < -1.0] = -0.99
#     plot_image_colormap(T_array, title=f"Transmission - {hv_file}")


#     E_field = np.arcsin(T_array)
#     plot_image_colormap(E_field, title=f"Electric Field - {hv_file}")
