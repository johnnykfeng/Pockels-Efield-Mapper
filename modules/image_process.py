import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import cv2
from scipy.signal import find_peaks


def png_to_array(image_path, dtype=np.float32):
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
    img_array = img_array.astype(dtype)
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

def find_bad_pixels(img_array, lower_threshold=101, upper_threshold=20e3):
    """
    Find bad pixels in an image array.
    """
    dim_pixels = np.where(img_array < lower_threshold)
    bright_pixels = np.where(img_array > upper_threshold)
    bad_pixels = np.concatenate((dim_pixels, bright_pixels), axis=1)
    bad_pixel_coords = list(zip(bad_pixels[1], bad_pixels[0]))
    return bad_pixel_coords


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
    # Convert to float to handle negative values
    img_array = img_array.astype(float)
    img_array[(img_array > 0) & (img_array < 1.0)] = 1.0
    img_array[(img_array < 0) & (img_array > -1.0)] = -1.0
    img_array[img_array == 0] = 1.0
    return img_array

def canny_edge_method(image, threshold1=100, threshold2=300):
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    # find the edges of the sensor
    edges = cv2.Canny(image, threshold1, threshold2)
    return edges


def find_horizontal_edges(image, edge_threshold1=100, edge_threshold2=300, mean_threshold=1.0):
    """
    Find the most likely horizontal edges of a bright rectangle in an image.
    
    Args:
        image: Input image array
        threshold1: Lower threshold for Canny edge detection
        threshold2: Upper threshold for Canny edge detection
    
    Returns:
        top_edge: Row index of the top edge
        bottom_edge: Row index of the bottom edge
    """
    # Get edges using existing function
    edges = canny_edge_method(image, edge_threshold1, edge_threshold2)
    
    # Sum edges horizontally to find rows with strong horizontal edges
    horizontal_edge_strength = np.sum(edges, axis=1)

    min_separation = 50
    
    # Find peaks in the horizontal edge strength
    peaks = {}
    for i, value in enumerate(horizontal_edge_strength):
        if (value > horizontal_edge_strength[i-1] and 
            value > horizontal_edge_strength[i+1] and
            value > np.mean(horizontal_edge_strength) * mean_threshold):
            peaks[i] = value
    
    filtered_peaks = {} # Remove peaks that are too close to each other
    for peak in peaks.keys():
        if not any(abs(peak - existing) < min_separation for existing in filtered_peaks):
            filtered_peaks[peak] = peaks[peak]
    
    # Find the 2 largest peak values
    sorted_peaks = dict(sorted(filtered_peaks.items(), key=lambda x: x[1], reverse=True))
    top_peaks = {k: sorted_peaks[k] for k in list(sorted_peaks.keys())[:2]}

    if len(top_peaks) < 2:
        raise ValueError("Not enough peaks found")
    
    top_edge = min(top_peaks.keys())
    bottom_edge = max(top_peaks.keys())
    
    return top_edge, bottom_edge, horizontal_edge_strength

def find_horizontal_edges_robust(image, edge_threshold1=100, edge_threshold2=300, mean_threshold=1.0):
    """
    DON'T USE THIS FUNCTION
    """
    # Get edges
    edges = canny_edge_method(image, edge_threshold1, edge_threshold2)
    
    # Method 1: Sum edges horizontally
    horizontal_edge_strength = np.sum(edges, axis=1)
    
    # Method 2: Find where intensity changes significantly (gradient)
    if image.dtype != np.uint8:
        image_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    else:
        image_uint8 = image
    
    # Calculate gradient magnitude
    grad_y = cv2.Sobel(image_uint8, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.abs(grad_y)
    gradient_strength = np.sum(grad_magnitude, axis=1)
    
    # Combine both methods
    combined_strength = horizontal_edge_strength * gradient_strength
    
    # Find peaks with minimum separation
    min_separation = 10  # Minimum pixels between edges
    peaks = []
    
    # Find local maxima
    for i in range(1, len(combined_strength) - 1):
        if (combined_strength[i] > combined_strength[i-1] and 
            combined_strength[i] > combined_strength[i+1] and
            combined_strength[i] > np.mean(combined_strength) * mean_threshold):
            peaks.append(i)
    
    # Filter peaks to ensure minimum separation
    filtered_peaks = []
    for peak in peaks:
        if not any(abs(peak - existing) < min_separation for existing in filtered_peaks):
            filtered_peaks.append(peak)
    
    if len(filtered_peaks) < 2:
        # Fallback: use the two highest values
        sorted_indices = np.argsort(combined_strength)[::-1]
        filtered_peaks = sorted_indices[:2]
        filtered_peaks.sort()
    
    # Return top and bottom edges
    top_edge = filtered_peaks[0]
    bottom_edge = filtered_peaks[-1]
    
    return top_edge, bottom_edge, combined_strength

def find_vertical_edges(image, edge_threshold1=100, edge_threshold2=300, mean_threshold=5.0):
    """
    Find the most likely vertical edges of a bright rectangle in an image.
    
    Args:
        image: Input image array
        threshold1: Lower threshold for Canny edge detection
        threshold2: Upper threshold for Canny edge detection
    
    Returns:
        left_edge: Column index of the left edge
        right_edge: Column index of the right edge
    """
    # Get edges using existing function
    edges = canny_edge_method(image, edge_threshold1, edge_threshold2)
    # Sum edges vertically to find columns with strong vertical edges
    vertical_edge_strength = np.sum(edges, axis=0).astype(np.float32)
    # Find peaks in the vertical edge strength
    peaks = {}
    for idx, value in enumerate(vertical_edge_strength):
        if (value > vertical_edge_strength[idx-1] and 
            value > vertical_edge_strength[idx+1] and
            value > np.mean(vertical_edge_strength) * mean_threshold):
            peaks[idx] = value

    min_separation = 200
    filtered_peaks = {}
    for peak in peaks.keys():
        if not any(abs(peak - existing) < min_separation for existing in filtered_peaks):
            filtered_peaks[peak] = peaks[peak]

    # Find the 2 largest peak values (top peaks)
    sorted_peaks = dict(sorted(filtered_peaks.items(), key=lambda x: x[1], reverse=True))
    top_peaks = {k: sorted_peaks[k] for k in list(sorted_peaks.keys())[:2]}

    if len(top_peaks) < 2:
        raise ValueError("Not enough peaks found")
    
    # Return the left and right edges
    left_edge = min(top_peaks.keys())
    right_edge = max(top_peaks.keys())
    
    return left_edge, right_edge, vertical_edge_strength


def find_sensor_edges(image_path, 
                      edge_threshold1=50, 
                      edge_threshold2=100, 
                      mean_threshold=6.0,
                      plot=True):
    if isinstance(image_path, str):
        image = png_to_array(image_path)
    elif isinstance(image_path, np.ndarray):
        image = image_path
    else:
        raise ValueError("Image path must be a string or a numpy array")
    
    edges = canny_edge_method(image, threshold1=edge_threshold1, threshold2=edge_threshold2)
    # Find both horizontal and vertical edges
    top_edge, bottom_edge, horizontal_edge_strength = find_horizontal_edges(image, 
                                                  edge_threshold1=edge_threshold1, 
                                                  edge_threshold2=edge_threshold2, 
                                                  mean_threshold=mean_threshold)
    left_edge, right_edge, vertical_edge_strength = find_vertical_edges(image, 
                                                edge_threshold1=edge_threshold1, 
                                                edge_threshold2=edge_threshold2, 
                                                mean_threshold=mean_threshold)

    if plot:
        fig, axs = plt.subplots(3, 2, figsize=(12, 8))
        
        axs[0, 0].imshow(image, cmap='gray')
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axhline(y=top_edge, color='red', linestyle='--', alpha=0.7)
        axs[0, 0].axhline(y=bottom_edge, color='red', linestyle='--', alpha=0.7)
        axs[0, 0].axvline(x=left_edge, color='blue', linestyle='--', alpha=0.7)
        axs[0, 0].axvline(x=right_edge, color='blue', linestyle='--', alpha=0.7)

        axs[0, 1].imshow(edges, cmap='gray')
        axs[0, 1].set_title('Canny Edge Method')

        axs[1, 0].plot(np.sum(edges, axis=1), color='green', alpha=0.7)
        axs[1, 0].set_title('Horizontal Edge Strength')
        axs[1, 0].grid(True, linestyle='--', alpha=0.5)
        axs[1, 0].axvline(x=top_edge, color='red', linestyle='--', alpha=0.7)
        axs[1, 0].axvline(x=bottom_edge, color='red', linestyle='--', alpha=0.7)
        axs[1, 0].axhline(y=np.mean(horizontal_edge_strength)*mean_threshold, color='black', linestyle='--', alpha=0.7)
        axs[1, 0].set_xlabel('Row Index')
        axs[1, 0].set_ylabel('Edge Strength')

        axs[1, 1].plot(np.sum(edges, axis=0), color='green', alpha=0.7)
        axs[1, 1].set_title('Vertical Edge Strength')
        axs[1, 1].grid(True, linestyle='--', alpha=0.5)
        axs[1, 1].axvline(x=left_edge, color='blue', linestyle='--', alpha=0.7)
        axs[1, 1].axvline(x=right_edge, color='blue', linestyle='--', alpha=0.7)
        axs[1, 1].axhline(y=np.mean(vertical_edge_strength)*mean_threshold, color='black', linestyle='--', alpha=0.7)
        axs[1, 1].set_xlabel('Column Index')
        axs[1, 1].set_ylabel('Edge Strength')

        axs[2, 0].imshow(edges, cmap='gray')
        axs[2, 0].set_title('Canny Edges and Detected Edges')
        axs[2, 0].axhline(y=top_edge, color='red', linestyle='--', alpha=0.5)
        axs[2, 0].axhline(y=bottom_edge, color='red', linestyle='--', alpha=0.5)  
        axs[2, 0].axvline(x=left_edge, color='blue', linestyle='--', alpha=0.5)
        axs[2, 0].axvline(x=right_edge, color='blue', linestyle='--', alpha=0.5)

        axs[2, 1].imshow(image, cmap='gray')
        axs[2, 1].set_title('Detected Rectangle')
        # Draw rectangle around detected edges
        rect = plt.Rectangle((left_edge, top_edge), 
                            right_edge - left_edge, 
                            bottom_edge - top_edge, 
                            fill=False, color='green', linewidth=2, alpha=0.7)
        axs[2, 1].add_patch(rect)
    else:
        fig = None

    return top_edge, bottom_edge, left_edge, right_edge, fig

if __name__ == "__main__":

    # Load the image in grayscale
    image_path = r'R:\Pockels_data\NEXT GEN POCKELS\Photo-Pockels_D420222_2025-06-20\calib_parallel_on.png'

    edge_threshold1 = 50
    edge_threshold2 = 100
    mean_threshold = 6.0

    top_edge, bottom_edge, left_edge, right_edge, fig = find_sensor_edges(image_path, 
                                                                          edge_threshold1=edge_threshold1, 
                                                                          edge_threshold2=edge_threshold2, 
                                                                          mean_threshold=mean_threshold,
                                                                          plot=True)

    plt.tight_layout()
    plt.show()
