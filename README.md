# Pockels Image Analyzer

Pockels Image Analyzer is a Streamlit-based web application designed to analyze and visualize data from PNG images. The application is particularly useful for analyzing images related to electro-optic effects, such as those observed in Pockels cells.

## Features

- **Image Upload**: Upload PNG images for analysis.
- **Image Cropping**: Crop images to a specified range using sliders.
- **Dead Pixel Detection and Imputation**: Detect dead pixels in the image and optionally impute them.
- **Color Mapping**: Apply various color maps to the images for better visualization.
- **Transmission Calculation**: Calculate the transmission array from the uploaded data images using calibration images.
- **E-field Calculation**: Calculate the electric field from the transmission array using specified coefficients.
- **Interactive Plots**: Generate interactive plots using Plotly for detailed analysis.

## How to Use

1. **Upload Calibration Images**: Upload calibration images (e.g., `calib_parallel_on`, `calib_parallel_off`, `calib_cross_on`) using the file uploader.
2. **Set Parameters**: Use the sidebar to set various parameters such as crop range, color map, dead pixel threshold, and coefficients for calculations.
3. **Upload Data Images**: Upload data images for analysis.
4. **Analyze and Visualize**: The application will process the images, detect dead pixels, calculate transmission and electric field arrays, and generate interactive plots for visualization.

## Sidebar Parameters

- **Crop Range**: Set the crop range for the x and y axes.
- **Color Map**: Select a color map for visualization.
- **Dead Pixel Threshold**: Set the threshold for detecting dead pixels.
- **Fix Dead Pixels**: Option to impute dead pixels.
- **Coefficients**: Input values for wavelength, refractive index, thickness, and electro-optic coefficient.

## Example

1. Upload calibration images and data images.
2. Set the desired parameters in the sidebar.
3. View the processed images and plots in the main area of the application.

## References

- Cola, A., Dominici, L., & Valletta, A. (2022). Optical Writing and Electro-Optic Imaging of Reversible Space Charges in Semi-Insulating CdTe Diodes. Sensors, 22(4). https://doi.org/10.3390/s22041579
