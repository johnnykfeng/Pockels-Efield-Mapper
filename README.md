# Pockels Electric Field Mapper

A Python-based application for analyzing electric field distributions in semiconductor detectors using the Pockels electro-optic effect. This tool processes polarized light images to compute 2D electric field maps in CdZnTe (CZT) and related semiconductor materials.

## Overview

The Pockels effect describes the linear electro-optic response of certain materials where an applied electric field induces birefringence. This application leverages this phenomenon to non-destructively map electric field distributions in semiconductor X-ray detectors by analyzing the transmission of polarized light through the material.

### Key Scientific Principle

The electric field is calculated from transmission images using:
- **Alpha coefficient**: `α = √3 π n₀³ d r₄₁ / (2λ)`
- **Reference field**: `E_ref = λ / (√3 n₀³ d r₄₁)`
- **Field calculation**: `E = √(arcsin(T)) / α`

Where:
- `n₀`: refractive index
- `d`: path length (mm)
- `r₄₁`: electro-optic coefficient (m/V)
- `λ`: wavelength (nm)
- `T`: transmission ratio

## Features

### Core Functionality
- **Multi-configuration support**: Pre-configured settings for XMED and CZT-10mm detectors
- **Image processing pipeline**: Automated cropping, bad pixel detection/correction, and noise filtering
- **Electric field computation**: Real-time calculation from calibration and bias images
- **Interactive visualization**: Plotly-based heatmaps with customizable color scales and ranges
- **Data export**: Multiple format support (PNG, CSV, PDF reports)

### Analysis Capabilities
- **2D Electric field mapping**: High-resolution spatial field distribution
- **Statistical analysis**: Comprehensive image statistics and histograms
- **Bad pixel management**: Automatic detection and interpolation of dead/hot pixels
- **Boundary box analysis**: Focused analysis on detector active regions
- **Multi-bias analysis**: Process voltage series from 0V to 1100V+

### User Interface
- **Streamlit web application**: Intuitive browser-based interface
- **Real-time parameter adjustment**: Interactive sliders for all analysis parameters
- **Live preview**: Instant visualization updates with parameter changes
- **Sample data included**: Built-in test dataset for immediate use

## Installation

### Prerequisites
- Python 3.11 or higher
- Git (for cloning the repository)

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/Pockels-Efield-Mapper.git
cd Pockels-Efield-Mapper
```

2. **Install dependencies:**
```bash
# Using pip
pip install -r requirements.txt

# Or using uv (recommended)
uv sync
```

3. **Run the application:**
```bash
streamlit run app.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

### Dependencies

Core packages:
- **streamlit**: Web application framework
- **numpy**: Numerical computations
- **matplotlib**: Static plotting
- **plotly**: Interactive visualization
- **pandas**: Data manipulation
- **pillow**: Image processing

## Usage

### Getting Started

1. **Launch the application** using `streamlit run app.py`
2. **Select detector configuration** (XMED or CZT-10mm)
3. **Choose data source**:
   - **Sample Data**: Use included test dataset
   - **Data Uploader**: Upload your own PNG images

### Data Requirements

#### Calibration Images (Required)
- `calib_parallel_on.png`: Parallel polarizers (maximum transmission)
- `calib_parallel_off.png`: Parallel polarizers blocked
- `calib_cross_on.png`: Crossed polarizers (minimum transmission)

#### Bias Images
- Series of images at different bias voltages
- Naming convention: `bias_[voltage]V_xray_0mA.png`
- Example: `bias_100V_xray_0mA.png`, `bias_500V_xray_0mA.png`

### Analysis Workflow

1. **Configure Parameters**:
   - Set wavelength, refractive index, path length
   - Adjust electro-optic coefficient (r₄₁)
   - Define image cropping and boundary regions

2. **Process Calibration Images**:
   - Review transmission ratios
   - Check for bad pixels
   - Validate calibration quality

3. **Analyze Bias Images**:
   - Compute electric field maps
   - Generate comparative visualizations
   - Export results in multiple formats

4. **Generate Reports**:
   - PDF compilation of all results
   - Individual plot exports
   - CSV data tables

### Configuration Files

The application uses TOML configuration files for different detector setups:

#### `config/CZT_10mm.toml`
```toml
path_length = 10  # mm
wavelength = 1550  # nm
n0 = 2.8  # refractive index
r41 = 5.5  # 1e-12 m/V electro-optic coefficient
# Cropping and boundary parameters...
```

#### `config/XMED.toml`
Similar structure optimized for XMED detector geometry.

## Project Structure

```
Pockels-Efield-Mapper/
├── app.py                      # Main Streamlit application
├── compute_efield_page.py      # E-field computation interface
├── process_image_page.py       # Image cropping/processing tools
├── processed_efields_page.py   # Results visualization
├── modules/
│   ├── pockels_math.py        # Core electro-optic calculations
│   ├── image_process.py       # Image processing utilities
│   └── plotting_modules.py    # Visualization functions
├── config/
│   ├── CZT_10mm.toml         # 10mm CZT detector configuration
│   ├── XMED.toml             # XMED detector configuration
│   └── pockels_parameters.toml # Literature parameter values
├── SAMPLE_DATA/               # Test dataset
└── requirements.txt           # Python dependencies
```

## Scientific Background

This application implements methods described in:

> Cola, A., Dominici, L., & Valletta, A. (2022). Optical Writing and Electro-Optic Imaging of Reversible Space Charges in Semi-Insulating CdTe Diodes. *Sensors*, 22(4). https://doi.org/10.3390/s22041579

### Materials Supported
- **CdZnTe (CZT)**: Cadmium Zinc Telluride detectors
- **CdTe**: Cadmium Telluride crystals
- **CdZnTeSe**: Quaternary semiconductor compounds

### Typical Applications
- **X-ray detector characterization**: Mapping charge collection efficiency
- **Space charge analysis**: Visualizing trapped charge distributions
- **Electric field uniformity**: Quality control for detector manufacturing
- **Bias optimization**: Determining optimal operating voltages

## Advanced Features

### Bad Pixel Correction
- Automatic detection of dead (< threshold) and hot (> threshold) pixels
- Interpolation using surrounding pixel values
- Configurable thresholds for different detector types

### Statistical Analysis
- Comprehensive image statistics (mean, std, min, max)
- Histogram analysis with configurable binning
- Percentile-based color scaling for consistent visualization

### Export Options
- **PNG**: High-resolution images with custom colormaps
- **CSV**: Raw numerical data for external analysis
- **PDF**: Multi-page reports with all analysis results

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed with correct versions
2. **Sample data not found**: Check that SAMPLE_DATA directory exists
3. **Memory issues**: Large images may require cropping for processing
4. **Color scaling**: Adjust percentile ranges if images appear over/under-saturated

### Performance Optimization
- Use image cropping to reduce processing time
- Enable bad pixel correction only when necessary
- Batch process multiple bias voltages for efficiency

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

### Development Setup
```bash
# Clone your fork
git clone https://github.com/your-username/Pockels-Efield-Mapper.git

# Install development dependencies
uv sync --dev

# Run tests (if available)
pytest

# Start development server
streamlit run app.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Scientific methodology based on work by Cola et al. (2022)
- CZT detector parameters from various literature sources
- Sample data courtesy of [Institution/Lab Name]

## Citation

If you use this software in your research, please cite:

```bibtex
@software{pockels_efield_mapper,
  title={Pockels Electric Field Mapper},
  author={[Your Name]},
  year={2025},
  url={https://github.com/your-username/Pockels-Efield-Mapper}
}
```

## Contact

For questions, issues, or collaboration opportunities:
- GitHub Issues: [Project Issues](https://github.com/your-username/Pockels-Efield-Mapper/issues)
- Email: [your.email@institution.edu]

---

**Keywords**: Pockels effect, electro-optic imaging, semiconductor detectors, CZT, electric field mapping, X-ray detectors, polarized light imaging
