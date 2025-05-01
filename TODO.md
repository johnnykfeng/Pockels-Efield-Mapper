- Save data in all file formats, .png, .bin, .csv

- Do full Pockels measurement for 3 sensors
   - use full camera image size
   - 50 V steps

- Calculate 2D space-charge density from E-field map
   - First calculate the Ey field using the method from paper
   - Then use the method from paper to calculate the space-charge density
   - Read Prague group's thesis as well


- calculate average over columns and rows of E-field image

- Make a better workflow
   1. Crop the raw images
   2. Separate all sets of images and data in appropriate subfolders
   3. 

2025-04-30
- Apply a metadata organization...
- Create function that extracts metadata from filename
- def metadata_from_filename(filename:str):
   ...
   return Bias, XrayFlux, LEDFlux, ...
- Efield_data = {Bias: [0,... 1100V], XrayFlux: [0], LEDFlux: [0, .., n], Efield_array: []}
- Create an ALL_DATA file, either in hdf or just plain dictionary
   - Calibration -> parallel_on, parallel_off, cross_on
   - Bias, XrayFlux, LEDflux  -> Raw image, Numerator, Denominator, Transmission, Efield


* END GAME:
   - Create a one page static PDF report or HTML dashboard showing all the results
   - Include all the E-field figures
   - The Ez-field profile map and space charge plots