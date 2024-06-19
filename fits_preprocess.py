import os
import numpy as np
from astropy.io import fits

def load_fits_data(image_path):
    with fits.open(image_path, mode='update') as hdul:
        header = hdul[0].header
        image_data = hdul[0].data
    return header, image_data

def analyze_background(image_data):
    """Perform a basic background analysis to find mean and standard deviation of the background."""
    # Assuming the image has more background than stars, use a simple statistical approach
    background_pixels = image_data.flatten()
    mean_background = np.mean(background_pixels)
    std_background = np.std(background_pixels)
    return mean_background, std_background

def update_fits_header_with_bg_info(fits_file_path, mean_bg, std_bg):
    """Update the FITS header with calculated mean and std background values."""
    with fits.open(fits_file_path, mode='update') as hdul:
        header = hdul[0].header
        header['MEAN_BG'] = mean_bg
        header['STD_BG'] = std_bg
        hdul.flush()  # Write changes to the file

def preprocess_fits_folder(folder_path):
    fits_files = [f for f in os.listdir(folder_path) if f.endswith('.fits')]
    
    for fits_filename in fits_files:
        full_path_fits = os.path.join(folder_path, fits_filename)
        header, image_data = load_fits_data(full_path_fits)
        
        mean_bg, std_bg = analyze_background(image_data)
        update_fits_header_with_bg_info(full_path_fits, mean_bg, std_bg)
        
        print(f"Updated {fits_filename} with MEAN_BG: {mean_bg} and STD_BG: {std_bg}")

# Example usage
if __name__ == "__main__":
    folder_path = r'C:\Users\USER\Desktop\finalGPbegad\fits'  # Specify the path to your folder
    preprocess_fits_folder(folder_path)
