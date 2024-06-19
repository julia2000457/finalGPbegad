import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def load_fits_data(image_path):
    with fits.open(image_path) as hdul:
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

def check_image_integrity(header, image_data):
    """Check the correspondence between header information and image characteristics."""
    # Extract expected mean and standard deviation from header
    expected_mean = header.get('MEAN_BG', None)
    expected_std = header.get('STD_BG', None)
    
    if expected_mean is None or expected_std is None:
        print("Header does not contain expected mean or std background values.")
        return False

    # Perform background analysis on the image data
    mean_bg, std_bg = analyze_background(image_data)

    # Compare the results
    mean_diff = abs(mean_bg - float(expected_mean))
    std_diff = abs(std_bg - float(expected_std))
    
    print(f"Mean Background (Header): {expected_mean}")
    print(f"Mean Background (Image): {mean_bg}")
    print(f"Standard Deviation Background (Header): {expected_std}")
    print(f"Standard Deviation Background (Image): {std_bg}")
    
    # Define acceptable thresholds
    mean_threshold = 0.1 * float(expected_mean)  # 10% of the expected mean
    std_threshold = 0.1 * float(expected_std)    # 10% of the expected std

    if mean_diff < mean_threshold and std_diff < std_threshold:
        print("Image integrity check passed.")
        return True
    else:
        print("Image integrity check failed.")
        return False

def plot_image(image_data, title="FITS Image"):
    plt.figure(figsize=(8, 8))
    plt.imshow(image_data, cmap='gray')
    plt.colorbar()
    plt.title(title)
    plt.show()

def analyze_fits_folder(folder_path):
    fits_files = [f for f in os.listdir(folder_path) if f.endswith('.fits')]
    
    for fits_filename in fits_files:
        full_path_fits = os.path.join(folder_path, fits_filename)
        header, image_data = load_fits_data(full_path_fits)
        
        print(f"Analyzing file: {fits_filename}")
        if check_image_integrity(header, image_data):
            print(f"The image {fits_filename} is consistent with its header information.")
        else:
            print(f"The image {fits_filename} is NOT consistent with its header information.")
        
        # Optionally, plot the image for visual inspection
        plot_image(image_data, title=f"FITS Image: {fits_filename}")

# Example usage
if __name__ == "__main__":
    folder_path = r'C:\Users\USER\Desktop\finalGPbegad\fits'  # Specify the path to your folder
    analyze_fits_folder(folder_path)
