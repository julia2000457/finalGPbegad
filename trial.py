import numpy as np
import cv2
import os
import csv
import matplotlib.pyplot as plt
from conversion import convert_fits_to_image
from scipy.optimize import curve_fit

def iterative_thresholding(img):
    """
    Apply iterative thresholding to the input image and return the thresholded image along with the guessed threshold.
    """
    # Initial guess for threshold
    threshold = 120

    # Loop until convergence
    while True:
        # Threshold the image using the current threshold
        _, thresholded_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        # Calculate the mean pixel values for foreground and background
        foreground_mean = np.mean(img[thresholded_img == 255])
        background_mean = np.mean(img[thresholded_img == 0])

        # Update the threshold using Otsu's method
        new_threshold = (foreground_mean + background_mean) / 2

        # Check for convergence
        if abs(threshold - new_threshold) < 0.5:
            break

        # Update the threshold
        threshold = new_threshold

    return thresholded_img, threshold


def refine_thresholded_image(thresholded_img):
    # Remove small noise with morphological opening
    kernel = np.ones((1, 1), np.uint8)
    refined_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_OPEN, kernel, iterations=2)

    # Close small holes in the foreground with morphological closing
    refined_img = cv2.morphologyEx(refined_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    return refined_img


def apply_mask(img, mask):
    """
    Apply the binary mask to the original image, setting the background to black.
    """
    masked_img = img.copy()
    masked_img[mask == 0] = 0
    return masked_img

if __name__ == "__main__":
    # Example image file path
    image_path = r"E:\finalGPbegad\images\NEOS_SCI_2024001000555.png"
    
    # Read the image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    
    if img is None:
        print(f"Failed to read image: {image_path}")
    else:
        # Apply iterative thresholding
        thresholded_img, threshold = iterative_thresholding(img)
        re = refine_thresholded_image(thresholded_img)
        ree = apply_mask(thresholded_img, refine_thresholded_image)

       
        # Example: Save or further process thresholded_img
        output_path = r"E:\finalGPbegad\images\NEOS_SCI_2024001000555_thresholded.png"
        cv2.imwrite(output_path, ree)
        print(f"Saved thresholded image: {output_path}")