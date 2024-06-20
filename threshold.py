import cv2
import os
import numpy as np

def iterative_thresholding_folder(input_folder, output_folder, sub_image_size=(64, 64)):
    """
    Apply iterative thresholding to images in the input folder and save the thresholded images in the output folder.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each file in the input folder
    for filename in os.listdir(input_folder):
        # Read the image
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Skipping file {filename}: Not a valid image.")
            continue

        # Apply iterative thresholding to sub-images
        thresholded_img = apply_thresholding_to_sub_images(img, sub_image_size)

        # Write the thresholded image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, thresholded_img)

        print(f"Thresholded image saved: {output_path}")

def apply_thresholding_to_sub_images(img, sub_image_size):
    """
    Divide the image into sub-images, apply iterative thresholding to each sub-image, and combine them.
    """
    h, w = img.shape
    sub_h, sub_w = sub_image_size

    # Create an empty image to store the thresholded result
    thresholded_img = np.zeros((h, w), dtype=np.uint8)

    # Iterate over sub-images
    for i in range(0, h, sub_h):
        for j in range(0, w, sub_w):
            sub_img = img[i:i+sub_h, j:j+sub_w]
            if sub_img.size == 0:
                continue

            # Apply median filter to remove hot pixels or cosmic rays
            filtered_sub_img = cv2.medianBlur(sub_img, 3)

            # Apply iterative thresholding
            thresholded_sub_img, _ = iterative_thresholding(filtered_sub_img)

            # Place the thresholded sub-image back into the result image
            thresholded_img[i:i+sub_h, j:j+sub_w] = thresholded_sub_img

    return thresholded_img

def iterative_thresholding(img):
    """
    Apply iterative thresholding to the input image and return the thresholded image along with the guessed threshold.
    """
    # Initial guess for threshold
    threshold = 128

    # Loop until convergence
    while True:
        # Threshold the image using the current threshold
        _, thresholded_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        # Calculate the mean pixel values for foreground and background
        foreground_pixels = img[thresholded_img == 255]
        background_pixels = img[thresholded_img == 0]

        # Check if foreground or background is empty
        if foreground_pixels.size == 0 or background_pixels.size == 0:
            break

        foreground_mean = np.mean(foreground_pixels) if foreground_pixels.size > 0 else 0
        background_mean = np.mean(background_pixels) if background_pixels.size > 0 else 0

        # Update the threshold using the means of foreground and background
        new_threshold = (foreground_mean + background_mean) / 2

        # Check for convergence
        if abs(threshold - new_threshold) < 0.1:
            break

        # Update the threshold
        threshold = new_threshold

    return threshold
    # return thresholded_img, threshold

# Example usage
if __name__ == "__main__":
    input_folder = r'C:\Users\USER\Desktop\finalGPbegad\images'  # Specify the path to your input folder
    output_folder = r'C:\Users\USER\Desktop\finalGPbegad\threshold'  # Specify the path to your output folder
    iterative_thresholding_folder(input_folder, output_folder)
