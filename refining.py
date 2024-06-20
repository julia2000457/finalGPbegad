import numpy as np
import os
import cv2
from scipy.ndimage import label, find_objects

def process_images(input_folder, output_folder, threshold=5, sub_image_size=(64, 64)):
    """
    Process images in the input folder to detect stars and debris, and save the processed images in the output folder.
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

        # Apply the detection and labeling algorithm
        labeled_img, num_objects = detect_and_label_objects(img, threshold, sub_image_size)

        # Write the labeled image to the output folder
        output_path = os.path.join(output_folder, f"labeled_{filename}")
        cv2.imwrite(output_path, labeled_img)

        print(f"Labeled image saved: {output_path}, Number of objects detected: {num_objects}")

def detect_and_label_objects(img, threshold, sub_image_size):
    """
    Detect and label objects (stars, debris, etc.) in the input image.
    """
    h, w = img.shape
    sub_h, sub_w = sub_image_size

    # Create an empty image to store the labeled result
    labeled_img = np.zeros((h, w), dtype=np.int32)

    # Variable to keep track of the current label
    current_label = 1

    # Iterate over sub-images
    for i in range(0, h, sub_h):
        for j in range(0, w, sub_w):
            sub_img = img[i:i+sub_h, j:j+sub_w]
            if sub_img.size == 0:
                continue

            # Apply median filter to remove hot pixels or cosmic rays
            filtered_sub_img = cv2.medianBlur(sub_img, 3)

            # Threshold the sub-image
            _, binary_sub_img = cv2.threshold(filtered_sub_img, threshold, 1, cv2.THRESH_BINARY)

            # Label connected regions in the sub-image
            sub_labeled_img, num_features = label(binary_sub_img)

            # Relabel the sub-image with the global labels
            for sub_label in range(1, num_features + 1):
                sub_labeled_img[sub_labeled_img == sub_label] = current_label
                current_label += 1

            # Place the labeled sub-image back into the result image
            labeled_img[i:i+sub_h, j:j+sub_w] = sub_labeled_img

    # Perform the second step to combine close objects
    labeled_img = combine_close_objects(labeled_img)

    return labeled_img, current_label - 1

def combine_close_objects(labeled_img):
    """
    Combine close objects in the labeled image to ensure accurate grouping.
    """
    objects_slices = find_objects(labeled_img)
    num_objects = len(objects_slices)

    # Compare each object with every other object
    for i in range(num_objects):
        if objects_slices[i] is None:
            continue
        for j in range(i + 1, num_objects):
            if objects_slices[j] is None:
                continue

            # Check if the objects are close to each other
            obj1_slice = objects_slices[i]
            obj2_slice = objects_slices[j]

            if are_objects_close(obj1_slice, obj2_slice):
                # Combine the objects
                labeled_img[labeled_img == j + 1] = i + 1
                objects_slices[j] = None

    return labeled_img

def are_objects_close(obj1_slice, obj2_slice, proximity_threshold=5):
    """
    Check if two objects are close to each other based on their slices.
    """
    if obj1_slice is None or obj2_slice is None:
        return False

    for dim in range(len(obj1_slice)):
        if abs(obj1_slice[dim].start - obj2_slice[dim].stop) < proximity_threshold or \
           abs(obj1_slice[dim].stop - obj2_slice[dim].start) < proximity_threshold:
            return True
    return False

# Example usage
if __name__ == "__main__":
    input_folder = r'C:\Users\USER\Desktop\finalGPbegad\threshold'  # Specify the path to your input folder
    output_folder = r'C:\Users\USER\Desktop\finalGPbegad\refined'  # Specify the path to your output folder
    process_images(input_folder, output_folder)
