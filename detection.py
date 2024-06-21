import os
import cv2
import numpy as np
import csv
from skimage import feature
from conversion import convert_fits_to_image  # Ensure these modules are available
from threshold import iterative_thresholding

class DebrisAnalyzer:

    def __init__(self, fits_directory, images_directory, threshed_directory, highlighted_directory, csv_file_path):
        self.fits_directory = fits_directory
        self.images_directory = images_directory
        self.threshed_directory = threshed_directory
        self.highlighted_directory = highlighted_directory
        self.csv_file_path = csv_file_path

        # Ensure the highlighted directory exists
        if not os.path.exists(self.highlighted_directory):
            os.makedirs(self.highlighted_directory)

    def moment_of_inertia(self, xWidth, yHeight, xCG, yCG):
        Ixx = sum((y - yCG)**2 for y in yHeight)
        Iyy = sum((x - xCG)**2 for x in xWidth)
        Ixy = sum((x - xCG)*(y - yCG) for x, y in zip(xWidth, yHeight))
        return Ixx, Iyy, Ixy

    def main_inertia(self, Ixx, Iyy, Ixy, yHeight, xWidth):
        Imain1 = 0.5 * (Ixx + Iyy + np.sqrt((Ixx - Iyy)**2 + 4*(Ixy)**2))
        Imain2 = 0.5 * (Ixx + Iyy - np.sqrt((Ixx - Iyy)**2 + 4*(Ixy)**2))
        epsilonn = 30
        final_inertia = Imain1 / Imain2
        if final_inertia > epsilonn:
            return 'Debris'
        else:
            return 'Celestial Object'

    def process_images(self):
        fitsfiles = os.listdir(self.threshed_directory)

        # Open the CSV file in write mode
        with open(self.csv_file_path, 'w', newline='') as csvfile:
            # Create a CSV writer
            csvwriter = csv.writer(csvfile)

            # Write the header row
            csvwriter.writerow(['Image', 'Object ID', 'Area', 'Edges', 'Center_x', 'Center_y', 'Width', 'Height', 'lbp_mean', 'lbp_std', 'Prediction'])

            for fits_filename in fitsfiles:
                # Full path to the FITS file
                # full_path_fits = os.path.join(self.fits_directory, fits_filename)

                # Output PNG filename (assuming the same name with a different extension)
                # Construct full paths for input FITS file and output image file
                fits_file_path = os.path.join(self.fits_directory, fits_filename)
                output_image_filename = os.path.join(self.images_directory, os.path.splitext(fits_filename)[0] + '.png')
                convert_fits_to_image(self.fits_directory, self.images_directory)  # Corrected function call

                image = cv2.imread(output_image_filename)
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Apply the iterative thresholding algorithm to the image
                thresholded_img, optimal_threshold = iterative_thresholding(image)

                # Threshold the image using the optimal threshold
                thresholded_img = (img_gray >= optimal_threshold).astype(np.uint8) * 255

                # Perform connected component analysis
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_img, connectivity=8)

                # Reset object_id for each new image
                object_id = 1

                for label in range(1, num_labels):
                    area = stats[label, cv2.CC_STAT_AREA]
                    x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
                    center_x, center_y = centroids[label]

                    # Extract the region of interest (ROI)
                    roi = img_gray[y:y+h, x:x+w]

                    # Compute Local Binary Pattern (LBP) features
                    lbp_features = feature.local_binary_pattern(roi, P=8, R=1, method='uniform')
                    lbp_mean = np.mean(lbp_features)
                    lbp_std = np.std(lbp_features)

                    # Calculate number of edges using Canny edge detection within the component
                    edges = cv2.Canny(roi, 30, 100)
                    edge_count = np.count_nonzero(edges)

                    # Calculate moment of inertia
                    xWidth = list(range(w))
                    yHeight = list(range(h))
                    Ixx, Iyy, Ixy = self.moment_of_inertia(xWidth, yHeight, center_x - x, center_y - y)
                    prediction = self.main_inertia(Ixx, Iyy, Ixy, yHeight, xWidth)

                    # Print the coordinates of the bounding box
                    print(f"Object {object_id} in {fits_filename}: Bounding Box (x, y, w, h) = ({x}, {y}, {w}, {h})")

                    # Highlight celestial objects in the output image and write the row to the CSV file
                    if prediction == 'Celestial Object':
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue bounding box for celestial objects
                        cv2.putText(image, str(object_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # Write the row to the CSV file
                        csvwriter.writerow([fits_filename, object_id, area, edge_count, center_x, center_y, w, h, lbp_mean, lbp_std, prediction])

                    # Increment Object ID
                    object_id += 1

                # Save the output image with highlighted celestial objects in the new directory
                output_highlighted_image_filename = os.path.join(self.highlighted_directory, os.path.splitext(fits_filename)[0] + '_highlighted.png')
                cv2.imwrite(output_highlighted_image_filename, image)

# Example usage
if __name__ == "__main__":
    # Instantiate DebrisAnalyzer 
    analyzer = DebrisAnalyzer(
        fits_directory=r"E:\finalGPbegad\fits",
        images_directory=r"E:\finalGPbegad\images",
        threshed_directory=r"C:\Users\Aziz\Desktop\julia\Space-Debris-Project-1\OOP\Detection\images_Preprocessing\iter_images", 
        highlighted_directory=r"E:\finalGPbegad\Highlighted",
        csv_file_path=r"E:\finalGPbegad\sim_debris.csv"
    )
    # Debugging: Print out the paths
    print("Threshed directory:", analyzer.threshed_directory)
    print("Highlighted directory:", analyzer.highlighted_directory)
    # Process images
    analyzer.process_images()


# import os
# import cv2
# import numpy as np
# import csv
# from skimage import feature
# from conversion import convert_fits_to_image  # Ensure these modules are available
# from threshold import iterative_thresholding

# class DebrisAnalyzer:
#     def __init__(self, fits_directory, images_directory, threshed_directory, highlighted_directory, csv_file_path):
#         self.fits_directory = fits_directory
#         self.images_directory = images_directory
#         self.threshed_directory = threshed_directory
#         self.highlighted_directory = highlighted_directory
#         self.csv_file_path = csv_file_path

#         # Ensure the directories exist
#         for directory in [self.images_directory, self.threshed_directory, self.highlighted_directory]:
#             if not os.path.exists(directory):
#                 os.makedirs(directory)

#     def moment_of_inertia(self, xWidth, yHeight, xCG, yCG):
#         Ixx = sum((y - yCG)**2 for y in yHeight)
#         Iyy = sum((x - xCG)**2 for x in xWidth)
#         Ixy = sum((x - xCG)*(y - yCG) for x, y in zip(xWidth, yHeight))
#         return Ixx, Iyy, Ixy

#     def main_inertia(self, Ixx, Iyy, Ixy, yHeight, xWidth):
#         Imain1 = 0.5 * (Ixx + Iyy + np.sqrt((Ixx - Iyy)**2 + 4*(Ixy)**2))
#         Imain2 = 0.5 * (Ixx + Iyy - np.sqrt((Ixx - Iyy)**2 + 4*(Ixy)**2))
#         epsilonn = 20
#         final_inertia = Imain1 / Imain2
#         if final_inertia > epsilonn:
#             return 'Debris'
#         else:
#             return 'Celestial Object'

#     def process_images(self):
#         fitsfiles = os.listdir(self.fits_directory)

#         # Open the CSV file in write mode
#         with open(self.csv_file_path, 'w', newline='') as csvfile:
#             # Create a CSV writer
#             csvwriter = csv.writer(csvfile)

#             # Write the header row
#             csvwriter.writerow(['Image', 'Object ID', 'Area', 'Edges', 'Center_x', 'Center_y', 'Width', 'Height', 'lbp_mean', 'lbp_std', 'Prediction'])

#             for fits_filename in fitsfiles:
#                 # Full path to the FITS file
#                 full_path_fits = os.path.join(self.fits_directory, fits_filename)

#                 # Convert FITS to image
#                 convert_fits_to_image(self.fits_directory, self.images_directory)

#                 # Output PNG filename (assuming the same name with a different extension)
#                 output_image_filename = os.path.join(self.images_directory, os.path.splitext(fits_filename)[0] + '.png')

#                 # Read the converted image
#                 image = cv2.imread(output_image_filename)
#                 if image is None:
#                     print(f"Error: Could not read image {output_image_filename}")
#                     continue

#                 img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#                 # Apply the iterative thresholding algorithm to the image
#                 thresholded_img, optimal_threshold = iterative_thresholding(img)

#                 # Threshold the image using the optimal threshold
#                 thresholded_img = (img >= optimal_threshold).astype(np.uint8) * 255

#                 # Optional: Clean up the thresholded image using morphological operations
#                 kernel = np.ones((3, 3), np.uint8)
#                 thresholded_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_OPEN, kernel)
#                 thresholded_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_CLOSE, kernel)

#                 # Save the thresholded image
#                 thresholded_image_filename = os.path.join(self.threshed_directory, os.path.splitext(fits_filename)[0] + '_threshed.png')
#                 cv2.imwrite(thresholded_image_filename, thresholded_img)

#                 num_labels_iterative, labels_iterative, stats_iterative, centroids_iterative = cv2.connectedComponentsWithStats(
#                     thresholded_img, connectivity=8)

#                 # Reset object_id for each new image
#                 object_id = 1

#                 for label in range(1, num_labels_iterative):
#                     area_iterative = stats_iterative[label, cv2.CC_STAT_AREA]
#                     component_mask = (labels_iterative == label).astype(np.uint8)
#                     center_x, center_y = centroids_iterative[label]
#                     # Multiply the component mask with the edges to get edges within the component
#                     edges = cv2.Canny(img, 30, 100)
#                     edges_in_component = cv2.bitwise_and(edges, edges, mask=component_mask)

#                     # Get the coordinates of the bounding box for the current object
#                     x, y, w, h, area = stats_iterative[label]

#                     # Count the number of edges in the component
#                     edge_count = np.count_nonzero(edges_in_component)

#                     # Extract the region of interest (ROI)
#                     roi = img[y:min(y+h, img.shape[0]), x:min(x+w, img.shape[1])]

#                     # Compute Local Binary Pattern (LBP) features
#                     lbp_features = feature.local_binary_pattern(roi, P=8, R=1, method='uniform')
#                     lbp_mean = np.mean(lbp_features)
#                     lbp_std = np.std(lbp_features)

#                     # Ensure xWidth and yHeight are iterable (lists)
#                     xWidth = list(range(w))
#                     yHeight = list(range(h))

#                     Ixx, Iyy, Ixy = self.moment_of_inertia(xWidth, yHeight, center_x, center_y)
#                     prediction = self.main_inertia(Ixx, Iyy, Ixy, yHeight, xWidth)

#                     # Highlight celestial objects in the output image and write the row to the CSV file
#                     if prediction == 'Celestial Object':
#                         cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue bounding box for celestial objects
#                         cv2.putText(image, str(object_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                         # Write the row to the CSV file
#                         csvwriter.writerow([fits_filename, object_id, area_iterative, edge_count, center_x, center_y, w, h, lbp_mean, lbp_std, prediction])
                    
#                     # Increment Id
#                     object_id += 1

#                 # Save the output image with highlighted celestial objects in the new directory
#                 output_highlighted_image_filename = os.path.join(self.highlighted_directory, os.path.splitext(fits_filename)[0] + '_highlighted.png')
#                 cv2.imwrite(output_highlighted_image_filename, image)

# # Example usage
# if __name__ == "__main__":
#     # Instantiate DebrisAnalyzer
#     analyzer = DebrisAnalyzer(
#         fits_directory=r"E:\finalGPbegad\fits",
#         images_directory= r"E:\finalGPbegad\images",
#         threshed_directory=r"E:\finalGPbegad\iter_images", 
#         highlighted_directory=r"E:\finalGPbegad\Highlighted",
#         csv_file_path=r"E:\finalGPbegad\sim_debris.csv"
#     )
#     # Debugging: Print out the paths
#     print("FITS directory:", analyzer.fits_directory)
#     print("Images directory:", analyzer.images_directory)
#     print("Threshed directory:", analyzer.threshed_directory)
#     print("Highlighted directory:", analyzer.highlighted_directory)
#     # Process images
#     analyzer.process_images()
