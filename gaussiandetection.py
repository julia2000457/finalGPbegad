import numpy as np
import cv2
import os
import csv
import matplotlib.pyplot as plt
from conversion import convert_fits_to_image
from scipy.optimize import curve_fit

class DebrisAnalyzer:

    def __init__(self, fits_directory, images_directory, threshed_directory, highlighted_directory, curves_directory, csv_file_path):
        self.fits_directory = fits_directory
        self.images_directory = images_directory
        self.threshed_directory = threshed_directory
        self.highlighted_directory = highlighted_directory
        self.curves_directory = curves_directory
        self.csv_file_path = csv_file_path

        # Ensure the directories exist
        for directory in [self.images_directory, self.threshed_directory, self.highlighted_directory, self.curves_directory]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    @staticmethod
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

    @staticmethod
    def refine_thresholded_image(thresholded_img):
        # Remove small noise with morphological opening
        kernel = np.ones((1, 1), np.uint8)
        refined_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_OPEN, kernel, iterations=2)

        # Close small holes in the foreground with morphological closing
        refined_img = cv2.morphologyEx(refined_img, cv2.MORPH_CLOSE, kernel, iterations=2)

        return refined_img

    @staticmethod
    def apply_mask(img, mask):
        """
        Apply the binary mask to the original image, setting the background to black.
        """
        masked_img = img.copy()
        masked_img[mask == 0] = 0
        return masked_img

    @staticmethod
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

    def save_gaussian_curve(self, bin_centers, hist, popt, fits_filename, label):
        plt.figure()
        plt.hist(bin_centers, bins=len(bin_centers), weights=hist, alpha=0.6, label='Intensity Histogram')
        plt.plot(bin_centers, self.gaussian(bin_centers, *popt), 'r--', label='Gaussian Fit')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(f'Gaussian Fit for Blob {label} in {fits_filename}')
        curve_filename = os.path.join(self.curves_directory, f"{os.path.splitext(fits_filename)[0]}_blob{label}_gaussian.png")
        plt.savefig(curve_filename)
        plt.close()

    def process_images(self):
        fitsfiles = os.listdir(self.fits_directory)
        threshold_sigma = 120  # Define an appropriate threshold_sigma based on your data

        # Open the CSV file in write mode
        with open(self.csv_file_path, 'w', newline='') as csvfile:
            # Create a CSV writer
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Image', 'Object ID', 'Area', 'Edges', 'Center_x', 'Center_y', 'Width', 'Height', 'Prediction'])

            for fits_filename in fitsfiles:
                # Convert FITS to image
                output_image_filename = os.path.join(self.images_directory, os.path.splitext(fits_filename)[0] + '.png')
                convert_fits_to_image(self.fits_directory, self.images_directory)

                # Read the image
                image = cv2.imread(output_image_filename)
                if image is None:
                    print(f"Failed to read image: {output_image_filename}")
                    continue

                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Apply iterative thresholding
                thresholded_img, threshold = self.iterative_thresholding(img_gray)

                # Refine the thresholded image
                refined_thresholded_img = self.refine_thresholded_image(thresholded_img)

                # Apply the mask to the original image
                final_img = self.apply_mask(img_gray, refined_thresholded_img)

                iter_image_filename = os.path.join(self.threshed_directory, os.path.splitext(fits_filename)[0] + '_iter.png')
                cv2.imwrite(iter_image_filename, final_img)
                print(f"Saved thresholded image: {iter_image_filename}")

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(refined_thresholded_img, connectivity=8)

                blobs_intensity_profiles = []

                for label in range(1, num_labels):  # Skip the background label 0
                    blob_mask = (labels == label)
                    blob_intensity_values = img_gray[blob_mask]  # Extract intensity values from the original grayscale image
                    blobs_intensity_profiles.append(blob_intensity_values)

                blob_classifications = []

                for label, intensity_values in enumerate(blobs_intensity_profiles, start=1):
                    hist, bin_edges = np.histogram(intensity_values, bins=10, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                    try:
                        popt, _ = curve_fit(self.gaussian, bin_centers, hist, p0=[1., np.mean(intensity_values), np.std(intensity_values)])
                        amp, mu, sigma = popt

                        # Save Gaussian curve plot
                        self.save_gaussian_curve(bin_centers, hist, popt, fits_filename, label)

                        if sigma < threshold_sigma:
                            blob_classifications.append('Star')
                        else:
                            blob_classifications.append('Not a Star')
                    except RuntimeError:
                        blob_classifications.append('Not a Star')

                object_id = 1

                for label in range(1, num_labels):
                    x, y, w, h, area = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT], stats[label, cv2.CC_STAT_AREA]
                    center_x, center_y = centroids[label]
                    prediction = blob_classifications[label - 1]

                    if prediction == 'Star':
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(image, str(object_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    csvwriter.writerow([fits_filename, object_id, area, '-', center_x, center_y, w, h, prediction])
                    object_id += 1

                output_highlighted_image_filename = os.path.join(self.highlighted_directory, os.path.splitext(fits_filename)[0] + '_highlighted.png')
                cv2.imwrite(output_highlighted_image_filename, image)
                print(f"Saved highlighted image: {output_highlighted_image_filename}")

# Example usage
if __name__ == "__main__":
    analyzer = DebrisAnalyzer(
        fits_directory=r"E:\finalGPbegad\fitsaya",
        images_directory=r"E:\finalGPbegad\images",
        threshed_directory=r"E:\finalGPbegad\iter_images",
        highlighted_directory=r"E:\finalGPbegad\Highlighted",
        curves_directory=r"E:\finalGPbegad\Curves",
        csv_file_path=r"E:\finalGPbegad\sim_debris.csv"
    )
    analyzer.process_images()
