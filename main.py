from conversion import convert_fits_to_image
from detection import *

fits_path=r"C:\Users\USER\Desktop\finalGPbegad\fits"
images_path=r"C:\Users\USER\Desktop\finalGPbegad\images"
threshed_path=r"C:\Users\USER\Desktop\finalGPbegad\threshold"

#convert_fits_to_image(fits_path,images_path)
csv_file=r"C:\Users\USER\Desktop\finalGPbegad\sim_debris.xlsx"
analyzer = DebrisAnalyzer( threshed_path, csv_file)