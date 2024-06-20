import numpy as np
from scipy.ndimage import gaussian_filter, label
from astropy.io import fits
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
import requests
import time
from skimage.feature import peak_local_max


import requests
import json
R = requests.post('http://nova.astrometry.net/api/login', data={'request-json': json.dumps({"apikey": "yaumjfujcxtspqvo"})})
print(R.text)

import numpy as np
from scipy.ndimage import gaussian_filter, label
from astropy.io import fits
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
import requests
import time
from skimage.feature import peak_local_max
import json

def detect_stars(image_data, threshold=5, min_distance=5):
    """Detect stars in the image data using Gaussian filter and peak local maxima."""
    # Apply a Gaussian filter to smooth the image
    smoothed_data = gaussian_filter(image_data, sigma=2)
    
    # Find stars using peak local maxima
    coordinates = peak_local_max(smoothed_data, threshold_abs=threshold, min_distance=min_distance)
    
    # Create a binary image with detected stars
    star_mask = np.zeros_like(smoothed_data, dtype=bool)
    star_mask[tuple(coordinates.T)] = True
    
    # Label connected regions
    labeled, num_features = label(star_mask)
    
    return labeled, num_features

def fetch_star_catalog(ra, dec, radius=1):
    """Fetch star catalog data around given RA and Dec."""
    vizier = Vizier(columns=['RAJ2000', 'DEJ2000'])
    catalog = "I/284/out"
    coords = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
    
    # Query the catalog around the coordinates with the given radius
    result = vizier.query_region(coords, radius=radius*u.deg, catalog=catalog)
    
    if result:
        return result[0]
    else:
        return None

def login_to_astrometry(api_key):
    """Login to astrometry.net and get session key."""
    login_url = 'http://nova.astrometry.net/api/login'
    data = {'request-json': json.dumps({"apikey": api_key})}
    
    response = requests.post(login_url, data=data)
    
    if response.status_code == 200:
        response_json = response.json()
        session_key = response_json.get('session')
        if session_key:
            return session_key
        else:
            print("Login failed, session key not found.")
            print("Response text:", response.text)
            return None
    else:
        print("Failed to login to astrometry.net")
        print("Response status code:", response.status_code)
        print("Response text:", response.text)
        return None

def solve_astrometry(image_path, session_key):
    """Solve astrometry using astrometry.net API."""
    upload_url = 'http://nova.astrometry.net/api/upload'
    data = {'request-json': json.dumps({'session': session_key})}
    
    with open(image_path, 'rb') as image_file:
        files = {'file': image_file}
        response = requests.post(upload_url, data=data, files=files)
    
    if response.status_code == 200:
        response_json = response.json()
        if 'subid' in response_json:
            job_id = response_json['subid']
            return job_id
        else:
            print("Upload failed, server returned an error.")
            print("Response text:", response.text)
            return None
    else:
        print("Failed to upload image to astrometry.net")
        print("Response status code:", response.status_code)
        print("Response text:", response.text)
        return None

def check_astrometry_status(job_id, session_key):
    """Check the status of the astrometry job."""
    status_url = f'http://nova.astrometry.net/api/jobs/{job_id}'
    data = {'request-json': json.dumps({'session': session_key})}
    
    while True:
        response = requests.get(status_url, params=data)
        if response.status_code == 200:
            response_json = response.json()
            status = response_json.get('status')
            if status == 'success':
                return response_json
            elif status == 'failure':
                print("Astrometry job failed.")
                return None
            else:
                print("Astrometry job status:", status)
                time.sleep(5)
        else:
            print("Failed to check astrometry job status.")
            print("Response status code:", response.status_code)
            print("Response text:", response.text)
            return None

def extract_ra_dec_from_header(header):
    """Extract RA and Dec from FITS header."""
    ra = header.get('CRVAL1') or header.get('RA')
    dec = header.get('CRVAL2') or header.get('DEC')
    return ra, dec

def process_image(image_path, api_key):
    """Process the image to detect stars, fetch catalog data, and solve astrometry."""
    # Load image data and header
    with fits.open(image_path) as hdul:
        image_data = hdul[0].data
        header = hdul[0].header
    
    # Extract RA and Dec from header
    ra, dec = extract_ra_dec_from_header(header)
    if ra is None or dec is None:
        print("RA and Dec not found in FITS header.")
        return
    
    print(f"Extracted RA: {ra}, Dec: {dec} from FITS header.")
    
    # Detect stars
    labeled, num_features = detect_stars(image_data)
    print(f"Detected {num_features} stars in the image.")
    
    # Fetch star catalog data
    catalog = fetch_star_catalog(ra, dec, radius=1)
    
    if catalog:
        print("Catalog data fetched successfully.")
        # Here you would typically match detected stars with catalog data
    
    # Login to astrometry.net and get session key
    session_key = login_to_astrometry(api_key)
    
    if not session_key:
        print("Failed to obtain session key.")
        return
    
    # Solve astrometry
    job_id = solve_astrometry(image_path, session_key)
    
    if job_id:
        print(f"Astrometry job submitted, job ID: {job_id}")
        result = check_astrometry_status(job_id, session_key)
        
        if result:
            print("Astrometry solved successfully:", result)
        else:
            print("Astrometry solving failed or did not complete.")
    else:
        print("Failed to submit astrometry job.")

if __name__ == "__main__":
    # Path to the FITS image file
    image_path = r"C:\Users\USER\Desktop\finalGPbegad\fits\NEOS_SCI_2024001000555.fits"
    
    # Your astrometry.net API key
    api_key = 'kicxdposandvygae'
    
    process_image(image_path, api_key)


