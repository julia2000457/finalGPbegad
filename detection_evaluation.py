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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from astropy import wcs
from threshold import iterative_thresholding

def detect_stars(image_data, min_distance=5):
    smoothed_data = gaussian_filter(image_data, sigma=2)
    threshold=iterative_thresholding(image_data)
    coordinates = peak_local_max(smoothed_data, threshold_abs=threshold, min_distance=min_distance)
    star_mask = np.zeros_like(smoothed_data, dtype=bool)
    star_mask[tuple(coordinates.T)] = True
    labeled, num_features = label(star_mask)
    return labeled, coordinates

def fetch_star_catalog(ra, dec, radius=1):
    vizier = Vizier(columns=['RAJ2000', 'DEJ2000'])
    catalog = "I/284/out"
    coords = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
    result = vizier.query_region(coords, radius=radius*u.deg, catalog=catalog)
    if result:
        return result[0]
    else:
        return None

def login_to_astrometry(api_key):
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
    ra = header.get('CRVAL1') or header.get('RA')
    dec = header.get('CRVAL2') or header.get('DEC')
    return ra, dec

def convert_pixel_to_world(coords, wcs_info):
    return wcs_info.pixel_to_world(coords[:, 1], coords[:, 0])

def match_stars(detected_coords, catalog_coords, tolerance=1.0):
    detected = SkyCoord(ra=detected_coords.ra, dec=detected_coords.dec)
    catalog = SkyCoord(ra=catalog_coords['RAJ2000'], dec=catalog_coords['DEJ2000'])
    idx, d2d, _ = detected.match_to_catalog_sky(catalog)
    matched = d2d < tolerance * u.arcsec
    return matched

def evaluate_detection_for_object(detected_coords, catalog_coords, tolerance=1.0):
    detected = SkyCoord(ra=detected_coords.ra, dec=detected_coords.dec)
    catalog = SkyCoord(ra=catalog_coords['RAJ2000'], dec=catalog_coords['DEJ2000'])
    idx, d2d, _ = detected.match_to_catalog_sky(catalog)
    matched = d2d < tolerance * u.arcsec
    
    y_true = np.ones(len(detected_coords), dtype=int)
    y_pred = matched.astype(int)
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    return precision, recall, f1, accuracy

def process_image(image_path, api_key):
    with fits.open(image_path) as hdul:
        image_data = hdul[0].data
        header = hdul[0].header
    
    ra, dec = extract_ra_dec_from_header(header)
    if ra is None or dec is None:
        print("RA and Dec not found in FITS header.")
        return
    
    print(f"Extracted RA: {ra}, Dec: {dec} from FITS header.")
    
    labeled, detected_coords = detect_stars(image_data)
    print(f"Detected {len(detected_coords)} stars in the image.")
    
    w = wcs.WCS(header)
    detected_coords_world = convert_pixel_to_world(detected_coords, w)
    
    catalog = fetch_star_catalog(ra, dec, radius=1)
    if catalog:
        print("Catalog data fetched successfully.")
        precision, recall, f1, accuracy = evaluate_detection(detected_coords_world, catalog)
        print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}, Accuracy: {accuracy}")
    
    session_key = login_to_astrometry(api_key)
    if not session_key:
        print("Failed to obtain session key.")
        return
    
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
    image_path = r"E:\finalGPbegad\fits\NEOS_SCI_2024001000312.fits"
    api_key = 'kicxdposandvygae'
    process_image(image_path, api_key)
