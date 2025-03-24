# Pipeline Script
# BoMeyering 2024

import os
import cv2
from typing import List, Tuple, Union
from tqdm import tqdm
from glob import glob
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path

from src.api_calls import invoke_endpoints
from src.img_utils import show_image, draw_bounding_boxes, overlay_preds, map_preds, point_transform, scale_midpoints, scale_points, get_marker_midpoints, back_transform_mask
from src.veg_indices import clahe_channel

def get_filenames(root_dir: Union[str, Path]='assets/images') -> dict:
    """
    Lists all of the jpg images in a given folder

    Args:
        root_dir (Union[str, Path]): Root directory where the images are stored

    Returns:
        dict: img_root_dir and a list of images basenames
    """
    filenames = []
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
        filenames.extend(glob(ext, root_dir=root_dir))
    
    result_dict = {
        'root_dir': root_dir,
        'filenames': filenames
    }

    return result_dict

def class_proportions(preds: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Calculate the proportions of each PGC class in the predictions

    Args:
        preds (np.ndarray): The integer predictions as a numpy array of shape (H, W)

    Returns:
        Tuple[np.ndarray, dict]: a 1-dimensional numpy array of the proportions of each class, along with a dictionary with keys corresponding to the class name.
    """
    counts = np.bincount(preds.flatten(), minlength=9)
    total_elements = preds.size
    props = counts / total_elements
    
    keys = ['background', 'quadrat', 'pgc_grass', 'pgc_clover', 'broadleaf_weed', 'maize', 'soybean', 'other_vegetation']
    prop_dict = {k: v for k, v in zip(keys, props)}
    
    return props, prop_dict


def run_pipeline(filenames_dict: dict) -> None:
    """
    Run the PGC View Image Analysis Pipeline

    Args:
        images (List[str]): A list of image file paths.
    """

    # Marker class map
    marker_map = {
        1: 'marker',
        2: 'quadrat'
    }
    
    # HSV threshold settings
    brown_lo = (0, 0, 45)
    brown_hi = (30, 110, 255)
    green_lo = (32, 45, 45)
    green_hi = (115, 255, 255)
    
    # PGC class to RGB mapping
    mapping = {
        0: (0, 0, 0), # soil
        1: (255, 255, 255), # quadrat
        2: (255, 0, 0), # grass
        3: (0, 255, 0), # clover
        4: (0, 0, 255), # weeds
        5: (0, 75, 55), # corn
        6: (12, 142, 194), # soybean
        7: (133, 12, 194) # other_vegetation
    }

    IMAGE_DIR = filenames_dict['root_dir']
    IMAGE_NAMES = filenames_dict['filenames']

    # Results Dataframe
    results = pd.DataFrame(columns=['filename', 'num_markers', 'prediction_zone', 'background', 'quadrat', 'pgc_grass', 
                                    'pgc_clover', 'broadleaf_weed', 'maize', 'soybean', 
                                    'other_vegetation', 'active_grass', 'dormant_grass'])

    
    # Main loop
    pbar = tqdm(IMAGE_NAMES)
    for filename in pbar:
        img_path = os.path.join(IMAGE_DIR, filename)
        if os.path.exists(img_path):
            pbar.set_description(f"Processing image {filename}")
            
            # Invoke endpoints and pull out data
            filename, marker_data, pgc_preds = invoke_endpoints(img_path)
            marker_preds = marker_data['coordinates']
            marker_classes = marker_data['classes']

            # Read in the raw image and set height and width
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            num_markers = len(marker_preds)
            try:
                if num_markers <= 2:
                    print('Fewer than 2 markers predicted, returning predictions over entire image.')
                    # Resize raw image to inference size
                    img_resized = cv2.resize(img, (1024, 1024))
                    # Get segmentation preds and extract grass mask
                    pgc_preds = pgc_preds.astype(np.uint8)
                    grass_mask = np.zeros((1024, 1024))
                    # Get grass predictions
                    idx = np.where(pgc_preds==2)
                    grass_mask[idx] = 255
                    grass_mask = grass_mask.astype(np.uint8)


                    preds_transformed = cv2.resize(pgc_preds, (1024, 1024))
                    props, props_dict = class_proportions(preds_transformed)

                    # Assign metadata to props_dict
                    props_dict['filename'] = filename
                    props_dict['prediction_zone'] = 'whole_image'

                    # Green Brown Pixel Discrimination
                    clahe_img = clahe_channel(img_resized, 20)
                    hsv_img = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2HSV)
                    hsv_gr_mask = cv2.inRange(hsv_img, green_lo, green_hi) * grass_mask
                    hsv_br_mask = cv2.inRange(hsv_img, brown_lo, brown_hi) * grass_mask

                    # Create green/brown images and overlay
                    green = cv2.bitwise_and(img_resized, img_resized, mask=hsv_gr_mask)
                    brown = cv2.bitwise_and(img_resized, img_resized, mask=hsv_br_mask)
                    color_mask = map_preds(preds_transformed, mapping)
                    overlay = overlay_preds(img_resized, color_mask, alpha=.4)

                else:
                    # Get marker midpoint coords
                    mdpts = get_marker_midpoints(marker_preds)
                
                    # Scale the bbox points and midpoints back up to original size
                    sc_preds = scale_points(marker_preds, (w, h))
                    sc_mdpts = scale_midpoints(mdpts, (w, h))

                    # Get segmentation preds and extract grass mask
                    pgc_preds = pgc_preds.astype(np.uint8)
                    grass_mask = np.zeros((1024, 1024))

                    # Get grass predictions
                    idx = np.where(pgc_preds==2)
                    grass_mask[idx] = 255
                    grass_mask = grass_mask.astype(np.uint8)

                    # Resize raw image to inference size
                    img_resized = cv2.resize(img, (1024, 1024))

                    transformed, M = point_transform(img, sc_mdpts, output_shape=(1024, 1024))
                    grass_mask_transformed, M = point_transform(grass_mask, mdpts, (1024, 1024))
                    preds_transformed, M = point_transform(pgc_preds, mdpts, (1024, 1024))
                    props, props_dict = class_proportions(preds_transformed)
                    
                    # Assign results to props_dict
                    props_dict['filename'] = filename
                    props_dict['prediction_zone'] = 'marker_roi'

                    # Green Brown Pixel Discrimination
                    clahe_img = clahe_channel(transformed, 20)
                    hsv_img = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2HSV)
                    hsv_gr_mask = cv2.inRange(hsv_img, green_lo, green_hi) * grass_mask_transformed
                    hsv_br_mask = cv2.inRange(hsv_img, brown_lo, brown_hi) * grass_mask_transformed
                
                    # Create green/brown images and overlay
                    green = cv2.bitwise_and(transformed, transformed, mask=hsv_gr_mask)
                    brown = cv2.bitwise_and(transformed, transformed, mask=hsv_br_mask)
                    color_mask = map_preds(preds_transformed, mapping)
                    overlay = overlay_preds(transformed, color_mask, alpha=.4)

                # Assign the number of markers in the results
                props_dict['num_markers'] = num_markers

                # Calculate green/brown props if pgc_grass
                if props_dict['pgc_grass'] > 0:
                    green_fraction = (hsv_gr_mask > 0).sum()
                    brown_fraction = (hsv_br_mask > 0).sum()
                    total_grass_px = green_fraction + brown_fraction
                    green_fraction /= total_grass_px
                    brown_fraction /= total_grass_px
                    props_dict['active_grass'] = green_fraction
                    props_dict['dormant_grass'] = brown_fraction
                else:
                    props_dict['active_grass'] = 0
                    props_dict['dormant_grass'] = 0
                

                    
                overlay_out = Path('output') / ('overlay_' + filename)
                cv2.imwrite(str(overlay_out), overlay)
                
                prop_df = pd.DataFrame([props_dict])
                results = pd.concat([results, prop_df], ignore_index=True)
            
            except RuntimeError:
                print("Fewer than 3 corner points detected. Not transforming the image.")
        else:
            print(f"File '{filename}' does not exist. Skipping")
    print(results)

    # New Polars dataframe addition
    formatted_results = pl.from_pandas(results)

    # Calculate aggregated results (Add erroneous predictions to correct classes)
    formatted_results.with_columns(
        [
            (pl.col("broadleaf_weed") + pl.col("pgc_clover") + pl.col("other_vegetation")).alias("broadleaf_weed"),
            (pl.col("background") + pl.col("quadrat")).alias("background")
        ]    
    )[['filename', 'num_markers', 'prediction_zone', 'pgc_grass', 'broadleaf_weed', 'background', 'active_grass', 'dormant_grass']] #subset needed columns
    
    formatted_results = formatted_results.with_columns(
        (pl.col('pgc_grass') + pl.col('broadleaf_weed') + pl.col('background')).alias('roi_sums')
    ).with_columns(  # Calculate roi_sums
        (pl.col('pgc_grass') / pl.col('roi_sums')).alias('Pgc_Cov_prop'),  # Update Pgc_Cov_prop
        (pl.col('broadleaf_weed') / pl.col('roi_sums')).alias('Wd_Cov_prop'),  # Update Wd_Cov_prop
        (pl.col('background') / pl.col('roi_sums')).alias('So_Cov_prop')  # Update So_Cov_prop
    ).drop(pl.col('roi_sums', 'pgc_grass', 'broadleaf_weed', 'background', 'quadrat', 'pgc_clover', 'maize', 'soybean', 'other_vegetation'))\
    .rename(
        {
            'active_grass': 'Pgc_ActFrac_prop',
            'dormant_grass': "Pgc_DorFrac_prop"
        }
    )

    formatted_results.write_csv('output/formatted_results.csv')
    results.to_csv('output/output_results.csv')

if __name__ == '__main__':
    images = get_filenames('assets/images')

    run_pipeline(images)
