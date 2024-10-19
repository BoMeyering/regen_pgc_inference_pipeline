"""
Main inference script
BoMeyering 2024
"""
from typing import Union, Tuple, List
from glob import glob
from tqdm import tqdm
from pathlib import Path
import torch
import os
import numpy as np
import pandas as pd
import cv2
import requests
import random
from src.img_utils import show_image, draw_bounding_boxes, overlay_preds, map_preds, point_transform, scale_midpoints, scale_points, get_marker_midpoints, back_transform_mask
from src.veg_indices import clahe_channel
from src.transforms import get_pgc_transforms, get_marker_transforms
import torch.nn.functional as F

MARKER_ENDPOINT = "https://pgcview.org/api/v1/predict_markers"
PGC_ENDPOINT = "https://pgcview.org/api/v1/predict_pgc"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_filenames(root_dir: Union[str, Path]) -> List:
    """
    Lists all of the jpg images in a given folder

    Args:
        root_dir (Union[str, Path]): Root directory where the images are stored

    Returns:
        list: list of images basenames
    """
    filenames = []
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
        filenames.extend(glob(ext, root_dir=root_dir))
        
    return filenames

def get_models(mdm_path: Union[str, Path], pgc_path: Union[str, Path]) -> Tuple[torch.nn.Module, torch.nn.Module]:
    marker_model = torch.load(mdm_path, map_location=device)
    pgc_model = torch.load(pgc_path, map_location=device)
    
    return marker_model, pgc_model

def invoke_endpoints(path, **kwargs):
    """
    Invoke the marker and pgc prediction endpoints and return the results
    """
    # Make the marker model request
    files_body = {'file': open(path, 'rb')}
    marker_response = requests.post(MARKER_ENDPOINT, files=files_body, **kwargs)

    # Reopen img as binary
    files_body = {'file': open(path, 'rb')}
    pgc_response = requests.post(PGC_ENDPOINT, files=files_body, **kwargs)

    # Format responses
    print(marker_response.json().keys())
    filename = marker_response.json()['filename']
    data = marker_response.json()['data']

    marker_data = {
        'coordinates': np.array(marker_response.json()['data']['coordinates']),
        'classes': np.array(marker_response.json()['data']['classes'])
    }
    pgc_data = np.array(pgc_response.json()['data'])

    return filename, marker_data, pgc_data


@torch.no_grad()
def run_inference(model: torch.nn.Module, img: torch.tensor) -> torch.tensor:
    """
    Wrapper for model inference

    Args:
        model (torch.nn.Module): The Torch.nn.Module model object.
        img (torch.tensor): The image used for inference formatted as a torch.tensor.

    Returns:
        torch.tensor: The raw output from the model.
    """
    out = model(img)
    
    return out.squeeze()

@torch.no_grad()
def format_pgc_preds(preds: torch.tensor) -> np.ndarray:
    """
    Format raw logits into a segmentation map

    Args:
        preds (torch.tensor): Raw logit predictions of shape (N, C, H, W)

    Returns:
        np.ndarray: A numpy array of integer classes of shape (H, W)
    """
    # Turn the logits into probabilities
    probs = F.softmax(preds, dim=0)
    # Map the highest probability into the class integer
    class_map = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)

    return class_map


def class_proportions(preds):
    
    counts = np.bincount(preds.flatten(), minlength=9)
    total_elements = preds.size
    props = counts / total_elements
    
    keys = ['background', 'quadrat', 'pgc_grass', 'pgc_clover', 'broadleaf_weed', 'maize', 'soybean', 'other_vegetation']
    prop_dict = {k: v for k, v in zip(keys, props)}
    
    return props, prop_dict



if __name__ == '__main__':

    # Set pipeline parameters
    # Directories
    IMAGE_DIR = 'assets/images'

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
    
    # Grab filenames from IMAGE_DIR
    filenames = get_filenames(IMAGE_DIR)
    print(filenames)
    
    # marker_model, pgc_model = get_models(MDM_PATH, PGC_PATH)
    
    # marker_transforms = get_marker_transforms()
    # seg_transforms = get_val_transforms()
    
    
    results = pd.DataFrame(columns=['filename', 'background', 'quadrat', 'pgc_grass', 
                                    'pgc_clover', 'broadleaf_weed', 'maize', 'soybean', 
                                    'other_vegetation', 'active_grass', 'dormant_grass'])
    
    
    for file in filenames:
        print(file)
        file_path = os.path.join(IMAGE_DIR, file)
        filename, marker_data, pgc_preds = invoke_endpoints(file_path)
        marker_preds = marker_data['coordinates']
        marker_classes = marker_data['classes']

    #     path = os.path.join(IMAGE_DIR, file)
        img = cv2.imread(file_path)
        h, w = img.shape[:2]
        
    #     # Get normalized images for inference
    #     marker_img = marker_transforms(image=img)['image'].to(device).unsqueeze(0)
    #     pgc_img = seg_transforms(image=img)['image'].to(device).unsqueeze(0)
        
    #     # Get marker detections and filter
    #     pts = run_inference(marker_model, marker_img)
    #     row_idx = pts[:, 4] > .5
    #     pts = pts[row_idx].cpu().numpy()
        
    #     # print(pts)
        if len(marker_preds) <= 2:
            print('Fewer than 2 markers predicted')
            continue
        
        # Get marker midpoint coords
        mdpts = get_marker_midpoints(marker_preds)
        
        # Scale the bbox points and midpoints back up to original size
        sc_preds = scale_points(marker_preds, (w, h))
        sc_mdpts = scale_midpoints(mdpts, (w, h))

        # print("PREDS")
        # print(marker_preds)
        # print("SCPREDS")
        # print(sc_preds)
        # print("MDPTS")
        # print(mdpts)
        # print("SCMDPTS")
        # print(sc_mdpts)
        
        # Get segmentation preds and extract grass mask
        pgc_preds = pgc_preds.astype(np.uint8)
        grass_mask = np.zeros((1024, 1024))

        # Get grass predictions
        idx = np.where(pgc_preds==2)
        grass_mask[idx] = 255
        grass_mask = grass_mask.astype(np.uint8)

        # show_image(grass_mask)
        
        # Resize raw image to inference size
        img_resized = cv2.resize(img, (1024, 1024))
        
        try:
            # show_image(img)
            transformed, M = point_transform(img, sc_mdpts, output_shape=(1024, 1024))
            # show_image(transformed)
            
            # show_image(img_resized)
            grass_mask_transformed, M = point_transform(grass_mask, mdpts, (1024, 1024))
            # show_image(grass_mask_transformed)
            
            # print(pgc_preds)
            # print(pgc_preds.shape)
            # show_image(pgc_preds*35)

            preds_transformed, M = point_transform(pgc_preds, mdpts, (1024, 1024))
            # show_image(preds_transformed*35)
            props, props_dict = class_proportions(preds_transformed)

            props_dict['filename'] = file

    #         grass_mask, M = point_transform(grass_mask, mdpts, (1024, 1024))
    #         # print(grass_mask)
    #         # print(grass_mask.shape)
            
    #         props, prop_dict = class_proportions(pred_transformed)
    #         prop_dict['filename'] = file
            
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
            
            # show_image(green)
            # show_image(brown)
            # show_image(overlay)

            # Calculate green/brown props
            green_fraction = (hsv_gr_mask > 0).sum()
            brown_fraction = (hsv_br_mask > 0).sum()
            total_grass_px = green_fraction + brown_fraction
            green_fraction /= total_grass_px
            brown_fraction /= total_grass_px
            props_dict['active_grass'] = green_fraction
            props_dict['dormant_grass'] = brown_fraction
            
    #         big_mask = back_transform_mask(img, mask=grass_mask, pts=sc_mdpts.astype(np.float32))
            
    #         masked_img = cv2.bitwise_and(img, img, mask=big_mask)
    #         # show_image(masked_img)
            
            
    #         # cv2.namedWindow('grass')
    #         # cv2.namedWindow('green')
    #         # cv2.namedWindow('brown')
    #         # cv2.imshow('grass', grass_mask*255)
    #         # cv2.imshow('green', hsv_gr_mask)
    #         # cv2.imshow('brown', hsv_br_mask)
    #         # cv2.waitKey()
    #         # cv2.destroyAllWindows()
            
    #         # Save images
    #         roi_out = Path('output') / ('roi_' + file)
    #         clahe_out = Path('output') / ('clahe_' + file)
    #         green_out = Path('output') / ('green_' + file)
    #         brown_out = Path('output') / ('brown_' + file)
            overlay_out = Path('output') / ('overlay_' + file)
    #         grass_mask_out = Path('output') / ('grass_' + file)
    #         cv2.imwrite(str(roi_out), roi_transformed)
    #         cv2.imwrite(str(clahe_out), clahe_img)
    #         cv2.imwrite(str(green_out), green)
    #         cv2.imwrite(str(brown_out), brown)
            cv2.imwrite(str(overlay_out), overlay)
    #         cv2.imwrite(str(grass_mask_out), masked_img)
            
            prop_df = pd.DataFrame([props_dict])
            results = pd.concat([results, prop_df], ignore_index=True)
            
        except RuntimeError:
            print("Fewer than 3 corner points detected. Not transforming the image.")
            show_image(img_resized)
    print(results)
    
    results.to_csv('output_results.csv')
        
        

        