# Image Utils
# BoMeyering 2024

import cv2
import json
import torch
import albumentations
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from typing import Tuple
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment


def show_image(img: np.ndarray, window_name: str='test'):
    """Create a gui window to quickly show an OpenCV image"""

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def draw_bounding_boxes(img, pts: torch.tensor, pts_range=1024):
    """
    Draw bounding boxes on an image
    """
    if img.shape[:2] != (1024, 1024):
        h, w = img.shape[0:2]
        thickness = 10
    else:
        h, w = 1024, 1024
        thickness = 3
    
    img_copy = img.copy()
    for bbox in pts:
        pt_list = bbox[0:4].tolist()
        print(pt_list)
        x1, y1, x2, y2 = (int(i) for i in pt_list) 
        # x1, x2  = (int(x*w/1024) for x in [x1, x2])
        # y1, y2 = (int(y*h/1024) for y in [y1, y2])
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 100, 100), thickness)
        
    return img_copy

def map_preds(preds, mapping):
    """preds = integer mask of shape (H, W)"""

    h, w = preds.shape
    color_mask = np.zeros(shape=(h, w, 3)).astype(np.uint8)
    for i in np.unique(preds):
        idx = np.where(preds==i)
        rgb = mapping.get(i)
        color_mask[idx] = np.array(rgb)
    
    return color_mask

def overlay_preds(img, color_mask, alpha, gamma=0.0):
    beta = 1-alpha
    overlay = cv2.addWeighted(img, alpha, color_mask, beta, gamma)
    return overlay

def order_points(pts: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """_summary_

    Args:
        pts (np.ndarray): point array with x, y coordinates of shape (3, 2) or (4, 2)
        img_shape (Tuple[int, int]): _description_

    Returns:
        np.ndarray: _description_
    """

    # Unpack the image shape
    height, width = img_shape
    if len(pts) <= 2:
        raise ValueError("Fewer than three points were passed to 'pts'.")
    elif np.any(pts > np.max(img_shape)):
        raise ValueError('One or more of the coordinates in pts is outside the bounds of the image.')
    # elif np.any(np.max(pts, axis=0) > img_shape[::-1]):
    #     raise ValueError('One or more of the coordinates in pts is outside the bounds of the image.')
    elif np.any(pts < 0):
        raise ValueError('One or more of the coordinates in pts is negative.')        

    # Set up the iamge corner array and compute distance matrix. Start at top left origin and work clockwise
    img_corners = np.array([
        [0, 0],
        [width, 0], 
        [width, height],
        [0, height]
    ])

    # Calculate the distance between all of the points
    d_mat = distance_matrix(img_corners, pts)

    # Optimize the linear sums to find the best matching corner using distance a cost
    row_ind, col_ind = linear_sum_assignment(d_mat)

    if len(row_ind) < 4:
        rect = pts[col_ind].astype('float32')
        temp_rect = np.zeros((4, 2)).astype('float32')
        temp_rect[row_ind] = rect
        if 0 not in row_ind:
            p_vec = temp_rect[2] - temp_rect[1]
            point = temp_rect[3] - p_vec
            print(point)
            temp_rect[0] = point
            return temp_rect
        elif 1 not in row_ind:
            p_vec = temp_rect[3] - temp_rect[0]
            point = temp_rect[2] - p_vec
            print(point)
            temp_rect[1] = point
            return temp_rect
        elif 2 not in row_ind:
            p_vec = temp_rect[0] - temp_rect[3]
            point = temp_rect[1] - p_vec
            print(point)
            temp_rect[2] = point
            return temp_rect
        elif 3 not in row_ind:
            p_vec = temp_rect[1] - temp_rect[2]
            point = temp_rect[0] - p_vec
            print(point)
            temp_rect[3] = point
            return temp_rect
    else:
        # Reorder the points
        rect = pts[col_ind].astype('float32')

        return rect
    

def point_transform(img: np.ndarray, pts: np.ndarray, output_shape: Tuple[int, int]=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make a linear transform from a set of 4 points to a new coordinate space.

    Args:
        img (np.ndarray): a 3 channel images stored as a Numpy array
        pts (np.ndarray): point array with x, y coordinates of shape (3, 2) or (4, 2)
        output_shape (Tuple[int, int], optional): a tuple of two integers corresponding to the desired height and width of the transformed image. Defaults to None.

    Returns:
        transformed (np.ndarray): The original image transformed to the new points
        M (np.ndarray): The transformation matrix
    """
    # Order and impute points if 3 <= len(pts) < 4 
    rect = order_points(pts, img.shape[:2])
    tl, tr, br, bl = rect
    pt_array = np.array([tl, tr, br, bl])
    
    # Calculate a "natural" transformed image shape
    if not output_shape:
        height, width = natural_shape(pt_array)
    else:
        height, width = output_shape
    
    # Create the target array for the transform
    dst = np.array(
        [
            [0, 0],
            [width, 0,],
            [width, height],
            [0, height]
        ],
        dtype = "float32"
    )

    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Transform the image using nearest interpolation
    transformed = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_NEAREST)
    
    return transformed, M

def natural_shape(pt_array: np.ndarray) -> Tuple[int, int]:
    """
    Find the 'natural' shape of an image for a point transformation.

    Args:
        pt_array (np.ndarray): A numpy array of shape (4, 2) containing ordered points from the top left going in clockwise orientation

    Returns:
        Tuple[int, int]: The "natural" height and width of the the new transformed ROI
    """
    top_width = np.linalg.norm(pt_array[1]-pt_array[0])
    bottom_width = np.linalg.norm(pt_array[3]-pt_array[2])
    left_height = np.linalg.norm(pt_array[3]-pt_array[1])
    right_height = np.linalg.norm(pt_array[2]-pt_array[0])

    width = np.max(int(top_width), int(bottom_width))
    height = np.max(int(left_height), int(right_height))

    return height, width

def scale_points(pts: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
    """
    Scales a set of bounding box coordinates from a (1024, 1024) reference to a new size.

    Args:
        pts (np.ndarray): A numpy array of raw model output as x1, y1, x2, y2 bbox coordinates with scores
        new_size (Tuple[int, int]): a tuple of a new coordinate space to scale to

    Returns:
        np.ndarray: the scaled bbox points as integers.
    """
    # Grab the (x, y) coordinates
    coords = pts[:, 0:4]
    # Expand the size tuple and calculate the resized coordinates
    scale = new_size * 2
    sc_coords = ((coords * np.array([scale])) / 1024).astype(np.int32)
    
    # Copy the new coordinates into the scaled points array and return
    sc_pts = pts.copy()
    sc_pts[:, 0:4] = sc_coords
    
    return sc_pts

def scale_midpoints(mdpts: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
    """
    Scales a set of x,y coordinates from a (1024, 1024) reference to a new size.

    Args:
        mdpts (np.ndarray): A numpy array of x, y coordinates
        new_size (Tuple[int, int]): a tuple of a new coordinate space to scale to

    Returns:
        np.ndarray: the scaled midpoints as integers.
    """
    
    sc_mdpts = ((mdpts * np.array([[*new_size]])) / 1024).astype(np.int32)
    
    return sc_mdpts

def get_marker_midpoints(pts: np.ndarray, clip: bool=False) -> np.ndarray:
    """
    Take a set of marker bounding box coordinates and return the midpoints of each one.

    Args:
        pts (np.ndarray): A numpy array of shape (N, 4), in the format (x1, y1, x2, y2).
        clip (bool, optional): Clip the midpoints between 0 and 1024. Defaults to False.

    Returns:
        np.ndarray: A numpy array of shape (N, 2) with coordinates in the format of (cx, cy)
    """
    if clip:
        pts[:, :4] = pts[:, :4].clip(min=0, max = 1024)
    
    # Calculate the mid x and y coordinates
    cx = ((pts[:, 2] + pts[:, 0]) / 2).astype(np.int32)
    cy = ((pts[:, 3] + pts[:, 1]) / 2).astype(np.int32)
    
    mdpts = np.array([[x, y] for x, y in zip(cx, cy)])
    
    return mdpts

def back_transform_mask(img: np.ndarray, mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Back transform a modified ROI image onto the original image.

    Args:
        img (np.ndarray): The original, unmodified image.
        mask (np.ndarray): A binary image or mask ROI to apply to the original image.
        pts (np.ndarray): A set of ROI marker midpoints.

    Returns:
        np.ndarray: _description_
    """
    
    img_h, img_w = img.shape[:2]
    mask_h, mask_w = mask.shape
    mask_pts = np.float32([[0, 0], [mask_w, 0],
                       [mask_w, mask_h], [0, mask_h]])
    
    pts = order_points(pts, img.shape[:2])

    # Create a new white mask the size of img
    one_mask = np.ones_like(img)[:, :, 0].astype(np.uint8)*255
    
    # Apply Perspective Transform Algorithm
    M = cv2.getPerspectiveTransform(mask_pts, pts)
    
    # Get transform matrix and transform
    mask = cv2.bitwise_not(mask)
    transformed_mask = cv2.warpPerspective(mask, M, (w, h))

    combo_mask = ((cv2.bitwise_xor(one_mask, transformed_mask) > 0)*255).astype(np.uint8)
    
    return combo_mask