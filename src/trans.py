# Marker Transformation
# BoMeyering 2024

import numpy as np
import cv2
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from typing import Tuple

def order_points(pts: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """_summary_

    Args:
        pts (np.ndarray): _description_
        img_shape (Tuple[int, int]): _description_

    Returns:
        np.ndarray: _description_
    """
    
    # Unpack the image shape
    height, width = img_shape
    
    if np.any(np.max(pts, axis=0) > img_shape[::-1]):
        raise ValueError('One or more of the coordinates in pts is outside the bounds of the image.')
    elif np.any(pts < 0):
        raise ValueError('One or more of the coordinates in pts is negative.')
    elif len(pts) <= 2:
        raise ValueError("Only two points were passed to 'pts'.")
    
    # Set up the iamge corner array and compute distance matrix
    img_corners = np.array([
        [0, 0],
        [width, 0], 
        [width, height],
        [0, height]
    ])
    
    d_mat = distance_matrix(img_corners, pts)

    # Optimize the linear sums to find the best matching corner using distance a cost
    row_ind, col_ind = linear_sum_assignment(d_mat)

    print(col_ind)
    print(row_ind)
    
    if len(row_ind) < 4:
        rect = pts[col_ind].astype('float32')
        temp_rect = np.zeros((4, 2)).astype('float32')
        temp_rect[row_ind] = rect
        if 0 not in row_ind:
            p_vec = temp_rect[2] - temp_rect[1]
            point = temp_rect[3] - p_vec
            temp_rect[0] = point
            return temp_rect
        elif 1 not in row_ind:
            p_vec = temp_rect[3] - temp_rect[0]
            point = temp_rect[2] - p_vec
            temp_rect[1] = point
            return temp_rect
        elif 2 not in row_ind:
            p_vec = temp_rect[0] - temp_rect[3]
            point = temp_rect[1] - p_vec
            temp_rect[2] = point
            return temp_rect
        elif 3 not in row_ind:
            p_vec = temp_rect[1] - temp_rect[2]
            point = temp_rect[0] - p_vec
            temp_rect[3] = point
            return temp_rect
    else:
        # Reorder the points
        rect = pts[col_ind].astype('float32')

        return rect

def point_transform(img: np.ndarray, pts: np.ndarray, output_shape: Tuple[int, int]=None) -> np.ndarray:
    
    rect = order_points(pts, img.shape[:2])
    tl, tr, br, bl = rect
    
    if not output_shape:
        wA = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        wB = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width = max(int(wA), int(wB))
        
        hA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        hB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        height = max(int(hA), int(hB))
    else:
        height, width = output_shape
        
    dst = np.array(
        [
            [0, 0],
            [width, 0,],
            [width, height],
            [0, height]
        ],
        dtype = "float32"
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    
    transformed = cv2.warpPerspective(img, M, (width, height))
    
    return transformed
    
def show_image(img):
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    img = cv2.imread('assets/images/72649807-5735-451d-834a-5ab80be1396f.jpg')
    # show_image(img)
    
    
    points0 = np.array([[687, 2267], [3980, 231], [4066, 2349]]) # missing 0
    points1 = np.array([[687, 2267], [4066, 2349], [839, 22]]) # missing 1
    points2 = np.array([[687, 2267], [3980, 231], [839, 22]]) # missing 2
    points3 = np.array([[3980, 231], [4066, 2349], [839, 22]]) # missing 3
    points = np.array([[687, 2267], [3980, 231], [4066, 2349], [839, 22]])
    # rect = order_points(pts=points, img_shape=img.shape[:2])
    # print(rect)
    transformed = point_transform(img, points)
    show_image(transformed)