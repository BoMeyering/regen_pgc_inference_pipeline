# Four Point Image Transforms
# BoMeyering 2024

import numpy as np
import cv2
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from typing import Tuple

# def interpolate_points(pts: np.ndarray, img_shape: Tuple[int, int]) 
# 
def order_points(pts: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
	"""
	Get a numpy array of 4 x, y coordinates.
	Finds the distance matrix between the image corners and the points.
	Uses a linear sum operation to find the closest image corner to each point.
	Optimizes such that each point belongs to only one corner.

	Arguments:
		pts (np.ndarray): A 2D Numpy array of shape (4, 2) containing x, y point coordinates.
		shape: (Tuple[int, int]): A tuple of two integers of the image shape

	Returns:
		rect (np.ndarray): A 2d Numpy array of shape (4, 2) containing the ordered point coordinates.
	"""
    #Unpack the image shape
    height, width = img_shape

    if np.any(np.max(pts, axis=0) > img_shape[::-1]):
	    raise ValueError('One or more of the coordinates in pts is outside the bounds of the image.')
    elif np.any(pts < 0):
	    raise ValueError('One or more of the coordinates in pts is negative.')
    # Set up the image corner array and compute distance matriximg_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    d_mat = distance_matrix(img_corners, pts)
    
    print(d_mat)
    
    # Optimize the linear sums to get the best corner
	
    _, col_ind = linear_sum_assignment(d_mat)

    # Reorder the points
    rect = pts[col_ind].astype('float32')
    print(rect)
    return rect

def point_transform(img: np.ndarray, pts: np.ndarray, output_shape: Tuple[int, int]=None) -> np.ndarray:
	"""
	Get a numpy array image and an array of 4 x, y coordinates.
	Return a transformed image that with new corners found in pts.

	Arguments:
		img: (np.ndarray): A 3D Numpy array image of shape (H, W, C)
		pts (np.ndarray): A 2D Numpy array of shape (4, 2) containing x, y point coordinates.
		output_shape: (Tuple[int, int] Optional): A tuple of two integers of the image shape

	Returns:
		transformed (np.ndarray): A 3D Numpy array of shape (H, W, C) that has been transformed to the points.
	"""
	rect = order_points(pts, img.shape[:2])
	tl, tr, br, bl = rect

	# Calculate the output shapes from the longest sides of the transform
	if not output_shape:
		wA = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
		wB = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
		width = max(int(wA), int(wB))

		hA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
		hB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
		height = max(int(hA), int(hB))
	else:
		height, width = output_shape

	# Get Dst transform and transform matrix M
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

	# Transform the image
	transformed = cv2.warpPerspective(img, M, (width, height))

	return transformed

def show_img(img):
	cv2.namedWindow('test', cv2.WINDOW_NORMAL)
	cv2.imshow('test', img)
	cv2.waitKey()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	img = cv2.imread('assets/images/72649807-5735-451d-834a-5ab80be1396f.jpg')
	show_img(img)

    # points in [[x, y]] format
	# points = np.array([[687, 2267], [3980, 231], [839, 22], [4066, 2349]])
	points = np.array([[687, 2267], [3980, 231], [4066, 2349]])
	transformed = point_transform(img, points)
	show_img(transformed)
